"""Every search filter must be one the engine will actually evaluate.

Three ways a filter can be silently ignored, all of which produce plausible
results rather than errors — which is why they survived:

1. **An undeclared ``$vectorSearch`` filter path.** ``recall`` and
   ``hybrid_search`` pre-filter on ``memory_type`` and ``tags``; neither was
   declared as a ``filter`` field in ``memories_vector_index``. An undeclared
   path is not rejected, the branch simply matches nothing, so both narrowing
   arguments the API advertises guaranteed an empty result. It reads as "this
   user has no memories of that type".

2. **An unsupported pre-filter operator.** ``$vectorSearch`` supports ``$eq``/
   ``$ne``, the range operators, ``$in``/``$nin``, ``$exists``, and the logical
   operators — and nothing else. Tag filtering used ``$all``. Same silent
   outcome as (1).

3. **A restriction applied to one branch of a ``$rankFusion``.** Hybrid search
   put ``memory_type`` and ``tags`` in the vector pre-filter but not in the
   full-text ``compound.filter``. The unfiltered branch contributed matches that
   ignored the restriction and fusion mixed them into one ranked list, so a
   scoped search returned documents of every other type — wrong results rather
   than missing ones, which is worse.

These tests are written against the *built pipelines* and the *shipped index
definitions* rather than against a copied list, so a new filter added to a
service without a matching index field fails here.

REQ-E-144 (an operation's scope is the scope it states).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_memory.config import MemoryConfig
from agent_memory.core.collections import get_search_indexes
from agent_memory.services.memory import MemoryService, tag_filter, tag_fts_clauses
from agent_memory.services.search_pipeline import TEXT_BRANCH, VECTOR_BRANCH

# The complete set of MQL operators `$vectorSearch` accepts in `filter`.
# Sourced from the MongoDB Vector Search reference: equality, range, in-set,
# existence, and logical. Anything else is accepted by the driver, ignored by the
# engine, and returns nothing.
SUPPORTED_PREFILTER_OPERATORS = frozenset(
    {
        "$eq", "$ne",
        "$gt", "$lt", "$gte", "$lte",
        "$in", "$nin",
        "$exists",
        "$not", "$nor", "$and", "$or",
    }
)


def _config(**overrides) -> MemoryConfig:
    defaults = {"mongodb_connection_string": "mongodb://localhost:27017", "_env_file": None}
    defaults.update(overrides)
    return MemoryConfig(**defaults)


def _service():
    """A ``MemoryService`` whose collection records the pipeline it is given."""
    col = MagicMock()
    cursor = AsyncMock()
    cursor.to_list = AsyncMock(return_value=[])
    col.aggregate = AsyncMock(return_value=cursor)
    providers = MagicMock()
    providers.embedding = AsyncMock()
    providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 4)
    return MemoryService(col, _config(), providers), col


def _declared_filter_paths(index_name: str) -> set[str]:
    """The ``filter``-type paths a shipped vectorSearch index actually declares."""
    idx = next(i for i in get_search_indexes() if i["name"] == index_name)
    return {f["path"] for f in idx["definition"]["fields"] if f["type"] == "filter"}


def _declared_token_fields(index_name: str) -> set[str]:
    """The ``token``-mapped fields of a shipped search index.

    ``token``, specifically: an analyzed ``string`` field cannot back an exact
    ``equals``, so a field mapped as ``string`` is as unusable for filtering as
    one that is absent.
    """
    idx = next(i for i in get_search_indexes() if i["name"] == index_name)
    fields = idx["definition"]["mappings"]["fields"]
    return {name for name, spec in fields.items() if spec.get("type") == "token"}


def _filter_paths(node, prefix: str = "") -> set[str]:
    """Every document path a `$vectorSearch` filter restricts on.

    Walks the logical operators, so `{"$and": [{"tags": "a"}, {"tags": "b"}]}`
    reports `{"tags"}` rather than `{"$and"}`.
    """
    paths: set[str] = set()
    if isinstance(node, list):
        for item in node:
            paths |= _filter_paths(item, prefix)
        return paths
    if not isinstance(node, dict):
        return paths
    for key, value in node.items():
        if key.startswith("$"):
            paths |= _filter_paths(value, prefix)
        else:
            paths.add(key)
    return paths


def _filter_operators(node) -> set[str]:
    """Every `$`-prefixed operator appearing anywhere in a filter."""
    ops: set[str] = set()
    if isinstance(node, list):
        for item in node:
            ops |= _filter_operators(item)
        return ops
    if not isinstance(node, dict):
        return ops
    for key, value in node.items():
        if key.startswith("$"):
            ops.add(key)
        ops |= _filter_operators(value)
    return ops


def _recall_filter(col) -> dict:
    return col.aggregate.call_args.args[0][0]["$vectorSearch"]["filter"]


def _fusion_branches(col) -> tuple[dict, list[dict]]:
    """The vector pre-filter and full-text filter clauses of a built fusion."""
    pipes = col.aggregate.call_args.args[0][0]["$rankFusion"]["input"]["pipelines"]
    vs_filter = pipes[VECTOR_BRANCH][0]["$vectorSearch"]["filter"]
    fts_clauses = pipes[TEXT_BRANCH][0]["$search"]["compound"]["filter"]
    return vs_filter, fts_clauses


def _clause_paths(clauses: list[dict]) -> set[str]:
    """The paths an Atlas Search `compound.filter` list restricts on."""
    paths = set()
    for clause in clauses:
        for spec in clause.values():
            if "path" in spec:
                paths.add(spec["path"])
    return paths


class TestEveryPreFilterPathIsIndexed:
    """Finding 3, first half: a filter on an undeclared path matches nothing."""

    async def test_recall_filters_only_on_declared_paths(self) -> None:
        """TC-SEARCH-IDX-001: asserted against the shipped index definition.

        Every argument at once, because the defect was that the *combination* a
        real caller uses had two paths the index never declared.
        """
        svc, col = _service()
        await svc.recall(
            "u1", "q", tier=["ltm"], memory_type="factual", tags=["a", "b"]
        )
        used = _filter_paths(_recall_filter(col))
        declared = _declared_filter_paths("memories_vector_index")
        assert used <= declared, (
            f"recall pre-filters on undeclared paths {sorted(used - declared)}; "
            f"those branches match nothing and the recall returns empty"
        )

    async def test_hybrid_search_filters_only_on_declared_paths(self) -> None:
        # TC-SEARCH-IDX-002
        svc, col = _service()
        await svc.hybrid_search("u1", "q", memory_type="factual", tags=["a", "b"])
        used = _filter_paths(_fusion_branches(col)[0])
        declared = _declared_filter_paths("memories_vector_index")
        assert used <= declared, (
            f"hybrid search pre-filters on undeclared paths {sorted(used - declared)}"
        )

    async def test_evolve_memory_filters_only_on_declared_paths(self) -> None:
        """TC-SEARCH-IDX-003: the enrichment worker's own search counts too.

        It runs unattended, so an always-empty result here means duplicates are
        never merged and nothing reports a problem.
        """
        svc, col = _service()
        await svc.evolve_memory("u1", "content", [0.1] * 4)
        used = _filter_paths(_recall_filter(col))
        declared = _declared_filter_paths("memories_vector_index")
        assert used <= declared, f"undeclared: {sorted(used - declared)}"

    @pytest.mark.parametrize("path", ["memory_type", "tags"])
    def test_the_two_missing_paths_are_now_declared(self, path) -> None:
        """TC-SEARCH-IDX-004: named individually, since these are the finding."""
        assert path in _declared_filter_paths("memories_vector_index")


class TestOnlySupportedOperatorsReachAPreFilter:
    """Finding 3, second half: `$all` is not a supported pre-filter operator."""

    async def test_recall_uses_only_supported_operators(self) -> None:
        """TC-SEARCH-OP-001: multiple tags, which is where `$all` was used."""
        svc, col = _service()
        await svc.recall("u1", "q", tier=["ltm"], memory_type="f", tags=["a", "b"])
        used = _filter_operators(_recall_filter(col))
        unsupported = used - SUPPORTED_PREFILTER_OPERATORS
        assert not unsupported, (
            f"unsupported $vectorSearch pre-filter operators {sorted(unsupported)}; "
            f"the engine ignores these and the branch returns nothing"
        )

    async def test_hybrid_search_uses_only_supported_operators(self) -> None:
        # TC-SEARCH-OP-002
        svc, col = _service()
        await svc.hybrid_search("u1", "q", tier=["stm"], memory_type="f", tags=["a", "b"])
        used = _filter_operators(_fusion_branches(col)[0])
        assert not used - SUPPORTED_PREFILTER_OPERATORS

    def test_all_of_semantics_are_preserved(self) -> None:
        """TC-SEARCH-OP-003: the replacement must still mean *all* the tags.

        An `$or` — or a single `{"tags": {"$in": tags}}` — would be supported and
        would silently widen the filter to any-of, which is the failure a
        mechanical operator swap invites.
        """
        built = tag_filter(["a", "b", "c"])
        assert built == {"$and": [{"tags": "a"}, {"tags": "b"}, {"tags": "c"}]}
        assert "$or" not in str(built) and "$in" not in str(built)

    def test_a_single_tag_needs_no_wrapper(self) -> None:
        # TC-SEARCH-OP-004: the common case stays readable in a log.
        assert tag_filter(["a"]) == {"tags": "a"}


class TestBothFusionBranchesCarryTheSameRestriction:
    """Finding 3, third half: a one-branch filter is not a filter.

    `$rankFusion` merges the two ranked lists. Whatever the full-text branch
    matches enters the result set regardless of what the vector pre-filter said.
    """

    async def test_memory_type_reaches_the_full_text_branch(self) -> None:
        """TC-SEARCH-FUSE-001: the defect — scoped search, unscoped results."""
        svc, col = _service()
        await svc.hybrid_search("u1", "q", memory_type="factual")
        vs_filter, fts_clauses = _fusion_branches(col)
        assert vs_filter["memory_type"] == "factual"
        assert {"equals": {"path": "memory_type", "value": "factual"}} in fts_clauses, (
            "the full-text branch ignores memory_type, so fusion returns other types"
        )

    async def test_tags_reach_the_full_text_branch(self) -> None:
        # TC-SEARCH-FUSE-002
        svc, col = _service()
        await svc.hybrid_search("u1", "q", tags=["work", "urgent"])
        _, fts_clauses = _fusion_branches(col)
        for tag in ("work", "urgent"):
            assert {"equals": {"path": "tags", "value": tag}} in fts_clauses, (
                f"tag {tag!r} is missing from the full-text branch"
            )

    async def test_the_two_branches_restrict_the_same_fields(self) -> None:
        """TC-SEARCH-FUSE-003: the invariant, rather than a field checklist.

        `deleted_at`/`is_deleted` are the one legitimate asymmetry: the same
        soft-delete condition spelled for each branch's own index. Every other
        field must appear in both, so a filter added to one branch in future
        fails here.
        """
        svc, col = _service()
        await svc.hybrid_search(
            "u1", "q", tier=["stm", "ltm"], memory_type="factual", tags=["a", "b"]
        )
        vs_filter, fts_clauses = _fusion_branches(col)
        soft_delete = {"deleted_at", "is_deleted"}
        vector_side = _filter_paths(vs_filter) - soft_delete
        text_side = _clause_paths(fts_clauses) - soft_delete
        assert vector_side == text_side, (
            f"only the vector branch restricts {sorted(vector_side - text_side)}; "
            f"only the full-text branch restricts {sorted(text_side - vector_side)}"
        )

    async def test_both_branches_still_scope_by_tenant(self) -> None:
        """TC-SEARCH-FUSE-004: the filter that was never missing, kept asserted.

        Isolation was correct before this change and has to stay correct after
        it — this is the one clause whose absence from either branch would be a
        cross-tenant read rather than a wrong-looking result.
        """
        svc, col = _service()
        await svc.hybrid_search("u1", "q", memory_type="factual", tags=["a"])
        vs_filter, fts_clauses = _fusion_branches(col)
        assert vs_filter["user_id"] == "u1"
        assert {"equals": {"path": "user_id", "value": "u1"}} in fts_clauses


class TestFullTextFilterFieldsAreTokenMapped:
    """An `equals` filter needs `token`; `string` is analyzed and will not match."""

    async def test_every_full_text_filter_path_is_a_token_field(self) -> None:
        """TC-SEARCH-FTS-001: against the shipped mapping, not a copied list."""
        svc, col = _service()
        await svc.hybrid_search("u1", "q", memory_type="factual", tags=["a", "b"])
        used = _clause_paths(_fusion_branches(col)[1])
        declared = _declared_token_fields("memories_fts_index")
        assert used <= declared, (
            f"full-text filter paths not mapped as `token`: {sorted(used - declared)}; "
            f"an analyzed field cannot back an exact `equals`"
        )

    @pytest.mark.parametrize("field", ["memory_type", "tags"])
    def test_the_two_new_fields_are_tokens(self, field) -> None:
        # TC-SEARCH-FTS-002
        assert field in _declared_token_fields("memories_fts_index")

    def test_the_clause_helper_produces_one_equals_per_tag(self) -> None:
        """TC-SEARCH-FTS-003: `compound.filter` is an AND, so this is all-of."""
        assert tag_fts_clauses(["a", "b"]) == [
            {"equals": {"path": "tags", "value": "a"}},
            {"equals": {"path": "tags", "value": "b"}},
        ]


class TestEpisodicSearchHoldsTheSameContract:
    """The episodic tier already did this correctly; assert it stays that way."""

    def _service(self):
        from agent_memory.services.episodic import EpisodicService

        col = MagicMock()
        cursor = AsyncMock()
        cursor.to_list = AsyncMock(return_value=[])
        col.aggregate = AsyncMock(return_value=cursor)
        providers = MagicMock()
        providers.embedding = AsyncMock()
        providers.embedding.generate_embedding = AsyncMock(return_value=[0.1] * 4)
        return (
            EpisodicService(col, _config(), providers, worker=MagicMock()),
            col,
        )

    async def test_episodic_filters_only_on_declared_paths(self) -> None:
        # TC-SEARCH-EP-001
        svc, col = self._service()
        await svc.search("u1", "q", thread_id="t1", agent_name="planner")
        used = _filter_paths(_fusion_branches(col)[0])
        declared = _declared_filter_paths("episodes_vector_index")
        assert used <= declared, f"undeclared: {sorted(used - declared)}"

    async def test_episodic_branches_restrict_the_same_fields(self) -> None:
        # TC-SEARCH-EP-002: same invariant, other tier.
        svc, col = self._service()
        await svc.search("u1", "q", thread_id="t1", agent_name="planner")
        vs_filter, fts_clauses = _fusion_branches(col)
        assert _filter_paths(vs_filter) == _clause_paths(fts_clauses)
