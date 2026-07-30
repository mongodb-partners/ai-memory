"""Tests for the episodic projection helpers. REQ-E-090..095."""

from types import SimpleNamespace

from agent_memory.core.projection import (
    build_search_text,
    coerce_content,
    is_final_step,
    project_files,
    project_messages,
    project_todos,
    truncate,
)


def _msg(**kwargs):
    """An attribute-bearing message, the shape most agent frameworks use."""
    defaults = {
        "type": "ai",
        "content": "",
        "tool_calls": [],
        "tool_call_id": None,
        "usage_metadata": None,
        "additional_kwargs": {},
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


class TestCoerceContent:
    """REQ-E-091: content collapses to a string."""

    def test_string_passes_through(self):
        # TC-EP-PROJ-001
        assert coerce_content(_msg(content="hello")) == "hello"

    def test_none_becomes_empty_string(self):
        # TC-EP-PROJ-002
        assert coerce_content(_msg(content=None)) == ""

    def test_text_blocks_are_joined(self):
        # TC-EP-PROJ-003
        content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
        assert coerce_content(_msg(content=content)) == "ab"

    def test_non_text_blocks_are_dropped(self):
        # TC-EP-PROJ-004: a tool_use block carries no readable text.
        content = [
            {"type": "text", "text": "keep"},
            {"type": "tool_use", "name": "search", "input": {}},
            {"type": "image", "source": {}},
        ]
        assert coerce_content(_msg(content=content)) == "keep"

    def test_bare_strings_in_a_list_are_kept(self):
        # TC-EP-PROJ-005
        assert coerce_content(_msg(content=["a", {"type": "text", "text": "b"}])) == "ab"

    def test_unexpected_type_is_stringified(self):
        # TC-EP-PROJ-006
        assert coerce_content(_msg(content=42)) == "42"

    def test_dict_message_is_supported(self):
        # TC-EP-PROJ-007: dicts are the shape messages arrive in over HTTP.
        assert coerce_content({"type": "ai", "content": "from a dict"}) == "from a dict"


class TestTruncate:
    """REQ-E-092: truncation is visible and reports the original size."""

    def test_short_text_is_untouched(self):
        # TC-EP-PROJ-010
        assert truncate("abc", 10) == "abc"

    def test_long_text_gets_a_marker(self):
        # TC-EP-PROJ-011
        out = truncate("x" * 100, 10)
        assert out.startswith("x" * 10)
        assert "original_size=100" in out

    def test_zero_cap_disables_truncation(self):
        # TC-EP-PROJ-012
        assert truncate("x" * 100, 0) == "x" * 100

    def test_negative_cap_disables_truncation(self):
        # TC-EP-PROJ-013
        assert truncate("x" * 100, -1) == "x" * 100


class TestProjectMessages:
    """REQ-E-090: exactly seven keys, in order."""

    def test_seven_keys_in_order(self):
        # TC-EP-PROJ-020: key order is part of the stored document contract.
        out = project_messages([_msg(content="hi")], cap=1000)
        assert list(out[0].keys()) == [
            "type",
            "content",
            "tool_calls",
            "tool_call_id",
            "usage",
            "model_id",
            "finish_reason",
        ]

    def test_type_defaults_to_ai(self):
        # TC-EP-PROJ-021
        out = project_messages([SimpleNamespace(content="hi")], cap=1000)
        assert out[0]["type"] == "ai"

    def test_provider_metadata_is_lifted(self):
        # TC-EP-PROJ-022: model id and stop reason live on the envelope.
        message = _msg(
            content="hi",
            usage_metadata={"input_tokens": 5, "output_tokens": 2},
            additional_kwargs={"model_id": "claude-x", "stop_reason": "end_turn"},
        )
        out = project_messages([message], cap=1000)[0]
        assert out["usage"] == {"input_tokens": 5, "output_tokens": 2}
        assert out["model_id"] == "claude-x"
        assert out["finish_reason"] == "end_turn"

    def test_content_is_truncated(self):
        # TC-EP-PROJ-023
        out = project_messages([_msg(content="x" * 50)], cap=10)
        assert "original_size=50" in out[0]["content"]

    def test_dict_messages_project_fully(self):
        # TC-EP-PROJ-024: the getattr-only version silently produced empty docs.
        raw = [
            {
                "type": "human",
                "content": "what is the weather",
                "tool_calls": [{"name": "get_weather", "args": {}}],
                "tool_call_id": "call_1",
                "usage_metadata": {"input_tokens": 3},
                "additional_kwargs": {"model_id": "m", "stop_reason": "tool_use"},
            }
        ]
        out = project_messages(raw, cap=1000)[0]
        assert out["type"] == "human"
        assert out["content"] == "what is the weather"
        assert out["tool_calls"] == [{"name": "get_weather", "args": {}}]
        assert out["tool_call_id"] == "call_1"
        assert out["usage"] == {"input_tokens": 3}
        assert out["model_id"] == "m"
        assert out["finish_reason"] == "tool_use"

    def test_malformed_additional_kwargs_does_not_raise(self):
        # TC-EP-PROJ-025
        out = project_messages([_msg(additional_kwargs="not a mapping")], cap=1000)[0]
        assert out["model_id"] is None
        assert out["finish_reason"] is None

    def test_empty_input_yields_empty_list(self):
        # TC-EP-PROJ-026
        assert project_messages([], cap=1000) == []


class TestProjectTodos:
    """REQ-E-093: three keys, status clamped."""

    def test_valid_todo_projects(self):
        # TC-EP-PROJ-030
        out = project_todos([{"id": "1", "content": "do it", "status": "completed"}])
        assert out == [{"id": "1", "content": "do it", "status": "completed"}]

    def test_unknown_status_clamps_to_pending(self):
        # TC-EP-PROJ-031: a malformed todo must not cost the whole logged turn.
        out = project_todos([{"id": "1", "content": "x", "status": "wat"}])
        assert out[0]["status"] == "pending"

    def test_text_is_accepted_as_a_content_alias(self):
        # TC-EP-PROJ-032
        out = project_todos([{"id": "1", "text": "aliased", "status": "pending"}])
        assert out[0]["content"] == "aliased"

    def test_non_list_input_yields_empty_list(self):
        # TC-EP-PROJ-033
        assert project_todos("not a list") == []
        assert project_todos(None) == []

    def test_non_mapping_entries_are_skipped(self):
        # TC-EP-PROJ-034
        out = project_todos(["nope", {"id": "1", "content": "y", "status": "pending"}])
        assert len(out) == 1


class TestProjectFiles:
    """REQ-E-094: files_touched, last write per path, sorted."""

    def _write(self, name, path, **args):
        return _msg(tool_calls=[{"name": name, "args": {"file_path": path, **args}}])

    def test_write_tool_produces_an_entry(self):
        # TC-EP-PROJ-040
        out = project_files([self._write("write_file", "a.md", content="hello")])
        assert out == [
            {"path": "a.md", "size": 5, "content_hash": None, "op": "write"}
        ]

    def test_edit_tool_is_labelled_edit(self):
        # TC-EP-PROJ-041
        out = project_files([self._write("edit_file", "a.md", new_string="hi")])
        assert out[0]["op"] == "edit"
        assert out[0]["size"] == 2

    def test_last_write_per_path_wins(self):
        # TC-EP-PROJ-042: write then edit reports one entry, op=edit.
        messages = [
            self._write("write_file", "a.md", content="hello"),
            self._write("edit_file", "a.md", new_string="hi"),
        ]
        out = project_files(messages)
        assert len(out) == 1
        assert out[0]["op"] == "edit"

    def test_results_are_sorted_by_path(self):
        # TC-EP-PROJ-043: so two logs of the same turn compare equal.
        messages = [
            self._write("write_file", "z.md", content="z"),
            self._write("write_file", "a.md", content="a"),
        ]
        assert [entry["path"] for entry in project_files(messages)] == ["a.md", "z.md"]

    def test_read_only_tools_are_ignored(self):
        # TC-EP-PROJ-044
        assert project_files([self._write("read_file", "a.md")]) == []

    def test_non_ai_messages_are_ignored(self):
        # TC-EP-PROJ-045: only the assistant issues tool calls.
        message = _msg(
            type="human",
            tool_calls=[{"name": "write_file", "args": {"file_path": "a.md"}}],
        )
        assert project_files([message]) == []

    def test_path_alias_is_accepted(self):
        # TC-EP-PROJ-046
        message = _msg(tool_calls=[{"name": "write_file", "args": {"path": "a.md"}}])
        assert project_files([message])[0]["path"] == "a.md"

    def test_missing_path_is_skipped(self):
        # TC-EP-PROJ-047
        message = _msg(tool_calls=[{"name": "write_file", "args": {}}])
        assert project_files([message]) == []

    def test_custom_tool_names_are_honored(self):
        # TC-EP-PROJ-048: op is derived from set membership, not a hardcoded name.
        message = _msg(
            tool_calls=[{"name": "save_doc", "args": {"file_path": "a.md", "content": "x"}}]
        )
        out = project_files(
            [message],
            fs_write_tools=frozenset({"save_doc"}),
            fs_create_tools=frozenset({"save_doc"}),
        )
        assert out[0]["op"] == "write"

    def test_dict_messages_are_supported(self):
        # TC-EP-PROJ-049
        raw = [
            {
                "type": "ai",
                "tool_calls": [
                    {"name": "write_file", "args": {"file_path": "a.md", "content": "xy"}}
                ],
            }
        ]
        assert project_files(raw) == [
            {"path": "a.md", "size": 2, "content_hash": None, "op": "write"}
        ]


class TestIsFinalStep:
    """REQ-E-095: only a turn that ended in an answer is a final step."""

    def test_ai_without_tool_calls_is_final(self):
        # TC-EP-PROJ-050
        assert is_final_step([{"type": "ai", "content": "done", "tool_calls": []}])

    def test_ai_with_tool_calls_is_not_final(self):
        # TC-EP-PROJ-051
        proj = [{"type": "ai", "content": "", "tool_calls": [{"name": "search"}]}]
        assert is_final_step(proj) is False

    def test_no_ai_message_is_not_final(self):
        # TC-EP-PROJ-052
        assert is_final_step([{"type": "human", "content": "hi"}]) is False

    def test_the_last_ai_message_decides(self):
        # TC-EP-PROJ-053
        proj = [
            {"type": "ai", "content": "", "tool_calls": [{"name": "search"}]},
            {"type": "tool", "content": "result"},
            {"type": "ai", "content": "answer", "tool_calls": []},
        ]
        assert is_final_step(proj) is True


class TestBuildSearchText:
    """REQ-E-095: question plus answer, nothing in between."""

    def test_first_human_and_last_ai_are_joined(self):
        # TC-EP-PROJ-060
        proj = [
            {"type": "human", "content": "Q"},
            {"type": "ai", "content": "", "tool_calls": [{"name": "s"}]},
            {"type": "ai", "content": "A"},
        ]
        assert build_search_text(proj, cap=1000) == "Q\n\nA"

    def test_missing_human_yields_empty_string(self):
        # TC-EP-PROJ-061: an empty return suppresses embedding and search_text.
        assert build_search_text([{"type": "ai", "content": "A"}], cap=1000) == ""

    def test_missing_ai_yields_empty_string(self):
        # TC-EP-PROJ-062
        assert build_search_text([{"type": "human", "content": "Q"}], cap=1000) == ""

    def test_cap_truncates_without_a_marker(self):
        # TC-EP-PROJ-063: a marker inside embedded text would pollute the vector.
        proj = [{"type": "human", "content": "x" * 50}, {"type": "ai", "content": "y"}]
        out = build_search_text(proj, cap=10)
        assert out == "x" * 10
        assert "truncated" not in out

    def test_zero_cap_disables_truncation(self):
        # TC-EP-PROJ-064
        proj = [{"type": "human", "content": "x" * 50}, {"type": "ai", "content": "y"}]
        assert len(build_search_text(proj, cap=0)) == 53
