"""Packaging contract for agent-memory v4. REQ-E-081."""

import pathlib
import tomllib
from typing import ClassVar

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class TestPyprojectToml:
    @pytest.fixture(autouse=True)
    def _load(self):
        path = ROOT / "pyproject.toml"
        assert path.exists(), "pyproject.toml not found at project root"
        with open(path, "rb") as f:
            self.data = tomllib.load(f)

    def test_build_backend_is_hatchling(self):
        assert self.data["build-system"]["build-backend"] == "hatchling.build"

    def test_project_name_and_version(self):
        assert self.data["project"]["name"] == "agent-memory"
        assert self.data["project"]["version"].startswith("4.")

    def test_requires_python_at_least_311(self):
        assert ">=3.11" in self.data["project"]["requires-python"]

    def test_runtime_dependencies_present(self):
        deps = self.data["project"]["dependencies"]
        names = [d.split(">")[0].split("=")[0].split("<")[0].split("[")[0] for d in deps]
        for required in ["fastmcp", "pymongo", "boto3", "pydantic", "pydantic-settings"]:
            assert required in names, f"Missing dependency: {required}"

    def test_optional_dependency_groups(self):
        opt = self.data["project"]["optional-dependencies"]
        for group in ["openai", "anthropic", "rest", "all", "dev"]:
            assert group in opt, f"Missing optional-dependency group: {group}"

    def test_rest_extra_pulls_fastapi(self):
        rest = self.data["project"]["optional-dependencies"]["rest"]
        names = [d.split(">")[0].split("=")[0] for d in rest]
        assert "fastapi" in names

    def test_console_script_entry_point(self):
        scripts = self.data["project"]["scripts"]
        assert scripts.get("agent-memory") == "agent_memory.__main__:main"

    def test_wheel_packages_agent_memory(self):
        wheel = self.data["tool"]["hatch"]["build"]["targets"]["wheel"]
        assert "agent_memory" in wheel["packages"]

    def test_version_is_at_least_4_2(self):
        """The scorer seam is additive, so a minor bump."""
        major, minor = self.data["project"]["version"].split(".")[:2]
        assert (int(major), int(minor)) >= (4, 2)

    def test_training_extra_exists(self):
        opt = self.data["project"]["optional-dependencies"]
        assert "training" in opt
        names = [d.split(">")[0].split("=")[0].split("[")[0] for d in opt["training"]]
        for required in ["scikit-learn", "numpy"]:
            assert required in names

    def test_training_extra_is_not_in_all(self):
        """`all` is 'every provider', not 'every dependency'. scikit-learn +
        pandas + datasets is ~200MB of transitive weight for a feature that runs
        offline and is never imported by the library."""
        all_names = [
            d.split(">")[0].split("=")[0].split("[")[0]
            for d in self.data["project"]["optional-dependencies"]["all"]
        ]
        for excluded in ["scikit-learn", "numpy", "pandas", "datasets"]:
            assert excluded not in all_names

    def test_runtime_dependencies_exclude_the_scientific_stack(self):
        """The load-bearing constraint of the whole design. If numpy lands in
        runtime deps, the pure-Python scorer stopped being necessary and someone
        should have said so out loud."""
        names = [
            d.split(">")[0].split("=")[0].split("<")[0].split("[")[0]
            for d in self.data["project"]["dependencies"]
        ]
        for excluded in ["numpy", "scipy", "scikit-learn", "pandas", "torch"]:
            assert excluded not in names

    def test_sdist_includes_the_package(self):
        """`/agent_memory` covers data/importance/. Asserted so a switch to a
        narrower include list has to notice."""
        include = self.data["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        assert "/agent_memory" in include


class TestImportanceArtifactsShip:
    """A wheel without the artifacts makes IMPORTANCE_SCORER=local a startup
    failure for every installed user — and it is invisible from a source checkout,
    where the files are on disk regardless. Hence a real build.
    """

    # Only `lexical` ships: it is the one artifact that is trained, and
    # `_BUNDLED_ARTIFACTS` is empty so every deployment selects it.
    EXPECTED: ClassVar[set[str]] = {"lexical.json"}

    def test_artifacts_exist_in_the_source_tree(self):
        found = {
            p.name for p in (ROOT / "agent_memory/data/importance").glob("*.json")
        }
        assert self.EXPECTED <= found

    def test_artifacts_are_in_the_built_wheel(self, tmp_path):
        import subprocess
        import zipfile

        result = subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if result.returncode != 0:
            pytest.skip(f"wheel build unavailable: {result.stderr[-400:]}")

        wheels = list(tmp_path.glob("*.whl"))
        assert wheels, "no wheel produced"
        with zipfile.ZipFile(wheels[0]) as zf:
            names = {
                pathlib.PurePosixPath(n).name
                for n in zf.namelist()
                if "data/importance" in n
            }
        assert self.EXPECTED <= names, f"artifacts missing from wheel: {names}"


class TestLibraryStaysDependencyFree:
    """REQ-E-161. Grep rather than import-check: an `import numpy` inside a
    function body would not fail an import of the module, and the scorer is
    exactly where someone would be tempted to add one."""

    FORBIDDEN = ("numpy", "scipy", "sklearn", "pandas", "torch")

    def test_no_scientific_imports_under_agent_memory(self):
        import re

        offenders = []
        pattern = re.compile(
            r"^\s*(?:import|from)\s+(" + "|".join(self.FORBIDDEN) + r")\b", re.M
        )
        for path in (ROOT / "agent_memory").rglob("*.py"):
            for match in pattern.finditer(path.read_text()):
                offenders.append(f"{path.relative_to(ROOT)}: {match.group(0).strip()}")
        assert not offenders, (
            "agent_memory must import no scientific stack — the local scorer is "
            "pure Python so that enabling it costs no install weight:\n"
            + "\n".join(offenders)
        )


class TestLintGateIsRunnable:
    """CI's lint step is `uv run --no-sync ruff check`, whole repo.

    Undeclared, that command resolved to whatever `ruff` happened to be on PATH:
    a developer's Homebrew build locally, and *nothing* on a clean runner, where
    `uv run ruff` exits 2 with "Failed to spawn: ruff". The job cannot pass, and it
    fails for a reason that has nothing to do with the diff under review — the
    failure mode a lint gate is supposed to be immune to.
    """

    @pytest.fixture(autouse=True)
    def _load(self):
        with open(ROOT / "pyproject.toml", "rb") as f:
            self.data = tomllib.load(f)

    def _dev_specs(self) -> list[str]:
        return self.data["project"]["optional-dependencies"]["dev"]

    def test_ruff_is_declared(self):
        """`uv sync --all-extras` must install the linter CI then invokes."""
        names = [
            s.split(">")[0].split("=")[0].split("<")[0].split("[")[0].strip().lower()
            for s in self._dev_specs()
        ]
        assert "ruff" in names, (
            "ruff is not declared in the dev extra, so CI's `uv run ruff check` has "
            "nothing to resolve and exits 2 rather than linting"
        )

    def test_ruff_is_pinned_to_a_bounded_range(self):
        """An unpinned linter makes CI's verdict depend on its release date. Ruff's
        default rule selection and `target-version` inference both change between
        releases: 0.9 reported this package clean where 0.16 flagged 28 findings,
        with no source change in between."""
        spec = next(s for s in self._dev_specs() if s.lower().startswith("ruff"))
        assert "<" in spec, (
            f"ruff spec {spec!r} has no upper bound: a new release can turn CI red "
            "without a code change"
        )

    def test_lint_rules_are_configured_explicitly(self):
        """Without `[tool.ruff.lint].select`, the enabled rules are whatever the
        installed ruff defaults to — which is version-dependent, so the pin above
        is only half the fix."""
        select = self.data["tool"]["ruff"]["lint"]["select"]
        assert select, "[tool.ruff.lint].select is empty"
        assert self.data["tool"]["ruff"]["target-version"] == "py311", (
            "target-version must be pinned; ruff otherwise infers it from "
            "requires-python, and the inference has changed between releases"
        )

    def test_ignores_stay_narrow(self):
        """`ignore` is the cheapest way to make a finding disappear.

        RUF002 is ignored deliberately: en dashes in *prose docstrings* are not a
        homoglyph attack, and "fixing" them would mean degrading the punctuation of
        every docstring. Its siblings RUF001 and RUF003 cover identifiers, string
        literals, and inline comments, where an ambiguous character genuinely can
        hide something — the config comment says so, and this asserts it, because
        the tempting response to two RUF001/RUF003 findings is to widen the ignore
        by two entries rather than fix them.
        """
        ignored = set(self.data["tool"]["ruff"]["lint"].get("ignore", []))
        for rule in ("RUF001", "RUF003"):
            assert rule not in ignored, (
                f"{rule} was added to `ignore`. Unlike RUF002 it applies to "
                "identifiers, literals, and inline comments, where an ambiguous "
                "unicode character is worth flagging — fix the finding instead"
            )
        per_file = self.data["tool"]["ruff"]["lint"].get("per-file-ignores", {})
        for pattern, rules in per_file.items():
            assert not pattern.startswith("tests"), (
                f"per-file-ignores exempts {pattern!r} from {rules}. The tests are "
                "what certify the library; exempting them re-opens the gap the "
                "whole-repo gate was widened to close"
            )

    def test_the_repo_actually_passes_the_gate(self):
        """The end-to-end assertion: run the gate.

        Deliberately the *project's* ruff (`.venv/bin/ruff`) rather than whatever
        `shutil.which` finds. An earlier draft of this test used PATH and failed
        against a developer's Homebrew ruff 0.9.10 on five UP038 findings — a rule
        newer ruff removed, because rewriting `isinstance(x, (A, B))` to `A | B` is
        slower at runtime. Asserting against an arbitrary PATH ruff would make this
        test demand changes the pinned linter does not want, which is the drift the
        pin exists to prevent.

        No path argument, matching CI. This used to pass `agent_memory/`, and the
        two together meant the 134 findings in tests/ and scripts/ were invisible
        to both the gate and the test asserting the gate passes.
        """
        import subprocess
        import sys

        ruff = pathlib.Path(sys.executable).parent / "ruff"
        if not ruff.exists():
            pytest.skip(
                "ruff not installed in this interpreter's environment; "
                "run `uv sync --all-extras`"
            )
        result = subprocess.run(
            [str(ruff), "check"],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"`ruff check` failed:\n{result.stdout}{result.stderr}"
        )

    def test_the_gate_covers_the_tests_too(self):
        """CI must lint the whole repo, not just the library.

        The gate ran `ruff check agent_memory/` for long enough that tests/,
        examples/, and scripts/ accumulated 134 findings, two of which were real
        defects in *assertions*: a `pytest.raises(match="bad.json")` whose
        unescaped `.` also matched "badXjson", and a stale import in
        test_transport.py. A test is what certifies the library, so an unlinted
        test suite is the one place a weakened check has nothing above it.

        Asserted against the workflow text because that is the thing CI runs; a
        passing local `ruff check` says nothing about the argument in the YAML.
        """
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        lint_lines = [
            line.strip() for line in ci.splitlines()
            if "ruff check" in line and not line.strip().startswith("#")
        ]
        assert lint_lines, "no `ruff check` invocation found in ci.yml"
        for line in lint_lines:
            # Everything after `ruff check` must be flags, not a path narrowing
            # the scope back to one directory.
            args = line.split("ruff check", 1)[1].split()
            paths = [a for a in args if not a.startswith("-")]
            assert not paths, (
                f"ci.yml lints only {paths} — the gate must cover the whole repo, "
                "or unlinted directories drift the way tests/ already did once"
            )

    def test_the_gate_actually_reaches_the_test_files(self):
        """Scope is a property of the run, not of the command line.

        The check above closes one door: a path argument in the workflow. This one
        closes the other — `extend-exclude = [..., "tests"]` in pyproject narrows
        the gate to exactly the same thing while `ci.yml` still reads
        `ruff check` with no arguments. Asked `--show-files` instead of trusting
        either the YAML or the config, because that is ruff answering what it will
        actually look at.
        """
        import subprocess
        import sys

        ruff = pathlib.Path(sys.executable).parent / "ruff"
        if not ruff.exists():
            pytest.skip("ruff not installed in this interpreter's environment")
        result = subprocess.run(
            [str(ruff), "check", "--show-files"],
            cwd=ROOT, capture_output=True, text=True,
        )
        walked = {
            pathlib.Path(line).resolve() for line in result.stdout.splitlines() if line
        }
        # This very file, plus one from each directory that was previously dark.
        for rel in (
            "tests/unit/test_packaging.py",
            "tests/unit/test_transport.py",
            "scripts/train_importance.py",
            "examples/memory-ui/server/cache_key.py",
        ):
            assert (ROOT / rel).resolve() in walked, (
                f"ruff does not lint {rel} — the gate's scope has been narrowed by "
                "config even though ci.yml still passes no path"
            )


class TestDockerfile:
    @pytest.fixture(autouse=True)
    def _load(self):
        path = ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile not found"
        self.text = path.read_text()
        self.lines = self.text.splitlines()

    def test_base_image_is_python_slim(self):
        assert any("python:3.11-slim" in line for line in self.lines)

    def test_copies_agent_memory_package(self):
        assert any("COPY agent_memory/" in line for line in self.lines)

    def test_runs_console_script(self):
        assert "agent-memory" in self.text
