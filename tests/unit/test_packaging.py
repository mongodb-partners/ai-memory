"""Packaging contract for agent-memory v4. REQ-E-081."""

import pathlib
import tomllib

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
    EXPECTED = {"lexical.json"}

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
