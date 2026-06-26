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
