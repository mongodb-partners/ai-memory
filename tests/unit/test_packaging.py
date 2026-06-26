"""Tests for pyproject.toml, Dockerfile, docker-compose.yml, and .dockerignore."""

import pathlib
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# pyproject.toml
# ---------------------------------------------------------------------------

class TestPyprojectToml:
    """Verify pyproject.toml is valid and correctly configured."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = ROOT / "pyproject.toml"
        assert path.exists(), "pyproject.toml not found at project root"
        with open(path, "rb") as f:
            self.data = tomllib.load(f)

    def test_build_backend_is_hatchling(self):
        assert self.data["build-system"]["build-backend"] == "hatchling.build"

    def test_project_name(self):
        assert self.data["project"]["name"] == "memory-mcp"

    def test_requires_python_at_least_311(self):
        assert ">=3.11" in self.data["project"]["requires-python"]

    def test_runtime_dependencies_present(self):
        deps = self.data["project"]["dependencies"]
        dep_names = [d.split(">")[0].split("=")[0].split("<")[0] for d in deps]
        for required in [
            "fastmcp",
            "pymongo",
            "boto3",
            "pydantic",
            "pydantic-settings",
            "tavily-python",
            "python-dotenv",
        ]:
            assert required in dep_names, f"Missing dependency: {required}"

    def test_dev_dependencies_present(self):
        dev = self.data["project"]["optional-dependencies"]["dev"]
        dep_names = [d.split(">")[0].split("=")[0] for d in dev]
        assert "pytest" in dep_names
        assert "pytest-asyncio" in dep_names

    def test_console_script_entry_point(self):
        scripts = self.data["project"]["scripts"]
        assert "memory-mcp" in scripts
        assert scripts["memory-mcp"] == "agent_memory.__main__:main"

    def test_force_include_maps_all_packages(self):
        force = self.data["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
        expected_targets = [
            "agent_memory/__init__.py",
            "agent_memory/__main__.py",
            "agent_memory/server.py",
            "agent_memory/core",
            "agent_memory/providers",
            "agent_memory/services",
            "agent_memory/tools",
        ]
        for target in expected_targets:
            assert target in force.values(), f"Missing force-include target: {target}"


# ---------------------------------------------------------------------------
# Dockerfile
# ---------------------------------------------------------------------------

class TestDockerfile:
    """Verify Dockerfile structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = ROOT / "Dockerfile"
        assert path.exists(), "Dockerfile not found"
        self.lines = path.read_text().splitlines()
        self.text = path.read_text()

    def test_base_image_is_python_slim(self):
        assert any("python:3.11-slim" in line for line in self.lines)

    def test_workdir_is_app(self):
        assert any("WORKDIR /app" in line for line in self.lines)

    def test_exposes_port_8000(self):
        assert any("EXPOSE 8000" in line for line in self.lines)

    def test_cmd_uses_uv_run(self):
        assert any('CMD ["uv", "run", "memory-mcp"]' in line for line in self.lines)

    def test_uv_sync_frozen(self):
        assert "uv sync --frozen" in self.text

    def test_installs_uv_binary(self):
        assert any("ghcr.io/astral-sh/uv" in line for line in self.lines)

    def test_copies_pyproject_toml(self):
        assert any("COPY pyproject.toml" in line for line in self.lines)

    def test_copies_uv_lock(self):
        assert any("uv.lock" in line for line in self.lines)

    def test_copies_source_directories(self):
        for directory in ["core/", "providers/", "services/", "tools/"]:
            assert any(
                f"COPY {directory}" in line for line in self.lines
            ), f"Missing COPY for {directory}"


# ---------------------------------------------------------------------------
# docker-compose.yml
# ---------------------------------------------------------------------------

class TestDockerCompose:
    """Verify docker-compose.yml structure."""

    @pytest.fixture(autouse=True)
    def _load(self):
        try:
            import yaml
        except ImportError:
            pytest.skip("pyyaml not installed")
        path = ROOT / "docker-compose.yml"
        assert path.exists(), "docker-compose.yml not found"
        with open(path) as f:
            self.data = yaml.safe_load(f)

    def test_service_name(self):
        assert "memory-mcp" in self.data["services"]

    def test_build_context_is_current_dir(self):
        assert self.data["services"]["memory-mcp"]["build"] == "."

    def test_env_file(self):
        assert self.data["services"]["memory-mcp"]["env_file"] == ".env"

    def test_restart_policy(self):
        assert self.data["services"]["memory-mcp"]["restart"] == "unless-stopped"

    def test_memory_limit(self):
        limits = self.data["services"]["memory-mcp"]["deploy"]["resources"]["limits"]
        assert limits["memory"] == "2G"

    def test_health_check_present(self):
        hc = self.data["services"]["memory-mcp"]["healthcheck"]
        assert hc["interval"] == "30s"
        assert hc["timeout"] == "10s"
        assert hc["retries"] == 3
        assert hc["start_period"] == "60s"

    def test_health_check_targets_port_8000(self):
        hc = self.data["services"]["memory-mcp"]["healthcheck"]
        test_cmd = " ".join(hc["test"]) if isinstance(hc["test"], list) else hc["test"]
        assert "8000" in test_cmd

    def test_health_check_uses_get_health(self):
        hc = self.data["services"]["memory-mcp"]["healthcheck"]
        test_cmd = " ".join(hc["test"]) if isinstance(hc["test"], list) else hc["test"]
        assert "GET" in test_cmd
        assert "/health" in test_cmd
        assert "status == 200" in test_cmd

    def test_network_is_standalone_bridge(self):
        nets = self.data["networks"]
        assert "memory-mcp-network" in nets
        assert nets["memory-mcp-network"]["driver"] == "bridge"

    def test_service_joins_network(self):
        nets = self.data["services"]["memory-mcp"]["networks"]
        assert "memory-mcp-network" in nets


# ---------------------------------------------------------------------------
# .dockerignore
# ---------------------------------------------------------------------------

class TestDockerignore:
    """Verify .dockerignore excludes sensitive/unnecessary files."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = ROOT / ".dockerignore"
        assert path.exists(), ".dockerignore not found"
        self.lines = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

    def test_excludes_env_files(self):
        assert ".env" in self.lines
        assert ".env.*" in self.lines

    def test_includes_env_example(self):
        assert "!.env.example" in self.lines

    def test_excludes_git(self):
        assert ".git" in self.lines

    def test_excludes_pycache(self):
        assert "__pycache__" in self.lines

    def test_excludes_tests(self):
        assert "tests/" in self.lines

    def test_excludes_venv(self):
        assert ".venv" in self.lines


# ---------------------------------------------------------------------------
# .env.example
# ---------------------------------------------------------------------------

class TestEnvExample:
    """Verify .env.example documents required variables."""

    @pytest.fixture(autouse=True)
    def _load(self):
        path = ROOT / ".env.example"
        assert path.exists(), ".env.example not found"
        self.text = path.read_text()

    def test_contains_mongodb_connection_string(self):
        assert "MONGODB_CONNECTION_STRING" in self.text

    def test_contains_aws_credentials(self):
        assert "AWS_ACCESS_KEY_ID" in self.text
        assert "AWS_SECRET_ACCESS_KEY" in self.text
        assert "AWS_REGION" in self.text
