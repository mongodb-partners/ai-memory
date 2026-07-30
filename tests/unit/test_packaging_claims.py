"""The package's claims about itself must be true.

Two of them were not, and neither could fail visibly:

* **The version fallback went stale.** ``version.py`` carried ``4.1.0`` while
  ``pyproject.toml`` said ``4.2.0``. The fallback fires only in an uninstalled
  source tree — exactly where nobody checks — and the value it produces is served
  by ``/health`` and stamped on every audit record. A wrong version in an audit
  trail misattributes behaviour to a release that did not have it.

* **``Typing :: Typed`` was declared without a ``py.typed`` marker.** The
  classifier tells a consumer's type checker that annotations here are
  authoritative; without the marker, PEP 561 says to ignore them. So mypy in a
  downstream project silently treated this fully-annotated library as untyped, and
  the only symptom was type errors it failed to catch.

Both are the same kind of defect: a statement about the package that nothing
verifies, in a place where being wrong is invisible. This file is the check.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


class TestTheVersionIsWrittenOnce:
    def test_the_fallback_matches_pyproject(self, pyproject) -> None:
        """The one assertion that makes a second copy of a version safe to keep.

        `version.py` reads installed distribution metadata, so `pyproject.toml`
        is the single source — except in an uninstalled source tree, where the
        literal takes over. Keeping the literal is the right call (a bare `git
        clone` should not report `unknown`), but only if it is checked.
        """
        from agent_memory.version import _FALLBACK

        assert _FALLBACK == pyproject["project"]["version"], (
            f"version.py's _FALLBACK is {_FALLBACK!r} but pyproject.toml says "
            f"{pyproject['project']['version']!r}. An uninstalled source tree "
            "would report the stale one from /health and every audit record."
        )

    def test_the_exported_version_is_the_real_one(self, pyproject) -> None:
        """Installed in this environment, so metadata — not the fallback — wins.

        A mismatch here means the installed distribution is out of date with the
        working tree, which makes every other version assertion untrustworthy.
        """
        from agent_memory.version import __version__

        assert __version__ == pyproject["project"]["version"]

    def test_the_config_reports_it_too(self, pyproject) -> None:
        """`app_version` is what `/health` actually serves."""
        from agent_memory.config import MemoryConfig

        config = MemoryConfig(
            mongodb_connection_string="mongodb://localhost:27017", _env_file=None
        )
        assert config.app_version == pyproject["project"]["version"]


class TestTheTypedClaimIsBacked:
    def test_the_marker_exists(self) -> None:
        """PEP 561: without this file a consumer's type checker ignores every
        annotation in the package, whatever the classifier says."""
        assert (_ROOT / "agent_memory" / "py.typed").is_file(), (
            "pyproject declares 'Typing :: Typed' but agent_memory/py.typed is "
            "missing, so downstream type checkers ignore this library's types"
        )

    def test_the_marker_is_empty(self) -> None:
        """PEP 561 defines the file as a marker. Content in it is not read, and
        a comment someone adds later would be silently ignored rather than
        wrong — worth pinning so nobody expects it to do something."""
        assert (_ROOT / "agent_memory" / "py.typed").read_bytes() == b""

    def test_the_classifier_is_still_declared(self, pyproject) -> None:
        """The inverse drift: dropping the classifier while keeping the marker
        would be just as inconsistent, in the harder-to-notice direction."""
        assert "Typing :: Typed" in pyproject["project"]["classifiers"]

    def test_the_build_config_ships_the_marker(self, pyproject) -> None:
        """A marker in the repo but not in the built distribution fixes nothing —
        a consumer's type checker reads the *installed* package.

        This asserts the configuration that decides inclusion, not the output of
        a build: hatchling ships everything under a directory named in
        ``packages``, so ``agent_memory`` being listed is what puts ``py.typed``
        in the wheel. A build was run by hand to confirm it
        (``agent_memory-4.2.0-py3-none-any.whl`` contains
        ``agent_memory/py.typed``), but repeating that here would make a unit
        test invoke the build backend, and asserting against the *installed*
        distribution is not an option either — an editable install lists a
        ``.pth`` shim rather than package files, so the assertion would fail for
        a correctly configured project.

        What is left unguarded is a change to hatchling's package-data default.
        That is upstream behaviour, not a mistake this repo can make, and the
        sdist allow-list below is the same reasoning applied to the other target.
        """
        wheel = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"]
        assert "agent_memory" in wheel["packages"], (
            "py.typed only reaches consumers because hatchling ships package "
            "data for every directory in `packages`; agent_memory must be listed"
        )

    def test_the_sdist_allow_list_covers_the_package(self, pyproject) -> None:
        """The sdist uses an allow-list (deliberately — a live ``.env`` sits in
        this working tree and a missed exclude pattern would publish credentials
        irreversibly). The cost of that choice is that a new top-level path is
        omitted by default, so the package directory itself has to be asserted.
        """
        include = pyproject["tool"]["hatch"]["build"]["targets"]["sdist"]["include"]
        assert "/agent_memory" in include
