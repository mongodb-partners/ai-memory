"""The single source of truth for the package version.

Read from installed distribution metadata so ``pyproject.toml`` stays the only
place the number is written. Falls back to a literal when the package is being
run from a source tree that was never installed (a plain ``git clone`` + ``PYTHONPATH``),
where ``importlib.metadata`` has nothing to find.
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

_FALLBACK = "4.1.0"

try:
    __version__: str = _dist_version("agent-memory")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = _FALLBACK

__all__ = ["__version__"]
