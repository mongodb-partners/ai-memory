"""The single source of truth for the package version.

Read from installed distribution metadata so ``pyproject.toml`` stays the only
place the number is written. Falls back to a literal when the package is being
run from a source tree that was never installed (a plain ``git clone`` + ``PYTHONPATH``),
where ``importlib.metadata`` has nothing to find.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version

# Must equal `version` in pyproject.toml. It went stale at 4.1.0 while the
# project moved to 4.2.0, and nothing failed: the fallback only fires in an
# uninstalled source tree, which is precisely where nobody checks the number —
# so `/health` and every audit record served a version that had not existed for
# a release. `tests/unit/test_packaging_claims.py` asserts the two agree, which
# is the only thing that makes a second copy of a version safe to keep.
_FALLBACK = "4.2.0"

try:
    __version__: str = _dist_version("agent-memory")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = _FALLBACK

__all__ = ["__version__"]
