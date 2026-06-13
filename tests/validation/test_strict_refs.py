"""Strict-mode guard: reference caches must exist when PROGENAX_STRICT_REFS=1."""
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIMEPY_CACHE = REPO_ROOT / "validation" / "data" / "limepy_reference"


@pytest.mark.skipif(
    os.environ.get("PROGENAX_STRICT_REFS") != "1",
    reason="strict reference mode not requested",
)
def test_limepy_reference_cache_present():
    """In strict (nightly/release) mode the committed LIMEPY cache must exist.

    Guards against a checkout/packaging mistake silently disabling the
    reference-parity suite (audit T4/H2).
    """
    npz_files = sorted(LIMEPY_CACHE.glob("*.npz"))
    assert npz_files, f"no LIMEPY reference .npz under {LIMEPY_CACHE}"
