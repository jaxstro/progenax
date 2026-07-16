"""P0 Task 0.1 — the experimental gravoturb package must be importable.

This is the RED test that drives standing up the standalone package skeleton at
``src/experimental/gravoturb/`` (importable as ``gravoturb``, NOT shipped in
the progenax wheel). See the approved clean-room TDD plan, P0.
"""

import pytest

pytestmark = pytest.mark.experimental


def test_package_imports():
    """The standalone experimental package imports and is self-describing."""
    import gravoturb

    assert gravoturb.__doc__, "gravoturb must have a module docstring"


def test_subpackages_import():
    """The four documented layers import as subpackages."""
