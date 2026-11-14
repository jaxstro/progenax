"""Progenax equation data; loading and symbolic machinery live in jaxstro.registry."""

from pathlib import Path


def registry_root() -> Path:
    """Return the packaged source-bundle root without importing SymPy."""
    return Path(__file__).resolve().parent
