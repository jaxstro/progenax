"""ICViz figure specs — one declaration per figure, consumed by the registry/CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt

# Repo root = parents[2] of this file (laboratory/icviz/specs.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
# Master outputs (PDF + PNG + WebP) land here — gitignored, paper-local.
PLOTS_DIR = REPO_ROOT / "laboratory" / "icviz" / "plots"
# The CLI copies ONLY the WebP here for the MyST site to embed (committed).
SITE_FIGURE_DIR = REPO_ROOT / "docs" / "website" / "10-theory" / "figures"


@dataclass(frozen=True)
class FigureSpec:
    """A single ICViz figure.

    Attributes:
        name: CLI id (kebab-case; ``--only <name>``).
        builder: zero-argument callable returning a live matplotlib Figure
            (styling applied; the CLI owns saving/closing).
        stem: output filename stem under FIGURE_DIR (no suffix).
        page: website-root-relative theory page the WebP embeds into
            (documentation of intent; the CLI does not edit pages).
        seed: the fixed PRNG seed the builder uses (reproducibility contract —
            builders must not draw entropy from anywhere else).
        caption: one-line description (shows in ``--list``).
    """

    name: str
    builder: Callable[[], plt.Figure]
    stem: str
    page: str
    seed: int
    caption: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def output_stem(self) -> Path:
        return PLOTS_DIR / self.stem

    @property
    def site_webp(self) -> Path:
        return SITE_FIGURE_DIR / f"{self.stem}.webp"
