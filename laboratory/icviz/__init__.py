"""ICViz — progenax's publication-quality figure laboratory (StarViz's sibling).

Modular seaborn+matplotlib figure library for the theory-page / methods-paper
figure suite: one module per model family, one function per figure, declared
in ``registry.py``, rendered by the CLI to PDF + PNG (gitignored, paper-local)
and WebP (committed; embedded by the MyST site).
"""

from .registry import FIGURES
from .specs import PLOTS_DIR, SITE_FIGURE_DIR, FigureSpec
from .style import PALETTE, save_figure_formats, setup_style

__all__ = [
    "FIGURES",
    "PLOTS_DIR",
    "SITE_FIGURE_DIR",
    "FigureSpec",
    "PALETTE",
    "save_figure_formats",
    "setup_style",
]
