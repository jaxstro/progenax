"""ICViz plot style — the StarViz theme ported to progenax (ADR-0021 sibling).

Ported from ``startrax/laboratory/starviz/starviz/style.py`` so figures across
the jaxstro ecosystem share ONE visual identity (Anna's methods-paper style:
seaborn ``paper``/``ticks``, the 8-color custom palette, trimmed despine).

Export contract (Anna, 2026-07-11): every figure renders to THREE formats at
one stem — PDF (paper-ready vector, gitignored), PNG (350 dpi raster,
gitignored), WebP (committed; the format the MyST site embeds).
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

PALETTE = [
    "#355C7D",  # slate blue
    "#6C5B7B",  # muted purple
    "#2A9D8F",  # teal
    "#E9C46A",  # gold
    "#F4A261",  # light orange
    "#E76F51",  # coral
    "#457B9D",  # steel blue
    "#8AB17D",  # sage
]
NEGATIVE = "#8E5A7F"
POSITIVE = "#2A9D8F"
NEUTRAL = "#3A3A3A"


@dataclass(frozen=True)
class ExportSpec:
    """Figure export knobs (mirrors StarViz's ExportSpec)."""

    dpi: int = 350
    webp_quality: int = 92
    bbox_inches: str = "tight"
    pad_inches: float = 0.02
    facecolor: str = "white"


def setup_style(font_scale: float = 0.95) -> list[str]:
    """Apply the shared seaborn theme; returns the palette for convenience.

    ``font_scale`` is slightly larger than StarViz's 0.86 default because the
    website renders figures wider than the compact proposal layout StarViz
    targets; pass 0.86 for paper-column-width exports.
    """
    sns.set_theme(
        context="paper",
        style="ticks",
        palette=PALETTE,
        font_scale=font_scale,
        rc={
            "figure.dpi": 150,
            "savefig.dpi": 350,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#2B2B2B",
            "axes.linewidth": 0.75,
            "axes.titleweight": "normal",
            "axes.labelcolor": "#252525",
            "xtick.color": "#303030",
            "ytick.color": "#303030",
            "grid.color": "#EAEAEA",
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.35,
            "patch.linewidth": 0.35,
            "mathtext.fontset": "dejavuserif",
        },
    )
    return list(PALETTE)


def polish_axes(
    ax: plt.Axes, *, grid_axis: Literal["x", "y", "both"] | None = None
) -> None:
    """StarViz's axis finish: optional feather grid, short ticks, trimmed spines."""
    if grid_axis is None:
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color="#EAEAEA", linewidth=0.45)
    ax.tick_params(length=3.0, width=0.7, pad=1.5)
    sns.despine(ax=ax, trim=False)


def save_figure_formats(
    fig: plt.Figure,
    output_stem: Path,
    *,
    spec: ExportSpec | None = None,
) -> tuple[Path, ...]:
    """Export one live figure to PDF + PNG + WebP at ``output_stem``.

    WebP goes through a rendered PNG buffer + Pillow (quality=92, method=6):
    deterministic and independent of matplotlib's own webp support. Closes the
    figure when done.
    """
    export = spec or ExportSpec()
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    common = dict(
        bbox_inches=export.bbox_inches,
        pad_inches=export.pad_inches,
        facecolor=export.facecolor,
    )
    paths: list[Path] = []

    pdf = output_stem.with_suffix(".pdf")
    fig.savefig(pdf, **common)
    paths.append(pdf)

    png = output_stem.with_suffix(".png")
    fig.savefig(png, dpi=export.dpi, **common)
    paths.append(png)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=export.dpi, **common)
    buf.seek(0)
    webp = output_stem.with_suffix(".webp")
    Image.open(buf).convert("RGB").save(
        webp, "WEBP", quality=export.webp_quality, method=6
    )
    paths.append(webp)

    plt.close(fig)
    return tuple(paths)
