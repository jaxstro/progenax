"""Shared publication-figure style for the progenax validation scripts.

Single source of truth for the look of every ``scripts/validate_*.py`` figure
(ApJ/AAS house style): serif text + Computer-Modern math, Okabe-Ito colourblind
palette, inward ticks, no in-figure titles (the paper / validation-page caption
carries the title), PNG (raster, for the website) + PDF (vector, for the paper).

Usage::

    from _plotstyle import OI, apply_pub_style, panel_label, save_fig
    apply_pub_style()
    ...
    save_fig(fig, output_dir, "king_concentration")
"""
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe qualitative palette.
OI = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "vermilion": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000",
}

# Publication rcParams (ApJ/AAS): serif text + CM math, clean inward ticks,
# vector-friendly. Figures carry no in-figure title; panel + axis labels suffice.
PUB_RCPARAMS = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def apply_pub_style():
    """Apply the shared publication rcParams to the global matplotlib state."""
    plt.rcParams.update(PUB_RCPARAMS)


def panel_label(ax, text, loc="upper left"):
    """Bold (a)/(b)/... tag in a figure corner, on a subtle white patch so it
    never collides with data or the legend."""
    x, ha = (0.035, "left") if "left" in loc else (0.965, "right")
    y, va = (0.96, "top") if "upper" in loc else (0.06, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, fontsize=11, fontweight="bold",
            va=va, ha=ha,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))


def save_fig(fig, output_dir, stem):
    """Save publication PNG (raster, website) + PDF (vector, paper) and close."""
    fig.savefig(f"{output_dir}/{stem}.png")
    fig.savefig(f"{output_dir}/{stem}.pdf")
    plt.close(fig)
