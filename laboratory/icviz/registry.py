"""ICViz figure registry — the one place a figure is declared."""

from __future__ import annotations

from .imfs import SEED as _IMF_SEED
from .imfs import build_imf_classic_slopes
from .specs import FigureSpec

FIGURES: dict[str, FigureSpec] = {
    spec.name: spec
    for spec in [
        FigureSpec(
            name="imf-classic-slopes",
            builder=build_imf_classic_slopes,
            stem="imf_classic_slopes",
            page="10-theory/imfs/classic.md",
            seed=_IMF_SEED,
            caption=(
                "The four classic IMFs (analytic + sampled) and their autodiff "
                "local slope S(m): Kroupa/Chabrier kinks vs Maschberger smooth."
            ),
            tags=("imf", "pilot"),
        ),
    ]
}
