"""ICViz figure registry — the one place a figure is declared."""

from __future__ import annotations

from .imfs import SEED as _IMF_SEED
from .imfs import build_imf_classic_slopes
from .profiles import SEED as _PROFILE_SEED
from .profiles import build_density_residuals, build_family_portrait
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
        FigureSpec(
            name="profile-family-portrait",
            builder=build_family_portrait,
            stem="profile_family_portrait",
            page="10-theory/spatial-profiles/index.md",
            seed=_PROFILE_SEED,
            caption="Every released density family, half-mass normalized: Plummer, King W0=3/7/12, EFF, Michie.",
            tags=("profiles",),
        ),
        FigureSpec(
            name="profile-density-residuals",
            builder=build_density_residuals,
            stem="profile_density_residuals",
            page="10-theory/spatial-profiles/index.md",
            seed=_PROFILE_SEED,
            caption="Sampled vs analytic densities with Poisson residuals (Plummer/King/EFF, N=2e5 each).",
            tags=("profiles",),
        ),
    ]
}
