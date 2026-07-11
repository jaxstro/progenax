"""ICViz figure registry — the one place a figure is declared."""

from __future__ import annotations

from .dfs import SEED as _DF_SEED
from .dfs import (
    build_beta_anisotropy,
    build_eddington_triptych,
    build_king_lowered_maxwellian,
    build_plummer_dispersion_oracles,
)
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
        FigureSpec(
            name="king-lowered-maxwellian",
            builder=build_king_lowered_maxwellian,
            stem="king_lowered_maxwellian",
            page="10-theory/velocity-dfs/king-dfs.md",
            seed=_DF_SEED,
            caption="What lowering means: King f(v|W) vs the pure Maxwellian at W=1/3/6.",
            tags=("dfs",),
        ),
        FigureSpec(
            name="plummer-dispersion-oracles",
            builder=build_plummer_dispersion_oracles,
            stem="plummer_dispersion_oracles",
            page="10-theory/velocity-dfs/plummer-dfs.md",
            seed=_DF_SEED,
            caption="sigma_r sampled vs closed form + sigma_los vs the Dejonghe Eq. 43 oracle.",
            tags=("dfs",),
        ),
        FigureSpec(
            name="beta-anisotropy",
            builder=build_beta_anisotropy,
            stem="beta_anisotropy",
            page="10-theory/velocity-dfs/rotation-anisotropy.md",
            seed=_DF_SEED,
            caption="Realized beta(r): OM identity exact for Plummer/EFF; Michie below its ceiling.",
            tags=("dfs",),
        ),
        FigureSpec(
            name="eddington-triptych",
            builder=build_eddington_triptych,
            stem="eddington_triptych",
            page="10-theory/velocity-dfs/plummer-dfs.md",
            seed=_DF_SEED,
            caption="The Eddington pipeline rho(Psi) -> d2rho/dPsi2 -> f(E), numerical dots on the law.",
            tags=("dfs",),
        ),
    ]
}
