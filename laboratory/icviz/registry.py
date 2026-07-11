"""ICViz figure registry — the one place a figure is declared."""

from __future__ import annotations

from .binaries_phase import build_binary_phase_space
from .dfs import SEED as _DF_SEED
from .dfs import (
    build_beta_anisotropy,
    build_eddington_triptych,
    build_king_lowered_maxwellian,
    build_phase_space_hexbin,
    build_plummer_dispersion_oracles,
)
from .engines import build_engine_a_segregation, build_engine_b_mix
from .imfs import SEED as _IMF_SEED
from .imfs import build_imf_classic_slopes
from .moe import build_moe_pqe
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
        FigureSpec(
            name="phase-space-hexbin",
            builder=build_phase_space_hexbin,
            stem="phase_space_hexbin",
            page="10-theory/velocity-dfs/index.md",
            seed=13,
            caption="The DF on the (r, v_r) plane: Plummer's open envelope vs King/EFF pinched shut at r_t.",
            tags=("dfs",),
        ),
        FigureSpec(
            name="binary-phase-space",
            builder=build_binary_phase_space,
            stem="binary_phase_space",
            page="10-theory/imfs/binary.md",
            seed=21,
            caption="Binaries in phase space, resolved vs unresolved: envelope punctures vs hidden dispersion inflation (EFF young / King old).",
            tags=("binaries",),
        ),
        FigureSpec(
            name="moe-pqe",
            builder=build_moe_pqe,
            stem="moe_pqe",
            page="10-theory/imfs/multiplicity-statistics.md",
            seed=41,
            caption="Moe P-q-e: the non-separable q distribution (twin excess at short P) + the (P, e) Roche ceiling.",
            tags=("binaries",),
        ),
        FigureSpec(
            name="engine-a-segregation",
            builder=build_engine_a_segregation,
            stem="engine_a_segregation",
            page="10-theory/spatial-profiles/lowered-model-family.md",
            seed=31,
            caption="Engine A: per-component densities at delta=0 (degenerate) vs delta=1/2 (segregated equilibrium).",
            tags=("engines",),
        ),
        FigureSpec(
            name="engine-b-mix",
            builder=build_engine_b_mix,
            stem="engine_b_mix",
            page="10-theory/populations/eddington-engine.md",
            seed=31,
            caption="Engine B: Plummer halo + EFF core in one shared potential — sampled densities on the prescribed curves, Q_j = 1/2 oracle.",
            tags=("engines",),
        ),
    ]
}
