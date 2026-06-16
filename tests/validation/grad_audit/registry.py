"""The grad-audit case registry: every public entry point x direction x param.
Tiers are added incrementally (see the implementation plan)."""
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (  # noqa: F401-adjacent — carries float64 on import
    ChabrierIMF,
    EFFProfile,
    EFFVelocityDF,
    KingProfile,
    KingVelocityDF,
    Maschberger,
    MichieProfile,
    MichieVelocityDF,
    LogNormalPeriod,
    LogUniformPeriod,
    SanaOBPeriod,
    PlummerProfile,
    PlummerVelocityDF,
    PowerLawIMF,
    ThermalEccentricity,
    UniformEccentricity,
    jeans_dispersion,
    project_dispersion,
    df_moment_dispersion,
    build_spatial_ic,
    build_cluster,
    build_king_cluster,
    build_eff_cluster,
    build_michie_cluster,
    build_limepy_cluster,
    ClusterParams,
    build_cluster_from_params,
)
from progenax.binaries import (
    IndependentCompanions,
    LogisticThermalEccentricity,
    MoeEccentricity,
)
from progenax.stellar import (
    inverse_zams_luminosity,
    zams_effective_temperature,
    zams_luminosity,
    zams_radius,
    zams_surface_gravity,
)
from progenax.binaries.assembly import resolve_binary_components
from progenax.binaries.companions import MoeCompanions
from progenax.binaries.kepler import KeplerElements
from progenax.builders import Systems, build_binary_cluster
from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.diagnostics.q_approx import q_approx
from progenax.diagnostics.segregation_approx import lambda_msr_approx
from progenax.imf.differentiable import log_prob_masses
from progenax.imf.params import IMFParams
from progenax.imf.smooth import Schechter
from progenax.kinematics.rotation import (
    apply_differential_rotation,
    apply_solid_body_rotation,
)
from tests.validation.grad_audit.binners import (
    binned_number_density,
    binned_sigma1d,
    binned_sigma_beta,
)
from tests.validation.grad_audit.core import Case, EdgeConfig
from tests.validation.grad_audit.reductions import identity_sum, mean_mass, mean_radius, mean_speed

_KEY = jax.random.PRNGKey(0)
_MASSES = jnp.ones(400)
# Split the frozen key so positions and velocities never reuse the same PRNG draw
# (repo anti-pattern: one key for pos AND vel correlates the two samplings).
_KEY_POS, _KEY_VEL = jax.random.split(_KEY)

# Plummer scale radius from r_h (a = r_h * sqrt(2^(2/3)-1)); the Merritt (1985, Eq. 46)
# Osipkov-Merritt bound is r_a >= 0.75 a, below which the OM phase-space DF goes negative.
_PLUMMER_A_FACTOR = float(jnp.sqrt(2 ** (2 / 3) - 1))


def _plummer_positions(r_h):
    return PlummerProfile(r_h=r_h).sample_positions(_MASSES, _KEY_POS)


def _plummer_velocities(r_h):
    positions = PlummerProfile(r_h=r_h).sample_positions(_MASSES, _KEY_POS)
    df = PlummerVelocityDF(r_h=r_h)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _plummer_velocities_om(r_a):
    # Isotropic positions (fixed r_h=1) with an Osipkov-Merritt anisotropic velocity DF;
    # the audited parameter is the anisotropy radius r_a (constructor kwarg anisotropy_radius).
    positions = PlummerProfile(r_h=1.0).sample_positions(_MASSES, _KEY_POS)
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


# Edge value just above the Merritt bound 0.75 a (for r_h=1): a small margin keeps the
# symmetric FD lower probe (r_a - h) on the valid side of the bound while still probing it.
_OM_BOUND = 0.75 * _PLUMMER_A_FACTOR  # ~= 0.57482 for r_h=1
_OM_EDGE = _OM_BOUND + 2e-3           # ~= 0.57682, lower FD probe (h=1e-4) stays >= bound


# ---------------------------------------------------------------------------
# King (1966) profile + DF (Task 1.2)
# ---------------------------------------------------------------------------
# The King ODE auto-sizes its integration domain from a CONCRETE W0
# (_auto_ode_domain), but under jax.grad W0 is a tracer and the domain falls
# back to the historical default (xi_max=300, n=2000) -- which is too small for
# W0 >= ~10 and trips the eager pinned-r_t ValueError. So the W0-parameterised
# closure passes an EXPLICIT domain sized for the largest probed W0 (the W0=12
# edge needs xi_t ~ 548), keeping both the W0=7 baseline and the W0=12 edge valid
# under tracing. The r_c closure fixes W0=7.0 (concrete) so it can auto-size.
_KING_XI_MAX = 700.0       # > xi_t(W0=12) ~ 548, with margin
_KING_N_ODE = 5000


def _king_positions_W0(W0):
    profile = KingProfile.from_W0_rc(
        W0=W0, r_c=1.0, xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE
    )
    return profile.sample_positions(_MASSES, _KEY_POS)


def _king_positions_rc(r_c):
    # W0 fixed (concrete) -> auto-sized ODE domain is valid; r_c is the audited param.
    profile = KingProfile.from_W0_rc(W0=7.0, r_c=r_c)
    return profile.sample_positions(_MASSES, _KEY_POS)


def _king_velocities_W0(W0):
    profile = KingProfile.from_W0_rc(
        W0=W0, r_c=1.0, xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE
    )
    positions = profile.sample_positions(_MASSES, _KEY_POS)
    df = KingVelocityDF(W0=W0, r_c=1.0, xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _king_velocities_rc(r_c):
    # r_c channel of the King lowered-Maxwellian velocity DF (Task 4.2a). W0 fixed
    # CONCRETE at 7.0 so the King ODE auto-sizes its integration domain (no explicit
    # xi_max/n_ode needed -- the W0-tracing domain-overflow problem only afflicts the
    # _king_velocities_W0 closure). r_c is the audited param; it sets the spatial scale
    # that maps into the velocity normalisation sigma ~ sqrt(G M / r_c). MEASURED at
    # r_c=1.0: AD=-1.906048e-1 vs FD=-1.906048e-1 (|ratio-1|=6.3e-9) -- machine-exact.
    # This is the registry replacement for test_df_gradients.py::TestKingDFGradients's
    # r_c part (registry previously had only the W0 channel of this DF).
    profile = KingProfile.from_W0_rc(W0=7.0, r_c=r_c)
    positions = profile.sample_positions(_MASSES, _KEY_POS)
    df = KingVelocityDF(W0=7.0, r_c=r_c)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _king_r_t(W0):
    # The King tidal radius r_t = r_c * xi_t, with xi_t from _find_tidal_radius's
    # linear-interpolation crossing. Task 1.2b feeds UNCLAMPED psi to the interp,
    # so d r_t/dW0 now flows through the diffrax solve (the implicit-function-
    # theorem result to grid accuracy). A FINE explicit grid (xi_max=400,
    # n=8000) makes the linear-interp FD estimate grid-converged so AD~FD.
    return jnp.atleast_1d(
        KingProfile.from_W0_rc(
            W0=W0, r_c=1.0, xi_max=400.0, n_ode_points=8000
        ).r_t
    )

# ---------------------------------------------------------------------------
# Michie-King anisotropic profile + DF (Task 1.3)
# ---------------------------------------------------------------------------
# solve_michie_profile RAISES (concrete-input guard) if the model does not truncate
# within xi_max -- i.e. if r_a/r_c is too small (over-anisotropic, 1/r^2 radial-orbit
# tail, infinite mass / no finite r_t). The prompt's suggested r_a=2.0 trips this at
# W0=7 (measured psi(xi_max)=1.13 > 0). r_a=8.0 (ra_hat=8) is comfortably inside the
# truncating regime: r_t ~ 56, stable across W0=7 +- h (no guard trips on the FD probes).
_MICHIE_R_A = 8.0


def _michie_positions_W0(W0):
    profile = MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=_MICHIE_R_A)
    return profile.sample_positions(_MASSES, _KEY_POS)


def _michie_velocities_W0(W0):
    profile = MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=_MICHIE_R_A)
    positions = profile.sample_positions(_MASSES, _KEY_POS)
    df = MichieVelocityDF(W0=W0, r_c=1.0, r_a=_MICHIE_R_A)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _michie_r_t(W0):
    # Michie/LIMEPY r_t is now differentiable (Task 1.2b unclamped-psi crossing fed to
    # _find_tidal_radius, same as King). d r_t/dW0 flows through the diffrax solve to
    # grid accuracy. Measured at W0=7, r_a=8: AD=78.89 vs FD=79.26 (|ratio-1|=4.7e-3) --
    # the honest FD band for a linear-interp estimate of a smooth derivative, so tol=1e-2
    # (matches KingProfile.r_t).
    return jnp.atleast_1d(
        MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=_MICHIE_R_A).r_t
    )


# Fixed query radius for the Michie density-observable channel (interior, r=1.5 < r_t~56).
_MICHIE_RHO_QUERY = jnp.array([1.5])


def _michie_log_density_rc(r_c):
    # Michie/LIMEPY density-OBSERVABLE channel in r_c (Task 4.2a). The registry already
    # has MichieProfile.sample_positions(W0) but NOT the r_c channel NOR a density
    # observable -- this case fills BOTH gaps in one clean replacement for the r_c part
    # of test_michie_physics.py::TestMichieDifferentiability::test_grad_profile_observable
    # (which FD-tests d log rho(r=1.5) / d{W0, r_c}; W0 of the SAMPLER is already covered,
    # r_c of the DENSITY is the unique uncovered piece). The observable is
    # log(density(r=1.5)) at a fixed interior query, matching the scattered test verbatim
    # (W0=7, r_a=8). MEASURED at r_c=1.0: AD=2.209859e+0 vs FD=2.209859e+0
    # (|ratio-1|=1.3e-8) -- machine-exact closed-form density; tol=1e-5 (the density band).
    p = MichieProfile.from_W0_rc(W0=7.0, r_c=r_c, r_a=_MICHIE_R_A)
    return jnp.atleast_1d(jnp.log(p.density(_MICHIE_RHO_QUERY)[0] + 1e-30))


def _michie_log_density_W0(W0):
    # Michie/LIMEPY density-OBSERVABLE channel in W0 (Task 4.2b review-fix). The W0 half of
    # test_michie_physics.py::test_grad_profile_observable runs through the CLOSED-FORM
    # density() formula, a DIFFERENT code path than MichieProfile.sample_positions(W0) (the
    # stochastic inverse-CDF sampler) -- so the sampler-W0 case does NOT subsume it (a
    # stop_gradient on density()'s W0 dependence would pass the sampler case but fail this).
    # Same observable as the r_c case: log(density(r=1.5)) at a fixed interior query (r_c=1.0,
    # r_a=8). MEASURED at W0=7: AD=2.573508e-2 vs FD=2.573507e-2 (|ratio-1|=2.3e-7) --
    # machine-exact closed-form density; tol=1e-5 (the density band).
    p = MichieProfile.from_W0_rc(W0=W0, r_c=1.0, r_a=_MICHIE_R_A)
    return jnp.atleast_1d(jnp.log(p.density(_MICHIE_RHO_QUERY)[0] + 1e-30))


# ---------------------------------------------------------------------------
# EFF (Elson-Fall-Freeman 1987) profile + Eddington DF (Task 1.3)
# ---------------------------------------------------------------------------
# EFFProfile has NO eager guard: gamma and r_t are free parameters (r_t is prescribed,
# not solved), already differentiable. The gamma=2.01 edge (just above the 3-D slope=2
# divergent-mass regime) does NOT trip any guard -- it samples cleanly (measured
# ratio 1.000 at gamma=2.01 +- h). EFFVelocityDF's only guard is the Osipkov-Merritt
# negative-DF check (concrete r_a only); the isotropic gamma case is unguarded. gamma=5
# is the mild-truncation ~virial point (the gamma=3 default is ~8% sub-virial by
# construction, a documented limitation, not a gradient hazard).
def _eff_positions_gamma(gamma):
    return EFFProfile(a=1.0, gamma=gamma, r_t=10.0).sample_positions(_MASSES, _KEY_POS)


def _eff_positions_rt(r_t):
    return EFFProfile(a=1.0, gamma=3.0, r_t=r_t).sample_positions(_MASSES, _KEY_POS)


def _eff_positions_a(a):
    # EFF scale-radius channel of the POSITION sampler (Task 4.2b review-fix). The registry
    # had EFFProfile.sample_positions in gamma + r_t, and EFFVelocityDF.sample_velocities in
    # a, but NOT the position sampler's a channel -- so test_profile_gradients.py::
    # test_eff_grad_a had no equal/stronger registry case and was held back by the migrate
    # interlock. gamma=3.0 (the scattered test's value), r_t=10.0; a flows through the
    # inverse-CDF radial draw. MEASURED at a=1.0: AD=1.143671e+0 vs FD=1.143689e+0
    # (|ratio-1|=1.5e-5) -- clean within tol=1e-3 (the EFF inverse-CDF band, matching the
    # sibling EFFProfile.sample_positions gamma/r_t cases).
    return EFFProfile(a=a, gamma=3.0, r_t=10.0).sample_positions(_MASSES, _KEY_POS)


def _eff_velocities_gamma(gamma):
    positions = EFFProfile(a=1.0, gamma=gamma, r_t=10.0).sample_positions(_MASSES, _KEY_POS)
    df = EFFVelocityDF(a=1.0, gamma=gamma, r_t=10.0)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


def _eff_velocities_a(a):
    # EFF scale-radius channel of the Eddington velocity DF (Task 4.2a). gamma fixed at
    # 5.0 (the mild-truncation ~virial point, matching _eff_velocities_gamma's audited
    # value) and r_t=10.0; a is the audited scale that flows through BOTH the positions
    # and the Eddington-DF velocity normalisation. MEASURED at a=1.0: AD=-3.160765e-1 vs
    # FD=-3.160774e-1 (|ratio-1|=2.8e-6) -- clean within tol=1e-3 (the Eddington-table +
    # inverse-CDF band, matching the gamma case). Replaces test_df_gradients.py::
    # TestEFFDFGradients's `a` part (registry previously had only the gamma channel).
    positions = EFFProfile(a=a, gamma=5.0, r_t=10.0).sample_positions(_MASSES, _KEY_POS)
    df = EFFVelocityDF(a=a, gamma=5.0, r_t=10.0)
    return df.sample_velocities(positions, _MASSES, _KEY_VEL, G=STELLAR.G)


# ---------------------------------------------------------------------------
# build_spatial_ic end-to-end (Task 1.4) — the headline params->IC path
# ---------------------------------------------------------------------------
# This is the flagship CLAUDE.md promise: jax.grad(loss)(r_h) flows through the
# WHOLE assembly — profile.sample_positions x velocity_df.sample_velocities ->
# to_com_frame -> virial_scale to Q=0.5. r_h enters BOTH the profile and the DF,
# and virial scaling couples r_h into the velocities as well (it rescales speeds
# to hit Q=T/|V|, and |V| depends on the positions which scale with r_h). G is
# explicit (STELLAR.G); build_spatial_ic splits the frozen key internally (the
# same pos/vel decorrelation the per-channel cases do by hand). tol=1e-3: the
# path is NOT pure closed-form (virial rescale + COM centering), so the FD band
# is wider than the bare-profile 1e-5.
def _build_spatial_ic_rh(r_h):
    profile = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h)
    return build_spatial_ic(profile, _MASSES, df, _KEY, G=STELLAR.G).positions


def _build_spatial_ic_rh_velocities(r_h):
    # Same end-to-end path, reducing the VELOCITIES: virial scaling injects r_h
    # into the speeds (|V| ~ G M / r_h scaling), so d<speed>/d r_h flows through
    # the virial_scale factor as well as the raw DF draw.
    profile = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h)
    return build_spatial_ic(profile, _MASSES, df, _KEY, G=STELLAR.G).velocities


# ---------------------------------------------------------------------------
# IMF samplers + mass-function summary (Task 1.5) — the params->IC path through
# the inverse-CDF/Newton mass samplers, plus the params->summary mass-function
# Fisher channel. Two suspected hazards are probed explicitly:
#
#   H4: PowerLawIMF.cdf clips m to [m_min, m_max] (power_law.py:~227) before the
#       unnormalised CDF. Probed by d(cdf)/d(m_min) at a FIXED query just inside
#       the support (m=0.101, i.e. m_min0 + 1e-3). MEASURED ratio 1.0000000
#       (AD -13.321 vs FD -13.321) -> BENIGN: the clip is inactive for an in-support
#       query, so m_min flows through the lower integration bound cleanly. Below
#       support (m=0.099 < m_min) AD=FD=0 (correct: F=0 below the support edge).
#
#   H6: ChabrierIMF.ppf Newton clamp jnp.clip(m_new, m_min, m_max) (chabrier.py:371).
#       Probed by d(ppf)/d(m_c) at a tiny u=1e-12 (sample pinned at the m_min floor).
#       MEASURED ratio 0.99967 (AD 9.344e-11 vs FD 9.347e-11) -> BENIGN: the gradient
#       is LIVE and tracks FD even at the boundary; the clamp does not zero it (the
#       sampled mass still moves smoothly with m_c through the Newton residual).
#
#   alpha=1.0: the exp_safe double-where removable-singularity guards keep the VJP
#       FINITE at exactly alpha=1, but the value/gradient there is BRANCH-LIMITED:
#       - PowerLawIMF.ppf  @alpha=1.0: AD=0 vs FD=-1.384e4 (the log branch is
#         alpha-independent, so AD=0) -> known_blocked.
#       - PowerLawIMF.mean_mass @alpha=1.0: AD=-52.2 vs FD=-35.6 (ratio 1.47); the
#         Z-denominator's alpha=1 branch flips while the numerator's does not, so AD
#         is finite-but-inconsistent -> known_blocked.
#       - IMFParams log_prob NLL @alpha3=1.0: AD=23.8 vs FD=-274.7 -> known_blocked.
#       In ALL cases alpha=1.0 is FINITE (no NaN) and alpha=1+-1e-3 is FD-consistent.
# ---------------------------------------------------------------------------
# Fixed uniform draw for the ppf/sample cases (same N as the spatial cases). The
# ppf cases reduce identity_sum over this vector; the sample cases reduce mean_mass.
_U = jax.random.uniform(_KEY, (_MASSES.shape[0],))
# Salpeter single-segment bounds (PowerLawIMF.salpeter defaults).
_PL_M_MIN, _PL_M_MAX = 0.1, 100.0


def _powerlaw_ppf_alpha(alpha):
    # Single-segment Salpeter; alpha is the (traced) slope. exponents is NOT static
    # on PowerLawIMF, so the tracer flows through the inverse-CDF closed form.
    return PowerLawIMF(exponents=[alpha], breakpoints=[],
                       m_min=_PL_M_MIN, m_max=_PL_M_MAX).ppf(_U)


def _powerlaw_sample_alpha(alpha):
    # Sampled masses (reparam: fixed key -> fixed u -> ppf), reduced by mean_mass.
    return PowerLawIMF(exponents=[alpha], breakpoints=[],
                       m_min=_PL_M_MIN, m_max=_PL_M_MAX).sample(_KEY, _MASSES.shape[0])


def _powerlaw_mean_mass_alpha(alpha):
    # params->summary: analytic E[m] for the single-segment power law. The alpha=1.0
    # edge is branch-limited (Z's removable singularity), known_blocked.
    return jnp.atleast_1d(
        PowerLawIMF(exponents=[alpha], breakpoints=[],
                    m_min=_PL_M_MIN, m_max=_PL_M_MAX).mean_mass()
    )


# H4 probe: d(cdf)/d(m_min) at a FIXED in-support query mass = m_min0 + 1e-3 = 0.101.
_H4_QUERY = jnp.array([_PL_M_MIN + 1e-3])


def _powerlaw_cdf_mmin(m_min):
    return PowerLawIMF(exponents=[2.35], breakpoints=[],
                       m_min=m_min, m_max=_PL_M_MAX).cdf(_H4_QUERY)


def _powerlaw_ppf_mmin(m_min):
    # d(ppf)/d(m_min) for the Salpeter single-segment power law (Task 4.2a). m_min sets
    # the LOWER support edge, so it enters the inverse-CDF closed form directly (every
    # sampled mass shifts with the floor). Fixed slope alpha=2.35, fixed uniform draw _U,
    # reduced by identity_sum over the sampled masses. MEASURED at m_min=0.1:
    # AD=1.572808e+3 vs FD=1.572808e+3 (|ratio-1|=2.4e-9) -- machine-exact; tol=1e-5
    # (closed-form band). Replaces test_imf_gradients.py::test_powerlaw_ppf_grad_mmin
    # (registry had cdf[H4](m_min) and ppf(alpha) but not ppf(m_min)).
    return PowerLawIMF(exponents=[2.35], breakpoints=[],
                       m_min=m_min, m_max=_PL_M_MAX).ppf(_U)


def _chabrier_ppf_mc(m_c):
    return ChabrierIMF(m_c=m_c).ppf(_U)


def _chabrier_ppf_sigma(sigma):
    return ChabrierIMF(sigma=sigma).ppf(_U)


def _chabrier_ppf_alpha(alpha):
    return ChabrierIMF(alpha=alpha).ppf(_U)


# H6 probe: d(ppf)/d(m_c) with u pinned to the m_min floor (tiny u). The Newton clamp
# at chabrier.py:371 is the suspect; measured BENIGN (live gradient, ratio ~1).
_H6_U = jnp.array([1e-12])


def _chabrier_ppf_mc_boundary(m_c):
    return ChabrierIMF(m_c=m_c).ppf(_H6_U)


def _maschberger_ppf_mu(mu):
    return Maschberger(mu=mu).ppf(_U)


def _maschberger_ppf_alpha(alpha):
    return Maschberger(alpha=alpha).ppf(_U)


def _maschberger_ppf_beta(beta):
    return Maschberger(beta=beta).ppf(_U)


def _schechter_ppf_alpha(alpha):
    return Schechter(alpha=alpha).ppf(_U)


# params->summary mass-function Fisher channel: d(NLL)/d(alpha3) for the 4-segment
# IMFParams log_prob over a FIXED Kroupa-sampled mass set. alpha3 is a traced leaf.
_IMF_MASSES = PowerLawIMF.kroupa().sample(_KEY, _MASSES.shape[0])


def _imf_logprob_nll_alpha3(alpha3):
    params = IMFParams(alpha0=jnp.array(0.3), alpha1=jnp.array(1.3),
                       alpha2=jnp.array(2.3), alpha3=alpha3)
    return jnp.atleast_1d(-jnp.sum(log_prob_masses(_IMF_MASSES, params)))


# ---------------------------------------------------------------------------
# Differentiable summary diagnostics (Task 1.6) — the params->summary path
# through the CW04 substructure-Q surrogate (q_approx) and the Allison+2009
# mass-segregation surrogate (lambda_msr_approx). Both are reduced by an
# identity sum over the scalar statistic.
#
#   q_approx: the substructure-Q is a DIMENSIONLESS length-scale ratio, so it
#       is SCALE-INVARIANT — d(Q)/d(r_h) is genuinely ~0 (measured |AD|~2e-10 for
#       a Plummer r_h rescale, pure float64 round-off: Q(r_h=0.5..4) is constant
#       to 1e-9). r_h is therefore the WRONG audited param (it produces a true
#       known_zero, not a live-gradient consistent case with teeth). We instead
#       audit the EFF slope gamma, which MORPHS THE SHAPE (radial concentration)
#       and so moves Q with a large live gradient: Q(gamma=2.5..5)=1.03..1.38,
#       d(Q)/d(gamma) ~ 9.2e-2 (measured AD 9.239e-2 vs FD 9.237e-2, ratio
#       1.000183). Mutation: wrapping the positions in stop_gradient before
#       q_approx drives AD->0 with a live FD (9.24e-2) -> hazard, proving teeth.
#
#   lambda_msr_approx: the soft Lambda_MSR (MST-ratio surrogate). The intentional
#       non-diff site at segregation_approx.py:145 — stop_gradient(median(min(dist)))
#       — sets the softmin scale ONLY; the gradient SHOULD still flow through `dist`.
#       Confirmed: a geometry param (the massive-core scale) gives a large, finite,
#       NON-ZERO d(Lambda)/d(core_scale) (AD=-4.22 at core_scale=0.2), so the
#       stop_gradient does NOT block the real gradient through dist — the design
#       refinement holds. FD-consistency depends on the softmin temperature beta:
#       at the default beta=0.1 the stopped median-scale's omitted derivative is a
#       real ~27% AD/FD gap (measured ratio 0.727 at core_scale=0.3); as beta->0
#       the softmin -> true 1-NN min and the scale only sets sharpness, so AD
#       reconciles with FD (ratio 0.994 @ beta=0.03, 0.997 @ beta=0.01). We audit
#       at beta=0.03 (sharp softmin), core_scale=0.2 (tight core, strong signal):
#       AD=-4.2157 vs FD=-4.2156, ratio 1.000026 -> the gradient flows through dist,
#       is non-zero, AND is FD-consistent.
_SEG_N_MASSIVE = 40
# Segregated-cluster masses: a light halo + a heavy central core. The massive bin
# (m > m_cut=2) is the central core, so a TIGHTER core (smaller core_scale) is MORE
# segregated -> larger Lambda. m_cut=2 sits cleanly between the 0.5 and 10.0 masses.
_SEG_MASSES = jnp.concatenate(
    [jnp.full(_MASSES.shape[0] - _SEG_N_MASSIVE, 0.5),
     jnp.full(_SEG_N_MASSIVE, 10.0)]
)


def _q_approx_gamma(gamma):
    # CW04 substructure-Q surrogate over EFF-sampled positions; gamma morphs the
    # radial concentration so Q moves (r_h would be scale-invariant -> ~0 gradient).
    positions = EFFProfile(a=1.0, gamma=gamma, r_t=10.0).sample_positions(_MASSES, _KEY_POS)
    return jnp.atleast_1d(q_approx(positions, method="naive"))


def _lambda_msr_core_scale(core_scale):
    # Soft Lambda_MSR over a segregated cluster: light halo (sigma=1) + heavy core
    # (sigma=core_scale). core_scale flows through the positions into the softmin NN
    # distances `dist`. Audited at the DEFAULT beta=0.1: the median softmin scale's
    # derivative now flows (the stop_gradient was removed -- it had omitted ~27% of the
    # true gradient at this beta), so AD is FD-consistent.
    k_halo, k_core = jax.random.split(_KEY_POS)
    halo = jax.random.normal(k_halo, (_MASSES.shape[0] - _SEG_N_MASSIVE, 3)) * 1.0
    core = jax.random.normal(k_core, (_SEG_N_MASSIVE, 3)) * core_scale
    positions = jnp.concatenate([halo, core], axis=0)
    return jnp.atleast_1d(
        lambda_msr_approx(positions, _SEG_MASSES, m_cut=2.0, tau=0.5, beta=0.1)
    )


# ---------------------------------------------------------------------------
# MultiComponentCluster Engine A: sample_cluster (Task 2.1) — params->IC through
# the lowered-isothermal multimass equilibrium. from_imf bins the IMF into n_comp
# components, runs the eigenvalue solve for the central density fractions alpha_j,
# then sample_cluster draws each star's (component, position, velocity). W0, g,
# delta are the three traceable equilibrium drivers (W0 = central dimensionless
# potential, g = truncation sharpness exponent, delta = Gieles-Zocchi mass-
# segregation exponent w_j = mu_j^-delta). G is explicit (STELLAR.G).
#
# Baseline: W0=5, g=1, delta=0.5, n_comp=3, n_stars=400 (small for the inner-loop
# gate). reduce=mean_radius weights the OUTER particles, so it is sensitive to the
# ψ=0 boundary shell — the H2 probe target.
#
# CATEGORICAL-ASSIGNMENT DISCRETENESS (the design hazard, OUT of scope): each star
# is assigned a component by jax.random.categorical(k_assign, log(N_frac_j)) with a
# FIXED key (sampling.py:37). N_frac_j depends on (W0, g, delta), so in principle a
# param nudge can flip a star's argmax(log N_frac + gumbel) assignment DISCONTINUOUSLY
# and inject FD noise that is NOT an autodiff bug. MEASURED: at h=1e-4 the assignment
# is STABLE — ZERO flips out of 400 stars for ALL of W0, g, delta at these baselines
# (the fixed gumbel key + tiny h never crosses an argmax boundary). So the FD here is
# the pure smooth position-physics derivative, uncontaminated by discreteness. This is
# the case design that side-steps the discreteness (small h + stable assignment),
# rather than weakening tol to paper over noise.
#
# H2 PROBE (ψ=0 / r≤r_t boundary masks + max(ψ,0) clamps + shared r_t kink):
# the masks live at multicomponent.py:272 (the jnp.where(psi_grid>0, rho, 0)
# xi-grid support mask) and multicomponent.py:286 (the jnp.where(r_grid<=r_t, rho, 0)
# radial truncation mask used to build the mass-CDF). The max(ψ,0) clamps that a
# sampled star could hit are at sampling.py:64-66 (the per-star W_i=rescale*max(interp
# ψ(r),0) on the sampling path) and limepy.py:275 (psi_grid=max(psi_raw,0), the
# model-build grid clamp). CLEARED BENIGN. (1) The shared tidal radius r_t IS
# differentiable in W0 (Tier-1 unclamped-ψ fix, commit 9d3365a): measured AD=1.583973
# vs FD=1.583973, ratio 1.000000 at W0=5 (and 1.000000 at W0=7) through the from_imf
# path. (2) NO sampled star ever lands where ψ(r)≤0: the mass-CDF draw places stars by
# mass, and the boundary shell carries ~zero mass. MEASURED AT THE W0=3 EDGE (the most
# extended cluster, r_t=4.26 — boundary most stretched): min ψ(r_sampled)=0.153 (max
# sampled radius 3.66 < r_t=4.26), 0/400 stars hit the sampling.py:64 max(ψ,0) clamp
# (none even within 1e-6 of ψ=0), and 0/400 categorical assignments flip between W0=3
# and W0=3+1e-4. (At the W0=5 baseline the margin is wider still: min ψ=0.227.) The
# boundary is approached more closely at W0=3 but never reached; even if it were, the
# jnp.where masks contribute ZERO gradient at the edge because the masked-out density
# past r_t is ~0 there, so the where-discontinuity carries no live derivative. The
# W0=3 edge is FD-consistent (|ratio-1|=2.6e-4), confirming the boundary is benign.
# No H2 hazard.
#
# tol=1e-3: max measured |ratio-1| is 1.1e-4 (g) and 2.6e-4 (W0=3 edge); 1e-3 is the
# honest band for the trapezoid mass-CDF + IMF-binning eigenvalue solve + categorical
# reparam, with comfortable margin.
_CLUSTER_IMF = PowerLawIMF.kroupa()
_CLUSTER_N_COMP = 3
_CLUSTER_N_STARS = 400


def _cluster_engine_a(W0, g, delta):
    # n_comp=3 log-spaced bins; the eigenvalue solve auto-sizes from CONCRETE
    # other params, but ALL THREE of (W0, g, delta) appear traced across the three
    # cases, so pass the default xi_max=300/n_ode=2000 explicitly (the W0/g/delta
    # baselines are mild — xi_t(W0=5)~9 — so the historical default domain is ample;
    # no W0>=10 domain-overflow as in the King cases).
    return MultiComponentCluster.from_imf(
        _CLUSTER_IMF, n_comp=_CLUSTER_N_COMP, W0=W0, g=g, delta=delta,
        xi_max=300.0, n_ode_points=2000)


def _cluster_sample_W0(W0):
    return _cluster_engine_a(W0, 1.0, 0.5).sample_cluster(
        _KEY, _CLUSTER_N_STARS, G=STELLAR.G).positions


def _cluster_sample_g(g):
    return _cluster_engine_a(5.0, g, 0.5).sample_cluster(
        _KEY, _CLUSTER_N_STARS, G=STELLAR.G).positions


def _cluster_sample_delta(delta):
    return _cluster_engine_a(5.0, 1.0, delta).sample_cluster(
        _KEY, _CLUSTER_N_STARS, G=STELLAR.G).positions


# ---------------------------------------------------------------------------
# MultiComponentCluster Engine A: from_components(w_j) (Task B1) — the DIRECT
# velocity-scale-ratio channel (the Fisher target). from_components builds the
# lowered-isothermal multimass equilibrium from per-component w_j (rescale_j =
# w_j^-2; smaller w_j = colder = more concentrated) rather than from the IMF-binning
# / equipartition path of from_imf (which the sample_cluster[EngineA] W0/g/delta
# cases cover). w_j is the leaf an inference would treat as the free per-component
# velocity scale, so it is the primary Fisher parameter for direct-component models.
#
# SCALAR-theta contract: w_j is a shape-(n_comp) vector but audit_entry_point drives a
# SCALAR theta. We vary the SECOND (heavy) component's ratio and hold the first fixed
# via jnp.stack([w0_fixed, w1]) — the Engine-B _cluster_b_sample_ra jnp.stack pattern.
# This perturbs the physically-meaningful RATIO w_j[1]/w_j[0] (uniform-scaling all of
# w_j would be a near-degenerate global rescale); w0=1.0 fixed, w1 (heavy) is theta.
# Config from tests/unit/cluster/test_multicomponent.py::test_differentiable_in_w_j
# (alpha_j=[0.6,0.4], m_j=[0.5,2.0], W0=7, g=1, r_c=1, n_ode=1500, n_grid=600) — the
# known-clean-sampling config. reduce=mean_speed (w_j sets the velocity scales, so the
# speeds carry the gradient).
#
# MEASURED (theta0=w1=0.8, h=1e-4, mean_speed over velocities), 3 PRNG seeds:
#   seed 0: AD=1.161003e-1  FD=1.161044e-1  |ratio-1|=3.5e-5  flips=0/400
#   seed 1: AD=1.099080e-1  FD=1.099169e-1  |ratio-1|=8.1e-5  flips=0/400
#   seed 2: AD=1.057982e-1  FD=1.057937e-1  |ratio-1|=4.2e-5  flips=0/400
# |AD|~0.11 >> eps=1e-9 (live, non-zero gradient); seed-stable (AD 0.106..0.116, FD
# tracks); 0/400 categorical-assignment flips at +-h for ALL seeds (FD is discreteness-
# free — the fixed gumbel key + tiny h never crosses an argmax boundary, same as the
# from_imf Engine-A cases). max |ratio-1| over seeds = 8.1e-5; tol=1e-3 is the honest
# band for the trapezoid mass-CDF + lowered-isothermal solve + categorical reparam,
# matching the sibling sample_cluster[EngineA] cases (comfortable >10x margin).
_CLUSTER_A_W0_FIXED = 1.0  # the light component's w_j, held fixed


def _cluster_a_w_j(w1):
    # Heavy-component velocity-scale ratio w_j[1] is the scalar theta; w_j[0] fixed.
    w_j = jnp.stack([jnp.asarray(_CLUSTER_A_W0_FIXED, dtype=jnp.float64), w1])
    model = MultiComponentCluster.from_components(
        alpha_j=jnp.array([0.6, 0.4]), w_j=w_j, m_j=jnp.array([0.5, 2.0]),
        W0=7.0, g=1.0, r_c=1.0, n_ode_points=1500, n_grid=600)
    return model.sample_cluster(_KEY, _CLUSTER_N_STARS, G=STELLAR.G).velocities


# ---------------------------------------------------------------------------
# MultiComponentCluster Engine A: from_mass_segregation(r_a) (Task B2) — the
# Osipkov-Merritt ANISOTROPY-RADIUS channel of the equipartition constructor.
# from_mass_segregation builds w_j = mu_j^(-delta) (Gieles & Zocchi 2015) from the
# masses, then sets the per-component anisotropy radius ra_hat_j = (r_a/r_c)*mu_j^eta
# (eta=0 -> mass-independent, the paper default). r_a is the SCALAR anisotropy radius
# that an inference would treat as the free OM scale (beta(r)=r^2/(r^2+r_a^2)); it is
# the Fisher parameter for anisotropic equipartition models. r_a is ALREADY a scalar
# (no jnp.stack needed, unlike the w_j vector case). Config from the known-clean
# tests/unit/cluster/test_multicomponent.py::test_aniso_sample_differentiable_in_ra
# (alpha_j=[0.6,0.4], m_j=[1.0,4.0], W0=7, g=1, delta=0.4, eta=0, r_c=1, xi_max=800,
# n_ode=2000, n_grid=600). Anisotropy lives in the VELOCITIES, so reduce=mean_speed
# (measured cleaner than the existing test's mean-squared-RADIAL reduction, whose
# single-projection-per-star variance gives a 3-16% AD/FD band vs mean_speed's <0.2%).
#
# theta0=4.0 (NOT the test's 10.0): the OM gradient grows AND the FD band tightens as
# r_a shrinks toward the cluster scale (more anisotropy = stronger, cleaner live signal).
# At r_a=10 (mild anisotropy, beta only reaches ~0.1 over the cluster) |AD|~8e-4 with a
# ~2.5e-2 ratio band; at r_a=4 |AD|~1.3e-2 (comfortably live) with a ~1.5e-3 band.
#
# REALIZABILITY: Engine-A's anisotropic build is TABLE-based (no concrete negative-DF
# ValueError like Engine B's eddington_engine guard); r_a=4 (ra_hat_j=[4,4]) is well
# inside the truncating regime and BOTH central-FD probes r_a +- h build cleanly
# (verified: r_a in {3.9996, 4.0, 4.0004} all sample OK, no raise). The over-anisotropy
# bound (1/r^2 radial-orbit divergence) is far below r_a=4 for this W0=7 config.
#
# CATEGORICAL-ASSIGNMENT DISCRETENESS (out of scope, same as the w_j / sample_cluster
# Engine-A cases): each star's component is jax.random.categorical(...) off a FIXED
# gumbel key. MEASURED 0/400 flips at +-h for ALL seeds, so the FD is the pure smooth
# velocity-physics derivative, discreteness-free.
#
# MEASURED (theta0=r_a=4.0, h=4e-4, mean_speed over velocities), 5 PRNG seeds:
#   seed 0: AD=1.255278e-2  FD=1.253345e-2  |ratio-1|=1.5e-3  flips=0/400
#   seed 1: AD=1.343451e-2  FD=1.341555e-2  |ratio-1|=1.4e-3  flips=0/400
#   seed 2: AD=1.357858e-2  FD=1.355928e-2  |ratio-1|=1.4e-3  flips=0/400
#   seed 3: AD=1.324524e-2  FD=1.322511e-2  |ratio-1|=1.5e-3  flips=0/400
#   seed 4: AD=1.365211e-2  FD=1.363375e-2  |ratio-1|=1.4e-3  flips=0/400
# |AD|~1.3e-2 >> eps=1e-9 (live, non-zero); seed-stable; max |ratio-1| over 5 seeds =
# 1.54e-3. tol=3e-3: the measured band is 1.5e-3, so the sibling 1e-3 would NOT cover it
# (mean_speed over a stochastic anisotropic mass-CDF sample is intrinsically FD-noisier
# than a closed form); 3e-3 is ~2x the measured max -- the honest band, NOT a weakened tol
# to mask a mismatch (a blocked gradient would give |ratio-1|~1, the silent-zero signature).
# The FD floor here is ROUNDING-limited, not truncation-limited (the table-based anisotropic
# build + categorical-reparam draw quantizes mean_speed): h=4e-4 is a signal/rounding-noise
# sweet-spot (h=1e-4 is WORSE, |ratio-1|~2.6e-2), and a wide-secant H=5e-2 cross-check at the
# test's r_a=10 gives |ratio-1|=2e-4 -- so AD is provably correct, the band is pure FD noise.
# This architectural noise source (table+categorical) is why the sibling Engine-B r_a (density-
# defined Eddington quadrature, no table) is machine-exact at tol=1e-3 while this one needs 3e-3.
_CLUSTER_A_RA_CFG = dict(
    alpha_j=jnp.array([0.6, 0.4]), m_j=jnp.array([1.0, 4.0]),
    W0=7.0, g=1.0, delta=0.4, eta=0.0, r_c=1.0,
    xi_max=800.0, n_ode_points=2000, n_grid=600)


def _cluster_a_r_a(r_a):
    # Scalar anisotropy radius r_a is theta; from_mass_segregation sets ra_hat_j =
    # (r_a/r_c)*mu_j^eta internally. reduce=mean_speed (anisotropy is in the velocities).
    model = MultiComponentCluster.from_mass_segregation(r_a=r_a, **_CLUSTER_A_RA_CFG)
    return model.sample_cluster(_KEY, _CLUSTER_N_STARS, G=STELLAR.G).velocities


# ---------------------------------------------------------------------------
# build_binary_cluster END-TO-END (Task B3) — the flagship binary-cluster Fisher
# path: IMF -> companion(P,q,e) composition -> system COMs (build_spatial_ic,
# virial-scaled to Q) -> resolve_binary_components places each binary's two stars
# around its COM. r_h is the spatial-scale leaf an inference treats as free; it is
# the headline parameter for the binary-cluster forward model.
#
# r_h enters ONLY the spatial assembly (PlummerProfile.sample_positions x
# PlummerVelocityDF.sample_velocities, then virial-scaled to Q=0.5). The IMF mass
# sampling and the companion (P, q, e) draws are r_h-INDEPENDENT — masses/orbits do
# not depend on r_h — so the is_binary multiplicity mask and every categorical /
# Heaviside selection are r_h-invariant. The discreteness CANNOT move with r_h, so
# the is_binary/is_real mask has 0 flips at +-h BY CONSTRUCTION (confirmed below),
# leaving a pure smooth positions channel: r_h linearly rescales the system COMs.
#
# COMPACT CHOICE: compact=True (the eagerly-compacted ICResult: the real
# n_systems + n_binaries particles, no ghost slots). MEASURED cleaner than
# compact=False (the masked 2N ResolvedBinaries, whose single-star "secondary"
# ghost slots sit at COM + the m2=0 relative offset and dilute the reduction):
#   compact=True  |ratio-1| ~ 5e-14..2e-12  (machine-exact)
#   compact=False |ratio-1| ~ 1e-10          (clean too, but ICResult is the
#                                             flagship user-facing output)
# The audit engine never jits (it jax.grad's the closure and calls the FD probes as
# plain Python), so the ICResult's data-dependent compacted shape is fine — and
# since the is_binary mask is identical at r_h +- h (0 flips), the compacted shape
# is the SAME at both FD probes, so the central FD is well-defined.
#
# Config = the known-clean test_binary_cluster.py::test_grad_through_r_h:
# IndependentCompanions(fbin=0.5, q~Flat(q_min=0.2), P~LogUniform[2,4], e~Thermal),
# target=Systems(100), units=STELLAR, Q=0.5 (default). reduce=mean_radius (r_h sets
# the spatial scale, so the positions carry the gradient).
#
# MEASURED (theta0=r_h=1.0, h=1e-4, mean_radius over ICResult.positions), 5 seeds:
#   seed 0: AD=1.346409e+0  FD=1.346409e+0  |ratio-1|=6.6e-13  flips=0/400
#   seed 1: AD=2.016999e+0  FD=2.016999e+0  |ratio-1|=5.1e-14  flips=0/400
#   seed 2: AD=1.713738e+0  FD=1.713738e+0  |ratio-1|=2.0e-12  flips=0/400
#   seed 5: AD=1.744199e+0  FD=1.744199e+0  |ratio-1|=9.0e-13  flips=0/400
#   seed 7: AD=1.932542e+0  FD=1.932542e+0  |ratio-1|=5.7e-14  flips=0/400
# |AD|~1.3-2.0 >> eps=1e-9 (live, non-zero; sign positive: bigger r_h -> bigger
# cluster -> bigger mean radius); seed-stable; 0/400 is_real-mask flips at +-h for
# ALL seeds (r_h-invariant multiplicity, as argued above). max |ratio-1| over seeds
# = 2.0e-12. The end-to-end virial-scale + COM-center path is r_h-LINEAR here (r_h
# just scales the Plummer COMs and the resolve offsets ride along), so unlike the
# stochastic cluster samplers this is essentially machine-exact. tol=1e-5 is a
# comfortable >5e6x margin over the measured 2e-12 (NOT a weakened tol — a blocked
# gradient would give |ratio-1|~1, the silent-zero signature, not 2e-12).
def _independent_companions_b3():
    # The known-clean IndependentCompanions config from test_binary_cluster.py.
    return IndependentCompanions(
        binary_fraction=ConstantBinaryFraction(0.5),
        q_distribution=FlatMassRatio(q_min=0.2),
        period_distribution=LogUniformPeriod(log_P_min=2.0, log_P_max=4.0),
        eccentricity_distribution=ThermalEccentricity(),
    )


def _build_binary_cluster_rh(r_h):
    # r_h threads BOTH the Plummer profile and the Plummer velocity DF (same r_h, as
    # the equilibrium requires); IMF + companion draws are r_h-independent. compact=True
    # -> the real-particle ICResult; reduce its positions with mean_radius.
    ic = build_binary_cluster(
        profile=PlummerProfile(r_h=r_h),
        velocity_df=PlummerVelocityDF(r_h=r_h),
        primary_imf=PowerLawIMF.kroupa(),
        companion_model=_independent_companions_b3(),
        target=Systems(100),
        key=_KEY,
        units=STELLAR,
        compact=True,
    )
    return ic.positions


# ---------------------------------------------------------------------------
# Cluster convenience builders (build_cluster + aliases + ClusterParams wrapper).
# The thin sugar layer over build_spatial_ic. ALL MEASURED 2026-06-14 (N=400, key 0):
#   build_cluster[Plummer] r_h               AD=+1.4409e0  FD=+1.4409e0  |ratio-1|=2.7e-13
#   build_cluster[Plummer+OM] r_a            AD=+2.7280e-2 FD=+2.7280e-2 |ratio-1|=4.7e-6
#   build_cluster[Plummer+rotation] omega    AD=+6.4243e-1 FD=+6.4243e-1 |ratio-1|=4.2e-9
#   build_king_cluster r_c                   AD=+5.7318e0  FD=+5.7318e0  |ratio-1|=4.9e-12
#   build_eff_cluster gamma                  AD=-1.5850e0  FD=-1.5850e0  |ratio-1|=1.8e-5
#   build_michie_cluster W0                  AD=+5.4395e0  FD=+5.4415e0  |ratio-1|=3.7e-4
#   build_limepy_cluster W0                  AD=+7.7293e-1 FD=+7.7287e-1 |ratio-1|=7.6e-5
#   build_cluster_from_params r_h            AD=+1.4409e0  FD=+1.4409e0  |ratio-1|=2.7e-13
# build_cluster is pure sugar (bit-identical to build_spatial_ic in the base case), so the
# Plummer r_h case is machine-exact like build_spatial_ic[Plummer]. The modifier channels
# anisotropy_radius (matched OM DF) and omega (apply_solid_body_rotation overlay) are
# FD-consistent. The per-FAMILY alias cases prove King/EFF/Michie/LIMEPY flow THROUGH the
# new builder path (build_cluster's own cases are Plummer-only).
#
# KING audits r_c (NOT W0): KingProfile.from_W0_rc AUTO-sizes its ODE domain, so a TRACED
# W0 (AD, fallback grid) and a CONCRETE W0 (FD, auto grid) would use different grids (a grid
# artifact, not a real gradient bug). With W0 concrete=7.0 the domain is consistent for both
# AD and FD, and r_c is machine-exact (4.9e-12). Michie (fixed xi_max=800) and LIMEPY (fixed
# xi_max=300) use FIXED domains, so their W0 IS consistent (measured clean).
#
# TIDAL_RADIUS is INTENTIONALLY NOT a consistent Case here: it flows through
# apply_tidal_truncation's straight-through surrogate (an exact hard cut forward + a logistic
# grad backward), which is DELIBERATELY not FD-consistent (the AD is the smooth surrogate
# ~109.6; the finite-N FD is a discrete bin-crossing staircase 0/33/67/117). apply_tidal_
# truncation is EXEMPT_HELPER in the manifest for exactly this reason, and build_cluster's
# tidal channel inherits that. It is instead covered by a dedicated LIVE-gradient teeth test
# (test_grad_audit.py::test_cluster_tidal_gradient_has_teeth) that asserts |AD|>eps + finite
# (catching a silent-zero regression) WITHOUT a false FD-consistency claim.
# ---------------------------------------------------------------------------
def _bc_plummer_rh(r_h):
    return build_cluster(PlummerProfile(r_h=r_h), masses=_MASSES, key=_KEY).positions


def _bc_plummer_om(r_a):
    # OM anisotropy threaded via the matched Plummer DF; r_a=0.7 is above the Merritt
    # bound 0.75a (~0.575 for r_h=1), so the FD probe stays in the realizable regime.
    return build_cluster(PlummerProfile(r_h=1.0), masses=_MASSES, key=_KEY,
                         anisotropy_radius=r_a).velocities


def _bc_plummer_omega(omega):
    # Solid-body rotation overlay (apply_solid_body_rotation); omega enters the velocities.
    return build_cluster(PlummerProfile(r_h=1.0), masses=_MASSES, key=_KEY,
                         rotation=omega).velocities


def _bk_rc(r_c):
    # King FAMILY through the alias; audit r_c with W0 CONCRETE=7.0 (consistent ODE domain).
    return build_king_cluster(masses=_MASSES, W0=7.0, r_c=r_c, key=_KEY).positions


def _beff_gamma(gamma):
    # EFF family through the alias; gamma morphs the density slope (no ODE, precomputed CDF).
    return build_eff_cluster(masses=_MASSES, a=1.0, gamma=gamma, r_t=10.0, key=_KEY).positions


def _bmich_W0(W0):
    # Michie family through the alias; fixed xi_max=800 -> W0 grid consistent for AD and FD.
    return build_michie_cluster(masses=_MASSES, W0=W0, r_c=1.0, r_a=8.0, key=_KEY).positions


def _blim_W0(W0):
    # LIMEPY family through the alias; fixed xi_max=300 -> W0 grid consistent for AD and FD.
    return build_limepy_cluster(masses=_MASSES, W0=W0, g=1.0, r_c=1.0, key=_KEY).positions


def _bcfp_rh(r_h):
    # build_cluster_from_params: scalar r_h -> ClusterParams PyTree -> wrapper -> ICResult.
    # The PyTree-theta path; r_h is the profile's traced leaf (machine-exact, like build_cluster).
    return build_cluster_from_params(
        ClusterParams(profile=PlummerProfile(r_h=r_h)), masses=_MASSES, key=_KEY).positions


# ---------------------------------------------------------------------------
# MultiComponentCluster Engine B: from_density_profiles -> sample_cluster
# (Task 2.2) — params->IC through the density-defined shared-Psi Eddington/OM
# build (Poisson quadrature + Abel/Eddington inversion, NO ODE) and the per-star
# (component, position, velocity) draw. The audited params are the traceable
# scale leaves on the PRESCRIBED component profiles (multicomponent.py:407).
#
# REALIZABLE CONFIG (reused verbatim from tests/validation/test_engine_b_physics.py
# _headline_model): a Plummer halo + EFF(gamma=5) core,
#   profiles      = [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)]
#   mass_fractions= [0.6, 0.4],  m_j = [0.5, 1.0]
# This is the science-headline mix and a PROVEN shared-potential equilibrium
# (theory Q_j = 0.5 +- 3e-3; physics test test_plummer_halo_eff_core_equilibrium).
# The core scale a=0.8 is itself a realizability constraint (a=0.4 has NO
# equilibrium — genuinely negative Eddington DF; see _headline_model's docstring
# and TestEngineB), so we start from a known-good, comfortably-realizable point.
#
# REALIZABILITY MARGIN AT THE FD PROBES (eddington_engine.py:212-216 raises on a
# CONCRETE negative DF; the central FD evaluates the build at theta +- h on
# CONCRETE values, so BOTH probes must stay realizable). MEASURED min f / max|f|
# (f_min_j) at every probe point — all positive with margin:
#   r_h   in {1.99980, 2.0, 2.00020}: f_min_j = [~0.0164, ~1.185e-4] (>0)
#   gamma in {4.99950, 5.0, 5.00050}: f_min_j = [~0.0164, ~1.185e-4] (>0)
#   r_a   in {2.99970, 3.0, 3.00030}: f_min_j = [~0.0848, ~1.185e-4] (>0)
# (The OM r_a=3.0 f_min_halo=0.085 reproduces test_om_beta_profile_realized's
# stated [+0.085, +1.2e-4].) The tightest margin is the EFF core's ~1.185e-4 —
# small but strictly positive and essentially flat across all probes, so neither
# FD probe trips the realizability ValueError. AD-vs-FD identity is a property of
# the computational graph (both sides share the same grids), so the gate uses the
# REDUCED resolution n_r=3000, n_e=500 (same as the physics test's
# test_gradients_ad_vs_fd) to stay in the inner-loop budget.
#
# CATEGORICAL-ASSIGNMENT DISCRETENESS (the design hazard, OUT of scope, same as
# Engine A): each star's component is jax.random.categorical(...) with a FIXED key
# off N_frac_j ∝ mass_fractions_j / m_j. mass_fractions and m_j are CONCRETE here
# (the audited leaves are the profile scales), and a profile-scale nudge can in
# principle reshape N_frac via the mass-normalization — but MEASURED: 0/400
# assignment flips at h for ALL THREE params (r_h, gamma, r_a), so the FD is the
# pure smooth position/velocity-physics derivative, discreteness-free.
#
# tol=1e-3: max measured |ratio-1| is 3.0e-4 (gamma); 1e-3 is the honest band for
# the Poisson-quadrature + Eddington-inversion + mass-CDF + categorical reparam,
# with comfortable margin (matches the Engine A cases).
_CLUSTER_B_MF = jnp.array([0.6, 0.4])
_CLUSTER_B_MJ = jnp.array([0.5, 1.0])
# Reduced resolution (AD==FD is a graph property; both sides use the same grids).
_CLUSTER_B_N_R, _CLUSTER_B_N_E, _CLUSTER_B_N_GRID = 3000, 500, 1000


def _cluster_engine_b(profiles, r_a_j=None):
    return MultiComponentCluster.from_density_profiles(
        profiles, _CLUSTER_B_MF, m_j=_CLUSTER_B_MJ, r_a_j=r_a_j,
        n_r=_CLUSTER_B_N_R, n_e=_CLUSTER_B_N_E, n_grid=_CLUSTER_B_N_GRID)


def _cluster_b_sample_rh(r_h):
    # Halo Plummer scale r_h (the prescribed-density leaf) -> positions.
    profiles = [PlummerProfile(r_h=r_h), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)]
    return _cluster_engine_b(profiles).sample_cluster(
        _KEY, _CLUSTER_N_STARS, G=STELLAR.G).positions


def _cluster_b_sample_gamma(gamma):
    # Core EFF slope gamma (the prescribed-density leaf) -> positions.
    profiles = [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=gamma, r_t=9.0)]
    return _cluster_engine_b(profiles).sample_cluster(
        _KEY, _CLUSTER_N_STARS, G=STELLAR.G).positions


def _cluster_b_sample_ra(r_a):
    # Halo Osipkov-Merritt anisotropy radius r_a_j[0] (core isotropic, r_a=inf).
    # OM anisotropy lives in the VELOCITIES (beta(r)=r^2/(r^2+r_a^2)), so reduce
    # the speeds. r_a=3.0 is comfortably inside the realizable regime (f_min_halo
    # = 0.085); the FD probes r_a +- h both stay realizable (see block above).
    # |AD|=1.77e-3 is small *physically*, not weakly: r_a=3.0 >> r_h=2.0 puts the
    # halo in the MILD-anisotropy regime (beta(r) only reaches ~0.3 at r~r_a), so
    # d<speed>/dr_a is genuinely small. The teeth are in the ratio, not the
    # magnitude: |ratio-1|=2.7e-9 (a blocked gradient would give |ratio-1|~1, FD
    # finite & AD~0 -- the silent-zero signature -- not a near-perfect match).
    # NOTE the core leaf r_a_j[1]=inf: under jax.grad the whole stacked r_a_j is a
    # tracer, so _validate_engine_b_inputs's concrete >0 guard skips BOTH leaves
    # (two-tier contract); the inf is the isotropic-core sentinel, valid by design.
    profiles = [PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)]
    r_a_j = jnp.stack([r_a, jnp.asarray(jnp.inf)])
    return _cluster_engine_b(profiles, r_a_j=r_a_j).sample_cluster(
        _KEY, _CLUSTER_N_STARS, G=STELLAR.G).velocities


# ---------------------------------------------------------------------------
# Binary entry points (Task 2.3): params->IC through the orbital-mechanics layer.
#
# (a) KeplerElements.to_state — the elements -> Cartesian (position, velocity)
#     map. Kepler's equation M=E-e·sin(E) is solved by a FIXED-iteration (50)
#     Newton scheme in jax.lax.scan (kepler.py:312, differentiable); the
#     y_p = a·sqrt(max(1-e²,1e-12))·sin E and E_dot = n/max(1-e·cos E,1e-12)
#     double-where guards keep the value+grad finite as e->1 without a hard floor
#     on the dimensional a³. We audit d<position-radius>/d(e) (mean_radius over the
#     single (3,) position) at a moderate e0=0.5, and an e=0.999 EDGE (near the
#     parabolic limit — the slowest Newton convergence, where the prompt expects
#     machine-precision grads). MEASURED:
#       to_state mean_radius @ e=0.5  : AD=4.439570e-1 vs FD=4.439570e-1 (ratio 1.0000000)
#       to_state mean_radius @ e=0.999: AD=9.995240e-1 vs FD=9.995240e-1 (ratio 1.0000000)
#     The position map is closed-form-ish (Newton + trig rotation), FD-consistent to
#     machine precision at BOTH e — even the e=0.999 edge is clean (50 Newton iters
#     reach convergence; the max(1-e²,1e-12) guard is inactive at e=0.999 since
#     1-e²=2.0e-3 >> 1e-12). tol=1e-5 (the closed-form band).
#
#     VELOCITY-channel note (honest scoping): we audit the POSITION channel because
#     it is the cleaner FD signal near the parabolic limit. The velocity channel was
#     re-measured at e=0.999 (mean_speed over the single (3,) velocity): with the
#     engine's CENTRAL FD it is also machine-precision clean — AD=-7.484823e-2 vs
#     FD=-7.484823e-2 (ratio 1.0000000). Only under a COARSE one-sided/forward FD
#     step does it soften (forward h=1e-3 -> ratio ~0.99993, the ~0.9995 the review
#     saw with an even coarser one-sided probe). The softening is the curvature of
#     the d/de[sqrt(1-e²)] truncation term, which is steepest as e->1, so a one-sided
#     stencil mis-estimates the slope there. The velocity gradient is NOT blocked —
#     it is correct (central FD ratio 1.0), just FD-noisier under a coarse one-sided
#     stencil — hence the position channel is the audited (cleaner) signal.
# ---------------------------------------------------------------------------
def _kepler_to_state_e(e):
    # Fixed a/inc/Omega/omega/M0; vary the eccentricity e. M_total + G explicit.
    el = KeplerElements(a=1.0, e=e, i=0.3, Omega=0.4, omega=0.5, M0=1.0)
    return el.to_state(M_total=2.0, G=STELLAR.G).position


# (a2/a3) KeplerElements.to_state — the semi-major-axis a and mean-anomaly M0 columns
#     of the elements->Cartesian Jacobian (Task 4.2a). The registry previously audited
#     only the e column; the differentiable-IC binary pipeline depends on the a and M0
#     columns too. Same closed-form-ish map (fixed-iteration Newton Kepler solve + trig
#     rotation), reduced by mean_radius over the single (3,) position (matching the e
#     case). MEASURED, M_total=2.0, G=STELLAR.G explicit:
#       to_state mean_radius @ a=1.5 : AD=9.163137e-1 vs FD=9.163137e-1 (|ratio-1|=2.4e-13)
#       to_state mean_radius @ M0=1.0: AD=3.144025e-1 vs FD=3.144025e-1 (|ratio-1|=2.1e-9)
#     Both machine-precision clean; tol=1e-5 (closed-form band). These replace the a and
#     M0 parts of test_binary_physics.py::TestKeplerTransformGradients (the registry
#     position channel is an equal/stronger FD replacement -- that test FD-checks d|v|^2/da
#     and d|r|/dM0; the position-radius channel here is the cleaner near-degenerate signal).
def _kepler_to_state_a(a):
    el = KeplerElements(a=a, e=0.3, i=0.3, Omega=0.4, omega=0.5, M0=1.0)
    return el.to_state(M_total=2.0, G=STELLAR.G).position


def _kepler_to_state_M0(M0):
    el = KeplerElements(a=1.0, e=0.3, i=0.3, Omega=0.4, omega=0.5, M0=M0)
    return el.to_state(M_total=2.0, G=STELLAR.G).position


# (a4) KeplerElements.from_state — the INVERSE map (state->elements), the
#     angular-momentum / eccentricity-vector method in kepler_inverse. We build a
#     concrete reference state from a known orbit, then audit how the recovered
#     semi-major axis a moves as a velocity-magnitude scale s rescales the input
#     velocity (scaling v rescales the orbital energy E=-GM/2a, so a moves smoothly).
#     This is the exact transform/param test_binary_physics.py::
#     TestKeplerTransformGradients::test_from_state_grad_wrt_velocity_scale FD-checks
#     (d a / d(v-scale)); we reduce identity_sum over the recovered a. MEASURED at s=1.0,
#     M_total=2.0, G=STELLAR.G: AD=2.365317e+0 vs FD=2.365317e+0 (|ratio-1|=8.0e-8) --
#     clean; tol=1e-4 (the from_state vector-algebra band, matching the scattered test's
#     rel<1e-4 tolerance). The base orbit uses a NON-DEGENERATE geometry (i=0.3, Omega=0.4,
#     omega=0.5, matching the other Kepler cases) so from_state's angular-momentum/
#     eccentricity-vector inversion runs its general path, not a coordinate-plane special
#     case; the recovered a is rotation-invariant (a depends only on E=v^2/2-GM/|r|), so the
#     measured value is unchanged by the orientation -- this hardens the codepath, not the number.
_KEPLER_FROM_STATE_BASE = KeplerElements(
    a=1.0, e=0.3, i=0.3, Omega=0.4, omega=0.5, M0=1.0).to_state(M_total=2.0, G=STELLAR.G)
_KEPLER_R0 = _KEPLER_FROM_STATE_BASE.position
_KEPLER_V0 = _KEPLER_FROM_STATE_BASE.velocity


def _kepler_from_state_vscale(s):
    return jnp.atleast_1d(
        KeplerElements.from_state(_KEPLER_R0, s * _KEPLER_V0,
                                  M_total=2.0, G=STELLAR.G).a
    )


# (b) resolve_binary_components — the binary->spatial connector. A SMALL fixed
#     binary population (N=50, all is_binary=True, fixed COM pos/vel, fixed m1/m2,
#     fixed angles) with ONE traced element: the semi-major axis a (broadcast from
#     a scalar). Each system's two components are placed at COM ± δr via
#     to_binary_state, so d<component-radius>/d(a) flows through the per-system
#     vmap(to_binary_state). reduce=mean_radius over the (2N,3) resolved positions.
#     MEASURED: resolve mean_radius @ a=0.5: AD=8.437447e-2 vs FD=8.437447e-2
#     (ratio 1.0000000008). The single-star sanitization (a_safe/e_safe/m2_safe) is
#     INACTIVE here (all is_binary=True), so the path is the pure smooth orbital
#     placement; FD-consistent to machine precision. tol=1e-5 (closed-form band).
#     (The seed is PRNGKey(11); an earlier draft used PRNGKey(0), which gives
#     AD=5.872715e-2 — a stale number that predated the seed change.)
_RB_N = 50
_RB_KEY = jax.random.PRNGKey(11)
_RB_KP, _RB_KV = jax.random.split(_RB_KEY)
_RB_COM_POS = jax.random.normal(_RB_KP, (_RB_N, 3))
_RB_COM_VEL = jax.random.normal(_RB_KV, (_RB_N, 3)) * 0.1
_RB_M1 = jnp.full((_RB_N,), 1.0)
_RB_M2 = jnp.full((_RB_N,), 0.5)
_RB_IS_BINARY = jnp.ones((_RB_N,), dtype=bool)
_RB_E = jnp.full((_RB_N,), 0.3)
_RB_INC = jnp.full((_RB_N,), 0.4)
_RB_OMEGA = jnp.full((_RB_N,), 0.5)
_RB_OMEGA_P = jnp.full((_RB_N,), 0.6)
_RB_M_ANOM = jnp.full((_RB_N,), 1.0)


def _resolve_binary_a(a_scalar):
    a = jnp.full((_RB_N,), a_scalar)
    return resolve_binary_components(
        _RB_COM_POS, _RB_COM_VEL, _RB_M1, _RB_M2, _RB_IS_BINARY,
        a, _RB_E, _RB_INC, _RB_OMEGA, _RB_OMEGA_P, _RB_M_ANOM, G=STELLAR.G,
    ).positions


# (b2) resolve_binary_components — MIXED is_binary (Fisher-integrity coverage). Same
#     fixed N=50 population as (b), but the mask is half True / half False (25 binaries
#     + 25 singles). This ACTIVATES the single-star sanitization path
#     (a_safe=1/e_safe=0/m2_safe=0; assembly.py:85-87): the 25 single slots feed
#     sanitized constant elements into to_binary_state, so their d/da is exactly 0 —
#     a FINITE (zero) gradient, NOT a NaN. We confirm jax.jacfwd of the per-system
#     resolved positions wrt the per-element a is NaN-free, with d/da=0 on every single
#     slot and a live non-zero d/da on every binary slot. This directly verifies the
#     sanitization does not leak NaN into a Fisher matrix.
#     MEASURED: mixed resolve mean_radius @ a=0.5: AD=3.019178e-2 vs FD=3.019178e-2
#     (ratio 1.0000000002), positions all finite. tol=1e-5 (closed-form band).
_RB_MIXED = jnp.arange(_RB_N) % 2 == 0  # 25 binaries (even idx) + 25 singles (odd)


def _resolve_binary_a_mixed(a_scalar):
    a = jnp.full((_RB_N,), a_scalar)
    return resolve_binary_components(
        _RB_COM_POS, _RB_COM_VEL, _RB_M1, _RB_M2, _RB_MIXED,
        a, _RB_E, _RB_INC, _RB_OMEGA, _RB_OMEGA_P, _RB_M_ANOM, G=STELLAR.G,
    ).positions


# (c) MoeCompanions.sample — a sampler that MIXES a smooth grid-CDF-reparameterized
#     (P, q, e) draw with a DISCRETE is_binary = uniform(kb) < f_bin(m1) Heaviside
#     mask and a m2 = where(is_binary, m1·q, 0) gate (companions.py:153-157). The
#     MoePeriod / MoeDiStefano2017Full / MoeEccentricity samplers are all grid-CDF
#     inverses (smooth, properly reparameterized — moe_di_stefano.py:318,378), so q,
#     P, e are smooth in M1. None of the constructor leaves are traceable (q_min,
#     logP_min, n_grid are floats/ints; the Table-13 grids are module constants), so
#     the only smooth knob is m1 itself — we vary a SCALAR multiplier s on a fixed
#     Kroupa mass set and reduce <e> (mean of the sampled eccentricities). e is
#     gated by NEITHER the is_binary mask NOR the m2 where, so it isolates the pure
#     P->e coupling gradient.
#
#     CLASSIFICATION = consistent/clean (HONEST measurement, not cherry-picked):
#       - is_binary FLIP COUNT at s=1±h (h=1e-4): 0/400 at +h, 0/400 at -h (n_binary
#         =98). The fixed-key Heaviside mask does NOT cross threshold for any star
#         under the small nudge, so the discrete selection injects ZERO FD noise.
#       - <e> wrt s: AD=-1.023290e-4 vs FD=-1.023290e-4 (ratio 1.0000000, |AD|=
#         1.02e-4 >> eps). The grid-CDF P->e coupling gradient flows cleanly.
#       - is_circular FLIP COUNT (MoeEccentricity.sample's etap1=eta+1<=1e-6 branch,
#         eccentricity.py:276, e->0 where circular): 10/400 stars are ALREADY deeply
#         circular at s=1.0 (min etap1=-698 << 1e-6, far below threshold, not near it),
#         but 0/400 FLIP between s=1.0 and s=1±h. The deeply-circular stars stay
#         circular (their de/ds=0 is correct: e is pinned to 0) and no near-threshold
#         star crosses, so the discrete jnp.where(is_circular,0,e) injects ZERO extra
#         FD noise — exactly why <e> stays ratio 1.0000000.
#       - (cross-check) <a> ratio 1.0000000; <m2=m1·q> ratio 1.0000000 too — even the
#         MASKED/GATED m2 channel is FD-consistent here BECAUSE 0 mask flips means the
#         where() gate is locally smooth. So the discrete is_binary mask does NOT
#         block the q/a/e/m2 gradient at this baseline.
#     The discrete selection (the Heaviside in f_bin, the lax.cond/where in the
#     single-slope MoeDiStefano2017, the twin/power-law where in the two-slope Full)
#     would only contaminate FD if a param nudge crossed a selection boundary; at this
#     baseline with the grid-CDF reparameterization + 0 mask flips, none do. <e> is
#     the cleanest channel (mask-independent), so it is the audited Case; tol=1e-3
#     (the grid-CDF inverse-interp band, matching the other reparam samplers).
_MOE_N = _MASSES.shape[0]  # 400
_MOE_BASE_M1 = PowerLawIMF.kroupa().sample(jax.random.PRNGKey(7), _MOE_N)
_MOE_DAY = 86400.0 / STELLAR.time_scale_cgs  # day in STELLAR time units (Myr)
_MOE_MODEL = MoeCompanions()


def _moe_companions_mean_e(s):
    # Scalar multiplier s on the fixed Kroupa primaries; reduce <e> (mask-independent).
    m1 = _MOE_BASE_M1 * s
    _, comp = _MOE_MODEL.sample(_KEY, m1, G=STELLAR.G, day_in_time_units=_MOE_DAY)
    return jnp.atleast_1d(jnp.mean(comp.e))


# ---------------------------------------------------------------------------
# Rotation kinematic overlays (Task 2.4): params->IC through the additive
# streaming-rotation transforms (kinematics/rotation.py). Both are pure
# velocity overlays — v_out = v_base + v_rot(theta, positions) — over a FIXED
# Plummer position sample (r_h=1) and a FIXED z-axis (axis=[0,0,1], nonzero so
# _normalized_rotation_axis does NOT raise). We start from ZERO base velocities
# so the audited channel is the PURE rotation field (no Plummer-draw speed term
# in the denominator of mean_speed), isolating d<speed>/d(theta) of the overlay
# itself. reduce=mean_speed.
#
# (a) apply_solid_body_rotation vs omega. v_rot = omega·(axis × r), so for omega>0
#     mean_speed = omega·mean(|axis × r|) is EXACTLY LINEAR in omega — the gradient
#     is the closed form d<speed>/d(omega) = mean(|axis × r|), independent of omega.
#     VERIFIED: AD = mean(|axis × r|) = 1.1226200071 to <1e-12 (closed-form match),
#     and AD=1.122620e+00 vs central FD=1.122620e+00 (|ratio-1|=6.9e-13, machine-
#     exact — the linear overlay has zero FD truncation error). tol=1e-5 (closed-form
#     band, with enormous margin).
#
# (b) apply_differential_rotation vs v_peak. v_phi(R)=v_peak·(R/R_peak)·exp(1-R/R_peak)
#     is LINEAR in v_peak (v_peak only scales the curve), so the overlay speed is again
#     linear in v_peak and AD is machine-exact: AD=7.635384e-01 vs FD=7.635384e-01
#     (|ratio-1|=5.8e-13). tol=1e-5.
#
# (c) apply_differential_rotation vs R_peak. R_peak enters x=R/R_peak inside BOTH the
#     (R/R_peak) prefactor and the exp(1-R/R_peak) — genuinely NONLINEAR, so this is
#     the real teeth: the central FD carries truncation error. Even so it is FD-
#     consistent to ~5.5e-8: AD=-4.356504e-02 vs FD=-4.356503e-02 (|ratio-1|=5.5e-8).
#     The gradient is negative (R_peak0=1.0 is below the position-weighted mean R, so
#     pushing the peak outward lowers the mean rotation speed of the inner-weighted
#     sample). tol=1e-5 (the nonlinear FD band, with comfortable margin).
_ROT_AXIS = jnp.array([0.0, 0.0, 1.0])
_ROT_POS = PlummerProfile(r_h=1.0).sample_positions(_MASSES, _KEY_POS)
_ROT_VEL = jnp.zeros_like(_ROT_POS)  # zero base -> audit the pure rotation overlay


def _solid_body_omega(omega):
    return apply_solid_body_rotation(_ROT_VEL, _ROT_POS, omega, _ROT_AXIS)


def _differential_v_peak(v_peak):
    return apply_differential_rotation(_ROT_VEL, _ROT_POS, v_peak, 1.0, _ROT_AXIS)


def _differential_R_peak(R_peak):
    return apply_differential_rotation(_ROT_VEL, _ROT_POS, 1.0, R_peak, _ROT_AXIS)


# ---------------------------------------------------------------------------
# Tier 3 — the binned-kinematic Fisher path (Task 3.1): params -> binned summary
# statistic (sigma_1d(r), beta(r)) via the FROZEN-EDGE binners vendored in
# tests/validation/grad_audit/binners.py. This is the Fisher channel itself:
# F = JᵀJ with J = d(binned summary)/d(theta), so a silently-zero or wrong column
# of J would be a confidently-wrong forecast. The data-side bin geometry r_edges is
# STATIC (the observer freezes the bins; gradients flow through the velocities that
# fill them) — correct and in-scope-as-frozen.
#
# These ARE legitimately differentiable: the per-bin sigma_k and beta_k have
# CONTINUOUS values that move smoothly as r_h / r_a scale the velocity magnitudes.
# The bin-membership (searchsorted on frozen edges) is a sub-leading discreteness on
# top of the smooth dispersion gradient. We isolate the smooth channel by choosing an
# FD step small enough that NO star crosses a frozen edge across theta +- h (the same
# "small h + stable membership" design the Engine-A/B categorical-flip cases use,
# NOT a weakened tol).
#
# CONFIG: a Plummer IC built end-to-end (build_spatial_ic: profile x DF ->
# COM-center -> virial-scale to Q=0.5), N=2000 stars (well-occupied bins under the
# n_min=30 floor), and K=7 frozen edges [0, 0.4, 0.8, 1.2, 1.8, 2.6, 4.0, 8.0]
# spanning the populated radii. Baseline occupancy @ r_h=1: bins
# [206, 588, 435, 332, 206, 116, 85] (32 stars beyond r=8). G is explicit (STELLAR.G).
#
# FD-STEP / BIN-CROSSING DIAGNOSTIC (the headline measurement). At the engine
# DEFAULT h_rel=1e-4 the symmetric FD probe r_h=1 +- 1e-4 makes exactly ONE star
# (#821, r=7.99977, sitting right on the r=8.0 outer edge) cross bin 6 -> out-of-range.
# That single Heaviside flip injects a discrete jump that corrupts the COARSE central
# FD (measured ratio 0.86 at h_rel=1e-4 — NOT an autodiff bug). Shrinking the FD step
# to h_rel=1e-5 moves the probe inside the edge so 0/2000 stars cross at +-h, and the
# central FD then matches AD to MACHINE PRECISION:
#   sigma_1d(r) (identity_sum over sig_hat), r_h=1, h_rel=1e-5:
#       AD=-3.345969e+00 vs FD=-3.345969e+00 (ratio 1.0000000); per-bin all ratios 1.0;
#       AD/FD ratio stable at 1.000000 for h in {1e-5, 1e-6, 1e-7} and scales with N
#       (ratio 1.0 at N=2000/4000/8000) — the gradient is correct, the 0.86 was pure
#       coarse-FD edge-straddle. tol=1e-3 (comfortable margin over the machine-exact
#       residual; the band for the virial-rescale + COM + frozen-edge binner path).
#   beta(r) (identity_sum over beta_hat), Osipkov-Merritt DF, r_a=2.0, ENGINE-DEFAULT
#       h_rel=1e-4 (the beta channel needs NO override: r_a moves only velocities, so bin
#       membership is r_a-invariant — 0/2000 crossings by construction, the sigma edge-
#       straddle cannot happen here): AD=-1.000759e+00 vs FD=-1.000754e+00 (ratio
#       1.0000054, the anisotropy headline channel). r_a=2.0 is comfortably above the
#       Merritt (1985) bound r_a >= 0.75 a ~ 0.575 for r_h=1, so the OM phase-space DF
#       stays positive at r_a +- h. tol=1e-3.
# MUTATION CHECK (proving the case has Fisher teeth): wrapping the sampled (pos, vel)
# in jax.lax.stop_gradient before binned_sigma1d drives AD -> 0.000000e+00 exactly
# while the live FD stays -3.89e+00 (at h_rel=1e-4) / -3.35e+00 (at h_rel=1e-5) — see
# test_grad_audit.py::test_binned_sigma_mutation_has_teeth. The collapse to 0 proves
# the case genuinely tests the params->summary gradient (not a trivial constant).
_BK_N = 2000
_BK_MASSES = jnp.ones(_BK_N)
_BK_GROUP = jnp.zeros(_BK_N, dtype=jnp.int32)
# Frozen radial bin edges (STATIC — the data-side bin geometry; in-scope-as-frozen).
_BK_R_EDGES = jnp.array([0.0, 0.4, 0.8, 1.2, 1.8, 2.6, 4.0, 8.0])
_BK_N_MIN = 30
# r_a=2.0 >> Merritt bound 0.75 a (~0.575 for r_h=1): mild anisotropy, DF stays positive.
_BK_OM_R_A = 2.0


def _binned_sigma1d_rh(r_h):
    # build_spatial_ic end-to-end (positions AND velocities in r_h via virial scaling);
    # the FROZEN-edge binner returns per-bin sig_hat that moves smoothly with r_h.
    profile = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h)
    ic = build_spatial_ic(profile, _BK_MASSES, df, _KEY, G=STELLAR.G)
    sig_hat, _se, _w, _n = binned_sigma1d(
        ic.positions, ic.velocities, _BK_GROUP, 1, _BK_R_EDGES, n_min=_BK_N_MIN)
    return sig_hat


def _binned_beta_ra(r_a):
    # Isotropic Plummer positions (r_h=1) with an Osipkov-Merritt anisotropic DF;
    # the audited param is the anisotropy radius r_a (it enters the velocities only).
    profile = PlummerProfile(r_h=1.0)
    df = PlummerVelocityDF(r_h=1.0, anisotropy_radius=r_a)
    ic = build_spatial_ic(profile, _BK_MASSES, df, _KEY, G=STELLAR.G)
    res = binned_sigma_beta(
        ic.positions, ic.velocities, _BK_R_EDGES, component_id=None, n_min=_BK_N_MIN)
    return res.beta_hat


# --- N(r) number-density Fisher channel: the MODEL side, NOT the data ---------
# The N(r) Fisher gradient lives in the MODEL's expected per-shell occupancy
# p_k(theta) = F(r_{k+1}) - F(r_k), where F = PlummerProfile.enclosed_mass_fraction
# is the analytic Plummer CDF M(<r)/M = r^3/(r^2+a^2)^(3/2) (the new public method,
# Task 3.1 N(r)). A Poisson number-density likelihood writes mu_k = N_total*p_k(theta)
# and differentiates THROUGH mu_k (= through p_k), NOT through the frozen observed
# counts. p_k is smooth closed-form in r_h (via a = r_h*sqrt(2^(2/3)-1)), so this is
# the clean, machine-exact Fisher column for the radial-profile channel.
# MEASURED @ r_h=1 over the frozen _BK_R_EDGES: AD=-2.691265e-2 vs FD=-2.691265e-2
# (ratio 1.0000000) -> consistent, tol=1e-5 (the closed-form analytic band).
def _model_pk_rh(r_h):
    profile = PlummerProfile(r_h=r_h)
    # Expected per-shell occupancy over the FROZEN Tier-3 edges (model, not data).
    return (profile.enclosed_mass_fraction(_BK_R_EDGES[1:])
            - profile.enclosed_mass_fraction(_BK_R_EDGES[:-1]))


# --- N(r) DATA side: the frozen binned count, correctly non-differentiable -----
# PINNED known-limitation. binned_number_density is a sum of indicator functions on
# FROZEN edges: N_k = sum_i 1[r_edge_k <= r_i < r_edge_{k+1}]. Its derivative wrt any
# model parameter is a.e. exactly 0 (the indicators are piecewise-constant in r_h),
# so jax.grad returns AD=0 — a FINITE, correct-by-design answer: the N(r) gradient
# lives in the MODEL p_k above, not in the frozen data. The finite-difference probe,
# by contrast, is a NONZERO discrete step (MEASURED FD=-5.000e3 @ r_h=1, h=1e-4): as
# r_h nudges, individual stars cross the frozen bin edges and the integer count jumps.
# That FD step is the data-side bin-crossing the audit treats as frozen/out-of-scope,
# NOT an autodiff bug. We therefore classify this expect="known_blocked" (passes iff
# AD is FINITE — AD=0 is finite and correct), NOT "known_zero": known_zero requires
# BOTH |AD|<eps AND |FD|<eps, but here |FD|=5e3 >> eps, so known_zero would mis-flag
# the correct AD=0 as a hazard. known_blocked is the right pin: it documents that the
# binned DATA count is correctly non-differentiable.
def _binned_number_density_rh(r_h):
    profile = PlummerProfile(r_h=r_h)
    df = PlummerVelocityDF(r_h=r_h)
    ic = build_spatial_ic(profile, _BK_MASSES, df, _KEY, G=STELLAR.G)
    return binned_number_density(ic.positions, _BK_R_EDGES)


# ---------------------------------------------------------------------------
# Task B4: binary period distributions — the pure params->IC period draws (no G,
# no units; period in days). Each is a smooth inverse-CDF / location-scale
# reparameterization at the FIXED module key _KEY, so the observable
# mean(log10(periods)) is a differentiable function of the distribution param and
# central FD is clean. Reduce = mean(log10(.)) inline (matches the scattered
# tests' observable in tests/unit/binaries/test_population.py). N=4000 (inner-loop):
# the AD-vs-FD RATIO (the gate quantity) is clean at N=2000/4000/20000 for all three
# (residual ~1e-10..1e-13 at every N). The gradient VALUE itself converges with N for
# the two finite-mean cases (Sana ~0.792, LogUniform ~0.49 drift <1% across N; LogNormal's
# is exactly 1.0 at any N), so reduced N is a pure compute saving with no AD/FD-fidelity
# cost. Three PRNG seeds measured to confirm seed-stability; the Case freezes the module key.
#
#   (1) SanaOBPeriod.power — p(logP) ∝ (logP)^power inverted via jnp.power. The
#       FD-tested target (test_sana_power_gradient_matches_finite_difference). The
#       index threads through the inverse-CDF power, a non-trivial gradient.
#       MEASURED (theta0=-0.55, h=1e-4, mean(log10 P) over N=4000), 3 seeds:
#         seed 0: AD=7.920134e-1  FD=7.920134e-1  |ratio-1|=1.4e-9
#         seed 1: AD=8.044817e-1  FD=8.044817e-1  |ratio-1|=1.4e-9
#         seed 2: AD=7.984597e-1  FD=7.984597e-1  |ratio-1|=1.4e-9
#       |AD|~0.79 >> eps (live); seed-stable; max |ratio-1| = 1.4e-9. tol=1e-5
#       (>7000x margin over the measured residual; a blocked gradient -> |ratio-1|~1).
#
#   (2) LogNormalPeriod.mu_log_P — log10 P = mu + sigma*z (z = erfinv reparam of u),
#       so d<log10 P>/dmu = 1 EXACTLY (CLOSED-FORM ANALYTIC; the location shifts every
#       sample by the same dmu). MEASURED (theta0=4.8, h=4.8e-4), 3 seeds:
#         seed 0: AD=1.000000e+0  FD=1.000000e+0  |ratio-1|=7.4e-14
#         seed 1: AD=1.000000e+0  FD=1.000000e+0  |ratio-1|=7.4e-14
#         seed 2: AD=1.000000e+0  FD=1.000000e+0  |ratio-1|=2.7e-12
#       AD=1.0 to machine precision, exactly the closed form. tol=1e-5.
#
#   (3) LogUniformPeriod.log_P_max — log10 P = lo + u*(hi - lo), so
#       d<log10 P>/d(hi) = <u> ≈ 0.5 (CLOSED-FORM ANALYTIC; the mean uniform draw).
#       MEASURED (theta0=8.0, h=8e-4), 3 seeds:
#         seed 0: AD=4.938855e-1  FD=4.938855e-1  |ratio-1|=6.7e-13
#         seed 1: AD=5.042832e-1  FD=5.042832e-1  |ratio-1|=4.0e-13
#         seed 2: AD=5.078402e-1  FD=5.078402e-1  |ratio-1|=1.5e-12
#       <u>~0.49-0.51 (the finite-N mean uniform draw, -> 0.5); seed-stable. tol=1e-5.
_PERIOD_N = 4000  # inner-loop sample size (N-stable for the smooth reparam'd mean)
_log10_mean = lambda periods: jnp.mean(jnp.log10(periods))  # the scattered-test observable


def _sana_period_power(power):
    return SanaOBPeriod(power=power).sample(_KEY, _PERIOD_N)


def _lognormal_period_mu(mu_log_P):
    return LogNormalPeriod(mu_log_P=mu_log_P).sample(_KEY, _PERIOD_N)


def _loguniform_period_logpmax(log_P_max):
    return LogUniformPeriod(log_P_max=log_P_max).sample(_KEY, _PERIOD_N)


# Task B5: binary eccentricity distributions — the pure e_max->ecc draws (bounded
# [0,1]; no G, no units). Each samples at the FIXED module key _KEY; the observable
# is mean(e) (NOT mean(log10), unlike the periods), reduced inline. N=4000 (inner
# loop): the AD-vs-FD RATIO (the gate quantity) is machine-clean at this N for all
# four (residual ~1e-12..1e-13; flat in N). Three PRNG seeds measured to confirm
# seed-stability; each Case freezes the module key. Two are closed-form analytic
# location-scale derivatives (Thermal->2/3, Uniform->1/2), two are FD-targets where
# e_max threads a period/mass-conditional sampler (Moe, LogisticThermal).
#
#   (1) ThermalEccentricity.e_max — e = e_max*sqrt(u), so d<e>/de_max = <sqrt(u)>
#       (CLOSED-FORM ANALYTIC) -> 2/3 in the N->inf limit (<sqrt u> for u~U(0,1)).
#       MEASURED (theta0=0.99, h=1e-4, mean(e) over N=4000), 3 seeds:
#         seed 0: AD=6.619815e-1  FD=6.619815e-1  |ratio-1|=1.2e-13
#         seed 1: AD=6.708214e-1  FD=6.708214e-1  |ratio-1|=1.0e-12
#         seed 2: AD=6.728200e-1  FD=6.728200e-1  |ratio-1|=4.4e-13
#       <sqrt u>~0.66-0.67 (the finite-N draw, -> 2/3); seed-stable. tol=1e-5.
#
#   (2) UniformEccentricity.e_max — e = u*e_max (e_min=0), so d<e>/de_max = <u>
#       (CLOSED-FORM ANALYTIC) -> 1/2. MEASURED (theta0=0.9, h=9e-5), 3 seeds:
#         seed 0: AD=4.938855e-1  FD=4.938855e-1  |ratio-1|=6.7e-13
#         seed 1: AD=5.042832e-1  FD=5.042832e-1  |ratio-1|=4.0e-13
#         seed 2: AD=5.078402e-1  FD=5.078402e-1  |ratio-1|=9.2e-13
#       <u>~0.49-0.51 (-> 0.5); seed-stable. tol=1e-5.
#
#   (3) MoeEccentricity.e_max — e = e_max(P) * u^(1/(eta+1)) with e_max(P) the Roche
#       ceiling clipped to the e_max FIELD. FIXED P=1e8 d, M1=20 Msun (scattered
#       test config): at this long P the raw Roche relation >= 1, so the clip
#       e_max(P)=min(roche, e_max) BINDS on the field and d<e>/de_max is the live
#       <u^(1/(eta+1))> (eta=0.873 -> exponent ~0.534). At shorter P the period cap
#       is non-binding and the field gradient is legitimately 0 (not tested here).
#       DISCRETENESS CHECK: the is_circular branch (etap1<=1e-6 -> e=0) flips ZERO
#       stars at e_max0+/-h (eta=0.873 >> -1 everywhere at P=1e8); measured circular
#       count = 0 at 0.9899 / 0.99 / 0.9901, so no FD contamination from the gate.
#       MEASURED (theta0=0.99, h=1e-4, mean(e) over N=4000), 3 seeds:
#         seed 0: AD=6.471313e-1  FD=6.471313e-1  |ratio-1|=1.2e-12
#         seed 1: AD=6.562164e-1  FD=6.562164e-1  |ratio-1|=2.2e-13
#         seed 2: AD=6.583287e-1  FD=6.583287e-1  |ratio-1|=2.4e-14
#       |AD|~0.65 >> eps (live); seed-stable; max |ratio-1| = 1.2e-12. tol=1e-5.
#
#   (4) LogisticThermalEccentricity.e_max — e = blend(P)*e_max*sqrt(u), so
#       d<e>/de_max = <blend(P)*sqrt(u)>. FIXED periods = logspace(0.5, 3.5, N)
#       (scattered test config) span the circular->thermal transition, so the blend
#       half-suppresses the thermal scale -> AD~0.33 (vs Thermal's 0.66). The e_max
#       field threads linearly through every sample (mass-independent).
#       MEASURED (theta0=0.99, h=1e-4, mean(e) over N=4000), 3 seeds:
#         seed 0: AD=3.308037e-1  FD=3.308037e-1  |ratio-1|=4.7e-13
#         seed 1: AD=3.362300e-1  FD=3.362300e-1  |ratio-1|=1.3e-13
#         seed 2: AD=3.363630e-1  FD=3.363630e-1  |ratio-1|=9.7e-13
#       |AD|~0.33 >> eps (live); seed-stable; max |ratio-1| = 9.7e-13. tol=1e-5.
_ECC_N = 4000  # inner-loop sample size (N-stable for the bounded ecc mean)
_mean_ecc = lambda e: jnp.mean(e)  # the scattered-test observable (mean ecc, not log)
_MOE_PERIODS = jnp.full(_ECC_N, 1e8)   # long-P: e_max FIELD binds (Roche cap >= 1)
_MOE_MASSES = jnp.full(_ECC_N, 20.0)   # 20 Msun O-star (eta=0.873, non-circular)
_LT_PERIODS = jnp.logspace(0.5, 3.5, _ECC_N)  # spans the circular->thermal transition


def _thermal_ecc_emax(e_max):
    return ThermalEccentricity(e_max=e_max).sample(_KEY, _ECC_N)


def _uniform_ecc_emax(e_max):
    return UniformEccentricity(e_min=0.0, e_max=e_max).sample(_KEY, _ECC_N)


def _moe_ecc_emax(e_max):
    return MoeEccentricity(e_max=e_max).sample(_KEY, _MOE_PERIODS, _MOE_MASSES)


def _logistic_thermal_ecc_emax(e_max):
    return LogisticThermalEccentricity(e_max=e_max).sample(_KEY, _LT_PERIODS)


# Task B6: IndependentCompanions.e_max — the SAME mean(e) observable as B5's bare
# ThermalEccentricity case, but routed through the FULL IndependentCompanions
# ASSEMBLY (companions.py:91-112): the kb is_binary = uniform < f_bin Heaviside draw,
# the q->m2 = where(is_binary, m1·q, 0) gate, the period->a Kepler map, and the
# 7-field CompanionElements pytree. This confirms the e_max gradient SURVIVES the
# assembly graph — a stop_gradient anywhere in the connector (or an e field
# accidentally gated by is_binary) would zero it. COMPLEMENTARY to B5: B5 audits
# ThermalEccentricity.sample in ISOLATION; B6 audits the same e_max flowing through
# the companion-model assembly. Config reproduces test_companions.py::
# test_grad_fd_accurate_eccentricity: m1 = full(4000, 2.0), FlatMassRatio(q_min=0.2),
# LogUniformPeriod(2,4), ConstantBinaryFraction(0.5), ThermalEccentricity(e_max).
# REDUCTION = mean(comp.e) over ALL N (matches the scattered test). e is NOT gated by
# is_binary (companions.py:110 — e is sampled unconditionally; only m2 is gated), so
# singles carry the full thermal scale and mean(e) over all N is the clean observable.
# The closure REBUILDS IndependentCompanions with a fresh ThermalEccentricity(e_max=
# theta) so the traced e_max threads the assembly.
def _independent_companions_e_max(e_max):
    from progenax.binaries import IndependentCompanions
    from progenax.imf.binary import ConstantBinaryFraction, FlatMassRatio

    model = IndependentCompanions(
        binary_fraction=ConstantBinaryFraction(0.5),
        q_distribution=FlatMassRatio(q_min=0.2),
        period_distribution=LogUniformPeriod(log_P_min=2.0, log_P_max=4.0),
        eccentricity_distribution=ThermalEccentricity(e_max=e_max),
    )
    _, comp = model.sample(_KEY, _IC_M1, G=STELLAR.G, day_in_time_units=_MOE_DAY)
    return comp.e


_IC_M1 = jnp.full(_ECC_N, 2.0)  # fixed 2 Msun primaries (scattered-test config)


# ---------------------------------------------------------------------------
# ZAMS stellar relations (Tout+1996) — P3 grad-audit cases. The four forward
# functions are elementwise rational/composite functions of mass; the inverse is
# the differentiable fixed-iteration Newton/scan invert dM/dL. All are scalar-in /
# scalar-out at a single mass, so we wrap in jnp.atleast_1d and reduce by
# identity_sum (the params->summary scalar-observable form). theta0 is a
# representative main-sequence mass M=5 (well inside the fitted 0.1-100 Msun range,
# off any clip), so the gradient is the smooth interior derivative.
#
# INVERSE M1 (critical): the inverse case MUST probe an IN-RANGE L_target. We pick
# L_target = zams_luminosity(5.0) ~ 530.13 Lsun so the audited point is M~5 (NOT a
# clipped plateau at M>150 where dM/dL is a dead zero). Measured dM/dL = 2.64e-3 at
# this L (live, non-zero); the round-trip inverse(zams_luminosity(5))~5 is exact.
#
# MEASURED (theta0 as below, h_rel=1e-4, identity_sum), all clean:
#   zams_luminosity            mass=5    AD=3.785187e+2  FD=3.785187e+2  |ratio-1|=4.9e-9
#   zams_radius                mass=5    AD=3.002906e-1  FD=3.002906e-1  |ratio-1|=9.1e-10
#   zams_effective_temperature mass=5    AD=2.073397e+3  FD=2.073397e+3  |ratio-1|=1.4e-9
#   zams_surface_gravity       mass=5    AD=-1.210520e-2 FD=-1.210520e-2 |ratio-1|=1.1e-9
#   inverse_zams_luminosity    L~530.13  AD=2.641877e-3  FD=2.641877e-3  |ratio-1|=1.9e-9
# All closed-form / smooth-Newton => tol=1e-5 is comfortable (>100x margin; a blocked
# gradient would give |ratio-1|~1, the silent-zero signature, not ~1e-9).
_ZAMS_MASS = 5.0  # representative MS mass, interior to the fitted 0.1-100 Msun range
_ZAMS_INVERSE_L = float(zams_luminosity(jnp.asarray(_ZAMS_MASS)))  # ~530.13 Lsun -> M~5


def _zams_luminosity_mass(mass):
    return jnp.atleast_1d(zams_luminosity(mass))


def _zams_radius_mass(mass):
    return jnp.atleast_1d(zams_radius(mass))


def _zams_teff_mass(mass):
    return jnp.atleast_1d(zams_effective_temperature(mass))


def _zams_logg_mass(mass):
    return jnp.atleast_1d(zams_surface_gravity(mass))


def _inverse_zams_L(L_target):
    # In-range L_target (~530.13 Lsun => M~5); the Newton/scan invert is differentiable
    # via the fixed-iteration lax.scan. Scalar in -> scalar out, wrapped to (1,).
    return jnp.atleast_1d(inverse_zams_luminosity(L_target))


# ---------------------------------------------------------------------------
# Dispersion forward models (Phase 0 Task 8) — jeans_dispersion (3-D anisotropic
# Jeans) and project_dispersion (B&M82 LOS/PM projection). Both are reverse-mode
# differentiable forward models that expose the EQUILIBRIUM velocity dispersion of
# a (potential, anisotropy) pair — Task 7 proved AD-vs-FD consistency. These
# params->summary cases probe the two leaves an inference would treat as free: the
# Osipkov-Merritt anisotropy radius r_a and the total mass M. The closures sample at
# INTERIOR radii r in [0.5, 2.0] (well inside the s-grid [1e-4 r_max, r_max] with
# r_max = 30 a ~ 23 for r_h=1, so jnp.interp never clamps -> no silent-zero edge) and
# reduce the sigma vector by identity_sum.
#
# MEASURED (theta0 as below, h_rel=1e-4 default, identity_sum over sigma at the three
# interior radii), STELLAR.G:
#   jeans_dispersion[Plummer+OM]      r_a=2.0  AD=-7.779763e-2 FD=-7.779763e-2 |ratio-1|=1e-8
#   jeans_dispersion[Plummer]         M=400.0  AD= 1.938686e-3 FD= 1.938686e-3 |ratio-1|<1e-9
#   project_dispersion[Plummer+OM]    r_a=2.0  AD= 4.027117e-2 FD= 4.027117e-2 |ratio-1|=1e-8
#   project_dispersion[Plummer+OM].pm_t r_a=2.0 AD=9.151713e-2 FD=9.151713e-2 |ratio-1|=1e-8
# All |AD| >> eps=1e-9 (live, non-zero) and machine-exact FD-consistent (the integrands
# are smooth trapezoid quadratures in r_a / M with no branch crossings), so tol=1e-3 is a
# comfortable margin (the honest band for the trapezoid Jeans + B&M82 u-quadrature).
_DISP_R_INTERIOR = jnp.array([0.5, 1.0, 2.0])
_DISP_R_A = 2.0  # OM anisotropy radius (>> 0.75 a ~ 0.575, well inside the valid domain)


def _jeans_dispersion_om_r_a(r_a):
    # Plummer OM radial dispersion sigma_r in the anisotropy radius r_a.
    prof = PlummerProfile(r_h=1.0)
    return jeans_dispersion(prof, r_a, _DISP_R_INTERIOR, 400.0, STELLAR.G).sigma_r


def _jeans_dispersion_M(M):
    # Plummer Jeans radial dispersion sigma_r in total mass M (sigma_r^2 ∝ G M).
    prof = PlummerProfile(r_h=1.0)
    return jeans_dispersion(prof, _DISP_R_A, _DISP_R_INTERIOR, M, STELLAR.G).sigma_r


def _project_dispersion_om_r_a(r_a):
    # B&M82 projected line-of-sight dispersion sigma_los in the anisotropy radius r_a.
    prof = PlummerProfile(r_h=1.0)
    return project_dispersion(prof, r_a, _DISP_R_INTERIOR, 400.0, STELLAR.G).sigma_los


def _project_dispersion_om_r_a_pmt(r_a):
    # B&M82 projected tangential proper-motion dispersion sigma_pm_t (the beta-carrying
    # channel: kernel (1 - beta)) in the anisotropy radius r_a.
    prof = PlummerProfile(r_h=1.0)
    return project_dispersion(prof, r_a, _DISP_R_INTERIOR, 400.0, STELLAR.G).sigma_pm_t


# Michie-King DF (built ONCE; W0=6.0, r_c=1.0, r_a=5.0 — mild anisotropy, interior radii
# well inside the bound region). df_moment_dispersion integrates the DF's 2nd velocity
# moments by polar quadrature (smooth, no boundary mask), so sigma_r is a clean trapezoid
# function of M (sigma^2 ∝ G M). M-gradient ONLY — the W0 path is the deferred Michie-W0
# limitation (docs/plans/2026-06-16-michie-king-equilibrium-gradient-redesign-deferred.md).
# MEASURED (theta0=400.0, identity_sum over sigma_r at the three interior radii), STELLAR.G:
# AD/FD |ratio-1| ~ 4.2e-4 -- consistent (tol=1e-3).
_DISP_MICHIE_DF = MichieVelocityDF(W0=6.0, r_c=1.0, r_a=5.0)


def _df_moment_dispersion_M(M):
    # Michie DF-moment radial dispersion sigma_r in total mass M (sigma_r^2 ∝ G M).
    return df_moment_dispersion(_DISP_MICHIE_DF, _DISP_R_INTERIOR, M, STELLAR.G).sigma_r


REGISTRY: list[Case] = [
    Case(id="PlummerProfile.sample_positions", direction="params->IC",
         fn=_plummer_positions, param="r_h", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    Case(id="PlummerVelocityDF.sample_velocities", direction="params->IC",
         fn=_plummer_velocities, param="r_h", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    Case(id="PlummerVelocityDF+OM.sample_velocities", direction="params->IC",
         fn=_plummer_velocities_om, param="r_a", theta0=2.0, reduce=mean_speed,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("r_a=0.75a", _OM_EDGE),)),
    # King profile positions: differentiable in W0 (profile SHAPE) and r_c.
    Case(id="KingProfile.sample_positions", direction="params->IC",
         fn=_king_positions_W0, param="W0", theta0=7.0, reduce=mean_radius,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("W0=12", 12.0),)),
    Case(id="KingProfile.sample_positions", direction="params->IC",
         fn=_king_positions_rc, param="r_c", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # King lowered-Maxwellian velocity DF: differentiable in W0.
    Case(id="KingVelocityDF.sample_velocities", direction="params->IC",
         fn=_king_velocities_W0, param="W0", theta0=7.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # King DF r_c channel (Task 4.2a): W0=7 concrete (ODE auto-sizes), vary r_c.
    # AD=-1.906048e-1 vs FD=-1.906048e-1 (|ratio-1|=6.3e-9), machine-exact. tol=1e-3.
    # Replaces test_df_gradients.py::TestKingDFGradients r_c part.
    Case(id="KingVelocityDF.sample_velocities", direction="params->IC",
         fn=_king_velocities_rc, param="r_c", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # The King tidal radius r_t: now differentiable in W0 (Task 1.2b) via the
    # unclamped-psi linear-interp crossing in _find_tidal_radius. d r_t/dW0 ~ 48
    # at W0=8 (smooth, large; measured AD 47.999 vs FD 48.024); AD~FD to grid
    # accuracy. tol=1e-2 is the honest FD
    # band for a linear-interp estimate of a smooth derivative (measured ratio
    # within ~1e-3; margin to 1e-2).
    Case(id="KingProfile.r_t", direction="params->IC",
         fn=_king_r_t, param="W0", theta0=8.0, reduce=jnp.sum,
         expect="consistent", tol=1e-2),
    # --- Michie-King anisotropic profile + DF (Task 1.3) ---
    Case(id="MichieProfile.sample_positions", direction="params->IC",
         fn=_michie_positions_W0, param="W0", theta0=7.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # Michie r_t: now differentiable (Task 1.2b). AD 78.89 vs FD 79.26 at W0=7
    # (|ratio-1|=4.7e-3); tol=1e-2 is the linear-interp FD band (matches King r_t).
    Case(id="MichieProfile.r_t", direction="params->IC",
         fn=_michie_r_t, param="W0", theta0=7.0, reduce=jnp.sum,
         expect="consistent", tol=1e-2),
    Case(id="MichieVelocityDF.sample_velocities", direction="params->IC",
         fn=_michie_velocities_W0, param="W0", theta0=7.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # Michie density-OBSERVABLE channel in r_c (Task 4.2a): log rho(r=1.5) vs r_c.
    # Fills the registry gap (had sample_positions(W0), not r_c, not a density obs).
    # AD=2.209859e+0 vs FD=2.209859e+0 (|ratio-1|=1.3e-8), closed-form. tol=1e-5.
    # Replaces the r_c part of test_michie_physics.py::test_grad_profile_observable.
    Case(id="MichieProfile.density[log rho(r)]", direction="params->summary",
         fn=_michie_log_density_rc, param="r_c", theta0=1.0, reduce=identity_sum,
         expect="consistent", tol=1e-5),
    # Michie density-OBSERVABLE channel in W0 (Task 4.2b review-fix): log rho(r=1.5) vs W0.
    # The W0 part of the density() formula is a DIFFERENT code path than the W0 sampler
    # case, so it needs its own pin. AD=2.573508e-2 vs FD=2.573507e-2 (|ratio-1|=2.3e-7),
    # closed-form. tol=1e-5. Completes the migration of test_grad_profile_observable.
    Case(id="MichieProfile.density[log rho(r)]", direction="params->summary",
         fn=_michie_log_density_W0, param="W0", theta0=7.0, reduce=identity_sum,
         expect="consistent", tol=1e-5),
    # --- EFF profile + Eddington DF (Task 1.3) ---
    # EFF positions: differentiable in the slope gamma (with a gamma=2.01 near-divergent
    # edge -- unguarded, samples cleanly) AND in the prescribed truncation radius r_t.
    Case(id="EFFProfile.sample_positions", direction="params->IC",
         fn=_eff_positions_gamma, param="gamma", theta0=3.0, reduce=mean_radius,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("gamma=2.01", 2.01),)),
    Case(id="EFFProfile.sample_positions", direction="params->IC",
         fn=_eff_positions_rt, param="r_t", theta0=10.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # EFF POSITION-sampler scale-radius channel `a` (Task 4.2b review-fix): gamma=3, r_t=10.
    # Distinct from EFFVelocityDF.sample_velocities(a) (velocity observable) -- this is the
    # position inverse-CDF in a. AD=1.143671e+0 vs FD=1.143689e+0 (|ratio-1|=1.5e-5). tol=1e-3.
    # Lets test_profile_gradients.py::test_eff_grad_a migrate (closes the interlock skip).
    Case(id="EFFProfile.sample_positions", direction="params->IC",
         fn=_eff_positions_a, param="a", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # EFF Eddington velocity DF: differentiable in gamma (gamma=5 mild-truncation
    # ~virial point; the gamma=3 default is documented ~8% sub-virial, not a hazard).
    Case(id="EFFVelocityDF.sample_velocities", direction="params->IC",
         fn=_eff_velocities_gamma, param="gamma", theta0=5.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # EFF DF scale-radius channel `a` (Task 4.2a): gamma=5 (virial point), r_t=10,
    # vary a (flows through positions + Eddington-DF normalisation). AD=-3.160765e-1
    # vs FD=-3.160774e-1 (|ratio-1|=2.8e-6), clean. tol=1e-3 (Eddington-table band).
    # Replaces test_df_gradients.py::TestEFFDFGradients `a` part.
    Case(id="EFFVelocityDF.sample_velocities", direction="params->IC",
         fn=_eff_velocities_a, param="a", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # --- build_spatial_ic end-to-end (Task 1.4): the headline assembly path ---
    # profile x DF -> sample -> COM-center -> virial-scale to Q=0.5, all in r_h.
    # positions channel (COM centering preserves the radial scaling with r_h):
    Case(id="build_spatial_ic[Plummer]", direction="params->IC",
         fn=_build_spatial_ic_rh, param="r_h", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # velocities channel (virial scaling couples r_h into the speeds):
    Case(id="build_spatial_ic[Plummer].velocities", direction="params->IC",
         fn=_build_spatial_ic_rh_velocities, param="r_h", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # --- IMF samplers (params->IC) + mass-function summary (Task 1.5) ---
    # PowerLawIMF.ppf Salpeter single-segment: closed-form inverse CDF, tol=1e-5.
    # alpha=1.0 is the branch-limited point (AD=0 vs live FD) -> known_blocked edge;
    # alpha=0.999 is FD-consistent (ratio 1.0000000).
    Case(id="PowerLawIMF.ppf[Salpeter]", direction="params->IC",
         fn=_powerlaw_ppf_alpha, param="alpha", theta0=2.35, reduce=identity_sum,
         expect="consistent", tol=1e-5,
         edges=(EdgeConfig("alpha=0.999", 0.999),
                EdgeConfig("alpha=1.0", 1.0, expect="known_blocked"))),
    # PowerLawIMF.sample (reparam ppf), reduced by mean_mass over the sampled set.
    Case(id="PowerLawIMF.sample[Salpeter]", direction="params->IC",
         fn=_powerlaw_sample_alpha, param="alpha", theta0=2.35, reduce=mean_mass,
         expect="consistent", tol=1e-3),
    # ChabrierIMF.ppf (Newton solver): differentiable in m_c, sigma, alpha.
    Case(id="ChabrierIMF.ppf", direction="params->IC",
         fn=_chabrier_ppf_mc, param="m_c", theta0=0.08, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    # H6 BOUNDARY probe (Task 4.2a fix): d(ppf)/d(m_c) with u pinned to the m_min floor
    # (_H6_U=1e-12), the Newton clamp jnp.clip(m_new, m_min, m_max) (chabrier.py:371)
    # suspect. This is now a STANDALONE Case using the dedicated _chabrier_ppf_mc_boundary
    # closure -- the previous EdgeConfig("u->m_min[H6]", 0.08) was a NO-OP: the edge
    # mechanism (core.py audit_entry_point) only overrides theta/tol/expect, it always
    # calls case.fn, so the edge reused the baseline _chabrier_ppf_mc (full _U draw) at
    # theta=0.08==theta0 and emitted a DUPLICATE row that never ran the boundary probe.
    # MEASURED BENIGN: AD=9.343597e-11 vs FD=9.346690e-11 (|ratio-1|=3.3e-4) -- the
    # gradient is LIVE and FD-consistent even at the m_min floor; the Newton clamp does
    # NOT zero it. |AD|=9.34e-11 is genuinely tiny (the sample is pinned at the floor, so
    # m_c barely moves it) but strictly non-zero and FD-matched -- the teeth are in the
    # ratio, not the magnitude. We therefore set eps=1e-12 (below |AD|) so this live-but-
    # tiny gradient classifies clean rather than tripping the eps=1e-9 silent-zero default
    # (a blocked gradient would give |ratio-1|~1, not a near-perfect match). tol=1e-3.
    Case(id="ChabrierIMF.ppf[H6 boundary]", direction="params->IC",
         fn=_chabrier_ppf_mc_boundary, param="m_c", theta0=0.08, reduce=identity_sum,
         expect="consistent", tol=1e-3, eps=1e-12),
    Case(id="ChabrierIMF.ppf", direction="params->IC",
         fn=_chabrier_ppf_sigma, param="sigma", theta0=0.69, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    Case(id="ChabrierIMF.ppf", direction="params->IC",
         fn=_chabrier_ppf_alpha, param="alpha", theta0=2.3, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    # Maschberger.ppf: analytical inverse CDF (closed form) in mu, alpha, beta.
    Case(id="Maschberger.ppf", direction="params->IC",
         fn=_maschberger_ppf_mu, param="mu", theta0=0.2, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    Case(id="Maschberger.ppf", direction="params->IC",
         fn=_maschberger_ppf_alpha, param="alpha", theta0=2.3, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    Case(id="Maschberger.ppf", direction="params->IC",
         fn=_maschberger_ppf_beta, param="beta", theta0=1.4, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    # Schechter.ppf: grid-CDF + Newton (BaseIMF.ppf); tol=1e-3 for the grid/Newton band.
    Case(id="Schechter.ppf", direction="params->IC",
         fn=_schechter_ppf_alpha, param="alpha", theta0=2.3, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    # H4 explicit: d(PowerLawIMF.cdf)/d(m_min) at an in-support query (m=0.101).
    # MEASURED ratio 1.0000000 (AD/FD -13.321) -> BENIGN (clip inactive in-support).
    Case(id="PowerLawIMF.cdf[H4]", direction="params->IC",
         fn=_powerlaw_cdf_mmin, param="m_min", theta0=_PL_M_MIN, reduce=identity_sum,
         expect="consistent", tol=1e-5),
    # PowerLawIMF.ppf(m_min) (Task 4.2a): d(ppf)/d(m_min), the lower-support-edge column
    # of the inverse-CDF Jacobian. Salpeter single-segment, fixed alpha=2.35 + fixed _U.
    # AD=1.572808e+3 vs FD=1.572808e+3 (|ratio-1|=2.4e-9), closed-form. tol=1e-5.
    # Replaces test_imf_gradients.py::test_powerlaw_ppf_grad_mmin (registry had
    # cdf[H4](m_min) and ppf(alpha) but not ppf(m_min)).
    Case(id="PowerLawIMF.ppf[m_min]", direction="params->IC",
         fn=_powerlaw_ppf_mmin, param="m_min", theta0=_PL_M_MIN, reduce=identity_sum,
         expect="consistent", tol=1e-5),
    # --- params->summary mass-function channel ---
    # PowerLawIMF.mean_mass (analytic E[m]) in the single-segment slope alpha.
    # alpha=1.0 is branch-limited (Z removable singularity): AD=-52.2 vs FD=-35.6
    # (finite but inconsistent) -> known_blocked; alpha=0.999 is FD-consistent.
    Case(id="PowerLawIMF.mean_mass", direction="params->summary",
         fn=_powerlaw_mean_mass_alpha, param="alpha", theta0=2.35, reduce=identity_sum,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("alpha=0.999", 0.999),
                EdgeConfig("alpha=1.0", 1.0, expect="known_blocked"))),
    # IMFParams log_prob NLL (the 4-segment mass-function Fisher channel) in alpha3.
    # alpha3=1.0 branch-limited: AD=23.8 vs FD=-274.7 (finite) -> known_blocked edge.
    Case(id="IMFParams.log_prob_nll", direction="params->summary",
         fn=_imf_logprob_nll_alpha3, param="alpha3", theta0=2.35, reduce=identity_sum,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("alpha3=1.0", 1.0, expect="known_blocked"),)),
    # --- Differentiable summary diagnostics (Task 1.6) ---
    # CW04 substructure-Q surrogate, audited in the EFF slope gamma (Q is
    # scale-invariant so r_h gives ~0; gamma morphs the shape). AD 9.239e-2 vs
    # FD 9.237e-2 (ratio 1.000183). tol=1e-3 for the kNN-softmin finite-N band.
    Case(id="q_approx[EFF]", direction="params->summary",
         fn=_q_approx_gamma, param="gamma", theta0=3.0,
         reduce=lambda x: jnp.sum(jnp.atleast_1d(x)),
         expect="consistent", tol=1e-3),
    # Soft Lambda_MSR mass-segregation surrogate, audited in the massive-core scale
    # (flows through `dist`) at the DEFAULT beta=0.1, theta0=0.3 -- the config where the
    # old stop_gradient(median) softmin scale omitted ~27% of the gradient (was a
    # ratio~0.73 hazard). With the stop_gradient removed the median scale's derivative
    # flows: AD=-0.2853 vs FD=-0.2852 (ratio 1.0004), finite + non-zero, consistent.
    Case(id="lambda_msr_approx", direction="params->summary",
         fn=_lambda_msr_core_scale, param="core_scale", theta0=0.3,
         reduce=lambda x: jnp.sum(jnp.atleast_1d(x)),
         expect="consistent", tol=1e-3),
    # --- MultiComponentCluster Engine A sample_cluster (Task 2.1) ---
    # params->IC through the lowered-isothermal multimass equilibrium + per-star
    # (component, position, velocity) draw. Three traceable equilibrium drivers.
    # W0 (central potential): AD=4.709932e-1 vs FD=4.710074e-1 (|ratio-1|=3.0e-5).
    # Carries the H2 boundary edge at W0=3 (extended, r_t=4.26): AD=3.672140e-1 vs
    # FD=3.671194e-1 (|ratio-1|=2.6e-4) — the ψ=0 masks + shared r_t are benign
    # (r_t flows ratio 1.000000; min ψ(r_sampled)=0.153 with max radius 3.66 < r_t,
    # so 0/400 sampled stars reach ψ(r)≤0 or hit the max(ψ,0) clamp). 0/400 categorical
    # assignment flips at h=1e-4, so FD is discreteness-free.
    Case(id="MultiComponentCluster.sample_cluster[EngineA]", direction="params->IC",
         fn=_cluster_sample_W0, param="W0", theta0=5.0, reduce=mean_radius,
         expect="consistent", tol=1e-3,
         edges=(EdgeConfig("W0=3[H2-boundary]", 3.0),)),
    # g (truncation sharpness): AD=1.459179e-1 vs FD=1.459341e-1 (|ratio-1|=1.1e-4).
    Case(id="MultiComponentCluster.sample_cluster[EngineA]", direction="params->IC",
         fn=_cluster_sample_g, param="g", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # delta (Gieles-Zocchi mass-segregation exponent w_j=mu_j^-delta):
    # AD=8.716899e-1 vs FD=8.716973e-1 (|ratio-1|=8.5e-6) — the cleanest of the three
    # (delta enters via w_j, no boundary interplay).
    Case(id="MultiComponentCluster.sample_cluster[EngineA]", direction="params->IC",
         fn=_cluster_sample_delta, param="delta", theta0=0.5, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # --- MultiComponentCluster Engine A from_components(w_j) (Task B1) ---
    # The DIRECT velocity-scale-ratio channel (the Fisher target): vary the heavy
    # component's w_j[1] (light w_j[0]=1.0 fixed via jnp.stack), reduce mean_speed.
    # MEASURED 3 seeds at w1=0.8: |ratio-1| in {3.5e-5, 8.1e-5, 4.2e-5}, |AD|~0.11,
    # 0/400 categorical flips at +-h (FD discreteness-free). tol=1e-3 (sibling band).
    Case(id="MultiComponentCluster.from_components[EngineA]", direction="params->IC",
         fn=_cluster_a_w_j, param="w_j", theta0=0.8, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # --- MultiComponentCluster Engine A from_mass_segregation(r_a) (Task B2) ---
    # The OM ANISOTROPY-RADIUS channel of the equipartition constructor (Fisher target).
    # Scalar r_a -> ra_hat_j=(r_a/r_c)*mu_j^eta; reduce mean_speed (anisotropy in vels).
    # MEASURED 5 seeds at r_a=4.0: |ratio-1| in {1.4e-3..1.5e-3}, |AD|~1.3e-2 (live),
    # 0/400 categorical flips at +-h (FD discreteness-free); both FD probes realizable
    # (table build, no Engine-B-style negative-DF raise). tol=3e-3 (~2x measured 1.5e-3
    # band; mean_speed over a stochastic anisotropic sample is FD-noisier than closed form).
    Case(id="MultiComponentCluster.from_mass_segregation[EngineA]", direction="params->IC",
         fn=_cluster_a_r_a, param="r_a", theta0=4.0, reduce=mean_speed,
         expect="consistent", tol=3e-3),
    # --- build_binary_cluster end-to-end (Task B3) ---
    # The flagship IMF->companion(P,q,e)->spatial Fisher path; r_h is the spatial-scale
    # leaf. r_h enters ONLY the assembly (masses/orbits r_h-independent), so the is_binary
    # multiplicity mask is r_h-invariant -> 0/400 flips at +-h BY CONSTRUCTION. compact=True
    # (real-particle ICResult). MEASURED 5 seeds at r_h=1.0: |ratio-1| <= 2.0e-12, |AD|~1.3-2.0
    # (live, +ve), 0/400 mask flips. r_h-LINEAR end-to-end -> machine-exact; tol=1e-5.
    Case(id="build_binary_cluster", direction="params->IC",
         fn=_build_binary_cluster_rh, param="r_h", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    # --- Task B4: binary period distributions ---
    # Pure params->IC period draws (no G/units). Observable mean(log10 P) at the fixed
    # module key; smooth reparam'd ppf -> central FD clean + N-stable. See the B4 block
    # above for the per-seed measured table.
    # (1) SanaOBPeriod.power: p(logP) ∝ (logP)^power inverse-CDF. MEASURED 3 seeds at
    # power=-0.55: |ratio-1| ~ 1.4e-9, |AD|~0.79 (live). tol=1e-5 (>7000x margin).
    Case(id="SanaOBPeriod.sample", direction="params->IC",
         fn=_sana_period_power, param="power", theta0=-0.55, reduce=_log10_mean,
         expect="consistent", tol=1e-5),
    # (2) LogNormalPeriod.mu_log_P: CLOSED-FORM d<log10 P>/dmu = 1 exactly (location
    # shift). MEASURED 3 seeds at mu=4.8: AD=1.000000, |ratio-1| <= 2.7e-12. tol=1e-5.
    Case(id="LogNormalPeriod.sample", direction="params->IC",
         fn=_lognormal_period_mu, param="mu_log_P", theta0=4.8, reduce=_log10_mean,
         expect="consistent", tol=1e-5),
    # (3) LogUniformPeriod.log_P_max: CLOSED-FORM d<log10 P>/d(hi) = <u> ≈ 0.5.
    # MEASURED 3 seeds at hi=8.0: AD~0.49-0.51, |ratio-1| <= 1.5e-12. tol=1e-5.
    Case(id="LogUniformPeriod.sample", direction="params->IC",
         fn=_loguniform_period_logpmax, param="log_P_max", theta0=8.0, reduce=_log10_mean,
         expect="consistent", tol=1e-5),
    # --- Task B5: binary eccentricity distributions ---
    # Pure e_max->ecc draws (bounded [0,1]; no G/units). Observable mean(e) at the fixed
    # module key. See the B5 block above for the per-seed measured table + the Moe
    # circular-flip diagnostic (0 flips at the chosen long-P config).
    # (1) ThermalEccentricity.e_max: CLOSED-FORM d<e>/de_max = <sqrt(u)> -> 2/3.
    # MEASURED 3 seeds at e_max=0.99: AD~0.66-0.67, |ratio-1| <= 1.0e-12. tol=1e-5.
    Case(id="ThermalEccentricity.sample", direction="params->IC",
         fn=_thermal_ecc_emax, param="e_max", theta0=0.99, reduce=_mean_ecc,
         expect="consistent", tol=1e-5),
    # (2) UniformEccentricity.e_max: CLOSED-FORM d<e>/de_max = <u> -> 0.5.
    # MEASURED 3 seeds at e_max=0.9: AD~0.49-0.51, |ratio-1| <= 9.2e-13. tol=1e-5.
    Case(id="UniformEccentricity.sample", direction="params->IC",
         fn=_uniform_ecc_emax, param="e_max", theta0=0.9, reduce=_mean_ecc,
         expect="consistent", tol=1e-5),
    # (3) MoeEccentricity.e_max: FD-target. P=1e8 d, M1=20 Msun -> Roche cap binds on
    # the e_max field; <u^(1/(eta+1))> live (eta=0.873). 0 circular flips at +/-h.
    # MEASURED 3 seeds at e_max=0.99: AD~0.65, |ratio-1| <= 1.2e-12. tol=1e-5.
    Case(id="MoeEccentricity.sample", direction="params->IC",
         fn=_moe_ecc_emax, param="e_max", theta0=0.99, reduce=_mean_ecc,
         expect="consistent", tol=1e-5),
    # (4) LogisticThermalEccentricity.e_max: FD-target. e = blend(P)*e_max*sqrt(u) over
    # logspace(0.5,3.5) periods -> AD~0.33 (blend half-suppresses the thermal scale).
    # MEASURED 3 seeds at e_max=0.99: AD~0.33, |ratio-1| <= 9.7e-13. tol=1e-5.
    Case(id="LogisticThermalEccentricity.sample", direction="params->IC",
         fn=_logistic_thermal_ecc_emax, param="e_max", theta0=0.99, reduce=_mean_ecc,
         expect="consistent", tol=1e-5),
    # --- Task B6: IndependentCompanions.e_max (the FINAL D4 hole) ---
    # Same mean(e) observable as B5's ThermalEccentricity.sample, but routed THROUGH
    # the full IndependentCompanions ASSEMBLY (is_binary Heaviside draw + q->m2 gate +
    # period->a Kepler map + 7-field CompanionElements pytree). Confirms the e_max
    # gradient SURVIVES the assembly graph (a stop_gradient in the connector, or an e
    # field gated by is_binary, would zero it). COMPLEMENTARY to B5's bare-dist case.
    # DISCRETENESS: is_binary depends on the kb-uniform draw + fbin, NOT on e_max, so
    # it is e_max-invariant by construction -> 0 flips at +/-h (MEASURED: n_binary=
    # 2023/4000, flips(+h)=0, flips(-h)=0). e is NOT gated by is_binary, so mean(e)
    # over ALL N is FD-clean (singles carry the full thermal scale; max e_single=0.99).
    # AD ~ <sqrt u> ~ 0.66 -> 2/3 (the thermal location-scale derivative flowing
    # through the assembly). MEASURED (theta0=0.99, h=1e-4, mean(comp.e) over N=4000),
    # 3 seeds:
    #   seed 0: AD=6.681601e-1  FD=6.681601e-1  |ratio-1|=1.0e-12
    #   seed 1: AD=6.667383e-1  FD=6.667383e-1  |ratio-1|=9.6e-13
    #   seed 2: AD=6.624480e-1  FD=6.624480e-1  |ratio-1|=2.4e-13
    # |AD|~0.66 >> eps (live); seed-stable; max |ratio-1| = 1.0e-12. tol=1e-5.
    Case(id="IndependentCompanions.sample", direction="params->IC",
         fn=_independent_companions_e_max, param="e_max", theta0=0.99, reduce=_mean_ecc,
         expect="consistent", tol=1e-5),
    # --- MultiComponentCluster Engine B from_density_profiles (Task 2.2) ---
    # params->IC through the density-defined shared-Psi Eddington/OM build (Poisson
    # quadrature + Eddington inversion, NO ODE) + per-star draw. Realizable headline
    # mix (Plummer halo + EFF core) reused from test_engine_b_physics._headline_model.
    # halo Plummer scale r_h (prescribed-density leaf) -> positions:
    # AD=6.767522e-1 vs FD=6.767957e-1 (|ratio-1|=6.4e-5); f_min_j > 0 at r_h +- h.
    Case(id="MultiComponentCluster.sample_cluster[EngineB]", direction="params->IC",
         fn=_cluster_b_sample_rh, param="r_h", theta0=2.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # core EFF slope gamma (prescribed-density leaf) -> positions:
    # AD=-1.204971e-1 vs FD=-1.205328e-1 (|ratio-1|=3.0e-4 — the widest of the three,
    # sets the tol); 0/400 categorical flips at h; f_min_j > 0 at gamma +- h.
    Case(id="MultiComponentCluster.sample_cluster[EngineB]", direction="params->IC",
         fn=_cluster_b_sample_gamma, param="gamma", theta0=5.0, reduce=mean_radius,
         expect="consistent", tol=1e-3),
    # halo Osipkov-Merritt anisotropy radius r_a_j[0] -> VELOCITIES (anisotropy lives
    # in the speeds): AD=1.768007e-3 vs FD=1.768007e-3 (|ratio-1|=2.7e-9 — essentially
    # exact). |AD|=1.77e-3 is small but >> eps=1e-9 (a live, non-zero gradient). r_a=3.0
    # is well inside the realizable OM regime (f_min_halo=0.085; FD probes r_a +- h both
    # realizable). Core stays isotropic (r_a=inf).
    Case(id="MultiComponentCluster.sample_cluster[EngineB]", direction="params->IC",
         fn=_cluster_b_sample_ra, param="r_a", theta0=3.0, reduce=mean_speed,
         expect="consistent", tol=1e-3),
    # --- Binary entry points (Task 2.3) ---
    # (a) KeplerElements.to_state — elements -> Cartesian position, vary e.
    # Fixed-iteration Newton Kepler solve + trig rotation: closed-form-ish, FD-
    # consistent to machine precision (AD=4.439570e-1 vs FD=4.439570e-1, ratio
    # 1.0000000 at e=0.5). The e=0.999 EDGE (near-parabolic, slowest Newton
    # convergence) is ALSO machine-precision clean (AD=9.995240e-1 vs FD=
    # 9.995240e-1, ratio 1.0000000) — the max(1-e²,1e-12) guard is inactive
    # (1-e²=2.0e-3 at e=0.999). tol=1e-5 (closed-form band). We audit POSITION
    # (cleaner near e->1); the velocity channel at e=0.999 is also central-FD clean
    # (AD=-7.484823e-2 vs FD=-7.484823e-2, ratio 1.0000000) and only softens under a
    # coarse one-sided FD (forward h=1e-3 -> ~0.99993) from the d/de[sqrt(1-e²)]
    # truncation-term curvature — NOT blocked, just FD-noisier (see header note).
    Case(id="KeplerElements.to_state", direction="params->IC",
         fn=_kepler_to_state_e, param="e", theta0=0.5, reduce=mean_radius,
         expect="consistent", tol=1e-5,
         edges=(EdgeConfig("e=0.999", 0.999),)),
    # (a2) to_state a-column: AD=9.163137e-1 vs FD=9.163137e-1 (|ratio-1|=2.4e-13),
    # machine-exact. tol=1e-5. Replaces test_binary_physics.py::
    # TestKeplerTransformGradients::test_to_state_grad_wrt_a.
    Case(id="KeplerElements.to_state", direction="params->IC",
         fn=_kepler_to_state_a, param="a", theta0=1.5, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    # (a3) to_state M0-column: AD=3.144025e-1 vs FD=3.144025e-1 (|ratio-1|=2.1e-9),
    # machine-exact. tol=1e-5. Replaces test_to_state_grad_wrt_M0.
    Case(id="KeplerElements.to_state", direction="params->IC",
         fn=_kepler_to_state_M0, param="M0", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    # (a4) from_state (state->elements inverse): d(recovered a)/d(v-scale).
    # AD=2.365317e+0 vs FD=2.365317e+0 (|ratio-1|=8.0e-8), clean. tol=1e-4 (the
    # from_state vector-algebra band, matching the scattered test's rel<1e-4).
    # Replaces test_from_state_grad_wrt_velocity_scale.
    Case(id="KeplerElements.from_state", direction="params->IC",
         fn=_kepler_from_state_vscale, param="v_scale", theta0=1.0, reduce=identity_sum,
         expect="consistent", tol=1e-4),
    # (b) resolve_binary_components — binary->spatial connector, vary a (N=50, all
    # binaries). Pure smooth orbital placement (sanitization inactive): AD=8.437447e-2
    # vs FD=8.437447e-2 (ratio 1.0000000008). tol=1e-5 (closed-form band).
    Case(id="resolve_binary_components", direction="params->IC",
         fn=_resolve_binary_a, param="a", theta0=0.5, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    # (b2) resolve_binary_components MIXED is_binary (Fisher-integrity check). 25
    # binaries + 25 singles activates the sanitization path (assembly.py:85-87); the
    # single slots give d/da=0 (FINITE, not NaN), binary slots a live gradient. NaN-free
    # jacfwd confirmed. AD=3.019178e-2 vs FD=3.019178e-2 (ratio 1.0000000002). tol=1e-5.
    Case(id="resolve_binary_components[mixed]", direction="params->IC",
         fn=_resolve_binary_a_mixed, param="a", theta0=0.5, reduce=mean_radius,
         expect="consistent", tol=1e-5),
    # (c) MoeCompanions.sample — smooth grid-CDF (P,q,e) draw mixed with a DISCRETE
    # is_binary Heaviside mask + m2 gate. Audited via <e> (mask-independent P->e
    # coupling) wrt a scalar m1-multiplier. CLASSIFIED consistent/clean by MEASUREMENT:
    # 0/400 is_binary flips at s=1±h (the discrete selection injects no FD noise), and
    # <e> AD=-1.023290e-4 vs FD=-1.023290e-4 (ratio 1.0000000, |AD|=1.02e-4 >> eps).
    # The grid-CDF reparameterization makes the coupling gradient flow; the mask does
    # NOT block it (even <m2=m1·q> is ratio 1.0000000 since 0 mask flips keeps the gate
    # locally smooth). tol=1e-3 (the grid-CDF inverse-interp band).
    Case(id="MoeCompanions.sample", direction="params->IC",
         fn=_moe_companions_mean_e, param="m1_scale", theta0=1.0,
         reduce=lambda x: jnp.sum(jnp.atleast_1d(x)),
         expect="consistent", tol=1e-3),
    # --- Rotation kinematic overlays (Task 2.4) ---
    # (a) apply_solid_body_rotation in omega: EXACTLY LINEAR overlay, AD = closed-form
    # mean(|axis × r|) = 1.1226200071 (verified <1e-12); AD=1.122620e+00 vs FD=
    # 1.122620e+00 (|ratio-1|=6.9e-13, machine-exact). tol=1e-5 (closed-form band).
    Case(id="apply_solid_body_rotation", direction="params->IC",
         fn=_solid_body_omega, param="omega", theta0=0.5, reduce=mean_speed,
         expect="consistent", tol=1e-5),
    # (b) apply_differential_rotation in v_peak: LINEAR in v_peak (scales the curve),
    # AD=7.635384e-01 vs FD=7.635384e-01 (|ratio-1|=5.8e-13, machine-exact). tol=1e-5.
    Case(id="apply_differential_rotation", direction="params->IC",
         fn=_differential_v_peak, param="v_peak", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-5),
    # (c) apply_differential_rotation in R_peak: NONLINEAR (R_peak in both the (R/R_peak)
    # prefactor and exp(1-R/R_peak)) — the real teeth. AD=-4.356504e-02 vs FD=
    # -4.356503e-02 (|ratio-1|=5.5e-8, the nonlinear central-FD band). Negative grad:
    # R_peak0=1.0 below the position-weighted mean R, so widening the peak lowers the
    # inner-weighted mean speed. tol=1e-5 (with margin to the 5.5e-8 residual).
    Case(id="apply_differential_rotation", direction="params->IC",
         fn=_differential_R_peak, param="R_peak", theta0=1.0, reduce=mean_speed,
         expect="consistent", tol=1e-5),
    # --- Tier 3: binned-kinematic Fisher path (Task 3.1) ---
    # sigma_1d(r): params->summary through build_spatial_ic + the frozen-edge binner.
    # h_rel=1e-5 keeps the FD probe off the r=8 edge-crossing (0/2000 flips), so the
    # central FD matches AD to machine precision: AD=-3.345969e+00 vs FD=-3.345969e+00
    # (ratio 1.0000000). At the default h_rel=1e-4 one boundary star (#821, r=7.99977)
    # crosses out-of-range and corrupts the coarse FD to ratio 0.86 — a coarse-FD
    # edge-straddle, NOT an autodiff bug (per-bin ratios all 1.0; ratio stable in
    # h<=1e-5 and across N). tol=1e-3. Mutation-checked (stop_gradient -> AD=0).
    Case(id="binned_sigma1d[Plummer]", direction="params->summary",
         fn=_binned_sigma1d_rh, param="r_h", theta0=1.0, reduce=identity_sum,
         expect="consistent", tol=1e-3, h_rel=1e-5),
    # beta(r): the anisotropy-Fisher headline channel. Osipkov-Merritt DF, vary r_a
    # (>> Merritt bound 0.75a~0.575 for r_h=1). Uses the ENGINE-DEFAULT h_rel=1e-4 (NOT
    # the sigma case's 1e-5 override): r_a scales only the VELOCITIES, so positions and
    # bin membership are r_a-invariant and the sigma case's edge-straddle cannot occur
    # (0/2000 bin crossings at +-h by construction). Measured at the default h_rel=1e-4:
    # AD=-1.000759e+00 vs FD=-1.000754e+00 (ratio 1.0000054), clean within tol=1e-3.
    Case(id="binned_sigma_beta[Plummer+OM]", direction="params->summary",
         fn=_binned_beta_ra, param="r_a", theta0=_BK_OM_R_A, reduce=identity_sum,
         expect="consistent", tol=1e-3),
    # N(r) number-density Fisher channel — the MODEL side. The Fisher gradient lives
    # in the expected per-shell occupancy p_k(r_h) = F(r_{k+1}) - F(r_k) via the new
    # PlummerProfile.enclosed_mass_fraction CDF (a Poisson likelihood differentiates
    # mu_k = N*p_k, NOT the frozen counts). Closed-form analytic -> machine-exact:
    # AD=-2.691265e-2 vs FD=-2.691265e-2 (ratio 1.0000000). tol=1e-5 (closed-form band).
    Case(id="PlummerProfile.enclosed_mass_fraction[N(r) model]",
         direction="params->summary",
         fn=_model_pk_rh, param="r_h", theta0=1.0, reduce=identity_sum,
         expect="consistent", tol=1e-5),
    # N(r) DATA side — PINNED known-limitation: the binned count is correctly NON-diff.
    # binned_number_density is a sum of frozen-edge indicators, so its autodiff gradient
    # is identically 0 (AD=0, FINITE — the right answer; the N(r) gradient is in the
    # model p_k above). The FD is a NONZERO discrete bin-crossing step (FD=-5.000e3 @
    # r_h=1) the audit treats as frozen/out-of-scope. known_blocked passes iff AD is
    # finite (AD=0 qualifies); known_zero would WRONGLY flag this a hazard since the FD
    # step is far from 0. This PINS that frozen data is out-of-scope and the Fisher
    # gradient lives in the model — the arc's headline principle.
    Case(id="binned_number_density[data, pinned non-diff]",
         direction="params->summary",
         fn=_binned_number_density_rh, param="r_h", theta0=1.0, reduce=identity_sum,
         expect="known_blocked", tol=1e-3),
    # --- cluster convenience builders (cluster-builders arc, measured 2026-06-14) ---
    Case(id="build_cluster[Plummer]", direction="params->IC", fn=_bc_plummer_rh,
         param="r_h", theta0=1.0, reduce=mean_radius, tol=1e-5),         # |ratio-1|=2.7e-13
    Case(id="build_cluster[Plummer+OM]", direction="params->IC", fn=_bc_plummer_om,
         param="anisotropy_radius", theta0=0.7, reduce=mean_speed, tol=1e-3),  # 4.7e-6
    Case(id="build_cluster[Plummer+rotation]", direction="params->IC", fn=_bc_plummer_omega,
         param="omega", theta0=0.3, reduce=mean_speed, tol=1e-4),       # |ratio-1|=4.2e-9
    Case(id="build_king_cluster", direction="params->IC", fn=_bk_rc,
         param="r_c", theta0=1.0, reduce=mean_radius, tol=1e-5),        # |ratio-1|=4.9e-12
    Case(id="build_eff_cluster", direction="params->IC", fn=_beff_gamma,
         param="gamma", theta0=3.0, reduce=mean_radius, tol=1e-3),      # |ratio-1|=1.8e-5
    # tol=1e-3 is the tightest margin (2.7x over measured): the Michie W0 channel runs
    # through the diffrax ODE solve, so its FD floor tracks the solver tolerance / n_ode_points
    # default. If those defaults change this case is the most likely to flicker (King sidesteps
    # this by auditing the non-ODE r_c channel; Michie has no comparably clean non-ODE scale).
    Case(id="build_michie_cluster", direction="params->IC", fn=_bmich_W0,
         param="W0", theta0=7.0, reduce=mean_radius, tol=1e-3),         # |ratio-1|=3.7e-4
    Case(id="build_limepy_cluster", direction="params->IC", fn=_blim_W0,
         param="W0", theta0=5.0, reduce=mean_radius, tol=1e-3),         # |ratio-1|=7.6e-5
    Case(id="build_cluster_from_params[ClusterParams]", direction="params->IC", fn=_bcfp_rh,
         param="r_h", theta0=1.0, reduce=mean_radius, tol=1e-5),        # |ratio-1|=2.7e-13
    # --- ZAMS stellar relations (Tout+1996) — P3 ---
    # Forward L/R/T_eff/log g differentiable in mass at the interior MS point M=5;
    # the inverse audits dM/dL at an IN-RANGE L_target (~530.13 Lsun -> M~5), NOT a
    # clipped plateau (M1). All closed-form / smooth-Newton -> tol=1e-5.
    Case(id="zams_luminosity", direction="params->summary", fn=_zams_luminosity_mass,
         param="mass", theta0=_ZAMS_MASS, reduce=identity_sum, tol=1e-5),   # |ratio-1|=4.9e-9
    Case(id="zams_radius", direction="params->summary", fn=_zams_radius_mass,
         param="mass", theta0=_ZAMS_MASS, reduce=identity_sum, tol=1e-5),   # |ratio-1|=9.1e-10
    Case(id="zams_effective_temperature", direction="params->summary", fn=_zams_teff_mass,
         param="mass", theta0=_ZAMS_MASS, reduce=identity_sum, tol=1e-5),   # |ratio-1|=1.4e-9
    Case(id="zams_surface_gravity", direction="params->summary", fn=_zams_logg_mass,
         param="mass", theta0=_ZAMS_MASS, reduce=identity_sum, tol=1e-5),   # |ratio-1|=1.1e-9
    Case(id="inverse_zams_luminosity", direction="params->summary", fn=_inverse_zams_L,
         param="L_target", theta0=_ZAMS_INVERSE_L, reduce=identity_sum, tol=1e-5),  # |ratio-1|=1.9e-9
    # --- Dispersion forward models (Phase 0 Task 8) ---
    # jeans_dispersion (3-D anisotropic Jeans) + project_dispersion (B&M82 LOS/PM); all
    # FD-consistent at interior radii r in [0.5,2.0] (Task 7). tol=1e-3 (smooth quadratures).
    Case(id="jeans_dispersion[Plummer+OM]", direction="params->summary",
         fn=_jeans_dispersion_om_r_a, param="r_a", theta0=_DISP_R_A,
         reduce=identity_sum, expect="consistent", tol=1e-3),       # |ratio-1|=1e-8
    Case(id="jeans_dispersion[Plummer]", direction="params->summary",
         fn=_jeans_dispersion_M, param="M", theta0=400.0,
         reduce=identity_sum, expect="consistent", tol=1e-3),       # |ratio-1|<1e-9
    Case(id="project_dispersion[Plummer+OM]", direction="params->summary",
         fn=_project_dispersion_om_r_a, param="r_a", theta0=_DISP_R_A,
         reduce=identity_sum, expect="consistent", tol=1e-3),       # |ratio-1|=1e-8
    Case(id="project_dispersion[Plummer+OM].pm_t", direction="params->summary",
         fn=_project_dispersion_om_r_a_pmt, param="r_a", theta0=_DISP_R_A,
         reduce=identity_sum, expect="consistent", tol=1e-3),       # |ratio-1|=1e-8
    # df_moment_dispersion (exact Michie DF 2nd-moment quadrature); M-gradient only (W0
    # deferred). FD-consistent at interior radii r in [0.5,2.0]. tol=1e-3 (smooth quadrature).
    Case(id="df_moment_dispersion[Michie]", direction="params->summary",
         fn=_df_moment_dispersion_M, param="M", theta0=400.0,
         reduce=identity_sum, expect="consistent", tol=1e-3),       # |ratio-1|~4.2e-4
]
