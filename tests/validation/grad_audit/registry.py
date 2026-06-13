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
    PlummerProfile,
    PlummerVelocityDF,
    PowerLawIMF,
    build_spatial_ic,
)
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.diagnostics.q_approx import q_approx
from progenax.diagnostics.segregation_approx import lambda_msr_approx
from progenax.imf.differentiable import log_prob_masses
from progenax.imf.params import IMFParams
from progenax.imf.smooth import Schechter
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


def _eff_velocities_gamma(gamma):
    positions = EFFProfile(a=1.0, gamma=gamma, r_t=10.0).sample_positions(_MASSES, _KEY_POS)
    df = EFFVelocityDF(a=1.0, gamma=gamma, r_t=10.0)
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
    # EFF Eddington velocity DF: differentiable in gamma (gamma=5 mild-truncation
    # ~virial point; the gamma=3 default is documented ~8% sub-virial, not a hazard).
    Case(id="EFFVelocityDF.sample_velocities", direction="params->IC",
         fn=_eff_velocities_gamma, param="gamma", theta0=5.0, reduce=mean_speed,
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
         expect="consistent", tol=1e-3,
         # H6 boundary probe: u pinned to the m_min floor (Newton clamp suspect).
         # MEASURED BENIGN (live gradient, ratio ~1) -> consistent, no hazard_id.
         edges=(EdgeConfig("u->m_min[H6]", 0.08),)),
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
]
