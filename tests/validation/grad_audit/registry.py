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
]
