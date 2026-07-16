"""Phase 1: FK12 multi-freefall star placement (design 2026-07-16, gate AC-IC7).

The placement PMF is ``p_⋆ ∝ w(s_turb) · e^{(3/2)·s_total}`` — the normalized FK12 Eq. 7
integrand (t_ff ∝ ρ^{-1/2}, Eq. 8; ε/φ_t cancel in the PMF), with the collapse-eligibility
gate w on the BM19 transition s_t (the s_t-for-s_crit substitution; per-paper note
federrath-klessen-2012). The former free f_sub knob is replaced by two DERIVED quantities:
``tail_star_fraction`` (hard, from the actual placement PMF — the f_sub successor) and
``collapse_eligible_fraction`` (smooth, ungated — the differentiable analytic hook).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax import PlummerProfile

pytestmark = [pytest.mark.experimental, pytest.mark.unit]


# ── the PMF itself ──


def test_pmf_is_freefall_weighted():
    """On a uniform eligibility field (w→1), p ∝ ρ^{3/2}: a cell with 4× the density
    gets 4^{3/2}=8× the probability."""
    from gravoturb.realization.placement import multi_freefall_pmf

    s = jnp.log(jnp.array([1.0, 4.0]))          # densities 1 and 4
    s_t = -1e3                                   # everything deep inside the tail: w≈1
    p = multi_freefall_pmf(s, s_t, mask_sharpness=8.0, s_density=s)
    assert p.shape == s.shape
    np.testing.assert_allclose(float(p[1] / p[0]), 8.0, rtol=1e-10)
    np.testing.assert_allclose(float(p.sum()), 1.0, rtol=1e-12)


def test_pmf_gate_suppresses_ineligible_cells():
    """With a sharp mask, cells below s_t get ~zero weight even when dense in s_total."""
    from gravoturb.realization.placement import multi_freefall_pmf

    s_turb = jnp.array([-2.0, 2.0])              # below / above the transition
    s_t = 0.0
    s_total = jnp.array([3.0, 3.0])              # equally dense in TOTAL density
    p = multi_freefall_pmf(s_turb, s_t, mask_sharpness=50.0, s_density=s_total)
    assert float(p[0]) < 1e-8
    np.testing.assert_allclose(float(p[1]), 1.0, atol=1e-8)


def test_pmf_separates_eligibility_from_placement_density():
    """The gate reads s_turb (LOCAL overdensity); the ρ^{3/2} reads s_total (envelope
    included) — the envelope/substructure decoupling carried over from two_population."""
    from gravoturb.realization.placement import multi_freefall_pmf

    s_turb = jnp.array([1.0, 1.0])               # equally eligible
    s_total = jnp.array([0.0, jnp.log(4.0)])     # envelope makes cell 2 4× denser
    p = multi_freefall_pmf(s_turb, s_t=0.0, mask_sharpness=8.0, s_density=s_total)
    np.testing.assert_allclose(float(p[1] / p[0]), 8.0, rtol=1e-10)


# ── the derived tail fraction ──


def test_collapse_eligible_fraction_monotone_in_alpha():
    """Heavier-tailed clouds (lower alpha) put more star-forming weight in the
    collapsing tail: collapse_eligible_fraction decreases with alpha (the PDF-grounded
    direction, matching BM19 f_dense). The mach-response direction is regime-dependent
    (AC8: ∂f_dense/∂ℳ < 0 at the fiducial), so it is characterized by the printed
    AC-IC7 table, not asserted here."""
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import collapse_eligible_fraction

    def fel(mach, alpha, seed=0):
        fld = build_turbulent_field(mach, 0.5, alpha, 3.0, (24, 24, 24),
                                    jax.random.PRNGKey(seed))
        return float(collapse_eligible_fraction(fld.s, fld.s_t, mask_sharpness=8.0,
                                                s_density=fld.s))

    assert fel(mach=8.0, alpha=1.5) > fel(mach=8.0, alpha=2.5)
    assert 0.0 < fel(mach=8.0, alpha=1.8) < 1.0


def test_tail_star_fraction_is_the_placement_pmf_fraction():
    """tail_star_fraction = Σ_{s>s_t} p under the ACTUAL gated PMF — near 1 for a sharp
    gate, and strictly larger than collapse_eligible_fraction (review finding: the two
    quantities differ by >2x at the fiducial; they must never be conflated again)."""
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import (
        collapse_eligible_fraction,
        multi_freefall_pmf,
        tail_star_fraction,
    )

    fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, (24, 24, 24), jax.random.PRNGKey(0))
    f_tail = float(tail_star_fraction(fld.s, fld.s_t, mask_sharpness=8.0))
    f_elig = float(collapse_eligible_fraction(fld.s, fld.s_t, mask_sharpness=8.0))
    # exact cross-check against the PMF itself
    p = multi_freefall_pmf(fld.s, fld.s_t, mask_sharpness=8.0)
    direct = float(jnp.sum(jnp.where(fld.s > fld.s_t, p, 0.0)))
    np.testing.assert_allclose(f_tail, direct, rtol=1e-12)
    assert f_tail > 0.8            # sharp gate: stars land in eligible cells
    assert f_tail > f_elig + 0.1   # the two quantities are materially different


def test_collapse_eligible_fraction_differentiable_in_s_t():
    """d collapse_eligible_fraction / d s_t is finite and negative (raising the
    threshold empties the tail) — the smooth analytic hook."""
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import collapse_eligible_fraction

    fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, (16, 16, 16), jax.random.PRNGKey(0))
    g = jax.grad(lambda st: collapse_eligible_fraction(fld.s, st, mask_sharpness=8.0,
                                                       s_density=fld.s))(fld.s_t)
    assert jnp.isfinite(g) and float(g) < 0.0


def test_collapse_eligible_fraction_ad_matches_fd():
    """AD-vs-FD on the smooth analytic quantity (design policy): d/d s_t and
    d/d mask_sharpness agree with central finite differences to <1e-6 rel."""
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import collapse_eligible_fraction

    fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, (16, 16, 16), jax.random.PRNGKey(0))

    for argname, val in [("s_t", float(fld.s_t)), ("mask_sharpness", 8.0)]:
        if argname == "s_t":
            f = lambda x: collapse_eligible_fraction(fld.s, x, mask_sharpness=8.0)
        else:
            f = lambda x: collapse_eligible_fraction(fld.s, fld.s_t, mask_sharpness=x)
        ad = float(jax.grad(f)(val))
        h = 1e-5 * max(abs(val), 1.0)
        fd = (float(f(val + h)) - float(f(val - h))) / (2 * h)
        np.testing.assert_allclose(ad, fd, rtol=1e-6, err_msg=argname)


# ── sampling behaviour ──


def test_multi_freefall_stars_trace_denser_gas_than_rho_placement():
    """ρ^{3/2} weighting concentrates stars in denser cells than plain ∝ρ placement:
    the mean log-density at star cells must be strictly higher."""
    from gravoturb.realization.pipeline import build_turbulent_field
    from gravoturb.realization.placement import (
        sample_positions,
        sample_positions_multi_freefall,
    )

    fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, (32, 32, 32), jax.random.PRNGKey(3))

    def mean_s_at_stars(pos):
        idx = jnp.clip((pos * 32).astype(int), 0, 31)
        return float(jnp.mean(fld.s[idx[:, 0], idx[:, 1], idx[:, 2]]))

    pos_mff = sample_positions_multi_freefall(
        fld.s, fld.s_t, 8.0, 4000, jax.random.PRNGKey(7))
    pos_rho = sample_positions(fld.s, fld.s_t, 8.0, 0.0, 4000, jax.random.PRNGKey(7))
    assert mean_s_at_stars(pos_mff) > mean_s_at_stars(pos_rho) + 0.1


def test_envelope_control_reproduces_rho_15_weighted_plummer():
    """AC-IC7(a) in miniature: turbulence OFF ⇒ placement ∝ ρ_env^{3/2}. With w≈1
    everywhere (s_t → −∞ on a zero field), the sampled radial distribution must match
    an INDEPENDENT numpy reference draw (shared oracle in gravoturb.validation.oracles:
    numpy ρ_env^{3/2} cell weights + numpy jitter — no gravoturb realization code in the
    oracle path beyond the FREEFALL_EXPONENT constant): two-sample KS < 0.015 at
    N=M=40000."""
    from gravoturb.realization.envelope import apply_spherical_envelope
    from gravoturb.realization.placement import (
        FREEFALL_EXPONENT,
        sample_positions_multi_freefall,
    )
    from gravoturb.validation.oracles import (
        ks_two_sample,
        rho_weighted_reference_positions,
    )

    shape, box = (48, 48, 48), 4.0
    prof = PlummerProfile(r_h=0.5)
    s_turb = jnp.zeros(shape)
    s_tot = apply_spherical_envelope(s_turb, prof, box)
    pos = sample_positions_multi_freefall(
        s_turb, -1e3, 8.0, 40000, jax.random.PRNGKey(11),
        box_size=box, s_density=s_tot)
    r_star = np.linalg.norm(np.asarray(pos) - box / 2, axis=1)

    # independent numpy reference: same cell geometry, weights from profile.density
    ref = rho_weighted_reference_positions(
        prof, shape, box, FREEFALL_EXPONENT, 40000, np.random.default_rng(2026))
    r_ref = np.linalg.norm(ref, axis=1)

    ks = ks_two_sample(r_star, r_ref)
    assert ks < 0.015, f"two-sample KS {ks:.4f}"


# ── builder integration + spec guards ──


def _specs(placement, f_sub=None, beta=3.0):
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

    return dict(
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=beta),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0,
                              shape=(16, 16, 16)),
        velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
        composition=(CompositionSpec(placement=placement, f_sub=f_sub)
                     if f_sub is not None else CompositionSpec(placement=placement)),
    )


def test_composition_spec_mode_guards():
    from gravoturb.specs import CompositionSpec

    CompositionSpec()  # default: multi_freefall, no f_sub
    CompositionSpec(placement="two_population", f_sub=0.3)
    with pytest.raises(ValueError, match="f_sub"):
        CompositionSpec(placement="multi_freefall", f_sub=0.3)  # f_sub is derived here
    with pytest.raises(ValueError, match="f_sub"):
        CompositionSpec(placement="two_population")  # legacy mode needs f_sub
    with pytest.raises(ValueError, match="placement"):
        CompositionSpec(placement="nonsense")


def test_builder_multi_freefall_reports_derived_fraction():
    import jax.numpy as jnp
    from gravoturb.cluster import build_cluster_ic
    from jaxstro.units import STELLAR

    ic = build_cluster_ic(jnp.ones(400), **_specs("multi_freefall"),
                          G=STELLAR.G, key=jax.random.PRNGKey(0))
    assert ic.tail_star_fraction is not None
    assert ic.collapse_eligible_fraction is not None
    assert 0.0 < float(ic.collapse_eligible_fraction) < float(ic.tail_star_fraction) <= 1.0
    assert np.all(np.isfinite(np.asarray(ic.positions)))


def test_builder_two_population_unchanged():
    """Legacy mode still runs and reports no derived fractions."""
    import jax.numpy as jnp
    from gravoturb.cluster import build_cluster_ic
    from jaxstro.units import STELLAR

    ic = build_cluster_ic(jnp.ones(400), **_specs("two_population", f_sub=0.3),
                          G=STELLAR.G, key=jax.random.PRNGKey(0))
    assert ic.tail_star_fraction is None
    assert ic.collapse_eligible_fraction is None
    assert abs(float(ic.Q_virial) - 0.5) < 1e-2
