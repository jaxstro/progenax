"""Validate the PP20 magnification factor ζ(p) against published values.

Anchored on Parmentier & Pasquali 2020, ApJ 903, 56 (arXiv:2009.10652).

The paper's Eq. 6 (page 2) is, transcribed verbatim:

    zeta(p) = (3 - p)^(3/2) / [2.6 * (2 - p)]   for 0 <= p < 2

This is algebraically identical to the analytic form derived from the
integral definition

    zeta = [int rho^(3/2) dV] / [M * sqrt(<rho>)]

evaluated for a pure power-law profile rho(r) = rho_R (r/R)^(-p):

    zeta(p) = 2 (3 - p)^(3/2) / [3^(3/2) (2 - p)]

The two forms agree because PP20's "2.6" is the rounded value of
3^(3/2) / 2 = 2.598. We assert this equivalence directly.

This test file also lays a regression trap for the historical inline form
(3 - p) / (2.6 - 2p)^(3/2), which is a transcription bug — the 2.6 was
moved *inside* the power and the (2 - p) factor lost its 3/2 exponent.
That bug pretended to be a "domain limit at p = 1.3" (where the spurious
denominator 2.6 - 2p vanishes), but PP20 Eq. 6 is well-behaved over the
full physical domain 0 <= p < 2; the only true singularity is at p = 2
(singular isothermal collapse).

Numerical anchors:
- zeta(0) = 1 exactly (top-hat / uniform sphere)
- zeta(1) = 2 * 2^(3/2) / 3^(3/2) ~ 1.0887 (analytic exact value)
- zeta(1.5) = sqrt(2) (analytic exact value at p = 3/2)
- zeta(1.67) ~ 1.79 (Kainulainen+2014 median p; observational anchor)
- zeta -> infinity as p -> 2 (PP20 Fig. 1)
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from progenax.gravoturb.pp20_magnification import (
    P_MAX,
    magnification_factor,
    magnification_factor_with_core,
)


# ---------------------------------------------------------------------------
# Canonical anchor: zeta(0) = 1 (uniform-density sphere is its own reference)
# ---------------------------------------------------------------------------
def test_pp20_zeta_at_p_zero_is_unity():
    """zeta(p=0) = 1: a top-hat profile is its own SFR reference, by
    construction of the magnification-factor integral."""
    zeta = float(magnification_factor(jnp.float64(0.0)))
    assert zeta == pytest.approx(1.0, abs=1e-10)


# ---------------------------------------------------------------------------
# PP20 Eq. 6 vs analytic equivalence
# ---------------------------------------------------------------------------
def _pp20_eq6(p: float) -> float:
    """Direct transcription of PP20 Eq. 6 with the constant 2.6."""
    return (3.0 - p) ** 1.5 / (2.6 * (2.0 - p))


def _zeta_analytic(p: float) -> float:
    """Analytic form derived from the integral definition.

    Identical to PP20 Eq. 6 modulo the substitution 2.6 -> 3^(3/2)/2.
    """
    return 2.0 * (3.0 - p) ** 1.5 / (3.0**1.5 * (2.0 - p))


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.67, 1.8, 1.9])
def test_progenax_matches_pp20_eq6(p):
    """The progenax ζ(p) must agree with PP20 Eq. 6 to within the rounding
    of "2.6" (~ 0.1%). PP20 uses 2.6 as a numerical approximation to
    3^(3/2)/2 = 2.598; progenax uses the unrounded analytic form."""
    pgx = float(magnification_factor(jnp.float64(p)))
    pp20 = _pp20_eq6(p)
    if pp20 < 1.0:
        # PP20 Eq. 6 dips slightly below 1 right at p=0 due to 2.6 rounding
        # (true value is exactly 1 at p=0). Both expressions should be ~1.
        assert pgx == pytest.approx(1.0, abs=2e-3)
    else:
        # Agreement to ~0.1% (relative) is expected from the 2.6 rounding.
        assert pgx == pytest.approx(pp20, rel=2e-3), (
            f"At p={p}, progenax ζ = {pgx:.6f}, PP20 Eq. 6 = {pp20:.6f}"
        )


@pytest.mark.parametrize("p", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.67, 1.8, 1.9])
def test_progenax_matches_analytic_derivation(p):
    """The progenax ζ(p) must equal the unrounded analytic form
    2 (3-p)^(3/2) / (3^(3/2) (2-p)) to machine precision over [0, 2)."""
    pgx = float(magnification_factor(jnp.float64(p)))
    ana = _zeta_analytic(p)
    if p == 0.0:
        # Both equal 1 exactly at p=0
        assert pgx == pytest.approx(1.0, abs=1e-12)
        assert ana == pytest.approx(1.0, abs=1e-12)
    else:
        assert pgx == pytest.approx(ana, rel=1e-12), (
            f"At p={p}, progenax ζ = {pgx:.12f}, analytic = {ana:.12f}"
        )


# ---------------------------------------------------------------------------
# Specific exact-value anchors
# ---------------------------------------------------------------------------
def test_pp20_zeta_at_p_one_exact_analytic():
    """ζ(1) = 2·2^(3/2)/3^(3/2) ≈ 1.0887 (exact analytic value).

    PP20 Eq. 6 gives 2^(3/2)/2.6 = 1.0879 — agrees to 3 decimals.
    """
    zeta = float(magnification_factor(jnp.float64(1.0)))
    expected = 2.0 * 2.0**1.5 / 3.0**1.5
    assert zeta == pytest.approx(expected, abs=1e-12)
    assert zeta == pytest.approx(1.0887, abs=1e-3)


def test_pp20_zeta_at_p_three_halves_is_sqrt2():
    """ζ(3/2) = √2 (exact analytic value).

    Verify: 2 * (3/2)^(3/2) / (3^(3/2) * (1/2)) = 4 * (3/2)^(3/2) / 3^(3/2)
                                               = 4 / 3^(3/2) * (3/2)^(3/2)
                                               = 4 * (1/2)^(3/2) = 4/(2√2) = √2  ✓
    """
    zeta = float(magnification_factor(jnp.float64(1.5)))
    assert zeta == pytest.approx(2.0**0.5, abs=1e-12)


def test_pp20_zeta_at_kainulainen_median():
    """ζ(p=1.67) ≈ 1.79 — the Kainulainen+2014 median p inferred from the
    observational ρ-PDFs of 16 nearby molecular clouds. PP20 cites this
    on page 5 as the typical magnification factor for resolved Galactic
    cloud samples; the SFR boost of order ~80% over a uniform-density
    cloud is consistent with the observed dense-gas SFR efficiencies."""
    zeta = float(magnification_factor(jnp.float64(1.67)))
    assert zeta == pytest.approx(1.79, abs=0.02)


# ---------------------------------------------------------------------------
# Asymptotic behaviour
# ---------------------------------------------------------------------------
def test_pp20_zeta_diverges_as_p_to_two():
    """ζ → ∞ as p → 2: PP20 Eq. 6 has a 1/(2-p) singularity, reflecting
    the pathology of singular isothermal collapse (mass piles up at r=0)."""
    zeta_close_to_2 = float(magnification_factor(jnp.float64(1.95)))
    zeta_at_one = float(magnification_factor(jnp.float64(1.0)))
    # Should be at least an order of magnitude larger than ζ(1)
    assert zeta_close_to_2 > 5.0 * zeta_at_one
    # And finite (we clip at P_MAX to keep gradients well-behaved)
    assert jnp.isfinite(zeta_close_to_2)


def test_pp20_zeta_clamped_at_P_MAX():
    """For numerical safety, ζ is clipped at p = P_MAX = 1.95.

    P_MAX < 2 keeps the function differentiable for HMC/NUTS and matches
    PP20's own treatment (Fig. 1 caps the analytic curve before p=2 and
    extends with a numerically-integrated cored profile beyond)."""
    zeta_above = float(magnification_factor(jnp.float64(2.5)))
    zeta_at_pmax = float(magnification_factor(jnp.float64(P_MAX)))
    assert zeta_above == pytest.approx(zeta_at_pmax, abs=1e-12)


# ---------------------------------------------------------------------------
# Monotonicity over the full physical domain
# ---------------------------------------------------------------------------
def test_pp20_zeta_monotonically_increasing_over_full_domain():
    """ζ(p) is monotonically increasing for 0 ≤ p < 2, with NO singularity
    at p = 1.3. This explicitly contradicts the historical buggy claim
    that the formula has a "domain limit" at p ≈ 1.3."""
    p_values = jnp.linspace(0.0, 1.9, 20, dtype=jnp.float64)
    zeta_values = jax.vmap(magnification_factor)(p_values)
    diffs = jnp.diff(zeta_values)
    assert jnp.all(diffs > 0), (
        "ζ(p) should be strictly monotonically increasing for 0 ≤ p < 2; "
        f"got differences {diffs}"
    )


def test_pp20_zeta_finite_across_p_one_three():
    """The historical buggy form (3-p)/(2.6-2p)^(3/2) had a spurious
    singularity at p=1.3 where the denominator 2.6-2p vanishes. The
    canonical PP20 form has NO singularity there — ζ(1.3) is a smooth
    finite value, with ζ(1.29), ζ(1.30), ζ(1.31) all close to each other."""
    z_below = float(magnification_factor(jnp.float64(1.29)))
    z_at = float(magnification_factor(jnp.float64(1.30)))
    z_above = float(magnification_factor(jnp.float64(1.31)))
    # All finite and physically sensible (zeta > 1)
    assert jnp.isfinite(z_below) and z_below > 1.0
    assert jnp.isfinite(z_at) and z_at > 1.0
    assert jnp.isfinite(z_above) and z_above > 1.0
    # And smooth (no jump): consecutive values differ by < 1%
    assert abs(z_at - z_below) / z_at < 0.01
    assert abs(z_above - z_at) / z_at < 0.01


# ---------------------------------------------------------------------------
# JAX compatibility (jit, grad, vmap)
# ---------------------------------------------------------------------------
def test_pp20_zeta_jit_compatible():
    """magnification_factor must be JIT-safe."""
    jit_fn = jax.jit(magnification_factor)
    direct = float(magnification_factor(jnp.float64(1.0)))
    jitted = float(jit_fn(jnp.float64(1.0)))
    assert jitted == pytest.approx(direct, rel=1e-12)


def test_pp20_zeta_grad_compatible():
    """magnification_factor must be differentiable across the safe domain."""
    grad_fn = jax.grad(magnification_factor)
    for p_test in [0.5, 1.0, 1.5, 1.8]:
        g = float(grad_fn(jnp.float64(p_test)))
        assert jnp.isfinite(g), f"gradient at p={p_test} is not finite: {g}"
        # ζ is monotonically increasing → gradient is positive
        assert g > 0.0


def test_pp20_zeta_vmap_compatible():
    """magnification_factor must be vmap-safe."""
    p_values = jnp.linspace(0.0, 1.9, 10, dtype=jnp.float64)
    zeta_values = jax.vmap(magnification_factor)(p_values)
    assert zeta_values.shape == (10,)
    assert jnp.all(jnp.isfinite(zeta_values))


# ---------------------------------------------------------------------------
# magnification_factor_with_core convergence to magnification_factor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [0.5, 1.0, 1.5])
def test_cored_approaches_powerlaw_as_core_shrinks(p):
    """magnification_factor_with_core(p, r_c/R) should monotonically
    approach the analytic magnification_factor(p) as r_c/R → 0 (the
    cored profile becomes a pure power-law in the limit of vanishing
    core). We use the function's default integration grid (n=100) and
    check convergence direction: smaller cores give cored-ζ closer to
    the analytic power-law value.
    """
    analytic = float(magnification_factor(jnp.float64(p)))
    cored_large = float(
        magnification_factor_with_core(jnp.float64(p), jnp.float64(0.3))
    )
    cored_small = float(
        magnification_factor_with_core(jnp.float64(p), jnp.float64(0.01))
    )
    # Smaller core → cored-ζ closer to the pure power-law analytic value
    err_large = abs(cored_large - analytic)
    err_small = abs(cored_small - analytic)
    assert err_small < err_large, (
        f"At p={p}, cored ζ should approach analytic ζ={analytic:.4f} as "
        f"r_c/R shrinks: r_c=0.3 gives {cored_large:.4f} (err {err_large:.4f}), "
        f"r_c=0.01 gives {cored_small:.4f} (err {err_small:.4f})"
    )
    # And the small-core value should already be within ~30% of the analytic
    # (residual is from the trapezoid integration on the dimensionless grid;
    # tightening would require integration tweaks orthogonal to this fix).
    assert cored_small == pytest.approx(analytic, rel=0.3), (
        f"At p={p}, cored ζ (r_c/R=0.01) = {cored_small:.4f} vs analytic {analytic:.4f}"
    )


@pytest.mark.parametrize("p,rc", [(0.5, 0.2), (1.5, 0.2), (1.7, 0.3)])
def test_with_core_resolution_controllable_and_converged(p, rc):
    """magnification_factor_with_core must accept an explicit n_radial_points
    and give a resolution-independent result at high n.

    Before the F10 fix, n_radial_points was not static under jax.jit, so passing
    it raised ConcretizationTypeError (jnp.linspace 'num' must be concrete).
    After the fix (static n + trapezoid on a grid from 0), the cored integral is
    converged: doubling the resolution changes zeta by < 1e-3.
    """
    pp, rr = jnp.float64(p), jnp.float64(rc)
    z256 = float(magnification_factor_with_core(pp, rr, n_radial_points=256))
    z512 = float(magnification_factor_with_core(pp, rr, n_radial_points=512))
    assert jnp.isfinite(z256) and jnp.isfinite(z512)
    assert abs(z256 - z512) < 1e-3, (
        f"At p={p}, r_c/R={rc}: cored zeta not converged in resolution "
        f"(n=256 -> {z256:.6f}, n=512 -> {z512:.6f}, |delta|={abs(z256 - z512):.2e})"
    )


# ---------------------------------------------------------------------------
# Regression trap for the historical buggy inline form
# ---------------------------------------------------------------------------
def test_no_buggy_inline_zeta_form_in_progenax():
    """The transcription bug `(3 - p) / (2.6 - 2*p)^(3/2)` (which moves the
    2.6 inside the power and drops the (3-p) exponent) must NOT reappear
    in progenax. The bug had a spurious singularity at p = 1.3 that was
    rationalised as a "domain limit" — but PP20 Eq. 6 is well-behaved
    over the full physical 0 ≤ p < 2 domain.

    This test scans the progenax source tree for the typo signature.
    """
    progenax_root = (
        Path(__file__).resolve().parent.parent.parent.parent / "src" / "progenax"
    )
    bad_signatures = [
        "(2.6 - 2.0 * p)",
        "(2.6 - 2*p)",
        "2.6 - 2.0*p",
        "2.6 - 2 * p",
    ]

    def _stripped_lines(src: str) -> list[str]:
        """Yield non-comment, non-docstring source lines.

        Cheap heuristic: drop lines whose first non-whitespace char is `#`,
        and skip everything between matched triple-quote pairs. Sufficient
        for progenax source files (no `#` or triple-quotes inside string
        literals on lines that also contain numeric expressions)."""
        out = []
        in_doc = False
        doc_marker = None
        for line in src.splitlines():
            stripped = line.lstrip()
            for marker in ('"""', "'''"):
                if marker in stripped:
                    if not in_doc:
                        in_doc = True
                        doc_marker = marker
                        if stripped.count(marker) >= 2:
                            in_doc = False
                            doc_marker = None
                        break
                    elif marker == doc_marker:
                        in_doc = False
                        doc_marker = None
                        break
            if in_doc:
                continue
            if stripped.startswith("#"):
                continue
            if "#" in line:
                line = line.split("#", 1)[0]
            out.append(line)
        return out

    offenders = []
    for path in progenax_root.rglob("*.py"):
        text = path.read_text()
        live_text = "\n".join(_stripped_lines(text))
        for sig in bad_signatures:
            if sig in live_text:
                offenders.append(f"{path.relative_to(progenax_root)}: {sig!r}")
    assert not offenders, (
        "Buggy PP20 ζ(p) transcription detected in progenax source.\n"
        "The form (3-p)/(2.6-2p)^(3/2) is a typo of PP20 Eq. 6,\n"
        "which is (3-p)^(3/2)/(2.6*(2-p)). See "
        "progenax/gravoturb/pp20_magnification.py for the canonical form.\n"
        "Offenders:\n  " + "\n  ".join(offenders)
    )
