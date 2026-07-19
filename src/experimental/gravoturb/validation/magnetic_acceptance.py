"""Acceptance suite for the magnetized-turbulence layers (ADR-0060..0063; AC-MAG1..8).

``MagneticSpec`` adds μ_Φ-primary magnetic support to ``build_cluster_ic``: L1 magnetic σ_s²
(Molina 2012 / F&K12), the s_crit collapse-threshold channel (magnetothermal Jeans, F&K12 Eq.21;
+ ambipolar flux-loss closure), L2 velocity anisotropy (Hu & Lazarian 2021 ℳ_A^{-4/3}), and the L3
divergence-free vector B field for RMHD seeding.

Each ``ac_mag*`` PRINTS an expected-vs-measured table with a PASS/FAIL verdict and returns
``{"passed": bool, ...}``. "Validated" = a number one of these committed functions just printed.
numpy is permitted here (validation side).

Run:
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.magnetic_acceptance
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.realization.magnetic import (
    anisotropy_ratio_theory,
    magnetic_field_grid,
    sigma_s_squared_magnetic,
)
from gravoturb.specs import (
    CloudSpec,
    CompositionSpec,
    GasSpec,
    GeometrySpec,
    MagneticSpec,
    VelocitySpec,
)
from gravoturb.theory.density_pdf import sigma_s_squared
from progenax import PlummerProfile

G = STELLAR.G
_MASSES = jnp.linspace(0.3, 8.0, 300)


def _kw(c_s=1.0, sfe=0.02, shape=(24, 24, 24)):
    return dict(
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=1.0), box_size=4.0, shape=shape),
        velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=c_s),
        composition=CompositionSpec(),
        gas=GasSpec(sfe=sfe),
        G=G, units=STELLAR, key=jax.random.PRNGKey(0),
    )


def _build(mu_phi, **mag):
    return build_cluster_ic(_MASSES, **_kw(), magnetic=MagneticSpec(mu_phi=mu_phi, **mag))


def ac_mag1_hydro_limit():
    """β₀→∞ (μ_Φ→∞) recovers the hydro σ_s² field, and magnetic=None is byte-identical."""
    print("\n=== AC-MAG1 — hydro limit + magnetic=None byte-identity ===")
    hydro = build_cluster_ic(_MASSES, **_kw())
    weak = _build(1.0e5, realize="scalar")
    dvar = abs(float(jnp.var(weak.fields.s_turb.s) - jnp.var(hydro.fields.s_turb.s)))
    dvar_rel = dvar / float(jnp.var(hydro.fields.s_turb.s))
    none_build = build_cluster_ic(_MASSES, **_kw(), magnetic=None)
    ident = all(
        (a == b if isinstance(a, str) else bool(jnp.array_equal(a, b)))
        for a, b in zip(jax.tree_util.tree_leaves(hydro), jax.tree_util.tree_leaves(none_build))
    )
    ok = dvar_rel < 1e-3 and ident
    print(f"  weak-field σ_s² deviation from hydro: {dvar_rel:.2e}  (< 1e-3)")
    print(f"  magnetic=None tree-identical to hydro: {ident}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "dvar_rel": dvar_rel, "identical": ident}


def ac_mag2_molina_sigma_s():
    """Magnetic σ_s² matches Molina 2012 / F&K12: ln(1+b²ℳ²·β₀/(β₀+1)) at the realized β₀."""
    print("\n=== AC-MAG2 — magnetic σ_s² anchor (Molina 2012 / F&K12 Eq.4) ===")
    b, mach = 0.5, 8.0
    ok = True
    print(f"  {'β₀':>10} {'σ_s²(meas)':>12} {'σ_s²(F&K12)':>12} {'|Δ|':>10}")
    for beta0 in (0.1, 1.0, 10.0):
        meas = float(sigma_s_squared_magnetic(mach, b, beta0))
        want = float(jnp.log(1.0 + (b * mach) ** 2 * beta0 / (beta0 + 1.0)))
        d = abs(meas - want)
        ok = ok and d < 1e-12
        print(f"  {beta0:10.2f} {meas:12.5f} {want:12.5f} {d:10.1e}")
    hydro = float(sigma_s_squared(mach, b))
    print(f"  β₀→∞ hydro reference σ_s² = {hydro:.5f}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok}


def ac_mag3_divergence_free():
    """L3 vector field ∇·B = 0 to machine precision (spectral divergence / field scale)."""
    print("\n=== AC-MAG3 — divergence-free vector B field (realize='field') ===")
    ic = _build(5.0, realize="field", anisotropy="fixed", anisotropy_value=2.0)
    B = ic.magnetic.B_field
    n = B.shape[1]
    kk = jnp.fft.fftfreq(n) * n
    KX, KY, KZ = jnp.meshgrid(kk, kk, kk, indexing="ij")
    Bk = jnp.fft.fftn(B, axes=(1, 2, 3))
    div = float(jnp.max(jnp.abs(KX * Bk[0] + KY * Bk[1] + KZ * Bk[2])) / jnp.max(jnp.abs(Bk)))
    ok = div < 1e-10
    print(f"  ‖∇·B‖/‖B̂‖ = {div:.2e}  (< 1e-10)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "divergence": div}


def ac_mag4_tangle_amplitude():
    """δB/B₀ ∝ ℳ_A: the turbulent tangle rms scales linearly with the Alfvén Mach number."""
    print("\n=== AC-MAG4 — tangle amplitude δB rms scales linearly with ℳ_A ===")
    b0, key = 2.0, jax.random.PRNGKey(3)

    def rms(mach_a):
        B = magnetic_field_grid((32, 32, 32), b0, mach_a, key)
        dB = B - jnp.mean(B, axis=(1, 2, 3), keepdims=True)
        return float(jnp.sqrt(jnp.mean(jnp.sum(dB**2, axis=0))))

    r_lo, r_hi = rms(0.5), rms(2.0)
    ratio = r_hi / r_lo
    ok = abs(ratio - 4.0) < 1e-3 and abs(r_hi - 2.0 * b0) / (2.0 * b0) < 1e-3
    print(f"  rms(δB; ℳ_A=2)/rms(δB; ℳ_A=0.5) = {ratio:.4f}  (expected 4.00)")
    print(f"  rms(δB; ℳ_A=2) = {r_hi:.4f}  (expected ℳ_A·B₀ = {2.0*b0:.4f})")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "ratio": ratio}


def ac_mag5_anisotropy_theory():
    """Velocity anisotropy closure = Hu & Lazarian 2021 (arXiv:2012.06039) r_A = ℳ_A^{-4/3}."""
    print("\n=== AC-MAG5 — anisotropy closure r_A = ℳ_A^{-4/3} (Hu & Lazarian 2021) ===")
    ok = True
    print(f"  {'ℳ_A':>8} {'r_A(meas)':>12} {'ℳ_A^-4/3':>12}")
    for m_a in (0.3, 0.5, 1.0, 2.0):
        meas = float(anisotropy_ratio_theory(m_a))
        want = m_a ** (-4.0 / 3.0) if m_a <= 1.0 else 1.0
        ok = ok and abs(meas - want) < 1e-9
        print(f"  {m_a:8.2f} {meas:12.5f} {want:12.5f}")
    print(f"  (isotropic r_A=1 at trans/super-Alfvénic ℳ_A≥1)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok}


def ac_mag6_s_crit_reduces_sf():
    """s_crit channel: magnetic support monotonically reduces the collapse-eligible fraction;
    strongly sub-critical clouds cease star formation (SFE unreachable)."""
    print("\n=== AC-MAG6 — s_crit reduces collapse-eligible fraction (F&K12 Eq.21) ===")
    hydro = build_cluster_ic(_MASSES, **_kw())
    f_hydro = float(hydro.ledger.collapse_eligible_fraction)
    rows = []
    print(f"  {'μ_Φ':>8} {'ℳ_A':>7} {'f_elig':>9} {'/hydro':>8}")
    print(f"  {'hydro':>8} {'-':>7} {f_hydro:9.4f} {1.0:8.3f}")
    for mu in (10.0, 5.0, 3.0, 2.0):
        ic = _build(mu, realize="scalar")
        f = float(ic.ledger.collapse_eligible_fraction)
        rows.append(f)
        print(f"  {mu:8.1f} {float(ic.magnetic.mach_alfven):7.2f} {f:9.4f} {f/f_hydro:8.3f}")
    monotone = all(rows[i] > rows[i + 1] for i in range(len(rows) - 1)) and rows[0] < f_hydro
    # too much flux -> SF ceases (SFE unreachable at c_s=0.2, sfe=0.3, mu_phi=0.3)
    ceased = False
    try:
        build_cluster_ic(_MASSES, **_kw(c_s=0.2, sfe=0.3),
                         magnetic=MagneticSpec(mu_phi=0.3, realize="scalar"))
    except (RuntimeError, ValueError):
        ceased = True
    ok = monotone and ceased
    print(f"  monotone decrease with field strength = {monotone}")
    print(f"  strongly sub-critical → SF ceases (SFE unreachable) = {ceased}")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "f_hydro": f_hydro, "rows": rows, "ceased": ceased}


def ac_mag7_ambipolar_recovery():
    """Ambipolar flux loss (non-ideal) recovers collapse vs the ideal flux-frozen threshold."""
    print("\n=== AC-MAG7 — ambipolar flux-loss recovers collapse vs ideal ===")
    ideal = _build(3.0, realize="scalar", collapse_threshold="ideal")
    ambi = _build(3.0, realize="scalar", collapse_threshold="ambipolar",
                  flux_loss_density=1.0, flux_loss_sharpness=4.0)
    f_i = float(ideal.ledger.collapse_eligible_fraction)
    f_a = float(ambi.ledger.collapse_eligible_fraction)
    ok = f_a > f_i
    print(f"  f_elig(ideal)     = {f_i:.5f}")
    print(f"  f_elig(ambipolar) = {f_a:.5f}   (> ideal: dense gas sheds flux, collapses)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "f_ideal": f_i, "f_ambipolar": f_a}


def ac_mag8_differentiable():
    """AD gradient of the magnetic σ_s² through μ_Φ agrees with finite differences."""
    print("\n=== AC-MAG8 — differentiability: d(σ_s²)/d(μ_Φ) AD vs FD ===")
    from gravoturb.realization.magnetic import (
        beta_from_mass_to_flux,
        sigma_s_squared_magnetic,
    )

    mach, b = 8.0, 0.5

    def sig(mu):
        beta0 = beta_from_mass_to_flux(mu, mach=mach, c_s=0.3, m_half=1e3, r_h=1.5, G=G)
        return sigma_s_squared_magnetic(mach, b, beta0)

    mu0 = 1.3
    ad = float(jax.grad(sig)(mu0))
    h = 1e-4
    fd = float((sig(mu0 + h) - sig(mu0 - h)) / (2 * h))
    rel = abs(ad - fd) / abs(fd)
    ok = np.isfinite(ad) and ad > 0 and rel < 1e-5
    print(f"  AD = {ad:.6e}   FD = {fd:.6e}   |Δ|/FD = {rel:.2e}  (< 1e-5)")
    print(f"  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "ad": ad, "fd": fd}


def main():
    print("=" * 72)
    print(" MAGNETIZED-TURBULENCE ACCEPTANCE SUITE (AC-MAG1..8; ADR-0060..0063)")
    print("=" * 72)
    results = {
        "AC-MAG1": ac_mag1_hydro_limit(),
        "AC-MAG2": ac_mag2_molina_sigma_s(),
        "AC-MAG3": ac_mag3_divergence_free(),
        "AC-MAG4": ac_mag4_tangle_amplitude(),
        "AC-MAG5": ac_mag5_anisotropy_theory(),
        "AC-MAG6": ac_mag6_s_crit_reduces_sf(),
        "AC-MAG7": ac_mag7_ambipolar_recovery(),
        "AC-MAG8": ac_mag8_differentiable(),
    }
    print("\n" + "=" * 72)
    n_pass = sum(r["passed"] for r in results.values())
    for name, r in results.items():
        print(f"  {name}: {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  OVERALL: {n_pass}/{len(results)} passed")
    print("=" * 72)
    return results


if __name__ == "__main__":
    main()
