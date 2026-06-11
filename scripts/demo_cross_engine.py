#!/usr/bin/env python
r"""Cross-engine agreement demo (B1): one King model built TWICE, two engines agree.

The methods-paper "two independent derivations agree" credibility figure. We
build ONE King model (W0=5, g=1, r_c=1 pc) by two INDEPENDENT engines and show
they agree on the three quantities that pin a self-consistent equilibrium:

  * rho(r): Engine A's total dimensionless density vs the prescribed King
    density that Engine B inverts. Both use the unit-CENTRAL-density convention
    (rho(r)/rho_0, =1 at the centre), so they overlay directly; we additionally
    re-normalize each by its innermost-interior value before differencing so the
    residual is a pure SHAPE comparison free of any central-amplitude offset.
  * sigma_1d(r): Engine A's analytic lowered-isothermal moment oracle
    (s_j sqrt(I4/I2/3)) vs Engine B's Eddington speed-moment oracle
    (sqrt((m2/m0)/3)), overlaid with the per-bin sigma measured from N=2e4
    same-key samples of each engine.
  * f(E): Engine A's lowered-exponential DF shape vs Engine B's
    Eddington-inverted f_j_grid[0]. Engine B's E_grid is the PHYSICAL relative
    energy (0..Psi_0, G=1 model units); Engine A's lowered_exponential(g, .) is
    a function of the DIMENSIONLESS lowered-potential argument (0..W0). Under
    the linear map hat E = (W0/Psi_0) E they share the King DF shape, so we plot
    both on hat E and match by PEAK -- a SHAPE comparison, the only meaningful
    one across the two conventions. The ~5% edge/cusp residual is the Eddington
    inverter recovering the closed-form lowered exponential to grid accuracy.

This is assembly of already-validated machinery (mirrors
tests/validation/test_engine_b_physics.py::test_king_density_engine_b_matches_engine_a
and the analytic oracles in test_multimass_equilibrium_physics.py /
test_engine_b_physics.py); there are NO STOP points. The gates ARE the contract.

Measured gate values (this build, N=2e4, key=PRNGKey(0)) are printed in the
PASS/FAIL table at the end and re-asserted as exit status. The validated ledger
anchors are ~2e-4 (radial KS) / ~3e-4 (sigma-dev); the gates sit at 0.02.

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_cross_engine.py
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from jaxstro.units import STELLAR
from progenax import KingProfile
from progenax.cluster.multicomponent import MultiComponentCluster
from progenax.profiles.limepy import lowered_exponential

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig

apply_pub_style()

OUTPUT_DIR = "validation/plots"
G = STELLAR.G  # pc^3 Msun^-1 Myr^-2 -> lengths pc, velocities pc/Myr
N_SAMPLE = 20000
W0, GG, R_C = 5.0, 1.0, 1.0


# ---------------------------------------------------------------------------
# model builds (verbatim from the gold reference test)
# ---------------------------------------------------------------------------


def build_models():
    """The SAME King(W0=5, g=1, r_c=1) by two independent engines.

    Engine B: prescribed King density -> Poisson quadrature -> Eddington
    inversion. Engine A: lowered-isothermal DF -> coupled Poisson ODE.
    """
    king = KingProfile.from_W0_rc(W0=W0, r_c=R_C)
    mB = MultiComponentCluster.from_density_profiles(
        [king], jnp.array([1.0]), m_j=jnp.array([1.0]))
    mA = MultiComponentCluster.from_components(
        alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]), m_j=jnp.array([1.0]),
        W0=W0, g=GG, r_c=R_C)
    return king, mA, mB


def _com_arrays(ic):
    """COM-frame numpy (positions, velocities, masses) from an ICResult."""
    p = np.asarray(ic.positions - jnp.average(ic.positions, axis=0, weights=ic.masses))
    v = np.asarray(ic.velocities - jnp.average(ic.velocities, axis=0, weights=ic.masses))
    return p, v, np.asarray(ic.masses)


# ---------------------------------------------------------------------------
# analytic sigma_1d(r) oracles (verbatim recipes)
# ---------------------------------------------------------------------------


def sigma_engine_a(mA, r, m_total):
    r"""Engine A analytic sigma_1d(r): s_j sqrt(I4/I2/3) on the lowered DF.

    Recipe from test_multimass_equilibrium_physics.py: the velocity scale
    s = sqrt(G M / (9 r_c mu_tot)), s_j = s w_j; W_j(r) = rescale_j psi(r)
    (psi interpolated on the ODE grid); E = lowered_exponential(g, W_j - u^2/2);
    sigma_j = s_j sqrt(integral u^4 E / integral u^2 E / 3). Single component
    here (j=0, w_j=1, rescale_j=1). ``m_total`` is the PHYSICAL total cluster
    mass actually realized by the sample -- sample_cluster sets the velocity
    scale from M = sum_i m_i (not sum_j m_j), so the oracle must use the same M
    to land in physical pc/Myr.
    """
    s = float(jnp.sqrt(G * m_total / (9.0 * mA.r_c * mA.mu_tot)))
    s_j = s * float(mA.w_j[0])
    rescale0 = float(mA.rescale_j[0])

    def sigma_at(rr):
        W_j = rescale0 * float(jnp.interp(rr, mA.xi_grid * mA.r_c,
                                          mA.psi_grid, left=W0, right=0.0))
        if W_j <= 0.0:
            return np.nan
        u = jnp.linspace(0.0, jnp.sqrt(2.0 * W_j), 400)
        E = lowered_exponential(mA.g, W_j - u**2 / 2.0)
        return s_j * float(jnp.sqrt(jnp.trapezoid(u**4 * E, u)
                                    / jnp.trapezoid(u**2 * E, u) / 3.0))

    return np.array([sigma_at(float(rr)) for rr in np.asarray(r)])


def sigma_engine_b(mB, r, m_total, n_w=400):
    r"""Engine B analytic sigma_1d(r): sqrt((m2/m0)/3) from the Eddington DF.

    Recipe from test_engine_b_physics.py: at each radius the speed moments
    m0 = integral w^2 f dw, m2 = integral w^4 f dw of the (clamped) Eddington
    f_j on the shared (dimensionless) Psi give <w^2> = m2/m0, so the
    DIMENSIONLESS sigma_1d(r) = sqrt((m2/m0)/3). f_row clamped at 0 exactly as
    the speed sampler clamps grid ringing. Engine B's sampler scales speeds by
    v = sqrt(G M / (4 pi mu)) w (eddington_engine; 4 pi mu == 1 for the
    mass-normalized densities), so we multiply by that PHYSICAL velocity scale
    with M = m_total to match the sampled pc/Myr.
    """
    st = mB.engine_b
    f_row = st.f_j_grid[0]
    v_scale = float(jnp.sqrt(G * m_total / (4.0 * jnp.pi * st.mu)))
    Psi_at = np.asarray(jnp.interp(jnp.asarray(r), st.r_poisson, st.Psi_poisson,
                                   left=st.Psi_poisson[0], right=0.0))

    def sigma_at(Psi_r):
        if Psi_r <= 0.0:
            return np.nan
        w = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_r), n_w)
        f_at = jnp.maximum(jnp.interp(Psi_r - 0.5 * w**2, st.E_grid, f_row), 0.0)
        m0 = jnp.trapezoid(w**2 * f_at, w)
        m2 = jnp.trapezoid(w**4 * f_at, w)
        return v_scale * float(jnp.sqrt((m2 / (m0 + 1e-300)) / 3.0))

    return np.array([sigma_at(float(P)) for P in Psi_at])


# ---------------------------------------------------------------------------
# sampling + gate measurement
# ---------------------------------------------------------------------------


def sample_both(mA, mB):
    """Same-key N=2e4 draw from both engines -> (rA, vsq_A, rB, vsq_B, m_total)."""
    key = jax.random.PRNGKey(0)
    icA = mA.sample_cluster(key, n_stars=N_SAMPLE, G=G)
    icB = mB.sample_cluster(key, n_stars=N_SAMPLE, G=G)
    _, vA, mA_i = _com_arrays(icA)
    _, vB, _ = _com_arrays(icB)
    rA = np.asarray(jnp.linalg.norm(icA.positions, axis=1))
    rB = np.asarray(jnp.linalg.norm(icB.positions, axis=1))
    m_total = float(mA_i.sum())  # physical M = sum_i m_i (sets the velocity scale)
    return rA, np.sum(vA**2, axis=1), rB, np.sum(vB**2, axis=1), m_total


def radial_ks(rA, rB):
    """Two-sample KS distance between the two radial samples (gate < 0.02)."""
    grid = np.sort(np.concatenate([rA, rB]))
    return float(np.max(np.abs(
        np.searchsorted(np.sort(rA), grid, side="right") / len(rA)
        - np.searchsorted(np.sort(rB), grid, side="right") / len(rB))))


def binned_sigma_dev(rA, vsqA, rB, vsqB):
    """Per-bin sampled sigma_1d for A and B over interior 5%-90% quantile bins.

    Returns (centers, sigA, sigB, max_dev) where sigX = sqrt(<v^2>/3) in bin
    and max_dev = max |sigma_B/sigma_A - 1| (gate < 0.02).
    """
    edges = np.quantile(rA, np.linspace(0.05, 0.90, 7))
    centers, sigA, sigB = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selA = (rA >= lo) & (rA < hi)
        selB = (rB >= lo) & (rB < hi)
        centers.append(0.5 * (lo + hi))
        sigA.append(np.sqrt(vsqA[selA].mean() / 3.0))
        sigB.append(np.sqrt(vsqB[selB].mean() / 3.0))
    centers, sigA, sigB = map(np.array, (centers, sigA, sigB))
    max_dev = float(np.max(np.abs(sigB / sigA - 1.0)))
    return centers, sigA, sigB, max_dev


# ---------------------------------------------------------------------------
# panel painters (each draws a main axis + its residual strip)
# ---------------------------------------------------------------------------


def panel_rho(ax, res, king, mA):
    r"""rho(r) panel: Engine A total_density vs prescribed King density.

    Both unit-central-density (rho/rho_0). Residual = fractional difference
    after re-normalizing each by its innermost interior value (pure shape).
    """
    r = np.linspace(0.02 * float(mA.r_t), 0.98 * float(mA.r_t), 400)
    rho_A = np.asarray(mA.total_density(jnp.asarray(r)))
    rho_B = np.asarray(king.density(jnp.asarray(r)))  # prescribed King (Engine B input)
    # Re-normalize each to its innermost value (central convention) for a pure
    # SHAPE residual immune to any central-amplitude offset.
    nA, nB = rho_A / rho_A[0], rho_B / rho_B[0]
    frac = nB / np.where(nA > 0, nA, np.nan) - 1.0
    max_frac = float(np.nanmax(np.abs(frac[nA > 1e-6])))

    ax.semilogy(r, rho_A, color=OI["blue"], lw=1.7,
                label=r"Engine A: $\rho_{\rm tot}(r)$ (ODE + lowered DF)")
    ax.semilogy(r, rho_B, color=OI["vermilion"], lw=1.5, ls="--",
                label=r"Engine B: prescribed King $\rho(r)$")
    ax.set_ylabel(r"$\rho(r)/\rho_0$")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    res.plot(r, 100.0 * frac, color=OI["green"], lw=1.2)
    res.axhline(0.0, color="0.6", lw=0.8, ls=":")
    res.set_ylabel(r"$\Delta\rho\,[\%]$", fontsize=7)
    res.set_xlabel(r"$r$ [pc]")
    return max_frac


def panel_sigma(ax, res, mA, mB, r_grid, samp, m_total):
    r"""sigma_1d(r) panel: A & B analytic oracles + measured per-bin sigma."""
    centers, sigA_s, sigB_s, _ = samp
    sigA = sigma_engine_a(mA, r_grid, m_total)
    sigB = sigma_engine_b(mB, r_grid, m_total)
    good = np.isfinite(sigA) & np.isfinite(sigB)

    ax.plot(r_grid[good], sigA[good], color=OI["blue"], lw=1.7,
            label=r"Engine A oracle $s_j\sqrt{I_4/I_2/3}$")
    ax.plot(r_grid[good], sigB[good], color=OI["vermilion"], lw=1.5, ls="--",
            label=r"Engine B oracle $\sqrt{(m_2/m_0)/3}$")
    ax.plot(centers, sigA_s, ls="none", marker="o", ms=4, color=OI["blue"],
            mfc="white", label=r"sampled A ($N=2\times10^4$)")
    ax.plot(centers, sigB_s, ls="none", marker="s", ms=4, color=OI["vermilion"],
            mfc="white", label=r"sampled B")
    ax.set_ylabel(r"$\sigma_{1d}$ [pc Myr$^{-1}$]")
    ax.legend(frameon=False, fontsize=6.5, loc="upper right")

    # residual: oracle ratio sigma_B/sigma_A - 1 over the analytic grid.
    ratio = sigB[good] / sigA[good] - 1.0
    res.plot(r_grid[good], 100.0 * ratio, color=OI["green"], lw=1.2)
    res.axhline(0.0, color="0.6", lw=0.8, ls=":")
    res.set_ylabel(r"$\sigma_B/\sigma_A-1\,[\%]$", fontsize=6.5)
    res.set_xlabel(r"$r$ [pc]")


def panel_fe(ax, res, mA, mB):
    r"""f(E) panel: Engine A lowered-exponential shape vs Engine B Eddington f.

    Engine B's E_grid is the PHYSICAL relative energy (0..Psi_0 in G=1 model
    units); Engine A's lowered_exponential(g, .) is a function of the
    DIMENSIONLESS lowered potential argument W (0..W0). The two share the same
    King DF shape under the linear map W = (W0/Psi_0) E, so we plot both on the
    dimensionless energy axis E_hat = (W0/Psi_0) E and match by PEAK (each
    normalized to its own maximum on the shared support) -- the only meaningful
    comparison across the two unit conventions. Residual is the peak-normalized
    difference; the small (~5%) edge/cusp residual is the Eddington inverter
    recovering the closed-form lowered exponential to grid accuracy.
    """
    st = mB.engine_b
    E = np.asarray(st.E_grid)
    Psi0 = float(st.Psi_poisson[0])
    E_hat = E * (W0 / Psi0)  # physical -> dimensionless lowered-potential argument
    f_B = np.asarray(jnp.maximum(st.f_j_grid[0], 0.0))
    f_A = np.asarray(jnp.maximum(lowered_exponential(mA.g, jnp.asarray(E_hat)), 0.0))

    sel = (f_A > 0) & (f_B > 0) & (E > 0)
    nA = f_A / f_A[sel].max()
    nB = f_B / f_B[sel].max()

    ax.plot(E_hat[sel], nA[sel], color=OI["blue"], lw=1.7,
            label=r"Engine A: lowered exp. $E_\gamma(g, \hat E)$")
    ax.plot(E_hat[sel], nB[sel], color=OI["vermilion"], lw=1.5, ls="--",
            label=r"Engine B: Eddington $f_0(\hat E)$")
    ax.set_ylabel(r"$f/f_{\rm max}$ (peak-matched)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    res.plot(E_hat[sel], 100.0 * (nB[sel] - nA[sel]), color=OI["green"], lw=1.2)
    res.axhline(0.0, color="0.6", lw=0.8, ls=":")
    res.set_ylabel(r"$\Delta f\,[\%]$", fontsize=7)
    res.set_xlabel(r"$\hat E = (W_0/\Psi_0)\,E$ (dimensionless)")
    return float(np.max(np.abs(nB[sel] - nA[sel])))


# ---------------------------------------------------------------------------
# figure + driver
# ---------------------------------------------------------------------------


def make_figure(king, mA, mB, samp, m_total):
    """Three stacked panels (rho, sigma, f) each with a residual strip below."""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(6.6, 9.4))
    gs = GridSpec(6, 1, height_ratios=[3, 1, 3, 1, 3, 1], hspace=0.08, figure=fig)
    ax_rho, res_rho = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax_sig, res_sig = fig.add_subplot(gs[2]), fig.add_subplot(gs[3])
    ax_fe, res_fe = fig.add_subplot(gs[4]), fig.add_subplot(gs[5])
    for ax in (ax_rho, ax_sig, ax_fe):
        ax.tick_params(labelbottom=False)

    r_grid = np.linspace(0.02 * float(mA.r_t), 0.95 * float(mA.r_t), 300)
    max_rho = panel_rho(ax_rho, res_rho, king, mA)
    panel_sigma(ax_sig, res_sig, mA, mB, r_grid, samp, m_total)
    max_fe = panel_fe(ax_fe, res_fe, mA, mB)

    for ax, tag in ((ax_rho, "(a)"), (ax_sig, "(b)"), (ax_fe, "(c)")):
        panel_label(ax, tag, loc="upper right" if ax is ax_fe else "upper left")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_fig(fig, OUTPUT_DIR, "demo_cross_engine")
    return max_rho, max_fe


def main():
    print("=" * 78)
    print("CROSS-ENGINE AGREEMENT (B1): King W0=5, r_c=1 pc by two engines")
    print("(units: STELLAR -- lengths pc, masses Msun, velocities pc/Myr)")
    print("=" * 78)

    king, mA, mB = build_models()
    print(f"\n  r_t:  Engine A = {float(mA.r_t):.4f} pc   "
          f"Engine B = {float(mB.r_t):.4f} pc")

    # theory virial sanity gate (both engines).
    QA = float(np.asarray(mA.component_virial_ratios())[0])
    QB = float(np.asarray(mB.component_virial_ratios())[0])
    print(f"  theory Q_j:  Engine A = {QA:.5f}   Engine B = {QB:.5f}"
          f"  (gate 0.5 +- 3e-3)")

    rA, vsqA, rB, vsqB, m_total = sample_both(mA, mB)
    ks = radial_ks(rA, rB)
    samp = binned_sigma_dev(rA, vsqA, rB, vsqB)
    max_sig_dev = samp[3]
    print(f"\n  radial KS distance         = {ks:.5f}  (gate < 0.02)")
    print(f"  max |sigma_B/sigma_A - 1|  = {max_sig_dev:.5f}  (gate < 0.02)")

    max_rho, max_fe = make_figure(king, mA, mB, samp, m_total)
    print(f"  rho shape max frac diff    = {max_rho:.2e}  (diagnostic)")
    print(f"  f(E) peak-matched max diff = {max_fe:.2e}  (diagnostic)")

    rows = [
        ("radial KS distance", f"{ks:.5f}", "< 0.02", ks < 0.02),
        ("max |sigma_B/sigma_A - 1|", f"{max_sig_dev:.5f}", "< 0.02",
         max_sig_dev < 0.02),
        ("Engine A theory Q_j", f"{QA:.5f}", "0.5 +- 3e-3", abs(QA - 0.5) < 3e-3),
        ("Engine B theory Q_j", f"{QB:.5f}", "0.5 +- 3e-3", abs(QB - 0.5) < 3e-3),
    ]

    print("\n" + "-" * 78)
    print(f"  {'CHECK':<32s} {'measured':>12s} {'gate':>14s}   status")
    print("-" * 78)
    all_ok = True
    for name, measured, gate, ok in rows:
        all_ok &= ok
        print(f"  {name:<32s} {measured:>12s} {gate:>14s}   {'PASS' if ok else 'FAIL'}")
    print("-" * 78)
    print(f"  saved {OUTPUT_DIR}/demo_cross_engine.{{png,pdf}}")
    print("=" * 78)
    print("  CROSS-ENGINE DEMO: ALL PASS" if all_ok
          else "  CROSS-ENGINE DEMO: FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
