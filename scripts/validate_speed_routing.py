#!/usr/bin/env python
"""Visual fidelity evidence for the table-routed speed draws (Batch A).

Since Batch A, the standalone velocity DFs (`KingVelocityDF`,
`MichieVelocityDF`, `LIMEPYVelocityDF`) draw speeds from precomputed CDF
tables by default (`speed_method="table"`), with the exact per-star
quadrature retained as the oracle (`speed_method="quadrature"`). The unit
suite gates this distributionally (KS D < 0.02; mean/second-moment ratios
within 2%/3%; beta(r) preserved to atol 0.06 -- see the TableRouting classes
in tests/unit/kinematics/). This script produces the corresponding FIGURES
for the validation pages, at the same model points the tests pin:

  * King   W0=5, r_c=1, r_t=10           (isotropic)        G = 1
  * LIMEPY W0=5, g=1, r_c=1, r_a=4       (OM-anisotropic)   G = 1
  * Michie W0=7, r_c=1, r_a=8            (Michie-King)      G = STELLAR.G

Each figure overlays the table-path and quadrature-oracle speed
distributions (panel a) and, for the anisotropic models, the measured
beta(r) profiles (panel b). Printed criteria mirror the test gates; exits
nonzero if any criterion fails.

Outputs: validation/plots/speed_routing_{king,limepy,michie}.{png,pdf}

Usage:
    env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_speed_routing.py
"""
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
from _plotstyle import OI, apply_pub_style, panel_label, save_fig  # noqa: E402

from jaxstro.units import STELLAR  # noqa: E402

PLOT_DIR = Path(__file__).parent.parent / "validation" / "plots"
N = 20_000          # oracle-N convention: KS 95% critical D ~ 0.0136 < 0.02
KS_GATE = 0.02
MEAN_GATE = 0.02
M2_GATE = 0.03
BETA_GATE = 0.06


def _speeds(vel):
    return np.asarray(jnp.linalg.norm(vel, axis=1))


def _beta_profile(pos, vel, r_edges):
    """Measured beta(r) = 1 - sigma_t^2 / (2 sigma_r^2) in radial shells."""
    r = jnp.linalg.norm(pos, axis=1)
    r_hat = pos / (r[:, None] + 1e-30)
    v_r = jnp.sum(vel * r_hat, axis=1)
    v_t2 = jnp.sum(vel**2, axis=1) - v_r**2
    betas, centers = [], []
    for lo, hi in zip(r_edges[:-1], r_edges[1:]):
        m = (r >= lo) & (r < hi)
        if int(jnp.sum(m)) < 50:
            continue
        betas.append(float(1.0 - jnp.mean(v_t2[m]) / (2.0 * jnp.mean(v_r[m] ** 2))))
        centers.append(0.5 * (float(lo) + float(hi)))
    return np.array(centers), np.array(betas)


def _draw(profile, df_table, df_quad, G, seed=0):
    """Positions (shared) + velocities from both paths, test-suite style."""
    masses = jnp.ones(N)
    pos = profile.sample_positions(masses, jax.random.PRNGKey(seed))
    key = jax.random.PRNGKey(seed + 1)
    v_t = df_table.sample_velocities(pos, masses, key, G=G)
    v_q = df_quad.sample_velocities(pos, masses, key, G=G)
    return pos, v_t, v_q


def _criteria(name, s_t, s_q):
    """KS + moment-ratio criteria (the unit-test gates); returns rows."""
    D = ks_2samp(s_t, s_q).statistic
    mean_dev = abs(s_t.mean() / s_q.mean() - 1.0)
    m2_dev = abs((s_t**2).mean() / (s_q**2).mean() - 1.0)
    return [
        (f"{name} speed KS D", D, KS_GATE),
        (f"{name} mean-speed ratio dev", mean_dev, MEAN_GATE),
        (f"{name} <u^2> ratio dev", m2_dev, M2_GATE),
    ], D


def _speed_panel(ax, s_t, s_q, D):
    bins = np.linspace(0.0, max(s_t.max(), s_q.max()), 60)
    ax.hist(s_q, bins=bins, density=True, histtype="step", lw=1.8,
            color=OI["vermilion"], label="quadrature oracle")
    ax.hist(s_t, bins=bins, density=True, histtype="step", lw=1.4,
            color=OI["blue"], label="table (default)")
    ax.annotate(f"KS $D = {D:.4f}$\n($N = {N:,}$, gate $< {KS_GATE}$)",
                xy=(0.97, 0.78), xycoords="axes fraction", ha="right",
                fontsize=8)
    ax.set_xlabel("speed $|v|$ [model units]")
    ax.set_ylabel("probability density")
    ax.legend(loc="upper right")


def _beta_panel(ax, pos, v_t, v_q, r_edges):
    c_t, b_t = _beta_profile(pos, v_t, r_edges)
    c_q, b_q = _beta_profile(pos, v_q, r_edges)
    ax.plot(c_q, b_q, "o-", color=OI["vermilion"], label="quadrature oracle")
    ax.plot(c_t, b_t, "s--", color=OI["blue"], label="table (default)")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("$r$ [model units]")
    ax.set_ylabel(r"$\beta(r) = 1 - \sigma_t^2 / 2\sigma_r^2$")
    ax.legend(loc="upper left")
    return float(np.max(np.abs(b_t - b_q)))


def run_king(rows):
    import matplotlib.pyplot as plt
    from progenax import KingVelocityDF
    from progenax.profiles.king import KingProfile

    kw = dict(W0=5.0, r_c=1.0, r_t=10.0)
    prof = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
    _, v_t, v_q = _draw(prof, KingVelocityDF(**kw),
                        KingVelocityDF(**kw, speed_method="quadrature"), G=1.0)
    crit, D = _criteria("King", _speeds(v_t), _speeds(v_q))
    rows += crit
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    _speed_panel(ax, _speeds(v_t), _speeds(v_q), D)
    save_fig(fig, PLOT_DIR, "speed_routing_king")


def run_limepy(rows):
    import matplotlib.pyplot as plt
    from progenax.kinematics.limepy_df import LIMEPYVelocityDF
    from progenax.profiles.limepy import LIMEPYProfile

    kw = dict(W0=5.0, g=1.0, r_c=1.0, r_a=4.0)
    prof = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0)
    pos, v_t, v_q = _draw(prof, LIMEPYVelocityDF(**kw),
                          LIMEPYVelocityDF(**kw, speed_method="quadrature"),
                          G=1.0)
    crit, D = _criteria("LIMEPY", _speeds(v_t), _speeds(v_q))
    rows += crit
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    _speed_panel(axa, _speeds(v_t), _speeds(v_q), D)
    panel_label(axa, "(a)")
    r = np.asarray(jnp.linalg.norm(pos, axis=1))
    edges = np.quantile(r, np.linspace(0.02, 0.98, 9))
    beta_dev = _beta_panel(axb, pos, v_t, v_q, edges)
    panel_label(axb, "(b)", loc="lower right")
    rows.append(("LIMEPY max |beta_t - beta_q|", beta_dev, BETA_GATE))
    fig.tight_layout()
    save_fig(fig, PLOT_DIR, "speed_routing_limepy")


def run_michie(rows):
    import matplotlib.pyplot as plt
    from progenax import MichieVelocityDF
    from progenax.profiles.michie import MichieProfile

    kw = dict(W0=7.0, r_c=1.0, r_a=8.0)
    prof = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
    pos, v_t, v_q = _draw(prof, MichieVelocityDF(**kw),
                          MichieVelocityDF(**kw, speed_method="quadrature"),
                          G=STELLAR.G)
    crit, D = _criteria("Michie", _speeds(v_t), _speeds(v_q))
    rows += crit
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    _speed_panel(axa, _speeds(v_t), _speeds(v_q), D)
    panel_label(axa, "(a)")
    r = np.asarray(jnp.linalg.norm(pos, axis=1))
    edges = np.quantile(r, np.linspace(0.02, 0.98, 9))
    beta_dev = _beta_panel(axb, pos, v_t, v_q, edges)
    panel_label(axb, "(b)", loc="lower right")
    rows.append(("Michie max |beta_t - beta_q|", beta_dev, BETA_GATE))
    fig.tight_layout()
    save_fig(fig, PLOT_DIR, "speed_routing_michie")


def main() -> int:
    apply_pub_style()
    print("=" * 72)
    print("SPEED-ROUTING FIDELITY (table default vs exact quadrature oracle)")
    print(f"N = {N:,} per draw (oracle-N convention)")
    print("=" * 72)
    rows = []
    run_king(rows)
    run_limepy(rows)
    run_michie(rows)

    all_pass = True
    for name, val, gate in rows:
        ok = val < gate
        all_pass &= ok
        print(f"  {name:<34} {val:11.5f}   < {gate:<6}  "
              f"{'PASS' if ok else 'FAIL'}")
    print("-" * 72)
    for stem in ("speed_routing_king", "speed_routing_limepy",
                 "speed_routing_michie"):
        print(f"  saved validation/plots/{stem}.{{png,pdf}}")
    print("=" * 72)
    print("  SPEED-ROUTING FIDELITY: ALL PASS" if all_pass
          else "  SPEED-ROUTING FIDELITY: FAILED")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
