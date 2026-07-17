"""AC-IC12 — identifiability on ENVELOPED catalogs (Phase 5; printed-artifact discipline).

The Phase-5 question: can (ℳ, α, β) be recovered from catalogs a real cluster would
give us — enveloped, position-sampled — with the periodic-box-calibrated machinery,
after the A1 windowing treatment? Channels (design 2026-07-16 + A1):

- latent channels (POT tail → α; 2-pt band powers → β) operate on the
  ENVELOPE-SUBTRACTED field ŝ = s_total − ln ρ_env (exact — the envelope is a known
  forward-model component; measured residual band-power distortion ≤2% lowest bin);
- the stellar CIC count channel (→ ℳ) uses the A1 masked, intensity-detrended
  statistic on the ACTUAL star catalog (histogrammed positions), with the model's
  homogeneous prediction evaluated at the shot-matched HARMONIC-mean effective n̄
  over masked cells (leading-order shot term 1/n̄; residual approximation error is
  absorbed by the enveloped-mock var_v, and the A1 transfer test bounds the bias
  below seed scatter);
- var_v and bp_precision are FIXED-FIDUCIAL constants computed on ENVELOPED mocks
  through the same measurement path (truth-independent — SBC-valid pattern).

Forward-bias matching: catalogs use placement='two_population' with f_sub=0 (pure
smooth ∝ρ sampling — the count model's linear-intensity assumption). Inference from
multi-freefall (ρ^{3/2}-weighted) catalogs needs a matching count model and is a
DOCUMENTED limitation, not silently ignored.

Run (the focused Anna-approved grid; several hours of NUTS):
    PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync \
        python -m gravoturb.validation.identifiability_acceptance
"""

import jax
import jax.numpy as jnp
import numpy as np
from jaxstro.units import STELLAR

from gravoturb.cluster import build_cluster_ic
from gravoturb.diagnostics.measure import (
    envelope_cell_intensity,
    measure_exceedances,
    measure_log_count_variance_detrended,
)
from gravoturb.inference.covariance import measured_bandpowers, mock_precision
from gravoturb.inference.hmc import run_nuts, to_constrained, to_unconstrained
from gravoturb.inference.model import K_EDGES, build_logdensity
from gravoturb.inference.priors import BM19Prior
from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
from progenax import PlummerProfile

# fiducial + measurement configuration (24³ grid = the count-model FFT grid)
SHAPE, BOX, R_H = (32,) * 3, 4.0, 0.5
B_FIX = 0.5
N_STARS = 5000
CELL = 4
N_MIN = 10.0
S_THR_MARGIN = 0.25  # 24³+0.5 starved the POT tail (~8 exceedances → α prior-pulled)
G = STELLAR.G
_PROFILE = PlummerProfile(r_h=R_H)

FID = (8.0, 1.8, 3.0)  # (mach, alpha, beta)


def _build_catalog(mach, alpha, beta, seed):
    """An enveloped catalog through the REAL builder (virial_target keeps the
    velocity stream irrelevant to the density channels)."""
    return build_cluster_ic(
        jnp.ones(N_STARS),
        cloud=CloudSpec(mach=mach, b=B_FIX, alpha=alpha, beta=beta),
        geometry=GeometrySpec(profile=_PROFILE, box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
        composition=CompositionSpec(placement="two_population", f_sub=0.0),
        G=G, key=jax.random.PRNGKey(seed),
    )


def _measure(ic):
    """The A1 envelope-aware data bundle from one catalog."""
    # latent channels on the envelope-subtracted field: ŝ = s_total − ln ρ_env,
    # which is EXACTLY s_turb by construction (the envelope is a known additive
    # log-component; with real data the envelope-fit residual would enter here)
    s_hat = np.asarray(ic.fields.s_turb.s)
    s_t = float(ic.fields.s_turb.s_t)
    s_thr = s_t + S_THR_MARGIN
    if s_hat.max() > s_thr:
        exc_counts, exc_edges, s_max, _ = measure_exceedances(s_hat, s_thr, 12)
    else:  # honest empty tail (same convention as the SBC driver)
        s_max = s_thr + max(S_THR_MARGIN, 1.0)
        exc_edges = np.linspace(s_thr, s_max, 13)
        exc_counts = np.zeros(12)
    band_powers = measured_bandpowers(s_hat, SHAPE, K_EDGES)

    # catalog CIC counts (actual star positions, box frame) + A1 detrended statistic
    pos_box = np.asarray(ic.stars.positions) + np.asarray(ic.ledger.frame.origin)
    n_coarse = SHAPE[0] // CELL
    edges = np.linspace(0.0, BOX, n_coarse + 1)
    counts, _ = np.histogramdd(pos_box, bins=(edges, edges, edges))
    nb_global = N_STARS / n_coarse**3
    n_cells = envelope_cell_intensity(_PROFILE, BOX, SHAPE, CELL, nb_global)
    v_meas = measure_log_count_variance_detrended(counts, n_cells, n_min=N_MIN)
    # shot-matched effective n_bar (harmonic mean over masked cells)
    masked = n_cells[n_cells >= N_MIN]
    n_eff = float(1.0 / np.mean(1.0 / masked))
    return {
        "exc_counts": exc_counts, "exc_edges": exc_edges,
        "log_count_vars": (v_meas,), "n_bars": (n_eff,),
        "band_powers": band_powers,
    }, float(s_thr), float(s_max)


def _fiducial_precisions(n_real_var=12, n_real_bp=48, key=None):
    """Truth-independent var_v + bp_precision on ENVELOPED mocks through the SAME
    measurement path (fixed fiducial; Hartlap for the band powers)."""
    key = key if key is not None else jax.random.PRNGKey(2**20)
    vs, bps = [], []
    for i in range(max(n_real_var, n_real_bp)):
        ic = _build_catalog(*FID, seed=10_000 + i)
        data, _, _ = _measure(ic)
        if i < n_real_var:
            vs.append(float(data["log_count_vars"][0]))
        bps.append(np.asarray(data["band_powers"]))
    var_v = float(np.var(vs, ddof=1))
    bp_prec = mock_precision(bps[:n_real_bp])
    return var_v, bp_prec


def ac_ic12_coverage(grid=None, n_rep=16, n_warmup=300, n_samples=500):
    """Coverage + bias over the focused grid (fiducial + one-at-a-time excursions)."""
    grid = grid or [FID,
                    (6.0, 1.8, 3.0), (10.0, 1.8, 3.0),
                    (8.0, 1.5, 3.0), (8.0, 2.2, 3.0),
                    (8.0, 1.8, 2.5), (8.0, 1.8, 3.5)]
    print("\n=== AC-IC12 — identifiability on enveloped catalogs (A1-windowed) ===")
    print(f"  grid: {grid}\n  n_rep={n_rep}, NUTS {n_warmup}+{n_samples}")
    var_v, bp_prec = _fiducial_precisions()
    print(f"  enveloped-mock var_v = {var_v:.3e}; bp_precision ready (Hartlap)")
    prior = BM19Prior()
    names = ["mach", "alpha", "beta"]
    all_rows = []
    for theta in grid:
        cover = np.zeros(3)
        bias, widths = [], []
        for rep in range(n_rep):
            ic = _build_catalog(*theta, seed=1000 * hash(theta) % 100_000 + rep)
            data, s_thr, s_max = _measure(ic)
            data["var_vs"] = (var_v,)
            logd = build_logdensity(prior, data, b=B_FIX, s_thr=s_thr, s_max=s_max,
                                    shape=SHAPE, cell_sizes=(CELL,),
                                    bp_precision=bp_prec)
            z0 = to_unconstrained(jnp.asarray(theta)) + 0.1 * jax.random.normal(
                jax.random.PRNGKey(rep), (3,))
            draws = run_nuts(logd, z0, jax.random.fold_in(jax.random.PRNGKey(77), rep),
                             n_warmup, n_samples)
            post = np.asarray(jax.vmap(to_constrained)(draws))
            lo, hi = np.percentile(post, [5, 95], axis=0)
            cover += (np.asarray(theta) >= lo) & (np.asarray(theta) <= hi)
            bias.append(post.mean(axis=0) - np.asarray(theta))
            widths.append(post.std(axis=0))
        cover /= n_rep
        bias = np.array(bias)
        widths = np.array(widths)
        row = (theta, cover, bias.mean(axis=0), bias.std(axis=0), widths.mean(axis=0))
        all_rows.append(row)
        print(f"  θ={theta}: 90% coverage " +
              " ".join(f"{n}={c:.2f}" for n, c in zip(names, cover)) +
              "  bias " +
              " ".join(f"{n}={m:+.3f}±{s:.3f}" for n, m, s in
                       zip(names, bias.mean(axis=0), bias.std(axis=0))) +
              "  ⟨σ_post⟩ " +
              " ".join(f"{n}={w:.3f}" for n, w in zip(names, widths.mean(axis=0))))
    # gate (calibration-aware): coverage within binomial 3σ of 0.90, AND
    # |bias| < 0.5·⟨σ_post⟩ — prior shrinkage with HONEST wide intervals is correct
    # Bayesian behavior under weak data (the α-channel prior pull); the gate flags
    # MIScalibration (bias comparable to the claimed width, or bad coverage), not
    # weak channels. Channel informativeness is reported via ⟨σ_post⟩.
    ok = True
    se = np.sqrt(0.9 * 0.1 / n_rep)
    for theta, cover, bmean, bstd, wmean in all_rows:
        ok = ok and bool(np.all(np.abs(cover - 0.90) < 3 * se + 1e-9))
        ok = ok and bool(np.all(np.abs(bmean) < 0.5 * wmean))
    print(f"  coverage within 3σ_binomial({3*se:.2f}) of 0.90 and |bias| < 0.5·⟨σ_post⟩ "
          f"everywhere = {ok}  {'PASS' if ok else 'FAIL'}")
    return {"passed": ok, "rows": all_rows, "var_v": var_v}


def main():
    print("=" * 78)
    print("GRAVOTURB ENVELOPED-CATALOG IDENTIFIABILITY (AC-IC12)  |  Phase 5")
    print("=" * 78)
    r = ac_ic12_coverage()
    print("\n" + "=" * 78)
    print(f"SUMMARY  AC-IC12: {'PASS' if r['passed'] else 'FAIL'}")
    print("  (multi-freefall catalog inference needs a rho^{3/2} count model — "
          "documented limitation)")
    print("=" * 78)
    return r


if __name__ == "__main__":
    main()
