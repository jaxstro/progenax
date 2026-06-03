# Ticket: BM19 correlated-field dense-tail undersampling

**Opened:** 2026-06-02 · **Status:** ✅ **RESOLVED 2026-06-02 (commit `d654e4f`)** · **Source:** 2026-06-01 audit finding **M3**

> **Resolution.** The undersampling was **not** a finite-resolution / correlation
> sampling-fidelity issue (a sub-grid cascade was considered and *measured* to be
> resolution-insensitive at β=4, so rejected). The real cause was a **copula bug**:
> `gaussian_to_bm19` used `u = Φ(g)` assuming the GRF `g` is standard normal, but at
> the physical slope β=4 the realized `g` is non-Gaussian, so `Φ(g)` mismaps and the
> dense tail collapses (the measured `0.022 ± 0.020` symptom). Fixed by switching to a
> **rank / empirical-CDF copula** (`copula="rank"`, now the default) — `u` is exactly
> uniform → exact BM19 marginal at any β, the monotone rank preserves the turbulent
> spatial correlation, and gradients still flow through the CDF table. Measured at
> Mach=2/α=2/β=4 (`f_dense=0.0568`): `f_tail` `0.029 ± 0.040` → **`0.054 ± 0.000`**.
> A resolution guard warns when `s_t` is so extreme the tail is unresolvable at the
> grid. Design: [`docs/plans/2026-06-02-bm19-rank-copula-design.md`]; validation:
> `tests/validation/test_bm19_field_tail.py`. The diagnosis below is retained as a record.

The forward BM19 *formula* was never in doubt (lognormal+power-law PDF,
`s_t=(α−½)σ_s²`, verified to <1e-13 by direct quadrature); the defect was entirely in
the **sampled 3-D field**, specifically the Gaussian-copula remap.

## Summary

`init_bm19_density_field` builds a 3-D gas density field by exponentiating a
Gaussian random field (GRF) with a turbulent power spectrum `P(k) ∝ k^{−β}`, then
selects the dense tail by a mass-fraction threshold. The realized dense-tail mass
fraction `f_tail` matches BM19 theory **only in the near-uniform / well-mixed
limit** (small β, many independent cells). At the physically motivated slope
**β ≈ 4** the dense tail is spatially correlated — carried by a *handful* of
adjacent cells — so at fixed grid resolution it is **undersampled**, with large
field-to-field scatter.

## Quantitative evidence (audit §3)

| Quantity | Field | Measured | Theory |
| --- | --- | --- | --- |
| dense-tail mass fraction `f_tail` | `init_bm19_density_field`, β=4, 8 seeds | **0.022 ± 0.020** | 0.057 |

The realized `f_tail` is ~2.6× low and has ~90% relative scatter (0.020 on 0.022):
a single rare overdensity dominates each realization. The existing BM19 validation
exercises a near-uniform field, so it passes against the well-mixed theory and gives
**false confidence** in the production (correlated-field) regime.

## Root cause

At β≈4 most spectral power is at the largest scales, so the high-density tail
concentrates into a few correlated cells rather than spreading across many
independent ones. The well-mixed prediction `f_tail ≈ f_dense(s_t, σ_s²)` assumes
cell-independent sampling; correlation breaks that assumption and the fixed grid
cannot resolve the tail's internal structure. This is intrinsic to sampling a
correlated field at finite resolution, not an error in the PDF or threshold.

## Proposed resolution path (research)

1. **Characterize** realized `f_tail(β, N_grid, seed)`: sweep β ∈ [2,4], grid
   resolution, and seeds; quantify mean bias and scatter vs the well-mixed theory.
2. **Convergence**: show whether `f_tail → f_dense` as `N_grid → ∞` at fixed β
   (resolution study), and estimate the resolution needed for β≈4 to reach a target
   tolerance.
3. **Validation test** (the M3 deliverable): a `tests/validation` test that runs
   `init_bm19_density_field(β≈4)` over several seeds and asserts the realized `f_tail`
   **distribution** (mean + scatter band), anchored to the measured regime — not to a
   single run, and not weakened to the well-mixed value.
4. **Caveat + guard**: document the well-mixed assumption in
   `10-theory/gravoturbulence/` and either raise the default field resolution for
   tail-dominated work or emit a runtime warning when `f_sub` is large at low
   resolution.
5. **(Stretch)** spatial-correlation-aware tail allocation: sub-grid or
   importance-sampled dense-tail placement that preserves the correlated structure
   without requiring a prohibitively fine global grid.

## Acceptance criteria

- A seeded, regime-anchored β≈4 validation test exists and passes (asserts the true
  correlated-field `f_tail` distribution, with the bias/scatter documented — *not*
  the well-mixed value).
- The well-mixed caveat is stated in the theory docs, and tail-dominated runs are
  guarded (resolution default or warning).

## References

- Burkhart & Mocz (2019), ApJ 879, 129 — density-PDF / dense-tail framework.
- Audit: `docs/website/90-development-log/code-reviews.md` finding **M3** (§3 evidence).
- Code: `src/progenax/cluster/fdf_density.py` (`init_bm19_density_field`, GRF field
  generation, dense-tail mass-fraction selection).
