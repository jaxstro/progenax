# Design: BM19 dense-tail fix via rank (empirical-CDF) copula

**Date:** 2026-06-02 · **Status:** validated (brainstormed with Anna) · **Resolves:** audit **M3**

## Problem

`init_bm19_density_field` builds a turbulent gas field whose *one-point* PDF is meant
to match the BM19 lognormal+power-law law via a Gaussian copula. The audit (M3)
measured the realized dense-tail mass fraction far below theory with huge scatter
(e.g. `f_tail = 0.022 ± 0.020` vs `f_dense ≈ 0.057`), so fractal-substructure ICs
have weaker-than-physical clumping and a validation test masked it.

## Root-cause investigation (measured, not assumed)

The field standardizes the GRF `g` to unit **variance** per realization, then applies
`u = Φ(g)` (standard-normal CDF) and `s = F_V⁻¹(u)`. The bug: at the physical slope
**β≈4**, almost all spectral power is at the lowest k, so the realized `g` is **not
Gaussian** (a sum of ~1–2 sinusoids → narrow, bounded marginal). Standardizing the
*variance* does not fix the wrong *shape*, so `Φ(g)` never reaches the extreme `u`
near 1 and the inverse-CDF never produces `s > s_t`.

Empirical evidence (this investigation):

| Test | Result |
| --- | --- |
| `f_tail` vs resolution N=32→192, β=4 (Φ-copula) | **0.0000 at every N** — *not* resolution-limited |
| `f_tail` vs β at N=64 (Φ-copula) | β=2: 0.023±0.040 (overshoot+scatter); β=3,4: 0.000 |
| **Rank-copula**, β=4, N=128 | **0.0047 ± 0.0000** vs theory 0.0054 (87%, zero scatter) |

So: (1) it is a **copula non-Gaussianity** bug, not a finite-DOF/resolution problem
(a sub-grid cascade — the originally-considered fix — was *proven* resolution-
insensitive at β=4 and is rejected); (2) the **rank / empirical-CDF copula** fixes it.

## Fix: rank (empirical-CDF) copula — "Gaussian anamorphosis"

Replace `u = Φ(g)` with `u = empirical_CDF(g) = (rank(g) − 0.5)/N_cells`. Then `u` is
**exactly uniform by construction at any β**, so `s = F_V⁻¹(u)` reproduces the BM19
marginal exactly. Properties:

- **Exact marginal → realized `f_tail → f_dense`**, and the per-realization scatter
  collapses (the marginal is forced), directly curing the `0.022 ± 0.020` symptom.
- **Spatial correlation preserved:** the rank map is monotone per cell, so high-`g`
  cells stay high-`s` cells — the β-slope turbulent correlation ξ(r) is untouched.
- **Differentiable where it matters:** hard `argsort`-based rank (O(N log N), exact)
  acts on the *frozen* random realization; gradients w.r.t. the physical params
  (σ_s², s_t, α) flow through the BM19 CDF table `F_V⁻¹`, unaffected by the rank.
  Consistent with the existing `stop_gradient` on the field.

**Rejected — soft-rank:** O(N²) (infeasible at 64³ cells), needs a temperature that
re-blurs the exact marginal (the whole point), and its extra gradients are discarded
by the existing `stop_gradient`. Hard rank is correct here.

**Residual:** for extreme `s_t` where the tail count-probability `1 − F_V(s_t) < ~1/N³`,
even uniform `u` cannot place a cell — genuinely unresolvable at that grid. Handle
with a **resolution guard** (warn / suggest larger `grid_size`), not a cascade.

## Implementation

1. **Core:** add `copula: str = "rank"` to `init_bm19_density_field` (and the
   `gaussian_to_bm19` remap) — `"rank"` (empirical CDF, default) vs `"phi"` (legacy
   normal-CDF, retained for comparison). Rank via `argsort(argsort(g.ravel()))`.
2. **Guard:** when `1 − F_V(s_t) < C / grid_size³` (C≈ a few), emit a `UserWarning`
   that the dense tail is unresolved at this resolution; suggest a larger `grid_size`.
3. **Validation (`tests/validation`):** β≈4, several seeds, assert realized `f_tail`
   matches `f_dense_bm19_full` within a regime band AND scatter ≪ the Φ-copula's
   (oracle = the analytic BM19 mass integral; numerical-method-validation).
4. **Caveat + ticket:** document the rank-copula + resolution limit in the
   gravoturbulence theory docs; update `docs/notes/2026-06-02-bm19-field-tail-ticket.md`
   → resolved (link this design).

## Out of scope

End-to-end `∂(stellar f_tail)/∂(Mach)` with an **un**frozen field (would also require
differentiating the discrete cell→star sampling) — a separate architectural effort.
