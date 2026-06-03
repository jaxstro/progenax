# Decision: King density-potential relation + nondimensionalization (subsumes audit M6)

**Opened:** 2026-06-02 · **Status:** Implemented in Batch 2 (B2.0), awaiting Anna's sign-off
**Scope:** `src/progenax/profiles/king.py` · **Severity:** Major (newly discovered; not in the 2026-06-01 audit)

## Summary

While preparing the Batch-2 King *velocity* DF (true lowered-Maxwellian), I found that
`KingProfile` solved Poisson with the **wrong 3-D density-potential relation**. The audit
verified the *K-function's definition* matched King (1966) but never checked its *use* in the
volume-density Poisson RHS. Two coupled errors were present and mutually masking:

1. **Density relation.** The code used King's K-function (an incomplete-gamma / projected-style
   function) as the 3-D volume density: `rho_tilde = [K(W0) − K(W0−psi)] / K(W0)`. The correct
   King lowered-Maxwellian volume density is

   ```
   rho_hat(W) = e^W erf(sqrt W) − (2/sqrt pi) sqrt(W) (1 + 2W/3)
   ```

   (Binney & Tremaine 2008, Eq. 4.131; King 1966). The code's relation over-extends the profile.

2. **Nondimensionalization (audit M6).** The ODE used `psi'' + (2/xi)psi' = −rho_tilde`, omitting
   the standard factor of 9. King (1966)/BT08 carry `= −9 rho_tilde`, with `xi = r/r_c` and
   `r_c` the King core radius `r_0 = sqrt(9 sigma^2 / 4 pi G rho_0)`.

## Evidence

**Density relation (literature-free oracle — direct velocity integration of the DF):**
`rho(W) ∝ int_0^{sqrt(2W)} v^2 (e^{W−v^2/2}−1) dv` matches `rho_hat(W)` to 6 decimals at every W,
and the code's K-form by **2.9× at psi=3.5, ~30× near the tidal edge**:

| psi | direct integral | rho_hat (BT08) | code K-form |
|-----|-----------------|----------------|-------------|
| 3.5 | 0.023908 | 0.023908 | 0.069194 |
| 0.5 | 0.000057 | 0.000057 | 0.001736 |

**Concentration c(W0)=log10(r_t/r_c) vs King (1966) Table II** — only the corrected density
**and** factor-of-9 reproduce the table (~1%); the two old errors partially canceled, hiding the bug:

| W0 | current code | corrected (rho_hat + factor 9) | King Table II |
|----|--------------|--------------------------------|---------------|
| 1 | 0.48 | 0.296 | 0.30 |
| 3 | 0.82 | 0.672 | 0.67 |
| 5 | 1.08 | 1.029 | 1.03 |
| 7 | 1.41 | 1.528 | 1.53 |
| 9 | 1.86 | 2.119 | 2.12 |

## Decision (implemented)

Replace the density relation with `king_lowered_maxwellian_density(rho_hat)` at all three sites
(`_king_poisson_rhs`, `KingProfile.__init__`, `KingProfile.density`) and adopt the standard
factor-of-9 RHS, so `r_c` **is** the King core radius. This **subsumes M6** (no separate
factor-of-9 decision needed) and is validated against King Table II.

`king_K_function` was initially retained, then **removed in Batch 4** (commit `629770b`): once it
was no longer the volume density it had zero production callers, and its docstring still
advertised the disproven density relation `ρ/ρ₀ = [K(W0)−K(W0−ψ)]/K(W0)`. Its W=0 gradient-safety
(audit C2) is now covered on `king_lowered_maxwellian_density` instead.

## Blast radius (please confirm acceptable)

- **King sampled positions change** for everyone: for a given `(W0, r_c)`, `from_W0_rc` now yields
  a different `r_t` (e.g. W0=7, r_c=1: r_t 25.9 → 33.7) and a more centrally-concentrated profile
  matching the true King model.
- **ODE domain widened**: the corrected high-W0 models extend further (xi_t(W0=9) ≈ 131), so
  `solve_king_profile`/`from_W0_rc` defaults were bumped `xi_max 100→300`, `n_points 500→2000`.
  Very high W0 (>~10) still needs an explicit larger `xi_max` (pre-existing "grid-edge" minor).
- **No suite regressions**: full suite `847 passed, 1 skipped`. The only test that needed updating
  was the Batch-1 King CDF quadrature reference (re-pointed to `rho_hat` — it validates the
  trapezoid rule, not the density form).

## Provenance

- King, I. R. (1966), AJ 71, 64 — model, density relation, **Table II** concentrations.
- Binney & Tremaine (2008), *Galactic Dynamics* 2nd ed., §4.3, **Eq. 4.131** — `rho_hat(W)` and the
  factor-of-9 nondimensionalization.

## Sign-off

- [x] **Anna confirms (2026-06-02)** the corrected King model (changed sampled positions) is
  acceptable for launch. Cross-checked the density relation and factor-of-9 against King (1966)
  directly.
