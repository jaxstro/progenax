# Pre-release SoTA-design + validation agenda — progenax

**Opened:** 2026-06-03 · **Status:** 📋 **OPEN — next-session agenda** · **Owner:** Anna

## Goal

Harden and finalize progenax to **publication quality for a methods paper + public
release**. The 2026-06-01 audit and its 2026-06-03 follow-up resolved every correctness
Critical/Major (package at **A**); test coverage is **91 %** and CI coverage is decoupled
from the pass/fail gate. What remains before release is a **deliberate, per-module
SoTA-design + validation pass** (done carefully, module by module, in a dedicated session)
plus the evolved-dynamics science validation.

## A. Per-module SoTA-design + validation pass (next session)

Go through each module deliberately — design review (responsibilities, public API,
numerics), validation (vs theory / literature, FD gradient-checks on public entry points,
units, provenance of constants), and docstrings — applying the audit's per-lane discipline
*deliberately* rather than reactively. Concrete known items to fold in:

1. **`cluster/fdf.py` (767 LOC) — deliberate split.** The only genuinely-large file; two
   clear concerns (displacement-layer physics vs IC-pipeline orchestration). Split as a
   *design* exercise (thoughtful module boundaries + a byte-identical RNG check on
   `generate_fractal_ic`), not a mechanical chop. (From the file-length ticket.)

2. **`imf/smooth.py` CDF numerics — cumulative-shared-grid integration.**
   `TaperedPowerLaw` / `Schechter` build their CDF via `_scalar_cdf_unnorm`, which
   **re-grids `[m_min, m_val]` independently for every upper limit** (`_linear_trapz_integrate`)
   — O(N²), and O(1e-4)-relative quadrature wiggle between adjacent CDF points (the current
   `test_cdf_unnorm_array_*` monotonicity tests therefore use a quadrature-aware floor).
   **SoTA fix:** a cumulative integral on a **single shared grid**
   (`concat([0], cumsum(0.5·(f[1:]+f[:-1])·dr))`) — monotone *by construction*, O(N),
   matching the **M5 King/EFF pattern** already in the codebase. Add **log-spacing** for the
   `m^−α` spike near `m_min`, and/or use the **analytic power-law primitive**
   (`∫ m^−α dm = m^{1−α}/(1−α)`) so numerics handle only the taper/exponential factor; then
   **tighten the monotonicity test to exact**. (Note: `Maschberger` already uses an exact
   analytic inverse — unaffected. Not a bug today — the wiggle is ~100× below Monte-Carlo
   sampling error — but it's the right SoTA design and removes a tolerance band.)

3. **General, per module:** confirm numerical order/convergence, differentiability
   (FD-vs-autodiff on public entry points), unit consistency, constant provenance, and
   docstring accuracy.

## B. Evolved-dynamics science validation (pending gravax)

The audit's **#1 stated limitation**: the M1/M2/M6 King/EFF/Plummer equilibria were verified
only at *t = 0* (`Q ≈ 0.5`). Once **gravax is independently cleaned + verified**, evolve the
generated ICs (PEFRL / IAS15) over many crossing times + a relaxation time and confirm
**stationarity** — density profile, Lagrangian radii (r₁₀/r₅₀/r₉₀), `Q(t)`, no secular
expansion/collapse. `build_spatial_ic` is now genuinely differentiable, feeding gradient-based
inference. Produces the validation scripts + publication plots the "Definition of Complete"
expects.

## C. Policy / state recorded this cycle

- **File-length:** 500 LOC is **preferred, not hard**; cohesive files ≤ ~600 are accepted.
  Only `cluster/fdf.py` (767) warrants a split (→ A.1); the other borderline files
  (507–573) are kept. See `docs/notes/2026-06-02-file-length-followup-ticket.md`.
- **`cluster/fractal_gw_legacy.py`** — deprecated but public + live in `core.py`; a separate
  **retire-vs-split** decision (migrating `core.py` to the FDF path is behavior-changing).
- **Coverage tooling:** `--cov=<submodule>` aborts (jaxlib/abseil double-init); use
  `--cov=progenax`. CI coverage is a non-fatal step. See
  `docs/notes/2026-06-03-coverage-jaxlib-narrow-scope.md`.
