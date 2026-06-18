# df_moment_dispersion W₀-gradient — discriminating experiment + lock (design)

**Date:** 2026-06-18
**Branch:** `feat/df-moment-w0-gradient-lock` (off `main` @ 351f6ec)
**Status:** design ratified with Anna (brainstorming); experiment RUN, decision = H0
**Owner:** Anna Rosen (single HITL); agent executes under per-step approval

## 1. Context & goal

The #4 Michie-W₀ arc (ADR-0016, merged) made the OED/Fisher *jeans* dispersion
forward models (`jeans_dispersion` / `project_dispersion`) W₀-differentiable for all
truncated models by replacing a C⁰ piecewise-linear back-interpolation with a
monotone C¹ PCHIP (`_pchip_interp`). It left an explicitly-deferred follow-up: the
**separate** `df_moment_dispersion` path (the exact Michie DF 2nd-moment "Tier B"
dispersion) carries its **own** `jnp.interp` on a W₀-dependent solve, and STATUS
flagged it as a *candidate same-cause C⁰-kink* — **UNVERIFIED**.

**Goal:** determine whether ∂(σ_r, σ_t, σ_1d, β)/∂W₀ through `df_moment_dispersion`
is AD-correct; if a real kink exists, fix it on merit; then **lock** the W₀ axis
(it is currently `# W0 deferred` in the grad-audit registry, `registry.py:2084` /
`manifest.py:120`).

**Honest relevance (no-overselling).** STATUS said this "feeds OED/Fisher
W₀-differentiability like the Michie arc did." On inspection that is **not quite
right**: the OED Fisher forward model is `project_dispersion`, which reads the
*jeans* tables (`_jeans_tables` / `_sigma_r2_from_tables`), **not**
`df_moment_dispersion`. So this thread is about **completeness of the
differentiability audit** (every public differentiable entry point clean — the
Fisher-integrity goal of the grad-audit arc) and `df_moment_dispersion` being a
trustworthy public symbol. It does **not** unblock OED directly.

## 2. Mechanism analysis (predicted, then confirmed)

The suspect is `dispersion.py:625`:
`W = jnp.interp(r_i / df.r_c, df.xi_grid, df.psi_grid, left=df.W0, right=0.0)`.

- In the Michie *jeans* arc the kink came from interpolating onto an `s`-grid whose
  **endpoint was `r_t(W₀)`** — *moving nodes*. A fixed query radius crossed grid
  nodes as W₀ varied → slope jump → C⁰ kink. PCHIP fixed it.
- Here, `xi_grid` is built at `king.py:237` as
  `jnp.linspace(1e-6, xi_max, n_points)` with **`xi_max` / `n_points` static**
  (not traced from W₀). The **nodes are fixed**; only the `psi` *values* move with
  W₀. For a fixed query `jnp.interp` always selects the same two bracketing nodes,
  so its W₀-derivative is a smooth linear blend of `dψ/dW₀` — **no node-crossing
  kink.** The moments are then smooth polar-quadrature integrals of `W` (no
  boundary mask). The remaining theoretical suspect is the `max(ψ_raw,0)` /
  `max(W,0)` clamp near `r_t` (C⁰ in W₀), which could only bite radii near `r_t`.

**Prediction:** clean at interior; at worst a clamp kink near `r_t`.

## 3. Discriminating experiment (pre-registered)

| Slot | Filled |
|---|---|
| **H0 (null)** | ∂σ/∂W₀ is AD-correct everywhere; "W0 deferred" was conservative. |
| **H1 (clamp kink)** | kink localized near `r_t`; interior clean. |
| **H2 (deep kink)** | kink at deep interior → deeper cause (ODE/`mu`/`wmax`). |
| **Observable** | rel err `|g_ad − g_fd|/(|g_ad|+|g_fd|)` vs central-FD step `h ∈ {1e-3,1e-4,1e-5,1e-6}`, at `r/r_t ∈ {0.1,0.5,0.9}`, W₀=6, r_a=5, fixed `xi_max=800` (W₀-consistent domain). |
| **Decision rule** | **H0** if min-over-h rel < 1e-4 at all three radii. **H1** if interior+mid < 1e-4 but near-`r_t` ≥ 1e-3 and non-converging as h↓. **H2** if deep-interior (0.1 `r_t`) ≥ 1e-3. Else inconclusive→redesign. Convergence = central FD must approach AD as h↓. |

**Valid FD-step discipline (Anna-directed).** "AD-correct" requires the central FD
to *converge* to AD as h↓ — a single coarse step carries truncation error and a
single tiny step carries round-off. The valid window for this path is **h ≈
1e-5…1e-4** (floor ~1e-8); h=1e-2 shows pure truncation (rel ~1e-4) that vanishes
as h↓. The grad-audit engine default `h_rel=1e-4` → `h=6e-4` at W₀=6 lands at rel
~1e-7, inside the window.

## 4. Results — H0 accepted

Measurement-only scratch runs (float64 on via `import progenax`):

| W₀ | radii | channels | best rel-err range |
|---|---|---|---|
| 5.0 | 0.1, 0.5, 0.9 r_t | σ_r, σ_t, σ_1d, β | 2e-11 … 8e-9 |
| 6.0 | 0.1, 0.5, 0.9 r_t | σ_r, σ_t, σ_1d, β | 9e-11 … 4e-8 |
| 7.0 (near-divergence, r_t=545) | 0.3 r_t | σ_r | converges to 1e-8 at h=1e-4 |

Central FD converges to AD as h↓ at **all** radii and channels (classic
truncation/round-off bowl, sweet spot ~1e-5); the M-case sanity passes (rel ~1e-10).
Even at W₀=7 (near the Michie mass-divergence, `r_t`→545) AD is correct — the only
"large" number (1.2e-4 at h=1e-2) is coarse-FD truncation that vanishes as h↓.

**Decision: H0.** No kink; the STATUS "same-cause" hypothesis is **falsified**,
exactly as the fixed-node inspection predicted. **No `src/` fix is needed.**

## 5. Scope — lock-only (no src change)

1. **Grad-audit registry W₀ case.** Add `_df_moment_dispersion_W0(W0)` (builds
   `MichieVelocityDF(W0=W0, r_c=1.0, r_a=5.0, xi_max=800.0, n_ode_points=3000)`
   inside so W₀ flows through; reduces `σ_r` over interior + near-`r_t` radii via
   `identity_sum`) + a `Case(id="df_moment_dispersion[Michie]", param="W0",
   theta0=6.0, expect="consistent", tol=1e-3)` (engine `h_rel=1e-4` → valid step).
   Add the `("df_moment_dispersion[Michie]", "W0")` manifest description; flip the
   `# W0 deferred` notes. Net registry cases +1 (audited count unchanged — symbol
   already AUDITED).
2. **Regression test** in `tests/unit/kinematics/test_dispersion.py` mirroring
   `test_grad_jeans_michie_wrt_W0`: `test_grad_df_moment_michie_wrt_W0`
   (`_assert_ad_fd(f, 6.0)`, default h=6e-4, measured rel ~1e-7) **plus** a
   high-W₀ AD-correct test paralleling `test_grad_jeans_michie_high_W0_ad_correct`
   (W₀=7: AD correct, coarse-FD disagreement is an FD artifact that shrinks with h).
3. **ADR** (`/adr`, ADR-0017): record the falsified same-cause hypothesis, the
   fixed-node-vs-moving-node mechanism distinction, and the confirmed-clean finding.
4. **Regen** grad-audit JSON (`scripts/audit_gradients.py`) + dashboard re-stamp
   (`scripts/build_test_dashboard.py --emit --render`) + ratchet bump
   (`test_dashboard_gen.py` case count). **No `src/progenax/` edit → coverage
   staleness gate is NOT triggered** (it keys on src changes); only the grad-audit
   JSON + dashboard freshness gates need regen.

## 6. Definition of done / gates

- New W₀ registry case + 2 regression tests pass; `audit_gradients.py` exit 0,
  `0 hazard(s)`, case count +1.
- Dashboard JSON fresh (staleness gate green); ratchet updated.
- FULL released-core gate green (`-n auto`, XLA thread caps) — expected unchanged
  pass count + the 2 new unit tests.
- ADR-0017 recorded; STATUS updated; design + this finding committed.
- **No `src/` change.** If review disputes H0, the fix path (PCHIP-on-`W`, or
  clamp-smoothing) re-opens under a separate explicit go.

## 7. Provenance of the falsified hypothesis

The STATUS "candidate same-cause C⁰-kink (UNVERIFIED)" is retained here as a
falsified premise (cf. ADR-0016's retained no-op premise): the diagnosis structure
(a W₀-dependent interp) was shared, but the *mechanism* (moving endpoint nodes)
is absent in the DF-moment path (fixed linspace nodes), so the conclusion does not
transfer. Verified by the experiment in §4, not asserted.
