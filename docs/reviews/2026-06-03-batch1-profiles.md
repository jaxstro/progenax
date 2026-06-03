# Batch 1 review — profiles (`plummer` · `king` · `eff` · `uniform` · `api`) — gate packet

**Date:** 2026-06-03 · **Branch:** `sota/batch1-profiles` · **Base:** local `main @ 8c7c000` ·
**Reviewer:** Claude Opus 4.8 + astro-review panel (2 agents) · **Engine:** manual + panel
(physics-heavy) · **Status:** 🚦 **AT GATE 1**

## Scope

`profiles/` 1,396 LOC: `plummer.py` (155), `king.py` (508), `eff.py` (197), `uniform.py` (161),
`api.py` (315). Tests: `tests/unit/profiles/*` (7 files) + `tests/validation/test_{plummer,king,eff}_physics.py`.

## What's solid (verified)

- **Plummer:** scale-radius `a = r_h·sqrt(2^(2/3)−1)` correct & consistent (NOT the historical
  inversion); inverse-CDF radii; `Φ(0) = −GM/a`. **Uniform:** `r = R·u^(1/3)` correct.
- **M5 cumulative-shared-grid CDF** (king.py:349, eff.py:91): genuinely fixed — single shared
  grid, monotone by construction (verified strictly increasing), normalized to 1.0, 2nd-order
  convergence measured (`test_cdf_quadrature.py`). No O(N²), no tolerance bands.
- **Differentiability is real, not just finite:** panel ran FD-vs-autodiff on all samplers —
  Plummer d/dr_h 8e-13, EFF d/da 2e-6, **King d/dW0 (through the diffrax ODE + tidal radius +
  CDF) 6e-6**. The `argmax` in `_find_tidal_radius` does not block the W0 gradient (rides the
  linear-interp `t`). King ODE = diffrax Tsit5 (rtol 1e-8), differentiable/jit/vmap-safe.

## Findings ledger

| id | finding | sev | disposition |
|----|---------|-----|-------------|
| **P1** | **King concentration `c(W0)` — RESOLVED, non-issue.** Panel agent 1 flagged the Table-II test as *circular* with "genuine Table II = 0.84/1.18/1.48/1.76". **Verified against the actual King (1966) AJ 71, 64 PDF (Table II, page 73, `log c` column): W₀=3→0.672, 5→1.029, 7→1.528, 9→2.119** — these EXACTLY match the test's reference values. Code reproduces them to ≤0.03. King defines `r_c` via the factor-of-9 normalization (Eq. 15), exactly as the code does. Agent 1's reference values were **hallucinated** (appear nowhere in the paper); the code + test are **correct**. | — (non-issue) | **No fix.** Adversarial verification against the source caught a reviewer false-positive. |
| **P1b** | The Table-II test's `W₀=1 → 0.30` anchor is not in King Table II (which starts at W₀=2.5); only that one point lacks a paper citation. | Minor | fix-now: drop or replace with a tabulated W₀ (e.g. 2.5→0.590), or annotate as extrapolated |
| **P2** | `api.py:287` EFF γ=3 enclosed-mass uses `arctan(x)/arctan(x_t)` — **wrong**; correct is `[asinh(x)−x/√(1+x²)]/…` (~70% error). Used for mass-segregation energy ordering. | Major | fix-now (RED first) |
| **P3** | `api.py:286` `if gamma == 3.0:` → **TracerBoolConversionError under `jit`**, and `grad` wrt `gamma` is **silently 0.0**. | Major | fix-now (`jnp.where`/`lax.cond`) |
| **P4** | No **FD-vs-autodiff grad-checks** on any public sampler — existing grad tests assert only finiteness. (Gradients verified correct by panel, so this is hardening, not a bug.) | Major (gap) | fix-now: add central-difference grad-checks on `sample_positions` wrt r_h/a/γ/W0 |
| **P5a** | `king.py:473` `density()` docstring describes the *old, rejected* K-function form; implementation uses the (correct) lowered-Maxwellian. Docstring contradicts code. | Minor | fix-now (docstring) |
| **P5b** | `eff.py:24` docstring cites "EFF87 Eq.1" without noting it is the *projected* SB form applied here as a 3-D volume density (a common IC convention). | Minor | fix-now (docstring note) |
| **P5c** | `@jax.jit` on bound methods `_sample_radii` (plummer.py:92, uniform.py:97) — fragile (re-traces per instance); King/EFF correctly don't. | Minor | ticket |
| **P5d** | Test nits: `test_jit_compatible` smoke tests assert only finiteness (not jit==eager); tighten King CDF-quadrature tol toward achievable 1e-6; truncation tests use 1% slop where exact holds; one redundant Table-II test. | Minor | fix-now where trivial, else ticket |

## Decisions requested at Gate 1

- **D-P1 (science):** Confirm the King (1966) Table II concentrations. My read: code + test are
  **correct** and the panel agent supplied wrong reference values. Options: (a) code correct —
  no fix; (b) agent correct — code has a Major `r_c`↔`r₀` bug; (c) verify definitively (install
  LIMEPY / check the paper).
- **D-P2 (scope):** fix P2 + P3 (EFF potential + jit/grad) and add P4 grad-checks now; P5
  docstrings fix-now, P5c/P5d-nontrivial → tickets?

## Resolution (Gate 1 approved)

| id | resolution |
|----|------------|
| P1 | **No fix** — King concentration verified CORRECT vs the King 1966 PDF (Table II `log c`: W₀=3→0.672, 5→1.029, 7→1.528, 9→2.119, reproduced to ≤0.03). Reviewer false-positive. |
| P2+P3 | **Fixed (true potentials).** `compute_profile_potential`: King → `Φ=−σ²·ψ(r)` (ψ from ODE, σ² consistent with KingVelocityDF); EFF → true `Φ=−G[M(<r)/r + 4π∫ᵣ^rt ρs ds]` from the exact enclosed mass; Plummer exact. No `gamma==3.0` branch → jit/grad-safe. |
| P4 | **Fixed.** `test_profile_gradients.py` — FD-vs-autodiff grad-checks on all samplers (Plummer r_h, EFF a/γ, King r_c/W₀-through-ODE). |
| P5a | **Fixed.** king.py `density()` docstring now describes the lowered-Maxwellian (was the rejected K-function form). |
| P5b | **Fixed.** eff.py docstring + theory/per-paper notes: EFF87 Eq.1 = surface brightness; progenax γ is a 3-D slope. |
| P1b | **Fixed.** Dropped the W₀=1 Table-II anchor (not tabulated; Table II starts at 2.5). |
| P5c/P5d | **Ticketed** → [docs/notes/2026-06-03-profiles-jit-on-bound-methods-ticket.md](../notes/2026-06-03-profiles-jit-on-bound-methods-ticket.md). |

Plus paper-grounding (per Anna's no-assumptions workflow): king-1966.md, elson-fall-freeman-1987.md,
plummer-1911.md verified/expanded vs the PDFs; **removed the fabricated King-vs-LIMEPY comparison**
an earlier session invented (the test references LIMEPY zero times).

## Verification (Gate 2 evidence)

- **Profiles subset:** `pytest tests/unit/profiles/ tests/validation/test_{king,eff,plummer}_physics.py`
  → **128 passed**.
- **Full suite (regression):** `pytest tests/` → **990 passed, 0 failed** (was 979) — the api.py
  true-potential change feeds cluster mass-seg; no regression.
- **True-potential RED→GREEN:** King `Φ(r_t)≈0` + monotone; EFF jit-safe + `grad wrt γ` finite/nonzero;
  EFF enclosed mass matches the profile CDF to 1e-9; Plummer exact to 1e-12.
- **MyST build:** clean (King + EFF + Plummer doc edits; cite keys + xrefs resolve).
- **Diff:** 10 modified + 4 new files (+277 / −141).

## Gate status

- **G1 — findings + fix plan:** ✅ approved (D-P1 verified non-issue; D-P2 fix-bugs+grad-checks+docstrings; true-potential redesign).
- **G2 — diff + local verification before commit:** 🚦 **awaiting Anna** (this packet).
- **G3 — merge to local main:** ☐
