# Differentiability Gradient-Audit Harness — Design

**Date:** 2026-06-13
**Status:** Design validated (Phase 1 brainstorm, HITL with Anna). Awaiting Phase 2 plan.
**Branch:** `feat/differentiability-audit`

## Why this exists

progenax's value proposition (`docs/website/60-science-demos/throughline.md`) is that a
**differentiable** forward model turns cluster inference into information geometry: every
initial condition is `jax.grad`-able, so the Fisher matrix is computable and forecasts what a
survey can and cannot measure *before* observing.

The Fisher information is `F = JᵀJ` (see `scripts/_demo_inference.py::fisher_information_gn`).
A silently **zeroed** column of `J` makes that parameter look infinitely well-constrained or
makes `F` rank-deficient — a confident **wrong** uncertainty with **no error raised**. A NaN at
least crashes loudly; a zeroed gradient does not. **Gradient integrity = Fisher integrity.**

This arc builds a systematic AD-vs-FD gradient-audit harness over every public sampling/physics
entry point, classifies each into clean / known-limitation / hazard, fixes the confirmed
hazards (audit-first triage with Anna), and tracks the campaign in a living validation website
doc whose numbers are the *same* numbers the test gate asserts on.

## Protected gradient directions (scope)

Two first-class guarantees:

1. **params → binned summary statistic** — the Fisher/inference path:
   `d(σ(r), N(r), β(r), mass function) / d(r_h, W0, α, r_t, r_a, …)`.
2. **params → sampled IC** — the CLAUDE.md promise: `jax.grad(loss)(r_h)` through
   `build_spatial_ic`; `d(sampled positions/velocities/masses) / d(params)`.

**Out of scope:** the DATA side is always frozen and never differentiated. A clip on the frozen
random quantile `u` (e.g. `chabrier.py:325`) is harmless. Only clips/kinks on **param-dependent**
quantities on a protected path matter.

## Decisions locked in the Phase-1 brainstorm

### D1 — Architecture: shared core + two thin layers
One pure engine, two consumers, so the gate and the website table can never disagree.

```
tests/validation/grad_audit/
  core.py        # audit_entry_point(case) -> AuditResult   (the shared engine)
  registry.py    # REGISTRY: list[Case]                      (entry points × directions × params)
  reductions.py  # mean_radius, mean_speed, mean_mass, ...   (per-channel scalars)
tests/validation/test_grad_audit.py   # thin: @parametrize(REGISTRY) -> assert per expect-class
scripts/audit_gradients.py            # thin: REGISTRY -> validation/data/grad_audit_results.json
docs/website/50-validation/differentiability-audit.md   # cites the JSON numbers
```

The website doc cites `grad_audit_results.json`, which is produced by the same `REGISTRY`/engine
the pytest gate asserts on. The "every entry must cite a real measured number" rule is therefore
enforced **by construction**, not by diligence.

### D2 — Outcome taxonomy: explicit `expect`, limitations PINNED (not skipped)
A zero gradient is a bug on a protected path but correct-and-intended at king `r_t` and
segregation `stop_gradient`. So each case declares an expectation, and the assertion differs:

- `consistent` → assert finite **and** `|ratio−1| < tol` **and** `|AD| > eps` (silent-zero guard).
- `known_zero` → assert `|AD| < eps` **and** `|FD| < eps`. **Pinned**: fails loudly if the
  gradient ever stops being zero — alerting us to an unannounced physics change.
- `known_blocked` → assert finite; record AD; compare against a *coarse* FD only (STE surrogates,
  measure-zero branch points). Do not demand `AD≈FD`.

### D3 — AD-vs-FD numerical policy: per-class tol + relative step
- Central finite differences, relative step `h = h_rel · max(|θ|, 1)`, `h_rel = 1e-4`
  (overridable per case).
- Per-class tolerance (measured-first: freeze at measured ±3σ over ≥3 seeds after the first run):

  | Entry-point class | default tol `|ratio−1|` | rationale |
  |---|---|---|
  | closed-form (Plummer CDF, IMF ppf/cdf, Kepler) | `1e-5` | float64 analytic |
  | ODE/quadrature (King, Michie, Eddington, MCC)  | `1e-3` | diffrax adaptive-step noise |
  | sampled → binned-summary (T3)                  | `1e-3` | binning + finite-N |
  | STE (`known_blocked`)                          | coarse only | surrogate ≈ FD-of-step |

- The **measured ratio is always recorded** to JSON, regardless of which band it passed — so the
  website shows the real number and a closed-form path silently degrading from 1e-9 to 9e-4 is
  visible even though both "pass" 1e-3.
- Reverse-mode `jax.grad`/`jacrev` (ODE `custom_vjp`-safe, mirroring
  `_demo_inference.fisher_information_gn`).

### D4 — params→IC reduction: per-channel physical scalar
The sampled N×3 output (frozen key ⇒ deterministic in θ) is reduced to a scalar matched to the
channel the parameter actually moves: `mean‖r‖` (positions), `mean‖v‖ ∝ dispersion` (velocities),
`mean(m)` (masses). The `consistent` silent-zero guard (`|AD| > eps`) flags a reduction that
accidentally cancels so we replace it rather than get a false pass.

### D5 — Coverage: full surface, built in tier order (checkpoint per tier)
The target is *every* public entry point, built in four tiers so we checkpoint with Anna.

**T1 — headline samplers + packaged summaries + pinned limitations (~16 cases)**

| Entry point | Dir | param(s) | expect | edge probe |
|---|---|---|---|---|
| `PlummerProfile.sample_positions` | →IC | r_h | consistent | — |
| `PlummerVelocityDF` (+OM) | →IC | r_h, r_a | consistent | r_a→0.75a (Merritt bound) |
| `KingProfile.from_W0_rc`→positions | →IC | W0, r_c | consistent | W0=12 (high-conc core) |
| `KingVelocityDF.sample_velocities` | →IC | W0 | consistent | W0=12 |
| `Michie` profile+DF | →IC | W0, r_a | consistent | — |
| `EFFProfile`/`EFFVelocityDF` | →IC | γ, r_t, r_a | consistent | γ=2.01 (near-divergent) |
| `build_spatial_ic` (e2e: virial+COM) | →IC | r_h | consistent | — |
| `PowerLawIMF.sample`/`.ppf`/`.cdf` | →IC / →summary | α, m_min | consistent | α=1±1e-3; cdf@m_min (H4) |
| `ChabrierIMF.sample` | →IC | m_c, σ, α | consistent | sample→m_min (H6) |
| `Maschberger`/`Schechter.ppf` | →IC | μ, α, β | consistent | — |
| `IMF.logpdf` / `mean_mass` | →summary | α | consistent | α=1±1e-3 |
| `q_approx`, `lambda_msr_approx` | →summary | positions(r_h) | consistent | — |
| **king `r_t` via `from_W0_rc`** | →IC | W0 | **known_zero** | PIN d r_t/dW0 = 0 |
| **`segregation_approx` scale** | →summary | — | **known_zero** | PIN stop_gradient |
| **`apply_tidal_truncation`** | →IC | r_t | **known_blocked** | STE ≈ coarse-FD |
| **IMF ppf at α=1.0 exactly** | →IC | α | **known_blocked** | branch-limited point (AD=0) |

**T2 (~8):** MultiComponentCluster Engine A & B `sample_cluster` (W0, g, δ, w_j / r_h_j, γ_j, r_a_j)
— probes **H2** (ψ=0 mask) and the shared `r_t` kink; `KeplerElements.to_state` (e=0.999);
`resolve_binary_components`; `MoeCompanions.sample`; rotation overlays (ω).

**T3 (~4):** `build_spatial_ic → binned σ(r)/β(r)/N(r)` via a frozen-edge binner vendored into the
harness — the literal Fisher path the demos use (the binners themselves are scripts-local).

**T4 — consolidation (the release gradient-gate).** Inventory every existing gradient test;
migrate its AD-vs-FD assertion into the registry; delete the redundant finite-only smoke tests
(fixes audit **T6**: "7/9 in `test_jax_compatibility.py` assert sign/finiteness only — a wrong-by-2×
gradient passes"); **keep co-located** the non-gradient properties (the `find_alpha` IFT
forward-value regression `REF_ALPHA`; the Engine-B analytic β(r) physics anchor; the α=1 kink
check). Result: the registry is the single, published gradient coverage map (audit §11).

### D6 — Confirmed-but-unfixed hazard encoding: `xfail(strict=True)` + hazard-id
Audit-first means we confirm some hazards before fixing them, but the FULL gate must stay green to
commit the harness. A confirmed hazard becomes
`@pytest.mark.xfail(strict=True, reason="HAZARD <id>, pending triage")`:

- suite stays green; hazard is documented + tracked (website status ⏳);
- when later fixed, the test **xpasses**, which under `strict` **fails** — forcing us to drop the
  marker and flip the website row to ✅-fixed. A self-cleaning ratchet.
- `known_zero`/`known_blocked` cases are **not** xfail — they are real passing pins.

## Case schema

```python
Case(
  id="build_spatial_ic[Plummer]",   # stable key; the website-table row key
  direction="params->IC",            # or "params->summary"
  fn=...,                            # θ -> output  (closure over a frozen key/data)
  param="r_h", theta0=1.0,           # baseline (generic) value
  reduce=mean_radius,                # scalar reduction (IC dir); identity for summary
  expect="consistent",               # consistent | known_zero | known_blocked
  tol=1e-3, h_rel=1e-4,              # per-class override
  edges=[EdgeConfig(...)],           # curated boundary probes, each → a hazard-map id
  hazard_id=None,                    # set when an edge confirms a hazard (xfail link)
)
```

`AuditResult` (returned by `audit_entry_point`, serialized to JSON):
`{id, direction, param, theta, finite, ad, fd, ratio, abs_ad, status}` where
`status ∈ {clean, known-limitation, hazard-pending, fixed}` is **computed** from
`(expect, finite, |ratio−1|<tol, |ad|>eps)`.

## Status computation (never hand-set)

```
expect=consistent    & finite & |ratio−1|<tol & |ad|>eps      → clean ✅
expect=consistent    & (¬finite | ratio off | ad≈0)            → HAZARD 🔧 → xfail(strict,id) → ⏳
expect=known_zero    & |ad|<eps & |fd|<eps                     → known-limitation ⚠ (pinned)
expect=known_blocked & finite                                  → known-limitation ⚠ (surrogate)
xfail case later passes                                        → strict-fail → drop marker → fixed ✅
```

## Preliminary probe results (measured 2026-06-13, float64, central FD)

Early de-risking of the suspected in-scope hazards (full evidence in the Phase-2 RED steps):

| Probe | Measured | Reading |
|---|---|---|
| H4 `power_law.cdf` d/d m_min, m_query=m_min+1e-3 | AD/FD = 1.000000 | clip benign where the value is live |
| H4 same, m_query<m_min | AD=FD=0 | correct (F=0 below support) |
| H6 `chabrier.ppf` d/d m_c, u∈{1e-4,1e-2,0.2} | AD/FD = 1.000000 | Newton clamp `:371` not biting |
| α=1.0 exactly, `ppf` d/dα | AD=0 vs FD=−2.37e-4 | documented branch-limited point (→ `known_blocked`) |
| IMF gradient suite (`test_imf_gradients.py`) | 25 passed | interior path clean |

**Reading:** H4/H6 are *preliminarily benign* — reclassified from "suspected hazard" to
"verify-benign"; the harness still pushes them to the exact boundary (u→1e-12 pinning a sample at
`m_min`) before marking ✅. The α=1-exactly point is a known measure-zero branch limitation, so the
IMF-ppf α edge splits: `α=1±1e-3` → `consistent` (tight), `α=1.0` → `known_blocked` (finite-only).
After this micro-audit the live suspected-hazard surface narrows toward **H2 (multicomponent ψ=0
masks)** and the **shared `r_t` kink** (king + MCC Engine A via `_find_tidal_radius`).

## Corrected hazard map (Phase-0 re-verification, 2026-06-13)

All sites freshly re-grepped present; PR #6 (`7d1f402`) confirmed in `main`.

| id | site | on protected path? | verdict |
|---|---|---|---|
| H1 | `multicomponent.py:273-276` | →IC, →summary | mask at :276 is a no-op (`r_grid=linspace(0,r_t)`); the issue is `r_t` from `_find_tidal_radius` → d/dW0=0 (= king root cause) |
| H2 | `multicomponent.py:266,478` | →summary (N_frac, Q_j) | boundary masks at ψ=0; ρ̂→0 there so small but non-zero/unmeasured — **measure** |
| H3 | `multicomponent.py:454,504,512` | eval / total_density | same `r_t`-kink family |
| H4 | `power_law.py:227` cdf clip | →summary | **preliminarily benign** (measured above) |
| H5 | `chabrier.py:342,349,360` | Newton initial-guess | benign (washes out over 30 iters) |
| H6 | `chabrier.py:371` Newton clamp | →IC | **preliminarily benign** (measured above) |
| H7 | `tidal.py:106,172` | →IC | **not a hazard** — `custom_jvp` STE by design (`known_blocked` pin) |

**Intentional (do NOT change without Anna):** `king.py:282` argmax r_t (d/dW0=0, deferral doc
`docs/plans/2026-06-08-king-differentiable-tidal-radius-deferred.md`); `segregation_approx.py:145`
`stop_gradient` (sets softmin sharpness only; gradients flow through `dist`).

## Artifacts (the 5-requirement Definition of Done)

1. `tests/validation/test_grad_audit.py` — the parametrized gate.
2. `scripts/audit_gradients.py` — emits `validation/data/grad_audit_results.json` + a markdown table.
3. `validation/plots/grad_audit_*.png` — AD-vs-FD scatter (ratio per case) + per-direction summary.
4. Quantitative table printed by the script (id | dir | param | ratio | status).
5. `.claude-work/TASK_*_COMPLETE.md` completion doc.

Plus the living website doc `docs/website/50-validation/differentiability-audit.md`, wired into
`myst.yml` under the validation section, with: a *gradient integrity = Fisher integrity* intro; the
status table (numbers from `results.json`); an **"intentional non-differentiable sites"**
subsection (king `r_t`, segregation, α=1 branch point) with rationale; and a **changelog of fixes**.
Site builds to **0 warnings**.

## Fix strategies (triaged AFTER measurement — audit-first)

Candidate strategies per hazard class, each flagged because it **changes sampled physics** and
must be ratified by Anna before implementation:

- **Smooth/soft truncation** over a small width (replace a hard `where(r≤r_t,…,0)` with a logistic
  taper) — changes the model near `r_t`; width is a physics choice.
- **Implicit-function / STE handling** for `r_t` (precedents in-tree: `find_alpha_for_masses`
  custom_vjp/IFT; `tidal.py` STE) — makes `r_t` differentiable without a hard cut. The deferred
  king `r_t` option (b) if Anna later chooses to fix it.
- **Remove a harmful output clip** — only if measurement shows it zeros a live gradient.

No fix is implemented before its hazard is *measured-confirmed* and the strategy *ratified*.

## Workflow & gates

- TDD strict: RED (failing/edge test shown against current code) → GREEN. Never weaken a test; if a
  fix changes expected physics, derive the new expectation analytically in the test.
- FAST gate per task; FULL gate per tier checkpoint (commands in `progenax/CLAUDE.md`).
- Units STELLAR; G explicit. JAX-native (`jnp`, `lax.scan`/`fori_loop`, no `while_loop`).
- Branch `feat/differentiability-audit`; commit per task; **no push/merge without Anna's go**; ONE
  final PR when CI is green and Anna approves.
