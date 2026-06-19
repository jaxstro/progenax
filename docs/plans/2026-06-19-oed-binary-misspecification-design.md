# Binary-misspecification robustness OED — design

**Date:** 2026-06-19
**Branch:** `feat/oed-binary-misspecification`
**Status:** design ratified (brainstorming, HITL, plan-mode approved); plan + implementation pending
**Arc:** converts the OED demos from "recovers the obvious" into a referee-proof **bias** result —
does a c-optimal-for-M design computed under the idealized binary-free model survive when the
true cluster has binaries, and what design protects M against that misspecification?

## Motivation

The three shipped OED demos (Stage-1 `r_a`, Stage-2 `M_dyn`, Stage-3 `W₀`) are strong
*engineering* but each headline is either already-known physics (anisotropy→outskirts,
concentration→core) or contingent on an arbitrary RV/PM error split that flips with the budget.
None is a result a referee couldn't dismiss as "Fisher-OED recovers the obvious."

This arc fixes that with a result a referee **cannot** wave away: a **parameter bias**.
Binaries inflate the line-of-sight velocity dispersion (mass/period-dependent; the Moe & Di
Stefano P–q–e engine → a flux-weighted blend kernel `K_orb`). A design optimized for the
dynamical mass M under the binary-free model walks into a biased M̂ — *larger than its own
forecast σ* (false confidence) — and an OED design that accounts for binaries removes the bias.
This monetizes machinery nobody else has wired to OED.

### Honest caveats (load-bearing — do not oversell)

1. **Single-epoch.** Binaries are an unresolvable statistical inflation; we handle them via the
   *second moment* (σ²) + radial leverage, not via per-epoch RV-variability detection
   (multi-epoch is the explicit next arc; the API is built multi-epoch-ready).
2. **Second-moment model.** The f_bin↔σ_cluster degeneracy is broken by **radial leverage**
   (flat binary pedestal vs cluster σ²(R) profile), *not* by the non-Gaussian K_orb tails
   (using the full velocity histogram via `predict_vlos_counts` is a stronger-identification
   extension, deliberately deferred).
3. **Mass-follows-light, RV-only.** No DM halo (tracer≠mass deferred); the RV-only
   mass-anisotropy degeneracy (M↔r_a) is handled by photometric tight priors on (γ, a).
4. **Informax-bound, OUT of v0.1.0.** Ships as scripts + a `60-science-demos` page; **no
   `src/progenax/` change** → released-core coverage/staleness gate untouched.

## Decisions (ratified in brainstorming, one at a time)

| # | Question | Decision |
|---|----------|----------|
| 1 | First slice | **Binary-misspecification robustness** (fuses workstreams 1 + 3a) |
| 2 | Epochs | **Single-epoch** now, **multi-epoch-ready** design-variable API |
| 3 | Headline biased parameter | **M (dynamical mass)** — the canonical binary M/L systematic |
| 4 | Robust mechanism | **Both** — measure-and-marginalize f_bin (fix) + min-max/maximin (comparison) |
| 5 | Survey regime | **RV-only**, σ_bin/σ_cluster large (no arbitrary channel split; dwarf/cluster M/L literature) |
| 6 | Density model | **EFF** (analytic density → ODE-free/OOM-safe; slope γ = bonus concentration-bias param; faithful Moe / young-massive-cluster regime) |
| 7 | Framing | **σ_bin/σ_cluster sweep across system mass** (YMC operating point headline + sweep) |

**Reconciliation with [[oed-sota-capabilities-arc]]:** this arc IS that design's **Stage B**
(robust/min-max) ∪ **Stage D target-2** (f_bin pluggable block), re-scoped through the bias
lens. **Stage A (df_moment C¹ retrofit) is dropped as OBE** (ADR-0017 proved
`df_moment_dispersion`'s W₀ gradient AD-correct). Binaries lead instead of Stage-C
native-Michie (cheaper; bias headline; native-Michie sky projection doesn't exist → deferred).

## Adversarial verification against the live repo (done before ratifying — 3 Explore agents)

- **EFF is ODE-free, exactly like Plummer.** `dispersion.py:38` explicitly groups "the
  *analytic-density* profiles (Plummer, EFF)" against the King/Michie `custom_vjp` ODE solvers;
  `EFFProfile.density` is closed-form `(1+r²/a²)^(−γ/2)` (`eff.py:195`); `project_dispersion`
  does quadrature of `profile.density` on a uniform `r_t` grid (`dispersion.py:318`). → the
  cross-model MC has **no diffrax tape to batch** (sidesteps the Stage-3 28 GB OOM).
- **EFF concentration parameter.** `EFFProfile.gamma` (`eff.py:54`) is a shape/concentration
  knob (γ=3 young clusters, γ=5→Plummer) → the bias vector spans (M, γ, a, r_a) with no King ODE.
- **OM-on-EFF is exact** (`dispersion.py:446`: EFF intrinsically isotropic); the mock sampler is
  `EFFVelocityDF` (Eddington + OM, build-once table, validated Q≈0.50, `eff_df.py:69`).
- **The B12 σ_los kernel exists only in scripts**, not `src/`: `scripts/_demo_binaries.py`
  (`sample_blend_velocities`→`build_korb_kernel`→`predict_vlos_counts`, differentiable in
  (σ, f_b)) + `scripts/demo_binary_dynamical_mass.py` (`build_mock_vlos` with an `f_b` knob).
  Built from `KeplerElements.to_binary_state` (`binaries/kepler.py:179`), `MoeJointOrbit`
  (`imf/binary/moe_di_stefano.py:390`), ZAMS flux weights (`progenax.stellar`).
- **f_bin differentiability:** use `DifferentiableBinaryFraction` (sigmoid,
  `imf/differentiable_binary.py:45`); **avoid** `MassDependentBinaryFraction` (hard step,
  non-differentiable).
- **Absent (must build, scripts-level):** any cross-model evaluation (optimize-under-A,
  score-under-B); the f_bin Fisher block; the maximin criterion. The OED backbone itself
  (`_demo_oed.py`: additive `fisher`, ONE jacrev, ln-θ ADR-0011, c/D/A, multi-start Adam,
  `lax.map` calibration) is reused verbatim.

## Architecture

Scripts-only, mirroring Stage-1/2/3. **No `src/progenax/` change.**

- `scripts/_demo_oed_binary.py` — core: EFF-OM RV-only forward model, the binary σ²-inflation
  term, the f_bin Fisher block, the marginalize + maximin criteria, the σ_bin/σ_cluster sweep,
  and the cross-model calibration MC.
- `scripts/demo_oed_binary.py` — gated CLI (exit 0), figures.
- extend `scripts/_demo_binaries.py` — expose `V_bin = Var(K_orb)` as a reusable
  population scalar (Moe + primary-mass distribution), build-once.
- `tests/` — scripts-level unit tests + AD-vs-FD gate (f_bin block + kernel) + @slow env-gated
  cross-model MC.
- `docs/website/60-science-demos/optimal-design/binary-robustness.md` — MyST page, 0 warnings.

### Forward / Fisher backbone (reused from Stage-1, RV-only)

```text
theta = (M, r_a, gamma, a, f_bin)       # ln-theta metric (ADR-0011)
predict_sigma_obs2(theta, R, G):
    prof = EFFProfile(a=a, gamma=gamma, r_t=R_T)   # native EFF params; no r_h inversion
    sig_cluster = project_dispersion(prof, r_a, R, M, G).sigma_los   # RV channel only
    return sig_cluster**2 + f_bin * V_bin                            # + eps**2 at the gate
```

- **θ uses the EFF *scale radius* `a`, not a derived `r_h`.** `EFFProfile` is natively
  `(a, gamma, r_t)` with no closed-form `r_h(a, γ, r_t)`, so parameterizing by `a` directly
  avoids a spurious unpinned inversion. `a` *is* the concentration scale; the half-mass radius
  is recoverable post-hoc. (Pinned in Phase 0 Task 0.2; the MyST page (T4.2) states this.)
- **Observable model:** σ²_obs(R) = M·h(R; r_a, γ, a) + f_bin·V_bin + ε². `M` scales the
  cluster term (amplitude), (r_a, γ, a) set the radial **shape** `h(R)`, and binaries add a
  ~flat **pedestal** `f_bin·V_bin` (V_bin a population scalar; per-bin V_bin(R) is the
  mass-segregation extension). Radial leverage separates amplitude vs shape vs offset.
- **Priors / degeneracy structure:** M = **target** (no prior); **γ, a photometrically
  pinned** (tight priors — measured from the surface-brightness profile); **r_a, f_bin =
  kinematic nuisances** the radial RV allocation must disentangle. The f_bin column of J
  (∂σ_los/∂ln f_bin = f_bin·V_bin/(2σ_los)) concentrates info in the **outskirts** (low σ_los);
  M info is everywhere → the optimal design uses the **core↔outskirts contrast** to break
  M↔f_bin. (Reported bias vector: M headline, plus r_a, γ for free.)
- ONE `jax.jacrev` of the cluster term → `J = ∂σ/∂lnθ` (reverse-mode by policy); additive
  `F = Σ_b n_eff,b · M_b` with `M_b = 2 J Jᵀ/(σ²+ε²)`; c-optimal on the M index. All lifted
  from `_demo_oed.py`.

### Mechanism 1 — measure-and-marginalize f_bin (the fix)

Extend the Fisher with the ∂σ²/∂f_bin block (the Stage-D pluggable block); the binary-aware
optimal design is c-optimal-for-M over the **marginalized** Fisher (f_bin a free nuisance).
The cross-model MC fit uses the binary-aware model (M̂ unbiased, σ(M) honestly inflated).

### Mechanism 2 — min-max / maximin (the comparison)

Design to minimize the **worst-case** marginalized (F⁻¹)_MM over f_bin ∈ [0, f_max]. Cheap:
the cluster jacrev `J` is computed once; only the σ_los(f_bin) denominator updates on the f_bin
grid (no re-jacrev). Answers "what if you refuse to trust Moe's f_bin?".

### Multi-epoch-ready API

The design vector is per-radial-bin allocation now; structure the additive Fisher over
(bin × channel × **epoch**) so an "epochs per bin" axis (binary *detection* via RV variability,
the ratified Stage-3 cadence design space) slots in later without refactor.

## The discriminating result (pre-registered — honesty bar)

Run `research-workflow:discriminating-experiment-design` FIRST (Phase 0).

- **H1 — Fragility (the bias).** Binary-free-optimal design + binary-free fit on
  binary-contaminated mock → **accept if |bias(M̂)/M| > 2·σ_forecast** at the YMC operating
  point. *If H1 fails even at the YMC point, the item is DESCOPED.*
- **H2 — OED payoff.** Binary-aware-optimal design buys a target σ(M) with fewer stars than the
  binary-free design under the same binary-aware fit → **accept if precision-gain ≥ 1.3×**.
- **H3 — Non-obvious allocation.** Binary-aware allocation is **not** a monotone rescaling of
  the binary-free one (quantified, e.g. per-bin weight rank change / KL).
- **Sweep (reported, not gated).** bias(M̂) and remedy effectiveness vs σ_bin/σ_cluster across
  system mass → the regime of validity.

### Pre-registration — LOCKED 2026-06-19 (Phase 0 Task 0.3)

Locked **before** Phase 1 runs the MC (this commit is the lock). Operating point (pinned in
Task 0.2): EFF-OM YMC γ=2.7, a=1 pc, r_t=18 pc, r_a=3 pc, **M=4×10⁵ M☉**, f_bin=0.5,
ε_RV=1 km/s, RV-only, 12 log-spaced radial bins 0.2a→0.95 r_t, N_total stars (set in Phase 1).
Pinned scales: σ_cluster,central=8.98 km/s, σ_bin=9.73 km/s (σ_bin/σ_cluster=1.08 at center,
rising to ~24 in the outskirts).

**H1 — the bias (six-slot):**

| Slot | Filled in |
|------|-----------|
| **H1** | Fitting the binary-free model to binary-contaminated RV data biases M̂ **high**, by **more than the design's own forecast σ(M)** (false confidence). |
| **H0 (rival)** | The binary-free fit absorbs the flat f_bin·V_bin pedestal into the nuisances (γ, a, r_a) / the radial profile, so M̂ stays unbiased within its forecast σ — i.e. binaries at f_bin=0.5 don't materially bias M at this operating point. |
| **Observable** | fractional bias `bias(M̂)/M = (⟨M̂⟩ − M)/M` from the cross-model MC (generate WITH Moe binaries, fit WITHOUT), vs the binary-free Fisher `σ_forecast(M)/M`. |
| **Signature** | **H1:** `bias(M̂)/M > 0` and `> 2·σ_forecast/M`. **H0:** `abs(bias(M̂)/M) ≤ σ_forecast/M`. |
| **Smallest run** | the pinned operating point; the naive design = c-optimal-for-M radial allocation under the binary-free Fisher; n_draws cross-model MC chosen so 2·SEM of the bias ≪ the bias (start 48–64 draws, the prior-calibration scale; increase only if SEM straddles the threshold). |
| **Decision rule (fixed now)** | **ACCEPT H1** if mean `bias(M̂)/M > 2·σ_forecast/M` AND positive AND the 2·SEM band does not straddle that threshold. **REJECT H1 → H0 → DESCOPE the arc** if `abs(bias(M̂)/M) ≤ σ_forecast/M`. Between → inconclusive: raise n_draws or redesign (do **not** move the threshold). |

**H2 — OED payoff (conditional on H1 accept).** Observable: `N_binary-free / N_binary-aware` at
fixed σ(M) (equivalently the σ(M) ratio at fixed N), both under the binary-aware marginalized
fit. **ACCEPT** if precision-gain **≥ 1.3×**; reject if < 1.3×. Reported regardless.

**H3 — non-obvious allocation (conditional on H1 accept).** Observable: per-bin design-weight
comparison (binary-aware vs binary-free). **ACCEPT** if the binary-aware allocation is **not** a
monotone rescaling of the binary-free one (per-bin weight rank order changes, i.e. it pulls
budget toward the f_bin-constraining radii); else report as a null finding.

**Honesty note:** a REJECT/null on any of H1/H2/H3 is a reportable finding, not a failure
(null-result integrity). H1 reject descopes the arc — that is an acceptable outcome of a
de-risk gate.

## Validation gate (thresholds locked; never weakened — fix the root cause)

1. **AD-vs-FD** on the f_bin Fisher block and the binary σ²-inflation term: `rel < 1e-3`
   (Richardson idiom where a fixed-step FD is truncation-limited), per `gradient-validation`.
2. **σ_bin numerics PIN (Phase 0):** compute V_bin for a Moe massive-primary population and
   confirm σ_bin/σ_cluster is large enough at the YMC point for H1 to bite.
3. **Cross-model @slow calibration MC** (env-gated `PROGENAX_RUN_OED_*`, OUT of CI): build-once
   EFF Eddington sampler + K_orb pool; per draw via `jax.lax.map` (sequential) → EFF particles +
   injected Moe binaries → σ_los → bin → subsample → +ε → ln-θ GN MAP fit → collect M̂; assert
   realized bias/variance match the forecast (binary-aware fit unbiased; binary-free fit biased
   per H1).
4. `numerical-method-validation` on the K_orb kernel + quadrature; `verification-gate` close-out.

## Performance / OOM plan (`enforce-jax-performance` — non-negotiable)

EFF analytic density → no ODE. Build-once K_orb pool + EFF sampler table (per truth), reused per
draw; per-bin blocks cached (design-independent); MC via `jax.lax.map` (not vmap-over-fit-tape);
f_bin grid re-uses cached blocks; jit hot paths (`static_argnames`); persistent XLA cache.
**Estimate cost + smoke-test peak RSS before any long run**; never run heavy jobs concurrently.
Every subagent spec writing a hot loop MUST require jit + build-once + a reported wall-clock;
review verifies it.

## Provenance (no fabrication)

Verify the Moe & Di Stefano (2017) P–q–e constants used by `_demo_binaries.py`/`MoeJointOrbit`
against the held PDF / per-paper note (`no-assumptions-verify-against-pdfs`,
`paper-grounding-workflow`); ZAMS = Tout+1996 (already internalized in `progenax.stellar`);
Kepler `to_binary_state` reused as-is.

### Phase 0 Task 0.1 verdict (2026-06-19): PASS

`scripts/_demo_binaries.py` hardcodes **no** Moe constants — it rides `MoeJointOrbit.default()`
(packaged) + `Maschberger(α=2.3, 0.08–100 M☉)` + Tout+1996 `zams_luminosity` + Kepler
`period_to_semimajor_axis`/`to_binary_state`. The Moe Table-13 grids in
`src/progenax/imf/binary/moe_di_stefano.py` were **re-verified cell-by-cell against the actual
PDF** (`docs/core-papers/Moe_2017_ApJS_230_15.pdf`, p.52, Table 13) this session — every value of
`_GAMMA_LARGEQ`, `_GAMMA_SMALLQ`, `_F_TWIN`, `_COMPANION_FREQ` matches exactly (the `<0.03` twin
cells correctly → 0; representative mass nodes 1.0/3.2/6.7/12/20 M☉ for the five Table-13 bins).
The kernel uses the **faithful** `MoeJointOrbit`, so the documented R3 twin-over-weight of the
period-averaged `MoeDiStefano2017` reduction does **not** apply. Eccentricity η(P,M₁) (Eqs. 17–18)
+ Roche ceiling (Eq. 3) live in `MoeEccentricity` per the PDF-verified per-paper note (§9.2). No
unsourced/fabricated constant on the binary grad path.

## Science deliverable

The binary-aware vs binary-free optimal allocation maps (radius), the M-bias bar (naive vs
robust, MC-verified), the precision-gain factor, the σ_bin/σ_cluster sweep, and the marginalize
vs maximin comparison. Publication-quality figures. A wrong sub-prediction is a finding, not a
failure (null-result integrity).

## ADR

Record (ADR-0019+): the binary σ²-inflation forward-model + f_bin pluggable Fisher block; the
EFF-OM RV-only modeling choice; the marginalize-vs-maximin robustness decision.

## Phases (HITL checkpoint between each — Phase 1 is the minimal-falsifiable slice)

0. Gate & provenance (discriminating-experiment-design; PIN σ_bin numerics; verify Moe constants).
1. Bias demonstration (H1) — EFF-OM RV-only backbone + binary σ²-term + cross-model MC. *Stop if H1 fails.*
2. Marginalize fix (H2/H3) — f_bin block + binary-aware design + calibration.
3. Min-max comparison — maximin over the f_bin grid.
4. Sweep + docs — σ_bin/σ_cluster sweep + MyST page + ADRs + completion doc.

## Out of scope (YAGNI)

Multi-epoch binary *detection* (epochs-vs-stars); the full-histogram (K_orb-tail) identification;
PM channel / channel allocation; tracer≠mass / DM-halo dwarfs; mass-segregated per-bin V_bin(R);
the joint/Pareto criterion (Item 2); native-Michie T-optimality (Item 1, anisotropy). No
`src/progenax/` API; no released-core promotion (informax-bound).
