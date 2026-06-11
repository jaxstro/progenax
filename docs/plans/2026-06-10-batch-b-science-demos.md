# Batch B: Science Demos Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> subagent-driven-development) to implement this plan task-by-task.

**Goal:** Build the methods-paper demo suite: the cross-engine agreement
figure, the self-consistent IMF(α)+equipartition(δ) joint-recovery demo
(MLE + NUTS + wrong-IMF bias curve + robustness grid), and the halo+core
(t, r_a, r_h) recovery demo, with a new `60-science-demos` website section.

**Architecture:** One scripts-local physics-direct likelihood layer
(`scripts/_demo_inference.py`): mock stars sampled at truth → binned
kinematic summaries with finite-N errors → Gaussian likelihood against the
model's ANALYTIC dispersion predictions (gradients flow through the analytic
side only; no resampling in the loss). Demos are gated CLIs in the
`validate_*` house style. Design doc (Anna-approved):
`docs/plans/2026-06-10-batch-b-science-demos-design.md`.

> **Plan amendments — 2026-06-10 (Phase-1 verification, Anna-approved):**
> 1. **Mass-likelihood form pinned (Task 3 STOP resolved).** `sample_cluster`
>    assigns each star its component's *representative* mass label
>    (`sampling.py:39` `m_i = model.m_j[c]`) — `ic.masses` holds only J=4
>    discrete values, so the design's `sum(logpdf(m_obs))` cannot run on it.
>    **Anna's decision: Option A (global IMF sample)** — draw N observed masses
>    globally from `Maschberger(α_true)` over the full `M_RANGE`, independent of
>    kinematic group; mass term = `jnp.sum(Maschberger(α).logpdf(m_obs))`, NO
>    truncation correction. α stays self-consistent at the population level (one
>    α drives both the mass histogram and the equipartition groups). Pages must
>    state: observed masses are a global IMF sample; per-star mass↔group
>    correlation is not modeled (clean-mock choice).
> 2. **Maschberger-Jacobian hedge RETIRED.** Maschberger is already fully
>    differentiable in α: analytic `ppf` (`smooth.py:120-163`) and normalized
>    `logpdf = _logpdf_unnorm(α) − _log_norm(α)` via the analytic `_primitive`
>    at the bounds (`base.py:124-126`). No package TDD addition; no `argmax`/
>    `argsort` in the α path; the loss never resamples, so even `ppf`
>    differentiability is off the critical path.
> 3. **Engine B β(r) is analytic (Task 7).** The Engine B moment recipe returns
>    a combined ⟨v²⟩, so σ_1d = √(⟨v²⟩/3) from the recipe and
>    **β(r) = r²/(r²+r_a²)** closed-form in r_a (OM definition), not from the
>    f-moments. Headline `r_a=3.0` (`test_engine_b_physics.py:375`), m_j=[0.5,1.0].

**Tech stack:** JAX, equinox, optax (Adam MLE; optimistix NOT available),
blackjax NUTS (pattern: `src/experimental/gravoturb_fdf/inference/hmc.py:38-55`),
`scripts/_plotstyle.py` figures.

**Verified call sites (2026-06-10 exploration; re-verify line numbers):**
- `Maschberger` — `src/progenax/imf/smooth.py:50-174`; fields incl.
  `alpha: 2.3` (high-mass slope), all differentiable; closed-form `ppf`;
  normalized `logpdf(m)` inherited (base.py:124-126).
- `MultiComponentCluster.from_imf(imf, n_comp, W0, g, delta, m_range, ...)`
  — `src/progenax/cluster/multicomponent.py:362-393`; `_bin_imf` static
  log-edges (numpy.geomspace — fine: edges fixed, IMF-param gradients flow
  through the trapezoid integrals); runs `find_alpha_for_masses` (n_iter=30
  fixed-scan eigenvalue solve — the expensive part, see Task 3 budget gate).
- Engine A per-group σ oracle (VERBATIM recipe):
  `tests/validation/test_multimass_equilibrium_physics.py:70-86` —
  σ_j(r) = s_j √[∫u⁴E du / ∫u²E du / 3], E = lowered_exponential(g, W_j−u²/2),
  W_j(r) = rescale_j · ψ(r), s_j = s·w_j, s² = G M /(9 r_c μ_tot).
- Engine B per-component moment oracle:
  `tests/validation/test_engine_b_physics.py:146-162` — speed moments of the
  f_j(E) rows + OM (1/3 + (2/3)/(1+(r/r_a_j)²)) weighting.
- blackjax + optax: `experimental` extras (pyproject.toml:29-30) — demo pages
  must state `pip install progenax[experimental]`.

**Gates (run from repo root):** FAST gate as in CLAUDE.md; demo scripts exit
nonzero on recovery-gate failure. **Git:** branch `feat/batch-b-demos` off
main; commit per task; NO push/merge without Anna's go.

**Hard rules:** JAX-native; TDD for the helper layer; demo gates are real
(3σ recovery, occupancy floors) — never weaken to pass; every figure via
`_plotstyle`; runtime-budget checkpoints are STOP points, not suggestions.

---

## Task 1: `scripts/_demo_inference.py` — shared likelihood layer (TDD)

**Files:** Create `scripts/_demo_inference.py`;
Test `tests/unit/test_demo_inference.py` (import via
`sys.path.insert(0, "<repo>/scripts")` in the test module — precedent: none,
this is the first tested scripts-helper; keep the path hack in the test file).

**Step 1 — failing tests** (write all, watch fail with ImportError):

```python
class TestBinnedSigma1d:
    def test_isotropic_gaussian_recovers_sigma(self):
        # 3 groups x 50k stars, v ~ N(0, sigma_j^2 I3): binned sigma_1d
        # must recover sigma_j within 3 SE in every populated bin.
    def test_se_scaling(self):
        # SE == sigma_hat / sqrt(2 n_bin) (the Gaussian dispersion SE).
    def test_empty_bins_masked(self):
        # bins with < n_min members return weight 0 (not NaN) in the chi2.

class TestChi2Loglike:
    def test_perfect_model_gives_zero_chi2(self): ...
    def test_gradient_flows_to_predict_params(self):
        # jax.grad of the loglike w.r.t. a predict_fn parameter is finite/nonzero.

class TestAdamMLE:
    def test_recovers_quadratic_minimum(self):
        # loss (x-3)^2: mle_adam returns ~3.0; fisher_cov returns ~[[0.5]]
        # for loglike -(x-3)^2 (hessian 2 -> cov 0.5).
```

**Step 3 — implementation** (signatures; bodies are straightforward JAX):

```python
def binned_sigma1d(pos, vel, group_ids, n_groups, r_edges, n_min=30):
    """Per-group binned sigma_1d(r) from |v|^2/3 (isotropic estimator).
    Returns (sig_hat[J,K], se[J,K], weight[J,K] in {0,1}, n[J,K])."""

def binned_sigma_beta(pos, vel, r_edges, component_id=None, n_min=50):
    """sigma_1d(r) + beta(r) (radial/tangential split) per component
    (component_id=None -> single). For the Engine B demo."""

def gaussian_loglike(data, predict_fn):
    """data = (sig_hat, se, weight); returns loglike(theta) =
    -0.5 * sum(weight * ((sig_hat - predict_fn(theta)) / se)**2)."""

def mle_adam(negloglike, z0, n_steps=400, lr=3e-2):
    """jit(value_and_grad) + optax.adam over a lax.scan of n_steps.
    Returns (z_hat, loss_trace). Fixed steps (differentiable-friendly,
    deterministic); caller checks the trace plateaued."""

def fisher_cov(negloglike, z_hat):
    """inv(jax.hessian(negloglike)(z_hat)) with a symmetric solve; raises
    (concrete) if not positive definite -- report, don't mask."""

# Reparametrizations (mirror gravoturb hmc.py:19-35):
def logit(x, lo, hi): ...
def expit(z, lo, hi): ...
```

**Steps 2/4/5:** run tests (fail → pass), then
`git commit -m "feat(demos): shared physics-direct likelihood layer (TDD)"`.

---

## Task 2: B1 — cross-engine agreement figure

**Files:** Create `scripts/demo_cross_engine.py`;
page comes in Task 8.

Build ONE King model twice:
```python
mA = MultiComponentCluster.from_components(
    alpha_j=jnp.array([1.0]), w_j=jnp.array([1.0]), m_j=jnp.array([1.0]),
    W0=5.0, g=1.0, r_c=1.0)
mB = MultiComponentCluster.from_density_profiles(
    [KingProfile.from_W0_rc(W0=5.0, r_c=1.0)], jnp.array([1.0]),
    m_j=jnp.array([1.0]))
```
Figure (3 panels + residual strips): ρ(r) overlay (A: `total_density`;
B: prescribed density), σ_1d(r) overlay (both oracles above), f(E) overlay
(A: analytic lowered_exponential DF shape; B: the `engine_b` f-row — match
normalizations by peak or integral, state which). Sample BOTH at N=2×10⁴
(same key) → radial KS + max σ-dev measured fresh and printed as gates:
KS < 0.02, σ-dev < 0.02 (the ledger anchors measured 2e-4/3e-4 — print
measured values; gate at the validated thresholds).
Output: `validation/plots/demo_cross_engine.{png,pdf}`; ALL PASS table;
commit `feat(demos): cross-engine King agreement figure (B1)`.

---

## Task 3: B2a — truth dataset + jitted joint loss + RUNTIME BUDGET GATE

**Files:** Create `scripts/demo_delta_recovery.py` (data + loss only this task).

**Truth config (module constants):** `ALPHA_TRUE=2.3, DELTA_TRUE=0.4,
W0_TRUE=5.0, G_MODEL=1.0 (model units), N_COMP=4, M_RANGE=(0.1, 20.0),
N_STARS=100_000, R_EDGES = quantile-based 8 bins` (fixed after one draw,
then frozen constants — document). IMF truth:
`Maschberger(alpha=ALPHA_TRUE, m_min=0.1, m_max=20.0)` (class bounds = draw
bounds so `logpdf` normalization matches the sample).

**Mock data:** `model = MultiComponentCluster.from_imf(imf, N_COMP, W0_TRUE,
g=1.0, delta=DELTA_TRUE, m_range=M_RANGE)`;
`ic = model.sample_cluster(key, N_STARS, G=G_MODEL)` (carries `component_id`,
used to BIN the kinematic summaries σ̂_j(r)). **Observed masses (Option A,
amendment 1):** `ic.masses` holds only the J discrete `m_j` labels, so draw a
SEPARATE global mass sample `m_obs = imf.ppf(jax.random.uniform(key2, (N_STARS,)))`
from the SAME truth `Maschberger(α_true)` over the SAME `M_RANGE` — this is the
observed-mass dataset for the mass-likelihood channel (it is NOT the kinematic
group label; document on the page that per-star mass↔group correlation is not
modeled).

**Joint negloglike(z)** with z unconstrained for
θ = (α ∈ [1.5, 3.2], δ ∈ [0, 1], W₀ ∈ [3, 8]):
- kinematics: `gaussian_loglike(binned data, predict_fn)` where predict_fn
  rebuilds `from_imf(Maschberger(alpha), N_COMP, W0, g=1, delta, M_RANGE)`
  inside the traced function and evaluates the Engine A σ_j oracle at the
  bin centers (verbatim recipe, vectorized over groups × bins);
- mass term (Option A, amendment 1): `jnp.sum(Maschberger(alpha, m_min,
  m_max).logpdf(m_obs))` over the global mass sample — NO truncation
  correction (full-range IMF-slope likelihood; analytic/differentiable in α).

**RUNTIME BUDGET GATE (STOP point):** measure ONE warm
`jit(value_and_grad(negloglike))` call. If > 5 s, STOP and report to Anna
with options (reduce n_ode_points to 1500, n_iter to 15 with re-verified
solve residual < 2e-3, or reduced ensemble scope) — do NOT silently degrade
accuracy. Record the measured cost in the script docstring.
Commit: `feat(demos): B2 truth data + jitted joint (alpha,delta,W0) likelihood`.

---

## Task 4: B2b — joint MLE headline + gates + data-vs-fit figure

`mle_adam` from 4 dispersed inits (loss-trace plateau check); pick best.
`fisher_cov` errors in constrained space via the Jacobian of expit.
Gates: |θ̂−θ_true| < 3σ̂ componentwise; top-group occupancy ≥ 300 (else the
config is wrong — STOP); loss trace plateaued (last-10% improvement < 1%).
Figure `demo_delta_recovery_fit.png`: (a) per-group σ̂(r) data + best-fit
curves; (b) observed mass histogram + fitted Maschberger pdf.
Commit: `feat(demos): B2 joint MLE recovery (alpha,delta,W0) with Fisher errors`.

---

## Task 5: B2c — Fisher degeneracy panel + NUTS corner

- Fisher panel `demo_delta_recovery_fisher.png`: 2σ ellipses in (α, δ) from
  (i) kinematics-only Fisher (mass term dropped) — the degeneracy — and
  (ii) the joint Fisher. Quote the kinematics-only correlation coefficient.
- NUTS (blackjax; copy the run_nuts pattern from
  `src/experimental/gravoturb_fdf/inference/hmc.py:38-55` — do NOT import
  experimental code into a released-core script; vendor the ~20-line wrapper
  into `_demo_inference.py` with attribution comment): 3-param corner
  `demo_delta_recovery_corner.png`, n_warmup=300, n_samples=600 budgeted
  against the measured eval cost (BUDGET GATE: projected wall-time
  > 45 min → STOP and report options). Overlay MLE + truth; report
  divergence count (must be 0) and that posterior mean ≈ MLE within 1σ.
Commit: `feat(demos): B2 Fisher degeneracy panel + NUTS corner`.

---

## Task 6: B2d — wrong-IMF bias curve + robustness grid (ensembles)

- Bias curve: α_assumed ∈ {1.9, 2.1, 2.3, 2.5, 2.7}; kinematics-only refit
  of (δ, W₀) per α_assumed × N_SEEDS seeds (N_SEEDS scaled to the measured
  eval cost; target whole-panel < 30 min; report the chosen N_SEEDS).
  Figure: δ̂(α_assumed) ± seed scatter, truth lines, quoted slope dδ̂/dα ±
  uncertainty from the seed ensemble (REPORTED, not gated).
- Robustness grid: α_true ∈ {1.9, 2.3, 2.7} (fresh truth datasets), full
  joint refit; gate: each within 3σ.
Figure `demo_delta_recovery_bias.png` (2 panels).
Commit: `feat(demos): B2 wrong-IMF bias curve + robustness grid`.

---

## Task 7: B3 — halo+core (t, r_a, r_h) recovery

**Files:** Create `scripts/demo_halo_core.py`.

Truth: `from_density_profiles([PlummerProfile(r_h=2.0),
EFFProfile(a=0.8, gamma=5.0, r_t=9.0)], jnp.array([0.6, 0.4]),
m_j=jnp.array([0.5, 1.0]), r_a_j=jnp.array([3.0, jnp.inf]))` — **OM headline
r_a=3.0 confirmed** (`test_engine_b_physics.py:375`; realizability f_min_j ≈
[+0.085, +1.2e-4] there, so r_a=3.0 is used as-is). N=3×10⁴ mock stars.
Observables: per-component σ̂(r) (`binned_sigma_beta` with component_id) +
halo β̂(r). predict_fn: rebuild Engine B inside the traced loss from
θ = (t ∈ logit[0.3,0.9], log r_a, log r_h); σ_1d(r) = √(⟨v²⟩/3) from the f_j-row
moment recipe (test_engine_b_physics.py:146-162), and **β(r) = r²/(r²+r_a²)
closed-form in r_a** (amendment 3 — the OM definition, not from the f-moments). Realizability: report the traced-build
`engine_b` f_min diagnostic at θ̂ (must be ≥ −1e-3). MLE + Fisher only.
Gates: 3σ recovery on all three; f_min check. Figure
`demo_halo_core.png`: data-vs-fit σ/β panels + Fisher ellipses.
Commit: `feat(demos): B3 halo+core (t, r_a, r_h) recovery`.

---

## Task 8: Website section `60-science-demos/`

**Files:** Create `docs/website/60-science-demos/{index.md, cross-engine.md,
imf-equipartition.md, halo-core.md}`; `figures/` copies of the 6 demo PNGs;
Modify `docs/website/myst.yml` (new TOC group after Validation);
Modify `docs/website/50-validation/index.md` (one cross-link line).

Pages in paper-section prose; every number from the scripts' printed tables;
honest-limitations admonitions (NUTS not SBC-calibrated — future work;
mass-term construction choice documented; `experimental` extras requirement).
Build gate: `myst build` exit 0, 0 warnings, **148 pages** (144 + 4).
Commit: `docs(website): science-demos section (B1-B3) wired into TOC`.

---

## Task 9: Close-out

FULL gate (slow incl.) — record `N passed` (expect 1163 + Task-1 helper
tests). Memory gates ALL PASS. All three demo scripts end-to-end ALL PASS
(fresh, captured). myst 148 pages / 0 warnings. Completion doc
`.claude-work/BATCH_B_SCIENCE_DEMOS_COMPLETE.md` (recovery tables, runtime
budgets vs measured, the dδ̂/dα number, lessons). STATUS.md (`next:` =
gravax forward chain OR paper assembly — Anna's call). STOP: merge to local
main + any push ONLY on Anna's explicit go.

---

## Out of scope (do not drift)

gravax forward chain; promoting `_demo_inference.py` to package API; SBC of
the NUTS posterior; observational realism (LOS projection, errors,
incompleteness) — the demos are clean-mock methods showcases, say so on the
pages.
