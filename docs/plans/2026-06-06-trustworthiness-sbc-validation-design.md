# gravoturb_fdf trustworthiness hardening — SBC + HMC diagnostics + figures (DESIGN / HANDOFF)

> **Status: DESIGN, brainstormed with Anna 2026-06-06.** This is the handoff for a *new session*
> to start the trustworthiness-hardening arc. Next session: read this, run
> `superpowers:writing-plans` to turn it into a TDD task plan, then `test-driven-development`.

## Context

The gravoturb_fdf differentiable-inference arc (Phases 1–7) is complete and on PR #5
(`gravoturb-fdf-clean-room`): the BM19 tail slope α is recoverable via the peaks-over-threshold
(POT) block (AC16), with the σ(α)-vs-N_tail forecast (AC17). Anna's directive (2026-06-06):
**harden trustworthiness before claiming anything in a paper.**

**Honest assessment of where we stand:**
- **Solid (high trust):** the *machinery*. 245 experimental + 815 released-core tests, strict TDD,
  gradient validation, analytic predictions validated against the realization simulator as an
  independent oracle (AC11–AC13), POT posterior width matched the Fisher to 1%, docs build clean.
  Every formula is PDF-grounded.
- **Soft (caveated, not externally validated):** AC16 is **injection–recovery** (mock drawn from the
  *same* BM19 model it fits) at a *single* injected θ; the correlation penalty (`N_eff ≈ N_tail/6`)
  makes the per-cluster constraint ~2.5× weaker than the idealized Fisher.
- **Untested / missing:** no calibration across the prior; no HMC convergence diagnostics
  (R̂/ESS/divergences); the 3-pt null test (the core phase-randomness assumption) is deferred; no
  external/real-sim validation; no publication figures.

**Decision (Anna):** trustworthiness-first, sequenced as **① SBC + diagnostics + figures (this
doc) → ② 3-pt null test → ③ external / misspecification validation.**

## What SBC proves — and what it does NOT (the honest scope)

Simulation-Based Calibration upgrades AC16 from *"the posterior covers one injected θ"* to *"the
posterior is calibrated across the whole prior."* That is a major trust win for the **inference
engine**. It does **not** test model misspecification — the data are still drawn from BM19 — so the
"injection-recovery, not real data" caveat **remains**; that is workstream ③. State this boundary in
the paper: a green SBC means *self-consistent and calibrated under the assumed model*, not *the model
matches real clouds*.

## Design — workstream ① components (TDD, experimental-only; released-core 815 stays green)

**1. Explicit proper priors** — `inference/priors.py`. SBC requires drawing θ from a *proper* prior;
the HMC today has only the bounded reparametrization + Jacobian (implicitly improper). Define
explicit priors on `(ℳ, α, β)` over physical ranges (e.g. log-uniform on ℳ∈[2,20], β∈[2,11/3];
α∈(1, α_max(ℳ)] bounded so `s_t(θ) ≤ s_thr` stays in the POT-valid region — reuse the
`pot_validity_barrier` logic). Wire the prior log-density into the HMC logdensity (replacing the
bare Jacobian). This is itself a hardening step.

**2. SBC driver** — `inference/sbc.py`. For K trials: draw θ*~prior → simulate the mock (POT
exceedances via `rank_copula_field` + `measure_exceedances`, CIC counts via `sample_cic_counts`) →
run NUTS → compute the **rank** of θ* among the L posterior draws, per parameter. Calibrated ⇔ ranks
~ Uniform(0, L). Use a reduced-grid SBC config (calibration, not absolute precision) to keep K NUTS
runs tractable; parallelize across trials. Follow Talts et al. (2018): thin posterior draws to
~independent, K≈100–200.

**3. HMC diagnostics** — `inference/diagnostics.py` (or wrap `arviz`). ≥4 dispersed chains;
report **R̂, bulk/tail-ESS, divergence count, BFMI, max-tree-depth saturation**. blackjax exposes
the per-step info; arviz computes R̂/ESS/rank plots from stacked chains.

**4. Figures via jaxstroviz (DRY — Anna's directive).** jaxstroviz
(`~/projects/jaxstro-dev/jaxstroviz`) is the shared ecosystem viz package with a strict
**compute/plot separation** (`analysis/` computes, `plots/` visualizes; `set_paper`/`newfig`
helpers). It **already has** much of what we need — **reuse, don't reinvent:**
- `analysis/inference.py`: `compute_parameter_recovery`, `compute_coverage_probability`,
  `compute_bias_significance`.
- `plots/inference.py`: `plot_parameter_recovery` (recovery scatter), `plot_recovery_correlation`,
  `plot_posterior_1d` / `plot_posterior_2d` (corner), `plot_inference_summary`.
- `plots/substructure.py`: `plot_two_point_correlation`, etc. for FDF-specific figures.

**Add to jaxstroviz only the genuinely new pieces** (and unit-test them in jaxstroviz):
- `analysis/inference.py`: `compute_sbc_ranks(thetas_true, posterior_samples)` and a small
  `sbc_uniformity_test` (χ²/KS p-value); `compute_hmc_diagnostics(chains)` (or an arviz wrapper).
- `plots/inference.py`: `plot_sbc_rank_histogram(ax, ranks, n_bins, ...)` (the rank histogram with
  the uniform band — the one genuinely new figure type); add `plot_calibration_curve` only if the
  existing coverage util doesn't already cover nominal-vs-actual.

gravoturb_fdf's `validation/` computes the arrays (ranks, recovery, σ(α)) and **calls these
jaxstroviz functions** to render; jaxstroviz is an optional `[viz]` dep on the validation side only
(progenax core must NOT import jaxstroviz). Save PNGs to `validation/plots/` per the "Definition of
Complete."

**Figure set:** SBC rank histograms (per param) · empirical coverage curve · recovery scatter
(θ̂±σ vs θ_true across the prior = the parameter-space recovery map) · a representative (ℳ,α,β)
corner showing the broken degeneracy · σ(α)-vs-N_tail (AC17 as a figure).

**5. New acceptance criteria:**
- **AC18 — SBC uniformity:** per-parameter rank histograms pass a uniformity test (χ²/KS, p>0.05)
  over K≈100–200 trials. (Slow-marked; reduced config in the test wrapper, full K in `main()`.)
- **AC19 — HMC convergence:** ≥4 chains at the production AC16 config; R̂<1.01, bulk/tail-ESS above
  threshold (e.g. >400), divergences <1%.

**New optional deps:** `arviz` (R̂/ESS/SBC), `matplotlib` (jaxstroviz already uses it). Add to the
`[experimental]` / `[viz]` extras; declare jaxstroviz as the figure dep on the validation side.

## After ① — the rest of the trustworthiness arc

- **② 3-point null test (Phase 8).** Predict the marginal-induced 3-pt analytically; measure it on
  the simulator's mocks; the null test must PASS on phase-random mocks (self-consistency) and becomes
  a filament-detector / discovery channel on real data. Tests the assumption that licenses 1pt+2pt.
- **③ External / misspecification validation.** Feed the inference data NOT drawn from BM19 (a public
  gravo-turbulent sim snapshot, or a deliberately misspecified generative model) and measure how
  recovery degrades. The biggest "is the model right for reality" test; needs external data.

## Rules in force (non-negotiable)

HITL — Anna approves at every design fork; strict TDD (RED→GREEN, watch it fail); verify formulae
against held PDFs (never assert from memory); JAX-native core (numpy/scipy/matplotlib only on the
validation/figure side); evidence-before-done (paste fresh command output for any pass/works claim);
commit each task; **NO git push / NO PR #5 merge without Anna's explicit go.** Figures are reusable
jaxstroviz functions (DRY), not one-off scripts.

## READ FIRST (next session)

1. This doc + memory `gravoturb-fdf-differentiable-inference.md` (Phase history) +
   `gravoturb-fdf-clean-room.md`.
2. `src/experimental/gravoturb_fdf/inference/{hmc,likelihood,fisher}.py` and
   `validation/acceptance.py` (`ac16_hmc_recovery`, `ac17_alpha_forecast`).
3. jaxstroviz `analysis/inference.py` + `plots/inference.py` (the reusable functions above).
4. The website chapter `docs/website/10-theory/gravoturbulence/differentiable-inference.md`.

## Verification (uv)

```bash
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q   # 815 invariant
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -m gravoturb_fdf.validation.acceptance   # AC1–AC19
# jaxstroviz figure functions: unit-test in jaxstroviz; render gravoturb_fdf figures to validation/plots/
```
