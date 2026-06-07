# Handoff: gravoturb_fdf count-model tail-robustness (the M-channel bias)

> **For the next Claude session.** Anna wants a **pedagogical, one-thing-at-a-time** walkthrough
> BEFORE any design or code: explain what we built and why, what the current bug is, what we
> tried and ruled out — checking her understanding at each step — and *then* brainstorm the
> **state-of-the-art, scientifically-correct** fix together. Do **not** jump to a solution.
> Use the `superpowers:brainstorming` skill for the design phase. **No hacks** (Anna's words);
> fix the physics. This is real astrophysics research, not a tech demo.

## 0. How to run the session (Anna's explicit request)

1. Read this whole doc + the references in §8, and skim the code in §7.
2. Then **explain to Anna, one piece at a time** (Socratic, multiple-choice where natural,
   one question/topic per message), in roughly this order — confirming she follows before moving on:
   (a) what the gravoturb_fdf inference layer *is* and the physics (§2);
   (b) the three inference channels α/β/M and which are healthy (§3);
   (c) the counts-in-cells (CIC) forward model in detail — Route A vs Route B (§4);
   (d) the bug: the M-channel bias, with the evidence (§5);
   (e) what we tried and **ruled out** (§6);
   (f) the root cause — the fat-tail tail-sensitivity (§5.4).
3. **Only then** brainstorm the SoTA fix (§9): explore both repair philosophies + any hybrid,
   research the cosmology counts-in-cells literature, settle on a design, write it to
   `docs/plans/YYYY-MM-DD-...-design.md`, get Anna's approval, then implement TDD with oracle
   validation across the M prior.

## 1. One-paragraph summary

The gravoturb_fdf differentiable-inference engine infers turbulent-cloud parameters
θ=(ℳ, b, α, β) from a star field. Three channels constrain them: **α** (PDF tail slope) via a
peaks-over-threshold tail block, **β** (density power-spectrum slope) via the log-density 2-point,
and **ℳ** (sonic Mach) via the counts-in-cells (CIC) 1-point distribution P(N) through
σ_s²=ln(1+(bℳ)²). We added Simulation-Based Calibration (SBC, AC18). SBC **passes for α and β
but rejects ℳ** (rank-uniformity p≈0.005): the CIC forward model **over-predicts the count
over-dispersion, growing with Mach (+9% at ℳ=3 → +36% at ℳ=16)**, because it builds P(N) from
**analytic, infinite-tail linear-density moments** (⟨e^s⟩, ⟨e^{2s}⟩) that a *finite* star field
can never realize. This is the known **fat-tail tail-sensitivity** of linear-density statistics
(⟨ρ²⟩ is tail-dominated; diverges for α≤2). It is a genuine forward-model limitation, not a
sampler/prior/numerics bug. **The task: redesign the ℳ/CIC channel to be tail-robust** (no hacks),
validated against an oracle across the whole ℳ prior, with AC18-ℳ then uniform.

## 2. What gravoturb_fdf is (the physics) — for step 2(a)

- **Experimental, repo-only** subsystem at `progenax/src/experimental/gravoturb_fdf/` (NOT in the
  released wheel). ~3000 LOC. JAX-native, float64, differentiable.
- **BM19 model** (Burkhart & Mocz 2019; Burkhart 2018): a supersonically-turbulent cloud's
  volume density PDF is **lognormal body + power-law high-density tail**. In log-density
  s = ln(ρ/ρ̄): a Gaussian of variance σ_s²=ln(1+(bℳ)²) for s<s_t, then p(s)∝e^{-αs} for s>s_t,
  with transition s_t=(α−½)σ_s². See per-paper notes `docs/website/99-bibliography/per-paper/`
  (burkhart-2018, burkhart-mocz-2019) and `docs/core-papers/Burkhart_2018_ApJ_863_118.pdf`.
- **Differentiable physics-direct inference** (the design Anna chose — avoid SBI): don't
  differentiate the stochastic simulator; **predict the summary statistic analytically as a
  smooth function of θ** and differentiate THAT (cosmology playbook: differentiable theory +
  likelihood + HMC). Design docs: `docs/plans/2026-06-05-gravoturb-fdf-differentiable-inference-design.md`
  and `...-gaussianization-predicted-statistics-tdd-plan.md`. Memory: `gravoturb-fdf-differentiable-inference`.
- **A GRF is fully specified by 1pt+2pt**, so restricting inference to the 1-point PDF + 2-point
  makes the phase-randomness assumption honest (3-pt is a held-out null test, never in the fit).

## 3. The three inference channels — for step 2(b)

| Param | What it is | Constrained by | Status |
|-------|------------|----------------|--------|
| **α** | PDF power-law tail slope | **POT tail block** `tail_exceedance_loglike` (truncated-exponential fit of log-density exceedances above s_thr). **Shift-immune** (the lognormal norm cancels → α-only, geometry-free). | ✅ SBC PASS (p=0.18). Task 6 made it valid by dropping a truth-keyed barrier (see §6). |
| **β** | GRF density power-spectrum slope P(k)∝k^−β | **log-density 2-point** ξ_s(r) via Coles&Jones/Szapudi–Pan **Gaussianization** (Hermite series; tail-robust because it works in **log** space). | ✅ SBC PASS (p=0.12). |
| **ℳ** | sonic Mach number | **CIC 1-point** P(N) via σ_s²=ln(1+(bℳ)²) → the count over-dispersion. **b is FIXED** (the data constrains (ℳ,b) only through σ_s², so b is not separately identifiable — AC15 ℳ–b degeneracy). | ❌ **SBC REJECTS** (p=0.005). The subject of this handoff. |

The pattern is already telling: the two channels that work live in **log space** (α via
shift-immune exceedances; β via log-density Gaussianization). The broken channel (ℳ) rides on
**linear-density** moments. That is the clue.

## 4. The counts-in-cells (CIC) forward model — for step 2(c)

Stars are placed ∝ the **linear** density ρ (a Cox/doubly-stochastic Poisson process,
intensity λ(x)=n̄·ρ̃(x), ρ̃=ρ/⟨ρ⟩). Counts-in-cells variance:
**σ²_N(R) = N̄ + N̄²·ξ̄(R)**, ξ̄(R)=Var(ρ̃ smoothed at cell scale R). Two routes
(`src/experimental/gravoturb_fdf/theory/cic.py`):

- **Route A** — `cell_averaged_xi_rho` → `cic_variance`. ξ̄(R) from the exact marginal-induced
  **linear**-density Gaussianization series (Hermite of exp(s_of_g)). Used in `data_vector`
  (the Gaussian-likelihood / Fisher path, AC15).
- **Route B** — `count_distribution` (used by **`count_loglike`**, the AC16/AC18 path!). A
  compound-Poisson P(N): σ_s²(R)=`smoothed_log_variance` (tail-robust, log-space) → effective
  Mach ℳ_eff=`effective_mach` → smoothed BM19 PDF p_R(s)=`smoothed_pdf` → then
  P(N)=∫ Poisson(N | λ=n̄·e^s/μ) p_R(s) ds, μ=⟨e^s⟩ on the s-grid.

**The subtlety that bites:** Route B sets the *width* via the tail-robust σ_s²(R), but it then
forms the count over-dispersion from **μ=⟨e^s⟩ and the e^s Poisson mixture** — i.e.
Var(N)=N̄+N̄²(⟨e^{2s}⟩/μ²−1), and ⟨e^s⟩, ⟨e^{2s}⟩ are **tail-dominated linear moments** integrated
analytically over the (infinite-tail) BM19 PDF on the s-grid [s_min,s_max]. A finite star field
can't realize those extreme cells, so the analytic P(N) is **over-dispersed**, increasingly at
high σ_s² (high ℳ). Same root cause as Route A, entering through μ and the e^s mixture.

## 5. The bug — for step 2(d) and (f)

### 5.1 What SBC found (AC18, `inference/sbc.py` + `validation/acceptance.py::ac18_sbc_rank_uniformity`)
At the test config (n_trials=30, shape=(24)³, density_shape=(64)³, n_warmup=120, n_samples=200,
n_thin=4): **ℳ χ² rank-uniformity p=0.0049 → FAIL**; α p=0.176 PASS; β p=0.117 PASS. Ranks span
the full {0..50} (not a clipped artifact). With n_bins=2 the χ² is a low/high split → the
signature is a **slope/bias** in the ℳ posterior (truth ranks systematically toward one end),
the classic SBC signature of a **biased marginal** (NOT ∪/∩ over/under-dispersion).

### 5.2 Direct likelihood bias (forward evals, no NUTS; via `count_loglike` = Route B)
argmax_ℳ of Σ count_loglike(ℳ | α*,β*, mock(ℳ*)) vs injected ℳ*, fixed α*=2.5, β*=3.0,
b=0.4, shape=(24)³ — averaged over 5 realizations:
- ℳ*=12: mean bias **+0.23 ± 1.11** (n=5); ℳ*=16: **+0.64 ± 1.29**.
- Single-draw values were misleading (one unlucky key gave +1.59/+2.15) — the MEAN bias is
  modest but POSITIVE (likelihood peaks above ℳ*), with large per-realization scatter. A small
  consistent positive bias across the prior is enough to produce the AC18 rank slope.

### 5.3 Oracle check of the model (corrected — the decisive evidence)
Model σ²_N(ℳ) (Route A `cic_variance` **with the box window `w2`**) vs measured count variance
from a **96³ oracle field** (mean of 4 realizations), b=0.4, α=2.5, β=3.0, cell=4:

| ℳ | model σ²_N | measured (oracle) | error |
|---|-----------|-------------------|-------|
| 3 | 10.40 | 9.52 | **+9%** |
| 5 | 18.03 | 15.98 | +13% |
| 8 | 32.37 | 27.34 | +18% |
| 12 | 54.90 | 43.08 | +27% |
| 16 | 80.31 | 59.17 | **+36%** |

The model **over-predicts** the count variance, growing with ℳ. (NB: my *first* oracle run was
buggy — I called `cell_averaged_xi_rho` **without** `w2`, giving a spurious ~Poisson model and a
fake −95%. The corrected run above, with `w2=box_window_sq_grid(shape,c)`, is the real result.
Lesson for the next session: always pass `w2` for cubic CIC cells.)

### 5.4 Root cause (the physics)
The CIC over-dispersion depends on the **second linear moment** ⟨ρ²⟩=e^{σ_s²} (and ⟨e^{2s}⟩ in
Route B). For a fat tail p(ρ)∝ρ^{−(α+1)}, **⟨ρ²⟩ is tail-dominated and diverges for α≤2**
(α=2.5 here is finite but tail-sensitive). At high ℳ, σ_s² is large and ⟨ρ²⟩ is carried by rare
extreme cells that **no realizable field samples** (even 96³ under-realizes it). So the analytic
model over-predicts the variance of any finite observable → ℳ biased. This is a **fundamental
property of linear-density statistics on fat-tailed fields**, not a coding bug. The cell scale R
was meant to "regularize" this (cell-averaging is smoothing), but the regularization is
insufficient across the high-ℳ end of the prior. **AC16 passes** because it's a single point at
ℳ=5 where the bias is small (+13% variance, ≈+0.05 in ℳ); SBC across ℳ∈[2,20] exposes it.

## 6. What we tried and RULED OUT — for step 2(e)

- **Prior / reparam Jacobian** (`inference/priors.py`, `hmc.py`): VERIFIED CORRECT. ℳ is
  LogUniform → `logpdf` returns −log(ℳ)−norm (∝1/ℳ); reparam ℳ=exp(z₀), `log_jacobian=Σz`
  contributes +z₀; net **flat in log ℳ** as intended. Not the cause. (β also log-uniform and
  PASSES, consistent.)
- **POT-validity barrier (I2, Task 6)**: the earlier truth-keyed `s_thr=s_t(θ*)+margin` barrier
  WAS a real SBC artifact (a trial-dependent prior skewing α-ranks). We DROPPED it (Anna-approved)
  — α now passes. The likelihood is shift-immune so dropping it doesn't bias α. **That fix is
  done and correct; the ℳ failure is separate.** (Commit 266c516.)
- **Quadrature / model internal resolution**: identical argmax at n_max∈{10,16,20},
  n_s∈{400,1024,2048}. NOT a truncation/quadrature issue.
- **Bigger field grid**: a HACK (Anna vetoed) AND ineffective — multi-realization (24³ vs 48³)
  showed the **mean** bias unchanged (only the scatter shrinks). ⟨ρ²⟩ needs astronomically many
  cells at high σ_s²; you cannot brute-force this.
- **Sampler/init/thinning**: the signature is a *slope* (bias), not ∪/∩ (which init/thinning
  would cause). Dispersed NUTS init was added (Task 6) regardless. Not the cause.

## 7. Key files to read (step 1)

- `src/experimental/gravoturb_fdf/theory/cic.py` — **the heart**: `count_distribution` (Route B),
  `smoothed_log_variance`, `effective_mach`, `smoothed_pdf`, `cell_averaged_xi_rho`,
  `cic_variance`, `linear_hermite_coefficients`, `_windowed_series_variance`.
- `src/experimental/gravoturb_fdf/theory/pdf.py` — `bm19_volume_pdf`, the CDF table, moments.
- `src/experimental/gravoturb_fdf/theory/bm19.py` — `sigma_s_squared`, `transition_density`,
  `f_dense_*` (the mass fraction above s_t — a candidate tail-robust statistic).
- `src/experimental/gravoturb_fdf/theory/gaussianization.py` — the log-space Hermite machinery
  (why α/β are tail-robust).
- `src/experimental/gravoturb_fdf/inference/likelihood.py` — `count_loglike`, `data_vector`,
  `tail_exceedance_loglike`, `gaussian_loglike`.
- `src/experimental/gravoturb_fdf/inference/sbc.py` — `sbc_ranks`, `build_logdensity`, `_build_mock`.
- `src/experimental/gravoturb_fdf/field/{field,sampling}.py` — `gaussian_random_field`,
  `rank_copula_field`, `sample_cic_counts` (the data generator).
- `src/experimental/gravoturb_fdf/validation/{acceptance,measure}.py` — AC1–AC18, oracle helpers.

## 8. References / SoTA to research (step 3)

- **Held PDFs** in `docs/core-papers/`: Burkhart 2018, Federrath 2010, Padoan-Nordlund 2011,
  Kainulainen 2014, the Szapudi/Neyrinck/Carron Gaussianization set, Talts 2018, Sailynoja 2022.
- The next session should **research the cosmology counts-in-cells literature** for SoTA
  tail-robust CIC likelihoods on lognormal/fat-tailed density fields — e.g. finite-volume /
  finite-N corrections to ⟨ρ²⟩, the gravitational quasi-equilibrium distribution (Saslaw–Hamilton),
  Sheth's CIC, shifted-lognormal / generalized-extreme-value count models, and robust-moment or
  bulk-only count estimators. (Use WebSearch/the literature tools; ground any formula in a real
  paper per the `no-assumptions-verify-against-pdfs` memory.)

## 9. The open design question — the brainstorm (step 3)

Two repair philosophies (Anna asked to explore BOTH with trade-offs before choosing):

**(A) Make the model predict the FINITE-realizable counts.** Keep stars-as-counts (linear
observable), but compute the model's P(N) moments (μ=⟨e^s⟩, over-dispersion) **consistently with
what a finite field of the data's size realizes** — a finite-population / tail-truncation
treatment matched to the realized field (same spirit as the POT data-derived threshold). The
model stops claiming variance the observable can't contain. *Trade-off:* physically honest (the
observable IS finite), keeps the ℳ-carrier; but ties the model to a realization-dependent cutoff
(must be done so it stays differentiable + SBC-valid, i.e. identical in generation and inference).

**(B) Switch ℳ to a tail-ROBUST statistic.** Demote the tail-sensitive count over-dispersion;
constrain ℳ (σ_s²) from a tail-robust quantity — e.g. the **log-density** cell variance σ_s²(R)
(already β's tail-robust choice; mirrors the Gaussianization philosophy), the **bulk** of P(N)
(low–moderate N, excluding the unrealizable extreme-N tail), or a robust functional (log-counts,
quantiles, f_dense). *Trade-off:* aligns with the two channels that already work (log space);
but the observable is linear star counts, so the connection from counts → a tail-robust ℳ
estimator must be made carefully (and must not double-count β).

**Hybrid** worth weighing: keep P(N) but fit only its tail-robust *bulk* for ℳ while α stays on
the POT tail and β on the log-density 2-pt — i.e. give each parameter the statistic where it is
identifiable AND tail-tame.

**Constraints / success criteria (non-negotiable):**
- **No hacks** (Anna). Fix the physics; no bigger-grid brute force, no test-weakening.
- **JAX-native + differentiable** (HMC needs `jax.grad` through the likelihood).
- **Preserve α (shift-immune POT) and β (Gaussianization)** — they pass; don't regress them.
- **SBC-valid**: any data-derived quantity (e.g. a finite-N cutoff) must be applied identically
  in mock generation and inference (the lesson from the POT s_thr / I2 fix).
- **Validate against the oracle across the whole ℳ prior** (the §5.3 table must flatten to ~few-%
  at all ℳ), and **AC16 must still pass**, and **AC18-ℳ must become uniform**.
- **Released-core invariant**: `pytest tests/unit tests/integration tests/validation -m "not slow"`
  stays **814**. Experimental suite: `PYTHONPATH=src:src/experimental … pytest tests/experimental`.

## 10. Repo / branch / env state (as of this handoff)

- **progenax** branch `gravoturb-fdf-sbc-validation` (LOCAL only, nothing pushed). HEAD `e8d0222`.
  Commits this arc: `a58f867` (Säilynoja grounding) → `266c516` (Task 6 SBC driver + I2
  barrier-drop) → `c13bc15` (jaxstroviz first-class dep) → `e8d0222` (AC18, **xfail**, surfaces
  the ℳ-bug). Working tree CLEAN.
- **jaxstroviz** branch `gravoturb-fdf-sbc-figures` (LOCAL only). HEAD `184cc03`. The **entire
  figure gallery is DONE** (F2a–F4: residual primitive, pdf, spectrum, counts, tail, field,
  SBC ECDF-diff via arviz, forecast, hmc, Q-ladder, released profile-comparison). 848 tests pass
  (807 released + 41 experimental). jaxstroviz `main` is already PUSHED (`b816165`).
- **Env**: progenax now declares jaxstroviz as a first-class `[experimental]` dep (uv.sources +
  matplotlib installed; `uv.lock` updated, `uv lock --check` clean). Run experimental:
  `PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q`.
- **Deferred (blocked on this redesign):** AC18-ℳ pass, AC19 (HMC convergence — Task 8), Task 9
  (the figure-gallery orchestrator `validation/figures.py` + docs sweep + completion doc). The
  jaxstroviz plotters AC19 needs (`plot_hmc_trace/rank`) already exist.
- **Memories**: `gravoturb-fdf-sbc-figures-arc` (arc state), `gravoturb-fdf-differentiable-inference`
  (inference design history), `no-assumptions-verify-against-pdfs`, `hitl-approve-everything`,
  `sota-pass-git-workflow`. A new memory `gravoturb-fdf-count-model-tail-robustness` records this bug.

## 11. Honest framing for Anna

SBC did **exactly its job**: it caught a real, physically-meaningful forward-model limitation
(the linear-density count statistic's fat-tail sensitivity at high Mach) that the single-point
AC16 recovery could not see. That is a **success of the trustworthiness machinery**, not a
setback. The figure gallery + SBC + diagnostics infrastructure is built and green; what remains
is the *scientific* improvement of the ℳ/CIC channel — the right thing to design carefully and to
state SoTA-correctly, which is why Anna is taking it into a fresh, focused brainstorm.
