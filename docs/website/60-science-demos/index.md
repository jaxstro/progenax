---
title: Science demos
description: "End-to-end methods demonstrations that turn progenax's differentiable initial conditions into inference — recovering cluster birth and structural parameters from mock observations, each a gated CLI with measured recovery tables. See 'The scientific throughline' for the synthesis: what is measurable, what is degenerate, and what will bias you."
---

# Science demos

The [validation section](../50-validation/index.md) proves that each progenax
model reproduces its analytic or published ground truth. This section answers a
different question: **can you run the models *backwards* — recover the
parameters of a cluster from mock observations of it — and does the recovery
land on truth?** That is the use case the whole package is built for: a
*differentiable* forward model whose gradients drive maximum-likelihood and
Hamiltonian-Monte-Carlo inference.

```{tip}
**Read [The scientific throughline](throughline.md) first** for the synthesis — the
demos sort onto three axes (*measurable*, *degenerate*, *biased*), and the valuable
results turn out to be the degeneracies and systematics, not the point estimates.
```

Each demo is a standalone gated CLI in `scripts/` (the same `validate_*` house style
— it exits nonzero if any recovery gate fails).

**Batch B** (kinematic recovery, the $\sigma(r)$ channel):

```{list-table}
:header-rows: 1

* - Demo
  - Question
  - Recovered
* - [Cross-engine agreement](cross-engine.md) (B1)
  - Do the DF-defined and density-defined engines build the *same* cluster?
  - (none — a consistency check)
* - [IMF + equipartition](imf-equipartition.md) (B2)
  - Can one IMF slope $\alpha$ be measured jointly from masses *and* kinematics?
  - $(\alpha,\ \delta,\ W_0)$
* - [Halo + core](halo-core.md) (B3)
  - Can a two-family cluster's mass split + anisotropy be recovered?
  - $(t,\ r_a,\ r_h)$
```

**Batch C** (binaries, environment IMF, structural recovery, diagnostics — paper
seeds + methods showcases):

```{list-table}
:header-rows: 1

* - Demo
  - Question
  - Result
* - [Binary energy budget](binary-energy-budget.md) (B9)
  - How big is the primordial-binary energy reservoir?
  - dwarfs $|W|$ 12–1900×; environment-dependent
* - [King concentration](king-concentration.md) (B11)
  - Recover $(W_0, r_c)$ from star **counts** alone?
  - yes — with a $\rho=-0.91$ degeneracy
* - [Differentiable diagnostics](diff-diagnostics.md) (B10)
  - Do JAX $Q$ / $\Lambda_{\rm MSR}$ surrogates track the exact statistics?
  - yes (substructure regime); usable as a loss
* - [Birth environment](birth-environment.md) (B5)
  - Read the birth conditions off the IMF?
  - $\alpha_3$ yes; environment **rank-1 unrecoverable**
* - [Binary mass function](binary-mass-function.md) (B4)
  - Recover $f_b$ from the unresolved-binary distortion?
  - yes — ignoring Moe $P$–$q$ coupling biases it $-3.6\sigma$
* - [Binary dynamical mass](binary-dynamical-mass.md) (B12)
  - Remove the binary-inflated virial-mass bias?
  - yes — from the velocity wings ($1.28\times$ bias removed; dispersion-only is rank-1)
* - [Anisotropy](anisotropy.md) (B6)
  - Measure the anisotropy radius from $\beta(r)$?
  - yes (OM); a Michie cluster mis-fits $12.9\times$
* - [Tidal radius](tidal-radius.md) (B7)
  - Recover $r_t$ → Galactocentric distance?
  - from the count-limited outskirts (93% of the info)
* - [Rotation + projection](rotation.md) (B8)
  - Recover the rotation rate from $\langle v_{\rm los}\rangle$?
  - only $\omega\sin i$ — rank-1 with inclination
* - [Optimal experimental design](optimal-design.md) (B14)
  - *Pre-data:* where to spend a star budget to pin $r_a$?
  - PMs to the outskirts; $3.66\times$ fewer stars at equal precision
```

## The shared method: physics-direct differentiable inference

All three demos share one inference layer (`scripts/_demo_inference.py`). The
design avoids both expensive likelihood-free / simulation-based inference and
any non-differentiable resampling inside the loss. The recipe:

1. **Forward-sample mock stars at truth.** Draw $N$ positions+velocities from a
   progenax model (here $N = 3\times10^4$ to $10^5$). This is done *once*, to
   make the data; it is **not** repeated inside the optimizer.

2. **Compress to binned kinematic summaries.** Reduce the stars to per-group (or
   per-component) binned 1-D velocity dispersions $\hat\sigma_{1\mathrm D,j}(r)$
   on a frozen set of radial bins, each carrying a finite-$N$ standard error.
   For an isotropic pooled estimator over the three velocity components,

   ```{math}
   :label: sci-sigma-estimator
   \hat\sigma_{1\mathrm D}^2 = \frac{1}{3}\,\overline{|\mathbf v|^2}
   = \frac{\sigma_r^2 + \sigma_t^2}{3},
   \qquad
   \mathrm{SE}(\hat\sigma) = \frac{\hat\sigma}{\sqrt{6 n}},
   ```

   where the $\sqrt{6n}$ (not $\sqrt{2n}$) comes from pooling all *three*
   one-D components into a $\chi^2_{3n}$ inner sum. **This compression is what
   makes the cost $N$-independent:** the likelihood sees a few hundred binned
   numbers, never the $10^5$ stars.

3. **Likelihood against analytic predictions.** Compare the binned data to the
   model's *analytic* expectation with a Gaussian (χ²) likelihood,

   ```{math}
   :label: sci-loglike
   \ln \mathcal L(\theta) = -\tfrac12 \sum_{j,k} w_{jk}
   \left(\frac{\hat\sigma_{jk} - \sigma_{jk}^{\rm pred}(\theta)}{\mathrm{SE}_{jk}}\right)^2 ,
   ```

   with $w_{jk}\in\{0,1\}$ masking under-occupied bins. **Gradients flow only
   through the analytic prediction** $\sigma^{\rm pred}(\theta)$ — the data side
   is a fixed constant — so $\nabla_\theta \ln\mathcal L$ is exact and cheap, and
   there is no reparametrization-gradient noise from resampling.

4. **Optimize and quantify.** Adam (`mle_adam`, a fixed-step `lax.scan` —
   deterministic and differentiable-friendly) from several dispersed starts;
   the Fisher information from the Gauss-Newton $J^{\mathsf T}J$ of the residual
   vector (`fisher_information_gn` via `jax.jacrev`, which is robust through the
   ODE/quadrature `custom_vjp`s, unlike a full Hessian); and, where affordable,
   full posterior sampling with a vendored blackjax NUTS.

```{important}
**The predictor is the unbiased *binned expectation*, not a bin-centre
evaluation.** Where the dispersion and number density vary steeply across a wide
outer bin, evaluating $\sigma(\theta)$ at the bin centre is biased. Instead each
demo predicts the number-weighted bin average

​   $$\mathbb E[\hat\sigma_{jk}^2] = \frac{\int_{\rm bin} n_j(r)\,\sigma_j^2(r)\,dr}{\int_{\rm bin} n_j(r)\,dr},$$

with $n_j(r)$ the model's own number density — a like-with-like comparison to
the binned data, computed by a cumulative-trapezoid-at-edges trick that stays
differentiable.
```

## Honest scope — these are clean-mock methods showcases

```{warning}
The demos demonstrate the **inference machinery on mock data drawn from the same
model family**, with deliberately idealized observations. They do **not** claim
recovery from real data. With few exceptions, the demos do not model:

- **line-of-sight projection** — most kinematic observables are the full 3-D
  $\sigma_{1\mathrm D}(r)$ and $\beta(r)$, not projected/proper-motion quantities
  ([B8](rotation.md) is the exception that *introduces* a projection helper — the
  realism-axis bridge);
- **measurement errors or selection / incompleteness** — every star is observed
  with its exact phase-space coordinates;
- **model misspecification of the family** — truth and fit usually share the same
  generative model (the deliberate exceptions are [B2d](imf-equipartition.md)'s
  wrong-IMF curve, [B4](binary-mass-function.md)'s wrong $P$–$q$ coupling, and
  [B6](anisotropy.md)'s OM-vs-Michie misfit — each *reported* as the result);
- **SBC-calibrated posteriors** — where NUTS is run, the corner is non-divergent
  with posterior mean ≈ MLE, but simulation-based calibration is future work.

Adding full observational realism (projection, errors, incompleteness) across all
demos is the natural next arc; see [the throughline](throughline.md) for why the
*degeneracies* survive it while the *measurable* numbers are optimistic ceilings.
```

## Running the demos

The inference layer needs the `experimental` optional dependencies (blackjax for
NUTS, optax for Adam):

```bash
env -u VIRTUAL_ENV uv pip install -e ".[dev,experimental]"

# each demo is a gated CLI (exits nonzero on a recovery-gate failure)
# --- Batch B (kinematic recovery) ---
python scripts/demo_cross_engine.py        # B1 (seconds)
python scripts/demo_delta_recovery.py      # B2 headline MLE + Fisher (minutes)
python scripts/demo_delta_recovery.py --run-nuts   # + the ~52 min NUTS corner
python scripts/demo_delta_recovery_bias.py # B2 wrong-IMF curve + robustness grid
python scripts/demo_halo_core.py           # B3 (~4 min, MLE-compile-dominated)
# --- Batch C (binaries, environment IMF, structure, diagnostics) ---
python scripts/demo_binary_energy_budget.py  # B9 (~30 s)
python scripts/demo_king_concentration.py    # B11 (~30 s)
python scripts/demo_diff_diagnostics.py      # B10 (~6 s)
python scripts/demo_birth_environment.py     # B5 (~7 s)
python scripts/demo_binary_mass_function.py  # B4 (ZAMS relations in-package via progenax.stellar)
python scripts/demo_binary_dynamical_mass.py # B12 (~90 s; kinematic companion to B4)
python scripts/demo_anisotropy.py            # B6 (~9 s)
python scripts/demo_tidal_radius.py          # B7 (~5 s)
python scripts/demo_rotation.py              # B8 (~4 s)
```

Figures are written to `validation/plots/` (a gitignored, regeneratable
artifact); the copies embedded on these pages live in
`docs/website/60-science-demos/figures/`.
