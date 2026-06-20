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
* - [Optimal experimental design](optimal-design/index.md) (B14)
  - *Pre-data:* where to spend a star budget to pin $r_a$, and how deep to weigh $M$?
  - PMs to the outskirts ($3.66\times$ fewer stars); an **interior** optimal survey depth for $M_{\rm dyn}$
```

## The shared method: physics-direct differentiable inference

The demos share one inference layer (`scripts/_demo_inference.py`): forward-sample
mock stars at truth *once*, compress them to binned kinematic summaries with honest
finite-$N$ standard errors, evaluate a Gaussian/Poisson likelihood against the
model's *analytic* prediction (so gradients flow only through the prediction, not a
resampled data side), then optimize with Adam and quantify with the Gauss-Newton
Fisher (and, where affordable, blackjax NUTS). The full four-step recipe — with the
$\sqrt{6n}$ pooled-dispersion estimator, the likelihood, and the binned-expectation
predictor — is derived once in
[**The scientific throughline → the shared method**](throughline.md#sec-shared-method).

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
