---
title: The scientific throughline
description: "The unifying idea behind the science demos: a differentiable forward model turns cluster inference into information geometry, telling you BEFORE you observe what is measurable, what is degenerate, and what will bias you. The valuable results are the degeneracies and systematics, not the point estimates — organized here into a measurable / degenerate / biased map of cluster observables."
---

# The scientific throughline

Every demo in this section is the **same inverse problem**: *given a present-day
snapshot of a star cluster, which birth and structural parameters can you actually
recover, and what are the fundamental limits?* That is the inverse of what progenax
does in the forward direction (parameters → initial conditions), run through one
[physics-direct differentiable inference engine](#sec-shared-method):
forward model → analytic predicted statistic → Gaussian/Poisson likelihood → MLE +
**Fisher information**.

The quoted measurements are retained demonstrations, not a fresh release
qualification. Fisher information describes local sensitivity under a specified
forward model, likelihood, covariance and fiducial parameter point. It can expose
local null directions; differentiability alone does not establish an accurate
observation model, globally identified parameters or calibrated uncertainties.

The demos sort onto three axes.

(sec-shared-method)=
## The shared method: physics-direct differentiable inference

The demos share one inference layer (`scripts/_demo_inference.py`). The
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

   Here $\sigma_t^2$ sums both tangential components. The stated standard
   error is a Gaussian, independent, equal-variance component approximation:
   only under those assumptions does the pooled sum have a $\chi^2_{3n}$ law.
   Non-Gaussian DFs, anisotropy and within-bin dispersion variation require
   separate uncertainty checks. **This compression is what
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
   is a fixed constant — so $\nabla_\theta \ln\mathcal L$ is a derivative of that numerical prediction and adopted likelihood.
   Its accuracy still depends on numerical convergence and the likelihood
   assumptions; freezing data avoids resampling noise inside this loss.

4. **Optimize and quantify.** Adam (`mle_adam`, a fixed-step `lax.scan` —
   deterministic and differentiable-friendly) from several dispersed starts;
   the Fisher information from the Gauss-Newton $J^{\mathsf T}J$ of the residual
   vector (`fisher_information_gn` via `jax.jacrev`, which is robust through the
   ODE/quadrature `custom_vjp`s, unlike a full Hessian); and, where affordable,
   posterior sampling through the installed BlackJAX NUTS implementation.
   Convergence and calibration require their own recorded diagnostics.

:::{important}
**The predictor averages the binned squared-velocity moment.** Where the dispersion and number density vary steeply across a wide
outer bin, evaluating $\sigma(\theta)$ at the bin center is biased. Instead each
demo predicts the number-weighted bin average

```{math}
\mathbb E[\hat\sigma_{jk}^2] = \frac{\int_{\rm bin} n_j(r)\,\sigma_j^2(r)\,dr}{\int_{\rm bin} n_j(r)\,dr}.
```

with $n_j(r)=4\pi r^2\nu_j(r)$ the radial shell number weight and
$\nu_j$ the three-dimensional number density. The displayed measure must
include that shell factor. The prediction uses the square root of this
mean squared moment, not exactly the expectation of the finite-sample
$\hat\sigma$ itself. That square-root bias and the Gaussian error approximation
are separate from bin-centering and numerical quadrature error.
:::

## Measurable — clean recovery, then the catch

These establish that the machinery recovers truth, and then find what the point
estimate hides.

```{list-table}
:header-rows: 1

* - Demo
  - Recovered
  - The catch
* - [IMF + equipartition](imf-equipartition.md) (B2)
  - $(\alpha,\,\delta,\,W_0)$ from masses **and** kinematics
  - The revised single-catalogue count/kinematic experiment remains under its
    stated qualification; older mass-channel gains are historical exposures.
* - [Halo + core](halo-core.md) (B3)
  - $(t,\,r_a,\,r_h)$ for a two-family cluster
  - the three are *near-orthogonal* (Fisher correlations $\sim$0) — a best case, by contrast with B11 below
* - [King concentration](king-concentration.md) (B11)
  - $(W_0,\,r_c)$ from **star counts alone**
  - $\rho(W_0, r_c) = -0.91$: a more-concentrated/smaller-core model mimics the same counts. Tight marginals **hide a strong covariance** — counts need kinematics to break it
* - [Anisotropy](anisotropy.md) (B6)
  - the Osipkov–Merritt $r_a$ from $\beta(r)$
  - fit the OM form to a *Michie* cluster and you get a **$12.9\times$ $\chi^2$ inflation and a biased $r_a$** — "the cluster has anisotropy radius $X$" is meaningless without naming the DF family
```

## Degenerate — the Fisher matrix is rank-deficient

The strongest results: the inverse problem is *fundamentally* (not statistically)
ill-posed, and the differentiable Fisher exposes it exactly.

[**Birth-environment archaeology**](birth-environment.md) (B5) is the deepest. The
environment-dependent IMF {cite:p}`Marks2012,Jerabkova2018` maps
$(\,[\mathrm{Fe/H}],\,\log M_{\rm ecl},\,\mathrm{sfe})\to\alpha_3$, the high-mass
slope. $\alpha_3$ is *trivially* measurable (a $\sim$40$\sigma$ top-heavy detection
at $N=10^4$), but the map is **three-to-one**, so that one-slope channel has Fisher rank at most one; at the retained
active-branch fiducial it has rank one. A clipped or constant branch can
have rank zero. Extra observables and external constraints change the target. A continuum of birth
environments produces the *identical* mass function. You cannot read a cluster's
birth metallicity, mass, or star-formation efficiency off its IMF slope without an
external constraint on two of the three. This is a caution for any claim that
infers birth conditions from a present-day mass function.

[**Rotation + projection**](rotation.md) (B8) is the same rank-deficiency from
*geometry*: a cluster rotating at rate $\omega$ viewed at inclination $i$ has a mean
line-of-sight velocity $\propto \omega\sin i$, so $\langle v_{\rm los}\rangle$
constrains only the **product** — the $(\omega, i)$ Fisher is rank 1. Cluster
rotation amplitudes from line-of-sight velocities are lower limits, degenerate with
inclination in this mean-field model. Independent inclination information
would need another model/observable; inferring it from flattening additionally
requires an intrinsic-shape model.

## Biased — wrong-physics systematics

[**Unresolved-binary mass function**](binary-mass-function.md) (B4) is the headline.
Unresolved binaries blend photometrically (their summed light inflates the inferred
mass), distorting the observed mass function. A controlled experiment — shuffle the
mass ratio $q$ globally while retaining the other draws — changes both
$q$–$P$ and $q$–primary-mass dependence. The separation selection also depends
on mass, period and mass ratio. It does not isolate $P$–$q$ coupling alone. **Ignoring it biases the binary fraction low by $\sim$5% ($3.6\sigma$ at
survey scale)**, because the blended (close) subset is more equal-mass under the
real coupling than an independent model assumes. It is a *systematic* — fixed
fractional bias whose significance grows as $\sqrt N$. The reported bias belongs to that mock population, global-shuffle protocol
and selection; it is not a universal bias for every independent-$q$ model.

The [wrong-IMF curve](imf-equipartition.md) (B2d) and the OM-vs-Michie misfit
([B6](anisotropy.md)) are the same lesson in other channels: assume the wrong
generative physics and the recovered parameter shifts in a quantifiable,
detectable way.

## Honest diagnostics — what the observables actually carry

- [**Binary energy budget**](binary-energy-budget.md) (B9): the internal binary
  binding energy is a *separate reservoir* that dwarfs the cluster potential
  (12–1900×) and is environment-dependent — the seed of binary-burning / core-collapse
  arrest. The naive resolved virial ratio is contaminated by orbital phase and is
  **not** the cluster's $Q$; virialize binary ICs on the system centers-of-mass.
- [**Differentiable diagnostics**](diff-diagnostics.md) (B10): the JAX surrogates for
  the CW04 substructure $Q$ {cite:p}`Cartwright2004` and the Allison mass-segregation
  ratio {cite:p}`Allison2009` have selected value/ranking comparisons. The
  segregation surrogate sharp limit is an NN ratio, not exact MST Lambda;
  ranking agreement does not certify its likelihood or gradient target.
- [**Tidal radius**](tidal-radius.md) (B7): $93\%$ of the truncation-radius
  information lives in the few outermost (count-limited) bins, so the Poisson
  treatment matters in this mock. Converting the imposed cutoff to a
  Galactocentric radius additionally assumes tidal filling, known masses,
  a circular orbit and a point-mass host; counts alone do not measure an orbit.
- [**Cross-engine agreement**](cross-engine.md) (B1): two independent engines build
  the *same* cluster — the consistency check the recoveries rest on.

## What this is, and what it is not

```{warning}
**These are idealized mock demonstrations.** Most omit projection,
measurement error and selection. Their numerical forecasts are conditional
on the adopted observables and likelihoods. The stated null directions hold
for those forward maps; additional observables, priors or a different physical
model can change identifiability. A local Fisher forecast is not a posterior
calibration or a universal information bound for all observations.

**progenax is initial-conditions only.** Every result here is a *birth-state*
inference. Real clusters are dynamically processed (binary burning, segregation
growth, tidal stripping), which *adds* degeneracy on top of these; the evolved-cluster
version needs an N-body forward chain.

**Some forward models use approximations** (Tout ZAMS labels, a point-mass
Jacobi conversion, a fixed EFF slope). Their influence on inference must be
checked for the intended population and observation model; qualitative robustness
is not established merely by naming an approximation.
```

## The implication

The reason to build *differentiable* cluster initial conditions is not faster point
estimation. It is that **the Fisher information of a differentiable forward model is a
forecasting instrument**: for a given observable and sample size, it tells you what a
cluster survey can and cannot claim. These demos are a first map of that information
content for star-cluster observables — and the most valuable entries on the map are
the ones that say *"this is degenerate"* (birth environment $\not\leftrightarrow$ IMF
slope; rotation $\not\leftrightarrow$ inclination) and *"this is biased if you assume
the wrong physics"* (binary fraction under the wrong $P$–$q$ coupling; anisotropy under
the wrong DF). Those are cautions an observer can carry to real data.
