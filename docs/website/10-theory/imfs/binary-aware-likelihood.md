---
title: Binary-aware likelihood
description: The system-mass likelihood, its primary-mass quadrature, and the limits of the retained binary-aware recovery evidence.
---

# The binary-aware likelihood

```{seealso}
This chapter focuses on the **likelihood** — the inference-time object
that turns observed system masses into a posterior on $\alpha$. For
the parent chapter that situates the likelihood in its scientific
context (binary contamination, "confidently wrong" regime, observation
operators), see [](binary.md). For the binary statistics that the
likelihood marginalises over, see [](multiplicity-statistics.md) and
[](mass-ratio-distributions.md).
```

Let $M$ be an observed total system mass in solar masses, $N$ the number
of observed systems, and $\alpha$ the dimensionless IMF slope. Here
$\xi(m\mid\alpha)$ is the normalized **primary-mass probability density**
per solar mass on $[m_{\min},m_{\max}]$, zero outside that support.
The binary probability is $f_b(m_1)$ at primary mass $m_1$, and
$g(q\mid m_1)$ is a normalized conditional density for the dimensionless
mass ratio $q=m_2/m_1$ on $[q_{\min},1]$.

The single-star likelihood

```{math}
\mathcal{L}_{\mathrm{naive}}(\alpha)
\;=\; \prod_{i=1}^{N} \xi(M_{\mathrm{sys},i} \mid \alpha)
```

assumes every observed system mass is a single star. The
**binary-aware likelihood** corrects this by marginalising over the
latent binary status and mass ratio of each observation. For each $q$,
the candidate primary mass is $m_1=M/(1+q)$:

```{math}
:label: ba-likelihood
p(M \mid \alpha) \;=\;
\underbrace{[1 - f_b(M)]\,\xi(M \mid \alpha)}_{\text{single}}
\;+\;
\underbrace{\int_{q_{\min}}^{1}
  \frac{\xi(m_1 \mid \alpha)\,f_b(m_1)\,g(q\mid m_1)}{1+q}
  \,\mathrm{d}q}_{\text{binary}}.
```

Each term has direct physical meaning. The single-star contribution is
the density for a system containing one star drawn from the primary
IMF at mass $M$. The binary contribution
integrates over every possible mass-ratio $q$ for which the observed
$M$ could be a binary: the primary had mass $M / (1+q)$, the secondary
had $q \cdot M / (1+q)$, and the IMF and mass-ratio distribution
combine with $f_b(m_1)$ to give the joint probability density. This
fraction stays **inside** the integral: different candidate primary
masses have different probabilities of being binary. Only a constant
binary fraction can be factored out. The Jacobian
$1/(1+q)$ comes from the change of variables $m_{\mathrm{sys}} = m_1
(1+q)$.

This page derives the integrand, walks through the 128-point
Gauss-Legendre quadrature used by the recovery scripts, and describes its cost.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers wanting the inference-time mechanics of the binary-aware likelihood — the quadrature, vectorisation, and gradients; no prior Bayesian-inference literature assumed.
**Prerequisites:** the [binary-aware IMF chapter](binary.md) (the scientific motivation and the system mass function this likelihood inverts).
**You'll get:** the conditional binary weight, the 128-point Gauss-Legendre quadrature, the $N \times K$ vectorised tile, and the distinction between likelihood derivatives and posterior qualification.
:::

## Why no closed form

The recovery scripts use numerical integration for the IMF and
mass-dependent binary statistics. Special choices can have analytic
integrals; the general implementation does not require one:

1. **The IMF $\xi$** — closed form for {cite:t}`Maschberger2013`,
   piecewise for {cite:t}`Kroupa2001`.
2. **The mass-ratio distribution $g(q \mid m_1)$** —
   {eq}`gq-form` from [](mass-ratio-distributions.md), piecewise in
   $m_1$ via the {cite:t}`MoeDiStefano2017` mass-bin lookup.
3. **The Jacobian $1/(1+q)$** — trivial.

The piecewise mass-bin lookups in $f_b(m_1)$ and $g(q \mid m_1)$
introduce boundaries that depend on the
candidate primary mass $m_1 = M/(1+q)$, which depends on the
integration variable $q$. progenax evaluates the integral
numerically.

## The Gauss-Legendre quadrature

For an integration variable $t$ on $[a,b]$, $K$-point Gauss-Legendre
quadrature uses nodes $x_k$ and weights $w_k$ on $[-1,1]$:

```{math}
:label: gl-quadrature
\int_a^b h(t)\,\mathrm{d}t
\;\approx\;
\frac{b-a}{2}\,\sum_{k=1}^{K} w_k\,
h\!\left(\frac{b-a}{2}x_k+\frac{a+b}{2}\right).
```

Both recovery scripts precompute $K=128$ nodes and weights at module
load. They use **quadrature over primary mass**, with
$a=\max(M/2,m_{\min})$ and $b=\min(M/(1+q_{\min}),m_{\max})$.
The binary integrand is $\xi(m_1\mid\alpha)f_b(m_1)
g(M/m_1-1\mid m_1)/m_1$, as derived in {eq}`psys`.
The contribution is zero if $b\le a$. Systems can extend to
$2m_{\max}$ even though primaries stop at $m_{\max}$.

```{warning} Quadrature accuracy needs its own comparison
The empirical mass-bin boundaries make the integrand piecewise smooth.
A fixed node count alone does not establish a relative accuracy, and
agreement of AD with finite differences does not test integration error.
The retained recovery evidence does not qualify a universal
$10^{-10}$ quadrature budget across masses and IMF slopes.
```

## Vectorisation: the $N \times K$ tile

For $N$ observed systems and $K$ quadrature points, the likelihood
evaluation is a tensor contraction:

```{note}
The block below shows the existing script calculation; its bindings are
`Maschberger`, `MassDependentBinaryFraction`, `MoeDiStefano2017`, the
128-point `GL_NODES`/`GL_WEIGHTS`, and the scripts' mass bounds and
`Q_MIN`. The public `BinaryIMF` exposes sampling helpers such as
`sample_systems`; the likelihood remains owned by the validation scripts.
```

```python
@jax.jit
def log_likelihood(alpha, M_sys):
    imf = Maschberger(alpha=alpha, m_min=M_MIN, m_max=M_MAX)
    bf = MassDependentBinaryFraction()
    qd = MoeDiStefano2017(q_min=Q_MIN)
    lo = jnp.maximum(M_sys / 2, M_MIN)                   # (N,)
    hi = jnp.minimum(M_sys / (1 + Q_MIN), M_MAX)         # (N,)
    half = (hi - lo) / 2
    m1 = half[:, None] * GL_NODES + ((hi + lo) / 2)[:, None]
    q = jnp.clip(M_sys[:, None] / m1 - 1, Q_MIN, 1)
    g = jax.vmap(jax.vmap(qd.pdf_given_primary))(q, m1)   # (N, K)
    integrand = bf(m1) * jnp.exp(imf.logpdf(m1)) * g / m1
    p_binary = jnp.where(hi > lo, half * (integrand @ GL_WEIGHTS), 0)
    p_single = (1 - bf(M_sys)) * jnp.exp(imf.logpdf(M_sys))
    log_p = jnp.log(jnp.maximum(p_single + p_binary, 1e-30))
    return jnp.sum(jnp.where(jnp.isfinite(log_p), log_p, -1e10))
```

The tile carries $N\times K$ candidate primaries. `vmap` evaluates the
conditional mass-ratio density, and the weighted sum reduces its
quadrature axis. The shown JIT wrapper and the scripts evaluate the
same mathematical calculation.

## Numerical-stability notes

Three details that matter at large $N$:

1. **Probability-space sums.** The IMF is evaluated through `logpdf`
   and exponentiated. The scripts sum the quadrature and the
   single/binary mixture in probability space, rather than using
   `logsumexp` throughout.
2. **Finite log-density guards.** Both scripts floor the summed density
   at $10^{-30}$ before taking its logarithm, and replace nonfinite
   log values with $-10^{10}$. These guards are numerical approximations:
   an impossible observation receives a finite penalty instead of
   $-\infty$. They require review before general inference use.
3. **$q_{\min}$ floor.** The integral lower limit is $q_{\min} = 0.1$
   matching {cite:t}`MoeDiStefano2017`'s observational completeness. progenax
   scripts do not extrapolate this empirical calibration below its chosen
   support. Whether a density remains integrable at zero depends on its
   mass-ratio law; the change-of-variables Jacobian is finite there.

## Computational cost

For $N$ observed systems and $K = 128$ quadrature nodes, each
likelihood evaluation requires $N \times K$ IMF + mass-ratio
evaluations:

```{list-table}
:header-rows: 1

* - $N$
  - Candidate primary evaluations, $N \times K$
* - 100
  - $1.3\!\times\!10^4$
* - 1{,}000
  - $1.3\!\times\!10^5$
* - 10{,}000
  - $1.3\!\times\!10^6$
* - 30{,}000
  - $3.8\!\times\!10^6$
```

These are evaluation counts, not timings. The tile requires storage
proportional to $NK$, with additional intermediates for reverse AD.
NUTS uses a variable number of likelihood/gradient evaluations per draw;
chain cost cannot be obtained by multiplying this count by the sample count.
No matched CPU/A100 timing exposure is retained here.

## Differentiability

The scripts differentiate the discretized log likelihood with respect
to $\alpha$, holding masses, binary statistics and quadrature nodes fixed:

- $\xi(m \mid \alpha)$ — analytic in $\alpha$ for {cite:t}`Maschberger2013`,
  piecewise-analytic for {cite:t}`Kroupa2001`.
- $g(q \mid m_1)$ and $f_b(m_1)$ — the empirical mass-bin lookups use
  `jnp.where` and remain fixed as $\alpha$ changes. They are discontinuous
  at their primary-mass boundaries; no sigmoid-blend option is used here.
- The Gauss-Legendre quadrature is a fixed weighted sum.

`jax.grad(log_likelihood)(alpha, M_sys)` returns $\partial \log
\mathcal{L}/\partial \alpha$ for the numerical objective. The MLE script
checks this derivative against central finite differences. Its local
uncertainty uses finite-difference curvature at the fit; an observed
Hessian at arbitrary parameters is not the expected Fisher information.
Neither check establishes quadrature convergence or gradients through
moving support and empirical mass-bin boundaries.

## Connection to posterior inference

A posterior log density combines this deterministic likelihood with
an explicitly chosen prior. Ordinary NUTS requires a deterministic
gradient; a noisy population-expectation estimator is not a drop-in
replacement. New recovery qualification uses BlackJAX, with its priors,
convergence design and independent catalogues specified before execution.

The retained binary-only posterior artifact comes from the historical
NumPyro script: two chains of 1,000 samples each. It does not establish
four-chain convergence, repeated-catalogue coverage or the separate
cluster IMF/equipartition demo. See the
[validation owner](../../50-validation/binary-imf.md#honest-scope)
for the saved diagnostics and their scope.

## What if the binary statistics are wrong?

The likelihood above assumes *known* $f_b(m_1)$ and $g(q \mid m_1)$.
In real applications, the {cite:t}`MoeDiStefano2017` calibration carries
its own uncertainties: $\pm 0.04$ for solar-type binary fractions,
growing to $\pm 0.10$ for O-stars. Misspecification of these
statistics reintroduces bias on $\alpha$ at large $N$.

The architectural extension is a **hierarchical likelihood**: jointly
infer $\alpha$ alongside hyper-parameters of the binary statistics
(an overall $f_b$ multiplicative factor, a $\gamma$ offset). The
extra parameters would propagate calibration uncertainty, but their
identifiability and convergence require their own qualification. progenax does
not currently export a `BinaryIMF.with_inferred_binary_stats()` helper;
this is a planned likelihood-layer extension rather than a live API.

## Implementation, validation & references

- **In code:** the binary-aware forward model lives in
  `src/progenax/imf/binary/imf.py` (`BinaryIMF`); the `log_likelihood`
  block on this page mirrors `scripts/validate_binaries.py::aware_loglike`
  and the historical recovery script's `log_system_mass_likelihood`.
  The public `BinaryIMF` exposes sampling helpers rather than this likelihood. See the
  [IMF API](../../30-api/imf.md).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md)
  — retained MLE figures and a historical two-chain artifact; general
  quadrature accuracy and posterior coverage remain unqualified.
- **Primary sources:** the likelihood structure and "confidently wrong"
  framing are original to progenax; the binary-statistics calibration is
  {cite:t}`MoeDiStefano2017`. Full notes
  in the [bibliography](../../99-bibliography/per-paper/moe-distefano-2017.md);
  the scientific context is at [](binary.md).
