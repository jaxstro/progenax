---
title: Binary-aware likelihood
description: The Gauss-Legendre quadrature that marginalises over binary status and mass ratio at inference time, producing an unbiased α posterior even at LSST-scale sample sizes.
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

The single-star likelihood

```{math}
\mathcal{L}_{\mathrm{naive}}(\alpha)
\;=\; \prod_{i=1}^{N} \xi(M_{\mathrm{sys},i} \mid \alpha)
```

assumes every observed system mass is a single star. The
**binary-aware likelihood** corrects this by marginalising over the
latent binary status and mass ratio of each observation:

```{math}
:label: ba-likelihood
p(M \mid \alpha) \;=\;
\underbrace{(1 - \bar f_b)\,\xi(M \mid \alpha)}_{\text{single}}
\;+\;
\underbrace{\bar f_b \int_{q_{\min}}^{1}
  \frac{\xi\!\bigl(\tfrac{M}{1+q} \mid \alpha\bigr)}{1+q}\,
  g\!\bigl(q \mid \tfrac{M}{1+q}\bigr)\,\mathrm{d}q}_{\text{binary}}
```

Each term has direct physical meaning. The single-star contribution is
the probability that the observed mass $M$ is *just one star*, with
that star drawn from the IMF at mass $M$. The binary contribution
integrates over every possible mass-ratio $q$ for which the observed
$M$ could be a binary: the primary had mass $M / (1+q)$, the secondary
had $q \cdot M / (1+q)$, and the IMF and mass-ratio distribution
combine to give the joint probability density. The Jacobian
$1/(1+q)$ comes from the change of variables $m_{\mathrm{sys}} = m_1
(1+q)$.

This page derives the integrand, walks through the 128-point
Gauss-Legendre quadrature progenax uses, and quantifies the cost.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers wanting the inference-time mechanics of the binary-aware likelihood — the quadrature, vectorisation, and gradients; no prior Bayesian-inference literature assumed.
**Prerequisites:** the [binary-aware IMF chapter](binary.md) (the scientific motivation and the system mass function this likelihood inverts).
**You'll get:** why the integral has no closed form, the 128-point Gauss-Legendre quadrature, the $N \times K$ vectorised tile, numerical-stability notes, and how it plugs into NUTS / NumPyro.
:::

## Why no closed form

The integrand of {eq}`ba-likelihood` involves three pieces that all
have closed-form expressions individually, but no closed form for
their product:

1. **The IMF $\xi$** — closed form for {cite:t}`Maschberger2013`,
   piecewise for {cite:t}`Kroupa2001`.
2. **The mass-ratio distribution $g(q \mid m_1)$** —
   {eq}`gq-form` from [](mass-ratio-distributions.md), piecewise in
   $m_1$ via the {cite:t}`MoeDiStefano2017` mass-bin lookup.
3. **The Jacobian $1/(1+q)$** — trivial.

The piecewise mass-bin lookup in $g(q \mid m_1)$ is what kills any
hope of a closed form. The integrand is a sum of products of
analytic pieces, but the *boundaries* of each piece depend on the
candidate primary mass $m_1 = M/(1+q)$, which depends on the
integration variable $q$. progenax evaluates the integral
numerically.

## The Gauss-Legendre quadrature

For a smooth integrand on $[a, b]$, $K$-point Gauss-Legendre
quadrature gives roughly $14$ digits of accuracy at $K = 128$. The
integrand of {eq}`ba-likelihood` is smooth on the interior $(q_{\min},
1)$ with weak edge behaviour at $q \to 1$ (where the Jacobian $1/(1+q)
\to 1/2$ and the twin Gaussian peaks). Gauss-Legendre with $K = 128$
nodes is the right tool:

```{math}
:label: gl-quadrature
\int_{q_{\min}}^{1} h(q)\,\mathrm{d}q
\;\approx\;
\frac{1 - q_{\min}}{2}\,\sum_{k=1}^{K} w_k\,h\!\biggl(\frac{(1 - q_{\min})\,x_k + (1 + q_{\min})}{2}\biggr)
```

where $\{x_k, w_k\}$ are the standard Gauss-Legendre nodes and weights
on $[-1, 1]$. progenax precomputes $\{x_k, w_k\}$ at module-import
time and passes them as a fixed array to the JIT-compiled likelihood.

The 128-point choice is calibrated against analytic test cases (where
the integrand reduces to closed-form sub-cases) to give $\le 10^{-10}$
relative error across the full $\alpha \in [0.5, 4.0]$ range and
$M \in [m_{\min}, m_{\max}]$. Smaller $K$ shows degradation near
$q_{\min}$ where the integrand can be steep.

## Vectorisation: the $N \times K$ tile

For $N$ observed systems and $K$ quadrature points, the likelihood
evaluation is a tensor contraction:

```{note}
Implementation status: the block below is schematic likelihood
pseudocode. The current public `BinaryIMF` exposes sampling helpers such
as `sample_systems`; it does not expose this exact `log_prob` method as
a copy-paste API.
```

```python
@jax.jit
def log_likelihood(alpha, M_sys):
    # q_nodes, w_nodes: shape (K,) — fixed
    q = q_nodes_scaled                                  # (K,)

    # Candidate primary masses for each (M_sys, q) pair
    m1 = M_sys[:, None] / (1.0 + q[None, :])            # (N, K)

    # IMF + mass-ratio distribution at each (m1, q)
    log_xi = imf.logpdf(m1, alpha=alpha)                # (N, K)
    log_g = mass_ratio_log_prob(q[None, :], m1)         # (N, K)

    # Integrand and quadrature
    log_integrand = log_xi + log_g - jnp.log(1.0 + q[None, :])
    binary_integral = jnp.sum(w_nodes * jnp.exp(log_integrand), axis=1)

    # Combine single + binary contributions
    f_b = binary_fraction(M_sys)                        # (N,)
    p = (1 - f_b) * jnp.exp(imf.logpdf(M_sys, alpha)) + f_b * binary_integral

    return jnp.sum(jnp.log(p))
```

The whole pipeline is one JIT trace and one device call — no Python
loops, no rejection sampling, no per-particle branching. Gradients
flow analytically through every line.

## Numerical-stability notes

Three details that matter at large $N$:

1. **Log-space evaluation.** `log_xi + log_g - log(1+q)` is computed in
   log-space, then `jnp.exp` is applied before the quadrature sum.
   The intermediate log values can be large negative (for unlikely
   mass-ratio combinations), but never produce overflow.
2. **`logsumexp` for the final mixture.** The single-star and binary
   contributions can differ by orders of magnitude at the tails of the
   mass distribution. progenax uses `jax.scipy.special.logsumexp` for
   the final $\log[(1-f_b)\xi + f_b\,I]$ to avoid catastrophic
   cancellation.
3. **$q_{\min}$ floor.** The integral lower limit is $q_{\min} = 0.1$
   matching {cite:t}`MoeDiStefano2017`'s observational completeness. progenax
   does not extrapolate below this; setting $q_{\min} = 0$ would make
   the integral diverge for some IMF parameter values.

## Computational cost

For $N$ observed systems and $K = 128$ quadrature nodes, each
likelihood evaluation requires $N \times K$ IMF + mass-ratio
evaluations:

```{list-table}
:header-rows: 1

* - $N$
  - $N \times K$ ops
  - CPU time
  - GPU time (A100)
* - 100
  - $1.3\!\times\!10^4$
  - $\sim 1$ ms
  - $\sim 0.05$ ms
* - 1{,}000
  - $1.3\!\times\!10^5$
  - $\sim 5$ ms
  - $\sim 0.1$ ms
* - 10{,}000
  - $1.3\!\times\!10^6$
  - $\sim 50$ ms
  - $\sim 0.5$ ms
* - 30{,}000
  - $3.8\!\times\!10^6$
  - $\sim 150$ ms
  - $\sim 1$ ms
```

Per chain (NUTS, 1500 steps) at $N = 30{,}000$:
- CPU: $\sim 35$ minutes per chain.
- GPU (A100): $\sim 30$ seconds per chain — a $\sim 70\times$ speedup.

The $N \times K$ tile is embarrassingly parallel via `jax.vmap`, which
is what makes the GPU advantage so dramatic.

## Differentiability

Every step in {eq}`ba-likelihood` is JAX-differentiable:

- $\xi(m \mid \alpha)$ — analytic in $\alpha$ for {cite:t}`Maschberger2013`,
  piecewise-analytic for {cite:t}`Kroupa2001`.
- $g(q \mid m_1)$ — power-law and Gaussian both analytic; the mass-bin
  lookup uses `jnp.where` (smooth fallback for boundary
  derivatives optional via sigmoid-blend).
- $\bar f_b(m_1)$ — `jnp.interp` on the {cite:t}`MoeDiStefano2017` Table 13
  values, differentiable via piecewise-linear gradients.
- The Gauss-Legendre quadrature is a fixed weighted sum.

`jax.grad(log_likelihood)(alpha, M_sys)` returns $\partial \log
\mathcal{L}/\partial \alpha$ in one backward pass; `jax.hessian`
returns the Fisher-information matrix needed for Laplace-approximation
posteriors. Both work without any rewriting.

## Connection to NUTS / NumPyro

progenax exposes the binary-aware likelihood through a NumPyro
`numpyro.factor` statement:

```python
import numpyro
import numpyro.distributions as dist

def model(M_sys):
    alpha = numpyro.sample("alpha", dist.Uniform(0.5, 4.0))
    numpyro.factor("ll", log_likelihood(alpha, M_sys))

# Run NUTS
from numpyro.infer import MCMC, NUTS
mcmc = MCMC(NUTS(model), num_warmup=500, num_samples=1000, num_chains=4)
mcmc.run(jax.random.PRNGKey(0), M_sys=observed_masses)
```

NUTS adaptively chooses step sizes and trajectory lengths; the
binary-aware likelihood's smooth gradients keep step-size adaptation
stable across the full prior range. Production runs (the validation
suite at [](../../50-validation/binary-imf.md)) use 4 chains × 1500
post-warmup samples, which is sufficient for $\hat R < 1.01$ on
$\alpha$ at $N = 30{,}000$.

## What if the binary statistics are wrong?

The likelihood above assumes *known* $f_b(m_1)$ and $g(q \mid m_1)$.
In real applications, the {cite:t}`MoeDiStefano2017` calibration carries
its own uncertainties: $\pm 0.04$ for solar-type binary fractions,
growing to $\pm 0.10$ for O-stars. Misspecification of these
statistics reintroduces bias on $\alpha$ at large $N$.

The architectural extension is a **hierarchical likelihood**: jointly
infer $\alpha$ alongside hyper-parameters of the binary statistics
(an overall $f_b$ multiplicative factor, a $\gamma$ offset). The
extra parameters cost 2–3 NUTS steps in convergence rate but produce
posteriors that account for the calibration uncertainty. progenax does
not currently export a `BinaryIMF.with_inferred_binary_stats()` helper;
this is a planned likelihood-layer extension rather than a live API.

## Implementation, validation & references

- **In code:** the binary-aware forward model lives in
  `src/progenax/imf/binary/imf.py` (`BinaryIMF`); the `log_likelihood`
  block on this page is schematic pseudocode (the public `BinaryIMF`
  exposes sampling helpers, not this exact `log_prob`). See the
  [IMF API](../../30-api/imf.md).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md)
  — 4 chains × 1500 post-warmup samples demonstrate unbiased $\alpha$
  recovery ($\hat R < 1.01$) at $N = 30{,}000$.
- **Primary sources:** the likelihood structure and "confidently wrong"
  framing are original to progenax; the binary-statistics calibration is
  {cite:t}`MoeDiStefano2017`, sampled with NUTS via NumPyro. Full notes
  in the [bibliography](../../99-bibliography/per-paper/moe-distefano-2017.md);
  the scientific context is at [](binary.md).
