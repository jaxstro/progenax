---
title: Inferring β from the projected map — log₊ band-powers and the analytic shot transfer
description: A first-principles derivation of the differentiable, SBC-calibrated estimator for the turbulence power-spectrum slope β from a 2-D projected star catalogue. Why raw band-powers fail, why log₊ (not rank-Gaussianization) is the right observable, and the analytic Poisson-shot transfer that keeps the β-response analytic at all stellar densities.
---

# Inferring β from the projected map

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent + fractal-density-field (FDF) pipeline is the standalone **`gravoturb_fdf`**
package — a follow-up-paper feature **excluded from the released progenax wheel** (repo-only, under
`src/experimental/`). Fresh validation: `src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`.
```

```{admonition} Status — active research, honestly scoped
:class: note
The high-stellar-density regime is **SBC-calibrated** (single-cluster β rank-uniformity *p* = 0.82).
The analytic Poisson-shot transfer derived below extends this to low density; its one approximation
(a lognormal projected-density marginal) is **under SBC test**. Numbers here are first-hand and
re-runnable (`src/experimental/gravoturb_fdf/validation/_d0*`, `_v3_logp_sbc.py`).
```

```{seealso}
This page runs the [differentiable-inference](differentiable-inference.md) idea in **projection**:
the 3-D forward model ([](density-pdf-fundamentals.md), [](bm19.md), [](pdf-and-fdf.md)) seen as a
2-D star map. The pivot is the turbulence power-spectrum slope $\beta$ ($P(k)\propto k^{-\beta}$).
```

A young cluster's projected star map is a noisy, squashed shadow of the 3-D turbulent gas that made
it. We want one number from that shadow: the natal turbulence slope $\beta$ — the rigorous,
differentiable successor to the heuristic $Q$/MST substructure metrics. The generative chain is

```{math}
:label: betaproj-chain
g \;[\,P(k)=k^{-\beta}\,] \;\xrightarrow{\text{BM19 copula}}\; s \;\xrightarrow{\rho=e^{s}}\;
\rho \;\xrightarrow{\text{LOS sum}}\; \Sigma \;\xrightarrow{\text{Poisson}}\; N(\mathbf{x}),
```

a Gaussian field $g$ → log-density $s$ → density $\rho$ → projected (column) density $\Sigma$ →
star counts $N$. Every arrow after the first either destroys information about $\beta$ or breaks the
statistical properties an estimator needs. The art is finding the summary whose **mean is analytic
in $\beta$** *and* whose **likelihood is tractable**.

## Why the obvious estimator fails

The natural summary is the angular power spectrum (band-powers) of the count map. It fails as a
Gaussian-likelihood target for a sharp, first-principles reason.

A band-power is *quadratic* in the field — an average of $\sim N_{\text{modes}}$ squared Fourier
amplitudes. For a **Gaussian** field each is $\chi^2_2$, so the band-power's skewness is
$\sqrt{8/N_{\text{modes}}}$ and **falls** as you go to smaller scales (more modes). The measured
band-powers do the opposite: their skewness **grows** with $k$ and is so heavy-tailed that the
sample estimate is dominated by the single largest realization (we measure values scattering from
$\sim 3$ to $\gtrsim 10$ at fixed parameters). A statistic whose skew *grows* with $k$ at large
$N_{\text{modes}}$ cannot be reflecting the estimator's $\chi^2$ — it is reflecting the **field's
own non-Gaussianity**: the BM19 power-law tail puts rare, dense clumps into the map, and those
clumps dominate the small-scale power (a large connected trispectrum). A Gaussian likelihood on
these raw band-powers is therefore *structurally* mis-specified, and no covariance or mean
correction repairs a wrong distribution shape.

The textbook analytic rescues assume *Gaussian-field* (Wishart) band-power statistics —
Hamimeche–Lewis-style variable transforms, or a lognormal/log likelihood. Because the offending
non-Gaussianity is in the *field*, not the estimator, they under-correct exactly where it matters
(the high-$k$ tail): we measure the Hamimeche–Lewis transform leaving per-bin skew $0.8\!-\!1.7$,
no better than a plain log. The fix cannot be a cleverer *likelihood* for the raw statistic; it must
be a better *observable*.

## The right observable: the log₊ map

The information about $\beta$, and the Gaussianity, both live in the **log-density**. In $s$-space
the band-power slope tracks $\beta$ almost perfectly; the exponentiation $\rho=e^{s}$ is what
compresses the slope *and* manufactures the heavy tail. So the observable should *undo* the
exponentiation. The {cite:t}`Neyrinck2009` "$\log_+$" transform does exactly that on a count map:

```{math}
:label: betaproj-logplus
\mathcal{A}(\mathbf{x}) \;=\; \log_+\!\bigl(N(\mathbf{x})\bigr) \;=\;
\begin{cases} \ln\!\bigl(N/\bar N\bigr), & N>\bar N,\\[2pt] N/\bar N - 1, & N\le \bar N,\end{cases}
```

a count-safe logarithm ($\bar N$ the mean count). Measuring its band-powers, we find the per-bin
skewness driven to $\approx 0$ across all but the lowest $k$ — a *bona fide* Gaussian-likelihood
target. Two properties make $\log_+$ decisively better than rank-Gaussianization (the transform the
earlier attempt used):

1. **It is deterministic and differentiable** — a fixed function of $N$ — whereas a rank map is a
   sort (non-differentiable, sample-dependent).
2. **Its forward-model transfer is $\beta$-stable** (below), so the $\beta$-response can stay
   analytic; the rank transfer is strongly $\beta$-dependent, which forces a fitted, noisy
   $\beta$-response — the precise failure mode that mis-calibrated the earlier estimator.

## The analytic forward model

We predict the *mean* log₊ band-power as a smooth, differentiable function of $\theta=(\beta,
\mathcal{M},\dots)$ and fit it — the cosmology playbook (predict the statistic, never
backpropagate the simulator; see [](differentiable-inference.md)). The clustering backbone is the
analytic **projected log-density 2-point**: take the log-density Mehler series
{cite:p}`SzapudiPan2004,ColesJones1991`

```{math}
:label: betaproj-xis
\xi_s(r) \;=\; \sum_{n\ge 1} \frac{c_n^2}{n!}\,\rho_g(r)^{\,n},
\qquad c_n=\langle\, s(g)\,\mathrm{He}_n(g)\,\rangle,
```

with $\rho_g(r;\beta)$ the normalized Gaussian correlation of $P(k)=k^{-\beta}$ and $\mathrm{He}_n$
the probabilists' Hermite polynomials, then project it along the line of sight (a discrete Limber
sum) and bin in $|\mathbf{k}|$. Call this $A_s(k;\beta,\mathcal M)$. It is exact for the *field*:
measured against simulations it reproduces the projected-log-density slope to better than 1 % across
the whole $\beta$ prior.

The observable, though, is the log₊ of *shot-noised, projected* counts, not the log-density field.
At **high stellar density** the two coincide up to a $\beta$-independent per-bin transfer
$T(k)$, calibrated once at a fiducial $\theta_{\text{fid}}$:

```{math}
:label: betaproj-mu-highN
\mu(k;\beta) \;=\; A_s(k;\beta,\mathcal M_{\text{fid}})\;\times\;T_{\text{fid}}(k).
```

Because $A_s$ carries the $\beta$-response and $T$ is a *constant*, the slope information is never
fitted — and because $A_s$ is a **smooth, deterministic** function of $\beta$, we may tabulate it
and interpolate inside the sampler without injecting any noise (the opposite of interpolating a
Monte-Carlo simulation mean, which corrupts the very slope we are measuring). With a fixed-fiducial
Hartlap-corrected covariance and a logit-reparametrized NUTS sampler, this passes
simulation-based calibration {cite:p}`Talts2018`: single-cluster β rank-uniformity $p=0.82$, with
the recovered $\sigma(\beta)\approx0.084$ per cluster set by cosmic variance (it tightens only by
stacking, not by adding stars).

## The analytic shot transfer (low stellar density)

At realistic, sparser densities the simple fixed transfer breaks: Poisson noise suppresses the
$\beta$-response by a $\beta$-*dependent* amount (we measure the transfer's variation across $\beta$
growing from $\sim5\%$ to $15\!-\!46\%$ as $\bar N$ drops). Fitting that $\beta$-dependence from
simulations would re-introduce the noisy-slope trap. Instead we model the shot **analytically**.

Condition on the gas field. Two pixels' counts are sums over *disjoint* sets of independent Poisson
cells, so given the field they are independent. The autocovariance of $\mathcal A=\log_+(N)$ then
splits exactly into a clustering piece and a single zero-lag (white) piece:

```{math}
:label: betaproj-split
P_{\mathcal A}(k) \;=\; \underbrace{P_{\text{clust}}(k)}_{\text{band-power of }m(\Sigma)}
\;+\; \underbrace{W_{\text{shot}}}_{\text{$k$-independent}} .
```

The two ingredients are deterministic functions of the **compound-Poisson** model
($N\,|\,\Sigma \sim \mathrm{Poisson}(\bar n_{3d}\Sigma)$, with $\bar n_{3d}$ the *known* mean stars
per cell):

```{math}
:label: betaproj-m
m(\Sigma) \;=\; \mathbb{E}\!\left[\log_+ N \,\middle|\, \Sigma\right]
            \;=\; \sum_{N\ge 0}\log_+(N)\,\mathrm{Poisson}(N;\bar n_{3d}\Sigma),
```
```{math}
:label: betaproj-wshot
W_{\text{shot}} \;=\; \mathbb{E}_{\Sigma}\!\left[\mathrm{Var}\!\left(\log_+ N \,\middle|\,
\Sigma\right)\right].
```

Equation {eq}`betaproj-m` is the **Poisson smoothing** of the log: at large $\bar n_{3d}\Sigma$ it
returns $\ln(\Sigma/\bar\Sigma)$ (the noiseless limit, slope $\to\beta$); at small counts it bends
over — that bend *is* the shot suppression, now written in closed form. Equation
{eq}`betaproj-wshot` is the familiar white shot floor, here the conditional variance of $\log_+$.

To get $P_{\text{clust}}(k)$ we need the 2-point of the field $m(\Sigma(\mathbf x))$. Since $m$ is a
pointwise transform of the projected density $\Sigma$, the **same Mehler machinery** applies — we
just feed it the new map:

```{math}
:label: betaproj-clust
P_{\text{clust}}(k) \;=\; \mathrm{FFT}_{2\mathrm D}\!\Bigl[\;\sum_{n\ge1}\frac{a_n^2}{n!}\,
\rho_\Sigma(r_\perp)^{\,n}\Bigr],\qquad a_n=\langle\, m(\Sigma(g))\,\mathrm{He}_n(g)\,\rangle,
```

where $\rho_\Sigma(r_\perp)=\xi_\Sigma/\xi_\Sigma(0)$ is the normalized projected-density
correlation ($\xi_\Sigma=\mathrm{Limber}[\xi_\rho]$, analytic in $\beta,\mathcal M$). The single
modelling approximation is the **marginal** of $\Sigma$ used to build the map $\Sigma(g)$ and the
expectations {eq}`betaproj-m`–{eq}`betaproj-wshot`: a line-of-sight sum of correlated lognormals is
itself *approximately* lognormal {cite:p}`ColesJones1991`, matched to the analytic mean and variance
of $\Sigma$. Everything else is exact.

The payoff: $\mu(k;\beta)=P_{\text{clust}}(k)+W_{\text{shot}}$ is **fully analytic and
differentiable in $\beta$ at any stellar density** — the shot's $\beta$-dependence comes from the
physics (Poisson + the $\beta$-dependent $\Sigma$ statistics), never from a fitted surface. At high
$\bar n_{3d}$ it reduces to {eq}`betaproj-mu-highN`; at low $\bar n_{3d}$ it bends correctly. Whether
the lognormal-marginal approximation is accurate *enough* is exactly what SBC at low density decides
— the honest, falsifiable test rather than a tuned fudge.

## What this buys, and what it costs

This is the differentiable, SBC-calibrated $Q$/MST successor for $\beta$, built from analytic
predicted statistics rather than a learned simulator surrogate {cite:p}`Bairagi2026` — interpretable
and gradient-friendly by construction. The honest scope is set by physics, not method: $\sigma(\beta)
\approx 0.2$ per cluster is **cosmic-variance limited** (one cloud is one realization of a random
field), improving as $\sim 1/\sqrt{K}$ only by **stacking** a population of $K$ clusters. The
amplitude $\mathcal M$ is forecast-grade; the tail slope $\alpha$ is depth-gated in pure 2-D
projection (line-of-sight averaging Gaussianizes the 1-point tail). Those are statements about the
*information in a projected map*, and they bound any method — analytic or learned — equally.
```
