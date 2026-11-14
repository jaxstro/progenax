---
title: IMF statistics validation
description: "Validation suite for canonical IMFs (Salpeter, Kroupa, Chabrier, Maschberger): recovered slopes vs published alpha, PDF/CDF coverage with a KS test, mean-mass accuracy against a fine-grid reference, and differentiability of the inference layer."
---
# IMF statistics validation

The released IMF samplers — piecewise power-law ({cite:t}`Salpeter1955`,
{cite:t}`Kroupa2001`), the {cite:t}`Chabrier2003` lognormal+power-law, and the
{cite:t}`Maschberger2013` smooth bridging form — plus `TaperedPowerLaw` and
`Schechter`, exported by the public `progenax.imf` submodule — use inverse-CDF
sampling without rejection. Power laws and Maschberger have analytic inverses;
Chabrier uses fixed Newton iteration, and TaperedPowerLaw/Schechter also use a
numerically integrated CDF. Test file:
`tests/validation/test_imf_physics.py` (see the [test dashboard](test-dashboard.md)
for the live per-suite count); figures:
`scripts/validate_imfs.py` (5 figures, PASS/FAIL per panel, PNG + PDF). Theory at
[](../10-theory/imfs/classic.md).

:::{admonition} Scope of the recorded comparison
:class: important
The retained figures report sampled-slope recovery, distribution checks,
mean-mass accuracy and likelihood derivatives for their stated configurations.
For example, Salpeter's fitted slope is $2.347\pm0.004$ against $2.35$, and
the mean-mass relative difference is at most $1.2\times10^{-4}$ for the six
listed configurations against a 200k-point log-grid integral. Smooth-family
means and Newton inverses are numerical approximations. These comparisons do
not establish exact arithmetic, arbitrary-parameter accuracy or posterior
convergence. The figures and numbers below are retained measurements, not a
fresh campaign.
:::

## What is verified

`test_imf_physics.py` checks PDF slopes, continuity, selected sample properties
and six mean-mass configurations. The sampled-slope fits, histogram deviations,
KS tests and likelihood-gradient panels below belong to
`scripts/validate_imfs.py`; they are not all assertions in that pytest file.
The script uses $N=10^5$ draws per family for slope/PDF panels, with seed 42;
the KS panels use $N=2\times10^4$ and seeds 42, 43 and 44. Here $N$ is the
sample count, $\alpha$ a dimensionless mass-function slope, and KS compares
an empirical cumulative distribution with the implementation's CDF.

```{list-table}
:header-rows: 1

* - Property
  - Criterion or comparison reference
  - Measured
  - Anchor
* - Salpeter slope recovered (sample MLE)
  - $|\Delta\alpha| < 0.05$
  - $2.347\pm0.004$
  - {cite:t}`Salpeter1955` $\alpha=2.35$
* - Kroupa 3 segment slopes recovered
  - $|\Delta\alpha| \leq 0.08$
  - $0.294 / 1.297 / 2.301$
  - {cite:t}`Kroupa2001` $0.3/1.3/2.3$
* - Chabrier high-mass tail slope
  - $|\Delta\alpha| < 0.06$
  - $2.315\pm0.010$
  - {cite:t}`Chabrier2003` $\alpha=2.3$
* - Maschberger asymptotic slope
  - $|\Delta\alpha| < 0.15$
  - $2.358\pm0.158$
  - {cite:t}`Maschberger2013` $\alpha=2.3$
* - PDF coverage (hist vs analytic $m\,\xi$)
  - median rel $< 0.05$
  - $0.018 / 0.020 / 0.019$
  - Chabrier / Kroupa / Maschberger
* - CDF goodness-of-fit (KS)
  - 5% critical reference $D_{\rm crit}=0.0096$; script gates the $p$-value below
  - $0.0049 / 0.0072 / 0.0082$
  - Salpeter / Maschberger / Chabrier
* - KS $p$-value
  - $p > 0.01$
  - $0.71 / 0.26 / 0.13$
  - statistical consistency with the implemented CDF
* - `mean_mass()` vs 200k log-grid ref
  - rel $< 10^{-2}$ (6 IMFs)
  - $\leq 1.2\times10^{-4}$
  - finer quadrature of the same implemented PDF
* - Linear-grid failure mode (adversarial)
  - log grid $\ll$ linear grid
  - $1.2\times10^{-6}$ vs $1.3\times10^{-2}$
  - steep $m^{-2.35}$ low-mass spike
* - Inference-layer gradient (AD vs FD)
  - rel err $< 10^{-3}$
  - $\leq 1\times10^{-5}$
  - `IMFParams` $\partial\,$NLL$/\partial\alpha$
* - Truncated power-law clipping
  - all samples in $[m_{\min}, m_{\max}]$
  - no rejection (inverse CDF)
  - `TruncatedIMF` renormalization
```

The mean-mass pytest gate is relative difference $<10^{-2}$, unchanged here.
The finer reference resolves the integration independently of `mean_mass()`'s
own grid, but shares `logpdf`; it does not independently validate the physical
IMF formula. CDF/PPF round trips, normalization and truncated-support checks
also live in `tests/unit/imf/test_imf_core.py`; numerical smooth-family cases
live in `tests/unit/imf/test_smooth.py`. First-order sampler derivatives have a
separate owner at [](differentiability-audit.md).

:::{note} Provenance
The implemented constants were checked against the four primary papers directly
(per-paper notes: [Salpeter (1955)](../99-bibliography/per-paper/salpeter-1955.md),
[Kroupa (2001)](../99-bibliography/per-paper/kroupa-2001.md),
[Chabrier (2003)](../99-bibliography/per-paper/chabrier-2003.md),
[Maschberger (2013)](../99-bibliography/per-paper/maschberger-2013.md)):
Salpeter $\alpha=2.35$ (Eq. 5, the logarithmic MF $\Rightarrow$ $dN/dm$
slope $2.35$); Kroupa breaks $0.08/0.5$ and slopes $0.3/1.3/2.3$ (Eq. 2); Chabrier
single-star disk $m_c=0.079$, $\sigma=0.69$, high-mass $x=1.3 \Rightarrow \alpha=2.3$
(Table 1, as recorded in the source note, with the $1/(m\ln 10)$ Jacobian to $dN/dm$); Maschberger $\mu=0.2$,
$\alpha=2.3$, $\beta=1.4$ and the closed-form quantile (Table 1). The recovered-slope
figure below checks sampling against those declared constants. Their
physical provenance comes from the separate primary-source comparison;
matching an implementation's own CDF does not independently establish it. See the per-paper notes in
[the bibliography](../99-bibliography/index.md).
:::

## Figures

Generated by `scripts/validate_imfs.py` (PASS/FAIL per panel; PNG + PDF vector).

:::{figure} figures/imf_recovered_slopes.png
:label: val-imf-slopes
:width: 60%
:align: center

**Recovered slopes (headline).** Sampling $10^5$ stars and re-fitting the slope by
bounded-power-law maximum likelihood recovers every published $\alpha$ along the
$y=x$ line: Salpeter $2.35$, the three Kroupa segments $0.3/1.3/2.3$, the Chabrier
high-mass tail $2.3$, and the Maschberger asymptotic $2.3$. Error bars are bootstrap
standard errors — wide for Maschberger because only $\sim120$ stars exceed
$20\,M_\odot$ in the window. The four near-coincident points at the top right all sit
at $\alpha\simeq2.3$–$2.35$.
:::

:::{figure} figures/imf_pdf_overlay.png
:label: val-imf-pdf
:width: 95%
:align: center

**PDF shape coverage.** Sampled histograms (points) overlay the analytic
$m\,\xi(m)$ (solid) across the full mass range for (a) the Chabrier
lognormal+power-law, (b) the Kroupa piecewise power law, and (c) the Maschberger
smooth bridging form — median relative deviation $\leq2\%$ in every case. The
lognormal turnover, the segment breaks at $0.08$ and $0.5\,M_\odot$, and the smooth
peak are all reproduced.
:::

:::{figure} figures/imf_cdf_ks.png
:label: val-imf-cdf
:width: 65%
:align: center

**CDF coverage and the KS test.** Empirical CDFs (points) lie on the analytic
$F(m)$ (solid) for both the analytic-PPF samplers (Salpeter, Maschberger) and the
Newton-PPF sampler (Chabrier). The Kolmogorov–Smirnov statistic is below the $0.05$
critical value $D_{\rm crit}=0.0096$ in all three cases ($p=0.71/0.26/0.13$) — the
samples are statistically consistent with the implemented CDFs at this
exposure, including the iterative Chabrier inverse. A KS test does not
prove an exact distribution or bound every quantile error.
:::

:::{figure} figures/imf_mean_mass_accuracy.png
:label: val-imf-meanmass
:width: 95%
:align: center

**Mean-mass accuracy and the linear-grid failure mode (adversarial).** (a) The
differentiable `mean_mass()` matches a 200k-point log-grid reference to
$\leq1.2\times10^{-4}$ for all six configurations, with differences of
$\sim10^{-10}$ for the closed-form power-law means in the retained comparison.
(b) Why the log grid matters: estimating $\langle m\rangle$ for the steep
Salpeter spike on a **linear** grid stalls at $\sim1\%$ error even at $N=4000$
points, while the log grid converges to $10^{-6}$ in this comparison. The
smooth-family production means use 4000 log-spaced points; power laws use an
analytic mean.
:::

:::{figure} figures/imf_gradient_validation.png
:label: val-imf-gradient
:width: 95%
:align: center

**Inference-layer gradient validation.** For the differentiable `IMFParams`
likelihood, autodiff (line) and central finite difference (circles) of
$\partial\,$NLL$/\partial\alpha$ agree to max rel err $\leq10^{-5}$ for (a) the
high-mass slope $\alpha_3$ and (b) $\alpha_2$. The gradient's zero crossing (star)
is the maximum-likelihood slope, recovering the input ($2.32$, $2.31$ vs the
generating $2.30$), supporting this likelihood's first derivatives at the
plotted points. This comparison does not establish HMC convergence.
:::

## Differentiability

For a fixed random key, inverse-CDF sampling exposes pathwise derivatives of
the implemented quantile map. Analytic power-law and Maschberger inverses,
Chabrier's 30-step Newton solve, and the inherited 20-step smooth-family solve
are different numerical contracts. TaperedPowerLaw and Schechter interpolate a
4000-node log-grid CDF. Endpoints, density joins, singular parameters and moving
hard support require their own checks; neither a fixed loop nor a finite AD
value establishes derivative accuracy.
The `progenax.imf.differentiable` layer (`log_prob_masses`,
`sample_masses_from_params`, `individual_mass_nll` over `IMFParams`) is the
gradient-based inference target validated in @val-imf-gradient.

## How to run

```bash
# tests with explicit float64 in the project environment
JAX_ENABLE_X64=true .venv/bin/python -m pytest tests/validation/test_imf_physics.py -q

# regenerate the five figures with PASS/FAIL tables
JAX_ENABLE_X64=true .venv/bin/python scripts/validate_imfs.py
```

## What this suite does *not* test

- **Binary contamination bias** — the forward model + binary-aware likelihood are
  covered separately at [](binary-imf.md).
- **Environment dependence** — the IGIMF / `BirthEnvironment` mapping is documented
  at [](../10-theory/imfs/environment.md) but is not part of this validation suite.
- **Wider parameter domains and higher derivatives** — these default and selected
  configurations do not qualify arbitrary IMF parameters, all boundaries or
  second/higher derivatives.
- **Posterior convergence** — likelihood-gradient agreement is not an R-hat,
  effective-sample-size or posterior-predictive assessment.

## References

{cite:t}`Salpeter1955`, {cite:t}`Kroupa2001`, {cite:t}`Chabrier2003`,
{cite:t}`Maschberger2013`. Per-paper notes in
[the bibliography](../99-bibliography/index.md); theory at
[](../10-theory/imfs/classic.md).
