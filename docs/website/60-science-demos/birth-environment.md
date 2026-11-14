---
title: Birth-environment archaeology (B5)
description: "A historical synthetic birth-mass inference demo: only the high-mass slope responds to the environment, so the mass likelihood constrains at most one combination of metallicity, cluster mass and star-formation efficiency."
---

# Birth-environment archaeology (B5)

The environment-dependent IMF {cite:p}`Marks2012,Jerabkova2018` predicts that
**metal-poor, massive clusters form top-heavy** — a shallower high-mass slope
$\alpha_3$. This demo asks the inverse question: given an unevolved synthetic
birth-mass catalogue, *what about the cluster's birth conditions can you recover?*
It assumes a complete census with exact masses, without stellar evolution or selection.

```{note}
The numerical results and figure below are the historical 2026-06-12 CPU/float64
run recorded in `scripts/demo_birth_environment.py`. They have not been rerun
or qualified for optimizer convergence and repeated-catalogue coverage here.
The structural rank argument follows from the stated forward model; the
reported fit uncertainties use observed likelihood curvature.
```

## The forward model

`BirthEnvironment` carries three birth parameters — metallicity $[\mathrm{Fe/H}]$,
embedded-cluster mass $\log_{10}(M_{\rm ecl}/M_\odot)$, and star-formation efficiency
$\mathrm{sfe}=M_{\rm ecl}/M_{\rm cl}$ — and `env_to_imf_params` (the Jerabkova generalized relation) maps
them to the high-mass IMF slope $\alpha_3$, holding the low-mass slopes at their
canonical Kroupa {cite:p}`Kroupa2001` values. The truth here is a metal-poor,
massive cluster:

```{math}
:label: b5-truth
[\mathrm{Fe/H}] = -1.5,\quad \log M_{\rm ecl} = 6.5,\quad \mathrm{sfe}=0.3
\;\xrightarrow{\;\text{Jerabkova}\;}\; \alpha_3 = 1.625
```

— strongly **top-heavy** versus the canonical $\alpha_3=2.3$. We sample
$N=10^5$ stars from this IMF and ask what they reveal.

## Inputs and assumptions

The fit recovers **one parameter**, the high-mass IMF slope $\alpha_3$. The three
*birth-environment* axes are the scientific quantities of interest, but — as the
next sections show — they are **not** individually recoverable; they are held at
truth and enter only through $\alpha_3$.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b5-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $\alpha_3$
  - High-mass IMF slope — **the one observable**, read from the masses via $\sum_i\log p(m_i\mid\alpha_3)$ (`A3_BOX=(1.0,3.0)`).
  - **recovered** (truth $\approx1.62$)
* - $[\mathrm{Fe/H}]$, $\log M_{\rm ecl}$, sfe
  - The three `BirthEnvironment` axes that map to $\alpha_3$ via `env_to_imf_params`; gradients $(0.057,-0.248,+0.588)$ — $\log M_{\rm ecl}$ is the strongest lever.
  - known / fixed at truth (`-1.5, 6.5, 0.3`); **degenerate** (see below)
* - $\alpha_3^{\rm canon}$
  - Canonical (Kroupa) slope the top-heavy detection is measured against (sets $\Delta\alpha_3$ for the forecast).
  - known / fixed (`ALPHA3_CANON=2.3`)
* - IMF (low-mass)
  - Four-segment broken-power-law slopes/bounds in `IMFParams`, sampled by `sample_masses_from_params`; `env_to_imf_params` varies **only** $\alpha_3$.
  - known / fixed
* - $N_\star$
  - Number of sampled stars; sets the Fisher normalization and the $N^{-1/2}$ precision floor.
  - known / fixed ($10^5$)
* - `N_ADAM`, `n_emp`, `SEED`, `COND_GATE`, `RECOVERY_NSIG`
  - Adam steps (400), empirical-CRLB draw size ($10^4$), seeds, the rank-deficiency condition gate ($10^8$), recovery pull gate ($3\sigma$).
  - numerical choices
```

:::{important}
:label: imp-b5-degeneracy
**With only $\alpha_3$ varying, the masses constrain at most one local combination
of the three birth parameters.** Write $\boldsymbol\theta$ for those parameters
and $I_{\alpha_3}=\mathbb E[(\partial_{\alpha_3}\log p)^2]$ for the scalar
information of the catalogue likelihood $p$. Then
```{math}
\mathcal F_{\rm env}
= I_{\alpha_3}\,(\nabla_{\boldsymbol\theta}\alpha_3)
                  (\nabla_{\boldsymbol\theta}\alpha_3)^\top,
\qquad \operatorname{rank}(\mathcal F_{\rm env})\le1.
```
The rank is one in an informative active interior, with two zero eigenvalues;
it is zero where the mapping is locally constant. This is structural: more
stars shrink $\sigma_{\alpha_3}$ but the two flat directions stay flat at any $N$.
Externally fixing the other two parameters is one way to recover the remaining one
— the same role that an externally-calibrated $\epsilon$ plays in B12. Note the
gradient magnitudes (and which axis is the weak one) are a *local* linearization at
the fiducial point; the Jeřábková relation is nonlinear.
:::

## What you CAN read off the masses: $\alpha_3$

A direct maximum-likelihood fit of the high-mass slope (the per-star IMF
log-likelihood $\sum_i \log p(m_i\mid\alpha_3)$, differentiable in $\alpha_3$)
recovers it cleanly:

```{list-table}
:header-rows: 1

* - quantity
  - truth
  - recovered
* - $\alpha_3$
  - $1.6247$
  - $1.6249 \pm 0.0054$  (pull $+0.04\sigma$)
```

The script constructs an $N^{-1/2}$ forecast by dividing the fitted observed
curvature by $N$, then rescaling it to other sample sizes. Its 12 independent
$N=10^4$-star refits give a recorded scatter $0.0144$ against the curvature-based
forecast $0.0170$ (ratio $0.85$). This is a bounded consistency exposure, not an
independently derived expected-information Cramér–Rao bound or a calibrated
coverage measurement. Under the complete-census model the historical forecast
gives a $\sim40\sigma$ separation from the canonical slope at $N=10^4$.

## What you CANNOT: the birth environment

The map $(\,[\mathrm{Fe/H}],\,\log M_{\rm ecl},\,\mathrm{sfe})\to\alpha_3$ is
many-to-one. In the active interior the masses constrain one local combination — the gradient
direction

```{math}
:label: b5-grad
\frac{\partial\alpha_3}{\partial([\mathrm{Fe/H}],\,\log M_{\rm ecl},\,\mathrm{sfe})}
= (0.057,\; -0.248,\; 0.588),
```

The script illustrates this with the curvature-based outer product $\,\mathcal F_{\rm env} =
(\nabla_{\rm env}\alpha_3)(\nabla_{\rm env}\alpha_3)^\top/\sigma_{\alpha_3}^2\,$ is
**rank 1**. Its eigenvalues come out

```{math}
:label: b5-eig
\lambda(\mathcal F_{\rm env}) = (-1.8\times10^{-12},\;\; 1.8\times10^{-12},\;\;
1.4\times10^{4}),
```

— two of them numerical zero relative to the nonzero eigenvalue. The reported
condition number $\sim10^{304}$ uses the script's $10^{-300}$ denominator floor;
the exact rank-deficient matrix has infinite condition number. There are
**two flat directions**: a continuum of metallicities, cluster masses, and
efficiencies all produce the *same* $\alpha_3$ and hence the *same* mass spectrum.
Recovering the full environment needs two independent additional constraints
that resolve those directions. Fixing the other two parameters is a sufficient
choice when the remaining dependence is invertible locally.

This is the honest scientific message: *the IMF slope is an observable; the birth
environment is an inference with two irreducible degeneracies.*

## Interpretation — why the degeneracy is fundamental

**It is the structure of the map, not measurement noise.** The two zero eigenvalues
in {eq}`b5-eig` come from the scalar map itself, so they do not shrink with
better data. More stars *do* tighten $\sigma_{\alpha_3}$ — the one well-measured
eigenvalue (the $1.4\times10^{4}$) grows $\propto N$ — but the two machine-precision
zeros **stay zero at any sample size**. This is qualitatively different from a "we
need a bigger survey" degeneracy: no amount of data recovers the birth environment
from the masses alone. That is what makes it a *floor* rather than a *forecast*.

**Reading the one direction you do constrain.** The gradient {eq}`b5-grad` is the
single combination of birth parameters that the mass spectrum measures, and its
structure is physical. The signs match the environment-dependent IMF:
$\partial\alpha_3/\partial[\mathrm{Fe/H}] > 0$ (metal-poor $\Rightarrow$ smaller
$\alpha_3$ $\Rightarrow$ top-heavier) and $\partial\alpha_3/\partial\log M_{\rm ecl} < 0$
(more massive $\Rightarrow$ top-heavier). And per a *natural* variation of each
parameter ($\Delta[\mathrm{Fe/H}]\sim1$ dex, $\Delta\log M_{\rm ecl}\sim1$ dex,
$\Delta\mathrm{sfe}\sim0.3$), the embedded-cluster mass and the star-formation
efficiency are the **strong** levers and the metallicity the **weak** one — so in
this regime the high-mass IMF slope encodes cluster mass and efficiency far more
than metallicity.

**The implication for IMF archaeology.** This is a caution for the IGIMF /
top-heavy-IMF program. Inferring a cluster's birth conditions from the assumed unevolved
mass census reads a **one-dimensional** quantity ($\alpha_3$) and projects it back
onto a **three-dimensional** birth-parameter space — a non-invertible projection. The
honest statement is *"the IMF is top-heavy, consistent with $\alpha_3 = X$"*, **not**
*"therefore it formed at metallicity $Y$, embedded-cluster mass $Z$, and efficiency
$W$."* Breaking the degeneracy requires supplying **two of the three independently**
— e.g. a spectroscopic $[\mathrm{Fe/H}]$ *and* a dynamical $M_{\rm ecl}$ — at which
point the third follows from $\alpha_3$.

## Figure

:::{figure} figures/demo_birth_environment.png
:label: sci-birth-environment
:width: 100%

**Birth-environment archaeology** (`scripts/demo_birth_environment.py`, historical
2026-06-12 run; panel gates reported passing in its run record).
**(a)** The sampled high-mass IMF $dN/d\log m$ with the top-heavy truth slope
($\alpha_3=1.62$) and the canonical $2.3$ for reference. **(b)** Forecast: the
curvature-based $\sigma(\alpha_3)\propto N^{-1/2}$ forecast (green, labelled
CRLB in the retained figure) with the 12-refit scatter (black square)
and the $|\Delta\alpha_3|/3$ detection threshold.
**(c)** The environment degeneracy: $\alpha_3$ over the $([\mathrm{Fe/H}],\log
M_{\rm ecl})$ plane (sfe fixed); the recovered-$\alpha_3$ **ridge** (vermilion)
passes through the truth ★, and every point on it is an equally good fit.
:::

## Caveats

```{warning}
- **Mass channel only, complete census.** The likelihood assumes every star is
  observed with a known mass and no selection/incompleteness. The strong $\alpha_3$
  constraint partly comes from the high-mass *fraction* (the normalization), which a
  realistic high-mass-only survey would not deliver; the slope alone is weaker.
- **The "$N\sim60$" number is an unqualified extrapolation.** The script builds
  its $N^{-1/2}$ forecast from one fitted curvature; the independent-refit
  exposure is at $N=10^4$. Small-sample coverage and optimizer convergence
  have not been established by that calculation.
- **Default (high-mass-only) environment dependence.** Only $\alpha_3$ varies with
  the environment here (`include_lowmass_variation=False`); turning on the Marks
  low-mass-slope variation can add a second independent slope direction in
  active interiors. All three slopes still depend on at most metallicity and
  one density combination, so the three-parameter mass-only information has
  rank at most two. Clipping can reduce that rank further.
- **Clean Jerabkova relation.** Truth and fit share the same $\alpha_3(\text{env})$
  map; the demo isolates the *inverse-problem structure*, not relation
  uncertainty.
- **Perfect, complete mass census.** Every star's mass is known exactly — there is
  no mass-measurement error and no $\epsilon$-analogue (contrast B12); the
  $\alpha_3$ uncertainty is the inverse-Hessian (`fisher_cov`) Gaussian
  approximation, and the MLE is seeded at the truth (a convergence check, not a
  multimodality test).
- **The degeneracy figure is a 2-D slice.** Panel (c) plots $\alpha_3$ over
  $([\mathrm{Fe/H}],\log M_{\rm ecl})$ at **fixed** sfe; the visualized ridge is one
  slice of the true 3-D degenerate surface.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_birth_environment.py
```

## References

The environment-dependent IMF relations are {cite:t}`Marks2012` and
{cite:t}`Jerabkova2018`; the canonical low-mass slopes are {cite:t}`Kroupa2001`.
The `BirthEnvironment` + `env_to_imf_params` API is documented on the
[environment-dependent IMF](../10-theory/imfs/environment.md) theory page.
