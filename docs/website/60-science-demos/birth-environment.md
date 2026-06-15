---
title: Birth-environment archaeology (B5)
description: "A paper-seed inference demo on the environment-dependent IMF. Given a present-day stellar mass spectrum, the high-mass slope alpha3 is cleanly recovered (a ~40-sigma top-heavy detection at N=1e4), but the birth environment (metallicity, embedded-cluster mass, star-formation efficiency) is formally UNrecoverable: the environment->alpha3 map is three-to-one, so the environment-space Fisher information is rank 1 (two flat directions). You can read the IMF slope off the masses; you cannot read the birth conditions off the slope."
---

# Birth-environment archaeology (B5)

The environment-dependent IMF {cite:p}`Marks2012,Jerabkova2018` predicts that
**metal-poor, massive clusters form top-heavy** — a shallower high-mass slope
$\alpha_3$. This demo asks the inverse question honestly: given a present-day
stellar mass spectrum, *what about the cluster's birth conditions can you actually
recover?* The answer is a sharp two-part contrast.

## The forward model

`BirthEnvironment` carries three birth parameters — metallicity $[\mathrm{Fe/H}]$,
embedded-cluster mass $\log M_{\rm ecl}$, and star-formation efficiency
$\mathrm{sfe}$ — and `env_to_imf_params` (the Jerabkova generalized relation) maps
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
  - Maschberger low-mass slopes/bounds held canonical; `env_to_imf_params` varies **only** $\alpha_3$.
  - known / fixed
* - $N_\star$
  - Number of sampled stars; sets the Fisher normalization and the $N^{-1/2}$ precision floor.
  - known / fixed ($10^5$)
* - `N_ADAM`, `n_emp`, `SEED`, `COND_GATE`, `RECOVERY_NSIG`
  - Adam steps (400), empirical-CRLB draw size ($10^4$), seeds, the rank-deficiency condition gate ($10^8$), recovery pull gate ($3\sigma$).
  - numerical choices
```

```{important}
:label: imp-b5-degeneracy
**The environment→$\alpha_3$ map is three-to-one, so the masses constrain exactly
one combination of the three birth parameters.** The environment-space Fisher
$\mathcal F_{\rm env}=(\nabla\alpha_3)(\nabla\alpha_3)^\top/\sigma_{\alpha_3}^2$ is
the outer product of a single gradient vector — **rank 1, with two exactly-zero
eigenvalues** (cond $\sim10^{304}$). This is *structural*, not noise-limited: more
stars shrink $\sigma_{\alpha_3}$ but the two flat directions stay flat at any $N$.
Recovering any single birth parameter requires **externally fixing the other two**
— the same role that an externally-calibrated $\epsilon$ plays in B12. Note the
gradient magnitudes (and which axis is the weak one) are a *local* linearization at
the fiducial point; the Jeřábková relation is nonlinear.
```

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

The uncertainty scales as the textbook $\sigma(\alpha_3)\propto N^{-1/2}$, and we
**validate the Cramér–Rao bound empirically**: refitting $\alpha_3$ on independent
$N=10^4$-star draws gives a measured scatter $0.0144$ against the analytic CRLB
$0.0170$ (ratio $0.85$, within the 12-sample noise). At a realistic
$N=10^4$ complete sample the top-heavy slope is a **$\sim 40\sigma$ detection** —
$\alpha_3$ is *easy*.

## What you CANNOT: the birth environment

The map $(\,[\mathrm{Fe/H}],\,\log M_{\rm ecl},\,\mathrm{sfe})\to\alpha_3$ is
**three-to-one**. The masses constrain exactly one combination — the gradient
direction

```{math}
:label: b5-grad
\frac{\partial\alpha_3}{\partial([\mathrm{Fe/H}],\,\log M_{\rm ecl},\,\mathrm{sfe})}
= (0.057,\; -0.248,\; 0.588),
```

so the environment-space Fisher information $\,\mathcal F_{\rm env} =
(\nabla_{\rm env}\alpha_3)(\nabla_{\rm env}\alpha_3)^\top/\sigma_{\alpha_3}^2\,$ is
**rank 1**. Its eigenvalues come out

```{math}
:label: b5-eig
\lambda(\mathcal F_{\rm env}) = (-1.8\times10^{-12},\;\; 1.8\times10^{-12},\;\;
1.4\times10^{4}),
```

— two of them machine-precision zero (condition number $\sim 10^{304}$). There are
**two flat directions**: a continuum of metallicities, cluster masses, and
efficiencies all produce the *same* $\alpha_3$ and hence the *same* mass spectrum.
Recovering any single birth parameter requires an **external constraint** on the
other two (e.g. an independent $\log M_{\rm ecl}$ from the cluster luminosity, or a
spectroscopic $[\mathrm{Fe/H}]$).

This is the honest scientific message: *the IMF slope is an observable; the birth
environment is an inference with two irreducible degeneracies.*

## Interpretation — why the degeneracy is fundamental

**It is the structure of the map, not measurement noise.** The two zero eigenvalues
in {eq}`b5-eig` come from the three-to-one map itself, so they do not shrink with
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
top-heavy-IMF program. Inferring a cluster's birth conditions from its present-day
mass function reads a **one-dimensional** quantity ($\alpha_3$) and projects it back
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

**Birth-environment archaeology** (`scripts/demo_birth_environment.py`, ALL PASS).
**(a)** The sampled high-mass IMF $dN/d\log m$ with the top-heavy truth slope
($\alpha_3=1.62$) and the canonical $2.3$ for reference. **(b)** Forecast: the
analytic CRLB $\sigma(\alpha_3)\propto N^{-1/2}$ (green) with the empirical
validation point (black square) and the $|\Delta\alpha_3|/3$ detection threshold.
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
- **The "$N\sim60$" asymptotic number is optimistic.** It is the small-$N$
  extrapolation of the CRLB; the empirically validated point is $N=10^4$, and the
  small-sample regime does not achieve the bound. The robust statement is the
  $N^{-1/2}$ scaling and the $\sim40\sigma$ detection at $N=10^4$.
- **Default (high-mass-only) environment dependence.** Only $\alpha_3$ varies with
  the environment here (`include_lowmass_variation=False`); turning on the Marks
  low-mass-slope variation would add weak constraints from $\alpha_1,\alpha_2$ but
  not break the leading degeneracy.
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
