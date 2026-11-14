---
title: Recovering a cluster’s IMF and equipartition
description: "Recover IMF slope, equipartition and concentration from a synthetic cluster’s component counts and kinematics using a differentiable model and BlackJAX."
---

# Recovering a cluster’s IMF and equipartition

Can we recover the IMF slope and dynamical parameters of a **four-component
multimass model** from one synthetic cluster? This example fits the component
counts and velocity dispersions of the model it generates. It demonstrates
parameter recovery within that discrete model. It does not establish recovery
from observed clusters or from a continuously sampled stellar-mass population.

```{note} What you will learn
- How a binned IMF sets a cluster’s mass-component populations.
- How component counts and velocity dispersions constrain the IMF and internal dynamics.
- What this idealized synthetic-data example can—and cannot—tell you about a real cluster.
```

## The synthetic cluster

The example contains 100,000 stars in four logarithmically spaced mass bins.
It uses a Maschberger IMF with high-mass slope $\alpha=2.3$, characteristic
mass $m_c=0.2\,M_\odot$, and low-mass shape parameter $\beta=1.4$.
Among the IMF shape parameters, only $\alpha$ is fitted. The dynamical
parameters controlling mass-dependent velocities and concentration are fitted
alongside it, as described below.

| Component | IMF mass interval [$M_\odot$] | Representative mass [$M_\odot$] | Stars in this catalogue |
|---|---:|---:|---:|
| 0 | 0.100–0.376 | 0.197 | 65,349 |
| 1 | 0.376–1.414 | 0.679 | 27,287 |
| 2 | 1.414–5.318 | 2.468 | 6,211 |
| 3 | 5.318–20.000 | 9.214 | 1,153 |

Bin boundaries and representative masses are rounded in the table. Within each
bin, the representative mass is the IMF-weighted mean: the total IMF mass in
that interval divided by its expected number of stars.

```{important} What a star’s mass means here
The mass bins affect the **returned stellar catalogue**, not just the internal
equilibrium calculation. Every star in a component receives that component’s
representative mass.
There is **no additional continuous mass draw within a bin**. For example,
all 1,153 stars in component 3 have mass approximately $9.214\,M_\odot$ in
this catalogue. This is a four-component approximation to the IMF, not a
catalogue of individually sampled continuous stellar masses that was binned afterward.

These representative masses determine the equilibrium and velocity normalization.
The fit uses component counts and velocities; it does not treat the representative
masses as additional exact mass measurements. Candidate IMF slopes change the
representative masses while the bin boundaries stay fixed.
```

:::{figure} figures/imf_equip_cluster.png
:label: sci-b2-cluster
:width: 100%
:alt: Left, projected star counts in the synthetic cluster. Right, cumulative projected radial distributions of its four mass components.

**The synthetic cluster.** Left: all 100,000 stars projected onto the $x$–$y$
plane, with logarithmic colour showing stars per hexagon. Right: the fraction
of each component enclosed within projected radius $R=\sqrt{x^2+y^2}$.
Distances are divided by the model’s core scale $r_c$. A curve rising at
smaller radius indicates a more centrally concentrated component. These are
empirical distributions of the same catalogue used below, not fitted curves
or uncertainty bands. Line colours and styles distinguish the four mass bins.
:::

## The self-consistent physics

A multimass LIMEPY cluster {cite:p}`Gieles2015` is built from an
[initial mass function](../10-theory/imfs/index.md). progenax bins the IMF into
$J=4$ mass groups; each group $j$ has a representative mass $m_j$ and a population
weight set by the IMF. The equilibrium model represents mass-dependent
kinematics through the component velocity scales

```{math}
:label: b2-equipartition
\mu_j = \frac{m_j}{\bar m},
\qquad
s_j = s\,\mu_j^{-\delta}.
```

Here $\bar m$ is the central-density-weighted mean stellar mass, $\mu_j$ is
a dimensionless mass ratio, $s$ is the common velocity normalization, and
$\delta$ controls how strongly the velocity scale decreases with stellar mass.
At $\delta=0$ the scales are equal; at $\delta=1/2$, $m_j s_j^2$ is constant.
The actual velocity dispersions also depend on the truncated distribution
function, so this scale relation alone does not imply exact kinetic-energy
equipartition everywhere. The model prescribes an equilibrium; it does not
simulate the relaxation process that produces it. See the
[multimass equipartition theory](../10-theory/spatial-profiles/multimass-equipartition.md)
for the distribution-function interpretation {cite:p}`Gieles2015,Peuten2017`.

**The high-mass IMF slope.** The component populations are obtained by binning a Maschberger (2013) IMF
whose high-mass behaviour is the power law

```{math}
:label: b2-imf
\frac{dN}{dm}\Big|_{m\gg m_c}\ \propto\ m^{-\alpha},
\qquad \alpha_{\rm true} = 2.3\ (\approx\text{Salpeter}),
```

Here $N$ is the expected number of stars and $m$ is stellar mass;
{cite:t}`Maschberger2013` gives the full smooth form. The same $\alpha$ determines the representative masses and population weights
of the kinematic components. This links the observed component counts to their
velocity dispersions through the equilibrium model.

## The joint likelihood

Let $n_j$ be the observed number of stars in component $j$, and let
$p_j(\theta)$ be its normalized sampling probability at parameters
$\theta=(\alpha,\delta,W_0)$. Here $\alpha$ is the IMF high-mass slope,
$\delta$ controls the component velocity scales, and $W_0$ is the dimensionless
central potential. Up to a parameter-independent constant, the count likelihood is

```{math}
:label: b2-joint
\log\mathcal L_{\mathrm{counts}}(\theta)=\sum_j n_j\log p_j(\theta).
```

Integrating the IMF over the four mass intervals sets the target component
populations. A steeper high-mass slope shifts the population toward lighter
components, so the observed counts constrain $\alpha$.

The numerical equilibrium solve matches these targets to finite precision.
The likelihood therefore uses the solved model’s actual component number
fractions, normalized exactly as in the categorical sampler, rather than
substituting ideal bin integrals. No continuous
IMF density is evaluated on representative masses. Mass-component boundaries are
fixed; their representative masses $m_j(\theta)$ vary with the IMF parameters.

Conditional on the observed counts, the total mass used to predict the velocity
scale is $M(\theta)=\sum_j n_j m_j(\theta)$. Thus the velocity normalization changes
with the candidate IMF rather than being fixed to its generating value.
The observed component labels, positions and velocities stay fixed throughout
the fit: each trial rebuilds the model, without resampling or relabelling the
observations. $M(\theta)$ is a model prediction conditional on the observed
counts; the catalogue’s individual mass entries are not an additional fitted
data channel.

For isotropic velocities, the measured one-dimensional dispersion is
$\hat\sigma=\sqrt{\langle |\mathbf v|^2\rangle/3}$, where $\mathbf v$ is
a star’s velocity in the known cluster rest frame. The kinematic term compares
each populated radial bin’s measured
one-dimensional dispersion $\hat\sigma_{jk}$ with its density-weighted model
prediction $\sigma^{\mathrm{pred}}_{jk}(\theta)$. Here $k$ labels a radial bin, $\mathrm{SE}_{jk}$ is its estimated standard error,
and $w_{jk}$ is one for cells with at least 30 stars and zero otherwise. Then the working log likelihood is

```{math}
\log\mathcal L(\theta)=\log\mathcal L_{\mathrm{counts}}(\theta)
-\frac12\sum_{j,k}w_{jk}
\left[\frac{\hat\sigma_{jk}-\sigma^{\mathrm{pred}}_{jk}(\theta)}
{\mathrm{SE}_{jk}}\right]^2.
```

```{warning} Approximate conditional kinematics
This is a summary likelihood, not the full phase-space likelihood. It uses
a Gaussian-velocity approximation for the dispersion errors and neglects
correlations between summaries. The truncated model is not exactly Gaussian.
Radial bins are selected from the catalogue and held fixed during fitting.
Repeated-catalogue recovery and uncertainty coverage remain to be checked.
The generating centre and rest frame are assumed known; no fitted recentring
is applied. Individual representative masses are not additional observations.
```

## Inputs and execution

The example uses 100,000 stars, four mass components over
$[0.1,20]\,M_\odot$, generating parameters $(\alpha,\delta,W_0)=(2.3,0.4,5)$,
isotropic velocities and truncation order $g=1$. The radial summaries use
16 equal-count bins and retain cells with at least 30 stars. Numerical model
units use $G=1$ consistently in sampling and prediction.

The fit bounds are $1.5<\alpha<3.2$, $0<\delta<0.7$ and $3<W_0<8$.
These are the demo's inference bounds, not a universal physical stability limit.
BlackJAX samples a density uniform in these physical parameters before applying
the likelihood; the transformation Jacobian is included. Posterior runs are
opt-in and use four independent BlackJAX chains, saving draws and R-hat,
bulk/tail effective sample sizes and divergence diagnostics. Local information estimates are not calibrated credible intervals.

## Single-catalogue recovery

The fit and figure values below retain the September 28, 2026 single-catalogue
exposure (random seed 0, 100,000 stars), recorded in
`docs/reviews/recovery-artifacts/single-catalogue-record.json`. They were
not regenerated for this release audit. The final full driver and its
four-chain posterior path have not been rerun; selected objective checks
and the retained fit are separate evidence. The best retained fit is:

| Parameter | True value | Recovered | Local information scale |
|---|---:|---:|---:|
| IMF slope $\alpha$ | 2.3000 | 2.2968 | 0.0041 |
| Equipartition $\delta$ | 0.4000 | 0.3977 | 0.0344 |
| Concentration $W_0$ | 5.0000 | 4.9938 | 0.0212 |

The local information scale estimates how sharply the objective constrains
each parameter near the best fit. The largest offset from the true values is
0.784 of this scale. These estimates are not posterior credible intervals,
and one recovered catalogue does not establish their statistical coverage.
The fit uses three starting points; two reach similar objectives and one remains
higher. The lowest objective determines the reported fit.

:::{figure} figures/imf_equip_single_catalogue_fit.png
:label: sci-b2-single-catalogue-fit
:width: 100%
:alt: Left, velocity dispersions for four mass groups with fitted profiles; right, component counts from the same cluster and fitted expected counts.

**Single-catalogue fit.** Left: binned velocity dispersions and pointwise fitted
profile curves, in model units; the likelihood uses bin-averaged predictions.
Right: observed component counts and fitted expectations. The printed scales
are local information estimates. Both panels use the same 100,000 stars.
:::

## Assumptions and limits

```{important} Equilibrium recovery, not cluster history
This example contains no stellar or dynamical evolution. Its fitted IMF describes
the population used to generate the catalogue. In an evolved cluster, the
present-day mass function differs from the birth IMF because stars lose mass,
become remnants and escape. Recovering a birth IMF requires an explicit model
for those processes; a successful equilibrium fit alone cannot reconstruct it.
```


- The component labels, cluster centre and rest frame are known. No measurement
  errors, missing stars or observational selection are included.
- The fit uses three-dimensional positions and velocities. The projected image
  is an illustration, not the observational data model used by the likelihood.
- Four mass groups approximate a continuous IMF. Increasing their number may
  change the result; mass-bin convergence has not yet been established.
- The Gaussian kinematic summary likelihood and local uncertainty estimates
  require repeated-catalogue checks before interpreting them as calibrated errors.
- The count-and-kinematic objective passes the recorded float64 derivative
  checks. The four-chain BlackJAX path is available, but no posterior result for
  this likelihood is reported here.

### From model components to observations

Finite mass components approximate a population's dynamics; they do not imply
that real stars have only those masses. Observational mass bins and numerical
model components are separate choices. A finite-component model can be compared
with observations by projecting its distributions and integrating over the
selected stellar populations. Continuously sampling a particle catalogue is
not required to predict these observables.

An observational application must include the survey footprint, completeness,
measurement uncertainties and the mapping from photometry to stellar populations.
It must also distinguish visible tracers from remnants that contribute gravity
but little or no light. These ingredients are outside this demo. For an example
combining proper motions, line-of-sight velocities and photometric mass functions,
see [Dickson et al. (2023)](https://doi.org/10.1093/mnras/stad1254).

### Continuous masses and dynamical components

A continuous-mass population generator and a finite-component equilibrium solver
are distinct modeling choices. This example uses discrete masses for both.
It therefore cannot represent within-bin stellar-mass variation for applications
such as individual-star evolution.

An extension could retain continuously sampled stellar masses while using mass
components to approximate their dynamics. That would require a defined rule for
sampling positions and velocities conditional on stellar mass, together with
checks of the resulting density, potential and velocity moments. Assigning new
masses after drawing the present catalogue would not by itself establish a
self-consistent equilibrium. Increasing the number of discrete components is a
resolution check; it does not change the current sampler into a continuous-mass
sampler.

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py --illustrate-only # draw the cluster
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py --check-only # verify derivatives
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py # MLE + local information
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py --run-nuts # opt-in BlackJAX posterior
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery_bias.py       # wrong-IMF curve + grid
```

Use `--seed` to select an independent catalogue and `--n-comp` to change mass-bin
resolution. These options support qualification; changing resolution does not
by itself demonstrate convergence. The default remains four groups and seed 0.

Related: [multimass equilibrium validation](../50-validation/multimass-equilibrium.md)
and [the IMF models](../10-theory/imfs/index.md).
