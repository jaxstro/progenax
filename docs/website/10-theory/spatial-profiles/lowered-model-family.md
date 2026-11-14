---
title: "The differentiable lowered-model family (Engine A)"
description: "progenax's implemented, differentiable generalization of King/Wilson/Woolley into one lowered-isothermal family (continuous truncation parameter g, multi-component velocity scales, per-component anisotropy), following the Gieles & Zocchi (2015) formalism — implemented natively in JAX, not as an external dependency."
---
# The differentiable lowered-model family (Engine A)

:::{admonition} Status — implemented
:class: tip
This family is implemented; its [validation page](../../50-validation/multimass-equilibrium.md)
states the tested configurations and remaining limits. It is "Engine A" of the unified
[`MultiComponentCluster`](../populations/index.md) model: every component is a
lowered-isothermal DF riding one shared self-consistent potential, solved by
the coupled-Poisson core `solve_multicomponent_limepy(alpha_j, rescale_j, W0,
g, ra_hat_j)`. The complementary density-defined route ("Engine B",
[Eddington inversion in a shared potential](../populations/eddington-engine.md))
covers the case where the *densities* are prescribed instead of the DF.
:::

:::{admonition} Who this page is for
:class: note
**Audience:** students & researchers who already know the single-mass King model and want the unified, differentiable lowered-isothermal family (Woolley/King/Wilson + multi-mass + anisotropy).
**Prerequisites:** the [King profile](king.md) (the $g = 1$ member this generalizes) and the [virial-$Q$ convention](../../20-architecture/q-virial-convention.md); the [multi-component populations](../populations/index.md) overview gives the Engine A/B framing.
**You'll get:** the continuous truncation parameter $g$, how $n$ mass components share one self-consistent potential via velocity-scale ratios $w_j$, and why progenax reimplemented LIMEPY natively to make $g$ and the segregation parameters *differentiable*.
:::

## The idea

The classical truncated-isothermal models — Woolley (an abruptly truncated Maxwellian),
{cite:t}`King1966` (a linear lowered Maxwellian), and Wilson (a quadratic
lowered form) — are members of **one** family distinguished by how sharply the
distribution function is truncated near the escape energy. {cite:t}`Gieles2015`
made this explicit with a single continuous **truncation parameter** $g$: their
LIMEPY DF reads

```{math}
:label: limepy-df-family
f(E) \;\propto\; E_\gamma(g,W), \qquad
W = \frac{E_{\rm cut}-E}{\sigma^2}, \qquad W>0,
\qquad f(E)=0\ \text{for}\ W\le0,
```

[↗ model card](#card-limepy-df-family)

Here $E$ is specific energy, $E_{\rm cut}$ the cutoff energy, $\sigma$ a
velocity scale, and $W$ the dimensionless relative binding energy. The
lowered exponential is ({cite:t}`Gieles2015`, Eq. 2)

```{math}
E_\gamma(g,W)=
\begin{cases}
e^W, & g=0,\\
e^W P(g,W), & g>0,
\end{cases}
\qquad P(g,W)=\frac{\gamma(g,W)}{\Gamma(g)}.
```

The regularized lower incomplete gamma function $P$ is used only for $g>0$;
the separate $g=0$ branch is required. The three integer corners are
$e^W$ (Woolley), $e^W-1$ (King), and $e^W-1-W$ (Wilson). The implementation
sets the DF to zero at $W=0$ as well as outside the support; the value at
that single energy does not change the Woolley integrals. The same framework extends
naturally to **multiple mass components** (each mass group its own
$\sigma_j$, sharing one potential) and to **radial anisotropy** (an
Michie angular-momentum factor).

## The coupled multi-component equilibrium

progenax generalizes the single-mass solve to $n$ components sharing one
dimensionless potential $\psi(\xi)$, $\xi = r/r_c$. The model-defining insight
is that a component's **single free Engine-A scale is its velocity-scale
ratio**

```{math}
:label: wj-def
w_j \;=\; \frac{s_j}{s}, \qquad \texttt{rescale}_j \;=\; w_j^{-2},
```

so component $j$ sees the shared potential at the rescaled depth
$W_j(\xi) = \texttt{rescale}_j\,\psi(\xi)$ and contributes a density
$\hat\rho_j \propto E_\gamma\!\big(g+\tfrac32,\, \texttt{rescale}_j\,\psi\big)$
(the "lowered exponential" of {cite:t}`Gieles2015`, Appendix B), giving the
coupled Poisson equation

```{math}
:label: multimass-poisson
\frac{1}{\xi^2}\frac{\mathrm{d}}{\mathrm{d}\xi}
\Big(\xi^2\frac{\mathrm{d}\psi}{\mathrm{d}\xi}\Big)
\;=\; -\,9\sum_j \alpha_j\,\hat\rho_j(\xi),
```

[↗ model card](#card-multimass-poisson)

with $\alpha_j$ the central density fractions ($\sum_j\alpha_j = 1$). A
*colder* component ($w_j < 1$) feels a deeper effective well and concentrates
— as a **genuine equilibrium**, not a post-hoc radial reshuffling. Mass
segregation is then the equipartition convenience

```{math}
:label: mass-seg-law
w_j \;=\; \mu_j^{-\delta}, \qquad \mu_j = m_j/\bar m, \quad
\bar m = \sum_j m_j\,\alpha_j,
```

[↗ model card](#card-mass-seg-law)

({cite:t}`Gieles2015`, Eqs. 24–26): $\delta = 1/2$ is the standard
partial-equipartition choice, $\delta = 0$ collapses every component to the
single-mass model exactly (the cleanest oracle). The representative stellar
masses $m_j$ are otherwise **decoupled labels** — the structure depends only
on $(\alpha_j, w_j, \texttt{ra\_hat}_j, W_0, g)$, so equal-mass populations of
different concentration (GC 1G/2G, halo+core, binaries-vs-singles) set $w_j$
directly.

Per-component radial anisotropy enters as a Michie angular-momentum factor
with its own anisotropy radius $\hat r_{a,j} = r_{a,j}/r_c$
({cite:t}`Michie1963`; {cite:t}`Merritt1985`); the anisotropic density is
evaluated through the selected density-table or direct-quadrature path.
Finite table and quadrature errors have independent comparisons in the
validation owner. This is a Michie-type DF, not the density-preserving
OM construction; their anisotropy laws differ.

:::{admonition} A misprint worth knowing about
:class: note
The density index of the lowered-isothermal family is
$E_\gamma(g+\tfrac32, \hat\phi)$ — printed **correctly** in
{cite:t}`Gieles2015` main-text Eq. 8 (p. 578). What the published 2018
**erratum** (MNRAS 474, 3997) corrects are the $\hat\phi \to 0$
*limiting expressions* Eqs. 20–21 (whose corrected right-hand sides carry
exactly these $E_\gamma(g+\tfrac32,\cdot)$ / $\Gamma(g+\tfrac52)$ indices)
and a typo in the $\sigma_S^2(r)$ Eq. 41 — the LIMEPY code itself was
always correct. progenax verified the index three independent ways
(main-text Eq. 8, the erratum, and the King $g=1$ corner) before
implementing `limepy_density_hat`. (An earlier revision of this note
mislocated the misprint as "main text $g+\tfrac12$ vs Appendix B" —
corrected 2026-07-11 against both PDFs.)
:::


```{figure} ../figures/engine_a_segregation.webp
:label: fig-engine-a-segregation
:width: 100%

Mass segregation as a genuine equilibrium (seed 31, $3\times10^5$ stars,
$m_j = 0.3/1/3\,\mathrm{M_\odot}$). At $\delta = 0$ every component's
density collapses onto the same profile — the single-mass oracle, visually
exact. At $\delta = \frac12$ the heavy component concentrates (and the
whole model restructures: note the smaller $r_t$) while the light component
extends: heavier components feel a DEEPER effective well $W_j = \mu_j^{2\delta} W_0$
— segregation is in the DF, not a positional reshuffle.
Regenerate: `python -m laboratory.icviz --only engine-a-segregation`.
```

## Entry points

The family is exposed through three `MultiComponentCluster` constructors
(see [](../populations/index.md) for the engine map and
[](../populations/two-component.md) for a worked example):

The constructor alternatives below assume a chosen scalar IMF `imf` and a
JAX random `key`; they are interface examples, not a new recovery experiment.
Sampling takes the gravitational constant explicitly.

```python
from progenax import MultiComponentCluster
from jaxstro.units import STELLAR

# General: per-component velocity-scale ratios (GC 1G/2G, halo+core, ...)
model = MultiComponentCluster.from_components(
    alpha_j=[0.5, 0.5], w_j=[0.7, 1.0], m_j=[1.0, 1.0], W0=7.0, g=1.0)

# Mass segregation as equipartition: w_j = mu_j^(-delta)
model = MultiComponentCluster.from_mass_segregation(
    alpha_j=[0.6, 0.4], m_j=[0.3, 3.0], W0=7.0, g=1.0, delta=0.5)

# IMF-driven: bin the IMF, eigenvalue-solve for the alpha_j that
# reproduce the per-bin mass budget
model = MultiComponentCluster.from_imf(imf, n_comp=8, W0=7.0, g=1.0, delta=0.5)

ic = model.sample_cluster(key, n_stars=50_000, G=STELLAR.G)   # ICResult with component_id
```

`sample_cluster` draws every star from its component's lowered DF at the
rescaled potential $W_j(r)$ and velocity scale $s_j = s\,w_j$ — **no external
virial rescale**. The continuum model has component virial relations, while
numerical representations and finite samples require their own checks.
The retained unequal-mass two-population regression reports theory
$Q_j = 0.5 \pm 2\times 10^{-3}$, sampled global $Q$ within $\pm 0.04$ and
per-component $Q_j$ within $\pm 0.07$ at test resolution. The single-mass
corners were validated against the released models directly: $g=1$ isotropic
$\equiv$ [King](king.md), $g=1$ anisotropic $\equiv$
[Michie](../velocity-dfs/michie-king.md), with the retained unscaled energy-ratio checks.

## Check yourself

:::{dropdown} 1. Why does $r_t$ shrink when segregation turns on?
Compare the two panels' radial extents in {numref}`fig-engine-a-segregation`.
Turning on $\delta$ makes the heavy component colder ($w_j < 1$), deepening
the central potential it generates — at fixed $W_0$ the whole model becomes
more concentrated and its dimensionless truncation moves inward.
:::

:::{dropdown} 2. Verify the $\delta = 0$ degeneracy numerically
Sample `from_mass_segregation(..., delta=0.0)` and run a two-sample KS test
between the radii of any two components — it should be consistent with one
distribution (the components differ only by their decoupled mass labels).
:::


## DF tables and radial sampling

Density and conditional-speed tables accelerate the coupled solve and sampling.
The density table uses the dimensionless relative potential $W$ and radius ratio
$p=r/r_a$, with coordinates $\sqrt{W}$ and $\log(1+p^2W)$. The speed tables
integrate each speed interval with interior Gauss–Legendre nodes. Direct density
quadrature remains selectable as an independent comparison path.

### Potential and radial sampling

The tidal radius $r_t$ is the first zero of the relative potential. A monotone
cubic interpolant of the unclipped ODE potential supplies both radial-density
integration and conditional velocity sampling. Keeping that same potential
representation preserves the zero at the tidal boundary. The public potential
array remains nonnegative.

The radial grid retains saved potential boundaries, subdivides each interval
into 16 cells, and adds the requested minimum radial resolution. Component mass
CDFs integrate a piecewise-linear radial probability density; sampling inverts
that same integrated density. Repeated boundary nodes have zero width and mass.
Refining the radial grid controls integration error; it does not improve an
under-resolved ODE solution.

:::{seealso} Accuracy and differentiation
The [Engine A validation page](../../50-validation/multimass-equilibrium.md)
owns current table resolutions, comparisons, acceptance budgets and measured
results. [Differentiability rules](../../20-architecture/differentiability.md#engine-a-ad)
explain adjoint selection and the distinction between fixed-realization and
population-expectation gradients. Current performance scaling remains unqualified.
:::

## Why progenax reimplemented it (rather than depend on LIMEPY)

{cite:t}`Gieles2015` is the reference *formalism*, and the published `limepy`
code is the standard numpy/scipy implementation. progenax does **not** wrap or
depend on it, for one decisive reason: **`limepy` is not differentiable.** The
entire progenax thesis is JAX-native, end-to-end-differentiable initial
conditions, so that structural parameters can be **inferred** from data by
gradient descent or HMC (see [](../../20-architecture/differentiability.md)).

Differentiation through construction and sampling supports parameter inference
where the selected observable and numerical path have been qualified. A finite
gradient alone does not establish accuracy, and fixed-random-draw derivatives do
not include changes in discrete component probabilities. See the
[bounded derivative checks](../../50-validation/multimass-equilibrium.md#engine-a-matrix)
for the current end-to-end evidence and limitations.

## Delivered scope

```{list-table}
:header-rows: 1

* - Capability
  - Status
  - Measured
* - King ($g=1$), single mass
  - ✅ released ([King](king.md))
  - $c(W_0)$ vs King (1966) Table II, $\Delta \le 0.02$
* - Continuous $g$ (Woolley $\to$ King $\to$ Wilson)
  - ✅ delivered (`solve_multicomponent_limepy`, differentiable in $g$)
  - $g=1$ corner $\equiv$ King (iso) / Michie (aniso), $Q = 0.5$ unscaled
* - Multi-mass / multi-component equilibrium (per-component $s_j = s\,w_j$)
  - ✅ delivered (`from_components` / `from_mass_segregation` / `from_imf`)
  - theory $Q_j = 0.5 \pm 2\times 10^{-3}$; sampled global $\pm 0.04$,
    per-component $\pm 0.07$
* - Radial anisotropy per component ($\hat r_{a,j}$, Michie-type)
  - ✅ delivered
  - sampled $\beta_j(r) \equiv$ DF quadrature $\beta$ within $0.06$
* - DF-table acceleration (oracle-backed)
  - Implemented; accuracy checks retained
  - Current speed and memory scaling remain unqualified
* - Differentiable structural inference $(W_0, g, w_j, \delta, r_a)$
  - Bounded first-derivative checks
  - See the [qualification matrix](../../50-validation/multimass-equilibrium.md#engine-a-matrix); full-domain inference is not qualified
```

## Relationship to the other roadmap item

This is independent of, but complementary to, the now-**resolved differentiable
tidal radius** $\partial r_t/\partial W_0$
([](../../20-architecture/differentiability.md#roadmap-differentiable-rt)): the
scalar $r_t$ already carries a implicit gradient of the numerical interpolant through the unclamped
$\psi=0$ crossing. In the unified family $r_t$ is a function of $(g, W_0)$; the
same implicit-function-theorem treatment of the $\psi=0$ crossing carries over
to the $g$ generalization unchanged.

## Implementation, validation & references

- **In code:** `src/progenax/cluster/multicomponent.py`
  (`MultiComponentCluster` and the coupled-Poisson core
  `solve_multicomponent_limepy`), with the lowered-density kernel in
  `src/progenax/profiles/limepy_multimass.py` and the DF-table layer in
  `src/progenax/profiles/limepy_tables.py`. See the
  [`MultiComponentCluster` API](../../30-api/cluster.md).
- **Validated in:** [multimass equilibrium](../../50-validation/multimass-equilibrium.md)
  — per-component $Q_j$, the single-mass King/Michie corners, and the
  AD-vs-FD gradient checks for $(W_0, g, w_j, \delta, r_a)$.
- **Primary sources:** {cite:t}`Gieles2015` (the LIMEPY lowered-model
  formalism); single-model members are {cite:t}`King1966` (with Woolley
  1954 / Wilson 1975 for the $g = 0, 2$ endpoints), and anisotropy
  follows {cite:t}`Michie1963` and {cite:t}`Merritt1985`. Full notes in
  the bibliography:
  [Gieles & Zocchi 2015](../../99-bibliography/per-paper/gieles-zocchi-2015.md)
  and [King 1966](../../99-bibliography/per-paper/king-1966.md).
