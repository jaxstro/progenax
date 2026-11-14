---
title: Multi-component populations
description: "progenax's unified MultiComponentCluster — one differentiable model, two equilibrium engines (DF-defined lowered-isothermal family and density-defined Eddington inversion), every component individually virial in one shared potential with no external rescale."
---

# Multi-component populations

```{seealso}
For single-component clusters built from one IMF + one profile + one
DF, see the page groups [](../spatial-profiles/index.md),
[](../velocity-dfs/index.md), and [](../imfs/index.md). This section
covers the case where a single IC contains *multiple* stellar
populations sharing **one** self-consistent gravitational potential.
```

A **multi-component population** is an IC containing stars from more
than one population, all orbiting in the same potential. Real clusters
routinely need this:

- **Globular-cluster multiple populations (1G/2G).** Many old GCs host
  two chemically distinct populations with different radial
  distributions and kinematics — typically a centrally concentrated 2G
  inside a more extended 1G.
- **Halo + core decompositions.** An observed surface-brightness
  profile decomposed into an extended halo plus a compact core, each
  with its own prescribed density shape.
- **Mass segregation as equilibrium.** Multi-mass clusters in partial
  equipartition: heavier components are dynamically colder and sit
  deeper in the shared well.
- **Binaries vs. singles.** Binary systems behave as a dynamically
  colder, more concentrated component than the single stars.

progenax packages all of these as **one** class —
`MultiComponentCluster` — with **two equilibrium engines** selected by
the constructor:

```{list-table}
:header-rows: 1

* - Engine
  - Components are defined by…
  - Constructors
  - Theory page
* - **A** — DF-defined (lowered-isothermal family)
  - Their distribution functions: each component is a
    {cite:t}`Gieles2015`-family lowered DF with its own velocity-scale
    ratio $w_j$ (and optionally its own anisotropy radius
    $\hat r_{a,j}$), coupled through one Poisson solve.
  - `from_components`, `from_mass_segregation`, `from_imf`
  - [](../spatial-profiles/lowered-model-family.md)
* - **B** — density-defined (Eddington inversion)
  - Their prescribed *densities*: Plummer/EFF/King density shapes with
    mass-fraction amplitudes; the shared potential is one direct
    quadrature pass and each component's DF is recovered by Eddington
    inversion in that shared potential (optionally Osipkov–Merritt
    anisotropic).
  - `from_density_profiles`
  - [](eddington-engine.md)
```

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning how progenax builds a single IC containing multiple stellar populations in one shared self-consistent potential; no prior multi-component-equilibrium literature assumed.
**Prerequisites:** [spatial profiles](../spatial-profiles/index.md) and [velocity DFs](../velocity-dfs/index.md) (a multi-component cluster is the synthesis of both).
**You'll get:** the two equilibrium engines (DF-defined Engine A vs. density-defined Engine B), when to choose which, and the per-component "no rescale" principle ($Q_j = 0.5$ from the DF).
:::

## When to choose which engine

**Choose Engine A when the DF family *is* the model.** The
lowered-isothermal family (Woolley/King/Wilson, continuous truncation
parameter $g$) is the physically motivated description of relaxed,
tidally truncated clusters; the per-component velocity-scale ratios
$w_j$ (mass segregation: $w_j = \mu_j^{-\delta}$) make multi-mass
equipartition a *built-in* equilibrium property. Fitting $g$, $W_0$,
$\delta$, or $r_a$ to data is an Engine-A problem.

**Choose Engine B when observed or prescribed densities are the
input.** If the science starts from a density decomposition — "this
cluster is a Plummer halo of half-mass radius 2 pc plus an EFF core" —
Engine B takes those shapes verbatim, derives the one shared potential
they jointly generate, and asks Eddington whether each component can
exist as an equilibrium there. (It honestly refuses when the answer is
no — see the
[realizability discussion](eddington-engine.md).)

The two engines overlap at exactly one configuration — a single King
component ($g = 1$ in Engine A; King *density* in Engine B) — and that
overlap is the cross-engine trust anchor: two independent codepaths,
sampled with the **same seed** (a paired comparison, so any difference
is engine machinery rather than Monte-Carlo scatter), agree to a radial
KS distance of $2\times 10^{-4}$ and a velocity-dispersion-profile
deviation of $3\times 10^{-4}$.

:::{seealso} Engine A accuracy and gradients
The [Engine A validation page](../../50-validation/multimass-equilibrium.md)
records the retained physical checks and nine-case IMF-to-sampling derivative
matrix. [Adjoint and randomness conventions](../../20-architecture/differentiability.md#engine-a-ad)
distinguish fixed-realization derivatives from population-expectation gradients.
Those bounded checks do not qualify every component mixture or inference target.
:::

## The core principle: per-component equilibrium, no rescale

For a stationary component with a vanishing boundary pressure term, the
continuum virial relation is

```{math}
:label: per-component-virial
Q_j \;=\; \frac{T_j}{|W_j|} \;=\; 0.5
\qquad\text{with}\qquad
W_j = -\int \rho_j\, r\,\frac{\mathrm{d}\Phi}{\mathrm{d}r}\, \mathrm{d}^3r
\;\;\text{(Clausius, shared $\Phi$)},
```

where $T_j$ is component kinetic energy, $W_j$ its gravitational virial term,
$\rho_j$ its density and $\Phi$ the shared gravitational potential. This is a
continuum relation; finite samples have statistical scatter, and hard-truncated
Engine B profiles require the [boundary qualification](eddington-engine.md).
The sampler applies no external virial rescale. This matters because joint rescaling cannot
fix a multi-component IC: rescaling all velocities to set the *global*
$Q = 0.5$ moves every individually-correct component *away* from its
own equilibrium. Independent quadrature and sampled-moment comparisons are recorded on the
engine-specific validation pages.

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
* - [](two-component.md)
  - Worked two-component examples through **both** engines: an Engine-A
    cold/hot pair and the Engine-B halo+core headline, including the
    honest physics (an unrealizable mix, and the truncation-edge
    $Q_j$ plateau).
* - [](eddington-engine.md)
  - Engine B theory and methods: the shared-potential quadrature,
    per-component Eddington inversion, realizability, derived domains,
    and hybrid sampling.
* - [](../spatial-profiles/lowered-model-family.md)
  - Engine A theory and methods: the lowered-isothermal family, the
    coupled multi-component Poisson solve, and the DF-table
    performance layer.
```

## Composability

Each component composes with the modifier layers:

- Tidal truncation ([](../tidal-and-substructure/tidal.md)) applies to
  the union (or per-component if needed).
- Mass segregation has **two** routes: the Engine-A equipartition
  equilibrium (`from_mass_segregation` — segregation as a *true*
  equilibrium), and the labeled **primordial** non-equilibrium
  generator `energy_sorted_segregation`
  ([](../tidal-and-substructure/mass-segregation.md)) when the science
  calls for segregation imposed on a single-mass profile.
- Rotation stays a post-hoc transform (`apply_solid_body_rotation`,
  `apply_differential_rotation`,
  [](../velocity-dfs/rotation-anisotropy.md)) for either engine.

`sample_cluster` returns an `ICResult` whose `component_id` labels each
star's generating component, so per-population diagnostics (radial
profiles, $\sigma_j(r)$, $Q_j$, $\Lambda_{\rm MSR}$) are one mask away.

## Finite-population expectations

The total mass of a sampled catalogue sets its velocity normalization. A
mass-weighted velocity observable therefore retains a correlation between
each star's mass and the catalogue mass. Let $N$ be a fixed positive star count,
$m_j$ the representative mass of component $j$, and $\pi_j$ its normalized
number probability. Independent component labels give
$M=\sum_{i=1}^N m_i$, with one-star moments
$\mu_1=\sum_j\pi_jm_j$ and $\mu_2=\sum_j\pi_jm_j^2$.
Expanding the square separates the $N$ self terms from the $N(N-1)$ cross terms:

:::{include} ../../15-model-reference/_catalogue-mass-squared.md
:::

This number-weighted $\mu_1$ differs from the central-density-weighted mean
mass used by the Engine A segregation law. The expression has units of mass
squared and integrates the labels exactly.

For Engine A, squared speeds scale as
$|\boldsymbol v_i|^2=A_0 M w_j^2 u_i^2$ conditional on component $j$.
Here $u_i$ is dimensionless speed, $w_j$ the component velocity-scale ratio,
and $A_0=G/(9r_c\mu_{\mathrm{tot}})$, where $G$ is the gravitational constant,
$r_c$ the core scale radius and $\mu_{\mathrm{tot}}$ the dimensionless
equilibrium mass integral. Define $q_j=\mathbb{E}[w_j^2u_i^2\mid j]$,
$K_1=\sum_j\pi_jm_jq_j$ and $K_2=\sum_j\pi_jm_j^2q_j$.
With kinetic energy $T=\tfrac12\sum_i m_i|\boldsymbol v_i|^2$ in the model
rest frame, conditioning on one star's label gives
$\mathbb{E}[M\mid j]=m_j+(N-1)\mu_1$, hence

:::{include} ../../15-model-reference/_catalogue-kinetic.md
:::

The units are mass times velocity squared. Replacing $M$ by an independent
mean $N\mu_1$ drops the covariance contribution $A_0(K_2-\mu_1K_1)$.
These local derivations assume no recentering, selection or virial rescale.
`estimate_mass_squared_and_kinetic` integrates the labels and samples the
conditional kinetic moment. Its [bounded numerical qualification](../../50-validation/multimass-equilibrium.md#squared-mass-and-mass-weighted-kinetic-expectation)
does not certify broader domains or precision for the general scalar API.

## Implementation, validation & references

- **In code:** `src/progenax/cluster/multicomponent.py`
  (`MultiComponentCluster`, both engines), with the Engine-B
  shared-potential quadrature in
  `src/progenax/profiles/density_poisson.py` and the Eddington inverter
  in `src/progenax/kinematics/eddington.py`. See the
  [cluster API](../../30-api/cluster.md); each chapter below carries its
  exact module path.
- **Validated in:** [multimass equilibrium](../../50-validation/multimass-equilibrium.md)
  (Engine A), [Engine B (Eddington)](../../50-validation/engine-b-eddington.md),
  and [two-component](../../50-validation/two-component.md).
- **Primary sources:** self-consistent multi-mass lowered-isothermal
  equilibria follow {cite:t}`Gieles2015` (§4.1 multi-mass coupling);
  Eddington inversion with Osipkov–Merritt anisotropy follows
  {cite:t}`Merritt1985`; layered multi-population ICs are standard
  N-body practice ({cite:t}`Kuepper2011`'s McLuster), but that layered
  single-DF-per-component approach is *not* a joint equilibrium — the
  shared-potential treatment here is what replaces it. Full notes in the
  [bibliography](../../99-bibliography/index.md).
