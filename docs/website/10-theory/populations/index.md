---
title: Multi-component populations
description: "progenax's unified MultiComponentCluster — one differentiable model, two equilibrium engines (DF-defined lowered-isothermal family and density-defined Eddington inversion), every component individually virial in one shared potential with no external rescale."
---

# Multi-component populations

```{seealso}
For single-component clusters built from one IMF + one profile + one
DF, see the chapter families [](../spatial-profiles/index.md),
[](../velocity-dfs/index.md), and [](../imfs/index.md). This chapter
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

## The core principle: per-component equilibrium, no rescale

Every `MultiComponentCluster` IC satisfies, **per component**,

```{math}
:label: per-component-virial
Q_j \;=\; \frac{T_j}{|W_j|} \;=\; 0.5
\qquad\text{with}\qquad
W_j = -\int \rho_j\, r\,\frac{\mathrm{d}\Phi}{\mathrm{d}r}\, \mathrm{d}^3r
\;\;\text{(Clausius, shared $\Phi$)},
```

and this **emerges from the DF** — there is no external virial rescale
anywhere in the pipeline. This matters because joint rescaling cannot
fix a multi-component IC: rescaling all velocities to set the *global*
$Q = 0.5$ moves every individually-correct component *away* from its
own equilibrium. (Precisely this failure — feeding each sub-population's
DF the full cluster mass, then relying on a rescale — was the physics
bug that retired progenax's legacy two-component generator.) Both
engines instead prove $Q_j = 0.5$ by exact quadrature oracles that are
deliberately independent of the sampled draws, and the sampled clusters
are checked *unscaled*.

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

## References

Self-consistent multi-mass lowered-isothermal equilibria follow
{cite:t}`Gieles2015` (Section 4.1 for the multi-mass coupling);
Eddington inversion with Osipkov–Merritt anisotropy follows
{cite:t}`Merritt1985`. Multi-population N-body initial conditions are
standard practice — {cite:t}`Kuepper2011`'s McLuster supports layered
multi-population ICs — but the layered single-DF-per-component
approach is *not* a joint equilibrium; the shared-potential treatment
on this page is what replaces it in progenax.
