---
title: Fractal substructure
description: Theory of clumpy, hierarchical young-cluster substructure — the Goodwin & Whitworth (2004) recursive-tree fractal, the Cartwright & Whitworth (2004) Q diagnostic, and the fractal-dimension D ↔ Q relationship. The differentiable generator that once lived here was removed in the 2026-06 rewrite; turbulent-density ICs are now the experimental gravoturb_fdf method.
---

# Fractal substructure

```{admonition} Implementation removed — theory chapter only
:class: warning
This chapter is now **theory only**. The differentiable *Fractal Displacement Field*
generator it used to document (`cluster.fdf`, `FractalLayer`, `generate_fractal_ic`,
`fractal_gw_legacy`) was **removed in the 2026-06 clean-room rewrite** and has **no
released successor** in progenax. (The legacy string-dispatch cluster generator
`generate_cluster_ic` that once hosted the fractal layer was itself retired in the
2026-06 unified redesign.) What survives is the **diagnostic** side (measuring
substructure) and a *different* turbulent-IC method:

- **CW04 $Q$ diagnostic** — `progenax.diagnostics.substructure.compute_q_parameter`
  (+ the differentiable kNN approximation `progenax.diagnostics.q_approx`); see
  [](../../20-architecture/jax-native-substructure-q.md).
- **Turbulent-density ICs** — the experimental **`gravoturb_fdf`** package (a *density*-field
  gravoturbulent method, **not** a displacement-field fractal; repo-only, not in the wheel).
```

Stars do not form in smooth, spherically symmetric distributions. Young
clusters inherit clumpy, hierarchical spatial structure from the
turbulent molecular clouds that birthed them, and that substructure
controls early dynamical evolution {cite:p}`Allison2009`. The
{cite:t}`Goodwin2004` recursive-tree fractal generator is the standard
prescription for seeding this substructure into N-body initial
conditions — and it is also a useful lens on *why* differentiable IC
generation is hard, which is the theory this chapter keeps.

## Why Goodwin–Whitworth doesn't differentiate in JAX

The {cite:t}`Goodwin2004` algorithm is conceptually elegant — recursive
subdivision with stochastic survival — but every step that gives it its
distinctive clumpiness is incompatible with JAX's differentiable model:

```{list-table}
:header-rows: 1

* - GW04 feature
  - JAX-incompatibility
* - **Bernoulli survival** ($p = N_{\mathrm{div}}^{(D-3)/3}$)
  - Discrete 0/1 decisions; gradient with respect to $D$ is zero almost everywhere
* - **Variable cardinality**
  - Number of survivors is stochastic; array shapes change per realisation, breaking JIT
* - **Hard sphere rejection**
  - $r \le 1$ cuts produce discontinuous boundaries
* - **Subsampling to $N_\star$**
  - `replace=True/False` logic is non-differentiable; produces gradient discontinuities at cardinality changes
* - **Recursive tree control flow**
  - `while_loop`-style descent with random termination; not `scan`-able with a fixed iteration count
```

For pure forward Monte Carlo, GW04 is fine. For *gradient-based*
inference,

```{math}
\mathrm{IC}\;\xrightarrow{\text{N-body}}\;\mathrm{evolved}\;\xrightarrow{\text{render}}\;\mathrm{mock\,obs.}\;\xrightarrow{\nabla}\;\mathcal{L}
```

every backpropagation step would zero out at the GW04 boundary — which is
the motivation for building substructure from a smooth, differentiable
field instead of a discrete tree.

## The "statistics, not algorithm" insight

GW04 realisations are characterised observationally not by their
recursion tree but by *summary statistics*:

- The {cite:t}`Cartwright2004` Q parameter
  ([](../../20-architecture/jax-native-substructure-q.md)).
- Azimuthal density variation $\sigma_\Sigma / \langle\Sigma\rangle$,
  with $\sigma_\Sigma/\langle\Sigma\rangle \approx 1.45 - 0.46\,D$
  {cite:p}`Kuepper2011`.
- Two-point correlation function $\xi(r)$ (rarely used directly).

Any generator that reproduces the same $(Q_{\mathrm{CW}},\,\sigma_\Sigma/\langle\Sigma\rangle)$
distributions as a function of one continuous parameter recovers
everything observationally relevant about GW04 substructure. This is the
key reason a *differentiable* generator can stand in for the discrete
tree: match the statistics, not the algorithm.

## Physical motivation: turbulent fragmentation

A differentiable substructure generator is arguably *more* physically
motivated than the GW04 tree, not less. Stars form in
supersonic-turbulent molecular clouds where the velocity field has a
power-law spectrum

```{math}
:label: turbulence-spectrum
P(k)\;\propto\;k^{-\beta}
```

with $\beta = 11/3$ for incompressible Kolmogorov turbulence, $\beta = 4$
for highly compressible Burgers turbulence, and $\beta \approx 3.8$–$4.0$
for the observed ISM {cite:p}`Kritsuk2011,FederrathKlessen2012`. (Note
this is the *velocity*-field spectrum; the gravoturbulent *density*
spectrum behaves differently — it flattens with Mach, see
[](../gravoturbulence/pdf-and-fdf.md).) Stars inherit the spatial
structure of the dense cores carved out of these turbulent flows: a
smooth equilibrium profile perturbed by a power-law-spectrum field
directly represents this picture, with steeper spectra producing
clumpier (more small-scale) structure. The GW04 tree, by contrast, is a
purely abstract construction with no direct connection to ISM physics.

## Fractal dimension $D$ and its CW04 $Q$ signature

The observational handle on fractal substructure is the relationship
between the {cite:t}`Goodwin2004` fractal dimension $D$ and the
{cite:t}`Cartwright2004` $Q$ parameter: lower $D$ (more hierarchical
clumping) gives lower $Q$; a uniform sphere sits at $Q \approx 0.79$.
Representative anchors (Plummer base, $N_\star \sim 1000$):

```{list-table} GW04 fractal-dimension ladder and its CW04 signature.
:header-rows: 1

* - GW04 $D$
  - $Q_{\mathrm{CW}}$
  - $\sigma_\Sigma/\langle\Sigma\rangle$
* - 1.6
  - 0.45
  - 0.71
* - 2.0
  - 0.55
  - 0.53
* - 2.4
  - 0.65
  - 0.36
* - 3.0 (uniform)
  - 0.79
  - 0.07
```

These are theory/reference anchors, not fresh output from a committed
script. The CW04 $Q$ estimator that produces them is validated against
{cite:t}`Cartwright2004` Table 1 (uniform sphere $Q = 0.79 \pm 0.02$;
see [](../../20-architecture/jax-native-substructure-q.md)). The
experimental `gravoturb_fdf` package's headline calibration reproduces
the *direction* of this ladder — $Q$ decreasing as more stars are drawn
from the dense turbulent tail — measured with realization bands (its
`VALIDATION_SUMMARY.md`, AC7).

## Non-equilibrium kinematics

Fractal/clumpy ICs are intentionally **non-equilibrium**: the standard
equilibrium velocity DFs (Plummer, King) assume a smooth density field,
so they are inconsistent with clumpy positions. The
{cite:t}`Allison2009` "cool clumpy" setup ($Q_{\mathrm{vir}} \approx 0.3$,
where $Q_{\mathrm{vir}} \equiv T/|V|$ — see
[](../../20-architecture/q-virial-convention.md)) is a deliberate use of
this non-equilibrium pathway to study rapid dynamical mass segregation —
substructure that erases itself within $\sim 1$ Myr of evolution. See
[](mass-segregation.md) for the segregation side, which **is** released:
the primordial `energy_sorted_segregation` generator and the equilibrium
`MultiComponentCluster.from_mass_segregation` constructor.

## References

The GW04 baseline is {cite:t}`Goodwin2004`; the substructure diagnostic is
{cite:t}`Cartwright2004` (Q parameter) with the azimuthal-variation
relation from {cite:t}`Kuepper2011`. The turbulent-fragmentation physical
picture follows {cite:t}`FederrathKlessen2012` and {cite:t}`Kritsuk2011`.
The cool-fractal dynamical-segregation pathway is {cite:t}`Allison2009`;
the primordially-segregated comparison case is {cite:t}`Baumgardt2008`.
