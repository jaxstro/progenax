---
title: Fractal substructure (FDF method)
description: progenax's Fractal Displacement Field method — a differentiable replacement for the Goodwin & Whitworth (2004) recursive-tree algorithm, calibrated to reproduce the same CW04 Q and azimuthal-density statistics.
---

# Fractal substructure: the FDF method

```{seealso}
This chapter introduces the **Fractal Displacement Field (FDF)**
method, progenax's differentiable replacement for the {cite:t}`Goodwin2004`
recursive-tree algorithm. For the diagnostic Q parameter that quantifies
the resulting substructure, see [](../../20-architecture/jax-native-substructure-q.md).
For the dynamical evolution that *erases* the substructure, see
[](mass-segregation.md) (which discusses the Allison+2009 "rapid
segregation from cool fractals" channel).
```

Stars do not form in smooth, spherically symmetric distributions. Young
clusters inherit clumpy, hierarchical spatial structure from the
turbulent molecular clouds that birthed them, and that substructure
controls early dynamical evolution {cite:p}`Allison2009`. The
{cite:t}`Goodwin2004` recursive-tree fractal generator is the standard
prescription for seeding this substructure into N-body initial
conditions. It is also fundamentally non-differentiable in JAX.

progenax implements the **Fractal Displacement Field (FDF)** method as
a differentiable replacement: a smooth equilibrium IC is perturbed by a
power-spectrum-controlled displacement field whose parameters
$(\chi, \lambda_{\mathrm{frac}}, \sigma_u)$ are calibrated to reproduce
the same CW04 Q and azimuthal-density statistics that GW04 produces at
fractal dimension $D$. The method gives up bit-for-bit reproducibility
of GW04 realisations to gain end-to-end gradient flow through a critical
IC parameter — fractal strength can now sit alongside virial Q,
$\alpha_{\mathrm{IMF}}$, and $\lambda_{\mathrm{seg}}$ inside an HMC
posterior chain.

## Why Goodwin-Whitworth doesn't work in JAX

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

For pure forward Monte Carlo, GW04 is fine — progenax retains a legacy
implementation at `progenax.cluster.fractal_gw_legacy` for cross-checks.
For *gradient-based* inference,

```{math}
\mathrm{IC}\;\xrightarrow{\text{N-body}}\;\mathrm{evolved}\;\xrightarrow{\text{render}}\;\mathrm{mock\,obs.}\;\xrightarrow{\nabla}\;\mathcal{L}
```

every backpropagation step would zero out at the GW04 boundary. We need
a method that produces *statistically equivalent* substructure while
keeping every parameter differentiable.

## The "statistics, not algorithm" insight

GW04 realisations are characterised observationally not by their
recursion tree but by *summary statistics*:

- The {cite:t}`Cartwright2004` Q parameter
  ([](../../20-architecture/jax-native-substructure-q.md))
- Azimuthal density variation $\sigma_\Sigma / \langle\Sigma\rangle$,
  with $\sigma_\Sigma/\langle\Sigma\rangle \approx 1.45 - 0.46\,D$
  {cite:p}`Kuepper2011`
- Two-point correlation function $\xi(r)$ (rarely used directly)

A different generator that hits the same $(Q_{\mathrm{CW}},\,\sigma_\Sigma/\langle\Sigma\rangle)$
distributions as a function of one continuous parameter recovers
everything observationally relevant about GW04 substructure. That is
exactly what FDF does.

## Physical motivation: turbulent fragmentation

The FDF method is arguably *more* physically motivated than the GW04
tree, not less. Stars form in supersonic-turbulent molecular clouds
where the velocity field has a power-law spectrum

```{math}
:label: turbulence-spectrum
P(k)\;\propto\;k^{-\beta}
```

with $\beta = 11/3$ for incompressible Kolmogorov turbulence, $\beta = 4$
for highly compressible Burgers turbulence, and $\beta \approx 3.8$–$4.0$
for the observed ISM {cite:p}`Kritsuk2011,FederrathKlessen2012`. Stars
inherit the spatial structure of the dense cores carved out of these
turbulent flows. A power-law-spectrum displacement field applied to a
smooth equilibrium profile *directly represents* this picture: turbulent
modes shake stars from their smooth-distribution starting positions,
with more shaking on smaller scales (clumpier) when the spectrum is
steeper.

The GW04 tree, by contrast, is a purely abstract mathematical
construction with no direct connection to ISM physics.

## The FDF method

The construction has four steps.

**Step 1 — sample base positions from a smooth equilibrium profile.**
Use any of progenax's spatial profiles (Plummer, King, EFF; see
[](../spatial-profiles/index.md)). This produces $N_\star$ smooth-IC
positions $\mathbf{x}_i^{(0)}$ via standard inverse-CDF sampling.

**Step 2 — construct a vector displacement field with a tunable power
spectrum.** Choose $M$ wavevectors $\mathbf{k}_n = k_n\,\hat{\mathbf{k}}_n$
with magnitudes log-spaced in $[k_{\min}, k_{\max}]$ (typical range:
$k_{\min} \sim 1/R_{\mathrm{half}}$ down to $k_{\max} \sim 20/R_{\mathrm{half}}$),
random isotropic directions $\hat{\mathbf{k}}_n$, random phases
$\varphi_n$, and random polarisation directions $\hat{\mathbf{a}}_n$.
All of these are drawn once and *frozen* per realisation. The
displacement at position $\mathbf{x}$ is

```{math}
:label: displacement
\mathbf{u}(\mathbf{x};\,\chi,\,\sigma_u)
\;=\;\sum_{n=1}^{M} A_n(\chi,\,\sigma_u)\,\hat{\mathbf{a}}_n\,\cos(\mathbf{k}_n\cdot\mathbf{x} + \varphi_n)
```

where the mode amplitudes follow the power law

```{math}
:label: amplitudes
A_n(\chi,\,\sigma_u)\;=\;C(\chi,\,\sigma_u)\,k_n^{-\beta(\chi)/2},
\qquad
\beta(\chi)\;=\;\beta_0 + \beta_1\,(3 - \chi)
```

with $\beta_0 \approx 2.0$, $\beta_1 \approx 1.5$ (set by calibration),
and $C$ chosen so that $\sum_n A_n^2 = \sigma_u^2$. The clumpiness
parameter $\chi \in [1.5, 3.0]$ controls the spectral slope:

```{list-table}
:header-rows: 1

* - $\chi$
  - $\beta(\chi)$
  - Behaviour
* - 3.0
  - 2.0
  - Maximum large-scale power; smooth, near-uniform
* - 2.0
  - 3.5
  - Moderate substructure
* - 1.5
  - 4.25
  - Maximum small-scale power; highly clumpy
```

**Step 3 — apply with fractal fraction $\lambda_{\mathrm{frac}}$.**

```{math}
:label: apply-fdf
\mathbf{x}_i^{(1)}\;=\;\mathbf{x}_i^{(0)} + \lambda_{\mathrm{frac}}\,\mathbf{u}_i,
\qquad \mathbf{u}_i = \mathbf{u}(\mathbf{x}_i^{(0)};\,\chi,\,\sigma_u).
```

$\lambda_{\mathrm{frac}} = 0$ recovers the smooth profile;
$\lambda_{\mathrm{frac}} = 1$ applies the full displacement; intermediate
values produce smooth blends suitable for gradient inference.

**Step 4 — preserve the radial profile (recommended).** The displacement
in step 3 changes the radial CDF. progenax provides three modes for
radial-profile preservation, the recommended default being the
**rank-based remap**:

1. Compute displaced radii $r_i^{(1)} = |\mathbf{x}_i^{(1)}|$ and sort.
2. Sort the original (target) radii $r_i^{(0)}$.
3. Star at displaced-rank $k$ receives target-radius rank $k$:
   $\mathbf{x}_i^{\mathrm{final}} = \hat{\mathbf{x}}_i^{(1)} \cdot r^{\mathrm{target}}_{\mathrm{sorted},k}$.

The remap preserves the radial CDF *exactly* per-star while keeping the
clumpy angular structure. Sorting is non-differentiable in the
permutation, but gradients flow through the *values* being sorted —
acceptable for inference since the permutation only changes
discontinuously at coincidence loci of measure zero.

```{admonition} Three radial-preservation options
:class: note

`radial_mode='full'` — apply the displacement directly; let the radial
CDF deviate. Useful for studying how substructure feeds back on the
radial profile during evolution.

`radial_mode='tangential'` — project out the radial component of
$\mathbf{u}$ and re-normalise to original radius. Preserves radial CDF
*per-star* exactly. Drags points along radial rays for large
displacements, slightly twisting the angular structure. Expert option.

`radial_mode='remap'` (default) — apply full 3D displacement, then
rank-remap radii. Exact CDF + best preservation of clumpy angular
structure.
```

## Differentiability: what gets gradients

The FDF construction has explicit *frozen* and *gradient-receiving*
components:

```{list-table}
:header-rows: 1

* - Frozen (no gradient)
  - Gradient-receiving (differentiable)
* - Wavevector directions $\hat{\mathbf{k}}_n$
  - Clumpiness $\chi$
* - Phases $\varphi_n$
  - Fractal fraction $\lambda_{\mathrm{frac}}$
* - Polarisation directions $\hat{\mathbf{a}}_n$
  - Amplitude scale $\sigma_u$
* - Base profile random draws
  - Profile parameters ($r_h$, …)
* - Sort permutations (radial remap)
  - Virial ratio $Q_{\mathrm{vir}}$
```

The separation is enforced via `jax.lax.stop_gradient` on the stochastic
`FractalField` dataclass. Gradients of any downstream observable with
respect to $\chi$, $\lambda_{\mathrm{frac}}$, and $\sigma_u$ are
well-defined and HMC-friendly.

## Velocity coupling: coherent kinematics

Stars in the same dense clump should share correlated velocities — they
fragmented out of the same converging flow. progenax implements
**coherent velocity assignment** by deriving local velocity fields from
the same displacement-mode basis used for positions:

```{math}
:label: velocity-coupling
\mathbf{v}_i\;=\;\mathbf{v}_i^{\mathrm{equil}}\;+\;\lambda_{\mathrm{vel}}\cdot\sigma_v\,\dot{\mathbf{u}}(\mathbf{x}_i^{(0)};\,\chi,\,\sigma_u)
```

where $\dot{\mathbf{u}}$ is the time-derivative of the displacement
field (a different power-law-spectrum vector field built from the same
modes), and $\sigma_v$ is set by virial scaling after summing
$\mathbf{v}_i^{\mathrm{equil}} + \lambda_{\mathrm{vel}}\cdot\dot{\mathbf{u}}$.
The full velocity assignment is then renormalised to the target
$Q_{\mathrm{vir}}$ via [](../../20-architecture/q-virial-convention.md).

```{warning}
**Fractal ICs are intentionally non-equilibrium.** The standard
equilibrium velocity DFs (Plummer, King) assume a smooth density field;
applying them to FDF positions produces inconsistent kinematics. progenax
handles this by routing fractal ICs through `assign_velocities_and_virialize`
rather than the equilibrium DFs. The {cite:t}`Allison2009` "cool clumpy"
setup ($Q_{\mathrm{vir}} \approx 0.3$) is a deliberate use of this
non-equilibrium pathway to study rapid dynamical mass segregation —
see [](mass-segregation.md).
```

## Calibrating $\chi$ to GW04 fractal dimension $D$

The FDF clumpiness parameter $\chi$ does not directly equal a GW04
fractal dimension $D$. Calibration is done empirically: generate FDF
realisations across $\chi \in [1.5, 3.0]$ and GW04 realisations across
$D \in [1.6, 3.0]$, measure $Q_{\mathrm{CW}}$ for each, and fit the
$\chi(D)$ mapping that minimises the discrepancy.

```{list-table} Approximate calibration map (Plummer base, $N_\star = 1000$).
:header-rows: 1

* - GW04 $D$
  - $Q_{\mathrm{CW}}$
  - FDF $\chi$
  - $\sigma_\Sigma/\langle\Sigma\rangle$
* - 1.6
  - 0.45
  - 1.6
  - 0.71
* - 2.0
  - 0.55
  - 2.0
  - 0.53
* - 2.4
  - 0.65
  - 2.4
  - 0.36
* - 3.0
  - 0.79 (uniform)
  - 3.0
  - 0.07
```

The calibration is a one-time offline procedure. Current coverage is
unit/offline-backed rather than a dedicated
`tests/validation/test_fractal_substructure.py` suite. See
[](../../50-validation/fractal-substructure.md) for the rendered
validation-status notes.

## Implementation in progenax

The high-level user-facing cluster API lives in `progenax.cluster`:

```python
import jax
from jaxstro.units import STELLAR
from progenax.cluster import FractalLayer, SpatialStructureParams, generate_cluster_ic
from progenax.imf import PowerLawIMF

cluster = generate_cluster_ic(
    key=jax.random.PRNGKey(42),
    N_stars=1000,
    M_total=1000.0,
    R_half=1.0,
    imf_params=PowerLawIMF.kroupa(),
    structure_params=SpatialStructureParams(
        base_profile="plummer",
        fractal=FractalLayer(
            D=2.0,
            lambda_frac=1.0,
            virial_ratio=0.3,   # Allison+2009-style cool fractal
        ),
    ),
    G=STELLAR.G,
)
```

For lower-level FDF experiments, `progenax.cluster.fdf` exports
`FractalDisplacementLayer`, `init_fractal_field`,
`apply_displacement`, and `generate_fractal_ic`. There is no public
`generate_fdf_positions` helper in this checkout.

See [](../../50-validation/fractal-substructure.md) for the current
validation-status page.

## Composing with mass segregation

FDF positions and Baumgardt mass segregation are conceptually layered,
but the current high-level `generate_cluster_ic` implementation accepts
only one of them at a time. Passing both `fractal` and
`mass_segregation` raises `ValueError`. Their interaction still has
astrophysical content:

- $\lambda_{\mathrm{frac}} > 0,\,\lambda_{\mathrm{seg}} = 0,\, Q_{\mathrm{vir}} = 0.3$ — cool clumpy IC. Reproduces the {cite:t}`Allison2009` setup; dynamical mass segregation emerges within $\sim 1$ Myr of evolution.
- $\lambda_{\mathrm{frac}} = 0,\,\lambda_{\mathrm{seg}} > 0,\, Q_{\mathrm{vir}} = 0.5$ — primordially segregated smooth profile. Reproduces the {cite:t}`Baumgardt2008` setup.
- $\lambda_{\mathrm{frac}} > 0,\,\lambda_{\mathrm{seg}} > 0$ — composite. Useful for simultaneous-fit posteriors but degenerate after $\sim 1$ crossing time of evolution.

See [](mass-segregation.md) for the segregation side of the layered
construction and the doubly-segregated regime.

## References

The FDF method is original to progenax. The physical motivation follows
{cite:t}`FederrathKlessen2012` and {cite:t}`Kritsuk2011`. The GW04 baseline
is {cite:t}`Goodwin2004`; the calibration target observables are
{cite:t}`Cartwright2004` (Q parameter) and {cite:t}`Kuepper2011` (azimuthal
variation). The cool-fractal dynamical-segregation pathway is
{cite:t}`Allison2009`.
