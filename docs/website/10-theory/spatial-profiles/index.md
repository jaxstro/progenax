---
title: Spatial density profiles
description: The spatial density profiles in progenax — Plummer, King, EFF, and the anisotropic Michie–King — and when to use each.
---

# Spatial density profiles

A spatial density profile specifies how mass is distributed in space.
The profiles compared here are spherical; velocity isotropy or anisotropy is a
property of the accompanying distribution function. Progenax exposes density and
velocity components separately, but equilibrium constrains how they can be combined.

The profiles do not share a single size parameter. Plummer accepts the half-mass
radius $r_h$; King and Michie–King use $W_0$ and $r_c$, with an additional anisotropy
radius for Michie–King. EFF uses scale radius $a$, three-dimensional density slope
$\gamma$, and truncation radius $r_t$. The latter is the imposed outer boundary:
EFF density is zero for $r>r_t$. It is not automatically a tidal radius derived
from an external gravitational field. Use each model's enclosed-mass relation to
compare profiles at the same $r_h$.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers choosing a spatial density profile and learning how progenax's profiles compose with velocity DFs and modifiers; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [IC philosophy](../ic-philosophy.md) (the density, velocity, and mass conventions) — a good entry point; the individual profile pages ([Plummer](plummer.md), [King](king.md), [EFF](eff.md)) go deeper on each.
**You'll get:** what a spatial density profile is, which of the four progenax profiles to use when, the shared `SpatialProfile` API contract, and how profiles pair with equilibrium velocity DFs.
:::

```{list-table}
:header-rows: 1

* - Profile
  - $\rho(r)$ form
  - Use when
* - [](plummer.md)
  - $\rho(r) = \frac{3 M}{4\pi a^3}\,\bigl[1 + (r/a)^2\bigr]^{-5/2}$
  - You want a smooth, untruncated profile and analytic mass, potential, velocity dispersion, and isotropic distribution function.
* - [](king.md)
  - Lowered isothermal sphere; ODE-defined, parameterised by central potential $W_0$.
  - You want a lowered-isothermal model with a finite outer radius.
* - [](eff.md)
  - $\rho(r) = \rho_0\,\bigl[1 + (r/a)^2\bigr]^{-\gamma/2}$
  - You want a variable outer density slope and an imposed truncation radius. Here $\gamma$ describes 3-D density, not the projected surface-brightness slope of the original EFF convention.
* - [](../velocity-dfs/michie-king.md)
  - Self-consistent King with a Gaussian-in-$J^2$ anisotropy term (Michie 1963); ODE-defined, distinct from isotropic King.
  - You want a finite lowered model with radial velocity anisotropy. Michie–King anisotropy grows away from the core and returns toward isotropy at its finite edge. Density and DF are solved together.
```

```{figure} ../figures/profile_family_portrait.webp
:label: fig-profile-family-portrait
:alt: Plummer, King, EFF, and Michie density shapes versus radius normalized by each half-mass radius; each curve is divided by its own near-central density.
:width: 88%

Density profiles compared at matched half-mass radius. The curves illustrate
differences in central structure and outer envelopes. A density curve alone does
not determine velocity anisotropy. The ordinate divides each density by its value at $10^{-3}r_h$; it does not
match total masses. Radii use a numerically integrated $r_h$, with the Plummer
integral evaluated out to $30$ input half-mass radii. King curves use
$W_0=3,7,12$; EFF uses $\gamma=3.5$ and $r_t/a=60$; Michie–King uses
$W_0=7$ and $r_a/r_c=8$. King and Michie–King cutoffs come from their models. Regenerate: `python -m laboratory.icviz --only profile-family-portrait`.
```

Choose the family by the required density structure, outer boundary, and velocity
model. Plummer supplies an analytic baseline; King supplies a lowered-isothermal
model with a finite boundary; EFF allows a variable outer slope; Michie–King
couples density to an anisotropic DF.

## Common API contract

Every spatial profile satisfies the `SpatialProfile` protocol
([](../../20-architecture/protocols.md)):

```python
class SpatialProfile(Protocol):
    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKey,
    ) -> Float[Array, "N 3"]:
        """Draw N positions from ρ(r). Returns (N, 3) Cartesian."""
        ...

    def characteristic_radius(self) -> Float[Array, ""]:
        """A representative scale (r_h for Plummer, r_t for King/EFF),
        used e.g. for softening defaults."""
        ...
```

The protocol's contract is just these two methods — there is no shared
`r_h` field and no `density`/`cumulative_mass` requirement (the concrete
classes provide `density` where it is closed-form, e.g. Plummer and EFF,
but the protocol does not mandate it). Sampling methods differ by profile.
Plummer uses an analytic inverse CDF; other profiles use numerical constructions.
The common protocol specifies the interface, not a shared interpolation scheme
or gradient guarantee.

## Gravitational potential outside a finite profile

A density cutoff does not remove the cluster's exterior gravitational field.
For a spherical component of total mass $M$ and no density at $r>r_t$,
with gravitational constant $G$, Poisson's equation gives $\mathrm{d}\Phi/\mathrm{d}r=GM/r^2$. Thus

```{math}
:label: profile-vacuum-potential
\Phi(r)=-\frac{GM}{r}+C,\qquad r\ge r_t,
```

where $\Phi$ is specific potential, with units of squared speed, and $C$ fixes
its additive zero. The radial acceleration is $-GM/r^2$ for either zero.
The functional API `compute_profile_potential` retains each model's convention:

| Model | Potential zero | Exterior specific potential |
|---|---|---|
| King | $\Phi(r_t)=0$ | $GM(1/r_t-1/r)$ |
| EFF | $\Phi(\infty)=0$ | $-GM/r$ |

King's exterior potential is positive in its tidal-zero convention. Potential
sign alone does not determine whether an orbit is bound; compare its specific
energy with the relevant cutoff energy in the same convention. The King DF's
nonnegative lookup $W$ is used for density and velocity sampling inside its
support; clamping that lookup is not a vacuum gravitational potential.
Interior ODE and mass-table interpolation remain numerical approximations.
See the [functional potential API](../../30-api/profiles.md#api-profiles-compute_profile_potential)
and the owning [King](king.md) and [EFF](eff.md) pages.

## Composability with velocity DFs and modifiers

Each profile pairs with a matching equilibrium velocity DF for
in-equilibrium ICs:

```{list-table}
:header-rows: 1

* - Profile
  - Equilibrium velocity DF
  - Out-of-equilibrium pairing?
* - Plummer
  - `progenax.kinematics.PlummerVelocityDF` ([](../velocity-dfs/plummer-dfs.md))
  - Plummer + isothermal velocities → a different velocity prescription; whether it is subvirial depends on velocity normalization
* - King
  - `progenax.kinematics.KingVelocityDF` ([](../velocity-dfs/king-dfs.md))
  - King + Plummer DF → mismatched IC for testing relaxation
* - EFF
  - `progenax.kinematics.EFFVelocityDF`
  - EFF + isotropic-Maxwellian → approximate but useful starting state
* - Michie–King
  - `progenax.kinematics.MichieVelocityDF` ([](../velocity-dfs/michie-king.md))
  - Solved self-consistently as a pair (`MichieProfile` + `MichieVelocityDF`); the density is intrinsic to the anisotropic DF
```

Modifiers can change spatial structure, mass–position correlations, or radial
support. Assess the final modified realization against the intended model.
See [mass segregation](../tidal-and-substructure/mass-segregation.md),
[fractal substructure](../tidal-and-substructure/fractal.md), and
[truncation](../tidal-and-substructure/tidal.md).

## Sampler fidelity at a glance

```{figure} ../figures/profile_density_residuals.webp
:label: fig-profile-density-residuals
:width: 100%

Sampled radial densities ($N = 2\times 10^5$ per family, seed 7) against
the analytic curves, spanning 6–7 decades, with residuals inside the
$\pm 1/\sqrt{N_{\rm bin}}$ Poisson bands (shaded). Bins with fewer than
10 stars are dropped. Quantitative gates: [Plummer](../../50-validation/plummer-equilibrium.md),
[King](../../50-validation/king-profile.md), [EFF](../../50-validation/eff-profile.md).
Regenerate: `python -m laboratory.icviz --only profile-density-residuals`.
```

## Check yourself

:::{dropdown} 1. Read the portrait
Both King and the implemented EFF profiles have finite cutoffs. Before ranking
their outer densities, identify each cutoff in units of $r_h$. Which comparison
remains meaningful inside their shared radial range?
:::


:::{dropdown} 2. The $r_h$ trap
Build `KingProfile.from_W0_rc(W0=7.0, r_c=1.0)` and `PlummerProfile(r_h=1.0)`.
Are they the same size? Compute the King model's half-mass radius from its
cumulative mass profile and compare it with the Plummer input radius. Matching
numerical values of different scale parameters does not match half-mass radii.
:::


## Implementation, validation & references

- **In code:** the profiles live under `src/progenax/profiles/`
  (`plummer.py`, `king.py`, `eff.py`, `michie.py`) with their paired DFs
  under `src/progenax/kinematics/`. See the
  [profiles API](../../30-api/profiles.md) and the
  [kinematics API](../../30-api/kinematics.md); the per-profile theory
  pages ([Plummer](plummer.md), [King](king.md), [EFF](eff.md)) carry the
  exact module paths.
- **Validated in:** [Plummer](../../50-validation/plummer-equilibrium.md),
  [King](../../50-validation/king-profile.md),
  [EFF](../../50-validation/eff-profile.md), and
  [Michie anisotropy](../../50-validation/michie-anisotropy.md).
- **Primary sources:** the isotropic profiles are {cite:t}`Plummer1911`,
  {cite:t}`King1966`, and {cite:t}`ElsonFallFreeman1987`; the anisotropic
  Michie–King model is {cite:t}`Michie1963` (+ {cite:t}`King1966` cutoff).
  The lowered-model family {cite:t}`Gieles2015` contains King and
  Michie–King and their multi-mass generalizations; it does not make a
  prescribed finite EFF density a member of the same DF family (see the
  [lowered-model family](lowered-model-family.md)). Full notes in the
  [bibliography](../../99-bibliography/index.md).
