---
title: Spatial density profiles
description: The spatial density profiles in progenax — Plummer, King, EFF, and the anisotropic Michie–King — and when to use each.
---

# Spatial density profiles

A **spatial density profile** $\rho(r)$ is one of the three orthogonal
ingredients in every progenax IC ([](../ic-philosophy.md)). progenax
ships four spatial profiles — the isotropic Plummer / King / EFF plus the
self-consistent anisotropic **Michie–King**. They do **not** share a
single size parameter: Plummer is parameterised by the half-mass radius
$r_h$; King and Michie–King by the central potential $W_0$ and core radius
$r_c$ (Michie–King adds the anisotropy radius $r_a$); and EFF by the scale
radius $a$, outer slope $\gamma$, and truncation radius $r_t$. Convert
between the conventions using each profile's own $r_h$ relation (e.g.
{eq}`plummer-rh-a`) when you need cross-profile comparability.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers choosing a spatial density profile and learning how progenax's profiles compose with velocity DFs and modifiers; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [IC philosophy](../ic-philosophy.md) (the three orthogonal IC ingredients) — a good entry point; the individual profile pages ([Plummer](plummer.md), [King](king.md), [EFF](eff.md)) go deeper on each.
**You'll get:** what a spatial density profile is, which of the four progenax profiles to use when, the shared `SpatialProfile` API contract, and how profiles pair with equilibrium velocity DFs.
:::

```{list-table}
:header-rows: 1

* - Profile
  - $\rho(r)$ form
  - Use when
* - [](plummer.md)
  - $\rho(r) = \frac{3 M}{4\pi a^3}\,\bigl[1 + (r/a)^2\bigr]^{-5/2}$
  - You want a smooth, untruncated profile and a closed-form everything (mass, potential, velocity dispersion, distribution function).
* - [](king.md)
  - Lowered isothermal sphere; ODE-defined, parameterised by central potential $W_0$.
  - You want a *tidally truncated* cluster matching observed Galactic globular clusters; need a finite outer radius.
* - [](eff.md)
  - $\rho(r) = \rho_0\,\bigl[1 + (r/a)^2\bigr]^{-\gamma/2}$
  - You want power-law outer falloff matching young massive cluster surface brightness; need a free outer-slope parameter.
* - [](../velocity-dfs/michie-king.md)
  - Self-consistent King with a Gaussian-in-$J^2$ anisotropy term (Michie 1963); ODE-defined, distinct from isotropic King.
  - You want a tidally-truncated cluster with *radial velocity anisotropy* increasing outward (the Michie–King model). Density and DF are solved together.
```

```{figure} ../figures/profile_family_portrait.webp
:label: fig-profile-family-portrait
:width: 88%

The model-selection table, drawn: every released density family,
half-mass-normalized. Plummer's untruncated $r^{-5}$ tail; the King
$W_0 = 3 \to 12$ sequence (each $\sim$4–5 units of $W_0$ buys a decade of
core-to-tidal contrast $c = \log_{10} r_t/r_c$: $0.67 \to 1.53 \to 2.74$);
EFF's shallow power-law halo crossing Plummer at large radii; Michie
(dashed) as "King, but more extended" — radial anisotropy visible as pure
structure. Regenerate: `python -m laboratory.icviz --only
profile-family-portrait`.
```

Plummer is the default (closed-form everything). King fits old, tidally
truncated globulars ($W_0 \sim 5$–$9$); EFF fits young massive clusters
whose power-law halos King's exponential edge cannot capture
{cite:p}`ElsonFallFreeman1987`; Michie–King adds self-consistent radial
anisotropy to the King picture.

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
but the protocol does not mandate it). `sample_positions` uses an
inverse-CDF sampler over a fixed grid so that the *positions themselves*
are differentiable in the size parameter ($r_h$ for Plummer, $r_c$/$W_0$
for King, $a$/$\gamma$ for EFF) — useful when the loss function depends on
post-sampling spatial moments.

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
  - Plummer + isothermal velocity → cold-collapse IC
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

All four profiles compose with the modifier layers — mass segregation
([](../tidal-and-substructure/mass-segregation.md)), fractal
substructure ([](../tidal-and-substructure/fractal.md)), and tidal
truncation ([](../tidal-and-substructure/tidal.md)) — without changing
the underlying $\rho(r)$.

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
At $r = 3\,r_h$, rank the families in {numref}`fig-profile-family-portrait`
by $\rho$ before looking closely. Why must EFF ($\gamma = 3.5$) eventually
exceed *every* King model, however concentrated? (Truncation beats any
power law: King's $\rho \to 0$ at finite $r_t$, EFF's tail only falls as
$r^{-3.5}$.)
:::

:::{dropdown} 2. The $r_h$ trap
Build `KingProfile.from_W0_rc(W0=7.0, r_c=1.0)` and `PlummerProfile(r_h=1.0)`.
Are they the same size? Compute the King model's half-mass radius (integrate
its density, or sample and take the median radius) — you'll find
$r_h \approx 3.9\,r_c$, so this King cluster is nearly four times larger
than the Plummer sphere despite the unit scale parameter. This is why the
portrait normalizes by $r_h$ and why cross-profile comparisons must too.
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
  The lowered-model family {cite:t}`Gieles2015` unifies these under a
  single multi-mass family that progenax implements natively (see the
  [lowered-model family](lowered-model-family.md)). Full notes in the
  [bibliography](../../99-bibliography/index.md).
