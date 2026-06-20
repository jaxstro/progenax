---
title: Spatial density profiles
description: The production-grade spatial density profiles in progenax — Plummer, King, EFF, and the anisotropic Michie–King — and when to use each.
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

Plummer is the default. King is the right choice for old globular
clusters where the King-radius / tidal-radius ratio is observationally
constrained ($W_0 \sim 5$–$9$ for most Galactic GCs). EFF is the right
choice for young massive clusters in the LMC and elsewhere where the
power-law outer slope is a free parameter not well captured by King's
exponential cutoff {cite:p}`ElsonFallFreeman1987`.

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

## References

The isotropic profiles are due to {cite:t}`Plummer1911`, {cite:t}`King1966`,
and {cite:t}`ElsonFallFreeman1987`; the anisotropic Michie–King model is
{cite:t}`Michie1963` (+ {cite:t}`King1966` cutoff). The lowered-model
family formalized by {cite:t}`Gieles2015` unifies these lowered models under
a single multi-mass family, which progenax plans to implement natively as its
own differentiable generalization; see
[](lowered-model-family.md) for the roadmap.
