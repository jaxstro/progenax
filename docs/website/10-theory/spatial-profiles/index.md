---
title: Spatial density profiles
description: The production-grade spatial density profiles in progenax — Plummer, King, EFF, and the anisotropic Michie–King — and when to use each.
---

# Spatial density profiles

A **spatial density profile** $\rho(r)$ is one of the three orthogonal
ingredients in every progenax IC ([](../ic-philosophy.md)). progenax
ships four production-grade profiles — the isotropic Plummer / King /
EFF (parameterised by half-mass radius $r_h$ for cross-profile
comparability) plus the self-consistent anisotropic **Michie–King**
($W_0$, $r_c$, $r_a$):

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
    r_h: Float[Array, ""]   # Half-mass radius (always exposed)

    def density(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """ρ(r) at radius r."""
        ...

    def cumulative_mass(self, r: Float[Array, "..."]) -> Float[Array, "..."]:
        """M(<r) / M_total."""
        ...

    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKey,
        *,
        truncate: float | None = None,
    ) -> Float[Array, "N 3"]:
        """Draw N positions from ρ(r). Returns (N, 3) Cartesian."""
        ...
```

`density` and `cumulative_mass` are differentiable analytically (Plummer)
or via the implicit-function theorem applied to the inverse-CDF lookup
(King, EFF). `sample_positions` uses a fixed-iteration `lax.scan`
inverse-CDF sampler so that the *positions themselves* are
differentiable in $r_h$ — useful when the loss function depends on
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
