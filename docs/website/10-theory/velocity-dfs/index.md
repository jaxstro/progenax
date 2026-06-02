---
title: Velocity distribution functions
description: Velocity DFs in progenax — Plummer / King / EFF equilibrium distributions, plus anisotropy and rotation extensions.
---

# Velocity distribution functions

A spatial density profile alone does not produce a complete IC: every
particle needs a velocity. The **velocity distribution function** (DF)
specifies how velocities are sampled given particle positions and
masses. progenax implements the canonical equilibrium DFs for each of
its three spatial profiles, plus extensions for radial anisotropy
({cite:t}`Plummer1911` Osipkov-Merritt) and rigid/differential rotation.

```{list-table}
:header-rows: 1

* - Velocity DF
  - Pairs with profile
  - Defines
* - [](plummer-dfs.md)
  - [](../spatial-profiles/plummer.md)
  - $f(E) \propto (-E)^{7/2}$ from Eddington inversion of Plummer
* - [](king-dfs.md)
  - [](../spatial-profiles/king.md)
  - Lowered-Maxwellian $f(E) = \rho_1 (2\pi\sigma_0^2)^{-3/2} [e^{(\Phi_t - E)/\sigma_0^2} - 1]$
* - `EFFVelocityDF`
  - [](../spatial-profiles/eff.md)
  - Numerically-evaluated isotropic DF from EFF $\rho(r)$ via Eddington inversion
* - [](rotation-anisotropy.md)
  - any of the above
  - Osipkov-Merritt anisotropy radius $r_a$, solid-body rotation $\Omega$, differential-rotation profile $\Omega(r)$
```

## What "equilibrium" means

A spatial profile $\rho(r)$ paired with the wrong velocity DF produces
a *non-equilibrium* IC: the cluster will dynamically relax towards a
new equilibrium state via violent relaxation on the first crossing
time. For most production work this is undesirable — you wanted a
Plummer cluster at $t = 0$, not a Plummer-like distribution heading
toward something else. The fix is to use the matched equilibrium DF.

```{admonition} How to check if a DF is in equilibrium with a profile
:class: note
A density profile $\rho(r)$ and a velocity DF $f(\mathbf{x}, \mathbf{v})$
are in equilibrium if the **Jeans equation** holds:

```{math}
\frac{1}{\rho}\frac{\mathrm{d}(\rho\sigma_r^2)}{\mathrm{d}r}
+ \frac{2\,\beta\,\sigma_r^2}{r}
+ \frac{\mathrm{d}\Phi}{\mathrm{d}r} \;=\; 0
```

with $\beta = 1 - \sigma_t^2/\sigma_r^2$ the anisotropy parameter (zero
for isotropic). progenax's validation suite
([](../../50-validation/methodology.md)) checks this directly for
sampled positions+velocities.
```

## When you actually want a non-equilibrium IC

Three deliberate non-equilibrium configurations are useful enough to
deserve mention:

1. **Subvirial / cold cluster**, $Q_{\mathrm{vir}} \approx 0.3$ — the
   {cite:t}`Allison2009` cool-fractal setup. Cluster contracts on a
   crossing time and produces *dynamical* mass segregation within
   $\sim 1$ Myr.
2. **Supervirial / hot cluster**, $Q_{\mathrm{vir}} \approx 0.75$ —
   post-gas-expulsion. Cluster expands, may unbind partially.
3. **Mismatched-DF cluster** — Plummer profile with a King DF, or
   vice versa. Useful for testing the *rate* of violent relaxation
   without contaminating the experiment with non-equilibrium energy
   ratio.

In all three cases, the spatial profile and base DF are sampled
*independently*, then the velocity field is rescaled to the target
$Q_{\mathrm{vir}}$ via [](../../20-architecture/q-virial-convention.md).
Each non-equilibrium configuration is a deliberate scientific choice;
progenax exposes the parameters so the configuration can be made
explicit at IC construction time.

## Common API contract

Every velocity DF satisfies the `VelocityDF` protocol:

```python
class VelocityDF(Protocol):
    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKey,
        *,
        G: float,
    ) -> Float[Array, "N 3"]:
        """Draw N velocities consistent with the DF given positions+masses."""
        ...

    def velocity_dispersion(
        self, r: Float[Array, "..."], *, G: float
    ) -> Float[Array, "..."]:
        """Closed-form or numerically-evaluated σ_r(r)."""
        ...
```

`sample_velocities` is differentiable in $r_h$ via the chain $r_h \to
\sigma_r(r) \to \mathbf{v}$. `velocity_dispersion` is differentiable
analytically (Plummer) or via implicit-function / fixed-quadrature
machinery (King, EFF). All three DFs are JIT-compatible and
vectorisable via `jax.vmap`.

## References

The Eddington-inversion machinery ({cite:t}`Plummer1911`,
{cite:t}`King1966`) is standard textbook material; for a clean
review see {cite:t}`Aarseth1974` Section 3 and
{cite:t}`Gieles2015` Section 2 (covering LIMEPY's multi-mass
extension).
