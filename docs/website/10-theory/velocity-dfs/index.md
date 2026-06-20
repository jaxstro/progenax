---
title: Velocity distribution functions
description: Velocity DFs in progenax — Plummer / King / EFF equilibrium distributions, plus anisotropy and rotation extensions.
---

# Velocity distribution functions

A spatial density profile alone does not produce a complete IC: every
particle needs a velocity. The **velocity distribution function** (DF)
specifies how velocities are sampled given particle positions and
masses. progenax implements the canonical equilibrium DFs for its
spatial profiles (Plummer, King, EFF), the self-consistent radially
anisotropic **Michie–King** DF, plus extensions for Osipkov–Merritt
radial anisotropy ({cite:t}`Merritt1985`) and rigid/differential rotation.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers choosing a velocity distribution function and learning how progenax turns positions into equilibrium velocities; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [spatial profiles](../spatial-profiles/index.md) (a DF samples velocities *given* sampled positions) and the [IC philosophy](../ic-philosophy.md) (virial $Q$, units). The per-DF pages ([Plummer](plummer-dfs.md), [King](king-dfs.md), [Michie–King](michie-king.md)) go deeper.
**You'll get:** what a velocity DF is, what "equilibrium" means (and when you want a non-equilibrium IC), the shared `sample_velocities` contract, and how DFs extend to anisotropy, rotation, and multi-component potentials.
:::

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
* - [](michie-king.md)
  - King (anisotropic, self-consistent)
  - Michie–King $f(E,J) \propto e^{-J^2/2r_a^2\sigma^2}[e^{-E/\sigma^2}-1]$ — radial anisotropy, distinct density
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

Every velocity DF satisfies the `VelocityDF` protocol, whose **single
required method** is `sample_velocities`:

```python
class VelocityDF(Protocol):
    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKey,
        G: float,
    ) -> Float[Array, "N 3"]:
        """Draw N velocities consistent with the DF given positions+masses."""
        ...
```

That is the whole contract — there is **no** `velocity_dispersion` method
on the protocol or on the concrete DF classes; the dispersion profiles
shown in the per-DF chapters are derived analytically (Plummer) or built
internally by the sampler (King, EFF, Michie) and are not part of the
public surface. `sample_velocities` is differentiable in the model
parameters via the chain (e.g. $r_h \to \sigma_r(r) \to \mathbf{v}$), and
every DF is JIT-compatible and vectorisable via `jax.vmap`.

## Multiple components in one potential

Everything above is **single-component**: one DF in its own
self-consistent potential. For clusters with *multiple* populations
sharing **one** potential — GC 1G/2G, halo+core, multi-mass
equipartition — the right tool is `MultiComponentCluster`
([](../populations/index.md)), which provides two equilibrium
engines: the DF-defined lowered-isothermal family
([Engine A](../spatial-profiles/lowered-model-family.md)) and
density-defined Eddington inversion in a shared potential
([Engine B](../populations/eddington-engine.md)). Per-component
equilibrium ($Q_j = 0.5$) emerges from the DFs with no external
rescale.

```{admonition} Why trust two engines?
:class: tip
The engines share one configuration — a single King component
($g = 1$ DF in Engine A; King *density* in Engine B) — built through
entirely disjoint numerics. They agree to a radial KS distance of
$2\times 10^{-4}$ and a velocity-dispersion-profile deviation of
$3\times 10^{-4}$: the cross-engine trust anchor for the
multi-component machinery.
```

## Implementation, validation & references

- **In code:** the velocity DFs live under `src/progenax/kinematics/`
  (`plummer_df.py`, `king_df.py`, `eff_df.py`, `michie_df.py`, plus
  `rotation.py` and the Osipkov–Merritt anisotropy options). See the
  [kinematics API](../../30-api/kinematics.md); the per-DF chapters
  ([Plummer](plummer-dfs.md), [King](king-dfs.md),
  [Michie–King](michie-king.md), [anisotropy & rotation](rotation-anisotropy.md))
  carry the exact module paths.
- **Validated in:** [Plummer equilibrium](../../50-validation/plummer-equilibrium.md),
  [King profile](../../50-validation/king-profile.md),
  [EFF profile](../../50-validation/eff-profile.md),
  [Michie anisotropy](../../50-validation/michie-anisotropy.md), and
  [OM / rotation anisotropy](../../50-validation/rotation-om-anisotropy.md).
- **Primary sources:** the Eddington-inversion machinery
  ({cite:t}`Plummer1911`, {cite:t}`King1966`) is standard textbook
  material; the anisotropy extensions are {cite:t}`Merritt1985`
  (Osipkov–Merritt) and {cite:t}`Michie1963` (self-consistent King),
  and {cite:t}`Gieles2015` covers the multi-mass lowered-model family.
  Full notes in the [bibliography](../../99-bibliography/index.md).
