---
title: Glossary
description: Definitions of every term used in the progenax docs — virial Q, half-mass radius, IMF, mass segregation, freefall-density factor, magnification factor, and more.
---
# Glossary

Definitions of common terms used in the progenax docs.

## Cluster dynamics

half-mass radius
    The radius $r_h$ enclosing half the total cluster mass:
    $M(<r_h) = M_{\mathrm{total}}/2$. progenax parameterises every
    spatial profile by $r_h$ for cross-profile comparability. See
    [](../10-theory/ic-philosophy.md).

scale radius
    The internal length scale of a spatial profile (Plummer's $a$,
    King's $r_c$, EFF's $a$). Profile-specific; converted to/from
    $r_h$ via closed-form or numerical mappings.

tidal radius
    The Jacobi radius $r_J$ at which the cluster's gravity balances
    the host galaxy's tidal field. Stars beyond $r_J$ are stripped.
    Computed by `progenax.tidal.jacobi_radius`. See
    [](../10-theory/tidal-and-substructure/tidal.md).

virial Q
    $Q_{\mathrm{vir}} \equiv T/|V|$, the ratio of kinetic to absolute
    potential energy. Equilibrium value: $0.5$ (virial theorem).
    progenax convention. See [](../20-architecture/q-virial-convention.md).

CW04 Q
    Cartwright & Whitworth (2004) substructure parameter: $Q = \bar
    m / \bar s$ from the minimum spanning tree. Distinct from the
    virial Q. See [](../20-architecture/jax-native-substructure-q.md).

virial equilibrium
    Dynamical state in which $2T + V = 0$, equivalently $Q_{\mathrm{vir}} =
    0.5$. progenax's default IC state.

subvirial
    $Q_{\mathrm{vir}} < 0.5$; cluster is collapsing. The {cite:t}`Allison2009`
    cool-fractal setup uses $Q \approx 0.3$.

supervirial
    $Q_{\mathrm{vir}} > 0.5$; cluster is expanding. Models post-gas-expulsion
    states.

centre-of-mass (COM) frame
    Frame in which $\sum_i m_i\,\mathbf{r}_i = 0$ and
    $\sum_i m_i\,\mathbf{v}_i = 0$. progenax always returns ICs
    in this frame.

## Mass functions

IMF
    Initial mass function $\xi(m) = \mathrm{d}N/\mathrm{d}m$. The
    birth-mass distribution of stars. See
    [](../10-theory/imfs/index.md).

Salpeter slope
    $\alpha = 2.35$. The high-mass slope of the IMF, established
    by {cite:t}`Salpeter1955`.

mass ratio
    $q = m_2/m_1 \in [0, 1]$ for a binary. Distribution $g(q | M_1)$
    follows {cite:t}`MoeDiStefano2017`.

binary fraction
    Probability that a primary has at least one companion. Mass-dependent;
    $\sim 0.5$ for solar-type, $\sim 0.9$ for O-type.

twin excess
    Narrow Gaussian peak at $q \approx 1$ in the {cite:t}`MoeDiStefano2017`
    mass-ratio distribution. Solar-type stars show the strongest
    excess, $f_{\mathrm{twin}} \approx 0.10$.

confidently wrong
    The regime where a misspecified likelihood produces a posterior
    whose 95% CI shrinks below the bias and *excludes* the true
    parameter value. Demonstrated for binary IMF inference at
    $N \gtrsim 10^4$ in [](../10-theory/imfs/binary.md).

## Substructure

fractal dimension
    $D \in [1.6, 3.0]$. Parameter of the {cite:t}`Goodwin2004`
    fractal IC. $D = 3$ uniform; $D = 1.6$ highly clumpy.

FDF method
    progenax's **Fractal Displacement Field** — differentiable
    replacement for the GW04 recursive tree. See
    [](../10-theory/tidal-and-substructure/fractal.md).

mass segregation
    Spatial arrangement where massive stars preferentially occupy
    central / low-energy orbits. Primordial (set at IC time, see
    {cite:t}`Baumgardt2008`) vs dynamical (emerges via two-body
    relaxation, {cite:t}`Allison2009`).

Λ_MSR
    {cite:t}`Allison2009` MST ratio for quantifying mass segregation.
    $\Lambda \sim 1$ for unsegregated; $\Lambda > 1$ for segregated.

## Gravoturbulence

density PDF
    Volume-density distribution $p_V(\rho)$ in a turbulent self-gravitating
    cloud. Lognormal core + power-law tail per
    {cite:t}`FederrathKlessen2012,Burkhart2018`.

Mach number
    Sonic Mach $\mathcal{M} = v_{\mathrm{turb}}/c_s$. Sets the
    lognormal variance via $\sigma_s^2 = \ln(1 + b^2\,\mathcal{M}^2)$.

forcing parameter
    $b \in [1/3, 1]$. Turbulence-driving geometry: $b = 1/3$
    solenoidal, $b = 1$ compressive, $b \approx 0.4$ natural mix.

freefall-density factor (FDF)
    The kernel $\rho/t_{\mathrm{ff}}(\rho) \propto \rho^{3/2}$
    that weights local density by its star-forming efficiency. See
    [](../10-theory/gravoturbulence/freefall-density-factor.md).

magnification factor
    $\zeta$ = SFR boost a centrally-concentrated cloud gets over a
    uniform top-hat. {cite:t}`ParmentierPasquali2020` Eq. 6 gives
    the closed form for power-law profiles. See
    [](../10-theory/gravoturbulence/pp20.md).

BM19 framework
    {cite:t}`Burkhart2018,Burkhart2021` forward model: turbulence
    parameters → density PDF → SFR. See
    [](../10-theory/gravoturbulence/bm19.md).

## JAX programming

PyTree
    A nested Python structure (dict, list, tuple, custom class)
    that JAX can trace through. progenax classes are PyTrees via
    `equinox.Module`.

JIT
    Just-in-time compilation via `@jax.jit`. Compiles a Python
    function to XLA, eliminating Python overhead for hot paths.

vmap
    `jax.vmap`. Vectorises a function over an axis without writing
    a loop. progenax uses this extensively for parallelisation
    over particles.

grad
    `jax.grad`. Automatic differentiation. The foundation of
    progenax's HMC inference capability.

scan
    `jax.lax.scan`. Fixed-iteration sequential loop primitive. Used
    instead of `while_loop` for differentiability. See
    [](../20-architecture/differentiability.md).

while-loop antipattern
    Using a data-dependent `jax.lax.while_loop` in code that needs
    gradients. Fixed-shape JAX loops are acceptable when gradients and
    static-shape compilation remain well-defined. See
    [](../20-architecture/differentiability.md).

## Architecture

SpatialProfile protocol
    Runtime-checkable protocol every spatial profile satisfies:
    `sample_positions` and `characteristic_radius`. See
    [](../20-architecture/protocols.md).

VelocityDF protocol
    Runtime-checkable protocol every velocity DF satisfies:
    `sample_velocities`. See
    [](../20-architecture/protocols.md).

IMFProtocol
    Runtime-checkable protocol every IMF satisfies: `logpdf`, `cdf`,
    `ppf`, `sample`, and `mean_mass`. See
    [](../20-architecture/protocols.md).

three-brick state
    A planned architecture pattern described in the design docs. The
    current public code does not expose `SystemParams` or
    `ParticleSystem`. See [](../20-architecture/three-brick-state.md).

DEFAULT_UNITS
    Per-package default unit system (STELLAR for progenax) used
    only by convenience wrappers. Core APIs require explicit units.
    See [](../20-architecture/units-policy.md).
