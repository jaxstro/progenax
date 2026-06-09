---
title: King profile
description: The King (1966) lowered isothermal sphere — its ODE-defined density profile, the central potential parameter W₀, the tidal radius, and its place in the lowered-model family.
---

# The King profile

The King model {cite:p}`King1966` is the canonical *tidally truncated*
spatial density profile for old globular clusters. Unlike Plummer, it
has no closed-form $\rho(r)$ — the profile is defined implicitly via
an ODE in the dimensionless gravitational potential. In exchange,
King provides what Plummer cannot: a finite outer radius matching the
observed tidal cutoff, a single dimensionless concentration parameter
$W_0$ that captures the full cluster geometry, and the family of
{cite:t}`King1966` model density profiles fit to most Galactic globular
clusters in the second half of the 20th century.

This chapter derives the King profile from its defining lowered-isothermal
DF, sets up the ODE that progenax integrates with `diffrax`, lists the
$W_0 \to (r_c, r_t, c, r_h)$ mapping, and notes the
{cite:t}`Gieles2015` lowered-model-family generalisation — progenax's planned
own differentiable extension when multi-mass anisotropy matters (not currently
implemented).

## The lowered-isothermal distribution function

The King profile starts from a *distribution function*, not a density
profile. Define a dimensionless potential

```{math}
:label: king-W
W(r) \;\equiv\; \frac{\Phi(r_t) - \Phi(r)}{\sigma_0^2}
```

where $\sigma_0$ is the central one-dimensional velocity dispersion
and $r_t$ is the tidal (truncation) radius — the location where the
gravitational potential of the cluster matches the host galaxy's
tidal potential. By construction $W(r_t) = 0$ and $W$ increases
monotonically inward, with $W_0 \equiv W(0)$ the central value.

The lowered-isothermal DF is

```{math}
:label: king-df
f(E) \;=\;
\begin{cases}
  \rho_1\,(2\pi\sigma_0^2)^{-3/2}\,\bigl[\,e^{(\Phi(r_t)-E)/\sigma_0^2} - 1\,\bigr], & E < \Phi(r_t) \\
  0, & E \ge \Phi(r_t)
\end{cases}
```

The "$-1$" in the bracket is the *lowering*: it removes the high-energy
tail of the Maxwell-Boltzmann distribution that would otherwise extend
to $E \to \infty$, producing a finite cluster with a sharp tidal cutoff.
{cite:t}`King1966` derived this form to model the observation that
globular clusters appear truncated rather than extending as
infinite-mass isothermal spheres.

Integrating $f(E)$ over velocity space gives the density profile:

```{math}
:label: king-rho-W
\rho(W) \;=\; \rho_1 \cdot
\bigl[\,e^W\,\mathrm{erf}(\sqrt{W}) - \sqrt{4W/\pi}\,(1 + \tfrac{2}{3}W)\,\bigr]
```

where $\mathrm{erf}$ is the error function. $\rho(W = 0) = 0$ (cluster
edge) and $\rho(W = W_0) = \rho_0$ at the centre. Note that $\rho$ is
not yet a function of *radius* — that requires solving the Poisson
equation for $W(r)$.

## The defining ODE

Combining {eq}`king-rho-W` with the spherical Poisson equation,

```{math}
:label: king-poisson
\frac{1}{r^2}\frac{\mathrm{d}}{\mathrm{d}r}\biggl(r^2 \frac{\mathrm{d}\Phi}{\mathrm{d}r}\biggr) \;=\; 4\pi G\,\rho(\Phi)
```

and non-dimensionalising via $\xi = r / r_c$ where $r_c$ is the King
*core radius*

```{math}
:label: king-rc
r_c^2 \;=\; \frac{9\,\sigma_0^2}{4\pi G\,\rho_0},
```

yields the dimensionless King ODE:

```{math}
:label: king-ode
\frac{1}{\xi^2}\frac{\mathrm{d}}{\mathrm{d}\xi}\biggl(\xi^2 \frac{\mathrm{d}W}{\mathrm{d}\xi}\biggr)
\;=\; -\frac{9\,\rho(W)}{\rho_0}
```

with boundary conditions $W(0) = W_0$ and $W'(0) = 0$. The ODE
integrates outward from $\xi = 0$ until $W$ reaches zero — that
defines the tidal radius $r_t = \xi_t \cdot r_c$. The $W_0$ parameter
fully determines the cluster's dimensionless structure.

```{admonition} Why no closed form
:class: note
The right-hand side of {eq}`king-ode` involves $\rho(W)$ in the form
{eq}`king-rho-W`, which combines $e^W$ and $\mathrm{erf}(\sqrt{W})$ with
algebraic terms. No combination of standard special functions inverts
this ODE, so $W(\xi)$ must be evaluated numerically. progenax
integrates {eq}`king-ode` once per $W_0$ value with `diffrax`'s adaptive
Tsit5 stepper, then uses the result as a lookup table for inverse-CDF
sampling.
```

## The concentration parameter $c$

The shape of a King cluster is captured by the dimensionless
concentration parameter

```{math}
:label: king-c
c \;\equiv\; \log_{10}\!\left(\frac{r_t}{r_c}\right) \;=\; \log_{10}(\xi_t).
```

For Galactic globular clusters, observational fits give $c$ in the
range $\sim 0.7$–$2.5$, corresponding to $W_0 \sim 5$–$10$:

```{list-table} $W_0 \to (\xi_t, c, r_h/r_c)$ mapping for the King profile. The $\xi_t$ and $c$ columns reproduce King (1966) Table II; $r_h/r_c$ is computed from the integrated mass profile.
:header-rows: 1

* - $W_0$
  - $\xi_t = r_t/r_c$
  - $c = \log_{10} \xi_t$
  - $r_h / r_c$
  - Cluster type
* - 3
  - 4.70
  - 0.67
  - 1.26
  - Diffuse, low-density
* - 5
  - 10.70
  - 1.03
  - 2.00
  - Typical low-concentration GC
* - 7
  - 33.7
  - 1.53
  - 3.92
  - Average Milky Way GC
* - 9
  - 131
  - 2.12
  - 15.4
  - High-concentration GC
```

The $r_h$ column shows that — unlike Plummer's $a/r_h \approx 0.766$
constant — the King profile's scale-to-half-mass mapping depends on
$W_0$. progenax stores this mapping as a precomputed lookup table and
exposes it through `progenax.profiles.solve_king_profile(W_0)` which
returns `(xi_grid, psi_grid)`, the dimensionless radius grid and
dimensionless potential trajectory used by the profile implementation.

## Inverse-CDF sampling

Once $W(\xi)$ is integrated, the cumulative mass

```{math}
:label: king-mcum
M(<\xi) \;=\; 4\pi r_c^3 \rho_0 \int_0^\xi \xi'^2\,\frac{\rho(W(\xi'))}{\rho_0}\,\mathrm{d}\xi'
```

is computed by trapezoidal integration on the same grid the ODE
solver produced. progenax then inverts this CDF to sample radii:

```python
u = jax.random.uniform(key, (N,))                        # u ~ U(0,1)
xi_samples = jnp.interp(u, cumulative_mass_normalized, xi_grid)
r_samples = xi_samples * r_c                              # r in physical units
```

The whole chain is differentiable in $r_c$ (and therefore in $r_h$ via
the $r_h(W_0, r_c)$ mapping). It is **also** differentiable in $W_0$:
`diffrax` propagates $\partial\psi/\partial W_0$ through the ODE solve, so the
density profile and any shape-based observable carry correct $W_0$ gradients
(validated against finite differences in
[](../../50-validation/king-profile.md)). The one exception is the *scalar*
tidal radius $r_t$: its $W_0$-derivative is zeroed by the `argmax` zero-crossing
in `_find_tidal_radius`, so $W_0$ inference should target the profile *shape*
(differentiable) rather than the scalar $r_t$ readout. This makes joint
gradient-based / HMC inference of $(W_0, r_c)$ — and, with the velocity DF,
$M_{\rm tot}$ — feasible.

## The lowered-model family

{cite:t}`Gieles2015` showed that King is one member of a one-parameter family of
lowered-isothermal models: a continuous truncation parameter $g$ interpolates
between Woolley ($g = 0$), King ($g = 1$), and Wilson ($g = 2$) models, and the
framework extends to multi-mass and anisotropic clusters. progenax will
implement this family **natively** as its own differentiable, JAX-native
generalization — so that $g$ (and the multi-mass / anisotropy parameters) can be
*inferred* — rather than depending on the external (non-differentiable) `limepy`
package. See the [roadmap](lowered-model-family.md).

For most production work — single-mass populations or coarse mass
binning — the released King profile suffices. The unified family becomes the
right tool when the science target is the radial velocity-anisotropy profile or
the multi-mass equipartition state.

## Tidal physics

The King profile's "tidal radius" $r_t$ is a *defining* property of
the model — the place where $W = 0$ and the cluster terminates. In
real space, the physical tidal radius (Jacobi radius) of a cluster
on a circular orbit at galactocentric radius $R$ is

```{math}
:label: jacobi
r_J \;\approx\; \biggl(\frac{M_{\mathrm{cl}}}{3 M_{\mathrm{gal}}(<R)}\biggr)^{\!1/3} R
```

`progenax.tidal.jacobi_radius` computes $r_J$ given a Galactic mass
model; equating $r_t = r_J$ then fixes the King concentration self-consistently.
Tidal physics is documented in detail at
[](../tidal-and-substructure/tidal.md).

## Implementation in progenax

```python
from progenax.profiles import KingProfile, solve_king_profile
from progenax.kinematics import KingVelocityDF
from jaxstro.units import STELLAR

# Solve the King ODE once for W_0 = 7
xi_grid, psi_grid = solve_king_profile(W0=7.0)      # diffrax Tsit5
profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
df = KingVelocityDF(W0=7.0, r_c=1.0, r_t=profile.r_t)

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

The `solve_king_profile` call is the ODE helper for the King profile; it
returns the `(xi_grid, psi_grid)` arrays used by `KingProfile`.

See [](../../30-api/profiles.md) for the full signature and
[](../../50-validation/king-profile.md) for the regression suite, which
validates $c(W_0)$ against {cite:t}`King1966` Table II and the volume
density against an independent direct-velocity-integral oracle.

## Domain of validity and limitations

1. **Single-mass only.** The lowered-isothermal DF assumes one mass
   species. Multi-mass clusters in equilibrium have radially-varying
   velocity dispersions per mass group; the planned
   [lowered-model family](lowered-model-family.md) will handle this natively,
   while progenax's standard King does not.

2. **Spherical and isotropic.** No rotation, no anisotropy. For
   anisotropic generalisations see [](../velocity-dfs/rotation-anisotropy.md)
   and the anisotropic [Michie](../velocity-dfs/michie-king.md) model; rotating
   King-like profiles are part of the planned lowered-model family.

3. **No primordial substructure.** King is a smooth equilibrium
   profile. Turbulent/fractal substructure — relevant to cool-clumpy
   initial conditions {cite:p}`Allison2009` — is provided separately by
   the experimental `gravoturb_fdf` subsystem
   ([](../tidal-and-substructure/fractal.md)), not by the released King
   profile, and deliberately breaks the equilibrium assumption.

4. **Tidal radius is fixed at IC time.** A cluster's tidal radius
   evolves with its galactocentric orbit; progenax's King IC fixes
   $r_t$ at $t = 0$ and the user must apply post-evolution truncation
   manually if the orbit drives the cluster across a tidal-radius
   threshold.

## References

The original lowered-isothermal model is {cite:t}`King1966`; the LIMEPY
generalisation is {cite:t}`Gieles2015`. The numerical-integration
scheme follows the standard {cite:t}`Aarseth1974` approach; progenax
uses `diffrax` as the JAX-native ODE backend.
