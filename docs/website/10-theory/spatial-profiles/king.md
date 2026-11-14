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
{cite:t}`Gieles2015` lowered-model family implemented by LIMEPY and Engine A
for multimass and anisotropic models.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the King lowered-isothermal model and tidal truncation; no prior globular-cluster-dynamics literature assumed.
**Prerequisites:** the [Plummer profile](plummer.md) (the closed-form baseline this generalizes) and the [virial-$Q$ convention](../../20-architecture/q-virial-convention.md).
**You'll get:** why the King profile starts from a *distribution function*, the ODE progenax integrates to get $\rho(r)$, the $W_0 \to (r_c, r_t, c, r_h)$ mapping, and how the tidal radius and concentration are defined.
:::

## The lowered-isothermal distribution function

The King profile starts from a *distribution function*, not a density
profile. Define a dimensionless potential

```{math}
:label: king-W
W(r) \;\equiv\; \frac{\Phi(r_t) - \Phi(r)}{\sigma_0^2}
```

where $\sigma_0$ is the DF velocity-scale parameter, not the actual central
one-dimensional dispersion at finite $W_0$. The model's truncation radius
$r_t$ is where the relative potential vanishes; matching it to a Galactic
Jacobi radius requires an external tidal model. By construction $W(r_t)=0$,
and $W_0\equiv W(0)$ is the central dimensionless potential. Here $\Phi$
and $E$ below are potential and energy per unit mass, and $\rho_1$ is a
density normalization.

The functional potential API sets $\Phi(r_t)=0$. With total cluster mass $M$
and gravitational constant $G$, outside the density support it returns the spherical vacuum continuation
$\Phi(r)=GM(1/r_t-1/r)$, which is positive for $r>r_t$ and still gives inward
gravity. The clamped nonnegative $W$ table used for density and sampling is a
separate numerical quantity. See the [exterior-potential convention](index.md#profile-vacuum-potential).

The lowered-isothermal DF is

```{math}
:label: king-df
f(E) \;=\;
\begin{cases}
  \rho_1\,(2\pi\sigma_0^2)^{-3/2}\,\bigl[\,e^{(\Phi(r_t)-E)/\sigma_0^2} - 1\,\bigr], & E < \Phi(r_t) \\
  0, & E \ge \Phi(r_t)
\end{cases}
```

[↗ model card](#card-king-df)

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

[↗ model card](#card-king-rho-W)

where $\mathrm{erf}$ is the error function. $\rho(W = 0) = 0$ (cluster
edge) and $\rho(W = W_0) = \rho_0$ at the center. Note that $\rho$ is
not yet a function of *radius* — that requires solving the Poisson
equation for $W(r)$.

## The defining ODE

Combining {eq}`king-rho-W` with the spherical Poisson equation,

```{math}
:label: king-poisson
\frac{1}{r^2}\frac{\mathrm{d}}{\mathrm{d}r}\biggl(r^2 \frac{\mathrm{d}\Phi}{\mathrm{d}r}\biggr) \;=\; 4\pi G\,\rho(\Phi)
```

and nondimensionalizing via $\xi=r/r_c$, where $r_c$ is the King scale
radius (not exactly an observational half-central-surface-density radius),

```{math}
:label: king-rc
r_c^2 \;=\; \frac{9\,\sigma_0^2}{4\pi G\,\rho_0},
```

[↗ model card](#card-king-rc)

yields the dimensionless King ODE:

```{math}
:label: king-ode
\frac{1}{\xi^2}\frac{\mathrm{d}}{\mathrm{d}\xi}\biggl(\xi^2 \frac{\mathrm{d}W}{\mathrm{d}\xi}\biggr)
\;=\; -\frac{9\,\rho(W)}{\rho_0}
```

with boundary conditions $W(0)=W_0$ and $W'(0)=0$. The first zero
$\xi_t$ defines the truncation radius $r_t=\xi_t r_c$. The parameter
$W_0$ determines the dimensionless structure within this spherical,
single-mass model.

### Regular central boundary

Spherical symmetry permits an even expansion $W=W_0+a\xi^2+O(\xi^4)$.
Then $W''+2W'/\xi=6a+O(\xi^2)$, while $\rho(W_0)/\rho_0=1$.
Consequently $6a=-9$, giving

```{math}
:label: king-center-series
\begin{aligned}
W(\xi)&=W_0-\frac{3}{2}\xi^2+O(\xi^4),\\
W'(\xi)&=-3\xi+O(\xi^3),\\
W''(0)&=-3.
\end{aligned}
```

The geometric term has the finite limit $2W'/\xi\to-6$. The factor
$9$ in the Poisson normalization therefore remains unchanged. At a positive
starting radius $\epsilon$, both initial values must approximate this
regular solution: $W(\epsilon)=W_0-3\epsilon^2/2+O(\epsilon^4)$ and
$W'(\epsilon)=-3\epsilon+O(\epsilon^3)$. The central curvature alone
does not determine the error in a saved profile, tidal radius, or derivative;
those also depend on the start, adaptive solve, saved grid, and interpolation.

```{admonition} Why no closed form
:class: note
The right-hand side of {eq}`king-ode` involves $\rho(W)$ in the form
{eq}`king-rho-W`, which combines $e^W$ and $\mathrm{erf}(\sqrt{W})$ with
algebraic terms. Progenax evaluates $W(\xi)$ numerically rather than using a closed-form
radial solution. It
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

[↗ model card](#card-king-c)

For Galactic globular clusters, observational fits give $c$ in the
range $\sim 0.7$–$2.5$, with $W_0 \sim 5$–$9$ for most clusters (the
high-concentration / near-core-collapse tail reaching higher):

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
exposes it through `progenax.profiles.solve_king_profile(W0)`, which
returns the **3-tuple** `(xi_grid, psi_clamped, psi_raw)`: the
dimensionless radius grid, the potential clamped to $\psi \ge 0$ (used for
the density and CDF), and the *unclamped* potential (negative just past
$r_t$; used only for the differentiable tidal-radius crossing). The
profile implementation consumes the first two.

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

At a fixed random key, the construction supports continuous scale and
shape derivatives on documented paths. Diffrax's selected adjoint propagates
potential sensitivities. The scalar tidal radius uses the unclamped potential
and an implicit derivative of a monotone-cubic zero crossing; it requires a
bracketed simple zero. Neither AD nor a finite derivative removes ODE, grid,
or interpolation error. See the [derivative contract](../../20-architecture/differentiability.md#roadmap-differentiable-rt)
and [retained checks](../../50-validation/king-profile.md).

## The lowered-model family

{cite:t}`Gieles2015` showed that King is one member of a one-parameter family of
lowered-isothermal models: a continuous truncation parameter $g$ interpolates
between Woolley ($g = 0$), King ($g = 1$), and Wilson ($g = 2$) models, and the
framework extends to multimass and anisotropic clusters. Progenax implements
this family in its LIMEPY profiles and Engine A. The external `limepy` package
serves as a comparison rather than a runtime dependency. See the
[lowered-model family](lowered-model-family.md) and its validation matrix.

For most production work — single-mass populations or coarse mass
binning — the released King profile suffices. The unified family becomes the
right tool when the science target is the radial velocity-anisotropy profile or
the multi-mass equipartition state.

## Tidal physics

The King profile's "tidal radius" $r_t$ is a *defining* property of
the model — the place where $W = 0$ and the cluster terminates. In
the point-mass host approximation, the Jacobi radius of a cluster
on a circular orbit at distance $R$ from a dominant host mass $M_{\mathrm{gal}}$ is

```{math}
:label: jacobi
r_J \;\approx\; \biggl(\frac{M_{\mathrm{cl}}}{3 M_{\mathrm{gal}}(<R)}\biggr)^{\!1/3} R
```

`progenax.tidal.jacobi_radius` implements this point-mass, circular-orbit
approximation. An extended host potential can change the coefficient; for an
isothermal halo use `jacobi_radius_isothermal`. Matching $r_t$ to $r_J$ is an
additional tidal-model constraint, not a condition solved by the King constructor.
Tidal physics is documented in detail at
[](../tidal-and-substructure/tidal.md).

## Implementation in progenax

```python
import jax
import jax.numpy as jnp
from progenax.profiles import KingProfile, solve_king_profile
from progenax.kinematics import KingVelocityDF
from jaxstro.units import STELLAR

key_pos, key_vel = jax.random.split(jax.random.PRNGKey(0))

# Solve the King ODE once for W_0 = 7 (returns a 3-tuple; psi_raw unused here)
xi_grid, psi_grid, _ = solve_king_profile(W0=7.0)   # diffrax Tsit5
profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
df = KingVelocityDF(W0=7.0, r_c=1.0)                 # r_t derived from W0

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

The `solve_king_profile` call is the ODE helper for the King profile; it
returns the 3-tuple `(xi_grid, psi_clamped, psi_raw)` — here we keep the
clamped potential and discard `psi_raw`. `KingVelocityDF` takes only
`(W0, r_c)` and re-solves the ODE internally (deriving $r_t$ from $W_0$), so
no `r_t` argument is passed.

## Domain of validity and limitations

1. **Single-mass only.** The lowered-isothermal DF assumes one mass
   species. Multi-mass clusters in equilibrium have radially-varying
   velocity dispersions per mass group; use the implemented
   [lowered-model family](lowered-model-family.md) for that model rather than
   interpreting the standard King profile as a multimass equilibrium.

2. **Spherical and isotropic.** No rotation, no anisotropy. For
   anisotropic generalizations see [](../velocity-dfs/rotation-anisotropy.md)
   and the anisotropic [Michie](../velocity-dfs/michie-king.md) model. Additive
   rotation overlays change the velocities and energy; they do not construct a
   stationary rotating King DF.

3. **No primordial substructure.** King is a smooth equilibrium
   profile. Turbulent/fractal substructure — relevant to cool-clumpy
   initial conditions {cite:p}`Allison2009` — is provided separately by
   the experimental `gravoturb` subsystem
   ([](../tidal-and-substructure/fractal.md)), not by the released King
   profile, and deliberately breaks the equilibrium assumption.

4. **Tidal radius is fixed at IC time.** A cluster's tidal radius
   evolves with its galactocentric orbit; progenax's King IC fixes
   $r_t$ at $t = 0$ and the user must apply post-evolution truncation
   manually if the orbit drives the cluster across a tidal-radius
   threshold.

## Implementation, validation & references

- **In code:** `src/progenax/profiles/king.py` — the King ODE solve
  (`solve_king_profile`), $W_0 \to (r_c, r_t, c)$ mapping, and inverse-CDF
  sampling; the paired DF is in `src/progenax/kinematics/king_df.py`. See
  the [`KingProfile` API](../../30-api/profiles.md).
- **Validated in:** [King profile](../../50-validation/king-profile.md) —
  validates $c(W_0)$ against {cite:t}`King1966` Table II and the volume
  density against an independent direct-velocity-integral oracle.
- **Primary sources:** {cite:t}`King1966` (the lowered-isothermal model),
  with the {cite:t}`Gieles2015` LIMEPY generalization (see the
  [lowered-model family](lowered-model-family.md)); the integration scheme
  follows the standard {cite:t}`Aarseth1974` approach (progenax uses
  `diffrax` as the JAX-native ODE backend). Full notes in the
  [bibliography](../../99-bibliography/per-paper/king-1966.md).
