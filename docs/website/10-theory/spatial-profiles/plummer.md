---
title: Plummer profile
description: The Plummer density, mass, potential, and isotropic distribution function, with sampling methods and implementation limits.
---

:::{admonition} Who this page is for
:class: note

- **Audience:** Students and researchers learning the Plummer model and cluster initial conditions.
- **Prerequisites:** Basic calculus and Newtonian gravity; no prior stellar-dynamics background.
- **You’ll learn:** The density, mass and potential; how the scale and half-mass radii relate; and how to sample positions.
:::

## The density profile

The Plummer model {cite:p}`Plummer1911` describes a spherical cluster with a
finite central density and an extended envelope whose density falls as $r^{-5}$.
Its analytic mass profile, potential, and isotropic distribution function make it
a useful starting point for understanding cluster initial conditions.

The Plummer profile is

```{include} ../../15-model-reference/_plummer-density.md
```

[↗ model card](#card-plummer-rho)

where $r$ is distance from the cluster center, $M$ is the total mass,
and $a$ is the **Plummer scale radius**: the
characteristic length that sets the size of the density profile. The ratio $r/a$
measures distance in units of that length. At fixed mass, increasing $a$ spreads
the same mass over a larger volume and lowers the central density as $a^{-3}$.
The scale radius marks the transition from the nearly flat central density at
$r\ll a$ to the steep outer envelope at $r\gg a$; it is not an outer boundary.
The profile is centrally smooth ($\rho(0) = 3M/(4\pi a^3) < \infty$),
asymptotically $\rho \propto r^{-5}$ at $r \gg a$, and integrates to
finite total mass even though $r$ extends to infinity. The outer slope and absence of a finite boundary are model assumptions;
choose a profile whose structure matches the intended physical problem.

See the implementation notes below for the distinction between physical mass
density and the dimensionless shape returned by `PlummerProfile.density`.

:::{figure} ../figures/plummer_projected_density.webp
:label: fig-plummer-projected-density
:alt: Two panels show the projected Plummer density and its radial surface-density curve. Dashed markers identify the Plummer scale radius a, also the projected half-mass radius; dotted markers identify the three-dimensional half-mass radius r_h. Both density scales are logarithmic.
:width: 100%

**Left:** projected surface density $\Sigma(R)=M[\pi a^2]^{-1}(1+R^2/a^2)^{-2}$,
obtained by integrating the volume density along the line of sight.
Here $R=\sqrt{x^2+y^2}$ is projected radius and $x,y$ are coordinates on the image.
**Right:** the same density as a radial profile. Both panels show $\Sigma/\Sigma(0)$
on logarithmic scales. Dashed markers indicate $a$; dotted markers indicate the
three-dimensional half-mass radius $r_h$ ([radius definitions below](#plummer-projected-radius)).
The image spans $\pm5a$ on each axis; its edge is a display boundary, not a physical truncation.
:::

## Closed-form mass and potential

Integrating {eq}`plummer-rho` against the spherical volume element
gives the cumulative mass:

```{math}
:label: plummer-mcum
M(<r) \;=\; M\,\frac{r^3}{(r^2 + a^2)^{3/2}}
```

[↗ model card](#card-plummer-mcum)

The total mass is recovered as $r \to \infty$: $M(<\infty) = M$. Solving
$M(<r_h) = M/2$ for the half-mass radius gives the relation that
recurs throughout progenax,

```{include} ../../15-model-reference/_plummer-scale.md
```

[↗ model card](#card-plummer-rh-a)

so that progenax's user-facing `r_h` parameter unambiguously specifies
the radius enclosing half the cluster mass, regardless of the internal
scale-radius convention.

(plummer-projected-radius)=
:::{admonition} Why the projected half-mass radius is different
:class: note

The $r_h$ derived above encloses half the mass inside a **sphere**:
$r_h\simeq1.305\,a$. In an image, we instead measure projected radius $R$.
A circle on that image includes material along the entire line of sight,
including stars outside the sphere of the same radius.

For Plummer, integrating the projected surface density gives

```{math}
M_{\mathrm{proj}}(<R)=M\frac{R^2}{R^2+a^2},
\qquad M_{\mathrm{proj}}(<a)=\frac{M}{2}.
```

Thus the **scale radius $a$ also equals the projected half-mass radius** for
this model. It is smaller than the three-dimensional half-mass radius $r_h$.
The two markers in the projected-density figure show these distinct lengths.
:::

The gravitational potential $\Phi(r)$, with $G$ the gravitational constant, is

```{math}
:label: plummer-phi
\Phi(r) \;=\; -\frac{G\,M}{\sqrt{r^2 + a^2}}
```

with central value $\Phi(0) = -GM/a$ and asymptotic $\Phi(r) \to 0$ as
$r \to \infty$. The escape speed at radius $r$ is

```{math}
:label: plummer-vesc
v_{\mathrm{esc}}(r) \;=\; \sqrt{-2\Phi(r)}
\;=\; \sqrt{\frac{2GM}{\sqrt{r^2 + a^2}}}\,.
```

## Inverse-CDF sampling

For Monte Carlo IC generation {cite:p}`Aarseth1974`, progenax uses inverse-CDF sampling on
{eq}`plummer-mcum`. Setting $u = M(<r) / M$ for $u \sim \mathcal{U}(0, 1)$
and inverting:

```{include} ../../15-model-reference/_plummer-inverse.md
```

The analytic inverse applies for $0<u<1$.
Choose a direction $\hat{\mathbf n}$ uniformly over the sphere, independently
of $u$, and set the position vector $\mathbf{x}=r(u)\hat{\mathbf n}$.
Holding both random draws fixed, positions scale linearly with $r_h$:

```{math}
\frac{\partial\mathbf{x}}{\partial r_h}=\frac{\mathbf{x}}{r_h}
```

This is a fixed-draw derivative; it does not establish a population-expectation
gradient. The inverse-CDF radius diverges as $u\to1$.


```{figure} ../figures/plummer_density_to_radius.webp
:label: fig-plummer-density-to-radius
:alt: Three analytic curves show central density declining with radius, shell probability peaking away from the center, and cumulative mass reaching one half at the half-mass radius.
:width: 100%

The Plummer density peaks at the center, but the probability of drawing a radius
includes the spherical shell volume. Integrating this radial probability gives
the cumulative mass fraction; inverse-CDF sampling reverses that mapping.
Here $F(r)=M(<r)/M$ is the cumulative mass fraction, so $F(r_h)=1/2$.
The radial probability density $p_r(r)=4\pi r^2\rho(r)/M$ gives the probability
$p_r(r)\,\mathrm{d}r$ of sampling a radius in $[r,r+\mathrm{d}r]$.
Curves use the dimensionless radius $x=r/a$ and its probability density
$p_x(x)=a\,p_r(ax)$.
These are continuum relations, before numerical endpoint clipping.
```

## Velocities for this density

The density and potential determine the isotropic equilibrium velocity distribution
through Eddington inversion. The [Plummer velocity-DF page](../velocity-dfs/plummer-dfs.md)
derives that distribution, its velocity moments, and the conditional speed law used
for sampling. Density alone does not specify velocity anisotropy.

## Closed-form summary

The continuum expressions above are collected here for reference:

```{list-table}
:header-rows: 1

* - Quantity
  - Closed form
* - Density $\rho(r)$
  - $\dfrac{3M}{4\pi a^3}\,(1 + r^2/a^2)^{-5/2}$
* - Cumulative mass $M(<r)$
  - $M\,r^3/(r^2 + a^2)^{3/2}$
* - Potential $\Phi(r)$
  - $-GM/\sqrt{r^2+a^2}$
* - Velocity moments and DF
  - See [](../velocity-dfs/plummer-dfs.md)
```

The half-mass relation $a = r_h\sqrt{2^{2/3}-1}$ {eq}`plummer-rh-a`
connects the length conventions. The [validation page](../../50-validation/plummer-equilibrium.md)
records the scope of the checks performed.

(plummer-implementation)=
## Implementation in progenax

:::{admonition} Numerical convention
:class: note

`PlummerProfile.density(r)` returns the dimensionless shape
$[1+(r/a)^2]^{-5/2}$. Multiply it by $3M/(4\pi a^3)$ to obtain physical
mass density. Position sampling evaluates the inverse CDF and isotropic directions
with array operations. Uniform radial draws are clipped to $[10^{-7},1-10^{-7}]$
before inversion, limiting the sampled radial range. The velocity method is described on the linked DF page.
:::

The [profiles API](../../30-api/profiles.md) and
[kinematics API](../../30-api/kinematics.md) document signatures; the
[architecture pages](../../20-architecture/index.md) describe JAX integration.
The [first Plummer tutorial](../../00-getting-started/first-plummer-sphere.md)
provides the runnable example, including float64 setup and matched position and
velocity sampling.

## Domain of validity and limitations

The continuum Plummer model has no finite outer boundary and fixes the outer
density slope at $-5$. These assumptions make it useful for controlled baseline
calculations, but they may not match the cluster or tidal environment being modeled.

:::{warning} Truncation changes the model
Removing outer particles changes the sampled distribution and can disturb the
equilibrium associated with the original DF. If a finite boundary or a different
outer slope is central to the problem, choose and validate the density–velocity
model together. See [](eff.md), [](king.md), and
[](../tidal-and-substructure/tidal.md).
:::

(implementation-validation-references)=
:::{admonition} Implementation and validation
:class: seealso

- **API:** [`PlummerProfile`](../../30-api/profiles.md#api-profiles-plummerprofile)
  and [`PlummerVelocityDF`](../../30-api/kinematics.md).
- **Checks and limits:** [Plummer validation](../../50-validation/plummer-equilibrium.md).
- **Equation provenance:** [Spatial model reference](../../15-model-reference/spatial_profiles.md#card-plummer-rho).
:::
