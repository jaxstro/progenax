---
title: Plummer velocity distribution functions
description: The isotropic Plummer distribution function f(E) ∝ (-E)^(7/2), its derivation by Eddington inversion, and progenax's sampling implementation.
---

# Plummer velocity distribution functions

The Plummer density has an analytic isotropic equilibrium distribution function.
Expressing density in terms of the relative potential makes the Eddington inversion
particularly simple: $\rho\propto\Psi^5$ gives $f(\mathcal E)\propto\mathcal E^{7/2}$.
This page derives that result, connects it to velocity moments, and explains
Progenax's conditional speed sampler.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the Plummer equilibrium DF and, through it, how Eddington inversion turns a density into velocities; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [Plummer profile](../spatial-profiles/plummer.md) (the density this DF inverts); the [velocity-DF overview](index.md) for the shared contract.
**You'll get:** the derivation of $f(\mathcal{E}) \propto \mathcal{E}^{7/2}$, the closed-form velocity moments, how the isotropic sampler uses the Beta distribution, and how the isotropic DF extends to anisotropy and rotation.
:::

## Eddington inversion in one paragraph

For a spherical, isotropic system, the DF depends only on the binding
energy $\mathcal{E} \equiv -E = -\tfrac{1}{2}v^2 + \Psi(r)$, where
$\Psi \equiv -\Phi$ is the **relative potential** (positive for a bound
system, $\Psi \to 0$ at infinity). The Eddington formula

```{math}
:label: eddington
f(\mathcal{E}) \;=\; \frac{1}{\sqrt{8}\,\pi^2}
\biggl[\,\int_0^{\mathcal{E}} \frac{\mathrm{d}^2\rho}{\mathrm{d}\Psi^2}\,\frac{\mathrm{d}\Psi}{\sqrt{\mathcal{E} - \Psi}} + \frac{1}{\sqrt{\mathcal{E}}}\,\biggl(\frac{\mathrm{d}\rho}{\mathrm{d}\Psi}\biggr)_{\!\Psi=0}\,\biggr]
```

[↗ model card](#card-eddington)

inverts the spatial-density profile $\rho(\Psi)$ — the density
expressed as a function of the relative potential — into the DF $f(\mathcal{E})$.
The boundary term vanishes when $(d\rho/d\Psi)_{\Psi=0}=0$. For Plummer,
$\rho\propto\Psi^5$, so $d\rho/d\Psi\propto\Psi^4$ and this condition holds.
Density approaching zero alone does not establish that its derivative vanishes.

## The Plummer DF

Substitute the Plummer density relation

```{math}
\rho(\Psi) \;=\; \frac{3 M}{4\pi a^3}\,\biggl(\frac{\Psi\,a}{GM}\biggr)^{\!5}
```

into {eq}`eddington`. The boundary term vanishes ($d\rho/d\Psi\propto\Psi^4\to0$ as $\Psi\to0$), and the second derivative is a pure power:
$\mathrm{d}^2\rho/\mathrm{d}\Psi^2 = \tfrac{3M}{4\pi a^3}\,
(a/GM)^5\, 20\,\Psi^3$. The Abel integral is then a Beta function,

```{math}
\int_0^{\mathcal{E}} \frac{\Psi^3\,\mathrm{d}\Psi}{\sqrt{\mathcal{E}-\Psi}}
\;=\; B\!\left(4, \tfrac12\right)\mathcal{E}^{7/2}
\;=\; \frac{32}{35}\,\mathcal{E}^{7/2},
```

and collecting constants ($\tfrac{1}{\sqrt8 \pi^2} \cdot \tfrac{3}{4\pi}
\cdot 20 \cdot \tfrac{32}{35} = \tfrac{24\sqrt2}{7\pi^3}$) yields

```{math}
:label: plummer-df-final
\boxed{\;\;f(\mathcal{E}) \;=\; \frac{24\sqrt{2}}{7\pi^3}\,\frac{a^2}{G^5\,M^4}\,\mathcal{E}^{7/2},\qquad \mathcal{E} > 0\;\;}
```

[↗ model card](#card-plummer-df-final)

with $\mathcal{E} = -E = GM/\sqrt{r^2 + a^2} - \tfrac{1}{2}v^2$. The
$\mathcal{E}^{7/2}$ exponent is the signature of the Plummer profile —
it follows directly from $\rho \propto \Psi^5$ and the Abel-integral
$\frac{1}{2}$-power kernel of {eq}`eddington`. The sampling method below distinguishes the isotropic law from the numerically inverted
Osipkov–Merritt speed law.


```{figure} ../figures/eddington_triptych.webp
:label: fig-eddington-triptych
:width: 100%

The inversion, step by step: the density as a function of the relative
potential, its second derivative (the Abel-kernel ingredient), and the
resulting DF — with the output of the **numerical** inverter
(`eddington_invert`, the same code Engine B uses) as dots on the closed-form
law. The dots compare the numerical inversion with the analytic DF; this
figure does not report a maximum-error statistic.
Regenerate: `python -m laboratory.icviz --only eddington-triptych`.
```

With signed specific energy $E=-\mathcal E$, the same DF is

```{math}
:label: plummer-df
f(E) \;=\; \frac{24\sqrt{2}}{7\pi^3}\,\frac{a^2}{G^5\,M^4}\,(-E)^{7/2},
\qquad E < 0
```

## Closed-form moments

The second velocity moments of the Plummer DF are

```{math}
:label: plummer-vmoments
\begin{aligned}
\langle v^2\rangle(r) &\;=\; \frac{GM}{2\,\sqrt{r^2 + a^2}} \\
\sigma_r^2(r) = \langle v_r^2\rangle &\;=\; \frac{GM}{6\,\sqrt{r^2 + a^2}} \\
\sigma_\theta^2(r) = \sigma_\phi^2(r) &\;=\; \sigma_r^2(r) \quad\text{(isotropic)}
\end{aligned}
```

For the isotropic, non-streaming model, $\langle v^2\rangle=3\sigma_r^2$.
The combined tangential variance is $\sigma_\theta^2+\sigma_\phi^2=2\sigma_r^2$.
The radial dispersion can also be written as

```{math}
:label: plummer-sigma-r
\sigma_r^2(r) \;=\; \frac{G\,M}{6\,\sqrt{r^2 + a^2}}
```

The relation $\langle v^2\rangle=|\Phi|/2$ holds pointwise for this model.
Integrating gives $T=\tfrac12\int\rho\langle v^2\rangle\,d^3x
=-\tfrac14\int\rho\Phi\,d^3x=-V/2$, where
$V=\tfrac12\int\rho\Phi\,d^3x$. Finite particle realizations fluctuate around
this continuum relation. See [](../ic-philosophy.md).

```{figure} ../figures/plummer_dispersion_oracles.webp
:label: fig-plummer-dispersion-oracles
:width: 100%

Left: sampled $\sigma_r(r)$ ($N = 2\times10^5$, seed 11) compared
with the closed form. Error bars use a Gaussian approximation for uncertainty
in the sampled standard deviation, $\hat\sigma_r/\sqrt{2(n-1)}$, with
$n$ samples in each radial bin. The theoretical curve is evaluated at bin
centres. Right: `project_dispersion` compared with {cite:t}`Dejonghe1987`
Eq. 43. Residual panels use fixed plotting limits; the figure alone does not
establish a maximum-error bound.
Regenerate: `python -m laboratory.icviz --only plummer-dispersion-oracles`.
```

The central velocity dispersion is

```{math}
\sigma_0^2 \;=\; \sigma_r^2(0) \;=\; \frac{GM}{6\,a}
```

which (combined with {eq}`plummer-rh-a`) gives a one-line conversion
between cluster mass, scale radius, and central velocity dispersion —
useful for sanity-checking observational fits.

## Conditional speed distribution and sampling

At fixed radius, the velocity-space volume element contributes a factor $v^2$:

```{math}
:label: plummer-fv
p(v\mid r)\propto v^2\mathcal E(r,v)^{7/2},\qquad
\mathcal E(r,v)=\frac{GM}{\sqrt{r^2+a^2}}-\frac12v^2.
```

[↗ model card](#card-plummer-fv)

Define $q=v/v_{\mathrm{esc}}$. Then $p(q)\propto q^2(1-q^2)^{7/2}$
on $[0,1]$. With $z=q^2$, the change of variables gives
$z\sim\mathrm{Beta}(3/2,9/2)$. Here $\mathrm{Beta}(\alpha,\beta)$ is the
probability distribution on $[0,1]$ with density
$z^{\alpha-1}(1-z)^{\beta-1}/B(\alpha,\beta)$, where $B$ is the beta function.

The [normalized speed law and escape-speed records](../../15-model-reference/velocity_dfs.md#card-plummer-speed-pdf)
link these relations to symbolic checks and sampler-scale tests.

The isotropic implementation calls `jax.random.beta`, computes
$v=v_{\mathrm{esc}}\sqrt z$, and samples an isotropic direction. The
Osipkov–Merritt branch evaluates its analytic DF on a local dimensionless
speed grid and numerically inverts the resulting cumulative distribution.

```{figure} ../figures/plummer_velocity_components.webp
:label: fig-plummer-velocity-components
:alt: A schematic local spherical basis labels three equal velocity variances; the adjacent normalized speed distribution vanishes at zero and at escape speed.
:width: 100%

Isotropy gives equal variance in each velocity component. The speed distribution
includes the velocity-space volume factor $v^2$; for Plummer, this produces
$q^2\sim\mathrm{Beta}(3/2,9/2)$, where $q=v/v_{\mathrm{esc}}$.
The basis diagram is schematic; the curve is the analytic conditional speed law.
```

## Implementation in progenax

`PlummerVelocityDF(r_h, anisotropy_radius=None)` selects the isotropic branch
by default. A finite `anisotropy_radius` selects the Osipkov–Merritt branch.
The sampler evaluates squared radii from `positions` and uses the total supplied mass
to set the potential and velocity scale.

Given arrays and a fresh random key from the
[first Plummer tutorial](../../00-getting-started/first-plummer-sphere.md), call
`df.sample_velocities(positions, masses, key, G)`.
This call does not assign binary fractions or construct a mass-dependent
equipartition model. See the [API reference](../../30-api/kinematics.md) for
arguments, shapes, and units.

## Validation scope

The [Plummer validation page](../../50-validation/plummer-equilibrium.md)
records the tested quantities, configurations, criteria, and measured outcomes.
These include radial sampling, velocity moments, speed support, and fixed-draw
gradients. A global energy-ratio check is not a full Jeans-equation or stationarity test.

## Check yourself

:::{dropdown} 1. The 7/2 exponent by power counting
Without integrating: $\rho \propto \Psi^n$ feeds the Abel kernel
$\int_0^{\mathcal{E}} \Psi^{n-2}\,(\mathcal{E}-\Psi)^{-1/2}\mathrm{d}\Psi
\propto \mathcal{E}^{\,n - 3/2}$. Plummer has $n = 5$; check that Dejonghe's
$q$-family (which generalizes $\rho \propto \Psi^{5-q}(1+r^2)^{-q/2}$)
recovers the same $7/2$ at $q = 0$.
:::

:::{dropdown} 2. The central projected dispersion
Using $\sigma_{\mathrm{los}}^2(0)=3\pi GM/(64a)$ and
$\sigma_r^2(0)=GM/(6a)$, derive
$\sigma_{\mathrm{los}}(0)/\sigma_r(0)=\sqrt{9\pi/32}$.
Why does the ratio contain neither mass nor scale radius? Explain why a projected
central measurement differs from the local central dispersion.
:::


## Anisotropy and rotation extensions

The Plummer DF above is isotropic ($\beta = 0$) and non-rotating.
progenax extends it with two compositions:

- **Osipkov-Merritt anisotropy**: introduces a free anisotropy radius
  $r_a$ such that $\beta(r) = r^2/(r^2 + r_a^2)$. The DF is no longer
  a function of $\mathcal{E}$ alone but of $Q = \mathcal{E} - L^2/(2 r_a^2)$,
  the augmented integral of motion. progenax implements this as an
  intrinsic DF option, `PlummerVelocityDF(r_h=..., anisotropy_radius=r_a)`,
  using the analytic OM Plummer DF (Merritt 1985, Eq. 45), which is
  non-negative only for $r_a \ge 0.75\,a$ (Merritt 1985, Eq. 46). See
  [](rotation-anisotropy.md).
- **Rotation**: the array-based rotation functions add streaming velocities
  to sampled velocities. See [](rotation-anisotropy.md) and the
  [rotation API](../../30-api/kinematics.md) for their arguments and restrictions.

OM changes the distribution to $f(Q)$; additive rotation is a different construction.
Neither should be described as an unchanged isotropic DF.

## Domain of validity and limitations

The default sampler uses one shared spatial and velocity prescription for all
particles. Supplying unequal particle masses does not introduce mass-dependent
velocity dispersions or equipartition. Multi-mass models require an explicit
prescription for component distributions and their shared potential; see
[](../spatial-profiles/multimass-equipartition.md).

The continuum isotropic model is spherical and untruncated. Removing outer particles
changes the sampled model. A finite boundary and its velocity distribution must be
treated together when equilibrium is required.

## Implementation, validation & references

- **In code:** `src/progenax/kinematics/plummer_df.py` (the isotropic
  $f(\mathcal{E})$, the isotropic Beta speed sampler, and the optional
  Osipkov–Merritt anisotropy) — see the
  [`PlummerVelocityDF` API](../../30-api/kinematics.md).
- **Validated in:** [Plummer equilibrium](../../50-validation/plummer-equilibrium.md)
  — test exposures, criteria, outcomes and limitations.
- **DF source:** {cite:t}`Merritt1985`; the Eddington calculation above gives
  the isotropic normalization explicitly.
- **Primary sources:** {cite:t}`Plummer1911` (the density model);
  modern textbook treatments follow Binney & Tremaine *Galactic
  Dynamics* §4, and {cite:t}`Aarseth1974` is a clean N-body
  initialisation reference — full notes in the
  [bibliography](../../99-bibliography/per-paper/plummer-1911.md). The
  mass-first sampling contract follows the IC redesign at
  [](../../90-development-log/2026-02-12-ic-redesign.md).
