---
title: Plummer velocity distribution functions
description: The isotropic Plummer (1911) distribution function f(E) ∝ (-E)^(7/2), its derivation by Eddington inversion, and progenax's sampling implementation.
---

# Plummer velocity distribution functions

The isotropic equilibrium DF for the Plummer profile is one of the
oldest closed-form distribution functions in stellar dynamics. It
falls out of {cite:t}`Plummer1911`'s original analysis as a direct
consequence of the smooth $(1 + r^2/a^2)^{-5/2}$ density profile,
and provides the canonical example of how the Eddington inversion
machinery turns a 3D density profile into a 6D phase-space DF for
spherical isotropic systems.

This chapter derives $f(E) \propto (-E)^{7/2}$ from the Plummer density
profile, lists the closed-form moments, walks through progenax's
sampling implementation, and connects to the anisotropy and rotation
extensions in [](rotation-anisotropy.md).

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the Plummer equilibrium DF and, through it, how Eddington inversion turns a density into velocities; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [Plummer profile](../spatial-profiles/plummer.md) (the density this DF inverts); the [velocity-DF overview](index.md) for the shared contract.
**You'll get:** the derivation of $f(\mathcal{E}) \propto \mathcal{E}^{7/2}$, the closed-form velocity moments, why progenax samples by inverse-CDF rather than rejection, and how the isotropic DF extends to anisotropy and rotation.
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
(An earlier revision of this page wrote the integration variable as $\Phi$;
the integral runs over the positive relative potential, so $\Psi$ is the
honest symbol.) The boundary term vanishes for systems with $\rho \to 0$
as $\Psi \to 0$ (i.e. cluster density vanishing at infinity, which
Plummer satisfies). For Plummer the full inversion gives a closed-form
DF in $\mathcal{E}$ alone.

## The Plummer DF

Substituting Plummer's

```{math}
\rho(\Psi) \;=\; \frac{3 M}{4\pi a^3}\,\biggl(\frac{\Psi\,a}{GM}\biggr)^{\!5}
```

into {eq}`eddington` and integrating yields

```{math}
:label: plummer-df-final
\boxed{\;\;f(\mathcal{E}) \;=\; \frac{24\sqrt{2}}{7\pi^3}\,\frac{a^2}{G^5\,M^4}\,\mathcal{E}^{7/2},\qquad \mathcal{E} > 0\;\;}
```

[↗ model card](#card-plummer-df-final)

with $\mathcal{E} = -E = GM/\sqrt{r^2 + a^2} - \tfrac{1}{2}v^2$. The
$\mathcal{E}^{7/2}$ exponent is the signature of the Plummer profile —
it follows directly from $\rho \propto \Phi^5$ and the Abel-integral
$\frac{1}{2}$-power kernel of {eq}`eddington`. progenax's sampler uses
this exact form, with no truncation or smoothing.

## Closed-form moments

The first three velocity moments of the Plummer DF are

```{math}
:label: plummer-vmoments
\begin{aligned}
\langle v^2\rangle(r) &\;=\; \frac{GM}{2\,\sqrt{r^2 + a^2}} \\
\sigma_r^2(r) = \langle v_r^2\rangle &\;=\; \frac{GM}{6\,\sqrt{r^2 + a^2}} \\
\sigma_t^2(r) = \langle v_t^2\rangle &\;=\; \sigma_r^2(r) \quad\text{(isotropic)}
\end{aligned}
```

with $\langle v^2 \rangle = \sigma_r^2 + 2\sigma_t^2 = 3\sigma_r^2$. The
ratio $\langle v^2 \rangle / |\Phi(r)| = \tfrac{1}{2}$ at all radii —
this is the *local* statement of the virial theorem and the proximate
reason a Plummer cluster sampled from {eq}`plummer-df-final` lands at
$Q_{\mathrm{vir}} = 0.5$ at the DF level — realised as $0.5$ up to the
finite-$N$ Monte-Carlo fluctuation, with no rescale
([](../ic-philosophy.md)).

The central velocity dispersion is

```{math}
\sigma_0^2 \;=\; \sigma_r^2(0) \;=\; \frac{GM}{6\,a}
```

which (combined with {eq}`plummer-rh-a`) gives a one-line conversion
between cluster mass, scale radius, and central velocity dispersion —
useful for sanity-checking observational fits.

## Sampling: speed and rejection

progenax samples velocities from {eq}`plummer-df-final` via inverse-CDF
on the speed distribution at each particle's position. At fixed $r$, the
DF integrated over angles gives

```{math}
:label: plummer-fv
f(v \mid r) \;\propto\; v^2\,\mathcal{E}(r, v)^{7/2},\qquad
\mathcal{E}(r, v) = \frac{GM}{\sqrt{r^2+a^2}} - \tfrac{1}{2}v^2
```

[↗ model card](#card-plummer-fv)

with $v \in [0, v_{\mathrm{esc}}(r)]$. The distribution is bounded,
unimodal, and smooth, so progenax uses a fixed-iteration
inverse-CDF lookup table rather than rejection sampling. The lookup
table is constructed once per call (vectorised over $r$) and inverted
via `jnp.interp`, so the cost per particle is dominated by the
single bisection step — no Python loop, no rejection retries, fully
JIT-compatible.

```{admonition} Why not rejection sampling?
:class: note
Rejection sampling has a *variable* per-particle cost: in the worst
case a single rejection can repeat indefinitely. Even bounding the
rejection rate at 99% leaves a long tail of slow particles. JAX's
`vmap` requires equal cost per element, so rejection sampling either
forces a fixed iteration count (defeating the purpose) or falls back
to Python-level retries (defeating JIT). progenax uses inverse-CDF
exclusively — the fixed iteration count is built into the table
inversion.
```

## Implementation in progenax

```python
from progenax.kinematics import PlummerVelocityDF
from jaxstro.units import STELLAR
import jax, jax.numpy as jnp

df = PlummerVelocityDF(r_h=1.0)        # Half-mass radius in pc
masses = jnp.ones(1000)
key = jax.random.PRNGKey(42)
key_pos, key_vel = jax.random.split(key)   # never reuse a key

# Positions from the matched profile (any sampled positions work)
from progenax.profiles import PlummerProfile
positions = PlummerProfile(r_h=1.0).sample_positions(masses, key_pos)

# Sample velocities consistent with f(E) ∝ E^(7/2)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)

# The DF is an exact equilibrium, so Q_vir -> 0.5 with no rescale
# (finite-N: 0.5 +/- ~5e-3 at N=1e4, the Monte-Carlo fluctuation)
```

`PlummerVelocityDF` is an Equinox module. Its single Python parameter
is `r_h`; everything else (the central potential depth, the inverse-CDF
lookup table) is derived analytically from $r_h$ inside the class.
Differentiability flows through $r_h \to \Phi_0 \to f(\mathcal{E}) \to
\mathbf{v}$.

```{admonition} The mass-first contract
:class: note
`sample_velocities` takes `masses` as a positional argument, not a
keyword. This is the **masses-first** API contract progenax adopts
across the package — see [](../../20-architecture/protocols.md). The
masses array determines $N$ and (for the binary IMF) the per-particle
binary fraction; passing it first matches the order in which a
practitioner would naturally write a forward-model line.
```

## Sanity checks

The Plummer DF satisfies four checks that progenax's validation suite
([](../../50-validation/plummer-equilibrium.md)) verifies on every
release:

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Source of truth
* - Total kinetic energy / $|V|$
  - $0.5 \pm 5\!\times\!10^{-3}$
  - Virial theorem (statistical from finite $N = 10^4$)
* - Mean radial dispersion at $r_h$
  - 1% of analytic value
  - {eq}`plummer-vmoments`
* - Anisotropy parameter $\beta(r)$
  - $|\beta| < 0.02$
  - Isotropy by construction
* - Bound fraction
  - $> 99.9\%$
  - All particles have $\mathcal{E} > 0$ for $\mathbf{v} < v_{\mathrm{esc}}$
```

The first row implements the full Jeans-equation check: Plummer
positions sampled from `PlummerProfile` plus velocities from
`PlummerVelocityDF` produce $Q_{\mathrm{vir}} = 0.5$ to 0.5%, with
the residual being the finite-$N$ fluctuation of the Monte Carlo
energy estimate. No virial rescaling is needed — equilibrium is
exact at the DF level.

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
- **Solid-body / differential rotation**: adds a tangential velocity
  component $\mathbf{v}_\phi(\mathbf{r}) = \mathbf{\Omega}(r) \times \mathbf{r}$
  to the isotropic Plummer velocities. Implemented via
  `apply_solid_body_rotation(df, omega)` and
  `apply_differential_rotation(df, omega_profile)`.

Both extensions are documented in detail at
[](rotation-anisotropy.md). They compose freely with each other and
with non-Plummer profiles (e.g. King + Osipkov-Merritt is a common
combination for old globular clusters with observed radial anisotropy).

## Domain of validity and limitations

1. **Single-mass equilibrium only.** The Eddington DF assumes one
   stellar mass species. Multi-mass equilibrium DFs require
   simultaneous treatment of all mass groups — the multi-mass
   generalisation is formalized by {cite:t}`Gieles2015`, which progenax
   plans to implement natively as its own differentiable generalization
   (see [](../spatial-profiles/lowered-model-family.md)).

2. **Spherical and isotropic by default.** Anisotropy and rotation
   extensions exist (above) but are layered on top of the isotropic
   DF; they do not change the underlying $f(\mathcal{E})$.

3. **Untruncated.** The DF extends to $r \to \infty$. For tidally
   truncated Plummer-like systems use [](../spatial-profiles/king.md)
   instead, or apply post-sampling truncation via
   `apply_tidal_truncation` ([](../tidal-and-substructure/tidal.md)).

## Implementation, validation & references

- **In code:** `src/progenax/kinematics/plummer_df.py` (the isotropic
  $f(\mathcal{E})$, the inverse-CDF speed sampler, and the optional
  Osipkov–Merritt anisotropy) — see the
  [`PlummerVelocityDF` API](../../30-api/kinematics.md).
- **Validated in:** [Plummer equilibrium](../../50-validation/plummer-equilibrium.md)
  — the regression suite that locks the velocity moments, isotropy
  $|\beta| < 0.02$, and the unscaled $Q_{\mathrm{vir}} = 0.5$.
- **Primary sources:** {cite:t}`Plummer1911` (the model and its DF);
  modern textbook treatments follow Binney & Tremaine *Galactic
  Dynamics* §4, and {cite:t}`Aarseth1974` is a clean N-body
  initialisation reference — full notes in the
  [bibliography](../../99-bibliography/per-paper/plummer-1911.md). The
  mass-first sampling contract follows the IC redesign at
  [](../../90-development-log/2026-02-12-ic-redesign.md).
