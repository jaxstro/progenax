---
title: Anisotropy and rotation
description: Osipkov-Merritt radial anisotropy and solid-body / differential rotation extensions to progenax's isotropic equilibrium velocity DFs.
---

# Anisotropy and rotation

The default equilibrium velocity DFs in progenax — Plummer, King, EFF —
are spherical, isotropic, and non-rotating. Real star clusters are
often *not*: globular clusters show modest radial anisotropy in their
outer regions {cite:p}`King1966`, young massive clusters frequently
rotate, and post-merger or tidally-stressed systems can have
substantial velocity anisotropy in either direction.

progenax extends each isotropic DF with two compositional layers:

1. **Osipkov-Merritt radial anisotropy** — a one-parameter
   ($r_a$, the *anisotropy radius*) generalisation that makes the
   velocity ellipsoid radially biased outside $r_a$ and stays
   isotropic within.
2. **Rotation** — solid-body or differential, applied as an additive
   tangential velocity on top of the isotropic base.

The two layers attach differently: anisotropy is *intrinsic to the DF*
(an `anisotropy_radius` constructor argument on `PlummerVelocityDF` /
`EFFVelocityDF`), while the rotation overlays are *plain functions* applied
to the already-sampled `(velocities, positions)` arrays (or declared via
`RotationParams` in the pipeline). The composition is order-independent (anisotropy
modifies the velocity ellipsoid; rotation adds a bulk flow), so the
final IC has the right $\beta(r)$ profile *and* the right rotation
curve.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers adding radial anisotropy or rotation to an isotropic equilibrium DF; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [Plummer DF](plummer-dfs.md) (the isotropic base these layers modify). For the *self-consistent* anisotropic King model see [Michie–King](michie-king.md).
**You'll get:** the Osipkov–Merritt construction ($\beta(r) = r^2/(r^2+r_a^2)$) and its non-negativity bound, solid-body and differential rotation as additive tangential flows, and how the two layers compose.
:::

## Osipkov-Merritt anisotropy

The Osipkov-Merritt construction replaces the energy-only DF
$f(\mathcal{E})$ with a DF that depends on the augmented integral

```{math}
:label: om-Q
Q \;\equiv\; \mathcal{E} - \frac{L^2}{2\,r_a^2}
```

[↗ model card](#card-om-Q)

where $L = |\mathbf{r}\times\mathbf{v}|$ is the specific angular
momentum and $r_a$ is the **anisotropy radius**. The DF $f(Q)$ —
constructed by Eddington-style inversion using the augmented density
$\rho_Q(r) \equiv (1 + r^2/r_a^2)\,\rho(r)$ — produces a velocity
ellipsoid whose anisotropy is

```{math}
:label: beta-r
\beta(r) \;\equiv\; 1 - \frac{\sigma_t^2(r)}{\sigma_r^2(r)}
\;=\; \frac{r^2}{r^2 + r_a^2}
```

[↗ model card](#card-beta-r)

Three regimes:

```{list-table}
:header-rows: 1

* - $r$
  - $\beta(r)$
  - Velocity ellipsoid
* - $\ll r_a$
  - $\to 0$
  - Isotropic ($\sigma_r = \sigma_t$)
* - $= r_a$
  - $0.5$
  - Moderately radial
* - $\gg r_a$
  - $\to 1$
  - Fully radial ($\sigma_t \to 0$)
```

The anisotropy radius $r_a$ is the only new parameter. Setting $r_a
\to \infty$ recovers the isotropic base DF; setting $r_a = 0$ produces
fully radial orbits at all radii (which is generally unphysical and
unstable).

## Implementation: the `anisotropy_radius` DF parameter

Osipkov-Merritt anisotropy is an intrinsic property of the velocity DF:
pass `anisotropy_radius` ($=r_a$) to a DF. `None` (default) gives the
isotropic DF; a float gives the radially anisotropic OM DF for the
*same* density.

```python
from jaxstro.units import STELLAR
from progenax.kinematics import PlummerVelocityDF, EFFVelocityDF

# r_a in the same length units as r_h / a
om_plummer = PlummerVelocityDF(r_h=1.0, anisotropy_radius=2.0)
# gamma=5 reduces EFF to Plummer, so this OM-EFF starts in clean equilibrium
# (Q ~ 0.5). A steep gamma=3 EFF, strongly truncated, is a few percent
# sub-virial even before anisotropy — see the EFF chapter.
om_eff = EFFVelocityDF(a=1.0, gamma=5.0, r_t=10.0, anisotropy_radius=2.0)

velocities = om_plummer.sample_velocities(positions, masses, key, G=STELLAR.G)
```

Internally (Merritt 1985):

1. Build $f(Q)$ from the augmented density $\rho_Q = (1+r^2/r_a^2)\rho$
   — the closed-form Eq. 45 for Plummer, the numerical Eddington
   inversion of $\rho_Q$ for EFF. The potential $\Psi(r)$ is unchanged
   (set by the true mass density).
2. At each radius sample a speed $s$ from $s^2 f(\Psi - s^2/2)$ by
   inverse-CDF — the same machinery as the isotropic DF.
3. Split $s$ *isotropically in the stretched frame*
   $w_t = v_t\sqrt{1+r^2/r_a^2}$, then un-stretch the tangential
   component, which realises $\beta(r)=r^2/(r^2+r_a^2)$ exactly.

The DF is JIT-compatible and differentiable in $r_a$ (and $r_h$/$a$).

```{warning}
**Non-negativity bound on $r_a$.** Too small an $r_a$ asks for more
radial anisotropy than the density can support with $f(Q)\ge 0$. For
Plummer the exact bound is $r_a \ge 0.75\,a$ (Merritt 1985, Eq. 46);
progenax **refuses** a smaller $r_a$ at construction (raises
`ValueError`) rather than silently clamping an unphysical DF, and EFF
likewise rejects an $r_a$ that drives $f(Q)$ negative. Separately, OM
DFs with very small $r_a$ are prone to the radial-orbit instability
under evolution — the non-negativity bound is necessary but not
sufficient for long-term stability.
```

```{note}
**King anisotropy** is *not* the augmented-density OM construction:
the radially anisotropic King model is the self-consistent
**Michie (1963)** model, whose lowered DF
$f \propto e^{-J^2/2r_a^2\sigma^2}[e^{-E/\sigma^2}-1]$ is **not** a
function of $Q$ alone and yields a *different*, more centrally-radial
density profile. It is implemented as the separate self-consistent pair
`MichieProfile` + `MichieVelocityDF` (`KingVelocityDF` stays isotropic).
See [](michie-king.md) and [](../../99-bibliography/per-paper/michie-1963.md).
```

## Solid-body rotation

The simplest rotation prescription adds a tangential velocity
$\mathbf{v}_\phi(\mathbf{r}) = \mathbf{\Omega} \times \mathbf{r}$ to
the isotropic base velocities, where $\mathbf{\Omega}$ is a constant
angular-velocity vector (typically aligned with the cluster's
$z$-axis). The resulting per-particle velocity is

```{math}
:label: solid-rotation
\mathbf{v}_i \;=\; \mathbf{v}_i^{\mathrm{iso}} + \mathbf{\Omega}\times\mathbf{r}_i.
```

This adds a bulk-rotation kinetic-energy contribution

```{math}
:label: rot-energy
T_{\mathrm{rot}} \;=\; \tfrac{1}{2}\sum_i m_i\,|\mathbf{\Omega}\times\mathbf{r}_i|^2
\;=\; \tfrac{1}{2}\,\Omega^2 I_\perp
```

with $I_\perp$ the moment of inertia perpendicular to $\mathbf{\Omega}$.
Because solid-body rotation adds *only* tangential velocity, the
random-motion velocity dispersion is preserved — but the total
kinetic energy increases, which means the resulting cluster is
*supervirial* by $T_{\mathrm{rot}}/|V|$. You can rescale the speeds to a
target $Q_{\mathrm{vir}}$ via `virial_scale` after rotation is applied,
but note the source module's own warning: a uniform speed rescale does
**not** restore stationarity — it cannot remove the injected angular
momentum $L_z$, so a rotation-overlaid IC is a deliberately rotating,
*non-stationary* initial condition.

```{note}
**The virial rescaling preserves rotation direction.** `virial_scale`
multiplies all velocities by a single scalar, which scales both the
random-motion and bulk-rotation components proportionally. The
*ratio* $T_{\mathrm{rot}} / T_{\mathrm{rand}}$ is preserved; only
the absolute energy is renormalised. This is the right behaviour
for inferring rotation strength from observations, where the
constraint typically comes from the *fraction* of kinetic energy in
rotation rather than the absolute rotation rate.
```

```python
from progenax import PlummerProfile, PlummerVelocityDF, virial_scale
from progenax.kinematics import apply_solid_body_rotation

profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)

# Overlay: add the streaming component v_rot = omega x r. The overlays are
# plain FUNCTIONS on (velocities, positions) arrays — not DF wrappers.
velocities = apply_solid_body_rotation(
    velocities, positions, omega=0.5, axis=jnp.array([0.0, 0.0, 1.0])
)

# Optional: rescale speeds to a target Q_vir (returns the scaled velocities).
# CAVEAT: the uniform rescale cannot remove the injected L_z — the rotating
# IC is deliberately NON-STATIONARY (see kinematics/rotation.py's docstring).
velocities = virial_scale(positions, velocities, masses, Q_target=0.5, G=STELLAR.G)
```

## Differential rotation

Differential rotation generalises {eq}`solid-rotation` by allowing
$\mathbf{\Omega}$ to depend on position:

```{math}
:label: diff-rotation
\mathbf{v}_i \;=\; \mathbf{v}_i^{\mathrm{iso}} + \mathbf{\Omega}(r_i) \times \mathbf{r}_i.
```

progenax implements a *peaked* rotation curve in cylindrical radius
$R = \sqrt{x^2 + y^2}$:

```{math}
:label: omega-power-law
v_\phi(R) \;=\; v_{\mathrm{peak}}\,\frac{R}{R_{\mathrm{peak}}}\,
  \exp\!\Bigl(1 - \frac{R}{R_{\mathrm{peak}}}\Bigr)
```

which is solid-body ($v_\phi \propto R$, $\Omega$ constant) in the core,
reaches $v_{\mathrm{peak}}$ exactly at $R = R_{\mathrm{peak}}$, and decays
exponentially outward — no rotation at the centre or in the far outskirts,
the shape observed in rotating globular clusters. (An earlier revision of
this page described a user-supplied power-law $\Omega(R)$ callable; the
shipped API takes the two scalars below.)

```python
from progenax.kinematics import apply_differential_rotation

velocities = apply_differential_rotation(
    velocities, positions, v_peak=0.5, R_peak=1.0,
    axis=jnp.array([0.0, 0.0, 1.0]),
)
```

The overlay is a plain function on the sampled `(velocities, positions)`
arrays, fully `vmap`/JIT-compatible and differentiable in
$v_{\mathrm{peak}}$ and $R_{\mathrm{peak}}$ (both are grad-audited
registry cases).

## Composability

Anisotropy is intrinsic to the DF; rotation and an optional virial
rescale are layered by the velocity pipeline:

```python
from progenax.kinematics import (
    PlummerVelocityDF, VelocityModel, RotationParams, sample_velocities_pipeline,
)

model = VelocityModel(
    df=PlummerVelocityDF(r_h=1.0, anisotropy_radius=2.0),   # radial anisotropy
    rotation=RotationParams(solid_body=True, pattern_speed=0.3),
)
velocities = sample_velocities_pipeline(key, positions, masses, model, G=STELLAR.G)
```

The DF sets the velocity ellipsoid ($\sigma_r$, $\sigma_t$); rotation
adds a bulk flow on top without changing the random-motion
dispersions.

## Domain of validity

1. **Osipkov-Merritt non-negativity.** progenax refuses an $r_a$ below
   the Plummer bound $r_a \ge 0.75\,a$ (Merritt 1985, Eq. 46) at
   construction (raises `ValueError`), and EFF rejects an $r_a$ that
   drives $f(Q)$ negative. Even above the bound, very small $r_a$ is
   prone to the radial-orbit instability within a relaxation time
   under evolution.
2. **Solid-body rotation up to $\Omega/\Omega_{\mathrm{break-up}} \sim 0.5$**
   is physically reasonable. Above that, the cluster flattens
   significantly and a spherical IC is a poor approximation.
3. **Differential-rotation profiles must be smooth** — discontinuities
   in $\Omega(R)$ produce delta-function torques on the cluster and
   non-equilibrium starting states. If your science requires a
   sharp transition (e.g. a rotating core embedded in a
   non-rotating envelope), use a smooth-blend $\Omega(R)$ rather
   than a step function.

## Implementation, validation & references

- **In code:** the rotation overlays live in
  `src/progenax/kinematics/rotation.py`
  (`apply_solid_body_rotation`, `apply_differential_rotation`); the
  Osipkov–Merritt anisotropy is an intrinsic option on the DFs in
  `src/progenax/kinematics/` (`plummer_df.py`, `eff_df.py`). See the
  [kinematics API](../../30-api/kinematics.md).
- **Validated in:** [OM / rotation anisotropy](../../50-validation/rotation-om-anisotropy.md)
  — the realised $\beta(r) = r^2/(r^2+r_a^2)$, the non-negativity bound,
  and the rotation-curve recovery.
- **Primary sources:** the Osipkov–Merritt construction is
  {cite:t}`Merritt1985` (the closed-form Plummer bound is Eq. 46);
  textbook context is Binney & Tremaine *Galactic Dynamics* §4, and the
  rotation prescriptions follow the N-body initialisation literature
  {cite:p}`Aarseth1974,Kuepper2011`. For a self-consistent rotating
  King extension, the lowered-model family {cite:t}`Gieles2015` is the
  natural route (see the
  [lowered-model family](../spatial-profiles/lowered-model-family.md)),
  not yet implemented. Full notes in the
  [bibliography](../../99-bibliography/per-paper/merritt-1985.md).
