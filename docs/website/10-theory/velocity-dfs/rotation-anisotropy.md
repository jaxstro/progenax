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

Both layers are *decorators* on the underlying DF: any of
`PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF` can be wrapped
with either or both. The composition is order-independent (anisotropy
modifies the velocity ellipsoid; rotation adds a bulk flow), so the
final IC has the right $\beta(r)$ profile *and* the right rotation
curve.

## Osipkov-Merritt anisotropy

The Osipkov-Merritt construction replaces the energy-only DF
$f(\mathcal{E})$ with a DF that depends on the augmented integral

```{math}
:label: om-Q
Q \;\equiv\; \mathcal{E} - \frac{L^2}{2\,r_a^2}
```

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

## Implementation: the `apply_osipkov_merritt` decorator

```python
from progenax.kinematics import PlummerVelocityDF, apply_osipkov_merritt

base_df = PlummerVelocityDF(r_h=1.0)
om_df = apply_osipkov_merritt(base_df, r_a=2.0)   # r_a in same units as r_h

# Sample velocities — same API as the base DF
velocities = om_df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

`apply_osipkov_merritt` returns a new Equinox module that wraps the
base DF and overrides `sample_velocities` to produce $f(Q)$ samples.
The implementation:

1. At each particle's position, compute the local potential $\Phi(r)$
   and the augmented density $\rho_Q(r) = (1 + r^2/r_a^2)\rho(r)$.
2. Sample $Q$ from the Eddington-inverted $f(Q)$ via the same
   inverse-CDF table machinery used for the base DF.
3. Convert $(Q, r)$ to $(v, \cos\theta_v)$ where $\theta_v$ is the
   angle between $\mathbf{v}$ and $\mathbf{\hat r}$.
4. Add an isotropic tangential angle in the plane perpendicular to
   $\mathbf{\hat r}$.

The decorator is JIT-compatible and differentiable in $r_a$ as well
as $r_h$.

```{warning}
**Stability constraint on $r_a$.** Osipkov-Merritt DFs become
*radial-orbit-instability* unstable when $r_a$ is too small relative
to the spatial scale. For Plummer, $r_a / a \gtrsim 0.5$ is generally
stable; below that, evolved configurations show bar-like instabilities
within $\sim 1$ relaxation time. progenax does not enforce this
bound at construction time; the validation suite checks for negative
$f(Q)$ values which is a clear signal of a physically inconsistent
choice.
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
*supervirial* by $T_{\mathrm{rot}}/|V|$. progenax handles this by
rescaling the velocities to a target $Q_{\mathrm{vir}}$ via
`virial_scale` after rotation is applied.

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
from progenax.kinematics import apply_solid_body_rotation

base_df = PlummerVelocityDF(r_h=1.0)
rot_df = apply_solid_body_rotation(base_df, omega=jnp.array([0.0, 0.0, 0.5]))

velocities = rot_df.sample_velocities(positions, masses, key, G=STELLAR.G)
# Then virial-rescale to target Q_vir
positions, velocities, masses = virial_scale(positions, velocities, masses, G=STELLAR.G)
```

## Differential rotation

Differential rotation generalises {eq}`solid-rotation` by allowing
$\mathbf{\Omega}$ to depend on position:

```{math}
:label: diff-rotation
\mathbf{v}_i \;=\; \mathbf{v}_i^{\mathrm{iso}} + \mathbf{\Omega}(r_i) \times \mathbf{r}_i.
```

The most common parameterisation is a power-law in cylindrical
radius $R = \sqrt{x^2 + y^2}$:

```{math}
:label: omega-power-law
\Omega(R) \;=\; \Omega_0\,\biggl(\frac{R}{R_0}\biggr)^{-q}
```

with $q = 0$ recovering solid-body rotation, $q = 1$ giving
constant-circular-velocity flat rotation, and $q = 2$ giving
Keplerian fall-off. progenax exposes the rotation profile as a
user-supplied callable for full flexibility:

```python
from progenax.kinematics import apply_differential_rotation

def omega_profile(R):
    return 0.5 * (R / 1.0) ** -1.0   # Flat rotation curve

rot_df = apply_differential_rotation(base_df, omega_profile=omega_profile)
velocities = rot_df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

The decorator takes the callable and applies it via `jax.vmap` over
particles. The callable must be JAX-traceable (use `jnp` operations
internally) so the resulting DF remains JIT-compatible and
differentiable in any parameters of the rotation profile.

## Composability

All three extensions compose:

```python
df = PlummerVelocityDF(r_h=1.0)
df = apply_osipkov_merritt(df, r_a=2.0)
df = apply_solid_body_rotation(df, omega=jnp.array([0.0, 0.0, 0.3]))
```

The order is *anisotropy first, rotation second* by convention —
anisotropy modifies the velocity ellipsoid, rotation adds a bulk
flow on top. Reversing the order gives the same numerical result
because rotation does not change $\sigma_r$ or $\sigma_t$.

## Domain of validity

1. **Osipkov-Merritt requires $r_a / r_h \gtrsim 0.5$** for stability.
   Below that, expect radial-orbit instability within a relaxation
   time. progenax does not enforce this at construction; the
   validation suite checks $f(Q) > 0$ which catches the most pathological
   regime.
2. **Solid-body rotation up to $\Omega/\Omega_{\mathrm{break-up}} \sim 0.5$**
   is physically reasonable. Above that, the cluster flattens
   significantly and a spherical IC is a poor approximation.
3. **Differential-rotation profiles must be smooth** — discontinuities
   in $\Omega(R)$ produce delta-function torques on the cluster and
   non-equilibrium starting states. If your science requires a
   sharp transition (e.g. a rotating core embedded in a
   non-rotating envelope), use a smooth-blend $\Omega(R)$ rather
   than a step function.

## References

The Osipkov-Merritt construction is standard textbook material;
Binney & Tremaine *Galactic Dynamics* §4 gives a clean derivation.
The solid-body and differential-rotation prescriptions follow the
N-body initialisation literature {cite:p}`Aarseth1974,Kuepper2011`. For
King-cluster rotation specifically, the {cite:t}`Gieles2015` LIMEPY
family includes a self-consistent rotating extension that progenax
does not currently implement.
