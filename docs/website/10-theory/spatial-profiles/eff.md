---
title: EFF profile
description: The Elson, Fall & Freeman (1987) profile — a power-law spatial density model for young massive star clusters in the LMC, parameterised by a free outer-slope parameter γ.
---

# The EFF profile

The {cite:t}`ElsonFallFreeman1987` (EFF) profile is the standard
choice for **young massive star clusters** — typically $\lesssim 100$
Myr old, cluster mass $M_{\mathrm{cl}} \gtrsim 10^4\,\Msun$ — where
the surface-brightness profile shows a power-law fall-off whose slope
is *not well captured* by Plummer's fixed $r^{-5}$ outer asymptote or
by King's finite DF-derived cutoff. EFF was originally fit to 10 LMC
clusters of ages $10^7$–$10^9$ yr {cite:p}`ElsonFallFreeman1987`; it
is now used routinely for young Galactic and LMC/SMC clusters
where ages and dynamical states preclude King-like tidal truncation.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the EFF power-law profile for young massive clusters; no prior literature on cluster surface-brightness fitting assumed.
**Prerequisites:** the [Plummer profile](plummer.md) (EFF has its untruncated density shape at $\gamma=5$); the [King profile](king.md) is useful for the truncation contrast but optional.
**You'll get:** the EFF density and its free slope $\gamma$, why progenax's $\gamma$ is a 3-D (not surface) slope, the truncation and mass behavior, and how velocities come from Eddington inversion rather than a closed form.
:::

## The density profile

The EFF profile is

```{math}
:label: eff-rho
\rho(r) \;=\; \rho_0\,\biggl[1 + \biggl(\frac{r}{a}\biggr)^{\!2}\,\biggr]^{-\gamma/2}
```

[↗ model card](#card-eff-rho)

with three parameters: central density $\rho_0$, scale radius $a$, and
power-law slope $\gamma$. The structural similarity to Plummer (which
is exactly EFF at $\gamma = 5$) is intentional — EFF generalises Plummer
by exposing the outer slope as a free parameter.

```{admonition} progenax's $\gamma$ is a 3-D density slope, not EFF87's surface slope
:class: important
{cite:t}`ElsonFallFreeman1987` Eq. 1 defines $\mu(r) = \mu_0(1+r^2/a^2)^{-\gamma/2}$ as the
**projected surface brightness** (their $\gamma \in [2.2, 3.2]$, median 2.6, is the *surface*
slope). progenax adopts the **same functional form as the 3-D volume density** $\rho(r)$ above —
the standard N-body/IC-code convention (e.g. McLuster). So **progenax's $\gamma$ is a 3-D
density slope**, offset by $\approx 1$ from EFF87's observed surface slope (Abel projection of a
3-D power law $r^{-\gamma}$ gives a surface power law $r^{-(\gamma-1)}$). Thus EFF87's median
surface slope $\gamma_{\rm EFF}\approx 2.6$ corresponds to a progenax 3-D slope $\approx 3.6$;
the mass-convergence thresholds in the table below ($\gamma > 3$) are 3-D-density statements.
See [](../../99-bibliography/per-paper/elson-fall-freeman-1987.md).
```

```{list-table} EFF behaviour vs. $\gamma$.
:header-rows: 1

* - $\gamma$
  - Outer slope at $r \gg a$
  - Cluster type
* - 2
  - $\rho \propto r^{-2}$
  - Mass diverges; not physically valid
* - 3
  - $\rho \propto r^{-3}$
  - Mass logarithmically divergent at infinity; truncation needed
* - 4
  - $\rho \propto r^{-4}$
  - Typical young massive cluster
* - 5
  - $\rho \propto r^{-5}$
  - Plummer (special case)
* - 6
  - $\rho \propto r^{-6}$
  - Rare; very compact clusters
```

```{admonition} The *untruncated* EFF needs $\gamma > 3$ — but progenax's EFF is always truncated
:class: important
For the **untruncated** profile ($r_t \to \infty$) the total-mass integral
converges only for $\gamma > 3$: at $\gamma = 3$ it diverges logarithmically and at
$\gamma < 3$ polynomially. progenax's `EFFProfile`, however, **always carries a
finite truncation radius $r_t$** (constructor argument, default $r_t = 10$), so the
sampled mass $M(<r_t)$ is finite for *any* $\gamma$ — including the **default
$\gamma = 3$** and shallower slopes. There is therefore **no $\gamma > 3$ validation
and no `ValueError`** in `EFFProfile`; the truncation does the work that finite mass
would otherwise require. The closed-form $M_{\rm total}$ below ({eq}`eff-mtotal`) is the
$r_t \to \infty$ limit and applies only when $\gamma > 3$; the actual sampled mass is the
truncated $M(<r_t)$.
```

## Enclosed mass

For the density $\rho(r) = \rho_0\,(1 + r^2/a^2)^{-\gamma/2}$ the enclosed mass
has **no elementary closed form for general $\gamma$**; the exact result is a
hypergeometric function,

```{math}
:label: eff-mcum
M(<r) \;=\; \frac{4\pi}{3}\,\rho_0\,r^3\;
  {}_2F_1\!\left(\tfrac{3}{2},\,\tfrac{\gamma}{2};\,\tfrac{5}{2};\,-\frac{r^2}{a^2}\right).
```

[↗ model card](#card-eff-mcum)

The total (untruncated) mass converges only for $\gamma > 3$, where the integral
evaluates to a ratio of Gamma functions,

```{math}
:label: eff-mtotal
M_{\mathrm{total}} \;=\; \lim_{r\to\infty} M(<r)
  \;=\; 4\pi\,\rho_0\,a^3\,\frac{\sqrt{\pi}}{4}\,
        \frac{\Gamma\!\big(\tfrac{\gamma-3}{2}\big)}{\Gamma\!\big(\tfrac{\gamma}{2}\big)}.
```

[↗ model card](#card-eff-mtotal)

As $\gamma \to 3^{+}$ the factor $\Gamma\!\big(\tfrac{\gamma-3}{2}\big)$ diverges:
a shallow outer slope makes the profile so extended that its *untruncated* mass is
infinite. This is exactly why a finite truncation radius $r_t$ is required for
shallow $\gamma$ — the sampled mass is always the truncated $M(<r_t)$.

Two special cases are worth keeping as sanity anchors:

- **$\gamma=5$** has the Plummer density shape at the same $a$. For the
  untruncated mathematical profile, the hypergeometric reduces to the
  elementary Plummer form
  $M(<r) = \tfrac{4\pi}{3}\rho_0 a^3\,\hat r^{3}\,(1+\hat r^2)^{-3/2}$ with
  $\hat r = r/a$, giving $M_{\mathrm{total}} = \tfrac{4\pi}{3}\rho_0 a^3$ and the
  Plummer half-mass radius $r_h = a/\sqrt{2^{2/3}-1} \approx 1.305\,a$.
- **$\gamma = 4$** integrates elementarily to
  $M(<r) = 2\pi\rho_0 a^3\big[\arctan\hat r - \hat r/(1+\hat r^2)\big]$, with
  $M_{\mathrm{total}} = \pi^2\rho_0 a^3$.

```{note}
There is **no closed-form $r_h \leftrightarrow a$ relation for general $\gamma$**.
The relation $r_h=a/\sqrt{2^{2/3}-1}\simeq1.305a$ applies to the
**untruncated** $\gamma=5$ model. The implemented finite cut changes the
normalized mass distribution: its half-mass radius instead satisfies
$M(<r_h)=M(<r_t)/2$, even at $\gamma=5$. For a chosen $(\gamma,r_t)$,
invert the numerical CDF or the enclosed-mass expression with that finite
normalization. Equal density shapes at equal $a$ do not make finite and
untruncated samples identical.
```

The functional potential API uses gravitational constant $G$, the total
truncated mass $M=M(<r_t)$, and the zero $\Phi(\infty)=0$. Beyond the density cutoff it returns
$\Phi(r)=-GM/r$, with $\Phi(r_t)=-GM/r_t$; the exterior field remains
nonzero. The interior potential is obtained numerically from the enclosed mass
and outer shells. See the [exterior-potential convention](index.md#profile-vacuum-potential).

## Sampling

`EFFProfile` is constructed from the scale radius $a$ directly
(`EFFProfile(a, gamma, r_t)`) — it does not take $r_h$. Because {eq}`eff-mcum` has
no elementary inverse, `EFFProfile.sample_positions` builds a **numerical
cumulative-mass table**: it evaluates $4\pi r^2\rho(r)$ on a square-stretched grid
$r = r_t\,u^2$ (which concentrates points in the core), forms the running
trapezoidal integral, normalizes it to a CDF, and draws radii by inverse-CDF
interpolation (`jnp.interp`) of uniform deviates. The sampled mass is therefore the
truncated $M(<r_t)$.

Sampling stays **differentiable in $a$, $\gamma$, and $r_t$**: the density enters
the grid analytically, so gradients flow through the tabulated CDF and the
interpolation. This is the practical advantage of EFF over King for fitting a
cluster's outer slope — $\gamma$ can be inferred via HMC alongside $r_h$ and the
IMF parameters, whereas the King concentration $W_0$ requires re-solving the
Poisson ODE at each gradient evaluation.

## Velocities: numerical Eddington inversion

{cite:t}`ElsonFallFreeman1987` is a **surface-brightness** model only — the
paper fits the *projected* light profile $\mu(r)$ (Eq. 1) and its enclosed
projected luminosity (Eq. 2). It gives **no distribution function and no
velocity-dispersion formula**: there is no "EFF Eq. 7" for $\sigma_r^2(r)$.
Velocities in progenax are therefore *not* read from a closed form. Instead,
treating {eq}`eff-rho` as the 3-D volume density, `EFFVelocityDF` builds the
numerical isotropic candidate DF $f(E)$ by Eddington inversion of the interior
EFF density against its self-consistent potential $\Psi(r)$,

```{math}
:label: eff-eddington
f(\mathcal{E}) \;=\; \frac{1}{\sqrt{8}\,\pi^2}
\left[\int_0^{\mathcal{E}} \frac{\mathrm{d}^2\rho}{\mathrm{d}\Psi^2}\frac{\mathrm{d}\Psi}{\sqrt{\mathcal{E}-\Psi}}
+ \frac{1}{\sqrt{\mathcal{E}}}\left(\frac{\mathrm{d}\rho}{\mathrm{d}\Psi}\right)_{\!\Psi=0}\right],
```

[↗ model card](#card-eff-eddington)

evaluated numerically on a tabulated grid at initialisation; speeds are then
drawn per particle from $g(v)\propto v^2 f(\Psi(r) - v^2/2)$ by inverse-CDF.
The construction is differentiable in $\gamma$, $a$, and $r_t$ through the
tabulated inversion. Because the EFF density is *empirical* (not derived from a
DF), a sharply truncated EFF is only **approximately** stationary: mild
truncation at $\gamma=5$ approaches the untruncated Plummer case in retained
checks, whereas the shallower $\gamma=3$ default left strongly truncated is a few percent
sub-virial — intrinsic to truncating an empirical profile, not a DF error.
A finite interior density at $r_t$ cannot be represented exactly by a regular
ergodic DF with zero escape energy: the accessible velocity volume vanishes
as $\Psi\to0$. The sharp-edge construction therefore approximates the
requested density; its numerical and physical limits remain separate from
the formal inversion identity. For a DF-defined finite boundary, use an
admissible lowered model and its validation exposure.

## Implementation in progenax

```python
import jax
import jax.numpy as jnp
from progenax.profiles import EFFProfile
from progenax.kinematics import EFFVelocityDF
from jaxstro.units import STELLAR

key_pos, key_vel = jax.random.split(jax.random.PRNGKey(0))

profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)   # a = scale radius, r_t = truncation
df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)      # match a, gamma, r_t

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

`EFFProfile` stores a numerical mass CDF. `EFFVelocityDF` numerically inverts
the density and samples a tabulated speed CDF. Its derivative accuracy depends
on the chosen parameter domain and resolution; JAX tracing alone does not
qualify all derivatives. For radial anisotropy,
pass `anisotropy_radius` (Osipkov–Merritt; see [](../velocity-dfs/rotation-anisotropy.md)).

```{warning}
**Match $\gamma$ in profile and velocity DF.** Like the $r_h$ matching
required for Plummer/King equilibrium, EFF requires the profile's
$\gamma$ and the velocity DF's $\gamma$ to agree. Mismatched values
produce non-equilibrium starting states.
```

## When to use EFF over Plummer or King

```{list-table}
:header-rows: 1

* - Use EFF when…
  - …because
* - Modelling young massive clusters
  - Observed surface-brightness profiles match power-law outer fall-off, not Plummer's $r^{-5}$ or King's finite DF-derived cutoff
* - Fitting $\gamma$ as a free parameter
  - $\gamma$ enters EFF analytically; no ODE re-solve per gradient step (unlike King's $W_0$)
* - LMC/SMC cluster work
  - {cite:t}`ElsonFallFreeman1987` calibrated EFF on 10 LMC clusters of ages $10^7$–$10^9$ yr
* - Pre-relaxation / "post-formation" clusters
  - No assumption of dynamical equilibrium with the host galaxy's tidal field
```

```{list-table}
:header-rows: 1

* - Use Plummer instead when…
  - …because
* - You want closed-form everything
  - Plummer's DF is analytic; EFF's requires quadrature
* - The outer slope doesn't matter
  - Use the analytic baseline only when its fixed envelope and lack of a finite boundary suit the intended problem
* - Computational speed is critical
  - Compare construction and warm sampling for the intended size and hardware; see the [performance evidence](../../50-validation/performance-memory.md)
```

```{list-table}
:header-rows: 1

* - Use King instead when…
  - …because
* - Modelling old globular clusters
  - Tidal truncation is well-defined and observationally constrained
* - The truncation radius is physically meaningful
  - King's $r_t$ is the self-consistent tidal boundary where the DF vanishes; EFF's $r_t$ is an imposed cutoff on an otherwise power-law (formally infinite) density
* - The host-galaxy tidal field is the dominant boundary
  - A lowered DF supplies a finite model edge; matching that edge to an external Jacobi surface is a separate physical constraint
```

## Domain of validity and limitations

1. **$\gamma > 3$ is required only for the *untruncated* profile** to have
   finite total mass. progenax's `EFFProfile` always truncates at a finite
   $r_t$, so the sampled mass is finite for any $\gamma$ (the default is
   $\gamma = 3$); the constructor performs no $\gamma$ validation.
2. **Truncation is intrinsic, not optional.** The profile is sampled on
   $r \in [0, r_t]$ by construction; choose $r_t$ to match the cluster.
   Further tidal trimming can be layered via `apply_tidal_truncation`.
3. **Anisotropy requires an admissible DF.** `EFFVelocityDF` supports
   Osipkov–Merritt radial anisotropy via `anisotropy_radius`; its augmented
   density and non-negativity checks have their own numerical limits.
   `KingVelocityDF` remains isotropic; the self-consistent anisotropic King
   family is Michie. See [](../velocity-dfs/rotation-anisotropy.md).
4. **The standalone profile is one density component.** For prescribed EFF
   components in a shared potential, use Engine B
   (`MultiComponentCluster.from_density_profiles`) and check joint DF
   realizability. A post-sampling segregation transform is a different,
   generally nonstationary construction.

## Implementation, validation & references

- **In code:** `src/progenax/profiles/eff.py` (density, truncated mass,
  numerical inverse-CDF sampling); velocities come from the Eddington
  inversion in `src/progenax/kinematics/eddington.py` via
  `src/progenax/kinematics/eff_df.py`. See the
  [`EFFProfile` API](../../30-api/profiles.md).
- **Validated in:** [EFF profile](../../50-validation/eff-profile.md) —
  the regression suite, including the $\gamma = 5$ reduction to Plummer
  and the Eddington-DF virial check.
- **Primary sources:** {cite:t}`ElsonFallFreeman1987` defines the projected
  light profile. The DF-defined lowered families of {cite:t}`Gieles2015`
  are separate models with different boundary assumptions; EFF is not a
  member of that lowered-isothermal DF family. Full notes in the
  [bibliography](../../99-bibliography/per-paper/elson-fall-freeman-1987.md).
