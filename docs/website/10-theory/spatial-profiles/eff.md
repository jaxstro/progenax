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
by King's exponential cutoff. EFF was originally fit to 10 LMC
clusters of ages $10^7$–$10^9$ yr {cite:p}`ElsonFallFreeman1987`; it
is now used routinely for young Galactic and LMC/SMC clusters
where ages and dynamical states preclude King-like tidal truncation.

## The density profile

The EFF profile is

```{math}
:label: eff-rho
\rho(r) \;=\; \rho_0\,\biggl[1 + \biggl(\frac{r}{a}\biggr)^{\!2}\,\biggr]^{-\gamma/2}
```

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

## Closed-form mass

The cumulative mass for $\gamma > 3$ is

```{math}
:label: eff-mcum
M(<r) \;=\; \frac{4\pi\,\rho_0\,a^3}{\gamma - 3}\,\biggl[1 - \biggl(1 + \frac{r^2}{a^2}\biggr)^{\!(3-\gamma)/2}\,\biggr]
```

with total mass

```{math}
:label: eff-mtotal
M_{\mathrm{total}} \;=\; \lim_{r\to\infty} M(<r) \;=\; \frac{4\pi\,\rho_0\,a^3}{\gamma - 3}.
```

The half-mass-radius condition $M(<r_h) = M/2$ gives

```{math}
:label: eff-rh-a
r_h \;=\; a\,\sqrt{\,2^{2/(\gamma - 3)} - 1\,}.
```

At $\gamma = 5$, this *untruncated* EFF half-mass relation gives $r_h = a$.
This does not reproduce the Plummer scale-radius convention even though both
profiles have an outer $r^{-5}$ density falloff; the profiles share an
asymptotic slope but use different normalisations and mass profiles.

```{important}
{eq}`eff-rh-a` is the **untruncated, analytic** $r_h \leftrightarrow a$
relation. progenax's `EFFProfile` is constructed from the **scale radius $a$
directly** (`EFFProfile(a, gamma, r_t)`) — it does *not* take $r_h$ and does
not apply this mapping internally. The relation is provided for context (e.g.
converting a literature $r_h$ to an $a$ by hand); with a finite $r_t$ the true
half-mass radius is the truncated one, which differs from {eq}`eff-rh-a`.
```

```{admonition} Why EFF and Plummer don't match at $\gamma = 5$
:class: note
The EFF density is $\rho \propto (1 + r^2/a^2)^{-\gamma/2}$. The
Plummer density is $\rho \propto (1 + r^2/a^2)^{-5/2}$. These are
the same functional form *with the same scale radius $a$* only when
$\gamma = 5$. The half-mass radii therefore differ by a constant — at
$\gamma = 5$, EFF gives $r_h = a$, Plummer gives $r_h \approx 1.31\,a$
({eq}`plummer-rh-a` inverted). This is not a bug; it is a difference
of convention between the two papers. progenax keeps each profile's
native parameterisation — `PlummerProfile(r_h=...)` is built from the
half-mass radius, while `EFFProfile(a=..., gamma=..., r_t=...)` is built
from the scale radius $a$ — so the two are not interchangeable as a single
`r_h`.
```

## Inverse-CDF sampling

Inverting {eq}`eff-mcum` is straightforward. Setting $u = M(<r) / M$:

```{math}
:label: eff-inverse-cdf
r(u) \;=\; a\,\sqrt{\,(1 - u)^{2/(3 - \gamma)} - 1\,}
```

Differentiable analytically in $u$, $a$, and $\gamma$. progenax's
`EFFProfile.sample_positions` uses this expression directly via
`vmap`, with `lax.scan` not needed because the inversion is one-shot.

The differentiability in $\gamma$ is the practical advantage of EFF
over King: when fitting a cluster's outer slope from observational
data, $\gamma$ can be inferred via HMC alongside $r_h$ and the IMF
parameters. The King concentration $W_0$ requires re-solving the ODE
for each gradient evaluation, whereas $\gamma$ enters EFF analytically.

## Velocities: Eddington inversion (no closed form)

{cite:t}`ElsonFallFreeman1987` is a **surface-brightness** model only — the
paper fits the *projected* light profile $\mu(r)$ (Eq. 1) and its enclosed
projected luminosity (Eq. 2). It gives **no distribution function and no
velocity-dispersion formula**: there is no "EFF Eq. 7" for $\sigma_r^2(r)$.
Velocities in progenax are therefore *not* read from a closed form. Instead,
treating {eq}`eff-rho` as the 3-D volume density, `EFFVelocityDF` builds the
exact isotropic ergodic DF $f(E)$ by **Eddington inversion** of the (truncated)
EFF density against its self-consistent potential $\Psi(r)$,

```{math}
:label: eff-eddington
f(\mathcal{E}) \;=\; \frac{1}{\sqrt{8}\,\pi^2}
\left[\int_0^{\mathcal{E}} \frac{\mathrm{d}^2\rho}{\mathrm{d}\Psi^2}\frac{\mathrm{d}\Psi}{\sqrt{\mathcal{E}-\Psi}}
+ \frac{1}{\sqrt{\mathcal{E}}}\left(\frac{\mathrm{d}\rho}{\mathrm{d}\Psi}\right)_{\!\Psi=0}\right],
```

evaluated numerically on a tabulated grid at initialisation; speeds are then
drawn per particle from $g(v)\propto v^2 f(\Psi(r) - v^2/2)$ by inverse-CDF.
The construction is differentiable in $\gamma$, $a$, and $r_t$ through the
tabulated inversion. Because the EFF density is *empirical* (not derived from a
DF), a sharply truncated EFF is only **approximately** stationary: mild
truncation (e.g. $\gamma = 5$, which is exactly Plummer) is virial to $\sim1\%$,
whereas the steep $\gamma = 3$ default left strongly truncated is a few percent
sub-virial — intrinsic to truncating an empirical profile, not a DF error.
For a strict lowered-DF equilibrium use the King model.

## Implementation in progenax

```python
from progenax.profiles import EFFProfile
from progenax.kinematics import EFFVelocityDF
from jaxstro.units import STELLAR

profile = EFFProfile(a=1.0, gamma=3.0, r_t=10.0)   # a = scale radius, r_t = truncation
df = EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0)      # match a, gamma, r_t

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

`EFFProfile` is a fully-vectorised Equinox module. `EFFVelocityDF` builds the
exact isotropic ergodic DF $f(E)$ of the (truncated) EFF density by **Eddington
inversion** at initialisation, then samples speeds per particle from a tabulated
inverse-CDF; it is fully JIT-compatible and differentiable. For radial anisotropy,
pass `anisotropy_radius` (Osipkov–Merritt; see [](../velocity-dfs/rotation-anisotropy.md)).

```{warning}
**Match $\gamma$ in profile and velocity DF.** Like the $r_h$ matching
required for Plummer/King equilibrium, EFF requires the profile's
$\gamma$ and the velocity DF's $\gamma$ to agree. Mismatched values
produce non-equilibrium starting states.
```

See [](../../30-api/profiles.md) for the signature and
[](../../50-validation/eff-profile.md) for the regression suite.

## When to use EFF over Plummer or King

```{list-table}
:header-rows: 1

* - Use EFF when…
  - …because
* - Modelling young massive clusters
  - Observed surface-brightness profiles match power-law outer fall-off, not Plummer's $r^{-5}$ or King's exponential cutoff
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
  - For dynamical-evolution studies the central concentration dominates; outer slope effects on relaxation are sub-leading
* - Computational speed is critical
  - Plummer is ~$3\times$ faster than EFF per IC realisation
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
  - King's lowered-isothermal DF assumes tidal-field equilibrium
```

## Domain of validity and limitations

1. **$\gamma > 3$ is required only for the *untruncated* profile** to have
   finite total mass. progenax's `EFFProfile` always truncates at a finite
   $r_t$, so the sampled mass is finite for any $\gamma$ (the default is
   $\gamma = 3$); the constructor performs no $\gamma$ validation.
2. **Truncation is intrinsic, not optional.** The profile is sampled on
   $r \in [0, r_t]$ by construction; choose $r_t$ to match the cluster.
   Further tidal trimming can be layered via `apply_tidal_truncation`.
3. **Only isotropic equilibrium DF.** Anisotropic versions of EFF
   exist in the literature {cite:p}`ElsonFallFreeman1987` but are not
   currently in progenax. For anisotropy, use the King DF with the
   Osipkov-Merritt extension ([](../velocity-dfs/rotation-anisotropy.md)).
4. **Single-mass only.** Multi-mass populations need post-segregation
   via [](../tidal-and-substructure/mass-segregation.md).

## References

The original derivation is {cite:t}`ElsonFallFreeman1987`. The
power-law outer slope motivates several other young-cluster profile
families (Wilson, Woolley) that the lowered-model family of
{cite:t}`Gieles2015` generalises uniformly (see
[](lowered-model-family.md)). EFF remains the simplest of the family
that captures the key observation — power-law outer fall-off with a
free slope.
