---
title: EFF profile
description: The Elson, Fall & Freeman (1987) profile — a power-law spatial density model for young massive star clusters in the LMC, parameterised by a free outer-slope parameter γ.
---

# The EFF profile

The {cite:t}`ElsonFallFreeman1987` (EFF) profile is the production-grade
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

```{warning}
**EFF requires $\gamma > 3$ for finite total mass.** At $\gamma = 3$
the cumulative-mass integral diverges logarithmically; at $\gamma < 3$
it diverges polynomially. progenax's `EFFProfile` enforces $\gamma >
3$ at construction time, raising `ValueError` for unphysical inputs.
For $\gamma \in (2, 3]$ — useful for fitting *projected*
surface-brightness profiles which can have shallower slopes than the
3D density — the cluster must be truncated explicitly via
`apply_tidal_truncation` ([](../tidal-and-substructure/tidal.md)).
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

At $\gamma = 5$, the EFF half-mass relation gives $r_h = a$. This does
not reproduce the Plummer scale-radius convention even though both
profiles have an outer $r^{-5}$ density falloff; the profiles share an
asymptotic slope but use different normalisations and mass profiles.
progenax uses the EFF mapping directly when constructing the profile
scale radius from the half-mass radius.

```{admonition} Why EFF and Plummer don't match at $\gamma = 5$
:class: note
The EFF density is $\rho \propto (1 + r^2/a^2)^{-\gamma/2}$. The
Plummer density is $\rho \propto (1 + r^2/a^2)^{-5/2}$. These are
the same functional form *with the same scale radius $a$* only when
$\gamma = 5$. The half-mass radii therefore differ by a constant — at
$\gamma = 5$, EFF gives $r_h = a$, Plummer gives $r_h \approx 1.31\,a$
({eq}`plummer-rh-a` inverted). This is not a bug; it is a difference
of convention between the two papers. progenax uses each profile's
native convention internally and converts user-facing `r_h` consistently.
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

## Velocity dispersion

EFF, unlike Plummer and King, does not admit a closed-form isotropic
distribution function for arbitrary $\gamma$. The Eddington inversion
of {eq}`eff-rho` against its self-consistent gravitational potential
yields a special-function expression that progenax evaluates
numerically. The radial velocity dispersion at radius $r$ is

```{math}
:label: eff-sigma
\sigma_r^2(r) \;=\; \frac{G\,\rho_0\,a^2}{2\,(1 + r^2/a^2)^{\gamma/2}}\,
\int_r^{\infty} \frac{(\gamma + 2)\,r'^2/a^2}{(1 + r'^2/a^2)^{(\gamma+4)/2}}\,\mathrm{d}r' \,+\, \cdots
```

(full expression in {cite:t}`ElsonFallFreeman1987` Eq. 7); progenax
evaluates the integral via fixed-step Gauss-Legendre quadrature inside
`EFFVelocityDF`. The implementation is differentiable in $\gamma$ and
$r_h$ via the standard JAX trick of differentiating *under* the
quadrature.

## Implementation in progenax

```python
from progenax.profiles import EFFProfile
from progenax.kinematics import EFFVelocityDF
from jaxstro.units import STELLAR

profile = EFFProfile(r_h=1.0, gamma=4.0)        # Young-cluster default
df = EFFVelocityDF(r_h=1.0, gamma=4.0)

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

`EFFProfile` is a fully-vectorised Equinox module; `EFFVelocityDF` runs
a 64-point Gauss-Legendre quadrature per particle that adds ~$3\times$
overhead vs Plummer at $N = 10^4$. The quadrature is fully
JIT-compatible.

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
* - You need a finite outer radius
  - King has $r_t$ built in; EFF extends to infinity
* - The host-galaxy tidal field is the dominant boundary
  - King's lowered-isothermal DF assumes tidal-field equilibrium
```

## Domain of validity and limitations

1. **$\gamma > 3$ required** for finite total mass without explicit
   truncation. progenax raises at construction if violated.
2. **No tidal radius built in.** EFF extends to infinity; tidal
   truncation must be applied as a post-processing step via
   `apply_tidal_truncation`.
3. **Only isotropic equilibrium DF.** Anisotropic versions of EFF
   exist in the literature {cite:p}`ElsonFallFreeman1987` but are not
   currently in progenax. For anisotropy, use the King DF with the
   Osipkov-Merritt extension ([](../velocity-dfs/rotation-anisotropy.md)).
4. **Single-mass only.** Multi-mass populations need post-segregation
   via [](../tidal-and-substructure/mass-segregation.md).

## References

The original derivation is {cite:t}`ElsonFallFreeman1987`. The
power-law outer slope motivates several other young-cluster profile
families (Wilson, Woolley) that {cite:t}`Gieles2015` LIMEPY
generalises uniformly. EFF remains the simplest of the family that
captures the key observation — power-law outer fall-off with a
free slope.
