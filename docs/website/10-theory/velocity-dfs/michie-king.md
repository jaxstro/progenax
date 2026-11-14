---
title: Michie-King anisotropic model
description: The self-consistent radially anisotropic King model (Michie 1963 anisotropy + King 1966 cutoff) as implemented by MichieProfile + MichieVelocityDF.
---

# Michie-King anisotropic model

The isotropic [King model](../spatial-profiles/king.md) is the workhorse truncated
globular-cluster model, but real clusters show **radial velocity anisotropy** in their
outskirts. The **Michie-King** model adds Michie's (1963) angular-momentum term to King's
(1966) lowered-Maxwellian cutoff, giving a one-parameter family (the anisotropy radius
$r_a$) of self-consistent, tidally truncated, radially anisotropic models.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the self-consistent radially anisotropic King model and how progenax samples it; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [King DF](king-dfs.md) (the lowered-Maxwellian cutoff this extends) and the [anisotropy idea](rotation-anisotropy.md) (what $\beta(r)$ and the anisotropy radius $r_a$ mean).
**You'll get:** the Michie–King DF $f(E, J)$, why it differs from the Osipkov–Merritt construction (it re-solves Poisson, changing the density), the finite-domain checks needed to establish a tidal crossing, and how to pair `MichieProfile` + `MichieVelocityDF`.
:::

## The distribution function

```{math}
:label: mk-df
f(E, J) \propto \exp\!\left(-\frac{J^2}{2 r_a^2 \sigma^2}\right)
   \left[\exp\!\left(-\frac{E}{\sigma^2}\right) - 1\right],\qquad E \le 0,
```

[↗ model card](#card-mk-df)

with $E=\tfrac12 v^2 + \Phi$, $J = r v_t$. The Gaussian $\exp(-J^2/2r_a^2\sigma^2)$
depopulates high-angular-momentum (tangential) orbits — increasingly so at large $r$
(since $J^2\propto r^2$) — so the velocity ellipsoid is isotropic at the center and
can become radially biased at intermediate radii:

```{math}
\beta(r) \equiv 1 - \frac{\sigma_t^2}{\sigma_r^2}:\quad \beta(0)\approx 0,\quad
\beta(r_t^-)\to0\ \text{for a finite truncation},\qquad
r_a\to\infty \Rightarrow \beta\equiv0\ \text{(King)}.
```

Here $\sigma_t^2$ is a **single** tangential component (the convention used
throughout progenax). Binney & Tremaine write $\beta = 1 - \sigma_t^2/(2\sigma_r^2)$
with $\sigma_t^2 \equiv \sigma_\theta^2 + \sigma_\phi^2$ the **two-component**
tangential dispersion; the factor of 2 cancels, so both give the *same* $\beta(r)$
(see [the Merritt 1985 note](../../99-bibliography/per-paper/merritt-1985.md)).

Unlike the [Osipkov-Merritt](../../99-bibliography/per-paper/merritt-1985.md)
construction (which holds a *given* density fixed and inverts for $f$), Michie **specifies**
$f$ and re-solves Poisson, so the density itself changes — hence $\beta(r)$ is the
DF-implied profile, **not** the OM form $r^2/(r^2+r_a^2)$. Near a finite
edge, allowed speeds vanish, so the angular factor tends to one and the
velocity distribution returns toward isotropy. Thus $r_a$ is not generally
the radius where Michie anisotropy equals one half.

## Self-consistency

The density is the velocity integral of {eq}`mk-df`, which depends on radius explicitly
through the anisotropy term, giving a radius-dependent King ODE — the King-1966
nondimensionalization of Michie (1963) Eq. 5.8, whose anisotropy weight
$e^{-Cz^2\eta^2(1-\mu^2)}$ is our $e^{-J^2/2r_a^2\sigma^2}$ ($\xi=r/r_c$,
$\hat r_a = r_a/r_c$):

```{math}
:label: mk-poisson
\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2\frac{d\psi}{d\xi}\right)
  = -9\,\frac{\hat\rho(\psi,\ \xi/\hat r_a)}{\hat\rho(W_0,0)},
\qquad \psi(0)=W_0,\ \psi'(0)=0.
```

[↗ model card](#card-mk-poisson)

`progenax` solves this with `diffrax` (differentiable), the velocity scale fixed
self-consistently as $\sigma^2 = GM/(9 r_c \mu)$ — so the ICs are virial ($Q\approx0.5$)
without any external rescale.

```{warning}
**Finite extent must be checked.** Strong radial anisotropy can produce an
extended or nontruncating model. A positive potential at the numerical endpoint
alone does not distinguish an infinite model from an insufficient integration
domain. The current solver raises a diagnostic for concrete Python scalar
inputs when its endpoint-potential check fails; that guard is skipped during
JAX tracing. Verify a zero crossing and domain convergence for the requested
configuration. The retained $W_0=7$ examples are much more extended than the
isotropic King model; this does not qualify an arbitrary anisotropy radius.

```

## Usage

```python
from jaxstro.units import STELLAR
from progenax import MichieProfile, MichieVelocityDF

profile = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)   # positions
df = MichieVelocityDF(W0=7.0, r_c=1.0, r_a=8.0)               # velocities (2-D sampler)

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

The default sampler caches an anisotropic speed-marginal CDF table and draws
the angle conditional on speed. `speed_method="quadrature"` retains the
per-star two-dimensional marginal/conditional inverse-CDF reference. Neither
is an OM stretch of an energy-only DF: {eq}`mk-df` depends separately on energy
and angular momentum. Selected fixed-key derivatives have retained checks;
ODE, table, quadrature, and interpolation convergence remain relevant.

## Validation

- **Isotropic limit** — $r_a\to\infty$ reproduces the validated King profile and DF
  (matching $\xi_t$ and density).
- **Anisotropy** — the realized moments are compared with the Michie DF
  reference at the tested radii; central and finite-edge isotropic limits
  differ from the OM anisotropy law.
- **Virial** — $Q=T/|V|\approx0.5$ unscaled.
- **Differentiability** — finite, FD-matching gradients w.r.t. the model parameters.

## Implementation, validation & references

- **In code:** `src/progenax/kinematics/michie_df.py` (the 2-D
  marginal-then-conditional sampler) paired with
  `src/progenax/profiles/michie.py` (`solve_michie_profile`,
  `MichieProfile`) — see the [kinematics API](../../30-api/kinematics.md)
  and the [profiles API](../../30-api/profiles.md).
- **Validated in:** [Michie anisotropy](../../50-validation/michie-anisotropy.md)
  — the isotropic-limit (→ King), the realized $\beta(r)$ profile, the
  unscaled virial $Q \approx 0.5$, and FD-matching gradients.
- **Primary sources:** {cite:t}`Michie1963` (the angular-momentum
  anisotropy term) and {cite:t}`King1966` (the lowered-Maxwellian
  cutoff) — full notes in the
  [Michie (1963)](../../99-bibliography/per-paper/michie-1963.md) and
  [King (1966)](../../99-bibliography/per-paper/king-1966.md) per-paper
  bibliography pages.
