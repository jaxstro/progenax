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

## The distribution function

```{math}
:label: mk-df
f(E, J) \propto \exp\!\left(-\frac{J^2}{2 r_a^2 \sigma^2}\right)
   \left[\exp\!\left(-\frac{E}{\sigma^2}\right) - 1\right],\qquad E \le 0,
```

with $E=\tfrac12 v^2 + \Phi$, $J = r v_t$. The Gaussian $\exp(-J^2/2r_a^2\sigma^2)$
depopulates high-angular-momentum (tangential) orbits — increasingly so at large $r$
(since $J^2\propto r^2$) — so the velocity ellipsoid is isotropic at the centre and
becomes radial outward:

```{math}
\beta(r) \equiv 1 - \frac{\sigma_t^2}{2\sigma_r^2}:\quad \beta(0)\approx 0,\quad
\beta \nearrow \text{ outward},\qquad r_a\to\infty \Rightarrow \beta\equiv 0 \text{ (King)}.
```

Unlike the [Osipkov-Merritt](../../99-bibliography/per-paper/merritt-1985.md)
construction (which holds a *given* density fixed and inverts for $f$), Michie **specifies**
$f$ and re-solves Poisson, so the density itself changes — hence $\beta(r)$ is the
DF-implied profile, **not** the OM form $r^2/(r^2+r_a^2)$.

## Self-consistency

The density is the velocity integral of {eq}`mk-df`, which depends on radius explicitly
through the anisotropy term, giving a radius-dependent King ODE (Michie 1963, Eq. 5.8;
$\xi=r/r_c$, $\hat r_a = r_a/r_c$):

```{math}
\frac{1}{\xi^2}\frac{d}{d\xi}\!\left(\xi^2\frac{d\psi}{d\xi}\right)
  = -9\,\frac{\hat\rho(\psi,\ \xi/\hat r_a)}{\hat\rho(W_0,0)},
\qquad \psi(0)=W_0,\ \psi'(0)=0.
```

`progenax` solves this with `diffrax` (differentiable), the velocity scale fixed
self-consistently as $\sigma^2 = GM/(9 r_c \mu)$ — so the ICs are virial ($Q\approx0.5$)
without any external rescale.

```{warning}
**Maximum anisotropy.** Below a $W_0$-dependent threshold (roughly $r_a \lesssim 3$–$4\,r_c$
at $W_0=7$) the radial orbits build a $1/r^2$ density tail and the model **never truncates**
— infinite mass, no tidal radius (the radial-orbit pathology). `solve_michie_profile` /
`MichieProfile` **raise `ValueError`** for such an over-anisotropic choice. Valid models are
far more extended than King ($\xi_t$: 34 isotropic → ~545 at $r_a=5\,r_c$, $W_0=7$).
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

Velocities are drawn by a **2-D marginal-then-conditional inverse-CDF** over $(v_r, v_t)$
(no Osipkov-Merritt stretch trick — {eq}`mk-df` is not a function of one integral). The
whole pipeline (ODE solve + sampler) is JIT-compatible and differentiable in $W_0$, $r_c$,
$r_a$ (FD-verified).

## Validation

- **Isotropic limit** — $r_a\to\infty$ reproduces the validated King profile and DF
  (matching $\xi_t$ and density).
- **Anisotropy** — realised $\beta(r)$ isotropic at the centre, increasing outward.
- **Virial** — $Q=T/|V|\approx0.5$ unscaled.
- **Differentiability** — finite, FD-matching gradients w.r.t. the model parameters.

## References

- [](../../99-bibliography/per-paper/michie-1963.md) — Michie (1963), MNRAS 125, 127.
- [](../../99-bibliography/per-paper/king-1966.md) — King (1966), AJ 71, 64.
