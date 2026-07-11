---
title: Michie (1963)
description: Annotated reference for R. W. Michie — On the distribution of high energy stars in spherical stellar systems (the anisotropic lowered-Maxwellian / "Michie-King" model).
---

# Michie (1963)

```{admonition} On the distribution of high energy stars in spherical stellar systems
:class: note

**Author.** Richard W. Michie (University of California, Berkeley).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **125** (2), 127–139
(1963). Received 1962 June 1.

**ADS.** [1963MNRAS.125..127M](https://ui.adsabs.harvard.edu/abs/1963MNRAS.125..127M)
```

## Abstract (paraphrased)

Derives a distribution function $f(r,v,\mu;t)$ for an isolated spherical stellar system
from the Boltzmann equation with Fokker–Planck encounters, **self-consistently** with
Poisson's equation. The orbits are *not* assumed isotropic: the model becomes
increasingly **radially anisotropic** at large radius. The analysis focuses on the
high-energy tail (stars near the escape energy), where the velocity-space flux is shown
to be nearly constant, giving the energy cutoff. This is the origin of the anisotropic
lowered-Maxwellian now universally combined with King's (1966) cutoff as the
**"Michie–King" model**.

## The distribution function (verified against the paper, §4–5)

```{note}
Re-verified equation-by-equation against the PDF on 2026-06-16 (Eqs. 4.0–4.3, 4.8, 5.0,
5.3, 5.4–5.9) as the provenance grounding for the packaged differentiable Michie DF
second-moment forward model `df_moment_dispersion` (progenax `kinematics/dispersion.py`).
The anisotropy term $\exp(-\beta J^2)$ (Eq. 4.0, $J^2=r^2v^2(1-\mu^2)$) maps to
$\exp(-s^2 u_t^2/2)$ ($s=r/r_a$, $u_t=v_t/\sigma$, $\beta\equiv 1/2r_a^2\sigma^2$); the
$[\exp(-E/\sigma^2)-1]$ cutoff is King (1966), **not** Michie's Fokker–Planck $Q$ (Eq. 4.8).
```

Michie writes (Eqs. 4.0–4.3, 5.0)

```{math}
:label: michie-df
f(r,v,\mu;t) = A\,\exp\!\left[-\tfrac{m}{m_0}\left(\alpha E + \beta J^2\right)\right] Q,
\qquad E = \tfrac12 v^2 + \Phi(r),\quad J^2 = r^2 v^2 (1-\mu^2),
```

with $\mu = \cos\theta$ ($\theta$ the angle between $\mathbf r$ and $\mathbf v$),
$\Phi(0)=0$, and $\alpha,\beta$ model constants. The two ingredients:

- **Anisotropy** — the Gaussian factor $\exp(-\tfrac{m}{m_0}\beta J^2)$ depopulates
  high-angular-momentum (circular) orbits, the more so at large $r$ (since $J^2\propto
  r^2$). This makes the velocity ellipsoid increasingly **radial outward** — Michie's
  central result. The strength is set by $\beta/\alpha$.
- **Cutoff** — $Q(E)$ (Eq. 4.8) is a Fokker–Planck-derived **high-energy depopulation
  function** ($Q\to1$ at $E=0$, $Q\to0$ at the escape energy $E_e$), *not* a lowered
  Maxwellian. It carries a slight $J^2$-dependent correction (Eq. 5.3).

:::{admonition} What "Michie-King" means — and what progenax implements
:class: important
Michie's *literal* cutoff $Q$ (Eq. 4.8) is more elaborate than King's. Modern IC codes
(Gunn & Griffin 1979; LIMEPY, Gieles & Zocchi 2015) use the **Michie anisotropy term**
$\exp(-J^2/2r_a^2\sigma^2)$ **with King's (1966) lowered-Maxwellian cutoff**
$[\exp(-E/\sigma^2)-1]$:

```{math}
:label: michie-king-df
f(E,J) \propto \exp\!\left(-\frac{J^2}{2 r_a^2 \sigma^2}\right)
   \left[\exp\!\left(-\frac{E}{\sigma^2}\right) - 1\right],\qquad E \le 0,
```

This is the standard **Michie–King** model. progenax adopts {eq}`michie-king-df`
(Michie 1963 anisotropy + King 1966 cutoff), not Michie's full $Q$ — the $Q$ refinement
is a high-energy-tail correction irrelevant to equilibrium ICs.
:::

## Self-consistency: the anisotropic King ODE (verified, §5)

In dimensionless variables $\phi=\alpha\Phi$, $z^2 = r^2 A\alpha G(4\pi)^2(2/\alpha)^{3/2}m_0$,
$\eta = v\sqrt{\alpha/2}$ (Eqs. 5.4–5.6), Poisson's equation becomes (Eq. 5.8)

```{math}
:label: michie-poisson
\frac{1}{z^2}\frac{d}{dz}\!\left(z^2\frac{d\phi}{dz}\right)
 = e^{-\phi}\int_0^{\eta_e}\!\!\int_{-1}^{1}
   e^{-\eta^2}\,e^{-C z^2\eta^2(1-\mu^2)}\,Q\,\eta^2\,d\mu\,d\eta.
```

The right-hand side is the **density**, which now depends on $z$ *explicitly* through the
anisotropy term $e^{-Cz^2\eta^2(1-\mu^2)}$ ($J^2\propto z^2\eta^2(1-\mu^2)$) — so the King
ODE becomes radius-dependent and the resulting **density profile differs from the
isotropic King model**. The single **model parameter** is (Eq. 5.9)

```{math}
:label: michie-C
C = \frac{\beta}{A(2\alpha)^{1/2}G(4\pi)^2}.
```

$C\to 0$ recovers the isotropic King model; anisotropy becomes important where
$Cz^2\sim\tfrac12$. In modern units $C \propto 1/r_a^2$: the anisotropy radius $r_a$ (the
radius where $\beta\to 0.5$) is the same parameter as Michie's $C$, rescaled.

## Use in progenax

- `progenax.kinematics` / `progenax.profiles` — the **implemented Michie–King anisotropic
  model** (`MichieProfile` + `MichieVelocityDF`): solve {eq}`michie-poisson` for
  $\rho(r),\Phi(r),r_t$ given $(W_0, r_a)$, then sample $(v_r,v_t)$ from
  {eq}`michie-king-df`. Distinct from the isotropic [](king-1966.md) (different density)
  and from the [](merritt-1985.md) Osipkov–Merritt construction (which holds the density
  fixed; Michie does not).
- The realised anisotropy increases outward, $\beta(r)\to 1$ at large $r$ inside $r_t$.

## Notes

Michie's $\exp(-\beta J^2)$ Gaussian-in-$J^2$ anisotropy is the **same functional form**
Merritt (1985) notes for Eddington's (1914) generalised isothermal sphere; the difference
is that Osipkov–Merritt holds a *given* density fixed and inverts for $f(Q)$, whereas
Michie *specifies* $f(E,J)$ and solves Poisson for a new, self-consistent (more radial)
density. The Michie–King model is the anisotropic, tidally-truncated workhorse for
globular-cluster ICs.
