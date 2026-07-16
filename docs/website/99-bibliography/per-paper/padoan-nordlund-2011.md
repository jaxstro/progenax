---
title: Padoan & Nordlund (2011)
description: Annotated reference for P. Padoan & Å. Nordlund — the star formation rate of supersonic magnetohydrodynamic turbulence.
---

# Padoan & Nordlund (2011)

```{admonition} The star formation rate of supersonic magnetohydrodynamic turbulence
:class: note

**Authors.** P. Padoan, Å. Nordlund

**Reference.** *The Astrophysical Journal* **730, 40** (2011).

**DOI.** [10.1088/0004-637X/730/1/40](https://doi.org/10.1088/0004-637X/730/1/40)

**Verified.** Equations + prefactors checked against the held PDF (pp. 1–4, Eqs. 1, 3–4, 8–11, 18; re-verified 2026-06).
```

## The big idea

PN11 derive the **critical density for gravitational collapse** in supersonic turbulence
from first principles, rather than from the turbulent-pressure / sonic-scale argument of
Krumholz & McKee (2005). The physical picture: turbulence drives shocks; the post-shock
gas is dense and, *if* a post-shock region's Bonnor–Ebert mass falls below its actual
mass, it collapses. The **minimum** density at which this happens is the critical density
$\rho_\mathrm{crit}$. The star-formation rate then follows from the mass fraction of the
turbulent density PDF lying above $\rho_\mathrm{crit}$.

In `gravoturb` PN11 is kept as a **clearly-labelled classical alternative** to the
BM19 transition density $s_t$ — *not* the default path.

## Core equations

**Shock jump → characteristic post-shock density.** Balancing thermal and ram pressure
for a shock of velocity $v_0/2$ (Eq. 3) gives the post-shock density (Eq. 4)

$$
\frac{\rho_\mathrm{HD}}{\rho_0} = \frac{\mathcal{M}_{S,0}^2}{4},
$$

where $\mathcal{M}_{S,0}$ is the rms sonic Mach number.

**Critical density for collapse (hydrodynamic case, Eq. 8).** Setting the Bonnor–Ebert
mass equal to the mass of a post-shock region, $M_\mathrm{HD}(\rho_\mathrm{cr})=M_\mathrm{BE}(\rho_\mathrm{cr})$,
yields

$$
\frac{\rho_\mathrm{cr,HD}}{\rho_0}
   = 0.067\,\theta^{-2}\,\alpha_\mathrm{vir}\,\mathcal{M}_{S,0}^{2},
$$

where $\theta\le1$ is the fraction of the cloud size set by the turbulence integral
scale. PN11 adopt $\theta = 0.35$ (their Section 2, after Wang & George 2002), giving the
**numerical critical density** (Eq. 11)

$$
\frac{\rho_\mathrm{cr,HD}}{\rho_0}
   = 0.547\,\alpha_\mathrm{vir}\,\mathcal{M}_{S,0}^{2}
   \qquad (0.067\times0.35^{-2}=0.547).
$$

In log-density this is the PN11 critical threshold

$$
s_\mathrm{crit} = \ln\!\left(0.067\,\theta^{-2}\,\alpha_\mathrm{vir}\,\mathcal{M}_{S,0}^{2}\right).
$$

**Virial parameter (Eqs. 1, 9).** Equivalently
$\alpha_\mathrm{vir} = 5\sigma_{v,\mathrm{1D}}^2 R/(GM)$ (Bertoldi & McKee 1992; Eq. 1) or
$\alpha_\mathrm{vir} = 5 v_0^2 / (\pi G \rho_0 L_0^2)$ (Eq. 9, a uniform sphere of radius
$L_0/2$, mean density $\rho_0$, 3-D rms velocity $v_0$). `gravoturb`'s
`virial_parameter` helper uses the Eq. 1 form. $\alpha_\mathrm{vir}\approx1$ is virial
equilibrium; the SFR *decreases* with increasing $\alpha_\mathrm{vir}$ (more turbulent
support relative to gravity) and increases with $\mathcal{M}_{S,0}$.

**MHD generalisation (Eq. 18).** Including magnetic pressure ($\beta_0 = 2c_s^2/v_{A,0}^2$
the ratio of gas to magnetic pressure),

$$
\frac{\rho_\mathrm{cr,MHD}}{\rho_0}
  = 0.067\,\theta^{-2}\,\alpha_\mathrm{vir}\,\mathcal{M}_{S,0}^{2}\,
    \frac{\left(1+0.925\,\beta_0^{-3/2}\right)^{2/3}}{\left(1+\beta_0^{-1}\right)^{2}},
$$

which reduces to the HD case as $\beta_0\to\infty$. `gravoturb` implements only the
HD form (Eq. 8/11); the MHD form is noted but not coded.

## Use in progenax

- `experimental/gravoturb/theory/collapse_threshold.py` — `critical_overdensity` (Eq. 8),
  `critical_log_density` (classical-alternative threshold), `THETA_PN11 = 0.35`.

Validation: prefactor $0.067\,\theta^{-2}=0.547$ at $\theta=0.35$; $\partial s_\mathrm{crit}/\partial\mathcal{M}=2/\mathcal{M}$.

## Notes

- **Prefactor provenance.** The correct prefactor is **0.547** (PN11 Eq. 11). A prior
  implementation used **0.242** (~2.3× too small) — corrected in the clean-room rewrite.
- PN11's $\rho_\mathrm{crit}$ is *distinct* from the Krumholz–McKee (2005) turbulent-
  pressure form and from the Federrath–Klessen (2012) $(\pi^2\phi_x^2/5)$ form; BM19 also
  recast their own $s_t$ in terms of an $\alpha_\mathrm{vir}\mathcal{M}^2$ critical density
  (BM19 Eq. 11/15), so the forms are related but not identical.
