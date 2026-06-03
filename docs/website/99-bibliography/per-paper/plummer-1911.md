---
title: Plummer (1911)
description: Annotated reference for H. C. Plummer — On the problem of distribution in globular star clusters.
---

# Plummer (1911)

```{admonition} On the problem of distribution in globular star clusters
:class: note

**Authors.** H. C. Plummer, M.A.

**Reference.** *Monthly Notices of the Royal Astronomical Society* **71**, 460–470 (1911,
March; Plate 8).

**ADS.** [1911MNRAS..71..460P](https://ui.adsabs.harvard.edu/abs/1911MNRAS..71..460P) ·
**DOI.** [10.1093/mnras/71.5.460](https://doi.org/10.1093/mnras/71.5.460)
```

## Abstract (paraphrased)

Seeks a *physical* basis for the spatial distribution of stars in globular clusters from star
counts. Plummer adopts Schuster's (1883) closed-form solution of the polytropic
(convective-equilibrium) gas sphere for $\gamma = 1.2$ — equivalently the $n = 5$ polytrope —
as the space-density law, and shows it reproduces the projected star counts of ω Centauri and
other clusters.

## The density law (verified against the paper, §5–6)

Plummer carefully distinguishes the **space density** $\phi(r)$ (stars per volume), the cylinder
counts $F(r)$, and the **projected** counts $f(r)$ / $\Sigma(r)$ (perpendicular to the line of
sight). Schuster's $\gamma = 1.2$ polytrope solution (his Eq. 11) gives the **space density**

```{math}
:label: plummer-rho
\phi(r) = N\,(1 + r^2)^{-5/2}   \qquad\text{(Plummer Eq. 11–12, in units of the scale radius } a=1\text{)}
```

i.e. the canonical $\rho(r) = \rho_0\,[1 + (r/a)^2]^{-5/2}$. The projected (surface) count is the
*different* $f(r) = \tfrac{4}{3}N(1+r^2)^{-2}$, $\sigma(r) = \tfrac{4}{3}\pi N r^2 (1+r^2)^{-1}$
(Eq. 13). The model is fit to ω Centauri (his Table I) and a second cluster (Table II) with good
agreement to the counts $\Sigma(r) = 3540\,r/\sqrt{1+r^2}$.

```{admonition} What is — and isn't — in the 1911 paper
:class: note
Plummer (1911) introduces the **space-density law** (from Schuster's polytrope) and fits it to
star counts. The closed-form **potential** $\Phi(r) = -GM/\sqrt{r^2 + a^2}$ and the isotropic
**distribution function** $f(E) \propto (-E)^{7/2}$ are standard later results, obtained from the
density via Eddington inversion (Eddington 1916) — *not* derived in the 1911 paper, which
predates the inversion method. progenax's $\rho(r) \propto [1+(r/a)^2]^{-5/2}$ is faithful to
Plummer's space density $\phi$; `a` is Plummer's scale radius (his unit of length).
```

## Use in progenax

- [](../../10-theory/spatial-profiles/plummer.md) — derivation and progenax implementation
  (half-mass→scale-radius $a = r_h\sqrt{2^{2/3}-1} \approx 0.766\,r_h$)
- [](../../10-theory/velocity-dfs/plummer-dfs.md) — the matched Eddington-inversion DF
  $f(E) \propto (-E)^{7/2}$
- `progenax.profiles.PlummerProfile` — production-default spatial profile
- `progenax.kinematics.PlummerVelocityDF` — matched equilibrium velocity DF

## Notes

The simplest self-consistent cluster equilibrium with a closed-form density, potential, and DF —
a research tool (production ICs) and the canonical pedagogical example of Eddington inversion.
The $n=5$ polytrope is the unique polytrope with finite mass yet infinite extent.
