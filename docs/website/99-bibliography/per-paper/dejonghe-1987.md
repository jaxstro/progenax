---
title: Dejonghe (1987)
description: Annotated reference for H. Dejonghe — A completely analytical family of anisotropic Plummer models (the sigma_los oracle and the q-family of Plummer DFs).
---

# Dejonghe (1987)

```{admonition} A completely analytical family of anisotropic Plummer models
:class: note

**Author.** Herwig Dejonghe (School of Natural Sciences, Institute for Advanced Study, Princeton).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **224**, 13–39 (1987).
Received 1986 June 10; accepted 1986 June 11.

**DOI.** [10.1093/mnras/224.1.13](https://doi.org/10.1093/mnras/224.1.13) ·
**ADS.** [1987MNRAS.224...13D](https://ui.adsabs.harvard.edu/abs/1987MNRAS.224...13D)
```

## Abstract (paraphrased)

A one-parameter family of anisotropic distribution functions all consistent with the **same
Plummer mass density**, illustrating the DF indeterminacy of a given density. The family
parameter $q$ tunes the orbital structure: $q = 0$ is the isotropic model, $0 < q \le 2$
radially anisotropic ("radial clusters"), $q < 0$ tangential. Moments, energy distributions,
and observable line profiles are all analytic — making the paper a premier source of exact
oracles for Plummer-family kinematics.

## What progenax uses (all PDF-verified 2026-07-10/11)

Model units $G = M = a = 1$ throughout ($\psi = (1+r^2)^{-1/2}$, Eq. 13).

- **Eq. 14–16 — the isotropic chain.** $\rho = \tfrac{3}{4\pi}(1+r^2)^{-5/2}$,
  $\rho(\psi) = \tfrac{3}{4\pi}\psi^5$, and the isotropic DF
  $F(E) = \tfrac{3}{7\pi^3}(2E)^{7/2}$ — since $3 \cdot 2^{7/2} = 24\sqrt{2}$, this
  independently confirms progenax's Plummer DF coefficient
  $24\sqrt{2}/(7\pi^3)$ ([theory page](../../10-theory/velocity-dfs/plummer-dfs.md)).
- **Eq. 17 — the intrinsic dispersion.** $\sigma^2(r) = \tfrac16 (1+r^2)^{-1/2}$, i.e.
  $\sigma_r^2 = GM/(6\sqrt{r^2+a^2})$ — the `PlummerVelocityDF` moment oracle.
- **Eqs. 22a/22b/23 — the anisotropic moments.** $\sigma_r^2 = \tfrac{1}{6-q}(1+r^2)^{-1/2}$
  and Binney's $\beta(r) = \tfrac{q}{2}\,\tfrac{r^2}{1+r^2}$ (Eq. 23) — an OM-like
  monotone anisotropy profile.
- **Eq. 21 (p. 18) — the Merritt bridge.** The limiting member $q = 2$ is *identical* to the
  Merritt (1985) model with $r_a = 1$: an independent cross-check between the two anisotropic
  constructions progenax implements (see the
  [om_anisotropy model card](../../15-model-reference/velocity_dfs.md)).
- **Eq. 43 (p. 24) — THE projected-dispersion oracle.** Via the Binney & Mamon (1982)
  projection integral (his Eq. 42),

  ```{math}
  \sigma_p^2(r_p) \;=\; \frac{3\pi}{32}\,\frac{1}{6-q}\,
  \frac{1}{\sqrt{1+r_p^2}}\left(3 - \frac{5q}{4}\,\frac{r_p^2}{1+r_p^2}\right).
  ```

  At $q = 0$ this reduces to $\sigma_p^2 = \tfrac{3\pi}{64}(1+r_p^2)^{-1/2}$ — restoring
  units, $\sigma_{\rm los}^2(R) = \tfrac{3\pi}{64}\,GM/\sqrt{a^2+R^2}$, the **tight absolute
  oracle** used by `tests/validation/test_dispersion_physics.py` for
  `project_dispersion`. Formerly cited there as a "standard result"; now source-verified.
  Dejonghe notes all members pass through
  $\sigma_p^2 = 3\pi\sqrt{3/5}/64$ at $r_p = \sqrt{2/3}$ — a family-invariant point.

## Connections in progenax

- `progenax.kinematics.dispersion.project_dispersion` — the LOS/PM projection whose Plummer
  anchor is Eq. 43 ($q=0$); the projection kernel itself is Eq. 42 = Binney & Mamon (1982).
- [plummer_df model card](../../15-model-reference/velocity_dfs.md) — cites Eqs. 16/17/43.
- [Merritt (1985) note](merritt-1985.md) — the $q=2 \leftrightarrow r_a=1$ bridge.
