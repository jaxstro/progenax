---
title: Binney & Mamon (1982)
description: Annotated reference for J. Binney & G. A. Mamon — M/L and velocity anisotropy from observations of spherical galaxies — the line-of-sight projection integral (Eq. 7) behind progenax's project_dispersion.
---

# Binney & Mamon (1982)

```{admonition} M/L and velocity anisotropy from observations of spherical galaxies, or must M87 have a massive black hole?
:class: note

**Authors.** James Binney (present address Department of Theoretical Physics, Oxford)
& Gary A. Mamon (Princeton University Observatory).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **200**, 361–375 (1982).
Received 1981 December 7; in original form 1981 September 7.

**ADS.** [1982MNRAS.200..361B](https://ui.adsabs.harvard.edu/abs/1982MNRAS.200..361B)
```

## Abstract (paraphrased)

Given the projected surface brightness $\Sigma(R)$ and the line-of-sight velocity dispersion
$\sigma_v(R)$ of a spherical galaxy as functions of projected radius, Binney & Mamon construct
the *unique* constant mass-to-light-ratio model consistent with both, and recover as an **output**
the radial dependence of the velocity-anisotropy profile $\beta(r)$ — without assuming isotropy.
The method is an Abel-type inversion of the projected Jeans equation (their §2). Applied to M87,
it returns a physically reasonable, radially anisotropic model with constant $M/L_V\approx7.6$
that needs no central black hole. The single relation this paper contributes to progenax is the
**line-of-sight projection integral** (their Eq. 7), which maps an anisotropic 3-D model
$(l, \sigma_r, \beta)$ onto the observed $\sigma_v(R)\,\Sigma(R)$ on the sky.

## The load-bearing equations (verified against the paper, §2, p. 363)

**Anisotropy parameter (Eq. 1, p. 363).** With $\sigma_r,\sigma_\theta$ the radial and (single)
tangential principal dispersions,

```{math}
:label: bm82-beta
\beta \equiv \frac{\sigma_r^2 - \sigma_\theta^2}{\sigma_r^2}.
```

$\beta=0$ is isotropic; $\beta\to1$ is purely radial. (progenax supplies $\beta(r)$ from the
Osipkov–Merritt law $\beta=r^2/(r^2+r_a^2)$ of [](merritt-1985.md); BM82 itself is agnostic to the
functional form of $\beta$.)

**Surface-density / luminosity-density Abel pair (Eq. 2, p. 363).** The luminosity density
$l(r)$ is recovered from the surface brightness $\Sigma(R)$ by the standard Abel deprojection

```{math}
:label: bm82-abel
l(r) = -\frac{1}{\pi}\int_r^{R_t}\frac{d\Sigma}{dR}\,\frac{dR}{(R^2-r^2)^{1/2}}.
```

Its **forward** companion — projecting a 3-D density $l(r)$ (or $\rho(r)$) onto the sky — is the
inverse-Abel relation
$\Sigma(R) = 2\int_R^{R_t} l(r)\,r\,(r^2-R^2)^{-1/2}\,dr$,
which is what progenax evaluates for the projection normalisation $\Sigma$.

**Line-of-sight projection integral (Eq. 7, p. 363) — the central result for progenax.**
"Elementary geometry" gives the observed line-of-sight dispersion as

```{math}
:label: bm82-eq7
\tfrac12\,\sigma_v^2(R)\,\Sigma(R)
  = \int_R^{R_t}\frac{l(r)\,\sigma_r^2(r)}{(r^2-R^2)^{1/2}}\,r\,dr
  \;-\; R^2\int_R^{R_t}\frac{\beta(r)\,l(r)\,\sigma_r^2(r)}{r\,(r^2-R^2)^{1/2}}\,dr.
```

The two integrals share the measure $r/(r^2-R^2)^{1/2}$ (the first carries weight $r$, the
second weight $R^2/r$), so they combine into a **single anisotropic kernel**. Multiplying through
by 2:

```{math}
:label: bm82-los-kernel
\sigma_v^2(R)\,\Sigma(R)
  = 2\int_R^{R_t}\!\left(1 - \beta(r)\,\frac{R^2}{r^2}\right)
        l(r)\,\sigma_r^2(r)\,\frac{r}{(r^2-R^2)^{1/2}}\,dr.
```

{eq}`bm82-los-kernel` is the form implemented in `project_dispersion`: the line-of-sight kernel
$\bigl(1-\beta R^2/r^2\bigr)$, the weight $l\,\sigma_r^2$, the factor 2, and the
$r/(r^2-R^2)^{1/2}$ projection measure are *exactly* Eq. (7) regrouped. (progenax uses the mass
density $\rho$ as the projection weight under the paper's own mass-follows-light assumption
$\rho = A\,l$, Eq. 3; the constant $A$ cancels in $\sigma_v^2 = (\sigma_v^2\Sigma)/\Sigma$.)

```{admonition} Scope of BM82 — line-of-sight ONLY (verified, whole paper read)
:class: important
The entire paper concerns the **line-of-sight** velocity dispersion $\sigma_v(R)$. Equation (7)
is the *only* projection formula it contains; there is **no proper-motion** (on-sky radial /
tangential) projection anywhere in Binney & Mamon (1982) (pp. 361–375, including Appendices A–C,
verified page by page). The proper-motion kernels $(1-\beta+\beta R^2/r^2)$ and $(1-\beta)$ that
`project_dispersion` also returns are the standard later generalisation of the same projection
geometry (Leonard & Merritt 1989; Strigari, Bullock & Kaplinghat 2007; Mamon & Łokas 2005) — they
are **not** from BM82 and should be attributed to those works, not to this paper.
```

## Code cross-reference (`src/progenax/kinematics/dispersion.py`)

`project_dispersion` (def at line 677) folds the $r/(r^2-R^2)^{1/2}$ pole away analytically via
$r^2 = R^2 + u^2$ (so $r\,dr/(r^2-R^2)^{1/2} = du$) and integrates a smooth quadrature in $u$
(line 810, `r = sqrt(R_i**2 + u**2)`). With `ratio = R²/r²` (line 822) and
`w = rho * sigma_r2` (line 823):

| Quantity | Code (line) | Kernel in code | Paper source |
|---|---|---|---|
| $\Sigma(R)$ | `Sigma`, L829: `2*trapezoid(rho*jac, grid)` | $2\int \rho\, r/\sqrt{r^2-R^2}\,dr$ | Forward inverse-Abel of {eq}`bm82-abel` (Eq. 2) |
| $\sigma_{\rm los}^2$ | `S_los`, L830: `(1 - beta*ratio)*w` | $1 - \beta\,R^2/r^2$ | **BM82 Eq. 7** → {eq}`bm82-los-kernel` ✓ |
| $\sigma_{\rm pm,R}^2$ | `S_pmr`, L831: `(1 - beta + beta*ratio)*w` | $1 - \beta + \beta\,R^2/r^2$ | **NOT in BM82** (Leonard & Merritt 1989; Strigari+2007) |
| $\sigma_{\rm pm,T}^2$ | `S_pmt`, L832: `(1 - beta)*w` | $1 - \beta$ | **NOT in BM82** (same later PM works) |
| $\beta(r)$ | L820: `r**2/(r**2 + r_a2)` | OM law | [](merritt-1985.md), not BM82 |

The factor 2 (L829–832), the $l\,\sigma_r^2$ weight, and the substitution that removes
$1/\sqrt{r^2-R^2}$ all match Eq. (7) cell-by-cell. The isotropic limit ($r_a=$`None`
$\Rightarrow\beta=0$, L818) collapses all three kernels to 1, so
$\sigma_{\rm los}=\sigma_{\rm pm,R}=\sigma_{\rm pm,T}$ — the correct $\beta=0$ behaviour of Eq. (7).

### Verification verdict

**MATCH (with an attribution caveat).** The line-of-sight kernel `(1 - beta*ratio)` in `S_los`
(L830) is algebraically identical to Binney & Mamon (1982) Eq. (7), regrouped as
{eq}`bm82-los-kernel`; the $\Sigma$ normalisation is the forward partner of their Eq. (2) Abel
pair. No physics discrepancy in the $\sigma_{\rm los}$ / $\Sigma$ channels.

The **only** issue is *provenance, not formula*: the `project_dispersion` docstring (L678, L682,
L825 "B&M82 kernels") and the module header (L11–13, L21–22) credit *all three* projection
kernels to Binney & Mamon (1982), but BM82 contains the line-of-sight kernel **only**. The
proper-motion kernels $(1-\beta+\beta R^2/r^2)$ and $(1-\beta)$ are the standard later extension
(Leonard & Merritt 1989, ApJ 339, 195; Strigari, Bullock & Kaplinghat 2007; Mamon & Łokas 2005).
This is a docstring/citation fix for Phase 2 — **no source-code logic change** — flagged here for
the closeout, not fixed in this note's pass.

## Use in progenax

- `progenax.kinematics.dispersion.project_dispersion` — line-of-sight projection of the
  anisotropic Jeans model ({eq}`bm82-los-kernel`, BM82 Eq. 7) onto on-sky radii $R$, returning
  $\sigma_{\rm los}(R)$ and the projected surface density $\Sigma(R)$.
- Pairs with [](merritt-1985.md) (the Osipkov–Merritt $\beta(r)$ supplied to the kernel) and the
  3-D anisotropic Jeans solver that produces $\sigma_r^2(r)$.
- The isotropic-Plummer closed form $\sigma_{\rm los}^2(R) = (3\pi/64)\,GM/\sqrt{a^2+R^2}$
  (Dejonghe 1987) is the tight absolute validation anchor for the $\sigma_{\rm los}$ channel in
  `test_dispersion_physics.py`.

## Notes

- BM82's broader algorithm (the Abel inversion that recovers $\sigma_r(r)$, $\beta(r)$ and $M/L$
  *from* observed $\Sigma(R)$ and $\sigma_v(R)$, their Eqs. 8–20 and Appendices A–B) is the
  inverse problem; progenax uses only the **forward** projection, Eq. (7), as a differentiable
  forward model for optimal experimental design.
- The convention $\beta = (\sigma_r^2-\sigma_\theta^2)/\sigma_r^2$ (Eq. 1) uses a **single**
  tangential component $\sigma_\theta$; this is consistent with the Osipkov–Merritt
  $\beta(r)=r^2/(r^2+r_a^2)$ used in progenax (the single-vs-two-component factor of 2 is
  reconciled in [](merritt-1985.md)).
- BM82 restricts to $\beta<1$ throughout (p. 372): the radial component never exceeds the total
  available — a physical bound the projection inherits.
