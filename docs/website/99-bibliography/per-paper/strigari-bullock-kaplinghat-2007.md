---
title: Strigari, Bullock & Kaplinghat (2007)
description: Annotated reference for L. E. Strigari, J. S. Bullock & M. Kaplinghat — Determining the Nature of Dark Matter with Astrometry — the source of the two proper-motion projection kernels (1 − β + β R²/r²) and (1 − β) in progenax's project_dispersion.
---

# Strigari, Bullock & Kaplinghat (2007)

```{admonition} Determining the Nature of Dark Matter with Astrometry
:class: note

**Authors.** Louis E. Strigari (Center for Cosmology, University of California, Irvine;
McCue Fellow), James S. Bullock & Manoj Kaplinghat (Center for Cosmology, UC Irvine).

**Reference.** *The Astrophysical Journal Letters* **657**, L1–L4 (2007 March 1).
Received 2006 December 18; accepted 2007 January 19; published 2007 February 6.

**arXiv.** [astro-ph/0701581](https://arxiv.org/abs/astro-ph/0701581) ·
**DOI.** [10.1086/512976](https://doi.org/10.1086/512976) ·
**ADS.** [2007ApJ...657L...1S](https://ui.adsabs.harvard.edu/abs/2007ApJ...657L...1S)
```

## Abstract (paraphrased)

Strigari, Bullock & Kaplinghat show that adding stellar **proper motions** to the standard
line-of-sight (LOS) velocity-dispersion data of a dwarf spheroidal (dSph) galaxy is a powerful
probe of the dark-matter density profile. Allowing for a general (six-parameter) halo density
profile and a constant stellar velocity anisotropy $\beta$, they forecast (via a $6\times6$
Fisher matrix) that the log-slope of the dark-matter density at about twice the stellar core
(King) radius, $r_\star \simeq 2 r_{\rm King}$, can be measured to within $\pm0.2$ once the proper
motions of $\sim$200 stars are combined with $\sim$1000 LOS velocities — a factor of $\sim$5 better
than LOS data alone. The key physics is a manifest degeneracy between $\beta$ and the log-slope in
the LOS dispersion that is **broken** by the tangential information carried in the two on-sky
proper-motion components. For progenax, the load-bearing content is **§2 "Mass Modeling"**, where
they write the **three** observable projected dispersions — one LOS and two proper-motion — as
projections of the 3-D radial dispersion $\sigma_r(r)$, generalising the line-of-sight Jeans
projection of [](binney-mamon-1982.md) to the proper-motion channels.

## The load-bearing equations (verified against the paper, §2, p. L2)

The 3-D velocity is decomposed (§2, p. L1) into line-of-sight ($v_{\rm los} = v_r\cos\theta +
v_\theta\sin\theta$) and the two in-sky components parallel and tangential to the projected radius
vector $R$: $v_R = v_r\sin\theta + v_\theta\cos\theta$ (on-sky **radial**) and $v_t = v_\phi$
(on-sky **tangential**). The dispersions are $\sigma_i^2 \equiv \langle v_i^2\rangle$, with
$\sigma_\phi^2 = \sigma_\theta^2$ assumed. Solving the Jeans equation for the 3-D radial dispersion
$\sigma_r(r)$ and integrating along the line of sight gives the **three resulting observable
velocity dispersions** (their Eqs. 1–3, p. L2):

**Line-of-sight (Eq. 1, p. L2):**

```{math}
:label: sbk07-los
\sigma_{\rm los}^2(R) = \frac{2}{I_\star(R)}\int_R^\infty
  \left(1 - \beta\,\frac{R^2}{r^2}\right)
  \frac{\nu_\star\,\sigma_r^2\,r\,dr}{\sqrt{r^2 - R^2}}.
```

**Proper-motion radial / on-sky radial (Eq. 2, p. L2):**

```{math}
:label: sbk07-pmr
\sigma_R^2(R) = \frac{2}{I_\star(R)}\int_R^\infty
  \left(1 - \beta + \beta\,\frac{R^2}{r^2}\right)
  \frac{\nu_\star\,\sigma_r^2\,r\,dr}{\sqrt{r^2 - R^2}}.
```

**Proper-motion tangential (Eq. 3, p. L2):**

```{math}
:label: sbk07-pmt
\sigma_t^2(R) = \frac{2}{I_\star(R)}\int_R^\infty
  (1 - \beta)\,
  \frac{\nu_\star\,\sigma_r^2\,r\,dr}{\sqrt{r^2 - R^2}}.
```

Here $\beta(r) = 1 - \sigma_\theta^2/\sigma_r^2$ is the stellar velocity anisotropy, $I_\star(R)$
is the surface density of stars, and $\nu_\star(r)$ is the three-dimensional number (light) density
(§2, p. L2). The paper states explicitly that "it is clear from inspection that each component
depends on $\beta$ in a different fashion, and therefore they can be used together to constrain its
value" — which is precisely the $\beta$–$\gamma$ degeneracy break progenax's OED exploits.

```{admonition} Scope — Strigari+2007 contains ALL THREE kernels explicitly (verified, whole Letter read)
:class: important
Equations (1)–(3) (p. L2) of this Letter give, **in closed form**, the LOS kernel
$(1-\beta R^2/r^2)$ **and** the two proper-motion kernels $(1-\beta+\beta R^2/r^2)$ (radial) and
$(1-\beta)$ (tangential), all three sharing the **identical** projection measure
$r/\sqrt{r^2-R^2}$ and the weight $\nu_\star\,\sigma_r^2$, with the prefactor $2/I_\star(R)$. This
is exactly the form in `project_dispersion`. So unlike [](binney-mamon-1982.md) — which contains
the LOS kernel **only** — this Letter is a primary source for the two **proper-motion** kernels.
(The 4-page Letter does not *derive* Eqs. 1–3 from first principles; it presents them as the
"three resulting observable velocity dispersions", deferring the constant-$\beta$ Jeans solution to
standard treatments. The earliest first-principles derivation of the proper-motion projection is
Leonard & Merritt 1989, ApJ 339, 195; Strigari+2007 is the canonical dSph-astrometry statement of
the same formulae and an entirely adequate primary citation for the kernels as written.)
```

## Code cross-reference (`src/progenax/kinematics/dispersion.py`)

`project_dispersion` (def at line 677) folds the $r/\sqrt{r^2-R^2}$ pole away analytically via
$r^2 = R^2 + u^2$ (so $r\,dr/\sqrt{r^2-R^2} = du$) and integrates a smooth quadrature in $u$
(line 810, `r = sqrt(R_i**2 + u**2)`). With `ratio = R²/r²` (line 822),
`w = rho * sigma_r2` (line 823), and `beta = r²/(r²+r_a²)` ([](merritt-1985.md), line 820):

| Quantity | Code (line) | Kernel in code | Paper source |
|---|---|---|---|
| $\sigma_{\rm los}^2$ | `S_los`, L830: `(1 - beta*ratio)*w` | $1 - \beta\,R^2/r^2$ | Eq. (1) {eq}`sbk07-los` (= [](binney-mamon-1982.md) Eq. 7) ✓ |
| $\sigma_{\rm pm,R}^2$ | `S_pmr`, L831: `(1 - beta + beta*ratio)*w` | $1 - \beta + \beta\,R^2/r^2$ | **Eq. (2)** {eq}`sbk07-pmr` ✓ |
| $\sigma_{\rm pm,T}^2$ | `S_pmt`, L832: `(1 - beta)*w` | $1 - \beta$ | **Eq. (3)** {eq}`sbk07-pmt` ✓ |
| $\Sigma(R)$ | `Sigma`, L829: `2*trapezoid(rho*jac, grid)` | $2\int \rho\, r/\sqrt{r^2-R^2}\,dr$ | $I_\star(R)$ inverse-Abel (BM82 Eq. 2) |

Cell-by-cell: the factor 2 (L829–832), the $\nu_\star\,\sigma_r^2$ weight (here `rho * sigma_r2`,
$\rho = A\,\nu_\star$ under mass-follows-light, $A$ cancelling in the dispersion ratios), the
$\beta(r)$ multiplier, and the substitution that removes $1/\sqrt{r^2-R^2}$ all match Eqs. (1)–(3)
exactly. The isotropic limit ($r_a=$`None` $\Rightarrow\beta=0$, L818) collapses all three kernels
to 1, so $\sigma_{\rm los}=\sigma_{\rm pm,R}=\sigma_{\rm pm,T}$ — the correct $\beta=0$ behaviour of
Eqs. (1)–(3) (anisotropy lives entirely in the kernel ratios).

### Verification verdict

```{admonition} Verdict: MATCH — Strigari+2007 is the primary source for both PM kernels
:class: tip

- **$\sigma_{\rm pm,R}$ ⇒ MATCH.** `(1 - beta + beta*ratio)` (L831) is algebraically identical to
  Strigari+2007 **Eq. (2)** kernel $(1 - \beta + \beta R^2/r^2)$ (p. L2). ✓
- **$\sigma_{\rm pm,T}$ ⇒ MATCH.** `(1 - beta)` (L832) is identical to Strigari+2007 **Eq. (3)**
  kernel $(1 - \beta)$ (p. L2). ✓
- **$\sigma_{\rm los}$ ⇒ MATCH** (also given here as Eq. 1; the dedicated LOS primary remains
  [](binney-mamon-1982.md) Eq. 7).
- All three share the projection measure $r/\sqrt{r^2-R^2}$, the weight $\nu_\star\,\sigma_r^2$, and
  the factor 2 — exactly the `project_dispersion` implementation. No physics discrepancy.

**Phase 2 (Anna-gated) provenance fix.** The `project_dispersion` docstring (L11–13, L21–22, L678,
L682, L825 "B&M82 kernels") currently credits *all three* kernels to Binney & Mamon (1982), but
BM82 contains the LOS kernel **only**. Recommended attribution after this verification:
$\sigma_{\rm los}\to$ **Binney & Mamon (1982)** Eq. 7; $\sigma_{\rm pm,R}\to$ **Strigari+2007**
Eq. 2; $\sigma_{\rm pm,T}\to$ **Strigari+2007** Eq. 3 (earliest derivation: Leonard & Merritt 1989).
No `src/` edits were made in this task.
```

```{note}
All equations above were read from the PDF's **embedded text/vector layer** (the Letter is a
text-based PDF, not a scanned image), so no values are flagged UNVERIFIED-by-text-extract.
```

## Use in progenax

- `progenax.kinematics.dispersion.project_dispersion` — proper-motion projection of the anisotropic
  Jeans model: $\sigma_{\rm pm,R}(R)$ from {eq}`sbk07-pmr` (Strigari+2007 Eq. 2) and
  $\sigma_{\rm pm,T}(R)$ from {eq}`sbk07-pmt` (Strigari+2007 Eq. 3) onto on-sky radii $R$.
- Pairs with [](binney-mamon-1982.md) (the LOS channel, Eq. 7) and [](merritt-1985.md) (the
  Osipkov–Merritt $\beta(r)=r^2/(r^2+r_a^2)$ supplied to all three kernels).
- The two proper-motion channels are what make the dispersion forward model a genuinely
  three-method observable for the OED Fisher, breaking the $\beta$–log-slope degeneracy exactly as
  Strigari+2007 demonstrate (their §4 and Fig. 2).

## Notes

- **Anisotropy convention.** Strigari+2007 use $\beta(r) = 1 - \sigma_\theta^2/\sigma_r^2$ with a
  single tangential component $\sigma_\theta$ (and $\sigma_\phi^2=\sigma_\theta^2$) — the **same**
  convention as [](binney-mamon-1982.md) Eq. 1 and consistent with the Osipkov–Merritt law used in
  progenax. The Letter's Fisher forecast assumes a *constant* $\beta$; progenax instead supplies the
  radius-dependent OM $\beta(r)$, but the projection kernels {eq}`sbk07-los`–{eq}`sbk07-pmt` hold for
  an arbitrary $\beta(r)$ (the kernels are evaluated inside the radial integral).
- The Letter's headline science (the $6\times6$ Fisher matrix forecasting $\sigma(\gamma_\star)$,
  Table 1; the $\beta$–$\gamma$ contours, Fig. 2) is the **motivation** for progenax's OED
  dispersion arc — the same "add proper motions to break the anisotropy degeneracy" argument — but
  progenax implements only the **forward** projection (Eqs. 1–3), not the paper's specific
  six-parameter dark-halo parameterisation (their Eq. 4).
- Equations above verified against the published PDF.
