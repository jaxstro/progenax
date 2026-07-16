---
title: Lomax, Bates & Whitworth (2018)
description: Annotated reference for O. Lomax, M. L. Bates & A. P. Whitworth — modelling the structure of star clusters with fractional Brownian motion.
---

# Lomax, Bates & Whitworth (2018)

```{admonition} Modelling the structure of star clusters with fractional Brownian motion
:class: note

**Authors.** O. Lomax, M. L. Bates, A. P. Whitworth

**Reference.** *Monthly Notices of the Royal Astronomical Society* **480, 371** (2018).

**DOI.** [10.1093/mnras/sty1788](https://doi.org/10.1093/mnras/sty1788)

**Verified.** Equations below checked against the held PDF (2026-06).
```

## The big idea

Molecular clouds are well described by **fractional Brownian motion (FBM)** density
fields — self-similar random fields with a well-defined fractal dimension. Since clusters
are born from clouds, Lomax+ propose modelling *cluster* structure with the **same** FBM
fields, rather than with the discrete box-fractal (BF; Goodwin & Whitworth 2004) or radial
density-profile (RDP) models used to calibrate the CW04 $\mathcal{Q}$ parameter. An FBM
cluster is parametrised by just two numbers: the **drift/Hurst exponent $H$** (structural
roughness) and the **log-density standard deviation $\sigma$**. Crucially, FBM can produce
*both* centrally-concentrated and substructured clusters, giving a much better match to
real clusters than BF or RDP.

**This is the same construction `gravoturb` uses for its 3-D density field** (a
Gaussian random field with a power-law power spectrum, exponentiated to a lognormal) — so
Lomax+ 2018 is the published, peer-reviewed grounding for our *differentiable* field
method. We differ only in the final marginal map (see Notes).

## Core equations (their Section 2)

Work on an $E$-dimensional grid ($E=2$ or $3$); $\mathcal{U}$ denotes a uniform random
variate. **Generate the field spectrum** with a power-law amplitude (Eqs. 1–2):

$$
\hat f(\mathbf{k},H) = A(\mathbf{k},H)\,[\cos\varphi(\mathbf{k}) + i\sin\varphi(\mathbf{k})],
\qquad
A(\mathbf{k},H) = \mathcal{P}^{-1/2}\,\lVert\mathbf{k}\rVert^{-\beta/2},
\quad \beta = E + 2H,
$$

with $\mathcal{P} = \sum_{\mathbf{k}}\lVert\mathbf{k}\rVert^{-\beta}$ and $A(\mathbf 0)=0$.
**Hermitian-symmetric phases** (Eq. 3) make the inverse DFT real:

$$
\varphi(\mathbf{k}) = \chi(\mathbf{k}) - \chi(-\mathbf{k}),
\qquad \chi(\mathbf{k}) = 2\pi\mathcal{U}.
$$

Optionally **smooth** with a Gaussian kernel of width $h$ (a nuisance resolution
parameter; Eq. 4). Finally **exponentiate** the standardised field $f'$ to a lognormal
density (Eq. 5):

$$
g(\mathbf{r},H,h,\sigma) = \exp\!\left[\frac{\sigma\, f'(\mathbf{r},H,h)}
                                              {\sqrt{\langle f'^2\rangle}}\right],
$$

where $\sigma$ is the standard deviation of $\ln g$. Stellar positions are then sampled
with $g$ as the PDF; 3-D fields are projected to 2-D for analysis.

**Fractal dimension.** The FBM field has

$$
D = E - H ,
$$

so in 3-D ($E=3$): $H=1$ gives smooth/centrally-concentrated structure ($D=2$ surface-like),
while $H\to0$ gives rough, multi-clump structure ($D\to3$, space-filling sub-clumps).

**$\mathcal{Q}$ statistics (their Eq. 8).** Lomax+ use the Schmeja & Klessen (2006)
normalisation $R=\sqrt{A_\mathrm{hull}}$ for both:

$$
\bar m = \frac{1}{\pi R^E [N_m+1]^{(E-1)/E}}\sum_{i=1}^{N_m} m_i ,
\qquad
\bar s = \frac{1}{N_s R}\sum_i s_i ,
\qquad \mathcal{Q}=\bar m/\bar s .
$$

(The $R$ choice differs from CW04's $R=R_\mathrm{cluster}$ + $A=\pi R^2$, so the absolute
$\mathcal{Q}$ scale differs — see [](cartwright-2004.md).)

## The key result for us

Lomax+ show that **$\mathcal{Q}$ analysis is unable to estimate FBM parameters**: in the
$\mathcal{Q}$ (and $\bar m$–$\bar s$) plane, FBM clusters with different $(H,\sigma)$
overlap badly, and $H\sim1$ FBM clusters occupy the *same* region as smooth RDP clusters.
$\mathcal{Q}$ is therefore a **poor predictor of $H$ and/or $\sigma$**; they resort to a
**machine-learning regressor** trained on $(\bar m,\bar s)$ to recover $(H,\sigma)$ with
uncertainties.

> **Implication (AC7 risk).** Because the `gravoturb` density field *is* an FBM-type
> field, a clean monotonic $\mathcal{Q}(f_\mathrm{sub})$ calibration may be **optimistic** —
> $\mathcal{Q}$ may weakly discriminate our GRF-based substructure, exactly as Lomax+ found.
> This caution is recorded for the P3 / AC7 calibration.

## Use in progenax

- `experimental/gravoturb/realization/gaussian_field.py` (P2) — the GRF + exponentiation engine is
  the Lomax-2018 FBM construction; we map $(H,\sigma)\leftrightarrow(\beta,\sigma_s)$ via
  $\beta = E+2H$ and $\sigma_s$ from BM19.
- [](cartwright-2004.md) — the $\mathcal{Q}$ estimator Lomax+ analyse (different $R$ convention).

## Notes

- **What we keep / change.** We keep Lomax's spatial engine (GRF, $\beta=E+2H$,
  Hermitian → iFFT, differentiable in $H$). We **replace** the plain exponentiation (Eq. 5,
  a lognormal of width $\sigma$) with a **rank copula to the BM19 lognormal+power-law
  marginal**, so the field carries the physically-motivated dense tail rather than a pure
  lognormal. Both maps are differentiable in the cloud parameters.
- $H$ (Hurst) ↔ fractal dimension $D=E-H$ ↔ power-spectrum slope $\beta=E+2H$ are three
  views of the same roughness parameter.
