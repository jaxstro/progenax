---
title: Cartwright & Whitworth (2004)
description: Annotated reference for A. Cartwright & A. P. Whitworth — the statistical analysis of star clusters (the Q parameter).
---

# Cartwright & Whitworth (2004)

```{admonition} The statistical analysis of star clusters
:class: note

**Authors.** A. Cartwright, A. P. Whitworth

**Reference.** *Monthly Notices of the Royal Astronomical Society* **348, 589** (2004).

**DOI.** [10.1111/j.1365-2966.2004.07360.x](https://doi.org/10.1111/j.1365-2966.2004.07360.x) ·
**arXiv.** [astro-ph/0403474](https://arxiv.org/abs/astro-ph/0403474)

**Verified.** Eqs. 2–6, Table 1, and the MST/area normalisation checked against the held PDF (pp. 589–594; re-verified 2026-06 — fixed a transcribed $\mathcal{Q}$ uncertainty at $D=1.5$).
```

## The big idea

How do you tell, *objectively*, whether a star cluster is **centrally concentrated** (a
smooth radial density gradient) or **substructured** (multi-scale fractal clumping)? CW04
show that two normalised statistics together do it. The **normalised correlation length**
$\bar{s}$ (mean separation / cluster radius) decreases both as a cluster becomes more
centrally concentrated *and* as it becomes more fractal — so $\bar{s}$ alone is ambiguous.
The **normalised mean MST edge length** $\bar{m}$ also decreases in both cases, but with a
*different* sensitivity. Their ratio

$$
\mathcal{Q} = \frac{\bar{m}}{\bar{s}}
$$

breaks the degeneracy: **$\mathcal{Q} > 0.8$ ⇒ centrally concentrated** (radial gradient),
**$\mathcal{Q} < 0.8$ ⇒ substructured** (fractal).

## Core definitions

Project the cluster to 2-D. Let $N$ be the number of stars, $R_\mathrm{cluster}$ the
distance from the mean position to the furthest star, and $A$ the cluster area.

**Normalised correlation length.** With $\langle s\rangle$ the mean of all pairwise
separations,

$$
\bar{s} = \frac{\langle s\rangle}{R_\mathrm{cluster}} .
$$

**Normalised mean MST edge length.** The minimum spanning tree (MST) is the shortest set
of edges connecting all $N$ stars with no loops. Its expected total length for $N$ points
uniformly spread over area $A$ scales as $(N A)^{1/2}$ (Beardwood+ 1959), so the
size-independent normalisation is

$$
\bar{m} = \frac{L_\mathrm{MST}}{\sqrt{N\,A}},
\qquad A = \pi R_\mathrm{cluster}^2 ,
$$

with $L_\mathrm{MST}$ the **total** MST length. The $\sqrt{N}$ is **mandatory** — omitting
it collapses $\bar m$ (and was the root cause of a discredited $\mathcal{Q}\approx0.13$
headline elsewhere in this codebase).

## Calibration (their Table 1; 3-D models projected to 2-D, $100\le N\le300$)

| Model | profile | $\mathcal{Q}$ |
|---|---|---|
| 3D0 | uniform sphere | $0.79\pm0.02$ |
| 3D1 | $n\propto r^{-1}$ | $0.84\pm0.03$ |
| 3D2 | $n\propto r^{-2}$ | $0.93\pm0.03$ |
| 3D2.9 | $n\propto r^{-2.9}$ | $1.50\pm0.13$ |
| F3.0 | fractal $D=3.0$ | $0.80\pm0.02$ |
| F2.5 | fractal $D=2.5$ | $0.73\pm0.06$ |
| F2.0 | fractal $D=2.0$ | $0.61\pm0.08$ |
| F1.5 | fractal $D=1.5$ | $0.45\pm0.09$ |

Increasing **central concentration** pushes $\mathcal{Q}$ *above* 0.8; increasing
**fractal sub-clustering** (lower $D$) pushes it *below* 0.8. (The $\mathcal{Q}$
uncertainties are small — $\pm0.09$ at $D=1.5$; it is the *correlation length* $\bar s$
that carries the large $\pm0.18$ scatter at low $D$, Table 1 column 2.)

Radial models are sampled analytically via the inverse CDF $r = \{(3-\alpha)\mathcal{R}/3\}^{1/(3-\alpha)}$
(their Eq. 2) for $n\propto r^{-\alpha}$; fractal models use a discrete box-fractal tree
with Bernoulli maturation probability $p = \mathcal{N}_\mathrm{div}^{D-3} = 2^{D-3}$
($\mathcal{N}_\mathrm{div}=2$).

**Real clusters (their Table 1 / abstract).** Centrally concentrated: IC 348
$\mathcal{Q}=0.98$, $\rho$ Oph $0.85$ (radial slopes $\alpha\simeq2.2,\,1.2$); mildly
substructured: Chamaeleon $0.67$, IC 2391 $0.66$ ($D'\simeq2.25$); strongly substructured:
Taurus $0.45$ ($D'\simeq1.55$; treating its binaries as single systems raises this to
$0.58$, $D'\simeq1.9$).

## Use in progenax

- [](../../20-architecture/jax-native-substructure-q.md) — JAX-native kNN approximation + scipy reference.
- [](../../10-theory/tidal-and-substructure/fractal.md) — $\mathcal{Q}$ as the substructure truth metric.
- `experimental/gravoturb/diagnostics/q.py` — clean-room CW04 estimator (numpy/scipy, non-differentiable).
- `tests/experimental/fixtures/cw04_models.py` — analytic radial models for the AC5 validation.

Validation (AC5): the estimator reproduces the 3D0/3D1/3D2 anchors (0.79/0.84/0.93) within
$\sim$Table-1 scatter and is monotone in central concentration.

## Notes

- **Area convention (important).** CW04 normalise each cluster so that
  $R_\mathrm{cluster}=1$ (Fig. 3 caption: "scaled so that the distance from the centre to
  the most distant star is unity") and normalise $\bar m$ by $(N A)^{1/2}/(N-1)$ with "the
  area $A$ of the cluster" (Section 4.1). Operationally this is the circle area
  $A=\pi R_\mathrm{cluster}^2$, **not** the convex-hull area: using the hull biases
  $\mathcal{Q}$ high by $\sim+0.08$–$0.14$, whereas $A=\pi R_\mathrm{cluster}^2$ reproduces
  Table 1 to $<0.01$ (our AC5: 0.782/0.832/0.924 vs 0.79/0.84/0.93). Other authors (Schmeja
  & Klessen 2006; Lomax+ 2018) instead set $R=\sqrt{A_\mathrm{hull}}$, giving a *different*
  absolute $\mathcal{Q}$ scale.
- The CW04 $\mathcal{Q}$ is **distinct** from the virial ratio $Q_\mathrm{vir}=T/|V|$;
  progenax separates these into different modules.
- $\mathcal{Q}$ is a **poor estimator of fractional-Brownian-motion (FBM) parameters** —
  see [](lomax-2018.md) — a caution for any $\mathcal{Q}$-vs-structure calibration built
  on GRF/FBM density fields.
