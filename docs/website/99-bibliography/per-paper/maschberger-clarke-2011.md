---
title: Maschberger & Clarke (2011)
description: Annotated reference for Th. Maschberger & C. J. Clarke — substructure-robust mass-segregation diagnostics (median-MST Λ̃ and the local-surface-density m–Σ plane).
---

# Maschberger & Clarke (2011)

```{admonition} Global mass segregation in hydrodynamical simulations of star formation
:class: note

**Authors.** Th. Maschberger, C. J. Clarke.

**Reference.** *Monthly Notices of the Royal Astronomical Society* **416, 541–546** (2011).

**DOI.** [10.1111/j.1365-2966.2011.19067.x](https://doi.org/10.1111/j.1365-2966.2011.19067.x)

**Verified.** Checked against the published PDF (pp. 541–546, 2026-06-08): the median-MST Λ̃
(Eqs. 1–3) and the local-surface-density Σ estimator (Eq. 4) + the m–Σ KS test (§4).
```

## The problem

Allison's $\bar\Lambda$ (mean-MST; see [](allison-2009.md)) is **fragile to outliers in
substructured regions**: a *single* isolated massive star adds one long MST edge to the massive-star
tree, which can make $\bar\Lambda < 1$ ("inversely segregated") even when the massive stars are
overwhelmingly in subcluster centres (their Fig. 2). M&C give two more robust diagnostics.

## Diagnostic 1 — median-MST $\tilde\Lambda$ (Eqs. 1–3)

Over a moving window of 40 mass-ranked stars, $\bar\Lambda_{(i)}(\bar m_{(i)}) =
\bar\ell_{40}/\bar\ell_{(i),(i+40)}$ (mean edge length, Eq. 1, following Parker et al. 2011). The
**robust variant replaces the mean edge length with the *median*** (Eq. 3),

$$
\tilde\Lambda(\bar m_{(i)}) = \frac{\tilde\ell_{40}}{\tilde\ell_{(i),(i+40)}},
$$

which is far less sensitive to a few isolated massive outliers. Using mean *and* median together
flags both genuine segregation of the majority and the presence of outliers.

## Diagnostic 2 — local surface density / the m–Σ plane (§4, Eq. 4) ← used in progenax

For each star, the projected local stellar **surface number density** (von Hoerner 1963;
Casertano & Hut 1985) is

$$
\Sigma = \frac{k-1}{\pi\,r_k^{2}}, \qquad k = 6,
$$

with $r_k$ the distance to the $k$-th nearest neighbour ($k=6$ is "a good compromise between
locality and low-number fluctuations" for $N\in[30,1000]$). Mass segregation is read off the
**m–Σ plane**: massive stars sit at systematically higher Σ. M&C quantify the difference between the
high-mass and low-mass Σ distributions with a **two-sample KS test** (e.g. their $1000\,M_\odot$ run:
high- vs low-mass Σ differ at $p=0.025$). **This estimator is robust to substructure and to outliers**
— a single isolated massive star is simply one low-Σ point and does not dominate, unlike the MST mean.

## Physical results (for context)

SPH star-formation sims (Bonnell et al. 2003, 2008) show **global mass segregation from very early
times**, only mildly affected by subcluster merging. Up to $\approx2$–3 per cent of "massive" sinks
($m>2.5\,M_\odot$) are in relative isolation (formed in situ), but the majority are in dense centres.
(Softened gravity suppresses ejections, so isolated-massive counts are a lower bound.)

## Use in progenax

- `gravoturb_fdf.diagnostics.mass_density` (experimental) — implements **Eq. 4** (`local_surface_density`)
  and the m–Σ diagnostic (`mass_density_segregation`: Spearman ρ(m,Σ), high/low median-Σ ratio, KS p).
  This is the **substructure-robust mass-weighted metric** for the FDF cluster IC — it detects
  *primordial* mass–density correlation (massive stars placed in dense BM19 clumps by
  `gravoturb_fdf.masses.correlated_mass_assignment`) where CW04 $\mathcal{Q}$ on small massive subsets
  is too noisy. Validated in `tests/experimental/unit/test_mass_density.py` (exact-formula + uniform-density
  recovery + primordial-correlation detection).
- Complements [](allison-2009.md) ($\Lambda_{\mathrm{MSR}}$ concentration) and
  [](cartwright-2004.md) ($\mathcal{Q}$ geometry): Σ is the *local-density* axis of the mass-segregation
  picture.

## Notes

- The metric is computed **in projection** (2-D), matching M&C; for 3-D one would adapt the
  neighbour count / normalisation.
- $k=6$ and the $(k-1)$ numerator are the Casertano & Hut (1985) convention; do not "simplify" the
  $-1$ away (it is the unbiased surface-density estimator, not $k/\pi r_k^2$).
