---
title: Säilynoja et al. (2022)
description: Annotated reference for Säilynoja, Bürkner & Vehtari — the ECDF-difference graphical uniformity test with simultaneous confidence bands, the calibration plot gravoturb_fdf uses to visualize SBC rank uniformity.
---

# Säilynoja et al. (2022)

```{admonition} Graphical test for discrete uniformity and its applications in goodness-of-fit evaluation and multiple sample comparison
:class: note

**Authors.** Teemu Säilynoja, Paul-Christian Bürkner, Aki Vehtari

**Reference.** Statistics and Computing **32**, 32 (2022).

**DOI / arXiv.** [10.1007/s11222-022-10090-6](https://doi.org/10.1007/s11222-022-10090-6) ·
[arXiv:2103.10522](https://arxiv.org/abs/2103.10522)

**Verified.** Abstract, §1.1 (PIT, Eqs. 1–2), §2 (Eqs. 3–9), §2.1 pointwise bands (Eqs. 7–9),
§2.2 simulation method (Eqs. 10–14 + the boxed algorithm), §2.3 optimization method (Eqs.
15–24), and Figs. 1–3 (esp. Fig. 3d, the ECDF-difference plot) checked against the held PDF
(2026-06). The two facts `gravoturb_fdf` depends on: the **scaled-ECDF binomial law**
`N·F(z_i) ~ Binomial(N, z_i)` (Eq. 8) giving pointwise bands, and the **simulation-based
simultaneous band** — inflate the pointwise level to `γ` (Eq. 13, the α-percentile of the
per-trajectory minimum two-sided binomial tail) so the bands jointly cover `1-α`, then plot
the **ECDF difference** `F(z_i) − z_i` (Fig. 3d) for dynamic range.
```

## The big idea

A standard way to test whether a sample follows a reference distribution is the **probability
integral transform** (PIT): if `y_1,…,y_N ~ g` and `g` equals the reference `p`, then the PIT
values `u_i = ∫_{-∞}^{y_i} p(x) dx` are i.i.d. `Uniform(0,1)` (§1.1, Eq. 1). When `p`'s CDF is
unavailable but one can draw samples from it, an empirical PIT `u_i = (1/S) Σ_j 𝟙[x^i_j ≤ y_i]`
(Eq. 2) takes `S+1` discrete equally-spaced values and the test becomes one of **discrete
uniformity**.

The paper's contribution is a **graphical** uniformity test: overlay the empirical CDF (ECDF)
of the PIT values on **simultaneous** confidence bands valid under uniformity, and read off
*both* a yes/no test (does the ECDF stay inside?) and the *shape* of any discrepancy (where and
how it leaves). This is more informative than a single-number test (χ², KS) and, unlike the
histogram, free of bin-width artifacts. It is the method behind the SBC ECDF plots in
`bayesplot`/`arviz` and the natural visual companion to the {cite:t}`Talts2018` rank histogram.

## Core results

**Pointwise bands from the binomial (§2.1).** The ECDF at an evaluation point `z_i ∈ (0,1)` is

```{math}
:label: sailynoja-ecdf
F(z_i) = \frac{1}{N}\sum_{j=1}^{N}\mathbb{1}[u_j \le z_i].
```

Under uniformity each `u_j ≤ z_i` independently with probability `z_i`, so the **scaled ECDF is
binomial** (Eq. 8):

```{math}
:label: sailynoja-binom
N\,F(z_i) \sim \mathrm{Binomial}(N, z_i).
```

The `1-α` pointwise lower/upper bands are then the `γ/2` and `1-γ/2` binomial quantiles divided
by `N` — but using `γ = α` here only controls coverage *at each point individually* (Eq. 9).

**Why pointwise is not enough — and the simultaneous fix (§2.2).** ECDF values at nearby `z_i`
are **strongly dependent** (Fig. 4): stacking pointwise `1-α` intervals under-covers the whole
trajectory. The fix is to inflate the per-point level to a stricter `γ < α` chosen so the bands
hold *jointly*:

```{math}
:label: sailynoja-simultaneous
\Pr\!\big(L_i(\gamma) \le F(z_i) \le U_i(\gamma)\ \text{for all } i\big) = 1-\alpha.
```

`γ` is found by **simulation** (the boxed algorithm, Eqs. 11–14): draw `M` uniform samples of
size `N`; for sample `m` compute its ECDF `F^m(z_i)` and the trajectory's *smallest two-sided
binomial tail*

```{math}
:label: sailynoja-gamma
\gamma^m = 2\,\min_i\Big\{\min\big(\mathrm{Bin}(N F^m(z_i)\mid N, z_i),\;
1 - \mathrm{Bin}(N F^m(z_i)-1\mid N, z_i)\big)\Big\};
```

then set `γ` = the `100α` percentile of `{γ^1,…,γ^M}`, and form the bands

```{math}
:label: sailynoja-bands
L_i(\gamma),\,U_i(\gamma) =
\tfrac{1}{N}\mathrm{Bin}^{-1}\!\big(\tfrac{\gamma}{2}\mid N, z_i\big),\;
\tfrac{1}{N}\mathrm{Bin}^{-1}\!\big(1-\tfrac{\gamma}{2}\mid N, z_i\big).
```

A faster **optimization method** (§2.3, Eqs. 15–24) replaces the `M` simulations with a
derivative-free search over `γ`, using the **Markov** structure of the ECDF trajectory
(`F(z_{i+1})` depends only on `F(z_i)`) to evaluate `Pr(T(γ))` by a recursion (Eq. 23) — same
bands, ~10–60× faster. For figure generation the simulation method is ample.

**The ECDF-difference plot (Fig. 3d).** Because the simultaneous band is narrow for large `N`,
the recommended visualization plots the **difference** of the observed ECDF from the expected
uniform CDF, `F(z_i) − z_i`, with the band shifted by `−z_i`. The expected line becomes flat at
zero; departures (and which region they occur in) are far easier to see than on the raw ECDF.
Recommended evaluation points are the ordered **fractional ranks** `r̃_i = (1/N)Σ_j 𝟙[y_j ≤ y_i]`
(Eq. 5), which form a uniform partition of `[0,1]` independent of the `y` distribution.

**Discreteness (§2, §2.1).** When the PIT is genuinely discrete (here: SBC ranks take `L+1`
values), the binomial coverage is not exact — Brown et al. (2001) "lucky/unlucky" discreteness —
but the authors report the effect on the *simultaneous* band coverage is within `±1%` for
`N ∈ [50, 2000]`, and their fractional-rank construction "behaves better for the smallest and
largest ranks." The multiple-sample extension (§3) swaps the binomial for the **hypergeometric**
distribution; `gravoturb_fdf` uses only the **single-sample** test, so that machinery is not
needed here.

## Use in progenax

The experimental `gravoturb_fdf` trustworthiness arc (workstream ①; see
[](../../10-theory/gravoturbulence/differentiable-inference.md)) visualizes SBC calibration two
complementary ways. The {cite:t}`Talts2018` **rank histogram** + binomial band is one; this
paper's **ECDF-difference plot with simultaneous bands** is the second, sharper view:

- `compute_sbc_ecdf_diff` (jaxstroviz `experimental/analysis/sbc.py`) computes the
  fractional-rank PIT of the per-parameter SBC ranks, then the ECDF difference
  {eq}`sailynoja-ecdf` − `z_i` and the simulation-based simultaneous band
  {eq}`sailynoja-gamma`–{eq}`sailynoja-bands` with sample size `N = K` (number of SBC trials)
  by calling **`arviz_stats.ecdf_utils.ecdf_pit`** — the reference implementation of this paper,
  maintained by the authors' own group — which returns `(eval_points, ecdf, lower, upper)`.
- `plot_sbc_ecdf_diff` (jaxstroviz `experimental/plots/sbc.py`) draws the ECDF-difference curve,
  the shaded simultaneous band, and the zero reference — gallery figure 11's second panel,
  beside the Talts rank histogram.
- **Discreteness is handled consistently** with the [Talts](talts-2018.md) integer-aware χ²
  lesson: the rank histogram uses an integer-aware expected count, and the ECDF test uses the
  fractional-rank PIT with `N=K` binomial bands (defensible for our thinned `L ≳ 100`); the
  ±1% discrete-coverage caveat is documented in the analysis docstring.
- **Honest scope is inherited from SBC:** a calibrated ECDF-difference plot certifies the
  inference engine is self-consistent *under the BM19 model*, not that real clouds follow BM19.

## Notes

- The method generalizes {cite:t}`Talts2018`: SBC asks whether rank statistics are uniform;
  this paper supplies the *simultaneous* graphical test for that uniformity (and quantifies the
  multiple-comparison correction the χ² histogram band only approximates).
- Implemented in the R packages `bayesplot` (`ppc_pit_ecdf`) and the `SBC` package, and in
  `arviz` (`arviz_stats.ecdf_utils.ecdf_pit` / `arviz.plot_ecdf_pit`); the `gravoturb_fdf`
  figure side **reuses the arviz array-layer `ecdf_pit`** (the authors' reference
  implementation) rather than re-deriving the simulation, adding `arviz`/`xarray` to the
  jaxstroviz `[experimental]` extra.
- The newer {cite:t}`Sailynoja2026` ("Posterior SBC") *uses* this test to validate calibration
  *conditional on observed data* — a more honest, dataset-specific calibration check that is a
  natural future workstream beyond the prior-SBC AC18.
