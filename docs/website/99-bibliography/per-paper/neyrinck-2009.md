---
title: Neyrinck, Szapudi & Szalay (2009)
description: Annotated reference for Mark Neyrinck, István Szapudi & Alexander Szalay — log-density Gaussianization restores Fisher information to the power spectrum, the motivation for using the log-density 2-point in gravoturb.
---

# Neyrinck, Szapudi & Szalay (2009)

```{admonition} Rejuvenating the matter power spectrum: restoring information with a logarithmic density mapping
:class: note

**Authors.** Mark C. Neyrinck, István Szapudi, Alexander S. Szalay

**Reference.** *The Astrophysical Journal Letters* **698, L90–L93** (2009).

**DOI.** [10.1088/0004-637X/698/2/L90](https://doi.org/10.1088/0004-637X/698/2/L90)

**Verified.** Abstract, §1–2, and Figs. 1–3 checked against the held PDF (2026-06). The result
progenax relies on: **Gaussianizing the density restores information to the two-point function.**
```

## The big idea

As structure grows non-linear, the matter power spectrum $P_\delta(k)$ becomes a *bad* summary
statistic: gravitational evolution couples Fourier modes, the late-time density is dominated by a few
sharp peaks (haloes), and the cosmological information that was cleanly held in the linear power
spectrum **leaks into higher-order statistics** and into a large, "translinear" covariance. Neyrinck,
Szapudi & Szalay show that a single, simple operation undoes much of this damage: replace $\delta$
with the **log-density** $\log(1+\delta)$ before measuring the power spectrum.

The log-mapped power spectrum $P_{\log(1+\delta)}(k)$ has a shape "hardly departing from the linear
power spectrum for $k \lesssim 1\,h\,\mathrm{Mpc}^{-1}$ at all redshifts," and — the headline —
recovers **pristine Fisher information**, yielding about **10× more cumulative signal-to-noise** at
$z=0$ than the standard power spectrum over a range of scales.

## Why it works

The density field is statistically invariant under translations and rotations; all the cosmological
information of the Gaussian initial conditions lives in the *power spectrum*, with every higher
moment zero. Non-linear growth breaks this by making the one-point PDF non-Gaussian
($\delta$ develops a long tail and a hard floor at $\delta=-1$). Because the late-time PDF is
approximately **lognormal** ([](coles-jones-1991.md)), restoring Gaussianity to the one-point
distribution — by taking the log — pulls the strayed information *back* into the two-point function.
Phase correlations (which build genuine cosmic-web filaments) affect higher-order statistics but not
the power spectrum, so a monotone one-point remap recovers what it can without needing the phases.

## Use in progenax

- **Why the 2-point carrier is the log-density $\xi_s$, not the linear $\langle\rho\rho\rangle$.**
  This is the cosmology precedent for the choice made throughout [](../../10-theory/gravoturbulence/inference.md): the
  fat power-law tail makes linear-density 2-point statistics divergent / information-poor, while the
  log-density two-point is well-behaved and information-rich. The
  [`gaussianized_xi`](../../../../src/experimental/gravoturb/theory/log_correlations.py) series
  predicts exactly this $\xi_s(r)$.
- **The "predict the statistic, restore the information" philosophy** of the cosmology playbook that
  the differentiable-inference layer adopts.

## Notes

- Paper I of a pair; the discrete-field / galaxy extension and the Fisher-vs-resolution behaviour are
  in [](neyrinck-2011.md). The information-theoretic optimality of the transform is formalised by
  [](carron-szapudi-2013.md).
- The log transform is exactly reversible and preserves cell-by-cell ranking — the same property the
  FDF rank copula exploits.
