---
title: Kim & Ryu (2005)
description: Annotated reference for Kim & Ryu — the density power spectrum of compressible turbulent flows and its flattening with Mach number.
---

# Kim & Ryu (2005)

```{admonition} Density power spectrum of compressible hydrodynamic turbulent flows
:class: note

**Authors.** Jongsoo Kim, Dongsu Ryu

**Reference.** *The Astrophysical Journal Letters* **630, L45** (2005).

**ADS.** 2005ApJ...630L..45K (DOI not printed on the held PDF; cite by ADS bibcode)

**Verified.** Abstract, Figs. 1–2, and the 1D/3D slope tables checked against the held
PDF (2026-06). This is the **density** power spectrum (not the velocity spectrum) — the
distinction is the whole point.
```

## The big idea

The **density** field of supersonic turbulence is *not* a passive copy of the velocity
field. Kim & Ryu simulate driven isothermal compressible turbulence ($1\lesssim
\mathcal{M}_\mathrm{rms}\lesssim10$) and measure the **density power spectrum**
$P_\rho(k)\propto k^{-\beta}$. Their central result: **the density spectrum becomes
shallower (flatter) as the Mach number rises**, because supersonic turbulence sweeps mass
into thin sheets, filaments, and ultimately point-like peaks — structures whose Fourier
content is dominated by sharp edges (shallow spectra), not by a smooth cascade.

This is the opposite of the intuition you get from the *velocity* spectrum (Kolmogorov
$\to$ Burgers, which *steepens* from $k^{-5/3}$ toward $k^{-2}$). Conflating the two is a
common and consequential mistake.

## Core results

Let the shell-integrated density spectrum scale as $E_\rho(k)\propto k^{-s}$.

**One dimension** (compressive only): the profile changes from shock *discontinuities* at
$\mathcal{M}\sim1$ to delta-function *peaks* at $\mathcal{M}\gg1$:

$$
E_\rho \propto k^{-2}\ (\mathcal{M}\sim1,\ \text{step functions})
\;\longrightarrow\;
E_\rho \propto k^{0}\ (\mathcal{M}\gg1,\ \text{peaks}).
$$

(These $k^{-2}$ and $k^{0}$ limits are the Burgers shock / strong-shock mass-concentration
results of Saichev & Woyczynski 1996.)

**Three dimensions** (the relevant case), least-squares slopes over the inertial range:

| $\mathcal{M}_\mathrm{rms}$ | $E_\rho(k)$ slope $s$ | morphology |
|---|---|---|
| 1.2 (transonic) | $-1.73$ (≈ Kolmogorov $-5/3$) | discontinuity surfaces on a smooth field |
| 3.4 | $-1.08$ | emerging sheets/filaments |
| 7.3 | $-0.75$ | sheets + filaments |
| 12.0 | $-0.52$ | dense filaments + knots |

So the 3D density spectrum **flattens monotonically** with Mach number.

## A convention you must get right (E(k) vs P₃D(k))

There are two spectra in circulation and they differ by $k^2$:

- the **shell-integrated** spectrum $E(k)\propto k^{-s}$ (power summed over the
  $|\mathbf{k}|=k$ shell), and
- the **3D power-spectral density** $P_\mathrm{3D}(k)$ (power per Fourier mode), with
  $E(k) = 4\pi k^2 P_\mathrm{3D}(k)$, hence $P_\mathrm{3D}(k)\propto k^{-(s+2)}$.

Kim & Ryu quote $E_\rho$ (their transonic $-1.73$ is "close to Kolmogorov $-5/3$", which is
an $E(k)$ statement). The progenax FDF Gaussian random field is built and *measured* in
the **$P_\mathrm{3D}$** convention ($P_\mathrm{3D}\propto k^{-\beta}$, see
[field.py](../../../../src/experimental/gravoturb_fdf/field/field.py)). Translating:

$$
\beta_{P_\mathrm{3D}} = s + 2:\qquad
\mathcal{M}=1.2\!\to\!3.73,\;\;3.4\!\to\!3.08,\;\;7.3\!\to\!2.75,\;\;12\!\to\!2.52 .
$$

i.e. the physical 3D *density* spectrum sits at $\beta\approx2.5$–$3.7$ and **decreases**
with Mach — shallower than, and trending opposite to, a pure Kolmogorov/Burgers
**velocity** spectrum.

## Use in progenax

- This is the **correct primary reference for the density power-spectrum slope** $\beta$
  fed to the FDF GRF ([field.py](../../../../src/experimental/gravoturb_fdf/field/field.py)) —
  *not* the velocity-spectrum theories (Kolmogorov/Burgers).
- Cross-check for [](lomax-2018.md): Lomax's FBM field is generated with $\beta=E+2H=3+2H$
  (the *generating* Gaussian field), set by the desired fractal dimension $D=3-H$; the
  resulting *density* field then carries the shallower Kim & Ryu spectrum after the
  nonlinear (lognormal / copula) remap.

## Notes

- **Density ≠ velocity spectrum.** An earlier `spectral_slope_from_mach`
  ([turbulence.py](../../../../src/progenax/cluster/turbulence.py)) interpolated between
  Kolmogorov ($\beta=11/3$) and Burgers ($\beta=4$) — those are **velocity** slopes, and
  the implied *increase* of $\beta$ with Mach is **backwards** for the density spectrum.
  **Corrected in the 2026-06 grounding pass:** `spectral_slope_from_mach` now returns the
  Kim & Ryu density slope $\beta(\mathcal{M}) = 3.788 - 1.203\log_{10}\mathcal{M}$ (the
  least-squares fit of $s+2$ vs $\log_{10}\mathcal{M}$ over the four points above), clipped
  to $[2,\,11/3]$ (`BETA_DENSITY_FLOOR`, `BETA_KOLMOGOROV` in `cluster/constants.py`).
- The simulations neglect magnetic fields and self-gravity; both modify the slope (Padoan+
  2004 found magnetic fields shallow it further; self-gravity adds power at small scales,
  the high-density tail of [](burkhart-mocz-2019.md)).
- The shallow CNM H I power spectrum (Deshpande+2000) vs. the Kolmogorov electron-density
  spectrum (Armstrong+1995) is explained by this Mach dependence: H I is supersonic
  ($\mathcal{M}\sim$ few), the WIM transonic.
