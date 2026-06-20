---
title: Hurley, Tout & Pols (2002)
description: Annotated reference for J. R. Hurley et al. — binary-star evolution (BSE) including tidal circularisation and synchronisation; the tidal-circularisation timescale used in the eccentricity model.
---

# Hurley, Tout & Pols (2002)

```{admonition} Evolution of binary stars and the effect of tides on binary populations
:class: note

**Authors.** J. R. Hurley, C. A. Tout, O. R. Pols.

**Reference.** *Monthly Notices of the Royal Astronomical Society* **329**, 897–928 (2002).

**DOI.** [10.1046/j.1365-2966.2002.05038.x](https://doi.org/10.1046/j.1365-2966.2002.05038.x) ·
**ADS.** [2002MNRAS.329..897H](https://ui.adsabs.harvard.edu/abs/2002MNRAS.329..897H)

**Verified.** The tidal-circularisation timescale (Eqs. 28, 29, 41) below was checked against
the published PDF (p. 901).
```

## Abstract (paraphrased)

Presents the **Binary Star Evolution (BSE)** rapid population-synthesis algorithm, the binary
companion to the single-star (SSE) formulae of [](hurley-2000.md). On top of single-star
evolution it adds the physics of interacting binaries: mass transfer and Roche-lobe overflow,
common-envelope evolution, gravitational-radiation and magnetic-braking angular-momentum loss,
and — central to its use here — **tidal circularisation and synchronisation** of eccentric,
non-corotating orbits. Using the equilibrium-tide (convective damping) and dynamical-tide
(radiative damping) prescriptions, the paper quantifies the systematic effect of tidal
friction on synthesised binary populations and shows that orbits generally circularise before
Roche-lobe overflow.

## The tidal-circularisation timescale (Eqs. 28–29, 41, verified against the PDF)

For stars with **convective** envelopes (equilibrium tide), the circularisation rate is
(Eq. 28)

```{math}
:label: hurley2002-tcirc-conv
\frac{1}{\tau_{\rm circ}} = \frac{21}{2}\left(\frac{k}{T}\right)_{\!c}\,q_2\,(1 + q_2)\,
\left(\frac{R}{a}\right)^{8},
```

with $q_2 = m/M$ the companion-to-primary mass ratio, $R$ the (primary) stellar radius, $a$ the
semi-major axis, and $(k/T)_c$ the apsidal-motion-constant / damping-timescale factor set by
the convective eddy turnover (Eq. 30–31, after Rasio et al. 1996). Hurley et al. write it in
the equivalent Rasio form (Eq. 29)

```{math}
:label: hurley2002-tcirc-rasio
\frac{1}{\tau_{\rm circ}} = \frac{f_{\rm conv}}{\tau_{\rm conv}}\,\frac{M_{\rm env}}{M}\,
q_2\,(1 + q_2)\,\left(\frac{R}{a}\right)^{8}.
```

For stars with **radiative** envelopes (dynamical tide, after Zahn 1977) the scaling is steeper
in $R/a$ (Eq. 41):

```{math}
:label: hurley2002-tcirc-rad
\frac{1}{\tau_{\rm circ}} = \frac{21}{2}\left(\frac{GM}{R^{3}}\right)^{1/2}
q_2\,(1 + q_2)^{11/6}\,E_2\,\left(\frac{R}{a}\right)^{21/2}.
```

The key physics for progenax is the **very strong $(R/a)^8$ dependence** (convective case): the
circularisation rate plummets with separation, so only short-period orbits are tidally
circularised within the relevant lifetimes. This is exactly the schematic
$\tau_{\rm circ} \sim (a/R_\star)^8 / [q(1+q)]$ used in the eccentricity chapter — the
convective-envelope (equilibrium-tide) form, inverted.

## Use in progenax

- [](../../10-theory/binaries/eccentricity.md) — the tidal-circularisation timescale
  {eq}`hurley2002-tcirc-conv` motivating the short-period $e \to 0$ behaviour of the
  period-dependent eccentricity model.
- `progenax.binaries` — `MoeEccentricity` reproduces short-period circularisation
  *intrinsically* (the Roche ceiling and the $\eta \le -1$ branch drive $e \to 0$); progenax
  does **not** integrate the BSE tidal ODEs themselves.

## Notes

progenax uses Hurley+2002 only for the **scaling intuition** behind tidal circularisation —
the $(R/a)^8$ law that makes circularisation a short-period phenomenon — not as an implemented
tidal-evolution integrator. A full BSE-style tidal-evolution treatment (and the
stellar-evolution-aware Roche radius needed for evolved-star cutoffs) is a planned **startrax**
coupling. The single-star formulae underlying BSE are [](hurley-2000.md).
