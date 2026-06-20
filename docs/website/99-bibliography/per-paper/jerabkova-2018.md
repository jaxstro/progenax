---
title: Jeřábková et al. (2018)
description: Annotated reference for T. Jeřábková et al. — the metallicity- and SFR-dependent, time-evolving galaxy-wide IMF (IGIMF3) and its embedded-cluster α₃(x) relation.
---

# Jeřábková et al. (2018)

```{admonition} Impact of metallicity and star formation rate on the time-dependent, galaxy-wide stellar initial mass function
:class: note

**Authors.** T. Jeřábková, A. Hasani Zonoozi, P. Kroupa, G. Beccari, Z. Yan, A. Vazdekis,
Z.-Y. Zhang.

**Reference.** *Astronomy & Astrophysics* **620**, A39 (2018). Received 2018 March 20,
accepted 2018 August 24.

**DOI.** [10.1051/0004-6361/201833055](https://doi.org/10.1051/0004-6361/201833055) ·
**ADS.** [2018A&A...620A..39J](https://ui.adsabs.harvard.edu/abs/2018A%26A...620A..39J)
```

## Abstract (paraphrased)

Builds a comprehensive grid of galaxy-wide IMFs (gwIMF) under the Integrated Galactic IMF
(IGIMF) theory, for metallicities ${\rm [Fe/H]}\in(-3,1)$ and SFRs $10^{-5}$–$10^{5}\,M_\odot\,{\rm yr}^{-1}$.
The gwIMF is the sum of the IMFs of all embedded clusters forming in a galaxy over $\delta t\approx 10$ Myr;
each embedded cluster's stellar IMF varies with its density and metallicity following
[](marks-2012.md). The resulting gwIMF is bottom-light + top-heavy at high SFR and bottom-heavy
at high metallicity. Three IGIMF variants are defined (IGIMF1/2/3); IGIMF3 lets the full IMF vary.

## The IGIMF framework — how the galaxy-wide IMF is assembled (§3, verified)

The IGIMF idea: a galaxy does not form stars in one event but as a population of **embedded
clusters**, and the galaxy-wide IMF (gwIMF) is the *sum* of all their stellar IMFs over a
star-formation epoch $\delta t \approx 10$ Myr (the molecular-cloud free-fall/cycle time):

$$
\xi_{\rm gwIMF}(m) = \int_{M_{\rm ecl,min}}^{M_{\rm ecl,max}({\rm SFR})}
\xi_{\star}\!\big(m \,\big|\, \rho_{\rm cl}(M_{\rm ecl}),\,{\rm [Fe/H]}\big)\;
\xi_{\rm ecl}(M_{\rm ecl})\; dM_{\rm ecl}.
$$

Two ingredients:

1. **The embedded-cluster mass function (ECMF)** — how many clusters of each mass form
   (Eqs. 1–2): a power law $\xi_{\rm ecl}(M_{\rm ecl})\propto M_{\rm ecl}^{-\beta}$ with a
   galaxy-SFR-dependent slope $\beta = -0.106\,\log_{10}{\rm SFR} + 2$ (Weidner+2004), and an
   upper truncation $M_{\rm ecl,max}({\rm SFR})$ from $M_{\rm tot}={\rm SFR}\cdot\delta t$
   (Eq. 3). Higher SFR → more, more-massive clusters → top-heavier gwIMF.
2. **The embedded-cluster stellar IMF** — the multi-power-law (Eq. 4) whose slopes vary with
   the cluster's own $(\rho_{\rm cl}, {\rm [Fe/H]})$ following [](marks-2012.md) (below).

The IGIMF variants differ in *what* is allowed to vary: IGIMF1 (high-mass only), IGIMF2
(adds a metallicity-dependent ECMF), IGIMF3 (full stellar-IMF variation). progenax
implements the embedded-cluster stellar-IMF mapping (the integrand), not the galaxy
integral.

```{warning}
**Two unrelated quantities are both called $\beta$.** Here $\beta=-0.106\log_{10}{\rm SFR}+2$
is the **ECMF slope** (a cluster *mass*-function exponent, set by galaxy SFR). It is entirely
distinct from the turbulent **density power-spectrum slope** $\beta$ of [](kim-ryu-2005.md) /
`cluster.turbulence.spectral_slope_from_mach`. Neither $\beta$ enters the stellar-IMF slopes
$\alpha_i$ — those depend only on $(\rho_{\rm cl},{\rm [Fe/H]})$.
```

## The embedded-cluster α₃ and the density parameter x (verified, §3.2–3.3)

The stellar IMF in an embedded cluster is the multi-power-law (Eq. 4) with high-mass slope
(Eq. 6; attributed to Marks et al. 2012 + its 2014 erratum)

```{math}
:label: jerabkova-alpha3
\alpha_3 = \begin{cases} 2.3, & x < -0.87\\ -0.41\,x + 1.94, & x \ge -0.87\end{cases}
```

with the density/metallicity parameter (Eq. 7)

```{math}
:label: jerabkova-x
x = -0.14\,{\rm [Fe/H]} + 0.99\,\log_{10}\!\Big(\tfrac{\rho_{\rm cl}}{10^6\,M_\odot\,{\rm pc}^{-3}}\Big),
```

where (Eq. 8) $\rho_{\rm cl}=3M_{\rm cl}/(4\pi r_h^3)$, $r_h/{\rm pc}=0.1\,(M_{\rm ecl}/M_\odot)^{0.13}$
(Marks & Kroupa 2012), and $\epsilon = M_{\rm ecl}/M_{\rm cl} = 0.33$. The low-mass slopes follow
$\alpha_i = \alpha_{i,c}+\Delta\alpha\,{\rm [Fe/H]}$ (Eq. 10, $\Delta\alpha\approx0.5$). The embedded-cluster
mass function (ECMF) is a power law of slope $\beta=-0.106\log_{10}{\rm SFR}+2$ (Eqs. 1–2).

Jeřábková et al. also give a **concise mass-based** form of the density parameter (Eq. 9,
verified against the PDF, p. 6) that folds the whole density chain into a function of the
embedded-cluster mass $M_{\rm ecl}$ alone:

```{math}
:label: jerabkova-x-mass
x = -0.14\,{\rm [Fe/H]} + 0.6\log_{10}\!\Big(\tfrac{M_{\rm ecl}}{10^6\,M_\odot}\Big) + 2.83.
```

```{warning}
**The Eq. 9 "2.83" intercept reflects Jeřábková's 4π density convention, not progenax's.**
Eq. 9 is derived from Eq. 7 plus $\log_{10}\rho_{\rm ecl} = 0.61\log_{10}M_{\rm ecl} + 2.08$ and
$\log_{10}\rho_{\rm cl} = 0.61\log_{10}M_{\rm ecl} + 2.85$ (PDF p. 6). progenax instead uses the
internally-consistent **8π** half-mass density of [](marks-kroupa-2012.md) (which reproduces
Marks 2012 Table 1 exactly), under which the consistent intercept is **0.2161**, not 2.83. The
printed 2.83 is verified to be exactly what the paper prints — progenax's departure from it is
the deliberate 8π-convention fix documented in the
[environment chapter](../../10-theory/imfs/environment.md), not a transcription error.
```

## Use in progenax

- [](../../10-theory/imfs/environment.md) — the IGIMF embedded-cluster α₃ mapping.
- `progenax.imf.environment.alpha3_jerabkova_rho` / `x_jerabkova_rho` — Eqs. 6–7 directly.
- `progenax.imf.environment.alpha3_jerabkova_generalized` / `alpha3_jerabkova_mecl` — the
  mass-based form (x from $M_{\rm ecl}$ via the density chain).
- `JERABKOVA_COEFFICIENTS` — the verified constants ($-0.14$, $0.99$, $-0.41$, $1.94$, $-0.87$).

## Notes

```{admonition} Density-convention note (resolved — progenax follows the 8π definition)
:class: note
Eq. 8 writes $\rho_{\rm cl}=3M_{\rm cl}/(4\pi r_h^3)$, but this is **internally inconsistent**
with Jeřábková's own numerical relation $\log_{10}\rho_{\rm ecl}=0.61\log_{10}M_{\rm ecl}+2.08$,
which corresponds to the **8π** half-mass density. The authoritative definition is
[](marks-kroupa-2012.md) (A&A 543, A8, p. 2): $\rho_{\rm ecl}=3M_{\rm ecl}/(8\pi r_h^3)$.
progenax follows this 8π convention ($\rho_{\rm cl}=\rho_{\rm ecl}/\epsilon=3M_{\rm cl}/(8\pi r_h^3)$),
which **reproduces Marks (2012) Table 1 exactly** (NGC 104:
$3\cdot9.40\times10^6/(8\pi\cdot0.49^3)=9.54\times10^6$ = the tabulated $\rho_{\rm cl}$) — the
actual calibration basis for the α₃–ρ relation. The code is therefore consistent with the source;
Jeřábková's Eq. 8 "4π" is the inconsistency, not progenax.
```
