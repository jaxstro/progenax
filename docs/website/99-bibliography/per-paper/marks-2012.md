---
title: Marks et al. (2012)
description: Annotated reference for M. Marks et al. — Evidence for top-heavy stellar IMFs with increasing density and decreasing metallicity (the α₃ density/metallicity relations).
---

# Marks et al. (2012)

```{admonition} Evidence for top-heavy stellar initial mass functions with increasing density and decreasing metallicity
:class: note

**Authors.** M. Marks, P. Kroupa, J. Dabringhausen, M. S. Pawlowski (Argelander-Institut
für Astronomie / Max-Planck-Institut für Radioastronomie, Bonn).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **422**, 2246–2254
(2012). Accepted 2012 February 18.

**DOI.** [10.1111/j.1365-2966.2012.20767.x](https://doi.org/10.1111/j.1365-2966.2012.20767.x) ·
**ADS.** [2012MNRAS.422.2246M](https://ui.adsabs.harvard.edu/abs/2012MNRAS.422.2246M)
```

## Abstract (paraphrased)

Uses residual-gas-expulsion modelling of the low-mass present-day mass functions of Galactic
globular clusters (GCs) to infer their birth conditions (mass, radius, density, metallicity)
and the high-mass IMF slope $\alpha_3$ required to expel the gas. Finds the high-mass IMF must
become **top-heavy** (lower $\alpha_3$) with **increasing pre-cluster core density** and
**decreasing metallicity**. A *Fundamental Plane* in $(\alpha_3, \log\rho_{\rm cl}, {\rm [Fe/H]})$
captures both dependences. This is the empirical basis for the density/metallicity-dependent
high-mass IMF used in IGIMF theory.

## The physics: top-heaviness from the gas-expulsion energy budget (§2, verified)

The α₃ relations are **not** fitted ad hoc — they fall out of an energy argument. An
embedded cluster forms with star-formation efficiency
$\epsilon = M_{\rm ecl}/(M_{\rm ecl}+M_{\rm gas})$; the leftover gas must be expelled by
feedback from the massive (O/B) stars. Marks et al. ask: *what high-mass slope $\alpha_3$
delivers exactly enough energy to unbind the residual gas within a cluster crossing time?*

**Energy required.** For a Plummer cluster ($r_h = 1.305\,r_{\rm pl}$) that expels its gas,
the change in binding energy is (Eq. 5)

$$
E_{\rm OB}^{\rm req} = 1.305\,\frac{3G}{32}
\left(\frac{M_{\rm cl}^2}{r_{h,\rm i}} - \frac{M_{\rm ecl}\,M_{\rm cl}}{r_{h,\rm i}}\right),
$$

with $M_{\rm cl}=M_{\rm ecl}/\epsilon$ the cloud-core mass and $r_{h,\rm f}=r_{h,\rm i}\,M_{\rm cl}/M_{\rm ecl}$
the post-expulsion half-mass radius (Eq. 6, slow/adiabatic expulsion, Hills 1980).

**Energy supplied.** The radiative + mechanical power deposited by all stars is (Eqs. 8–9)

$$
\dot E = \int_{0.08\,M_\odot}^{m_{\max}} \dot E_*(m)\,\xi(m)\,dm,
\qquad
\log_{10}\!\frac{\dot E_*}{{\rm erg\,Myr^{-1}}} = 50 + 1.72\left(\log_{10}\tfrac{m}{M_\odot} - 1.55\right),
$$

so the budget is dominated by massive stars and is negligible for low-mass stars. Integrated
over a crossing time $\tau_{\rm cr}=\tfrac{2}{\sqrt G}M_{\rm cl}^{-1/2}r_h^{3/2}$ (Eq. 7),
$\alpha_3$ is chosen so that $E_{\rm OB}^{\tau_M}(\alpha_3)=E_{\rm OB}^{\rm req}$.

**Why it goes top-heavy with density.** A denser, more massive cluster sits in a deeper
potential with more gas to expel ($E_{\rm OB}^{\rm req}$ larger), so it needs *more* massive
stars — a *flatter* (smaller) $\alpha_3$. Lower metallicity raises the Jeans mass, also
favouring massive stars. Hence $\alpha_3$ **decreases with $\rho_{\rm cl}$ and with
decreasing [Fe/H]** — exactly the trends fit below.

## Inputs — what actually sets α₃ (and what does *not*)

The variation is driven by **three environmental quantities only**: the pre-cluster
**cloud-core density $\rho_{\rm cl}$**, the **metallicity [Fe/H]**, and the **cluster mass**
($M_{\rm cl}$/$M_{\rm ecl}$, which fix $\rho_{\rm cl}$ via the $r_h$–$M$ relation). It does
**not** involve any turbulence statistic — neither the density-PDF width $\sigma_s$ nor the
turbulent power-spectrum slope $\beta$. In progenax this means the environment-dependent IMF
slopes are independent of `cluster.turbulence.spectral_slope_from_mach`: the gravoturbulent
$\beta$ feeds only the experimental FDF *spatial* field, never the mass function.

## The canonical IMF and the α₃ relations (verified against the paper)

The stellar IMF is the canonical multi-power-law (Eq. 2; Kroupa 2001),
$\xi(m)\propto m^{-\alpha_i}$ with $\alpha_1=1.3$ ($0.08$–$0.5\,M_\odot$),
$\alpha_2=\alpha_3=2.3$ ($\ge 0.5\,M_\odot$). The star-formation efficiency is
$\epsilon = M_{\rm ecl}/(M_{\rm ecl}+M_{\rm gas})$, $0.1<\epsilon<0.5$ (Eq. 1).

**1-D relations (Eq. 11, Table 3, p. 2251).** Each environmental variable $\lambda$ gives

```{math}
:label: marks-table3
\alpha_3(\lambda) = \begin{cases} p_\lambda\,\lambda + q_\lambda, & \lambda \gtrless \lambda_{\rm lim}\\ 2.3, & \text{otherwise}\end{cases}
```

| $\lambda$ | $p_\lambda$ | $q_\lambda$ | $\lambda_{\rm lim}$ | branch |
|-----------|-------------|-------------|---------------------|--------|
| $\log_{10}(M_{\rm cl}/10^6 M_\odot)$ | $-0.94$ | $2.14$ | $0.68$ | $>$ |
| $\log_{10}(M_{\rm ecl}/10^6 M_\odot)$ | $-0.77$ | $1.59$ | $0.27$ | $>$ |
| $\log_{10}(\rho_{\rm cl}/10^6 M_\odot{\rm pc}^{-3})$ | $-0.43$ | $1.86$ | $0.095$ | $>$ |
| ${\rm [Fe/H]}$ | $0.66$ | $2.63$ | $-0.5$ | $<$ |

**Fundamental Plane (Eqs. 13–14, p. 2252).** With $\vartheta = 98°$,

```{math}
:label: marks-fp
x' = \cos\vartheta\,{\rm [Fe/H]} + \sin\vartheta\,\log_{10}\!\Big(\tfrac{\rho_{\rm cl}}{10^6 M_\odot{\rm pc}^{-3}}\Big),
\qquad
\alpha_3 = \begin{cases} -0.4072\,x' + 1.9383, & x' \ge 0.87\\ 2.3, & \text{otherwise}\end{cases}
```

with $\cos 98° = -0.139$, $\sin 98° = 0.990$. Density dominates metallicity in setting
$\alpha_3$ (smaller scatter in Fig. 3 than Fig. 4).

**Low-mass metallicity dependence (Eq. 12, p. 2251).**

```{math}
:label: marks-lowmass
\alpha_{1,2}({\rm [Fe/H]}) = \alpha_{1,2,c} + \Delta\alpha\,{\rm [Fe/H]}, \qquad \Delta\alpha \approx 0.5,
```

reproducing the Table 4 grid (e.g. ${\rm [Fe/H]}=-2 \Rightarrow \alpha_1=0.30,\ \alpha_2=1.30$).

## Use in progenax

- [](../../10-theory/imfs/environment.md) — the density/metallicity-dependent high-mass IMF.
- `progenax.imf.environment.alpha3_marks_plane` — Fundamental Plane {eq}`marks-fp`.
- `progenax.imf.environment.alpha3_marks_table3` — the four 1-D relations {eq}`marks-table3`.
- `progenax.imf.environment.lowmass_slopes_metallicity` — Eq. 12 low-mass slopes.
- `MARKS_COEFFICIENTS`, `MARKS_TABLE3_COEFFICIENTS` — the tabulated constants (all verified exact).

## Notes

The α₃(x) form was slightly revised in the **Marks et al. (2014) erratum** to
$-0.41\,x + 1.94$; that revised form is the one adopted by [](jerabkova-2018.md) Eq. 6 and
used in progenax's `JERABKOVA_COEFFICIENTS` (kept distinct from this paper's MNRAS Fundamental-Plane
fit $-0.4072/1.9383$). The radius–mass relation $r_h = 0.1\,(M_{\rm ecl}/M_\odot)^{0.13}$ pc used
in the density chain comes from the companion paper **Marks & Kroupa (2012), A&A 543, A8** (a
*different* paper), not this one.
