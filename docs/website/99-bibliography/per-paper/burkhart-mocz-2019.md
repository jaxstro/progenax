---
title: Burkhart & Mocz (2019)
description: Annotated reference for B. Burkhart & P. Mocz — the self-gravitating gas fraction and the critical density for star formation.
---

# Burkhart & Mocz (2019)

```{admonition} The self-gravitating gas fraction and the critical density for star formation
:class: note

**Authors.** B. Burkhart, P. Mocz

**Reference.** *The Astrophysical Journal* **879, 129** (2019).

**DOI.** [10.3847/1538-4357/ab25ed](https://doi.org/10.3847/1538-4357/ab25ed)

**Verified.** Equations + ranges checked against the held PDF (pp. 1–8, Eqs. 1–3, 11, 16–20, 24; Figs. 1, 5; re-verified 2026-06).
```

## The big idea

In a turbulent, self-gravitating molecular cloud the volume density $\rho$ is not
single-valued — it has a **probability distribution function (PDF)**. Where gravity is
weak the PDF is a **lognormal** set by supersonic turbulence; where gravity wins it grows
a **power-law tail** at high density (collapsing regions). BM19 model the PDF as a
**piecewise lognormal + power law** and show that the density where the two pieces join,
$s_t$, is *not* a free parameter but a **mathematically motivated critical density for
star formation** — the post-shock density where the Jeans length equals the sonic length.
The mass fraction in the power-law tail is the **self-gravitating (dense) gas fraction**
$f_\mathrm{dense}$, which sets the star-formation efficiency.

Throughout, work in the log-density $s \equiv \ln(\rho/\rho_0)$ ($\rho_0$ = mean density).

## Core equations

**Lognormal width (Eq. 1).** Supersonic isothermal turbulence gives a lognormal of variance

$$
\sigma_s^2 = \ln\!\left(1 + b^2 \mathcal{M}^2\right),
$$

with $\mathcal{M}$ the sonic Mach number and $b\in[1/3,1]$ the driving parameter
($1/3$ solenoidal → $1$ compressive). Mass conservation fixes the lognormal mean at
$s_0 = -\sigma_s^2/2$, so $\int e^{s} p_\mathrm{LN}\,ds = 1$.

**Transition density (Eq. 2).** The lognormal joins the power-law tail at

$$
s_t = \left(\alpha - \tfrac12\right)\sigma_s^2 ,
$$

where $\alpha$ is the slope of the tail $p_\mathrm{PL}(s)\propto e^{-\alpha s}$. This is
*derived*, not fitted. For the canonical collapsing value $\alpha = 3/2$ it reduces to
$s_t = \sigma_s^2$ (Eq. 16).

**PDF slope ↔ radial slope.** A spherical region $\rho\propto r^{-\kappa}$ produces a
density-PDF power law $p(s)\propto e^{-\alpha s}$ with

$$
\kappa = 3/\alpha .
$$

So $\alpha=3/2 \Leftrightarrow \kappa=2$ (isothermal collapse; Shu 1977) and
$\alpha=2 \Leftrightarrow \kappa=3/2$.

**Self-gravitating gas fraction (Eqs. 17–20).** Defining dense gas as all mass above $s_t$,

$$
f_\mathrm{dense} \equiv \frac{M_\mathrm{PL}}{M_\mathrm{LN} + M_\mathrm{PL}}
= \frac{\displaystyle\int_{s_t}^{\infty} e^{s}\,p_\mathrm{PL}(s)\,ds}
       {\displaystyle\int_{-\infty}^{s_t} e^{s}\,p_\mathrm{LN}(s)\,ds
        + \int_{s_t}^{\infty} e^{s}\,p_\mathrm{PL}(s)\,ds }.
$$

Demanding the PDF be **continuous** at $s_t$ fixes the tail amplitude
$C = p_\mathrm{LN}(s_t)\,e^{\alpha s_t}$, and the mass-weighted integrals evaluate to

$$
M_\mathrm{PL} = \frac{C\,e^{(1-\alpha)s_t}}{\alpha-1}, \qquad
M_\mathrm{LN} = \tfrac12\!\left[\,1 + \mathrm{erf}\!\left(
        \frac{s_t - \sigma_s^2/2}{\sqrt2\,\sigma_s}\right)\right].
$$

(BM19 Eq. 19/20 write the identical result multiplied through by $\alpha-1$.) The
$1/(\alpha-1)$ makes $f_\mathrm{dense}\to1$ as $\alpha\to1$ — the limit where the whole
cloud is self-gravitating — and is the term a numerical implementation must guard.

**Pure-lognormal limit.** Removing the power law gives
$f_\mathrm{dense}^{LN} = \tfrac12\,\mathrm{erfc}\!\big((s_t-\sigma_s^2/2)/(\sqrt2\sigma_s)\big)$;
the true $f_\mathrm{dense}$ exceeds it because the shallower tail adds high-density mass.

**Behaviour (their Fig. 5).** $f_\mathrm{dense}$ *decreases* with $\mathcal{M}$ (the PDF
widens, $s_t$ moves up) and *decreases* with $\alpha$ (steeper tail → less dense gas). The
instantaneous SFE is $\epsilon_\mathrm{inst}=\epsilon_0\,f_\mathrm{dense}$ (Eq. 24).

**The central result: only two parameters, no free critical density.** Because $s_t$ is
*derived* from $(\sigma_s,\alpha)$ via Eq. 2, the self-gravitating fraction in Eqs. 18–20 is
controlled by **just those two numbers** — BM19 stress there is "*no need to invoke a
critical density of collapse*." This is what lets the SFE be predicted from observables
(cloud Mach number / PDF width and the measured tail slope) rather than a tuned threshold.

**$s_t$ *is* a critical density (Eqs. 9–15).** Equating the Jeans length to the post-shock
sonic length gives a critical overdensity $\rho_\mathrm{crit}/\rho_0 = \exp(s_\mathrm{crit})
= \tfrac{\pi^2}{15}\,\alpha_\mathrm{vir}\,\mathcal{M}^2$ (their Eq. 11, with
$\alpha_\mathrm{vir}=5v_L^2 R/GM$). For virialised clouds ($\alpha_\mathrm{vir}\approx1$)
this matches the post-shock density $\rho_\mathrm{ps}/\rho_0=\mathcal{M}^2$ to within a
factor of a few, and BM19 show $s_t\approx s_\mathrm{crit}\approx s_\mathrm{ps}$ in the
$\alpha\simeq1.5$–2 limit. They validate $s_t$ and $f_\mathrm{dense}$ against AREPO
moving-mesh gravo-turbulent simulations ($b=1/3$, $\mathcal{M}=5,10,16$; their Figs. 4–5),
finding the dense fraction (and hence SFE) is **weakly anti-correlated with Mach number**.

## Use in progenax

- [](../../10-theory/gravoturbulence/bm19.md) — BM19 1-D PDF theory and the $\alpha$ window.
- [](../../10-theory/gravoturbulence/magnification-factor.md) — the $\alpha\leftrightarrow p$ ($\kappa=3/\alpha$) mapping.
- `experimental/gravoturb/theory/density_pdf.py` — `sigma_s_squared`, `transition_density`,
  `f_dense_bm19_full`, `f_dense_lognormal_limit`, `pdf_slope_to_radial`.
- `experimental/gravoturb/theory/density_cdf.py` — the BM19 volume PDF + inverse-CDF that
  imprints the BM19 marginal on the 3-D FDF field via the rank copula.

Validation: closed-form $f_\mathrm{dense}$ matches direct quadrature of Eq. 18 to
$\mathrm{rel}\,10^{-4}$ (AC1); mass conservation $\int e^{s}p_\mathrm{LN}\,ds=1$ to
$10^{-3}$ (AC2).

## Notes

- The companion **Paper I** (Burkhart 2018) defines the renormalisation constants $N,C$
  and the density shift $s_s$ (Eq. 3) for mass-conserving periodic-box simulations.
- $\alpha$ is the **PDF** slope, not the radial slope ($\kappa=3/\alpha$). The canonical
  collapsing window is $\alpha\in[1.5,2]$ (saturating toward $\alpha\simeq1.5$).
