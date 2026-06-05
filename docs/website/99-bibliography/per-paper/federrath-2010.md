---
title: Federrath et al. (2010)
description: Annotated reference for Federrath, Roman-Duval, Klessen, Schmidt & Mac Low — solenoidal vs compressive turbulence forcing and the density-dispersion–Mach relation.
---

# Federrath et al. (2010)

```{admonition} Comparing the statistics of interstellar turbulence in simulations and observations
:class: note

**Authors.** C. Federrath, J. Roman-Duval, R. S. Klessen, W. Schmidt, M.-M. Mac Low

**Reference.** *Astronomy & Astrophysics* **512, A81** (2010).

**DOI.** [10.1051/0004-6361/200912437](https://doi.org/10.1051/0004-6361/200912437) ·
**ADS.** 2010A&A...512A..81F

**Verified.** Equations 1, 10, 11, 18, 19 and the b-parameter range checked against the
held PDF (2026-06). **Correction:** the σ_s²–Mach relation is **Eq. 19**, not "Eq. 14"
(Eq. 14 is the Azzalini skewed-lognormal); the progenax docstring previously mis-cited it.
```

## The big idea

Supersonic turbulence stirs molecular-cloud gas into an enormous range of densities. How
*wide* that range is — the variance of the density PDF — is the single most important
input to analytic theories of star formation (the IMF, the SFR, the dense-gas fraction).
Federrath+2010 (FK10) show that this width is **not** set by the Mach number alone: it
depends just as strongly on **how the turbulence is driven**. Purely **solenoidal**
(divergence-free) forcing and purely **compressive** (curl-free) forcing, *at the same
Mach number*, produce density PDFs whose standard deviations differ by a factor of ~3.
This is encoded in a single **forcing parameter** $b$.

## Core relations

Work in the logarithmic density $s = \ln(\rho/\langle\rho\rangle)$ (FK10 Eq. 1). For
driven isothermal supersonic turbulence the volume-weighted PDF of $s$ is close to a
**lognormal** (Eq. 10):

$$
p_s(s)\,ds = \frac{1}{\sqrt{2\pi\sigma_s^2}}\,
\exp\!\left[-\frac{(s-\langle s\rangle)^2}{2\sigma_s^2}\right] ds .
$$

Mass conservation ($\int e^{s} p_s\,ds = \langle\rho\rangle/\langle\rho\rangle = 1$) fixes
the mean in terms of the variance (Eq. 11):

$$
\langle s\rangle = -\tfrac{1}{2}\sigma_s^2 .
$$

The **density-dispersion–Mach relation** is stated in two equivalent forms. The linear
form for the (non-log) density (Eq. 18, after Padoan, Nordlund & Jones 1997; Passot &
Vázquez-Semadeni 1998):

$$
\frac{\sigma_\rho}{\langle\rho\rangle} = b\,\mathcal{M},
$$

and — assuming the lognormal (Eq. 10) — its logarithmic counterpart (**Eq. 19**), the
relation progenax actually uses:

$$
\boxed{\;\sigma_s^2 = \ln\!\left(1 + b^2\mathcal{M}^2\right)\;}
$$

with the **same** parameter $b$. Here $\mathcal{M}=\sigma_v/c_s$ is the rms sonic Mach
number.

### The forcing parameter $b$

$b$ measures the fraction of compressive (longitudinal) power in the driving:

| Driving | $b$ | Notes |
|---|---|---|
| Solenoidal (divergence-free) | $\approx 1/3$ | natural floor in 3D (1 of 3 spatial modes is longitudinal) |
| Natural mixture ($\zeta=0.5$) | $\approx 0.4$ | progenax `B_DEFAULT` |
| Compressive (curl-free) | $\approx 1$ | maximal density contrast |

FK10 §3.6 establishes $b$ as a smooth function of the forcing parameter $\zeta$ and
reconciles the earlier disagreement (Padoan+1997 found $b\approx0.5$; Passot &
Vázquez-Semadeni 1998 found $b\approx1$) as different points along this $b(\zeta)$ curve.

### Departures from lognormality (intermittency)

FK10 emphasise that the PDF is *not perfectly* lognormal: there are non-Gaussian
skewness and kurtosis in the wings, caused by intermittency (rare strong shocks and
rarefactions). They model these with a **skewed lognormal** (Azzalini 1985; **Eq. 14**)
and a 4th-order expansion (Eq. 17). This is the reason a purely Gaussian/lognormal
description — and, by extension, a Gaussian random field — captures the *variance* but
not the coherent filaments and sheets of real supersonic turbulence.

## Use in progenax

- [cluster/turbulence.py](../../../../src/progenax/cluster/turbulence.py) —
  `sigma_ln_rho_from_mach` implements **Eq. 19**; `b_from_environment` interpolates
  $b\in[1/3, \sim0.7]$ with density (a *tentative* mapping, not from FK10).
- `experimental/gravoturb_fdf/theory/bm19.py` — `sigma_s_squared(mach, b)` is exactly
  Eq. 19; it is the entry point of the BM19 gravoturbulent density PDF
  ([](burkhart-mocz-2019.md)).

## Notes

- **Citation correction.** The σ_s²–Mach relation is FK10 **Eq. 19** (with Eq. 18 the
  linear $\sigma_\rho/\langle\rho\rangle=b\mathcal{M}$). The progenax docstring previously
  attributed it to "Eq. 14", which is in fact the Azzalini skewed-lognormal PDF — a
  citation bug fixed in the 2026-06 clean-room pass.
- **$b$ is a driving diagnostic, not a free knob.** It should sit in $[1/3, 1]$;
  values outside that range are unphysical for isothermal turbulence.
- **Intermittency is real and unmodelled here.** The lognormal (and any Gaussian random
  field built on it) reproduces $\sigma_s^2$ but not the non-Gaussian wings / coherent
  structures. This is a known limitation of the FDF field realisation.
- The **column-density** PDF (observable) has a smaller dispersion than the volumetric
  PDF because line-of-sight integration averages out fluctuations (FK10 §3.5).
