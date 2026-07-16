---
title: Federrath & Klessen (2012)
description: Annotated reference for C. Federrath & R. S. Klessen — the star formation rate of turbulent magnetized clouds; the six SFR-per-freefall-time models and the σ_s²–Mach–magnetization relation.
---

# Federrath & Klessen (2012)

```{admonition} The star formation rate of turbulent magnetized clouds: comparing theory, simulations, and observations
:class: note

**Authors.** C. Federrath, R. S. Klessen

**Reference.** *The Astrophysical Journal* **761, 156** (2012).

**DOI.** [10.1088/0004-637X/761/2/156](https://doi.org/10.1088/0004-637X/761/2/156) ·
**ADS.** 2012ApJ...761..156F

**Verified.** Eqs. 1–8, Table 1, and the four control parameters checked against the held
PDF (pp. 1–4; re-verified 2026-06).
```

## The big idea

FK12 is the **unifying comparison** of analytic star-formation-rate (SFR) theories. It
derives and compares **six** models for the dimensionless *SFR per free-fall time*
$\mathrm{SFR}_\mathrm{ff}$ — the Krumholz & McKee (KM), Padoan & Nordlund (PN), and
Hennebelle & Chabrier (HC) theories, plus **multi-freefall** versions of each — all as a
single integral over the **lognormal density PDF**. It then tests all six against MHD
simulations ($\mathcal{M}=3$–50, $\mathcal{M}_A=1$–$\infty$, solenoidal/mixed/compressive
forcing). The headline: the SFR depends on **four** parameters and the **multi-freefall
KM and PN** models fit best (to within a factor of 2).

## The four controlling parameters (their §1)

1. **Virial parameter** $\alpha_\mathrm{vir} = 2E_\mathrm{kin}/|E_\mathrm{grav}|$.
2. **Sonic Mach number** $\mathcal{M} = \sigma_v/c_s$.
3. **Turbulent forcing parameter** $b$ — fraction of energy in compressive modes:
   $b\approx1/3$ solenoidal (divergence-free), $b\approx0.4$ natural mixture,
   $b\approx1$ compressive (curl-free).
4. **Plasma $\beta = 2\mathcal{M}_A^2/\mathcal{M}^2 = P_\mathrm{th}/P_\mathrm{mag}$**
   (thermal-to-magnetic pressure; $\mathcal{M}_A$ the Alfvén Mach number).

> Comparing forcings, the SFR is **>10× higher for compressive than solenoidal** forcing
> at fixed $\mathcal{M}$; magnetic fields reduce the SFR by a factor of ~2.

## The density PDF and σ_s² (their §2.1–2.2)

The log-density $s=\ln(\rho/\rho_0)$ has a lognormal PDF (Eq. 1) with mean fixed by mass
conservation (Eq. 3), $s_0 = -\tfrac12\sigma_s^2$. The width depends on forcing, Mach
*and* magnetization. For the intermediate field scaling $B\propto\rho^{1/2}$ (Molina+2012),

$$
\sigma_s^2 = \ln\!\left(1 + b^2\mathcal{M}^2\,\frac{\beta}{\beta+1}\right)
\qquad\text{(Eq. 4)},
$$

which in the **hydrodynamic limit** ($\beta\to\infty$, no field) reduces to the relation
used throughout `gravoturb`:

$$
\boxed{\;\sigma_s^2 = \ln\!\left(1 + b^2\mathcal{M}^2\right)\;}
$$

(equivalently Eq. 5, $\sigma_s^2=\ln[1+b^2\mathcal{M}^2\,2\mathcal{M}_A^2/(\mathcal{M}^2+2\mathcal{M}_A^2)]$).
This is the same HD relation as Federrath+2010 Eq. 19 and BM19 Eq. 1.

## The SFR-per-freefall framework (their §2.3) and the six models

The SFR per free-fall time is the mass above a critical density, weighted by the *local*
free-fall rate (the **multi-freefall** insight — gas at different $\rho$ collapses at
different rates):

$$
\mathrm{SFR}_\mathrm{ff} = \frac{\epsilon}{\phi_t}\int_{s_\mathrm{crit}}^{\infty}
   \frac{t_\mathrm{ff}(\rho_0)}{t_\mathrm{ff}(\rho)}\,\frac{\rho}{\rho_0}\,p(s)\,ds
\qquad\text{(Eq. 7)},\qquad t_\mathrm{ff}(\rho)=\sqrt{3\pi/32G\rho}\;\;(\text{Eq. 8}).
$$

The six models (Table 1) differ only in the **critical density** $s_\mathrm{crit}$ (the
lower integration limit) and whether the $t_\mathrm{ff}$ factor is kept inside the integral
(multi-ff) or set to 1 (single-ff):

| Model | $\rho_\mathrm{crit}/\rho_0$ |
| --- | --- |
| KM / multi-ff KM | $(\pi^2/5)\,\phi_x^2\,\alpha_\mathrm{vir}\mathcal{M}^2(1+\beta^{-1})^{-1}$ |
| **PN** / multi-ff PN | $0.067\,\theta^{-2}\,\alpha_\mathrm{vir}\mathcal{M}^2\,f(\beta)$ |
| HC / multi-ff HC | $(\pi^2/5)\,y_\mathrm{cut}^{-2}\,\alpha_\mathrm{vir}\mathcal{M}^2(1+\beta^{-1})+\tilde\rho_\mathrm{crit,turb}$ |

Two facts that matter downstream:

- The **PN** critical density carries the prefactor **$0.067\,\theta^{-2}$** — the same
  form `gravoturb.theory.collapse_threshold` implements (HD: $0.547$ at $\theta=0.35$); see
  [](padoan-nordlund-2011.md).
- The **KM/HC** critical density carries the **$(\pi^2/5)\phi_x^2$** form — *distinct* from
  PN's, which is why the PN11 note flags them as different prefactor conventions.

## Use in progenax

- [](../../10-theory/gravoturbulence/density-pdf-and-fdf.md) — the lognormal+power-law PDF (FK12 §2.1) and the $\rho^{3/2}$ free-fall kernel (the $t_\mathrm{ff}(\rho)^{-1}\propto\rho^{1/2}$ weight in Eq. 7).
- [](burkhart-mocz-2019.md) — BM19 *simplifies* this framework by tying $s_\mathrm{crit}$ to the PDF transition density $s_t$, removing a free critical density.
- $b\in[1/3,1]$ grounds `cluster.turbulence.b_from_environment` and the FK10 forcing parameter.

## Placement-PMF corollary (gravoturb Phase 1, verified vs the PDF 2026-07-16)

The multi-freefall integrand of Eq. 7 — ``[t_ff(ρ₀)/t_ff(ρ)]·(ρ/ρ₀) = (ρ/ρ₀)^{3/2}`` via
Eq. 8 — is the *relative star-formation weight per cell*. Normalizing it into a placement
PMF cancels the ``ε/φ_t`` efficiency prefactors exactly, so **where** stars form needs no
efficiency knob (only **how many** does, and the IC generator takes N⋆ as input). gravoturb's
``placement='multi_freefall'`` uses ``p_⋆ ∝ w(s_turb)·e^{(3/2)s_total}`` with the eligibility
gate ``w`` on the BM19 transition ``s_t`` — the s_t-for-s_crit substitution described under
"Use in progenax" (BM19's derived transition replaces FK12's assumed critical density; FK12
itself does not license s_t, BM19 does). The derived tail-star fraction
``f_sub_derived = Σ_tail p_⋆ / Σ p_⋆`` then replaces the former free ``f_sub`` knob.

## Notes

- **The HD `gravoturb` path drops magnetic fields**: it uses $\sigma_s^2=\ln(1+b^2\mathcal{M}^2)$,
  i.e. FK12 Eq. 4 with $\beta\to\infty$. Magnetization ($\beta<\infty$) *narrows* the PDF
  and lowers the SFR by ~2× — not modelled here.
- Best-fit efficiencies from the simulations: SFE $=1$–$10\%$, local
  $\epsilon\approx0.3$–$0.7$ (best $\sim0.5$); the multi-ff KM and PN models match to a
  factor of 2 over two orders of magnitude in SFR.
- FK12 is the conceptual parent of BM19: BM19 keeps the lognormal+power-law PDF and the
  $\rho^{3/2}$ kernel but replaces the *assumed* critical density with the *derived*
  transition density $s_t$.
