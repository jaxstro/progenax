---
title: Density PDFs and the freefall-density factor
description: Combining the lognormal+power-law density PDF with the ρ^(3/2) freefall-density factor — the cloud-integrated SFR formula that BM19 forward chains consume.
---

# Density PDFs and the freefall-density factor

```{seealso}
This chapter ties together [](density-pdf-fundamentals.md) and
[](freefall-density-factor.md) into the cloud-integrated SFR
expression that the [](bm19.md) framework uses. For the per-cloud
*geometric* magnification ζ that captures the same physics for
power-law profiles, see [](pp20.md).
```

The density PDF $p_V(\rho)$ describes *what density structure* a cloud
has. The freefall-density factor (FDF) describes *how each density
contributes to star formation*. The cloud-integrated SFR is the
convolution of the two. This chapter is short and integrative: it
shows the cloud-SFR formula, identifies the *self-gravitating fraction*
that gates star formation, and connects to the dense-gas SFR
observations that motivated the framework.

## The full SFR formula

For a cloud with volume-density PDF $p_V(\rho)$, mean density
$\langle\rho\rangle$, and total mass $M$, the cloud-integrated SFR is

```{math}
:label: sfr-pdf-fdf
\mathrm{SFR}_{\mathrm{cloud}} \;=\; \varepsilon_{\mathrm{ff,int}}\,
\frac{M}{\langle t_{\mathrm{ff}}\rangle}\,
\int_{\rho_t}^{\infty}
\biggl(\frac{\rho}{\langle\rho\rangle}\biggr)^{\!3/2}\,p_V(\rho)\,\mathrm{d}\rho
```

with $\langle t_{\mathrm{ff}}\rangle = \sqrt{3\pi/(32\,G\,\langle\rho\rangle)}$
the mean-density free-fall time and $\varepsilon_{\mathrm{ff,int}}
\sim 0.01$ the intrinsic SFE per free-fall time. The integrand is
the freefall-density-factor kernel $\rho^{3/2}/\langle\rho\rangle^{3/2}$
weighted by the volume PDF.

The lower limit $\rho_t$ — the **transition density** at which the
PDF crosses from lognormal to power-law — encodes the physical
assumption that *only self-gravitating gas forms stars*. Below $\rho_t$,
turbulent fluctuations compress and re-expand the gas without forming
stars; above $\rho_t$, gravitational collapse dominates and the local
SFR follows the FDF kernel.

## The self-gravitating fraction

A useful auxiliary quantity is the **self-gravitating fraction**
$f_{\mathrm{dense}}$, the mass fraction of the cloud above the
transition density:

```{math}
:label: f-dense
f_{\mathrm{dense}} \;\equiv\; \int_{\rho_t}^{\infty} \frac{\rho}{\langle\rho\rangle}\,p_V(\rho)\,\mathrm{d}\rho
```

This is *not* the SFR-weighted fraction; it is the simple mass
fraction. The SFR uses the FDF-weighted version with $\rho^{3/2}$ in
the integrand. progenax computes both:

```python
from progenax.gravoturb.bm19_model import sigma_s_squared, transition_density, _f_dense_bm19_full_jit
from progenax.gravoturb import bm19_pipeline

mach = 10.0; b = 0.4; alpha = 2.0
sigma_s_sq = sigma_s_squared(mach, b)        # Lognormal variance
s_t = transition_density(sigma_s_sq, alpha)  # Transition log-density

# f_dense — mass fraction above s_t
f_dense = _f_dense_bm19_full_jit(sigma_s_sq, s_t, alpha)

# Full BM19 forward chain (includes f_dense and ζ)
result = bm19_pipeline(mach=mach, b=b, alpha=alpha, eta_survive=0.6)
print(f"f_dense = {result.f_dense:.3f}, f_sub = {result.f_sub:.3f}, ζ = {result.zeta:.3f}")
```

For typical Galactic-cloud parameters ($\mathcal{M} = 10$, $b = 0.4$,
$\alpha = 2$), $f_{\mathrm{dense}} \sim 0.05$–$0.15$ — a few percent
of cloud mass is in the dense star-forming tail at any instant.

## Connecting to PP20 ζ

For a *spatially uniform* density (no PDF spread), the FDF integral
in {eq}`sfr-pdf-fdf` reduces trivially to $M\,\sqrt{\langle\rho\rangle}$
and the cloud SFR is the "top-hat" reference. For a non-uniform
cloud, the integral is *larger* than the top-hat reference — by
exactly the magnification factor

```{math}
:label: zeta-pdf-fdf
\zeta \;=\; \frac{\int (\rho/\langle\rho\rangle)^{3/2}\,p_V(\rho)\,\mathrm{d}\rho \cdot M\sqrt{\langle\rho\rangle}}{M\,\sqrt{\langle\rho\rangle}}
\;=\; \int (\rho/\langle\rho\rangle)^{3/2}\,p_V(\rho)\,\mathrm{d}\rho.
```

The PP20 chapter ([](pp20.md)) evaluates this analytically for a
power-law radial profile $\rho(r) \propto r^{-p}$; this chapter's
PDF-based formulation is its **density-space dual**. They produce the
same number for the same physical cloud.

## Why both formulations matter

The two formulations apply to different observational situations:

```{list-table}
:header-rows: 1

* - Formulation
  - Best when you have…
  - Output
* - **Radial-profile** ([](pp20.md))
  - …a fitted radial profile $\rho(r) \propto r^{-p}$
  - $\zeta(p)$ analytic, fast
* - **PDF-based** (this chapter)
  - …a density-PDF observation (e.g. column-density PDF)
  - Numerical integral; consumes lognormal+power-law parameters
* - **Direct 3D** ([](direct-3d-zeta.md))
  - …a simulation snapshot or detailed observation
  - Direct sum over voxels; no parametric assumption
```

For inference: the radial-profile formulation is what most
observational papers report (clouds are characterised by $r_{\mathrm{eff}}$
and $\rho(r)$). The PDF-based formulation is what cloud simulations
output. The 3D formulation is what cosmological simulations output
when probed at the cloud scale.

progenax's [](bm19.md) forward chain uses the *PDF-based* formulation
because the {cite:t}`Burkhart2018,Burkhart2021` framework is
parameterised in PDF space.

## The α↔p mapping

The PDF tail-slope $\alpha$ and the radial-profile slope $p$ are not
independent: under spherical symmetry and a power-law correspondence
between volume and density {cite:p}`Kritsuk2011,FederrathKlessen2012`,

```{math}
:label: alpha-p-here
p \;=\; \frac{3}{\alpha}
```

This is the same relation noted in the
[PP20 α-to-p mapping](pp20.md#alpha-p).
For the {cite:t}`Burkhart2021` α window $[\alpha_{\mathrm{sat}},\alpha_0]
= [1.5, 3.0]$, the corresponding $p$ window is $[1.0, 2.0]$ — from
"marginally collapsing" to "singular isothermal" radial profiles.

The mapping is what lets progenax's BM19 forward chain hand off to
the PP20 magnification-factor calculation seamlessly: BM19 infers
$\alpha$ from cloud observations, the α↔p mapping converts it to a
radial-profile slope, and PP20 ζ(p) gives the geometric SFR boost.

## Domain of validity

1. **PDF parameterisation must be appropriate.** The lognormal
   +power-law form is well-validated for typical Galactic molecular
   clouds {cite:p}`Kainulainen2014`, but breaks down for very young
   ($t < t_{\mathrm{ff}}$) clouds where the power-law tail has not
   developed, and for very old ($t \gg t_{\mathrm{ff}}$) clouds with
   substantial feedback.
2. **Spherical/cylindrical symmetry assumed** in the α↔p mapping.
   For highly filamentary or sheet-like clouds the mapping is
   approximate.
3. **Single-cloud assumption.** The PDF describes one cloud. Multi-cloud
   superposition (line-of-sight integration in observations) requires
   a separate treatment.

## References

The PDF + FDF combination is {cite:t}`Burkhart2018`. The analytic
α↔p mapping is from {cite:t}`Kritsuk2011`. The intrinsic SFE
$\varepsilon_{\mathrm{ff,int}} \sim 0.01$ comes from
{cite:t}`FederrathKlessen2012`. The full forward-chain implementation
is at [](bm19.md).
