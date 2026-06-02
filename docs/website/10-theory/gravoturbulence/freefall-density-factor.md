---
title: Freefall-density factor (FDF)
description: The functional ρ/t_ff(ρ) ∝ ρ^(3/2) that weights local density by its star-forming efficiency — the kernel that turns the density PDF into a star formation rate.
---

# The freefall-density factor

The **freefall-density factor** (FDF) is the kernel that converts the
density-PDF picture from [](density-pdf-fundamentals.md) into a *star
formation rate*. Its form follows from elementary considerations:
the local star formation rate per unit volume scales as

```{math}
:label: sfr-local
\dot\rho_\star \;\propto\; \frac{\rho}{t_{\mathrm{ff}}(\rho)}
```

with $t_{\mathrm{ff}}$ the local free-fall time

```{math}
:label: tff
t_{\mathrm{ff}}(\rho) \;=\; \sqrt{\frac{3\pi}{32\,G\,\rho}}
\;\propto\; \rho^{-1/2}.
```

Combining {eq}`sfr-local` and {eq}`tff`:

```{math}
:label: fdf
\boxed{\;\;\dot\rho_\star \;\propto\; \rho \cdot \rho^{1/2} \;=\; \rho^{3/2}\;\;}
```

This is the FDF kernel. *High-density gas contributes disproportionately
to the cloud-integrated SFR* — a $\rho \to 10\rho$ region produces
$\sim 32\times$ more SFR per unit volume than the mean-density gas.
It is the proximate reason that
[](pp20.md) magnification factor exists at all: a density-PDF-weighted
cloud has higher SFR than a uniform-density cloud of the same mass.

This chapter unpacks the kernel, explains its connection to the
density PDF, and points to where progenax uses it.

## Why $\rho^{3/2}$ and not something else

Three alternative scalings are sometimes proposed:

```{list-table}
:header-rows: 1

* - Kernel
  - Implied $\dot\rho_\star$
  - Physical assumption
* - $\rho / t_{\mathrm{ff}}$
  - $\rho^{3/2}$
  - Local SFR set by free-fall collapse — **progenax default**
* - $\rho / t_{\mathrm{cross}}$
  - $\rho$
  - Local SFR set by turbulent crossing time (constant Mach)
* - $\rho / t_{\mathrm{cool}}$
  - Variable
  - Local SFR limited by cooling timescale (relevant for low-density warm phases)
```

The $\rho^{3/2}$ kernel is appropriate when *gravitational free-fall*
is the rate-limiting step for local star formation. This applies to
the dense cores ($\rho \gtrsim 10^4\,\mathrm{cm}^{-3}$) where most stars
actually form. The $\rho / t_{\mathrm{cross}}$ kernel would be
appropriate if turbulent feedback regulated SFR on the *crossing*
timescale rather than the free-fall — which is the case in some
massive-star-feedback-regulated regimes but not in the dense-core
regime relevant to dense-gas SFR observations.

The {cite:t}`FederrathKlessen2012` and {cite:t}`Burkhart2018` frameworks
both adopt the $\rho^{3/2}$ kernel; progenax inherits this convention.

## The cloud-integrated SFR

For a cloud with volume-density distribution $p_V(\rho)$ and total
volume $V_R$, the cloud-integrated SFR is

```{math}
:label: sfr-integrated
\mathrm{SFR}_{\mathrm{clump}} \;=\; \varepsilon_{\mathrm{ff}, \mathrm{int}}\,
\int_{V_R} \frac{\rho}{t_{\mathrm{ff}}(\rho)}\,\mathrm{d}V
\;\propto\; \int_{V_R} \rho^{3/2}\,\mathrm{d}V
```

with $\varepsilon_{\mathrm{ff}, \mathrm{int}}$ the **intrinsic
star-formation efficiency per free-fall time** — the fraction of a
free-fall time's mass that is actually converted to stars. Observations
constrain $\varepsilon_{\mathrm{ff}, \mathrm{int}} \sim 0.01$ for
typical Galactic dense gas {cite:p}`Burkhart2018`.

The $\int \rho^{3/2}\,\mathrm{d}V$ integral on the right is exactly
the SFR-weighted volume integral that appears in the [](pp20.md)
magnification-factor definition. The connection:

```{math}
:label: zeta-from-fdf
\zeta \;\equiv\; \frac{\int \rho^{3/2}\,\mathrm{d}V}{M\,\sqrt{\langle\rho\rangle}}
\;=\; \frac{\mathrm{SFR}_{\mathrm{clump}}}{\mathrm{SFR}_{\mathrm{TH}}}
```

ζ is the dimensionless ratio of the FDF integral to its top-hat
reference. The next two chapters build on this:

- [](pp20.md) evaluates {eq}`zeta-from-fdf` analytically for a power-law
  $\rho(r)$ profile.
- [](direct-3d-zeta.md) evaluates {eq}`zeta-from-fdf` numerically for
  an arbitrary 3D density field.

## Turning the density PDF into the SFR

For a cloud described by a volume-density PDF $p_V(\rho)$, the
SFR-weighted *fraction of mass in star-forming gas* is

```{math}
:label: f-sfr
f_{\mathrm{SFR}} \;=\; \frac{\int_{\rho_t}^{\infty} \rho^{3/2}\,p_V(\rho)\,\mathrm{d}\rho}{\langle\rho\rangle\,\sqrt{\langle\rho\rangle}}
```

with $\rho_t$ the transition density at which the PDF crosses from
lognormal to power-law ([](density-pdf-fundamentals.md), {eq}`s-t`).
The lower limit $\rho_t$ encodes a key physical assumption: only the
power-law tail (i.e. the *self-gravitating* high-density gas)
contributes to actual star formation. The lognormal core represents
turbulent fluctuations that compress and re-expand without forming
stars.

`progenax.gravoturb.bm19_model._f_dense_bm19_full_jit` evaluates
{eq}`f-sfr` for the {cite:t}`Burkhart2018` framework — see [](bm19.md)
for the full forward chain.

## Implementation in progenax

The FDF kernel is implicit in every gravoturbulence calculation.
progenax does not expose it as a standalone function (since it has
no free parameters beyond the constants), but it appears inside:

- `magnification_factor(p)` — analytic ζ for power-law profiles.
- `magnification_factor_with_core(p, r_c_over_R)` — numerical ζ for
  cored profiles.
- `zeta_fdf_direct(rho_grid, tail_weights)` — direct measurement
  from a 3D field.
- `_f_dense_bm19_full_jit(sigma_s_sq, s_t, alpha)` — the BM19 forward
  chain.

In each case, the integrand is $\rho^{3/2}$ (or its log-space
equivalent $e^{1.5 s}$) weighted by the appropriate volume element or
PDF.

## Domain of validity

1. **Free-fall regime.** The $\rho^{3/2}$ kernel assumes that local
   gravitational collapse on a free-fall timescale dominates over
   turbulent or thermal stabilisation. At very low density (diffuse
   warm gas) other timescales matter; at very high density (proto-stellar
   cores) hydrostatic-equilibrium effects matter. Both extremes are
   outside the gravoturbulent dense-tail regime.
2. **Isothermal assumption** — $t_{\mathrm{ff}}$ in {eq}`tff` ignores
   thermal-pressure support. For sub-thermal-Jeans-mass clumps,
   pressure support extends the actual collapse time beyond
   $t_{\mathrm{ff}}$. The standard treatment lumps this into
   $\varepsilon_{\mathrm{ff}, \mathrm{int}}$.
3. **No magnetic-field support** — magnetic fields slow collapse via
   ambipolar diffusion. {cite:t}`FederrathKlessen2012` provide a
   magnetic correction; progenax does not currently include it.
4. **No stellar feedback** — the FDF kernel assumes star formation
   does not back-react on the gas. For molecular clouds older than a
   few $t_{\mathrm{ff}}$, feedback becomes important and the kernel
   under-predicts the SFR (because stars have already disrupted the
   dense gas).

## References

The FDF kernel $\rho^{3/2}$ derives from elementary free-fall
arguments; {cite:t}`FederrathKlessen2012` give the canonical modern
treatment in the SFR-prediction context. {cite:t}`TanKrumholzMcKee2006`
is an earlier "single-mean-density" framework that uses the same
kernel without integrating over a density PDF. {cite:t}`Burkhart2018`
combines the kernel with the lognormal+power-law PDF to give the BM19
framework.
