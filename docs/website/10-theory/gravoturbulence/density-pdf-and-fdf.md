---
title: Density PDFs and the freefall-density factor
description: The Federrath & Klessen (2012) lognormal+power-law density PDF, the ρ^(3/2) freefall-density factor, and how the two combine into the cloud-integrated dense-gas SFR that the BM19 forward chain consumes — with the single α↔p mapping the gravoturbulence section relies on.
---

# Density PDFs and the freefall-density factor

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent + fractal-density-field (FDF) pipeline was rebuilt **clean-room** (2026-06) as
the standalone **`gravoturb`** package — a follow-up-paper feature **excluded from the released
progenax wheel**. Import it as `gravoturb` (repo-only, under `src/experimental/`), **not** as
`progenax.gravoturb` (removed in the 2026-06 rewrite). Fresh validation:
`src/experimental/gravoturb/VALIDATION_SUMMARY.md`.
```

A turbulent molecular cloud has a *distribution* of densities, not a single density. The shape of
that distribution — the **volume-density PDF** $p_V(\rho)$ — is the foundational object that every
subsequent chapter in the gravoturbulence section consumes. This chapter develops the two halves of
the framework and then joins them:

1. **The density PDF** — the {cite:t}`FederrathKlessen2012` lognormal core (set by turbulence)
   joined to a high-density power-law tail (set by self-gravity-driven collapse).
2. **The freefall-density factor (FDF)** — the kernel $\rho/t_{\mathrm{ff}}(\rho) \propto
   \rho^{3/2}$ that weights each density by its star-forming efficiency.
3. **The cloud-integrated SFR** — the convolution of the two, which yields the dense-mass fraction
   and the magnification factor that [](bm19.md) and [](magnification-factor.md) consume.

The chapter ends with the single, canonical **α↔p mapping** ($p = 3/\alpha$) that connects the
PDF-tail slope to the radial-profile slope — every downstream page refers back to this statement.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the molecular-cloud density PDF and the freefall-density factor that the rest of the (experimental) gravoturbulence section consumes; no prior turbulence literature assumed.
**Prerequisites:** [gravoturbulence overview](index.md) (the forward-chain framing and experimental scope).
**You'll get:** the Federrath–Klessen lognormal+power-law PDF, the $\rho^{3/2}$ FDF kernel, how they combine into the cloud-integrated SFR, and the canonical α↔p mapping ($p = 3/\alpha$).
:::

---

## Part 1 — The density PDF

### The lognormal core

In a *purely turbulent* (non-self-gravitating) supersonic medium, the density PDF is
well-described by a lognormal:

```{math}
:label: pdf-lognormal
p_V(s) \;=\; \frac{1}{\sqrt{2\pi\sigma_s^2}}\,\exp\!\biggl[-\frac{(s - \langle s\rangle)^2}{2\sigma_s^2}\biggr]
```

where $s \equiv \ln(\rho/\langle\rho\rangle)$ is the log-density,
$\langle s \rangle = -\sigma_s^2/2$ (so that $\langle\rho\rangle$ is the mean), and $\sigma_s^2$ is
the log-density variance. The lognormal form follows from the central limit theorem applied to the
cumulative product of compressions and rarefactions a fluid element experiences in a turbulent
cascade.

The variance scales with Mach number as

```{math}
:label: sigma-s
\sigma_s^2 \;=\; \ln\bigl[1 + b^2\,\mathcal{M}^2\bigr]
```

with $b$ the **turbulence-driving parameter** ($b \in [1/3, 1]$) and
$\mathcal{M} = v_{\mathrm{turb}} / c_s$ the sonic Mach number. The forcing parameter has
well-defined physical limits:

```{list-table}
:header-rows: 1

* - $b$
  - Forcing
  - Physical interpretation
* - $1/3$
  - Solenoidal (divergence-free)
  - Turbulence stirred by shear; rotational modes only
* - $0.4$
  - Natural mix
  - Default for ISM turbulence in observations
* - $1$
  - Compressive (curl-free)
  - Turbulence stirred by compression; e.g. expanding HII regions, supernova shells
```

For a typical Mach $\mathcal{M} = 10$ molecular cloud and $b = 0.4$,
$\sigma_s^2 = \ln[1 + 16] \approx 2.83$, so $\sigma_s \approx 1.7$. The lognormal therefore spans
about 4 e-folds in $s$ at the $\pm 2\sigma$ level — orders of magnitude in $\rho$.

### The power-law tail

When self-gravity becomes important — at densities high enough that the local freefall time is
shorter than the turbulent crossing time — the density PDF develops a **power-law tail** at high
$s$:

```{math}
:label: pdf-power-law
p_V(s) \;\propto\; e^{-\alpha\,s}\quad\text{for}\quad s \gtrsim s_t
```

where $\alpha$ is the tail slope and $s_t$ is the **transition density** where the PDF crosses from
lognormal to power-law. Physically, the power-law tail represents *self-gravitating gas* in
approximate isothermal collapse — the {cite:t}`Kritsuk2011` derivation maps $\rho \propto r^{-p}$ to
a power-law in $s$ with slope $\alpha = 3/p$ (the [α↔p mapping](#alpha-p) developed at the end of this
chapter).

### The transition density $s_t$

The matching point $s_t$ between lognormal and power-law is constrained by *continuity* of $p_V(s)$
{cite:p}`Burkhart2018`:

```{math}
:label: s-t
s_t \;=\; \Bigl(\alpha - \tfrac{1}{2}\Bigr)\,\sigma_s^2
```

This is a closed-form expression: given $\sigma_s^2$ (set by Mach + forcing) and $\alpha$ (a free
parameter), $s_t$ is determined. `gravoturb`'s `transition_density(alpha, sigma_s_sq)` returns
this value directly; it is differentiable in both arguments. (For $\alpha = 3/2$ it reduces to
$s_t = \sigma_s^2$, BM19 Eq. 16.)

For typical molecular-cloud parameters ($\sigma_s^2 \approx 2.83$, $\alpha \approx 2$),
$s_t = (\alpha - \tfrac{1}{2})\sigma_s^2 \approx 4.25$, meaning the lognormal-to-power-law
transition occurs around $\rho/\langle\rho\rangle \approx e^{4.25} \approx 70$. This matches the
{cite:t}`Kainulainen2014` observational dense-gas threshold $s_{\mathrm{th}} \approx 4.2$ (as adopted
by {cite:t}`ParmentierPasquali2020`).

### Evaluating the PDF in `gravoturb`

```python
import jax.numpy as jnp
from gravoturb.theory.density_pdf import sigma_s_squared, transition_density
from gravoturb.theory.density_cdf import bm19_volume_pdf

mach = 10.0          # Sonic Mach number
b = 0.4              # Forcing parameter
alpha = 2.0          # Power-law tail slope

sigma_s_sq = sigma_s_squared(mach, b)            # ≈ 2.833
sigma_s = jnp.sqrt(sigma_s_sq)
s_t = transition_density(alpha, sigma_s_sq)      # ≈ 4.250   (args: alpha, σ_s²)

# Evaluate the full PDF (it takes mach, b, alpha directly — σ_s² and s_t are internal)
s_grid = jnp.linspace(-5, 10, 200)
pdf_values = bm19_volume_pdf(s_grid, mach, b, alpha)
```

`bm19_volume_pdf` evaluates the lognormal piece for $s < s_t$ and the power-law piece for
$s \ge s_t$, with continuous matching at $s_t$ enforced by the closed-form {eq}`s-t`. The function
is JIT-compatible and differentiable in $\mathcal{M}$, $b$, and $\alpha$ — useful for inferring all
three from observed cloud properties.

---

## Part 2 — The freefall-density factor

The **freefall-density factor** (FDF) is the kernel that converts the density-PDF picture into a
*star formation rate*. Its form follows from elementary considerations: the local star formation
rate per unit volume scales as

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

This is the FDF kernel. *High-density gas contributes disproportionately to the cloud-integrated
SFR* — a $\rho \to 10\rho$ region produces $\sim 32\times$ more SFR per unit volume than the
mean-density gas. It is the proximate reason that the [magnification factor](magnification-factor.md)
exists at all: a density-PDF-weighted cloud has higher SFR than a uniform-density cloud of the same
mass.

### Why $\rho^{3/2}$ and not something else

Three alternative scalings are sometimes proposed:

```{list-table}
:header-rows: 1

* - Kernel
  - Implied $\dot\rho_\star$
  - Physical assumption
* - $\rho / t_{\mathrm{ff}}$
  - $\rho^{3/2}$
  - Local SFR set by free-fall collapse — **`gravoturb` default**
* - $\rho / t_{\mathrm{cross}}$
  - $\rho$
  - Local SFR set by turbulent crossing time (constant Mach)
* - $\rho / t_{\mathrm{cool}}$
  - Variable
  - Local SFR limited by cooling timescale (relevant for low-density warm phases)
```

The $\rho^{3/2}$ kernel is appropriate when *gravitational free-fall* is the rate-limiting step for
local star formation. This applies to the dense cores ($\rho \gtrsim 10^4\,\mathrm{cm}^{-3}$) where
most stars actually form. The $\rho / t_{\mathrm{cross}}$ kernel would be appropriate if turbulent
feedback regulated SFR on the *crossing* timescale rather than the free-fall — which is the case in
some massive-star-feedback-regulated regimes but not in the dense-core regime relevant to dense-gas
SFR observations.

The {cite:t}`FederrathKlessen2012` and {cite:t}`Burkhart2018` frameworks both adopt the $\rho^{3/2}$
kernel; `gravoturb` inherits this convention.

---

## Part 3 — Combining the PDF and the FDF: the cloud-integrated SFR

The density PDF $p_V(\rho)$ describes *what density structure* a cloud has. The FDF describes *how
each density contributes to star formation*. The cloud-integrated SFR is the convolution of the two.

### The full SFR formula

```{admonition} Normalization convention (used consistently throughout this section)
:class: important
Every density-space integral below is written in **dimensionless** form, with $\rho$ measured
relative to the volume-mean $\langle\rho\rangle$ — equivalently in log-density $s = \ln(\rho/\langle
\rho\rangle)$. SFR-weighting uses the kernel $(\rho/\langle\rho\rangle)^{3/2}$ (from {eq}`fdf`); pure
mass-weighting uses $(\rho/\langle\rho\rangle)$. This matches the `gravoturb` code, where
`f_dense_bm19_full` integrates the *mass-weighted* PDF in $s$-space and all densities are referred to
$\langle\rho\rangle$ (the $\langle e^s\rangle = 1$ convention enforced by `rank_copula_field`). The
overall dimensional rate is carried by the prefactors $M/\langle t_{\mathrm{ff}}\rangle$ and
$\varepsilon_{\mathrm{ff,int}}$, **not** by the integrand.
```

For a cloud with volume-density PDF $p_V(\rho)$, mean density $\langle\rho\rangle$, and total mass
$M$, the cloud-integrated SFR is

```{math}
:label: sfr-pdf-fdf
\mathrm{SFR}_{\mathrm{cloud}} \;=\; \varepsilon_{\mathrm{ff,int}}\,
\frac{M}{\langle t_{\mathrm{ff}}\rangle}\,
\int_{\rho_t}^{\infty}
\biggl(\frac{\rho}{\langle\rho\rangle}\biggr)^{\!3/2}\,p_V(\rho)\,\mathrm{d}\rho
```

with $\langle t_{\mathrm{ff}}\rangle = \sqrt{3\pi/(32\,G\,\langle\rho\rangle)}$ the mean-density
free-fall time and $\varepsilon_{\mathrm{ff,int}} \sim 0.01$ the **intrinsic star-formation
efficiency per free-fall time** — the fraction of a free-fall time's mass that is actually converted
to stars. Observations constrain $\varepsilon_{\mathrm{ff,int}} \sim 0.01$ for typical Galactic
dense gas {cite:p}`Burkhart2018`. The integrand is the dimensionless FDF kernel
$(\rho/\langle\rho\rangle)^{3/2}$ weighted by the volume PDF.

The lower limit $\rho_t$ — the **transition density** at which the PDF crosses from lognormal to
power-law — encodes the physical assumption that *only self-gravitating gas forms stars*. Below
$\rho_t$, turbulent fluctuations compress and re-expand the gas without forming stars; above
$\rho_t$, gravitational collapse dominates and the local SFR follows the FDF kernel. The lognormal
core represents turbulent fluctuations that compress and re-expand without forming stars.

### The self-gravitating fraction

A useful auxiliary quantity is the **self-gravitating (dense) fraction** $f_{\mathrm{dense}}$, the
mass fraction of the cloud above the transition density:

```{math}
:label: f-dense
f_{\mathrm{dense}} \;\equiv\; \int_{\rho_t}^{\infty} \frac{\rho}{\langle\rho\rangle}\,p_V(\rho)\,\mathrm{d}\rho
\;=\; \int_{s_t}^{\infty} e^{s}\,p_V(s)\,\mathrm{d}s .
```

This is *not* the SFR-weighted fraction; it is the simple **mass** fraction (kernel
$\rho/\langle\rho\rangle = e^s$, exponent 1). The SFR uses the FDF-weighted version with
$(\rho/\langle\rho\rangle)^{3/2}$ (exponent $3/2$) in the integrand. The two differ only in the
exponent of the dimensionless density kernel — both integrate the volume PDF, both are
dimensionless. `gravoturb` computes both:

```python
from gravoturb.theory.density_pdf import sigma_s_squared, transition_density, f_dense_bm19_full
from gravoturb.theory.dense_gas_sfr import magnification_factor

mach, b, alpha = 10.0, 0.4, 2.0
sigma_s_sq = sigma_s_squared(mach, b)        # lognormal variance        ≈ 2.833
s_t = transition_density(alpha, sigma_s_sq)  # transition log-density    ≈ 4.250  (args: alpha, σ_s²)

# f_dense — mass fraction in the dense power-law tail
f_dense = f_dense_bm19_full(mach, b, alpha)  # ≈ 0.057

# geometric magnification ζ for the implied radial slope p = 3/α
zeta = magnification_factor(3.0 / alpha)     # ζ(1.5) = √2 ≈ 1.414
print(f"f_dense = {f_dense:.3f}, ζ = {zeta:.3f}")
```

`f_dense_bm19_full` evaluates {eq}`f-dense` for the {cite:t}`Burkhart2018` framework by splitting
the integral into a lognormal body (an `erf` term) and a power-law tail
($M_{\mathrm{PL}} = C\,e^{(1-\alpha)s_t}/(\alpha-1)$, valid for $\alpha > 1$) and returning the
ratio $M_{\mathrm{PL}}/(M_{\mathrm{LN}} + M_{\mathrm{PL}})$. For typical Galactic-cloud parameters
($\mathcal{M} = 10$, $b = 0.4$, $\alpha = 2$), $f_{\mathrm{dense}} \sim 0.05$–$0.15$ — a few percent
of cloud mass is in the dense star-forming tail at any instant.

### Connecting to the magnification factor ζ

For a *spatially uniform* density (no PDF spread), the FDF integral in {eq}`sfr-pdf-fdf` reduces
trivially to the "top-hat" reference and the cloud SFR is that of a uniform cloud. For a non-uniform
cloud, the integral is *larger* than the top-hat reference — by exactly the magnification factor

```{math}
:label: zeta-pdf-fdf
\zeta \;=\; \int \biggl(\frac{\rho}{\langle\rho\rangle}\biggr)^{\!3/2}\,p_V(\rho)\,\mathrm{d}\rho .
```

This is the same dimensionless ratio that [](magnification-factor.md) evaluates **geometrically**
for a power-law radial profile $\rho(r) \propto r^{-p}$; this chapter's PDF-based formulation is its
**density-space dual**. They produce the same number for the same physical cloud.

### Which formulation when

The two formulations apply to different observational situations:

```{list-table}
:header-rows: 1

* - Formulation
  - Best when you have…
  - Output
* - **Radial-profile** ([](magnification-factor.md))
  - …a fitted radial profile $\rho(r) \propto r^{-p}$
  - $\zeta(p)$ analytic, fast
* - **PDF-based** (this chapter)
  - …a density-PDF observation (e.g. column-density PDF)
  - Numerical integral; consumes lognormal+power-law parameters
* - **Direct 3D** ([direct 3D ζ](#direct-3d))
  - …a simulation snapshot or detailed observation
  - Direct sum over voxels; no parametric assumption
```

For inference: the radial-profile formulation is what most observational papers report (clouds are
characterised by $r_{\mathrm{eff}}$ and $\rho(r)$). The PDF-based formulation is what cloud
simulations output. The 3D formulation is what cosmological simulations output when probed at the
cloud scale.

progenax's [](bm19.md) forward chain uses the *PDF-based* formulation because the
{cite:t}`Burkhart2018,BurkhartMocz2019` framework is parameterised in PDF space.

---

(alpha-p)=
## The α↔p mapping (stated once, used everywhere)

The PDF tail-slope $\alpha$ and the radial-profile slope $p$ are not independent. Under spherical
symmetry and a power-law correspondence between volume and density
{cite:p}`Kritsuk2011,FederrathKlessen2012`, the canonical relation is

```{math}
:label: alpha-p-mapping
\boxed{\;p \;=\; \frac{3}{\alpha}\;}
```

This is the single statement the rest of the gravoturbulence section refers back to: [](bm19.md)
applies it in step 5 of the forward chain, and [](magnification-factor.md) consumes the resulting
$p$ to compute $\zeta(p)$. For the {cite:t}`BurkhartMocz2019` canonical α window $[\alpha_{\mathrm{sat}},
\alpha_0] = [1.5, 3.0]$, the corresponding $p$ window is $[1.0, 2.0]$ — from "marginally collapsing"
to "singular isothermal" radial profiles. `tests/experimental/unit/test_bm19.py` verifies the
$p = 3/\alpha$ mapping (`pdf_slope_to_radial`).

The mapping is what lets the BM19 forward chain hand off to the magnification-factor calculation
seamlessly: BM19 infers $\alpha$ from cloud observations, {eq}`alpha-p-mapping` converts it to a
radial-profile slope, and $\zeta(p)$ gives the geometric SFR boost.

---

## Domain of validity

1. **Supersonic turbulence required** — at low Mach the lognormal variance shrinks below $\sim 1$
   and the lognormal approximation breaks down. `gravoturb`'s parameterisation works for
   $\mathcal{M} \gtrsim 3$.
2. **Self-gravity required for the power-law tail** — at very low density (diffuse warm neutral
   medium) there is no gravitating tail and the PDF is purely lognormal. The {cite:t}`Burkhart2018`
   framework assumes a power-law tail exists.
3. **Free-fall regime for the kernel** — the $\rho^{3/2}$ kernel assumes that local gravitational
   collapse on a free-fall timescale dominates over turbulent or thermal stabilisation. At very low
   density other timescales matter; at very high density (proto-stellar cores) hydrostatic-
   equilibrium effects matter. The isothermal $t_{\mathrm{ff}}$ of {eq}`tff` ignores
   thermal-pressure support; the standard treatment lumps this into $\varepsilon_{\mathrm{ff,int}}$.
4. **No magnetic-field or feedback support** — magnetic fields slow collapse via ambipolar
   diffusion ({cite:t}`FederrathKlessen2012` give a magnetic correction; `gravoturb` does not
   include it), and the kernel assumes star formation does not back-react on the gas. For clouds
   older than a few $t_{\mathrm{ff}}$, feedback becomes important and the kernel under-predicts SFR.
5. **Spherical/cylindrical symmetry** assumed in the α↔p mapping; for highly filamentary or
   sheet-like clouds the mapping is approximate.
6. **Single-cloud, static description.** The PDF describes one cloud at a time-averaged instant;
   multi-cloud line-of-sight superposition requires a separate treatment, and the PDF parameters
   themselves evolve on the cloud's free-fall timescale.
7. **3-D field realisation.** When a sampled 3-D field is built from the PDF
   (`gravoturb.realization.pipeline.build_fdf_field`), the one-point statistics are imposed by a
   **rank / empirical-CDF copula** (Gaussian anamorphosis), which reproduces the dense-tail mass
   fraction $f_{\mathrm{dense}}$ at any power-spectrum slope $\beta$. For a very extreme threshold
   $s_t$, where the tail count-probability $1 - F_V(s_t) \lesssim 1/N_{\mathrm{grid}}^3$, the tail
   is genuinely unresolved at that grid and a warning is emitted (see the [α wall in the inference
   chapter](inference.md)).

## Implementation, validation & references

- **In code:** `src/experimental/gravoturb/theory/density_pdf.py`
  (`sigma_s_squared`, `transition_density`, `f_dense_bm19_full`,
  `pdf_slope_to_radial`) and
  `src/experimental/gravoturb/theory/density_cdf.py` (`bm19_volume_pdf`);
  3-D field realisation is
  `src/experimental/gravoturb/realization/pipeline.py`. This experimental
  subsystem is repo-only with no generated website API page; the
  module reference is the package source and its `VALIDATION_SUMMARY.md`.
- **Validated in:** [gravoturbulent PP20](../../50-validation/gravoturbulent-pp20.md);
  the $p = 3/\alpha$ mapping is pinned by
  `tests/experimental/unit/test_bm19.py`.
- **Primary sources:** the lognormal core is {cite:t}`FederrathKlessen2012`
  (the $\sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)$ derivation); the
  power-law tail and the analytic α↔p mapping are {cite:t}`Kritsuk2011`;
  the transition-density matching and PDF+FDF combination are
  {cite:t}`Burkhart2018`; {cite:t}`TanKrumholzMcKee2006` is the earlier
  single-mean-density framework; observational verification of the form
  and dense-gas threshold is {cite:t}`Kainulainen2014`. Full notes in
  the [bibliography](../../99-bibliography/per-paper/federrath-klessen-2012.md).
  The full forward chain is [](bm19.md).
