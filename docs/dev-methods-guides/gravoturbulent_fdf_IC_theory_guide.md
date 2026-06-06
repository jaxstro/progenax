# A Gravoturbulent Framework for Cluster Substructure

## Physically-Grounded Initial Conditions via BM19 + FDF + Parmentier

---

## Purpose

This document develops a physically-motivated framework for generating star cluster initial conditions with environment-dependent substructure. We adopt the gravoturbulent density-PDF framework of Burkhart & Mocz (2019, hereafter BM19) and extend it in three linked pieces:

1. **From cloud-scale properties to a 1D density PDF**, which predicts the mass fraction of self-gravitating gas ($f_{\rm dense}$)

2. **From that 1D PDF to a 3D turbulent density field** with a geometrically-defined dense tail that sets the spatial clustering of stars and their Cartwright-Whitworth Q parameter

3. **From the geometry of the dense gas to a dense-gas SFR** via the Parmentier & Pasquali (2020) magnification factor $\zeta$

**Critical clarification:** We do *not* explicitly construct a piecewise lognormal+powerlaw PDF on a grid. Instead, we:

- **Realize a lognormal turbulent field** with the BM19-prescribed variance $\sigma_s$ and power spectrum slope $\beta$
- **Identify the self-gravitating tail** by direct thresholding on $s = \ln(\rho/\rho_0)$ at the BM19 transition density $s_t$
- **Sample stars** with fraction $f_{\rm sub}$ drawn from the dense tail

The BM19 framework tells us *how much* mass should reside in self-gravitating gas. The Fractal Density Field (FDF) layer tells us *where* that gas sits in 3D and what the resulting stellar clustering looks like. The Parmentier & Pasquali (2020) framework then tells us *how efficiently* that dense gas forms stars.

---

## What This Is

A well-posed, testable IC model rooted in gravoturbulent star-formation theory (BM19), realized as a two-layer turbulent + tail sampling scheme, with dense-gas SFR interpretation via the Parmentier magnification factor.

## What This Is Not

A claim that we derive cluster morphology from first principles. BM19 predicts the *mass fraction* of self-gravitating gas; the mapping to stellar Q requires 3D realizations and numerical calibration.

---

## Key Result

$$
\boxed{f_{\rm sub} = \eta_{\rm survive} \cdot f_{\rm dense}(\sigma_s, \alpha)}
$$

where:
- $\sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)$ — PDF width from turbulence (BM19 Eq. 1)
- $\alpha$ — powerlaw slope encoding collapse/feedback state (typically 1.5–2.5)
- $s_t = (\alpha - 1/2)\sigma_s^2$ — transition density (BM19 Eq. 2; **derived**, not assumed)
- $f_{\rm dense}$ — mass fraction above $s_t$ (BM19 Eq. 19-20)
- $\eta_{\rm survive}$ — feedback survival efficiency

**Crucially:** $f_{\rm sub}$ is **not another free parameter**. It is directly determined by the BM19 self-gravitating fraction and a survival factor—not chosen ad hoc.

In the **lognormal limit** ($\alpha \gtrsim 2.5$):

$$
f_{\rm dense} \approx \frac{1}{2} \operatorname{erfc}\left(\frac{s_t - \sigma_s^2/2}{\sqrt{2}\sigma_s}\right)
$$

*Note: This is the limiting form. In practice, we use the full piecewise LN+PL integral (§4.2) which is valid for all $\alpha > 1$.*

---

## Deliverable

> **What this framework provides:**
> 
> | **Inputs** | **Outputs** |
> |------------|-------------|
> | $\Sigma$ (surface density) | $f_{\rm sub}$ (substructure fraction) |
> | $\mathcal{M}$ (Mach number) | $Q$ (Cartwright-Whitworth parameter) |
> | $b$ (turbulent driving mode) | $\zeta_{\rm FDF}$ (magnification factor) |
> | $\alpha$ (collapse/feedback state) | Star positions in 3D |
> | $\eta_{\rm survive}$ (survival efficiency) | |
> | Cloud size $R$ | |
> 
> The key advance over traditional IC generators: **$Q$ is predicted from physics, not chosen as an arbitrary fractal dimension $D$.**
> 
> The mapping $f_{\rm sub} \to Q$ must be calibrated from FDF realizations (§12), but once calibrated, cluster ICs inherit their substructure from cloud-scale observables.

---

## Parameter Budget

| **Free "environment" parameters** | **Derived quantities** |
|-----------------------------------|------------------------|
| $\Sigma$ (surface density) | $\sigma_s$ (PDF width) |
| $R$ (cloud radius) | $s_t$ (transition density) |
| $\mathcal{M}$ (Mach number) | $f_{\rm dense}$ (self-gravitating fraction) |
| $b$ (turbulent driving mode) | $f_{\rm sub}$ (substructure fraction) |
| $\alpha$ (collapse/feedback state) | $\beta$ (power spectrum slope) |
| $\eta_{\rm survive}$ (survival efficiency) | $Q$ (Cartwright-Whitworth parameter) |
| | $p_{\rm eff}$ (effective density slope) |
| | $\zeta_{\rm FDF}$ (magnification factor) |

**In traditional fractal ICs, you pick $(D, Q)$. Here we pick $(\Sigma, R, \mathcal{M}, b, \alpha, \eta_{\rm survive})$ and $Q$ is emergent.**

> **Implementation note:** In v9.0, $\Sigma$ and $R$ are bookkeeping parameters that set physical units and cloud mass/radius. The BM19 core (`bm19_pipeline`) depends only on $(\mathcal{M}, b, \alpha, \eta_{\rm survive})$. Future extensions may link $\Sigma$ to typical $\alpha$ or $\mathcal{M}$ via observed $\Sigma$–SFR relations, but that is not hard-wired in the current implementation.

---

## Table of Contents

1. [The Physical Picture](#1-the-physical-picture)

**Part I: 1D Gravoturbulent PDF Theory (BM19)**

2. [The Piecewise Lognormal+Powerlaw PDF](#2-the-piecewise-lognormalpowerlaw-pdf)
3. [PDF Parameters from Cloud Physics](#3-pdf-parameters-from-cloud-physics)
4. [Transition Density and Self-Gravitating Fraction](#4-transition-density-and-self-gravitating-fraction)

**Part II: 3D Realization with FDF**

5. [Two-Layer Architecture Overview](#5-two-layer-architecture-overview)
6. [Lognormal Turbulent Fields](#6-lognormal-turbulent-fields)
7. [Geometric Tail Selection via Direct s-Thresholding](#7-geometric-tail-selection-via-direct-s-thresholding)
8. [From f_sub to Star Positions and Q](#8-from-f_sub-to-star-positions-and-q)

**Part III: Dense Gas SFR and Parmentier ζ**

9. [The Magnification Factor Framework](#9-the-magnification-factor-framework)
10. [Measuring p_eff and ζ from FDF Realizations](#10-measuring-p_eff-and-ζ-from-fdf-realizations)
11. [Unified Picture: BM19 + FDF + PP20](#11-unified-picture-bm19--fdf--pp20)

**Calibration and Strategy**

12. [Calibration and Validation](#12-calibration-and-validation)
13. [Publication Strategy](#13-publication-strategy)
14. [References](#14-references)

**Appendices**

- [Appendix A: Reference Implementation](#appendix-a-reference-implementation-in-progenax)
- [Appendix B: Classical PN11/FK12 Framework (Historical)](#appendix-b-classical-pn11fk12-framework-historical)
- [Appendix C: The Q Parameter in Detail](#appendix-c-the-q-parameter-in-detail)
- [Appendix D: Alternative—Local Virial Parameter Mask](#appendix-d-alternativelocal-virial-parameter-mask)

---

## 1. The Physical Picture

### 1.1 Why This Matters

When we simulate the dynamical evolution of star clusters—whether to understand mass segregation, binary populations, or long-term survival—the results depend critically on the **initial spatial distribution** of stars. A cluster born with stars smoothly distributed will evolve very differently from one born with stars clumped into dense subclusters separated by voids.

Yet for decades, initial condition generators have parameterized this substructure using an arbitrary "fractal dimension" $D$ with no physical basis. Users pick $D = 2.0$ or $D = 1.6$ based on intuition or empirical matching. There's been no principled way to connect $D$ to the **birth environment**—the properties of the molecular cloud from which the cluster formed.

This document addresses that gap by building a three-layer framework:

1. **BM19 (1D theory):** Predicts *how much* gas mass is self-gravitating from cloud properties
2. **FDF (3D realization):** Places that mass *where* it belongs spatially, producing measurable substructure
3. **Parmentier (SFR interpretation):** Predicts *how efficiently* that spatially-distributed dense gas forms stars

### 1.2 The Three-Layer Causal Chain

The theory naturally splits into three layers, each with distinct inputs, frameworks, and outputs:

```
═══════════════════════════════════════════════════════════════════
INPUTS: Cloud properties (Σ, M, b, ℳ, evolutionary state)
═══════════════════════════════════════════════════════════════════
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PART I: 1D Gravoturbulent PDF Theory (BM19)                    │
│                                                                 │
│  σ_s² = ln(1 + b²ℳ²)           → PDF width (turbulence)        │
│  α                              → powerlaw slope (evolution)    │
│  s_t = (α - 1/2)σ_s²           → transition density (derived)  │
│  f_dense(σ_s, α)               → self-gravitating gas fraction │
│                                                                 │
│  OUTPUT: "What fraction of gas mass is self-gravitating?"       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PART II: 3D Realization with FDF                               │
│                                                                 │
│  σ_s, β  → 3D lognormal turbulent density field ρ(x)           │
│  s(x) = ln(ρ/ρ₀)  → log-density contrast field                 │
│  s(x) > s_t  → BM19-consistent tail selection (direct)         │
│  f_sub = η_survive × f_dense  → star sampling split            │
│  FDF sampling  → Q(f_sub; σ_s, β, profile)                     │
│                                                                 │
│  OUTPUT: "Where is the dense gas, and how clumpy are stars?"    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  PART III: Dense Gas SFR (Parmentier)                           │
│                                                                 │
│  M_dg, p_eff  ← measured from ρ(x)                             │
│  ζ(p_eff)  → SFR magnification factor                          │
│  SFR_dg = ζ × ε_ff,int × M_dg / ⟨t_ff,dg⟩                      │
│                                                                 │
│  OUTPUT: "How efficiently does this geometry form stars?"       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
═══════════════════════════════════════════════════════════════════
OBSERVABLES: Q (substructure), SFR/M_dg (dense gas efficiency)
═══════════════════════════════════════════════════════════════════
```

### 1.3 Quantifying Substructure: The Q Parameter

To quantify cluster substructure, we use the **Cartwright-Whitworth Q parameter** (Cartwright & Whitworth 2004):

$$
Q = \frac{\bar{m}}{\bar{s}}
$$

where $\bar{m}$ is the normalized mean MST edge length and $\bar{s}$ is the normalized mean pairwise separation.

| Q value | Structure | Physical interpretation |
|---------|-----------|------------------------|
| Q ≈ 0.8 | Uniform/smooth | Volume-filling; no significant substructure |
| Q ≈ 0.6–0.7 | Mild substructure | Some clumping but not dominant |
| Q ≈ 0.45–0.55 | Strong substructure | Multiple well-separated subclusters |
| Q < 0.4 | Extreme substructure | Highly fragmented |

Our goal: replace arbitrary fractal dimension $D$ with physically-motivated $f_{\rm sub}$, yielding Q as an emergent prediction.

*For detailed derivations of Q, see Appendix C.*

### 1.4 What We're Trying to Predict

We predict **two complementary observables** from cloud properties:

1. **Substructure (Q):** How clumpy is the stellar distribution?
2. **Dense-gas SFR:** How efficiently does the dense gas form stars?

Both emerge from the same underlying turbulent density field, realized via FDF.

### 1.5 Scope and Limitations

This is a **one-zone, ensemble-averaged model** for predicting typical substructure given cloud properties.

**What the model captures:**
- The trend: high-$\Sigma$ → high $f_{\rm sub}$ → more substructure (lower Q)
- Order-of-magnitude estimates for $f_{\rm dense}$ from $(\mathcal{M}, b, \alpha)$
- A physically-motivated framework for choosing IC parameters
- Dense-gas SFR predictions consistent with Parmentier's observed bands

**What the model does not capture:**
- Local variations within a single cloud
- Time evolution during star formation
- Detailed magnetic field effects (BM19's $\sigma_s$–$\mathcal{M}$ relation can be modified by B fields; our implementation implicitly folds magnetization into $b$, but does not explicitly track Alfvén Mach number $\mathcal{M}_A$)
- Spatial topology (number of clumps, filamentarity)—these are not analytically predictable from the 1D PDF alone; they emerge numerically in FDF realizations and are then characterized (Q, $p_{\rm eff}$, etc.), not written in closed form

---

# Part I: 1D Gravoturbulent PDF Theory (BM19)

*This is the top layer of the causal chain (§1.2). Part I takes cloud properties as inputs and outputs the self-gravitating gas fraction $f_{\rm dense}$.*

## 2. The Piecewise Lognormal+Powerlaw PDF

### 2.1 The BM19 Framework

In supersonic isothermal turbulence, gas density fluctuates due to shocks and rarefactions. Burkhart & Mocz (2019) describe the probability distribution as a **piecewise PDF**:

$$
p(s) = \begin{cases}
p_{\rm LN}(s) & s < s_t \\[0.5em]
p_{\rm PL}(s) \propto e^{-\alpha s} & s \geq s_t
\end{cases}
$$

where $s \equiv \ln(\rho/\rho_0)$ is the log-density contrast.

**Physical interpretation:**
- **Low densities ($s < s_t$):** Turbulence dominates. The PDF is lognormal, set by supersonic shocks.
- **High densities ($s \geq s_t$):** Gravity dominates. The PDF develops a powerlaw tail from collapsing cores.

The transition density $s_t$ marks where self-gravity overcomes turbulent support.

### 2.2 The Lognormal Component

At low densities, the PDF is lognormal:

$$
p_{\rm LN}(s) = \frac{1}{\sqrt{2\pi}\sigma_s} \exp\left[-\frac{(s - s_0)^2}{2\sigma_s^2}\right]
$$

**Mass conservation** requires $s_0 = -\sigma_s^2/2$. This negative mean reflects that most *volume* is underdense, while *mass* concentrates in the high-density tail.

### 2.3 The Powerlaw Component

At high densities, gravitational collapse creates a powerlaw tail:

$$
p_{\rm PL}(s) \propto e^{-\alpha s}
$$

The slope $\alpha$ encodes the **collapse geometry**. For a density profile $\rho \propto r^{-\kappa}$, the PDF slope is $\alpha = 3/\kappa$:

| $\alpha$ | $\kappa$ | Profile | Physical state |
|----------|----------|---------|----------------|
| 1.5 | 2.0 | $\rho \propto r^{-2}$ | Isothermal sphere |
| 2.0 | 1.5 | $\rho \propto r^{-1.5}$ | Intermediate collapse |
| 3.0 | 1.0 | $\rho \propto r^{-1}$ | Early/shallow collapse |

### 2.4 Why This Framework Matters for FDF

**We do not explicitly construct this piecewise PDF.** Instead:

- BM19 tells us the *statistics* we're targeting: a lognormal body with width $\sigma_s$ and a self-gravitating tail containing mass fraction $f_{\rm dense}$
- FDF *realizes* these statistics as a 3D turbulent field plus a geometrically-selected tail

The 1D PDF framework provides the **target**; FDF provides the **realization**.

---

## 3. PDF Parameters from Cloud Physics

### 3.1 The PDF Width (BM19 Eq. 1)

The lognormal width is set by turbulence:

$$
\boxed{\sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)}
$$

| Symbol | Name | Typical Range | Physical Meaning |
|--------|------|---------------|------------------|
| $\mathcal{M}$ | Mach number | 5–30 | $\sigma_v / c_s$; turbulent velocity / sound speed |
| $b$ | Driving parameter | 0.3–1.0 | Nature of turbulent driving |
| $\sigma_s$ | PDF width | 1–3 | Amplitude of density fluctuations |

### 3.2 The Driving Parameter b

| Driving Mode | $b$ Value | Physical Origin |
|--------------|-----------|-----------------|
| **Solenoidal** (divergence-free) | $b \approx 1/3$ | Shear, vorticity |
| **Mixed** (natural) | $b \approx 0.4$ | Typical GMC |
| **Compressive** (curl-free) | $b \approx 1$ | SNe, H II regions, collisions |

**Environmental variation:**

| Environment | Expected $b$ | Physical reason |
|-------------|--------------|-----------------|
| Solar neighborhood GMCs | 0.3–0.5 | Mix of shear, spiral compression |
| Spiral arms | 0.4–0.6 | Cloud-cloud collisions |
| Central Molecular Zone | 0.5–0.7 | Strong tidal shear + bar inflows |
| Starburst galaxies | 0.6–0.9 | SNe-driven, highly compressive |

### 3.3 The Powerlaw Slope α: Evolutionary State

The powerlaw slope encodes the cloud's **evolutionary state**:

| α value | Physical state | Cloud age | Interpretation |
|---------|----------------|-----------|----------------|
| $\alpha \approx 3$ | Early collapse | $< 0.5\,t_{\rm ff}$ | Powerlaw just forming |
| $\alpha \approx 2$ | Active collapse | $\sim 1\,t_{\rm ff}$ | Gravity winning |
| $\alpha \approx 1.5$ | Saturated collapse | $\sim 1$–$2\,t_{\rm ff}$ | Isothermal cores |
| $\alpha > 2$ | Feedback-regulated | Late stages | Gas expelled |

**Key physics:** Without feedback, $\alpha$ evolves from ~3 to ~1.5 over one freefall time. Stellar feedback *steepens* the powerlaw by expelling dense gas.

### 3.4 Numerical Examples

| Environment | $\mathcal{M}$ | $b$ | $\sigma_s^2$ | $\sigma_s$ |
|-------------|---------------|-----|--------------|------------|
| Quiescent GMC | 5 | 0.4 | 1.61 | 1.27 |
| Typical GMC | 10 | 0.4 | 2.83 | 1.68 |
| Turbulent GMC | 15 | 0.4 | 3.61 | 1.90 |
| Starburst region | 30 | 0.5 | 5.42 | 2.33 |

*Values computed from $\sigma_s^2 = \ln(1 + b^2\mathcal{M}^2)$.*

---

## 4. Transition Density and Self-Gravitating Fraction

### 4.1 The Transition Density (BM19 Eq. 2)

A remarkable feature of BM19: the transition density $s_t$ is **not** a free parameter—it emerges from requiring PDF continuity:

$$
\boxed{s_t = \left(\alpha - \frac{1}{2}\right) \sigma_s^2}
$$

**Physical meaning:** $s_t$ corresponds to where the Jeans length equals the sonic scale—the post-shock critical density for gravitational instability.

**Why this matters:** Unlike the classical $s_{\rm crit}$ (which requires choosing $\phi_x$; see Appendix B), $s_t$ is fully determined by observable parameters.

### 4.2 The Self-Gravitating Gas Fraction (BM19 Eq. 19-20)

The mass fraction above $s_t$:

$$
\boxed{f_{\rm dense} = \frac{\int_{s_t}^{\infty} e^s p(s)\,ds}{\int_{-\infty}^{\infty} e^s p(s)\,ds}}
$$

Here $p(s)$ is the **volume-weighted** log-density PDF. The factor $e^s$ converts this into a mass-weighted integral.

We implement the **full piecewise integral** over the lognormal+powerlaw PDF:

**Lognormal part** ($s < s_t$):

The key identity: multiplying the volume-weighted lognormal $p_{\rm LN}(s)$ by $e^s$ yields another Gaussian, shifted to mean $s_0 + \sigma_s^2 = \sigma_s^2/2$ (using $s_0 = -\sigma_s^2/2$). This shifted Gaussian is already normalized to integrate to 1, so:

$$
M_{\rm LN}(-\infty, s_t) = \int_{-\infty}^{s_t} e^s \, p_{\rm LN}(s) \, ds = \Phi\left(\frac{s_t - \sigma_s^2/2}{\sigma_s}\right)
$$

where $\Phi$ is the standard Gaussian CDF. Equivalently:

$$
M_{\rm LN}(-\infty, s_t) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{s_t - \sigma_s^2/2}{\sqrt{2}\sigma_s}\right)\right]
$$

**Important:** There is no extra $e^{\sigma_s^2/2}$ prefactor here. The choice $s_0 = -\sigma_s^2/2$ ensures that $\int_{-\infty}^{\infty} e^s p_{\rm LN}(s)\,ds = 1$ (mass conservation).

**Powerlaw part** ($s \geq s_t$):
$$
M_{\rm PL} = \int_{s_t}^{\infty} e^s \, A e^{-\alpha s} \, ds = \frac{A}{\alpha - 1} e^{(1-\alpha)s_t} \quad \text{for } \alpha > 1
$$

where $A = p_{\rm LN}(s_t) \cdot e^{\alpha s_t}$ ensures PDF continuity at the transition.

The self-gravitating fraction is then:
$$
f_{\rm dense} = \frac{M_{\rm PL}}{M_{\rm LN}(-\infty, s_t) + M_{\rm PL}}
$$

**Compact closed form (for paper):**

$$
\boxed{f_{\rm dense} = \frac{1}{1 + \frac{(\alpha - 1)}{A}\, e^{(\alpha-1)s_t} \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{s_t - \sigma_s^2/2}{\sqrt{2}\sigma_s}\right)\right]}}
$$

where $A = p_{\rm LN}(s_t) \cdot e^{\alpha s_t}$ and $p_{\rm LN}(s_t) = \frac{1}{\sqrt{2\pi}\sigma_s}\exp\left[-\frac{(s_t + \sigma_s^2/2)^2}{2\sigma_s^2}\right]$.

This is the single expression we actually implement. The inputs are $(\sigma_s, \alpha)$, with $s_t = (\alpha - 1/2)\sigma_s^2$.

**Implementation note:** The JAX implementation (`f_dense_bm19_full`) computes this full integral analytically. **This is the default in the pipeline.** A lognormal-limit approximation (`f_dense_lognormal_limit`) is retained for comparison but is **not** used in production.

**Why renormalization matters:** The denominator $M_{\rm LN}(-\infty, s_t) + M_{\rm PL}$ is what makes the piecewise PDF self-consistent. By attaching a powerlaw tail at $s_t$, we effectively remove lognormal mass above $s_t$ and replace it with the powerlaw. The renormalization ensures the total mass-weighted PDF still integrates to 1.

**Unit test requirements:** 
- For $\alpha \to \infty$, `f_dense_bm19_full` should match `f_dense_lognormal_limit`
- For pure lognormal, total mass integral should equal 1 (not $e^{\sigma_s^2/2}$)

### 4.3 The Anti-Correlation with Mach Number

BM19's key prediction (validated against M51 PAWS data):

> **SFE is weakly anti-correlated with Mach number** for actively star-forming clouds.

This is counterintuitive: naively, higher $\mathcal{M}$ → wider PDF → more high-density gas.

**The resolution:** Higher $\mathcal{M}$ widens the PDF but *also* pushes $s_t$ higher (since $s_t \propto \sigma_s^2$). The net effect: mass fraction in the self-gravitating tail *decreases slightly* with increasing $\mathcal{M}$.

**Numerical illustration** (for $b = 0.4$, $\alpha = 2.0$):

| $\mathcal{M}$ | $\sigma_s$ | $s_t$ | $f_{\rm dense}$ |
|---------------|------------|-------|-----------------|
| 5 | 1.27 | 2.41 | 0.135 |
| 10 | 1.68 | 4.25 | 0.057 |
| 15 | 1.90 | 5.42 | 0.034 |
| 20 | 2.04 | 6.26 | 0.024 |
| 30 | 2.23 | 7.47 | 0.015 |

*Values computed from: $\sigma_s^2 = \ln(1 + b^2\mathcal{M}^2)$, $s_t = (\alpha - 1/2)\sigma_s^2$, and the full BM19 piecewise integral.*

The anti-correlation is weak but systematic: doubling $\mathcal{M}$ from 10 to 20 reduces $f_{\rm dense}$ by ~60%.

*Caveat: In real systems, $\alpha$ is not independent of $\mathcal{M}$ and environment. Variations in $\alpha$ and $\Sigma$ can dominate over this modest trend with Mach number. The table above illustrates the theoretical tendency at fixed $\alpha$, not a universal law.*

### 4.4 From f_dense to f_sub

Not all self-gravitating gas forms stars that survive in the cluster:

$$
\boxed{f_{\rm sub} = \eta_{\rm survive} \cdot f_{\rm dense}}
$$

where $\eta_{\rm survive} \in [0, 1]$ absorbs feedback effects, gas expulsion, and infant mortality.

| Environment | Expected $\eta_{\rm survive}$ | Reasoning |
|-------------|-------------------------------|-----------|
| Low-mass OC | 0.3–0.5 | Weak feedback; gas dispersal dominant |
| Moderate OC | 0.5–0.7 | Balanced feedback |
| High-mass YMC | 0.7–0.9 | Deep potential; survives feedback |

### 4.5 Summary: What Part I Provides

Part I (BM19) answers: **"Given cloud properties, what fraction of gas mass is self-gravitating?"**

$$
(\mathcal{M}, b, \alpha) \xrightarrow{\text{BM19}} (\sigma_s, s_t, f_{\rm dense}) \xrightarrow{\eta_{\rm survive}} f_{\rm sub}
$$

But this is a *scalar*—it tells us nothing about *where* that mass is located or how clumpy the resulting stars will be. That requires Part II.

---

# Part II: 3D Realization with FDF

*This is the middle layer of the causal chain (§1.2). Part II takes the 1D statistics from BM19 and realizes them as a 3D density field, outputting star positions and the Q parameter.*

## 5. Two-Layer Architecture Overview

### 5.1 The Problem: 1D Statistics Don't Give 3D Structure

BM19 provides $f_{\rm dense}$—a single number. But to generate cluster ICs, we need:

- **Where** the dense clumps are located
- **How many** clumps there are
- **How separated** they are (which determines Q)

A scalar can't encode topology. We need to **realize** the BM19 statistics in 3D space.

### 5.2 The FDF Solution: Two Layers

Our Fractal Density Field (FDF) implementation uses two conceptual layers:

| Layer | What it does | Key parameters |
|-------|--------------|----------------|
| **FractalDensityLayer** | Generates 3D turbulent density field | $\sigma_s$, $\beta$, base profile |
| **TailSubstructureLayer** | Identifies dense tail geometrically; samples stars | $f_{\rm sub}$, smoothing scale |

**Terminology note:** We use "Fractal" in the sense of *scale-free, turbulence-like structure*, not in the strict Hausdorff-dimension sense. Our fields are built as lognormal Gaussian random fields with a prescribed power spectrum $P(k) \propto k^{-\beta}$, which produces correlated, self-similar structure across scales. This is distinct from the Goodwin & Whitworth (2004) recursive fractal algorithm that uses an explicit fractal dimension $D$. To draw the comparison explicitly: **Our GRF-based FDF produces a statistical fractal-like field (lognormal + correlated), whereas Goodwin & Whitworth's "fractal tree" generator starts from an explicit fractal dimension $D$; our "$D$" is replaced by $(\sigma_s, \beta, f_{\rm sub})$.**

The key advantage: rather than choosing $D$ arbitrarily, we derive the field parameters from turbulence physics (BM19).

The two-layer implementation (see §2.4 for the core principle):

1. The **FractalDensityLayer** produces a field whose 1-point statistics are approximately lognormal (matching BM19's turbulent component)
2. The **TailSubstructureLayer** identifies high-density regions geometrically via direct $s > s_t$ thresholding (standing in for BM19's self-gravitating powerlaw tail)

### 5.3 Why This Works

The BM19 framework says: "A fraction $f_{\rm dense}$ of the mass lies above the self-gravitating threshold."

Our geometric interpretation: "A fraction $f_{\rm dense}$ of the mass lies in locally overdense regions that would collapse."

These are equivalent statements—one in PDF space, one in real space.

---

## 6. Lognormal Turbulent Fields

### 6.1 What FractalDensityLayer Actually Builds

The first layer generates a 3D density field $\rho(\mathbf{x})$ with:

1. **One-point statistics:** Approximately lognormal with variance $\sigma_s^2$ matching BM19
2. **Two-point statistics:** Power spectrum $P(k) \propto k^{-\beta}$ encoding spatial correlations
3. **Large-scale structure:** Optional base profile (uniform, Plummer, etc.)

### 6.2 Construction Algorithm

**Step 1:** Generate Gaussian random field $\delta(\mathbf{x})$ with power spectrum $P(k) \propto k^{-\beta}$

**Step 2:** Apply lognormal transform:
$$
\rho_{\rm turb}(\mathbf{x}) = \exp\left(\delta(\mathbf{x}) - \frac{\sigma_s^2}{2}\right)
$$

The shift $-\sigma_s^2/2$ ensures $\langle \rho_{\rm turb} \rangle = \rho_0$.

**Step 3:** Impose base profile:
$$
\rho(\mathbf{x}) = \mathcal{N} \cdot \rho_{\rm base}(r) \cdot \left[(1-\lambda) + \lambda \, \rho_{\rm turb}(\mathbf{x})\right]
$$

where $\mathcal{N}$ normalizes total mass and $\lambda$ controls turbulent amplitude.

### 6.3 The Power Spectrum Slope β

The power spectrum slope $\beta$ controls spatial correlations. Rather than treating it as a free parameter, we derive it from turbulence theory:

**Federrath+ 2010 scaling:**

| Regime | $\beta$ | Physical basis |
|--------|---------|----------------|
| Subsonic ($\mathcal{M} \lesssim 1$) | $\approx 11/3$ | Kolmogorov cascade |
| Supersonic ($\mathcal{M} \gg 1$) | $\approx 4$ | Burgers/shock-dominated |

The transition occurs smoothly around $\mathcal{M} \sim 1$–$2$:

$$
\beta(\mathcal{M}, b) \approx \frac{11}{3} + \left(4 - \frac{11}{3}\right) \cdot \frac{1}{2}\left[1 + \tanh\left(\frac{b\mathcal{M} - 1.5}{1.0}\right)\right]
$$

**Implementation:** The `power_spectrum_slope()` function computes $\beta$ from $(\mathcal{M}, b)$, and `bm19_pipeline()` includes it in the output. This means **$\beta$ is no longer a free parameter**—it's determined by the same turbulent Mach number that sets $\sigma_s$.

| Environment | $\mathcal{M}$ | $b$ | $\sigma_s$ | $\beta$ |
|-------------|---------------|-----|------------|---------|
| Quiescent GMC | 5 | 0.4 | 1.27 | 3.91 |
| Typical GMC | 10 | 0.4 | 1.68 | 4.00 |
| Turbulent GMC | 20 | 0.4 | 2.04 | 4.00 |
| Starburst | 30 | 0.5 | 2.33 | 4.00 |

*Values computed from $\sigma_s^2 = \ln(1 + b^2\mathcal{M}^2)$ and the Federrath+ scaling above.*

*Note: Other simulation suites give slightly different $\beta(\mathcal{M})$ scalings; our choice is meant as a physically motivated default that can be replaced if desired. The key constraint is consistency: whatever $\beta(\mathcal{M})$ relation is used should be held fixed across calibration and inference.*

### 6.4 What This Layer Provides

The FractalDensityLayer produces a field $\rho(\mathbf{x})$ that:

- Has the correct 1-point PDF width ($\sigma_s$ from BM19)
- Has realistic spatial correlations (power spectrum from turbulence theory)
- Can include a global density gradient (base profile)

This is the **lognormal body** of the BM19 framework, realized in 3D.

---

## 7. Geometric Tail Selection via Direct $s$-Thresholding

### 7.1 The Key Idea

BM19 defines the self-gravitating tail as mass above threshold $s_t$ in the 1D PDF. In our 3D implementation, we identify this tail **directly** by thresholding on the log-density contrast:

$$
s(\mathbf{x}) = \ln\left(\frac{\rho(\mathbf{x})}{\rho_0}\right)
$$

**BM19-consistent criterion:** A cell belongs to the self-gravitating tail if $s(\mathbf{x}) > s_t$.

This is a direct implementation of the theoretical criterion—no proxy or approximation.

### 7.2 The Direct $s$-Threshold Algorithm

**Step 1:** Convert density field to log-density:
$$
s(\mathbf{x}) = \ln\left(\frac{\rho(\mathbf{x})}{\langle\rho\rangle}\right)
$$

**Step 2:** Apply BM19's transition density threshold with soft sigmoid:
$$
w(\mathbf{x}) = \sigma\left[\kappa \cdot (s(\mathbf{x}) - s_t)\right]
$$

where $\sigma$ is the sigmoid function and $\kappa$ controls sharpness.

**Step 3:** Construct the tail PMF:
$$
p_{\rm tail}(\mathbf{x}) \propto w(\mathbf{x}) \cdot \rho(\mathbf{x})
$$

### 7.3 Why Direct Thresholding is Preferred

| Method | Theoretical basis | Implementation |
|--------|------------------|----------------|
| **Direct $s > s_t$** | Exact BM19 criterion | `compute_tail_pmfs_bm19()` |
| Local-overdensity proxy | Heuristic approximation | legacy `mode='pn11_legacy'` (deprecated) |

The local overdensity method ($\rho/\rho_{\rm smooth}$) was a proxy that attempted to identify "locally overdense" regions. But BM19's criterion is about **absolute** log-density, not relative to neighbors. Direct thresholding:

- Is theoretically consistent with BM19
- Requires no smoothing scale parameter
- Produces $f_{\rm tail}^{\rm actual}$ that matches $f_{\rm dense}^{\rm theory}$

### 7.4 Validation: $f_{\rm tail}$ Should Match $f_{\rm dense}$

For a density field with correct PDF statistics:
$$
f_{\rm tail}^{\rm actual} = \frac{\sum_{\mathbf{x}: s > s_t} \rho(\mathbf{x}) \, dV}{\sum_{\mathbf{x}} \rho(\mathbf{x}) \, dV} \approx f_{\rm dense}^{\rm BM19}
$$

This is a key consistency check: the geometric realization should match the 1D theory.

**Expected scatter:** For a single 3D realization at $128^3$ resolution, we expect agreement at the ~few percent level due to finite sampling of the PDF. An ensemble average over multiple random realizations should converge to $<1\%$ agreement with the analytic $f_{\rm dense}$. Deviations larger than ~5% for a single realization suggest either a bug or that the field's PDF is not properly lognormal.

### 7.5 Two Probability Mass Functions

From the tail weights, construct two normalized PMFs:

$$
p_{\rm tail}(\mathbf{x}) \propto w(\mathbf{x}) \cdot \rho(\mathbf{x})
$$

$$
p_{\rm smooth}(\mathbf{x}) \propto \rho(\mathbf{x})
$$

where $p_{\rm tail}$ is weighted toward the self-gravitating tail, and $p_{\rm smooth}$ covers the full domain.

**Normalization choice:** $p_{\rm smooth}$ is defined on the full domain, so some "smooth" stars may still land in dense regions. This mimics a scenario where star formation is enhanced but not exclusive in the dense tail.

### 7.6 Operational Modes

In production, we support two modes:

| Mode | Tail selection | Use case |
|------|----------------|----------|
| **BM19-consistent** | Direct $s > s_t$ threshold | Physics-driven ICs (preferred) |
| **Phenomenological** | User-specified $f_{\rm sub}$ | Exploratory studies |

In BM19-consistent mode (the default), the transition density $s_t$ comes directly from BM19 theory given $(\sigma_s, \alpha)$. The resulting $f_{\rm tail}$ should match $f_{\rm dense}$.

**Differentiability note:** The BM19-consistent path is **fully differentiable** in all physical parameters $(\mathcal{M}, b, \alpha, \eta_{\rm survive})$. Gradients can flow from a loss defined on dense-gas statistics (e.g., $f_{\rm tail}$ or $\zeta_{\rm FDF}$) back through $s_t \to \sigma_s \to (\mathcal{M}, b, \alpha)$. Gradients **cannot** flow through $Q$ (the MST-based measurement is non-differentiable). The realized differentiable inference layer (`gravoturb_fdf.inference`; AC11–AC17) therefore **sidesteps $Q$ entirely**: it predicts analytic summary statistics — the log-density 2-point (Gaussianization), counts-in-cells, and a peaks-over-threshold density-tail block — as smooth functions of $\theta$ and differentiates *those*, recovering $(\mathcal{M}, \alpha, \beta)$ including the BM19 tail slope $\alpha$. $Q$ is retained as a validation/demo diagnostic; the earlier fitted $Q$-surrogate prototype was retired. See [`differentiable-inference.md`](../website/10-theory/gravoturbulence/differentiable-inference.md) and the package `README.md` for the current API (some method names in this design guide predate the 2026-06 clean-room rewrite).

**Implementation note:** The legacy local-overdensity ranking (`mode='pn11_legacy'`) remains in the codebase for historical/phenomenological runs, but the default is `compute_tail_pmfs_bm19()` with direct thresholding on $s$.

---

## 8. From f_sub to Star Positions and Q

### 8.1 Star Sampling Algorithm

Given target number of stars $N_*$ and substructure fraction $f_{\rm sub} = \eta_{\rm survive} \cdot f_{\rm dense}$:

**Step 1:** Allocate stars:
$$
N_{\rm tail} = \text{round}(f_{\rm sub} \cdot N_*), \quad N_{\rm smooth} = N_* - N_{\rm tail}
$$

**Step 2:** Sample positions:
- Draw $N_{\rm tail}$ positions from $p_{\rm tail}(\mathbf{x})$ — stars in dense clumps
- Draw $N_{\rm smooth}$ positions from $p_{\rm smooth}(\mathbf{x})$ — stars in smooth component

**Step 3:** Add sub-voxel jitter within each cell.

**The only free substructure parameter is $f_{\rm sub}$**. Everything else—clump shapes, sizes, separations—is inherited from the turbulent field.

### 8.2 The f_sub → Q Mapping

Higher $f_{\rm sub}$ means more stars in dense clumps → lower Q (more substructure):

| $f_{\rm sub}$ | Expected Q | Structure |
|---------------|------------|-----------|
| 0.01 | 0.75–0.80 | Nearly smooth |
| 0.05 | 0.70–0.78 | Mild substructure |
| 0.10 | 0.60–0.72 | Moderate substructure |
| 0.20 | 0.50–0.65 | Strong substructure |
| 0.30 | 0.45–0.58 | Very clumpy |

*(The exact mapping depends on $\beta$ and the base profile; values here are illustrative of the calibration we expect to find in §12. Final numbers may shift.)*

**This mapping must be calibrated numerically** by running FDF realizations and measuring Q.

**Important caveats:**

1. **Q degeneracies:** Q is not a sufficient statistic for topology. Very centrally concentrated but smooth configurations can share Q values with multi-clump structures at very different spatial scales. We treat Q as our primary scalar summary, but we will also inspect MST maps and nearest-neighbor distributions during calibration to verify we're capturing the correct morphology. **We will illustrate this degeneracy explicitly in Paper A with side-by-side examples of configurations with identical Q but different topology (e.g., single concentrated clump vs. multiple separated clumps).**

2. **$N_*$ dependence:** The mapping $Q(f_{\rm sub})$ has some dependence on the number of stars, particularly at small $N_*$ where Poisson noise dominates. We will explicitly test $N_*$ scaling during calibration to verify the mapping stabilizes beyond some threshold (expected: $N_* \gtrsim 100$–$200$).

### 8.3 Why Q Emerges Rather Than Being Set

In traditional IC generators, you set $D$ (fractal dimension) and get Q as an output.

In our framework, you set $(\mathcal{M}, b, \alpha, \eta_{\rm survive})$ and get **both** $f_{\rm sub}$ (from BM19) and Q (from FDF realization). The substructure has a **physical origin**, not an arbitrary parameterization.

### 8.4 Summary: What Part II Provides

Part II (FDF) answers: **"Given the self-gravitating mass fraction, where is it located and how clumpy are the resulting stars?"**

$$
f_{\rm sub} \xrightarrow{\text{FDF}} \text{3D density field} \xrightarrow{\text{tail mask}} \text{star positions} \xrightarrow{\text{measure}} Q
$$

---

# Part III: Dense Gas SFR and Parmentier ζ

*This is the bottom layer of the causal chain (§1.2). Part III takes the realized density field from Part II and interprets it in terms of dense-gas star formation efficiency.*

## 9. The Magnification Factor Framework

### 9.1 The Question Parmentier Answers

Parts I and II tell us:
- *How much* gas is self-gravitating ($f_{\rm dense}$)
- *Where* that gas is located (FDF realization)

Parmentier & Pasquali (2020, hereafter PP20) ask a complementary question:

> **Given a mass $M_{\rm dg}$ of dense gas with a particular spatial structure, how efficiently does it form stars?**

### 9.2 The Magnification Factor ζ

PP20 define $\zeta$ as the ratio of a clump's SFR to that of a uniform-density ("top-hat") equivalent:

$$
\zeta \equiv \frac{\text{SFR}_{\rm clump}}{\text{SFR}_{\rm top-hat}}
$$

Centrally concentrated clumps have $\zeta > 1$ because their inner regions have shorter freefall times than the mean.

### 9.3 The Dense-Gas SFR Equation

The star formation rate of dense gas becomes:

$$
\boxed{\text{SFR}_{\rm dg} = \zeta(p, r_c/R) \cdot \epsilon_{\rm ff,int} \cdot \frac{M_{\rm dg}}{\langle t_{\rm ff,dg} \rangle}}
$$

where:
- $\epsilon_{\rm ff,int}$ — intrinsic SFE per freefall time (geometry-independent)
- $p$ — slope of radial density profile ($\rho \propto r^{-p}$)
- $r_c/R$ — relative size of any central constant-density core
- $\langle t_{\rm ff,dg} \rangle$ — mean freefall time of the dense gas

### 9.4 Analytic Form for ζ(p)

For a pure power-law profile $\rho \propto r^{-p}$ with $0 \le p < 2$ (PP20 Eq. 6; derivation from Tan et al. 2006):

$$
\boxed{\zeta(p) = \frac{(3 - p)^{3/2}}{2.6\,(2 - p)}}
$$

where $2.6 \approx 3^{3/2}/2 = 2.598$, so equivalently $\zeta(p) = 2(3-p)^{3/2}/[3^{3/2}(2-p)]$ (the two agree to $<0.1\%$ on $0 \le p < 2$). This expression is well-behaved on the whole physical domain: $\zeta(0) = 1$ (top-hat baseline), rising monotonically and diverging only as $p \to 2$ ($\alpha \to 3/2$), where the pure power-law mass integral itself diverges and a central core is required.

**ζ(p) from PP20 Eq. 6** (pure power law, no core):

| Profile slope $p$ | $\zeta(p)$ | $\alpha = 3/p$ |
|-------------------|------------|----------------|
| 0 (uniform) | 1.000 | ∞ |
| 1.0 | 1.089 | 3.0 |
| 1.5 | 1.414 (= √2) | 2.0 |
| 1.67 | 1.79 | 1.8 |
| → 2 | → ∞ | → 1.5 |

For $p \geq 2$ ($\alpha \le 3/2$) the pure power-law form diverges and a central core is required; $\zeta$ then depends on $r_c/R$ (use `magnification_factor_with_core` or measure $\zeta_{\rm FDF}$ directly).

> **When to prefer the direct measurement $\zeta_{\rm FDF}$:** The analytic $\zeta(p)$ assumes a *pure* power law with no core, and diverges as $p \to 2$. Real dense gas has finite central cores and departs from a single power law, so for realistic geometry — and for any $p$ near 2 — measure $\zeta_{\rm FDF}$ directly from the FDF density field via the freefall-weighted integral:
> $$\zeta_{\rm FDF} = \frac{\int_{\rm tail} \rho^{3/2} \, dV}{M_{\rm tail} \cdot \langle\rho_{\rm tail}\rangle^{1/2}}$$
> This automatically accounts for cores, deviations from pure power laws, and the actual geometry of the dense tail. See `zeta_fdf_direct()` (Appendix A).

### 9.5 The Connection: BM19 α ↔ PP20 p

The BM19 PDF slope $\alpha$ and PP20's radial profile slope $p$ are related via:

$$
\boxed{p = \frac{3}{\alpha} \quad \Leftrightarrow \quad \alpha = \frac{3}{p}}
$$

| BM19 $\alpha$ | PP20 $p$ | Physical state | $\zeta$ behavior |
|---------------|----------|----------------|------------------|
| 3.0 | 1.0 | Early collapse | Moderate boost |
| 2.0 | 1.5 | Active collapse | Significant boost |
| 1.5 | 2.0 | Isothermal cores | Very large (core-dependent) |

**Implementation note:** The analytic $\zeta(p) = (3-p)^{3/2}/[2.6\,(2-p)]$ (§9.4) is valid across $0 \le p < 2$, diverging only as $p \to 2$ ($\alpha \to 3/2$). For dense gas with finite cores, departures from a single power law, or $p$ near 2, measure $\zeta_{\rm FDF}$ directly (§10.3) rather than relying on the pure-power-law form.

**Key insight:** Advanced collapse (low $\alpha$) = steep radial profiles (high $p$) = large SFR boost ($\zeta \gg 1$).

### 9.6 The "Permitted Band" in (p, SFR/M_dg) Space

For given $\epsilon_{\rm ff,int}$ and $\langle t_{\rm ff,dg} \rangle$, real clouds must lie within a permitted region:

- **Lower bound:** $\zeta = 1$ (top-hat, uniform density)
- **Upper bound:** $\zeta(p)$ for pure power-law

PP20 show that:
- **Nearby clouds** (Kainulainen et al. 2014) populate this band with $\epsilon_{\rm ff,int} \approx 10^{-2}$
- **CMZ clouds** (Lu et al. 2019) lie a factor ~10 *below*, suggesting $\epsilon_{\rm ff,int} \approx 10^{-3}$ or an additional environmental factor

---

## 10. Measuring p_eff and ζ from FDF Realizations

### 10.1 Defining Dense Gas in FDF

For each FDF realization, define dense gas using one of:

| Method | Definition | Notes |
|--------|------------|-------|
| Density threshold | $n > 10^4$ cm$^{-3}$ | Matches observational convention |
| Log-density threshold | $s > s_{\rm dg}$ | Consistent with PDF framework |
| Tail mask | $T(\mathbf{x}) = 1$ | Uses our geometric selection |

The resulting $M_{\rm dg}$ should be consistent with observational definitions.

**Critical distinction: $f_{\rm dense}$ vs $M_{\rm dg}$**

| Quantity | Definition | Threshold | Source |
|----------|------------|-----------|--------|
| $f_{\rm dense}$ | BM19 self-gravitating fraction | $s > s_t$ (derived from $\sigma_s, \alpha$) | Theory |
| $M_{\rm dg}$ | Observational dense gas mass | $n > 10^4$ cm$^{-3}$ or tracer-based | Observation |

These are **not** tied to the same density threshold. $f_{\rm dense}$ is a theoretical self-gravity criterion; $M_{\rm dg}$ is typically defined by observational tracers (e.g., HCN, $n > 10^4$ cm$^{-3}$). Their correlation across $(\mathcal{M}, b, \alpha)$ is an **emergent prediction** of the framework.

*In calibration, we will explicitly track both: the BM19 tail fraction $f_{\rm dense}$ and the tracer-defined $M_{\rm dg}$, and check how they correlate across $(\mathcal{M}, b, \alpha)$.*

### 10.2 Measuring the Effective Density Slope p_eff

From the dense-gas region, measure $p_{\rm eff}$ via:

**Method A: Radial profile fitting**
1. Compute spherically-averaged $\rho(r)$ around the density peak
2. Fit $\rho \propto r^{-p_{\rm eff}}$ over the dense region

**Method B: PDF slope**
1. Measure the slope of the high-density tail of the $\rho$-PDF
2. Convert: $p_{\rm eff} = 3/\alpha_{\rm measured}$

### 10.3 Computing ζ_FDF Directly

> **Hierarchy of ζ estimates:**
> - **$\zeta_{\rm FDF}$ (direct measurement):** Ground truth; always valid. Use for any serious inference or comparison to PP20.
> - **$\zeta(p)$ (analytic formula, PP20 Eq. 6):** Valid for a pure power law across $0 \le p < 2$. Use as a closed-form estimate or sanity check; prefer $\zeta_{\rm FDF}$ when the dense gas has finite cores, departs from a single power law, or has $p$ near 2.

Compute $\zeta$ directly from the field:

$$
\zeta_{\rm FDF} = \frac{\sum_{\rm cells} \rho(\mathbf{x}) / t_{\rm ff}(\mathbf{x}) \, dV}{M_{\rm dg} / \langle t_{\rm ff,dg} \rangle}
$$

where $t_{\rm ff}(\mathbf{x}) = \sqrt{3\pi / (32 G \rho(\mathbf{x}))}$.

This bypasses the power-law assumption and measures the actual geometric boost.

### 10.4 Validation: Landing in the Permitted Band

For each FDF realization:
1. Measure $(p_{\rm eff}, \text{SFR}/M_{\rm dg})$
2. Plot on PP20's $(p, \text{SFR}/M_{\rm dg})$ diagram
3. Verify that synthetic clouds populate the correct band

**Expected behavior:**
- Nearby-cloud-like parameters → upper band ($\epsilon_{\rm ff,int} \sim 10^{-2}$)
- CMZ-like parameters → lower band ($\epsilon_{\rm ff,int} \sim 10^{-3}$)

---

## 11. Unified Picture: BM19 + FDF + PP20

### 11.1 The Complete Framework

The three frameworks provide complementary pieces:

| Framework | Question | Output |
|-----------|----------|--------|
| **BM19** | How much gas is self-gravitating? | $f_{\rm dense}(\sigma_s, \alpha)$ |
| **FDF** | Where is that gas, and how clumpy are stars? | $\rho(\mathbf{x})$, Q |
| **PP20** | How efficiently does that gas form stars? | $\zeta(p)$, SFR/$M_{\rm dg}$ |

### 11.2 Four Separable Physical Effects

The complete picture separates four distinct effects:

| Component | Symbol | Source | Physical meaning |
|-----------|--------|--------|------------------|
| Intrinsic efficiency | $\epsilon_{\rm ff,int}$ | Microphysics | How fast gas → stars at given $\rho$ |
| Geometric boost | $\zeta(p)$ | PP20 | Density gradient enhancement |
| Tail fraction | $f_{\rm dense}$ | BM19 | How much gas is self-gravitating |
| Feedback survival | $\eta_{\rm survive}$ | This work | What survives to final cluster |

### 11.3 The Measured vs. Intrinsic Efficiency

Observationally, one measures an *effective* efficiency:

$$
\epsilon_{\rm ff,meas} = \zeta \cdot \epsilon_{\rm ff,int}
$$

Our framework allows decomposing this into geometry ($\zeta$) and intrinsics ($\epsilon_{\rm ff,int}$).

### 11.4 One Realization, Multiple Observables

From a single FDF realization, we can predict:

1. **Cluster substructure (Q)** — from star positions
2. **Dense-gas mass ($M_{\rm dg}$)** — from density threshold
3. **Effective slope ($p_{\rm eff}$)** — from radial profile
4. **Magnification ($\zeta_{\rm FDF}$)** — from freefall-weighted integration
5. **Expected SFR/$M_{\rm dg}$** — from PP20 formula

All from the same underlying turbulent physics.

**Note:** $f_{\rm dense}$ and $M_{\rm dg}$ are generally not tied to the same density threshold; their correlation is an **emergent prediction** of the framework. This sets up a natural plot for Paper B: $f_{\rm dense}$ vs $M_{\rm dg}/M$ across $(\mathcal{M}, b, \alpha)$.

### 11.5 The Complete Causal Chain (Detailed)

```
Cloud-scale observables: Σ, ℳ, b, cloud age
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
        σ_s²(ℳ, b)                      α(age, feedback)
              │                               │
              └───────────────┬───────────────┘
                              ↓
                    s_t = (α - 1/2)σ_s²
                              ↓
                    f_dense(σ_s, s_t)
                              ↓
                    f_sub = η_survive × f_dense
                              ↓
              ┌───────────────┴───────────────┐
              ↓                               ↓
    FDF realization ρ(x)               p = 3/α → ζ(p)
              ↓                               ↓
    Direct s > s_t threshold          SFR/M_dg = ζ ε_ff / t_ff
              ↓                               
    Star sampling (f_sub split)        
              ↓                               
            Q(f_sub)                          
              ↓                               ↓
              └───────────────┬───────────────┘
                              ↓
           Observables: Q, SFR/M_dg, M_dg, p_eff
```

---

# Calibration and Strategy

## 12. Calibration and Validation

### 12.1 What Needs Calibration

| Quantity | Status | Method |
|----------|--------|--------|
| $\sigma_s(\mathcal{M}, b)$ | Established | BM19 Eq. 1 |
| $s_t(\sigma_s, \alpha)$ | Established | BM19 Eq. 2 |
| $f_{\rm dense}(\sigma_s, \alpha)$ | Established | BM19 Eq. 19-20 |
| $\zeta(p)$ | Established | PP20 Eq. 6 |
| $\eta_{\rm survive}$ | Unconstrained | Rad-hydro simulations |
| **$Q(f_{\rm sub})$** | **Needs calibration** | FDF numerical experiments |

### 12.2 Calibration Pipeline

We organize calibration into two complementary tracks:

---

**Track 1: BM19 Consistency (Theory Validation)**

Ensure that the 3D FDF realization faithfully represents the 1D BM19 statistics.

**Phase 1.1: f_tail vs f_dense consistency**

Check that $f_{\rm tail}^{\rm actual}(\mathcal{M}, b, \alpha)$ recovers the analytic $f_{\rm dense}(\mathcal{M}, b, \alpha)$ to within ~1% over the parameter grid.

1. Generate FDF realizations with varying $(\mathcal{M}, b, \alpha)$
2. Threshold at $s_t$ to get the dense tail
3. Measure $f_{\rm tail}^{\rm actual} = M_{\rm tail}/M_{\rm total}$
4. Compare to $f_{\rm dense}^{\rm BM19}$ from the analytic integral
5. Verify agreement across parameter space

*This becomes Paper A's "sanity plot" (e.g., Figure 2): 1D theory vs 3D realization consistency.*

---

**Track 2: Q Calibration and Robustness**

Establish the $f_{\rm sub} \to Q$ mapping and verify its stability.

**Phase 2.1: Calibrate Q(f_sub)**
1. Generate realizations varying $f_{\rm sub}$ at fixed turbulent parameters
2. Sample star positions with **fixed $N_* \ge 500$** to minimize Poisson noise
3. Measure Q
4. Fit $Q(f_{\rm sub}; \sigma_s, \beta)$ relation
5. Report Q bands (mean ± realization scatter), not single lines

**Phase 2.2: β sensitivity test**

Check sensitivity of $Q(f_{\rm sub})$ to the power spectrum slope:
1. Compare at least two $\beta(\mathcal{M})$ prescriptions (e.g., our default vs $\beta \pm 0.3$)
2. Verify that qualitative trends are preserved; quantify the calibration uncertainty
3. This provides a "robustness test" paragraph for the results paper

**Phase 2.3: N_* scaling test**
1. Vary $N_*$ from 100 to 2000 at fixed $f_{\rm sub}$
2. Verify that Q stabilizes beyond $N_* \gtrsim 200$–$500$
3. Document the regime where Poisson noise becomes negligible

---

**Track 3: Parmentier Validation (Paper B Preview)**

*For Paper A, keep this minimal; full exploration in Paper B.*

1. Measure $p_{\rm eff}$ and $\zeta_{\rm FDF}$ from realizations
2. Plot on PP20 $(p, \text{SFR}/M_{\rm dg})$ diagram
3. Verify nearby-cloud-like and CMZ-like parameters land in correct bands

---

**Track 4: External Comparison**
- STARFORGE clusters (known ICs)
- Observed clusters with measured Q and birth environment estimates

---

**Observational Translation (Future Work)**

The mapping between $f_{\rm dense}$ (BM19 self-gravitating fraction) and $M_{\rm dg}/M$ (tracer-defined dense gas mass) is an emergent prediction. For Paper A, we can note: "We can also define an observational dense gas threshold (e.g., $n > 10^4\,\text{cm}^{-3}$); full exploration of the $f_{\rm dense} \leftrightarrow M_{\rm dg}/M$ correlation is deferred to Paper B."

### 12.3 Expected Q(f_sub) Form

Based on limiting behavior:

$$
Q(f_{\rm sub}) = Q_{\rm max} - (Q_{\rm max} - Q_{\rm min}) \cdot f_{\rm sub}^\gamma
$$

where:
- $Q_{\rm max} \approx 0.8$ (uniform sphere limit)
- $Q_{\rm min} \approx 0.3$–$0.5$ (depends on power spectrum)
- $\gamma \approx 0.5$–$1.0$ (curvature parameter)

*We expect $Q_{\rm min}$ and the curvature $\gamma$ to depend primarily on the correlation structure of the field (set by $\beta$ and the base profile) and only weakly on $N_\star$ beyond finite-N noise. We will not overfit a single universal $Q(f_{\rm sub})$ curve; rather, we will characterize how the fit parameters vary with $(\sigma_s, \beta)$.*

---

## 13. Publication Strategy

### 13.1 Paper A: IC Framework

**Scope:**
1. BM19 framework for $f_{\rm dense}$
2. FDF two-layer implementation (lognormal field + geometric tail)
3. $Q(f_{\rm sub})$ calibration
4. One validation comparison (e.g., STARFORGE or observed cluster)

**Framing:** "We extend the Burkhart & Mocz (2019) 1D gravoturbulent PDF framework into 3D spatial initial conditions, realized as a lognormal turbulent field plus a geometrically-defined self-gravitating tail."

**Parmentier in Paper A:** Mention that our dense-gas geometry is compatible with PP20 and that we can place synthetic clouds in the PP20 $(p, \text{SFR}/M_{\rm dg})$ plane, but do not make this the central result. Keep focus on BM19 + FDF → physical ICs and Q.

### 13.2 Paper B: Parmentier Connection

**Scope:**
1. $\zeta_{\rm FDF}$ measurements across parameter space
2. Systematic placement in PP20 $(p, \text{SFR}/M_{\rm dg})$ space
3. Environment-dependent predictions (nearby vs. CMZ-like)
4. Interpretation: what sets $\epsilon_{\rm ff,int}$ variations?

**Framing:** "The same turbulent density fields that set cluster substructure also predict dense-gas star formation efficiency via the Parmentier magnification factor."

**Summary: Paper A vs Paper B**

| Paper | Primary x-axis | Primary y-axis | Headline deliverable |
|-------|---------------|----------------|---------------------|
| A | $f_{\rm sub}$ | Q | Physical IC generator for clusters |
| B | $p_{\rm eff}$ | SFR/$M_{\rm dg}$ | Geometry vs dense-gas efficiency |

### 13.3 What Remains

1. Run FDF calibration grid
2. Verify $f_{\rm tail}^{\rm actual} \approx f_{\rm dense}^{\rm BM19}$ across parameter space
3. Measure Q vs. $f_{\rm sub}$
4. Measure $p_{\rm eff}$ and $\zeta_{\rm FDF}$
5. Write Results sections

The theory is complete. What remains is numerical calibration.

---

## 14. References

**Core BM19 framework:**

- Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129. "The Self-gravitating Gas Fraction and the Critical Density for Star Formation" *[Primary reference]*

- Burkhart, B. 2018, ApJ, 863, 118. "The Star Formation Rate in the Gravoturbulent Interstellar Medium"

**Dense-gas geometry and magnification (PP20):**

- Parmentier, G. & Pasquali, A. 2020, ApJ, 903, 56. "A New Parameterization of the Star Formation Rate Dense Gas Mass Relation: Embracing Gas Density Gradients" *[Magnification factor ζ]*

- Parmentier, G. 2019, ApJ, 887, 179. "Star Formation Laws and the Turbulent ISM"

- Tan, J. C., Krumholz, M. R., & McKee, C. F. 2006, ApJL, 641, L121. "Equilibrium Star Cluster Formation"

**Observational validation data:**

- Kainulainen, J., Federrath, C., & Henning, Th. 2014, Science, 344, 182. "Unfolding the Laws of Star Formation"

- Lu, X. et al. 2019, ApJ, 872, 171. "Star Formation Rates in the CMZ"

- Kauffmann, J. et al. 2017, A&A, 603, A90. "The Galactic Center Molecular Cloud Survey"

**Density PDF theory:**

- Federrath, C., Klessen, R. S., & Schmidt, W. 2008, ApJL, 688, L79. "The density probability distribution in compressible isothermal turbulence"

- Federrath, C. et al. 2010, A&A, 512, A81. "Comparing the statistics of interstellar turbulence" *[Power spectrum slope β(M) relation]*

- Molina, F. Z. et al. 2012, MNRAS, 423, 2680. "The density variance–Mach number relation"

- Kim, J. & Ryu, D. 2005, ApJL, 630, L45. "A Multidimensional Code for Isothermal Magnetohydrodynamic Flows in Astrophysics" *[Supersonic PS slope]*

**Classical framework:**

- Padoan, P. & Nordlund, Å. 2011, ApJ, 730, 40. "The Star Formation Rate of Supersonic MHD Turbulence"

- Federrath, C. & Klessen, R. S. 2012, ApJ, 761, 156. "The star formation rate of turbulent magnetized clouds"

**Cloud properties:**

- Heyer, M. & Dame, T. M. 2015, ARA&A, 53, 583. "Molecular Clouds in the Milky Way"

- Heyer, M. H. & Brunt, C. M. 2004, ApJ, 615, L45. "The Universality of Turbulence in Galactic Molecular Clouds"

- Larson, R. B. 1981, MNRAS, 194, 809. "Turbulence and star formation in molecular clouds"

**Cluster structure:**

- Cartwright, A. & Whitworth, A. P. 2004, MNRAS, 348, 589. "A new method for analysing spatial distribution in young clusters"

- Goodwin, S. P. & Whitworth, A. P. 2004, A&A, 413, 929. "Dynamical evolution of fractal star clusters"

**Hydrodynamic simulations:**

- Grudić, M. Y. et al. 2021, MNRAS, 506, 2199. "STARFORGE: Towards a comprehensive numerical model of star cluster formation"

- Grudić, M. Y. et al. 2022, MNRAS, 512, 216. "Great balls of FIRE – II. Evolution and destruction of star clusters"

---

## Appendix A: Reference Implementation (in `progenax`)

The framework above is implemented in the `progenax.gravoturb` package (1D BM19 PDF,
PP20 magnification) and the `progenax.cluster` FDF layer (3D realization, tail
selection, sampling). Earlier revisions of this guide embedded a full copy of the
code here; it drifted from the package (most seriously, a stale $\zeta(p)$
transcription), so we now point to the canonical modules instead. Everything below is
JAX-native (`jax.jit` / `jax.grad` / `jax.vmap`) and differentiable in the physical
parameters $(\mathcal{M}, b, \alpha, \eta_{\rm survive})$.

### A.1 Part I — BM19 1D PDF (`progenax/gravoturb/bm19_model.py`)

| Function | Role | Equation |
|----------|------|----------|
| `sigma_s_squared(mach, b)` | PDF width $\sigma_s^2$ | BM19 Eq. 1: $\ln(1+b^2\mathcal{M}^2)$ |
| `transition_density(sigma_s_sq, alpha)` | transition density $s_t$ | BM19 Eq. 2: $(\alpha-\tfrac{1}{2})\sigma_s^2$ |
| `f_dense_bm19_full(sigma_s_sq, s_t, alpha)` | self-gravitating fraction (full piecewise LN+PL integral) | BM19 Eq. 19–20 |
| `f_dense_lognormal_limit(sigma_s_sq, s_t)` | $\alpha\to\infty$ limit (comparison only) | §Key Result |
| `power_spectrum_slope(mach, b)` | $\beta(\mathcal{M})$ interpolation (heuristic; see §6.3) | — |
| `bm19_pipeline(mach, b, alpha, eta_survive)` | end-to-end → `BM19Result` | Part I |

### A.2 Part III — PP20 magnification (`progenax/gravoturb/pp20_magnification.py`)

| Function | Role | Notes |
|----------|------|-------|
| `magnification_factor(p)` | analytic $\zeta(p)$ | **PP20 Eq. 6**: $(3-p)^{3/2}/[2.6\,(2-p)]$; $\zeta(0)=1$, valid $0\le p<2$ |
| `magnification_factor_with_core(p, r_c_over_R, ...)` | $\zeta$ for a cored profile | numerical integration |
| `zeta_fdf_direct(rho_grid, dense_weights)` | $\zeta$ measured from a density field | freefall-weighted; ground truth |
| `sfr_per_dense_gas(p, ...)` | $\mathrm{SFR}/M_{\rm dg}=\zeta\,\epsilon_{\rm ff}/t_{\rm ff}$ | PP20 |

> **Note.** `magnification_factor` is the correct, well-behaved PP20 Eq. 6 across
> $0\le p<2$. A pre-2026 transcription `(3-p)/(2.6-2p)^{3/2}` — with a spurious
> $p=1.3$ pole, once rationalised as a "domain limit" — was a typo, since corrected
> (see the fix-note in `pp20_magnification.py`, verified 2026-04-28 against PP20
> Eq. 6, p. 2). For cored / non-power-law geometry or $p$ near 2, prefer
> `zeta_fdf_direct`.

### A.3 BM19 field PDF + Part II FDF realization

The lognormal+powerlaw field PDF used to realize 3D fields lives in
`progenax/gravoturb/bm19_pdf.py` (`bm19_volume_pdf`, `build_bm19_cdf_table`,
`bm19_icdf`, `gaussian_to_bm19`, `validate_bm19_field`). The 3D density field, the
BM19-consistent direct $s>s_t$ tail selection (soft-sigmoid weights), PMF
construction, and star sampling live in `progenax/cluster/` — chiefly `fdf_density/`
(`density_field.py`, `field_init.py`, `pipeline.py`, `sampling.py`), `fdf_tail.py`,
`turbulence.py`, and the high-level `gravoturbulent.py`. The BM19-consistent path is
the default and is differentiable; the legacy local-overdensity proxy
(`mode='pn11_legacy'`) is retained for comparison only and is deprecated.

### A.4 Tests

Locked-in regression and gradient tests live in `tests/unit/physics/` (BM19
mass-conservation and lognormal-limit checks, PN11/PP20 formula and $\zeta$-anchor
tests, FD-vs-autodiff grad-checks) and `tests/unit/cluster/` (FDF field, tail
sampling, gravoturbulent presets). Run them with
`pytest tests/unit/physics tests/unit/cluster`.

---

## Appendix B: Classical PN11/FK12 Framework (Historical)

This appendix documents the classical Padoan-Nordlund (2011) and Federrath-Klessen (2012) framework, which preceded BM19.

### B.1 The Pure Lognormal PDF

Early gravoturbulent theories assumed a purely lognormal PDF:

$$
p(s) = \frac{1}{\sqrt{2\pi}\sigma_s} \exp\left[-\frac{(s + \sigma_s^2/2)^2}{2\sigma_s^2}\right]
$$

### B.2 The Parameterized Critical Density

These theories derive a critical density from $t_{\rm ff} < t_{\rm cross}$. **PN11**
(their Eq. 8) write it with a turbulence integral-scale factor $\theta$:

$$
\frac{\rho_{\rm crit}}{\rho_0} = 0.067\,\theta^{-2}\,\alpha_{\rm vir}\,\mathcal{M}^2,
\qquad s_{\rm crit} = \ln\!\left(0.067\,\theta^{-2}\,\alpha_{\rm vir}\,\mathcal{M}^2\right)
$$

with $\theta \approx 0.35$ (Wang & George 2002 correction; PN11 p. 3), giving prefactor
$0.067 \times 0.35^{-2} = 0.547$ (PN11 Eq. 11). **FK12** (and KM05) recast the same
physics with a geometric/magnetic factor $\phi_x$:

$$
s_{\rm crit} = \ln\!\left(\frac{\pi^2 \phi_x^2}{5}\,\alpha_{\rm vir}\,\mathcal{M}^2\right),
\qquad \phi_x \approx 0.17\text{–}0.5.
$$

progenax's `s_crit_pn11` (`gravoturb/pn11_model.py`) implements the **PN11 Eq. 8
form** ($0.067\,\theta^{-2}$), not the FK12 $\phi_x$ form.

**Problems with this classical approach (why we use BM19 instead):**
1. $\theta$ / $\phi_x$ must be calibrated, not derived
2. Different authors use different conventions and values
3. No connection to evolutionary state

### B.3 Comparison to BM19

| Aspect | PN11/FK12 | BM19 |
|--------|-----------|------|
| PDF form | Lognormal only | Piecewise LN+PL |
| Critical density | Parameterized ($\phi_x$) | Derived ($s_t$) |
| Evolutionary state | Not encoded | Encoded in $\alpha$ |
| Validation | Limited | M51 PAWS data |

**Why we use BM19:** The transition density $s_t$ is derived, $\alpha$ is observable, and the framework is validated against observations.

---

## Appendix C: The Q Parameter in Detail

### C.1 Definition

The Cartwright-Whitworth Q parameter:

$$
Q = \frac{\bar{m}}{\bar{s}}
$$

### C.2 Calculating $\bar{m}$ (MST)

1. Construct Minimum Spanning Tree connecting all N stars
2. Sum edge lengths: $L_{\rm MST} = \sum_{i=1}^{N-1} e_i$
3. Normalize: $\bar{m} = \frac{L_{\rm MST}}{(N-1) R_{\rm cluster}} \cdot \sqrt{N}$

### C.3 Calculating $\bar{s}$ (Pairwise)

1. Compute all $N(N-1)/2$ pairwise distances
2. Take mean: $\langle r \rangle = \frac{2}{N(N-1)} \sum_{i<j} |\mathbf{x}_i - \mathbf{x}_j|$
3. Normalize: $\bar{s} = \langle r \rangle / R_{\rm cluster}$

### C.4 Why Q Detects Substructure

| Property | Effect on $\bar{m}$ | Effect on $\bar{s}$ | Effect on Q |
|----------|---------------------|---------------------|-------------|
| Central concentration | ↓ | ↓ | ~unchanged |
| Substructure (clumps) | ↓↓ | ~ or ↑ | **↓↓** |

Substructure preferentially shrinks MST edges (local connectivity) while leaving pairwise separations large.

---

## Appendix D: Alternative—Local Virial Parameter Mask

This appendix describes an alternative collapse criterion not currently implemented.

### D.1 The Local Virial Approach

Instead of local overdensity, define collapse via local virial parameter:

$$
\alpha_{\rm vir,loc}(\mathbf{x}) = \frac{5 \sigma_\ell^2}{G \bar{\rho}_\ell(\mathbf{x}) \ell^2}
$$

where:
- $\bar{\rho}_\ell$ is the smoothed density on scale $\ell$
- $\sigma_\ell$ is the velocity dispersion on scale $\ell$ (e.g., from Larson's law)

### D.2 Collapse Mask

$$
C(\mathbf{x}) = \begin{cases} 1 & \text{if } \alpha_{\rm vir,loc}(\mathbf{x}) < 1 \\ 0 & \text{otherwise} \end{cases}
$$

### D.3 Why We Don't Currently Use This

1. Requires velocity field or Larson's law assumption
2. Local overdensity is simpler and sufficient
3. Results should be similar for realistic turbulent fields

This approach may be explored in future work for more physically detailed models.

---

*Document version: 9.1 — BM19 + FDF + PP20; calibration-ready (theory frozen December 2025; ζ and sourcing corrections June 2026)*

*v9.0 → v9.1 changes (June 2026 — progenax SoTA grounding pass): (1) **Corrected the PP20 $\zeta(p)$ formula** in §9.4/§9.5/§10.3 to the true Eq. 6 $(3-p)^{3/2}/[2.6\,(2-p)]$ ($\zeta(0)=1$, valid $0 \le p < 2$); the previous $(3-p)/(2.6-2p)^{3/2}$ form with a spurious $p=1.3$ pole was a transcription typo (corrected in `pp20_magnification.py`, verified 2026-04-28 against PP20 Eq. 6 p. 2), and the "never use the analytic formula / hard rule" narrative built on that pole was removed. (2) **Replaced the embedded Appendix A code dump with a pointer + API summary** to the canonical modules (`gravoturb/`, `cluster/`), from which the dump had drifted. (3) **Appendix B.2:** distinguished the PN11 Eq. 8 form ($0.067\,\theta^{-2}$, the form `s_crit_pn11` implements) from the FK12 $\phi_x$ form. (4) Fixed stale body references (`compute_sampling_pmfs` / `local_overdensity` → legacy `mode='pn11_legacy'`).*

*v8.9 → v9.0 changes: **Final consistency fixes per fourth reviewer.** (1) **Fixed sonic scale docstring**: removed inconsistent "$\ell_s = L/\mathcal{M}$" formula; clarified that we implement Larson/Burgers scaling $\ell_s = L \mathcal{M}^{-2}$ with explicit derivation. (2) **Enhanced $\zeta(p)$ domain warning**: added explicit regime-by-regime behavior in `magnification_factor()` docstring. (3) **Clarified Parameter Budget**: added note that $\Sigma$ and $R$ are bookkeeping parameters; BM19 core depends only on $(\mathcal{M}, b, \alpha, \eta)$. (4) **Added GRF vs Goodwin & Whitworth comparison**: explicit statement that our $(\sigma_s, \beta, f_{\rm sub})$ replace their fractal dimension $D$. (5) **Added Paper A figure promise**: Q degeneracy will be illustrated with side-by-side examples. (6) **Added $\zeta$ hierarchy box**: clarified that $\zeta_{\rm FDF}$ is ground truth; $\zeta(p)$ is sanity check only. (7) **Trimmed repetition**: simplified §5.2 reference to core principle. Status: **Theory frozen; remaining work is numerical calibration.***

*v8.8 → v8.9 changes: **Final structural polish per third reviewer.** (1) Added "Parameter Budget" table after Deliverable box listing free vs derived parameters. (2) Compressed §5.2 to avoid repeating "no explicit piecewise PDF" claim. (3) Expanded gradient flow note in §7.6 main text (not just docstrings) clarifying where AD works and doesn't. Status: **Ready for FDF implementation and Q calibration grids.***

*v8.7 → v8.8 changes: **Paper-ready polish per second reviewer feedback.** (1) Added "Deliverable" box after Key Result summarizing inputs/outputs. (2) Clarified that $f_{\rm sub}$ is "not ad hoc" in Key Result section. (3) Added magnetization/Alfvén Mach number note to §1.5 Limitations. (4) Added compact closed-form expression for $f_{\rm dense}$ in §4.2. (5) Added caveat that $\alpha$ and $\Sigma$ variations can dominate the Mach anti-correlation. (6) Boxed the $\zeta(p)$ hard rule for prominence. (7) Added parenthetical in §8.2 that Q values are illustrative. (8) Restructured §12.2 as two tracks (BM19 consistency + Q calibration) with β sensitivity test and $N_* \ge 500$ threshold. (9) Added gradient flow documentation to `compute_tail_pmfs_bm19()` docstring.*

*v8.6 → v8.7 changes: **Final polish per reviewer feedback.** (1) Added one-sentence reminder in §4.2 that renormalization $(M_{\rm LN} + M_{\rm PL})$ makes the piecewise PDF self-consistent. (2) Added numerical illustration table in §4.3 showing $f_{\rm dense}(\mathcal{M})$ anti-correlation for fixed $(b, \alpha)$. (3) Added "Fractal" terminology clarification in §5.2 distinguishing our GRF approach from strict Hausdorff-dimension fractals. (4) Added note in §6.3 that $\beta(\mathcal{M})$ is a replaceable default. (5) Added expected scatter note in §7.4 for $f_{\rm tail}$ vs $f_{\rm dense}$ agreement. (6) Added Q degeneracy and $N_*$ dependence caveats in §8.2. (7) Added diagnostic note to `tail_mass_fraction_from_s()` docstring clarifying it uses hard mask, not soft sigmoid. (8) Added two extra unit tests in A.4: σ_s–s_0 mass conservation (JAX-native) and f_tail vs f_dense smoke test. (9) Polished abstract to include $(Q, f_{\rm sub})$ in final sentence.*

*v8.5 → v8.6 changes: **Critical math fix + API cleanup per reviewer feedback.** (1) **Fixed BM19 mass integral normalization**: removed erroneous $e^{\sigma_s^2/2}$ factor from $M_{\rm LN}$; the correct formula is $M_{\rm LN}(-\infty, s_t) = \Phi[(s_t - \sigma_s^2/2)/\sigma_s]$ with no exponential prefactor. (2) Added explicit note that $p(s)$ is volume-weighted, $e^s$ converts to mass-weighted. (3) Tightened $\zeta(p)$ validity to a hard rule: "diagnostic only for $p \gtrsim 2$, always use $\zeta_{\rm FDF}$". (4) Added explicit $f_{\rm dense}$ vs $M_{\rm dg}$ table in §10.1. (5) Removed vestigial G parameter from `zeta_fdf_direct()`. (6) Added performance caveat to `local_overdensity()`. (7) Added A.4 Required Unit Tests section with mass conservation, lognormal limit, and gradient sign tests. (8) Added emergent $f_{\rm dense}$ vs $M_{\rm dg}$ correlation note in §11.4.*

*v8.4 → v8.5 changes: **Consistency and exposition improvements per reviewer feedback.** (1) Updated all remaining references to "local overdensity" to describe direct $s > s_t$ thresholding. (2) Fixed log_density_field docstring (arithmetic mean, not geometric). (3) Added boxed statement about using $\zeta_{\rm FDF}$ for $p \gtrsim 2$. (4) Enhanced §12.2 Phase 1 to explicitly describe the $f_{\rm tail}^{\rm actual} \approx f_{\rm dense}^{\rm BM19}$ sanity check. (5) Added note on tracking both $f_{\rm dense}$ and $M_{\rm dg}$ in calibration. (6) Added Q degeneracy note in §12.3. (7) Added Paper A vs B summary table. (8) Added causal chain back-references at start of Parts I/II/III. (9) Clarified deprecated local_overdensity path is retained for historical comparison only. (10) Expanded AD limitations note for deprecated method.*

*v8.0 → v8.1 changes: Aligned Purpose section with three-part structure. Added lognormal-limit implementation caveat. Added ζ(p) validity notes. Clarified f_dense vs M_dg distinction. Added sonic scale definition. Added operational modes (BM19-consistent vs phenomenological). Clarified p_smooth normalization. Refined Paper A/B publication balance.*

*v7.1 → v8.0 changes: Complete restructuring into three-part architecture (BM19 theory / FDF realization / Parmentier interpretation). Fixed implementation description to match actual code (local overdensity, not α_vir,loc). Added explicit "what BM19 gives vs what FDF does" distinction. Promoted Parmentier to full Part III. Rewrote causal chain as three-layer diagram. Demoted local virial parameter to Appendix D as future alternative.*

---

## Recommended Abstract

> We present a physically-motivated framework for generating star cluster initial conditions with environment-dependent substructure. Our approach combines three complementary elements: (1) the Burkhart & Mocz (2019) gravoturbulent PDF framework, which predicts the self-gravitating gas fraction $f_{\rm dense}$ from cloud properties; (2) a Fractal Density Field (FDF) implementation that realizes this fraction as a 3D lognormal turbulent field with a geometrically-defined dense tail; and (3) the Parmentier & Pasquali (2020) magnification factor formalism, which interprets the resulting geometry in terms of dense-gas star formation efficiency. We do not construct an explicit piecewise PDF; instead, we generate a lognormal turbulent field and identify the self-gravitating tail by direct thresholding on log-density $s = \ln(\rho/\rho_0)$ at the BM19 transition density $s_t$, sampling a fraction $f_{\rm sub} = \eta_{\rm survive} \cdot f_{\rm dense}$ of stars from this tail. The Cartwright-Whitworth Q parameter emerges as a prediction rather than an input. This provides a physics-based alternative to arbitrary fractal dimensions, with substructure $(Q, f_{\rm sub})$ tied to observable cloud properties: surface density, Mach number, and evolutionary state.
