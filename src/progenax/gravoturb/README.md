# Progenax Gravoturb Module

Differentiable gravoturbulent star formation physics for JAX.

This module implements the theoretical framework connecting turbulent molecular cloud properties to star formation rates and stellar cluster initial conditions.

## Overview

The gravoturb module provides three interconnected frameworks:

| Module | Framework | Key Output |
|--------|-----------|------------|
| `bm19_model.py` | Burkhart & Mocz (2019) | Dense gas fraction $f_{\text{dense}}$ |
| `bm19_pdf.py` | BM19 PDF sampling | CDF remap for density fields |
| `pp20_magnification.py` | Parmentier & Pfalzner (2020) | Magnification factor $\zeta_{\text{FDF}}$ |

All functions are:
- `@jax.jit` compatible
- Differentiable via `jax.grad`
- Vectorizable via `jax.vmap`

---

## Conventions & Notation

| Symbol | Definition |
|--------|------------|
| $s$ | Log-density contrast: $s \equiv \ln(\rho / \langle \rho \rangle)$ (natural log) |
| $\rho$ | Volume density (code uses dimensionless; physical units are user's responsibility) |
| $\langle \cdot \rangle$ | Volume average unless otherwise stated |
| $\alpha$ | Power-law slope of the PDF tail (BM19): $p(s) \propto e^{-\alpha s}$ for $s > s_t$ |
| $p$ | 3D radial density slope: $\rho(r) \propto r^{-p}$ |
| $\sigma_s$ | Standard deviation of log-density $s$ |
| $s_t$ | Transition log-density where PDF switches from lognormal to power-law |

### Regime of Validity

These modules implement gravoturbulent theory, not empirical fits to specific simulations.

**Tested parameter ranges:**
- Mach number $\mathcal{M}$: 5–30
- PDF tail slope $\alpha$: 1.5–3.0 (must be > 1 for convergence)
- Driving parameter $b$: 0.3–1.0 (0.4 = mixed driving)
- Grid resolution: $\geq 128^3$ for $< 5\%$ error in $f_{\text{dense}}$

Outside these ranges the code runs, but results are extrapolations.

---

## Module 1: BM19 Gravoturbulent Framework (`bm19_model.py`)

### Physical Background

Supersonic turbulence in molecular clouds creates a **piecewise density PDF**:
- **Lognormal core**: Low-to-moderate densities from turbulent compression
- **Powerlaw tail**: High densities from self-gravitating collapse

The BM19 framework predicts the **self-gravitating gas fraction** $f_{\text{dense}}$ from cloud properties.

### Key Equations

#### 1. PDF Width from Turbulence (Federrath+2010, adopted by BM19)

$$\sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)$$

Where:
- $\sigma_s$ = standard deviation of $s = \ln(\rho/\langle\rho\rangle)$
- $b$ = driving parameter (0.33 solenoidal → 1.0 compressive; default 0.4)
- $\mathcal{M}$ = turbulent Mach number ($\sigma_v / c_s$)

```python
from progenax.gravoturb import sigma_s_squared

sigma_s_sq = sigma_s_squared(mach=10.0, b=0.4)  # ~2.83
```

#### 2. Transition Density (BM19 Eq. 2)

$$s_t = \left(\alpha - \frac{1}{2}\right) \sigma_s^2$$

Where:
- $s_t$ = log-density at lognormal→powerlaw transition
- $\alpha$ = powerlaw slope (1.5-3.0; default 2.0)

This choice enforces continuity of the **volume-weighted** PDF at $s_t$ and fixes the normalization of the power-law tail.

```python
from progenax.gravoturb import transition_density

s_t = transition_density(sigma_s_sq=2.83, alpha=2.0)  # ~4.25
```

#### 3. Self-Gravitating Fraction (BM19 Eqs. 19-20)

The full piecewise integral:

$$f_{\text{dense}} = \frac{M(s > s_t)}{M_{\text{total}}} = \frac{\int_{s_t}^{\infty} e^s \, p(s) \, ds}{\int_{-\infty}^{\infty} e^s \, p(s) \, ds}$$

Here $p(s)$ is the **volume-weighted** PDF, and the factors of $e^s$ convert to mass weighting (since $\rho = \langle\rho\rangle e^s$).

The PDF is:

$$p(s) = \begin{cases}
p_{\text{LN}}(s) & s < s_t \\
A \, e^{-\alpha s} & s \geq s_t
\end{cases}$$

With $A$ chosen for continuity at $s_t$.

```python
from progenax.gravoturb import f_dense_bm19_full

f_dense = f_dense_bm19_full(sigma_s_sq=2.83, s_t=4.25, alpha=2.0)  # ~0.15
```

### Main Pipeline

The `bm19_pipeline()` function computes all quantities in one call:

```python
from progenax.gravoturb import bm19_pipeline

result = bm19_pipeline(
    mach=10.0,           # Turbulent Mach number
    b=0.4,               # Driving parameter
    alpha=2.0,           # Powerlaw slope
    eta_survive=0.6,     # Feedback survival efficiency
)

print(f"σ_s² = {result.sigma_s_sq:.3f}")  # PDF variance
print(f"s_t = {result.s_t:.3f}")          # Transition density
print(f"f_dense = {result.f_dense:.3f}")  # Self-gravitating fraction
print(f"f_sub = {result.f_sub:.3f}")      # Substructure fraction
print(f"ζ = {result.zeta:.3f}")           # PP20 analytic ζ (see note below)
```

> **Note**: `result.zeta` is the **analytic** PP20 magnification factor $\zeta_{\text{analytic}}(p)$ for the corresponding profile slope. For fully consistent 3D fields, prefer `zeta_fdf_direct()` measured on the actual density field.

### API Reference

| Function | Description | Returns |
|----------|-------------|---------|
| `sigma_s_squared(mach, b)` | PDF variance from turbulence | `Array` |
| `transition_density(sigma_s_sq, alpha)` | Lognormal→powerlaw transition | `Array` |
| `f_dense_bm19_full(sigma_s_sq, s_t, alpha)` | Full piecewise integral | `Array` |
| `f_dense_lognormal_limit(sigma_s_sq, s_t)` | Pure lognormal approximation | `Array` |
| `power_spectrum_slope(mach, b)` | Density power spectrum β | `Array` |
| `bm19_pipeline(mach, b, alpha, eta_survive)` | Complete calculation | `BM19Result` |

### Parameter Ranges

| Parameter | Tested Range | Notes |
|-----------|--------------|-------|
| Mach $\mathcal{M}$ | 5-30 | Valid but uncalibrated outside |
| $\alpha$ | 1.5-3.0 | Must be > 1.0 for convergence |
| $b$ | 0.3-1.0 | 0.4 = mixed driving (default) |

---

## Module 2: BM19 PDF Sampling (`bm19_pdf.py`)

### Physical Background

To generate 3D density fields with exact BM19 statistics, we use **Gaussian copula (CDF remap)**:

$$g(\mathbf{x}) \sim \mathcal{N}(0,1) \;\rightarrow\; u(\mathbf{x}) = \Phi(g) \;\rightarrow\; s(\mathbf{x}) = F_V^{-1}(u)$$

Where:
- $g(\mathbf{x})$ = Gaussian random field with turbulent power spectrum
- $\Phi$ = standard normal CDF
- $F_V$ = CDF of BM19 volume-weighted PDF
- $F_V^{-1}$ = inverse CDF (quantile function)

**Key insight**: The turbulent spatial correlations live in $g(\mathbf{x})$; the BM19 one-point statistics live in the inverse CDF mapping. This preserves the target power spectrum while enforcing the exact BM19 PDF.

### Volume vs Mass Weighting

**Critical distinction:**

- **Volume PDF** $p_V(s)$: Each voxel has equal volume → sample $s$ from this
- **Mass PDF** $p_M(s) = e^s \, p_V(s) / \langle e^s \rangle$: Weighted by density

When sampling voxels uniformly, we get the volume distribution. Mass integrals naturally include $e^s$ weighting.

Throughout, BM19 define the PDF in volume space. All $f_{\text{dense}}$ and $\zeta$ calculations use $e^s p_V(s)$ to convert to mass-weighted integrals, consistent with `bm19.py`.

### Key Functions

#### Build CDF Table

```python
from progenax.gravoturb import build_bm19_cdf_table

s_grid, F_grid = build_bm19_cdf_table(
    sigma_s_sq=2.83,
    s_t=4.25,
    alpha=2.0,
    n_grid=2000,  # Resolution for interpolation
)
```

#### Transform Gaussian to BM19

```python
from progenax.gravoturb import gaussian_to_bm19
import jax.random as random

key = random.PRNGKey(42)
g = random.normal(key, (128, 128, 128))  # Gaussian field

s = gaussian_to_bm19(g, sigma_s_sq=2.83, s_t=4.25, alpha=2.0,
                     s_grid=s_grid, F_grid=F_grid)
rho = jnp.exp(s)  # Density field with exact BM19 PDF
```

#### Validate Field Statistics

```python
from progenax.gravoturb import validate_bm19_field

stats = validate_bm19_field(s, sigma_s_sq=2.83, s_t=4.25, alpha=2.0)
print(f"f_tail_actual = {stats['f_tail_actual']:.3f}")
print(f"f_dense_theory = {stats['f_dense_theory']:.3f}")
print(f"Relative error = {stats['relative_error_percent']:.1f}%")
```

**Output fields:**
- `f_tail_actual`: Mass fraction in the tail from the 3D field (using soft selection)
- `f_dense_theory`: Analytic BM19 $f_{\text{dense}}$ from `f_dense_bm19_full()`
- `relative_error_percent`: $|f_{\text{actual}} - f_{\text{theory}}| / f_{\text{theory}} \times 100$

These tie directly to the E1/E2 validation experiments.

### API Reference

| Function | Description | Returns |
|----------|-------------|---------|
| `bm19_volume_pdf(s, sigma_s_sq, s_t, alpha)` | Volume-weighted PDF | `Array` |
| `build_bm19_cdf_table(sigma_s_sq, s_t, alpha, ...)` | Precompute CDF | `(s_grid, F_grid)` |
| `bm19_icdf(u, s_grid, F_grid)` | Inverse CDF lookup | `Array` |
| `gaussian_to_bm19(g, sigma_s_sq, s_t, alpha, ...)` | Full CDF remap | `Array` |
| `validate_bm19_field(s_field, ...)` | Check field statistics | `dict` |

---

## Module 3: PP20 Magnification Framework (`pp20_magnification.py`)

### Physical Background: The Magnification Factor

#### The Core Idea

In a turbulent molecular cloud:
- Gas density is **not uniform** — it follows a PDF (lognormal + powerlaw tail in BM19)
- Star formation preferentially occurs in **dense regions** where $\rho > \rho_{\text{threshold}}$
- The star formation rate per unit volume scales with freefall time:

$$\dot{\rho}_\star(\mathbf{x}) \propto \frac{\rho(\mathbf{x})}{t_{\text{ff}}(\mathbf{x})} \propto \frac{\rho}{\sqrt{1/(G\rho)}} \propto \rho^{3/2}$$

Dense gas forms stars **disproportionately fast** compared to diffuse gas.

The **magnification factor** $\zeta_{\text{FDF}}$ captures this enhancement:

> **$\zeta_{\text{FDF}}$ = how much faster this clumpy tail forms stars than if the same mass were smeared out uniformly at its mean density.**

Values like $\zeta \approx 1\text{--}3$ correspond to "turbulence boosts SFR by a factor of a few" — exactly the story PP20 tells.

### Deriving the $\zeta_{\text{FDF}}$ Formula

Focus on the **dense tail** of the PDF, selected by sigmoid weights $w(\mathbf{x}) \in [0,1]$ from BM19:

**Define the tail quantities:**

$$M_{\text{tail}} = \sum_{\mathbf{x}} w(\mathbf{x}) \, \rho(\mathbf{x}) \, dV \quad \text{(tail mass)}$$

$$V_{\text{tail}} = \sum_{\mathbf{x}} w(\mathbf{x}) \, dV \quad \text{(effective tail volume)}$$

$$\bar{\rho}_{\text{tail}} = \frac{M_{\text{tail}}}{V_{\text{tail}}} \quad \text{(tail mean density)}$$

**Actual SFR in the structured tail** (up to a constant factor):

$$\text{SFR}_{\text{actual}} \propto \sum_{\mathbf{x}} w(\mathbf{x}) \, \rho(\mathbf{x})^{3/2} \, dV$$

**Reference: top-hat tail** with same mass $M_{\text{tail}}$, same volume $V_{\text{tail}}$, uniform density $\bar{\rho}_{\text{tail}}$:

$$\text{SFR}_{\text{tophat}} \propto M_{\text{tail}} \cdot \sqrt{\bar{\rho}_{\text{tail}}}$$

**The magnification factor is the ratio:**

$$\boxed{\zeta_{\text{FDF}} = \frac{\text{SFR}_{\text{actual}}}{\text{SFR}_{\text{tophat}}} = \frac{\sum_{\mathbf{x}} w(\mathbf{x}) \, \rho(\mathbf{x})^{3/2}}{ M_{\text{tail}} \cdot \sqrt{\bar{\rho}_{\text{tail}}}}}$$

Note: $dV$ cancels between numerator and denominator, so the implementation works with dimensionless sums over the grid.

### Why $\zeta \geq 1$ Always

The **Cauchy-Schwarz / Jensen inequality** guarantees:

$$\langle \rho^{3/2} \rangle \geq \langle \rho \rangle^{3/2}$$

Equality holds **only** for uniform density. Any density variation increases $\zeta$. We enforce `zeta = jnp.maximum(zeta, 1.0)` to guard against tiny numerical undershoots.

### Typical Values

| Cloud Type | Mach | $\alpha$ | $p = 3/\alpha$ | $\zeta_{\text{FDF}}$ |
|------------|------|----------|----------------|----------------------|
| Diffuse GMC | 5 | 2.0 | 1.5 | ~1.1 |
| Typical GMC | 10 | 2.0 | 1.5 | ~1.3-1.5 |
| Dense GMC | 15 | 2.0 | 1.5 | ~1.5-2.0 |
| YMC progenitor | 25 | 1.8 | 1.67 | ~2.0-2.5 |

**Interpretation**: Turbulent structure boosts star formation by **30-150%** compared to a uniform cloud.

### Computing $\zeta$: Direct Measurement vs Analytic

#### Direct Measurement (RECOMMENDED)

```python
from progenax.gravoturb import zeta_fdf_direct

zeta = zeta_fdf_direct(
    rho_grid=rho,           # 3D density field [Nx, Ny, Nz]
    tail_weights=w,         # Soft tail mask [Nx, Ny, Nz]
)
```

**Advantages of `zeta_fdf_direct()`:**
- Works for any density distribution (no powerlaw assumption)
- Handles full BM19 piecewise PDF naturally
- Differentiable through soft weights
- Never diverges

#### Analytic (PP20 Eq. 6) — Reference Only

`magnification_factor(p)` implements PP20 Eq. 6 for a pure power-law density profile $\rho \propto r^{-p}$. It diverges at a critical slope $p_{\text{crit}}$ where the inner integral no longer converges (see PP20 Fig. 1).

**We do not recommend this for production science** — it is included for comparison and for reproducing PP20 figures.

```python
from progenax.gravoturb import magnification_factor

zeta = magnification_factor(p=0.75)  # Only reliable for p < 1.0
```

For $p \geq 1$ (typical in star formation where $\alpha \leq 3$), always use `zeta_fdf_direct()` instead.

### SFR Prediction

Once $\zeta$ is known, predict the dense-gas star formation rate:

$$\frac{\text{SFR}}{M_{\text{dg}}} = \zeta \cdot \frac{\epsilon_{\text{ff}}}{t_{\text{ff,dg}}}$$

Where:
- $M_{\text{dg}}$ = mass of dense gas (above threshold); external input
- $t_{\text{ff,dg}}$ = characteristic freefall time at dense threshold; external input
- $\epsilon_{\text{ff}}$ = intrinsic efficiency per freefall time (~1%)

```python
from progenax.gravoturb import sfr_per_dense_gas

sfr_per_mdg = sfr_per_dense_gas(
    zeta=1.5,
    epsilon_ff_int=0.01,  # 1% efficiency per freefall
    t_ff_dg=0.3,          # Dense gas freefall time [Myr]
)
# Returns: ~0.05 Myr^-1
```

### API Reference

| Function | Description | Returns |
|----------|-------------|---------|
| `zeta_fdf_direct(rho_grid, tail_weights)` | **RECOMMENDED**: Direct 3D measurement | `Array` |
| `sfr_per_dense_gas(zeta, epsilon_ff_int, t_ff_dg)` | SFR/M_dg prediction | `Array` |
| `magnification_factor(p)` | PP20 Eq. 6 (reference only, p < 1) | `Array` |
| `magnification_factor_with_core(p, r_c_over_R)` | Cored profile (reference) | `Array` |

---

## Design Principles

### 1. Differentiability First

All functions support `jax.grad()` for gradient-based inference:

```python
import jax

def loss(mach):
    result = bm19_pipeline(mach, alpha=2.0)
    return (result.f_dense - 0.1)**2  # Target 10% dense fraction

grad_fn = jax.grad(loss)
gradient = grad_fn(10.0)  # ∂loss/∂mach
```

### 2. Soft Thresholds

Hard thresholds break gradients. We use **soft sigmoid selection**:

$$w(s) = \sigma(\kappa \cdot (s - s_t)) = \frac{1}{1 + e^{-\kappa(s - s_t)}}$$

Where $\kappa$ controls sharpness. In FDF applications, these sigmoids define the collapsing tail of the density PDF, avoiding hard $s > s_t$ cuts that would break `jax.grad`.

**Recommended values**: $\kappa \approx 5\text{--}10$ provides smooth gradients while maintaining sharp enough selection. Tested with grid resolutions $\geq 128^3$ over $\mathcal{M} \in [5, 25]$, $\alpha \in [1.5, 2.5]$ (see E1/E2/E6 validation).

### 3. Validation-Driven Development

Each function is validated against:
- Analytic limits (e.g., $\alpha \to \infty$ gives pure lognormal)
- Literature values (BM19 Table 1, PP20 Fig. 1)
- Resolution convergence (128³ sufficient for <5% error)

---

## References

1. **Burkhart, B. & Mocz, P. 2019**, ApJ, 879, 129
   - Piecewise lognormal+powerlaw PDF framework
   - Self-gravitating fraction from turbulence

2. **Parmentier, G. & Pfalzner, S. 2020**, ApJ, 903, 56
   - Magnification factor for dense-gas SFR
   - Profile slope ↔ star formation connection

3. **Federrath, C. et al. 2010**, A&A, 512, A81
   - Turbulence-density PDF relationship
   - Driving parameter calibration

4. **Padoan, P. & Nordlund, Å. 2011**, ApJ, 730, 40
   - Turbulent fragmentation theory
   - Powerlaw tail emergence

### Validation & Tests

Code-level tests and validation for these modules:
- **Unit tests**: `progenax/tests/unit/physics/` (test files reference gravoturb module)
- **Validation suite**: `progenax/validation/bm19_fdf_suite/` (E1–E6 experiments)
- **Plots**: `progenax/validation/bm19_fdf_suite/plots/`

---

## Quick Reference

```python
# Complete workflow: cloud properties → SFR prediction
from progenax.gravoturb import (
    bm19_pipeline, gaussian_to_bm19, build_bm19_cdf_table,
    zeta_fdf_direct, sfr_per_dense_gas,
)
from progenax.cluster.fdf_tail import compute_tail_pmfs_bm19
import jax.random as random
import jax.numpy as jnp

# 1. BM19 parameters from cloud properties
result = bm19_pipeline(mach=10.0, b=0.4, alpha=2.0)

# 2. Generate density field with exact BM19 PDF
s_grid, F_grid = build_bm19_cdf_table(result.sigma_s_sq, result.s_t, 2.0)
key = random.PRNGKey(42)
g = random.normal(key, (128, 128, 128))
s = gaussian_to_bm19(g, result.sigma_s_sq, result.s_t, 2.0, s_grid, F_grid)
rho = jnp.exp(s)

# 3. Compute tail weights and magnification factor
pmf_result = compute_tail_pmfs_bm19(rho, result.s_t, kappa=10.0)
zeta = zeta_fdf_direct(rho, pmf_result.tail_weights)

# 4. Predict SFR
sfr_per_mdg = sfr_per_dense_gas(zeta, epsilon_ff_int=0.01, t_ff_dg=0.3)

print(f"Dense fraction: {result.f_dense:.1%}")
print(f"Magnification: ζ = {zeta:.2f}")
print(f"SFR/M_dg = {sfr_per_mdg:.4f} Myr⁻¹")
```
