# Fractal Density Field (FDF) Module

**Physically motivated fractal substructure for star cluster initial conditions.**

This module generates spatially substructured star cluster ICs using turbulent density fields, with parameters derived from ISM physics rather than arbitrary tuning.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Physical Motivation](#physical-motivation)
3. [The Parameter Chain](#the-parameter-chain)
4. [Configurable Parameters](#configurable-parameters)
5. [API Reference](#api-reference)
6. [Q Parameter and Validation](#q-parameter-and-validation)
7. [References](#references)

---

## Quick Start

```python
import jax
import jax.numpy as jnp
from progenax.cluster.fdf_config import env_to_fdf_layer
from progenax.cluster.fdf_density import generate_fractal_ic_density
from progenax.imf import PowerLawIMF

# Physically motivated: derive FDF params from cluster mass
layer = env_to_fdf_layer(log_mecl=jnp.array(4.0))  # 10^4 M☉ cluster
print(f"σ_ln_ρ = {layer.sigma_ln_rho:.2f}")  # ~1.0 (from Federrath+2010)

# Generate cluster IC
key = jax.random.PRNGKey(42)
imf = PowerLawIMF.kroupa()
cluster = generate_fractal_ic_density(
    key=key,
    N_stars=1000,
    M_total=1e4,
    R_half=1.0,
    imf_params=imf,
    layer=layer,
)
```

---

## Physical Motivation

### Why Turbulence-Based Parameters?

Stars form in turbulent molecular clouds. The spatial distribution of young stars inherits the fractal density structure imprinted by supersonic turbulence in the parent gas cloud. This module derives FDF parameters from the same physical conditions that determine the IMF.

**Key insight:** The fractal structure is set by *gas* turbulence at the *cloud* scale, not by stellar dynamics at the *cluster* scale.

### The Unified Physics Model

```
BirthEnvironment (M_ecl, [Fe/H], SFE)
    │
    ├──→ env_to_imf_params()     → IMFParams (α₀, α₁, α₂, α₃)
    │    (Marks+2012, Jerabkova+2018)
    │
    └──→ env_to_fdf_layer()      → FractalDensityLayer (χ, σ_ln_ρ)
         (Larson 1981, Federrath+2010)

Same environment → consistent IMF + spatial structure
```

---

## The Parameter Chain

### Step 1: Cloud Radius from Environment

The parent cloud radius is derived from cluster mass, star formation efficiency (SFE), and cloud density:

```
R_cloud = (3 M_gas / 4π ρ_cl)^(1/3)
```

where `M_gas = M_ecl / SFE`.

Cloud density `ρ_cl` can be:
- Computed from Marks+2012 r_h-M scaling (default)
- Provided explicitly via `log_rho_cl` parameter

**Expected ranges:**

| M_ecl   | SFE  | ρ_cl (Marks)    | R_cloud |
|---------|------|-----------------|---------|
| 10³ M☉  | 0.33 | ~10⁴ M☉/pc³     | ~1.5 pc |
| 10⁴ M☉  | 0.33 | ~10⁵ M☉/pc³     | ~2.5 pc |
| 10⁵ M☉  | 0.33 | ~10⁶ M☉/pc³     | ~4.0 pc |
| 10⁶ M☉  | 0.33 | ~10⁷ M☉/pc³     | ~6.5 pc |

### Step 2: Turbulent Velocity from Larson Relation

The turbulent velocity dispersion scales with cloud size (Larson 1981):

```
σ_v(R) = σ_v0 × (R / 1 pc)^α
```

**Default parameters:**
- `σ_v0 = 1.0 km/s` — normalization at 1 pc
- `α = 0.5` — velocity-size exponent

**References:**
- Larson (1981) MNRAS 194, 809
- Solomon et al. (1987) ApJ 319, 730
- Heyer & Brunt (2004) ApJ 615, L45

### Step 3: Mach Number

The turbulent Mach number:

```
M = σ_v / c_s
```

**Default:** `c_s = 0.2 km/s` (cold GMC at T ~ 10 K)

**Expected ranges (using Larson relation):**

| Cluster Type | R_cloud | σ_v     | Mach  |
|--------------|---------|---------|-------|
| Small OC     | ~1.5 pc | ~1.2    | ~6    |
| Large OC     | ~2.5 pc | ~1.6    | ~8    |
| YMC          | ~4.0 pc | ~2.0    | ~10   |
| GC           | ~6.5 pc | ~2.5    | ~13   |

### Step 4: Density Contrast σ_ln_ρ (Federrath+2010)

The width of the log-density PDF in supersonic turbulence:

```
σ²_ln_ρ = ln(1 + b² M²)
```

**Parameters:**
- `b = 0.4` — turbulence driving parameter (natural mixture)
  - `b ≈ 1/3`: Solenoidal (incompressible) driving
  - `b ≈ 1.0`: Compressive driving
  - `b ≈ 0.4`: Natural mixture (default)

**Physical interpretation:** σ_ln_ρ controls the *contrast* between clumps and voids. Higher values = more extreme density fluctuations.

**Expected ranges:**

| Cluster Type | Mach | σ_ln_ρ |
|--------------|------|--------|
| Small OC     | ~6   | ~1.1   |
| Large OC     | ~8   | ~1.3   |
| YMC          | ~10  | ~1.4   |
| GC           | ~13  | ~1.6   |

**Reference:** Federrath et al. (2010) A&A 512, A81, Eq. 14

### Step 5: Spectral Slope β

The power spectrum slope interpolates between turbulence regimes:

- **Subsonic (M << 1):** Kolmogorov β = 11/3 ≈ 3.67
- **Supersonic (M >> 1):** Burgers β ≈ 4.0

```python
# Smooth interpolation using tanh
t = 0.5 * (1 + tanh((M - 1) / 0.5))
β = β_Kolmogorov × (1 - t) + β_Burgers × t
```

**Physical interpretation:** β controls the *spatial scale distribution* of density fluctuations:
- Lower β → more power at small scales → more small clumps
- Higher β → more power at large scales → smoother structure

**References:**
- Kolmogorov (1941) — Incompressible turbulence
- Burgers (1948) — Shock-dominated turbulence

### Step 6: Chi Parameter χ

The internal `chi` parameter maps to spectral slope via:

```
β = BETA_0 + BETA_1 × (χ - 1.5)
```

Default: `BETA_0 = 2.0`, `BETA_1 = 1.5`

Inverse mapping (used in `env_to_fdf_layer()`):

```
χ = (β + 0.25) / 1.5
```

**Valid range:** χ ∈ [1.6, 3.0]

---

## Configurable Parameters

### Physical Constants (`fdf_config.py`)

| Constant | Default | Description | Reference |
|----------|---------|-------------|-----------|
| `C_S_DEFAULT` | 0.2 km/s | Sound speed in cold GMC (T ~ 10 K) | — |
| `B_DEFAULT` | 0.4 | Turbulence driving parameter | Federrath+2010 |
| `SIGMA_V0_DEFAULT` | 1.0 km/s | Larson normalization at 1 pc | Larson 1981 |
| `ALPHA_LARSON` | 0.5 | Larson velocity-size exponent | Larson 1981 |
| `BETA_KOLMOGOROV` | 11/3 ≈ 3.67 | Incompressible spectral slope | Kolmogorov 1941 |
| `BETA_BURGERS` | 4.0 | Shock-dominated spectral slope | Burgers 1948 |

### FractalDensityLayer Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chi` | 2.0 | Clumpiness parameter [1.6, 3.0] |
| `sigma_ln_rho` | 2.0 | Log-density fluctuation amplitude |
| `lambda_frac` | 1.0 | Blend fraction [0=smooth, 1=full turbulent] |
| `virial_ratio` | 0.5 | Target Q_vir = K/\|U\| |
| `base_profile` | "uniform" | Base density: "uniform" or "plummer" |
| `grid_size` | 64 | FFT grid resolution per dimension |
| `box_size_factor` | 4.0 | Box extends to ±factor × R_half |
| `sphere_radius_factor` | 2.5 | Uniform sphere radius = factor × R_half |
| `use_log_normal` | True | Lognormal (True) or Gaussian (False) density |

### env_to_fdf_layer() Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `log_mecl` | *required* | log₁₀(M_ecl / M☉) |
| `sfe` | 0.33 | Star formation efficiency |
| `log_rho_cl` | None | Override cloud density (log₁₀(ρ_cl / M☉ pc⁻³)) |
| `b` | 0.4 | Federrath driving parameter |
| `c_s` | 0.2 | Sound speed [km/s] |
| `sigma_v0` | 1.0 | Larson normalization [km/s] |
| `alpha` | 0.5 | Larson exponent |
| `base_profile` | "uniform" | Base density profile |

---

## API Reference

### High-Level API

```python
# Recommended: Derive parameters from cluster mass
from progenax.cluster.fdf_config import env_to_fdf_layer
layer = env_to_fdf_layer(log_mecl=jnp.array(4.0))

# Alternative: Specify fractal dimension D directly (uncalibrated!)
from progenax.cluster.fdf_density import density_layer_from_D
layer = density_layer_from_D(D=2.0, sigma_ln_rho=1.3)  # sigma_ln_rho REQUIRED
```

### BirthEnvironment Integration

```python
from progenax.imf.environment import BirthEnvironment

# Create environment
env = BirthEnvironment.from_cluster_mass(M_ecl=1e4, FeH=0.0, sfe=0.33)

# Access turbulence properties
print(f"R_cloud = {float(env.cloud_radius()):.2f} pc")
print(f"Mach = {float(env.turbulent_mach()):.1f}")
print(f"σ_ln_ρ = {float(env.sigma_ln_rho()):.2f}")
print(f"β = {float(env.spectral_slope()):.2f}")
```

### Direct Field Construction

```python
from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    init_turbulent_density_field,
    sample_positions_from_density,
)

# Manual layer construction
layer = FractalDensityLayer(
    chi=2.0,
    sigma_ln_rho=1.5,
    base_profile="uniform",
)

# Generate density field
key = jax.random.PRNGKey(42)
field = init_turbulent_density_field(key, R_half=1.0, layer=layer)

# Sample positions
positions = sample_positions_from_density(key, field, N_stars=1000)
```

---

## Q Parameter and Validation

### The Q Parameter (Cartwright & Whitworth 2004)

The Q parameter measures spatial clustering:

```
Q = m̄ / s̄
```

where:
- `m̄` = normalized mean MST edge length
- `s̄` = normalized mean pairwise separation

**CW04 calibration (Table 1):**

| Structure | Q |
|-----------|---|
| Uniform sphere ("3D0") | 0.79 ± 0.04 |
| Fractal D=1.5 | ~0.47 |
| Fractal D=2.0 | ~0.58 |
| Fractal D=2.5 | ~0.70 |
| Fractal D=3.0 | ~0.82 |

### Base Profile Effects

**Critical:** Q interpretation depends on base profile choice.

| Base Profile | Q Range | Notes |
|--------------|---------|-------|
| `"uniform"` | ~0.79-0.85 | Matches CW04 calibration |
| `"plummer"` | >> 1 | Central concentration inflates Q |

**For CW04-comparable Q values, always use `base_profile="uniform"`.**

### Validation Tests

```python
from progenax.diagnostics.substructure import compute_q_parameter

# Measure Q for uniform sphere (baseline)
Q = compute_q_parameter(positions)
# Expected: Q ≈ 0.79-0.85 for uniform base with any σ_ln_ρ
```

See `tests/unit/substructure/test_q_baselines.py` for validation tests.

---

## References

### Core Physics

1. **Larson (1981)** MNRAS 194, 809
   - Turbulent velocity-size relation: σ_v = σ_v0 × R^α

2. **Federrath et al. (2010)** A&A 512, A81
   - Density-Mach relation: σ²_ln_ρ = ln(1 + b²M²)
   - Turbulence driving parameter b

3. **Kolmogorov (1941)**
   - Incompressible turbulence: E(k) ∝ k^{-5/3}

4. **Burgers (1948)**
   - Shock-dominated turbulence: P(k) ∝ k^{-4}

### Substructure Diagnostics

5. **Cartwright & Whitworth (2004)** MNRAS 348, 589
   - Q parameter definition and calibration

6. **Goodwin & Whitworth (2004)** A&A 413, 929
   - Fractal dimension D convention

### Environment-Dependent IMF

7. **Marks et al. (2012)** MNRAS 422, 2246
   - r_h-M_ecl relation, Fundamental Plane

8. **Jerabkova et al. (2018)** A&A 620, A39
   - Mass-based IMF prescription

### Additional Reading

9. **Solomon et al. (1987)** ApJ 319, 730
   - GMC properties and Larson relations

10. **Heyer & Brunt (2004)** ApJ 615, L45
    - Velocity-size relation in molecular clouds

11. **Padoan & Nordlund (2002)** ApJ 576, 870
    - Turbulent fragmentation theory

---

## Calibration Status

### Parameter Calibration Table

| Parameter | Status | Source | Notes |
|-----------|--------|--------|-------|
| **σ_v(R)** | ✅ Calibrated | Larson (1981), Solomon+1987 | Velocity-size relation |
| **σ_ln_ρ(M)** | ✅ Calibrated | Federrath+2010 Eq. 14 | Density-Mach relation |
| **β(M)** | ✅ Calibrated | Kolmogorov/Burgers theory | Spectral slope |
| **b(ρ)** | ⚠️ Tentative | Federrath+2010 range | Heuristic ρ→b mapping |
| **χ(β)** | ❌ Uncalibrated | Inverse of arbitrary β₀, β₁ | Awaits Q(D) sweep |
| **D→χ** | ❌ Uncalibrated | Identity mapping | No Q(D) validation |

### Recommended Usage

```python
# ✅ CORRECT: Physics-based parameters via env_to_fdf_layer()
from progenax.cluster.fdf_config import env_to_fdf_layer
import jax.numpy as jnp

layer = env_to_fdf_layer(log_mecl=jnp.array(4.0))
# Derives σ_ln_ρ from Federrath+2010, b from environment
# χ mapping is tentative but σ_ln_ρ is physics-grounded

# ❌ UNCALIBRATED: Heuristic D→χ mapping (requires explicit sigma_ln_rho)
from progenax.cluster.fdf_density import density_layer_from_D
layer = density_layer_from_D(D=2.0, sigma_ln_rho=1.3)  # No hidden defaults
```

### What's Calibrated vs Not

**Well-Grounded (use with confidence):**

- Larson velocity-size relation: σ_v = σ_v0 × R^α
- Federrath density-Mach relation: σ²_ln_ρ = ln(1 + b²M²)
- Spectral slope Kolmogorov↔Burgers interpolation

**Tentative (use with caveat):**

- Environment-dependent b: b_from_environment() maps ρ_cl → b
- The ρ→b relationship is qualitative, not precisely measured

**Uncalibrated (placeholder only):**

- χ→β mapping: β = 2.0 + 1.5×(χ - 1.5) is arbitrary
- D→χ identity: D=χ not validated against CW04 Q(D)

**Version:** `v0.5_partially_calibrated` (2024-12)

---

## File Organization

```
progenax/src/progenax/cluster/
├── README.md              # This file
├── fdf_config.py          # CANONICAL: Physical constants, env_to_fdf_layer()
├── fdf_density.py         # Density-field FDF (production)
├── fdf.py                 # LEGACY: Displacement-field FDF (deprecated)
└── core.py                # ClusterState dataclass
```
