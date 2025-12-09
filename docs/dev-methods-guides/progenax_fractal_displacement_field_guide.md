# Fractal Displacement Field Implementation Guide for `progenax`

## A Differentiable Approach to Multi-Scale Cluster Substructure

**Version:** 1.2 (December 9, 2025)

**Supersedes:** Goodwin-Whitworth recursive tree method for JAX-native applications

**Physical basis:**
- Turbulent fragmentation in molecular clouds (Larson 1981, Mac Low & Klessen 2004)
- Power-law velocity/density spectra from supersonic turbulence (Federrath et al. 2010)
- Statistically reproduces Goodwin & Whitworth (2004) / McLuster $Q_\text{CW}$ and $\sigma_\Sigma/\langle\Sigma\rangle$ after calibration

---

## Table of Contents

1. [Motivation: Why Not Goodwin-Whitworth in JAX?](#1-motivation)
2. [Physical Foundation: Turbulent Star Formation](#2-physical-foundation)
3. [The Fractal Displacement Field Method](#3-method)
4. [Mathematical Formulation](#4-math)
5. [Implementation Guide](#5-implementation)
6. [Calibration to GW/McLuster Statistics](#6-calibration)
7. [Velocity Structure](#7-velocities)
8. [Validation Tests](#8-validation)
9. [v1 Scope and Integration](#9-scope)

---

## 1. Motivation: Why Not Goodwin-Whitworth in JAX? <a name="1-motivation"></a>

### 1.1 The Original Goodwin-Whitworth Algorithm

The Goodwin & Whitworth (2004) fractal generation algorithm works as follows:

1. **Initialize**: Place a single "parent" at the origin
2. **Subdivide**: Each parent spawns $N_\text{div}$ children at random positions within a sphere of radius $r_\text{child} = r_\text{parent} / N_\text{div}^{1/D}$
3. **Prune**: Each child survives with probability $p = N_\text{div}^{(D-3)/3}$
4. **Recurse**: Repeat for $g_\text{max}$ generations
5. **Harvest**: Collect all surviving leaves as star positions
6. **Subsample**: If $N_\text{leaves} \neq N_\text{stars}$, resample to target count

This produces beautifully clumpy, hierarchical structures with fractal dimension $D$.

### 1.2 Why This Breaks in JAX

The GW algorithm has several properties that are fundamentally incompatible with JAX's differentiable programming model:

| Feature | Problem for JAX |
|---------|-----------------|
| **Bernoulli survival** | Discrete 0/1 decisions create zero gradients almost everywhere |
| **Variable cardinality** | Number of survivors is stochastic; array shapes must be static for JIT |
| **Hard sphere rejection** | $r \leq 1$ cuts create discontinuous boundaries |
| **Subsampling to N** | `replace=True/False` logic is non-differentiable |
| **Recursive tree** | Control flow depends on random outcomes |

**Consequence**: For forward Monte Carlo, GW is fine. For differentiable inference—fitting cluster parameters by backpropagating through:

$$
\text{IC} \xrightarrow{\text{N-body}} \text{evolved state} \xrightarrow{\text{render}} \text{mock obs.} \xrightarrow{\nabla} \text{likelihood}
$$

...gradients are either zero or undefined.

### 1.3 The Key Insight

We don't need the *algorithm*—we need the *statistics*. 

The GW tree produces:
- Multi-scale clumpiness quantified by Cartwright-Whitworth $Q$
- Specific correlation functions $\xi(r)$
- Azimuthal density variations $\sigma_\Sigma / \langle \Sigma \rangle$

We can achieve *statistically similar properties* with a completely different, fully differentiable generator—then calibrate parameters to match GW outputs. The FDF method will not produce *identical* realizations to GW, but after calibration it reproduces the same $Q_\text{CW}$ and $\sigma_\Sigma/\langle\Sigma\rangle$ distributions as a function of "fractal dimension."

---

## 2. Physical Foundation: Turbulent Star Formation <a name="2-physical-foundation"></a>

### 2.1 The Turbulent Molecular Cloud Picture

Stars form in turbulent molecular clouds where:

1. **Supersonic turbulence** creates a hierarchy of density fluctuations
2. **Velocity field** has power-law spectrum: $P(k) \propto k^{-\beta}$
   - Kolmogorov (incompressible): $\beta = 11/3$
   - Burgers (highly compressible): $\beta = 4$
   - Observed ISM: $\beta \approx 3.8$–$4.0$
3. **Density structure** arises from converging flows
4. **Stars inherit** the spatial structure of dense cores

### 2.2 Why Displacement Fields Are Physically Motivated

The GW fractal tree is an *abstract* mathematical construction. A displacement field with power-law spectrum is arguably *more* physical:

- It directly represents the effect of turbulent velocity perturbations on stellar positions
- The spectral slope $\beta$ connects to turbulence physics
- Multi-scale clumpiness emerges naturally from superposed modes
- Coherent substructure in density and velocity are automatically coupled

**Key references:**
- Larson (1981): Velocity-size relation in molecular clouds
- Mac Low & Klessen (2004): Control of star formation by supersonic turbulence
- Federrath et al. (2010): Comparing compressive vs solenoidal turbulence driving

### 2.3 The Physical Story We're Encoding

Starting from a smooth equilibrium profile (the "pre-turbulent" state), we apply a displacement field that:

1. **Preserves total mass**: Same $N$ stars, same $M_\text{tot}$
2. **Introduces multi-scale clumpiness**: Power on scales from $R_\text{cluster}$ down to $\ell_\text{min}$
3. **Optionally preserves radial profile**: Tangential-only displacements or rank-based remap
4. **Couples to velocities**: Stars in dense regions share coherent motions

This is exactly what turbulent fragmentation does to a collapsing cloud.

---

## 3. The Fractal Displacement Field Method <a name="3-method"></a>

### 3.1 Overview

The method has four steps:

```
1. Sample smooth equilibrium profile → x₀
2. Construct displacement field u(x; θ) with tunable spectrum
3. Apply displacement: x₁ = x₀ + λ_frac · u(x₀)
4. Optionally remap radii to preserve target profile
```

All operations are smooth functions of continuous parameters.

### 3.2 Key Parameters

| Parameter | Symbol | Range | Physical Meaning |
|-----------|--------|-------|------------------|
| Clumpiness | $\chi$ | [1.5, 3.0] | Spectral slope control (calibrated to GW $D$) |
| Fractal fraction | $\lambda_\text{frac}$ | [0, 1] | Blend from smooth → clumpy |
| Amplitude scale | $\sigma_u$ | > 0 | RMS displacement magnitude |
| Number of modes | $M$ | 32–128 | Spectral resolution |
| Min wavenumber | $k_\text{min}$ | $\sim 1/R_\text{cluster}$ | Largest scale of structure |
| Max wavenumber | $k_\text{max}$ | $\sim 1/\ell_\text{min}$ | Smallest clump scale |

### 3.3 What Gets Gradients vs. What's Frozen

For differentiable inference, we separate:

**Frozen per realization** (no gradients):
- Wavevector directions $\hat{\mathbf{k}}_n$
- Phases $\varphi_n$
- Base profile random draws (radii, angles)

**Receive gradients**:
- Clumpiness $\chi$ (controls spectral slope)
- Fractal fraction $\lambda_\text{frac}$
- Amplitude scale $\sigma_u$
- Profile parameters ($R_\text{half}$, etc.)
- Virial ratio $Q_\text{vir}$

This separation is achieved via `jax.lax.stop_gradient` on the stochastic structure.

---

## 4. Mathematical Formulation <a name="4-math"></a>

### 4.1 Base Layer: Smooth Equilibrium Profile

Let the density profile be $\rho(r; \theta_\text{prof})$ with cumulative mass:

$$
M(<r) = 4\pi \int_0^r \rho(r') r'^2 \, dr'
$$

The radial CDF is:

$$
F(r) = \frac{M(<r)}{M_\text{tot}}
$$

**Sampling positions**:

1. Draw $u_i \sim \mathcal{U}(0, 1)$
2. Invert: $r_i = F^{-1}(u_i)$
3. Draw angles isotropically:
   $$
   \cos\theta_i \sim \mathcal{U}(-1, 1), \quad \phi_i \sim \mathcal{U}(0, 2\pi)
   $$
4. Construct direction:
   $$
   \hat{\mathbf{n}}_i = (\sin\theta_i \cos\phi_i, \sin\theta_i \sin\phi_i, \cos\theta_i)
   $$
5. Position:
   $$
   \mathbf{x}_i^{(0)} = r_i \, \hat{\mathbf{n}}_i
   $$

This is the existing `sample_density_profile` in progenax.

### 4.2 Displacement Field Construction

#### 4.2.1 Mode Setup

Choose $M$ wavevectors spanning the desired scale range:

**Wavenumber magnitudes** (log-spaced):
$$
k_n = k_\text{min} \left( \frac{k_\text{max}}{k_\text{min}} \right)^{(n-1)/(M-1)}, \quad n = 1, \ldots, M
$$

**Random directions** (unit sphere):
$$
\hat{\mathbf{k}}_n \sim \text{Uniform}(S^2)
$$

**Wavevectors**:
$$
\mathbf{k}_n = k_n \, \hat{\mathbf{k}}_n
$$

**Random phases**:
$$
\varphi_n \sim \mathcal{U}(0, 2\pi)
$$

**Random polarization directions** (for vector field):
$$
\hat{\mathbf{a}}_n \sim \text{Uniform}(S^2)
$$

All of these are drawn once and frozen.

#### 4.2.2 Spectral Amplitudes

Define a power-law spectrum controlled by clumpiness parameter $\chi$:

$$
P(k) \propto k^{-\beta(\chi)}
$$

The spectral slope mapping:
$$
\beta(\chi) = \beta_0 + \beta_1 (3 - \chi)
$$

where:
- $\beta_0 \approx 2.0$: baseline slope
- $\beta_1 \approx 1.5$: sensitivity to $\chi$

This gives:
- $\chi = 3.0$ → $\beta = \beta_0$ → more large-scale power → smoother
- $\chi = 1.5$ → $\beta = \beta_0 + 1.5 \beta_1$ → more small-scale power → clumpier

**Mode amplitudes**:
$$
A_n(\chi, \sigma_u) = C(\chi, \sigma_u) \, k_n^{-\beta(\chi)/2}
$$

where $C$ is a normalization constant ensuring:
$$
\sum_{n=1}^{M} A_n^2 = \sigma_u^2
$$

Solving:
$$
C(\chi, \sigma_u) = \sigma_u \left( \sum_{n=1}^{M} k_n^{-\beta(\chi)} \right)^{-1/2}
$$

> **Normalization note**: With this convention, the actual RMS displacement is $\langle|\mathbf{u}|^2\rangle^{1/2} \approx \sigma_u / \sqrt{2}$ because $\langle\cos^2\rangle = 1/2$ over positions and phases. We treat $\sigma_u$ as a dimensionless hyperparameter (in units of $R_\text{half}$) and let calibration absorb this order-unity factor. The exact RMS is set empirically when mapping $\chi \leftrightarrow D$.

#### 4.2.3 The Vector Displacement Field

The displacement at position $\mathbf{x}$ is:

$$
\mathbf{u}(\mathbf{x}; \chi, \sigma_u) = \sum_{n=1}^{M} A_n(\chi, \sigma_u) \, \hat{\mathbf{a}}_n \, \cos(\mathbf{k}_n \cdot \mathbf{x} + \varphi_n)
$$

This is:
- Smooth in $\mathbf{x}$
- Differentiable in $\chi$ and $\sigma_u$ (through $A_n$)
- Multi-scale: modes span $[k_\text{min}, k_\text{max}]$
- Statistically isotropic (random directions)

### 4.3 Applying the Displacement

Given base positions $\mathbf{x}_i^{(0)}$, compute displacements:

$$
\mathbf{u}_i = \mathbf{u}(\mathbf{x}_i^{(0)}; \chi, \sigma_u)
$$

Apply with fractal fraction $\lambda_\text{frac}$:

$$
\mathbf{x}_i^{(1)} = \mathbf{x}_i^{(0)} + \lambda_\text{frac} \, \mathbf{u}_i
$$

**Interpretation**:
- $\lambda_\text{frac} = 0$: Pure smooth equilibrium
- $\lambda_\text{frac} = 1$: Full displacement applied
- Intermediate: Smooth blend

### 4.4 Radial Profile Preservation Options

The full 3D displacement changes the radial distribution. Three options:

#### Option A: Full Displacement (Allow Radial Changes)

Just use $\mathbf{x}_i^{(1)}$ directly. The radial CDF will deviate from the target profile.

**Use case**: When you want the displacement to affect radial structure too.

**Status**: Supported but not recommended as default.

#### Option B: Tangential-Only Displacement (Expert/Experimental)

Project out the radial component:

$$
\hat{\mathbf{r}}_i = \frac{\mathbf{x}_i^{(0)}}{|\mathbf{x}_i^{(0)}|}
$$

$$
\mathbf{u}_i^\perp = \mathbf{u}_i - (\mathbf{u}_i \cdot \hat{\mathbf{r}}_i) \, \hat{\mathbf{r}}_i
$$

$$
\mathbf{x}_i^{(1)} = \mathbf{x}_i^{(0)} + \lambda_\text{frac} \, \mathbf{u}_i^\perp
$$

Then renormalize to original radius:

$$
\mathbf{x}_i^{(1)} \leftarrow \mathbf{x}_i^{(1)} \cdot \frac{r_i^{(0)}}{|\mathbf{x}_i^{(1)}|}
$$

**Result**: Radial CDF exactly preserved per-star; only angular structure is clumpy.

> **Caveat**: The renormalization step drags points along radial rays, which can twist the angular pattern if displacements are large. This mode is exact in radius but approximate in preserving angular structure fidelity. Use for special cases where strict per-star radial preservation is required.

**Status**: Supported as expert/experimental option.

#### Option C: Rank-Based Radial Remap (Recommended Default)

Allow full 3D displacement, then remap radii to match target profile:

1. Compute displaced radii: $r_i^{(1)} = |\mathbf{x}_i^{(1)}|$
2. Sort displaced radii: $\text{idx}_\text{frac} = \text{argsort}(r^{(1)})$
3. Sort target radii: $r^{\text{target}}_\text{sorted} = \text{sort}(r^{(0)})$
4. Assign: star at rank $k$ gets radius $r^{\text{target}}_{\text{sorted},k}$
5. Rescale directions:
   $$
   \mathbf{x}_i^{\text{final}} = \hat{\mathbf{x}}_i^{(1)} \cdot r_i^{\text{mapped}}
   $$

**Result**: Exact radial CDF match; clumpy angular structure preserved.

**Note on gradients**: Sorting is piecewise constant—gradients flow through the *values* being sorted, not the permutation. This is acceptable for inference.

---

## 5. Implementation Guide <a name="5-implementation"></a>

### 5.1 Data Structures

```python
from dataclasses import dataclass
from jax import Array
import jax.numpy as jnp

@dataclass(frozen=True)
class FractalField:
    """Frozen stochastic structure for displacement field.
    
    All fields are jax.Array. This structure is generated once per
    realization and frozen (no gradients flow through it).
    
    Attributes
    ----------
    k_vecs : Array, shape (M, 3)
        Wavevectors in 1/pc.
    phases : Array, shape (M,)
        Random phases in [0, 2π].
    base_vecs : Array, shape (M, 3)
        Unit polarization vectors for each mode.
    """
    k_vecs: Array    # (M, 3)
    phases: Array    # (M,)
    base_vecs: Array # (M, 3)


@dataclass(frozen=True)
class FractalDisplacementLayer:
    """Parameters for fractal displacement field layer.
    
    Attributes
    ----------
    chi : float
        Clumpiness parameter in [1.5, 3.0]. Controls spectral slope.
        chi=1.5: highly clumpy (more small-scale power)
        chi=3.0: smooth (more large-scale power)
        Calibrated to match Goodwin-Whitworth fractal dimension D.
    lambda_frac : float
        Fractal fraction in [0, 1]. Controls blend strength.
        lambda_frac=0: pure smooth profile
        lambda_frac=1: full displacement applied
    sigma_u : float
        Dimensionless displacement amplitude scale, in units of R_half.
        The actual RMS displacement is approximately sigma_u * R_half / sqrt(2).
        Typical values: 0.1–0.5. Exact mapping to Q_CW is set by calibration.
    n_modes : int
        Number of Fourier modes. More modes = finer structure.
        Default 64 is sufficient for most applications.
    k_min_factor : float
        Minimum wavenumber as fraction of 1/R_half.
        Default 0.5 gives modes on scales ~2×R_half.
    k_max_factor : float
        Maximum wavenumber as fraction of 1/R_half.
        Default 20 gives modes on scales ~R_half/20.
    radial_mode : str
        How to handle radial profile: 'full', 'tangential', 'remap'.
        Default 'remap' preserves exact radial CDF (recommended).
        'tangential' is expert/experimental.
    virial_ratio : float
        Target Q_vir = K/|U| after velocity assignment.
    coherent_velocities : bool
        If True, velocity field correlates with displacement field.
    lambda_vel : float
        Velocity coherence strength in [0, 1].
    
    Notes
    -----
    Unlike Goodwin-Whitworth D, chi is differentiable. The mapping
    chi → Q_CW (Cartwright-Whitworth) is established via calibration.
    
    The spectral slope is β(chi) = β₀ + β₁(3 - chi) where β₀ ≈ 2.0
    and β₁ ≈ 1.5 (calibration-dependent).
    
    sigma_u is defined "up to a factor of order unity" because the
    normalization uses Σ A_n² = σ_u² but ⟨cos²⟩ = 1/2. Calibration
    absorbs this factor when mapping chi → D.
    """
    chi: float = 2.0
    lambda_frac: float = 1.0
    sigma_u: float = 0.3  # Dimensionless, in units of R_half
    n_modes: int = 64
    k_min_factor: float = 0.5
    k_max_factor: float = 20.0
    radial_mode: str = "remap"
    virial_ratio: float = 0.5
    coherent_velocities: bool = True
    lambda_vel: float = 0.3
```

### 5.2 Field Initialization

```python
import jax
import jax.numpy as jnp
from jax import random

def init_fractal_field(
    key: random.PRNGKey,
    n_modes: int,
    R_half: float,
    k_min_factor: float = 0.5,
    k_max_factor: float = 20.0,
) -> FractalField:
    """Initialize frozen stochastic structure for displacement field.
    
    Parameters
    ----------
    key : PRNGKey
    n_modes : number of Fourier modes
    R_half : half-mass radius in pc (sets scale for k_min, k_max)
    k_min_factor : k_min = k_min_factor / R_half
    k_max_factor : k_max = k_max_factor / R_half
    
    Returns
    -------
    FractalField with k_vecs, phases, base_vecs
    
    Notes
    -----
    This structure should be frozen via stop_gradient before use
    in differentiable pipelines.
    
    **Inference note**: For gradient-based inference, the FractalField
    should be treated as frozen. The k_vecs depend on R_half at
    initialization, but because we apply stop_gradient, changes in
    R_half during inference affect only the amplitude scaling and
    base profile sampling—NOT the internal phase structure of the field.
    This is intentional: letting gradients propagate through changes
    in random phases would introduce noise. If you want different
    physical clump scales, regenerate the field; don't expect AD to
    smoothly move power between Fourier modes.
    """
    key_dir, key_phase, key_pol = random.split(key, 3)
    
    # Wavenumber range
    k_min = k_min_factor / R_half
    k_max = k_max_factor / R_half
    
    # Log-spaced wavenumber magnitudes
    t = jnp.linspace(0.0, 1.0, n_modes)
    k_mags = k_min * (k_max / k_min) ** t  # (M,)
    
    # Random directions on unit sphere
    raw_dirs = random.normal(key_dir, (n_modes, 3))
    k_dirs = raw_dirs / jnp.linalg.norm(raw_dirs, axis=1, keepdims=True)
    
    # Wavevectors
    k_vecs = k_mags[:, None] * k_dirs  # (M, 3)
    
    # Random phases
    phases = random.uniform(key_phase, (n_modes,)) * (2 * jnp.pi)
    
    # Random polarization directions
    raw_pol = random.normal(key_pol, (n_modes, 3))
    base_vecs = raw_pol / jnp.linalg.norm(raw_pol, axis=1, keepdims=True)
    
    return FractalField(
        k_vecs=k_vecs,
        phases=phases,
        base_vecs=base_vecs,
    )
```

### 5.3 Amplitude Computation

```python
# Calibration constants (from GW calibration, see §6)
BETA_0 = 2.0   # Baseline spectral slope
BETA_1 = 1.5   # Slope sensitivity to chi

def compute_amplitudes(
    field: FractalField,
    chi: float,
    sigma_u: float,
) -> Array:
    """Compute mode amplitudes from chi and sigma_u.
    
    Parameters
    ----------
    field : FractalField with frozen k_vecs
    chi : clumpiness parameter in [1.5, 3.0]
    sigma_u : displacement amplitude scale in *physical* units (same units
        as positions, typically pc). The caller should pass
        `sigma_u_physical = dimensionless_sigma_u * R_half`.
        Note: the actual RMS displacement is ~sigma_u/sqrt(2) due to
        ⟨cos²⟩ = 1/2. Treat sigma_u as a hyperparameter; calibration
        absorbs the order-unity factor.
    
    Returns
    -------
    a_vecs : (M, 3) amplitude vectors for each mode
    
    Notes
    -----
    This function is differentiable in chi and sigma_u.
    Gradients do NOT flow through field (should be stop_gradient'd).
    """
    # Wavenumber magnitudes
    k_mags = jnp.linalg.norm(field.k_vecs, axis=1)  # (M,)
    
    # Spectral slope from chi
    beta = BETA_0 + BETA_1 * (3.0 - chi)
    
    # Unnormalized amplitudes: A_n ∝ k_n^(-β/2)
    raw_amps = k_mags ** (-0.5 * beta)  # (M,)
    
    # Normalize to target RMS
    # Sum of A_n^2 should equal sigma_u^2
    norm = jnp.sqrt(jnp.sum(raw_amps ** 2))
    amps = sigma_u * raw_amps / norm  # (M,)
    
    # Amplitude vectors
    a_vecs = amps[:, None] * field.base_vecs  # (M, 3)
    
    return a_vecs
```

### 5.4 Displacement Field Evaluation

```python
def evaluate_displacement(
    positions: Array,
    field: FractalField,
    a_vecs: Array,
) -> Array:
    """Evaluate displacement field at given positions.
    
    Parameters
    ----------
    positions : (N, 3) positions in pc
    field : FractalField with k_vecs and phases
    a_vecs : (M, 3) amplitude vectors from compute_amplitudes
    
    Returns
    -------
    displacements : (N, 3) displacement vectors in pc
    
    Notes
    -----
    u(x) = Σ_n a_n cos(k_n · x + φ_n)
    
    This is fully differentiable in positions and a_vecs.
    """
    # k_n · x_i: shape (N, M)
    dot_products = jnp.einsum("nd,md->nm", positions, field.k_vecs)
    
    # Add phases: (N, M)
    arguments = dot_products + field.phases[None, :]
    
    # Cosine terms: (N, M)
    cos_terms = jnp.cos(arguments)
    
    # Sum over modes: (N, M) @ (M, 3) → (N, 3)
    displacements = cos_terms @ a_vecs
    
    return displacements
```

### 5.5 Applying Displacement with Radial Modes

```python
def apply_displacement(
    positions: Array,
    displacements: Array,
    lambda_frac: float,
    target_radii: Array,
    mode: str = "remap",
) -> Array:
    """Apply displacement field to positions.
    
    Parameters
    ----------
    positions : (N, 3) base positions from smooth profile
    displacements : (N, 3) displacement vectors
    lambda_frac : blend fraction in [0, 1]
    target_radii : (N,) target radii for 'remap' mode
    mode : 'full', 'tangential', or 'remap'
    
    Returns
    -------
    positions_out : (N, 3) displaced positions
    """
    if mode == "full":
        return positions + lambda_frac * displacements
    
    elif mode == "tangential":
        # Project out radial component
        r = jnp.linalg.norm(positions, axis=1, keepdims=True)
        r_hat = positions / jnp.maximum(r, 1e-10)
        
        # Tangential displacement
        u_radial = jnp.sum(displacements * r_hat, axis=1, keepdims=True)
        u_tangential = displacements - u_radial * r_hat
        
        # Apply tangential displacement
        pos_displaced = positions + lambda_frac * u_tangential
        
        # Renormalize to original radius
        r_new = jnp.linalg.norm(pos_displaced, axis=1, keepdims=True)
        pos_out = pos_displaced * (r / jnp.maximum(r_new, 1e-10))
        
        return pos_out
    
    elif mode == "remap":
        # Full displacement
        pos_displaced = positions + lambda_frac * displacements
        
        # Rank-based radial remap
        r_displaced = jnp.linalg.norm(pos_displaced, axis=1)
        
        # Sort indices
        # Note: sorting is piecewise-constant in the permutation; gradients
        # flow through the *values* being sorted, not which star is rank k.
        # We accept non-smooth gradients wrt permutations and only rely on
        # smoothness in the radii values themselves.
        idx_displaced = jnp.argsort(r_displaced)
        target_sorted = jnp.sort(target_radii)
        
        # Map: star at rank k gets target radius at rank k
        r_mapped = jnp.zeros_like(r_displaced)
        r_mapped = r_mapped.at[idx_displaced].set(target_sorted)
        
        # Rescale directions
        r_hat = pos_displaced / jnp.maximum(
            r_displaced[:, None], 1e-10
        )
        pos_out = r_hat * r_mapped[:, None]
        
        return pos_out
    
    else:
        raise ValueError(f"Unknown radial mode: {mode}")
```

### 5.6 Complete Generator

```python
from jaxstro.constants import G_stellar as G

def generate_fractal_ic(
    key: random.PRNGKey,
    N_stars: int,
    M_total: float,
    R_half: float,
    profile: str,
    frac_params: FractalDisplacementLayer,
    imf_params,
    field: FractalField = None,
) -> ClusterState:
    """Generate cluster IC with fractal displacement field.
    
    Parameters
    ----------
    key : PRNGKey
    N_stars : number of stars
    M_total : total mass in M_sun
    R_half : half-mass radius in pc
    profile : density profile type ('plummer', 'king', 'eff')
    frac_params : FractalDisplacementLayer parameters
    imf_params : IMF parameters from progenax.imf
    field : optional pre-initialized FractalField. If None, creates new one.
        For long-running inference loops, precompute and pass in a frozen
        FractalField to avoid reinitialization overhead.
    
    Returns
    -------
    ClusterState with masses, positions, velocities
    
    Notes
    -----
    **Inference pattern**: In long-running inference (HMC, NUTS), you may want
    to precompute a FractalField once and pass it in, rather than reinitializing
    each iteration. The field structure is frozen anyway; only chi, lambda_frac,
    and profile params receive gradients.
    
    **Velocity assignment**: Uses O(N²) potential energy calculation. This is
    acceptable for IC generation (N ~ 10³–10⁴) but is not intended for every
    time-step in an N-body loop.
    """
    # Split keys
    key_imf, key_pos, key_field, key_vel = random.split(key, 4)
    
    # Clamp lambda parameters to valid range [0, 1]
    lambda_frac = jnp.clip(frac_params.lambda_frac, 0.0, 1.0)
    lambda_vel = jnp.clip(frac_params.lambda_vel, 0.0, 1.0)
    
    # ─────────────────────────────────────────────────────────────
    # Step 1: Draw masses from IMF
    # ─────────────────────────────────────────────────────────────
    masses = sample_imf(key_imf, imf_params, N_stars)
    masses = masses * (M_total / jnp.sum(masses))
    
    # ─────────────────────────────────────────────────────────────
    # Step 2: Sample smooth base profile
    # ─────────────────────────────────────────────────────────────
    positions_base = sample_density_profile(key_pos, N_stars, profile, R_half)
    radii_base = jnp.linalg.norm(positions_base, axis=1)
    
    # ─────────────────────────────────────────────────────────────
    # Step 3: Initialize and freeze displacement field (or use provided)
    # ─────────────────────────────────────────────────────────────
    if field is None:
        field = init_fractal_field(
            key_field,
            n_modes=frac_params.n_modes,
            R_half=R_half,
            k_min_factor=frac_params.k_min_factor,
            k_max_factor=frac_params.k_max_factor,
        )
        # Freeze stochastic structure (no gradients)
        field = jax.tree_util.tree_map(jax.lax.stop_gradient, field)
    
    # ─────────────────────────────────────────────────────────────
    # Step 4: Compute amplitudes (differentiable in chi, sigma_u)
    # ─────────────────────────────────────────────────────────────
    sigma_u_physical = frac_params.sigma_u * R_half  # Convert to pc
    a_vecs = compute_amplitudes(field, frac_params.chi, sigma_u_physical)
    
    # ─────────────────────────────────────────────────────────────
    # Step 5: Evaluate and apply displacement
    # ─────────────────────────────────────────────────────────────
    displacements = evaluate_displacement(positions_base, field, a_vecs)
    
    positions = apply_displacement(
        positions_base,
        displacements,
        lambda_frac,  # Use clamped value
        target_radii=radii_base,
        mode=frac_params.radial_mode,
    )
    
    # Recenter positions to COM (finite-N realizations can drift)
    M_total_actual = jnp.sum(masses)
    x_com = jnp.sum(masses[:, None] * positions, axis=0) / M_total_actual
    positions = positions - x_com
    
    # ─────────────────────────────────────────────────────────────
    # Step 6: Assign velocities
    # NOTE: compute_potential_energy is O(N²); acceptable for IC generation
    # (N ~ 10³–10⁴), but not intended for every time-step in an N-body loop.
    # ─────────────────────────────────────────────────────────────
    velocities = assign_fractal_velocities(
        key_vel,
        positions,
        masses,
        field,
        a_vecs,
        frac_params,
        G,
        lambda_vel=lambda_vel,  # Use clamped value
    )
    
    return ClusterState(
        masses=masses,
        positions=positions,
        velocities=velocities,
    )
```

---

## 6. Calibration to GW/McLuster Statistics <a name="6-calibration"></a>

### 6.1 The Calibration Problem

The displacement field has parameters $(\chi, \sigma_u, \lambda_\text{frac})$ that don't directly map to GW's fractal dimension $D$. We need to establish:

$$
\chi \leftrightarrow D \quad \text{such that} \quad Q_\text{CW}(\chi) \approx Q_\text{CW}(D)
$$

where $Q_\text{CW}$ is the Cartwright-Whitworth substructure parameter.

### 6.2 Calibration Protocol

#### v1 Scope: Minimal Calibration Grid

For the initial implementation, use a small calibration grid:

> **Implementation note (for devs)**: For v1, we will run the calibration protocol **offline** (not as part of the public API) and commit a small lookup table to `progenax.data.fdf_calibration`. Runtime code should only implement the interpolation helpers (`chi_from_D`, `sigma_u_from_D`) and load this pre-computed table. Do NOT build a full auto-calibration framework into the library in this first pass.

```python
def v1_calibration_protocol():
    """
    v1 Calibration (minimal, get-it-working scope):
    
    1. Generate GW reference ensembles:
       - For D in [1.6, 2.0, 2.4, 2.8, 3.0]:  # 5 values only
         - Generate 50 realizations with N=1000
         - Compute Q_CW, σ_Σ/⟨Σ⟩
       
    2. Generate FDF ensembles:
       - Use chi = D as starting point (often close to optimal)
       - Tune sigma_u(D) by hand or small grid search:
         - sigma_u in [0.15, 0.25, 0.35, 0.45]
         - Find sigma_u that minimizes |Q_CW(FDF) - Q_CW(GW)|
       
    3. Store as lookup table:
       - D → (chi, sigma_u) with chi ≈ D
       
    Target: Q_CW match within ~15% is acceptable for v1.
    """
    pass


def full_calibration_protocol():
    """
    Full Calibration (for methods paper / production):
    
    1. Generate GW reference ensembles:
       - For D in [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]:
         - Generate 100 realizations with N=1000
         - Compute Q_CW, σ_Σ/⟨Σ⟩, correlation function ξ(r)
       
    2. Generate FDF ensembles:
       - For chi in [1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
         - For sigma_u in [0.1, 0.2, 0.3, 0.4, 0.5]:
           - Generate 100 realizations with N=1000, lambda_frac=1
           - Compute same statistics
    
    3. Find optimal mapping:
       - For each D, find (chi, sigma_u) that minimizes:
         |Q_CW(FDF) - Q_CW(GW)| + w * |σ_Σ(FDF) - σ_Σ(GW)|
       
    4. Fit parametric mapping:
       - chi(D) ≈ D (often close to identity)
       - sigma_u(D) may need adjustment
       
    5. Store calibration table in progenax.data
    """
    pass
```

### 6.3 Expected Calibration Results

Based on the physics, we expect:

| GW $D$ | FDF $\chi$ | $\sigma_u / R_\text{half}$ | $Q_\text{CW}$ |
|--------|------------|---------------------------|---------------|
| 1.6 | ~1.6 | ~0.4 | ~0.45 |
| 2.0 | ~2.0 | ~0.3 | ~0.55 |
| 2.4 | ~2.4 | ~0.25 | ~0.65 |
| 2.8 | ~2.8 | ~0.2 | ~0.75 |
| 3.0 | ~3.0 | ~0.1 | ~0.80 |

The $\chi \approx D$ correspondence is expected because both control the relative power on different scales.

### 6.4 Calibration Data Structure

```python
@dataclass(frozen=True)
class FDFCalibration:
    """Calibration mapping between GW D and FDF parameters.
    
    Loaded from progenax.data.fdf_calibration.
    """
    D_values: Array          # Reference GW D values
    chi_values: Array        # Corresponding chi values
    sigma_u_values: Array    # Corresponding sigma_u / R_half
    Q_CW_target: Array       # Target Q_CW for each D
    
    def chi_from_D(self, D: float) -> float:
        """Interpolate chi from target D."""
        return jnp.interp(D, self.D_values, self.chi_values)
    
    def sigma_u_from_D(self, D: float) -> float:
        """Interpolate sigma_u from target D."""
        return jnp.interp(D, self.D_values, self.sigma_u_values)
```

### 6.5 User-Facing API with D Parameter

```python
def fractal_layer_from_D(
    D: float,
    virial_ratio: float = 0.5,
    coherent_velocities: bool = True,
    lambda_frac: float = 1.0,
) -> FractalDisplacementLayer:
    """Create FractalDisplacementLayer from GW-style D parameter.
    
    Parameters
    ----------
    D : float in [1.5, 3.0]
        Target fractal dimension (GW convention)
    virial_ratio : target Q_vir
    coherent_velocities : whether to correlate velocities with structure
    lambda_frac : blend fraction
    
    Returns
    -------
    FractalDisplacementLayer with calibrated chi and sigma_u
    
    Notes
    -----
    Uses calibration table to map D → (chi, sigma_u) such that
    the resulting Q_CW matches GW fractals with the same D.
    """
    calibration = load_fdf_calibration()
    
    chi = calibration.chi_from_D(D)
    sigma_u = calibration.sigma_u_from_D(D)
    
    return FractalDisplacementLayer(
        chi=chi,
        lambda_frac=lambda_frac,
        sigma_u=sigma_u,
        virial_ratio=virial_ratio,
        coherent_velocities=coherent_velocities,
    )
```

---

## 7. Velocity Structure <a name="7-velocities"></a>

### 7.1 Base Velocities

For the smooth base profile, sample velocities from the equilibrium distribution function via `progenax.kinematics`:

```python
def sample_base_velocities(
    key: random.PRNGKey,
    positions: Array,
    profile: str,
    R_half: float,
    M_total: float,
    G: float,
) -> Array:
    """Sample velocities from equilibrium DF.
    
    Delegates to progenax.kinematics.api.sample_velocities.
    """
    return sample_velocities_for_profile(
        key, positions, profile, R_half, M_total, G
    )
```

### 7.2 Coherent Velocity Perturbation

To create coherent motions within subclumps (stars in dense regions moving together), we add a velocity perturbation correlated with the displacement field:

$$
\mathbf{v}_i = \mathbf{v}_i^{(\text{base})} + \lambda_\text{vel} \cdot \sigma_v \cdot \hat{\mathbf{u}}(\mathbf{x}_i)
$$

where:
- $\mathbf{v}_i^{(\text{base})}$ is from the equilibrium DF
- $\lambda_\text{vel}$ controls coherence strength
- $\sigma_v$ is the velocity dispersion scale
- $\hat{\mathbf{u}}$ is the normalized displacement field

**Physical interpretation**: Stars displaced in the same direction by the "turbulent" field also share a coherent velocity component.

```python
def assign_fractal_velocities(
    key: random.PRNGKey,
    positions: Array,
    masses: Array,
    field: FractalField,
    a_vecs: Array,
    frac_params: FractalDisplacementLayer,
    G: float,
    lambda_vel: float = None,
) -> Array:
    """Assign velocities with optional coherent structure.
    
    Parameters
    ----------
    key : PRNGKey
    positions : (N, 3) final positions in pc
    masses : (N,) stellar masses in M_sun
    field : frozen FractalField
    a_vecs : (M, 3) amplitude vectors
    frac_params : FractalDisplacementLayer with velocity params
    G : gravitational constant in units consistent with positions/masses
        (typically G_stellar giving velocities in pc/Myr)
    lambda_vel : optional override for velocity coherence strength [0, 1].
        If None, uses frac_params.lambda_vel.
    
    Returns
    -------
    velocities : (N, 3) in same units implied by G (typically pc/Myr for
        positions in pc, masses in M_sun, and G = G_stellar)
    
    Notes
    -----
    Uses O(N²) potential energy calculation internally. Acceptable for
    IC generation but not for every N-body timestep.
    """
    N = masses.shape[0]
    M_total = jnp.sum(masses)
    
    # Use override if provided, else use frac_params
    lam_vel = lambda_vel if lambda_vel is not None else frac_params.lambda_vel
    
    # Compute potential energy for virial scaling
    U_total = compute_potential_energy(positions, masses, G)
    K_target = frac_params.virial_ratio * jnp.abs(U_total)
    sigma_v = jnp.sqrt(2 * K_target / M_total)
    
    # Base velocities: isotropic Gaussian
    key, subkey = random.split(key)
    v_base = random.normal(subkey, (N, 3)) * sigma_v / jnp.sqrt(3)
    
    # Coherent perturbation from displacement field
    if frac_params.coherent_velocities and lam_vel > 0:
        # Evaluate displacement at final positions
        u = evaluate_displacement(positions, field, a_vecs)
        u_norm = jnp.linalg.norm(u, axis=1, keepdims=True)
        u_hat = u / jnp.maximum(u_norm, 1e-10)
        
        # Add coherent component
        v_coherent = lam_vel * sigma_v * u_hat
        velocities = v_base + v_coherent
    else:
        velocities = v_base
    
    # Rescale to exact target virial ratio
    K_actual = 0.5 * jnp.sum(masses[:, None] * velocities**2)
    scale = jnp.sqrt(K_target / K_actual)
    velocities = velocities * scale
    
    # Remove COM motion
    v_com = jnp.sum(masses[:, None] * velocities, axis=0) / M_total
    velocities = velocities - v_com
    
    return velocities
```

### 7.3 Alternative: Separate Velocity Field

For more control, use a separate velocity field $\mathbf{w}(\mathbf{x})$ with different spectral parameters:

```python
def init_velocity_field(
    key: random.PRNGKey,
    n_modes: int,
    R_half: float,
    # Can have different k_min, k_max, beta than position field
) -> FractalField:
    """Initialize separate field for velocity structure."""
    return init_fractal_field(key, n_modes, R_half)
```

This allows independent control of position and velocity structure scales.

---

## 8. Validation Tests <a name="8-validation"></a>

### 8.1 Statistical Match to GW

```python
def test_Q_CW_calibration():
    """Verify FDF reproduces GW Q_CW statistics."""
    from progenax.diagnostics import compute_q_parameter
    
    D_values = [1.6, 2.0, 2.4, 2.8]
    
    for D in D_values:
        # Generate GW reference
        Q_gw = []
        for seed in range(50):
            key = random.PRNGKey(seed)
            pos_gw = generate_gw_fractal(key, N=1000, D=D)
            Q_gw.append(compute_q_parameter(np.array(pos_gw)))
        Q_gw_mean = np.mean(Q_gw)
        
        # Generate FDF with calibrated params
        Q_fdf = []
        for seed in range(50):
            key = random.PRNGKey(seed + 1000)
            frac = fractal_layer_from_D(D)
            cluster = generate_fractal_ic(
                key, N_stars=1000, M_total=1000.0, R_half=1.0,
                profile="plummer", frac_params=frac, imf_params=kroupa_imf()
            )
            Q_fdf.append(compute_q_parameter(np.array(cluster.positions)))
        Q_fdf_mean = np.mean(Q_fdf)
        
        # Should match within ~10%
        assert abs(Q_fdf_mean - Q_gw_mean) / Q_gw_mean < 0.15, \
            f"D={D}: Q_GW={Q_gw_mean:.3f}, Q_FDF={Q_fdf_mean:.3f}"
```

### 8.2 Differentiability Test

```python
def test_gradient_flow():
    """Verify gradients flow through chi and lambda_frac."""
    
    def loss_fn(chi, lambda_frac):
        key = random.PRNGKey(0)
        
        frac = FractalDisplacementLayer(
            chi=chi,
            lambda_frac=lambda_frac,
            sigma_u=0.3,
        )
        
        cluster = generate_fractal_ic(
            key, N_stars=500, M_total=500.0, R_half=1.0,
            profile="plummer", frac_params=frac,
            imf_params=kroupa_imf()
        )
        
        # Some differentiable summary statistic
        # (mean radius of 10 most massive stars)
        mass_order = jnp.argsort(-cluster.masses)[:10]
        r_massive = jnp.mean(jnp.linalg.norm(
            cluster.positions[mass_order], axis=1
        ))
        
        return r_massive
    
    # Compute gradients
    grad_fn = jax.grad(loss_fn, argnums=(0, 1))
    
    chi, lambda_frac = 2.0, 0.5
    d_chi, d_lambda = grad_fn(chi, lambda_frac)
    
    # Gradients should be non-zero
    assert jnp.abs(d_chi) > 1e-6, f"d/d(chi) = {d_chi}, expected non-zero"
    assert jnp.abs(d_lambda) > 1e-6, f"d/d(lambda_frac) = {d_lambda}"
    
    print(f"✓ Gradients: d/d(chi)={d_chi:.4f}, d/d(λ_frac)={d_lambda:.4f}")
```

### 8.3 Radial Profile Preservation

```python
def test_radial_profile_remap():
    """Verify 'remap' mode preserves radial CDF."""
    key = random.PRNGKey(42)
    
    # Generate base profile radii
    key, subkey = random.split(key)
    pos_base = sample_density_profile(subkey, 1000, "plummer", R_half=1.0)
    r_base = jnp.linalg.norm(pos_base, axis=1)
    
    # Generate fractal IC with remap
    frac = FractalDisplacementLayer(
        chi=1.6, lambda_frac=1.0, sigma_u=0.4, radial_mode="remap"
    )
    cluster = generate_fractal_ic(
        key, N_stars=1000, M_total=1000.0, R_half=1.0,
        profile="plummer", frac_params=frac, imf_params=kroupa_imf()
    )
    r_frac = jnp.linalg.norm(cluster.positions, axis=1)
    
    # Sorted radii should match exactly
    r_base_sorted = jnp.sort(r_base)
    r_frac_sorted = jnp.sort(r_frac)
    
    assert jnp.allclose(r_base_sorted, r_frac_sorted, rtol=1e-5), \
        "Radial CDF not preserved in 'remap' mode"
```

### 8.4 Velocity Coherence

```python
def test_velocity_coherence():
    """Verify coherent velocities correlate with displacement."""
    key = random.PRNGKey(0)
    
    frac = FractalDisplacementLayer(
        chi=2.0, lambda_frac=1.0, sigma_u=0.3,
        coherent_velocities=True, lambda_vel=0.5,
    )
    
    cluster = generate_fractal_ic(
        key, N_stars=1000, M_total=1000.0, R_half=1.0,
        profile="plummer", frac_params=frac, imf_params=kroupa_imf()
    )
    
    # Neighbors should have correlated velocities
    # (This is a simplified test; full test would use k-NN)
    from scipy.spatial import cKDTree
    
    tree = cKDTree(np.array(cluster.positions))
    
    correlations = []
    for i in range(100):
        # Find 5 nearest neighbors
        _, neighbors = tree.query(cluster.positions[i], k=6)
        neighbors = neighbors[1:]  # Exclude self
        
        # Velocity correlation with neighbors
        v_i = cluster.velocities[i]
        v_neighbors = cluster.velocities[neighbors]
        
        # Cosine similarity
        cos_sim = jnp.mean(
            jnp.sum(v_i * v_neighbors, axis=1) / 
            (jnp.linalg.norm(v_i) * jnp.linalg.norm(v_neighbors, axis=1))
        )
        correlations.append(float(cos_sim))
    
    mean_corr = np.mean(correlations)
    
    # Should be positive (neighbors moving similarly)
    assert mean_corr > 0.1, f"Mean velocity correlation = {mean_corr:.3f}"
```

### 8.5 Lambda Blending Smoothness

```python
def test_lambda_frac_smoothness():
    """Verify smooth interpolation in lambda_frac."""
    key = random.PRNGKey(0)
    
    lambda_values = jnp.linspace(0.0, 1.0, 11)
    Q_values = []
    
    for lam in lambda_values:
        frac = FractalDisplacementLayer(
            chi=2.0, lambda_frac=float(lam), sigma_u=0.3
        )
        cluster = generate_fractal_ic(
            key, N_stars=1000, M_total=1000.0, R_half=1.0,
            profile="plummer", frac_params=frac, imf_params=kroupa_imf()
        )
        Q = compute_q_parameter(np.array(cluster.positions))
        Q_values.append(Q)
    
    Q_values = np.array(Q_values)
    
    # Q should vary smoothly (no jumps > 20% of range)
    Q_range = Q_values.max() - Q_values.min()
    max_jump = np.max(np.abs(np.diff(Q_values)))
    
    assert max_jump < 0.25 * Q_range, \
        f"λ_frac blending not smooth: max jump = {max_jump:.3f}"
    
    # Q should decrease with lambda_frac (more substructure)
    assert Q_values[0] > Q_values[-1], \
        "Q should decrease with increasing lambda_frac"
```

---

## 9. v1 Scope and Integration <a name="9-scope"></a>

### 9.1 v1 Implementation Scope

> **IMPLEMENTATION DIRECTIVE**: Implement exactly the items in this section first. Treat everything labeled "Full Calibration," "v2+," or "Alternative" as future work. Do not over-engineer.

To keep the initial implementation focused and shippable, v1 includes:

**Positions (implement these):**
- `FractalField` and `FractalDisplacementLayer` dataclasses (§5.1)
- `init_fractal_field` (§5.2)
- `compute_amplitudes` (§5.3)
- `evaluate_displacement` (§5.4)
- `apply_displacement` with modes `'full'` and `'remap'` (default) (§5.5)
- `'tangential'` mode: include but mark as experimental

**Velocities (implement these):**
- `assign_fractal_velocities` with base Gaussian + optional coherent component (§7)
- Global virial rescale to target $Q_\text{vir}$
- COM removal for both positions and velocities

**Generator (implement this):**
- `generate_fractal_ic` as specified (§5.6)
- Support optional pre-initialized `FractalField` argument

**Calibration (stub for now):**
- `load_fdf_calibration()` → can return trivial identity ($\chi = D$, nominal $\sigma_u$)
- `fractal_layer_from_D()` helper (§9.3)
- Actual calibration will be run offline and committed as static data

**Tests (implement these):**
- Gradient flow test (§8.2)
- Radial CDF preservation test (§8.3)
- Basic $Q_\text{CW}$ trend vs $\lambda_\text{frac}$ or $\chi$ (lightweight version of §8.1/§8.5)

**Deferred to v2+:**
- Full calibration grid with $\sigma_\Sigma/\langle\Sigma\rangle$ matching
- Separate velocity field with independent spectrum
- Advanced correlation function calibration
- Auto-calibration framework

### 9.2 Integration with Mass Segregation

The FDF method is **independent of mass segregation**. The recommended pipeline:

```python
# 1. Sample masses from IMF
masses = sample_imf(key_imf, imf_params, N_stars)

# 2. Generate fractal positions/velocities (FDF method)
#    This assigns random masses to displaced positions
cluster_frac = generate_fractal_ic(
    key, N_stars, M_total, R_half, profile,
    frac_params, imf_params
)

# 3. Optionally apply mass segregation as a separate transform
#    (using Method A or Method B from the segregation guide)
positions_seg, velocities_seg = assign_orbits_mass_seg(
    key_seg,
    cluster_frac.masses,
    orbit_positions=cluster_frac.positions,
    orbit_velocities=cluster_frac.velocities,
    energies=compute_energies(cluster_frac),
    method="blend",
    params=BlendingSegregationParams(lambda_seg=0.5),
)
```

This keeps the two capabilities modular and composable.

### 9.3 User-Facing API with D Parameter

For users who think in terms of GW-style fractal dimension $D$:

```python
def fractal_layer_from_D(
    D: float,
    virial_ratio: float = 0.5,
    coherent_velocities: bool = True,
    lambda_frac: float = 1.0,
    lambda_vel: float = 0.3,
) -> FractalDisplacementLayer:
    """Create FractalDisplacementLayer from GW-style D parameter.
    
    Parameters
    ----------
    D : float in [1.5, 3.0]
        Target fractal dimension (GW convention)
    virial_ratio : target Q_vir
    coherent_velocities : whether to correlate velocities with structure
    lambda_frac : blend fraction (0 = smooth, 1 = full fractal)
    lambda_vel : velocity coherence strength
    
    Returns
    -------
    FractalDisplacementLayer with calibrated chi and sigma_u
    
    Notes
    -----
    Uses calibration table to map D → (chi, sigma_u) such that
    the resulting Q_CW statistically matches GW fractals with the same D.
    
    This is the recommended user-facing API for most applications.
    """
    calibration = load_fdf_calibration()
    
    chi = calibration.chi_from_D(D)
    sigma_u = calibration.sigma_u_from_D(D)
    
    return FractalDisplacementLayer(
        chi=chi,
        lambda_frac=lambda_frac,
        sigma_u=sigma_u,
        virial_ratio=virial_ratio,
        coherent_velocities=coherent_velocities,
        lambda_vel=lambda_vel,
    )


# Usage example:
frac_params = fractal_layer_from_D(D=1.6, virial_ratio=0.3)
cluster = generate_fractal_ic(
    key, N_stars=1000, M_total=500.0, R_half=0.5,
    profile="plummer", frac_params=frac_params, imf_params=kroupa_imf()
)
```

### 9.4 Differentiable Inference Pipeline

The FDF design is shaped for gradient-based inference:

```python
def inference_loss(theta, key, data):
    """Example loss for fitting cluster parameters to data.
    
    theta includes: chi, lambda_frac, profile params, etc.
    All are continuous and differentiable.
    """
    # Unpack parameters
    chi, lambda_frac, R_half = theta['chi'], theta['lambda_frac'], theta['R_half']
    
    frac_params = FractalDisplacementLayer(
        chi=chi,
        lambda_frac=lambda_frac,
        sigma_u=calibration.sigma_u_from_chi(chi),
    )
    
    # Generate IC (differentiable in chi, lambda_frac, R_half)
    cluster = generate_fractal_ic(key, N, M, R_half, profile, frac_params, imf)
    
    # Forward model: IC → N-body → render → mock observables
    evolved = integrate_nbody(cluster, t_final)
    mock_obs = render_cluster(evolved, distance, extinction)
    
    # Compare to data
    return negative_log_likelihood(mock_obs, data)

# Run HMC/NUTS
posterior = run_hmc(inference_loss, theta_init, key, data)
```

The key differentiable knobs are:
- $\chi$: clumpiness / fractal dimension (via $\beta(\chi)$ in amplitudes)
- $\lambda_\text{frac}$: fractal fraction (linear blend)
- $R_\text{half}$: profile scale (through amplitude scaling and profile sampling)
- $Q_\text{vir}$: virial ratio (through velocity scaling)

---

## Appendix A: Comparison with GW2004

### A.1 Feature Comparison

| Feature | GW2004 Tree | FDF (This Guide) |
|---------|-------------|------------------|
| **Differentiable** | ✗ | ✓ |
| **Fixed N** | ✗ (needs subsampling) | ✓ |
| **Clumpiness control** | D (discrete) | χ (continuous, calibrated to D) |
| **Physical basis** | Abstract box subdivision | Turbulent displacement |
| **Radial profile** | Requires mapping | Native or remap |
| **Velocity coherence** | Via ancestry tree | Via field correlation |
| **JAX-native** | ✗ | ✓ |
| **Calibration needed** | No | Yes (χ → Q_CW) |
| **Exact match to GW** | N/A | No (statistical match only) |

### A.2 When to Use Which

**Use FDF (this guide) when:**
- Differentiable inference is required
- Cluster parameters need gradients
- Integration with JAX pipelines (Stellax/Gravax)
- Exact N_stars is important

**Use GW2004 (legacy) when:**
- Standalone forward Monte Carlo
- Exact replication of published GW/McLuster results
- No autodiff required

---

## Appendix B: Physical Connection to Turbulence

### B.1 Turbulent Velocity Power Spectrum

In supersonic isothermal turbulence (Federrath et al. 2010):

$$
E(k) \propto k^{-\beta}, \quad \beta \approx 3.8\text{–}4.0
$$

The density field inherits structure from converging velocity flows.

### B.2 Our Spectral Slope Mapping

We use:
$$
\beta(\chi) = \beta_0 + \beta_1 (3 - \chi)
$$

With $\beta_0 = 2.0$ and $\beta_1 = 1.5$:
- $\chi = 1.5$ → $\beta = 4.25$ (ISM-like, highly compressive)
- $\chi = 3.0$ → $\beta = 2.0$ (more power on large scales)

This range brackets observed turbulent spectra and maps naturally to GW $D$.

### B.3 Displacement vs Velocity Field

In turbulence, the *velocity* field has the power spectrum; stars form and *inherit* positions from the density peaks created by converging flows.

Our displacement field $\mathbf{u}(\mathbf{x})$ is conceptually the *integrated effect* of turbulent velocities on stellar positions during the star formation epoch:

$$
\mathbf{x}_\star \approx \mathbf{x}_0 + \int_0^{t_\text{SF}} \mathbf{v}_\text{turb}(\mathbf{x}(t)) \, dt
$$

The Fourier-mode representation with power-law spectrum captures this physics in a differentiable form.

---

## Appendix C: Parameter Quick Reference

### Clumpiness χ
```
χ = 1.5  → Highly clumpy (small-scale power dominant)
χ = 2.0  → Moderate substructure
χ = 2.5  → Mild substructure
χ = 3.0  → Nearly smooth (large-scale power dominant)

Calibrated to match GW D parameter.
Differentiable.
```

### Fractal Fraction λ_frac
```
λ_frac = 0.0  → Pure smooth equilibrium profile
λ_frac = 0.5  → 50% displacement applied
λ_frac = 1.0  → Full displacement applied

Differentiable. Smooth interpolation for inference.
```

### Amplitude Scale σ_u
```
σ_u = 0.1  → Subtle perturbations
σ_u = 0.3  → Typical clumpiness (default)
σ_u = 0.5  → Strong substructure

Dimensionless parameter in units of R_half.
Actual RMS displacement ≈ σ_u × R_half / √2.
Exact mapping to Q_CW set by calibration.
Differentiable.
```

### Number of Modes M
```
M = 32   → Coarse structure only
M = 64   → Default, adequate for most uses
M = 128  → Fine structure, higher cost

Static integer. More modes = finer clumps.
```

### Radial Mode
```
'full'       → Full 3D displacement (radial CDF changes)
'tangential' → Tangential only (exact radial CDF, expert/experimental)
'remap'      → Full + rank remap (exact radial CDF, RECOMMENDED DEFAULT)
```

### Velocity Coherence λ_vel
```
λ_vel = 0.0  → Independent random velocities
λ_vel = 0.3  → Moderate coherence (default)
λ_vel = 0.5  → Strong coherence
λ_vel = 1.0  → Velocity dominated by displacement field
```

---

## Appendix D: Key References

| Paper | Contribution | ADS |
|-------|--------------|-----|
| Goodwin & Whitworth (2004) | Original fractal tree method | [ADS](https://ui.adsabs.harvard.edu/abs/2004A&A...413..929G) |
| Cartwright & Whitworth (2004) | Q parameter for substructure | [ADS](https://ui.adsabs.harvard.edu/abs/2004MNRAS.348..589C) |
| Küpper et al. (2011) | McLuster implementation | [ADS](https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.2300K) |
| Larson (1981) | Velocity-size relation in clouds | [ADS](https://ui.adsabs.harvard.edu/abs/1981MNRAS.194..809L) |
| Mac Low & Klessen (2004) | Turbulent star formation review | [ADS](https://ui.adsabs.harvard.edu/abs/2004RvMP...76..125M) |
| Federrath et al. (2010) | Turbulence driving and statistics | [ADS](https://ui.adsabs.harvard.edu/abs/2010A&A...512A..81F) |

---

*Document v1.2 — Final pre-implementation revision. Added: R_half inference note (frozen field, gradients don't flow through k-vectors); sorting piecewise-constant comment in apply_displacement; optional pre-initialized FractalField parameter for long inference runs; explicit lambda clamping to [0,1]; O(N²) potential energy caveat; velocity units clarification; explicit implementation directive in §9.1 with exact list of v1 items. Calibration runs offline, commits static lookup table. Ready for Claude Code implementation.*
