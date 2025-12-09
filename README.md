# progenax

Differentiable initial conditions for N-body simulations in JAX.

Part of the **jaxstro ecosystem** - providing IC generation that can be differentiated through for gradient-based inference.

## Status

**Phase 1 Complete**: 9,400 LOC source code, 432 tests passing (unit: 310, integration: 42, validation: 80)

## Features

### Spatial Profiles

| Profile | Class | Description | Reference |
|---------|-------|-------------|-----------|
| **Plummer** | `PlummerProfile` | Plummer (1911) sphere ρ ∝ (1 + r²/a²)^(-5/2) | Plummer (1911) |
| **King** | `KingProfile` | King (1966) lowered isothermal model | King (1966) |
| **EFF** | `EFFProfile` | Elson-Fall-Freeman (1987) truncated power law | EFF (1987) |
| **King ODE** | `solve_king_profile()` | Numerical King profile via Diffrax | King (1966) |

### Velocity Distribution Functions

| DF | Class | Method | Notes |
|----|-------|--------|-------|
| **Plummer DF** | `PlummerVelocityDF` | Beta(3/2, 9/2) sampling | Exact, no rejection |
| **King DF** | `KingVelocityDF` | Gaussian + escape velocity cutoff | Lowered Maxwellian |
| **EFF DF** | `EFFVelocityDF` | Isotropic Gaussian σ ∝ √(GM/a) | Virial-based |

**Velocity Transforms:**

| Transform | Function | Description |
|-----------|----------|-------------|
| **Osipkov-Merritt** | `apply_osipkov_merritt()` | Radial anisotropy β(r) = r²/(r²+r_a²) |
| **Solid Body Rotation** | `apply_solid_body_rotation()` | v_rot = ω × r |
| **Differential Rotation** | `apply_differential_rotation()` | Peaked rotation curve |

### Initial Mass Functions

| IMF | Class | Description | Reference |
|-----|-------|-------------|-----------|
| **Power Law** | `PowerLawIMF` | Single/broken power laws | Salpeter (1955) |
| **Kroupa** | `PowerLawIMF.kroupa()` | 3-segment Kroupa (2001) | Kroupa (2001) |
| **Maschberger** | `Maschberger` | Smooth transitions | Maschberger (2013) |
| **Chabrier** | `ChabrierIMF` | Log-normal + power law | Chabrier (2003) |
| **Truncated** | `TruncatedIMF` | Mass limits wrapper | - |
| **Binary** | `BinaryIMF` | Primary + secondary sampling | - |
| **IGIMF** | `IGIMF` | Integrated galactic IMF | Weidner+ (2013) |
| **Environment** | `EnvironmentIMF` | Metallicity/SFR dependent | Jeans mass theory |

### Binary Orbital Mechanics

| Component | Class/Function | Description |
|-----------|----------------|-------------|
| **Keplerian Elements** | `KeplerElements` | (a, e, i, Ω, ω, M₀) + masses |
| **Orbital State** | `BinaryOrbitalState` | 6D position/velocity + masses |
| **Period Calculation** | `compute_period()` | T = 2π√(a³/GM) |
| **Inverse Period** | `period_to_semimajor_axis()` | a from T and M |
| **Batch Conversion** | `batch_elements_to_resolved()` | Vectorized element → state |

**Period Distributions:**

| Distribution | Class | Description |
|--------------|-------|-------------|
| Log-Uniform | `LogUniformPeriod` | Öpik's law |
| Log-Normal | `LogNormalPeriod` | Duquennoy & Mayor (1991) |
| Sana O/B | `SanaOBPeriod` | O-star periods (Sana+ 2012) |

**Eccentricity Distributions:**

| Distribution | Class | Description |
|--------------|-------|-------------|
| Thermal | `ThermalEccentricity` | f(e) = 2e (Heggie 1975) |
| Uniform | `UniformEccentricity` | f(e) = 1 |
| Moe+2017 | `MoeEccentricity` | Period-dependent blend |

### Analytical Test Cases

| System | Function | Description |
|--------|----------|-------------|
| **Kepler Orbit** | `two_body_kepler()` | Circular/eccentric two-body |
| **Figure-8** | `three_body_figure_eight()` | Moore (2001) periodic solution |
| **Earth-Sun** | `earth_sun_2body()` | JPL ephemerides |
| **Solar System** | `solar_system_inner_4()` | Sun + inner 4 planets |
| **Full Solar System** | `solar_system_full()` | Sun + 8 planets |
| **Harmonic Oscillator** | `harmonic_oscillator_1d()` | 1D/2D SHO |

### Utilities

| Category | Functions |
|----------|-----------|
| **IC Builder** | `build_spatial_ic()`, `ICResult` |
| **Energetics** | `compute_kinetic_energy()`, `compute_potential_energy()` |
| **Transforms** | `to_com_frame()`, `virial_scale()` |
| **Stellar Radii** | `compute_stellar_radii()` (3-regime M-R relation) |
| **Tidal Physics** | `jacobi_radius()`, `jacobi_radius_isothermal()`, `apply_tidal_truncation()` |
| **Fractal** | `generate_fractal_positions()`, `apply_fractal_overlay_radial()` |
| **Two-Component** | `TwoComponentConfig`, `generate_two_component_cluster()` |
| **Mass Segregation** | `apply_mass_segregation()`, `apply_mass_segregation_baumgardt()` |

## Installation

```bash
conda activate astro
pip install -e ".[dev]"
```

Or with UV (faster):

```bash
uv pip install -e ".[dev]"
```

## Quick Start

### Plummer Sphere IC

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF

# Create profile and velocity DF
profile = PlummerProfile(r_h=1.0)  # Half-mass radius = 1 pc
velocity_df = PlummerVelocityDF(r_h=1.0)

# Sample positions and velocities
N = 1000
masses = jnp.ones(N)  # 1000 Msun total
key = jax.random.PRNGKey(42)
key_pos, key_vel = jax.random.split(key)

positions = profile.sample_positions(masses, key_pos)
velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

### Two-Component Cluster

```python
from jaxstro.units import STELLAR
from progenax import (
    PlummerProfile, PlummerVelocityDF,
    TwoComponentConfig, generate_two_component_cluster,
)

# Extended halo + concentrated core
config = TwoComponentConfig(
    f_A=0.3,  # 30% in extended halo
    profile_A=PlummerProfile(r_h=2.0),
    profile_B=PlummerProfile(r_h=0.5),
    velocity_df_A=PlummerVelocityDF(r_h=2.0),
    velocity_df_B=PlummerVelocityDF(r_h=0.5),
)

positions, velocities, pop_id = generate_two_component_cluster(
    masses, config, key, G=STELLAR.G
)
```

### IMF Sampling

```python
import jax
from progenax.imf import PowerLawIMF, ChabrierIMF

# Kroupa IMF
imf = PowerLawIMF.kroupa()
key = jax.random.PRNGKey(42)
masses = imf.sample(key, 1000)  # Sample 1000 stellar masses

# Chabrier IMF
imf = ChabrierIMF()
masses = imf.sample(key, 1000)
```

### Binary Orbital Elements

```python
from jaxstro.units import PLANETARY
from progenax.binaries import KeplerElements, compute_period

# Create Keplerian elements
elements = KeplerElements(
    a=1.0,      # Semi-major axis [AU]
    e=0.3,      # Eccentricity
    i=0.1,      # Inclination [rad]
    Omega=0.0,  # Longitude of ascending node
    omega=0.0,  # Argument of periapsis
    M0=0.0,     # Mean anomaly at epoch
)

# Convert to Cartesian state
M_total = 2.0  # Solar masses
state = elements.to_state(M_total, G=PLANETARY.G)
r, v = state['position'], state['velocity']

# Compute orbital period
period = compute_period(a=1.0, M_total=2.0, G=PLANETARY.G)
```

### Analytical Test Case

```python
from jaxstro.units import PLANETARY
from progenax.analytical import two_body_kepler

# Create Earth-like orbit around Sun
ic = two_body_kepler(
    a=1.0,      # 1 AU
    e=0.017,    # Earth's eccentricity
    m1=1.0,     # 1 Msun
    m2=3e-6,    # ~1 Earth mass
    G=PLANETARY.G,
)

positions = ic.positions   # (2, 3)
velocities = ic.velocities # (2, 3)
masses = ic.masses         # (2,)
```

## Unit Systems

progenax uses **jaxstro.units** for gravitational constant management:

```python
from jaxstro.units import STELLAR, PLANETARY

# STELLAR.G for star clusters (~0.00450 pc³ Msun⁻¹ Myr⁻²)
velocities = velocity_df.sample_velocities(positions, masses, key, G=STELLAR.G)

# PLANETARY.G for binaries/planets (~39.478 AU³ Msun⁻¹ yr⁻² = 4π²)
orbit = two_body_kepler(a=1.0, e=0.3, m1=1.0, m2=1.0, G=PLANETARY.G)

# Default behavior: G=None uses jaxstro.units.DEFAULT.G (= STELLAR.G)
velocities = velocity_df.sample_velocities(positions, masses, key)  # Uses STELLAR.G
```

| Unit System | G Value | Units | Use Case |
|-------------|---------|-------|----------|
| `STELLAR` | ~0.00450 | pc³ Msun⁻¹ Myr⁻² | Star clusters, galaxies |
| `PLANETARY` | ~39.478 | AU³ Msun⁻¹ yr⁻² | Binaries, planets |
| `DEFAULT` | = STELLAR | - | Default for progenax |

## Architecture

```text
progenax/
├── src/progenax/
│   ├── __init__.py          # Public API (57+ exports)
│   ├── protocols.py         # 3 runtime-checkable protocols
│   ├── builders.py          # ICResult + build_spatial_ic
│   ├── populations.py       # Two-component clusters
│   ├── tidal.py             # Jacobi radius + truncation
│   ├── profiles/
│   │   ├── plummer.py       # PlummerProfile
│   │   ├── king.py          # KingProfile + ODE solver
│   │   ├── eff.py           # EFFProfile
│   │   └── mass_segregation.py
│   ├── kinematics/
│   │   ├── plummer_df.py    # PlummerVelocityDF (Beta sampling)
│   │   ├── king_df.py       # KingVelocityDF
│   │   ├── eff_df.py        # EFFVelocityDF
│   │   ├── anisotropy.py    # Osipkov-Merritt
│   │   └── rotation.py      # Solid body, differential
│   ├── imf/
│   │   ├── base.py          # BaseIMF with custom JVP
│   │   ├── power_law.py     # PowerLawIMF (Kroupa, Salpeter)
│   │   ├── chabrier.py      # ChabrierIMF
│   │   ├── environment.py   # EnvironmentIMF
│   │   ├── binary.py        # BinaryIMF
│   │   └── igimf.py         # IGIMF
│   ├── binaries/
│   │   ├── kepler.py        # KeplerElements, Kepler solver
│   │   ├── orbital_state.py # BinaryOrbitalState
│   │   └── population.py    # Period/eccentricity distributions
│   ├── substructure/
│   │   └── fractal.py       # Goodwin-Whitworth + overlays
│   └── analytical/
│       └── core.py          # Solar system, Kepler orbits
└── tests/
    ├── unit/                # 310 unit tests
    ├── integration/         # 42 integration tests
    └── validation/          # 80 physics validation tests
```

## Key Patterns

### Protocol-Based Composition

Any `SpatialProfile` can pair with any `VelocityDF`:

```python
from progenax.protocols import SpatialProfile, VelocityDF

profile: SpatialProfile = PlummerProfile(r_h=1.0)
df: VelocityDF = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)

# Mix Plummer positions with King velocities!
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

### Equinox Modules

All stateful classes are Equinox modules (immutable PyTrees):

```python
import equinox as eqx

class PlummerProfile(eqx.Module):
    r_h: Float[Array, ""]  # Half-mass radius
    a: Float[Array, ""]    # Scale radius (computed)
```

### Differentiability

All sampling is differentiable through JAX:

```python
def loss(r_h):
    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(positions, axis=1))

grad_fn = jax.grad(loss)
gradient = grad_fn(1.0)  # Fully differentiable!
```

### Critical Formulas

**Plummer Scale Radius:**
```python
# From half-mass radius to scale radius
a = r_h * jnp.sqrt(2**(2/3) - 1)  # ≈ 0.7664 * r_h
```

**Virial Ratio:**
```python
Q = T / |V|  # Q ≈ 0.5 for equilibrium (virial theorem: 2T + V = 0)
```

## Testing

```bash
# All tests
pytest tests/ -v                    # 432 tests, ~55s

# By tier
pytest tests/unit/ -v               # 310 unit tests
pytest tests/integration/ -v        # 42 integration tests
pytest tests/validation/ -v         # 80 physics validation tests

# Specific modules
pytest tests/unit/imf/ -v
pytest tests/validation/test_plummer_physics.py -v
```

### Physics Validation

From `tests/validation/`:

- **Virial ratio**: Q ≈ 0.5 (expected 0.5 for equilibrium)
- **Velocity dispersion**: <1% error at all radii
- **Bound particles**: 100% within escape velocity
- **Half-mass radius**: 49.9% within r_h (expected 50%)
- **Kepler orbits**: Period formula exact to 1e-10

## Dependencies

- `jax>=0.4.20`
- `jaxlib>=0.4.20`
- `equinox>=0.11.0`
- `jaxtyping>=0.2.25`
- `diffrax>=0.4.0` (for King profile ODE integration)
- `jaxstro` (core utilities)

## References

**Density Profiles:**
- Plummer (1911), MNRAS 71, 460
- King (1966), AJ 71, 64
- Elson, Fall & Freeman (1987), ApJ 323, 54

**Velocity Distribution Functions:**
- Dehnen (1993), MNRAS 265, 250 - Exact Plummer DF
- Binney & Tremaine (2008), "Galactic Dynamics"

**Initial Mass Functions:**
- Salpeter (1955), ApJ 121, 161
- Kroupa (2001), MNRAS 322, 231
- Chabrier (2003), PASP 115, 763
- Maschberger (2013), MNRAS 429, 1725

**Binary Populations:**
- Duquennoy & Mayor (1991), A&A 248, 485
- Sana+ (2012), Science 337, 444
- Moe & Di Stefano (2017), ApJS 230, 15

**Substructure:**
- Goodwin & Whitworth (2004), A&A 413, 929
- Kupper+ (2011), MNRAS 417, 2300 - McLuster

**N-body Methods:**
- Aarseth (2003), "Gravitational N-Body Simulations"

## License

MIT
