# CLAUDE.md - progenax

## Overview

Differentiable initial conditions for N-body simulations in JAX. Part of the **jaxstro ecosystem**.

**Status**: Phase 1 Complete - 9,400 LOC source, 432 tests (unit: 310, integration: 42, validation: 80)

## Quick Commands

```bash
conda activate astro
pip install -e ".[dev]"
pytest tests/ -v                    # All 432 tests (~55s)
pytest tests/unit/ -v               # 310 unit tests
pytest tests/integration/ -v        # 42 integration tests
pytest tests/validation/ -v         # 80 physics validation tests
```

## jaxstro Unit Systems (MANDATORY)

**ALL physics functions use jaxstro unit systems - NEVER hardcode G values.**

```python
from jaxstro.units import STELLAR, PLANETARY

# Star clusters: STELLAR.G (~0.00450 pc³ Msun⁻¹ Myr⁻²)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)

# Binaries/planets: PLANETARY.G (~39.478 AU³ Msun⁻¹ yr⁻² = 4π²)
orbit = two_body_kepler(a=1.0, e=0.3, m1=1.0, m2=1.0, G=PLANETARY.G)

# Default: G=None uses jaxstro.units.DEFAULT.G (= STELLAR.G)
velocities = df.sample_velocities(positions, masses, key)  # Uses STELLAR.G
```

| Unit System | G Value | Units | Use Case |
|-------------|---------|-------|----------|
| `STELLAR` | ~0.00450 | pc³ Msun⁻¹ Myr⁻² | Star clusters, galaxies |
| `PLANETARY` | ~39.478 | AU³ Msun⁻¹ yr⁻² | Binaries, planets |
| `DEFAULT` | = STELLAR | - | Default for progenax |

## Key Patterns

### Protocol-Based Composition

Any `SpatialProfile` pairs with any `VelocityDF`:

```python
from progenax.protocols import SpatialProfile, VelocityDF

profile: SpatialProfile = PlummerProfile(r_h=1.0)
df: VelocityDF = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)

# Mix Plummer positions with King velocities
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

### Equinox Modules

All stateful classes are Equinox modules (immutable PyTrees):

```python
class PlummerProfile(eqx.Module):
    r_h: Float[Array, ""]  # Half-mass radius
    a: Float[Array, ""]    # Scale radius (computed)
```

### Differentiability

All sampling uses `jax.lax.scan` with fixed iterations (NOT `while_loop`):

```python
def loss(r_h):
    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(positions, axis=1))

jax.grad(loss)(1.0)  # Fully differentiable!
```

## Module Quick Reference

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `profiles/` | Spatial density profiles | `PlummerProfile`, `KingProfile`, `EFFProfile` |
| `kinematics/` | Velocity DFs + transforms | `PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF` |
| `imf/` | Initial mass functions | `PowerLawIMF`, `ChabrierIMF`, `IGIMF`, `BinaryIMF` |
| `binaries/` | Orbital mechanics | `KeplerElements`, `BinaryOrbitalState` |
| `analytical/` | Test cases with exact solutions | `two_body_kepler()`, `three_body_figure_eight()` |
| `substructure/` | Fractal + substructure | `generate_fractal_positions()` |
| `protocols.py` | 3 runtime-checkable protocols | `SpatialProfile`, `VelocityDF`, `IMFProtocol` |
| `builders.py` | IC assembly utilities | `build_spatial_ic()`, `ICResult` |
| `populations.py` | Multi-component clusters | `TwoComponentConfig`, `generate_two_component_cluster()` |
| `tidal.py` | Tidal physics | `jacobi_radius()`, `apply_tidal_truncation()` |

## Critical Formulas

### Plummer Scale Radius

```python
# From half-mass radius to scale radius
a = r_h * jnp.sqrt(2**(2/3) - 1)  # ≈ 0.7664 * r_h

# WRONG (was a historical bug):
# a = r_h / jnp.sqrt(2**(2/3) - 1)  # INVERTED!
```

### Virial Ratio

```python
Q = T / |V|  # Q ≈ 0.5 for equilibrium (virial theorem: 2T + V = 0)
```

**Convention:**
- Q = 0.5: Virial equilibrium (2T + V = 0)
- Q < 0.5: Subvirial (cold, collapsing)
- Q > 0.5: Supervirial (hot, expanding)

### Kepler's Third Law

```python
T = 2 * jnp.pi * jnp.sqrt(a**3 / (G * M_total))  # Orbital period
```

## Common Issues

### Scale Radius Mismatch

Profile and velocity DF must use the **same** `r_h` value:

```python
# CORRECT
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)  # Same r_h!

# WRONG - will produce non-equilibrium ICs
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=2.0)  # Different r_h!
```

### JAX Float64

Enable float64 for precision-sensitive calculations:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

### Unit System Consistency

Always pass consistent G values through the pipeline:

```python
from jaxstro.units import STELLAR

# CORRECT - consistent G throughout
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
energy = compute_total_energy(positions, velocities, masses, G=STELLAR.G)

# WRONG - mixing unit systems
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
energy = compute_total_energy(positions, velocities, masses, G=PLANETARY.G)  # WRONG!
```

## Test Structure

```text
tests/
├── unit/                310 tests
│   ├── imf/             IMF tests (PowerLaw, Chabrier, IGIMF, Binary)
│   ├── profiles/        Profile tests (Plummer, King, EFF)
│   ├── kinematics/      Velocity DF tests + anisotropy
│   ├── analytical/      Analytical test case tests
│   └── binaries/        Binary orbital element tests
├── integration/         42 tests
│   ├── test_jax_compatibility.py    JIT/grad/vmap tests
│   └── test_physics_validation.py   Physics validation
└── validation/          80 tests
    ├── test_plummer_physics.py      Plummer equilibrium
    ├── test_binary_physics.py       Kepler's laws
    └── test_imf_statistics.py       IMF distributions
```

## Physics Validation Results

From `tests/validation/`:

| Test | Result | Expected |
|------|--------|----------|
| Virial ratio | Q = 0.995 | 1.0 |
| Velocity dispersion | <1% error | Analytical |
| Bound particles | 100% | 100% |
| Half-mass radius | 49.9% within r_h | 50% |
| Kepler period | Exact to 1e-10 | 2π√(a³/GM) |

## Public API (57+ exports)

All public symbols exported from `progenax.__init__`:

**Profiles**: `PlummerProfile`, `KingProfile`, `EFFProfile`, `solve_king_profile()`

**Velocity DFs**: `PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF`, `apply_osipkov_merritt()`, `apply_solid_body_rotation()`, `apply_differential_rotation()`

**IMFs**: `PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `TruncatedIMF`, `BinaryIMF`, `IGIMF`, `EnvironmentIMF`

**Binaries**: `KeplerElements`, `BinaryOrbitalState`, `compute_period()`, `period_to_semimajor_axis()`, `LogUniformPeriod`, `LogNormalPeriod`, `SanaOBPeriod`, `ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity`

**Analytical**: `two_body_kepler()`, `three_body_figure_eight()`, `earth_sun_2body()`, `solar_system_inner_4()`, `solar_system_full()`, `harmonic_oscillator_1d()`

**Utilities**: `build_spatial_ic()`, `ICResult`, `compute_kinetic_energy()`, `compute_potential_energy()`, `to_com_frame()`, `virial_scale()`, `compute_stellar_radii()`, `jacobi_radius()`, `apply_tidal_truncation()`, `generate_fractal_positions()`, `TwoComponentConfig`, `generate_two_component_cluster()`, `apply_mass_segregation()`

**Protocols**: `SpatialProfile`, `VelocityDF`, `IMFProtocol`

## TODO: Validation Plots (Pending jaxstroviz)

After jaxstroviz is ported:

- [ ] Plummer density profile (sampled vs analytical)
- [ ] King profile comparison with LIMEPY
- [ ] EFF profile truncation behavior
- [ ] Velocity dispersion radial profiles
- [ ] Virial equilibrium verification plots
- [ ] IMF mass distributions
- [ ] Binary orbital element distributions
