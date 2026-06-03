# CLAUDE.md - progenax

## Overview

Differentiable initial conditions for N-body simulations in JAX. Part of the **jaxstro ecosystem**.

**Status**: Phase 1 + 2026-06 audit hardening complete - 18,967 LOC source, 874 tests (unit: 742, integration: 24, validation: 108). King & EFF velocity DFs are now true equilibria (lowered-Maxwellian / Eddington inversion); see Physics Validation Results below.

## Quick Commands

```bash
conda activate astro
pip install -e ".[dev]"
pytest tests/ -v                    # All 874 tests (~55s; ~5min with coverage)
pytest tests/unit/ -v               # 742 unit tests
pytest tests/integration/ -v        # 24 integration tests
pytest tests/validation/ -v         # 108 physics validation tests
```

## Units Policy (progenax)

**DEFAULT_UNITS:** `STELLAR` (Msun, pc, Myr)

Rules:
- Core APIs require explicit `G` or `units` (or objects that carry units).
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.
- Do **not** use global context managers or `get_G()` in core code.

Example:
```python
from jaxstro.units import STELLAR
from progenax import DEFAULT_UNITS

# Core (explicit)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)

# Wrapper (optional)
velocities = df.sample_velocities(positions, masses, key, G=DEFAULT_UNITS.G)
```

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

### JAX Float64 (Automatic)

**progenax automatically enables float64** via `jaxstro.jaxconfig.enable_high_precision()` at import time. This is the standard approach across the jaxstro ecosystem - high precision is configured before any JAX arrays are created.

You don't need to do anything - just `import progenax` and you get float64.

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
├── unit/                742 tests
│   ├── imf/             IMF tests (PowerLaw, Chabrier, IGIMF, Binary)
│   ├── profiles/        Profile tests (Plummer, King, EFF)
│   ├── kinematics/      Velocity DF tests + anisotropy
│   ├── analytical/      Analytical test case tests
│   ├── binaries/        Binary orbital element + sampling-gradient tests
│   ├── cluster/         Fractal / FDF / tail-sampling tests
│   ├── physics/         PN11 / BM19 / PP20 gravoturbulence tests
│   ├── dynamics/        Virial / energy utilities
│   └── substructure/    Fractal substructure tests
├── integration/         24 tests
│   ├── test_jax_compatibility.py     JIT/grad/vmap tests
│   ├── test_units_through_pipeline.py  G threading (audit C1)
│   └── test_end_to_end.py            Full IC → energy checks
└── validation/          108 tests
    ├── test_plummer_physics.py      Plummer equilibrium
    ├── test_king_physics.py         King true-DF equilibrium + c(W0)
    ├── test_eff_physics.py          EFF Eddington-inversion DF
    ├── test_binary_physics.py       Kepler's laws
    └── test_imf_physics.py          IMF distributions
```

## Physics Validation Results

From `tests/validation/` (Q ≡ T/|V|, so 0.5 is equilibrium). King & EFF DFs
are sampled in detailed equilibrium with **no external virial rescale**:

| Test | Result | Expected |
|------|--------|----------|
| Plummer virial Q = T/\|V\| | 0.502 | 0.5 |
| King true-DF virial Q (unscaled) | ~0.51 | 0.5 |
| EFF Eddington-DF virial Q (γ=5, mild trunc) | ~0.50 | 0.5 |
| King c(W₀) vs King (1966) Table II | Δ ≤ 0.02 (≤0.004 for W₀≥5) | Table II |
| Kepler energy & angular momentum | conserved to ~1e-16 | exact |
| Bound particles | 100% | 100% |
| Binary period (Kepler III) | exact to 1e-10 | 2π√(a³/GM) |

## Public API

All public symbols exported from `progenax.__init__`:

**Profiles**: `PlummerProfile`, `KingProfile`, `EFFProfile`, `solve_king_profile()`

**Velocity DFs**: `PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF`, `apply_osipkov_merritt()`, `apply_solid_body_rotation()`, `apply_differential_rotation()`

**IMFs**: `PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `TruncatedIMF`, `BinaryIMF`, `IGIMF`, `EnvironmentIMF`

**Binaries**: `KeplerElements`, `BinaryOrbitalState`, `compute_period()`, `period_to_semimajor_axis()`, `LogUniformPeriod`, `LogNormalPeriod`, `SanaOBPeriod`, `ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity`

**Analytical**: `two_body_kepler()`, `three_body_figure_eight()`, `earth_sun_2body()`, `solar_system_inner_4()`, `solar_system_full()`, `harmonic_oscillator()`

**Utilities**: `build_spatial_ic()`, `ICResult`, `compute_kinetic_energy()`, `compute_potential_energy()`, `to_com_frame()`, `virial_scale()`, `compute_stellar_radii()`, `jacobi_radius()`, `apply_tidal_truncation()`, `generate_fractal_positions()`, `TwoComponentConfig`, `generate_two_component_cluster()`, `energy_sorted_segregation()`

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
