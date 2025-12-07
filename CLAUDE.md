# CLAUDE.md - progenax

## Overview

Differentiable initial conditions for N-body simulations. Part of the jaxstro ecosystem.

**Status**: Phase 1 complete (~8,000 LOC, 350+ tests passing)

## TODO: Validation Plots (Pending jaxstroviz)

After jaxstroviz is ported from gravax-legacy/graviz:

- [ ] Plummer density profile (sampled vs analytical)
- [ ] King profile comparison with LIMEPY
- [ ] EFF profile truncation behavior
- [ ] Velocity dispersion radial profiles
- [ ] Virial equilibrium verification plots
- [ ] IMF mass distributions
- [ ] Binary orbital element distributions

## Commands

```bash
conda activate astro
pip install -e ".[dev]"
pytest tests/ -v
pytest tests/integration/test_physics_validation.py -v -s  # Physics validation
```

## Key Patterns

### Explicit G Parameter (MANDATORY)

All physics functions take explicit `G` parameter - NO `get_G()` calls:

```python
# CORRECT
velocities = df.sample_velocities(positions, masses, key, G=1.0)

# WRONG - do not use
velocities = df.sample_velocities(positions, masses, key)  # No default G!
```

### Equinox Modules

All stateful classes are Equinox modules (immutable PyTrees):

```python
class PlummerProfile(eqx.Module):
    r_h: Float[Array, ""]
    a: Float[Array, ""]
```

### Protocol Composition

Spatial profiles and velocity DFs follow protocols:

```python
from progenax.protocols import SpatialProfile, VelocityDF

profile: SpatialProfile = PlummerProfile(r_h=1.0)
df: VelocityDF = PlummerVelocityDF(r_h=1.0)
```

## Module Quick Reference

| Module | Purpose |
|--------|---------|
| `imf/` | Initial mass functions (Kroupa, Chabrier, etc.) |
| `profiles/` | Spatial density profiles (Plummer, King, EFF) |
| `kinematics/` | Velocity distribution functions |
| `analytical/` | Test cases with exact solutions |
| `binaries.py` | Binary orbital elements |
| `builders.py` | IC assembly utilities |
| `protocols.py` | Protocol definitions |

## Critical Formulas

### Plummer Scale Radius

```python
# From half-mass radius to scale radius
a = r_h * jnp.sqrt(2**(2/3) - 1)  # ≈ 0.7664 * r_h

# WRONG (was a bug):
# a = r_h / jnp.sqrt(2**(2/3) - 1)  # This is INVERTED!
```

### Virial Ratio

```python
Q = 2*T / |V|  # Q ≈ 1.0 for equilibrium
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

### Differentiability

All sampling uses `jax.lax.scan` with fixed iterations (NOT `while_loop`):

```python
# Differentiable
def loss(r_h):
    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(positions, axis=1))

jax.grad(loss)(1.0)  # Works!
```

## Test Structure

```text
tests/
├── unit/              # ~300 unit tests
│   ├── imf/           # IMF tests
│   ├── profiles/      # Profile tests
│   ├── kinematics/    # Velocity DF tests
│   ├── analytical/    # Analytical test case tests
│   └── binaries/      # Binary orbital element tests
└── integration/       # ~50 integration tests
    ├── test_end_to_end.py
    └── test_physics_validation.py  # Physics validation
```

## Physics Validation Results

From `test_physics_validation.py`:

- **Virial ratio**: Q = 0.995 (expected 1.0)
- **Velocity dispersion**: <1% error at all radii
- **Bound particles**: 100% bound
- **Half-mass radius**: 49.9% within r_h (expected 50%)
