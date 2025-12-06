# CLAUDE.md - progenax

## Overview
Initial conditions for N-body simulations. JAX-native, differentiable.

## Commands
conda activate astro
pip install -e ".[dev]"
pytest tests/ -v

## Key Patterns
- ALL functions take `units: UnitSystem` parameter (NO get_G())
- Use `units.G` for gravitational constant
- Equinox modules for stateful classes
- Protocol-based composition (SpatialProfile + VelocityDF)

## Architecture
- `imf/` - Initial mass functions (differentiable)
- `profiles/` - Spatial density profiles
- `kinematics/` - Velocity distribution functions
- `analytical/` - Test cases with exact solutions
- `builders.py` - IC assembly utilities
