# progenax

Differentiable initial conditions for N-body simulations in JAX.

Part of the **jaxstro ecosystem** - providing IC generation that can be differentiated through for gradient-based inference.

## Status

**Phase 1 Complete**: Core implementation ported from gravax-legacy with 350+ tests passing.

### TODO: Validation Plots (Pending jaxstroviz)

The following validation plots will be added after the jaxstroviz plotting module is ported:

- [ ] Plummer density profile (sampled vs analytical)
- [ ] King profile comparison with LIMEPY
- [ ] EFF profile truncation behavior
- [ ] Velocity dispersion radial profiles
- [ ] Virial equilibrium verification plots
- [ ] IMF mass distributions (Kroupa, Chabrier, etc.)
- [ ] Binary orbital element distributions

## Features

### Initial Mass Functions (IMF)

| IMF | Module | Description |
|-----|--------|-------------|
| Power Law | `imf.power_law` | Single/broken power laws |
| Kroupa | `imf.smooth` | Kroupa (2001) with smooth transitions |
| Chabrier | `imf.chabrier` | Chabrier (2003) log-normal + power law |
| Truncated | `imf.truncated` | Mass limits on any IMF |
| Binary IMF | `imf.binary` | Primary + secondary mass sampling |
| IGIMF | `imf.igimf` | Integrated galactic IMF |
| Environmental | `imf.environment` | Metallicity-dependent variations |

### Spatial Profiles

| Profile | Module | Description |
|---------|--------|-------------|
| Plummer | `profiles.plummer` | Plummer (1911) sphere |
| King | `profiles.king` | King (1966) lowered isothermal |
| EFF | `profiles.eff` | Elson-Fall-Freeman (1987) |

### Velocity Distribution Functions

| DF | Module | Description |
|----|--------|-------------|
| Plummer DF | `kinematics.plummer_df` | Exact Beta sampling |
| King DF | `kinematics.king_df` | Rejection sampling |
| EFF DF | `kinematics.eff_df` | Energy-based sampling |

### Analytical Test Cases

| System | Module | Description |
|--------|--------|-------------|
| Circular Orbit | `analytical.circular_orbit` | Two-body periodic |
| Kepler Ellipse | `analytical.kepler_ellipse` | Eccentric orbits |
| Free Fall | `analytical.free_fall` | Radial infall |
| Figure-8 | `analytical.figure_eight` | Three-body periodic |
| Pythagorean | `analytical.pythagorean` | (3,4,5) triangle chaos |

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

### IMF Sampling

```python
import jax
from progenax.imf import KroupaIMF

imf = KroupaIMF()
key = jax.random.PRNGKey(42)
masses = imf.sample(key, n_samples=1000)  # Sample 1000 stellar masses
```

### Analytical Test Case

```python
from jaxstro.units import PLANETARY
from progenax.analytical import CircularOrbit

# Create two-body circular orbit (binary/planetary uses PLANETARY.G)
orbit = CircularOrbit(m1=1.0, m2=1.0, separation=1.0, G=PLANETARY.G)
positions, velocities, masses = orbit.get_state()
```

## Architecture

```text
progenax/
├── src/progenax/
│   ├── imf/              # Initial mass functions
│   │   ├── base.py       # IMFProtocol definition
│   │   ├── power_law.py  # Power law IMFs
│   │   ├── smooth.py     # Kroupa smooth IMF
│   │   ├── chabrier.py   # Chabrier IMF
│   │   └── ...
│   ├── profiles/         # Spatial density profiles
│   │   ├── plummer.py    # Plummer sphere
│   │   ├── king.py       # King model
│   │   └── eff.py        # EFF profile
│   ├── kinematics/       # Velocity DFs
│   │   ├── plummer_df.py # Plummer velocity DF
│   │   ├── king_df.py    # King velocity DF
│   │   └── eff_df.py     # EFF velocity DF
│   ├── analytical/       # Test cases
│   │   ├── circular_orbit.py
│   │   ├── kepler_ellipse.py
│   │   └── ...
│   ├── binaries.py       # Binary orbital elements
│   ├── builders.py       # IC assembly utilities
│   └── protocols.py      # Protocol definitions
└── tests/
    ├── unit/             # Unit tests (~300 tests)
    └── integration/      # Integration tests (~50 tests)
```

## Testing

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests (physics validation)
pytest tests/integration/ -v

# Specific module
pytest tests/unit/imf/ -v
```

## Key Patterns

### Unit System with jaxstro.units

All physics functions use jaxstro unit systems (no global state):

```python
from jaxstro.units import STELLAR, PLANETARY

# Star clusters use STELLAR.G (~0.00450 pc³ Msun⁻¹ Myr⁻²)
velocities = velocity_df.sample_velocities(positions, masses, key, G=STELLAR.G)

# Binaries/planets use PLANETARY.G (~39.478 AU³ Msun⁻¹ yr⁻²)
orbit = CircularOrbit(m1=1.0, m2=1.0, separation=1.0, G=PLANETARY.G)
```

### Protocol Composition

Spatial profiles and velocity DFs are composed via protocols:

```python
from progenax.protocols import SpatialProfile, VelocityDF

# Any SpatialProfile + compatible VelocityDF can be combined
profile: SpatialProfile = PlummerProfile(r_h=1.0)
df: VelocityDF = PlummerVelocityDF(r_h=1.0)
```

### Differentiability

All sampling is differentiable through JAX:

```python
def loss(r_h):
    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(positions, axis=1))

grad_fn = jax.grad(loss)
gradient = grad_fn(1.0)  # Works!
```

## Dependencies

- `jax>=0.4.20`
- `jaxlib>=0.4.20`
- `equinox>=0.11.0`
- `jaxtyping>=0.2.25`
- `diffrax>=0.4.0` (for King profile ODE integration)
- `jaxstro` (core utilities)

## References

- Plummer (1911), MNRAS 71, 460
- King (1966), AJ 71, 64
- Elson, Fall & Freeman (1987), ApJ 323, 54
- Kroupa (2001), MNRAS 322, 231
- Chabrier (2003), PASP 115, 763
- Aarseth (2003), "Gravitational N-Body Simulations"
- Binney & Tremaine (2008), "Galactic Dynamics"

## License

MIT
