---
title: Installation
description: Install progenax via UV (preferred) or pip — single-environment setup, optional dependencies, and the smoke test that verifies everything works.
---
# Installation

progenax is a JAX-native Python package. It is **not on PyPI**; it
(and its public sibling `jaxstro`) resolve from a standard
side-by-side checkout. The recommended installation tool is **UV**,
which is 10–100× faster than pip for this codebase.

## Quick install (UV — recommended)

Clone progenax and `jaxstro` next to each other (the `[tool.uv.sources]`
path dependency expects the sibling layout):

```bash
git clone https://github.com/jaxstro/jaxstro.git
git clone https://github.com/jaxstro/progenax.git
cd progenax
uv pip install -e ".[dev]"
```

The `[dev]` extra adds the test/lint tooling (pytest, pytest-cov,
pytest-xdist, ruff, mypy). For a minimal runtime-only install:

```bash
uv pip install -e .
```

## Pip alternative

```bash
git clone https://github.com/jaxstro/jaxstro.git
git clone https://github.com/jaxstro/progenax.git
cd progenax
pip install -e ".[dev]"
```

## Smoke test

After installation, verify the import + a basic IC works:

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF

masses = jnp.ones(100)
key_pos, key_vel = jax.random.split(jax.random.PRNGKey(0))  # never reuse a key
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
print(f"Generated {positions.shape[0]} particles, mean radius {jnp.linalg.norm(positions, axis=1).mean():.3f} pc")
```

Expected output:

```
Generated 100 particles, mean radius ~1.35 pc
```

If you see a `jaxstro` import error, the ecosystem isn't installed.
If you see a `jax` GPU-allocation message, JAX has detected your GPU
— this is fine. CPU-only is the default and sufficient for the
tutorials.

## Precision: progenax enables float64 at import

`import progenax` sets `jax_enable_x64=True` and
`jax_default_matmul_precision="highest"` **process-wide** (via
`jaxstro.jaxconfig.enable_high_precision()`). Cluster initial conditions are
energy-balance-sensitive, and every validation gate on this site runs in double
precision — float32 is **not** validated anywhere in progenax.

Two practical consequences if you embed progenax in a larger JAX application:

1. Arrays created *after* the import default to `float64` — roughly double the
   memory of a float32 pipeline. Import progenax early (before building other
   JAX state) so your whole process runs in one consistent precision regime.
2. Reverting x64 after import (`jax.config.update("jax_enable_x64", False)`)
   downcasts progenax's outputs with warnings and voids the validation
   guarantees. Don't mix regimes.

## GPU support

progenax runs on any JAX-supported device (CPU, GPU, TPU). For GPU:

```bash
# CUDA 12, NVIDIA GPU
uv pip install -U "jax[cuda12]"
```

```bash
# Apple Silicon Metal
uv pip install jax-metal
```

GPU/TPU acceleration can give substantial speedups for large $N$
relative to CPU (the exact factor depends on $N$, hardware, and the
operation). See [](differentiable-ic.md) for an example HMC chain
that benefits from GPU.

## Optional dependencies

```{list-table}
:header-rows: 1

* - Extra
  - Provides
* - `[dev]`
  - pytest, pytest-cov, pytest-xdist, ruff, mypy
* - `[experimental]`
  - the repo-only `gravoturb` inference layer (blackjax, optax, arviz, scipy, flowjax)
* - `[diagnostics]`
  - numpy + scipy, for the exact (non-differentiable) CW04 `compute_q_parameter` path
```

Install only what you need: `uv pip install -e ".[dev,experimental]"`.

## Next step

[](first-plummer-sphere.md) walks through generating your first IC.
