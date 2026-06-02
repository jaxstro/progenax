---
title: Installation
description: Install progenax via UV (preferred) or pip — single-environment setup, optional dependencies, and the smoke test that verifies everything works.
---
# Installation

progenax is a JAX-native Python package distributed through the
[jaxstro monorepo](https://github.com/drannarosen/jaxstro-dev). The
recommended installation tool is **UV**, which is 10–100× faster
than pip for this codebase.

## Quick install (UV — recommended)

```bash
git clone https://github.com/drannarosen/jaxstro-dev.git
cd jaxstro-dev/progenax
uv pip install -e ".[all]"
```

The `[all]` extra pulls in development tooling (pytest, black,
mypy), I/O (orbax-checkpoint, h5py), visualisation (matplotlib,
seaborn), and ML (optax, blackjax, numpyro). For minimal install:

```bash
uv pip install -e .
```

For the broader jaxstro ecosystem (gravax, fluxax, etc.):

```bash
cd jaxstro-dev
uv pip install -e ./jaxstro -e ./gravax -e ./progenax -e ./fluxax
```

## Pip alternative

```bash
git clone https://github.com/drannarosen/jaxstro-dev.git
cd jaxstro-dev/progenax
pip install -e ".[all]"
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
key = jax.random.PRNGKey(0)
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)
positions = profile.sample_positions(masses, key)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)
print(f"Generated {positions.shape[0]} particles, mean radius {jnp.linalg.norm(positions, axis=1).mean():.3f} pc")
```

Expected output:

```
Generated 100 particles, mean radius ~0.95 pc
```

If you see a `jaxstro` import error, the ecosystem isn't installed.
If you see a `jax` GPU-allocation message, JAX has detected your GPU
— this is fine. CPU-only is the default and sufficient for the
tutorials.

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

GPU acceleration gives roughly $100\times$ speedup at $N \sim 10^4$
relative to CPU. See [](differentiable-ic.md) for an example HMC
chain that benefits from GPU.

## Optional dependencies

```{list-table}
:header-rows: 1

* - Extra
  - Provides
* - `[dev]`
  - pytest, black, isort, flake8, mypy
* - `[io]`
  - orbax-checkpoint, h5py, pandas
* - `[viz]`
  - matplotlib, seaborn
* - `[astro]`
  - astropy, gala
* - `[ml]`
  - optax, blackjax, numpyro
* - `[all]`
  - All of the above
```

Install only what you need: `uv pip install -e ".[viz,ml]"`.

## Next step

[](first-plummer-sphere.md) walks through generating your first IC.
