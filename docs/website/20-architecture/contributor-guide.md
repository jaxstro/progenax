---
title: Contributor guide
description: How to add a new spatial profile, velocity DF, IMF, binary distribution, or validation suite to progenax — the protocol-based composition pattern walked end-to-end with a worked example.
---

# Contributor guide

This chapter is the practical guide to extending progenax. If you
want to add a new spatial profile, velocity DF, IMF, binary
distribution, modifier layer, or validation suite, this is where to
start. The chapter assumes you have read [](jax-native-philosophy.md),
[](differentiability.md), and [](protocols.md) — the *why* of
progenax's structure. This chapter is the *how*.

## Six contribution targets

```{list-table}
:header-rows: 1

* - Target
  - Protocol / file location
  - Worked example
* - **Spatial profile**
  - `SpatialProfile`, `progenax/profiles/`
  - [Adding a Sérsic profile](#sersic-walkthrough)
* - **Velocity DF**
  - `VelocityDF`, `progenax/kinematics/`
  - [Adding an Osipkov-Merritt extension](../10-theory/velocity-dfs/rotation-anisotropy.md)
* - **IMF**
  - `IMFProtocol`, `progenax/imf/`
  - [Adding a custom IMF](#imf-walkthrough)
* - **Binary distribution**
  - `progenax/binaries/`
  - Period / mass-ratio / eccentricity distributions
* - **Modifier layer**
  - `progenax/{tidal,cluster,populations}/`
  - Tidal truncation, fractal substructure, mass segregation
* - **Validation suite**
  - `tests/validation/`
  - Per-physics regression tests with quantitative pass/fail criteria
```

The pattern is the same for all six: write an `eqx.Module`, satisfy
the relevant protocol (or follow the modifier-layer convention),
add tests, document.

## Step-by-step: adding a new spatial profile

(sersic-walkthrough)=

Suppose you want to add a Sérsic-like profile

```{math}
\rho(r) \;\propto\; \exp\!\Bigl[-b_n\,(r/r_h)^{1/n}\Bigr]
```

with shape parameter $n$. The four steps:

### Step 1: write the Equinox module

```python
# progenax/profiles/sersic.py
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKey

class SersicProfile(eqx.Module):
    """Sérsic spatial density profile (work-in-progress contribution)."""

    r_h: Float[Array, ""]   # Half-mass radius
    n: Float[Array, ""]     # Sérsic shape parameter (n=4 = de Vaucouleurs)

    def density(self, r):
        b_n = 2.0 * self.n - 1.0/3.0   # Approximation; exact involves gamma fn
        return jnp.exp(-b_n * jnp.power(r / self.r_h, 1.0 / self.n))

    def cumulative_mass(self, r):
        # Numerical via trapezoidal on a log-spaced grid
        # ... (full implementation needs Newton solver for r_h consistency)
        return ...

    def sample_positions(self, masses, key):
        # Inverse-CDF on cumulative_mass
        N = masses.shape[0]
        u = jax.random.uniform(key, (N,))
        # Look up r corresponding to each u via interp on cumulative_mass grid
        return ...

    def characteristic_radius(self):
        # Radius generic builders use (e.g. for softening); r_h here
        return self.r_h
```

The class:
- Inherits from `eqx.Module` for PyTree compatibility.
- Has `r_h` as a required typed field.
- Adds custom fields (`n` for Sérsic).
- Implements the two `SpatialProfile` methods (`sample_positions`,
  `characteristic_radius`), plus the profile-specific helpers
  (`density`, `cumulative_mass`) that builders and validation use.

### Step 2: write conformance tests

```python
# tests/unit/profiles/test_sersic.py
import pytest
import jax
import jax.numpy as jnp
from progenax.profiles import SersicProfile
from progenax.protocols import SpatialProfile

def test_sersic_satisfies_protocol():
    p = SersicProfile(r_h=1.0, n=4.0)
    assert isinstance(p, SpatialProfile)

def test_sersic_density_shape():
    p = SersicProfile(r_h=1.0, n=4.0)
    r = jnp.linspace(0.1, 5.0, 100)
    rho = p.density(r)
    assert rho.shape == r.shape
    assert jnp.all(rho > 0)

def test_sersic_half_mass():
    """The half-mass radius condition M(<r_h) = M/2 must hold to 1%."""
    p = SersicProfile(r_h=1.0, n=4.0)
    M_half = p.cumulative_mass(jnp.float64(1.0))
    assert M_half == pytest.approx(0.5, rel=0.01)

def test_sersic_grad():
    """Sampling positions is differentiable in r_h."""
    def loss(r_h):
        p = SersicProfile(r_h=r_h, n=4.0)
        positions = p.sample_positions(jnp.ones(100), jax.random.PRNGKey(0))
        return jnp.sum(positions ** 2)

    grad = jax.grad(loss)(1.0)
    assert jnp.isfinite(grad)
```

### Step 3: write the validation script

```python
# tests/validation/test_sersic_physics.py
"""Validate SersicProfile against analytical/published results."""
def test_sersic_density_matches_textbook():
    """Sérsic n=4 = de Vaucouleurs should match standard textbook formula."""
    p = SersicProfile(r_h=1.0, n=4.0)
    # Textbook value at r = r_h is exp(-b_n) where b_n ≈ 7.67 for n=4
    rho_at_rh = float(p.density(jnp.float64(1.0)))
    expected = jnp.exp(-7.67)
    assert rho_at_rh == pytest.approx(expected, rel=0.05)
```

### Step 4: add the docs

Create `docs/website/10-theory/spatial-profiles/sersic.md` following
the chapter template ([](../10-theory/spatial-profiles/plummer.md)
is a good model). The new chapter should cover: motivation,
mathematical formulation, implementation, caveats, references.

Add the chapter to `docs/website/myst.yml`:

```yaml
- title: Spatial density profiles
  children:
    - file: 10-theory/spatial-profiles/index.md
    - file: 10-theory/spatial-profiles/plummer.md
    - file: 10-theory/spatial-profiles/king.md
    - file: 10-theory/spatial-profiles/eff.md
    - file: 10-theory/spatial-profiles/sersic.md   # NEW
```

And expose the new class in `progenax/profiles/__init__.py`:

```python
from progenax.profiles.sersic import SersicProfile
__all__ = [..., "SersicProfile"]
```

The class is now part of progenax. Any builder that accepts a
`SpatialProfile` will accept it — no further registration needed.

## Step-by-step: adding a custom IMF

(imf-walkthrough)=

The pattern is the same. The protocol is `IMFProtocol`. The required
methods are `logpdf`, `cdf`, `ppf`, `sample`, and `mean_mass`. See
[](../10-theory/imfs/classic.md) for examples in `progenax/imf/`.

A worked example: a "burst-mode" IMF that follows Salpeter at high
mass but has a distinct low-mass spike (e.g. modeling formation in
a single intense burst):

```python
# progenax/imf/burst.py
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKey

class BurstIMF(eqx.Module):
    alpha: Float[Array, ""]      # High-mass slope (Salpeter-like)
    m_burst: Float[Array, ""]    # Burst mass
    sigma_burst: Float[Array, ""]   # Burst width

    def sample(self, key, n):
        # Mix Salpeter + burst Gaussian via inverse-CDF
        ...

    def logpdf(self, m):
        # Log-PDF for likelihood evaluation
        ...

    def cdf(self, m):
        # Cumulative number fraction below m
        ...

    def ppf(self, u):
        # Inverse CDF (for inverse-CDF sampling)
        ...

    def mean_mass(self):
        # Expected stellar mass
        ...
```

The same four steps (module, conformance test, validation, docs)
apply.

## Modifier layers

Modifier layers (tidal truncation, fractal substructure, mass
segregation) are not protocols — they are *functions* that take and
return phase-space arrays:

```python
# Convention for modifier-layer functions:
@jax.jit
def apply_my_modifier(positions, velocities, masses, *, my_param, **kwargs):
    """Modifier description.

    Returns
    -------
    positions, velocities, masses : modified
        The masked or transformed phase-space arrays.
    """
    # Implementation
    return new_positions, new_velocities, new_masses
```

Modifier functions are JIT-compatible by default and differentiable
in their named parameters. They live under `progenax/{tidal,cluster,populations}/`
depending on physical category.

## Validation suite contributions

Validation tests live under `tests/validation/` and have a different
character from unit tests: they verify *physical correctness* against
analytical or published values, with quantitative pass/fail criteria.

The conventions:

```{list-table}
:header-rows: 1

* - Aspect
  - Convention
* - Filename
  - `test_<topic>_physics.py` for analytic checks; `test_<topic>_validation.py` for full forward-model
* - Sample size
  - $N \ge 10^4$ for statistical agreement; smaller for closed-form
* - Tolerance
  - Default: 1% relative for analytic checks; 5% for finite-N statistics
* - Pass/fail
  - `pytest.approx(expected, rel=tolerance)` or explicit `assert`
* - Plot output
  - Optional; if generated, save to `validation/plots/<test-name>.png`
```

A good validation test answers a specific physical question
("does the Plummer DF reproduce $Q_{\mathrm{vir}} = 0.5$?") with a
numerical answer ("yes, to within $5\!\times\!10^{-3}$ at $N = 10^4$").
Vague tests ("the IC looks reasonable") are not useful and not
admitted.

## Code-style conventions

```{list-table}
:header-rows: 1

* - Convention
  - Detail
* - Type annotations
  - Use `jaxtyping` for array shapes (e.g. `Float[Array, "N 3"]`); annotate every parameter and return
* - Docstrings
  - NumPy-style; every public function gets a `Parameters`, `Returns`, `Notes`, `References` block
* - Function length
  - ≤ 100 LOC per function; ≤ 50 preferred. Split if longer
* - File length
  - ≤ 500 LOC per file; ≤ 300 preferred
* - Data-class fields
  - ≤ 15 per Equinox module; ≤ 10 preferred
* - Import order
  - stdlib → JAX → third-party → progenax (alphabetical within each group)
* - Naming
  - `snake_case` functions, `PascalCase` classes, `UPPER_CASE` module-level constants
```

The 100/500/15 limits are enforced (or strongly encouraged) at code
review time. They exist because progenax is a research codebase that
will be modified by graduate students; readability scales inversely
with file length, function length, and field count.

## CI: what tests run

Every PR triggers:

1. **Unit tests** — `pytest tests/unit/`. Fast, runs every
   conformance and signature test.
2. **Integration tests** — `pytest tests/integration/`. Slower;
   runs end-to-end builders.
3. **Validation tests** — `pytest tests/validation/`. Slowest;
   physics-anchored regression on representative samples. Not run
   on every push; tagged `@pytest.mark.slow` so CI runs them on
   merge to main only.
4. **mystmd build** — `cd docs/website && myst build --html`.
   Verifies docs build and no broken cross-references.
5. **Linting** — `ruff check`, `mypy`, `pyright` (advisory).

A PR that breaks any of (1)-(3) is blocked. A PR that breaks (4) is
flagged but not auto-blocked. (5) is advisory; long-standing lint
issues exist and are tracked separately.

## References

The contributor pattern follows the protocol-based composition
documented in [](protocols.md). The differentiability constraints
contributors must respect are at [](differentiability.md). The
broader JAX-native conventions are at [](jax-native-philosophy.md).
