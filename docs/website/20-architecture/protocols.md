---
title: Protocol-based composition
description: progenax's three runtime-checkable protocols — SpatialProfile, VelocityDF, IMFProtocol — that enable mix-and-match composition of any spatial profile with any velocity DF and any IMF.
---

# Protocol-based composition

progenax exposes three **runtime-checkable protocols** that define
the interfaces for the orthogonal IC ingredients:

```{list-table}
:header-rows: 1

* - Protocol
  - Interface
  - Implementations
* - `SpatialProfile`
  - `sample_positions`, `characteristic_radius`
  - `PlummerProfile`, `KingProfile`, `EFFProfile`
* - `VelocityDF`
  - `sample_velocities`
  - `PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF`
* - `IMFProtocol`
  - `logpdf`, `cdf`, `ppf`, `sample`, `mean_mass`
  - `PowerLawIMF`, `Maschberger`, `ChabrierIMF`, `BinaryIMF`
```

The protocols use Python's `typing.Protocol` with
`@runtime_checkable`, so any class that *implements the methods* is
automatically a valid `SpatialProfile` (etc.) — no inheritance
required. This enables **mix-and-match composition**: any
`SpatialProfile` can pair with any `VelocityDF` (subject to
equilibrium constraints), and any combination can pair with any
`IMFProtocol`.

This chapter documents the three protocols, the reason for choosing
protocol-based composition over inheritance, and the rules for adding
new implementations.

## The three protocols in detail

### SpatialProfile

```python
from typing import Protocol, runtime_checkable
from jaxtyping import Array, Float, PRNGKey

@runtime_checkable
class SpatialProfile(Protocol):
    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKey,
        *,
        truncate: float | None = None,
    ) -> Float[Array, "N 3"]:
        """Draw N positions from ρ(r). Returns (N, 3) Cartesian."""
        ...

    def characteristic_radius(self) -> Float[Array, ""]:
        """Return the characteristic radius used by generic builders."""
        ...
```

The protocol requires position sampling and a characteristic radius.
Individual profile classes may expose additional analytic helpers such
as density or cumulative-mass functions, but those are not part of the
runtime-checkable `SpatialProfile` contract.

### VelocityDF

```python
@runtime_checkable
class VelocityDF(Protocol):
    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKey,
        *,
        G: float,
    ) -> Float[Array, "N 3"]:
        """Draw N velocities consistent with the DF given positions+masses."""
        ...
```

The signature follows the data-flow direction in IC builders:
positions and masses are already known when velocities are sampled.

### IMFProtocol

```python
@runtime_checkable
class IMFProtocol(Protocol):
    m_min: float
    m_max: float

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Log-PDF for likelihood evaluation."""
        ...

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative number fraction below mass m."""
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF."""
        ...

    def sample(
        self,
        key: PRNGKey,
        n: int,
    ) -> Float[Array, "N"]:
        """Draw N stellar masses."""
        ...

    def mean_mass(self) -> float:
        """Expected stellar mass."""
        ...
```

The `logpdf` method is what makes IMFs HMC-compatible — it allows
the IMF parameters (e.g. Maschberger's $\alpha$) to be inferred from
data. `sample(key, n)` uses inverse-CDF sampling where available; `cdf`
and `ppf` provide the forward and inverse distribution functions.

## Why protocols, not inheritance

The natural object-oriented alternative would be a `BaseProfile`
abstract class with `Plummer`, `King`, `EFF` as subclasses. progenax
deliberately rejects this for three reasons:

```{list-table}
:header-rows: 1

* - Reason
  - Detail
* - **Equinox compatibility**
  - Equinox modules cannot inherit from generic abstract bases. A protocol-based pattern lets each profile be its own self-contained `eqx.Module`, with no base class machinery.
* - **No "base implementation" anti-pattern**
  - With inheritance, there is always pressure to put "default" implementations in the base class. With protocols, each implementation must implement the methods explicitly — no accidental inheritance of the wrong default.
* - **External implementations are first-class**
  - A user writing a custom profile in their own code does not need to import progenax's base class. As long as the class has the required methods, it satisfies the protocol.
```

The third reason is the most consequential in practice. Researchers
extending progenax for their own science can write a custom profile
(e.g. a Sérsic-like profile not in progenax's defaults) and pass it
to any builder that accepts a `SpatialProfile` — without subclassing
or registering anything.

## Composability rules

The three protocols are **orthogonal** in the sense that any
combination of $(\text{SpatialProfile}, \text{VelocityDF}, \text{IMFProtocol})$
produces a valid IC. The combination is *physical* (i.e. in
equilibrium) only if profile and DF are matched:

```{list-table}
:header-rows: 1

* - Combination
  - Equilibrium?
  - Use case
* - Plummer profile + Plummer DF
  - Yes
  - Standard production IC
* - King profile + King DF
  - Yes
  - Tidally-truncated production IC
* - EFF profile + EFF DF
  - Yes
  - Young massive cluster
* - Plummer profile + King DF
  - **No**
  - Out-of-equilibrium starting state for relaxation studies
* - King profile + Plummer DF
  - **No**
  - Same — deliberate mismatch for studying violent relaxation
* - Plummer profile + Maxwellian
  - Approximately
  - Maxwellian is not the exact Plummer DF; gives small initial relaxation
```

The IMF is *fully* orthogonal to the profile + DF pairing — any IMF
works with any (matched or mismatched) spatial choice. The IMF only
sets the *masses*; the profile + DF set *positions* and *velocities*.

## Adding a new implementation

To add a new spatial profile (e.g. a hypothetical Sérsic-like
profile), the contract is:

1. **Inherit from `eqx.Module`** for PyTree compatibility.
2. **Implement the two required methods** (`sample_positions`,
   `characteristic_radius`).
3. **Expose any profile-specific radii** (`r_h`, `r_t`, `a`, …) as
   normal fields if users need them.
4. **Use `jax.numpy` only** in the implementation (no numpy/scipy).
5. **Use `jax.lax.scan` or vmap** for any iteration.

```python
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKey

class SersicProfile(eqx.Module):
    r_h: Float[Array, ""]
    n: Float[Array, ""]   # Sérsic index

    def density(self, r):
        # Sérsic density formula
        b_n = 2 * self.n - 1.0/3.0  # Approximation
        rho = jnp.exp(-b_n * jnp.power(r / self.r_h, 1.0 / self.n))
        return rho

    def cumulative_mass(self, r):
        # Numerical integration via lax.scan or analytic
        return ...   # Implementation here

    def sample_positions(self, masses, key, *, truncate=None):
        # Inverse-CDF on cumulative_mass
        return ...
```

Once the class is written, `isinstance(SersicProfile(...), SpatialProfile)`
returns `True` and the profile is usable with any progenax builder
that accepts `SpatialProfile`. No registration, no factory, no other
machinery.

## Validation: enforcing the contract

progenax's test suite includes "protocol-conformance" tests for
every shipped implementation:

```python
def test_plummer_satisfies_spatial_profile():
    p = PlummerProfile(r_h=1.0)
    assert isinstance(p, SpatialProfile)
    # Plus per-method tests verifying signatures and return shapes
```

Tests like these catch protocol-breaking changes (e.g. renaming
`density` to `evaluate_density`) at CI time before they reach a
release. They also serve as executable documentation of the protocol
contract.

## References

The Python `typing.Protocol` machinery is documented in PEP 544. The
"runtime-checkable protocol" pattern is increasingly standard in
modern Python data-science libraries. progenax's specific
mass-first / r_h-canonical design follows the IC-redesign decisions
documented at [](ic-redesign-history.md) and
[](../90-development-log/2026-02-12-ic-redesign.md).
