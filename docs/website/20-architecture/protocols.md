---
title: Protocol-based composition
description: Interfaces for spatial sampling, velocity sampling and scalar stellar-mass distributions, with separate structural and physical compatibility requirements.
---

# Protocol-based composition

Three primary **runtime-checkable protocols** define the interfaces for
spatial sampling, velocity sampling and scalar stellar-mass distributions.
Additional protocols describe binary companions and their distributions; see
the [protocol API](../30-api/protocols.md).

```{list-table}
:header-rows: 1

* - Protocol
  - Interface
  - Implementations
* - `SpatialProfile`
  - `sample_positions`, `characteristic_radius`
  - `PlummerProfile`, `KingProfile`, `MichieProfile`, `EFFProfile`
* - `VelocityDF`
  - `sample_velocities`
  - `PlummerVelocityDF`, `KingVelocityDF`, `MichieVelocityDF`, `EFFVelocityDF`
* - `IMFProtocol`
  - `m_min`, `m_max`, `logpdf`, `cdf`, `ppf`, `sample`, `mean_mass`
  - `PowerLawIMF`, `Maschberger`, `ChabrierIMF`, `TruncatedIMF`
```

The protocols use Python's `typing.Protocol` with `@runtime_checkable`.
Implementations need not inherit from them. A runtime check verifies the
presence of the required attributes; it does not check method signatures,
return shapes, numerical accuracy or physical compatibility. Those contracts
require execution tests and model-specific validation.

`BinaryIMF` composes a primary IMF with companion statistics and exposes
`sample_primaries`, `sample_mass_ratios`, `sample_systems` and
`sample_all_masses`. It does not implement this scalar `IMFProtocol`.

This chapter documents the three protocols, the reason for choosing
protocol-based composition over inheritance, and the rules for adding
new implementations.

## The three protocols in detail

### SpatialProfile

```python
from typing import Protocol, runtime_checkable
from jaxtyping import Array, Float, PRNGKeyArray

@runtime_checkable
class SpatialProfile(Protocol):
    def sample_positions(
        self,
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
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
        key: PRNGKeyArray,
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
        key: PRNGKeyArray,
        n: int,
    ) -> Float[Array, "n"]:
        """Draw n stellar masses in solar masses; n is static under JIT."""
        ...

    def mean_mass(self) -> Float[Array, ""]:
        """Expected stellar mass in solar masses."""
        ...
```

`logpdf` evaluates a normalized number-probability density in stellar mass.
It can supply an inference likelihood when the chosen parameter domain has
valid derivatives. HMC also requires an appropriate observation model,
priors and convergence checks. `sample(key, n)` uses the concrete IMF's
sampling method; `cdf` and `ppf` provide forward and inverse distribution
functions. Their precision and endpoint handling are implementation-specific.

The current exports `progenax.protocols.IMFProtocol` and
`progenax.imf.IMFProtocol` are separate structural definitions with the same
required attributes. Generic composition uses the former; `TruncatedIMF`
checks the latter. See their [protocol](../30-api/protocols.md#api-protocols-imfprotocol)
and [IMF](../30-api/imf.md#api-imf-imfprotocol) contracts.

## Why protocols, not inheritance

Protocols allow generic builders to accept implementations without requiring
a shared base class. They coexist with inheritance: `BaseIMF` is an Equinox
module that supplies normalization, numerical inversion and sampling modes
to several IMF subclasses.

```{list-table}
:header-rows: 1

* - Reason
  - Detail
* - **Equinox compatibility**
  - Stateful implementations can be Equinox modules while satisfying a protocol structurally. Protocols do not require an additional state-owning base class.
* - **Shared implementations remain explicit**
  - A protocol specifies required attributes without supplying numerical behavior. `BaseIMF` separately owns shared behavior where subclasses use it.
* - **External implementations are first-class**
  - A user writing a custom profile in their own code does not need to import progenax's base class. As long as the class has the required methods, it satisfies the protocol.
```

The third reason is the most consequential in practice. Researchers
extending progenax for their own science can write a custom profile
(e.g. a Sérsic-like profile not in progenax's defaults) and pass it
to any builder that accepts a `SpatialProfile` — without subclassing
or registering anything.

## Composability rules

The interfaces separate mass, position and velocity sampling. Satisfying them
does not guarantee that a combination produces a valid IC. Input domains,
array shapes, units and profile/DF scales must agree. A continuum equilibrium
also requires a DF consistent with the density and its potential:

```{list-table}
:header-rows: 1

* - Combination
  - Equilibrium?
  - Use case
* - Plummer profile + Plummer DF
  - Matched continuum model when scales and anisotropy are admissible
  - Standard production IC
* - King profile + King DF
  - Matched continuum model when concentration and scales agree
  - Tidally-truncated production IC
* - EFF profile + EFF DF
  - Requires matching density, potential and boundary treatment
  - Young massive cluster
* - Plummer profile + King DF
  - **No**
  - Out-of-equilibrium starting state for relaxation studies
* - King profile + Plummer DF
  - **No**
  - Same — deliberate mismatch for studying violent relaxation
* - Plummer profile + Maxwellian
  - A generic Maxwellian is not the Plummer equilibrium DF
  - Approximate kinematic initialization; stationarity requires separate evidence
```

In generic single-population assembly, the IMF supplies masses before spatial
and velocity sampling. These masses also set the gravitational velocity scale.
Multicomponent equilibrium couples component masses, fractions and DFs through
a shared potential; see [](../10-theory/spatial-profiles/multimass-equipartition.md).
Center-of-mass subtraction, tidal cuts, rotation and optional virial rescaling
are subsequent transformations, not stationarity certificates.

## Adding a new implementation

To add a new spatial profile (e.g. a hypothetical Sérsic-like
profile), the contract is:

1. **Use `eqx.Module`** for state that should participate in JAX PyTrees.
2. **Implement the two required methods** (`sample_positions`,
   `characteristic_radius`).
3. **Expose any profile-specific radii** (`r_h`, `r_t`, `a`, …) as
   normal fields if users need them.
4. **Use `jax.numpy` only** in the implementation (no numpy/scipy).
5. **Use fixed-shape JAX operations** for compiled sampling; validate the
   derivative path rather than inferring it from a loop construct.

```python
from progenax import PlummerProfile
from progenax.protocols import SpatialProfile

profile = PlummerProfile(r_h=1.0)
assert isinstance(profile, SpatialProfile)
```

This checks the interface of an existing implementation. A custom profile
must additionally demonstrate its position distribution, units, domains and
derivatives. `build_spatial_ic` accepts an explicit profile/DF pair;
`build_cluster` dispatches matched DFs for its supported profile classes and
does not accept arbitrary custom classes through this protocol alone.

## Validation: enforcing the contract

`tests/unit/test_protocols.py` checks Plummer, King and EFF profile/DF
classes, five scalar IMF instances and rejection of missing attributes.
The IMF suite adds other implementation-specific checks. An example
structural assertion is:

```python
def test_plummer_satisfies_spatial_profile():
    p = PlummerProfile(r_h=1.0)
    assert isinstance(p, SpatialProfile)
```

Tests like these detect missing required methods, such as renaming
`sample_positions`. Renaming an optional `density` helper does not violate
`SpatialProfile`. Signature, shape, distribution and derivative tests belong
to the concrete implementation and workflow owners; a structural check
cannot replace them.

## References

The Python `typing.Protocol` machinery is documented in PEP 544. The
"runtime-checkable protocol" pattern is increasingly standard in
modern Python data-science libraries. progenax's specific
mass-first / r_h-canonical design follows the IC-redesign decisions
documented at [](ic-redesign-history.md) and
[](../90-development-log/2026-02-12-ic-redesign.md).
