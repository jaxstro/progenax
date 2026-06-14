---
title: Units policy
description: progenax's strict explicit-units convention — no global context managers, no `get_G()` defaults, every core API requires explicit `units` or `G`. The architectural rationale and the convenience-wrapper exception.
---

# Units policy

progenax adopts a strict **explicit-units convention**: no global
state, no implicit unit defaults, no `get_G()`-style context managers
in core APIs. Every function that consumes a gravitational constant
takes either an explicit `G` value or an explicit `units` argument
carrying one. This is more verbose than the alternatives — and the
verbosity is the point.

This chapter documents the convention, lists the three unit systems
progenax supports, and explains why the alternatives (global $G$,
`get_G()` context managers, lazy unit defaults) were rejected.

## The three unit systems

```{list-table}
:header-rows: 1

* - System
  - Mass
  - Length
  - Time
  - Best for
* - **STELLAR** (default)
  - $\Msun$
  - pc
  - Myr
  - Star clusters, galaxies; G ≈ 0.00450 pc³ M☉⁻¹ Myr⁻²
* - **PLANETARY** / **BINARY**
  - $\Msun$
  - AU
  - yr
  - Planetary systems, binary stars; G ≈ 39.478 AU³ M☉⁻¹ yr⁻²
* - **CGS**
  - g
  - cm
  - s
  - Microphysics, gas dynamics; G = 6.674 × 10⁻⁸ cm³ g⁻¹ s⁻²
```

`progenax.DEFAULT_UNITS = STELLAR`. The convenience wrappers (high-level
builder functions) use `STELLAR` as the default; the core APIs require
the user to be explicit.

## Three rules

```{list-table}
:header-rows: 1

* - Rule
  - Implication
* - **Core APIs require explicit `G` or `units`**
  - No defaults; if you want STELLAR units, pass `units=STELLAR` or `G=STELLAR.G`
* - **Convenience wrappers may default to `DEFAULT_UNITS`**
  - But this resolution happens *only* at the API boundary — once inside the call stack, $G$ is explicit
* - **No global state, no context managers in core code**
  - A `with units.STELLAR:` block is forbidden; if it exists at all, it lives in `progenax.contrib` or `progenax.legacy`, never in the core
```

## Why explicit beats implicit

The conventional wisdom in scientific Python is that "users want
defaults" — i.e. that requiring users to pass units everywhere is
annoying. In progenax's experience the opposite is true at scale:

```{list-table}
:header-rows: 1

* - Pain point with implicit defaults
  - What progenax's explicit policy prevents
* - **Silent unit mismatch**
  - User samples positions in STELLAR, computes energy in PLANETARY → recovered $Q_{\mathrm{vir}}$ is off by $G_{\mathrm{stellar}}/G_{\mathrm{planetary}} \approx 8800$. Hours of debugging
* - **Notebook order-dependence**
  - Cell 5's `units.set_default(STELLAR)` affects cell 8's `compute_energy()`. Re-run cells in different order → different results
* - **Library-version coupling**
  - Library X v1.0 defaults to STELLAR; v1.1 defaults to CGS. Existing user code silently changes physics
* - **Test isolation**
  - Test A sets a global default; test B fails because it inherited it. Tests are not order-independent
```

The explicit-units convention has *zero* of these failure modes by
construction. It costs $\sim 10$ characters per call to write
`G=STELLAR.G`. For a research codebase that runs a $\sim 10^6$-step
HMC chain, those characters are negligible — and never producing the
"why is my energy 8800× wrong" diagnostic question is a substantial
quality-of-life win.

## Architectural shape

The explicit-units rule has a specific architectural shape: it
applies to **core APIs** but not to **convenience wrappers**. A
convenience wrapper *resolves* `units=None` to `DEFAULT_UNITS` at
the API boundary, then passes the resolved value through to the core:

The real `build_cluster` convenience builder is exactly this pattern. Its body resolves
`units=None` to `DEFAULT_UNITS` (= `STELLAR`) at the API surface, then threads the
explicit `G = units.G` into every core call (`build_spatial_ic`, the velocity DF) — the
core never sees a `None`:

```python
from progenax import build_cluster, PlummerProfile
from jaxstro.units import STELLAR

# units=None -> DEFAULT_UNITS (= STELLAR) is resolved inside build_cluster, which then
# passes the explicit G = units.G to the core build_spatial_ic.
ic = build_cluster(PlummerProfile(r_h=1.0), n=1000, key=key)                # units=None -> STELLAR
ic = build_cluster(PlummerProfile(r_h=1.0), n=1000, key=key, units=STELLAR)  # explicit units
```

The wrapper resolves `None` at the API surface; the core sees only explicit values. This
satisfies the "no globals in core" rule while still providing the ergonomic default for
users who don't want to think about units. (The same holds for the named aliases —
`build_plummer_cluster`, `build_king_cluster`, …, — which construct the profile and
delegate to `build_cluster`.)

## Each package defines its own DEFAULT_UNITS

In the broader jaxstro ecosystem, each package has its own
`DEFAULT_UNITS`:

```{list-table}
:header-rows: 1

* - Package
  - `DEFAULT_UNITS`
  - Rationale
* - **progenax**
  - STELLAR
  - Star-cluster ICs are most natural in STELLAR
* - **gravax**
  - STELLAR
  - N-body integration in cluster regimes
* - **fluxax**
  - STELLAR
  - Photometry; flux units in cgs are tracked separately
* - **stellax** (planned)
  - STELLAR
  - Stellar evolution; tracks evolve in Myr/$\Msun$
* - **planetax** (hypothetical)
  - PLANETARY
  - Planetary dynamics
```

Wrappers in different packages may resolve `units=None` differently;
core APIs in any package require explicit values. This decoupling
lets each package choose the most natural default for its science
domain.

## What about astropy units?

A reader from the astropy ecosystem might ask why progenax doesn't
use `astropy.units` Quantity objects. Two reasons:

1. **JAX compatibility.** Astropy Quantity is a numpy subclass that
   does not play well with `jax.numpy`. Round-tripping through
   `Quantity` breaks JIT and gradient flow.
2. **Performance overhead.** Astropy Quantity wraps every operation
   with unit-checking. For HMC chains running $\sim 10^6$ likelihood
   evaluations, this overhead is unacceptable.

The progenax approach is to use plain `jax.Array`s with
**convention-based units** — the `UnitSystem` object documents *what*
the numbers mean, but the numbers themselves are bare floats. This
is the same trade-off N-body codes have made for decades; progenax
is consistent with NBODY6, McLuster, COSMIC.

For unit-aware *post-processing* (e.g. converting an N-body output
to observed magnitudes for a comparison plot), use astropy. For
unit-aware *inside* progenax core, the bare-float convention is the
right answer.

## The audit pattern

The progenax test suite includes a "unit audit" that scans all
public-facing core functions for hidden defaults:

```python
def test_no_implicit_G_in_core():
    """Every core function must accept G explicitly (no default)."""
    public_modules = [progenax.profiles, progenax.kinematics, progenax.builders]
    for mod in public_modules:
        for name in dir(mod):
            f = getattr(mod, name)
            if callable(f) and inspect.signature(f).parameters.get('G') is not None:
                # If G is a parameter, it must not have a default
                assert f.__signature__.parameters['G'].default is inspect.Parameter.empty, (
                    f"{mod.__name__}.{name} has default G — violates explicit-units policy"
                )
```

A regression that introduces an implicit `G` default fails CI.

## When the rule has been broken

Two regressions historically caused real problems:

1. **2025-09**: a refactor introduced `G = 0.00450` as a default in
   `compute_potential_energy`. Users who passed CGS positions got
   silent factor-$\sim 8800$ errors in their energy diagnostics.
   Fixed by removing the default; test added to prevent regression.

2. **2025-11**: a context-manager-based units system was prototyped
   in the legacy package; it leaked state across tests in CI. Removed
   from the core; the prototype is preserved in `progenax.legacy`
   for reference.

Both failures motivated the strict no-defaults convention progenax
now enforces. The convention is "boring" — it doesn't enable any
fancy patterns — but it eliminates an entire class of debugging
sessions.

## References

The explicit-units convention is a deliberate departure from the
astropy / pint / unyt ecosystem. The closest analogue is the
{cite:t}`Aarseth1974` "internal units" tradition in N-body codes,
which uses bare floats with documented unit conventions. progenax's
specific architectural shape is documented at
[](../90-development-log/2026-02-12-ic-redesign.md).
