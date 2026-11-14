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
single-population builders) use `STELLAR` as the default; the core APIs require
an explicit gravitational constant or unit system.

## Dynamical arrays and stellar labels

`build_cluster` and its named aliases take supplied masses in the selected
`UnitSystem`'s mass unit. Profile lengths, softening, tidal and anisotropy radii
use its length unit; angular frequencies use its inverse time unit. Returned
positions, velocities and masses use those same dynamical units. The wrapper
does not infer or convert supplied array units.

IMF samplers return masses in $\Msun$. With `n=` and no IMF, each star has
physical mass $1\,\Msun$. These generated masses are converted to the selected
dynamical mass unit at the wrapper boundary. In CGS, `n=` therefore creates
one-solar-mass stars with numeric masses $1.9884\times10^{33}$ grams each.

Let $M_0$, $L_0$ and $T_0$ be the selected mass, length and time unit sizes
in grams, centimetres and seconds, and let $M_\odot$ be one solar mass in
grams. The dimensionless conversion factor $s_M$ is the number of solar
masses per native mass unit:

```{math}
s_M = \frac{M_0}{M_\odot}, \qquad
m_{\rm native} = \frac{m_{\rm IMF}}{s_M}, \qquad
m_{\rm ZAMS} = s_M m_{\rm native}, \qquad
G_{\rm native} = G_{\rm CGS}\frac{M_0 T_0^2}{L_0^3}.
```

Here $m_{\rm IMF}$ and $m_{\rm ZAMS}$ are numeric masses in solar units, and
$m_{\rm native}$ is the numeric dynamical mass. **`stellar_radii` stays in
$\Rsun$** in every system: it is a stellar label, not a position-unit radius.
To use it geometrically, convert $\Rsun$ to the chosen length unit separately.
The Tout fit's mass and metallicity domains still apply.

The low-level `build_spatial_ic` takes `G` explicitly and adds the keyword-only
`mass_scale_to_msun` for stellar-label evaluation. Its default 1.0 preserves
solar-mass callers; for CGS pass `CGS.mass_scale_cgs / STELLAR.mass_scale_cgs`.
`G` alone cannot determine a mass unit independently of length and time.
Multicomponent builders have a separate mass-label contract; their alternate-
unit behavior is not qualified by these single-population checks.

### Binary population boundaries

`build_binary_cluster` and `build_cataloged_binary_cluster` draw primary and
companion masses in $\Msun$. The empirical companion relations and
`TotalMass(M)` budget use those solar masses regardless of the selected
dynamical units. After the budget cut, the shared builder converts masses to
native units for COM sampling, orbital resolution and returned particle arrays.
Stellar radii remain in $\Rsun$.

The companion model computes the orbital semimajor axis before this mass
conversion. Its numerical gravitational constant is therefore
$G_{\rm companion}=G_{\rm native}/s_M$: it uses native length and time units
but solar masses. This is the same physical gravitational constant expressed
with the mass unit required by the empirical sampling boundary. Direct
companion-model calls must supply that consistent value.

Tests cover the retained Plummer construction with both Q paths, compact and
masked products, all three budgets, and independent/Moe companions. Inactive
ghost Cartesian coordinates are sanitization values; only real particles have
physical unit covariance. See [](primordial-system-handoff.md).

:::{admonition} EFF numerical scale convention
:class: note
The single-population EFF velocity table begins at $r_{\min}=10^{-5}a$, where
$a$ is the EFF scale radius. Its fractional inner cutoff therefore stays the
same when all lengths change units. The retained 16-entry STELLAR/CGS check
passes the existing $10^{-10}$ component-relative round-trip budget. This
numerical repair preserves the $a=1$ table and changes tables at other scales;
hard-truncation mismatch and wider-domain qualification remain separate.
See [](../50-validation/cluster-builders.md).
:::

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
  - Unit choices are passed as arguments; core code does not read a context manager
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

Explicit arguments remove dependence on a mutable global default. Bare arrays
still allow inconsistent input units, so explicit `G` is a convention to check.
Unit conversion and stellar-label contracts need their own tests.

## Architectural shape

The explicit-units rule has a specific architectural shape: it
applies to **core APIs** but not to **convenience wrappers**. A
convenience wrapper *resolves* `units=None` to `DEFAULT_UNITS` at
the API boundary, then passes the resolved value through to the core:

The real `build_cluster` convenience builder is exactly this pattern. Its body resolves
`units=None` to `DEFAULT_UNITS` (= `STELLAR`) at the API surface, then threads the
explicit `G = units.G` into the dynamical core. It also converts generated
solar-mass draws to native mass units and forwards `mass_scale_to_msun` for
stellar labels:

```python
from progenax import build_cluster, PlummerProfile
from jaxstro.units import STELLAR
import jax

jax.config.update("jax_enable_x64", True)
key = jax.random.PRNGKey(0)
ic = build_cluster(PlummerProfile(r_h=1.0), n=1000, key=key)                # units=None -> STELLAR
ic = build_cluster(PlummerProfile(r_h=1.0), n=1000, key=key, units=STELLAR)  # explicit units
```

The same physical one-parsec Plummer scale in CGS uses an explicitly converted
profile length. The wrapper converts the generated one-solar-mass stars:

```python
from jaxstro.units import CGS

ic_cgs = build_cluster(
    PlummerProfile(r_h=STELLAR.convert_length(1.0, to=CGS)),
    n=16, key=key, units=CGS, Q=None,
)
mass_msun = CGS.convert_mass(ic_cgs.masses, to=STELLAR)  # each value is 1.0
radius_rsun = ic_cgs.stellar_radii  # solar radii, independent of position units
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

`tests/unit/kinematics/test_api.py::TestExplicitGRequired` checks that omitting
`G` raises on the five velocity-DF surfaces, the velocity pipeline and
`MultiComponentCluster.sample_cluster`. It is a bounded surface check, not
a scan proving every public function's dimensional correctness.

`tests/unit/builders/test_cluster_units.py` represents the same physical
16-star Plummer fixture in STELLAR, PLANETARY and CGS. It checks supplied,
equal-star and IMF mass paths, both Q settings, CGS tidal/rotation labels,
the low-level conversion keyword, JIT and first derivatives. It reuses the
suite's exact-formula round-trip tolerance; it is not a new equilibrium or
stochastic-precision qualification. See [](../50-validation/cluster-builders.md).

`tests/unit/builders/test_binary_units.py` exercises the separate binary
construction boundary, including native-mass orbital periods, contact margins,
stable catalogue identities, JIT and first derivatives. Its deterministic
round trips do not qualify empirical extrapolation or dynamical stationarity.

## References

The explicit-units convention is a deliberate departure from the
astropy / pint / unyt ecosystem. The closest analogue is the
{cite:t}`Aarseth1974` "internal units" tradition in N-body codes,
which uses bare floats with documented unit conventions. progenax's
specific architectural shape is documented at
[](../90-development-log/2026-02-12-ic-redesign.md).
