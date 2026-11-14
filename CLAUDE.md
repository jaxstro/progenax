# CLAUDE.md - progenax

## Writing

Remove all mannered prose. State the rule, the measurement and the mechanism in plain
sentences. No rhetorical emphasis, no asides, no metaphors, no dramatised failure stories;
the number, the date and the mechanism carry the point. This applies to chat, docstrings,
commit messages, `STATUS.md` and everything under `docs/`.

## Overview

Differentiable initial conditions for N-body simulations in JAX. Part of the **jaxstro ecosystem**.

**Status:** research software; public release is not qualified. The
[README](README.md) and [release assessment](docs/website/95-release/index.md)
state current scope and outstanding gates. Matched King, Michie and
multi-component models use lowered-isothermal or Eddington DFs within their
documented domains. The finite-edge EFF approximation has an equilibrium
limitation. The binary population API composes primary IMFs, companion models
and spatial targets, including the Moe & Di Stefano period–mass-ratio–eccentricity
coupling. Retained numerical checks do not establish every parameter domain,
population-expectation gradient or inference application.

Gravoturb is an experimental, repository-only subsystem under
`src/experimental/`; it is excluded from the released wheel. Its density,
formation-rate and inference components have separate assumptions and
validation scopes. Experimental results do not qualify the released core.

## Quick Commands

Use **uv** (the project `.venv`); do not use conda. `env -u VIRTUAL_ENV` avoids an
outer-venv clash and `--no-sync` runs against the installed env without re-locking.

```bash
# Install (released core + experimental extras: blackjax, optax for the inference layer)
env -u VIRTUAL_ENV uv pip install -e ".[dev,experimental]"

# Released-core checks. Slow equilibrium cases are separate from the fast subset.
# Use pytest-xdist with XLA threads capped.
# FAST GATE: excludes @pytest.mark.slow
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
# FULL GATE: includes slow cases
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto

env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/ -q             # released-core unit tests
env -u VIRTUAL_ENV uv run --no-sync pytest tests/integration/ -q      # integration tests
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/ -q       # physics validation tests

# Experimental gravoturb subsystem (repo-only; needs src/experimental on the path).
# Consult tests/experimental/README.md for its separate test contract.
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync pytest tests/experimental -q -n auto
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -m gravoturb.validation.acceptance   # AC1-AC17
PYTHONPATH=src:src/experimental env -u VIRTUAL_ENV uv run --no-sync python -m gravoturb.validation.cluster_acceptance  # AC-IC0-IC7
```

## Units Policy (progenax)

**DEFAULT_UNITS:** `STELLAR` (Msun, pc, Myr)

Rules:
- Core APIs require explicit `G` or `units` (or objects that carry units).
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.
- Do **not** use global context managers or `get_G()` in core code.

Example:
```python
from jaxstro.units import STELLAR
from progenax import DEFAULT_UNITS

# Core (explicit)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)

# Wrapper (optional)
velocities = df.sample_velocities(positions, masses, key, G=DEFAULT_UNITS.G)
```

## Key Patterns

### Protocol-Based Composition

Any `SpatialProfile` pairs with any `VelocityDF`:

```python
from progenax.protocols import SpatialProfile, VelocityDF

profile: SpatialProfile = PlummerProfile(r_h=1.0)
df: VelocityDF = KingVelocityDF(W0=7.0, r_c=1.0, r_t=10.0)

# Mix Plummer positions with King velocities
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

### Equinox Modules

All stateful classes are Equinox modules (immutable PyTrees):

```python
class PlummerProfile(eqx.Module):
    r_h: Float[Array, ""]  # Half-mass radius
    a: Float[Array, ""]    # Scale radius (computed)
```

### Differentiability

All sampling uses `jax.lax.scan` with fixed iterations (NOT `while_loop`):

```python
def loss(r_h):
    profile = PlummerProfile(r_h=r_h)
    positions = profile.sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(positions, axis=1))

jax.grad(loss)(1.0)  # Fully differentiable!
```

## Module Quick Reference

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `profiles/` | Spatial density profiles | `PlummerProfile`, `KingProfile`, `MichieProfile`, `EFFProfile` |
| `kinematics/` | Velocity DFs + transforms | `PlummerVelocityDF`, `KingVelocityDF`, `MichieVelocityDF`, `EFFVelocityDF` |
| `imf/` | Initial mass functions + binary stats | `PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `BinaryIMF`, `MoeJointOrbit`; environment-dependent IMF via `BirthEnvironment` + `env_to_imf_params()` |
| `binaries/` | Orbital mechanics + connector + diagnostics | `KeplerElements`, `resolve_binary_components()`, `IndependentCompanions`, `MoeCompanions`, `binary_energy_budget()` |
| `analytical/` | Test cases with exact solutions | `two_body_kepler()`, `three_body_figure_eight()` |
| `diagnostics/` | Substructure + mass-segregation diagnostics | `compute_q_parameter()` (CW04 Q, A=πR²), `q_approx` (differentiable kNN), `energy_sorted_segregation()` |
| `protocols.py` | Runtime-checkable protocols | `SpatialProfile`, `VelocityDF`, `IMFProtocol`, `CompanionModel` |
| `builders.py` | IC assembly + binary-cluster composition | `build_spatial_ic()`, legacy `build_binary_cluster()`, opt-in `build_cataloged_binary_cluster()`, `Systems`/`Stars`/`TotalMass`, `ICResult` |
| `cluster/` | Multi-component equilibrium (Engine A: lowered-isothermal DF; Engine B: density-defined Eddington via `from_density_profiles`) + primordial segregation | `MultiComponentCluster`, `energy_sorted_segregation()` |
| `tidal.py` | Tidal physics | `jacobi_radius()`, `apply_tidal_truncation()` |

## Critical Formulas

### Plummer Scale Radius

```python
# From half-mass radius to scale radius
a = r_h * jnp.sqrt(2**(2/3) - 1)  # ≈ 0.7664 * r_h

# WRONG (was a historical bug):
# a = r_h / jnp.sqrt(2**(2/3) - 1)  # INVERTED!
```

### Virial Ratio

```python
Q = T / |V|  # Q ≈ 0.5 for equilibrium (virial theorem: 2T + V = 0)
```

**Convention:**
- Q = 0.5: Virial equilibrium (2T + V = 0)
- Q < 0.5: Subvirial (cold, collapsing)
- Q > 0.5: Supervirial (hot, expanding)

### Kepler's Third Law

```python
T = 2 * jnp.pi * jnp.sqrt(a**3 / (G * M_total))  # Orbital period
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

### JAX Float64 (Automatic)

**progenax automatically enables float64** via `jaxstro.jaxconfig.enable_high_precision()` at import time. This is the standard approach across the jaxstro ecosystem - high precision is configured before any JAX arrays are created.

You don't need to do anything - just `import progenax` and you get float64.

### Unit System Consistency

Always pass consistent G values through the pipeline:

```python
from jaxstro.units import STELLAR

# CORRECT - consistent G throughout
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
energy = compute_total_energy(positions, velocities, masses, G=STELLAR.G)

# WRONG - mixing unit systems
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
energy = compute_total_energy(positions, velocities, masses, G=PLANETARY.G)  # WRONG!
```

## Test Structure

```text
tests/                   released-core test tiers
├── unit/                unit tests
│   ├── imf/             IMF tests (PowerLaw, Chabrier, environment, Binary, Moe full P-q-e)
│   ├── profiles/        Profile tests (Plummer, King, EFF)
│   ├── kinematics/      Velocity DF tests + anisotropy
│   ├── analytical/      Analytical test case tests
│   ├── binaries/        Kepler + period/ecc/companion + assembly + diagnostics + energy-budget
│   ├── cluster/         MultiComponentCluster (Engine A + Engine B) + mass-segregation IC tests
│   ├── dynamics/        Virial / energy utilities
│   └── substructure/    CW04 Q diagnostic (compute_q_parameter, q_approx, baselines)
├── integration/         integration tests
│   ├── test_jax_compatibility.py     JIT/grad/vmap tests
│   ├── test_units_through_pipeline.py  G threading (audit C1)
│   ├── test_binary_cluster.py        build_binary_cluster (budgets + companions)
│   └── test_end_to_end.py            Full IC → energy checks
└── validation/          physics validation tests
    ├── test_plummer_physics.py      Plummer equilibrium
    ├── test_king_physics.py         King true-DF equilibrium + c(W0)
    ├── test_multimass_equilibrium_physics.py  MultiComponentCluster shared-potential equilibrium
    ├── test_engine_b_physics.py     Engine B anchors (King A-vs-B, headline Q_j, OM beta, DF fidelity, AD-vs-FD)
    ├── test_eff_physics.py          EFF Eddington-inversion DF
    ├── test_binary_physics.py       Kepler's laws
    ├── test_analytical_physics.py   Figure-eight closure/L=0 + two-body conservation + planet provenance
    └── test_imf_physics.py          IMF distributions

tests/experimental/      gravoturb (repo-only, PYTHONPATH=src:src/experimental)
├── unit/                BM19/PP20/PN11/PDF + GRF/copula/tail/sampling/pipeline + Q + grads
│                        + inference: Gaussianization/projection/CIC/Fisher/POT-tail/HMC
└── validation/          AC1-AC17 acceptance assertions
```

## Physics Validation Results

Retained examples from `tests/validation/` use Q ≡ T/|V|, with Q = 0.5 as
a necessary virial condition. The table is a historical summary, not a fresh
measurement or general equilibrium certificate. Matched-DF tests use no external
virial rescale; finite-edge EFF has the documented approximation limit:

| Test | Result | Expected |
|------|--------|----------|
| Plummer virial Q = T/\|V\| | 0.502 | 0.5 |
| King true-DF virial Q (unscaled) | ~0.51 | 0.5 |
| EFF Eddington-DF virial Q (γ=5, mild trunc) | ~0.50 | 0.5 |
| King c(W₀) vs King (1966) Table II | max \|Δlog₁₀c\| = 0.002 (hi-res solve: xi_max=400, n_ode_points=8000; defaults ≤0.012) | ≤ 0.03 (Table II) |
| Kepler energy & angular momentum | conserved to ~1e-16 | exact |
| Bound particles | 100% | 100% |
| Binary period (Kepler III) | exact to 1e-10 | 2π√(a³/GM) |
| Engine B King A-vs-B (σ_1d(r) dev / radial KS, N=2e4) | 3e-4 / 2e-4 | < 0.02 both |
| Engine B halo+core global Q (N=30k, unscaled) | 0.4976 | 0.5 ± 0.02 |
| Engine B OM β_halo(r) vs r²/(r²+r_a²) | max dev 0.028 | < 0.05 |

## Public API

All public symbols exported from `progenax.__init__`:

**Profiles**: `PlummerProfile`, `KingProfile`, `MichieProfile`, `EFFProfile`, `solve_king_profile()`, `solve_michie_profile()`

**Velocity DFs**: `PlummerVelocityDF`, `KingVelocityDF`, `EFFVelocityDF` (Plummer/EFF take an optional `anisotropy_radius` for Osipkov-Merritt radial anisotropy, β(r)=r²/(r²+r_a²)), `MichieVelocityDF` (self-consistent anisotropic King = Michie 1963 + King 1966 cutoff; pairs with `MichieProfile`), `apply_solid_body_rotation()`, `apply_differential_rotation()`

**IMFs**: `PowerLawIMF`, `ChabrierIMF`, `Maschberger`, `TruncatedIMF`, `BinaryIMF`; mass-ratio: `FlatMassRatio`, `PowerLawMassRatio`, `TwinPeakedMassRatio`, `MoeDiStefano2017`, `MoeDiStefano2017Full`, `MoePeriod`, `MoeJointOrbit`; fractions: `ConstantBinaryFraction`, `MassDependentBinaryFraction`. The environment-dependent stellar IMF (Marks+2012 / Jeřábková+2018 α₃ relations) is the functional `BirthEnvironment` + `env_to_imf_params()` API — **not** a galaxy-wide IGIMF integration and **not** an `IGIMF`/`EnvironmentIMF` class (those never existed; audit R7).

**Binaries**: `KeplerElements`, `BinaryOrbitalState`, `compute_period()`, `period_to_semimajor_axis()`, `LogUniformPeriod`, `LogNormalPeriod`, `SanaOBPeriod`, `ThermalEccentricity`, `UniformEccentricity`, `MoeEccentricity`; **connector/composition**: `resolve_binary_components()`, `ResolvedBinaries`, `CatalogedBinaryClusterIC`, `PrimordialSystemCatalog`, `CompanionElements`, `IndependentCompanions`, `MoeCompanions`; **diagnostics**: `relative_energy()`, `find_bound_pairs()`, `find_bound_multiples()`, `primordial_survival()`, `binary_energy_budget()`, `BinaryEnergyBudget`

**Analytical**: `two_body_kepler()`, `three_body_figure_eight()`, `earth_sun_2body()`, `solar_system_inner_4()`, `solar_system_full()`, `harmonic_oscillator()`

**Utilities**: `build_spatial_ic()`, `build_binary_cluster()`, `build_cataloged_binary_cluster()`, `Systems`, `Stars`, `TotalMass`, `ICResult`, `compute_kinetic_energy()`, `compute_potential_energy()`, `to_com_frame()`, `virial_scale()`, `compute_stellar_radii()`, `jacobi_radius()`, `apply_tidal_truncation()`, `energy_sorted_segregation()`, `MultiComponentCluster` (Engine A constructors `from_components`/`from_mass_segregation`/`from_imf`; Engine B `from_density_profiles` — prescribed Plummer/EFF/King densities, shared-Ψ Eddington/OM DFs)

(The fractal-substructure generator `generate_fractal_positions()` was removed in the 2026-06 clean-room rewrite; turbulent/fractal ICs now live in the experimental `gravoturb` package. The CW04 `Q` substructure *diagnostic* survives in `progenax.diagnostics`.)

**Protocols**: `SpatialProfile`, `VelocityDF`, `IMFProtocol`, `PeriodDistribution`, `EccentricityDistribution`, `BinaryFractionModel`, `CompanionModel`

## TODO: Validation Plots (Pending jaxstroviz)

After jaxstroviz is ported:

- [ ] Plummer density profile (sampled vs analytical)
- [ ] King profile comparison with LIMEPY
- [ ] EFF profile truncation behavior
- [ ] Velocity dispersion radial profiles
- [ ] Virial equilibrium verification plots
- [ ] IMF mass distributions
- [ ] Binary orbital element distributions

## User-facing scientific documentation

Follow `docs/reviews/site-ownership.md`, especially “User-facing authoring standard”.
Read the full page before targeted edits; preserve existing structure and voice.
Define symbols and physical conventions at first use. Remove stale bug/repair
narratives from teaching pages while retaining current limitations and historical
records. Use MyST callouts and figures deliberately; inspect the rendered reading
width. Keep theory, API, tutorials and validation in their established owners.

## Gravoturb storage limits

Keep current cloud inputs, required reference catalogs, plotting scripts and their
compact data, configurations, seeds, source, results, failures, and figure masters.
Retire superseded native fields and derived workspaces after checking the current
input graph and figure dependencies. Do not delete source or plotting code to
save space. Never reduce scientific precision to fit a disk budget. Preserve
failure results in compact records.

Before a new data-producing experiment, state retained and temporary byte
estimates in its existing run plan. New producers must use bounded storage and
clear owned scratch after verified output closeout. Do not accumulate numbered
full-field copies or unbounded profiler traces by default. Bulk arrays and
profiler output stay out of Git; keep compact plotting inputs and recipes for
regenerating retired fields. A storage policy does not authorize a new scientific
run or a change to its physical model.

## Test design: quality over quantity (Anna, 2026-10-05)

This rule is identical in the jaxstro-dev root and in every package's `CLAUDE.md` and `AGENTS.md`; keep the
wording the same everywhere.

A test exists to catch a named failure. A few strong tests are better than many weak ones. Inefficient test
design is not allowed in any jaxstro package, and each package's suite is to be brought under this rule file by
file.

**Every test**

- names in its docstring the failure it catches, and fails when that failure is introduced; check this once by
  mutation when it is not obvious;
- asserts against an independent value: an analytic or limiting solution, a vendored reference fixture, a
  conservation law, or a bound whose derivation is stated. A rerun of the same code path is not a reference;
- sets its tolerance from the quantity its error scales with, a few times the measured error. A near-zero
  difference is compared against the scale its roundoff comes from, not against itself (gravax: the contact gap
  against the contact radius, 2026-10-05);
- runs in seconds. A test that needs longer is marked `slow` and says why.

**Compile cost (JAX)**

- Do not call a function that contains `lax.scan`, `fori_loop`, `while_loop` or `cond` eagerly in a test. Call it
  through one module-level `jax.jit` with its static arguments named. An eager call rebuilds the loop body and
  compiles a new program on every call: gravax `test_reach.py` made 575 XLA compiles, 235 after the change, and
  its cold-cache wall went from 17.0 s to 9.2 s (2026-10-05).
- Values that vary between cases (thresholds, spans, seeds, eta, distances) are traced inputs of one compiled
  program: sweep them with `vmap` or `lax.map` inside one `jit`, or parametrize over traced values. A static
  field that differs between tests costs one compile per test.
- Tests in a file share static configurations. Build each distinct configuration once (a module-scoped fixture
  or a shared configuration module) and reuse it. A new static configuration needs a stated reason. Count the
  distinct compiled programs of a file when adding tests to it.
- An expensive result (an integration, a routed state, a replay) is computed once in a module-scoped fixture and
  checked by several tests, not recomputed in each.
- The persistent compilation cache stores every program: `jax_persistent_cache_min_compile_time_secs = 0` and
  `jax_persistent_cache_min_entry_size_bytes = 0` in `tests/conftest.py`. With a 0.5 s floor the small programs
  were recompiled on every run (gravax `test_interaction_policy.py`: 24.0 s to 7.1 s of wall for 4.6 MB of
  cache, 2026-10-05).

**Removal and hardening**

- Delete a test that can no longer fail: its scenario is unreachable, it duplicates another test, or it pins an
  implementation detail with no failure mode. Removing a weak test is progress, not lost coverage.
- Harden a loose test: replace "runs", "is finite" or "changed" with the quantitative claim and its tolerance.
- Never weaken a test to make it pass. Fix the code, or fix the test's premise and say why.

**Measure before changing a slow file.** Record per test the trace, lowering and XLA compile seconds, the cache
hits and the programs compiled (`jax.monitoring` duration listeners and `jax_log_compiles`). Gravax, 2026-10-05,
563 hierarchy unit tests with the cache off: XLA compile was 73% of the wall and execution 12%; 232 distinct
programs took over 0.3 s each to compile.
