# progenax

Differentiable initial conditions for N-body simulations in JAX.

Part of the **jaxstro ecosystem** — providing IC generation that can be differentiated through for gradient-based inference.

## Status

**Research software, release-candidate core.** King, Michie, EFF, and multi-component
(LIMEPY / density-defined Eddington) velocity DFs are true equilibria (lowered-Maxwellian /
Eddington inversion, no external virial rescale). The binary-population engine is finalized —
`build_binary_cluster` composes `primary_imf × companion_model × target` with the faithful
Moe & Di Stefano (2017) P–q–e coupling (`MoeCompanions`). The gravoturbulent + fractal-
density-field subsystem lives in the **experimental, repo-only `gravoturb` package**
(`src/experimental/`, **not** shipped in the wheel). Every public model carries a
machine-readable **provenance card** (equation-level citations, enforced by tests), and every
public entry point is **gradient-audited** (AD-vs-FD, ~98 registry cases, 0 hazards). For
current test/LOC counts see CI and `tests/README.md` (counts are not duplicated here to
avoid drift).

> **Not on PyPI yet.** progenax depends on the sibling package `jaxstro` (also unpublished),
> resolved from a side-by-side checkout. See [Installation](#installation).

## Documentation

The **MyST documentation site** (`docs/website/`, 200+ pages) is the single source of truth —
this README is only the quick tour. Build it locally:

```bash
cd docs/website && myst start    # live-reload dev server
cd docs/website && make gate     # full build + link/content gates
```

What lives there:

- **Theory** (`10-theory/`) — derivations for every profile, DF, IMF, and binary model, with
  publication-quality figures that double as correctness proofs (residual panels,
  escape-envelope oracles), worked Eddington inversions, and exercises.
- **Model reference** (`15-model-reference/`) — the generated **provenance glossary**: one
  card per model with its equations, primary sources (DOI/arXiv), implementing code, and
  validation tests. Generated from `docs/provenance/registry/*.yaml` and enforced by
  `tests/validation/provenance_cards/`.
- **Architecture, API reference, how-tos, validation dashboard** — `20-architecture/`,
  `30-api/` (generated, with model-card and gradient-verified badges), `40-howto/`,
  `50-validation/`.
- **Release checklist** (`95-release/checklist.md`) — the maintainer's living pre-flip /
  pre-tag punch list.

The figures are produced by the modular **ICViz** library (`laboratory/icviz/` — a
`FigureSpec` registry + CLI: `uv run python -m laboratory.icviz --only <name>`).

## Features

### Spatial Profiles

| Profile | Class | Description | Reference |
|---------|-------|-------------|-----------|
| **Plummer** | `PlummerProfile` | Plummer (1911) sphere ρ ∝ (1 + r²/a²)^(−5/2) | Plummer (1911) |
| **King** | `KingProfile` | King (1966) lowered isothermal model | King (1966) |
| **Michie** | `MichieProfile` | Anisotropic King (Michie 1963 + King 1966 cutoff) | Michie (1963) |
| **EFF** | `EFFProfile` | Elson–Fall–Freeman (1987) truncated power law | EFF (1987) |
| **LIMEPY** | `LIMEPYProfile` | Lowered-isothermal multi-mass family (single- or multi-component) | Gieles & Zocchi (2015) |
| **Uniform sphere** | `UniformSphereProfile` | Uniform-density ball (baselines/tests) | — |
| **King ODE** | `solve_king_profile()` | Numerical King profile via Diffrax | King (1966) |

### Velocity Distribution Functions

| DF | Class | Method | Notes |
|----|-------|--------|-------|
| **Plummer DF** | `PlummerVelocityDF` | Beta(3/2, 9/2) sampling | Exact, no rejection |
| **King DF** | `KingVelocityDF` | Lowered-Maxwellian f(E), σ self-consistent | True equilibrium (Q≈0.5 unscaled) |
| **Michie DF** | `MichieVelocityDF` | Self-consistent anisotropic King | True equilibrium; pairs with `MichieProfile` |
| **EFF DF** | `EFFVelocityDF` | Isotropic Eddington inversion f(E) from ρ(Ψ) | γ=3 default ~8% sub-virial (documented); γ≥5 ~equilibrium |
| **LIMEPY DF** | `LIMEPYVelocityDF` | Lowered-isothermal f(E) (multi-mass) | Parity vs canonical LIMEPY code |

`PlummerVelocityDF` and `EFFVelocityDF` take an optional `anisotropy_radius` (Osipkov–Merritt
radial anisotropy, β(r) = r²/(r²+r_a²)). Rotation overlays: `apply_solid_body_rotation()`,
`apply_differential_rotation()` (kinematic overlays — they inject energy/L_z and are NOT
stationary equilibria).

### Multi-component clusters

`MultiComponentCluster` builds a self-consistent multi-component equilibrium in a shared
potential (replaces the deleted `TwoComponentConfig`/`generate_two_component_cluster`):

| Constructor | Engine | Description |
|-------------|--------|-------------|
| `from_components()` | A | GZ15 multi-mass LIMEPY (one ψ ODE, Σ_j α_j ρ̂_j) |
| `from_imf()` | A | components binned from an IMF |
| `from_mass_segregation()` | A | primordial mass segregation |
| `from_density_profiles()` | B | prescribed Plummer/EFF/King densities → shared-Ψ Eddington/OM DFs |

### Initial Mass Functions

| IMF | Class | Description | Reference |
|-----|-------|-------------|-----------|
| **Power Law** | `PowerLawIMF` | Single/broken power laws | Salpeter (1955) |
| **Kroupa** | `PowerLawIMF.kroupa()` | 3-segment Kroupa (2001) | Kroupa (2001) |
| **Maschberger** | `Maschberger` | Smooth transitions | Maschberger (2013) |
| **Chabrier** | `ChabrierIMF` | Log-normal + power law | Chabrier (2003) |
| **Truncated** | `TruncatedIMF` | Mass-limit wrapper | — |
| **Binary** | `BinaryIMF` | Primary IMF + mass-dependent companions (q, f_b) | Moe & Di Stefano (2017) |

The **environment-dependent stellar IMF** (top-heavy α₃ from cluster density/metallicity) is
the functional `BirthEnvironment` + `env_to_imf_params()` API (Marks+2012 Fundamental Plane /
Jeřábková+2018 relations) — **not** a galaxy-wide IGIMF integration, and not an
`IGIMF`/`EnvironmentIMF` class.

### Binary Orbital Mechanics

| Component | Class/Function | Description |
|-----------|----------------|-------------|
| **Keplerian Elements** | `KeplerElements` | (a, e, i, Ω, ω, M₀) + masses |
| **Orbital State** | `BinaryOrbitalState` | 6D position/velocity + masses |
| **Period** | `compute_period()` | T = 2π√(a³/GM) |
| **Inverse Period** | `period_to_semimajor_axis()` | a from T and M |

**Period distributions:** `LogUniformPeriod` (Öpik), `LogNormalPeriod` (Duquennoy & Mayor 1991),
`SanaOBPeriod` (Sana+ 2012), `MoePeriod` (Moe & Di Stefano 2017).
**Eccentricity distributions:** `ThermalEccentricity` (f(e)=2e), `UniformEccentricity`,
`MoeEccentricity` (p(e) ∝ e^η + Roche cap).

**Binary-cluster composition** (`build_binary_cluster`): a primary IMF × a `CompanionModel`
(`IndependentCompanions` or the faithful `MoeCompanions`) × a population budget
(`Systems` / `Stars` / `TotalMass`). Connector `resolve_binary_components()` maps binary COMs
to 2N components. Diagnostics: `find_bound_pairs()`, `find_bound_multiples()`,
`primordial_survival()`, `binary_energy_budget()`.

### Analytical Test Cases

`two_body_kepler()`, `three_body_figure_eight()`, `earth_sun_2body()`,
`solar_system_inner_4()`, `solar_system_full()`, `harmonic_oscillator()`.

### Stellar ZAMS relations

Tout et al. (1996) metallicity-dependent ZAMS fits (differentiable, used for photometric
blending and mass–luminosity work): `zams_luminosity()`, `zams_radius()`,
`zams_effective_temperature()`, `zams_surface_gravity()`, `inverse_zams_luminosity()`.
(These are a placeholder for the planned `startrax` stellar-tracks package.)

### Utilities

`build_cluster()` — the one-call single-population builder: profile → matched velocity DF +
IMF/masses + optional Osipkov–Merritt anisotropy, tidal truncation (Plummer only; the other
families carry a native `r_t`), and rotation overlays via `RotationSpec`
(`build_cluster_from_params()` is the flat-parameter variant for inference loops).
`build_spatial_ic()` / `ICResult`, `compute_kinetic_energy()`, `compute_potential_energy()`,
`to_com_frame()`, `virial_scale()`, `compute_stellar_radii()` (Demircan & Kahraman 1991 M–R),
`jacobi_radius()` / `apply_tidal_truncation()`, `energy_sorted_segregation()`,
substructure diagnostics `compute_q_parameter()` (CW04 Q) and `q_approx` (differentiable kNN;
needs the `[diagnostics]` extra for the exact scipy path).

## Installation

progenax is **not on PyPI**; it (and its sibling `jaxstro`) resolve from a standard
side-by-side checkout. Use **uv** (preferred):

```bash
# Clone progenax and its public sibling jaxstro next to each other
git clone https://github.com/jaxstro/jaxstro.git
git clone https://github.com/jaxstro/progenax.git
cd progenax

# Install the released core + dev tools (uses the project .venv)
uv pip install -e ".[dev]"
```

Optional extras: `[experimental]` (the repo-only `gravoturb` inference layer — blackjax,
optax, …), `[diagnostics]` (numpy + scipy, for the exact non-differentiable CW04 Q path).

> **Import-time precision side effect.** `import progenax` enables JAX float64
> (`jax_enable_x64=True`) and sets `jax_default_matmul_precision="highest"`
> **process-wide** — cluster ICs are energy-balance-sensitive and are validated in
> double precision only. If you embed progenax in a float32 pipeline, be aware that
> arrays created after the import default to float64 (roughly double the memory), and
> that flipping x64 back off after import downcasts progenax's outputs with warnings —
> nothing in progenax is float32-validated. Import progenax early, before building
> other JAX state, so the precision regime is consistent.

## Quick Start

### Plummer sphere IC

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import PlummerProfile, PlummerVelocityDF

profile = PlummerProfile(r_h=1.0)        # half-mass radius = 1 pc
velocity_df = PlummerVelocityDF(r_h=1.0)

N = 1000
masses = jnp.ones(N)                      # 1000 Msun total
key_pos, key_vel = jax.random.split(jax.random.PRNGKey(42))
positions = profile.sample_positions(masses, key_pos)
velocities = velocity_df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

### One-call IC assembly

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import PlummerProfile, PlummerVelocityDF, build_spatial_ic

masses = jnp.ones(500)
ic = build_spatial_ic(
    PlummerProfile(r_h=1.0), masses, PlummerVelocityDF(r_h=1.0),
    key=jax.random.PRNGKey(0), G=STELLAR.G,
)
positions, velocities = ic.positions, ic.velocities
```

### One-call cluster builder (profile → matched DF)

```python
import jax
from jaxstro.units import STELLAR
from progenax import KingProfile, Maschberger, build_cluster

# King W0=7 cluster with Maschberger IMF masses, matched true-DF velocities
ic = build_cluster(
    KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
    key=jax.random.PRNGKey(7),
    n=1000, imf=Maschberger(),
    units=STELLAR,
    Q=None,   # trust the true-DF detailed equilibrium (no virial rescale)
)
```

### Multi-component cluster (extended halo + concentrated core)

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import MultiComponentCluster

# Two mass components in a shared self-consistent potential (Engine A, LIMEPY).
model = MultiComponentCluster.from_components(
    alpha_j=jnp.array([0.6, 0.4]),   # mass fractions
    w_j=jnp.array([1.0, 1.0]),       # relative velocity scales
    m_j=jnp.array([0.4, 5.0]),       # mean masses [Msun]
    W0=7.0, g=1.0, r_c=1.0,
)
ic = model.sample_cluster(jax.random.PRNGKey(1), n_stars=2000, G=STELLAR.G)
positions, velocities, masses = ic.positions, ic.velocities, ic.masses
```

### IMF sampling

```python
import jax
from progenax import PowerLawIMF, ChabrierIMF

k1, k2 = jax.random.split(jax.random.PRNGKey(42))
kroupa_masses = PowerLawIMF.kroupa().sample(k1, 1000)   # Kroupa (2001)
chabrier_masses = ChabrierIMF().sample(k2, 1000)        # Chabrier (2003)
```

### Binary orbital elements

```python
from jaxstro.units import PLANETARY
from progenax import KeplerElements, compute_period

elements = KeplerElements(a=1.0, e=0.3, i=0.1, Omega=0.0, omega=0.0, M0=0.0)
state = elements.to_state(M_total=2.0, G=PLANETARY.G)   # CartesianState
r, v = state.position, state.velocity
period = compute_period(a=1.0, M_total=2.0, G=PLANETARY.G)
```

### Analytical test case (Earth–Sun)

```python
from jaxstro.units import PLANETARY
from progenax import two_body_kepler

ic = two_body_kepler(M1=1.0, M2=3e-6, a=1.0, G=PLANETARY.G, e=0.017)
positions, velocities, masses = ic.positions, ic.velocities, ic.masses
```

### Differentiability

```python
import jax
import jax.numpy as jnp
from progenax import PlummerProfile

masses = jnp.ones(100)
key = jax.random.PRNGKey(0)

def loss(r_h):
    pos = PlummerProfile(r_h=r_h).sample_positions(masses, key)
    return jnp.mean(jnp.linalg.norm(pos, axis=1))

gradient = jax.grad(loss)(1.0)   # ∂⟨r⟩/∂r_h — fully differentiable
```

## Unit Systems

progenax uses **jaxstro.units** for explicit gravitational-constant management. Core APIs
require an explicit `G` (or `units`); convenience wrappers may resolve `G=None` to
`DEFAULT_UNITS` (= `STELLAR`).

| Unit System | G Value | Units | Use Case |
|-------------|---------|-------|----------|
| `STELLAR` | ~0.00450 | pc³ Msun⁻¹ Myr⁻² | Star clusters, galaxies |
| `PLANETARY` | ~39.478 | AU³ Msun⁻¹ yr⁻² | Binaries, planets |
| `DEFAULT_UNITS` | = STELLAR | — | progenax default (wrappers only) |

## Key Patterns

### Protocol-based composition

Any `SpatialProfile` pairs with any `VelocityDF` (9 runtime-checkable protocols total):

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import PlummerProfile, KingVelocityDF

masses = jnp.ones(500)
k_pos, k_vel = jax.random.split(jax.random.PRNGKey(0))
profile = PlummerProfile(r_h=1.0)
df = KingVelocityDF(W0=7.0, r_c=1.0)            # r_t derived from W0 (self-consistent)
positions = profile.sample_positions(masses, k_pos)
velocities = df.sample_velocities(positions, masses, k_vel, G=STELLAR.G)
```

### Equinox modules

All stateful classes are immutable Equinox PyTrees (e.g. `PlummerProfile` carries `r_h` and a
computed scale radius `a`), so sampling is differentiable through JAX and JIT/vmap-friendly.

### Critical formula (Plummer scale radius)

```python
import jax.numpy as jnp
r_h = 1.0
a = r_h * jnp.sqrt(2 ** (2 / 3) - 1)   # ≈ 0.7664 * r_h  (NOT the inverse!)
```

Virial ratio convention: **Q = T/|V|**, with **Q = 0.5** the equilibrium (2T + V = 0).

## Architecture

```text
src/progenax/
├── __init__.py            # public API
├── protocols.py           # 9 runtime-checkable protocols
├── builders.py            # ICResult + build_spatial_ic + build_binary_cluster
├── builders_cluster.py    # build_cluster / build_cluster_from_params + RotationSpec
├── numerics.py            # expm1/log1p-stable power-law kernels (α→1 removable singularity)
├── stellar.py             # Tout+1996 ZAMS relations (startrax placeholder)
├── tidal.py               # Jacobi radius + truncation
├── profiles/              # Plummer, King, Michie, EFF, LIMEPY, UniformSphere
├── kinematics/            # velocity DFs (+ Osipkov–Merritt anisotropy, rotation overlays)
├── imf/                   # PowerLaw, Chabrier, Maschberger, Binary; environment/ (BirthEnvironment)
├── binaries/              # Kepler elements/solver, period & eccentricity distributions, connector
├── cluster/               # MultiComponentCluster (Engine A + B), mass segregation
├── diagnostics/           # CW04 Q (compute_q_parameter), differentiable q_approx
└── analytical/            # solar system, Kepler orbits, figure-eight
src/experimental/          # gravoturb (repo-only; NOT in the wheel)
docs/website/              # the MyST documentation site (single source of truth)
docs/provenance/registry/  # machine-readable model cards (YAML) → glossary + enforcement tests
laboratory/icviz/          # ICViz figure library (publication figures for docs + methods paper)
```

## Testing

```bash
# FAST gate (inner loop): released-core, excluding slow
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  uv run --no-sync pytest tests/unit tests/integration tests/validation -m "not slow" -n auto

# FULL gate (phase/commit gate): includes the slow trust anchors
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  uv run --no-sync pytest tests/unit tests/integration tests/validation -n auto

# Experimental gravoturb subsystem (repo-only)
PYTHONPATH=src:src/experimental uv run --no-sync pytest tests/experimental
```

See `tests/README.md` for the 3-tier architecture and current counts.

### Physics validation (selected)

From `tests/validation/` (Q ≡ T/|V|; 0.5 = equilibrium), sampled with no external rescale:

- **Plummer** virial Q = 0.502 (expected 0.5)
- **King** true-DF virial Q ≈ 0.51 unscaled (lowered-Maxwellian)
- **EFF** Eddington-DF virial Q ≈ 0.50 unscaled (γ=5, mild truncation)
- **King c(W₀)** matches King (1966) Table II to |Δlog₁₀c| ≤ 0.002 (hi-res)
- **Engine B** King A-vs-B σ_1d / radial KS ≤ 3e-4 (N=2e4)
- **Kepler** energy & angular momentum conserved to ~1e-16; period exact to 1e-10

## Dependencies

`jax>=0.4.20`, `jaxlib>=0.4.20`, `equinox>=0.11.0`, `jaxtyping>=0.2.25`,
`diffrax>=0.4.0` (King ODE), and the sibling `jaxstro` (core utilities; side-by-side checkout).

## References

**Profiles/DFs:** Plummer (1911); King (1966); Michie (1963); Elson, Fall & Freeman (1987);
Gieles & Zocchi (2015); Dehnen (1993); Binney & Tremaine (2008); Merritt (1985).
**IMFs:** Salpeter (1955); Kroupa (2001); Chabrier (2003); Maschberger (2013);
Marks+ (2012); Jeřábková+ (2018).
**Binaries:** Duquennoy & Mayor (1991); Sana+ (2012); Moe & Di Stefano (2017).
**Substructure/Methods:** Cartwright & Whitworth (2004); Goodwin & Whitworth (2004);
Küpper+ (2011); Aarseth (2003).
**Stellar:** Tout+ (1996) ZAMS; Demircan & Kahraman (1991).

Every model's equation-level provenance (equations, primary sources with DOI/arXiv,
implementing code, validation tests) lives in the machine-readable registry
(`docs/provenance/registry/`) and its generated glossary on the docs site.

## License

Apache License 2.0 — see [LICENSE](LICENSE). Copyright 2026 Anna Rosen.
