---
title: IC redesign spec (2026-02-12)
description: Original design specification for progenax's protocol-based, composable, differentiable IC architecture. Absorbed verbatim from docs/specs/2026-02-12-progenax-ic-redesign-design.md.
---

# Progenax IC redesign: unified, composable, differentiable (2026-02-12)

```{admonition} API retirement notice (2026-06, Phase 1c)
:class: warning
Several symbols this historical spec references were **retired in the
2026-06 unified multi-component redesign**: `generate_cluster_ic`,
`ClusterState`, the `populations` module
(`generate_two_component_cluster`/`TwoComponentConfig`),
`MassSegregationLayer`/`lambda_seg`, and the fractal generator. The
current entry points are `build_spatial_ic` (single population),
`MultiComponentCluster` (multi-population shared-potential equilibrium,
Engines A and B), and `energy_sorted_segregation` (primordial). See
[](../10-theory/populations/index.md) and [](../30-api/cluster.md).
This page is preserved unedited as a historical record.
```

```{seealso}
This page is the *historical record* of the IC redesign specification.
For the current state of the architecture see
[](../20-architecture/protocols.md), [](../20-architecture/three-brick-state.md),
and [](../20-architecture/jax-native-philosophy.md). For the broader
dev-log narrative see [](index.md).
```

```{admonition} 2026-06 update — gravoturbulent/PN11/FDF portions superseded
:class: note
The **gravoturbulent, PN11, and fractal-density-field portions** of this 2026-02-12 plan
(e.g. "PN11 removal", `bm19_model.py`, `TurbulentDensityProfile`, the `cluster.fdf*`
modules) were **superseded by the 2026-06 clean-room rewrite**: that whole subsystem was
deleted from released progenax and rebuilt as the experimental, repo-only **`gravoturb_fdf`**
package. The replacement names this spec proposes (e.g. `TurbulentDensityProfile`,
`bm19_model.py`) were themselves later removed. Treat the gravoturbulent parts below as the
*2026-02-12 plan of record*, not the current state; see
`src/experimental/gravoturb_fdf/` and [](../10-theory/gravoturbulence/index.md).
```

**Goal:** Restructure progenax's initial conditions pipeline into a clean, protocol-driven architecture where every component — IMF, spatial profile, velocity DF, and modifier — can be swapped independently, composed freely, and (where physics allows) differentiated through.

**Architecture:** Contract-driven composition via runtime-checkable protocols. Each component satisfies a protocol with enforced invariants. Science-motivated presets bundle consistent components. Environment-dependent parameters flow from a single `BirthEnvironment` object.

**Scope:** IMF module unification, spatial structure (smooth + turbulent composition), velocity DFs, IC modifiers (mass segregation, rotation, tidal truncation), and top-level assembly. Binary star machinery (`binary.py`) is out of scope — it already follows clean patterns.

---

## Table of Contents

1. [Motivation](#ic-redesign-motivation)
2. [Protocol Hierarchy](#ic-redesign-protocol-hierarchy)
3. [Unified IMF Classes](#ic-redesign-unified-imf-classes)
4. [Spatial Pipeline](#ic-redesign-spatial-pipeline)
5. [Science-Motivated Presets & Top-Level API](#ic-redesign-presets-api)
6. [Deprecation Plan](#ic-redesign-deprecation-plan)
7. [Testing Strategy](#ic-redesign-testing-strategy)
8. [Scientific Review & References](#ic-redesign-review-references)

---

(ic-redesign-motivation)=
## 1. Motivation

### The Problem

The current IMF module has three inconsistent patterns for the same operation:

| Pattern | Location | Differentiable? | Satisfies IMFProtocol? |
|---------|----------|-----------------|----------------------|
| `BaseIMF` subclasses | `smooth.py` | Yes (JAX arrays) | Yes (via inheritance) |
| `PowerLawIMF` standalone | `power_law.py` | **No** (Python tuples) | Yes (duck typing) |
| `IMFParams` + functions | `params.py`, `differentiable.py` | Yes (JAX arrays) | **No** (pure data, no methods) |

This creates a concrete failure: `generate_cluster_ic()` in `cluster/core.py` calls `imf_params.sample(key, N)`, but `IMFParams` has no `.sample()` method. The cluster IC generator cannot use the differentiable IMF path.

The spatial module has a similar fragmentation: `FractalDensityLayer` has a `base_profile` string parameter (`"uniform"` or `"plummer"`) rather than accepting any `SpatialProfile` object. The turbulent field and smooth profile are entangled inside a single function rather than composed as independent modules.

### The Solution

One protocol per component, one implementation hierarchy per component, full composability:

```
BirthEnvironment ──┬──→ IMFProtocol (Maschberger, PiecewiseIMF, Chabrier, ...)
                   ├──→ SpatialProfile (Plummer, King, EFF, Turbulent, ...)
                   ├──→ VelocityDF (PlummerDF, KingDF, EFFDF, ...)
                   └──→ ICModifier[] (MassSegregation, Rotation, TidalTruncation, ...)
```

### Differentiability Staging

- **Now (v0.3):** IMF fully differentiable. Spatial sampling non-differentiable (frozen FFT field with `stop_gradient`). Velocity DFs non-differentiable (rejection sampling).
- **Later (v0.5+):** Spatial differentiable via reparameterizable density sampling. Velocity DFs differentiable via Eddington inversion with fixed-iteration solvers.
- **Architecture must not block future differentiability** — no Python loops in hot paths, no mutable state, all Equinox modules.

---

(ic-redesign-protocol-hierarchy)=
## 2. Protocol Hierarchy

Four runtime-checkable protocols define the composition contracts. Each protocol specifies what invariants implementors must satisfy.

### IMFProtocol (existing, unchanged)

```python
@runtime_checkable
class IMFProtocol(Protocol):
    """Mass sampling with normalized probability density."""
    m_min: float
    m_max: float

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]: ...
    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]: ...
    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]: ...
    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]: ...
    def mean_mass(self) -> float: ...
```

**Invariant:** `∫ exp(logpdf(m)) dm = 1` over `[m_min, m_max]`.

**Invariant:** `cdf(ppf(u)) ≈ u` for all `u ∈ [0, 1]` (inverse CDF roundtrip).

### SpatialProfile (existing, unchanged)

```python
@runtime_checkable
class SpatialProfile(Protocol):
    """3D position sampling from a density profile."""
    def sample_positions(
        self, masses: Float[Array, "N"], key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]: ...

    def characteristic_radius(self) -> Float[Array, ""]: ...
```

**Invariant:** Radial density of sampled positions matches the analytical density profile (tested via KS test on radial CDF).

### VelocityDF (existing, unchanged)

```python
@runtime_checkable
class VelocityDF(Protocol):
    """Velocity sampling from a distribution function."""
    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]: ...
```

**Invariant:** System is in virial equilibrium (Q = T/|V| ≈ 0.5) when paired with the matching spatial profile.

### ICModifier (new)

```python
@runtime_checkable
class ICModifier(Protocol):
    """Post-hoc transformation of positions and/or velocities."""
    def apply(
        self,
        positions: Float[Array, "N 3"],
        velocities: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "N 3"], Float[Array, "N 3"]]: ...
```

**Invariant:** Does not change `masses`. May change `positions` and/or `velocities`.

**Existing modifiers to wrap:**
- `apply_mass_segregation()` → `MassSegregationModifier`
- `apply_solid_body_rotation()` → `SolidBodyRotation`
- `apply_differential_rotation()` → `DifferentialRotation`
- `apply_osipkov_merritt()` → `OsipkovMerrittAnisotropy`
- `apply_tidal_truncation()` → `TidalTruncation`

---

(ic-redesign-unified-imf-classes)=
## 3. Unified IMF Classes

### 3.1 PiecewiseIMF(BaseIMF) — replaces PowerLawIMF + IMFParams + differentiable.py

The core change: store exponents as **JAX arrays**, not Python tuples. This makes the piecewise IMF natively differentiable while satisfying `IMFProtocol` via `BaseIMF`.

```python
class PiecewiseIMF(BaseIMF):
    """N-segment piecewise power-law IMF with differentiable exponents.

    Unifies PowerLawIMF (non-differentiable) and IMFParams+differentiable.py
    (no methods) into a single class.
    """
    # Inference targets (JAX arrays — traced by autodiff)
    exponents: Float[Array, "n_segments"]   # α values per segment

    # Fixed structure (static — not traced)
    breakpoints: tuple[float, ...] = eqx.field(static=True)
    m_min: float = eqx.field(static=True, default=0.01)
    m_max: float = eqx.field(static=True, default=150.0)

    # Pre-computed (derived from exponents, recomputed on construction)
    _continuity_factors: Float[Array, "n_segments"]
    _segment_integrals: Float[Array, "n_segments"]

    def _logpdf_unnorm(self, m): ...  # Piecewise log(m^(-α_i) * k_i)
    def _cdf_unnorm(self, m): ...     # Analytical piecewise integral

    @classmethod
    def kroupa(cls, m_max=150.0) -> "PiecewiseIMF":
        """Kroupa (2001) 4-segment: α = [0.3, 1.3, 2.3, 2.3]."""
        return cls(
            exponents=jnp.array([0.3, 1.3, 2.3, 2.3]),
            breakpoints=(0.08, 0.50, 1.00),
        )

    @classmethod
    def salpeter(cls, m_min=0.1, m_max=100.0) -> "PiecewiseIMF":
        """Salpeter (1955) single power-law: α = 2.35."""
        return cls(
            exponents=jnp.array([2.35]),
            breakpoints=(),
            m_min=m_min, m_max=m_max,
        )

    @classmethod
    def from_legacy_params(cls, params: "IMFParams") -> "PiecewiseIMF":
        """Migration path from deprecated IMFParams."""
        return cls(
            exponents=jnp.array([params.alpha0, params.alpha1,
                                  params.alpha2, params.alpha3]),
            breakpoints=(params.m_break0, params.m_break1, params.m_break2),
            m_min=params.m_min, m_max=params.m_max,
        )
```

**Key properties:**
- `exponents` is a JAX array → `jax.grad` flows through all α values
- `breakpoints` is static → structure doesn't change during tracing
- Analytical CDF and PPF (same math as current `differentiable.py`)
- Satisfies `IMFProtocol` via `BaseIMF` → works with `generate_cluster_ic()`

### 3.2 Existing BaseIMF subclasses (unchanged)

These already work correctly:

| Class | Params | CDF | PPF | Differentiable? |
|-------|--------|-----|-----|-----------------|
| `Maschberger` | α, β, μ | Analytical | Analytical | Yes |
| `TaperedPowerLaw` | α, m_peak, β | Numerical | Newton | Yes |
| `Schechter` | α, m_star | Numerical | Newton | Yes |

### 3.3 ChabrierIMF (minor update)

`ChabrierIMF` currently does not inherit `BaseIMF` but satisfies `IMFProtocol` via duck typing. It should stay as-is — its two-component structure (lognormal + power-law) doesn't fit the `_logpdf_unnorm`/`_cdf_unnorm` pattern cleanly. The protocol is sufficient.

### 3.4 env_to_imf() — replaces env_to_imf_params()

```python
def env_to_imf(
    env: BirthEnvironment,
    model: str = "jerabkova_generalized",
    imf_family: str = "piecewise",  # "piecewise" or "maschberger"
    **kwargs,
) -> IMFProtocol:
    """Environment → ready-to-use IMF object.

    Returns a PiecewiseIMF or Maschberger depending on imf_family.
    """
    if imf_family == "piecewise":
        alpha3 = alpha3_jerabkova_generalized(...)
        return PiecewiseIMF(
            exponents=jnp.array([0.3, 1.3, 2.3, alpha3]),
            breakpoints=(0.08, 0.50, 1.00),
        )
    elif imf_family == "maschberger":
        alpha3 = alpha3_jerabkova_generalized(...)
        return Maschberger(alpha=alpha3)
```

---

(ic-redesign-spatial-pipeline)=
## 4. Spatial Pipeline

### 4.1 TurbulentDensityProfile — composing smooth × turbulent

The central insight: real star-forming regions have a **smooth radial envelope** (matching Plummer, King, or EFF profiles observed in evolved clusters) modulated by **turbulent substructure** (from the parent molecular cloud's density field). The current `FractalDensityLayer` does this internally but with string-based dispatch and entangled logic.

The redesign formalizes this as **composition**:

```python
class TurbulentDensityProfile(eqx.Module):
    """Smooth profile × turbulent density field.

    Composes any SpatialProfile with a 3D turbulent field. The smooth profile
    provides the radial envelope; the turbulent field provides substructure.

    The combined density is: ρ(r) = ρ_smooth(r) × ρ_turb(r) / <ρ_turb>

    where <ρ_turb> normalizes the turbulent field to preserve the smooth
    profile's total mass.
    """
    smooth_profile: SpatialProfile
    sigma_ln_rho: float           # Log-density fluctuation amplitude
    spectral_slope: float         # Power spectrum slope β
    grid_size: int = eqx.field(static=True, default=64)
    lambda_frac: float = 1.0     # Blend: 0 = pure smooth, 1 = full turbulent

    def sample_positions(
        self, masses: Float[Array, "N"], key: PRNGKeyArray,
    ) -> Float[Array, "N 3"]:
        """Sample from combined smooth × turbulent density."""
        ...

    def characteristic_radius(self) -> Float[Array, ""]:
        """Delegates to smooth_profile."""
        return self.smooth_profile.characteristic_radius()
```

**Satisfies `SpatialProfile`** — can be used anywhere a profile is expected.

**Composition examples:**
```python
# Turbulent Plummer (young massive cluster)
spatial = TurbulentDensityProfile(
    smooth_profile=PlummerProfile(r_h=1.0),
    sigma_ln_rho=2.0,
    spectral_slope=-3.67,  # Kolmogorov
)

# Turbulent EFF (OB association)
spatial = TurbulentDensityProfile(
    smooth_profile=EFFProfile(a=2.0, gamma=2.5, r_t=20.0),
    sigma_ln_rho=1.5,
    spectral_slope=-4.0,   # Burgers
)

# Pure smooth (old globular cluster — no substructure)
spatial = KingProfile(W0=7.0, r_c=1.0, r_t=30.0)
```

### 4.2 BM19 Gravoturbulent Tail

The BM19 model (Burkhart & Mocz 2019) computes the fraction of gas in the gravitationally-collapsing dense tail of the turbulent density PDF. This controls which regions of the turbulent field form stars.

Currently implemented as `TailSubstructureLayer` — will become an `ICModifier`:

```python
class GravoturbulentTail(eqx.Module):
    """BM19 dense-tail selection for turbulent density fields.

    Applies the Burkhart & Mocz (2019) gravoturbulent model to select
    the fraction of positions from the dense tail of the log-normal
    density PDF, where self-gravity overcomes turbulent support.
    """
    f_sub: float              # Dense tail fraction [0, 1]
    s_t: float                # Density threshold (from BM19)
    kappa: float = 10.0       # Softmax temperature for tail selection

    @classmethod
    def from_environment(cls, env: BirthEnvironment) -> "GravoturbulentTail":
        """Derive f_sub and s_t from cloud physics."""
        result = bm19_pipeline(
            mach=env.turbulent_mach(),
            alpha=2.0,
            eta_survive=0.6,
        )
        return cls(f_sub=result.f_sub, s_t=result.s_t)
```

### 4.3 PN11 Removal

The Padoan & Nordlund (2011) model (`pn11_model.py`) is removed. BM19 subsumes its physics with a more complete treatment of the density PDF. Files to delete:

- `src/progenax/gravoturb/pn11_model.py`
- Any PN11 paths in `cluster/fdf_density.py` and `cluster/fdf_tail.py`

---

(ic-redesign-presets-api)=
## 5. Science-Motivated Presets & Top-Level API

### 5.1 ICPreset — bundles of consistent components

```python
class ICPreset(eqx.Module):
    """Bundle of physically-consistent IC components.

    Not a special class — just a convenience container. Every field
    satisfies its protocol, so components can still be swapped.
    """
    imf: IMFProtocol
    spatial: SpatialProfile
    velocity_df: VelocityDF
    modifiers: tuple[ICModifier, ...] = ()

    @classmethod
    def solar_neighborhood(cls) -> "ICPreset":
        """Low-density, solar metallicity.

        Kroupa IMF, Plummer profile, isotropic Plummer DF.
        Matches: open clusters, field star IMF studies.
        """
        return cls(
            imf=PiecewiseIMF.kroupa(),
            spatial=PlummerProfile(r_h=1.0),
            velocity_df=PlummerVelocityDF(r_h=1.0),
        )

    @classmethod
    def young_massive_cluster(cls, M_ecl=1e4, r_h=1.0) -> "ICPreset":
        """Dense, mass-segregated, turbulent substructure.

        Top-heavy IMF (env-dependent), Turbulent+EFF, mass segregation.
        Matches: Westerlund 1, Arches, NGC 3603.
        """
        env = BirthEnvironment.from_cluster_mass(M_ecl)
        imf = env_to_imf(env, imf_family="piecewise")
        return cls(
            imf=imf,
            spatial=TurbulentDensityProfile(
                smooth_profile=EFFProfile(a=r_h, gamma=2.5, r_t=10*r_h),
                sigma_ln_rho=float(env.sigma_ln_rho()),
                spectral_slope=float(env.spectral_slope()),
            ),
            velocity_df=EFFVelocityDF(a=r_h, gamma=2.5, r_t=10*r_h),
            modifiers=(MassSegregationModifier(lambda_seg=0.5),),
        )

    @classmethod
    def globular_cluster(cls, W0=7.0, r_c=1.0, r_t=30.0) -> "ICPreset":
        """Old, relaxed, King profile.

        Kroupa IMF, King profile + DF, full mass segregation.
        Matches: 47 Tuc, M15, ω Cen.
        """
        return cls(
            imf=PiecewiseIMF.kroupa(),
            spatial=KingProfile(W0=W0, r_c=r_c, r_t=r_t),
            velocity_df=KingVelocityDF(W0=W0, r_c=r_c, r_t=r_t),
            modifiers=(MassSegregationModifier(lambda_seg=1.0),),
        )

    @classmethod
    def starburst(cls, M_ecl=1e5) -> "ICPreset":
        """Extreme density, very top-heavy, strong turbulence.

        Matches: NGC 253, M82, Arp 220 super star clusters.
        """
        env = BirthEnvironment.from_cluster_mass(M_ecl, sfe=0.5)
        imf = env_to_imf(env, imf_family="piecewise")
        tail = GravoturbulentTail.from_environment(env)
        return cls(
            imf=imf,
            spatial=TurbulentDensityProfile(
                smooth_profile=PlummerProfile(r_h=0.5),
                sigma_ln_rho=float(env.sigma_ln_rho()),
                spectral_slope=float(env.spectral_slope()),
            ),
            velocity_df=PlummerVelocityDF(r_h=0.5),
        )
```

### 5.2 Top-Level API

```python
def generate_cluster_ic(
    key: PRNGKeyArray,
    N_stars: int,
    M_total: float,
    *,
    preset: ICPreset | None = None,
    imf: IMFProtocol | None = None,
    spatial: SpatialProfile | None = None,
    velocity_df: VelocityDF | None = None,
    modifiers: tuple[ICModifier, ...] = (),
    G: float | None = None,
) -> ClusterState:
    """Generate complete cluster initial conditions.

    Either pass a `preset` or individual components.

    Args:
        key: JAX PRNG key
        N_stars: Number of stars
        M_total: Total cluster mass [M☉]
        preset: Pre-bundled component set
        imf: Mass function (overrides preset)
        spatial: Density profile (overrides preset)
        velocity_df: Velocity distribution (overrides preset)
        modifiers: Post-hoc transformations (appended to preset)
        G: Gravitational constant (default: STELLAR.G)

    Returns:
        ClusterState with masses, positions, velocities
    """
    ...
```

---

(ic-redesign-deprecation-plan)=
## 6. Deprecation Plan

### Files to Remove

| File | Reason | Replacement |
|------|--------|-------------|
| `imf/differentiable.py` | Redundant standalone functions | `PiecewiseIMF` methods |
| `gravoturb/pn11_model.py` | Superseded by BM19 | `bm19_model.py` only |
| `cluster/fractal_gw_legacy.py` | Already deprecated GW2004 tree method | `TurbulentDensityProfile` |

### Classes to Deprecate

| Class | Replacement | Migration |
|-------|-------------|-----------|
| `PowerLawIMF` | `PiecewiseIMF` | `PiecewiseIMF.kroupa()` etc. |
| `IMFParams` | `PiecewiseIMF` | `PiecewiseIMF.from_legacy_params()` |
| `FractalDensityLayer` | `TurbulentDensityProfile` | Composition pattern |
| `TailSubstructureLayer` | `GravoturbulentTail` | `ICModifier` protocol |

### Functions to Deprecate

| Function | Replacement |
|----------|-------------|
| `log_prob_masses()` | `PiecewiseIMF(...).logpdf()` |
| `sample_masses_from_params()` | `PiecewiseIMF(...).ppf()` / `.sample()` |
| `individual_mass_nll()` | `-jnp.sum(PiecewiseIMF(...).logpdf(masses))` |
| `env_to_imf_params()` | `env_to_imf()` |
| `generate_fractal_positions()` | `TurbulentDensityProfile.sample_positions()` |

### API to Update

| Current | New |
|---------|-----|
| `env_to_imf_params(env) → IMFParams` | `env_to_imf(env) → IMFProtocol` |
| `generate_cluster_ic(imf_params=...)` | `generate_cluster_ic(imf=...)` |
| `SpatialStructureParams(base_profile="plummer")` | `TurbulentDensityProfile(smooth_profile=PlummerProfile(...))` |

---

(ic-redesign-testing-strategy)=
## 7. Testing Strategy

### Three-Tier Architecture (matching existing progenax convention)

#### Unit Tests (`tests/unit/`)

**IMF tests:**
- `PiecewiseIMF` normalization: `∫ exp(logpdf) dm ≈ 1.0` (within 1e-3)
- CDF roundtrip: `cdf(ppf(u)) ≈ u` for random u
- Consistency: `PiecewiseIMF.kroupa().logpdf(m)` matches `PowerLawIMF.kroupa().logpdf(m)` element-wise
- Gradient flow: `jax.grad(lambda α: -jnp.sum(PiecewiseIMF(exponents=α, ...).logpdf(masses)))(α₀)` returns finite
- JIT + vmap compatibility
- Edge cases: α near 1.0 (singularity), α at boundaries
- Factory methods: `.kroupa()`, `.salpeter()`, `.from_legacy_params()`

**Spatial tests:**
- `TurbulentDensityProfile` satisfies `SpatialProfile` protocol
- Composition: Turbulent+Plummer produces positions with Plummer-like radial envelope
- `lambda_frac=0` recovers pure smooth profile
- `lambda_frac=1` shows turbulent substructure (Q-parameter < 0.8)
- Grid size independence (64 vs 128 gives consistent statistics)

**Modifier tests:**
- Each modifier satisfies `ICModifier` protocol
- `MassSegregationModifier`: heavier stars more central (Λ_MSR > 1)
- `SolidBodyRotation`: v_rot ∝ r_perp
- Identity: modifier with λ=0 returns inputs unchanged

#### Integration Tests (`tests/integration/`)

- Protocol composition: Any `IMFProtocol` + any `SpatialProfile` + any `VelocityDF` produces valid `ClusterState`
- Preset roundtrip: `ICPreset.solar_neighborhood()` through `generate_cluster_ic()` gives virial equilibrium
- Environment pipeline: `BirthEnvironment.solar()` → `env_to_imf()` → `generate_cluster_ic()` end-to-end

#### Validation Tests (`tests/validation/`)

- Virial ratio: Q = T/|V| ≈ 0.5 ± 0.05 for equilibrium presets
- Density profile: Sampled radial CDF matches analytical (KS test p > 0.01)
- IMF recovery: NumPyro NUTS recovers α₃ within 68% CI
- Mass function: Sampled mass histogram matches theoretical ξ(m) (χ² test)

---

(ic-redesign-review-references)=
## 8. Scientific Review & References

### Why This Architecture is State-of-the-Art

Most N-body IC generators treat each physical component in isolation. MCLUSTER (Küpper et al. 2011) hard-codes profile choices. McLuster2 (Leveque et al. 2022) adds mass segregation but uses non-differentiable Goodwin & Whitworth (2004) fractal trees. AMUSE (Portegies Zwart et al. 2013) provides component libraries but without gradient flow. No existing package offers the combination of: (1) composable, protocol-driven components, (2) environment-dependent parameters from a unified physics model, (3) differentiability through the IC pipeline, and (4) turbulent spatial structure grounded in star formation theory.

### Component-Specific Justification

#### IMF: Piecewise with JAX-Array Exponents

The four-segment piecewise power-law IMF is the standard in star cluster modeling (Kroupa 2001). The environment dependence of the high-mass slope α₃ on cloud density and metallicity is well-established (Marks et al. 2012; Jerabkova et al. 2018). Making the exponents JAX arrays rather than Python floats is the minimal change needed for differentiability — it preserves the exact analytical CDF and PPF while enabling gradient-based inference.

The Maschberger (2013) smooth IMF provides a complementary parameterization where α controls the high-mass slope across the full mass range. This is advantageous for inference because the likelihood information is distributed more evenly across the mass function, rather than being concentrated above the 1 M☉ breakpoint.

#### Spatial: Smooth Profile × Turbulent Field Composition

Real star-forming regions exhibit a smooth radial envelope modulated by turbulent substructure. The radial profile follows an EFF (Elson, Fall & Freeman 1987) or King (1962) model at late times, but at birth the substructure reflects the parent molecular cloud's turbulent density field (Federrath & Klessen 2012).

Our `TurbulentDensityProfile` composes these two components explicitly:
- **Smooth envelope**: Any `SpatialProfile` (Plummer, King, EFF) provides the radial density
- **Turbulent field**: FFT-generated lognormal density field with physically-derived parameters (σ_ln_ρ from Mach number via Federrath et al. 2010; spectral slope β from turbulence regime)

This is physically superior to:
- **Pure smooth profiles** (miss birth substructure entirely)
- **Box fractal methods** (Goodwin & Whitworth 2004) that use recursive tree subdivision with no physical grounding — the fractal dimension D is a free parameter with no connection to cloud physics
- **Pure turbulent methods** that ignore the overall radial concentration

The composition pattern `ρ(r) = ρ_smooth(r) × ρ_turb(r) / ⟨ρ_turb⟩` preserves the smooth profile's mass normalization while adding substructure. The blend parameter `lambda_frac` enables smooth interpolation from pure analytical profile (λ=0, evolved clusters) to full turbulent substructure (λ=1, embedded clusters).

#### Gravoturbulent Tail: BM19

Burkhart & Mocz (2019) derive the star-forming gas fraction from the turbulent density PDF by computing the density threshold s_t above which self-gravity overcomes turbulent support. This gives a physically-motivated f_sub (dense tail fraction) from first principles, unlike the ad hoc f_sub values used in most IC generators.

We adopt BM19 exclusively over the Padoan & Nordlund (2011) model because BM19:
- Includes a more complete treatment of the lognormal + power-law density PDF tail
- Naturally connects to the Federrath et al. (2010) density-Mach relation
- Has been validated against magnetohydrodynamic turbulence simulations
- Provides the star formation rate per free-fall time (ε_ff) as a consistency check

#### Modifiers as Post-Processing

Mass segregation, rotation, and tidal truncation are applied as post-processing modifiers rather than being baked into the spatial sampling. This separation is physically motivated:
- **Primordial mass segregation** (Baumgardt et al. 2008) operates on the energy ordering of stars, not on the density field
- **Rotation** (Henault-Brunet et al. 2012) is a velocity-space transformation independent of density
- **Tidal truncation** (King 1966) depends on the external potential, not the internal density

The modifier pattern also enables clean testing: each modifier can be verified in isolation.

#### Environment-Dependent Parameters

The `BirthEnvironment` object (metallicity, embedded cluster mass, star formation efficiency) drives all environment-dependent quantities through calibrated relations:
- **IMF slope**: α₃ from Jerabkova et al. (2018) Eq. 3, calibrated on resolved Milky Way clusters
- **Turbulence amplitude**: σ_ln_ρ from Federrath et al. (2010) Eq. 7, calibrated on MHD simulations
- **Cloud radius**: Larson (1981) size-linewidth relation, R ∝ σ_v^(1/α)
- **Dense tail fraction**: BM19 f_sub from Mach number and virial parameter

This unified parameter derivation ensures physical consistency — the same cloud that produces a top-heavy IMF also has strong turbulent substructure, because both are driven by high density and Mach number.

### References

- Baumgardt, H., De Marchi, G., & Kroupa, P. 2008, ApJ, 685, 247 — Energy-based mass segregation
- Burkhart, B. & Mocz, P. 2019, ApJ, 879, 129 (BM19) — Gravoturbulent star formation rate from density PDF
- Chabrier, G. 2003, PASP, 115, 763 — Lognormal + power-law system IMF
- Elson, R. A. W., Fall, S. M., & Freeman, K. C. 1987, ApJ, 323, 54 — EFF profile for young LMC clusters
- Federrath, C., Roman-Duval, J., Klessen, R. S., Schmidt, W., & Mac Low, M.-M. 2010, A&A, 512, A81 — Density-Mach number relation
- Federrath, C. & Klessen, R. S. 2012, ApJ, 761, 156 — Turbulent star formation models
- Goodwin, S. P. & Whitworth, A. P. 2004, A&A, 413, 929 — Box fractal method (superseded)
- Henault-Brunet, V., et al. 2012, A&A, 545, L1 — Rotation in young massive clusters
- Jerabkova, T., Hasani Zonoozi, A., Kroupa, P., et al. 2018, A&A, 620, A39 — Environment-dependent IMF
- King, I. R. 1962, AJ, 67, 471 — King profile
- King, I. R. 1966, AJ, 71, 64 — Tidal truncation
- Kroupa, P. 2001, MNRAS, 322, 231 — Canonical multi-segment IMF
- Küpper, A. H. W., Maschberger, T., Kroupa, P., & Baumgardt, H. 2011, MNRAS, 417, 2300 — MCLUSTER IC generator
- Larson, R. B. 1981, MNRAS, 194, 809 — Turbulence scaling relations
- Leveque, A., et al. 2022, MNRAS, 514, 850 — McLuster2
- Marks, M., Kroupa, P., Dabringhausen, J., & Pawlowski, M. S. 2012, MNRAS, 422, 2246 — Environment-dependent IMF
- Maschberger, T. 2013, MNRAS, 429, 1725 — Smooth analytical IMF
- Moe, M. & Di Stefano, R. 2017, ApJS, 230, 15 — Binary star statistics
- Portegies Zwart, S., et al. 2013, CoPhC, 183, 456 — AMUSE framework
- Salpeter, E. E. 1955, ApJ, 121, 161 — Original power-law IMF

---

## Implementation Roadmap

| Phase | Deliverable | Depends On |
|-------|-------------|------------|
| **0** | Maschberger NUTS validation figure (proposal) | Nothing (use existing `Maschberger` class) |
| **1** | `PiecewiseIMF(BaseIMF)` + deprecate PowerLawIMF/IMFParams/differentiable.py | Phase 0 |
| **2** | `ICModifier` protocol + wrap existing modifiers | Phase 1 |
| **3** | `TurbulentDensityProfile` + remove PN11 | Phase 2 |
| **4** | `ICPreset` + refactored `generate_cluster_ic()` | Phases 1-3 |
| **5** | Full validation suite + publication figure | Phase 4 |

Phase 0 is the immediate deliverable for the LSST-DA proposal. Phases 1-4 are the full redesign. Phase 5 produces the validation evidence.
