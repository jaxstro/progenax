# Cluster-builder convenience API — design

**Date:** 2026-06-14 · **Status:** design ratified (brainstorm with Anna), implementation pending ·
**Branch (when built):** off `feat/provenance-credibility-audit` or a fresh `feat/cluster-builders`

## Motivation

progenax has a clean composable IC core — `build_spatial_ic(profile, masses, velocity_df, key, G,
Q=0.5, …) -> ICResult` — but a single-population cluster today takes **four wired steps** (sample an
IMF → construct a profile → construct a *matching* velocity DF → call `build_spatial_ic`), and the
profile/DF scale radius (`r_h`) **must be matched by hand** (a documented footgun, CLAUDE.md
"Scale Radius Mismatch"). The docs long advertised a `build_plummer_cluster(N, r_h, alpha, key)`
one-liner that was **never implemented** (a phantom caught in the 2026-06 provenance audit, Bucket A).

This design adds a thin, differentiable, units-explicit convenience layer **on top of** the
composable core — closing the footgun, giving a one-call onboarding example, and (the science driver)
exposing a clean `θ → ICResult` map that the B6/B7/B8 inference demos currently hand-wire.

## Ratified decisions (brainstorm)

1. **Shape = hybrid (A):** one generic engine `build_cluster(...)` + thin named aliases. Mirrors the
   existing house idiom (`env_to_imf_params(model=…)`, `MultiComponentCluster.from_*`). No N×M
   explosion.
2. **Parameterize by a profile *object*, not a `model=` string:** each family parameterizes
   differently (Plummer `r_h`; King `(W0, r_c)`; EFF `(r_h, γ)`; LIMEPY `(W0, g)`). Passing the
   profile keeps each family's clean constructor; `build_cluster` auto-pairs the matching DF.
3. **Modifier scope (1):** base + three optional modifier kwargs — `anisotropy_radius`,
   `tidal_radius`, `rotation`. Binaries stay in the separate `build_binary_cluster` (its
   `primary_imf × companion_model × target` API is too rich to inline).
4. **Dual mass/size spec (1):** explicit `masses=` (the fixed-data inference path) **or** `n=` +
   optional `imf=` (generative path); `n` without `imf` → equal **1 M⊙** masses (trivial one-liner).

## Core signature

```python
def build_cluster(
    profile: SpatialProfile, *, key: PRNGKeyArray,
    masses: Float[Array, "N"] | None = None,
    n: int | None = None,
    imf: IMFProtocol | None = None,
    units: UnitSystem | None = None,          # None -> DEFAULT_UNITS (STELLAR)
    Q: float = 0.5,
    anisotropy_radius: float | None = None,   # OM r_a; None -> isotropic
    tidal_radius: float | None = None,        # Jacobi r_t; None -> untruncated
    rotation: float | RotationSpec | None = None,  # omega overlay; None -> none
    revirialize: bool = False,                # re-virialize survivors after tidal cut
    softening: float = 0.0,
) -> ICResult: ...
```

**Mass/size resolution:** `masses` set → use it (`n`/`imf` must be unset, else error); elif `n` set →
`imf.sample(k_imf, n)` if `imf` else `jnp.ones(n)` (equal 1 M⊙); else error.

**Flow:** `df = matched_velocity_df(profile, anisotropy_radius)` → `ic = build_spatial_ic(profile,
masses, df, k_spatial, G=units.G, Q=Q, softening=softening)` → apply tidal/rotation modifiers →
`ICResult`. The single `key` is split internally into `(k_imf, k_spatial, k_rot)`.

## `matched_velocity_df` (new public primitive — the footgun killer)

`matched_velocity_df(profile, anisotropy_radius=None) -> VelocityDF` reads the profile's *own* scale
params and returns its equilibrium DF, so `r_h` can never desync:

| profile | matched DF |
|---------|-----------|
| `PlummerProfile(r_h)` | `PlummerVelocityDF(r_h, anisotropy_radius)` |
| `KingProfile(W0, r_c, r_t)` | `KingVelocityDF(W0, r_c, r_t)` |
| `EFFProfile(r_h, γ, …)` | `EFFVelocityDF(…, anisotropy_radius)` |
| `MichieProfile(W0, r_c, r_a)` | `MichieVelocityDF(…)` (anisotropy intrinsic) |
| `LIMEPYProfile(W0, g, …)` | `LIMEPYVelocityDF(…)` |

## Modifier semantics

- **`anisotropy_radius` (OM β = r²/(r²+r_a²)):** threaded into `matched_velocity_df`; valid only for
  **Plummer/EFF** DFs. Error if set for **King/LIMEPY** (isotropic) or **Michie** (anisotropy is
  intrinsic to the profile — pass it on `MichieProfile`). No silent ignore.
- **`tidal_radius` (Jacobi r_t):** applies `apply_tidal_truncation` — **shape-preserving and
  differentiable in r_t** (exact hard cut forward; logistic straight-through backward,
  `grad_width·r_t`). Truncated stars get **mass = 0** ("ghosts"; `N` fixed → stays jit/grad-safe;
  filter with `ic.masses > 0` for generative use).
  - ⚠️ **Caveat (audit S4):** survivors keep velocities drawn for the *untruncated* potential → the
    truncated set is **super-virial / non-stationary**. Default = **leave it + document loudly**;
    pair with `jacobi_radius(M, M_gal, R_gal)` to source r_t (the B7 tidal-inference entry point).
    Optional `revirialize=True` rescales survivors to Q.
  - **Recommended stationary route:** use **King/LIMEPY** — `r_t` is a native profile param, so
    `build_cluster(KingProfile(W0, r_c, r_t), …)` is a self-consistent truncated equilibrium (no S4).
  - Passing `tidal_radius` when the profile is already truncated (King/LIMEPY) → **error** (no
    double-truncation).
- **`rotation`:** `float ω` (solid-body about ẑ via `apply_solid_body_rotation`) or a small
  `RotationSpec(kind="solid"|"differential", omega, axis)` (differential → `apply_differential_rotation`).
  Differentiable in ω.

## Differentiability & units contract

- **Traced leaves** (grad/jit flow through): profile scalar params (`r_h`/`W0`/`r_c`/…),
  `anisotropy_radius`, `tidal_radius`, `rotation.omega`, `Q`.
- **Static:** `n`, `key`, `units`, profile *type*, `revirialize`.
- Headline use: `jax.grad(lambda r_h: loss(build_cluster(PlummerProfile(r_h), masses=m, key=k)))`.
- **Units:** convenience wrapper — `units=None → DEFAULT_UNITS` (STELLAR), per the mandatory units
  policy; the core still receives an explicit `G = units.G`.

## Placement, aliases, exports

- **New module** `src/progenax/builders_cluster.py` (`builders.py` is already 523 LOC, over the
  500-LOC limit). Holds `build_cluster`, `matched_velocity_df`, `RotationSpec`, and the aliases.
- **Named aliases (canonical 3):** `build_plummer_cluster(n|masses, r_h, …)`,
  `build_king_cluster(W0, r_c, …)`, `build_eff_cluster(r_h, γ, …)` — each constructs the profile and
  delegates to `build_cluster`. Michie/LIMEPY use the generic engine (aliases trivially added later).
- **Exports** added to `progenax.__init__.__all__`: `build_cluster`, `build_plummer_cluster`,
  `build_king_cluster`, `build_eff_cluster`, `matched_velocity_df`, `RotationSpec`.

## Test / validation plan (CLAUDE.md "Definition of Complete")

1. **Unit** (`tests/unit/builders/test_cluster_builders.py`): `matched_velocity_df` pairs all 5
   profiles scale-matched; mass-spec resolution (4 paths + 2 error cases); **`build_cluster` ≡ manual
   `build_spatial_ic` bit-identical** in the base case (proves pure sugar — no physics drift);
   anisotropy threading + unsupported-model errors; tidal masses→0 + double-truncate error; rotation
   L_z > 0; `units=None → STELLAR`; aliases ≡ `build_cluster` bit-identical.
2. **Differentiability + gradient-gate integration (headline):** `jax.grad` through `build_cluster`
   w.r.t. `r_h`, `r_a`, `r_t`, `ω` (AD-vs-FD agreement), jit-safe. **These are new public entry
   points → the grad-audit coverage ratchet (`tests/validation/grad_audit/manifest.py`) reds CI
   unless registered.** The plan MUST add them to `SYMBOL_CATEGORY` (AUDITED) + add registry Cases
   (one per differentiable knob). This keeps the gradient-gate green and is the right place for the
   AD-vs-FD assertions.
3. **Validation** (`validation/validate_cluster_builders.py`): Q ≈ 0.5 equilibrium for each alias,
   density-profile recovery, tidal-cut correctness, rotation L_z; publication-quality plots.
4. **Completion doc** `.claude-work/`.

## YAGNI — explicitly excluded (v1)

- Binaries (stay in `build_binary_cluster`).
- `model=`-string dispatch (profile object instead).
- Michie/LIMEPY named aliases (generic engine covers them).
- Multi-component (use `MultiComponentCluster.from_*`).

## Interaction with the Bucket-A provenance cleanup

The phantom `build_plummer_cluster` doc references (`methodology.md`, the `units-policy.md` /
`three-brick-state.md` architecture snippets) should be repointed to the **real** builder **after**
this lands. Until then, the other Bucket-A fixes (environment.md IGIMF/gwimf, the fractal/§7
deletions, archival snapshots) proceed independently.
