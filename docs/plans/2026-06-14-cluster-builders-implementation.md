# Cluster-builder convenience API — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Strict HITL — stop for Anna's review at every batch boundary (the `=== CHECKPOINT ===`
> markers). NEVER weaken a test/tolerance to pass; fix the physics. CI is exhausted → verify LOCALLY.
> No push/merge without Anna's explicit go.

**Goal:** Add a thin, differentiable, units-explicit convenience layer (`build_cluster` + 5 named
aliases + `matched_velocity_df` + `ClusterParams`/`build_cluster_from_params` + `RotationSpec`) on top
of the composable `build_spatial_ic` core, closing the `r_h`-desync footgun and exposing a clean
θ→ICResult map for inference.

**Architecture:** `build_cluster` is pure orchestration — resolve masses (`masses` or `n[+imf]`) →
`matched_velocity_df(profile, anisotropy_radius)` → `build_spatial_ic(...)` → optional
tidal/rotation modifiers → `ICResult`. It introduces **zero new physics**: the linchpin test proves
`build_cluster` is **bit-identical** to the manual `build_spatial_ic` composition in the base case, so
every equilibrium guarantee is inherited. Profiles/DFs are Equinox PyTrees, so `jax.grad` flows
through profile scalar params, `anisotropy_radius`, `tidal_radius`, and rotation `ω`.

**Tech Stack:** JAX (`jax.numpy`, `jax.grad`, `jax.lax`), Equinox (`eqx.Module`), jaxtyping; pytest
3-tier gate; the grad-audit registry (`tests/validation/grad_audit/`); matplotlib via
`scripts/_plotstyle.py`.

**Ratified design:** `docs/plans/2026-06-14-cluster-builder-api-design.md` (round-2 addendum). Read it
first. Relevant skills: @superpowers:test-driven-development, @superpowers:verification-before-completion,
@research-workflow:provenance-of-constants (grad-audit measured-tolerance discipline),
@jax-expert (PyTree/grad mechanics), @astro-code-dev (equilibrium/units correctness).

**Verified codebase facts (2026-06-14):**
- Core: `build_spatial_ic(profile, masses, velocity_df, key, G, Q=0.5, softening=0.0, id_offset=0)
  -> ICResult` splits `key` into `(key_pos, key_vel)` internally; `id_offset=0 → ids=None`.
- `builders.py` is 523 LOC (over the 500 limit) → ALL new code in `src/progenax/builders_cluster.py`.
- `ICResult` fields: `positions, velocities, masses, stellar_radii, ids, primordial_system_id,
  is_primordial_secondary, component_id`.
- Modifiers (reuse, do NOT reimplement):
  - `tidal.apply_tidal_truncation(positions, velocities, masses, r_t, grad_width=0.05)
    -> (positions, velocities, masses_truncated, keep_mask)` — exact hard cut fwd, logistic
    straight-through grad in `r_t`, masses→0 ghosts, fixed shape N. S4 super-virial warning is in
    its docstring.
  - `kinematics.rotation.apply_solid_body_rotation(velocities, positions, omega, axis)` and
    `apply_differential_rotation(velocities, positions, v_peak, R_peak, axis)` — additive overlays,
    deterministic (NO key), S3 non-stationary warning in the module docstring.
- `virial_scale(positions, velocities, masses, Q_target, G, softening=0.0) -> velocities` (for the
  `revirialize` opt-in).
- Profile/DF constructors (for `matched_velocity_df`):
  - `PlummerProfile(r_h)`; `PlummerVelocityDF(r_h=1.0, anisotropy_radius=None)`.
  - `EFFProfile(a, gamma, r_t, ...)`; `EFFVelocityDF(a=1.0, gamma=3.0, r_t=10.0, anisotropy_radius=None)`.
  - `KingProfile.from_W0_rc(W0, r_c, xi_max=None, n_ode_points=None, n_grid=1000)`; fields `W0, r_c, r_t`;
    `KingVelocityDF(W0=5.0, r_c=1.0, xi_max=None, n_ode_points=None, speed_method="table")` (auto-sizes ODE from W0).
  - `MichieProfile.from_W0_rc(W0, r_c, r_a, xi_max=800.0, n_ode_points=3000, n_grid=1000)`; fields
    `W0, r_c, r_a`; `MichieVelocityDF(W0=7.0, r_c=1.0, r_a=10.0, xi_max=800.0, n_ode_points=3000, speed_method="table")`.
  - `LIMEPYProfile.from_W0_rc(W0, g, r_c, r_a=None, xi_max=300.0, n_ode_points=2000, n_grid=1000)`;
    fields `W0, g, r_c, r_a` (r_a=inf when isotropic) + a static `is_aniso` flag;
    `LIMEPYVelocityDF(W0=5.0, g=1.0, r_c=1.0, r_a=None, xi_max=300.0, n_ode_points=2000, speed_method="table")`.
- Grad-audit: `Case(id, direction, fn, param, theta0, reduce=jnp.sum, expect="consistent", tol=1e-3,
  h_rel=1e-4, eps=1e-9, edges=(), hazard_id=None)`; `direction` ∈ {"params->IC","params->summary"};
  `fn` is `scalar_theta -> output_array`; reductions in `tests/validation/grad_audit/reductions.py`
  (`mean_radius, mean_speed, mean_mass, identity_sum`). The gate `test_grad_audit.py` runs every Case
  and asserts `status ∈ {clean, known-limitation}` (clean ⟺ finite ∧ |AD|>eps ∧ |AD/FD−1|<tol).
  Ratchet (`test_manifest_coverage.py`): `SYMBOL_CATEGORY` must cover all `__all__`; every `AUDITED`
  symbol must own a registry id (`cid == s` or `cid.startswith(s+".")` or `cid.startswith(s+"[")`);
  every `MUST_AUDIT` key must have a matching Case.
- Validation idiom: `scripts/validate_<name>.py` with `fig_*(output_dir)` functions calling
  `save_fig(fig, output_dir, "<name>")` from `scripts/_plotstyle.py`; plots written to
  `validation/plots/<subdir>/`. Demos: `scripts/demo_*.py`; Fisher helper
  `scripts/_demo_inference.py::fisher_information_gn`.

**OPEN for Anna at plan review (not blocking):** the kickoff said `validation/validate_cluster_builders.py`,
but every sibling validation *script* lives in `scripts/validate_*.py` (writing plots to
`validation/plots/`). This plan follows the **repo convention** (`scripts/validate_cluster_builders.py`
→ `validation/plots/cluster_builders/`). Flag if you want it under `validation/` instead.

---

## Batch 0: Branch + module skeleton

Already on `feat/cluster-builders` (off `main`). Design doc committed (`ba5ed46`).

### Task 0.1: Create the empty module + test package

**Files:**
- Create: `src/progenax/builders_cluster.py` (module docstring + imports only)
- Create: `tests/unit/builders/__init__.py` (empty)
- Create: `tests/unit/builders/test_cluster_builders.py` (imports only)

**Step 1:** Write the module header:

```python
"""Convenience cluster-IC builders (thin, differentiable sugar over build_spatial_ic).

Public API: build_cluster, matched_velocity_df, RotationSpec, ClusterParams,
build_cluster_from_params, and the named aliases build_{plummer,king,eff,michie,limepy}_cluster.

Design: docs/plans/2026-06-14-cluster-builder-api-design.md (round-2 addendum).
"""
from __future__ import annotations
from typing import Optional, Union

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .defaults import DEFAULT_UNITS
from .protocols import SpatialProfile, VelocityDF
from .builders import ICResult, build_spatial_ic, virial_scale, compute_stellar_radii
from .profiles import (
    PlummerProfile, EFFProfile, KingProfile, MichieProfile, LIMEPYProfile,
)
from .kinematics import (
    PlummerVelocityDF, EFFVelocityDF, KingVelocityDF, MichieVelocityDF, LIMEPYVelocityDF,
    apply_solid_body_rotation, apply_differential_rotation,
)

_ZHAT = jnp.array([0.0, 0.0, 1.0])

__all__ = [
    "build_cluster", "matched_velocity_df", "RotationSpec", "ClusterParams",
    "build_cluster_from_params", "build_plummer_cluster", "build_king_cluster",
    "build_eff_cluster", "build_michie_cluster", "build_limepy_cluster",
]
```

**Step 2:** Run `env -u VIRTUAL_ENV uv run --no-sync python -c "import progenax.builders_cluster"`
Expected: imports clean (no symbols defined yet — fine).

**Step 3:** Commit.

```bash
git add src/progenax/builders_cluster.py tests/unit/builders/
git commit -m "feat(builders_cluster): module skeleton + test package

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 0 → Anna ===`

---

## Batch 1: `matched_velocity_df` (the footgun-killer)

### Task 1.1: RED — pairing tests for all 5 profiles + error semantics

**Files:** Test: `tests/unit/builders/test_cluster_builders.py`

**Step 1:** Write the failing tests:

```python
import jax
import jax.numpy as jnp
import pytest
from jaxstro.units import STELLAR
from progenax import (
    PlummerProfile, EFFProfile, KingProfile, MichieProfile, LIMEPYProfile,
    PlummerVelocityDF, EFFVelocityDF, KingVelocityDF, MichieVelocityDF, LIMEPYVelocityDF,
)
from progenax.builders_cluster import matched_velocity_df


def test_matched_plummer_scale_matched():
    p = PlummerProfile(r_h=2.3)
    df = matched_velocity_df(p)
    assert isinstance(df, PlummerVelocityDF)
    assert float(df.r_h) == float(p.r_h)        # scale never desyncs


def test_matched_eff_scale_matched():
    p = EFFProfile(a=1.4, gamma=3.2, r_t=12.0)
    df = matched_velocity_df(p)
    assert isinstance(df, EFFVelocityDF)
    assert float(df.a) == float(p.a)
    assert float(df.gamma) == float(p.gamma)
    assert float(df.r_t) == float(p.r_t)


def test_matched_king_scale_matched():
    p = KingProfile.from_W0_rc(W0=7.0, r_c=1.3)
    df = matched_velocity_df(p)
    assert isinstance(df, KingVelocityDF)
    assert float(df.W0) == float(p.W0)
    assert float(df.r_c) == float(p.r_c)


def test_matched_michie_scale_matched():
    p = MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0)
    df = matched_velocity_df(p)
    assert isinstance(df, MichieVelocityDF)
    assert float(df.W0) == float(p.W0)
    assert float(df.r_a) == float(p.r_a)


def test_matched_limepy_isotropic_passes_none_r_a():
    p = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.5, r_c=1.0)   # isotropic -> profile.r_a = inf
    df = matched_velocity_df(p)
    assert isinstance(df, LIMEPYVelocityDF)
    assert not bool(jnp.isfinite(df.r_a))  # isotropic DF stores r_a = inf


def test_matched_limepy_anisotropic_threads_r_a():
    p = LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0, r_a=6.0, xi_max=800.0)
    df = matched_velocity_df(p)
    assert isinstance(df, LIMEPYVelocityDF)
    assert float(df.r_a) == pytest.approx(6.0)


def test_anisotropy_radius_valid_for_plummer_eff():
    df = matched_velocity_df(PlummerProfile(r_h=1.0), anisotropy_radius=0.9)
    assert df.anisotropy_radius is not None
    df2 = matched_velocity_df(EFFProfile(a=1.0, gamma=5.0, r_t=10.0), anisotropy_radius=3.0)
    assert df2.anisotropy_radius is not None


@pytest.mark.parametrize("profile", [
    KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
    MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0),
    LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0),
])
def test_anisotropy_radius_errors_for_non_om_models(profile):
    with pytest.raises(ValueError, match="anisotropy_radius"):
        matched_velocity_df(profile, anisotropy_radius=2.0)


def test_unknown_profile_type_errors():
    class Bogus:  # not a known profile
        pass
    with pytest.raises(TypeError, match="matched_velocity_df"):
        matched_velocity_df(Bogus())
```

**Step 2:** Run
`XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/builders/test_cluster_builders.py -q`
Expected: FAIL (matched_velocity_df not defined / ImportError).

### Task 1.2: GREEN — implement `matched_velocity_df`

**Files:** Modify: `src/progenax/builders_cluster.py`

**Step 3:** Implement:

```python
def matched_velocity_df(
    profile: SpatialProfile,
    anisotropy_radius: Optional[float] = None,
) -> VelocityDF:
    """Return the equilibrium velocity DF whose scale params match `profile` exactly.

    Kills the r_h-desync footgun: the DF reads the profile's OWN scale fields, so a
    Plummer profile can never be paired with a mismatched-r_h DF.

    `anisotropy_radius` (Osipkov-Merritt r_a; β(r)=r²/(r²+r_a²)) is valid ONLY for
    Plummer/EFF (whose base DFs are isotropic and OM-augmentable). King is isotropic;
    Michie/LIMEPY carry their anisotropy intrinsically (set `r_a` on the profile via
    `from_W0_rc(..., r_a=...)`), so passing `anisotropy_radius` for those is an ERROR
    (no silent ignore).

    Caveat: King/Michie/LIMEPY DFs re-solve their ODE at DEFAULT domains (consistent
    with the default profile constructors). A profile built with a custom `xi_max`
    cannot round-trip that domain (not stored as a field) — hand-compose for that case.
    """
    if isinstance(profile, PlummerProfile):
        return PlummerVelocityDF(r_h=profile.r_h, anisotropy_radius=anisotropy_radius)
    if isinstance(profile, EFFProfile):
        return EFFVelocityDF(a=profile.a, gamma=profile.gamma, r_t=profile.r_t,
                             anisotropy_radius=anisotropy_radius)
    # --- isotropic / intrinsically-anisotropic models: anisotropy_radius is invalid ---
    if anisotropy_radius is not None:
        raise ValueError(
            f"anisotropy_radius is only valid for Plummer/EFF DFs; got "
            f"{type(profile).__name__}. King is isotropic; for Michie/LIMEPY set the "
            f"anisotropy radius r_a on the profile (e.g. "
            f"{type(profile).__name__}.from_W0_rc(..., r_a=...))."
        )
    if isinstance(profile, KingProfile):
        return KingVelocityDF(W0=profile.W0, r_c=profile.r_c)
    if isinstance(profile, MichieProfile):
        return MichieVelocityDF(W0=profile.W0, r_c=profile.r_c, r_a=profile.r_a)
    if isinstance(profile, LIMEPYProfile):
        # LIMEPY stores r_a=inf for the isotropic model; the DF wants r_a=None there.
        # Branch on the STATIC is_aniso flag (not a traced jnp.isfinite).
        r_a = profile.r_a if profile.is_aniso else None
        return LIMEPYVelocityDF(W0=profile.W0, g=profile.g, r_c=profile.r_c, r_a=r_a)
    raise TypeError(
        f"matched_velocity_df: unknown profile type {type(profile).__name__}. "
        f"Supported: Plummer, EFF, King, Michie, LIMEPY."
    )
```

> **Executor note:** verify `LIMEPYProfile` actually exposes a static `is_aniso` attribute
> (`grep -n "is_aniso" src/progenax/profiles/limepy.py`). If it is named differently (e.g.
> `_is_aniso`) use that. If LIMEPY has no such flag, fall back to a static construction-time
> branch: store whether `r_a` was finite at build time. Do NOT use a traced `jnp.isfinite`.

**Step 4:** Run the Task 1.1 tests → Expected: PASS (all 10).

**Step 5:** Commit.

```bash
git add src/progenax/builders_cluster.py tests/unit/builders/test_cluster_builders.py
git commit -m "feat(builders_cluster): matched_velocity_df — equilibrium DF auto-pairing

Maps each profile to its scale-matched VelocityDF (kills the r_h-desync footgun).
anisotropy_radius valid only for Plummer/EFF; errors for King/Michie/LIMEPY.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 1 → Anna ===`

---

## Batch 2: `build_cluster` base (mass-spec + purity)

### Task 2.1: RED — mass-spec resolution + bit-identical purity + units default

**Files:** Test: `tests/unit/builders/test_cluster_builders.py` (append)

**Step 1:** Write failing tests:

```python
from progenax import build_spatial_ic, PowerLawIMF, DEFAULT_UNITS
from progenax.builders_cluster import build_cluster

_K = jax.random.PRNGKey(0)
_M = jnp.ones(200)


def _assert_ic_equal(a, b):
    for field in ("positions", "velocities", "masses", "stellar_radii"):
        assert bool(jnp.all(getattr(a, field) == getattr(b, field))), f"{field} differs"


def test_build_cluster_is_bit_identical_to_manual_base_case():
    # The linchpin: build_cluster(profile, masses, key) MUST equal the manual
    # build_spatial_ic composition exactly (pure sugar, no physics drift).
    p = PlummerProfile(r_h=1.0)
    ic = build_cluster(p, masses=_M, key=_K)                       # units=None -> STELLAR
    df = matched_velocity_df(p)
    manual = build_spatial_ic(p, _M, df, _K, G=STELLAR.G, Q=0.5)
    _assert_ic_equal(ic, manual)


def test_units_none_resolves_to_default_stellar():
    p = PlummerProfile(r_h=1.0)
    ic_none = build_cluster(p, masses=_M, key=_K, units=None)
    ic_stellar = build_cluster(p, masses=_M, key=_K, units=STELLAR)
    _assert_ic_equal(ic_none, ic_stellar)
    assert DEFAULT_UNITS is STELLAR  # documents the resolution target


def test_mass_spec_n_only_is_equal_one_msun():
    ic = build_cluster(PlummerProfile(r_h=1.0), n=128, key=_K)
    assert ic.masses.shape == (128,)
    assert bool(jnp.allclose(ic.masses, 1.0))


def test_mass_spec_n_plus_imf_samples():
    imf = PowerLawIMF.kroupa()
    ic = build_cluster(PlummerProfile(r_h=1.0), n=256, imf=imf, key=_K)
    assert ic.masses.shape == (256,)
    assert float(jnp.std(ic.masses)) > 0.0          # not all equal -> IMF actually sampled


def test_mass_spec_masses_array_used_verbatim():
    m = jnp.linspace(0.5, 3.0, 64)
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=m, key=_K)
    assert bool(jnp.all(ic.masses == m))


def test_mass_spec_error_both_masses_and_n():
    with pytest.raises(ValueError, match="masses.*or.*n|exactly one"):
        build_cluster(PlummerProfile(r_h=1.0), masses=_M, n=10, key=_K)


def test_mass_spec_error_neither_masses_nor_n():
    with pytest.raises(ValueError, match="masses.*or.*n|exactly one"):
        build_cluster(PlummerProfile(r_h=1.0), key=_K)


def test_mass_spec_error_imf_without_n():
    with pytest.raises(ValueError, match="imf.*requires.*n|n.*imf"):
        build_cluster(PlummerProfile(r_h=1.0), masses=_M, imf=PowerLawIMF.kroupa(), key=_K)
```

**Step 2:** Run → Expected: FAIL (build_cluster not defined).

### Task 2.2: GREEN — implement `build_cluster` base + mass-spec helper

**Files:** Modify: `src/progenax/builders_cluster.py`

**Step 3:** Implement the mass-spec resolver + the base `build_cluster` (modifiers added in Batch 3).
Key design: **split the key only when an IMF draw is needed** so the `masses`/`n`-only paths pass the
user key straight to `build_spatial_ic` (exact purity).

```python
def _resolve_masses(masses, n, imf, key):
    """Return (masses, key_spatial). Split the key ONLY when an IMF draw is needed."""
    if masses is not None:
        if n is not None:
            raise ValueError("pass exactly one of `masses` or `n` (got both).")
        if imf is not None:
            raise ValueError("`imf` requires `n` (the count to sample); pass `n=...`, not `masses=...`.")
        return masses, key
    if n is None:
        raise ValueError("pass exactly one of `masses` or `n` (got neither).")
    if imf is None:
        return jnp.ones(n), key                       # equal 1 M_sun, no PRNG needed
    key_imf, key_spatial = jax.random.split(key)
    return imf.sample(key_imf, n), key_spatial


def build_cluster(
    profile: SpatialProfile,
    *,
    key: PRNGKeyArray,
    masses: Optional[Float[Array, "N"]] = None,
    n: Optional[int] = None,
    imf=None,
    units=None,
    Q: float = 0.5,
    anisotropy_radius: Optional[float] = None,
    tidal_radius: Optional[float] = None,
    rotation: Optional[Union[float, "RotationSpec"]] = None,
    revirialize: bool = False,
    softening: float = 0.0,
) -> ICResult:
    """Build a single-population cluster IC from a profile object (see design doc)."""
    units = DEFAULT_UNITS if units is None else units
    masses, key_spatial = _resolve_masses(masses, n, imf, key)
    df = matched_velocity_df(profile, anisotropy_radius)
    ic = build_spatial_ic(profile, masses, df, key_spatial, G=units.G, Q=Q, softening=softening)

    if tidal_radius is None and rotation is None:
        return ic                                     # base case: bit-identical to build_spatial_ic
    return _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, units.G, softening)
```

Add a temporary stub so the base tests pass before Batch 3:

```python
def _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, G, softening):
    return ic  # filled in Batch 3
```

**Step 4:** Run the Task 2.1 tests → Expected: PASS (8).

**Step 5:** Commit.

```bash
git add src/progenax/builders_cluster.py tests/unit/builders/test_cluster_builders.py
git commit -m "feat(builders_cluster): build_cluster base — mass-spec + bit-identical purity

masses | n[+imf] resolution (key split only for IMF draws so the masses/n paths are
bit-identical to manual build_spatial_ic); units=None -> DEFAULT_UNITS (STELLAR).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 2 → Anna ===`

---

## Batch 3: Modifiers (`RotationSpec` + anisotropy/tidal/rotation)

### Task 3.1: RED — modifier behavior + error semantics

**Files:** Test: `tests/unit/builders/test_cluster_builders.py` (append)

**Step 1:** Write failing tests:

```python
from progenax.builders_cluster import RotationSpec


def test_anisotropy_threads_into_df_radial_bias():
    # OM anisotropy -> radial velocity bias (beta(r) > 0). Compare radial vs tangential
    # velocity variance at large radius for an anisotropic vs isotropic Plummer.
    p = PlummerProfile(r_h=1.0)
    iso = build_cluster(p, masses=_M, key=_K)
    ani = build_cluster(p, masses=_M, key=_K, anisotropy_radius=0.7)
    # Anisotropic build must differ from isotropic (threading actually happened).
    assert not bool(jnp.allclose(iso.velocities, ani.velocities))


def test_anisotropy_unsupported_model_errors():
    with pytest.raises(ValueError, match="anisotropy_radius"):
        build_cluster(KingProfile.from_W0_rc(W0=7.0, r_c=1.0), masses=_M, key=_K,
                      anisotropy_radius=2.0)


def test_tidal_zeroes_outer_masses():
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, tidal_radius=1.5)
    radii = jnp.linalg.norm(ic.positions, axis=1)
    assert bool(jnp.all(ic.masses[radii > 1.5] == 0.0))     # ghosts
    assert bool(jnp.any(ic.masses[radii <= 1.5] > 0.0))     # survivors kept


def test_tidal_double_truncation_errors_for_king():
    with pytest.raises(ValueError, match="already truncated|double"):
        build_cluster(KingProfile.from_W0_rc(W0=7.0, r_c=1.0), masses=_M, key=_K,
                      tidal_radius=5.0)


def test_tidal_double_truncation_errors_for_limepy():
    with pytest.raises(ValueError, match="already truncated|double"):
        build_cluster(LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0), masses=_M, key=_K,
                      tidal_radius=5.0)


def _Lz(ic):
    x, y = ic.positions[:, 0], ic.positions[:, 1]
    vx, vy = ic.velocities[:, 0], ic.velocities[:, 1]
    return float(jnp.sum(ic.masses * (x * vy - y * vx)))


def test_rotation_float_injects_positive_Lz():
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=0.3)
    assert _Lz(ic) > 0.0


def test_rotation_spec_solid_matches_float():
    ic_f = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=0.3)
    ic_s = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K,
                         rotation=RotationSpec(omega=0.3))
    _assert_ic_equal(ic_f, ic_s)


def test_rotation_spec_differential_injects_Lz():
    spec = RotationSpec(kind="differential", v_peak=2.0, R_peak=1.0)
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K, rotation=spec)
    assert _Lz(ic) > 0.0


def test_revirialize_rescales_survivors_to_Q():
    # After a tidal cut, survivors are super-virial (S4); revirialize=True restores Q≈0.5.
    from progenax import compute_kinetic_energy, compute_potential_energy
    ic = build_cluster(PlummerProfile(r_h=1.0), masses=jnp.ones(2000), key=_K,
                       tidal_radius=2.0, revirialize=True)
    keep = ic.masses > 0
    pos, vel, m = ic.positions[keep], ic.velocities[keep], ic.masses[keep]
    T = compute_kinetic_energy(vel, m)
    V = compute_potential_energy(pos, m, G=STELLAR.G)
    Q = float(T / jnp.abs(V))
    assert Q == pytest.approx(0.5, abs=0.05)
```

**Step 2:** Run → Expected: FAIL (RotationSpec undefined; modifiers are no-ops).

### Task 3.2: GREEN — `RotationSpec` + `_apply_modifiers`

**Files:** Modify: `src/progenax/builders_cluster.py`

**Step 3:** Implement `RotationSpec` and the real `_apply_modifiers` (replace the stub). Tidal
double-truncation guard keys off the profile type (King/LIMEPY carry a native `r_t`).

```python
class RotationSpec(eqx.Module):
    """Rotation overlay spec. kind='solid' uses omega; kind='differential' uses (v_peak, R_peak).

    Additive kinematic overlay — injects L_z and raises Q above 0.5 (audit S3, NOT a stationary
    equilibrium). Differentiable in omega / v_peak / R_peak.
    """
    kind: str = eqx.field(static=True, default="solid")
    omega: Optional[Float[Array, ""]] = None
    v_peak: Optional[Float[Array, ""]] = None
    R_peak: Optional[Float[Array, ""]] = None
    axis: Float[Array, "3"] = _ZHAT

    def __post_init__(self):
        if self.kind == "solid" and self.omega is None:
            raise ValueError("RotationSpec(kind='solid') requires omega=...")
        if self.kind == "differential" and (self.v_peak is None or self.R_peak is None):
            raise ValueError("RotationSpec(kind='differential') requires v_peak=... and R_peak=...")
        if self.kind not in ("solid", "differential"):
            raise ValueError(f"RotationSpec.kind must be 'solid' or 'differential', got {self.kind!r}")


_TRUNCATED_PROFILES = (KingProfile, LIMEPYProfile, MichieProfile, EFFProfile)  # native r_t; tidal_radius is Plummer-only (review I1, Anna-ratified)


def _apply_modifiers(ic, profile, tidal_radius, rotation, revirialize, Q, G, softening):
    positions, velocities, masses = ic.positions, ic.velocities, ic.masses

    if tidal_radius is not None:
        if isinstance(profile, _TRUNCATED_PROFILES):
            raise ValueError(
                f"{type(profile).__name__} is already tidally truncated (native r_t); passing "
                f"tidal_radius would double-truncate. For a stationary truncated equilibrium set "
                f"r_t on the profile instead (it is the recommended route — no audit-S4 issue)."
            )
        from .tidal import apply_tidal_truncation
        positions, velocities, masses, _keep = apply_tidal_truncation(
            positions, velocities, masses, tidal_radius)
        if revirialize:
            velocities = virial_scale(positions, velocities, masses, Q, G, softening)

    if rotation is not None:
        spec = RotationSpec(omega=rotation) if not isinstance(rotation, RotationSpec) else rotation
        if spec.kind == "solid":
            velocities = apply_solid_body_rotation(velocities, positions, spec.omega, spec.axis)
        else:
            velocities = apply_differential_rotation(
                velocities, positions, spec.v_peak, spec.R_peak, spec.axis)

    return ICResult(
        positions=positions, velocities=velocities, masses=masses,
        stellar_radii=compute_stellar_radii(masses), ids=ic.ids,
        primordial_system_id=ic.primordial_system_id,
        is_primordial_secondary=ic.is_primordial_secondary, component_id=ic.component_id,
    )
```

> **Executor note (anisotropy threading):** `anisotropy_radius` is already threaded in Batch 2 via
> `matched_velocity_df(profile, anisotropy_radius)` inside `build_cluster`, and the unsupported-model
> error is raised there. The `test_anisotropy_*` tests in this batch just confirm the end-to-end
> effect through `build_cluster`. No new code needed for anisotropy here.

**Step 4:** Run the full Batch-3 test set → Expected: PASS. If `test_revirialize_*` Q is off,
investigate the physics (do NOT loosen `abs=0.05` without measuring) — likely the survivor energy
recompute. Re-run the whole `test_cluster_builders.py` to confirm no regression.

**Step 5:** Commit.

```bash
git add src/progenax/builders_cluster.py tests/unit/builders/test_cluster_builders.py
git commit -m "feat(builders_cluster): modifiers — RotationSpec, tidal cut, rotation overlay

Tidal: apply_tidal_truncation (masses->0 ghosts) + double-truncation guard for King/LIMEPY
+ optional revirialize-to-Q. Rotation: solid (omega) / differential (v_peak,R_peak) overlays
(S3 non-stationary). Anisotropy threaded via matched_velocity_df.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 3 → Anna ===`

---

## Batch 4: Aliases + `ClusterParams` + `build_cluster_from_params` + exports

### Task 4.1: RED — aliases ≡ build_cluster, wrapper ≡ build_cluster, public API

**Files:** Test: `tests/unit/builders/test_cluster_builders.py` (append); and
`tests/unit/test_public_api.py` (extend the existing exported-symbol check — read it first).

**Step 1:** Write failing tests:

```python
from progenax.builders_cluster import (
    build_plummer_cluster, build_king_cluster, build_eff_cluster,
    build_michie_cluster, build_limepy_cluster,
    ClusterParams, build_cluster_from_params,
)


def test_plummer_alias_identical():
    ic_a = build_plummer_cluster(masses=_M, r_h=1.7, key=_K)
    ic_b = build_cluster(PlummerProfile(r_h=1.7), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_plummer_alias_n_path():
    ic = build_plummer_cluster(n=100, r_h=1.0, key=_K)
    assert ic.masses.shape == (100,)


def test_king_alias_identical():
    ic_a = build_king_cluster(masses=_M, W0=7.0, r_c=1.2, key=_K)
    ic_b = build_cluster(KingProfile.from_W0_rc(W0=7.0, r_c=1.2), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_eff_alias_identical():
    ic_a = build_eff_cluster(masses=_M, a=1.0, gamma=3.0, r_t=10.0, key=_K)
    ic_b = build_cluster(EFFProfile(a=1.0, gamma=3.0, r_t=10.0), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_michie_alias_identical():
    ic_a = build_michie_cluster(masses=_M, W0=7.0, r_c=1.0, r_a=8.0, key=_K)
    ic_b = build_cluster(MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_limepy_alias_identical():
    ic_a = build_limepy_cluster(masses=_M, W0=5.0, g=1.0, r_c=1.0, key=_K)
    ic_b = build_cluster(LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0), masses=_M, key=_K)
    _assert_ic_equal(ic_a, ic_b)


def test_cluster_params_wrapper_identical_to_build_cluster():
    params = ClusterParams(profile=PlummerProfile(r_h=1.3), tidal_radius=3.0, rotation=0.2)
    ic_w = build_cluster_from_params(params, masses=_M, key=_K)
    ic_d = build_cluster(PlummerProfile(r_h=1.3), masses=_M, key=_K,
                         tidal_radius=3.0, rotation=0.2)
    _assert_ic_equal(ic_w, ic_d)


def test_cluster_params_defaults_base_case():
    params = ClusterParams(profile=PlummerProfile(r_h=1.0))
    ic_w = build_cluster_from_params(params, masses=_M, key=_K)
    ic_d = build_cluster(PlummerProfile(r_h=1.0), masses=_M, key=_K)
    _assert_ic_equal(ic_w, ic_d)


def test_all_new_symbols_exported_from_progenax():
    import progenax
    for sym in ("build_cluster", "build_plummer_cluster", "build_king_cluster",
                "build_eff_cluster", "build_michie_cluster", "build_limepy_cluster",
                "matched_velocity_df", "RotationSpec", "ClusterParams",
                "build_cluster_from_params"):
        assert sym in progenax.__all__, f"{sym} missing from progenax.__all__"
        assert hasattr(progenax, sym), f"progenax.{sym} not importable"
```

**Step 2:** Run → Expected: FAIL.

### Task 4.2: GREEN — aliases, `ClusterParams`, wrapper

**Files:** Modify: `src/progenax/builders_cluster.py`, `src/progenax/__init__.py`

**Step 3:** Add the aliases + wrapper to `builders_cluster.py`:

```python
def build_plummer_cluster(*, key, masses=None, n=None, r_h=1.0, imf=None, **kw):
    return build_cluster(PlummerProfile(r_h=r_h), key=key, masses=masses, n=n, imf=imf, **kw)


def build_king_cluster(*, key, masses=None, n=None, W0=7.0, r_c=1.0, imf=None, **kw):
    return build_cluster(KingProfile.from_W0_rc(W0=W0, r_c=r_c), key=key,
                         masses=masses, n=n, imf=imf, **kw)


def build_eff_cluster(*, key, masses=None, n=None, a=1.0, gamma=3.0, r_t=10.0, imf=None, **kw):
    return build_cluster(EFFProfile(a=a, gamma=gamma, r_t=r_t), key=key,
                         masses=masses, n=n, imf=imf, **kw)


def build_michie_cluster(*, key, masses=None, n=None, W0=7.0, r_c=1.0, r_a=8.0, imf=None, **kw):
    return build_cluster(MichieProfile.from_W0_rc(W0=W0, r_c=r_c, r_a=r_a), key=key,
                         masses=masses, n=n, imf=imf, **kw)


def build_limepy_cluster(*, key, masses=None, n=None, W0=5.0, g=1.0, r_c=1.0, r_a=None,
                         imf=None, **kw):
    return build_cluster(LIMEPYProfile.from_W0_rc(W0=W0, g=g, r_c=r_c, r_a=r_a), key=key,
                         masses=masses, n=n, imf=imf, **kw)


class ClusterParams(eqx.Module):
    """Differentiable θ-PyTree: profile + named modifier knobs (see design doc).

    jax.grad over a ClusterParams gives joint gradients over the profile's float leaves AND
    any non-None modifier (the structure declares which params are free). `revirialize`/
    `softening` are static force-model config -> kwargs of build_cluster_from_params, not fields.
    """
    profile: SpatialProfile
    anisotropy_radius: Optional[Float[Array, ""]] = None
    tidal_radius: Optional[Float[Array, ""]] = None
    rotation: Optional[Union[float, RotationSpec]] = None
    Q: float = 0.5


def build_cluster_from_params(
    params: ClusterParams, *, key, masses=None, n=None, imf=None, units=None,
    revirialize: bool = False, softening: float = 0.0,
) -> ICResult:
    """Unpack a ClusterParams θ-PyTree into build_cluster (the inference forward map)."""
    return build_cluster(
        params.profile, key=key, masses=masses, n=n, imf=imf, units=units,
        anisotropy_radius=params.anisotropy_radius, tidal_radius=params.tidal_radius,
        rotation=params.rotation, Q=params.Q, revirialize=revirialize, softening=softening)
```

> **Executor note:** the `**kw` in each alias forwards `units, Q, anisotropy_radius, tidal_radius,
> rotation, revirialize, softening`. Keep `imf` explicit (not in `**kw`) so the signature documents
> the generative path. Verify each alias's `r_*`/`W0`/etc. names match the docstrings.

**Step 4:** Wire exports into `src/progenax/__init__.py`: add an import block and `__all__` entries
(place after the `from .builders import (...)` block):

```python
from .builders_cluster import (
    build_cluster, matched_velocity_df, RotationSpec, ClusterParams, build_cluster_from_params,
    build_plummer_cluster, build_king_cluster, build_eff_cluster,
    build_michie_cluster, build_limepy_cluster,
)
```
and append those 10 names to `__all__` under a new `# Cluster convenience builders` comment.

**Step 5:** Run Batch-4 tests + the whole `test_cluster_builders.py` + `tests/unit/test_public_api.py`
+ `tests/unit/test_documented_api.py` → Expected: PASS. (The public-API tests guard `__all__` ↔ exports.)

**Step 6:** Commit.

```bash
git add src/progenax/builders_cluster.py src/progenax/__init__.py tests/unit/builders/test_cluster_builders.py
git commit -m "feat(builders_cluster): 5 aliases + ClusterParams/build_cluster_from_params + exports

build_{plummer,king,eff,michie,limepy}_cluster delegate to build_cluster (bit-identical).
ClusterParams eqx.Module θ-PyTree + build_cluster_from_params wrapper for joint-param inference.
All 10 symbols exported from progenax.__all__.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 4 → Anna ===`

---

## Batch 5: Grad-audit gate registration (the ratchet)

**Goal:** register all 10 new symbols so the coverage ratchet stays green, and add ~10 measured
AD-vs-FD Cases. @research-workflow:provenance-of-constants — every Case's `tol` is set from a MEASURED
AD/FD/ratio printed in a provenance comment; NEVER weaken `tol` to pass a mismatch (a blocked gradient
shows `|ratio−1|~1`, the silent-zero signature, not a small band).

### Task 5.1: Measure the gradients first (no commit)

**Step 1:** Write a scratch script `/tmp/measure_cluster_grads.py` that, for each new Case closure,
prints `theta0, AD, FD, |ratio−1|, |AD|, flips` (flips = categorical-assignment flips at ±h if any;
for these single-population samplers there are none). Use the closures from Task 5.2. Run it and record
the numbers — they become the provenance comments and set each `tol`.

Run: `XLA_FLAGS=... env -u VIRTUAL_ENV uv run --no-sync python /tmp/measure_cluster_grads.py`

> Reductions: `mean_radius` for r_h/W0/gamma (positions), `mean_speed` for anisotropy_radius and
> rotation ω (velocities), `mean_radius` for tidal_radius (positions; the ghost masses don't enter
> `mean_radius`, but the truncation moves the surviving COM — confirm |AD|>eps; if ~0, reduce by a
> mass-weighted radius instead and document). Measure each and pick the reduction that gives a live,
> FD-consistent gradient.

### Task 5.2: GREEN — add closures + Cases + manifest entries

**Files:**
- Modify: `tests/validation/grad_audit/registry.py` (add closures + `REGISTRY` Cases)
- Modify: `tests/validation/grad_audit/manifest.py` (`SYMBOL_CATEGORY` + `MUST_AUDIT`)

**Step 2:** Add import + closures to `registry.py` (near the other builder closures):

```python
from progenax.builders_cluster import (
    build_cluster, build_king_cluster, build_eff_cluster, build_michie_cluster,
    build_limepy_cluster, ClusterParams, build_cluster_from_params,
)

# --- build_cluster modifier channels (Plummer base) -------------------------
def _bc_plummer_rh(r_h):
    return build_cluster(PlummerProfile(r_h=r_h), masses=_MASSES, key=_KEY).positions

def _bc_plummer_om(r_a):
    return build_cluster(PlummerProfile(r_h=1.0), masses=_MASSES, key=_KEY,
                         anisotropy_radius=r_a).velocities

def _bc_plummer_tidal(r_t):
    return build_cluster(PlummerProfile(r_h=1.0), masses=_MASSES, key=_KEY,
                         tidal_radius=r_t).positions

def _bc_plummer_omega(omega):
    return build_cluster(PlummerProfile(r_h=1.0), masses=_MASSES, key=_KEY,
                         rotation=omega).velocities

# --- per-family coverage THROUGH the alias builders -------------------------
def _bk_W0(W0):
    return build_king_cluster(masses=_MASSES, W0=W0, r_c=1.0, key=_KEY,
                              xi_max=_KING_XI_MAX, n_ode_points=_KING_N_ODE).positions
# (executor: build_king_cluster forwards **kw to build_cluster -> KingProfile.from_W0_rc;
#  confirm xi_max/n_ode_points reach from_W0_rc. If the alias does not forward ODE-domain
#  kwargs, add them to the alias signature OR fix W0 concrete and audit r_c instead.)

def _beff_gamma(gamma):
    return build_eff_cluster(masses=_MASSES, a=1.0, gamma=gamma, r_t=10.0, key=_KEY).positions

def _bmich_W0(W0):
    return build_michie_cluster(masses=_MASSES, W0=W0, r_c=1.0, r_a=_MICHIE_R_A, key=_KEY).positions

def _blim_W0(W0):
    return build_limepy_cluster(masses=_MASSES, W0=W0, g=1.0, r_c=1.0, key=_KEY).positions

# --- build_cluster_from_params (ClusterParams θ-PyTree) ----------------------
def _bcfp_rh(r_h):
    params = ClusterParams(profile=PlummerProfile(r_h=r_h))
    return build_cluster_from_params(params, masses=_MASSES, key=_KEY).positions

def _bcfp_tidal(r_t):
    params = ClusterParams(profile=PlummerProfile(r_h=1.0), tidal_radius=r_t)
    return build_cluster_from_params(params, masses=_MASSES, key=_KEY).positions
```

**Step 3:** Append Cases to `REGISTRY` (fill `tol` from Task 5.1 measurements; the comments below are
placeholders — replace with MEASURED AD/FD/ratio):

```python
    # --- cluster convenience builders (Task: cluster-builders arc) ---
    Case(id="build_cluster[Plummer]", direction="params->IC", fn=_bc_plummer_rh,
         param="r_h", theta0=1.0, reduce=mean_radius, tol=1e-5),       # MEASURED: AD=.. FD=.. ratio=..
    Case(id="build_cluster[Plummer+OM]", direction="params->IC", fn=_bc_plummer_om,
         param="anisotropy_radius", theta0=0.7, reduce=mean_speed, tol=3e-3),  # MEASURED ..
    Case(id="build_cluster[Plummer+tidal]", direction="params->IC", fn=_bc_plummer_tidal,
         param="tidal_radius", theta0=1.5, reduce=mean_radius, tol=1e-3),      # MEASURED ..
    Case(id="build_cluster[Plummer+rotation]", direction="params->IC", fn=_bc_plummer_omega,
         param="omega", theta0=0.3, reduce=mean_speed, tol=1e-4),             # MEASURED ..
    Case(id="build_king_cluster", direction="params->IC", fn=_bk_W0,
         param="W0", theta0=7.0, reduce=mean_radius, tol=1e-3),               # MEASURED ..
    Case(id="build_eff_cluster", direction="params->IC", fn=_beff_gamma,
         param="gamma", theta0=3.0, reduce=mean_radius, tol=1e-3),            # MEASURED ..
    Case(id="build_michie_cluster", direction="params->IC", fn=_bmich_W0,
         param="W0", theta0=7.0, reduce=mean_radius, tol=1e-3),               # MEASURED ..
    Case(id="build_limepy_cluster", direction="params->IC", fn=_blim_W0,
         param="W0", theta0=5.0, reduce=mean_radius, tol=1e-3),               # MEASURED ..
    Case(id="build_cluster_from_params[ClusterParams]", direction="params->IC", fn=_bcfp_rh,
         param="r_h", theta0=1.0, reduce=mean_radius, tol=1e-5),              # MEASURED ..
    Case(id="build_cluster_from_params[ClusterParams+tidal]", direction="params->IC", fn=_bcfp_tidal,
         param="tidal_radius", theta0=1.5, reduce=mean_radius, tol=1e-3),     # MEASURED ..
```

**Step 4:** Update `manifest.py`:
- `SYMBOL_CATEGORY`: add `build_cluster, build_king_cluster, build_eff_cluster,
  build_michie_cluster, build_limepy_cluster, build_cluster_from_params` → `AUDITED`;
  `build_plummer_cluster, matched_velocity_df` → `EXEMPT_HELPER`;
  `RotationSpec, ClusterParams` → `EXEMPT_CONTAINER`.
- `MUST_AUDIT`: add the 10 `(id, param)` keys exactly matching the Cases above, each with a one-line
  rationale.

**Step 5:** Run the ratchet + the gate:
```bash
XLA_FLAGS=... env -u VIRTUAL_ENV uv run --no-sync pytest \
  tests/validation/grad_audit/test_manifest_coverage.py \
  tests/validation/test_grad_audit.py -q -k "cluster or build_ or manifest or symbol or must_audit"
```
Expected: PASS. If any new Case is `hazard`, INVESTIGATE (do not pin/xfail without Anna). Re-run the
FULL `tests/validation/test_grad_audit.py` to confirm no regression in the existing 56 units.

**Step 6:** Commit.

```bash
git add tests/validation/grad_audit/registry.py tests/validation/grad_audit/manifest.py
git commit -m "test(grad-audit): register cluster builders — 10 measured AD-vs-FD cases

build_cluster (r_h/r_a/r_t/omega) + per-family alias cases (King W0, EFF gamma, Michie W0,
LIMEPY W0) + build_cluster_from_params (r_h, tidal_radius). SYMBOL_CATEGORY + MUST_AUDIT
updated (ratchet green); build_plummer_cluster + matched_velocity_df EXEMPT_HELPER (subsumed).

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 5 → Anna ===`

---

## Batch 6: Integration tests

### Task 6.1: end-to-end equilibrium + modifiers + jit/grad

**Files:** Create: `tests/integration/test_cluster_builders_integration.py`

**Step 1:** Write tests (read an existing `tests/integration/test_end_to_end.py` for the energy-check
idiom first):

```python
import jax, jax.numpy as jnp, pytest
from jaxstro.units import STELLAR
from progenax import (
    build_cluster, build_cluster_from_params, ClusterParams,
    PlummerProfile, EFFProfile, KingProfile, MichieProfile, LIMEPYProfile,
    compute_kinetic_energy, compute_potential_energy,
)

_K = jax.random.PRNGKey(3)


def _Q(ic):
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=STELLAR.G)
    return float(T / jnp.abs(V))


@pytest.mark.parametrize("profile", [
    PlummerProfile(r_h=1.0),
    EFFProfile(a=1.0, gamma=5.0, r_t=12.0),       # gamma=5 ~ virial
    KingProfile.from_W0_rc(W0=7.0, r_c=1.0),
    MichieProfile.from_W0_rc(W0=7.0, r_c=1.0, r_a=8.0),
    LIMEPYProfile.from_W0_rc(W0=5.0, g=1.0, r_c=1.0),
])
def test_each_profile_builds_near_virial(profile):
    ic = build_cluster(profile, masses=jnp.ones(3000), key=_K)   # Q=0.5 applied by build_spatial_ic
    assert _Q(ic) == pytest.approx(0.5, abs=0.05)
    assert jnp.all(jnp.isfinite(ic.positions)) and jnp.all(jnp.isfinite(ic.velocities))


def test_jit_through_build_cluster():
    @jax.jit
    def f(r_h):
        return build_cluster(PlummerProfile(r_h=r_h), masses=jnp.ones(200), key=_K).positions
    assert jnp.all(jnp.isfinite(f(1.0)))


def test_grad_through_build_cluster_from_params():
    m = jnp.ones(200)
    def loss(r_h):
        params = ClusterParams(profile=PlummerProfile(r_h=r_h), tidal_radius=3.0, rotation=0.1)
        ic = build_cluster_from_params(params, masses=m, key=_K)
        return jnp.mean(jnp.linalg.norm(ic.positions, axis=1))
    g = jax.grad(loss)(1.0)
    assert jnp.isfinite(g) and abs(g) > 1e-6
```

**Step 2:** Run `tests/integration/test_cluster_builders_integration.py` → iterate to GREEN (these
are slower — EFF/King/Michie/LIMEPY equilibria). If a profile's unscaled Q is off, that is a property
of the DF, not build_cluster — but build_cluster applies Q=0.5 virial scaling by default, so Q≈0.5 is
expected for all. Mark the slow ones `@pytest.mark.slow` if needed (read how the repo marks slow tests).

**Step 3:** Commit.

```bash
git add tests/integration/test_cluster_builders_integration.py
git commit -m "test(integration): build_cluster end-to-end — 5 profiles near-virial + jit/grad

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 6 → Anna ===`

---

## Batch 7: Validation script + publication plots

### Task 7.1: `scripts/validate_cluster_builders.py`

**Files:** Create: `scripts/validate_cluster_builders.py`; plots → `validation/plots/cluster_builders/`

Read `scripts/validate_plummer.py` + `scripts/_plotstyle.py` for the `fig_*(output_dir)` + `save_fig`
idiom and the CLI/`__main__` pattern. The script must PRINT expected-vs-measured tables with
absolute/relative error and an overall PASS/FAIL, and save publication-quality figures.

**Required content (functions):**
1. `table_virial_per_alias()` — for each of the 5 aliases, build N=5000, print measured Q vs expected
   0.5 with |Δ| and a PASS if `|Q−0.5|<0.03` (unscaled DFs) or `<0.02` (after the default Q=0.5 scale).
2. `fig_density_recovery(output_dir)` — sampled radial density vs the analytic profile for Plummer +
   King + EFF (log-log, residual panel).
3. `fig_tidal_cut(output_dir)` — radial mass profile with/without `tidal_radius`, showing the sharp
   cut at r_t and the ghost population (mass=0 beyond r_t).
4. `fig_rotation_Lz(output_dir)` — L_z(ω) linearity for solid-body rotation + a v_φ(R) curve for
   differential; annotate measured slope vs the analytic Σm R².
5. `fig_anisotropy_beta(output_dir)` — β(r) for an OM Plummer vs the analytic r²/(r²+r_a²).
6. `main()` — run all, print the summary table, save all figs, exit nonzero on any FAIL.

**Step 1–N:** Implement, then run:
`XLA_FLAGS=... env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_cluster_builders.py`
Expected: prints PASS for every row; writes ≥5 PNGs to `validation/plots/cluster_builders/`.

**Commit:**
```bash
git add scripts/validate_cluster_builders.py validation/plots/cluster_builders/
git commit -m "validation(cluster-builders): Q/density/tidal/rotation/anisotropy + pub plots

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 7 → Anna ===`

---

## Batch 8: Versatility demos

### Task 8.1: `scripts/demo_cluster_builders.py`

**Files:** Create: `scripts/demo_cluster_builders.py`. Read `scripts/demo_tidal_radius.py` +
`scripts/demo_rotation.py` + `scripts/_demo_inference.py` (`fisher_information_gn`) for the idiom +
run-record convention.

**Required demos (each prints a physical readout + PASS; gate the whole file):**
1. **Onboarding one-liner:** `build_plummer_cluster(n=1000, r_h=1.0, key=...)` → print N, Q, r_h.
2. **Differentiable θ→ICResult inference:** Fisher on `r_h`, `r_a`, `r_t`, `ω` via
   `build_cluster_from_params` + `fisher_information_gn` (the B-series pattern, now one-call). Print
   the Fisher info / σ(θ) for each knob.
3. **All 5 profiles via the generic engine:** loop, print Q + 10–90% Lagrangian radii.
4. **Each modifier with a physical readout:** β(r) (anisotropy), tidal cut fraction (tidal), L_z
   (rotation).
5. **Generative vs inference paths:** `n+imf` (generative) vs `masses=` (fixed-data) — print the
   mass-function summary for the generative draw and confirm the `masses=` path reproduces the input.

**Step 1–N:** Implement, then run:
`XLA_FLAGS=... env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_cluster_builders.py`
Expected: every demo prints PASS; capture the run record (stdout) into the completion doc.

**Commit:**
```bash
git add scripts/demo_cluster_builders.py
git commit -m "demo(cluster-builders): onboarding + differentiable inference + all profiles/modifiers

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 8 → Anna ===`

---

## Batch 9: Completion doc + full gate + STATUS

### Task 9.1: Full local gate

**Step 1:** Run the FULL released-core gate (CLAUDE.md FULL GATE):
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: all green (≈1243 prior + the new builder/integration/grad-audit tests). Record the count.

### Task 9.2: Completion doc + STATUS

**Files:** Create: `.claude-work/TASK_build_cluster_COMPLETE.md`; Modify: `STATUS.md`

**Step 2:** Write `.claude-work/TASK_build_cluster_COMPLETE.md` per CLAUDE.md "Definition of Complete":
implementation details (files/API), the matched_velocity_df mapping, scientific validation results
(the expected-vs-measured tables + plot references from Batch 7), the demo run records (Batch 8), test
results summary (counts per tier + the 10 grad-audit cases with measured AD/FD/ratio), lessons learned,
and the integration plan (repoint the Bucket-A phantom doc references to the real builder).

**Step 3:** Update `STATUS.md` (`next:`/`blocker:`/`due:`), then `brain "cluster-builder API landed:
build_cluster + 5 aliases + matched_velocity_df + ClusterParams; N tests green; grad-gate green"`.

**Step 4:** Commit.

```bash
git add .claude-work/TASK_build_cluster_COMPLETE.md STATUS.md
git commit -m "docs(cluster-builders): completion doc + STATUS — task complete, all gates green

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

`=== CHECKPOINT 9 → Anna: review, then decide PR/merge (no push without explicit go) ===`

---

## Verification matrix (Definition of Complete)

| Requirement | Batch | Artifact |
|---|---|---|
| Comprehensive unit tests | 1–4 | `tests/unit/builders/test_cluster_builders.py` (~35 tests) |
| Integration tests | 6 | `tests/integration/test_cluster_builders_integration.py` |
| Differentiability + grad-gate | 5 | 10 registry Cases + manifest, ratchet green |
| Validation script | 7 | `scripts/validate_cluster_builders.py` |
| Publication plots | 7 | `validation/plots/cluster_builders/*.png` (≥5) |
| Quantitative results | 7 | expected-vs-measured tables (PASS/FAIL) |
| Versatility demos | 8 | `scripts/demo_cluster_builders.py` |
| Completion doc | 9 | `.claude-work/TASK_build_cluster_COMPLETE.md` |

## Risks / watch-items
- **LIMEPY `is_aniso` flag name** (Task 1.2 executor note) — verify before relying on it.
- **Alias ODE-domain forwarding** (Task 5.2 `_bk_W0`) — King's W0-traced grad needs an explicit
  `xi_max/n_ode_points`; confirm the alias forwards them or audit `r_c` (concrete W0) instead.
- **`revirialize` survivor energy** (Task 3.1) — recompute T/|V| on the `masses>0` subset; the ghosts
  must not enter the virial scale (they don't — mass-weighted), but verify Q lands at 0.5.
- **Validation script location** — `scripts/` (repo convention) vs `validation/` (kickoff wording);
  confirm at plan review.
- **Slow integration tests** — EFF/King/Michie/LIMEPY equilibria; mark `@pytest.mark.slow` to keep the
  FAST GATE under budget.
