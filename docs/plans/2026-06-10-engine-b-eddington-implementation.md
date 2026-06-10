# Engine B (density-defined Eddington) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (or
> superpowers:subagent-driven-development) to implement this plan task-by-task.

**Goal:** `MultiComponentCluster.from_density_profiles(...)` — Plummer/EFF/King-density
components in ONE shared self-consistent potential, each component's DF by Eddington
inversion in the shared Ψ (+ optional per-component Osipkov-Merritt r_a_j), sampled to
`ICResult` with no external virial rescale.

**Architecture:** No ODE — the total density is prescribed, so the shared potential is one
cumulative-trapezoid pass over Σ_j (M_j/M_tot)·ρ̂_j(r). The new physics surface is (i) a
generic `eddington_invert` extracted from `_eff_eddington_table` and (ii) the f_j ≥ 0
realizability gate. Engine B reuses the existing position-CDF sampler fields verbatim
(`_r_grid`, `_cdf_j`, `N_frac_j`); only construction and the speed/direction stage dispatch
on a static `engine` field. Design: `docs/plans/2026-06-10-engine-b-eddington-design.md`
(all decisions locked with Anna — read it first).

**Tech Stack:** JAX (jax.numpy only in src/), Equinox, pytest(+xdist). float64 automatic.

---

## Context for an engineer with zero progenax background

- **Repo:** `~/projects/jaxstro-dev/progenax`, branch `feat/multimass-limepy-equilibrium`.
  Do NOT push. Commit after each task with the message given in the task.
- **Run everything:** prefix with `env -u VIRTUAL_ENV uv run --no-sync `. Parallel test
  runs add `XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"`
  and `-n auto`.
- **Rules (CLAUDE.md, non-negotiable):** jax.numpy only in src/ (no numpy/scipy); no
  Python data loops in hot paths (a Python loop over the SMALL STATIC component count is
  fine — Engine A precedent); everything differentiable; functions ≤100 LOC, files ≤500
  LOC; **never weaken a test to make it pass** — if a physics gate fails, the
  implementation (or grid resolution) is wrong, not the tolerance.
- **READ before coding:**
  - `src/progenax/kinematics/eff_df.py` — `_eff_eddington_table` (lines 40–108) is the
    function being split. Lines 48–61 build the EFF density (stays in eff_df); lines
    62–108 are the generic potential+inversion (move out, in two pieces: the potential
    integral belongs to the new density_poisson module; the inversion to eddington.py).
  - `src/progenax/kinematics/eddington.py` — `sample_speed_from_f_table`,
    `assign_om_directions` (reused as-is; one extension in Task 6).
  - `src/progenax/cluster/multicomponent.py` — `MultiComponentCluster` fields +
    `__init__` (the A-assembler), `_sample_cluster_arrays` (the jitted core),
    `component_virial_ratios` (the exact-quadrature oracle pattern to mirror).
  - `src/progenax/profiles/{plummer,eff,king}.py` — all three have `.density(r)`
    (unnormalized: Plummer (1+r²/a²)^{-5/2}, EFF (1+r²/a²)^{-γ/2}, King ρ̂(ψ(r))/ρ̂(W0)).
    Extents: `PlummerProfile` infinite; `EFFProfile.r_t`; `KingProfile.r_t`
    (natural, smooth edge). `KingProfile.from_W0_rc(W0, r_c)` is the convenience
    constructor.
- **Key physics identities used by tests:**
  - Isolated Plummer relative potential: Ψ(r) = GM/√(r²+a²) − GM/√(r_t²+a²).
  - Plummer enclosed mass: M(<r)/M = x³/(1+x²)^{3/2}, x = r/a.
  - Isotropic Plummer ergodic DF: f(E) ∝ E^{7/2} (BT2008 Eq. 4.83) — exact for the
    untruncated model; with r_t = 100a it holds to high accuracy at interior energies.
  - Per-component virial in a SHARED potential: W_j = −∫ ρ_j (dΦ/dr) r dV (Clausius);
    Q_j = T_j/|W_j| = 0.5 in equilibrium.

---

## Task 1 (phase 2a): extract `eddington_invert` — EFF bit-identical

**Files:**
- Modify: `src/progenax/kinematics/eddington.py` (add `eddington_invert`)
- Modify: `src/progenax/kinematics/eff_df.py` (`_eff_eddington_table` becomes a caller)
- Test: `tests/unit/kinematics/test_eddington_invert.py` (new)

**Step 1: capture the pre-refactor pins.** Run:

```bash
env -u VIRTUAL_ENV uv run --no-sync python - <<'EOF'
import jax.numpy as jnp
from progenax.kinematics.eff_df import _eff_eddington_table
for tag, ra in (("iso", None), ("om", 2.0)):
    r, Psi, E, f, mu = _eff_eddington_table(1.0, 4.0, 12.0, ra)
    print(tag, float(Psi[0]), float(mu),
          [float(f[i]) for i in (1, 250, 500, 750, 998)])
EOF
```

Paste the printed values into the regression test below as `PINS` (full repr precision).

**Step 2: write the failing tests**

```python
# tests/unit/kinematics/test_eddington_invert.py
"""Generic Eddington inversion: extraction regression + Plummer analytic oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Pre-refactor pins from _eff_eddington_table (captured at Task 1 Step 1).
PINS = {...}  # paste: {"iso": (Psi0, mu, [f1, f250, f500, f750, f998]), "om": (...)}


class TestExtractionRegression:
    @pytest.mark.parametrize("tag,ra", [("iso", None), ("om", 2.0)])
    def test_eff_table_bit_identical_after_refactor(self, tag, ra):
        """The refactored _eff_eddington_table (now calling eddington_invert)
        reproduces the pre-refactor values EXACTLY (same grids, same ops)."""
        from progenax.kinematics.eff_df import _eff_eddington_table
        r, Psi, E, f, mu = _eff_eddington_table(1.0, 4.0, 12.0, ra)
        Psi0, mu_pin, f_pins = PINS[tag]
        assert float(Psi[0]) == Psi0 and float(mu) == mu_pin
        for idx, pin in zip((1, 250, 500, 750, 998), f_pins):
            assert float(f[idx]) == pin, f"f[{idx}] drifted"


class TestPlummerAnalyticOracle:
    def test_isotropic_plummer_f_propto_E_3p5(self):
        """The strongest truth test: feed the inverter the ANALYTIC Plummer
        (rho, Psi, dPsi/dr) on a grid with r_t = 100 a and check
        f(E)/f(E_ref) == (E/E_ref)^{3.5} to rtol 1e-3 on interior energies
        E in [0.1, 0.8] Psi0. Bypasses all of our own potential numerics."""
        from progenax.kinematics.eddington import eddington_invert

        a, rt = 1.0, 100.0
        r = jnp.linspace(1e-5, rt, 20000)
        rho = (1.0 + (r / a) ** 2) ** (-2.5)
        drho = -5.0 * (r / a**2) * (1.0 + (r / a) ** 2) ** (-3.5)
        # G=1, total mass of the (untruncated) Plummer with rho_0=1:
        # M = (4/3) pi a^3 rho_0 * ... -> use the exact M(<r) instead:
        x = r / a
        Mr = (4.0 * jnp.pi * a**3 / 3.0) * x**3 / (1.0 + x**2) ** 1.5
        Psi = Mr[-1] / jnp.sqrt(rt**2 + a**2) * 0.0  # placeholder, see below
        Phi = -(4.0 * jnp.pi * a**3 / 3.0) / jnp.sqrt(r**2 + a**2) * 1.0
        # Exact untruncated Plummer: Phi = -G M_tot / sqrt(r^2+a^2),
        # M_tot = (4/3) pi a^3 (rho_0=1, but our rho above is rho/rho_0 with
        # the (3M/4pi a^3) prefactor absorbed -> M_tot = pi a^3 * (4/3) * 1? )
        # SIMPLEST CORRECT ROUTE (do this): M_tot = lim x->inf of Mr formula
        # with the SAME unnormalized rho used above:
        #   rho = (1+x^2)^{-5/2}  ->  M(<r) = (4 pi a^3/3) x^3 (1+x^2)^{-3/2}.
        M_tot = (4.0 * jnp.pi * a**3 / 3.0)
        Phi = -M_tot / jnp.sqrt(r**2 + a**2)
        Psi = Phi[-1] - Phi
        dPsi_dr = -Mr / r**2
        E_grid, f_grid = eddington_invert(r, rho, drho, Psi, dPsi_dr)

        Psi0 = float(Psi[0])
        sel = (np.asarray(E_grid) > 0.1 * Psi0) & (np.asarray(E_grid) < 0.8 * Psi0)
        E = np.asarray(E_grid)[sel]
        f = np.asarray(f_grid)[sel]
        i_ref = len(E) // 2
        np.testing.assert_allclose(
            f / f[i_ref], (E / E[i_ref]) ** 3.5, rtol=1e-3,
            err_msg="inverter does not reproduce the Plummer E^{7/2} law")

    def test_om_reduces_to_iso_at_infinite_ra(self):
        """r_a = inf augmentation weight is exactly 1 -> identical tables."""
        from progenax.kinematics.eddington import eddington_invert
        r = jnp.linspace(1e-5, 30.0, 4000)
        rho = (1.0 + r**2) ** (-2.5)
        drho = -5.0 * r * (1.0 + r**2) ** (-3.5)
        Mr = (4.0 * jnp.pi / 3.0) * r**3 / (1.0 + r**2) ** 1.5
        Phi = -(4.0 * jnp.pi / 3.0) / jnp.sqrt(r**2 + 1.0)
        Psi = Phi[-1] - Phi
        dPsi = -Mr / r**2
        E1, f1 = eddington_invert(r, rho, drho, Psi, dPsi, r_a=None)
        E2, f2 = eddington_invert(r, rho, drho, Psi, dPsi, r_a=jnp.inf)
        np.testing.assert_array_equal(np.asarray(f1), np.asarray(f2))
```

(Clean up the Plummer-oracle test body when writing it — the comment-trail above
documents the mass normalization derivation; keep only the final correct lines.)

**Step 3: run, verify fail** (`ImportError: cannot import name 'eddington_invert'`).

**Step 4: implement.**

In `eddington.py` add (signature is the contract — keep it):

```python
def eddington_invert(r_grid, rho_grid, drho_dr_grid, Psi_grid, dPsi_dr_grid,
                     r_a=None, n_e: int = 1000, n_u: int = 2000):
    """Generic (dimensionless) Eddington inversion in a given relative potential.

    Returns (E_grid, f_grid): the isotropic ergodic DF of rho in Psi, or with
    r_a set the Osipkov-Merritt f(Q) via the augmented density
    rho_Q = (1 + r^2/r_a^2) rho (Merritt 1985; r_a=inf or None -> isotropic).
    Raw (unclamped) f: callers detect genuine negativity; the speed sampler
    clamps grid-level ringing at use. Extracted VERBATIM from the validated
    _eff_eddington_table (Phase 2a) -- the r->0 double-where dPsi guard and the
    u = sqrt(E - Psi) substitution are gradient-safety load-bearing.
    """
```

Body = `_eff_eddington_table` lines 52–107 with: the OM augmentation applied to the
*passed* rho/drho (handle `r_a=None` OR `r_a=inf` → weight 1 via
`jnp.where(jnp.isfinite(r_a), ...)` when traced; plain `if r_a is None` for the None
case); `drho_dPsi` built from the PASSED `dPsi_dr_grid` with the same double-where +
`.at[0].set(neighbor)` guard; same `E_grid = linspace(1e-4 Psi0, 0.999 Psi0, n_e)`;
same `f_one` with the `u`-substitution and boundary term. **Copy, don't rewrite** —
every guard exists because a test failed without it.

Refactor `_eff_eddington_table` to: build r/rho/drho (lines 48–61 unchanged), compute
Phi/Psi/mu/dPsi_dr (lines 62–77 unchanged), then
`E_grid, f_grid = eddington_invert(r, rho, drho, Psi, dPsi_dr, r_a)` and return the
same 5-tuple. Delete the now-duplicated inversion lines from eff_df.py.

**Step 5: run** the new tests + the full EFF + kinematics file:

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit/kinematics/test_eddington_invert.py tests/unit/kinematics/ tests/validation/test_eff_physics.py -q
```

Expected: all pass, INCLUDING the bit-identical pins. If a pin fails, the refactor
changed an operation — fix the refactor, never the pin.

**Step 6: commit**

```bash
git add src/progenax/kinematics/ tests/unit/kinematics/test_eddington_invert.py
git commit -m "refactor(kinematics): extract generic eddington_invert; EFF tables bit-identical; Plummer E^{7/2} analytic oracle (Engine B 2a)"
```

---

## Task 2 (phase 2b): `density_poisson` — shared Ψ + derived domain

**Files:**
- Create: `src/progenax/profiles/density_poisson.py`
- Test: `tests/unit/profiles/test_density_poisson.py` (new)

**Step 1: write the failing tests** (complete file):

```python
# tests/unit/profiles/test_density_poisson.py
"""Prescribed-density shared potential + derived domain (Engine B, design c)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from progenax import PlummerProfile, EFFProfile, KingProfile


class TestComponentExtent:
    def test_plummer_is_infinite(self):
        from progenax.profiles.density_poisson import component_extent
        assert component_extent(PlummerProfile(r_h=1.0)) is None

    def test_eff_and_king_finite(self):
        from progenax.profiles.density_poisson import component_extent
        assert float(component_extent(EFFProfile(a=1.0, gamma=4.0, r_t=8.0))) == 8.0
        k = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        assert float(component_extent(k)) == pytest.approx(float(k.r_t))


class TestDeriveRt:
    def test_max_of_finite_extents(self):
        from progenax.profiles.density_poisson import derive_r_t
        rt, prov = derive_r_t([PlummerProfile(2.0), EFFProfile(a=0.3, gamma=5.0, r_t=8.0)],
                              jnp.array([0.7, 0.3]))
        assert float(rt) == 8.0 and "EFF" in prov

    def test_all_infinite_uses_f_enc_mass_radius(self):
        """Pure-Plummer mix: r_t = radius enclosing f_enc of the SUMMED mass.
        Single Plummer analytic check: M(<r)/M = x^3/(1+x^2)^{3/2} = f_enc."""
        from progenax.profiles.density_poisson import derive_r_t
        p = PlummerProfile(r_h=1.0)
        rt, prov = derive_r_t([p], jnp.array([1.0]), f_enc=0.995)
        c = 0.995 ** (2.0 / 3.0)
        x_exact = float(jnp.sqrt(c / (1.0 - c)))          # ~17.27
        assert float(rt) == pytest.approx(float(p.a) * x_exact, rel=2e-2), prov

    def test_explicit_override_wins(self):
        from progenax.profiles.density_poisson import derive_r_t
        rt, prov = derive_r_t([PlummerProfile(1.0)], jnp.array([1.0]), r_t=30.0)
        assert float(rt) == 30.0 and "override" in prov

    def test_king_conflict_with_override_raises(self):
        from progenax.profiles.density_poisson import derive_r_t
        k = KingProfile.from_W0_rc(W0=5.0, r_c=1.0)
        with pytest.raises(ValueError, match="King"):
            derive_r_t([k], jnp.array([1.0]), r_t=0.5 * float(k.r_t))


class TestSharedPotential:
    def test_single_plummer_matches_analytic(self):
        """Psi from the quadrature pass == GM/sqrt(r^2+a^2) - GM/sqrt(rt^2+a^2)
        (G=1 internal units) to rtol 1e-4 over the interior."""
        from progenax.profiles.density_poisson import shared_potential
        p = PlummerProfile(r_h=1.0)
        pot = shared_potential([p], jnp.array([1.0]), r_t=jnp.asarray(40.0))
        a = float(p.a)
        M = float(pot.M_cum_j[0, -1])                      # total (dimensionless) mass
        Psi_exact = M / jnp.sqrt(pot.r_grid**2 + a**2) - M / jnp.sqrt(40.0**2 + a**2)
        sel = np.asarray(pot.r_grid) > 0.05
        np.testing.assert_allclose(np.asarray(pot.Psi_grid)[sel],
                                   np.asarray(Psi_exact)[sel], rtol=1e-4)

    def test_two_components_mass_fractions_respected(self):
        """M_j(r_t) proportions == mass_fractions (each component's CDF is
        normalized; the FRACTIONS set the amplitudes)."""
        from progenax.profiles.density_poisson import shared_potential
        pot = shared_potential([PlummerProfile(2.0), PlummerProfile(0.5)],
                               jnp.array([0.7, 0.3]), r_t=jnp.asarray(60.0))
        Mj = np.asarray(pot.M_cum_j[:, -1])
        np.testing.assert_allclose(Mj / Mj.sum(), [0.7, 0.3], atol=5e-3)

    def test_truncated_mass_fraction_diagnostic(self):
        """Plummer truncated at 5a stores M(<rt)/M(inf) = x^3/(1+x^2)^{3/2}."""
        from progenax.profiles.density_poisson import shared_potential
        p = PlummerProfile(r_h=1.0)
        rt = 5.0 * float(p.a)
        pot = shared_potential([p], jnp.array([1.0]), r_t=jnp.asarray(rt))
        x = 5.0
        expect = x**3 / (1 + x**2) ** 1.5
        assert float(pot.trunc_frac_j[0]) == pytest.approx(expect, rel=1e-3)

    def test_mass_fractions_must_sum_to_one(self):
        from progenax.profiles.density_poisson import shared_potential
        with pytest.raises(ValueError, match="mass_fractions"):
            shared_potential([PlummerProfile(1.0), PlummerProfile(2.0)],
                             jnp.array([0.6, 0.6]), r_t=jnp.asarray(20.0))
```

**Step 2: run, verify fail** (ModuleNotFoundError).

**Step 3: implement `density_poisson.py`.** Structure (≤300 LOC):

- `component_extent(profile) -> Array | None`: isinstance dispatch (Plummer → None,
  EFF/King → their `r_t`). Raise TypeError for unknown profiles, naming the supported
  set.
- `derive_r_t(profiles, mass_fractions, r_t=None, f_enc=0.995) -> (Array, str)`:
  explicit override wins, BUT a King component whose natural r_t exceeds the override
  raises (no silent re-truncation; the design's edge-conflict rule). Else max of
  finite extents (provenance string names the profile class). Else (all infinite)
  bisection on the summed analytic enclosed mass for the f_enc radius — fixed 80-step
  `jax.lax.scan` bisection on [0, 1e4·max(scale radii)] (differentiable enough; the
  domain choice itself is a construction-time scalar).
- `shared_potential(profiles, mass_fractions, r_t, n_r=6000) -> SharedPotential`
  (a small eqx.Module): validate Σ mass_fractions (|Σ−1| > 1e-8 → ValueError);
  r_grid = linspace(1e-5·r_t/6000... use `jnp.linspace(1e-5, r_t, n_r)` exactly like
  the EFF table); per-component ρ̂_j via `_density_and_derivative(profile, r)`
  (analytic dρ/dr for Plummer/EFF; King via dρ̂/dW·dW/dr with dW/dr from the
  interpolated psi_grid gradient — numerical `jnp.gradient` on the grid is acceptable
  for King, NOT for Plummer/EFF where closed forms exist)
  [AMENDED 2026-06-10: proven wrong in Task 5 — jnp.gradient of interpolated psi
  produces a d²ρ/dΨ² staircase (f_min −0.679); replaced by the Poisson-identity
  dψ/dξ = −9ξ⁻²∫ρ̃s²ds quadrature in dccedbe. Never differentiate interpolated
  data.]; normalize each ρ̂_j so its
  truncated mass is 1, scale by mass_fractions → ρ_j; sum; one cumtrap pass (copy the
  `inner/tail/outer` pattern from `_eff_eddington_table` lines 62–71) → Φ, Ψ,
  dΨ/dr = −M(<r)/r², per-component M_cum_j, μ = Σ_j ∫ρ_j r² dr, and trunc_frac_j
  (analytic M(∞) for Plummer; EFF γ>3 analytic, γ≤3 → trunc_frac = 0.0 with the
  divergence documented in the field docstring — EFF always has finite r_t so this is
  a *diagnostic*, never a domain input).
- `SharedPotential(eqx.Module)` fields: `r_grid, Psi_grid, dPsi_dr_grid, rho_j_grid,
  M_cum_j, mu, trunc_frac_j, r_t` + static `r_t_provenance: str`.

**Step 4: run** the new file + profile suites:
`... pytest tests/unit/profiles/test_density_poisson.py tests/unit/profiles/ -q`

**Step 5: commit**

```bash
git commit -m "feat(profiles): prescribed-density shared potential + derived domain with provenance (Engine B 2b)"
```

---

## Task 3 (phase 2c-i): `from_density_profiles` constructor

**Files:**
- Create: `src/progenax/cluster/eddington_engine.py` (`_EngineBState` + builder)
- Modify: `src/progenax/cluster/multicomponent.py` (engine field + B branch)
- Test: extend `tests/unit/cluster/test_multicomponent.py` (new class `TestEngineB`)

**Step 1: failing tests** (append to test_multicomponent.py):

```python
class TestEngineB:
    def _model(self, **kw):
        from progenax.cluster.multicomponent import MultiComponentCluster
        defaults_ = dict(
            profiles=[PlummerProfile(r_h=2.0), EFFProfile(a=0.4, gamma=5.0, r_t=9.0)],
            mass_fractions=jnp.array([0.6, 0.4]), m_j=jnp.array([0.5, 1.0]))
        defaults_.update(kw)
        return MultiComponentCluster.from_density_profiles(**defaults_)

    def test_constructs_and_reports_domain(self):
        m = self._model()
        assert m.engine == "B"
        assert float(m.r_t) == 9.0                       # EFF extent wins (design c)
        assert "EFF" in m.engine_b.r_t_provenance
        assert bool(jnp.all(jnp.isfinite(m.engine_b.f_j_grid)))

    def test_f_min_diagnostic_stored_and_benign_here(self):
        m = self._model()
        fmin = np.asarray(m.engine_b.f_min_j)
        assert fmin.shape == (2,)
        assert np.all(fmin > -1e-3)                      # realizable mix (relative units)

    def test_a_fields_are_nan_tripwires(self):
        """Engine-A-only fields are NaN in B mode: accidental A-path use must
        poison results visibly, never silently."""
        m = self._model()
        for name in ("W0", "g", "mu_tot"):
            assert bool(jnp.isnan(getattr(m, name)))
        assert bool(jnp.all(jnp.isnan(m.alpha_j))) and bool(jnp.all(jnp.isnan(m.w_j)))

    def test_position_cdf_matches_component_masses(self):
        """_cdf_j is reused verbatim by the sampler: each row is a normalized
        M_j(<r); the Plummer row must match the analytic CDF."""
        m = self._model(profiles=[PlummerProfile(r_h=1.0)],
                        mass_fractions=jnp.array([1.0]), m_j=jnp.array([1.0]))
        a = float(PlummerProfile(r_h=1.0).a)
        x = np.asarray(m._r_grid) / a
        exact = x**3 / (1 + x**2) ** 1.5
        exact = exact / exact[-1]
        np.testing.assert_allclose(np.asarray(m._cdf_j[0]), exact, atol=2e-3)

    def test_unrealizable_om_raises_with_component_name(self):
        with pytest.raises(ValueError, match="component 0"):
            self._model(r_a_j=jnp.array([0.05, jnp.inf]))   # absurdly radial halo
```

**Step 2: run, verify fail** (`AttributeError: ... no attribute 'from_density_profiles'`).

**Step 3: implement.**

`eddington_engine.py`:

```python
class _EngineBState(eqx.Module):
    """Engine B tables + diagnostics (one field group on MultiComponentCluster)."""
    Psi_grid: Float[Array, "n_r"]          # shared relative potential on _r_grid
    E_grid: Float[Array, "n_e"]
    f_j_grid: Float[Array, "n_comp n_e"]   # per-component Eddington DFs (raw)
    mass_fractions: Float[Array, "n_comp"]
    r_a_j: Float[Array, "n_comp"]          # inf = isotropic (Engine A convention)
    mu: Float[Array, ""]                   # velocity-scale integral (EFF kappa pattern)
    f_min_j: Float[Array, "n_comp"]        # min f_j / max |f_j| (realizability margin)
    trunc_frac_j: Float[Array, "n_comp"]
    r_t_provenance: str = eqx.field(static=True)
```

`build_engine_b_state(profiles, mass_fractions, r_a_j, r_t, f_enc, n_r, n_e)`:
derive_r_t → shared_potential → per-component `eddington_invert(r, rho_j, drho_j, Psi,
dPsi_dr, r_a_j[j])` (Python loop over the static component count; one shared E_grid —
all components share Ψ0, so invert with the same `n_e` and assert the E_grids are
identical) → f_min_j = min(f_j)/max|f_j| → the GENUINE-negativity gate: for concrete
inputs (`isinstance` check on the originating Python floats is not possible here — use
`jax.core.is_concrete` equivalent: `not isinstance(f_min_j, jax.core.Tracer)`), raise
ValueError naming `component {j}` and the remedy text from the design doc; traced
builds skip the raise, always store the diagnostic. Returns `(_EngineBState, r_grid,
M_cum_j, m_j-independent N_frac inputs)`.

`multicomponent.py` changes (keep `__init__` ≤100 LOC by extracting, if needed, the
existing A-assembly into `_assemble_engine_a`):
- New fields: `engine: str = eqx.field(static=True)` (default "A") and
  `engine_b: Optional[_EngineBState] = None` (dynamic; None for A models — None is an
  empty pytree subtree, harmless under jit).
- `from_density_profiles(profiles, mass_fractions, m_j, r_a_j=None, r_t=None,
  f_enc=0.995, n_r=6000, n_e=1000, n_grid=1000)`: build the B state; fill SHARED
  fields meaningfully — `_r_grid` (the Poisson r-grid downsampled/interp to n_grid),
  `_cdf_j` (= M_cum_j normalized, interpolated to `_r_grid`), `N_frac_j ∝
  mass_fractions/m_j`, `r_t`, `m_j`, `residual=0` — and A-ONLY fields as NaN
  tripwires: `W0=g=r_c=mu_tot=nan`, `alpha_j=w_j=ra_hat_j=nan` arrays, `xi_grid=
  psi_grid=` NaN arrays of length 2 (shape-minimal).
- All existing A constructors set `engine="A"`, `engine_b=None` (defaults — verify no
  call-site churn).

**Step 4: run** `... pytest tests/unit/cluster/test_multicomponent.py -q -n auto`
(with XLA_FLAGS). ALL existing Engine A tests must pass untouched — the union must be
invisible to A.

**Step 5: commit**

```bash
git commit -m "feat(cluster): MultiComponentCluster.from_density_profiles - Engine B state, derived domain, f_j>=0 gate (2c-i)"
```

---

## Task 4 (phase 2c-ii): Engine B sampling + Q_j oracle

**Files:**
- Modify: `src/progenax/cluster/multicomponent.py` (`_sample_cluster_arrays` B branch;
  `component_virial_ratios` B branch)
- Modify: `src/progenax/kinematics/eddington.py` (`assign_om_directions` accepts a
  per-star r_a ARRAY; scalar/None behavior unchanged — add a regression test for
  scalar == array-of-same-scalar)
- Test: extend `TestEngineB`

**Step 1: failing tests**

```python
    def test_sampled_density_matches_each_component(self):
        """Position pipeline: per-component sampled radial mass CDF matches its
        own M_j(<r) (KS distance < 0.02 at N=20k per component)."""
        # sample, split by ic.component_id, compare ECDF of r to m._cdf_j rows.

    def test_engine_b_global_virial_is_half_unscaled(self):
        """THE headline gate: Plummer halo + EFF core, N=30k, |Q - 0.5| < 0.02
        with NO virial rescale anywhere in the pipeline."""

    def test_engine_b_theory_Qj_is_half(self):
        """component_virial_ratios (B branch, exact quadrature over the DF
        moments) returns Q_j = 0.5 +- 3e-3 for both components."""

    def test_speed_scale_uses_sampled_mass(self):
        """Doubling every stellar mass m_j doubles <v^2> (the Engine A lesson:
        velocity scale from the ACTUAL sampled Sigma m_i)."""
```

(Write these concretely following `TestFromComponents` patterns in the same file —
the acceptance numbers above are the contract.)

**Step 2: run, verify fail.**

**Step 3: implement.**

Sampler B branch in `_sample_cluster_arrays` (static `model.engine` Python branch —
compiled per engine, zero runtime dispatch):
1. component draw + position draw: UNCHANGED code path (`N_frac_j`, `_cdf_j`).
2. Ψ at each star: `jnp.interp(r_i, model._r_grid, engine_b.Psi_grid_on_r)`.
3. dimensionless speed: vmap `sample_speed_from_f_table(key_i, Psi_i, E_grid,
   f_j_grid[c_i])` (gather the star's component row).
4. physical scale: `v = sqrt(G * M_sampled / (4 pi mu)) * s` with
   `M_sampled = jnp.sum(m_i)` (the sampled masses, never an input M_total).
5. directions: `assign_om_directions(key, pos, speeds, r_a_per_star)` where
   `r_a_per_star = engine_b.r_a_j[c_i]` (inf rows → stretch 1 → isotropic; the
   array extension must broadcast `radii / r_a` elementwise — 3-line change).

`component_virial_ratios` B branch (exact quadrature, the oracle — NO reuse of
sampled quantities): for each j,
`T_j = (1/2) ∫ ρ_j(r) ⟨v²⟩_j(r) 4π r² dr` with ⟨v²⟩_j(r) from the DF speed moments
(`∫ s⁴ f_j(Ψ(r) − s²/2) ds / ∫ s² f_j ds`, w-grid quadrature per r — vmap over the
r-grid; for OM components include the augmented-density correction by computing the
moment in the stretched frame exactly as `_sample_speed_angle`'s oracle does), and
`W_j = −∫ ρ_j (dΦ/dr) r 4π r² dr` (Clausius in the shared potential). Q_j = T_j/|W_j|.

**Step 4: run** TestEngineB + the full multicomponent file (A regression) + fast gate.

**Step 5: commit**

```bash
git commit -m "feat(cluster): Engine B sampler (shared-Psi Eddington draws) + exact-quadrature Q_j oracle (2c-ii)"
```

---

## Task 5 (phase 2c-iii): cross-engine + cross-family anchors

**Files:**
- Test: `tests/validation/test_engine_b_physics.py` (new)

**Tests (the contract — write fully, mark the N≥20k ones `@pytest.mark.slow`):**

1. `test_king_density_engine_b_matches_engine_a`: `from_density_profiles([KingProfile
   .from_W0_rc(W0=5, r_c=1)], [1.0], m_j=[1.0])` vs `from_components(alpha_j=[1.0],
   w_j=[1.0], m_j=[1.0], W0=5, g=1, r_c=1)`. Gates: theory Q_j both 0.5±3e-3; sampled
   σ_1d(r) profiles agree to |σ_B/σ_A − 1| < 0.02 in interior bins (N=2e4, same-seed
   binning); sampled ρ(r) shapes agree (KS < 0.02). THE A-vs-B trust anchor.
2. `test_eff_gamma5_single_component_matches_plummer`: Engine B single
   EFF(γ=5, a) ≡ Engine B single Plummer(a math-matched: γ=5 EFF IS Plummer with
   a_Plummer = a_EFF) — same σ(r), same CDF.
3. `test_plummer_halo_eff_core_equilibrium`: the science headline (global Q
   0.5±0.02 unscaled, per-component Q_j 0.5±3e-3 theory; sampled per-component Q_j
   matches the exact-quadrature HYBRID prediction — ρ_presc weights × DF speed
   moments × prescribed-total Clausius field — within 3σ shot noise; the
   hard-truncated halo's plateau below 0.5 (predicted 0.4953, sampled
   0.4947±0.0014 over 18 seeds at N=16k) is verified physics, NOT a convergence
   target of 0.5). (amended 2026-06-10 with Anna's approval — predict-the-offset
   gate replaces "sampled Q_j → 0.5 with N".)
4. `test_om_beta_profile_realized`: r_a_j on the halo only → sampled β_halo(r) ≡
   r²/(r²+r_a²) in resolved bins (tol 0.05, seed-averaged ≥4 seeds), core stays β≈0.

**Run, fix, full-file pass, commit:**

```bash
git commit -m "validation(cluster): Engine B anchors - King A-vs-B, EFF(g=5)=Plummer, halo+core equilibrium, OM beta (2c-iii)"
```

---

## Task 6 (phase 2d): negative tests + gradients

**Files:**
- Test: extend `tests/unit/cluster/test_multicomponent.py` + `tests/validation/test_engine_b_physics.py`

1. `test_unrealizable_mix_raises_naming_component` (already partly in Task 3 via tiny
   r_a; add the message-content assertions: component index, f_min value, remedy text).
2. `test_traced_build_skips_raise_stores_diagnostic`: `jax.jit(lambda ra:
   build...)`-style construction with traced r_a_j does not raise; `f_min_j` is
   populated. (Use `eqx.filter_jit` over a scalar surrogate if full-model jit is
   awkward; the contract is "no raise under tracing".)
3. `test_gradients_ad_vs_fd`: `jax.grad` of a smooth scalar (e.g. theory T_j of
   component 0, or mean Ψ over the grid) w.r.t. (r_h of the halo, mass_fractions[0]
   via a reparametrized scalar, r_a_j[0]) — finite, nonzero, AD-vs-FD rtol 1e-3.
   Follow the FD pattern in `test_limepy_tables.py::test_differentiable_in_g_and_queries`.
4. `test_mass_fraction_sum_raise` + `test_king_override_conflict_raise` (constructor-
   level duplicates of the Task 2 unit gates, through the public API).

**Commit:** `git commit -m "test(cluster): Engine B realizability negatives + AD-vs-FD gradients (2d)"`

---

## Task 7 (phase 2e): validation script + close-out

1. **Write `scripts/validate_multicomponent_eddington.py`** (style: follow
   `scripts/validate_multimass_anisotropy.py` + `scripts/_plotstyle.py`): PASS/FAIL
   table printing (a) King A-vs-B σ(r) max deviation; (b) Plummer f(E) E^{7/2} max
   rel err; (c) halo+core Q, Q_j (theory + sampled ± σ over seeds); (d) sampled
   β_halo(r) vs OM curve; (e) AD-vs-FD gradient rel err; one 4–5 panel figure to
   `validation/plots/engine_b_eddington.{png,pdf}`. Exit 0 only if all PASS.
2. **Run it**: expected `ALL PASS`.
3. **Full gate** (record the count — expect 1064 + new tests):

```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```

4. **Write `.claude-work/PHASE_2_ENGINE_B_COMPLETE.md`** (implementation summary,
   measured anchor numbers, accuracy table, lessons; fill ALL numbers from real runs —
   no placeholders left at commit time, cf. the Phase 1.5 close-out lesson).
5. **Update** STATUS.md (`next:` → Phase 3) + CLAUDE.md (counts, Module Quick
   Reference `cluster/` row mentions Engine B, Public API adds `from_density_profiles`).
6. **Commit:** `git commit -m "docs(status): Phase 2 Engine B done (validation ALL PASS, gate <count>); resume at Phase 3"`. DO NOT PUSH.

---

## Honest scope / risks

- **Realizability is physics:** some prescribed mixes (shallow component in a
  concentrated companion's potential; too-small r_a_j) DO NOT EXIST as equilibria.
  The engine refuses with the named component; tests pin both the refusal and the
  traced-build diagnostic path.
- **Truncated empirical profiles** are approximately stationary at the edge (EFF
  caveat inherited): the halo+core global-Q gate is 0.02, not 0.002, for this reason;
  the per-component THEORY Q_j gate stays tight (3e-3) because the moment quadrature
  is truncation-consistent.
- **One shared E_grid** assumes all components feel the same Ψ0 — true by
  construction in one potential.
- **Speed tables (Phase 1.5-style) deferred**: the quadrature path is the oracle; add
  tables only if profiling shows Engine B sampling matters at N ≥ 1e5.
- If the King A-vs-B anchor disagrees beyond gates, STOP and report — it means one of
  the engines has a physics inconsistency; do not tune tolerances to pass.
