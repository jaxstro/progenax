# Batch 0 review — foundations (`protocols` · `dynamics` · `tidal`) — gate packet

**Date:** 2026-06-03 · **Branch:** `sota/batch0-foundations` · **Base:** `main @ 58254b2` ·
**Reviewer:** Claude Opus 4.8 · **Engine:** manual (tiny foundational modules — no panel) ·
**Status:** 🚦 **AT GATE 1** (findings + fix plan below; no code changed yet)

## Scope

| File | LOC | Public surface |
|------|-----|----------------|
| [protocols.py](../../src/progenax/protocols.py) | 123 | `SpatialProfile`, `VelocityDF`, `IMFProtocol` (runtime-checkable) |
| [dynamics/virial.py](../../src/progenax/dynamics/virial.py) | 98 | `compute_kinetic_energy`, `compute_potential_energy`, `compute_virial_ratio`, `rescale_velocities_to_virial` |
| [tidal.py](../../src/progenax/tidal.py) | 150 | `jacobi_radius`, `jacobi_radius_isothermal`, `apply_tidal_truncation`, `fill_factor_to_r_h` |

Baseline: `tests/unit/dynamics/test_virial.py` + `tests/unit/test_tidal.py` = **13 passed**.
No `test_protocols.py` exists.

## Validation evidence

### FD-vs-autodiff grad-checks (all public differentiable entry points) — ✅ PASS

| Entry point (scalar-parameterized) | autodiff | FD | rel-err |
|---|---|---|---|
| `compute_kinetic_energy` d/ds[s·v] | 2.603063e+01 | 2.603063e+01 | 1.3e-12 |
| `compute_potential_energy(soft=0.05)` d/ds[s·x] | 1.431411e-01 | 1.431411e-01 | 8.5e-11 |
| `compute_virial_ratio` d/ds[s·v] | 1.815319e+02 | 1.815319e+02 | 9.0e-13 |
| `jacobi_radius` d/dM | 8.582128e-04 | 8.582128e-04 | 1.9e-09 |
| `jacobi_radius_isothermal` d/dM | 1.032883e-03 | 1.032883e-03 | 1.9e-09 |
| `fill_factor_to_r_h` d/dff | 1.000000e+01 | 1.000000e+01 | 1.0e-12 |

### Softening=0 full-Jacobian grad (the suspected path) — ❌ FINDING F1

| `compute_potential_energy` source | soft=0 grad | soft=0.05 grad |
|---|---|---|
| **`builders.py` (PUBLIC, top-level export)** | **finite** ✅ (double-where) | finite |
| **`dynamics/virial.py` (LEAF, used by `cluster/`+`kinematics/api.py`)** | **NaN, 36/36** ❌ | finite |

The audit's softening=0 grad fix lives at [builders.py:133-137](../../src/progenax/builders.py#L133)
(double-where: safe diagonal before `sqrt`, then `inf` so the `i<j` sum drops it). It was
**never ported** to `dynamics/virial.py`, which still does `r2 + soft**2` then `r + 1e-10`.

### Protocol conformance (runtime `isinstance`) — ✅ PASS

`PlummerProfile`→`SpatialProfile` ✓; `PlummerVelocityDF`→`VelocityDF` ✓;
`Maschberger`, `ChabrierIMF`, `PowerLawIMF`, `TaperedPowerLaw`, `Schechter` → `IMFProtocol` ✓.
The protocols are sound and satisfied — they are simply **untested** (F6).

### Unit consistency — ❌ FINDING F3

`jacobi_radius_isothermal` docstring says `V_circ` is "km/s", but `G` is pc³·M⊙⁻¹·Myr⁻². With
`V=220` as-documented vs. the pc/Myr-consistent value, `r_J` = **30.99 vs 30.53 pc → 1.5%
silent bias**. Formula `r_J=(G·M·R²/2V²)^{1/3}` is itself correct (BT2008 §8.3.1).

### Constant / formula provenance — ✅ (one units caveat, F3)

| Quantity | Code | Source | Status |
|---|---|---|---|
| Jacobi radius `R(M_c/3M_g)^{1/3}` | tidal.py:47 | King (1962) AJ 67,471; BT2008 Eq 8.91 | ✓ |
| Isothermal Jacobi `(G M R²/2V²)^{1/3}` | tidal.py:78 | BT2008 §8.3.1 | ✓ formula; ✗ units (F3) |
| `PE = -G Σ_{i<j} m_i m_j/r_ij` | virial.py:42 | standard; G present ✓ | ✓ |
| `Q = T/\|V\|`, equilibrium 0.5 | virial.py | virial theorem 2T+V=0 | ✓ |

## Findings ledger

| id | finding | sev | proposed disposition |
|----|---------|-----|----------------------|
| **F1** | `dynamics/virial.compute_potential_energy` NaN grad at softening=0 | **Major** | fix-now, RED first |
| **F2** | Duplicated energy/virial physics: `builders.py` (public, safe) vs `dynamics/virial.py` (leaf, buggy) — two sources of truth | **Major** | **decision needed** (consolidate now vs fix-in-place + ticket) |
| **F3** | `jacobi_radius_isothermal` km/s vs pc/Myr → 1.5% bias | **Major** | fix-now (docstring→pc/Myr + fix test) — confirm approach |
| **F4** | `apply_tidal_truncation` boolean-mask → dynamic shape, not jit/vmap-able, non-diff mask | **Major (design)** | docstring "host-side" note now + **ticket** redesign (confirm no jitted caller) |
| **F5** | `test_virial.py` `test_formula_is_T_over_V` / `test_virial_equilibrium_is_half` test inline arithmetic, never call the module (mutation-insensitive) | Minor | fix-now: redirect at `compute_virial_ratio` |
| **F6** | No `test_protocols.py` — composition protocols have zero conformance tests | Minor | fix-now: add conformance test (cheap guard) |
| **F7** | `compute_kinetic_energy`/`compute_potential_energy` have no direct tests; no grad-checks in Batch 0 | Minor | fix-now: add direct + grad-check tests (incl. F1 regression) |
| **F8** | protocols.py:19 docstring "Used by build_ic()" — actual fn is `build_spatial_ic` | Minor (trivial) | fix-now |
| **F9** | `compute_virial_ratio`/`rescale_velocities_to_virial` not top-level; top-level virial fn is `builders.virial_scale`; IMF top-level-export claims in CLAUDE.md/API don't match (IMFs live in `progenax.imf`) | Minor | reconcile in F2 + WS2 docs |
| **F10** | `dynamics` PE adds unphysical `+1e-10` floor to `r` (vs builders' clean inf-diagonal); ~1e-11 value drift | Minor | resolved by F2 consolidation |

## Proposed fix plan (Gate 1 — pending approval, RED-first)

1. **F1+F2+F10 — consolidate energy/virial to one canonical leaf.** Make `dynamics/virial.py`
   the single source of truth and gradient-safe (port the double-where; drop the `+1e-10`
   floor). Have `builders.py` **re-export** `compute_kinetic_energy`/`compute_potential_energy`
   from `dynamics` (delete its duplicate bodies); keep top-level exports stable.
   *RED:* `jax.grad(dynamics.compute_potential_energy, soft=0)` finite; identity test that
   `builders.compute_potential_energy is dynamics.compute_potential_energy`.
   *Crosses into `builders.py` (Batch 7)* — needs your explicit go (decision D1 below).
2. **F3 — isothermal Jacobi units.** Docstring: `V_circ` must be in the same length/time
   units as `G` (pc/Myr for STELLAR); fix `test_tidal.py::test_formula` to pass a consistent
   value. *RED:* assert `r_J` equals the analytic value with consistent units.
3. **F4 — tidal truncation.** Add a docstring warning it is host-side (not jit/vmap-able);
   confirm no jitted/grad caller; **ticket** a shape-preserving (mask/where) redesign.
4. **F5+F6+F7 — test hardening.** Redirect the two tautological virial tests at the real
   functions; add direct `compute_kinetic_energy`/`compute_potential_energy` tests + grad-checks
   (incl. the softening=0 regression for F1); add `tests/unit/test_protocols.py` conformance.
   *Safety rails:* `--cov=progenax` not reduced; each new/kept test shown RED-sensitive.
5. **F8 — trivial docstring** fix.

### Decisions requested at Gate 1

- **D1 (F2):** consolidate energy/virial **now** (touches `builders.py`, a Batch-7 module, but
  is the SoTA-correct single-source-of-truth move) — **or** minimal fix-in-place in `dynamics`
  now + ticket the consolidation for Batch 7?
- **D2 (F3):** fix by **docstring + test** (require pc/Myr) — or add an internal km/s→pc/Myr
  conversion (heavier, changes the call contract)?
- **D3 (F4):** docstring note + ticket now — or redesign the truncation in this batch?

## Resolution (Gate 1 decisions D1–D3 applied)

| id | resolution |
|----|------------|
| F1 | **Fixed.** `dynamics/virial.compute_potential_energy` ported to the double-where guard; soft=0 grad now finite. |
| F2 | **Consolidated (D1).** `dynamics/virial.py` is the single canonical energy source; `builders.py` re-exports `compute_kinetic_energy`/`compute_potential_energy` (duplicate bodies deleted). Top-level API unchanged; `top is dynamics`, `builders is dynamics`. |
| F3 | **Fixed (D2).** `jacobi_radius_isothermal` docstring requires `V_circ` in pc/Myr (G's units); test pins the defining relation `r_J³ = G M R²/2V²`. |
| F4 | **Redesigned (D3, hybrid).** Zero-mass hard cut (shape-preserving, jit/vmap-safe) + `custom_jvp` logistic straight-through surrogate (width `0.05·r_t`); `r_t` now differentiable (grad finite, >0). |
| F5 | **Fixed.** Tautological virial tests redirected at `compute_virial_ratio`. |
| F6 | **Fixed.** `tests/unit/test_protocols.py` conformance added (profiles, DFs, 5 IMFs conform; protocols reject non-conformers). |
| F7 | **Fixed.** Direct `compute_kinetic_energy`/`compute_potential_energy` tests + FD-vs-autodiff grad-checks added. |
| F8 | **Fixed.** protocols.py docstring `build_ic` → `build_spatial_ic`. |
| F10 | **Fixed.** Unphysical `+1e-10` floor removed (resolved by F2 consolidation). |
| F9 | Deferred to WS2 (doc reconciliation of count/naming claims). |

## Verification (Gate 2 evidence)

- **Batch 0 subset:** `pytest tests/unit/dynamics/ tests/unit/test_tidal.py tests/unit/test_protocols.py`
  → **42 passed** (was 13).
- **Full suite (regression):** `pytest tests/` → **979 passed, 0 failed** (exit 0; 30 warnings; 197 s).
  Consolidation touched broadly-imported `builders.py`/`dynamics/virial.py` — no regression.
- **Key grad evidence (now passing tests):**
  - `dynamics.compute_potential_energy` soft=0 grad **finite** (F1 regression test).
  - `builders.compute_potential_energy is dynamics.compute_potential_energy` (F2 single source).
  - tidal `∂(retained mass)/∂r_t` **finite, >0**; surrogate matches `σ(1−σ)/w` (F4).
  - FD-vs-autodiff KE/PE rel-err ≤ 1e-5 (F7).
- **Coverage rail:** Batch 0 added tests only increase coverage of `protocols`/`dynamics`/`tidal`
  (new `test_protocols.py`, grad/energy tests); no test removed except *redirecting* two
  tautological asserts at the real function (coverage of `compute_virial_ratio` strictly up).

## Gate status

- **G1 — findings + fix plan:** ✅ approved (D1 consolidate, D2 docstring+test, D3 hybrid).
- **G2 — diff + evidence before commit:** 🚦 **awaiting Anna** (this packet).
- **G3 — branch + green CI before PR/merge:** ☐
