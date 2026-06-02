# Progenax Pre-Launch Hardening — TDD Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Each batch ends at a **human approval gate** (`research-workflow:high-impact-checkpoint`) — stop and report; do not cross a gate without Anna's OK. Never weaken a test to make it pass (`superpowers:verification-before-completion`).

**Goal:** Close every Critical/Major bug from the 2026-06-01 audit, upgrade the King/EFF velocity DFs to true equilibria, remove confirmed-stale code, refresh all docs/Claude files, and install a reusable HITL guardrail layer in the `research-workflow` plugin.

**Architecture:** Two repos. (1) `~/projects/claude-plugins/research-workflow` — add 4 enforcement hooks (cross-project HITL). (2) `~/projects/jaxstro-dev/progenax` worktree `hardening/audit-2026-06` — six TDD batches. Each fix is RED (failing test reproducing the audit finding) → GREEN (minimal fix) → commit.

**Tech Stack:** JAX (float64) + Equinox + diffrax + jaxtyping; pytest; `uv`; MyST (docs); Claude Code plugin hooks (prompt + command).

**⚠️ Hook activation:** Plugin hooks load at session start and cannot hot-swap. The hooks added in Phase A take effect **only after Claude Code is restarted**. For *this* effort, HITL is enforced by the per-batch gates + research-workflow skills; the hooks protect all future sessions.

---

## Operating model (every batch)

- Work in worktree `hardening/audit-2026-06` (already created). Per-batch commits; one PR to `main` at the end.
- TDD: write the failing test first; run it; confirm it fails for the right reason; implement; confirm green; commit.
- Stop at each **GATE** for Anna's review of diff + test evidence before the next batch.
- Stale code: **inventory → Anna confirms each item → hard delete** (Batch 4 only).

---

## Phase A — research-workflow plugin: 4 enforcement hooks

**Location:** `/Users/anna/projects/claude-plugins/research-workflow/` (NOT the progenax worktree). All hook scripts use `${CLAUDE_PLUGIN_ROOT}` for portable paths and are **path-scoped/self-limiting** (exit 0 = allow when the rule doesn't apply, so they stay inert during course work / quick edits).

### Task A0: git-init the plugin (version control the guardrails)
**Files:** `/Users/anna/projects/claude-plugins/research-workflow/`
- `git init`; add a `.gitignore` (ignore `.DS_Store`); commit the current 16-skill state as baseline `chore: baseline before hooks`.

### Task A1: scaffold hooks
**Files:** Create `hooks/hooks.json`; Modify `.claude-plugin/plugin.json` (add `"hooks": "./hooks/hooks.json"` if required by loader — verify against `@plugin-dev:plugin-structure`).
- `hooks/hooks.json` skeleton (plugin wrapper format):
```json
{
  "description": "research-workflow HITL enforcement: test-integrity, deletion, evidence-before-done, provenance",
  "hooks": {
    "PreToolUse": [
      {"matcher": "Edit|Write|MultiEdit", "hooks": [
        {"type": "command", "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/test_integrity.sh", "timeout": 15},
        {"type": "command", "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/provenance.sh", "timeout": 15}
      ]},
      {"matcher": "Bash", "hooks": [
        {"type": "command", "command": "bash ${CLAUDE_PLUGIN_ROOT}/hooks/deletion_gate.sh", "timeout": 10}
      ]}
    ],
    "Stop": [
      {"matcher": "*", "hooks": [
        {"type": "prompt", "prompt": "EVIDENCE-BEFORE-DONE: If the assistant's final turn claims work is 'fixed', 'passing', 'verified', 'complete', or 'done', verify the recent transcript contains a real command execution (pytest/validation/build) whose output supports the claim. If such a claim lacks fresh supporting evidence in the transcript, return decision 'block' with a reason instructing it to run the verification command and show output. Otherwise 'approve'. Be lenient for purely conversational/planning turns that make no completion claim."}
      ]}
    ]
  }
}
```

### Task A2: deletion gate (deterministic command hook) — implements "confirm before destructive ops"
**Files:** Create `hooks/deletion_gate.sh`; Test: `hooks/tests/test_deletion_gate.sh`
- **Step 1 (RED):** test feeds `{"tool_name":"Bash","tool_input":{"command":"rm -rf src/foo"}}`; expects exit 2 / `permissionDecision: ask` (or deny). Run `bash hooks/tests/test_deletion_gate.sh` → FAIL (script absent).
- **Step 3 (GREEN):** script reads stdin JSON (`jq -r '.tool_input.command'`), matches `\brm\b`, `git\s+rm`, `git\s+clean`, `>\s*\S` truncation, shred; if matched emit `{"hookSpecificOutput":{"permissionDecision":"ask"},"systemMessage":"Destructive op — confirm (Batch-4 stale-deletion gate)."}` and exit 0; else exit 0 (allow). Non-destructive commands pass untouched (self-limiting).
- **Step 5:** commit `feat(hooks): add deletion gate`.

### Task A3: test-integrity gate (command hook, path-scoped) — "never weaken tests to pass"
**Files:** Create `hooks/test_integrity.sh`; Test: `hooks/tests/test_test_integrity.sh`
- Self-limit: read `.tool_input.file_path`; if it does NOT match `test_*.py` or `/tests/`, exit 0 immediately (inert elsewhere — protects course work).
- For test files: inspect the edit (`.tool_input.old_string`/`.new_string` or content). Flag if the change **raises** an `rtol=`/`atol=` number, deletes an `assert`, or adds `@pytest.mark.skip`/`xfail`. On flag → `permissionDecision: ask` with a reason naming the suspected loosening. (Use a conservative regex; false-positives only cost a confirmation.)
- **Step 1 (RED):** test feeds an Edit to `tests/x_test.py` changing `rtol=0.05`→`rtol=0.5`; expect `ask`. RED → GREEN → commit.

### Task A4: provenance gate (command hook, path-scoped) — "constants need citations"
**Files:** Create `hooks/provenance.sh`; Test: `hooks/tests/test_provenance.sh`
- Self-limit: only fires when `.tool_input.file_path` matches a coefficient/constant pattern (`*/imf/environment.py`, `*/defaults.py`, `*calibration*.py`, `*constants*.py`).
- On such edits, require the new content to contain a citation token (`doi`, `arXiv`, `Table`, `Eq.`, `et al.`, a 4-digit year). If absent → `permissionDecision: ask` with reason "constant edit without a source citation." RED → GREEN → commit.

### Task A5: validate, document, GATE
- Run `${CLAUDE_PLUGIN_ROOT}/../../.claude/.../scripts/validate-hook-schema.sh hooks/hooks.json` (or `jq .` sanity) and `scripts/test-hook.sh`.
- Update plugin `README.md`: document each hook, its self-limiting scope, and the **restart-to-activate** requirement.
- Commit `docs(hooks): document HITL hooks`.
- **GATE A:** Anna reviews hooks; decide whether to restart Claude Code now (activate hooks for the rest of the effort) or after. *(Recommended: restart after Phase A so the hooks guard Batches 1–5.)*

---

## Phase B — progenax hardening (worktree `hardening/audit-2026-06`)

### Batch 0 — env + un-break + CI

**Task B0.1 — worktree env.** `uv venv` in the worktree; `uv pip install -e /Users/anna/projects/jaxstro-dev/jaxstro -e . pytest pytest-cov equinox diffrax`. Verify `python -c "import progenax"`.

**Task B0.2 — fix the non-collecting suite (M4).**
- Files: Modify `tests/integration/test_knobs_pipeline.py:13`.
- RED: `pytest tests/ -q` → ERROR at collection (`No module named 'progenax.profiles.mass_segregation'`).
- Fix: `from progenax.cluster.mass_segregation import apply_mass_segregation_baumgardt` (verify symbol exists; if renamed, use the real one).
- GREEN: `pytest tests/ -q` collects and runs (~812 passing). Commit `fix(tests): correct mass_segregation import so the suite collects`.

**Task B0.3 — CI + markers + coverage.**
- Create `.github/workflows/tests.yml` (uv install jaxstro+progenax editable; `pytest tests/ -m "not slow" -q`; upload coverage).
- Modify `pyproject.toml`: register `markers = ["slow", "validation"]`; add `[tool.coverage.run] source=["src/progenax"]`.
- GREEN: `pytest -m "not slow"` runs without the unknown-mark warning. Commit `ci: add test workflow, pytest markers, coverage config`.

**GATE 0:** green baseline in CI; hooks active (post-restart). Anna OK → Batch 1.

### Batch 1 — Critical + cheap bugs (strict TDD, one commit per fix)

**Task B1.1 — C1: thread `G` into velocity sampling.**
- Test: `tests/integration/test_units_through_pipeline.py` (new).
- RED:
```python
def test_build_spatial_ic_respects_G_without_virial_rescale():
    import jax, jax.numpy as jnp
    from jaxstro.units import PLANETARY
    from progenax import (PlummerProfile, PlummerVelocityDF, build_spatial_ic,
                          compute_kinetic_energy, compute_potential_energy)
    m = jnp.ones(800); key = jax.random.PRNGKey(0)
    ic = build_spatial_ic(PlummerProfile(1.0), m, PlummerVelocityDF(1.0),
                          key=key, G=PLANETARY.G, Q=None)
    T = compute_kinetic_energy(ic.velocities, m)
    V = compute_potential_energy(ic.positions, m, G=PLANETARY.G)
    assert abs(float(T/jnp.abs(V)) - 0.5) < 0.05   # was ~5.5e-5 before fix
```
- Run → FAIL (Q≈5.5e-5). Fix `src/progenax/builders.py:249` → `sample_velocities(positions, masses, key_vel, G=G)`. Run → PASS. Commit `fix(builders): pass G into velocity sampling (was silently STELLAR)`.

**Task B1.2 — C2: King K-function NaN gradient + differentiable constructor.**
- Test: `tests/unit/profiles/test_king_grad.py` (new).
- RED:
```python
def test_king_K_gradient_finite_at_zero():
    import jax
    from progenax.profiles.king import king_K_function
    assert jax.numpy.isfinite(jax.grad(king_K_function)(0.0))
def test_from_W0_rc_is_jittable():
    import jax
    from progenax.profiles.king import KingProfile
    jax.jit(lambda w: KingProfile.from_W0_rc(w, 1.0).r_t)(7.0)  # must not raise
```
- Run → FAIL (nan; ConcretizationTypeError). Fix `king_K_function` with the double-`where`/safe-`W_pos` pattern; drop `float(...)` in `_find_tidal_radius` (return the array); correct the `solve_king_profile` docstring (`n_points` is static, not the JIT blocker). Run → PASS. Commit `fix(king): finite K-function gradient and JIT/grad-safe constructor`.

**Task B1.3 — M5: cumulative trapezoid in EFF & King CDFs.**
- Test: `tests/unit/profiles/test_cdf_quadrature.py` (new) — sampled total mass vs a fine `jnp.trapezoid` reference, assert relative error < 1e-4 (left-Riemann gives ~6e-3).
- Fix `src/progenax/profiles/eff.py:~87` and `src/progenax/profiles/king.py:~344`:
  `M_cum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5*(integrand[1:]+integrand[:-1]))*dr])`.
- Commit `fix(profiles): use cumulative trapezoid for EFF/King mass CDF`.

**Task B1.4 — minors (one commit each):**
- `DensityField3D` `dV`: `fdf_density.py:~525,687` → `dx = 2*L_box/(Nx-1)`; test `∫ρ dV ≈ 1` within 1e-3.
- PN11 route: in `generate_fractal_ic_density` dispatch route `mode=="pn11"` like `"gravoturbulent"` (extract `s_t`); test that a PN11 env runs without `ValueError`.
- Docstrings (guarded): pin `DifferentiableBinaryFraction.from_moe2017()` constants in a test (`a≈-0.2799,b≈1.417,c≈0.4755`), then correct the stale `moe2017()` docstring (`differentiable_binary.py:150`) and the Chabrier "system IMF" → "single-star IMF" labels (`chabrier.py` module docstring + attribute comments + `_lognormal_pdf_unnorm`).

**GATE 1:** re-run the audit verification battery (`/tmp` battery → port into `tests/`); every Critical/Major-bug closed. Anna OK → Batch 2.

### Batch 2 — DF fidelity (King + EFF true equilibria) — `research-workflow:numerical-method-validation`

**Task B2.1 — King true lowered-Maxwellian DF.**
- File: rewrite `src/progenax/kinematics/king_df.py` (reuse `KingProfile.psi_grid`/`xi_grid`).
- Spec: at radius r, W(r)=ψ(r) (dimensionless, from ODE interp); speed pdf `g(v) ∝ v²[exp(W − v²/2σ²) − 1]` on `v∈[0, σ√(2W)]`; sample via fixed-grid tabulated inverse-CDF (differentiable, no `while_loop`); isotropic direction. Velocity scale σ from the self-consistent King model (M, r_c, ρ₀) — **coupled to Task B2.3**.
- Acceptance tests (`tests/validation/test_king_physics.py`): (a) σ_v(r) profile matches King-model moments within ~10%; (b) **virial Q from the sampled DF (no external rescale) = 0.5 ± 0.05**; (c) all particles bound (v<v_esc); (d) `jax.grad` through sampling is finite.

**Task B2.2 — EFF Eddington-inversion DF.**
- File: rewrite `src/progenax/kinematics/eff_df.py`. Tabulate `f(E)` via the Eddington formula from EFF ρ(Ψ); sample speed at r from `g(v)∝v² f(ψ(r)−v²/2)`.
- Acceptance: f(E)≥0 (assert physical), bound, σ_v(r) match, Q=0.5±0.05 unscaled, differentiable.

**Task B2.3 — M6: King nondimensionalization decision (`provenance-of-constants` + `decision-log-and-commits`).**
- Compute `c(W0)=log₁₀(r_t/r_c)` for W0∈{1,3,5,7,9}; compare to the King (1966) / BT08 table.
- If the missing factor-of-9 (`king.py:130` RHS `−ρ̃` vs standard `−9ρ̃`) is the cause: restore `−9*rho_tilde` (and center case), re-derive σ; else document the scaling convention. **Write the decision to `docs/notes/`; Anna signs off.**
- Acceptance: `c(W0)` matches the King table (or a documented, justified convention).

**GATE 2:** validation numbers/plots reviewed (dispersion profiles, unscaled Q, c(W0) table). Anna OK → Batch 3.

### Batch 3 — testing robustness
Each is RED-first (assert the stronger condition, watch it fail if applicable) → commit:
- Tighten `tests/conftest.py:86` `VIRIAL_RATIO` 0.20→0.05; fix any test that legitimately needs it.
- Tighten exactly-constructed `rtol≈0.3` checks → 0.05 (`test_fractal.py:354`, `test_cluster_ic.py:370`).
- Add **SanaOBPeriod −0.55 log-slope** test (`tests/unit/binaries/test_population.py`, N=5e4, slope −0.55±0.08).
- Add **ThermalEccentricity** canonical `⟨e⟩=2/3±0.01` (e_max=1).
- Add conservation tests (energy/momentum/COM, explicit tolerances) and **differentiability tests** for King/EFF DFs, binary period/eccentricity distributions, PN11 pipeline.
**GATE 3:** green CI + coverage report.

### Batch 4 — stale code (hard-delete on confirm)
- Build an **evidence-backed inventory** (grep ref-counts, `__init__` export status, test usage) for: `cluster/fractal_gw_legacy.py` (+ exported `generate_fractal_positions`), the `pn11_legacy` mode, any dead helpers. Present to Anna.
- **Not deletions:** `FDF_STUB` calibration is *in use* → instead warn at the `generate_cluster_ic` call site and surface `version` (M9).
- Hard-delete only per-item-approved entries (deletion-gate hook will prompt). Migrate `generate_fractal_positions` callers to the FDF path first if removing it.
**GATE 4:** per-item approval.

### Batch 5 — docs & Claude files
- `progenax/CLAUDE.md` + workspace `jaxstro-dev/CLAUDE.md`: correct LOC/test counts (~18,600 / 812), current module map, DF-fidelity status, no-`while_loop`/units reminders.
- `AGENTS.md`, `README.md`: refresh counts + quickstart.
- MyST: `10-theory/velocity-dfs/*` (King/EFF now true DFs), `imfs/classic.md` (Chabrier single-star label), getting-started counts, `30-api/*`. `myst build` clean.
- Update `docs/website/90-development-log/code-reviews.md` status lines → "resolved" per fixed item.
- Open `docs/notes/2026-06-02-bm19-field-tail-ticket.md` (ticket: correlated-field dense-tail undersampling; resolution + spatial-correlation research).
**GATE 5:** docs build clean + review.

### Final — verification gate & PR (`research-workflow:correct-cutover`)
- Full suite green in CI; validation scripts pass; provenance table + decision log committed; re-run the audit battery as acceptance tests.
- One PR `hardening/audit-2026-06 → main`; Anna merges.

---

## Verification (global)
- Per fix: named RED test fails before, passes after (shown in transcript — satisfies the evidence-before-done hook).
- Aggregate: `pytest tests/` collects + passes (≥ 812 + new), coverage reported.
- Science: King/EFF Q≈0.5 **unscaled**; σ_v(r) profiles match; King c(W0) matches table or documented.
- Docs: `myst build` succeeds; counts/labels correct; BM19 ticket filed.

## Out of scope (ticketed)
- BM19 turbulent-field dense-tail undersampling (research sub-effort).
- Global personal `~/.claude/CLAUDE.md`.
