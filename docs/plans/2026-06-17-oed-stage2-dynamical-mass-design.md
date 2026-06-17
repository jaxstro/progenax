# OED demo — Stage 2 (M_dyn via the magnitude-limit design knob) + OED section refactor — ratified design

**Date:** 2026-06-17
**Status:** RATIFIED (Anna HITL, brainstorming complete).
**Branch:** `feat/oed-stage2-dynamical-mass` (off `main`, which carries Stage 1 at `9294700`).
**Parent designs:** `docs/plans/2026-06-15-oed-dispersion-arc-design.md` (the arc),
`docs/plans/2026-06-16-oed-demo-stage1-design.md` (Stage 1).

This arc does two things: **(A)** refactor the single Stage-1 page into a proper **OED section**
(shared theory separated from per-example demos), and **(B)** build **Stage 2**, which promotes the
magnitude limit from Stage-1's *fixed completeness* to an optimisable **design knob**, headlining the
**dynamical mass** `M_dyn`.

Stage 3 (multi-epoch astrometry + explicit cost budget) remains deferred to its own later brainstorm.

## Honored decisions (do not re-litigate)

- ADRs 0001–0011 (c-optimality; two-phase; additive Fisher backbone; dimensionless ln-θ metric; …).
- The headline forward model stays **OM-Plummer** `project_dispersion` (clean grads). Stage 2 adds the
  magnitude-selection layer *around* it, not a new forward model.

## (A) The OED section refactor

The current `docs/website/60-science-demos/optimal-design.md` (Stage 1, on merged `main`) becomes a
subdirectory section `60-science-demos/optimal-design/`:

| Page | Content |
|------|---------|
| `index.md` | **What OED is** (the telescope-time hook) + the **"What OED can do with progenax" capability map** + an index table of the worked examples. Section-level, forward-looking. |
| `background.md` | **The shared formalism, once:** Fisher information & Cramér–Rao, the additive design-linear backbone `F=Σ n·c·M`, the dimensionless ln-θ metric (ADR 0011), the c/D/A criteria, and the B&M82 σ_r-vs-v_los projection geometry. Every example `{ref}`s this. |
| `anisotropy.md` | **Stage 1 (r_a)** — the application: the mock, "PMs to the outskirts", its 5 figures, its caveats. Theory delegated to `background.md`. (URL `.../optimal-design/anisotropy`; distinct from the top-level `anisotropy.md` = B6 *recovery* demo, cross-linked.) |
| `dynamical-mass.md` | **Stage 2 (M_dyn)** — new (this arc). |

`myst.yml` nests these under one **"Optimal experimental design"** parent; `index.md` (science-demos)
gets the section row. **Content relocates — nothing is lost**; `myst build` 0 content warnings is the
gate. The Stage-1 figures and run-record move under the section's `figures/`.

## (B) Stage 2 — the magnitude limit as a design knob

### What it measures

The headline is the **dynamical mass** `M` (the Plummer total mass, which enters every dispersion
through `GM`). r_a stays in the joint Fisher (θ = (r_a, M, r_h); M promoted from nuisance to
co-target). The question: **how deep must a survey go to weigh the cluster to a given precision, and
where is depth wasted?**

### The new design variable

The limiting magnitude **`m_lim`** gates *which stars exist to be observed*. A star of mass `m` at
distance `d` has apparent magnitude

```
m_app(m) = M_abs(m) + 5 log10(d / 10 pc),   M_abs from zams_luminosity (Tout+1996)
```

and is detectable iff `m_app ≤ m_lim`, i.e. mass `m ≥ m_min(m_lim, d)` (via the inverse ZAMS
relation). `m_lim` is added to the design vector and optimised.

### Three couplings depth introduces (new vs Stage 1)

1. **IMF → counts.** Detectable stars per radial bin = the **Chabrier** IMF integrated above
   `m_min` (`1 − cdf(m_min)`) × the projected density. Deeper `m_lim` → lower `m_min` → steeply more
   stars, *especially in the star-starved outskirts*.
2. **Magnitude → error.** Faint stars are noisy: a photon-noise-like scaling
   `ε(m_app) = ε₀ · 10^{0.2 (m_app − m_ref)}` raises the per-star RV and PM errors toward the faint
   end — the diminishing return on depth.
3. **Effective per-cell quantities.** Per (bin `b`, channel `c`): an IMF-weighted **effective error**
   `ε_eff,b,c(m_lim)` and an **availability weight** `avail_{b,c}(m_lim)` (how many stars depth
   unlocks there).

### The additive backbone survives (the key technical point)

The predicted dispersions `σ_pred(r)` are a property of the *potential* — **independent of `m_lim`**
(single-population, mass-follows-light: every star traces the same σ(r)). So the Jacobian
`J_{b,c} = ∂σ_pred/∂ln θ` is **still computed once**. `m_lim` enters only through the smooth scalars
above:

```
M_{b,c}(m_lim) = 2 J_{b,c} J_{b,c}^T / (σ²_{b,c} + ε_eff,{b,c}²(m_lim))
F(design)      = Σ_{b,c} n_{b,c} · avail_{b,c}(m_lim) · M_{b,c}(m_lim)
```

`J` once; `ε_eff(m_lim)` and `avail(m_lim)` are cheap differentiable IMF/ZAMS integrals. The
forward-mode ban never bites; the design is `[z (36 logits), m_lim (1 bounded scalar)]`, optimised
jointly with Adam (`m_lim` via `expit` into a magnitude range).

### Why this gives a genuine *interior* optimum — with no cost model

Hold the measured-star budget `N_total` fixed (as in Stage 1). Depth is then a real trade:
- **Too shallow** — only bright stars detectable, and bright stars are *scarce in the outskirts*, so
  you cannot place tracers where M_dyn's radial leverage lives.
- **Too deep** — you reach the outskirts, but those faint outer stars are so noisy (large ε) they
  barely constrain anything.

The optimiser finds the `m_lim` that best weighs the cluster: **σ(M_dyn)/M_dyn minimised at an
interior depth.** (Stage 3's explicit exposure cost will sharpen this; here the optimum comes from
the finite bright-star supply × faint-star noise, no money.)

## Shared helpers (DRY, Anna-requested)

Following the `_demo_inference.py` / `_demo_binaries.py` precedent, the **reusable selection/photometry
physics** is factored into a new shared helper, **not** buried in OED code:

```
scripts/_demo_selection.py   # NEW shared: apparent_mag(mass,d,Z), m_min(m_lim,d,Z) [inverse ZAMS],
                             #   photon_noise_error(m_app,...), detectable_counts(m_lim, imf, ...)
                             #   reusable by Stage 2 OED + B4 (binary mass fn) + B5 (IMF) + future
scripts/_demo_oed.py         # the shared OED CORE (Stage 1); small generalization so per_star_blocks /
                             #   fisher accept per-cell errors ε_eff (3,K) + availability weights
                             #   (backward-compatible: a (3,) eps still broadcasts). J still once.
```

`project_to_sky` consolidation (B8 reuse) is an optional, low-priority follow-up — not in this arc.

## File layout

```
scripts/_demo_selection.py                 # NEW shared selection/photometry helper
scripts/_demo_oed.py                       # MODIFY: per-cell ε + availability generalization
scripts/demo_oed_dynamical_mass.py         # NEW Stage-2 gated CLI (composes _demo_oed + _demo_selection)
tests/unit/test_demo_selection.py          # NEW (selection physics: mag, m_min round-trip, counts, error)
tests/unit/test_demo_oed_depth.py          # NEW (depth Fisher, interior optimum, AD-vs-FD on [z,m_lim], calibration)
docs/website/60-science-demos/optimal-design/{index,background,anisotropy,dynamical-mass}.md
docs/website/60-science-demos/optimal-design/figures/   # Stage-1 figures move here + Stage-2 figures
docs/website/myst.yml                      # nest the section
docs/website/60-science-demos/index.md     # section row
.claude-work/OED_DEMO_STAGE2_COMPLETE.md   # completion doc
```

## Figures (Stage 2)

1. **Headline** — `σ(M_dyn)/M_dyn` vs `m_lim`, the interior minimum, optimal depth annotated.
2. The depth trade decomposed — detectable counts vs `m_lim` (IMF, rising) and per-star error vs
   `m_lim` (rising) → their product, the information curve.
3. Optimal radial × channel allocation *at* the optimal depth (vs Stage-1's no-depth allocation).
4. `σ(M_dyn)` vs star budget at the optimal depth (the M_dyn frontier).
5. Calibration — realized `σ(M_dyn)` vs Fisher under a **magnitude-selected** mock draw.

## Gates (Definition of Complete)

1. `σ(M_dyn)/M_dyn` exhibits an **interior optimum** in `m_lim` — report optimal `m_lim` + σ(M)
   achieved vs too-shallow / too-deep.
2. **AD-vs-FD** on `∂(criterion)/∂[z, m_lim]` — the new `m_lim` gradient is the load-bearing
   differentiability check.
3. **Calibration** — realized `σ(M_dyn)` ≈ Fisher under a magnitude-selected mock ensemble (MC-error
   tolerance, principled like Stage 1's `2√(2/n)`).
4. Selection-helper unit tests (mag round-trip, counts monotonic in `m_lim`, error scaling).
5. Tests green; CLI exit 0 + run-record; **the whole OED section** `myst build` 0 warnings; completion
   doc; STATUS + brain.

## Scope / non-goals (Stage 2)

- Single-population, mass-follows-light (σ_pred is m_lim-independent — the backbone survives). Mass
  segregation / multi-mass kinematics are out of scope.
- No extinction; no crowding; photon-noise error model is illustrative, not a real survey ETC.
- Depth is a single global `m_lim` (not per-region). Explicit exposure **cost** and **epochs** are
  Stage 3.
- Chabrier IMF fixed (not itself a target here).

## Sequencing & HITL

1. `writing-plans` → bite-sized TDD plan (`docs/plans/2026-06-17-oed-stage2-dynamical-mass-plan.md`).
2. Subagent-driven TDD, independent code review per task + a final whole-arc review.
3. Full released-core gate green locally; CLI exit 0; section `myst build` 0 warnings.
4. Completion doc; STATUS + brain.
5. **Anna merge-go** → local `main` → push on her word → delete branch.

Anna approves every design decision, plan, and merge. Verify **locally**; nothing pushed/merged
without explicit go. Commit per task; messages end with the `Co-Authored-By` trailer.
