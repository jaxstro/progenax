# Spec — systematic validation audit + methods-paper figures (NEW SESSION)

**Date:** 2026-06-08 · **Branch:** `gravoturb-fdf-sbc-validation` · nothing pushed.
**For:** a fresh session. This is the single entry point. Read it, then work module-by-module.

## Why this exists

The validation/verification surface drifted: stale per-suite pages (claims that no longer match the
code), figures that were committed but not embedded / not build-verified, and broken/stale validate
scripts. Anna needs a **clean, clear, easy-to-follow V&V pipeline she can open in the browser** and
**publication-quality methods-paper figures**. This session does it **systematically, per module**.

## NON-NEGOTIABLE guardrails (trust is the point here)

- **Build-verify every figure.** A figure is not "embedded" until `cd docs/website && myst build --html`
  succeeds and the figure appears in `_build/site/public/` **and** in the page's content JSON. Never
  claim a figure renders without that evidence. (This is the exact mistake that triggered this audit.)
- **Evidence before done.** Every "validated/passing/works" claim must be backed by fresh command
  output shown in the transcript (pytest run, script run, build). No exceptions.
- **Ground papers against held PDFs** in `docs/core-papers/` (no formulas/numbers from memory). Fix
  the per-paper note in `docs/website/99-bibliography/per-paper/` as you go.
- **TDD** for any new/changed code; **released-core test count only grows, never breaks** (currently
  **822**; `pytest tests/unit tests/integration tests/validation -m "not slow"`). Experimental work
  stays under `src/experimental/`.
- **Per-module: brainstorm the plot set first** (superpowers:brainstorming) — don't just replot.
- **HITL:** Anna approves each module's plot set before building it.

## Per-module audit protocol (run for EACH module)

1. **Status check** — does a `tests/validation/test_<module>_physics.py` exist? Run it; record pass/fail.
   Does the released-core suite still pass? Is the validate script healthy (imports resolve)?
2. **Refactor/remove stale** — fix or delete broken/stale validate scripts (see inventory). Update the
   `50-validation/<module>.md` page so every claim matches the code (kill aspirational/“spot values”).
3. **Brainstorm the validation-plot set** (brainstorming skill) — what 1–4 plots prove this module's
   physics? Get Anna's sign-off.
4. **Build pub-quality figures** — a committed `scripts/validate_<module>.py` (or
   `validation/.../*.py`) that prints expected-vs-measured PASS/FAIL **and** writes the figures.
   Curate the verified ones into `docs/website/50-validation/figures/` (the committed exception to the
   `*.png` gitignore — see `.gitignore`).
5. **Embed + build-verify** — `:::{figure} figures/<name>.png` on the `50-validation/<module>.md` page;
   `myst build --html`; confirm the figure is in `_build/site/public/` and the page JSON. Show evidence.
6. **Update the index dashboard** (below).

## Module → page → tests inventory (status as of 2026-06-08)

| Module | 50-validation page | validation test | validate script | Status |
|---|---|---|---|---|
| **profiles/Plummer** | plummer-equilibrium.md | test_plummer_physics.py | validate_profiles.py | ✅ page current + figure embedded+build-verified (this session) |
| **cluster/two-component** | two-component.md | (unit + validate_cluster_ic) | validate_cluster_ic.py | ✅ figure embedded+build-verified |
| **diagnostics/Λ_MSR + cluster segregation** | mass-segregation.md | test_mass_segregation_physics.py (8) | validate_mass_segregation.py, validate_cluster_ic.py | ✅ current + 4 figures build-verified |
| profiles/King | king-profile.md | test_king_physics.py | validate_profiles.py | ⚠️ NOT audited — check page vs code, add/verify figure |
| profiles/EFF | eff-profile.md | test_eff_physics.py | validate_profiles.py | ⚠️ NOT audited |
| kinematics/velocity DFs | (within profile pages) | (in physics tests) | validate_profiles.py (isotropy) | ⚠️ NOT audited — may need its own page/figure |
| imf (Salpeter/Kroupa/Chabrier/Maschberger) | imf-statistics.md | test_imf_physics.py | validate_imfs.py | ⚠️ NOT audited |
| imf/binary + Moe + recovery | binary-imf.md | test_binary_physics.py | validate_binary_aware_recovery.py, validate_hmc_imf_recovery.py | ⚠️ NOT audited |
| imf/environment (IGIMF/EnvironmentIMF) | (no page?) | — | validate_env_imf.py | ⚠️ NOT audited — note IGIMF/EnvironmentIMF were NOT top-level exports (2026-06-08); confirm scope |
| binaries (Kepler, periods, ecc) | (binary-imf / analytical) | test_binary_physics.py | — | ⚠️ NOT audited |
| analytical test cases | analytical-test-cases.md | test_analytical_physics.py | — | ⚠️ NOT audited |
| dynamics (virial/energy) | physics-tests.md | — | — | ⚠️ NOT audited |
| tidal | tidal-truncation.md | (in king tests?) | — | ⚠️ NOT audited |
| diagnostics/CW04 Q + fractal | fractal-substructure.md | (substructure unit) | — | ⚠️ NOT audited — Q diagnostic exists + verified note; FDF generator removed |
| released-core `gravoturb` + PP20 | gravoturbulent-pp20.md | (pp20 suite) | — | ⚠️ NOT audited — note: distinct from experimental gravoturb_fdf |
| populations | two-component.md | test_populations (unit) | — | ⚠️ partial |

**Stale-script note:** `validation/imf/` also has `imf_env_gradient_flow.py`,
`imf_inference_vertical_slice.py`, `imf_vertical_slice.py` — health-check + classify (keep/refactor/remove)
during the imf audit. The 2026-06-08 round-1/2 cleanup already deleted 3 broken scripts and archived 23
banked-2D-β experimental scratch drivers (`src/experimental/gravoturb_fdf/validation/_banked_2d_beta/`).

## Validation index dashboard (`50-validation/index.md`)

Add a **status table** at the top: one row per module — *Validated? (tests pass) · Figure(s) on page? ·
Last verified date · Run command*. This is the "single place Anna opens to see the V&V state."
Keep it updated as each module is audited (✅ / ⚠️ / ❌). The existing "Map of the section" + tiers +
tolerances content stays below it.

## Methods-paper figures (start building)

Design + readiness in `docs/plans/2026-06-08-fdf-methods-paper-and-hardening-design.md`. **Buildable now**
(no gravax): **Figs 1–3** from the experimental `gravoturb_fdf` (hero/pipeline, BM19+β-recovery fidelity,
(m̄,s̄)+knobs) — curate from `gravoturb_fdf/validation/cluster_acceptance.py`. **Figs 4–5** (Λ_MSR(t),
knob→outcome) are DEFERRED to the gravax session (`docs/notes/2026-06-08-gravax-segregation-validation-followup.md`).
Publication styling: consistent fonts/sizes, panel labels, colourblind-safe, vector where possible
(the jaxstroviz port is the eventual home; interim = the validation scripts).

## Already done (don't redo)

- `50-validation/{mass-segregation, plummer-equilibrium, two-component}.md` are **current + figure-bearing
  + build-verified**; 6 curated figures committed under `50-validation/figures/`; the `.gitignore`
  exception for that dir is in place; `MaschbergerClarke2011` added to `references.bib`.
- Λ_MSR diagnostic validated (8 analytic tests); per-paper notes verified for Allison 2009, M&C 2011,
  Cartwright 2004; Tier-C (`correlated_mass_assignment`, M&C local-Σ metric) built + tested.

## Suggested module order (low-risk → high-value)

profiles (King, EFF) → kinematics/velocity DFs → imf (statistics) → analytical → binaries/binary-imf →
dynamics/tidal → diagnostics/fractal-Q → gravoturb/PP20 → then methods Figs 1–3. Update the index
dashboard after each.
