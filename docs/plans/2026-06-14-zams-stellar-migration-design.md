# ZAMS stellar relations — internalize Tout+1996 into progenax (drop the fluxax reach)

**Date:** 2026-06-14
**Status:** Design ratified (brainstorm, Anna HITL — 5 decisions confirmed). Next: TDD implementation plan.
**Branch (impl):** `feat/zams-stellar` (off local `main`).
**Lane:** Release-prep "Lane B" — self-containment + capability (NOT the pip-installability gate, which is the separate jaxstro/Lane-A decision).

## Problem & motivation

The B4 science demo (`scripts/demo_binary_mass_function.py`) reaches across the repo
boundary into the **private, unpublished** `fluxax` sibling for the Tout et al. (1996)
ZAMS mass→luminosity relations (`zams_luminosity`, `inverse_zams_luminosity` from
`fluxax.photometry`). The dependency is already cleanly guarded (one `importorskip` in one
script; fluxax is **not** a declared progenax dependency), but it means B4 cannot run in a
clean environment / CI without a manual `uv pip install -e ../fluxax`.

progenax already *anticipated* internalizing these relations: `compute_stellar_radii`
(`builders.py`) documents itself as a "TEMPORARY single-relation stand-in… REPLACED by the
`startrax` package once it lands: **first Tout et al. (1996) ZAMS radii**…". So bringing the
Tout ZAMS family in-house as a **placeholder-until-startrax** is the code's own documented
roadmap, not a detour.

The Tout relations are *stellar-structure* (M→L,R,T_eff,log g) — really startrax territory
(zero-age subset of the eventual Hurley+2000 tracks), distinct from fluxax's job (photometry:
L→flux→magnitudes/SED/bandpasses). This migration takes **only** the structure subset.

**Out of scope (explicit):** this does **not** make `pip install progenax` work — that is
gated entirely on the `jaxstro` packaging decision (audit R2/D2, Lane A), unrelated to fluxax.

## Ratified decisions (Anna HITL)

1. **Scope = full Tout+1996 ZAMS family**, drop PMS (Ushomirsky 1998 — different regime, no
   current use, YAGNI). Five functions: `zams_luminosity`, `zams_radius`,
   `zams_effective_temperature`, `zams_surface_gravity`, `inverse_zams_luminosity`.
2. **Home = new flat module `src/progenax/stellar.py`**; `zams_*` names lifted into
   `progenax.__all__` (flat, like the rest of the public API). A `stellar` (not `zams`) name
   gives a future home to co-locate `compute_stellar_radii` and pre-startrax relations. A
   subpackage is premature for five functions (YAGNI; the `tidal.py`/`numerics` precedent).
3. **`compute_stellar_radii` relationship = additive + doc-bridge.** Leave it byte-for-byte
   unchanged (still Demircan & Kahraman 1991, still the live N-body collision-radius path —
   the two relations disagree, e.g. R(1 M☉)=1.06 D&K91 vs ~0.89 Tout ZAMS, and unifying would
   change collision physics). Cross-reference both docstrings. Unification onto Tout is a
   future *physics* arc (startrax era), not a rider on this dependency drop.
4. **API style = light refactor to progenax idiom.** Array-aware jaxtyping signatures
   (`Float[Array, "..."]`) that broadcast naturally; the Newton inverse vmaps its scalar core
   internally so it takes an array like its neighbor `compute_stellar_radii(masses)`; drop the
   forced top-level `@jax.jit` (caller-jittable); keep `Z=0.02` solar default (preserves the
   metallicity dependence the CMD science wants). Same physics, same values — native. Pure
   functions (NOT an `eqx.Module` — these are stateless maps, like `compute_stellar_radii`).
5. **Definition-of-Complete = full release-grade treatment** (registry floor is mandatory;
   the standalone validation script + plots + website page + provenance artifact are the
   discretionary headline-model scaffolding, ratified IN).

## §1 — Module & API (`src/progenax/stellar.py`)

| Function | Maps | Units | Source |
|---|---|---|---|
| `zams_luminosity(mass, Z=0.02)` | M→L | L☉ | Tout 1996 Table 1 |
| `zams_radius(mass, Z=0.02)` | M→R | R☉ | Tout 1996 Table 2 |
| `zams_effective_temperature(mass, Z=0.02)` | M→T_eff | K | Stefan-Boltzmann(L,R) |
| `zams_surface_gravity(mass, Z=0.02)` | M→log g | dex (cgs) | g = G M / R² |
| `inverse_zams_luminosity(L, Z=0.02)` | L→M | M☉ | Newton via `lax.scan` |

- Forward functions broadcast over a mass array natively (the Tout rationals are elementwise);
  the inverse `vmap`s its scalar Newton core internally → array-in/array-out.
- Coefficients as **named module-level frozen constants** — `_TOUT_L_COEFFS` (Table 1: 7
  coefficient rows × degree-4 log-Z polynomial) and `_TOUT_R_COEFFS` (Table 2: 9 rows + scalar
  ν). These are what the provenance registry cites.
- CGS constants (`LSUN_ERG_S`, `RSUN_CM`, `SIGMA_SB`, `G_CGS`, `MSUN_G`) from
  `jaxstro.constants` (already a progenax dependency — no new deps).
- Fully differentiable: forward = smooth elementwise; inverse = unrolled-`scan` Newton
  (differentiable, like progenax's other `lax.scan` inverters).
- `compute_stellar_radii` **untouched**; both docstrings get a one-line cross-reference
  (collision-radius D&K91 vs photometric Tout ZAMS).

## §2 — Verification & provenance (paper-grounding first)

The held `Tout1996-ZAMS-Formulae.pdf` is read and Table 1 + Table 2 are verified
**cell-by-cell** against fluxax's transcription (fixing any error found) BEFORE the relations
ship. Output: `docs/core-papers/tout1996_zams_coefficients_verified.md`, mirroring the existing
`startrax_hurley2000_coefficients_verified.md`. The `Tout1996` bibkey already exists
(`references.bib:241`). The port is therefore a **provenance upgrade**, not a blind copy.

Anchors pinned from the PDF: Sun ZAMS (L≈0.69 L☉, R≈0.89 R☉, T_eff≈5670 K), the paper's stated
~5% MS accuracy, monotonicity of L(M)/R(M), and inverse round-trip to machine precision.

## §3 — Registry integration + fluxax decoupling

Five new `__all__` symbols flow through all four registries (first real dogfood of the
test-backbone we just shipped):

| Registry | Action |
|---|---|
| API-coverage | +5 `SYMBOL_TESTS` → asserting tests (112→117) |
| Differentiability | +5 `SYMBOL_CATEGORY`=AUDITED, +5 AD-vs-FD grad-audit cases (the Newton inverse ∂M/∂L is the notable one), regen `grad_audit_results.json` |
| Provenance | `stellar.py` joins the allowlist; `_TOUT_L_COEFFS`/`_TOUT_R_COEFFS` registered → Tout1996 Table 1/2 citations |
| Physics-validation | +5 `EXEMPT_NON_MODEL` (90→95), reason "stellar mass-relation, not an equilibrium model" |
| Dashboard | regen JSON + page; staleness gate. Adding `stellar.py` changes `src/progenax/` → coverage goes stale by the src-based rule → close-out runs the full `--cov` refresh (expected) |

**Decoupling:** `scripts/demo_binary_mass_function.py` swaps the fluxax import for
`from progenax.stellar import inverse_zams_luminosity, zams_luminosity`; the `importorskip`
guard + `uv pip install -e ../fluxax` instructions are deleted → **B4 runs clean / in CI with
no sibling.** Docs updated (binary-mass-function.md, 60-science-demos/index.md,
00-getting-started, the pyproject fluxax comment). Honest ecosystem-boundary note: progenax now
self-contains the ZAMS *structure* relations (startrax placeholder); **fluxax remains the
photometry package** — only the structure subset moved.

## §4 — Testing, DoD artifacts & phase sequence

**Tests.** `tests/unit/stellar/test_zams.py` (broadcasting, grad-finiteness, monotonicity,
Z-direction, inverse round-trip to machine precision, Sun anchor, valid-range) +
`tests/validation/test_zams_physics.py` (vs Tout published values + the ~5% accuracy claim).
Parity is checked against the **PDF-verified values**, not by importing fluxax (keeps the test
suite sibling-free).

**DoD artifacts.** `scripts/validate_zams.py` (+ L-M, R-M, T_eff-M, HR-diagram, inverse-residual
plots), `docs/website/50-validation/zams-relations.md` page, completion doc.

**Phase sequence** (subagent-driven, TDD, independent code-review + Anna HITL per phase; all
local on `feat/zams-stellar`; nothing pushed without Anna):

- **P0** branch off `main`.
- **P1** verify Tout coeffs vs held PDF → `tout1996_zams_coefficients_verified.md`.
- **P2** implement `stellar.py` + unit tests (RED→GREEN TDD).
- **P3** register across the 4 registries + dashboard regen.
- **P4** validation tier: `test_zams_physics.py` + `validate_zams.py` + plots + website page.
- **P5** decouple B4 demo + docs.
- **P6** close-out: full gate + coverage refresh + dashboard fresh + completion doc + STATUS/brain.
  CHECKPOINT → merge → local main on Anna's go.

**Footprint:** 5 public symbols, ~200 LOC in `stellar.py` + tests. **Net release effect:** B4
fully decoupled from fluxax; progenax self-contains verified ZAMS relations. (Restated caveat:
does NOT unblock `pip install progenax` — that's the jaxstro/Lane-A decision.)
