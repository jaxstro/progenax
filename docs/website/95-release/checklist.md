---
title: Release checklist
subtitle: What to do before tagging v0.1.0 — blockers first
description: >-
  Actionable, severity-ordered release checklist for progenax v0.1.0, each item
  tracing to a finding in the release-readiness audit report.
---

(release-checklist)=
# Release checklist

Every item traces to a finding in the [audit report](#release-audit-report). The
ordering is by severity: clear the blockers, then the should-fixes, then the
polish. Items are written so they can be checked off literally.

:::{note} A living checklist — several items are now done
This list began as the read-only audit's to-do list. Since then a number of items
have been actioned — some during the 2026-06-18 fix cycle, more during the v0.1.0
[documentation-hardening pass](#checklist-docs-hardening) — and are marked `[x]`
with a dated note. The genuine remaining blockers (re-enable CI, resolve the
`jaxstro`/PyPI dependency story) are **not** done and stay unchecked. Each fix is a
separate, maintainer-approved step.
:::

(checklist-slice-d)=
## 🚀 Slice D — the pre-flip / pre-tag punch list (maintainer executes)

*Added 2026-07-11 at the close of the pre-public improvement program (Slices A–C).
These are the concrete actions between "repo goes public" and "tag `v0.1.0`", in
order. Every claim below was re-verified against the repo on 2026-07-11.*

**Before or at the public flip:**

- [ ] **Re-enable the 3 GitHub Actions workflows.** All three are
  `disabled_manually` in the repo settings (the YAML itself is current):
  `tests.yml` (PR gate), `physics-validation.yml` (slow lane), `docs.yml`
  (docs gate). Re-enable in *Settings → Actions*, then confirm one green run on a
  real PR before trusting them.
- [ ] **Fix the stale Python-3.10 matrix leg first.** `pyproject.toml` declares
  `requires-python = ">=3.11"`, but `.github/workflows/physics-validation.yml`
  line 62 still runs a `"3.10"` matrix leg (and the comment above it, line 54,
  still says `>=3.10`). Change the matrix to `["3.11", "3.13"]` and fix the
  comment — otherwise the first CI run after re-enable fails on an
  unsupported interpreter, not on physics.
- [ ] **Add `CONTRIBUTING.md`.** Minimum useful content: uv-based setup
  (`uv pip install -e ".[dev]"` with the side-by-side `jaxstro` checkout), the
  two-tier local gate (FAST `-m "not slow"` / FULL), the JAX-native ground rules
  (no numpy/scipy on core paths, differentiability non-negotiable), and that PRs
  must pass `make -C docs/website gate` when they touch docs.

**At the `v0.1.0` tag:**

- [ ] **Add `CITATION.cff` + mint the Zenodo DOI.** File does not exist yet.
  Enable the GitHub–Zenodo integration *before* pushing the tag (Zenodo archives
  on tag-push), then backfill the minted DOI into `CITATION.cff` and the README.
  Template details in [release strategy](#release-strategy).
- [ ] **Add an sdist include/exclude.** The build backend is hatchling and only a
  `[tool.hatch.build.targets.wheel]` table exists — the sdist currently bundles
  ~20 MB of `docs/` (built figures included) and internal files. Add a
  `[tool.hatch.build.targets.sdist]` table with `exclude` (at minimum
  `docs/`, `laboratory/`, `audits/`, `papers/`, `.brain-drafts/`,
  `tests/experimental/`) or an explicit `include` allowlist, then verify with
  `uv build --sdist` + `tar -tzf dist/*.tar.gz | head -50`.
- [ ] **Fix the hardcoded path in `scripts/check.sh` (line 77).** The clean-venv
  wheel smoke installs `/Users/anna/projects/jaxstro-dev/jaxstro` verbatim.
  Replace with a repo-relative resolution, e.g.
  `"$(cd "$(dirname "$0")/../../jaxstro" && pwd)"`, so the gate script survives
  on any checkout (CI included).

**After the tag — get it on PyPI:**

- [ ] **Publish `progenax` to PyPI** *(added to scope 2026-07-11)*. This is
  gated on the dependency story: the wheel depends on the sibling `jaxstro`,
  which is also unpublished, so **`jaxstro` must reach PyPI first** (or the
  strategy in [release strategy](#release-strategy) — namespace package /
  vendor / git-URL — must be decided instead). Concrete steps once decided:
  1. Publish `jaxstro` (its own gate + tag).
  2. Replace the `[tool.uv.sources]` local-path dependency with a versioned
     PyPI pin (`jaxstro>=X.Y`).
  3. `uv build` → `twine check dist/*` → upload to **TestPyPI** and verify
     `pip install progenax` in a clean venv resolves end-to-end.
  4. Upload to PyPI (`twine upload` or a trusted-publisher GitHub Action on
     the tag).
  A GitHub source release does NOT wait on this — flip/tag first, PyPI after.

(checklist-blockers)=
## ❌ Blockers — must resolve to claim "released"

- [ ] **Resolve the `jaxstro` dependency story (R2).** An outsider cannot
  `pip install progenax` while the runtime depends on the unpublished `jaxstro`.
  Decision deferred to the maintainer (PyPI-first / namespace package / vendor /
  git-URL) — see [release strategy](#release-strategy). *Not a blocker for a
  GitHub source tag; a hard blocker for PyPI.*
- [ ] **Re-enable CI (D8).** Three GitHub Actions workflows are `disabled_manually`.
  Re-enable `tests.yml` (PR gate), `physics-validation.yml` (release-tag slow
  lane), and `docs.yml` (the docs gate — 0 broken `.md` links + 0 MyST content
  warnings, via `make -C docs/website gate`; added + validated locally 2026-06-20,
  including negative tests, but never run on GitHub). Confirm a green run on a real
  PR before tagging.

(checklist-important)=
## ⚠️ Should fix before a polished public launch

- [x] **Fix the stale `validate_king.py` (D2/T11).** *(done 2026-06-18)* Dropped the
  removed `r_t=` kwarg at both call sites; script exits 0 and writes all 5 figures.
- [x] **Document `MichieProfile` (D4).** *(done 2026-06-18)* Full docstrings added to
  `from_W0_rc`, `__init__`, `.sample_positions`, `.density`.
- [x] **Fill `TruncatedIMF.ppf` and the IMF mass-ratio `pdf`/`cdf`/`ppf` Returns
  docstrings (D4).** *(done 2026-06-18)*
- [x] **Decide the units-policy split (D4/A2).** *(done 2026-06-18 — breaking sweep)*
  Chose the explicit-$G$ sweep: the five `*VelocityDF.sample_velocities`,
  `MultiComponentCluster.sample_cluster`, and `sample_velocities_pipeline` now
  **require** explicit $G$ (no `G=None` default); `VelocityDF` protocol contract
  aligned; CHANGELOG breaking entry added.
- [x] **Resolve the `40-howto/` stubs (D6).** *(done 2026-06-18)* Authored the 4
  progenax-only how-tos (each code block run-verified). The originally
  backlogged `interface-with-gravax` page was replaced and promoted into the
  TOC with the cataloged-IC boundary on 2026-07-29.
- [x] **Fix the internal-doc links (D6).** *(done 2026-06-18; re-verified in the
  docs-hardening pass)* Rewrote/removed the `../../plans/` and `../../notes/` links
  in the real-source pages. The docs-hardening pass added a dedicated link-integrity
  gate (`scripts/check_links_and_counts.py`) — MyST silently passes bad `.md`
  targets, so this catches what the build does not — and drove the broken-link count
  to **0** (the 7 broken `.md` links it first surfaced, including the 6 expecting a
  `99-bibliography/index.md` landing, are all fixed).
- [x] **Fetch or downgrade two cited-but-not-held references (D6).** *(done
  2026-06-18 — held + verified)* B&M82, Baumgardt&Makino 2003, **and** Strigari+2007
  PDFs ingested; per-paper notes written; `project_dispersion`/`tidal.py` citations
  corrected to the verified primaries (σ_los→B&M82 Eq.7; σ_pmR/σ_pmT→Strigari+2007
  Eqs.2/3; `jacobi_radius_isothermal`→BM03 Eq.1; fill-factor BM03 mis-attribution
  removed). All formulas verified correct against the PDFs.
- [x] **Reconcile the repository URL (D7).** *(done 2026-06-18)* `myst.yml` +
  generated API pages now use `github.com/jaxstro/progenax`.
- [x] **Hold the OED demos out of v0.1.0 (D10).** *(executed in the docs-hardening
  pass)* The OED **code** is still held for the planned **informax** package. The
  **docs** were trimmed to a single public overview
  (`60-science-demos/optimal-design/index.md`), which states plainly that the OED
  tooling is prototyped and not part of v0.1.0; the five worked-example detail pages
  were set `hidden: true` (built and URL-reachable, kept out of the public nav,
  linked from the [unlisted-pages index](../_unlisted/index.md)).

(checklist-minor)=
## ℹ️ Minor / polish (non-blocking)

- [ ] **Re-verify the scanned-PDF anchors for the methods paper (D6).** EFF (1987),
  Plummer (1911), Michie (1963), Kroupa (2001) are scanned images; verify their
  load-bearing values visually before publication. (Demircan & Kahraman 1991 was
  re-verified during this audit.)
- [ ] **Reconcile the experimental test count.** Docs say 322; the suite collects
  351 (343 non-slow). Prefer "see CI for the live count" over a pinned number.
- [ ] **Trim the sdist (~20 MB).** It bundles `docs/` including built figures;
  optionally exclude `_build`/figures from the sdist.
- [x] **Scrub the public dev-log.** *(done in the docs-hardening pass)* The
  `90-development-log/` section was curated and modernised, its absolute home paths
  and internal-archive pointers were scrubbed, and the whole section was set
  `hidden: true` (kept out of the public nav; linked from the
  [unlisted-pages index](../_unlisted/index.md)). The companion absolute-path and
  internal-pointer leaks in the validation pages were scrubbed in the same pass.
- [ ] **Optionally auto-execute `docs/` code blocks** the way the README blocks
  are, to machine-protect the tutorials.

(checklist-verified)=
## ✅ Verified green — no action needed

- Full test gate: 1561 passed / 2 skipped / **0 xfailed**, exit 0; coverage 95.98%.
  (Audit-time was 1553 / 3 skip / 1 xfail / 95.96%; the OM-Plummer oracle closed the
  skip and the follow-on #4 arc closed the Michie-$W_0$ xfail — C¹ PCHIP back-interp,
  ADR 0016. The remaining skip is the env-gated strict-refs guard.)
- Physics anchors reproduce (Plummer/King/EFF $Q$, King $c(W_0)$, Engine-B,
  Kepler).
- Gradient integrity: 97 AD-vs-FD cases, 0 hazards; all 4 registries full.
- JAX-native discipline: no hardcoded $G$, no numpy/scipy on core paths,
  `while_loop` correctly fenced.
- `myst build`: 178 pages, 0 content warnings; no fabricated literature content.
  (172 at the 2026-06-17 audit; the docs-hardening pass added a bibliography
  landing, missing per-paper notes, and the unlisted-pages index while consolidating
  the gravoturbulence subsection.)
- Packaging: wheel clean, `twine` passes, LICENSE + metadata present, clean-venv
  import works.
- README examples execute (9 blocks) in CI.
- Nine of ten prior-audit blockers (R1, R3–R10) verified fixed.

(checklist-docs-hardening)=
## 📚 v0.1.0 documentation-hardening pass — release provenance

A dedicated pass made the docs site correct, honest, and warning-clean for the
v0.1.0 public source release. It ran in two phases:

1. **A hybrid adversarial audit** — a serial reviewer plus parallel reviewers over
   the theory, validation, and demos sections — to surface every drift, fabricated
   claim, stale count, and broken link as *leads*.
2. **A section-by-section hardening** in which each lead was verified against the
   actual `src/` symbol or the source PDF before any edit, then fixed:
   - **Correctness:** corrected API snippets (signatures, required `G`, return
     shapes, the King `r_t` differentiability statement) and citations against the
     primary sources.
   - **Test-count drift → dashboard:** replaced hardcoded current-tense counts with
     pointers to the [test dashboard](../50-validation/test-dashboard.md) / "see CI
     for the live count"; dated changelog/audit counts were left frozen and
     annotated.
   - **Link integrity:** added a link/count gate (`scripts/check_links_and_counts.py`)
     and drove broken `.md` links to **0**.
   - **Scope & navigation:** consolidated the gravoturbulence theory 10 → 5 pages and
     moved it to `hidden: true`; trimmed OED to one public overview with five hidden
     detail pages; curated, modernised, path-scrubbed, and hid the development log;
     added a `99-bibliography/index.md` landing and the missing per-paper notes.

This entry records the pass itself as release provenance; the specific outcomes are
checked off above. The genuine remaining blockers (CI re-enable, the `jaxstro`/PyPI
dependency story) are unaffected and still pending.
