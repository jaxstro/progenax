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

:::{note} This is a plan, not a record of work done
The audit was read-only by design — nothing below has been actioned yet. Each fix
is a separate, maintainer-approved step.
:::

(checklist-blockers)=
## ❌ Blockers — must resolve to claim "released"

- [ ] **Resolve the `jaxstro` dependency story (R2).** An outsider cannot
  `pip install progenax` while the runtime depends on the unpublished `jaxstro`.
  Decision deferred to the maintainer (PyPI-first / namespace package / vendor /
  git-URL) — see [release strategy](#release-strategy). *Not a blocker for a
  GitHub source tag; a hard blocker for PyPI.*
- [ ] **Re-enable CI (D8).** Both GitHub Actions workflows are `disabled_manually`.
  Re-enable `tests.yml` (PR gate) and `physics-validation.yml` (release-tag slow
  lane) so the advertised gate actually runs. Confirm a green run on a real PR
  before tagging.

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
  progenax-only how-tos (each code block run-verified); `interface-with-gravax`
  backlogged out of the TOC until gravax matures.
- [x] **Fix the internal-doc links (D6).** *(done 2026-06-18)* Rewrote/removed the
  `../../plans/` and `../../notes/` links in the real-source pages; `myst build`
  reports 0 broken xrefs.
- [x] **Fetch or downgrade two cited-but-not-held references (D6).** *(done
  2026-06-18 — held + verified)* B&M82, Baumgardt&Makino 2003, **and** Strigari+2007
  PDFs ingested; per-paper notes written; `project_dispersion`/`tidal.py` citations
  corrected to the verified primaries (σ_los→B&M82 Eq.7; σ_pmR/σ_pmT→Strigari+2007
  Eqs.2/3; `jacobi_radius_isothermal`→BM03 Eq.1; fill-factor BM03 mis-attribution
  removed). All formulas verified correct against the PDFs.
- [x] **Reconcile the repository URL (D7).** *(done 2026-06-18)* `myst.yml` +
  generated API pages now use `github.com/jaxstro/progenax`.
- [ ] **Hold the OED demos out of v0.1.0 (D10).** *(deferred — Anna 2026-06-18)*
  Held until the **informax** package is stood up; no TOC/scope change yet.

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
- [ ] **Scrub the public dev-log.** `90-development-log/` and a few validation
  pages contain absolute home paths and `.claude-work/` references — fine for an
  internal archive, but they read as dev notes if the section is public.
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
- `myst build`: 172 pages, 0 content warnings; no fabricated literature content.
- Packaging: wheel clean, `twine` passes, LICENSE + metadata present, clean-venv
  import works.
- README examples execute (9 blocks) in CI.
- Nine of ten prior-audit blockers (R1, R3–R10) verified fixed.
