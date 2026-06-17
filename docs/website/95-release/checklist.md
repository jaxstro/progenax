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

- [ ] **Fix the stale `validate_king.py` (D2/T11).** It calls
  `KingVelocityDF(..., r_t=...)`; the `r_t` field was removed. Drop the kwarg at
  the two call sites (lines ~248, ~451). Consider wiring the `validate_*.py`
  scripts into a smoke job so script-rot is caught.
- [ ] **Document `MichieProfile` (D4).** Add full docstrings (args/units/returns/
  differentiability) to `from_W0_rc`, `__init__`, `.sample_positions`, `.density`.
- [ ] **Fill `TruncatedIMF.ppf` and the IMF mass-ratio `pdf`/`cdf`/`ppf` Returns
  docstrings (D4).**
- [ ] **Decide the units-policy split (D4/A2).** Either tighten the five
  `sample_velocities` methods and `MultiComponentCluster.sample_cluster` to require
  explicit $G$, or add a one-line "intentional convenience-default (policy
  exception)" note to each class docstring — applied uniformly. (Deferred as a
  breaking sweep in the CHANGELOG; resolve before 1.0.)
- [ ] **Resolve the `40-howto/` stubs (D6).** Six pages are empty `TBD.`
  placeholders in the published TOC — author them or remove them from `myst.yml`.
- [ ] **Fix the internal-doc links (D6).** Ten-plus links target
  `../../plans/` and `../../notes/` outside the site root; they will deploy broken
  and expose internal filenames. Rewrite or remove before public hosting.
- [ ] **Fetch or downgrade two cited-but-not-held references (D6).** Binney &
  Mamon (1982) in `kinematics/dispersion.py` (load-bearing for `project_dispersion`)
  and Baumgardt & Makino (2003) in `tidal.py` — fetch the PDFs or cite a held /
  textbook source.
- [ ] **Reconcile the repository URL (D7).** `myst.yml` declares
  `github.com/drannarosen/progenax`, while `pyproject.toml` and the git remote use
  `github.com/jaxstro/progenax`. Pick one and make them agree.
- [ ] **Hold the OED demos out of v0.1.0 (D10).** Remove
  `60-science-demos/optimal-design/` from the release TOC and exclude
  `scripts/_demo_oed*.py` / `scripts/demo_oed*.py` from the release scope; they
  migrate to the planned **informax** package.

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

- Full test gate: 1553 passed / 3 skipped / 1 xfailed, exit 0; coverage 95.96%.
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
