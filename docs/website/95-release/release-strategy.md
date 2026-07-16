---
title: Release strategy
subtitle: Version, channel, the jaxstro dependency, and the path to a citable release
description: >-
  The recommended release strategy for progenax v0.1.0 — semantic version,
  release channel, the jaxstro ecosystem-dependency decision, CHANGELOG and
  DOI plan, public-API freeze, and how experimental and OED-demo code is handled.
---

(release-strategy)=
# Release strategy

The audit verdict is that the science and the mechanics are ready; the open
questions are about *channel* and *ecosystem packaging*, which are the
maintainer's to decide. This page lays out the recommended path and the decisions
that shape it.

(strategy-version)=
## Version: 0.1.0

Ship as **`0.1.0`** with the `Development Status :: 4 - Beta` classifier already
in `pyproject.toml`. This is honest: a first public release of research software
with a verified core and a known, deferred PyPI-installability story. Bumping to a
release candidate or a higher minor would over- or under-state where the package
is. Adopt [semantic versioning](https://semver.org/) going forward: the public
API (`progenax.__all__`, 122 symbols) becomes the compatibility surface a 1.0
would freeze.

(strategy-channel)=
## Channel: GitHub source tag now, PyPI when the ecosystem packaging is decided

The runtime depends on the sibling `jaxstro`, which is not yet on PyPI, so
`pip install progenax` cannot work for an outsider today. Two consequences:

- **A GitHub source tag (`v0.1.0`) is releasable now.** The documented
  side-by-side checkout (`git clone jaxstro` next to `progenax`,
  `uv pip install -e ".[dev]"`) is the supported install, and the installation
  docs already describe it. This is the recommended near-term release.
- **PyPI is gated on the `jaxstro` decision below.** Tagging the slow-physics CI
  lane (`push: tags: ["v*"]`) on the GitHub tag also exercises the headline trust
  anchors with `PROGENAX_STRICT_REFS=1`.

(strategy-jaxstro)=
## The `jaxstro` ecosystem-dependency decision (maintainer-owned)

progenax core needs only a small, stable slice of `jaxstro` (`jaxstro.units` +
`jaxstro.jaxconfig`). How that slice reaches an installer is **an open ecosystem
decision** the maintainer is still resolving — the leading direction is a
**`jaxstro` namespace package** spanning the ecosystem. The options, with
trade-offs:

| Option | What it means | Trade-off |
|---|---|---|
| **Namespace package** | Publish `jaxstro` as a PEP-420 namespace umbrella; `progenax` depends on the published namespace dist(s). | Cleanest long-term ecosystem story; needs the namespace layout settled first. *(Maintainer's leading direction.)* |
| **Publish `jaxstro` to PyPI first** | Pin `jaxstro>=0.1,<0.2`; release progenax after. | Simple, conventional; couples two release timelines. |
| **Vendor `units`+`jaxconfig`** | Copy the two modules into `progenax._vendor` for a standalone wheel. | Fastest path to a standalone PyPI wheel; costs ecosystem coherence (duplicated `UnitSystem` objects shared with gravax). |
| **git-URL dependency** | `jaxstro @ git+https://…` | Works for a GitHub soft-launch; **PyPI rejects URL deps**, so not a PyPI path. |

**Recommendation:** proceed with the GitHub source tag now; resolve the namespace
question on the ecosystem's own timeline, then publish progenax to PyPI against
whatever `jaxstro` distribution results. Do not block the verified v0.1.0 core on
the packaging decision.

(strategy-ci)=
## Re-enable CI as a release prerequisite

The CI configuration is sound (sharded core, wheel-smoke, gradient-gate, slow
lane, version matrix) but **both workflows are disabled**. Before tagging:

1. Re-enable `tests.yml` and `physics-validation.yml`.
2. Confirm a green run on a real PR (the aggregator `tests` check).
3. Optionally restore a nightly cron, or rely on the per-PR + release-tag lanes
   to conserve Actions minutes.

(strategy-changelog)=
## CHANGELOG and provenance

A `CHANGELOG.md` already exists and documents the `0.1.0` cycle's fixes against the
prior audit's finding IDs — keep it as the canonical changelog and stamp the
release date when tagging. This audit report, checklist, and strategy live under
`95-release/` as the release-readiness provenance, complementing the existing
[validation dashboard](#release-audit-report) and gradient-audit registries.

(strategy-citation)=
## Citable release (Zenodo / DOI)

For a research package that a methods paper will cite, mint a **DOI via the
GitHub–Zenodo integration** on the `v0.1.0` tag and add a `CITATION.cff` to the
repository root. The docs site already carries the ORCID and Apache-2.0 / CC-BY-4.0
licensing; a DOI closes the loop so users cite a fixed, archived version. This
pairs naturally with the methods-paper plan sketched in the prior audit.

(strategy-scope)=
## Experimental and OED-demo code at release

- **`gravoturb`** stays repo-only — excluded from the wheel and the public API,
  and labelled experimental throughout the docs. During the v0.1.0
  documentation-hardening pass its theory subsection was consolidated 10 → 5 pages
  and **moved out of the public navigation**: the five pages are kept in the repo
  and still build (they are `hidden: true` in the `myst.yml` toc and reachable by
  URL), but they no longer appear in the site nav and are linked instead from the
  [unlisted-pages index](../_unlisted/index.md). Mention `gravoturb` in release
  notes as future work (a separate paper), not as a shipped feature.
- **OED demos** keep a **single public overview page**
  (`60-science-demos/optimal-design/index.md`) — the telescope-time hook plus the
  four headline numbers and the one-paragraph "any differentiable forward model +
  likelihood = Fisher = OED" framing. The five worked-example detail pages
  (`background`, `anisotropy`, `dynamical-mass`, `concentration`,
  `binary-robustness`) were set `hidden: true` (built and URL-reachable, kept out
  of the public nav, linked from the [unlisted-pages index](../_unlisted/index.md)).
  The **OED tooling code itself is held out of v0.1.0** and is slated to migrate to
  the planned **informax** package (the CAREER Aim-1 prototype); the public
  overview says so plainly (tracked in the [checklist](#release-checklist)).

(strategy-docs-hosting)=
## Documentation hosting

The MyST site builds to 178 pages with zero content warnings and is the project's
single source of truth. (The count rose from the 172 of the 2026-06-17 audit as the
v0.1.0 docs-hardening pass added a bibliography landing, missing per-paper notes,
and the unlisted-pages index, while consolidating the gravoturbulence subsection.)
Deploy it (GitHub Pages or the existing MyST hosting) from the release tag — the
[internal-doc-link fixes](#release-checklist) that would otherwise deploy broken
were completed during the docs-hardening pass (the link gate now reports 0 broken
cross-references).

:::{note} What this strategy commits to
A **GitHub source release of v0.1.0 now** on a re-enabled, green CI, with a DOI for
citation; **PyPI deferred** until the `jaxstro` ecosystem-packaging decision lands;
**OED demos held** for informax; **`gravoturb` experimental**. None of this
blocks the verified scientific core from being released.
:::
