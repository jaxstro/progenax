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

The [current release verdict](index.md#release-readiness-verdict) is that the
current revision is not qualified for public release. This page records a
proposed distribution path. Scientific-domain, gradient, installation, artifact,
and documentation gates must close before a source tag or package publication.

(strategy-version)=
## Version: 0.1.0

Ship as **`0.1.0`** with the `Development Status :: 4 - Beta` classifier already
in `pyproject.toml`. This is honest: a first public release of research software
whose qualification and distribution are still in progress. A version number
does not establish readiness. Adopt [semantic versioning](https://semver.org/)
for the supported public API after its scope and compatibility policy are
reviewed; use the generated symbol index for the current surface.

(strategy-channel)=
## Proposed channels: source tag and PyPI

The runtime depends on the sibling `jaxstro`, which is not yet on PyPI, so
`pip install progenax` cannot work for an outsider today. Two consequences:

- **A GitHub source tag (`v0.1.0`) requires current qualification and approval.**
  The side-by-side checkout is the intended source-install route. Verify it in
  a clean environment using the [installation instructions](../00-getting-started/installation.md).
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

**Proposed sequence:** qualify the supported core and source installation, then
ask Anna to choose and approve the release channel. PyPI additionally needs a
resolvable `jaxstro` distribution. No tag, upload, or visibility change is
authorized by this strategy.

(strategy-ci)=
## Re-enable CI as a release prerequisite

Workflow configuration and remote enablement require fresh checks. For a private
repository, Anna must explicitly approve the repository and intended CI scope
before enabling, dispatching, rerunning, or triggering workflows, including
automatic triggers from pushes, tags, or pull requests. Local gates can run
without that remote action. Before an approved release, inspect all workflow
events and record the verification actually performed.

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

Build the current MyST site, run its link and API-freshness gates, execute
reviewed examples, and inspect rendered pages before any deployment. Hidden
TOC pages remain directly reachable; hiding navigation does not provide access
control. Public hosting requires a separate content-scope review and Anna's
explicit approval. Unpublished audit material belongs in the private tracker.

:::{note} What this strategy commits to
This is a proposal, not release authorization or a readiness verdict. Current
qualification, a supported installation, artifact inspection, privacy review,
and maintainer approval precede any release. OED tooling and experimental
Gravoturb remain outside the supported wheel scope.
:::
