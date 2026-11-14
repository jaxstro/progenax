---
title: Release strategy
subtitle: Version, channel, the jaxstro dependency, and the path to a citable release
description: >-
  Proposed release channels, the unresolved PyPI dependency decision, API scope,
  citation and publication steps for progenax v0.1.0.
---

(release-strategy)=
# Release strategy

The [current verdict](index.md#release-readiness-verdict) remains **not qualified
for release**. This page records the proposed release path. Local engineering
checks, scientific qualification and authorization to publish are separate.

(strategy-version)=
## Version: 0.1.0

Retain the proposed **0.1.0** version and the beta classifier in `pyproject.toml`.
Before a tag, approve the supported public API and its compatibility policy;
use the [generated symbol index](../30-api/full-symbol-index.md) for the current
surface. A version number does not establish qualification.

(strategy-channel)=
## Proposed channels: source tag and PyPI

- **Source checkout and wheel:** the documented source route uses the public
  `jaxstro` sibling. An isolated wheel installation with Python 3.13.7, JAX 0.11
  and public `jaxstro` revision `94b83705` passes its smoke check and six existing
  compatibility tests, including registry-resource loading, without an editable
  checkout fallback. This verifies that environment,
  not every supported version or platform. See the
  [installation guide](../00-getting-started/installation.md).
- **Source tag:** requires the remaining [qualification gates](checklist.md),
  final artifact review and Anna's approval. A tag may match a workflow trigger;
  inspect current remote state and obtain CI authorization before creating it.
- **PyPI:** remains gated on a resolvable `jaxstro` distribution and an end-to-end
  installation check through that channel. A wheel installed with a public Git
  source for `jaxstro` does not establish `pip install progenax` from PyPI.

(strategy-jaxstro)=
## The `jaxstro` ecosystem-dependency decision (maintainer-owned)

The source-checkout route is available; the PyPI distribution decision remains
open. Existing options are proposals, not selected implementation work:

| Option | What it means | Trade-off |
|---|---|---|
| Namespace package | Publish the required `jaxstro` distributions under a shared namespace. | Requires the ecosystem layout to be settled. |
| Publish `jaxstro` first | Depend on its published, versioned distribution. | Couples the two release timelines. |
| Vendor the required `jaxstro` surface | Audit units/configuration, quadrature, numerics, constants, spatial helpers and registry APIs before scoping a copy. | Requires a separate dependency and ownership decision; copying only `units` and `jaxconfig` is insufficient. |
| Git-URL dependency | Install `jaxstro` from its public Git source. | Supports a source-install route; not a PyPI dependency solution. |

No namespace migration, vendoring or dependency-policy change is authorized by
this strategy. Anna chooses the release channel and dependency policy.

(strategy-ci)=
## Local checks and remote CI authorization

The local release gate passed, including the primary suite, 24 core validation
CLIs, 19 serial artifact checks and fresh regeneration of the 99-case gradient
audit. The separate LIMEPY table timing test also passed serially without coverage.
The primary measurement uses Progenax `59b96f30` and local `jaxstro` `6451e183`;
it is distinct from the bounded installed-wheel check against public `jaxstro`
`94b83705`. The [checklist](checklist.md) records coverage, completed checks and
remaining qualification. These local results do not demonstrate a remote
workflow run or universal physical qualification.

Before a push, tag, pull request, dispatch or deployment, inspect repository
visibility, workflow events and current remote state. Obtain explicit
authorization for the intended CI scope; private-repository CI always requires
Anna's permission. Do not infer CI approval from permission to commit or push.
A publication requested with no CI must suppress every applicable automatic
trigger.

(strategy-changelog)=
## CHANGELOG and provenance

Keep `CHANGELOG.md` as the release-change owner and date it when a release is
approved. Record the exact source revision, environment, artifact identities,
checks and remaining limits. The [June audit](audit-report.md) retains its
historical scope; the [current verdict](index.md) owns current readiness.

(strategy-citation)=
## Citable release (Zenodo / DOI)

For an approved citable release, add `CITATION.cff` and archive the fixed version
with a DOI. Configure any GitHub–Zenodo integration deliberately before its
release trigger, then record the DOI in the citation metadata and README.
A proposed DOI or integration is not authorization to create a tag or upload.

(strategy-scope)=
## Experimental and OED-demo code at release

- **Gravoturb** remains outside the supported released core and package artifacts.
  Keep private research code, outputs and planning material out of public release
  inputs. Historical public material does not qualify the current experimental
  workflow or make it a shipped core feature.
- **OED tooling** remains outside the v0.1.0 supported feature set. The existing
  public overview and hidden examples describe prototype scope; the proposed
  migration to informax is future work. Hidden pages remain directly reachable
  and must meet the same public-content boundary.

(strategy-docs-hosting)=
## Documentation hosting

MyST CLI 1.10.1 with book theme 1.4.1 builds 215 HTML pages with zero content
warnings and zero broken links. Before deployment, verify the approved source
snapshot, generated API freshness, relevant runnable examples and rendered pages.
A clean build is an engineering check, not physical qualification or deployment
approval. Hiding a page from navigation does not make it private.

:::{note} What this strategy commits to
This remains a proposal. Current qualification, channel-specific installation,
artifact and privacy checks, and Anna's release approval precede a tag or upload.
Remote CI and deployment require their own explicit authorization.
:::
