---
title: Interface with gravax
description: Backlogged recipe — handing a progenax IC off to gravax for N-body evolution. Deferred until the gravax integration interface stabilizes.
---

(howto-interface-gravax)=
# Interface with `gravax`

```{note}
**This recipe is backlogged.** A worked handoff from a progenax `ICResult` to
a [gravax](https://github.com/jaxstro/gravax) N-body integrator is planned but
deferred until the gravax integration interface stabilizes. It is intentionally
**not** in the How-to navigation yet — this page exists only as a placeholder
for the inbound cross-references.
```

progenax produces *pure physical state* — positions, velocities, masses,
stellar radii — packaged as an immutable
[`ICResult`](../30-api/builders.md) PyTree with **no dependency on gravax**.
That state is exactly what an N-body integrator consumes, so the eventual
handoff is a thin adapter (`ParticleSystem.from_ic(ic, units=STELLAR)` in
gravax) rather than anything progenax must own.

Until that recipe lands, the [API reference](../30-api/builders.md) documents
the `ICResult` fields, and [](set-up-virial-cluster.md) shows how to build the
IC you would hand off.
