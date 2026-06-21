---
title: Binary populations
description: progenax's binary-population machinery — Kepler orbital elements, the four period-distribution families, and the three eccentricity distributions.
---

# Binary populations

A **binary population** is the kinematic complement to the binary IMF
discussed at [](../imfs/binary.md). The IMF chapter covers the
*statistical* framework — what fraction of stars are in binaries, what
mass ratios they have, and how that biases inferred IMF slopes. This
chapter family covers the **orbital mechanics**: given the binary
fraction and mass ratios, how to assign Keplerian orbits to each
binary, sample the orbital phase, and produce the resolved component
positions and velocities that an N-body integrator consumes.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers entering the binary-population track — learning how progenax turns a binary fraction and mass ratios into resolved Keplerian orbits; no prior binary-statistics literature assumed.
**Prerequisites:** the [binary IMF](../imfs/binary.md) (which stars are paired, and their mass ratios) is the natural companion, but this is also a good entry point for the orbital-mechanics track.
**You'll get:** what a "binary" is in progenax (provenance at $t=0$ vs. measured-from-state), the five composability axes of `build_binary_cluster`, and a map of the three chapters (elements, periods, eccentricities).
:::

## Map of the section

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
* - [](kepler-elements.md)
  - The seven orbital elements $(a, e, i, \Omega, \omega, M, t_p)$, their physical meaning, and the conversion from elements to phase-space coordinates.
* - [](period-distributions.md)
  - Four period-distribution families: log-uniform (Öpik), log-normal ({cite:t}`Sana2012` for OB-type, {cite:t}`DuquennoyMayor1991` for solar), {cite:t}`Sana2012` and {cite:t}`MoeDiStefano2017` empirical fits.
* - [](eccentricity.md)
  - Three eccentricity-distribution families: thermal ($f(e) = 2e$), uniform, period-dependent {cite:t}`MoeDiStefano2017`. Tidal-circularisation effects at short period.
```

## What a "binary" is in progenax

A binary in progenax is two particles whose relative orbit is a set of
`KeplerElements`. Both components live in the same `(positions, velocities,
masses)` arrays as single stars — secondaries are not "second-class" entries.
The primordial pairing is recorded as **provenance at $t=0$** on the
`ICResult` returned by `build_binary_cluster`:

```python
positions                 # (N, 3) component positions in the cluster frame
velocities                # (N, 3) component velocities
masses                    # (N,)   every star's mass, including secondaries
primordial_system_id      # (N,)   members of one system share an id
is_primordial_secondary   # (N,)   True on the secondary of each primordial binary
```

This labelling is the *birth* pairing. It goes stale under dynamical evolution
(encounters ionise soft binaries, three-body captures form new ones, exchanges
swap partners), so the **current** binary population of an evolved snapshot must
be *measured* from the phase-space state, not read off the labels:
`progenax.binaries.find_bound_pairs(positions, velocities, masses, G=...)`
returns the energy-bound mutual-nearest-neighbour pairs, and
`primordial_survival` compares them to the $t=0$ ids (survived / disrupted /
newly-formed). A single binary's elements are recovered with
`KeplerElements.from_state(r_rel, v_rel, M_total, G=...)`.

## Composability

`build_binary_cluster` composes **five independent axes**:

```{list-table}
:header-rows: 1

* - Axis
  - Role
* - `primary_imf`
  - The **primary** IMF — draws $m_1$ (companions are conditional, so the all-stars MF is derived; see [](../imfs/binary.md))
* - `companion_model`
  - Owns the binary statistics: multiplicity $f_b(m_1)$ **and** $q\to m_2$, $P\to a$, $e$, orientation
* - [spatial profile](../spatial-profiles/index.md)
  - Places the **system COMs** in 3D
* - [velocity DF](../velocity-dfs/index.md)
  - Sets each system COM's bulk velocity
* - `target`
  - Population-size budget: `Systems(n)` / `Stars(n)` / `TotalMass(M)`
```

The `companion_model` has two implementations: `IndependentCompanions`
(versatile independent $f_b \times q \times P \times e$ marginals — the
period-averaged default) and `MoeCompanions` (the faithful
{cite:t}`MoeDiStefano2017` joint $P$–$q$–$e$ interrelation, where the *same* $q$
sets $m_2$, so the coupling is self-consistent). `f_b` lives inside the
companion model — in Moe it is part of the model, set by the IMF masses.

## Resolved vs unresolved

`resolve_binary_components` places each binary's two components around its COM
using the barycentric split $m_1\,\delta r_1 + m_2\,\delta r_2 = 0$, so the COM
(and hence the cluster phase space) is preserved exactly. `build_binary_cluster`
returns one of two forms:

```python
from progenax.builders import build_binary_cluster, Systems
from progenax.binaries import MoeCompanions

# compact=True (default): eagerly compacted ICResult of real particles
ic = build_binary_cluster(profile, velocity_df, primary_imf,
                          MoeCompanions(), Systems(1000), key, units=STELLAR)

# compact=False: the masked, fixed-shape ResolvedBinaries (2N slots + is_real
# mask) — jit/grad-safe; required for differentiable IC generation
rb = build_binary_cluster(profile, velocity_df, primary_imf,
                         MoeCompanions(), Systems(1000), key, units=STELLAR,
                         compact=False)
```

For N-body integration the resolved components are what you want — the integrator
evolves the orbital motion (binaries are *collisional*: integrate with a
collisional scheme, $\varepsilon = 0$). The COM virialisation treats each binary
as a single CoM particle (the McLuster convention, {cite:t}`Kuepper2011` §A8) and
leaves the internal binary binding energy as a separate reservoir, which
`binary_energy_budget` reports explicitly.

## Connection to the binary IMF

The binary fraction $f_b(m_1)$ and the mass ratios from [](../imfs/binary.md)
decide *which* stars are paired and the secondary masses; the chapters in this
section decide the *orbital properties*. When consistency matters they come from
the same {cite:t}`MoeDiStefano2017` calibration — periods from
[](period-distributions.md), mass ratios from
[](../imfs/mass-ratio-distributions.md), eccentricities from [](eccentricity.md) —
and `MoeCompanions` samples them *jointly* to preserve the period-conditional
non-separability noted at [](../imfs/multiplicity-statistics.md).

## Implementation, validation & references

- **In code:** the binary subsystem lives under `src/progenax/binaries/`
  (`kepler.py`, `period.py`, `eccentricity.py`, `companions.py`,
  `assembly.py`, `diagnostics.py`); composition with the IMF + spatial
  profile + velocity DF is `build_binary_cluster` in
  `src/progenax/builders.py`. See the [binaries API](../../30-api/binaries.md)
  and the [builders API](../../30-api/builders.md); each chapter below
  carries its exact module path.
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md)
  (the binary-IMF + composition regression suite).
- **Primary sources:** Kepler-element machinery is standard textbook
  material; period distributions {cite:t}`Sana2012` (OB-type),
  {cite:t}`DuquennoyMayor1991` (solar), {cite:t}`MoeDiStefano2017`
  (joint $f(P)$); eccentricity distributions {cite:t}`MoeDiStefano2017`
  and earlier compilations. Full notes in the
  [bibliography](../../99-bibliography/index.md); each chapter below
  points at the specific result(s) used.
