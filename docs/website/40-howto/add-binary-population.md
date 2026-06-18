---
title: Add a binary population
description: "Recipe — assemble a star cluster with a primordial binary population using build_binary_cluster, composing a primary IMF with a companion model, and read its two-scale energy budget."
---

(howto-add-binaries)=
# Add a binary population

**Goal.** Build a cluster IC that contains **primordial binaries**, not just
single stars. `build_binary_cluster` composes five independent axes —
spatial profile × velocity DF × primary IMF × companion model × population
target — and resolves each binary's two components around its centre of mass.

## Inputs and assumptions

```{list-table} Recipe inputs
:header-rows: 1
:label: tbl-binaries-inputs

* - Axis
  - What it controls
  - Fiducial
* - `profile` × `velocity_df`
  - Spatial structure of the system **centres of mass**.
  - Plummer, $\rh = 1$ pc
* - `primary_imf`
  - The IMF of **primaries** (companions are attached on top).
  - $\alpha = 2.3$ power law, $0.1$–$20\,\Msun$
* - `companion_model`
  - The single owner of the binary statistics: $f_b \to$ multiplicity, $q \to m_2$, $P \to a$, $e$, orientation.
  - `IndependentCompanions` ($f_b = 0.5$)
* - `target`
  - What the population size holds fixed: `Systems(n)`, `Stars(n)`, or `TotalMass(M)`.
  - `Systems(1000)`
* - `units` → `G`
  - Unit system carrying `G` **and** the day→time-unit scale for periods. **Required.**
  - `STELLAR`
* - `Q`
  - Virial ratio of the **system COMs** (binaries treated as point masses).
  - 0.5
```

```{important}
:label: imp-binaries-conventions
**The companion model owns the binary statistics, and `Q` virializes the
COMs only.** `primary_imf` is the IMF of *primaries*; companions are drawn
conditionally ($m_2 = q\,m_1$), so the all-stars mass function is a *derived*
consequence, not the input IMF. The cluster is virialized treating each binary
as a single COM particle (the McLuster convention; Küpper+2011 §A8) — the
internal binary binding energy is a **separate reservoir** that `Q` never
touches. Measure it with `binary_energy_budget`, not with the naive resolved
virial ratio.
```

## Recipe

```python
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (
    PlummerProfile, PlummerVelocityDF,
    PowerLawIMF, IndependentCompanions, ConstantBinaryFraction,
    FlatMassRatio, LogUniformPeriod, ThermalEccentricity,
    build_binary_cluster, Systems,
    binary_energy_budget,
)

key = jax.random.PRNGKey(0)

profile     = PlummerProfile(r_h=1.0)
velocity_df = PlummerVelocityDF(r_h=1.0)
primary_imf = PowerLawIMF(exponents=[-2.3], breakpoints=[], m_min=0.1, m_max=20.0)

companions = IndependentCompanions(
    binary_fraction=ConstantBinaryFraction(f_bin=0.5),  # 50% of primaries are binaries
    q_distribution=FlatMassRatio(q_min=0.1),            # m2 = q * m1
    period_distribution=LogUniformPeriod(log_P_min=1.0, log_P_max=6.0),  # days
    eccentricity_distribution=ThermalEccentricity(),    # f(e) = 2e
)

ic = build_binary_cluster(
    profile, velocity_df, primary_imf, companions,
    target=Systems(1000), key=key, units=STELLAR, Q=0.5,
)

n_stars = ic.masses.shape[0]
n_bin   = int(jnp.sum(ic.is_primordial_secondary))
print(f"1000 systems -> {n_stars} resolved stars, {n_bin} binaries")

# Two-scale energy budget: COM virial vs internal binding.
budget = binary_energy_budget(
    ic.positions, ic.velocities, ic.masses, ic.primordial_system_id, G=STELLAR.G,
)
print(f"Q_com      = {budget.Q_com:.4f}   (cluster virial target)")
print(f"Q_resolved = {budget.Q_resolved:.4f}   (deflated -- mixes scales)")
print(f"E_internal = {budget.E_internal:.3e}   n_binaries = {budget.n_binaries}")
```

## Verified output

Measured (`PRNGKey(0)`, `Systems(1000)`):

```
1000 systems -> 1505 resolved stars, 505 binaries
Q_com      = 0.5000   (cluster virial target)
Q_resolved = 0.4579   (deflated -- mixes scales)
E_internal = -1.466e+07   n_binaries = 505
```

505 of the 1000 systems drew a companion ($f_b = 0.5$), giving 1505 resolved
stars. `Q_com = 0.5000` confirms the **system COMs** are virialized exactly;
`Q_resolved = 0.46` is lower because the deep internal binary binding energy
inflates $|W|$ on the resolved stars — which is why you read the COM scale,
not the resolved one.

```{tip}
- Swap `IndependentCompanions` for **`MoeCompanions`** to use the faithful
  Moe & Di Stefano (2017) $P$–$q$–$e$ coupling (no `f_b` argument — it is set
  by the primary masses).
- `Systems(n)` is the only **differentiable / `compact=False`** target (fixed
  shape). `Stars(n)` and `TotalMass(M)` have data-dependent counts and are
  eager-only (`compact=True`).
- `primordial_system_id` / `is_primordial_secondary` are **provenance at
  $t=0$** — they go stale under dynamical evolution. Measure the *current*
  binary population with `binaries.diagnostics.find_bound_pairs`.
```

## See also

- [](../10-theory/binaries/index.md) — binary orbital mechanics and population statistics.
- [](../60-science-demos/binary-energy-budget.md) — the two-scale energy budget in depth.
- [](../50-validation/binary-imf.md) — binary-population validation.
- [](../30-api/binaries.md) — full companion-model and diagnostics API.
