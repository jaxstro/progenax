# build_binary_cluster — SoTA IMF + binary composition (Batch 4k)

**Status:** IMPLEMENTED 2026-06-04 (Batch 4k, commit `11930ee`). Design brainstormed
with Anna in plan mode; grounded against her *Confidently Wrong* paper
(`papers/rosen-binary-imf-2026/`) and the McLuster methods paper
(`docs/core-papers/McLuster_Methods_2011.pdf` = Küpper, Maschberger, Kroupa &
Baumgardt 2011, MNRAS 417, 2300).

## What changed and why

`build_binary_cluster` was a fixed `n_systems` orchestrator that sampled q, P, e as
**independent marginals**. Two gaps (surfaced grounding against the paper):

1. **Counting.** `n_systems` does not let a dynamical-IC user fix the *star count* or
   *total mass*; companions (real gravitating bodies) were added on top, uncounted.
2. **Coupling.** The faithful Moe & Di Stefano (2017) P–q–e interrelation
   (`MoeJointOrbit`, Batch 4i) was built but never wired in; `q` (for `m2`) and `P`
   were drawn independently.

Now it composes **five independent axes**:

```
build_binary_cluster(profile, velocity_df, primary_imf, companion_model, target, ...)
```

`primary_imf` (IMF) × `companion_model` (binary statistics) × `profile` × `velocity_df`
(spatial) × `target` (population-size budget). `BinaryIMF.sample_systems` — the paper's
validated mass-only path — is **untouched**.

## A. Population-size budget (`target`)

| Target | Counts companions? | Result | Differentiable? | Provenance |
|---|---|---|---|---|
| `Systems(n)` | No | exactly `n` systems → `n + n_binary` stars | **Yes** (fixed shape, `compact=False`) | observational (paper: `N` = observed systems) |
| `Stars(n)` | Yes | `n` or `n+1` resolved stars | No (eager only) | McLuster (draws `N` stars, forms `N·b/2` binaries) |
| `TotalMass(M)` | Yes | total stellar mass `≥ M` | No (eager only) | McLuster-style mass filling |

**Caveats:**
- **Whole-system cuts** — `Stars`/`TotalMass` keep whole systems in draw order; a binary
  is **never split** to hit an exact count, so `Stars(n)` overshoots to `≤ n+1` stars and
  `TotalMass(M)` to `≤ one system` over `M`. This is deliberate (splitting a binary to hit
  a round number would drop a real companion); tests assert the bounds. The draw-order
  prefix is an unbiased sample of the system distribution.
- **Differentiability is `Systems`-only.** `Stars`/`TotalMass` have data-dependent system
  counts (binarity is stochastic) ⇒ dynamic shapes ⇒ eager only (`compact=True`). The
  masked `compact=False` path (jit/grad-safe IC generation) **requires `Systems(n)`** and
  raises `ValueError` otherwise. (Differentiating the IMF slope through the Bernoulli
  multiplicity draw is already blocked regardless, so this loses no real gradient.)
- `TotalMass` over-draws from a mean-system-mass presample and **tops up** in an eager
  loop until the budget is reached — robust to unlucky draws.

## B. Companion/orbit layer (`companion_model`)

A `CompanionModel` (protocol in `protocols.py`) is the **single owner** of the binary
statistics: given primary masses it decides multiplicity (`f_b → is_binary`) **and**
samples companion properties (`q → m2`, `P → a`, `e`, isotropic orientation), all keyed
on the primary masses.

- `IndependentCompanions(binary_fraction, q_distribution, period_distribution,
  eccentricity_distribution)` — versatile marginals; reproduces the **period-averaged**
  default used today and by the paper's mass-function forward model.
- `MoeCompanions(q_min=0.1)` — faithful Moe+2017: Moe's own mass-dependent
  `MassDependentBinaryFraction` **+** the joint `MoeJointOrbit` (`logP ~ MoePeriod(M1)`,
  `q ~ MoeDiStefano2017Full(M1, P)`, `e ~ MoeEccentricity(P, M1)`). The **same** `q` sets
  `m2 = m1·q`, so the P–q interrelation ("Mind your Ps and Qs") shows up self-consistently
  in the secondary masses (short-period binaries carry larger `q`; pinned by a test).

**`f_b` is folded into the companion model — there is no separate `binary_fraction` arg**
(Anna, 2026-06-04). In Moe, `f_b(m1)` is *part of* the model, set by the IMF masses we
pull; a separate arg would be redundant and could be made inconsistent with Moe's q/P/e.
`IndependentCompanions` keeps `f_b` as a configurable field (full versatility, e.g.
`ConstantBinaryFraction` + Moe q); `MoeCompanions` owns it internally.

**Period-averaged vs joint — which to use:** for a *mass-function* analysis the paper
deliberately marginalizes over period (period-averaged `q | M1`; use
`IndependentCompanions` with `MoeDiStefano2017`). For *dynamical ICs with realistic
orbits* use `MoeCompanions` (period-conditional joint). They serve different purposes; both
are first-class.

## C. Conventions (grounded)

**Primary IMF, conditional companions.** `primary_imf` is the IMF of *primaries*;
`m2 = q·m1` with `q | M1`. The all-stars mass function is therefore a **derived**
consequence, *not* the input IMF (Rosen, *Confidently Wrong*, §9.6: "the full set of
individual stars {m1, m2} does **not** follow the same IMF as the primaries"). The
conditional `q | M1` parameterization follows Moe & Di Stefano (2017) (their constraints
are defined conditional on primary mass). A Kroupa-style *individual-star* IMF (every star
drawn from ξ, then paired) is a **different** generative model — a deferred opt-in, not the
default.

**CoM virialization — McLuster convention (verbatim, §A8 p. 2316):** *"The binaries are
then replaced by a CoM particle for the rest of the procedure. Only in the very end, after
the density profile has been established and the velocities of the cluster members have
been scaled appropriately, the CoM particles get replaced by their two constituent
stars."* progenax does exactly this: `build_spatial_ic` virializes the system COMs
(binaries as point masses, `Q = 0.5` default — McLuster option `-Q`, p. 2313), then
`resolve_binary_components` places the two constituents around each COM. **Internal binary
binding energy is a separate reservoir untouched by `Q`** (scale separation: the global
virial balance lives on the COM scale). A `binary_energy_budget` diagnostic that reports
this reservoir is the next item (Batch 4j). Orbital orientation is isotropic (McLuster:
"The binary orbital plane is oriented randomly", p. 2316).

**Accuracy note — McLuster pairing differs.** McLuster draws `N` stars from the IMF then
*pairs* `N·b/2` of them (random, or ordered by mass above `msort`) — companions are drawn
from the IMF and counted in `N` (the `Stars(n)` philosophy). progenax's default is
*primary-constrained* (`m2 = q·m1`, Moe-conditional), which is **not** McLuster's pairing.
So McLuster is cited here for the **CoM-virialization** convention and the `Stars`/budget
philosophy, **not** for the primary-constrained recipe (that is Moe+2017 / the paper).

## D. Remaining caveats / deferred

- **One companion per primary.** Multiplicity is a single Bernoulli (`is_binary`); Moe's
  multiplicity frequency exceeds 1 for massive stars (triples/quadruples). This is the
  seam where higher-order multiples plug in later; for now it under-counts companions for
  O stars (conservative).
- **Equivalence gate is component-level, not bit-identical.** Moving `q`'s owner out of
  `sample_systems` into the companion layer changes the RNG routing by design; the default
  path is pinned to prior behaviour by the entropy-layout component test
  (`split(key,5) → [is_binary, q, P, e, orientation]`) + physics invariants (COM 1e-10,
  virial, Kepler-III), not a bit-identical assembled stream.
- `MoeCompanions.q_min` threads into `MoeDiStefano2017Full(q_min=...)` but not into the
  period/eccentricity sub-models (they have no q_min).

## Verification

Suite 1165 → 1194 (+12 `test_target_budget`, +12 `test_companions`, +5 integration), green
under jax 0.10.1 (full) and conda / jax 0.7.0. FD-grad-accurate on the eccentricity path;
grad-through-`r_h` finite; the P–q interrelation reproduced in the secondary masses.
