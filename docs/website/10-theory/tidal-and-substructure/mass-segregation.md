---
title: Mass segregation
description: Mass segregation two ways — the energy-ordered (Baumgardt-style) primordial generator and the differentiable equilibrium route via MultiComponentCluster.from_mass_segregation — plus the Λ_MSR diagnostic.
---

# Mass segregation

```{seealso}
This chapter covers **primordial** mass segregation seeded into the
initial conditions — distinct from **dynamical** segregation, which
emerges during evolution via two-body relaxation. The two are observationally
indistinguishable, which is exactly why it matters whether your IC
already has it. See {cite:t}`Allison2009` for the dynamical pathway and
[](../velocity-dfs/plummer-dfs.md) for the equilibrium DF that
underpins both.
```

Mass segregation refers to the spatial arrangement where massive stars
preferentially occupy lower-energy, more centrally concentrated orbits
than low-mass stars. It is observed in essentially every Galactic
globular cluster and in many young open clusters, but the
*provenance* — primordial (formed segregated) versus dynamical
(relaxed into segregation) — has been debated since
{cite:t}`Baumgardt2008` showed that primordial segregation can explain
both the IMF-slope-vs-concentration trend and low-mass star depletion in
old globular clusters, and {cite:t}`Allison2009` showed that subvirial
fractal initial conditions produce dynamical segregation on $\sim 1$ Myr
timescales — much shorter than the classical relaxation time.

progenax offers segregation **two ways**, with a clean division of
labour after the 2026-06 unified redesign:

1. **Primordial (non-equilibrium) generator** —
   `energy_sorted_segregation`, the energy-ordered
   {cite:t}`Baumgardt2008`-style construction described below. It is a
   discrete assignment (argsort + floor) and deliberately **not**
   differentiable.
2. **Equilibrium route (differentiable)** —
   `MultiComponentCluster.from_mass_segregation(delta)` (Engine A of the
   unified multi-component model): each mass component gets velocity-scale
   ratio $w_j = \mu_j^{-\delta}$ in ONE shared self-consistent potential,
   so *every* value of $\delta$ is a true equilibrium and the segregation
   knob $\delta$ is differentiable end-to-end. See
   [](../populations/index.md) and
   [](../spatial-profiles/lowered-model-family.md).

An earlier $\lambda_{\mathrm{seg}}$ catalog-blend (linear interpolation
between an unsegregated and a fully segregated catalog) was **retired**
in the redesign: its intermediate states drift from per-mass-group
virial balance, while the equilibrium route gives a smooth,
HMC-compatible segregation parameter *without* leaving equilibrium.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning primordial mass segregation — both the discrete energy-ordered generator and the differentiable equilibrium route; no prior cluster-dynamics literature assumed.
**Prerequisites:** [spatial profiles](../spatial-profiles/index.md) and the multimass-equipartition idea (the equilibrium route is a [`MultiComponentCluster`](../populations/index.md) constructor).
**You'll get:** the Baumgardt energy-ordered construction, the McLuster $S$-shuffle, the differentiable $\delta$-equipartition route, and the $\Lambda_{\mathrm{MSR}}$ diagnostic.
:::

## The Baumgardt + McLuster construction

The {cite:t}`Baumgardt2008` algorithm proceeds in four steps. We assume
the spatial profile and equilibrium distribution function are already
fixed (Plummer, King, or EFF; see [](../spatial-profiles/index.md) and
[](../velocity-dfs/index.md)).

**Step 1 — generate an orbit pool.** Draw $N_{\mathrm{pool}} = f_{\mathrm{pool}}
\cdot N_\star$ orbits from the equilibrium DF, with $f_{\mathrm{pool}}
\geq 4$ (more orbits than stars, because the energy-ranked binning
below distributes mass across phase space). Each orbit is a phase-space
sample $(\mathbf{r}_i, \mathbf{v}_i)$.

**Step 2 — sort orbits by specific energy.**

```{math}
:label: orbit-energy
E_i \;=\; \tfrac{1}{2}|\mathbf{v}_i|^2 \;+\; \Phi(\mathbf{r}_i)
```

[↗ model card](#card-orbit-energy)

Sort ascending; index 0 is the most-bound orbit, index $N_{\mathrm{pool}}-1$
is the least-bound.

**Step 3 — sort masses by descending mass.** Draw $N_\star$ masses from
the IMF, sort largest-to-smallest. The most massive star has *mass rank*
$i = 0$; the least massive has rank $N_\star - 1$.

**Step 4 — bin orbits by cumulative mass.** Define cumulative mass
fractions $\widetilde M_i = \sum_{j=0}^{i} m_j / M_{\mathrm{total}}$ and
energy bins per mass rank,

```{math}
:label: bin-bounds
\mathrm{bin}_{\mathrm{low}}[i] \;=\; \lfloor N_{\mathrm{pool}}\,\widetilde M_{i-1}\rfloor,
\qquad
\mathrm{bin}_{\mathrm{high}}[i] \;=\; \lfloor N_{\mathrm{pool}}\,\widetilde M_i\rfloor - 1.
```

For each mass rank $i$, sample one orbit uniformly from the energy-sorted
pool in the range $[\mathrm{bin}_{\mathrm{low}}[i],\, \mathrm{bin}_{\mathrm{high}}[i]]$.
The most massive star (rank 0) draws from the most-bound bin, the
second-most from the second bin, and so on.

```{admonition} Implementation departure (deliberate)
:class: note
The released `energy_sorted_segregation` replaces Step 4's *random*
per-bin draw with a **deterministic monotonic (isotonic-rounding)
assignment** of distinct orbit indices to mass ranks. The random draw
fails for steep IMFs: cumulative-mass bins collapse below one orbit,
forcing many low-mass ranks onto the *same* orbit (coincident stars,
$V = -\infty$). The deterministic assignment guarantees no orbit reuse
for any mass spectrum; realisation variety comes from the random orbit
pool. See the `energy_sorted_segregation` docstring and
[](../../50-validation/mass-segregation.md).
```

```{admonition} Why the orbit pool is oversized
:class: note
Massive stars "occupy more phase space" in this scheme — bin width is
proportional to $m_i / M_{\mathrm{total}}$. Without the $f_{\mathrm{pool}} \geq 4$
oversampling, the most-massive bin contains a single orbit and the
construction degenerates. $f_{\mathrm{pool}} = 4$ keeps every bin populated
to within Poisson noise for $N_\star \in [10^2, 10^5]$.
```

This construction has three useful mathematical properties:

1. **Density profile is preserved.** Each star ends up at the position
   of an orbit drawn from the equilibrium DF, so the density profile
   $\rho(r)$ is unchanged.
2. **Per-mass-group virial equilibrium is preserved.** The orbit pool
   samples the equilibrium DF, which means each mass group inherits the
   correct velocity dispersion for its radial range.
3. **Computational cost is $\mathcal{O}(N_{\mathrm{pool}}\,\log N_{\mathrm{pool}})$.**
   Two sorts dominate; the assignment is $\mathcal{O}(N_\star)$.

## Partial segregation: the McLuster S-shuffle

The {cite:t}`Kuepper2011` McLuster code parameterises *partial* segregation
through a strictness parameter $S \in [0, 1]$. The algorithm above
corresponds to $S = 1$ (complete segregation: most massive star $\to$
most-bound bin). For $S < 1$, McLuster permutes which physical star
receives each mass rank using

```{math}
:label: s-shuffle
j \;=\; \lfloor (N_\star - i)\,(1 - X^{1-S})\rfloor,
\qquad X \sim \mathcal{U}(0,1)
```

where $j$ indexes the next available star slot for mass rank $i$. The
limits are physically meaningful:

```{list-table}
:header-rows: 1

* - $S$
  - Behaviour
  - Result
* - $S = 0$
  - $j$ uniform in $[0, N - i]$
  - Random rank-to-star mapping; no segregation
* - $S = 1$
  - $j = 0$ always
  - Sequential mapping; rank $i$ to star $i$; complete segregation
* - $S = 0.5$
  - $j$ skewed toward 0
  - Partial segregation
```

```{warning}
**The S-shuffle alone does not produce partial segregation.** It permutes
which *star* gets which *rank*, but every rank still draws from its
own energy bin. To make $S$ control the *physical* degree of segregation
(measured e.g. by $\Lambda_{\mathrm{MSR}}$), the orbit-bin assignment
itself must also depend on $S$ — for instance by mixing the rank-indexed
bin with a random-bin draw. progenax does not implement the S-shuffle:
the released primordial generator is the fully ordered ($S = 1$)
construction, and *continuous* segregation strength is provided by the
equilibrium route described next.
```

## The differentiable route: equilibrium segregation

progenax expects gradient-based inference of segregation strength as a
first-class use case. The discrete McLuster $S$ parameter is awkward
for this — its effect on observables is non-monotonic in the partial
regime, and the floor function in {eq}`s-shuffle` is non-differentiable
even where it is monotonic. Instead of smoothing the discrete
construction, progenax provides a *first-principles* continuous knob:
`MultiComponentCluster.from_mass_segregation(delta)` (Engine A of the
unified multi-component model) builds $J$ mass components with
velocity-scale ratios

```{math}
:label: equipartition-law
w_j \;=\; \frac{s_j}{s} \;=\; \mu_j^{-\delta},
\qquad \mu_j = \frac{m_j}{\bar m},
```

the {cite:t}`Gieles2015`-style equipartition law: heavier components are
kinematically colder and sink in the shared self-consistent potential.
Crucially, *every* value of $\delta$ is a true shared-potential
equilibrium ($Q_j = 0.5$ per component, by construction), so the
segregation strength is a smooth physical parameter — differentiable
end-to-end through construction *and* sampling — rather than an
interpolation between catalogs. $\delta = 0$ is the unsegregated
corner; $\delta = 1/2$ is full Spitzer-style equipartition scaling.
The theory is developed in [](../populations/index.md) and
[](../spatial-profiles/lowered-model-family.md).

An earlier $\lambda_{\mathrm{seg}}$ *catalog blend* (linear
position-velocity interpolation between an unsegregated baseline and the
fully energy-ordered catalog) was retired in the 2026-06 redesign: its
intermediate states drift from per-mass-group virial balance (drift
peaking at $\sim 0.05$ in $Q_j$ near the midpoint), a defect the
equilibrium route does not share.

The *generators* are covered by unit tests in `tests/unit/cluster/`
(`test_mass_segregation.py`, `test_multicomponent.py`) and validation
tests (`tests/validation/test_segregation_equilibrium_physics.py`,
`test_multimass_equilibrium_physics.py`); the Λ_MSR *diagnostic* is
validated against analytic ground truth in
`tests/validation/test_mass_segregation_physics.py` (figures below).

## Quantifying mass segregation: $\Lambda_{\mathrm{MSR}}$

The {cite:t}`Allison2009` MST ratio is the diagnostic of choice for
quantifying segregation strength:

```{math}
:label: lambda-msr
\Lambda_{\mathrm{MSR}} \;=\; \frac{\langle\, \ell_{\mathrm{random}}\,\rangle}{\ell_{\mathrm{massive}}}
\;\pm\;\frac{\sigma_{\mathrm{random}}}{\ell_{\mathrm{massive}}}
```

[↗ model card](#card-lambda-msr)

where $\ell_{\mathrm{massive}}$ is the MST length of the $N_{\mathrm{m}}$
most massive stars (typically $N_{\mathrm{m}} = 10$–$20$),
$\langle\,\ell_{\mathrm{random}}\,\rangle$ is the mean MST length over
$\sim 50$–$200$ random subsets of the same size, and the error term is
the dispersion of the random subsets.

```{list-table} $\Lambda_{\mathrm{MSR}}$ regimes.
:header-rows: 1

* - $\Lambda_{\mathrm{MSR}}$
  - State
  - Meaning
* - $1.0 \pm \sigma$
  - No segregation
  - Massive stars distributed indistinguishably from random
* - $1.5$–$2.5$
  - Moderate
  - Typical Baumgardt $S = 1$ Plummer at $N_\star \approx 10^3$
* - $\gg 3$
  - Strong
  - All most-massive stars in inner core
```

The diagnostic is **validated against analytic ground truth** in
`tests/validation/test_mass_segregation_physics.py` (8 tests) with the human-facing
companion `scripts/validate_mass_segregation.py` — definition cross-checked against the
held {cite:t}`Allison2009` (ApJ 700 L99) PDF (see
[](../../99-bibliography/per-paper/allison-2009.md)).

:::{figure} ../../50-validation/figures/lambda_msr_regimes.png
:label: fig-lambda-msr-regimes
:width: 100%

$\Lambda_{\mathrm{MSR}}$ on three hand-constructed regimes with known answers:
unsegregated (random masses → $\Lambda\approx1$), maximally segregated (massive stars in a
tight core → $\Lambda\gg1$), and inverse (massive stars on the rim → $\Lambda<1$). ▲ marks the
$N_{\mathrm{massive}}$ set whose MST is $\ell_{\mathrm{massive}}$.
:::

:::{figure} ../../50-validation/figures/lambda_msr_monotonic_convergence.png
:label: fig-lambda-msr-convergence
:width: 100%

Left: $\Lambda_{\mathrm{MSR}}$ rises monotonically as the massive set is concentrated toward
the centre. Right: the estimator converges to the exact value (independent $N_{\mathrm{massive}}=2$
enumeration) with seed-to-seed scatter $\sigma(\Lambda)\propto 1/\sqrt{N_{\mathrm{random}}}$.
:::

```{warning}
**Λ_MSR is biased by binaries.** A massive binary appears as two stars
at the same position, giving an extremely short MST edge. For
binary-aware analyses, compute Λ_MSR on the *primary* mass rather than
the system mass, or use the local-density-ratio diagnostic
{cite:p}`Kuepper2011` which is less binary-sensitive. See the binary IMF
chapter ([](../imfs/binary.md)) for the related discussion of how
binary contamination affects single-star inferences.
```

:::{figure} ../../50-validation/figures/lambda_msr_binary_caveat.png
:label: fig-lambda-msr-binary
:width: 70%
:align: center

Quantifying the caveat: at $N_{\mathrm{massive}}=2$ a tight massive pair drives
$\Lambda_{\mathrm{MSR}}\propto 1/\text{(pair separation)}$, inflating it by $10^3$–$10^4$ as the
separation shrinks to $10^{-4}$ pc. The effect is milder at large $N_{\mathrm{massive}}$ (one of
$\sim\!N_{\mathrm{massive}}-1$ edges); mitigation is to use binary centre-of-mass positions.
:::

The implementation lives in `progenax.diagnostics.mass_segregation`,
which uses scipy's `minimum_spanning_tree` and is *not* part of the
JIT-able core IC pipeline (diagnostics are deliberately kept separate
from the differentiable core; see [JAX-native substructure design](../../20-architecture/jax-native-substructure-q.md) for
the broader JAX-native-MST design discussion).

## Other constructions (not implemented in progenax)

{cite:t}`Subr2008` parameterise mass segregation via interparticle
potential energies rather than total energies — physically equivalent in
limit but harder to implement without rejection sampling. progenax
adopts only the energy-ordered Baumgardt construction because:

1. It avoids rejection sampling (deterministic given a fixed orbit pool).
2. It composes with any equilibrium orbit pool (Plummer, King, EFF,
   LIMEPY) through the protocol API.
3. Continuous, differentiable segregation control is already provided by
   the separate equilibrium route
   (`MultiComponentCluster.from_mass_segregation`), so the primordial
   generator can stay an honest discrete construction.

The {cite:t}`Subr2008` construction may be added as an alternative
backend in a future version if science needs require it.

## Implementation in progenax

Both routes are public in `progenax`:

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import (
    MultiComponentCluster,
    PlummerProfile,
    PlummerVelocityDF,
    energy_sorted_segregation,
)
from progenax.profiles import compute_profile_potential

# --- Route 1: equilibrium segregation (differentiable in delta) ---
cluster = MultiComponentCluster.from_mass_segregation(
    alpha_j=jnp.array([0.5, 0.5]),
    m_j=jnp.array([0.3, 1.0]),
    W0=7.0, g=1.0, delta=0.5,
)
ic = cluster.sample_cluster(jax.random.PRNGKey(42), n_stars=1000, G=STELLAR.G)

# --- Route 2: primordial energy-ordered segregation (S = 1) ---
key_pos, key_vel, key_seg = jax.random.split(jax.random.PRNGKey(0), 3)
N, pool_factor = 1000, 4
masses = jnp.ones(N)  # or an IMF draw
pool_masses = jnp.ones(N * pool_factor)
profile = PlummerProfile(r_h=1.0)
df = PlummerVelocityDF(r_h=1.0)
pos_pool = profile.sample_positions(pool_masses, key_pos)
vel_pool = df.sample_velocities(pos_pool, pool_masses, key_vel, G=STELLAR.G)
masses_out, positions, velocities = energy_sorted_segregation(
    key_seg, masses, pos_pool, vel_pool,
    potential_fn=lambda p: compute_profile_potential(
        p, "plummer", jnp.sum(masses), 1.0, STELLAR.G),
)
```

The primordial sorter implements the discrete energy-ordered assignment
and is not itself differentiable; for gradient-based inference over
segregation strength use Route 1 ($\delta$ is the differentiable knob).
See [](../../50-validation/mass-segregation.md) for the current
validation status page.

:::{figure} ../../50-validation/figures/cluster_ic_energy_sorted_segregation.png
:label: fig-energy-sorted-segregation
:width: 100%

End-to-end check of `energy_sorted_segregation` (`scripts/validate_cluster_ic.py`): applying the
energy-ordered assignment to an unsegregated Kroupa-IMF Plummer pool drives the *independently
validated* $\Lambda_{\mathrm{MSR}}$ from $1.40$ to $14.3$ (left), and the most massive stars occupy
the most-bound orbits (right, Spearman $\rho(m,E)=-0.84$) — the construction described above produces
real, $\Lambda_{\mathrm{MSR}}$-detectable mass segregation.
:::

## Relation to fractal substructure

Mass segregation and fractal/clumpy substructure are conceptually
orthogonal pathways, but they live in different packages: turbulent
substructure ICs are the experimental, repo-only `gravoturb_fdf`
package, while both segregation routes are released progenax. Defining
"most bound" orbits inside a strongly clumpy potential is not
implemented anywhere; for most production use cases, choose *one* of:

- Clumpy + subvirial ($Q_{\mathrm{vir}} \approx 0.3$, experimental
  `gravoturb_fdf` ICs) — let dynamical segregation emerge during
  evolution {cite:p}`Allison2009`.
- Smooth + segregated ($Q_{\mathrm{vir}} = 0.5$,
  `MultiComponentCluster.from_mass_segregation` or
  `energy_sorted_segregation`) — primordial segregation in equilibrium.

See [](fractal.md) for the substructure side.

## Implementation, validation & references

- **In code:** the primordial energy-ordered generator is
  `energy_sorted_segregation` in
  `src/progenax/cluster/mass_segregation.py`; the differentiable
  equilibrium route is `MultiComponentCluster.from_mass_segregation` in
  `src/progenax/cluster/multicomponent.py`; the $\Lambda_{\mathrm{MSR}}$
  diagnostic is `src/progenax/diagnostics/mass_segregation.py`. See the
  [cluster API](../../30-api/cluster.md) and the
  [diagnostics API](../../30-api/diagnostics.md).
- **Validated in:** [mass segregation](../../50-validation/mass-segregation.md)
  — the $\Lambda_{\mathrm{MSR}}$ ground-truth suite and the end-to-end
  generator check; the equilibrium route is additionally pinned by
  [multimass equilibrium](../../50-validation/multimass-equilibrium.md).
- **Primary sources:** the energy-ordered construction follows
  {cite:t}`Baumgardt2008`; the partial-segregation $S$-shuffle follows
  {cite:t}`Kuepper2011`; the $\Lambda_{\mathrm{MSR}}$ diagnostic is
  {cite:t}`Allison2009`; the equipartition law is {cite:t}`Gieles2015`;
  {cite:t}`Subr2008` describes the alternative interparticle-energy
  construction not implemented here. Full notes in the
  [bibliography](../../99-bibliography/per-paper/baumgardt-2008.md).
