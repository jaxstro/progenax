---
title: Mass segregation
description: Energy-ordered (Baumgardt) primordial mass segregation with the McLuster S-shuffle, the λ_seg blending strategy that makes it differentiable, and the Λ_MSR diagnostic.
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

progenax implements primordial mass segregation via the energy-ordered
{cite:t}`Baumgardt2008` construction, with the partial-segregation
extension from {cite:t}`Kuepper2011`'s McLuster code. The key design
decision is to expose the segregation strength through a *smooth*
parameter $\lambda_{\mathrm{seg}} \in [0, 1]$ — differentiable through
`jax.grad`, rather than the discrete S parameter in McLuster — so that
it can sit inside an HMC posterior chain alongside $\alpha$, virial Q,
and the rest of the IC parameters.

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
bin with a random-bin draw. progenax v1 uses $S = 1$ internally and
exposes physical-strength control through the $\lambda_{\mathrm{seg}}$
parameter described next.
```

## The λ_seg blending strategy

progenax expects gradient-based inference of segregation strength as a
first-class use case. The discrete McLuster $S$ parameter is awkward
for this — its effect on observables is non-monotonic in the partial
regime, and the floor function in {eq}`s-shuffle` is non-differentiable
even where it is monotonic. progenax instead exposes a smooth blending
parameter

```{math}
:label: lambda-seg
(\mathbf{r}_i,\, \mathbf{v}_i)
\;=\; (1 - \lambda_{\mathrm{seg}})\,(\mathbf{r}_i,\, \mathbf{v}_i)_{\mathrm{baseline}}
\;+\; \lambda_{\mathrm{seg}}\,(\mathbf{r}_i,\, \mathbf{v}_i)_{\mathrm{Baumgardt}}
```

where the *baseline* is an unsegregated random rank-to-orbit mapping
and the *Baumgardt* state is the fully energy-ordered $S = 1$
construction above. $\lambda_{\mathrm{seg}} \in [0, 1]$ is a continuous
JAX-compatible parameter that produces the same Λ_MSR sweep as the
McLuster $S$ would have, but is differentiable end-to-end.

```{note}
The blend in {eq}`lambda-seg` is performed in *position-velocity space*,
not in *rank space*. A 50% blend means each star sits at a position
that is the linear combination of its baseline (random) and Baumgardt
(segregated) phase-space coordinates. This produces the smooth
parametric family of clusters that gradient-based inference needs.
```

Current unit tests in `tests/unit/cluster/test_cluster_ic.py`,
`tests/unit/cluster/test_mass_segregation.py`, and
`tests/unit/cluster/test_validation_suite.py` verify the implemented
behavior. There is not yet a dedicated
`tests/validation/test_mass_segregation.py` suite. The unit-backed
checks cover:

- $\lambda_{\mathrm{seg}} = 0$ gives $\Lambda_{\mathrm{MSR}} \approx 1$
  (no segregation) to within Poisson noise.
- $\lambda_{\mathrm{seg}} = 1$ recovers the Baumgardt prediction
  $\Lambda_{\mathrm{MSR}} \approx 1.7$–$2.5$ for typical Plummer
  parameters and $N_\star \in [10^3, 10^4]$.
- Intermediate $\lambda_{\mathrm{seg}}$ produces monotonically increasing
  $\Lambda_{\mathrm{MSR}}$, suitable for gradient inference.

## Quantifying mass segregation: $\Lambda_{\mathrm{MSR}}$

The {cite:t}`Allison2009` MST ratio is the diagnostic of choice for
quantifying segregation strength:

```{math}
:label: lambda-msr
\Lambda_{\mathrm{MSR}} \;=\; \frac{\langle\, \ell_{\mathrm{random}}\,\rangle}{\ell_{\mathrm{massive}}}
\;\pm\;\frac{\sigma_{\mathrm{random}}}{\ell_{\mathrm{massive}}}
```

where $\ell_{\mathrm{massive}}$ is the MST length of the $N_{\mathrm{m}}$
most massive stars (typically $N_{\mathrm{m}} = 10$–$20$),
$\langle\,\ell_{\mathrm{random}}\,\rangle$ is the mean MST length over
$\sim 50$–$200$ random subsets of the same size, and the error term is
the dispersion of the random subsets.

```{list-table} $\Lambda_{\mathrm{MSR}}$ regimes.
:header-rows: 1
:widths: 22 18 60

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

```{warning}
**Λ_MSR is biased by binaries.** A massive binary appears as two stars
at the same position, giving an extremely short MST edge. For
binary-aware analyses, compute Λ_MSR on the *primary* mass rather than
the system mass, or use the local-density-ratio diagnostic
{cite:p}`Kuepper2011` which is less binary-sensitive. See the binary IMF
chapter ([](../imfs/binary.md)) for the related discussion of how
binary contamination affects single-star inferences.
```

The implementation lives in `progenax.diagnostics.mass_segregation`,
which uses scipy's `minimum_spanning_tree` and is *not* part of the
JIT-able core IC pipeline (diagnostics are deliberately kept separate
from the differentiable core; see [](jax-native-substructure-q.md) for
the broader JAX-native-MST design discussion).

## Other constructions (not implemented in progenax)

{cite:t}`Subr2008` parameterise mass segregation via interparticle
potential energies rather than total energies — physically equivalent in
limit but harder to implement without rejection sampling. progenax
adopts only the energy-ordered Baumgardt construction because:

1. It avoids rejection sampling (deterministic given a fixed orbit pool).
2. It pairs naturally with $\lambda_{\mathrm{seg}}$ blending.
3. It maps cleanly onto fixed-iteration `jax.lax.scan`, the standard
   differentiable-loop pattern (see
   [](../../20-architecture/jax-native-philosophy.md)).

The {cite:t}`Subr2008` construction may be added as an alternative
backend in a future version if science needs require it.

## Implementation in progenax

The public high-level pipeline lives in `progenax.cluster`:

```python
import jax
from jaxstro.units import STELLAR
from progenax.cluster import (
    MassSegregationLayer,
    SpatialStructureParams,
    generate_cluster_ic,
)
from progenax.imf import PowerLawIMF

cluster = generate_cluster_ic(
    key=jax.random.PRNGKey(42),
    N_stars=1000,
    M_total=1000.0,
    R_half=1.0,
    imf_params=PowerLawIMF.kroupa(),
    structure_params=SpatialStructureParams(
        base_profile="plummer",
        mass_segregation=MassSegregationLayer(
            lambda_seg=0.7,
            pool_factor=4,
        ),
    ),
    G=STELLAR.G,
)
```

The lower-level sorter is
`progenax.cluster.mass_segregation.energy_sorted_segregation`. That
function implements the discrete energy-ordered assignment and is not
itself differentiable; smooth control comes from blending the
segregated catalog with the unsegregated baseline through
`MassSegregationLayer.lambda_seg`. See
[](../../50-validation/mass-segregation.md) for the current validation
status page.

## Composing with fractal substructure

Mass segregation and fractal substructure are separate layers in the
design, but the current `generate_cluster_ic` implementation allows
only one of them at a time. Passing both `mass_segregation` and
`fractal` raises `ValueError`, because defining the "most bound" orbits
inside a strongly clumpy potential is not yet implemented.

For most production use cases, choose *one* of:

- $\lambda_{\mathrm{frac}} > 0,\,\lambda_{\mathrm{seg}} = 0,\, Q_{\mathrm{vir}} = 0.3$ — let dynamical segregation emerge during evolution.
- $\lambda_{\mathrm{frac}} = 0,\,\lambda_{\mathrm{seg}} > 0,\, Q_{\mathrm{vir}} = 0.5$ — primordial segregation in a smooth profile.

See [](fractal.md) for the fractal-substructure side of the layered
construction.

## References

The energy-ordered construction follows {cite:t}`Baumgardt2008`; the
partial-segregation S-shuffle follows {cite:t}`Kuepper2011`; the
diagnostic $\Lambda_{\mathrm{MSR}}$ is from {cite:t}`Allison2009`.
{cite:t}`Subr2008` describes the alternative interparticle-energy
construction not implemented here. The original observational evidence
for primordial segregation in globular clusters is the focus of
{cite:t}`Baumgardt2008`.
