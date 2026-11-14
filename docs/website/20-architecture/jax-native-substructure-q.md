---
title: JAX-native CW04 substructure Q parameter
description: Design rationale and JAX-native implementation strategy for the Cartwright & Whitworth (2004) substructure Q parameter, including kNN-based approximation and exact Borůvka MST.
---

# JAX-native CW04 substructure Q parameter

```{seealso}
This chapter is about the {cite:t}`Cartwright2004` *substructure* Q
parameter — an MST-based diagnostic of clustering geometry. It is *not*
the *virial* Q = T/|V| convention used elsewhere in progenax, which is
documented in [](q-virial-convention.md). The two share a letter and
nothing else.
```

The {cite:t}`Cartwright2004` Q parameter quantifies the spatial
substructure of a stellar distribution as a single scalar derived from
the minimum spanning tree (MST). progenax computes Q at multiple stages —
during initial-condition validation, during dynamical evolution
diagnostics, and during fractal-substructure calibration
{cite:p}`Goodwin2004,Allison2009`. The reference scipy implementation in
`progenax.diagnostics.substructure.compute_q_parameter` is correct but
non-differentiable, non-JIT-compatible, and $\mathcal{O}(N^2)$ in memory.
This chapter records the design choice — a kNN-based JAX approximation alongside the exact SciPy reference;
a JAX Borůvka implementation remains a possible future path — and the rationale.

## What CW04 Q measures

Given $N$ stellar positions projected onto a plane, the CW04 Q is

```{math}
:label: q-cw04
Q \;\equiv\; \frac{\bar m}{\bar s},
\qquad
\bar m \;=\; \frac{L_{\mathrm{MST}}}{\sqrt{N\,A}},
\qquad
\bar s \;=\; \frac{\langle r_{ij}\rangle}{R_{\mathrm{cluster}}}
```

where $L_{\mathrm{MST}}$ is the total length of the 2D minimum spanning
tree, $A=\pi R_{\mathrm{cluster}}^2$ is the enclosing-circle area used here, $\langle r_{ij}\rangle$ is the mean
pairwise separation, and $R_{\mathrm{cluster}}$ is the maximum distance
from the centroid. The two normalizations $\sqrt{N A}$ and
$R_{\mathrm{cluster}}$ make $Q$ dimensionless and $N$-independent for
self-similar distributions.

```{list-table} CW04 Q regimes.
:header-rows: 1

* - Regime
  - $Q$ value
  - Geometry
* - Substructured
  - $Q < 0.79$
  - Fractal, clumpy; e.g. $Q \approx 0.61$ for $D = 2.0$ {cite:p}`Goodwin2004`
* - Uniform sphere
  - $Q \approx 0.79$
  - Reference baseline
* - Centrally concentrated
  - $Q > 0.79$
  - Power-law radial profile
```

For dynamical N-body evolution, $Q(t)$ traces the rate at which initial
substructure is erased by two-body relaxation and dynamical mixing
{cite:p}`Allison2009,Kuepper2011`. Tracking $\Delta Q / Q$ as a function
of crossing time is therefore a primary diagnostic of cluster
violent-relaxation and tidal evolution.

## Why MST is hard in JAX

A naive port of the scipy implementation to JAX runs into four
fundamental obstructions:

1. **Data-dependent control flow.** Both Prim's and Kruskal's
   algorithms iterate until the tree completes, with the iteration
   count depending on edge values. ordinary reverse-mode AD through a dynamic `jax.lax.while_loop` is
   unavailable without a custom derivative rule, and a fixed-iteration `scan` requires a worst-case
   bound on the loop count.
2. **Union-find sequentiality.** Cycle detection requires path-compressed
   disjoint-set operations, which are sequential by construction. JAX's
   parallelism cannot be applied.
3. **Discrete edge selection.** "Include this edge in the tree" is a
   binary decision; the gradient of $L_{\mathrm{MST}}$ with respect to
   particle positions is a sum over a *changing* edge set, which gives a piecewise differentiable length. Ties and changes in the selected
   tree can create nonsmooth boundaries; an exact derivative is distinct
   from a smoothed surrogate.
4. **Memory scaling.** A naive distance matrix is $\mathcal{O}(N^2)$ in
   memory, exhausting GPU memory at $N \gtrsim 10^4$.

```{list-table} MST algorithms ranked for JAX feasibility.
:header-rows: 1

* - Algorithm
  - Parallelisable?
  - JAX feasibility
  - Notes
* - **Prim's**
  - No (sequential priority queue)
  - Poor
  - Cannot vmap; data-dependent extract-min
* - **Kruskal's**
  - No (sequential union-find)
  - Poor
  - Same union-find bottleneck
* - **Borůvka's**
  - Yes (phases parallel)
  - Partial
  - $\mathcal{O}(\log N)$ phases, each parallel; requires fixed-iteration `scan`
* - **Dual-tree Borůvka**
  - Yes (Morton-coded)
  - Complex
  - Best asymptotic performance, hardest to write
```

Borůvka's algorithm is a possible parallel formulation: each
phase does $\mathcal{O}(E)$ parallel edge comparisons, components halve
each phase, and the total depth is $\mathcal{O}(\log N)$. Its weakness
is the per-phase union-find, which still has sequential bottlenecks
even with path compression replaced by `scan`-based merges.

## The kNN-based approximation

The production path in progenax avoids the MST entirely and approximates
$Q$ from a $k$-nearest-neighbour graph:

```{math}
:label: q-knn
Q_{\mathrm{approx}} \;=\; \kappa \cdot \frac{L_{\mathrm{NN}} / \sqrt{N\,A}}{\bar s},
\qquad
L_{\mathrm{NN}} \;=\; (N-1)\,\frac{1}{N}\sum_{i=1}^{N} d_{i,1}
```

where $d_{i,1}$ is each particle's nearest-neighbor distance and
$\kappa$ is a calibration factor. This approximates an MST's average edge
length with the mean nearest-neighbor distance; it is not an exact graph
identity. Dividing a sum by two would only deduplicate mutual nearest-neighbor
edges, and does not reproduce the implemented expression.

The approximation misses connections between separated groups and does not
reproduce all intra-group MST edges. Its error can depend on density shape,
clustering, geometry and particle count. A multiplicative calibration measured
on uniform spheres need not transfer to other populations, and an additive or
shape-dependent discrepancy does not cancel in a time-series ratio.

```{list-table} Implemented paths and performance scope.
:header-rows: 1

* - Path
  - Computation
  - Limit
* - Exact SciPy reference
  - Pairwise distances and sparse-graph MST
  - Quadratic distance storage; host-side diagnostic
* - JAX naive approximation
  - All-pairs nearest neighbor and mean separation
  - Quadratic storage; piecewise derivatives through distances
* - JAX fast approximation
  - Spatial-bin candidate nearest neighbors and deterministic pair subsampling
  - Accuracy depends on candidate coverage and pair sampling
* - Timing and peak memory
  - Depend on shape, precision, hardware, compilation and batching
  - This design page does not establish the former projected speedup numbers
```

Gradients flow through selected distances away from ties and zero-separation
singularities. Nearest-neighbor changes and spatial-bin/candidate changes can
introduce boundaries. Differentiability of a diagnostic does not establish
HMC mixing, likelihood accuracy, or a calibrated posterior; there is no
particle-count threshold that guarantees those properties.

## Implementation skeleton

The current owner is `progenax.diagnostics.q_approx`, which exports
`q_approx`, `q_approx_naive`, `q_approx_fast`, and `calibrate_q_approx`.
`method="auto"` selects the naive path for $N\le1000$ and the fast path
above that size. The fast path uses spatial bins and candidate neighbors;
it estimates mean separation from a fixed deterministic pair sample when
the full pair count exceeds 10,000. This avoids materializing the full
distance matrix on that path, but adds approximation error.

```python
import jax
from progenax.diagnostics.q_approx import q_approx

@jax.jit
def approximate_q(positions):
    return q_approx(positions, method="auto")
```

Both the exact reference and approximate paths use $A=\pi R^2$. There is
no convex-hull-to-circle correction to calibrate between those implemented
owners. Projection onto the first two coordinates is the default; the optional
unprojected calculation is not automatically the published projected statistic.

For a fixed-shape time series, the same function can be vectorized:

```python
@jax.jit
def approximate_q_timeseries(positions_t):
    return jax.vmap(approximate_q)(positions_t)
```

Batching may increase peak memory and compilation cost. Measure the actual
workload before choosing a time-series batch size or claiming a speedup.

## Calibration procedure

The existing `calibrate_q_approx` helper compares exact and approximate Q
on uniform-sphere realizations. The source records the default
$\kappa=1.287344$ from 100 realizations of 500 stars at seed 42.
That is a retained calibration of this reference population. It does not
establish a five-percent bound across fractal families, centrally concentrated
profiles, particle counts, or arbitrary input geometry. Validation for a new
scientific population must retain an exact-reference comparison and an
appropriate error criterion.

## When to use the exact Borůvka path

Three use cases require the exact MST rather than the kNN approximation:

1. **Reproducing published Q values.** Cross-checking a CW04 Q against
   {cite:t}`Cartwright2004` Table 1 or {cite:t}`Allison2009` Figure 4
   needs the exact algorithm.
2. **Calibrating $\kappa$.** The calibration procedure above requires
   the scipy MST; this is a one-time cost paid offline.
3. **Survey-grade absolute Q.** When the *absolute* value of $Q$ enters
   a published quantitative claim, the approximation error must be measured for the target population.

For these, progenax retains the scipy reference implementation. A
JAX-native Borůvka MST is feasible (the parallel-phase structure maps
to fixed-iteration `scan`) but adds substantial code complexity whose benefit and cost need measurement for the intended use. The current decision
is to defer Borůvka until a concrete use case requires it.

## Connection to other substructure diagnostics

The released substructure owner supplies CW04 Q and azimuthal variation:

```{list-table}
:header-rows: 1

* - Metric
  - What it measures
  - Status in progenax
* - **CW04 Q** (this chapter)
  - MST + mean separation
  - scipy reference + JAX kNN approximation
* - **Azimuthal variation** $\sigma_\Sigma / \langle\Sigma\rangle$
  - Surface-density fluctuation
  - NumPy reference; a family-specific empirical relation is $D \approx (1.45 - \sigma_\Sigma/\langle\Sigma\rangle)/0.46$ {cite:p}`Kuepper2011`
```

Q and the azimuthal-variation metric agree on the *trend* with
substructure but use different absolute scales. The fractal-substructure
chapter ([](../10-theory/tidal-and-substructure/fractal.md)) lays out
the conversion table.

## References

The CW04 Q definition follows {cite:t}`Cartwright2004`. The kNN
approximation is original to this work; the parallel Borůvka structure
is standard ({cite:t}`Cartwright2004`'s reference list cites the 1926
original). The fractal calibration uses {cite:t}`Goodwin2004`; the
azimuthal-variation alternative uses {cite:t}`Kuepper2011`.
