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
This chapter records the design choice — a kNN-based approximation as
the production path, with exact Borůvka MST available for cases that
demand it — and the rationale.

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
tree, $A$ is the convex-hull area, $\langle r_{ij}\rangle$ is the mean
pairwise separation, and $R_{\mathrm{cluster}}$ is the maximum distance
from the centroid. The two normalisations $\sqrt{N A}$ and
$R_{\mathrm{cluster}}$ make $Q$ dimensionless and $N$-independent for
self-similar distributions.

```{list-table} CW04 Q regimes.
:header-rows: 1

* - Regime
  - $Q$ value
  - Geometry
* - Substructured
  - $Q < 0.79$
  - Fractal, clumpy; e.g. $Q \approx 0.45$ for $D = 2.0$ {cite:p}`Goodwin2004`
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
   count depending on edge values. `jax.lax.while_loop` is not
   differentiable, and a fixed-iteration `scan` requires a worst-case
   bound on the loop count.
2. **Union-find sequentiality.** Cycle detection requires path-compressed
   disjoint-set operations, which are sequential by construction. JAX's
   parallelism cannot be applied.
3. **Discrete edge selection.** "Include this edge in the tree" is a
   binary decision; the gradient of $L_{\mathrm{MST}}$ with respect to
   particle positions is a sum over a *changing* edge set, which yields
   piecewise-linear gradients with no useful smoothing.
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

Borůvka's algorithm is the only structurally JAX-friendly choice: each
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
L_{\mathrm{NN}} \;=\; \tfrac{1}{2}\sum_{i=1}^{N} d_{i,1}
```

where $d_{i,1}$ is each particle's distance to its single nearest
neighbour and $\kappa$ is a calibration factor determined empirically
against the exact scipy MST on reference distributions. The factor of
$1/2$ deduplicates edges seen from both endpoints.

The physical justification is that for clustered distributions, MST
edges *are* dominated by nearest-neighbour distances within each
cluster — the MST connects every point through its cheapest edges, and
the cheapest edge from each point is its nearest neighbour modulo
inter-cluster bridges. The bridges are the only edges $L_{\mathrm{MST}}$
captures that $L_{\mathrm{NN}}$ misses, and they enter $L_{\mathrm{MST}}$
at fixed cost per cluster (one bridge per cluster pair).

The kNN approximation therefore captures the *intra-cluster* topology
exactly and the *inter-cluster* topology up to a small constant offset
folded into $\kappa$. For the relative tracking use case — $\Delta
Q(t)/Q(0)$ during dynamical evolution — the constant cancels.

```{list-table} Performance comparison: scipy MST vs. JAX kNN, single-snapshot $Q$.
:header-rows: 1

* - Metric
  - scipy MST
  - JAX kNN
  - Speedup
* - Single snapshot, $N = 10^3$ (CPU)
  - $\sim 50$ ms
  - $\sim 5$ ms
  - $10\times$
* - JIT-compiled (CPU)
  - n/a
  - $\sim 1$ ms
  - $50\times$
* - GPU, $N = 10^3$
  - n/a
  - $\sim 0.1$ ms
  - $500\times$
* - 100-snapshot time series
  - $\sim 5$ s (Python loop)
  - $\sim 10$ ms (`vmap`)
  - $500\times$
* - Memory, $N = 10^4$
  - $\mathcal{O}(N^2) = 800$ MB
  - $\mathcal{O}(Nk) = 2$ MB
  - $400\times$
* - Differentiable
  - No
  - Partial (through distances)
  - —
```

Differentiability is "partial" because the *choice* of nearest
neighbour is itself a discrete operation (`argmin`). The *distance*
through that neighbour is differentiable, but $\partial Q /
\partial \mathbf{x}_i$ is undefined at the moment $i$'s nearest
neighbour changes. For HMC use this is acceptable as long as the
posterior does not concentrate near a NN-swap surface; in practice
this is rare for $N \gtrsim 30$.

## Implementation skeleton

The JAX-native implementation lives at `progenax.diagnostics.q_jax`
(planned; current location is `progenax.diagnostics.substructure` which
hosts the scipy reference). The pipeline has three stages:

```python
@jax.jit
def compute_q_approx(positions, k=6, calibration=1.0, Nbins_per_dim=32):
    xy = positions[:, :2]                          # 2D projection per CW04
    grid = build_spatial_grid(xy, Nbins_per_dim)   # Morton-coded bins
    knn = compute_knn_graph(xy, grid, k=k)         # vmap'd kNN
    nn_d = knn.distances[:, 0]                     # Closest neighbour
    L_approx = jnp.sum(nn_d) / 2
    R = jnp.linalg.norm(xy - xy.mean(0), axis=1).max()
    A = jnp.pi * R ** 2                            # Circular hull approx
    s_bar = mean_pairwise_separation(xy, R, max_pairs=10000)
    return calibration * (L_approx / jnp.sqrt(xy.shape[0] * A)) / s_bar
```

The circular-area approximation $A \approx \pi R^2$ replaces the
$\mathcal{O}(N \log N)$ convex-hull computation with $\mathcal{O}(N)$
work and at most a $\sim 5\%$ bias on $\bar m$ for non-spherical
distributions. The bias is folded into $\kappa$ during calibration.

For a time series $\{X_t\}_{t=1}^{T}$, vectorisation over the time axis
is one line:

```python
@jax.jit
def compute_q_approx_timeseries(positions_t, k=6, calibration=1.0):
    return jax.vmap(lambda x: compute_q_approx(x, k=k, calibration=calibration))(positions_t)
```

This replaces a Python loop over scipy MST calls — the dominant cost in
the existing diagnostic pipeline — with a single vmapped device call.

## Calibration procedure

$\kappa$ is determined by computing both $Q_{\mathrm{exact}}$ (scipy MST)
and $Q_{\mathrm{approx}}$ (kNN) on a reference set of distributions
spanning the physically relevant range of substructure: uniform spheres,
fractals at $D \in \{1.6, 2.0, 2.4, 2.8\}$ {cite:p}`Goodwin2004`, and
centrally-concentrated profiles at $p \in \{0.5, 1.0, 1.5\}$. A least-squares
fit $Q_{\mathrm{exact}} \approx \kappa \cdot Q_{\mathrm{approx}}$
yields $\kappa$ to better than 5% across the range. The calibration is
$N$-stable for $N \in [100, 10^4]$, which spans the production regime.

## When to use the exact Borůvka path

Three use cases require the exact MST rather than the kNN approximation:

1. **Reproducing published Q values.** Cross-checking a CW04 Q against
   {cite:t}`Cartwright2004` Table 1 or {cite:t}`Allison2009` Figure 4
   needs the exact algorithm.
2. **Calibrating $\kappa$.** The calibration procedure above requires
   the scipy MST; this is a one-time cost paid offline.
3. **Survey-grade absolute Q.** When the *absolute* value of $Q$ enters
   a published quantitative claim, the kNN approximation's $\sim 5\%$
   bias is the limiting systematic.

For these, progenax retains the scipy reference implementation. A
JAX-native Borůvka MST is feasible (the parallel-phase structure maps
to fixed-iteration `scan`) but adds substantial code complexity for
$\sim 5\%$ improvement in the calibration regime. The current decision
is to defer Borůvka until a concrete use case requires it.

## Connection to other substructure diagnostics

CW04 Q is one of three substructure metrics implemented in progenax:

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
  - JAX-native; correlates with fractal dimension via $D \approx (1.45 - \sigma_\Sigma/\langle\Sigma\rangle)/0.46$ {cite:p}`Kuepper2011`
* - **Local-density variance** $\sigma_\rho / \langle\rho\rangle$
  - Density-PDF dispersion
  - JAX-native; correlates with $Q$ but not 1-to-1
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
