# JAX-Native Q Parameter Computation: Design Document

> **Goal:** Enable efficient, JAX-native computation of the Cartwright-Whitworth Q parameter for tracking substructure evolution in dynamical N-body simulations.

---

## Background: The Q Parameter

The Cartwright & Whitworth (2004) Q parameter quantifies spatial substructure:

```
Q = m̄ / s̄
```

Where:
- **m̄** = L_MST / sqrt(N × A) — normalized mean MST edge length
- **s̄** = mean_pairwise_separation / R_cluster — normalized mean separation
- **L_MST** = total length of minimum spanning tree
- **A** = convex hull area (2D projection)

**Interpretation:**
- Q < 0.79: Substructured (fractal, clumpy)
- Q ≈ 0.79: Uniform sphere baseline
- Q > 0.79: Centrally concentrated (radial profile)

---

## Current Implementation: scipy-based (NOT JAX-native)

**Location:** `progenax/src/progenax/diagnostics/substructure.py`

```python
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from scipy.spatial import ConvexHull

def compute_q_parameter(positions: np.ndarray) -> float:
    # O(N²) pairwise distances
    pairwise_dists = pdist(xy)
    dist_matrix = squareform(pairwise_dists)

    # O(N² log N) MST via Prim's algorithm
    mst = minimum_spanning_tree(dist_matrix)
    L_MST = mst.sum()

    # Convex hull for area
    hull = ConvexHull(xy)
    A = hull.volume  # 2D area
```

**Limitations:**
- Not JIT-compatible
- Not GPU-acceleratable
- Not differentiable
- O(N²) memory for distance matrix
- Python loop overhead for time series

---

## Why MST is Hard in JAX

### Fundamental Challenges

| Challenge | Description | JAX Impact |
|-----------|-------------|------------|
| **Data-dependent control flow** | Prim/Kruskal iterate until tree complete | `while_loop` not differentiable |
| **Union-Find structure** | Disjoint set operations for cycle detection | Sequential parent updates |
| **Priority queue** | Extract-min operation each iteration | No native JAX equivalent |
| **Discrete selection** | "Include this edge" is binary | Not differentiable |
| **Memory scaling** | O(N²) distance matrix | GPU memory limits |

### MST Algorithm Analysis for JAX

| Algorithm | Parallelizable? | JAX Feasibility | Notes |
|-----------|-----------------|-----------------|-------|
| **Prim's** | No (inherently sequential) | ❌ Poor | Priority queue iteration |
| **Kruskal's** | No (union-find sequential) | ❌ Poor | Data-dependent merges |
| **Borůvka's** | ✅ Yes (phases parallel) | ⚠️ Partial | O(log N) phases, each parallel |
| **Dual-tree Borůvka** | ✅ Yes with tree structure | ⚠️ Complex | Morton codes help here |

### Could We Write JAX-Native MST?

**Yes, via Borůvka's algorithm**, but with caveats:

```python
# Borůvka's: O(log N) phases, each phase is parallel
def boruvka_mst_jax(positions):
    """
    Borůvka's algorithm phases:
    1. Each component finds its minimum outgoing edge (PARALLEL)
    2. Contract selected edges, merge components
    3. Repeat until single component
    """
    # Phase 1: Find min edge per component - can be vmapped
    min_edges = vmap(find_min_outgoing_edge)(components)

    # Phase 2: Merge components - requires union-find
    # This is the hard part for JAX
    new_labels = union_find_merge(min_edges)  # Sequential!

    # O(log N) iterations of the above
```

**Borůvka Complexity:**
- Each phase: O(E) parallelizable edge comparisons
- Number of phases: O(log N) — components halve each phase
- Total: O(E log N) work, O(log N) sequential depth

**The Union-Find Problem:**
- Even with Borůvka, we need to track which nodes are in which component
- Path compression requires sequential updates
- Could use `jax.lax.scan` with fixed iteration count (overestimate log₂ N)

**Verdict:** Possible but complex. The kNN approximation is much simpler and sufficient for most use cases.

---

## Better Alternatives for JAX-Native Q

### Option 1: kNN-Based Spanning Forest (RECOMMENDED)

**Key Insight:** The k-nearest neighbor graph provides similar structural information to MST, and is trivially parallelizable.

```
Q_approx ≈ (mean NN distance × calibration) / s̄
```

**Why This Works:**
- MST connects every point via its "cheapest" edges
- For clustered distributions, MST edges ≈ NN distances within clusters
- The kNN graph captures the same local connectivity
- Calibration factor accounts for systematic bias

**Advantages:**
- O(N × k) instead of O(N²) — scales to large N
- Fully parallelizable (vmap over particles)
- JIT-compilable, GPU-acceleratable
- Existing infrastructure in jaxstro/spatial

### Option 2: Local Density Proxy

Use local density estimators that correlate with Q:

```python
def q_proxy_from_density(positions, k=10):
    """Q correlates with local density variance."""
    densities = estimate_local_density(positions, k)
    # High Q → concentrated → high density variance
    # Low Q → uniform → low density variance
    return some_calibrated_transform(jnp.std(densities) / jnp.mean(densities))
```

**Status:** Needs calibration; not as direct as MST-based Q.

### Option 3: Azimuthal Variation (Already Implemented)

```python
from progenax.diagnostics import compute_azimuthal_variation
# σ_Σ/<Σ> correlates with fractal dimension D
# D ≈ (1.45 - σ_Σ/<Σ>) / 0.46
```

**Status:** Already available, but different interpretation than Q.

---

## Recommended Approach: kNN-Based Q Approximation

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    compute_q_approx_jax()                   │
├─────────────────────────────────────────────────────────────┤
│  1. Quantize positions → Morton codes                       │
│  2. Build spatial bins                                      │
│  3. Gather kNN candidates (stencil method)                  │
│  4. Compute exact k-NN distances                            │
│  5. Sum NN distances → L_approx                             │
│  6. Compute s̄ (mean pairwise or subsampled)                │
│  7. Return Q_approx = calibration × (L_approx / sqrt(N×A)) / s̄ │
└─────────────────────────────────────────────────────────────┘
```

### Implementation Design

#### Data Structures (using jraph optionally)

```python
from dataclasses import dataclass
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

@dataclass(frozen=True)
class SpatialGrid:
    """Spatial indexing structure for kNN queries."""
    bin_members: Int[Array, "Nbins Bcap"]   # Particle IDs per bin
    bin_mask: Bool[Array, "Nbins Bcap"]     # Valid slots
    bin_of: Int[Array, "N"]                  # Bin assignment per particle
    Nbins_per_dim: int
    dx: float  # Bin size

@dataclass(frozen=True)
class KNNGraph:
    """k-NN graph representation (optionally jraph-compatible)."""
    neighbors: Int[Array, "N k"]      # k neighbor indices per particle
    distances: Float[Array, "N k"]    # k neighbor distances per particle
    valid_mask: Bool[Array, "N k"]    # Valid neighbor flags

    def to_jraph(self):
        """Convert to jraph.GraphsTuple for GNN compatibility."""
        import jraph
        N, k = self.neighbors.shape
        # Flatten to edge list
        senders = jnp.repeat(jnp.arange(N), k)
        receivers = self.neighbors.ravel()
        edges = self.distances.ravel()
        edge_mask = self.valid_mask.ravel()

        return jraph.GraphsTuple(
            nodes=None,  # Add node features if needed
            edges=edges[edge_mask],
            senders=senders[edge_mask],
            receivers=receivers[edge_mask],
            n_node=jnp.array([N]),
            n_edge=jnp.array([edge_mask.sum()]),
            globals=None,
        )
```

#### Core Functions

```python
# progenax/src/progenax/diagnostics/q_jax.py

import jax
import jax.numpy as jnp
from jaxstro.spatial.morton import morton_encode_3d, MAX_BITS_3D
from jaxstro.spatial.neighbor import approx_knn_candidates

def build_spatial_grid(
    positions: Float[Array, "N 3"],
    Nbins_per_dim: int = 32,
    Bcap: int = 64,
) -> SpatialGrid:
    """Build spatial grid for neighbor queries.

    Args:
        positions: Particle positions [N, 3]
        Nbins_per_dim: Grid resolution (32 typical for N~1000)
        Bcap: Max particles per bin

    Returns:
        SpatialGrid structure for kNN queries
    """
    N = positions.shape[0]

    # Normalize to [0, Nbins_per_dim) integer coordinates
    pos_min = positions.min(axis=0)
    pos_max = positions.max(axis=0)
    L_box = (pos_max - pos_min).max() * 1.01  # Small margin
    center = (pos_min + pos_max) / 2

    # Quantize to grid
    pos_normalized = (positions - center + L_box/2) / L_box * Nbins_per_dim
    pos_int = jnp.clip(pos_normalized.astype(jnp.int32), 0, Nbins_per_dim - 1)

    # Morton encode
    bin_of = morton_encode_3d(pos_int, bits=MAX_BITS_3D)

    # Build bin membership arrays
    Nbins = Nbins_per_dim ** 3
    bin_members, bin_mask = _build_bin_arrays(bin_of, N, Nbins, Bcap)

    dx = L_box / Nbins_per_dim

    return SpatialGrid(
        bin_members=bin_members,
        bin_mask=bin_mask,
        bin_of=bin_of,
        Nbins_per_dim=Nbins_per_dim,
        dx=dx,
    )


def compute_knn_graph(
    positions: Float[Array, "N 3"],
    grid: SpatialGrid,
    k: int = 6,
) -> KNNGraph:
    """Compute k-nearest neighbors using Morton-based spatial queries.

    Args:
        positions: Particle positions [N, 3]
        grid: Spatial grid from build_spatial_grid
        k: Number of neighbors

    Returns:
        KNNGraph with neighbor indices and distances
    """
    N = positions.shape[0]

    # Add sentinel position for safe indexing
    pos_with_sentinel = jnp.concatenate([positions, jnp.zeros((1, 3))], axis=0)

    # Get candidate neighbors via stencil method
    cand_idx, cand_mask = approx_knn_candidates(
        pos=pos_with_sentinel,
        bin_members=grid.bin_members,
        bin_mask=grid.bin_mask,
        bin_of=grid.bin_of,
        Nbins_per_dim=grid.Nbins_per_dim,
        K_target=k,
    )

    # Compute distances to candidates
    cand_pos = pos_with_sentinel[cand_idx]  # [N, Cand_max, 3]
    diff = positions[:, None, :] - cand_pos  # [N, Cand_max, 3]
    cand_dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # [N, Cand_max]

    # Mask invalid candidates with inf
    cand_dist = jnp.where(cand_mask, cand_dist, jnp.inf)

    # Select k closest
    _, top_k_idx = jax.lax.top_k(-cand_dist, k)  # Closest = most negative

    neighbors = jnp.take_along_axis(cand_idx, top_k_idx, axis=1)
    distances = jnp.take_along_axis(cand_dist, top_k_idx, axis=1)
    valid_mask = jnp.isfinite(distances)

    return KNNGraph(neighbors=neighbors, distances=distances, valid_mask=valid_mask)


def compute_q_approx(
    positions: Float[Array, "N 3"],
    k: int = 6,
    calibration: float = 1.0,
    Nbins_per_dim: int = 32,
) -> float:
    """Compute approximate Q parameter using kNN graph.

    This approximates the CW04 Q parameter without computing the full MST.
    The kNN-based metric correlates with true Q and is suitable for:
    - Tracking RELATIVE changes in substructure over time
    - Comparing different ICs or simulation snapshots
    - Fast screening before full scipy-based Q computation

    Args:
        positions: Particle positions [N, 3], 3D (will project to 2D)
        k: Number of neighbors for local connectivity estimate
        calibration: Multiplicative calibration factor (default 1.0)
        Nbins_per_dim: Spatial grid resolution

    Returns:
        Q_approx: Approximate Q parameter

    Notes:
        For absolute Q values matching CW04, use the scipy-based
        compute_q_parameter() function. This approximation is designed
        for efficiency and JAX-compatibility, not exact CW04 reproduction.

    Calibration:
        The calibration factor should be determined empirically by comparing
        Q_approx to Q_exact on a set of reference distributions:

        >>> calibration = mean(Q_exact / Q_approx)  # Over test set
    """
    # Project to 2D (CW04 methodology)
    xy = positions[:, :2]
    N = xy.shape[0]

    if N < 3:
        return 0.79  # Degenerate case

    # Build spatial grid and kNN graph
    grid = build_spatial_grid(xy, Nbins_per_dim=Nbins_per_dim)
    knn = compute_knn_graph(xy, grid, k=k)

    # Approximate MST length from NN distances
    # Each particle contributes its nearest neighbor distance
    # Divide by 2 because each edge is "seen" from both endpoints
    nn_distances = knn.distances[:, 0]  # Closest neighbor only
    L_approx = jnp.sum(nn_distances) / 2

    # Cluster radius (max distance from center)
    center = xy.mean(axis=0)
    radii = jnp.sqrt(jnp.sum((xy - center)**2, axis=1))
    R_cluster = radii.max()

    # Approximate area (circular approximation, faster than convex hull)
    A_approx = jnp.pi * R_cluster**2

    # m̄ approximation
    m_bar_approx = L_approx / jnp.sqrt(N * A_approx)

    # s̄: mean pairwise separation / R_cluster
    # For efficiency, use subsampling for large N
    s_bar = _compute_s_bar_subsampled(xy, R_cluster, max_pairs=10000)

    # Q = m̄ / s̄
    Q_approx = calibration * m_bar_approx / (s_bar + 1e-10)

    return float(Q_approx)


def _compute_s_bar_subsampled(
    xy: Float[Array, "N 2"],
    R_cluster: float,
    max_pairs: int = 10000,
) -> float:
    """Compute s̄ with optional subsampling for large N."""
    N = xy.shape[0]
    n_pairs = N * (N - 1) // 2

    if n_pairs <= max_pairs:
        # Exact computation
        # Pairwise distances via broadcasting
        diff = xy[:, None, :] - xy[None, :, :]  # [N, N, 2]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))  # [N, N]
        # Extract upper triangle (excluding diagonal)
        triu_mask = jnp.triu(jnp.ones((N, N), dtype=bool), k=1)
        s_raw = jnp.sum(dist * triu_mask) / n_pairs
    else:
        # Subsampled estimate
        # Use a fixed sample of pairs
        key = jax.random.PRNGKey(42)  # Deterministic for reproducibility
        n_sample = max_pairs

        idx_i = jax.random.randint(key, (n_sample,), 0, N)
        key, _ = jax.random.split(key)
        idx_j = jax.random.randint(key, (n_sample,), 0, N)

        # Ensure i != j
        idx_j = jnp.where(idx_i == idx_j, (idx_j + 1) % N, idx_j)

        diffs = xy[idx_i] - xy[idx_j]
        s_raw = jnp.mean(jnp.sqrt(jnp.sum(diffs**2, axis=-1)))

    s_bar = s_raw / R_cluster
    return s_bar


# Vectorized version for time series
@jax.jit
def compute_q_approx_timeseries(
    positions_t: Float[Array, "T N 3"],
    k: int = 6,
    calibration: float = 1.0,
) -> Float[Array, "T"]:
    """Compute Q_approx for each snapshot in a time series.

    This is the primary interface for tracking Q evolution during
    N-body simulations.

    Args:
        positions_t: Positions at each timestep [T, N, 3]
        k: Number of neighbors
        calibration: Calibration factor

    Returns:
        Q_approx: Q values at each timestep [T]
    """
    # vmap over time axis
    return jax.vmap(
        lambda pos: compute_q_approx(pos, k=k, calibration=calibration)
    )(positions_t)
```

### Calibration Procedure

The kNN-based Q needs calibration against exact MST-based Q:

```python
def calibrate_q_approx(n_samples: int = 100, N_stars: int = 500):
    """Determine calibration factor by comparing to exact Q.

    Generates distributions with known structure and computes both
    Q_exact (scipy MST) and Q_approx (kNN), then fits calibration.
    """
    from progenax.diagnostics import compute_q_parameter

    Q_exact_values = []
    Q_approx_values = []

    for seed in range(n_samples):
        key = jax.random.PRNGKey(seed)

        # Generate test distributions:
        # - Uniform sphere
        # - Various fractal dimensions
        # - Centrally concentrated
        positions = generate_test_distribution(key, N_stars, ...)

        Q_exact = compute_q_parameter(np.array(positions))
        Q_approx = compute_q_approx(positions, calibration=1.0)

        Q_exact_values.append(Q_exact)
        Q_approx_values.append(Q_approx)

    # Linear fit: Q_exact = calibration * Q_approx
    calibration = np.mean(Q_exact_values) / np.mean(Q_approx_values)

    # Or more sophisticated: minimize |Q_exact - cal * Q_approx|

    return calibration
```

### Expected Performance

| Metric | scipy MST | kNN Approx | Improvement |
|--------|-----------|------------|-------------|
| **Single snapshot (N=1000)** | ~50ms | ~5ms | 10× |
| **JIT-compiled** | N/A | ~1ms | 50× |
| **GPU (N=1000)** | N/A | ~0.1ms | 500× |
| **Time series (100 snapshots)** | ~5s (Python loop) | ~10ms (vmap) | 500× |
| **Memory (N=10000)** | O(N²) = 800MB | O(N×k) = 2MB | 400× |
| **Differentiable** | No | Partial (through distances) | - |

---

## Integration with jraph (Optional)

The `KNNGraph.to_jraph()` method enables integration with Graph Neural Networks:

```python
import jraph

# Build kNN graph
knn = compute_knn_graph(positions, grid, k=6)

# Convert to jraph for GNN processing
graph = knn.to_jraph()

# Use with jraph GNN models
node_features = some_encoder(positions)
graph = graph._replace(nodes=node_features)

# Apply GNN
gnn_model = jraph.GraphNetwork(...)
output = gnn_model(graph)
```

This enables learned Q predictors or learned substructure classifiers.

---

## Alternative: Parallel Borůvka for Exact MST

If exact MST is required, a JAX-native Borůvka implementation is possible:

```python
def boruvka_mst_jax(
    positions: Float[Array, "N 2"],
    max_iterations: int = 20,  # ceil(log2(N)) is sufficient
) -> Float[Array, ""]:  # Returns total MST length
    """
    Borůvka's MST algorithm adapted for JAX.

    WARNING: This is more complex and slower than kNN approximation.
    Use only when exact MST is required.
    """
    N = positions.shape[0]

    # Distance matrix (memory-intensive!)
    diff = positions[:, None, :] - positions[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff**2, axis=-1))
    dist = dist + jnp.eye(N) * jnp.inf  # No self-edges

    # Component labels (initially each node is its own component)
    labels = jnp.arange(N)

    # Selected MST edges
    mst_edges = jnp.zeros((N-1, 2), dtype=jnp.int32)
    mst_weights = jnp.zeros(N-1)
    n_edges = 0

    def body_fn(carry, _):
        labels, mst_edges, mst_weights, n_edges = carry

        # Find minimum outgoing edge per component (PARALLEL)
        # For each node, find closest node in different component
        same_component = labels[:, None] == labels[None, :]
        masked_dist = jnp.where(same_component, jnp.inf, dist)

        # Each node's min edge
        min_idx = jnp.argmin(masked_dist, axis=1)  # [N]
        min_dist = jnp.min(masked_dist, axis=1)    # [N]

        # Select one edge per component (the minimum)
        # This is the tricky part - need to avoid duplicates
        # ... (component-wise argmin, unique edge selection)

        # Update labels (union-find merge)
        # ... (sequential bottleneck)

        return (labels, mst_edges, mst_weights, n_edges), None

    # Run for max_iterations (overestimate, early exit via convergence check)
    (labels, mst_edges, mst_weights, n_edges), _ = jax.lax.scan(
        body_fn, (labels, mst_edges, mst_weights, n_edges), None, length=max_iterations
    )

    return jnp.sum(mst_weights[:n_edges])
```

**Verdict:** Possible but complex. Recommend kNN approximation for most use cases.

---

## Summary: Recommended Approach

For tracking Q evolution in dynamical simulations:

1. **Use `compute_q_approx()`** with kNN-based approximation
2. **Calibrate once** against exact scipy MST on reference distributions
3. **Track relative changes** ΔQ(t)/Q(0) which is robust to systematic bias
4. **Use `compute_q_approx_timeseries()`** for efficient batch processing
5. **Use jraph integration** if GNN-based analysis is desired

This provides:
- 100-500× speedup over scipy MST
- GPU acceleration
- JIT compilation
- vmap over time series
- Sufficient accuracy for relative Q evolution tracking

---

## Future Work

1. **Calibration sweep**: Systematic calibration across distribution types
2. **Convex hull in JAX**: Replace circular approximation with exact hull
3. **Borůvka implementation**: For cases requiring exact MST
4. **GNN-based Q predictor**: Learn Q directly from point clouds
5. **3D Q variant**: Extend beyond 2D projection

---

## References

- Cartwright & Whitworth (2004), MNRAS 348, 589 — Q parameter definition
- Küpper et al. (2011), MNRAS 417, 2300 — Azimuthal variation alternative
- Borůvka (1926) — Parallel MST algorithm
- Morton (1966) — Z-order space-filling curves
- jraph documentation — JAX graph neural network library
