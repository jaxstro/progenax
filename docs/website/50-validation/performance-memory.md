---
title: Performance & memory
description: "Measured costs for the public binary workflow, plus historical Batch A peak-RSS and timing evidence for blocked virial kernels and table-routed standalone velocity DFs."
---
# Performance & memory

The 2026-06 memory batch replaced the two kernels that could exhaust RAM on a
single workstation: the dense $(N,N,3)$ pairwise virial kernels in
`progenax.dynamics.virial`, and the per-star quadrature speed sampler in the
standalone anisotropic velocity DFs. The historical comparison below reports
measured peak RSS and call times: the benchmark checks out the pre-batch commit (`0dd1cd9`)
into a read-only git worktree and runs *both* code versions live in
subprocess-isolated cases on the same machine (Apple Silicon, macOS, float64,
eager mode; medians of 5 timed calls after one warm-up). Reproduce with:

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/benchmark_batch_a.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/profile_cluster_memory.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_speed_routing.py
```

The June JSON records the date, platform, architecture, pre-batch commit, and
repetition count. It does not identify the exact CPU model or dependency
versions, and it does not report compilation or first-call costs separately.
These retained cases are historical measurements, not a fresh release benchmark
or a performance guarantee for later source revisions.

## Peak memory: before vs after

:::{figure} figures/performance_memory.png
:name: fig-performance-memory
:width: 75%

Peak RSS (resident set size, GB; log scale) per kernel call, measured in
isolated subprocesses (2026-06-10, `scripts/benchmark_batch_a.py`). The dense
potential-energy kernel's peak depends on macOS memory pressure (20.9 GB in
this run; 32.8 GB in the original probe). The blocked kernel used 0.41 GB
at the same $N$.
:::

```{list-table}
:header-rows: 1

* - Kernel
  - $N$
  - pre-batch peak RSS
  - Batch A peak RSS
* - `compute_potential_energy`
  - 8,000
  - 6.06 GB
  - **0.38 GB** ($16\times$)
* - `compute_potential_energy`
  - 20,000
  - 20.9 GB (32.8 GB in the original probe)
  - **0.41 GB** ($\ge 50\times$)
* - aniso `LIMEPYVelocityDF.sample_velocities`
  - 20,000
  - 14.6 GB (10.9 GB in the original probe)
  - **2.02 GB** ($\ge 5.4\times$; construction-dominated — see below)
```

The pass gates live in `scripts/profile_cluster_memory.py` (7 stages,
including the $N = 10^5$ Engine A/B cluster samplers at 1.4–2.5 GB); the
script exits nonzero when a measured stage exceeds its gate. That supplies a
repeatable check when executed; the retained numbers do not show that it has
been rerun on the current source.

**Why it was so large.** The old energy kernel materialized all pairwise
separations at once (`positions[:,None,:] - positions[None,:,:]`, an
$(N,N,3)$ float64 array — 9.6 GB at $N = 2\times10^4$) and, running eagerly,
held several such intermediates simultaneously. The old anisotropic DF
sampler evaluated a 256-point speed grid × ~91-term lowered-isothermal series
per star under an eager `vmap` — a live $(N, 256, 91)$ block.

**The fixes.** (1) The virial kernels are now a `jax.lax.scan` over 256-row
blocks — one $(256, N, 3)$ transient at a time — with the per-block
computation wrapped in `jax.checkpoint`, so the **gradient** pass is also
$O(\mathrm{block}\cdot N)$ (without rematerialization, the scan's backward
pass silently stores per-block residuals and rebuilds the $O(N^2)$ footprint;
measured: `jax.grad` of the potential energy at $N=2\times10^4$ now peaks at
0.72 GB). Pair set and per-pair arithmetic are identical to the dense kernel
— verified against an independent dense oracle at rtol $10^{-13}$, gradient
parity $\leq 2\times10^{-14}$ — only the float64 summation *order* changes
across blocks. (2) The standalone DFs draw speeds from the same precomputed
CDF tables the cluster sampler already used (~14 MB), **cached on the module
at construction** and consumed through a **jitted sampling core**
(`eqx.filter_jit`, the `cluster/sampling.py` pattern) so the draw chain fuses
instead of materializing every eager intermediate. A staged profile
(2026-06-10, $N=2\times10^4$ aniso) attributes the remaining footprint:
**construction dominates** (~1.8 GB: two ODE solves + the $\mu$ integral +
the cached table build, paid once per model), the jitted draw adds only
+0.20 GB (eager it was +0.63 GB), and warm draws take **15 ms** after the
one-time ~0.5 s compile. The per-star numerical quadrature is retained as the
reference behind `speed_method="quadrature"` (also jitted now). A further
`lax.map` chunking of the table draw was evaluated and **rejected on
measurement**: it would shave part of a 0.2 GB component inside a
construction-dominated stage while adding scan overhead to a millisecond
hot path.

## Wall-clock: measured, including the honest small-$N$ crossover

:::{figure} figures/performance_walltime.png
:name: fig-performance-walltime
:width: 100%

Median wall-clock per call vs $N$ (log–log; 2026-06-10). **(a)**
`compute_potential_energy`: the blocked kernel is faster at $N=8{,}000$
($2.4\times$: 0.169 versus 0.070 s), while the dense broadcast is faster at
$N=2{,}000$ (0.013 versus 0.045 s). The crossover is not precisely located
by those two workloads. At $N=2\times10^4$ the blocked call took 0.216 s;
the dense call was memory-limited in the retained probe. **(b)** anisotropic LIMEPY sampling, after
the sampler-fusion follow-up (table cached at construction + jitted core):
warm draws cost **3–16 ms** across $N = 2\times10^3$–$2\times10^4$ — $55\times$
faster than the pre-batch quadrature at $N=10^4$ (0.497 versus 0.009 s) — with the
former small-$N$ table-build penalty eliminated (the build moved into
construction, paid once per model; a ~0.5 s jit compile is paid on the first
draw per $N$). The quadrature oracle (green) also runs in the jitted core
(0.33 s at $N=10^4$, `lax.map`-chunked to bound its memory).
:::

Timings are reported, never gated (they vary with machine load); the memory
ratios are gated (the benchmark requires $\ge 5\times$ at $N=8{,}000$;
the retained result is $16\times$).

## Distribution fidelity of the table-routed draws

Routing the speed marginal through CDF tables must not change the physics.
The angular conditional $\cos\theta\,|\,u$ uses the same numerical
sampler on both paths; only the speed marginal is routed through the table.
Agreement of those paths does not independently qualify their shared angular
quadrature. The standing gates (unit suite, `TableRouting`
classes): KS $D < 0.02$, mean/second-moment ratios within 2%/3%, $\beta(r)$
preserved to 0.06. Measured (2026-06-10, $N = 2\times10^4$ per draw,
`scripts/validate_speed_routing.py`, **ALL PASS**): King KS
$D = 1\times10^{-4}$, LIMEPY $3\times10^{-4}$ with
$\max|\Delta\beta| = 1\times10^{-5}$, Michie $6.7\times10^{-3}$ with
$\max|\Delta\beta| = 0.042$ (King/LIMEPY draws are variate-paired with the
oracle, hence near-exact; Michie's $(u_t, u_r)$ oracle consumes keys in a
different order, so its comparison is unpaired — the honest statistical
case).

:::{figure} figures/speed_routing_limepy.png
:name: fig-speed-routing-limepy
:width: 100%

Anisotropic LIMEPY ($W_0=5$, $g=1$, $r_a=4$): (a) speed distribution,
table default vs numerical quadrature reference, $N=2\times10^4$; (b) measured
$\beta(r)$ from the same draws — indistinguishable, as required (the angular
conditional uses the same numerical sampler). King and Michie versions of this figure are
embedded on the [King](king-profile.md) and
[Michie](michie-anisotropy.md) pages.
:::

## Public binary workflow: cold and warm costs

This separate 2026-09-30 exposure uses revision `366f6c4`, float64, JAX 0.11.0
on one Apple Silicon CPU host. The [](../40-howto/add-binary-population.md)
recipe supplies a Plummer half-mass radius of 1 pc, a declining primary IMF
$p(m)\propto m^{-2.3}$ over $0.1$–$20\,\Msun$ and independent companions.
Here $m$ is primary mass. The public `build_cataloged_binary_cluster` path
returns the full masked IC (`compact=False`), with twice as many particle slots
as systems. Key 0 gives 157 real stars at 100 systems and 1505 at 1000 systems.
With $T_{\rm COM}$ and $W_{\rm COM}$ the COM kinetic and gravitational
potential energies, the two values of $Q=T_{\rm COM}/|W_{\rm COM}|$ select unscaled COM velocities
(`None`) or the existing COM energy rescale (0.5).

Each of the eight cases runs in a fresh process, with persistent compilation
caching disabled. The inherited environment has `OMP_NUM_THREADS=12` and no
explicit `XLA_FLAGS`; these measurements are separate from the CI environment.
Forward execution returns the full IC and $S=2T/M$, where $T$ includes COM and
internal binary kinetic energy and $M$ is total resolved mass. Reverse execution
also returns $dS/d\lambda$ at $\lambda=1$, where $\lambda$ multiplies every
primary mass and both IMF bounds while keeping the companion draws fixed.
All outputs are synchronized. Warm times are medians of five calls.

| Systems / Q / mode | Lower + compile (s) | First call (ms) | Warm (ms) | RSS through warm (GB) | Compiler temporary (MB) |
|---|---|---|---|---|---|
| 100 / none / forward | 0.730 | 1.424 | 0.362 | 0.705 | 0.042 |
| 100 / none / reverse | 1.056 | 1.887 | 0.351 | 0.770 | 0.081 |
| 100 / 0.5 / forward | 0.810 | 7.814 | 0.388 | 0.717 | 0.116 |
| 100 / 0.5 / reverse | 1.109 | 4.896 | 0.610 | 0.772 | 0.723 |
| 1000 / none / forward | 0.751 | 5.988 | 2.238 | 0.707 | 0.372 |
| 1000 / none / reverse | 1.141 | 6.499 | 4.358 | 0.781 | 0.748 |
| 1000 / 0.5 / forward | 0.845 | 6.850 | 3.448 | 0.749 | 2.395 |
| 1000 / 0.5 / reverse | 1.426 | 21.338 | 11.380 | 0.850 | 17.238 |

```{note}
RSS is the process-lifetime high-water resident set measured through the end
of the timed calls, including preparation and compilation. It is not an
isolated execution allocation. Later eager validation raises the largest
process high-water RSS to 0.959 GB. Compiler temporary storage is a separate
compiler estimate. Lowering and compilation are recorded separately in the
raw artifact; neither is included in warm execution. These are two workload
sizes on one host, not parallel scaling, a speedup or a runner-capacity verdict.
Import/model preparation is outside these timers and is not timed separately;
the raw process wall time also includes the later validation.
```

All eight cases match eager construction at the existing relative budget
$10^{-10}$ with zero absolute slack. The reverse check is independent of AD:
at fixed periods, Kepler's relation gives orbital axes proportional to
$\lambda^{1/3}$ and squared internal speeds proportional to $\lambda^{2/3}$.
Squared COM speeds scale as $\lambda$. Thus, for this fixed-draw recipe,

$$
S(\lambda)=\lambda S_{\rm COM}(1)+\lambda^{2/3}S_{\rm int}(1),
\qquad
S'(1)=S_{\rm COM}(1)+\frac{2}{3}S_{\rm int}(1),
$$

where $S_{\rm COM}=2T_{\rm COM}/M$ and $S_{\rm int}=2T_{\rm int}/M$ are
in $\mathrm{pc^2\,Myr^{-2}}$, with $T_{\rm int}$ the internal binary kinetic
energy. The maximum relative derivative discrepancy
is $1.36\times10^{-13}$. This checks deterministic mass scaling, not
population-expectation precision, empirical calibration or stationarity.

The raw timings, source hashes, command and all validation results are in
the bounded workflow record (`docs/reviews/rung5-binary-workflow-costs.md`,
repository-only evidence).
Lowering and compilation dominate a single small draw; reuse the
compiled function when the model shape permits. The existing pair-energy
rescale does quadratic work; blocked checkpointing bounds its reverse
temporary storage. No numerical optimization follows from this exposure.

## What this enables

Blocked pair-energy kernels reduce transient storage to
$O(\mathrm{block\_size}\,N)$ while retaining quadratic pair work.
The retained measurements above concern their named particle counts,
source snapshots, precision and one host. They do not establish a universal
laptop capacity or cost for arbitrary $N$. Cached standalone DF tables can
reduce repeated-draw cost when the model is reused; construction, compilation
and numerical fidelity remain separate costs/checks. The memory profiling
script can be run locally when separately authorized for its workload; its
existence does not imply a current CI run or a fresh release benchmark.
