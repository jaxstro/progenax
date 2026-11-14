---
title: Mass segregation validation
description: Validation of the Λ_MSR diagnostic (analytic ground truth), the energy-ordered generator, the experimental mass-weighted local-Σ metric, and the new differentiable segregation observables (soft Λ_MSR / radial concentration / Σ–m) for gradient-based inference — with figures and run commands.
---

# Mass segregation validation

```{important}
Status (2026-06-10): the **$\Lambda_{\mathrm{MSR}}$ diagnostic is validated against
analytic ground truth** (`tests/validation/test_mass_segregation_physics.py`).
The energy-ordered generator is now explicitly labeled the **PRIMORDIAL**
(non-equilibrium) route and carries its own physics suite
(`tests/validation/test_segregation_equilibrium_physics.py`, §2; see the
[test dashboard](test-dashboard.md) for live per-suite counts); the
**`lambda_seg` blend was retired** in the 2026-06 unified redesign — the
differentiable, *equilibrium* segregation knob is now
`MultiComponentCluster.from_mass_segregation(delta)` (Engine A, validated at
[](multimass-equilibrium.md)). The **mass-weighted local-$\Sigma$ metric**
(experimental, repo-only) is validated against the held Maschberger & Clarke (2011)
PDF, and three **differentiable segregation observables**
(`progenax.diagnostics.segregation_approx`) enable gradient-based / HMC inference of
segregation. Their sharp limits and retained population comparisons have
different targets (§5); the NN ratio does not become the MST statistic.
Regenerate every figure with the commands in [Run it yourself](#run-it-yourself).
```

## 1. $\Lambda_{\mathrm{MSR}}$ diagnostic — validated against analytic ground truth

`progenax.diagnostics.compute_lambda_msr` (Allison et al. 2009;
$\Lambda_{\mathrm{MSR}}=\langle\ell_{\mathrm{random}}\rangle/\ell_{\mathrm{massive}}$,
definition verified against the held ApJ 700 L99 PDF — see
[](../99-bibliography/per-paper/allison-2009.md)). Tests in
`tests/validation/test_mass_segregation_physics.py`:

The three regimes below are **rearrangements of one shared, realistic baseline cluster**
(a centrally concentrated Plummer sphere): only the massive-star positions change, so the
panels are an honest like-for-like comparison, the massive stars are spatially *resolved*,
and the segregated $\Lambda$ sits in the **observed range** (Allison et al. 2009 measure
$\Lambda \sim$ a few for segregated young clusters such as the ONC). The unbounded
behaviour of $\Lambda$ — useful as a *mathematical* limit but not a physical configuration
— is checked separately (the delta-core row), so it never sets the headline number.

```{list-table}
:header-rows: 1

* - Check
  - Construction
  - Expected
  - Measured
* - Unsegregated
  - random masses on the baseline cluster
  - $\Lambda \approx 1$
  - $1.00 \pm 0.11$
* - Segregated (realistic)
  - massive stars in a **resolved** central core ($\sim0.2$ pc)
  - $\Lambda \sim$ a few
  - $4.8 \pm 0.7$
* - Inverse
  - massive stars on the outer half of the same cluster
  - $\Lambda < 1$
  - $0.80 \pm 0.10$
* - Estimator limit (delta core)
  - near-point massive core, $L_{\mathrm{massive}}\!\to\!0$
  - $\Lambda \to \infty$ (unbounded)
  - $\sim 390$ (scale-set, *not* physical)
* - Exact ($N_{\mathrm{massive}}=2$)
  - vs independent `scipy.pdist` enumeration
  - exact ratio
  - $3.047$ vs $3.046$ ✓
* - Estimator convergence
  - $\sigma(\Lambda)$ vs $N_{\mathrm{random}}$
  - $\propto 1/\sqrt{N}$
  - confirmed
* - Binary caveat
  - tight massive pair, $N_{\mathrm{massive}}=2$
  - spurious inflation
  - $\sim 10^3$–$10^4\times$
```

:::{figure} figures/lambda_msr_regimes.png
:label: val-lambda-regimes
:width: 100%

$\Lambda_{\mathrm{MSR}}$ on three states of **one shared Plummer cluster** (same grey
"all stars" distribution, same axes): (a) unsegregated ($\Lambda=1.05$) — massive stars
follow the field; (b) segregated ($\Lambda=5.70$) — the $N_{\mathrm{massive}}$ stars (▲)
sit in a *resolved* central core, a value in the observed range, not a scale-arbitrary
extreme; (c) inverse ($\Lambda=0.73$) — massive stars on the cluster outskirts. ▲ marks
the $N_{\mathrm{massive}}$ set used for $\ell_{\mathrm{massive}}$.
:::

```{note}
This figure was rebuilt (2026-06-09) after review. The earlier version (i) crushed the
massive stars into a $10^{-3}$ pc near-delta core — so they overlapped into a single
unresolvable point and reported a *scale-arbitrary* $\Lambda\approx410$ — and (ii) used a
**different** baseline distribution in each panel (a $0.05$ pc clump for the inverse case),
breaking the cross-panel comparison. The estimator itself was never in error (the exact
$N_{\mathrm{massive}}=2$ anchor below holds to $<0.1\%$); only the *illustration* was
unphysical. The three panels now share one realistic cluster and report observed-range
$\Lambda$ values.
```

:::{figure} figures/lambda_msr_monotonic_convergence.png
:label: val-lambda-convergence
:width: 100%

Left: $\Lambda$ rises monotonically with segregation. Right: convergence to the exact value,
$\sigma(\Lambda)\propto 1/\sqrt{N_{\mathrm{random}}}$.
:::

:::{figure} figures/lambda_msr_binary_caveat.png
:label: val-lambda-binary
:width: 70%
:align: center

The documented binary caveat, quantified: a tight massive pair drives
$\Lambda \propto 1/\text{separation}$ (use binary centre-of-mass positions to avoid).
:::

## 2. Primordial (energy-ordered) generator — physics suite + end-to-end check

`progenax.cluster.mass_segregation.energy_sorted_segregation` is the
**PRIMORDIAL** segregation generator: it assigns the most massive stars to the
most bound orbits of an equilibrium pool, in the spirit of
{cite:t}`Baumgardt2008` and McLuster (Küpper et al. 2011). One **deliberate,
documented departure** from the published recipe: Baumgardt+2008 draw orbits
*randomly within cumulative-mass bins*; progenax uses a deterministic isotonic
rounding of the bin-centre targets, which guarantees a distinct orbit per star
for ANY mass spectrum — the random per-bin sampler collapsed below one orbit
per bin for steep IMFs, producing coincident stars and $V = -\infty$ (the
orbit-reuse bug, fixed in this arc and regression-locked below). Realisation
variety comes from re-drawing the random pool.

Physics suite `tests/validation/test_segregation_equilibrium_physics.py`
(ported in the 2026-06 redesign from the retired blend's suite
onto the protocol-API composition; see the [test dashboard](test-dashboard.md)
for the live count):

```{list-table}
:header-rows: 1

* - Property
  - Tolerance (as tested)
  - Anchor
* - Retained full-ordering group virial comparison: seed-averaged
    $\max_j |Q_j - 0.5|$ over 4 mass groups, 10 seeds
  - drift $< 0.08$
  - finite group virial diagnostic in the retained configuration; energy
    ordering alone does not establish a self-consistent stationary DF
* - Global virial after finalisation
  - $|Q - 0.5| < 0.02$
  - exact rescale by construction
* - Orbit-reuse regression (steep Kroupa IMF)
  - $V$ finite; min pair separation $> 10^{-6}$
  - pre-fix: coincident stars, $V = -\infty$
* - Segregation signal
  - $\Lambda_{\mathrm{MSR}}^{\rm seg} > \Lambda_{\mathrm{MSR}}^{\rm unseg}$
  - the independently validated §1 diagnostic
```

End-to-end via `scripts/validate_cluster_ic.py` §3 (re-run 2026-06-10, PASS):
applying the generator to an unsegregated Kroupa Plummer pool drives the
*independently validated* $\Lambda_{\mathrm{MSR}}$ from $1.40 \to 4.74$ and
yields Spearman $\rho(m,E) = -0.92$ (massive stars in the most-bound orbits).
(Earlier versions of this page quoted $1.40 \to 14.3$, $\rho = -0.84$ —
numbers from before the orbit-reuse fix changed the deterministic assignment.)

:::{figure} figures/cluster_ic_energy_sorted_segregation.png
:label: val-energy-sorted
:width: 100%

`energy_sorted_segregation` produces real, $\Lambda_{\mathrm{MSR}}$-detectable
segregation: $\Lambda_{\mathrm{MSR}} = 1.40 \to 4.74$, $\rho(m,E) = -0.92$
(`scripts/validate_cluster_ic.py` §3, 2026-06-10 run).
:::

```{admonition} The `lambda_seg` blend is retired — the equilibrium route is Engine A
:class: important
The historical partial-segregation knob `lambda_seg` (a linear phase-space
blend between the unsegregated and fully-ordered catalogs) was **retired** in
the 2026-06 unified redesign: its intermediate states drift from per-mass-group
virial balance (drift peaking $\sim 0.05$ at $\lambda = 0.5$ in the Phase-0
error-budget study). The differentiable segregation knob is now
**`MultiComponentCluster.from_mass_segregation(delta)`** — the multi-mass
lowered-isothermal family with $w_j = \mu_j^{-\delta}$, a *true
shared-potential equilibrium for valid finite converged models* (theory
virial balance uses each component's force in that shared potential;
segregation ratio $1.0 \to 2.6$ over $\delta \in [0, 0.6]$). Validation:
[](multimass-equilibrium.md). `energy_sorted_segregation` survives as the
labeled primordial (non-equilibrium, non-differentiable) generator above.
```

## 3. Mass-weighted substructure metric (experimental, repo-only)

`gravoturb.diagnostics.mass_density` — Maschberger & Clarke (2011) local surface density
$\Sigma=(k-1)/(\pi r_k^2)$, $k=6$ (Eq. 4, verified vs the held PDF —
[](../99-bibliography/per-paper/maschberger-clarke-2011.md)) + the $m$–$\Sigma$ plane. Robust to
substructure (a *local* density), unlike CW04 $\mathcal{Q}$ on small massive subsets. Validated in
`tests/experimental/unit/test_mass_density.py` (6 tests: exact Eq. 4 formula; uniform-density
recovery; dense $>$ sparse; random masses $\to \rho\approx 0$; **primordial correlation detected**
$\rho(m,\Sigma)>0.5$ when massive stars are placed in dense clumps via
`gravoturb.realization.mass_assignment.correlated_mass_assignment`; reproducible).

## 4. Differentiability status — generator parameter vs. observable

There are **two distinct** "differentiable segregation" questions, and conflating them is
easy:

```{list-table}
:header-rows: 1

* - Quantity
  - Kind
  - Differentiable?
  - Use
* - Equipartition $\delta$ (`MultiComponentCluster.from_mass_segregation`)
  - forward-model **parameter**
  - **Yes** (AD $=$ FD through the table-backed solve, $2.15\times10^{-4}$)
  - dial *equilibrium* segregation strength into a model;
    $\partial(\text{model})/\partial\delta$ — see [](multimass-equilibrium.md)
* - Generator `energy_sorted_segregation` (primordial)
  - forward-model generator
  - No (`argsort`, floor — discrete assignment, by design)
  - labeled primordial route (§2); use $\delta$ for gradients
* - `compute_lambda_msr` (Allison)
  - **observable** (measurement)
  - No (SciPy MST + `argsort`)
  - validated diagnostic / oracle
* - $m$–$\Sigma$ metric, correlated placement
  - **observable**
  - No (kNN / ranking)
  - experimental diagnostic
* - **`segregation_approx`** (§5)
  - **observable** (measurement)
  - Piecewise local AD, away from rank/tie and other boundaries
  - gradient-based / HMC inference *from* positions+masses
```

The forward-model *parameter* route is differentiable: the equipartition
$\delta$ flows through the coupled multi-mass Poisson solve and the
`jax.lax.scan` sampler ([](multimass-equilibrium.md)). (The retired
`lambda_seg` blend was also differentiable — what it lacked was per-group
*equilibrium*, which is why $\delta$ replaced it.) What was missing on the
measurement side was a differentiable **observable**: a function
$f(\text{positions},\text{masses})\to\mathbb{R}$ whose gradient
is usable, so segregation can enter a likelihood / HMC *without* a samplable forward model.
Section 5 supplies local differentiable summary maps. Their use in a
likelihood still requires a sampling model, numerical derivative checks
and calibration for the selected physical population.

## 5. Differentiable segregation observables (released core)

`progenax.diagnostics.segregation_approx` provides three differentiable observables that
mimic the segregation estimators observers actually report, all sharing one **soft
mass-cut kernel** $w_i=\sigma\!\big((m_i-m_{\mathrm{cut}})/\tau\big)$ (the observer's
"massive bin"; $\tau\to0$ recovers the hard indicator $\mathbb{1}[m_i>m_{\mathrm{cut}}]$):

- **soft NN ratio** (`lambda_msr_approx`): the unweighted mean global
  soft nearest-neighbor distance divided by its soft-mass-weighted mean,
  with scale-relative temperature $\beta$. Neighbors are searched among
  **all** stars, including for massive-star weights. Its baseline is not
  a random-subset MST expectation, and its sharp limit is not Allison's
  $\Lambda_{\mathrm{MSR}}$.
- **radial concentration** (`radial_concentration_approx`):
  $C=\langle r\rangle_{\mathrm{massive}}/\langle r\rangle_{\mathrm{all}}$ about the
  soft-cut-weighted centroid ($C<1$ segregated).
- **soft $\Sigma$–$m$** (`sigma_m_approx`): $S=\mathrm{corr}_i(w_i,\log\Sigma_i)$ with the
  Maschberger–Clarke $k$-NN density; the $k$-NN radius uses `jnp.sort` (the **exact** order
  statistic. AD follows the selected ordering away from distance ties;
  it does not differentiate a changing neighbor identity at a tie.

All default to 2D-projected positions, with a 3D flag. The retained
comparisons below are from the existing figures; no population campaign
was rerun for this claim correction. Tests and scripts are
`tests/validation/test_segregation_approx_physics.py` + `tests/unit/diagnostics/`
(see the [test dashboard](test-dashboard.md) for live per-suite counts).

```{list-table}
:header-rows: 1

* - Property
  - Tolerance (as tested)
  - Measured
  - Anchor
* - soft radial $C \to$ exact ($\tau\to0$)
  - $|\text{soft}-\text{exact}|<10^{-3}$
  - $9.8\times10^{-12}$
  - hard mass-cut radial ratio
* - soft $\Sigma$–$m$ $\to$ exact ($\tau\to0$)
  - $<10^{-2}$
  - $4.1\times10^{-11}$
  - SciPy `cKDTree` $k$-NN $\Sigma$
* - soft $\Lambda \to$ exact NN-ratio ($\tau,\beta\to0$)
  - $<5\times10^{-2}$
  - $5.6\times10^{-3}$
  - hard 1-NN ratio
* - monotonic response (Spearman vs strength)
  - $|\rho|>0.8$
  - $C{:}\,0.99$, $S{:}\,0.98$, $\Lambda{:}\,0.81$
  - core-tightness sweep
* - rank-correlation vs exact $\Lambda_{\mathrm{MSR}}$
  - $|\rho|>0.8$
  - $0.98$ (Allison oracle)
  - `compute_lambda_msr`
* - Fisher information $\mathcal{I}(\theta)$
  - finite, $>0$
  - $C{:}\,649$, $S{:}\,143$, $\Lambda{:}\,132$
  - autodiff $\mathrm{d}\mu/\mathrm{d}\theta$, Var
* - $\partial(\text{obs})/\partial m_{\mathrm{cut}}$ (AD vs FD)
  - max $<10^{-3}$
  - $\le 8\times10^{-10}$
  - central finite-difference
* - segregation recovery (gradient descent)
  - $|\Delta\theta|<0.03$
  - $0.120\to0.120$
  - exact recovery
```

The figures compare sharp-limit targets (§5.1), response on the retained
synthetic family (§5.2), a local mean/variance information proxy
(§5.3–5.4), and selected derivatives and deterministic fitting (§5.5).
These are separate checks; none supplies a general inference guarantee.

### 5.1 Hard-limit convergence — are the surrogates faithful?

:::{figure} figures/seg_hard_limit_convergence.png
:label: val-seg-hardlimit
:width: 78%
:align: center

**The central correctness claim.** Absolute error $|\text{soft}-\text{exact}|$ vs the
softness, as $\tau$ (and $\beta$ for $\Lambda$) $\to 0$ (axis runs sharp-to-the-right).
Radial $C$ (orange) and $\Sigma$–$m$ $S$ (green) reach $\sim10^{-11}$; soft
$\Lambda_{\mathrm{MSR}}$ (blue) reaches $5.6\times10^{-3}$.
:::

**Interpretation.** The radial statistic approaches its hard-cut radial
ratio. For the two-level masses in this example, the soft weights have
only two values, so they are a positive affine transform of the hard
indicator. Pearson correlation is invariant to that transform (apart
from numerical regularizers); this explains the nearly constant
$\Sigma$–$m$ curve without requiring both weights to be near zero or one.
Radial concentration is not affine invariant, so residual low-mass
weights can change its centroid and mean radii.

The blue curve approaches the **global 1-NN ratio** checked by
`test_lambda_converges_to_exact_nn_ratio`, not an MST ratio. For example,
three collinear positions 0,1,3 with the first two selected have global
nearest-neighbor distances 1,1,2: their NN ratio is 4/3. The massive-pair
MST length is 1, while the mean of all three two-point subset lengths is 2,
giving MST $\Lambda=2$. Thus removing smoothing does not remove this
statistical difference. The retained 5.6e-3 discrepancy describes the
finite-softness NN comparison in that figure; it is not a universal MST
error floor or a general sub-percent calibration guarantee.

**Scope.** These are mathematical-limit checks for distinct observables.
MST rank correlation on selected cluster families is separate evidence.

### 5.2 Response to segregation — are they meaningful proxies?

:::{figure} figures/seg_response_curves.png
:label: val-seg-response
:width: 78%
:align: center

**Monotonic, literature-consistent response.** Each observable (normalised to $[0,1]$,
sign-flipped for $C$ so "up" $=$ more segregated) vs segregation strength $1-\theta$
(massive-star core tightness), with the exact Allison+2009 $\Lambda_{\mathrm{MSR}}$ (dashed)
overlaid. Spearman $\rho$: $C\,0.99$, $S\,0.98$, $\Lambda\,0.81$; all rank-correlate with
the exact $\Lambda_{\mathrm{MSR}}$ at $\rho=0.98$.
:::

**Interpretation.** As the massive stars are concentrated into a tighter core, each
observable responds in the physically correct direction: the massive population sits at
**smaller radii** ($C\downarrow$, plotted inverted so it rises), in **locally denser**
regions ($S\uparrow$), and with **shorter nearest-neighbour spacing** ($\Lambda\uparrow$).
All three are therefore genuine monotonic segregation proxies, and — critically — they are
*rank-consistent with the field-standard Allison+2009 $\Lambda_{\mathrm{MSR}}$* ($\rho=0.98$),
which is the validated oracle (§1). The radial measure (orange) is the steepest and
smoothest; the local NN-based $\Lambda$ (blue) is the noisiest and shallowest — a direct
preview of the Fisher ranking in §5.3. (The axis is normalised per-curve, so this panel
compares *shape and monotonicity*, not absolute scale.)

**Scope of the retained response.** The retained curves respond in the stated direction and track the
MST diagnostic in this family. That ranking does not establish general
MST equivalence or an inference-information ordering.

### 5.3 Fisher information — which observable should you infer with?

:::{figure} figures/seg_fisher_identifiability.png
:label: val-seg-fisher
:width: 62%
:align: center

**Identifiability ranking.** Fisher information
$\mathcal{I}(\theta)=(\mathrm{d}\mu/\mathrm{d}\theta)^2/\mathrm{Var}$ in the segregation
strength, evaluated at $\theta=0.3$ with $\mathrm{d}\mu/\mathrm{d}\theta$ from **autodiff**
over $40$ cluster realisations. Radial concentration $\mathcal{I}=649$ vs soft
$\Lambda_{\mathrm{MSR}}\,132$ and $\Sigma$–$m\,143$.
:::

**Interpretation.** The plotted quantity is a local mean/variance proxy,
$\mathcal I_\mu=(\partial_\theta\mu)^2/V$, with mean $\mu$ and variance $V$
estimated over the retained realizations. It equals the mean term of a
Gaussian-summary Fisher calculation when variance is treated as fixed.
A parameter-dependent Gaussian variance adds
$\tfrac12(\partial_\theta V/V)^2$; a non-Gaussian summary needs its
sampling distribution. The usual Cramér–Rao bound concerns regular
unbiased estimators, not a universal minimum posterior variance.

The retained values rank radial concentration highest at
$\theta=.3,N=400$ and the specified bimodal masses. AD computes the local
derivative of this numerical summary map; it does not establish sampling
error, global identifiability or calibrated posterior uncertainty. No
error bars on the information ranking or broader-population comparison
are supplied by this figure.

**Scope.** Use this as a configuration-specific summary comparison, not
a universal recommendation for HMC or survey precision.

### 5.4 Projection: 2D vs 3D — how much segregation signal survives?

:::{figure} figures/seg_2d_vs_3d.png
:label: val-seg-2d3d
:width: 96%

**Projected summary comparison.** (a) Each observable in 3D (solid) and
2D (dashed), normalized to its own 3D range. (b) The ratio of the
mean/variance information proxy in 2D to that in 3D is 0.50 for radial
$C$, 0.32 for $\Sigma$–$m$ and 1.19 for the soft NN ratio. Figure values
are retained; no uncertainty estimate is supplied here.
:::

**Interpretation.** Projection changes distances, neighbor ordering and
the sampling distributions of these compressed statistics. The proxy
ratios describe that retained configuration, not a fraction of the
information in the original full positions. In particular, the
three-dimensional and projected NN ratios are different compressions;
one need not be a deterministic function of the other.

```{caution}
Deterministic projection cannot increase the full data's Fisher
information under the data-processing assumptions. It can yield a
statistic with a larger mean/variance proxy than a different 3D statistic.
The retained 1.19 therefore does not prove an information gain, or
statistical consistency with unity without a sampling uncertainty.
```

**Scope.** No general projection-insensitivity or inference ranking is
established by these figure values.

### 5.5 Differentiability and end-to-end recovery

:::{figure} figures/seg_gradient_validation.png
:label: val-seg-grad
:width: 96%

**Selected mass-cut derivative checks.** (a) Autodiff $\partial(\text{obs})/\partial
m_{\mathrm{cut}}$ (lines) vs central finite-difference (points) for all three observables —
agreement to $\lesssim10^{-9}$. (b) Gradient descent on the radial observable recovers the
true segregation strength $\theta=0.12$ exactly.
:::

**Interpretation.** Panel (a) compares local derivatives with respect to
$m_{\mathrm{cut}}$ at the plotted points. That parameter changes soft
weights, not the neighbor ranking. Agreement there does not certify
position derivatives across ties, the MST statistic, a likelihood model
or HMC convergence. The near-flat Pearson curve follows the two-level
weight invariance discussed in §5.1.

Panel (b) fits a scalar summary within the specified synthetic family.
It is a retained deterministic fitting example, not independent-catalogue
coverage, a posterior or evidence of arbitrary physical segregation
recovery. Values and figure data are retained; this review does not
repeat the inference calculation.

**Scope.** The executed comparison supports the named local derivative
and fitting configurations. Numerical AD agreement and physical
qualification remain different claims.

### 5.6 Reading the results — recommendations

```{list-table}
:header-rows: 1

* - Question
  - Answer (from the figures)
* - Are the surrogates faithful?
  - Each has its own sharp target; the NN ratio does not become MST $\Lambda$ (§5.1).
* - Are they physically meaningful?
  - Yes — monotonic in segregation and rank-consistent with Allison+2009 $\Lambda_{\mathrm{MSR}}$ ($\rho=0.98$, §5.2).
* - **Which to use for HMC inference?**
  - The retained mean/variance proxy favors radial concentration; a scientific likelihood and coverage check are still needed (§5.3).
* - What does projection cost?
  - Retained summary-proxy ratios are .50,.32,1.19; these do not measure full-data information loss (§5.4).
* - Are they usable in gradients?
  - The plotted mass-cut derivatives agree with FD; ranking boundaries and general inference remain separate (§5.5).
```

```{admonition} Open research question — a better 2D↔3D segregation mapping
:class: note

Observers measure segregation in **projection**; theory and N-body live in **3D**.
The retained @val-seg-2d3d compares mean/variance proxies for projected
and 3D statistics. Their relation to full-data information and a useful
scientific deprojection model remains an open question. The natural follow-up: can a small **learned/fit deprojection
correction** $\Lambda_{\mathrm{3D}}\approx g(\Lambda_{\mathrm{2D}},\,\text{cluster shape})$
recover the 3D value, giving a *better substructure mapping* between 2D and 3D methods?
This — together with **data-space inference** (per-star position likelihoods) and a
**noisy mass proxy** (luminosity scatter / completeness) — is the milestone (B) follow-up.
```

(run-it-yourself)=
## Run it yourself

```bash
# (1) Λ_MSR diagnostic — analytic validation tests (released core)
pytest tests/validation/test_mass_segregation_physics.py -v

# (2) differentiable observables — validation + unit tests, and the 5 figures
pytest tests/validation/test_segregation_approx_physics.py tests/unit/diagnostics/ -v
python scripts/validate_segregation_approx.py     # -> seg_*.png (5 figures, PASS/FAIL)

# (3) primordial generator — physics suite, Λ_MSR diagnostic figures, end-to-end
pytest tests/validation/test_segregation_equilibrium_physics.py -v
python scripts/validate_mass_segregation.py     # -> lambda_msr_*.png
python scripts/validate_cluster_ic.py           # -> cluster_ic_*.png
#   figures land in validation/plots/; the curated copies embedded here live in
#   docs/website/50-validation/figures/

# (4) the EQUILIBRIUM mass-segregation route (Engine A, differentiable delta)
#     -> see multimass-equilibrium.md
python scripts/validate_multimass_equilibrium.py

# (5) experimental mass-weighted Σ metric + density-correlated placement
PYTHONPATH=src:src/experimental pytest tests/experimental/unit/test_mass_density.py tests/experimental/unit/test_masses.py -v
```

## References

Theory at [](../10-theory/tidal-and-substructure/mass-segregation.md). Diagnostic
{cite:t}`Allison2009`; energy-ordered generator {cite:t}`Baumgardt2008`; mass-weighted local-$\Sigma$
{cite:t}`MaschbergerClarke2011`. The differentiable-observables design is recorded
in the repository's planning notes.
