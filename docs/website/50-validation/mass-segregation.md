---
title: Mass segregation validation
description: Validation of the Λ_MSR diagnostic (analytic ground truth), the energy-ordered generator, the experimental mass-weighted local-Σ metric, and the new differentiable segregation observables (soft Λ_MSR / radial concentration / Σ–m) for gradient-based inference — with figures and run commands.
---

# Mass segregation validation

```{important}
Status (2026-06-09): the **$\Lambda_{\mathrm{MSR}}$ diagnostic is validated against
analytic ground truth** (`tests/validation/test_mass_segregation_physics.py`, 8 tests).
The energy-ordered **generator** is unit-tested + end-to-end checked. The
**mass-weighted local-$\Sigma$ metric** (experimental, repo-only) is validated against
the held Maschberger & Clarke (2011) PDF. **New:** three **differentiable segregation
observables** (`progenax.diagnostics.segregation_approx`) enable gradient-based / HMC
inference of segregation — validated against exact oracles in the soft $\to$ hard limit
(§5). Regenerate every figure with the commands in [Run it yourself](#run-it-yourself).
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

## 2. Energy-ordered generator — end-to-end check

`progenax.cluster.mass_segregation.energy_sorted_segregation` (+ `MassSegregationLayer`):
unit-tested in `tests/unit/cluster/` (shape/permutation validity, $S=0/1$ limits, massive-stars-
more-bound). End-to-end via `scripts/validate_cluster_ic.py`: applying it to an unsegregated
Kroupa Plummer pool drives the *independently validated* $\Lambda_{\mathrm{MSR}}$ from
$1.40 \to 14.3$ and yields Spearman $\rho(m,E)=-0.84$ (massive stars in the most-bound orbits).

:::{figure} figures/cluster_ic_energy_sorted_segregation.png
:label: val-energy-sorted
:width: 100%

`energy_sorted_segregation` produces real, $\Lambda_{\mathrm{MSR}}$-detectable segregation.
:::

```{caution}
The **partial**-segregation knob `lambda_seg` is a *linear phase-space blend* between the
unsegregated and fully-Baumgardt catalogs (differentiable in `lambda_seg`), **not** a
first-principles partial-equilibrium model — intermediate states are interpolated. Its
intermediate calibration is not yet analytically validated (see the hardening checklist in
`docs/plans/2026-06-08-fdf-methods-paper-and-hardening-design.md`).
```

## 3. Mass-weighted substructure metric (experimental, repo-only)

`gravoturb_fdf.diagnostics.mass_density` — Maschberger & Clarke (2011) local surface density
$\Sigma=(k-1)/(\pi r_k^2)$, $k=6$ (Eq. 4, verified vs the held PDF —
[](../99-bibliography/per-paper/maschberger-clarke-2011.md)) + the $m$–$\Sigma$ plane. Robust to
substructure (a *local* density), unlike CW04 $\mathcal{Q}$ on small massive subsets. Validated in
`tests/experimental/unit/test_mass_density.py` (6 tests: exact Eq. 4 formula; uniform-density
recovery; dense $>$ sparse; random masses $\to \rho\approx 0$; **primordial correlation detected**
$\rho(m,\Sigma)>0.5$ when massive stars are placed in dense clumps via
`gravoturb_fdf.masses.correlated_mass_assignment`; reproducible).

## 4. Differentiability status — generator parameter vs. observable

There are **two distinct** "differentiable segregation" questions, and conflating them is
easy:

```{list-table}
:header-rows: 1

* - Quantity
  - Kind
  - Differentiable?
  - Use
* - Generator `lambda_seg`
  - forward-model **parameter**
  - **Yes** (verified $\partial/\partial\lambda_{\mathrm{seg}}$ finite)
  - dial segregation strength *into* a model; $\partial(\text{model})/\partial\lambda_{\mathrm{seg}}$
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
  - **Yes** ($\partial/\partial\mathbf{x}$, $\partial/\partial m_{\mathrm{cut}}$)
  - gradient-based / HMC inference *from* positions+masses
```

The forward-model parameter `lambda_seg` has **always** been differentiable — that is what
powers `recover_lambda_seg_via_gradient_descent`. What was missing was a differentiable
**observable**: a function $f(\text{positions},\text{masses})\to\mathbb{R}$ whose gradient
is usable, so segregation can enter a likelihood / HMC *without* a samplable forward model.
That gap is now closed by §5 — mirroring how the differentiable `q_approx` surrogate closed
the same gap for CW04 $\mathcal{Q}$ substructure geometry.

## 5. Differentiable segregation observables (released core)

`progenax.diagnostics.segregation_approx` provides three differentiable observables that
mimic the segregation estimators observers actually report, all sharing one **soft
mass-cut kernel** $w_i=\sigma\!\big((m_i-m_{\mathrm{cut}})/\tau\big)$ (the observer's
"massive bin"; $\tau\to0$ recovers the hard indicator $\mathbb{1}[m_i>m_{\mathrm{cut}}]$):

- **soft $\Lambda_{\mathrm{MSR}}$** (`lambda_msr_approx`): the Allison+2009 MST ratio with a
  softmin nearest-neighbour distance ($(N{-}1)\langle d_{\mathrm{1NN}}\rangle$ MST proxy,
  scale-relative temperature $\beta$) and a **closed-form** random baseline (no Monte-Carlo).
- **radial concentration** (`radial_concentration_approx`):
  $C=\langle r\rangle_{\mathrm{massive}}/\langle r\rangle_{\mathrm{all}}$ about the
  mass-weighted centroid ($C<1$ segregated).
- **soft $\Sigma$–$m$** (`sigma_m_approx`): $S=\mathrm{corr}_i(w_i,\log\Sigma_i)$ with the
  Maschberger–Clarke $k$-NN density; the $k$-NN radius uses `jnp.sort` (the **exact** order
  statistic, which has a defined JVP — unlike `argsort`), so it is differentiable with no
  radius softening.

All default to **2D-projected** positions (observer-faithful), with a 3D flag. Validated in
`tests/validation/test_segregation_approx_physics.py` (13 tests) + `tests/unit/diagnostics/`
(27 tests).

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

The five figures below are read in order: first that the surrogates are *faithful*
(§5.1), then that they are *physically meaningful* segregation proxies (§5.2), then which
one is *best to infer with* (§5.3–5.4), and finally that they are *usable in gradient-based
inference* (§5.5). Each figure is followed by its physical interpretation and a correctness
verdict.

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

**Interpretation.** A differentiable surrogate is only trustworthy if it *reduces to the
estimator it claims to approximate* when the smoothing is removed. That is exactly what
this figure tests, and the three curves behave differently for a physically meaningful
reason:

- **$\Sigma$–$m$ (green, flat at $\sim10^{-11}$).** Its *only* approximation is the soft
  mass weight — the $k$-NN radius is computed with `jnp.sort`, the **exact** order
  statistic. With the bimodal test masses ($0.5$ and $10\,M_\odot$) the soft cut at
  $2\,M_\odot$ is already effectively hard even at $\tau=0.5$
  ($\sigma((0.5-2)/0.5)\approx0.05$, $\sigma((10-2)/0.5)\approx1$), so $S$ sits at
  machine precision throughout. **Correct, and exactly the expected behaviour.**
- **Radial $C$ (orange, converges with $\tau$).** Although each weight is near-binary, $C$
  *aggregates* the small residual weight over the $\sim$$380$ low-mass stars
  ($380\times0.05\approx19$, comparable to the $20$ massive ones), which shifts the
  mass-weighted centroid and mean radius. As $\tau\to0$ that residual vanishes and $C\to$
  the hard ratio at $\sim10^{-11}$. The convergence is *real and physical* — it measures
  the aggregate leakage of sub-threshold stars.
- **Soft $\Lambda$ (blue, slowest).** It carries a **second** smoothing — the softmin
  nearest-neighbour distance, controlled by $\beta$. softmin only becomes the true `min`
  as $\beta\to0$, and it does so at a finite rate, so $\Lambda$ floors at $5.6\times10^{-3}$
  of the hard 1-NN ratio (its intrinsic approximation, not an error). The monotone descent
  confirms both knobs are mutually consistent.

**Verdict: physically correct.** The hierarchy (exact $\to$ aggregation-limited $\to$
softmin-limited) is precisely what the construction predicts. In practice one calibrates
at a small but finite $\tau,\beta$ (as `calibrate_segregation_approx` does), trading a known
$\lesssim1\%$ bias for clean gradients.

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

**Verdict: physically correct and meaningful.** Every curve moves the right way and tracks
the published diagnostic; the differences in steepness are real information content, not
artefacts.

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

**Interpretation.** Fisher information is the inverse of the smallest achievable posterior
variance: $\mathrm{Var}(\hat\theta)\gtrsim1/\mathcal{I}$ (Cramér–Rao). A higher bar means a
*tighter, more confident* inference of the segregation strength from that one number. The
ranking has a clean physical reading:

- **Radial concentration wins ($\sim5\times$)** because it is a **global** statistic —
  *every* star's radius shifts when the core tightens, so the signal $\mathrm{d}\mu/\mathrm{d}\theta$
  is large while the realisation-to-realisation variance stays small.
- **$\Lambda_{\mathrm{MSR}}$ and $\Sigma$–$m$ are local** measures (nearest-neighbour
  spacing, $k$-NN density). They respond only through the immediate neighbourhood of the
  massive stars, which is both a weaker average signal and a noisier one (small-$N$ local
  estimates) — hence $\sim5\times$ less Fisher information.

This is the concrete, quantitative payoff of differentiability: the gradient
$\mathrm{d}\mu/\mathrm{d}\theta$ is computed *exactly* by autodiff rather than by noisy
finite differencing, so the identifiability comparison is itself trustworthy.

**Verdict: meaningful, with stated scope.** The ranking is evaluated at a single operating
point ($\theta=0.3$, $N=400$, bimodal masses); the *ordering* (global $\gg$ local) is
robust and physically expected, but the absolute factors are configuration-dependent and
should be re-measured for a specific survey/cluster regime before being quoted as a forecast.

### 5.4 Projection: 2D vs 3D — how much segregation signal survives?

:::{figure} figures/seg_2d_vs_3d.png
:label: val-seg-2d3d
:width: 96%

**Projection bias and information loss.** (a) Each observable in 3D (solid) vs 2D-projected
(dashed), normalised to its own 3D range. (b) The 2D/3D **Fisher ratio** — the fraction of
segregation information surviving projection: radial $C$ keeps $0.50$, $\Sigma$–$m$ $0.32$,
soft $\Lambda$ $1.19$ (dotted line = no loss).
:::

**Interpretation.** Observers never see 3D positions — they see the cluster projected on
the sky. This figure quantifies, *per observable*, how much segregation information that
projection destroys, using the 2D/3D Fisher ratio as "fraction of signal surviving":

- **Radial concentration: $\sim50\%$ survives.** Projecting a 3D radius $r$ to a sky radius
  $R\le r$ partially scrambles the radial ordering (foreground/background stars land at
  small projected radius), so the centrally-concentrated-massive-star signal is diluted by
  about half. Physically sensible.
- **$\Sigma$–$m$: only $\sim32\%$ survives** — the largest loss. A 3D local density becomes
  a 2D *surface* density, and line-of-sight superposition adds uncorrelated neighbours to
  each star's $k$-NN ball, injecting noise into the very quantity the correlation depends
  on. Reasonable that it degrades most.
- **Soft $\Lambda_{\mathrm{MSR}}$: ratio $\approx1.2$ — essentially projection-insensitive.**
  The nearest-neighbour *ratio* is dimensionless and local; 2D NN distances still separate
  a dense core from a diffuse halo about as well as 3D ones.

```{caution}
The $\Lambda$ ratio exceeding $1$ does **not** mean projection *adds* information about
segregation — a deterministic projection cannot increase the information in the data
(data-processing inequality). What is plotted is the Fisher of a *specific estimator's*
sampling distribution, and a different statistic (the 2D vs the 3D NN-ratio) can have a
marginally higher signal-to-variance if projection happens to suppress its variance as much
as its slope. With $\sim30$ realisations the $1.19$ is within sampling scatter of unity: the
honest statement is **"the local NN-ratio loses essentially no discriminating power under
projection,"** not that it gains any.
```

**Verdict: physically correct.** Global, line-of-sight-sensitive measures lose the most
($\Sigma$–$m$ $>$ radial); the dimensionless local ratio loses least. This is the expected
ordering, and it directly motivates the research question below.

### 5.5 Differentiability and end-to-end recovery

:::{figure} figures/seg_gradient_validation.png
:label: val-seg-grad
:width: 96%

**Gradients are exact and usable.** (a) Autodiff $\partial(\text{obs})/\partial
m_{\mathrm{cut}}$ (lines) vs central finite-difference (points) for all three observables —
agreement to $\lesssim10^{-9}$. (b) Gradient descent on the radial observable recovers the
true segregation strength $\theta=0.12$ exactly.
:::

**Interpretation.** Panel (a) is the differentiability certificate: autodiff and
finite-difference gradients coincide to machine precision, so the observables can be dropped
into a gradient-based optimiser or HMC sampler with confidence. The *shapes* are themselves
informative — $\partial C/\partial m_{\mathrm{cut}}$ and $\partial\Lambda/\partial
m_{\mathrm{cut}}$ peak near $3$–$4\,M_\odot$, because that is where the mass cut sweeps
*through* the bimodal mass gap and the "massive bin" membership changes fastest, while
$\partial S/\partial m_{\mathrm{cut}}\approx0$ (flat green) because the $\Sigma$–$m$
*correlation* is insensitive to exactly where the cut falls inside the gap. Panel (b) closes
the loop: descending on a single differentiable observable recovers the generating
segregation strength exactly — the minimal end-to-end demonstration that gradient-based
inference of segregation now works (the observable analogue of the long-standing
`recover_lambda_seg_via_gradient_descent`, which acts through the *forward model* instead).

**Verdict: correct.** Gradients are exact; inference through the observable succeeds.

### 5.6 Reading the results — recommendations

```{list-table}
:header-rows: 1

* - Question
  - Answer (from the figures)
* - Are the surrogates faithful?
  - Yes — each $\to$ its exact estimator as $\tau,\beta\to0$ (§5.1).
* - Are they physically meaningful?
  - Yes — monotonic in segregation and rank-consistent with Allison+2009 $\Lambda_{\mathrm{MSR}}$ ($\rho=0.98$, §5.2).
* - **Which to use for HMC inference?**
  - **Radial concentration** — $\sim5\times$ more identifiable (§5.3) and the most projection-robust *global* measure (§5.4).
* - What does projection cost?
  - $\sim50\%$ of the radial signal, $\sim68\%$ of $\Sigma$–$m$; the local $\Lambda$ ratio is essentially unaffected (§5.4).
* - Are they usable in gradients?
  - Yes — autodiff matches finite-difference to $\lesssim10^{-9}$; segregation strength is recovered exactly (§5.5).
```

```{admonition} Open research question — a better 2D↔3D segregation mapping
:class: note

Observers measure segregation in **projection**; theory and N-body live in **3D**.
Because these observables are differentiable, @val-seg-2d3d quantifies *how much
segregation information projection destroys* via the 2D/3D Fisher ratio — per observable,
not just qualitatively. The natural follow-up: can a small **learned/fit deprojection
correction** $\Lambda_{\mathrm{3D}}\approx g(\Lambda_{\mathrm{2D}},\,\text{cluster shape})$
recover the 3D value, giving a *better substructure mapping* between 2D and 3D methods?
This — together with **data-space inference** (per-star position likelihoods) and a
**noisy mass proxy** (luminosity scatter / completeness) — is the milestone (B) follow-up
(design: `docs/plans/2026-06-09-differentiable-segregation-observable-design.md`).
```

(run-it-yourself)=
## Run it yourself

```bash
# (1) Λ_MSR diagnostic — analytic validation tests (released core)
pytest tests/validation/test_mass_segregation_physics.py -v

# (2) differentiable observables — validation + unit tests, and the 5 figures
pytest tests/validation/test_segregation_approx_physics.py tests/unit/diagnostics/ -v
python scripts/validate_segregation_approx.py     # -> seg_*.png (5 figures, PASS/FAIL)

# (3) energy-ordered generator + Λ_MSR diagnostic figures
python scripts/validate_mass_segregation.py     # -> lambda_msr_*.png
python scripts/validate_cluster_ic.py           # -> cluster_ic_*.png
#   figures land in validation/plots/; the curated copies embedded here live in
#   docs/website/50-validation/figures/

# (4) experimental mass-weighted Σ metric + density-correlated placement
PYTHONPATH=src:src/experimental pytest tests/experimental/unit/test_mass_density.py tests/experimental/unit/test_masses.py -v
```

## References

Theory at [](../10-theory/tidal-and-substructure/mass-segregation.md). Diagnostic
{cite:t}`Allison2009`; energy-ordered generator {cite:t}`Baumgardt2008`; mass-weighted local-$\Sigma$
{cite:t}`MaschbergerClarke2011`. Differentiable observables design:
`docs/plans/2026-06-09-differentiable-segregation-observable-design.md`.
