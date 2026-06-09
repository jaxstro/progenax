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

```{list-table}
:header-rows: 1

* - Check
  - Construction
  - Expected
  - Measured
* - Unsegregated
  - random masses on a field
  - $\Lambda \approx 1$
  - $1.00 \pm 0.11$
* - Maximally segregated
  - massive stars in a tight core
  - $\Lambda \gg 1$
  - $\sim 410$
* - Inverse
  - massive stars on the rim
  - $\Lambda < 1$
  - $0.17$
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

$\Lambda_{\mathrm{MSR}}$ on three hand-constructed regimes (unsegregated $\to 1$,
maximally segregated $\to \gg 1$, inverse $\to <1$). ▲ marks the
$N_{\mathrm{massive}}$ set.
:::

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

:::{figure} figures/seg_hard_limit_convergence.png
:label: val-seg-hardlimit
:width: 80%
:align: center

**The central correctness claim.** Each soft observable converges to its exact
non-differentiable oracle as the softness $\tau$ (and $\beta$) $\to 0$: radial and
$\Sigma$–$m$ to $\sim10^{-11}$ (machine precision — their hard limits are exact), soft
$\Lambda_{\mathrm{MSR}}$ to $5.6\times10^{-3}$ of the hard 1-NN ratio.
:::

:::{figure} figures/seg_response_curves.png
:label: val-seg-response
:width: 80%
:align: center

**Response to segregation strength.** All three observables move monotonically as the
massive-star core tightens (normalised so "up" $=$ more segregated), tracking the exact
Allison+2009 $\Lambda_{\mathrm{MSR}}$ (dashed). The global measures (radial, $\Sigma$–$m$)
respond more sharply than the local NN-based $\Lambda$ proxy.
:::

:::{figure} figures/seg_fisher_identifiability.png
:label: val-seg-fisher
:width: 60%
:align: center

**Which observable to use for inference.** Fisher information
$\mathcal{I}(\theta)=(\mathrm{d}\mu/\mathrm{d}\theta)^2/\mathrm{Var}$ in the segregation
strength $\theta$, with $\mathrm{d}\mu/\mathrm{d}\theta$ from **autodiff** — the
differentiable payoff. **Radial concentration is $\sim5\times$ more identifiable**
($\mathcal{I}=649$) than soft $\Lambda_{\mathrm{MSR}}$ (132) or $\Sigma$–$m$ (143): the
recommended HMC summary.
:::

:::{figure} figures/seg_2d_vs_3d.png
:label: val-seg-2d3d
:width: 95%

**Projection bias (research question).** (a) Each observable in 3D (solid) vs 2D-projected
(dashed) across the segregation sweep. (b) The 2D/3D **Fisher ratio** — the fraction of
segregation information surviving projection: radial concentration keeps $\sim50\%$,
$\Sigma$–$m$ only $\sim32\%$, while the *local* NN-based $\Lambda$ proxy is essentially
unaffected ($\sim1.2$). See the open research question below.
:::

:::{figure} figures/seg_gradient_validation.png
:label: val-seg-grad
:width: 95%

**Differentiability + inference.** (a) Autodiff $\partial(\text{obs})/\partial
m_{\mathrm{cut}}$ matches central finite-difference to $\lesssim10^{-9}$ for all three.
(b) Gradient descent on a single observable (radial concentration) recovers the true
segregation strength $\theta$ exactly ($0.12\to0.12$) — the end-to-end "works for
inference" demonstration.
:::

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
