---
title: Multi-mass LIMEPY equilibrium (Engine A)
description: "Validation suite for MultiComponentCluster's coupled multi-mass lowered-isothermal (LIMEPY) equilibrium: per-component dispersion vs the analytic DF moment, per-group and global virial ratios across the equipartition range, the anisotropic Michie/Osipkov-Merritt sampler vs the DF's own beta(r), and the differentiable DF-table acceleration layer with quadrature oracles."
---
# Multi-mass LIMEPY equilibrium (Engine A)

**Engine A** (`MultiComponentCluster.from_components` /
`.from_mass_segregation` / `.from_imf`) is progenax's differentiable
multi-component **lowered-isothermal** family ({cite:t}`Gieles2015`, §2.2):
all components share a potential solved from the coupled Poisson equation.
This page checks continuum virial relations, sampled moments and table accuracy.
The [model page](../10-theory/spatial-profiles/lowered-model-family.md)
defines the component scales and segregation law. Tests live in
`tests/validation/test_multimass_equilibrium_physics.py` plus the DF-table
unit suites (`tests/unit/profiles/test_limepy_tables.py`; table threading in
`tests/unit/profiles/test_limepy_multimass.py`) — see the
[test dashboard](test-dashboard.md) for the test inventory. Figures + PASS/FAIL:
`scripts/validate_multimass_equilibrium.py`,
`scripts/validate_multimass_anisotropy.py`, `scripts/validate_df_tables.py`.

## Current validation checkpoint

```{note} Retained configurations, 2026-09-28
The equilibrium-physics tests and the equilibrium, anisotropy and DF-table
figure scripts pass their existing gates with float64 enabled. The figures below
show the retained exposure rerun on production revision `fb2cfa1`. The checks establish initial-condition accuracy
within their stated scope, not dynamical stability or full-domain qualification.

The sampled isotropic global virial ratios span 0.497–0.500 across the four
retained segregation settings. The anisotropic global ratios span approximately
0.499–0.500. The DF-table check gives a maximum absolute potential difference
of $1.56\times10^{-7}$ across three models and maximum relative AD/FD difference
of $1.97\times10^{-6}$ against its existing $10^{-3}$ gradient budget.
```

The anisotropy figure compares the sampled finite-width shells with
density-weighted DF shell moments. If $\rho_j(r)$ is component $j$'s density
and $\langle v_k^2\rangle_j(r)$ its local second velocity moment, the shell
reference is

```{math}
\langle v_k^2\rangle_{j,[a,b]} =
\frac{\int_a^b r^2\rho_j(r)\langle v_k^2\rangle_j(r)\,dr}
     {\int_a^b r^2\rho_j(r)\,dr},
```

where $a$ and $b$ are the shell boundaries and $k$ denotes the radial or
combined tangential component. Shell anisotropy is then
$\beta=1-\langle v_t^2\rangle/(2\langle v_r^2\rangle)$.
The reference is a ratio of population moments; a seed-averaged finite-sample
ratio can retain statistical bias. Sample positions and velocities use the
script's existing centre-of-mass subtraction.

```{important} Heavy-component coverage
The retained mixed-population sample resolves the two innermost heavy-component
shells in all eight seeds. The next shell has only two contributing seeds, and
outer shells are unresolved. Incomplete-seed points are labelled in the figure.
The existing absolute 0.04 anisotropy gate applies to resolved **light-component**
bins and retains its centre-point reference; it is reported separately from the
updated shell comparison. No new heavy-component acceptance threshold is inferred
from this plot. The separate conditional checks below assess both components; natural outer-shell
occupancy remains unresolved.
```

### Conditional-component qualification

A separate check samples each component directly: eight independent seeds with
60,000 stars per component and the same model, shell boundaries and DF quadrature.
The common velocity normalization cancels in the anisotropy ratio. A shell is
graded only when **every seed contains at least 400 stars**; its mean anisotropy
must differ from the density-weighted shell reference by less than 0.04.

| Component | Passing shells | Covered radius / tidal radius | Maximum absolute error | Unresolved shells |
|---|---|---|---|---|
| Light | 8 | 0.06–0.772 | 0.01053 | 2 |
| Heavy | 4 | 0.06–0.416 | 0.00740 | 6 |

All 12 adequately populated shells pass. The remaining eight are unresolved,
including shells whose measured error exceeds 0.04; they are not counted as passes.
This check does not qualify the complete radial domain. Reproduce it with
`scripts/validate_multimass_anisotropy.py --conditional-components results.json`.
The figure below retains the mixed-population exposure and its separate gates.

The complementary **shell-stratified check** draws 2,000 stars per shell and
component for each of eight seeds. Radial probabilities are uniform between the
component CDF values at the shell boundaries, then passed through the existing
radial inverse and conditional velocity sampler. Shell membership uses the model
centre; subtracting a sampled centre would change the conditioning.

All **20 component–shell comparisons pass** the same 0.04 budget over
0.06–0.95 tidal radii. Maximum absolute errors are 0.01972 (light) and
0.01987 (heavy). This checks conditional moments at rare outer radii; it does
not replace a natural-population occupancy test or cover radii outside that interval.

:::{figure} figures/seg_multimass_stratified.png
:label: val-multimass-stratified
:width: 100%
:alt: Two panels compare conditional shell anisotropy for light and heavy stars with density-weighted DF moments; all sample means lie within the absolute 0.04 budget.

**Conditional shell anisotropy, 2026-09-28.** Points show the eight-seed mean and
standard error; lines join density-weighted shell references at shell centres.
Shading marks the approved absolute error budget, not a confidence band.
Each component contributes 2,000 stars per shell per seed. This differs from the
natural-population exposure below. Regenerate the measurements with
`scripts/validate_multimass_anisotropy.py --stratified-shells results.json`;
render the saved record with `--plot-stratified results.json`.
:::

## What is verified — isotropic equilibrium

The retained isotropic tests use central dimensionless potential $W_0=7$,
truncation index $g=1$, stellar masses $(1,4)\,M_\odot$, and central density
fractions $(0.5,0.5)$. The segregation parameter $\delta$ controls the
component velocity-scale ratios. Here $Q=T/|V|$ denotes kinetic energy divided
by the magnitude of the gravitational virial term; $Q_j$ is its component counterpart.

| Observable | Acceptance criterion | Exposure |
|---|---|---|
| Continuum $Q_j$ | $\lvert Q_j-0.5\rvert<0.002$ | Both components; $\delta=0,0.3,0.5,0.6$ |
| Sampled global $Q$ | $\lvert\langle Q\rangle-0.5\rvert<0.03$ | Three seeds, 8,000 stars; $\delta=0,0.3,0.6$ |
| Core component dispersion | Relative difference below 7% | 40,000 stars; $\delta=0.5$ |
| Sampled component $Q_j$ | Absolute differences below 0.04 (light), 0.06 (heavy) | Five seeds, 8,000 stars; $\delta=0,0.3,0.5$ |
| Half-mass-radius ratio | Unity within 0.001 at $\delta=0$; increases across the sweep | $\delta=0,0.2,0.4,0.6$ |

```{note} Continuum and finite samples
A component's continuum virial relation and its sampled virial estimate are
separate quantities. Passing these finite-sample budgets does not establish the
cause of an offset or guarantee accuracy for a different population or sample size.
```

:::{figure} figures/seg_multimass_equilibrium.png
:label: val-multimass-equilibrium
:width: 100%

**Multi-mass equilibrium checks, 2026-09-28**
(`scripts/validate_multimass_equilibrium.py`). Component density and dispersion
profiles, the half-mass-radius ratio across segregation strength, and theoretical
and sampled virial ratios. The retained script gates pass; agreement of these
initial-condition diagnostics is not a dynamical stability test.
:::

## What is verified — anisotropic sampler

The anisotropic checks compare sampled radial and tangential moments with
quadrature of the component DF. See the
[model description](../10-theory/spatial-profiles/lowered-model-family.md)
for the physical family and component conventions.

| Observable | Acceptance criterion | Exposure |
|---|---|---|
| Global virial ratio | Absolute difference from 0.5 below 0.04 | Retained anisotropic segregation settings |
| Light-component shell anisotropy (figure script) | Absolute difference below 0.04 against the retained centre-point reference | Eight seeds of 60,000 stars; eight resolved bins |
| Light-component anisotropy (pytest) | Absolute difference below 0.05 against the retained median-radius reference | 60,000 stars; two radial-bias-region shells |

The plotted references use density-weighted shell moments; the existing test
references and thresholds remain explicit above. The figure also exposes
unresolved heavy-component shells rather than assigning them a passing result.

:::{figure} figures/seg_multimass_anisotropy.png
:label: val-multimass-anisotropy
:width: 100%

**Anisotropic sampler checks, 2026-09-28**
(`scripts/validate_multimass_anisotropy.py`). (a) Seed-averaged anisotropy with
standard errors and density-weighted shell references; the annotated heavy point
uses only two of eight seeds. (b) Light-component radial and one-dimensional
tangential dispersions against shell references. (c) Sampled global virial ratios.
(d) Local DF anisotropy for four anisotropy radii. The retained light-component
and global-virial gates pass; unresolved heavy shells are not qualified.
:::

## DF tables — the accelerated path is budget-asserted against its oracle

Three table primitives accelerate density and speed calculations
(`profiles/limepy_tables.py`). `AnisoDensityTable` interpolates the ratio to the
analytic isotropic density on a $160\times160$ grid in
$(\sqrt{W},\log(1+p^2W))$, where $W$ is the dimensionless relative potential
and $p$ is the radius divided by the component anisotropy radius.
`SpeedCDFTable` uses $256\times384$ potential/speed nodes;
`AnisoSpeedCDFTable` uses $192\times96\times384$ potential/anisotropy/speed
nodes with the same coupled anisotropy coordinate. Each speed interval uses
eight interior Gauss–Legendre nodes; cosine-spaced speed nodes resolve both
endpoints. The angular conditional remains a separate calculation.
Pointwise density quadrature is selectable with `aniso_method="quadrature"`.

```{note} Retained numerical checks
Float64 checks on 240 retained density queries give a maximum relative error
of $1.094\times10^{-6}$ against independent adaptive quadrature (budget
$10^{-5}$). With the earlier 48-node anisotropy-speed table, on the same 240 configurations the two speed moments differ by
at most 0.119% (budget 1.5%). These checks cover five truncation indices
$g\in\{0,0.5,1,2.5,3.5\}$, positive potentials down to $10^{-10}$ and
radius ratios up to about 200. They are discrete checks, not a guarantee
over every intervening parameter value.
```

The current figure script checks density interpolation against direct
quadrature, potential profiles against quadrature-built models at $W_0=5,7,9$,
and derivatives against finite differences. The potential budget is
$10^{-4}W_0$ and the derivative relative-error budget is $10^{-3}$.
`component_virial_ratios` uses direct velocity quadrature independently of the
speed-CDF interpolation checked by sampled moments.

Historical timing and memory measurements are not current performance guarantees.
In particular, the present small-sample break-even point remains unqualified.

:::{figure} figures/df_tables.png
:label: val-df-tables
:width: 100%

**DF-table checks, 2026-09-28** (`scripts/validate_df_tables.py`).
(a) Density errors for the retained 512-by-96 comparison grid and current
160-by-160 solver grid. (b) Table/quadrature potential differences for three
models. (c) Local warm timings; a documentation build overlapped this
run, so these are workload observations rather than an isolated performance
qualification. (d) Forward-value finite differences and reverse-AD derivatives,
with maximum relative difference $1.97\times10^{-6}$. All existing script gates
pass. These timings are not a current break-even or scaling guarantee.
:::

## Differentiability

All three Engine A constructors, including `from_imf`, accept an `adjoint`
keyword. Omitting it preserves checkpointed reverse-mode derivatives through
`jax.grad` and `jax.jacrev`. Pass `diffrax.ForwardMode()` for `jax.jvp` or
`jax.jacfwd`, or `diffrax.DirectAdjoint()` when the same configuration needs
both modes. Keep this construction choice static under JIT.

At the retained tight ODE tolerances, 36 comparisons of model
observables at $g=0,0.5,1$ agree with an independent SciPy solver's finite
differences to 0.175% (budget 0.3%). At $g=0$, the derivative is taken from
positive $g$, within the physical parameter domain. Independent integral
checks also cover the density and speed-table derivatives there.

(engine-a-matrix)=
### IMF-to-sampling qualification matrix

The retained matrix checks the derivative of **mean sampled speed** with respect
to the power-law IMF slope and segregation parameter $\delta$. The baseline uses
slope 2.3, stellar masses from 0.1 to 20 solar masses, two components,
$W_0=5$, $g=1$, $\delta=0.4$ and anisotropy radius $r_a=10$ pc with
core radius $r_c=1$ pc. Here $W_0$ sets the central dimensionless potential,
$g$ sets the truncation shape, and $\delta$ sets mass-dependent velocity scales.

| Case | Change from baseline |
|---|---|
| Baseline | None |
| Concentration | $W_0=7$ |
| Truncation boundary | $g=0$ |
| Intermediate truncation | $g=0.5$ |
| No mass-dependent velocity scaling | $\delta=0$ |
| Stronger mass-dependent velocity scaling | $\delta=0.6$ |
| Stronger anisotropy | $r_a=4$ pc |
| Isotropic | `r_a=None` |
| Component count | Four components |

Each case uses three seeds with 400 stars per seed, forward and reverse AD,
and finite-difference steps $10^{-4}$ and $10^{-5}$ in float64. The
$\delta=0$ comparison is one-sided. The ODE uses 1,500 saved nodes and
`xi_max=800`; the requested minimum radial resolution is 600 intervals.
The fit retains its $10^{-6}$ residual target and 60-iteration cap.

:::{note} Bounded result
All **216 derivative comparisons pass** the 0.3% relative-error budget.
The maximum discrepancy is 0.28497%, at $W_0=7$, so that case has a narrow
margin. The maximum fit residual is $8.414\times10^{-7}$; no component labels
change in the finite-difference comparisons. Forward and reverse derivatives
agree within $1.676\times10^{-12}$ absolute.

These measurements were made on the numerical prototype adopted into production
at `fb2cfa1`. Numerical code identity was checked during adoption, and a fresh
production run passed 21 radial, physical and expectation tests. This is not a
second run of the full matrix. One-at-a-time cases do not certify joint parameter
extremes, other IMF families, other observables or population-expectation precision.
:::

```{important} Differentiation scope
The IMF constructor differentiates its converged central-fraction constraint
with an implicit Jacobian–vector product (JVP). Public construction rejects a
nonfinite mass-fraction residual or one larger than `fit_tol`, including under
JIT and AD. This enforces the requested fit criterion; derivative accuracy can
require a tighter fit. These checks do not establish higher derivatives,
accuracy throughout the supported domain at default solver tolerances, or potential derivatives at the
escape boundary $W=0$. Positive-$W$ speed scaling and the analytic limiting
CDF are tested separately from the boundary itself.
```

### Population-moment gradients

`estimate_population_moments` supports isotropic and anisotropic Engine A
models. It averages over component assignments, integrates mass and radius
moments analytically, and estimates the squared-speed moment by conditional
sampling, including the correlation with sampled total stellar mass.

:::{note} Qualified anisotropic configuration
The retained model uses four components from a Maschberger IMF with high-mass
slope $\alpha=2.3$ over $0.1$–$20\,M_\odot$, central potential $W_0=5$,
truncation index $g=1$, segregation $\delta=0.4$, and anisotropy radius
$r_a/r_c=10$. Here $r_c$ is the core scale radius; the mass dependence of
anisotropy is disabled with `eta=0`. Construction uses `fit_tol=1e-10` and
`n_iter=60`, with float64 enabled.

For a target population of 100,000 stars, eight batches each average four
100,000-draw conditional integrations. Derivatives of mean squared speed in
$\alpha$, $\delta$, $W_0$ and $r_a$ all pass: discrepancy below 1%, relative
standard error below 0.25%, and agreement within three standard errors plus
the deterministic reference error. Maximum discrepancy is 0.0653%; maximum
relative standard error is 0.0739%. Forward and reverse AD agree in all four
directions. These integrations use the model centre and rest frame, without
recentering or a virial rescale.
:::

This qualifies the stated configuration and observable, not the complete
anisotropic domain, higher derivatives, arbitrary observables or Engine B.
Engine B has [separate population-moment checks](engine-b-eddington.md#population-moment-gradients). Reproduce this Engine A production check
with `scripts/qualify_anisotropic_population_moments.py --production` from the
repository root with `JAX_ENABLE_X64=true PYTHONPATH=src`; saved results are
protected from overwrite.

### Squared mass and mass-weighted kinetic expectation

`estimate_mass_squared_and_kinetic` returns two Engine A expectations:
$\mathbb{E}[M^2]$, where $M=\sum_i m_i$ is total catalogue mass, and
$\mathbb{E}[2T/N]$, where $T=\tfrac12\sum_i m_i|\boldsymbol v_i|^2$ is
kinetic energy and $N$ is the target star count. Their units are mass squared
and mass times velocity squared. Velocities are measured in the model rest
frame before recentering or virial rescaling. The kinetic observable uses the
star count as its denominator, not catalogue mass.

The retained isotropic model uses four representative-mass components from the
Maschberger IMF over $0.1$–$20\,M_\odot$, high-mass slope $\alpha=2.3$,
segregation parameter $\delta=0.4$, central dimensionless potential $W_0=5$
and truncation index $g=1$. Construction uses `fit_tol=1e-10`, `n_iter=60`;
$G=1$ is retained in model units. This is a 100,000-star population target.

The method integrates component labels analytically. Mass squared is deterministic;
kinetic energy retains conditional sampling and the correlation between each
star's mass and the total mass that sets its velocity scale. Eight independent
batches each average four integration realizations, with 25,000 draws per
component. Equal integration occupancy does not change the physical mixture.

| Check | Criterion | Retained result |
|---|---|---|
| Independent reference derivatives | Radial refinements 1/2/4 and two FD steps; relative difference <= 0.3% | All nonzero derivatives pass; mass delta/W0 zero identities pass the dimensionless $10^{-8}$ criterion |
| Kinetic derivatives in alpha, delta, W0 | Discrepancy <= 1%; relative SE <= 0.25%; consistency within three SE plus empirical reference error | All three pass; maximum discrepancy 0.0918%, maximum relative SE 0.1010% |
| Mass-squared identity | Exact independent categorical sum | Value and both AD modes pass |
| Forward/reverse replay | Same seed and configuration | Both observables pass in all three parameter directions |

The reference uses independent radial quadrature and analytic conditional DF
moments; it shares the equilibrium solution. Qualification was performed on the
conditional candidate at `e21094c`, then transferred to the named public method
at `a7763c5` with a same-seed replay. This transfer is checked separately from the
eight-batch precision exposure.

#### Independent absolute-reference comparison

The canonical NumPy/SciPy LIMEPY solver provides an independent equilibrium
and kinetic integral for this retained configuration. Center-normalized
dispersion alone would conceal a common velocity-scale error. The comparison
therefore uses each component's reference kinetic energy $K_j^{\rm ref}$ and
mass $M_j^{\rm ref}$, total reference mass $M_{\rm ref}$, gravitational constant
$G_{\rm ref}$, King radius $r_{0,\rm ref}$ and common squared speed scale
$s_{\rm ref}^2$. They determine the dimensionless conditional squared speed
$q_j$ and total mass integral $\mu_{\rm ref}$:

```{math}
q_j=\frac{2K_j^{\rm ref}}{M_j^{\rm ref}s_{\rm ref}^2},
\qquad
\mu_{\rm ref}=\frac{G_{\rm ref}M_{\rm ref}}
 {9r_{0,\rm ref}s_{\rm ref}^2}.
```

These enter the [finite-population identities](../10-theory/populations/index.md#finite-population-expectations)
with the solved component number probabilities. The reference uses the matching
central mean-mass convention. Numerical controls, actual fit residuals and source
identity are saved; the finest reference residual is below $4.4\times10^{-13}$.

| First-order comparison | Maximum nonzero relative derivative difference | Retained gate |
|---|---|---|
| Production grid: 2000 ODE nodes, `xi_max=300` | 0.0000944% | 0.3% |
| Resolved grid: 8000 ODE nodes, `xi_max=30` | 0.00000447% | 0.3% |

Both grids pass at FD steps $10^{-4}$ and $10^{-5}$ in $\alpha$, $\delta$ and
$W_0$: 24 derivative/identity cells including the scaled $10^{-8}$ mass-zero
identities. Reference fit/ODE refinement and FD precision pass separately.
The production kinetic value differs by -0.00000479% from the finest reference;
this is a measurement without a newly selected value-accuracy gate.
The comparison shares Progenax's IMF bins and observable definitions. It does
not independently qualify IMF binning, arbitrary domains or stochastic precision.
Cached outputs and fresh public forward AD are checked by
`tests/validation/test_limepy_reference_parity.py`; the four structural cases and
absolute tests pass with the adapter unit tests (18 tests, float64).

```{important} General scalar precision remains separate
The general `estimate_expectation` API retains its first-order scalar contract.
For these observables, the frozen 32-catalogue score exposure fails the stated
precision gates: relative SE is 17.68% for the mass-squared IMF-slope gradient
and 20.97% for the kinetic IMF-slope gradient. The kinetic segregation derivative also
misses its SE gate. The results do not establish estimator bias. The named
conditional method passes its bounded exposure; it does not qualify arbitrary
observables, wider parameter domains, higher derivatives or Engine B use of
this new method.
```

See the [generated method contract](../30-api/cluster.md#api-cluster-multicomponentcluster-estimate_mass_squared_and_kinetic).
Reproduce the frozen exposure without overwriting retained evidence:

```bash
JAX_ENABLE_X64=true PYTHONPATH=src .venv/bin/python scripts/qualify_general_observables.py \
  --conditional --reverse --output /tmp/progenax-mass-kinetic-check.json
```

## How to run

```bash
# retained physics tests (sampling and continuum checks; marked slow)
JAX_ENABLE_X64=true pytest tests/validation/test_multimass_equilibrium_physics.py -q

# DF-table unit suites (budgets + threading)
JAX_ENABLE_X64=true pytest tests/unit/profiles/test_limepy_tables.py tests/unit/profiles/test_limepy_multimass.py -q

# cached independent reference and adapter checks; no reference subprocess
JAX_ENABLE_X64=true PROGENAX_CLEAR_JAX_CACHES=1 PROGENAX_STRICT_REFS=1 PYTHONPATH=src \
  .venv/bin/python -m pytest tests/unit/test_limepy_reference_adapter.py \
  tests/validation/test_limepy_reference_parity.py -q

# nine-case mean-speed derivative matrix (run from the repository root)
JAX_ENABLE_X64=true PYTHONPATH=src .venv/bin/python scripts/check_population_mean_speed_ad.py --risk-matrix

# regenerate the figures with printed PASS/FAIL tables
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multimass_equilibrium.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multimass_anisotropy.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_df_tables.py
```

## Qualification limits

- No N-body evolution or dynamical-stability test is included.
- The segregation sweeps end at $\delta=0.6$; sampled heavy-component virial
  gates cover $\delta\le0.5$.
- Natural-population outer-shell occupancy remains sparse. The separate conditional
  moment check covers both components over 0.06–0.95 tidal radii; it does not
  qualify the unsampled central or outermost intervals.
- Small-sample table efficiency and current memory scaling need isolated measurements.
- The nine-case derivative matrix covers fixed realizations and two differentiated
  parameters. The population-moment check above separately qualifies one anisotropic
  configuration; other observables and the broader parameter domain remain unqualified.

Related validation: [density-defined Engine B](engine-b-eddington.md) and
[primordial mass segregation](mass-segregation.md).
