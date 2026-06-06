---
title: Differentiable inference — natal cloud parameters from cluster substructure
description: The physics-direct, differentiable inference layer of gravoturb_fdf — predict the summary statistic analytically and differentiate it (Gaussianization 2-point + counts-in-cells + a peaks-over-threshold tail block), then sample with HMC. Makes the BM19 density-PDF tail slope α inferable.
---

# Differentiable inference: natal cloud parameters from cluster substructure

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent + fractal-density-field (FDF) pipeline was rebuilt **clean-room** (2026-06) as
the standalone **`gravoturb_fdf`** package — a follow-up-paper feature **excluded from the released
progenax wheel**. Import it as `gravoturb_fdf` (repo-only, under `src/experimental/`). Fresh
validation: `src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`.
```

```{seealso}
This chapter builds on the forward model: the BM19 density PDF ([](density-pdf-fundamentals.md),
[](bm19.md)) and the fractal density field that turns it into a 3-D realization
([](pdf-and-fdf.md)). Here we run that chain *backwards* — from observed substructure to the natal
cloud parameters $\theta = (\mathcal{M}, b, \alpha, \beta)$.
```

A young star cluster is a fossil of the turbulent cloud that made it. Supersonic turbulence sets
the density power spectrum; self-gravity carves the dense, collapsing tail; newborn stars trace
that gas. So the cluster's *spatial substructure* encodes the natal parameters $\theta =
(\mathcal{M}, b, \alpha, \beta)$ — Mach number, turbulent forcing, density-PDF tail slope, and
power-spectrum slope (see [](density-pdf-fundamentals.md) for what each controls). The inference
problem inverts the arrow of causation: **observed substructure → natal cloud physics**. This is
galaxy-clustering-style inference applied to star clusters, with $\beta$ as the pivot (turbulence →
density spectrum → stellar clustering).

## The obstacle, and the breakthrough

The forward model is stochastic and full of non-differentiable steps: a rank sort imposes the
density marginal, categorical sampling places stars, and a minimum-spanning-tree computes the
Cartwright & Whitworth ({cite:t}`Cartwright2004`) $Q$ substructure metric. You cannot
backpropagate through it, so gradient-based inference looks impossible. The standard escape —
simulation-based inference with a neural network {cite:p}`Bairagi2026` — buries the physics and is
simulation-hungry.

The breakthrough is the **cosmology playbook**: do not differentiate the *simulator* — predict the
*summary statistic* analytically as a smooth function of $\theta$, and differentiate that.
Cosmologists never backpropagate through an $N$-body simulation; they predict the matter power
spectrum and fit it. Two facts make this exact here, not merely convenient:

1. **A Gaussian random field is fully specified by its 1-point and 2-point statistics.** A field
   that is a *monotone map* of a Gaussian likewise has vanishing higher-order correlations
   {cite:p}`Neyrinck2011,CarronSzapudi2013`, so the 1-point PDF and 2-point function are *sufficient
   statistics*. If inference uses only those, the phase-randomness of the model is *not* a
   misspecification — we use precisely the statistics it represents faithfully. This is the most
   powerful *and* the most honest use of the model.
2. **The copula separates value from arrangement.** The density *values* are an analytic function
   of $(\mathcal{M}, b, \alpha)$ on a fixed quantile grid; $\beta$ only sets the spatial
   *arrangement*. So $\partial/\partial(\mathcal{M}, b, \alpha)$ of any permutation-invariant
   observable flows cleanly, and $\beta$ enters analytically through the spectrum.

The realization simulator is retained, but demoted: it is the **ground-truth oracle** that
validates the analytic predictions, and the source of mocks and covariances. The CW04 $Q$ metric
is a validation/demo diagnostic ("we reproduce fractal clusters"), never a fit observable.

## The 2-point carrier: Gaussianization

The 2-point observable is the **log-density correlation** $\xi_s(r) = \langle s(\mathbf{x})\,
s(\mathbf{x}+\mathbf{r})\rangle$, where $s = \ln(\rho/\rho_0)$. We use the *log* density, not the
linear density, by necessity: the fat power-law tail makes $\langle\rho^2\rangle$ diverge for
$\alpha \le 2$ — the canonical collapsing slopes — so linear-density 2-point statistics are
formally ill-behaved, while $s$ has finite variance for any $\alpha$. The log-density is also the
*information-optimal* variable: Gaussianizing the field restores information to the two-point
function {cite:p}`Neyrinck2009`, and the log transform is the optimal local Gaussianizer for a
$k^{-\beta}$ spectrum {cite:p}`CarronSzapudi2013`.

The field is a Gaussian $g$ with spectrum $k^{-\beta}$, mapped monotonically to the BM19 marginal.
For any monotone transform $s = T(g)$, the classic Gaussianization result
{cite:p}`ColesJones1991,SzapudiPan2004` gives the transformed correlation as a Hermite series:

```{math}
:label: gaussianization
\xi_s(r) \;=\; \sum_{n\ge 1} \frac{c_n^2}{n!}\,\rho_g(r)^{\,n},
\qquad
c_n \;=\; \big\langle\, T(g)\,\mathrm{He}_n(g)\,\big\rangle ,
```

with $\mathrm{He}_n$ the probabilists' Hermite polynomials and $\rho_g(r)$ the normalized Gaussian
correlation, $\rho_g(0)=1$. The coefficients $c_n$ are 1-D integrals of the copula map, smooth in
$(\mathcal{M}, b, \alpha)$; $\rho_g(r)$ is the Fourier transform of $k^{-\beta}$, analytic in
$\beta$. So $\xi_s(r;\theta)$ is differentiable in *all four* parameters — **no sort, no
realization**. The series converges by $n_{\max} \approx 8$, and the predicted $\xi_s$ matches the
realization oracle to $\sim 0.1\%$.

## The stellar observable: counts-in-cells

Star positions are sampled from the density field, so the natural stellar observable is
**counts-in-cells** (CIC): partition the field into cubic cells and count stars per cell. The count
distribution is a local-Poisson average of the density PDF {cite:p}`SzapudiPan2004,CarronSzapudi2014`,
and the cell count variance is

```{math}
:label: cic-variance
\sigma_N^2(R) \;=\; \bar N \;+\; \bar N^2\, \bar\xi(R),
```

where $\bar N$ is the mean count and $\bar\xi(R)$ is the cell-averaged 2-point function at cell
scale $R$. The Poisson term $\bar N$ is shot noise; the clustering term $\bar N^2 \bar\xi(R)$
carries $\beta$ (the integrated 2-point), while the *shape* of the count distribution (its
over-dispersion) carries $(\mathcal{M}, b, \alpha)$. Counts-in-cells is tail-robust (finite $N$),
unlike the angular pair correlation, and the cell scale $R$ regularizes the fat tail.

## The α wall

The tail slope $\alpha$ — the rarest, densest, *collapsing* gas — resisted inference for two
compounding reasons.

**Stars do not carry $\alpha$.** Cell-averaging smooths over the density peaks and shot noise
buries what remains. An apparent $\alpha$-signal in star counts turned out to be a *sampling
artifact* (multinomial, with-replacement sampling over fine cells); a clean inhomogeneous-Poisson
sampler drops it to negligible. **$\alpha$ is a gas-density observable** — it needs a dust /
extinction / column-density map, not stars.

**The finite-field truncation.** To realize the field, the rank copula maps cell ranks to densities
via $u = (\mathrm{rank}+\tfrac12)/N$, then $s = F^{-1}(u)$. The largest density an $N$-cell field
can produce is $s_{\max} = F^{-1}(1 - 0.5/N)$. For a power-law tail $p(s) \propto e^{-\alpha s}$,

```{math}
:label: smax
s_{\max} \;\approx\; s_t \;+\; \frac{\ln N}{\alpha}.
```

```{admonition} Why brute force fails
:class: important
The tail's dynamic range grows only as $\ln N / \alpha$. Going from $64^3$ to $256^3$ — $64\times$
more cells — extends the tail by just $\ln(64)/\alpha \approx 1.7$ nats. You cannot resolve a
power-law slope by adding resolution; the lever arm grows logarithmically.
```

The realized tail is therefore *truncated*. Fitting the full *infinite-tail* BM19 PDF to a
truncated mock biases $\alpha$ high — the missing far-tail mass makes the realized slope look
steeper. The posterior peaks near $\alpha \approx 2.8$ against a truth of $2.5$, and goes flat for
$\alpha \gtrsim 3$, where the tail holds essentially no cells.

## The fix: peaks-over-threshold

The fix is the **peaks-over-threshold (POT)** estimator from extreme-value statistics — the
mathematics of flood and insurance risk. The key fact is that the exponential is *memoryless*:
above the transition $s_t$ the BM19 tail is exactly $C\,e^{-\alpha s}$, so the *exceedances* above
any fixed threshold $s_{\rm thr} \ge s_t$ are again exponential with rate $\alpha$. Keep only the
gas cells above $s_{\rm thr}$ and model those exceedances as a **truncated exponential** on
$[0, L]$, with $x = s - s_{\rm thr}$ and $L = s_{\max} - s_{\rm thr}$:

```{math}
:label: pot
p(x \mid 0 < x < L) \;=\; \frac{\alpha\,e^{-\alpha x}}{1 - e^{-\alpha L}} .
```

Four properties make this the right tool — each is an assertion in the test suite:

- **Exact, not asymptotic.** The lognormal normalization cancels in the conditional, so $\alpha$ is
  read off the exceedance slope *independent of* $\sigma_s^2(\mathcal{M},b)$. This **breaks the
  $\mathcal{M}$–$\alpha$ degeneracy** that crippled the full-PDF fit.
- **Shift-immune.** A global density zero-point shift moves $s$, $s_{\rm thr}$, and $s_{\max}$
  together; the exceedance $x$ is unchanged. The $\alpha$-block needs no $\langle e^s\rangle = 1$
  normalization.
- **Geometry-free.** It is a pure 1-point marginal — it never touches the spatial-correlation grid,
  so it carries no cross-grid forward bias.
- **Truncation modeled, not ignored.** The $(1 - e^{-\alpha L})$ normalizer accounts for the
  finite-field cutoff, removing the high-$\alpha$ bias at its source.

## The Fisher forecast

Because the exceedances are a clean truncated exponential, the Fisher information per exceedance is
analytic:

```{math}
:label: fisher-alpha
I(\alpha) \;=\; \frac{1}{\alpha^2} \;-\; \frac{L^2\,e^{-\alpha L}}{(1 - e^{-\alpha L})^2},
\qquad
\sigma(\alpha) \;=\; \frac{1}{\sqrt{N_{\rm tail}\, I(\alpha)}} .
```

As $L \to \infty$, $I \to 1/\alpha^2$, recovering the textbook Hill-estimator result
$\sigma(\alpha) = \alpha / \sqrt{N_{\rm tail}}$. At achievable grids $L \approx 2$–$3$, so the
truncation correction inflates $\sigma(\alpha)$ by $\sim 2$–$4\%$; the forecast uses the corrected
form. This converts the "wall" into an honest **survey-design forecast**: to measure $\alpha$ to a
target precision, you need a specific number of independent tail elements $N_{\rm tail}$.

## Putting it together: HMC recovery and forecast

The joint likelihood sums independent blocks — counts-in-cells {eq}`cic-variance` for
$(\mathcal{M}, \beta)$ and the POT exceedance likelihood {eq}`pot` for $\alpha$, plus a soft
barrier that keeps the chain in the valid region $s_t(\theta) \le s_{\rm thr}$. The free parameters
$(\mathcal{M}, \alpha, \beta)$ are sampled in an unconstrained reparametrization with the No-U-Turn
Sampler {cite:p}`HoffmanGelman2014` via blackjax {cite:p}`blackjax`; $b$ is fixed because the data
constrain $(\mathcal{M}, b)$ only through $\sigma_s^2 = \ln(1 + (b\mathcal{M})^2)$.

On a $160^3$ gas map (injection–recovery, $N_{\rm tail} = 510$):

```{list-table} AC16 — joint $(\mathcal{M}, \alpha, \beta)$ recovery on an injected mock
:header-rows: 1

* - parameter
  - posterior
  - truth
  - deviation
* - $\mathcal{M}$
  - $4.88 \pm 0.65$
  - $5.0$
  - $0.18\sigma$
* - $\alpha$
  - $\mathbf{2.533 \pm 0.112}$
  - $\mathbf{2.5}$
  - $\mathbf{0.30\sigma}$
* - $\beta$
  - $3.20 \pm 0.43$
  - $3.0$
  - $0.45\sigma$
```

The $\alpha$ posterior width ($0.112$) matches the truncation-corrected Fisher ($0.113$) to $1\%$ —
the likelihood is *calibrated*, not merely covering — and $\mathrm{corr}(\mathcal{M}, \alpha) =
-0.11$ confirms the degeneracy is broken. The forecast (AC17) validates {eq}`fisher-alpha` to
$\sim 3\%$ against independent draws, with the expected $\sqrt{N}$ scaling (empirical slope $-0.56$
vs ideal $-0.5$).

```python
import jax
from gravoturb_fdf.field.field import gaussian_random_field, rank_copula_field
from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density
from gravoturb_fdf.validation.measure import measure_exceedances
from gravoturb_fdf.inference.likelihood import tail_exceedance_loglike, pot_validity_barrier
from gravoturb_fdf.inference.fisher import sigma_alpha

mach, b, alpha, beta = 5.0, 0.4, 2.5, 3.0
key = jax.random.PRNGKey(0)

# one gas-density realization (the "observed" dust/extinction map)
s = rank_copula_field(gaussian_random_field((128, 128, 128), beta, key), mach, b, alpha)

# threshold in the power-law regime; reduce to exceedances above s_thr
s_thr = float(transition_density(alpha, sigma_s_squared(mach, b))) + 0.75
counts, edges, s_max, n_tail = measure_exceedances(s, s_thr, n_bins=12)

# the POT tail log-likelihood is differentiable in alpha (= theta[2])
import jax.numpy as jnp
theta = jnp.array([mach, b, alpha, beta])
ll = tail_exceedance_loglike(jnp.asarray(counts), jnp.asarray(edges), theta, s_thr, s_max)
forecast = sigma_alpha(alpha, s_max - s_thr, n_tail)      # truncation-corrected sigma(alpha)
print(f"N_tail = {n_tail},  sigma(alpha) forecast = {float(forecast):.3f}")
```

## Caveats and domain of validity

These define what the method does *not* claim.

1. **Injection–recovery, not real data.** AC16 draws the mock from the same BM19 model it fits. It
   validates the inference *machinery* (likelihood + sampler), not that real clouds carry this exact
   tail. The model *is* BM19 by assumption.
2. **The correlation penalty (the dominant real-world limit).** The forecast assumes $N_{\rm tail}$
   *independent* tail elements. A real $\beta = 3$ field's tail cells are spatially clustered — the
   dense gas sits in coherent peaks — so the realized scatter is $\sim 2.5\times$ the i.i.d. bound,
   i.e. $N_{\rm eff} \approx N_{\rm tail}/6$. A real map needs $\sim 6\times$ more tail resolution
   elements than the naive count, analogous to correlated-mode inflation in galaxy surveys.
3. **$\alpha$ requires a gas-density tracer** with enough dynamic range to resolve the tail.
4. **$\mathcal{M}$–$b$ degeneracy.** The data constrain $(\mathcal{M}, b)$ only through
   $\sigma_s^2 = \ln(1 + (b\mathcal{M})^2)$; the 4-parameter Fisher is rank-3 singular. Fix $b$ (or
   constrain it independently) to recover $\mathcal{M}$.
5. **2-D projection.** Real data are 2-D sky positions. The Limber projection introduces cluster
   distance/depth as a nuisance; the projected PDF is narrower than the volumetric one.
6. **Phase-random model.** Valid for 1pt+2pt inference, but do not invert higher-order /
   morphological statistics — real clouds have genuine filamentary phase coherence the GRF cannot
   produce. The 3-point function is reserved as a held-out null test and filament detector.
7. **Relative over absolute.** Absolute $\beta \to \mathcal{M}$ mapping needs validation against real
   gravo-turbulent simulations. Population *trends* and Fisher *forecasts* are far more defensible
   than absolute per-cluster Mach numbers.

## Example science questions

- **Natal turbulence of clusters.** Infer $\beta \to \mathcal{M}$ from observed substructure across
  thousands of LSST clusters: how supersonic was the gas that made each cluster?
- **The collapse state $\alpha$.** From Herschel/JWST/extinction column-density maps, measure the
  density-PDF tail slope — the fraction of gas actively collapsing, tied to star-formation
  efficiency.
- **$\beta$(environment).** Does natal turbulence vary with galactocentric radius, surface density,
  metallicity, or cluster mass? A hierarchical Bayesian study over the cluster population.
- **Joint $(\beta, \alpha_{\rm IMF})$.** Do turbulence morphology and the IMF share an environmental
  driver?
- **Substructure as a clock.** With a differentiable $N$-body engine, separate the initial $\beta$
  from dynamical age in a single snapshot.
- **Survey design.** The forecast answers "to measure $\alpha$ to $10\%$ in this cloud, you need a
  gas map with $N_{\rm eff} \approx X$ independent tail elements" — a concrete depth and resolution.

## References

The density-PDF framework is {cite:t}`FederrathKlessen2012` and {cite:t}`BurkhartMocz2019`; the
dense-gas SFR formalism is {cite:t}`Burkhart2018`. The fractal density field follows the
fractional-Brownian-motion construction of {cite:t}`Lomax2018`, with the density-spectrum slope
$\beta$ from {cite:t}`KimRyu2005`. The Gaussianization machinery is from {cite:t}`ColesJones1991`
and {cite:t}`SzapudiPan2004`, with the information-theoretic basis in
{cite:p}`Neyrinck2009,Neyrinck2011,CarronSzapudi2013,CarronSzapudi2014`; the FFT field generation
and 2-point estimators follow {cite:p}`CarronWolkSzapudi2014,Szapudi2005`. The substructure metric
is {cite:t}`Cartwright2004`. Inference uses the No-U-Turn Sampler {cite:p}`HoffmanGelman2014` via
blackjax {cite:p}`blackjax`, on {cite:p}`JAX`; the simulation-based-inference alternative is
{cite:t}`Bairagi2026`. Module reference and fresh validation numbers:
`src/experimental/gravoturb_fdf/README.md` and `VALIDATION_SUMMARY.md`.
