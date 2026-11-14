---
title: Mass-ratio distributions
description: The conditional mass-ratio distribution g(q | M₁) — power-law plus twin-excess parameterisation, period-averaged vs period-conditional treatments, and the per-mass γ values reduced from Moe & Di Stefano 2017 Table 13.
---

# Mass-ratio distributions

The conditional mass-ratio distribution $g(q \mid M_1)$, with
dimensionless $q=m_2/m_1$ and primary mass $M_1$ in solar masses, is the
component of the {cite:t}`MoeDiStefano2017` joint $f(M_1, q, P, e)$ that the
binary-aware likelihood ([](binary-aware-likelihood.md)) integrates
over at inference time. Getting it right is essential — a
misspecified $g(q)$ produces residual bias on the inferred IMF slope
even after correcting for the binary fraction.

This chapter documents the parametric form, the per-mass and
per-period values, and the period-averaged-vs-period-conditional
trade-off between the default `BinaryIMF` sampler and the full
period-conditioned companion models.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the conditional mass-ratio distribution $g(q \mid M_1)$ that the binary-aware likelihood integrates over; no prior binary-statistics literature assumed.
**Prerequisites:** the [classical IMFs](classic.md) (the single-star backbone); the broader [multiplicity statistics](multiplicity-statistics.md) for context.
**You'll get:** the power-law-plus-twin-excess form of $g(q)$, the per-mass $\gamma(M_1)$ and $f_{\mathrm{twin}}(M_1)$ values, the period-averaged-vs-period-conditional trade-off, and how progenax samples $q$.
:::

## The parametric form

The reduced `MoeDiStefano2017` model approximates the mass-ratio
statistics with a power law in $q$ and a narrow Gaussian component
near $q=1$. This is a progenax reduction of the source calibration:

```{math}
:label: gq-form
g(q \mid M_1) \;=\; (1 - f_{\mathrm{twin}})\,\frac{q^{\gamma(M_1)}}{Z_{\mathrm{pl}}}
                  \;+\; f_{\mathrm{twin}}\,
                  \frac{\mathcal{N}(q \mid 1,\,\sigma_{\mathrm{twin}})}{Z_{\mathrm{twin}}},
\qquad q\in[q_{\min},1].
```

with $Z_{\mathrm{pl}} = \int_{q_{\min}}^{1} q^\gamma\,\mathrm{d}q$ the
power-law normalization, $Z_{\mathrm{twin}}=\int_{q_{\min}}^1
\mathcal{N}(q\mid1,\sigma_{\mathrm{twin}})\,\mathrm{d}q$ the Gaussian
normalization, and $\sigma_{\mathrm{twin}}=0.03$ the dimensionless
twin-peak width. The density is zero outside the displayed support.
The twin Gaussian is sharply peaked at $q=1$; the power law dominates except in the
narrow $q \in [0.95, 1]$ window.

The two component weights $(1 - f_{\mathrm{twin}})$ and
$f_{\mathrm{twin}}$ guarantee normalization when both component densities
are normalized on that support. Both depend on $M_1$;
$\gamma$ also depends on $P$ in the full {cite:t}`MoeDiStefano2017`
description (see [period-conditional treatment](#period-conditional)
below).

```{note} Two different twin-weight conventions
`MoeDiStefano2017` uses a Gaussian mixture weight for all companions
on $[q_{\min},1]$. `MoeDiStefano2017Full` uses a uniform excess block
on $[0.95,1]$ and the Table-13 excess fraction **within the $q>0.3$
population**. Those weights are not interchangeable.
```

## The power-law slope $\gamma(M_1)$

Period-averaged single-slope $\gamma(M_1)$ — a reduction of {cite:t}`MoeDiStefano2017` Table 13 ($\gamma_{\mathrm{smallq}}/\gamma_{\mathrm{largeq}}$ averaged over $\log P$); a progenax approximation, not a verbatim Moe row.

```{list-table}
:header-rows: 1

* - Primary mass
  - $\gamma$
  - Behaviour at typical $q$
* - $M_1 < 0.8\,\Msun$
  - 0.4
  - Preference for near-equal masses ($q^{0.4}$ rises toward $q = 1$)
* - $0.8$–$1.2\,\Msun$
  - 0.3
  - Mild preference for equal masses
* - $1.2$–$3.5\,\Msun$
  - 0.0
  - Flat (uniform in $q$)
* - $M_1 > 3.5\,\Msun$
  - $-0.5$
  - Preference for unequal masses ($q^{-0.5}$ rises toward $q \to 0$)
```

The trend reverses across the mass spectrum: massive stars
preferentially have *low-$q$* companions, while low-mass stars prefer
*equal-mass* companions. Physically, this is consistent with
fragmentation cascades feeding low-mass companions to massive
primaries, while low-mass primaries form alongside near-equal-mass
companions in shared accretion environments.

## The twin fraction $f_{\mathrm{twin}}(M_1)$

Gaussian mixture weight $f_{\mathrm{twin}}(M_1)$ in the reduced model — a progenax approximation informed by {cite:t}`MoeDiStefano2017` Table 13, not the full model's conditional excess fraction.

```{list-table}
:header-rows: 1

* - Primary mass
  - $f_{\mathrm{twin}}$
  - Pattern
* - $M_1 < 0.8\,\Msun$
  - 0.05
  - Modest twin excess
* - $0.8$–$1.2\,\Msun$
  - **0.10**
  - Peak — solar-type strongest twin signature
* - $1.2$–$3.5\,\Msun$
  - 0.08
  - Slightly lower
* - $M_1 > 3.5\,\Msun$
  - 0.03
  - Massive stars rarely twin
```

Solar-type primaries (the $0.8$–$1.2\,\Msun$ row) have the largest
Gaussian mixture weight in this reduced model. This weight is not the
total fraction with $q>0.95$: the power law contributes there too,
and the truncated Gaussian extends below 0.95. The full model below
implements the distinct Table-13 excess-fraction convention.

## Period-averaged vs period-conditional

The values above define a period-independent reduction informed by
{cite:t}`MoeDiStefano2017`. The implementation stores these coarse
coefficients directly; it does not compute a weighted integral of the
full conditional law. A period mixture of power laws is generally not
a single power law. The reduced `BinaryIMF` is useful when:

1. The selected population can be represented adequately without an
   explicit period variable. Angular resolution, contrast and survey
   cadence can make even photometric samples period selective.
2. The period-conditional treatment requires sampling $P$ inside the
   forward model and marginalising over it in the likelihood — an
   extra integration dimension.

(period-conditional)=

For surveys that *are* period-selective — spectroscopic surveys
(short-period biased), visual-binary surveys (wide-period biased) —
the period-averaged $\gamma$ is wrong. A future
period-conditional likelihood layer should accept a period-selectivity
function $w(P)$ and replace the integrated $g(q \mid M_1)$ with the
*conditional* $g(q \mid M_1, P)$ from {cite:t}`MoeDiStefano2017` Table 11
(the per-$\log P$ $\gamma_{\mathrm{smallq}}/\gamma_{\mathrm{largeq}}$
breakdown for solar-type primaries, with the analogous early-type
tables). The current `BinaryIMF` does not export a
`with_period_conditional()` constructor. The conditional treatment
splits $\gamma$ further:

Illustrative coarse small-$q$ and large-$q$ slopes; this table is not
a verbatim source grid or the full model's interpolated coefficients.

```{list-table}
:header-rows: 1

* - Primary mass
  - $\gamma_{\mathrm{small}{-}q}$ ($0.1 < q < 0.3$)
  - $\gamma_{\mathrm{large}{-}q}$ ($0.3 < q < 1$)
  - Used as
* - $M_1 < 0.8\,\Msun$
  - 0.3
  - 0.4
  - $\gamma$ (avg)
* - $0.8$–$1.2\,\Msun$
  - 0.2
  - 0.3
  - $\gamma$ (avg)
* - $1.2$–$3.5\,\Msun$
  - $-0.4$
  - 0.0
  - $\gamma$ (avg)
* - $M_1 > 3.5\,\Msun$
  - $-1.0$
  - $-0.5$
  - $\gamma$ (avg)
```

The two-segment fit reflects the observation that low-$q$ companions
follow a different distribution than high-$q$ ones — the cascade-style
fragmentation that produces low-$q$ companions to massive primaries
saturates around $q \sim 0.3$. the reduced model uses the stored single-slope values in its
first table, which loses the small-$q$ structure. Its adequacy for a total mass-function
analysis or a selected survey requires a comparison under that survey's
period and mass-ratio selection.

The implemented `MoeDiStefano2017Full.pdf(q, masses, periods)` conditions
on primary mass and orbital period $P$ in days. At its default
$q_{\min}=0.1$ and break $q_b=0.3$, define the continuous unnormalized
power law $\phi(q)=q^{\gamma_s}$ below $q_b$ and
$\phi(q)=q_b^{\gamma_s-\gamma_l}q^{\gamma_l}$ above it. Here
$\gamma_s$, $\gamma_l$ and excess fraction $F$ come from the
mass/period grid. Let $I_A=\int_{q_{\min}}^{q_b}\phi(q)\,\mathrm{d}q$,
$I_B=\int_{q_b}^{1}\phi(q)\,\mathrm{d}q$, and
$A_t=F I_B/(1-F)$. Then

```{math}
:label: gq-full
g(q\mid M_1,P)=
\frac{\phi(q)+A_t\,\mathbf{1}_{[0.95,1]}(q)/0.05}{I_A+I_B+A_t},
\qquad q\in[q_{\min},1].
```

The indicator $\mathbf{1}_{[0.95,1]}$ is one inside the twin block
and zero outside. This local normalization gives
$A_t/(I_B+A_t)=F$, preserving the excess fraction in the large-$q$
population. The density is zero outside its support. See the
[full-model card](../../15-model-reference/binaries.md#card-moe_pqe)
for code/test bindings; the Gaussian reduction in {eq}`gq-form` is a
different sampler. The implemented grid interpolation clamps outside
its mass/period nodes; that numerical extrapolation is not a wider-domain
empirical qualification.

## Sampling

Sampling $q$ from {eq}`gq-form` is straightforward. progenax exposes
this through `BinaryIMF.sample_mass_ratios(key, m1)`; internally the
default `MoeDiStefano2017.sample_given_primary(key, m1)` does:

1. Draw a uniform $u \sim \mathcal{U}(0, 1)$.
2. With probability $1 - f_{\mathrm{twin}}(m_1)$: invert the
   power-law CDF $\int_{q_{\min}}^{q} q'^\gamma\,\mathrm{d}q' / Z_{\mathrm{pl}}$
   analytically.
3. Otherwise: draw from the twin Gaussian $\mathcal{N}(1,
   \sigma_{\mathrm{twin}})$, truncated at $q\in[q_{\min},1]$.

The reduced sampler uses a hard uniform comparison and `jnp.where`
to select the twin or power-law draw. It has no smooth-blend fallback;
fixed-draw derivatives do not account for probability moving between
these discrete branches.

## Likelihood evaluation

For the binary-aware likelihood ([](binary-aware-likelihood.md)),
$g(q \mid M_1)$ enters the integrand directly. The 128-point
Gauss-Legendre quadrature over candidate primary mass evaluates
$g(M/m_1-1\mid m_1)$ at each node; the script-owned likelihood
calls `MoeDiStefano2017.pdf_given_primary` for its mass-bin lookup.

The full likelihood structure and its computational cost are
documented at [](binary-aware-likelihood.md); this chapter only
covers the $g(q)$ ingredient.

## Implementation in progenax

```python
import jax
import jax.numpy as jnp
from progenax.imf import BinaryIMF, MoeDiStefano2017, PowerLawIMF

key = jax.random.PRNGKey(0)
m1 = jnp.array([0.5, 1.0, 2.0, 5.0, 20.0])
q = jnp.linspace(0.1, 1.0, 128)

q_dist = MoeDiStefano2017()
# pdf_given_primary takes a SCALAR primary mass (the power-law branch uses
# lax.cond, whose predicate must be scalar) — vmap over m1 for a grid:
pdf = jax.vmap(lambda m: q_dist.pdf_given_primary(q, m))(m1)  # shape (5, 128)

binary_imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa())
m1_sample, m2_sample, is_binary = binary_imf.sample_systems(key, 1000)
```

The `MoeDiStefano2017` methods implement the mass-binning above; they
are JIT-safe (`jnp.where` piecewise) and constant within primary-mass
bins, with discontinuities at their boundaries. The
`pdf_given_primary` expects a *scalar* $m_1$ per call — broadcast across
a mass array with `jax.vmap`, as above.

## Domain of validity

1. **Period-averaging assumption** — period-selective surveys need a
   future period-conditional likelihood layer; the current
   `BinaryIMF` is period-averaged.
2. **Empirical support.** The Moe calibration uses $q\ge0.1$; its
   default reduced and full implementations use that lower bound.
   An uncalibrated extension below it needs a physical assumption.
   Abstract power-law distributions can use other bounds; at
   $q_{\min}=0$, normalization requires $\gamma>-1$.
3. **Single-$\gamma$ approximation** in the default `BinaryIMF` —
   loses small-$q$ structure relevant to spectroscopic-binary
   inference. Treat the period-conditional treatment above as design
   guidance if the small-$q$ regime matters for your science.
4. **Metallicity assumed solar** — {cite:t}`Moe2019` finds
   metallicity-dependent close-binary fractions, which progenax does
   not currently incorporate into $g(q)$.

## Implementation, validation & references

- **In code:** `src/progenax/imf/binary/moe_di_stefano.py`
  (`MoeDiStefano2017` and `MoeDiStefano2017Full`) and
  `src/progenax/imf/binary/mass_ratio.py` (`PowerLawMassRatio`,
  `TwinPeakedMassRatio`), used by `BinaryIMF` in
  `src/progenax/imf/binary/imf.py` — see the [IMF API](../../30-api/imf.md).
- **Validated in:** [IMF statistics](../../50-validation/imf-statistics.md)
  and [binary-aware recovery](../../50-validation/binary-imf.md), which
  check the sampled $q$ distribution against $g(q \mid M_1)$.
- **Primary sources:** the functional form is {cite:t}`MoeDiStefano2017`
  Table 1 / Eq. 2 (per-mass $\gamma$ and $f_{\mathrm{twin}}$ from Table 13,
  the period-conditional split from Table 11); the metallicity dependence
  is {cite:t}`Moe2019` — full notes in the
  [bibliography](../../99-bibliography/per-paper/moe-distefano-2017.md).
  For the broader multiplicity context see
  [](multiplicity-statistics.md).
