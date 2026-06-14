---
title: Mass-ratio distributions
description: The conditional mass-ratio distribution g(q | M₁) — power-law plus twin-excess parameterisation, period-averaged vs period-conditional treatments, and the per-mass γ values reduced from Moe & Di Stefano 2017 Table 13.
---

# Mass-ratio distributions

The conditional mass-ratio distribution $g(q \mid M_1)$ is the
component of the {cite:t}`MoeDiStefano2017` joint $f(M_1, q, P, e)$ that the
binary-aware likelihood ([](binary-aware-likelihood.md)) integrates
over at inference time. Getting it right is essential — a
misspecified $g(q)$ produces residual bias on the inferred IMF slope
even after correcting for the binary fraction.

This chapter documents the parametric form, the per-mass and
per-period values, and the period-averaged-vs-period-conditional
trade-off that progenax exposes through `BinaryIMF`.

## The parametric form

For a primary of mass $M_1$, the {cite:t}`MoeDiStefano2017` mass-ratio
distribution combines a power-law in $q$ with a narrow Gaussian
"twin excess" near $q = 1$:

```{math}
:label: gq-form
g(q \mid M_1) \;=\; (1 - f_{\mathrm{twin}})\,\frac{q^{\gamma(M_1)}}{Z_{\mathrm{pl}}}
                  \;+\; f_{\mathrm{twin}}\,\mathcal{N}(q \mid 1,\,\sigma_{\mathrm{twin}})
```

with $Z_{\mathrm{pl}} = \int_{q_{\min}}^{1} q^\gamma\,\mathrm{d}q$ the
power-law normalisation and $\sigma_{\mathrm{twin}} \approx 0.03$ the
twin-peak width. The twin Gaussian is sharply peaked at $q = 1$ and
truncated at $q > 1$; the power-law dominates everywhere except the
narrow $q \in [0.95, 1]$ window.

The two component weights $(1 - f_{\mathrm{twin}})$ and
$f_{\mathrm{twin}}$ guarantee normalisation. Both depend on $M_1$;
$\gamma$ also depends on $P$ in the full {cite:t}`MoeDiStefano2017`
description (see [period-conditional treatment](#period-conditional)
below).

## The power-law slope $\gamma(M_1)$

```{list-table} Period-averaged single-slope $\gamma(M_1)$ — a reduction of {cite:t}`MoeDiStefano2017` Table 13 ($\gamma_{\mathrm{smallq}}/\gamma_{\mathrm{largeq}}$ averaged over $\log P$); a progenax approximation, not a verbatim Moe row.
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

```{list-table} Period-averaged twin-excess fraction $f_{\mathrm{twin}}(M_1)$ — reduced from {cite:t}`MoeDiStefano2017` Table 13 ($\mathcal{F}_{\mathrm{twin}}$ averaged over $\log P$).
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

Solar-type primaries (the $0.8$–$1.2\,\Msun$ row) show the strongest
twin excess. The "twin" terminology is precise:
$f_{\mathrm{twin}}$ is the fraction of systems with $q > 0.95$ *above*
the extrapolated power-law — not the total fraction of near-equal
mass systems. Conflating "twin" with "high-q" is a common
miscalibration in older population-synthesis codes.

## Period-averaged vs period-conditional

The values above are *period-averaged* — they integrate the full
{cite:t}`MoeDiStefano2017` $g(q \mid M_1, P)$ over all observed periods,
weighted by the period distribution. progenax's default `BinaryIMF`
uses these values because:

1. Most IMF-inference targets sample the full range of binary
   periods (resolved-cluster photometric surveys are not
   period-selective).
2. The period-conditional treatment requires sampling $P$ inside the
   forward model and marginalising over it in the likelihood — an
   extra integration dimension.

(period-conditional)=

For surveys that *are* period-selective — spectroscopic surveys
(short-period biased), visual-binary surveys (wide-period biased) —
the period-averaged $\gamma$ is wrong. A future
period-conditional likelihood layer should accept a period-selectivity
function $w(P)$ and replace the integrated $g(q \mid M_1)$ with the
*conditional* $g(q \mid M_1, P)$ from {cite:t}`MoeDiStefano2017` Tables 11 and
12. The current `BinaryIMF` does not export a
`with_period_conditional()` constructor. The conditional treatment
splits $\gamma$ further:

```{list-table} {cite:t}`MoeDiStefano2017` $\gamma$ split into small-$q$ and large-$q$ regimes (period-averaged).
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
saturates around $q \sim 0.3$. progenax's default uses only the
$\gamma_{\mathrm{large}{-}q}$ value (the period-averaged $\gamma$ in the
table above), which is appropriate for *total* mass-function analyses
but loses the small-$q$ structure relevant to spectroscopic surveys.

## Sampling

Sampling $q$ from {eq}`gq-form` is straightforward. progenax exposes
this through `BinaryIMF.sample_mass_ratios(key, m1)`; internally the
default `MoeDiStefano2017.sample_given_primary(key, m1)` does:

1. Draw a uniform $u \sim \mathcal{U}(0, 1)$.
2. With probability $1 - f_{\mathrm{twin}}(m_1)$: invert the
   power-law CDF $\int_{q_{\min}}^{q} q'^\gamma\,\mathrm{d}q' / Z_{\mathrm{pl}}$
   analytically.
3. Otherwise: draw from the twin Gaussian $\mathcal{N}(1,
   \sigma_{\mathrm{twin}})$, truncated at $q \in [\max(0.95, q_{\min}), 1]$.

The branch choice is implemented via `lax.cond` with a smooth-blend
fallback to keep the sampling differentiable in $f_{\mathrm{twin}}$
when the binary fraction is being inferred.

## Likelihood evaluation

For the binary-aware likelihood ([](binary-aware-likelihood.md)),
$g(q \mid M_1)$ enters the integrand directly. The 128-point
Gauss-Legendre quadrature over $q \in [q_{\min}, 1]$ evaluates
$g(q \mid M_1)$ at each node; for the period-averaged form, the
piecewise nature of $\gamma(M_1)$ is handled inside `BinaryIMF` via
mass-bin lookups.

The full likelihood structure and its computational cost are
documented at [](binary-aware-likelihood.md); this chapter only
covers the $g(q)$ ingredient.

## Implementation in progenax

```python
from progenax.imf import BinaryIMF, MoeDiStefano2017, PowerLawIMF

m1 = jnp.array([0.5, 1.0, 2.0, 5.0, 20.0])
q = jnp.linspace(0.1, 1.0, 128)

q_dist = MoeDiStefano2017()
pdf = q_dist.pdf_given_primary(q[None, :], m1[:, None])

binary_imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa())
m1_sample, m2_sample, is_binary = binary_imf.sample_systems(key, 1000)
```

The `MoeDiStefano2017` methods implement the mass-binning above; they
are JIT-safe (`jnp.where`
piecewise) and differentiable in $m_1$ if the user wants to fit
mass-bin boundaries (rare but supported).

## Domain of validity

1. **Period-averaging assumption** — period-selective surveys need a
   future period-conditional likelihood layer; the current
   `BinaryIMF` is period-averaged.
2. **$q \ge q_{\min}$ floor** — all distributions in this chapter
   assume $q \ge 0.1$, matching {cite:t}`MoeDiStefano2017`'s observational
   completeness threshold. Below $q = 0.1$, companion detection is
   unreliable, so the calibration cannot constrain that regime.
3. **Single-$\gamma$ approximation** in the default `BinaryIMF` —
   loses small-$q$ structure relevant to spectroscopic-binary
   inference. Treat the period-conditional treatment above as design
   guidance if the small-$q$ regime matters for your science.
4. **Metallicity assumed solar** — {cite:t}`Moe2019` finds
   metallicity-dependent close-binary fractions, which progenax does
   not currently incorporate into $g(q)$.

## References

The functional form and per-mass values are {cite:t}`MoeDiStefano2017` Tables
10–12. The period-conditional split is from the same source. The
metallicity dependence is {cite:t}`Moe2019`. For the broader
multiplicity context see [](multiplicity-statistics.md).
