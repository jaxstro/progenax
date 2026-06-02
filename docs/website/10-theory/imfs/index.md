---
title: Initial mass functions
description: progenax's IMF section — from canonical Salpeter / Kroupa / Chabrier / Maschberger through binary-aware Moe & Di Stefano and environment-dependent IGIMF.
---

# Initial mass functions

The **stellar initial mass function** $\xi(m) \equiv \mathrm{d}N/\mathrm{d}m$
is the birth-mass distribution of stars formed in a single
star-formation event. It governs chemical enrichment, supernova rates,
and the integrated colours of stellar populations
{cite:p}`Salpeter1955,Kroupa2001,Chabrier2003`. progenax implements the
IMF as one of its three orthogonal IC ingredients
([](../ic-philosophy.md)) and provides multiple parameterisations
ranging from textbook (Salpeter, Kroupa, Chabrier) through smooth
analytically-invertible (Maschberger) to physically-detailed
(binary-aware Moe & Di Stefano, environment-dependent IGIMF).

## Map of the IMF chapters

```{list-table}
:header-rows: 1

* - Chapter
  - Scope
  - Class in progenax
* - [](classic.md)
  - Salpeter, Kroupa, Chabrier, Maschberger, truncated power-law
  - `PowerLawIMF`, `Maschberger`, `ChabrierIMF`, `TruncatedIMF`
* - [](multiplicity-statistics.md)
  - {cite:t}`Moe2017` joint $f(M_1, q, P, e)$, mass-dependent binary fractions, three period regimes
  - Backing data for `BinaryIMF`
* - [](mass-ratio-distributions.md)
  - Power-law $q^\gamma$ + twin-excess parameterisation, period-conditional vs period-averaged
  - Used by `BinaryIMF`'s integrand
* - [](binary.md)
  - Full binary-aware IMF chapter — system mass function, the marginalisation likelihood, "confidently wrong" regime
  - `BinaryIMF`
* - [](binary-aware-likelihood.md)
  - The Gauss-Legendre quadrature that marginalises over $q$ at inference time
  - schematic likelihood; `BinaryIMF` currently exposes sampling helpers
* - [](observation-operators.md)
  - Mass addition vs. flux addition vs. multi-band CMD; how the choice scales the bias
  - Coordinates with the survey forward-model layer
* - [](environment.md)
  - Marks+12 cluster-scale variation + Jeřábková+18 IGIMF
  - `BirthEnvironment`, `env_to_imf_params`, `alpha3_*`
```

The chapters split a large topic into focussed pieces. New readers
should start with [](classic.md), then jump to [](binary.md) for the
binary-aware framework end to end before drilling into the
sub-chapters as needed.

## Common API contract

Every IMF in progenax satisfies the `IMFProtocol`:

```python
class IMFProtocol(Protocol):
    m_min: float
    m_max: float

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Log-PDF for likelihood evaluation."""
        ...

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Cumulative number fraction below mass m."""
        ...

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF."""
        ...

    def sample(
        self,
        key: PRNGKey,
        n: int,
    ) -> Float[Array, "N"]:
        """Draw N stellar masses."""
        ...

    def mean_mass(self) -> float:
        """Expected mass."""
        ...
```

`sample` uses inverse-CDF for the analytically-invertible IMFs
(Maschberger; truncated power-law) and a fixed-iteration Newton
solver for the rest. `logpdf` is the workhorse for the scalar IMF
families. Binary-aware inference is described conceptually in this
section, but the current `BinaryIMF` API exposes sampling helpers
(`sample_primaries`, `sample_mass_ratios`, `sample_systems`, and
`sample_all_masses`) rather than an exact `logpdf`. Environment-dependent
IMF support currently maps a `BirthEnvironment` to IMF parameters
rather than exposing an `EnvironmentIMF` class.

## Notation conventions

```{list-table}
:header-rows: 1

* - Symbol
  - Meaning
* - $\xi(m)$, $f(m)$
  - Stellar IMF (number per unit mass) — used interchangeably
* - $\alpha$
  - Power-law index, $\xi(m) \propto m^{-\alpha}$ at high mass
* - $m_1$
  - Primary mass in a binary
* - $q = m_2 / m_1$
  - Binary mass ratio, $q \in [0, 1]$
* - $f_b(m_1)$
  - Binary fraction (probability of having a companion) as a function of primary mass
* - $g(q \mid m_1)$
  - Conditional mass-ratio distribution
* - $\xi(m \mid \boldsymbol{\theta})$
  - IMF parameterised by some vector $\boldsymbol{\theta}$ (e.g. $\alpha$, $\rho_{\mathrm{cl}}$, [Fe/H])
```

The convention $\alpha = 2.35$ for the Salpeter slope is universal in
this section. {cite:t}`Marks2012` use a different segmentation
(3-segment vs progenax's 4-segment); the [](environment.md) chapter
documents the conversion in detail.

## Composability with profiles, velocities, and modifiers

IMFs compose orthogonally with [](../spatial-profiles/index.md) and
[](../velocity-dfs/index.md): any IMF can pair with any spatial profile
and velocity DF. The IMF determines the *masses* in `(masses, positions,
velocities)`; the profile determines the *positions*; the DF
determines the *velocities*. The mass-segregation modifier
([](../tidal-and-substructure/mass-segregation.md)) couples IMF and
profile by re-pairing high-mass particles to low-energy orbits, but
the underlying $\xi(m)$ is preserved.

## References

The canonical IMF references are {cite:t}`Salpeter1955`,
{cite:t}`Kroupa2001`, {cite:t}`Chabrier2003`, {cite:t}`Maschberger2013`.
For binary statistics: {cite:t}`Sana2012`, {cite:t}`Moe2017`,
{cite:t}`Moe2019`. For environment dependence: {cite:t}`Marks2012`,
{cite:t}`Jerabkova2018`. Each per-chapter `References` block points at
the specific result(s) used.
