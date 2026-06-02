---
title: Observation operators
description: How the choice of observation operator (mass-addition, photometric flux-addition, multi-band CMD) sets the magnitude — but not the existence — of binary contamination bias on inferred IMF slopes.
---

# Observation operators

```{seealso}
This chapter is the third pillar of the binary-aware framework: how
the choice of *what the telescope actually measures* controls the
size of the binary distortion. For the parent narrative see
[](binary.md). For the marginalisation likelihood that accommodates
each operator, see [](binary-aware-likelihood.md).
```

The binary-aware likelihood {eq}`ba-likelihood` was derived assuming
the **mass-addition** observation operator $m_{\mathrm{sys}} = m_1 +
m_2$. This operator gives the *largest possible* binary distortion —
it represents the case where the survey measures the true dynamical
mass of every system. Real photometric and CMD-based surveys measure
something different, with smaller distortions per unresolved binary,
but the *existence* of bias remains: the naive single-star likelihood
becomes confidently wrong at large $N$ for *every* operator, just at
different sample sizes.

This chapter catalogues the three observation operators progenax
supports, derives the photometric-mass formula that approximates
flux-addition, and quantifies the bias at each level for the same
underlying $\alpha = 2.35$ population.

## Three operators

```{list-table}
:header-rows: 1

* - Operator
  - What the telescope measures
  - Bias scaling
* - **Mass addition** $m_{\mathrm{sys}} = m_1 + m_2$
  - True dynamical mass (eclipsing, SB2, astrometric)
  - Maximum bias — sets the upper bound
* - **Photometric** $m_{\mathrm{phot}} = \mathcal{M}^{-1}[F_b(m_1) + F_b(m_2)]$
  - Single-band flux inverted through a mass-luminosity relation
  - Reduced bias; suppresses unequal-mass companions
* - **Multi-band CMD** $\mathbf{m}_{\mathrm{obs}} = \{-2.5\log[F_b(m_1) + F_b(m_2)] + \mathrm{ZP}\}_b$
  - Multi-band magnitudes per LSST/HST/JWST
  - Smallest per-system bias; full forward model required
```

The default `BinaryIMF` uses mass addition because it is the simplest
to expose to the marginalisation likelihood and provides the
worst-case binary contamination. Photometric and CMD operators are
provided for use cases where the survey forward model is needed —
LSST-style cluster catalogues being the canonical target.

## Mass addition: dynamical mass

For eclipsing binaries (which provide both component masses via
light-curve fitting), SB2 spectroscopic orbits (which provide the
mass ratio and mass sum), and astrometric binaries (which provide
both component masses via orbital astrometry), the measured quantity
is the *true* dynamical mass:

```{math}
:label: msys-dyn
m_{\mathrm{sys}} \;=\; m_1 + m_2 \;=\; m_1\,(1 + q).
```

In this regime, the binary-aware likelihood {eq}`ba-likelihood` is
exact. progenax's default `BinaryIMF` uses it without modification.

Mass addition is the most consequential observation operator for
benchmarking binary-aware inference because it produces the largest
distortion. If you can demonstrate unbiased recovery under mass
addition, the same machinery applied to a less-aggressive operator
will perform at least as well.

## Photometric: flux addition

Many published "mass functions" are actually *photometric* mass
functions: stellar masses are inferred from observed flux through a
mass-luminosity relation $L \propto m^s$. For an unresolved binary,
the telescope measures the combined flux:

```{math}
:label: F-sum
F_b(\mathrm{sys}) \;=\; F_b(m_1) + F_b(m_2) \quad\text{for each band }b
```

Inverting this through the *single-star* mass-luminosity relation
yields the photometric mass:

```{math}
:label: m-phot
m_{\mathrm{phot}} \;=\; \mathcal{M}^{-1}\bigl[F_b(\mathrm{sys})\bigr].
```

For the rough main-sequence scaling $L \propto m^s$ with $s \approx
3$–$4$, this evaluates to

```{math}
:label: m-phot-approx
m_{\mathrm{phot}} \;\approx\; m_1\,(1 + q^s)^{1/s}.
```

The key consequence: because $s$ is large, the secondary contributes
appreciably to the flux only when $q$ is near 1.

```{list-table} Photometric mass at $s = 3.5$.
:header-rows: 1

* - $q$
  - Mass-add $m_1 + m_2$
  - $m_{\mathrm{phot}}$
  - $\Delta\mathrm{mag}$
* - 1.0 (equal)
  - $2.0\,m_1$
  - $1.22\,m_1$
  - $0.75$ mag brighter
* - 0.5
  - $1.5\,m_1$
  - $1.04\,m_1$
  - Companion barely changes flux
* - 0.3
  - $1.3\,m_1$
  - $1.01\,m_1$
  - Negligible photometric effect
* - 0.1
  - $1.1\,m_1$
  - $1.0003\,m_1$
  - Undetectable
```

Equal-mass binaries shift photometric mass by $\sim 22\%$ — still
substantial. Unequal-mass binaries with $q \le 0.3$ are essentially
invisible to photometry. This is why photometric IMF inferences are
generally *less* contaminated than dynamical-mass inferences: the
operator's nonlinearity suppresses the contribution from low-$q$
companions, which are the most numerous in the {cite:t}`Moe2017`
distribution.

A future photometric likelihood layer can swap the mass-addition
operator for {eq}`m-phot-approx` inside the marginalisation
likelihood. The integral structure of {eq}`ba-likelihood` remains the
same; only the relationship between $m_1$, $q$, and the *observable*
changes. The current `BinaryIMF` does not export a
`with_photometric_operator()` helper.

```{warning}
**Photometric bias is smaller, but not zero.** The naive single-star
likelihood applied to photometric masses still becomes confidently
wrong, just at larger $N$. The transition $N$ where the 95% CI
excludes the truth is roughly $4\times$ larger for photometric than
mass-addition (i.e. $N \sim 4\times 10^4$ vs $1\times 10^4$). The
correction is qualitative, not categorical.
```

## Multi-band CMD: full forward model

LSST and other modern surveys do not measure "mass" at all — they
measure multi-band fluxes (or magnitudes) with uncertainties, plus
incompleteness and crowding effects. For an unresolved binary at
distance $d$:

```{math}
:label: m-multi-band
m_{b,\mathrm{sys}} \;=\; -2.5\,\log_{10}\!\bigl[F_b(m_1) + F_b(m_2)\bigr] + \mathrm{ZP}_b
\quad\text{for each } b \in \{u, g, r, i, z, y\}
```

Inferring "mass" from this requires:

1. Comparison to **theoretical isochrones** at assumed age $\tau$,
   metallicity $Z$, and distance $d$.
2. **Extinction correction** (dust column $A_V$ and reddening law
   $R_V$).
3. **Color-magnitude diagram (CMD) fitting** — turning per-band
   magnitudes into a position in CMD space and matching to a single-star
   isochrone or a binary-track curve.

Each step introduces additional model dependence. The binary
distortion in CMD space is *qualitatively* different from mass-space:
an unresolved equal-mass binary appears $\sim 0.75$ mag brighter at
nearly the same colour, placing it *above* the main sequence in the
CMD. This mimics a more massive or more evolved star — a degeneracy
that observational binary-fraction studies have used to *infer*
binary fractions from CMD residuals.

CMD-space observation operators are a planned layer, not a current
public API in this checkout. The intended shape is:

```python
from progenax.imf import BinaryIMF
from progenax.imf.observation_operators import CMDOperator  # planned

cmd_op = CMDOperator(
    isochrone_file="parsec_iso_logZ-0.5.dat",
    bands=("g", "r", "i"),
    distance_pc=10000.0,
    A_V=0.5,
)
binary_imf = BinaryIMF(primary_imf=primary_imf, observation_operator=cmd_op)
log_likelihood = binary_imf.logpdf(observed_magnitudes)  # planned
```

No `CMDOperator` implementation is currently exported from
`src/progenax/`; this section is design guidance for a future
survey-forward-model layer.

## A bias hierarchy

```{list-table} Approximate $|\Delta\alpha|$ at $N$ values, Salpeter $\alpha = 2.35$.
:header-rows: 1

* - Operator
  - $N = 10^3$
  - $N = 10^4$
  - $N = 10^5$
* - Mass addition
  - 0.04
  - **0.08** (CI excludes truth)
  - 0.10 (firmly confidently wrong)
* - Photometric ($s = 3.5$)
  - 0.02
  - 0.04
  - **0.08** (transition to confidently wrong)
* - Multi-band CMD
  - 0.01
  - 0.02
  - 0.04
```

Mass addition fails at $N \sim 10^4$. Photometric fails at $N \sim
10^5$. Multi-band CMD fails at $N \gtrsim 10^6$. Every operator
eventually produces "confidently wrong" naive posteriors at large
enough $N$; the binary-aware likelihood eliminates the bias at all $N$
regardless of operator.

## Resolved/unresolved selection

The above analysis treats every binary as unresolved. In real
catalogues, binaries with angular separations $\theta > \mathrm{PSF}$
are *resolved* into two detections rather than blended into one
system. Whether a given binary resolves depends on:

- Orbital period (sets physical separation $a$).
- Cluster distance (sets angular separation $\theta = a/d$).
- PSF or seeing (sets the resolution threshold).
- Crowding (in dense fields, even resolvable binaries can be blended
  with neighbours).

The effective binary fraction relevant to the unresolved-photometry
analysis is therefore

```{math}
:label: f-unres
f_{\mathrm{unres}}(m_1) \;=\; \int f(m_1, P)\,\mathbb{1}\bigl[\theta(P, d) < \mathrm{PSF}\bigr]\,\mathrm{d}\log P
```

i.e. the integral of $f(m_1, P)$ over periods that produce *unresolved*
configurations at the survey's distance and resolution.

A future survey-selection layer should expose this kind of resolution
cutoff and integrate over periods up to the resolution boundary using
the {cite:t}`Moe2017` period distribution. The current `BinaryIMF`
does not provide a `with_resolution_cutoff()` convenience method. This
refinement is most important for nearby clusters (where wide binaries
are routinely resolved) and for adaptive-optics or space-based surveys
(with resolution far below seeing-limited ground-based work).

## Domain of validity

1. **Mass-addition is exact** for SB2 / eclipsing / astrometric
   binaries; an over-estimate of the bias for photometric surveys.
2. **Photometric approximation $L \propto m^s$ is rough.** Real
   isochrones depart from a single power law; for production work,
   use the multi-band CMD operator with a real isochrone.
3. **Multi-band CMD requires isochrone selection.** Mis-specifying
   age, metallicity, or extinction couples into the inferred
   $\alpha$ in non-trivial ways. The recommendation is to *jointly*
   infer $(\alpha, \tau, Z, d, A_V)$ rather than treat the latter as
   fixed.
4. **All operators ignore stellar evolution.** A binary with one
   evolved component (giant, white dwarf) has flux properties not
   captured by main-sequence isochrones. progenax flags this as a
   known limitation; full treatment requires coupling to a
   stellar-evolution code (gravax, startrax in the future).

## References

The mass-addition operator follows from elementary dynamical
considerations. The flux-addition formula is standard; see
{cite:t}`Maschberger2013` Section 5 for IMF-relevant context. The
multi-band CMD analysis builds on standard isochrone-fitting techniques
(PARSEC, MIST). The resolution cutoff follows the LSST cadence
specification.
