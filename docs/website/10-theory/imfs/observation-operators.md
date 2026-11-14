---
title: Observation operators
description: Mass-addition and photometric observation operators, their distinct Jacobians, and the limits of current binary-aware inference evidence.
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
m_2$. It describes a total dynamical mass when the observations
constrain that quantity. Photometric and CMD-based surveys measure
combined fluxes instead. Their induced mass functions and inferred
slope biases depend on the stellar model, noise, selection and
likelihood; a per-system mass-shift bound does not order all those biases.

This chapter describes the implemented mass-addition model and two
planned photometric operators. It derives a simple flux-addition
approximation; it does not report executed photometric or CMD recovery.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning how the *measurement* (dynamical mass vs flux vs CMD) sets the size of binary contamination bias; no prior survey-forward-model literature assumed.
**Prerequisites:** the [binary-aware IMF chapter](binary.md) (the bias and likelihood these operators feed).
**You'll get:** the three observation operators (mass-addition, photometric flux-addition, multi-band CMD), the $m_{\mathrm{phot}} \approx m_1(1+q^s)^{1/s}$ formula, the assumptions needed to interpret bias, and which operators are live vs planned.
:::

## Three operators

```{list-table}
:header-rows: 1

* - Operator
  - What the telescope measures
  - Bias scaling
* - **Mass addition** $m_{\mathrm{sys}} = m_1 + m_2$
  - True dynamical mass (eclipsing, SB2, astrometric)
  - Direct total-mass observable; inference bias is model dependent
* - **Photometric** $m_{\mathrm{phot}} = \mathcal{M}^{-1}[F_b(m_1) + F_b(m_2)]$
  - Single-band flux inverted through a mass-luminosity relation
  - For a luminosity power law with exponent $s>1$, a smaller per-system mass shift
* - **Multi-band CMD** $\mathbf{m}_{\mathrm{obs}} = \{-2.5\log[F_b(m_1) + F_b(m_2)] + \mathrm{ZP}\}_b$
  - Multi-band magnitudes per LSST/HST/JWST
  - A vector observable; requires its own stellar and survey forward model
```

`BinaryIMF` supplies component masses; its system-mass helper adds
them. The recovery scripts use that observable. The photometric and
CMD operators below are proposed survey-forward-model layers, not
current public `BinaryIMF` implementations.

## Mass addition: dynamical mass

When a dynamical analysis constrains both component masses, its total
system mass is:

```{math}
:label: msys-dyn
m_{\mathrm{sys}} \;=\; m_1 + m_2 \;=\; m_1\,(1 + q).
```

The mass-addition change of variables in {eq}`ba-likelihood` is exact
for the stated primary and companion model. Observational uncertainty,
incompleteness and multiplicity misspecification are separate terms.
An SB2 radial-velocity orbit alone measures mass combinations involving
inclination; eclipses, astrometry or other information may be needed
to infer absolute masses. Success for this operator does not qualify
a photometric likelihood.

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

For $s>1$, $m_1\le m_{\mathrm{phot}}\le m_1+m_2$, with a reduced
secondary contribution at small $q$. This is a bound on each inferred
mass in this simplified stellar model, not on a fitted IMF slope.

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
  - $1.0245\,m_1$
  - $0.092$ mag brighter
* - 0.3
  - $1.3\,m_1$
  - $1.0042\,m_1$
  - $0.0159$ mag brighter
* - 0.1
  - $1.1\,m_1$
  - $1.00009\,m_1$
  - $0.000343$ mag brighter
```

In this example, equal-mass binaries shift the inferred mass by about
22%, while small-$q$ companions have a smaller effect. Detectability
depends on photometric precision, contrast and the stellar model;
these numbers alone do not determine a survey's slope bias.

A future photometric likelihood layer can swap the mass-addition
operator for {eq}`m-phot-approx` inside the marginalisation
likelihood. For $M=m_1 A(q)$ with
$A(q)=(1+q^s)^{1/s}$, the candidate primary becomes $M/A(q)$ and the
Jacobian becomes $1/A(q)$ rather than $1/(1+q)$. For a general
isochrone mapping the derivative of that mapping supplies the
Jacobian. The current `BinaryIMF` does not export a
`with_photometric_operator()` helper.

```{note}
A homogeneous mass operator need not change a scale-free IMF slope:
with mass-independent binary statistics and away from support
boundaries, $M=m_1 A(q)$ preserves a primary power-law index. The
[primary/system derivation](binary.md) gives this counterexample for
mass addition. Turnover, finite bounds and survey selection can
produce bias; its sign and size require the specified calculation.
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

## Evidence needed for operator comparisons

The retained recovery evidence uses mass addition. Photometric and
CMD recovery are unperformed here, so there is no measured bias table
or sample-size threshold for those operators. A comparison must hold
the primary population and binary statistics fixed, define the stellar
and survey models, and assess the resulting likelihood under each
observable. A correct forward likelihood can remove the selected
misspecification; it does not guarantee zero finite-sample bias,
identifiability or numerical accuracy.

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
the {cite:t}`MoeDiStefano2017` period distribution. The current `BinaryIMF`
does not provide a `with_resolution_cutoff()` convenience method. This
refinement is most important for nearby clusters (where wide binaries
are routinely resolved) and for adaptive-optics or space-based surveys
(with resolution far below seeing-limited ground-based work).

## Domain of validity

1. **Mass addition defines a total dynamical mass.** Whether a survey
   constrains it depends on the data and dynamical inference; it is
   not a general upper bound on photometric slope bias.
2. **Photometric approximation $L \propto m^s$ is rough.** Real
   isochrones depart from a single power law; for production work,
   use the multi-band CMD operator with a real isochrone.
3. **Multi-band CMD requires isochrone selection.** Mis-specifying
   age, metallicity, or extinction couples into the inferred
   $\alpha$ in non-trivial ways. The recommendation is to *jointly*
   infer $(\alpha, \tau, Z, d, A_V)$ rather than treat the latter as
   fixed.
4. **The simple main-sequence approximation omits evolution.** An
   evolved component can violate its mass-luminosity relation. A
   future CMD model must specify an age-dependent stellar model;
   flux addition itself does not require a main-sequence assumption.

## Implementation, validation & references

- **In code:** the live **mass-addition** operator is built into
  `src/progenax/imf/binary/imf.py` (`BinaryIMF`, $m_{\mathrm{sys}} =
  m_1 + m_2$); the photometric and multi-band CMD operators on this page
  are **planned** survey-forward-model layers (no `CMDOperator` /
  `with_photometric_operator()` is exported yet). See the
  [IMF API](../../30-api/imf.md) and the
  [add-binary-population recipe](../../40-howto/add-binary-population.md).
- **Validated in:** [binary-aware recovery](../../50-validation/binary-imf.md)
  exercises the mass-addition operator; the reduced-bias photometric / CMD operators have no
  dedicated validation page yet, since they are not implemented.
- **Primary sources:** the mass-addition operator follows from
  elementary dynamics; the flux-addition context is {cite:t}`Maschberger2013`
  §5, and the multi-band CMD analysis builds on standard isochrone
  fitting (PARSEC, MIST). Full notes in the
  [bibliography](../../99-bibliography/index.md).
