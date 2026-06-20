---
title: Küpper, Maschberger, Kroupa & Baumgardt (2011)
description: Annotated reference for A. H. W. Küpper et al. — the McLuster cluster-IC code and its method calibration; the partial-mass-segregation shuffle, fractal substructure, the COM-then-resolve binary convention, and the mass-segregation/substructure diagnostics.
---

# Küpper, Maschberger, Kroupa & Baumgardt (2011)

```{admonition} Mass segregation and fractal substructure in young massive clusters – I. The McLuster code and method calibration
:class: note

**Authors.** A. H. W. Küpper, T. Maschberger, P. Kroupa, H. Baumgardt
(Argelander-Institut für Astronomie, Bonn; ESO; IPAG Grenoble; Univ. Queensland).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **417**, 2300–2317 (2011).
Accepted 2011 July 8.

**DOI.** [10.1111/j.1365-2966.2011.19412.x](https://doi.org/10.1111/j.1365-2966.2011.19412.x) ·
**ADS.** [2011MNRAS.417.2300K](https://ui.adsabs.harvard.edu/abs/2011MNRAS.417.2300K)

**Verified.** The McLuster IC conventions used by progenax (the §A6 partial-segregation
shuffle and Eq. A1, the §A7 fractal construction, the §A8 COM-then-resolve binary convention)
were checked against the published PDF (pp. 2314–2316).
```

## Abstract (paraphrased)

Introduces and publicly releases **McLuster**, a code for generating star-cluster initial
conditions, and uses models of the young massive cluster R136 to **calibrate** the methods used
to detect and quantify mass segregation and substructure in (non-seeing-limited) $N$-body data.
The paper compares mass-segregation diagnostics — the mass-function-slope-vs-radius method,
Allison's $\Lambda$ minimum-spanning-tree parameter, colour gradients, and local stellar
surface density — and substructure diagnostics — the projected radial density profile, the
azimuthal density profile, and the Cartwright & Whitworth $\mathcal{Q}$ parameter. It finds the
mass-function-slope method and the azimuthal-profile method most practical for large data sets,
and discusses how **binaries** bias each measure (notably that $\mathcal{Q}$ is binary-sensitive
and depends on the radial density gradient). McLuster is progenax's primary cross-validation
reference for cluster ICs.

## What progenax actually uses

### Partial mass segregation — the energy-ordered shuffle (§A6, Eq. A1, verified)

McLuster applies **any degree** of primordial mass segregation to any density profile using the
method of Baumgardt et al. (2008a): it builds $N$ energy-ordered orbits and assigns masses to
them. Perfect ordering (most-massive star on the lowest-energy orbit) gives full segregation;
no ordering gives none. Intermediate (**partial**) segregation comes from a controlled shuffle
(Eq. A1):

```{math}
:label: mcluster-shuffle
j = (N - i)\left(1 - X^{\,1 - S}\right),
```

where $X \in [0,1)$ is random, $i$ indexes the mass-ranked stars (most massive first), and
$S \in [0,1]$ is the **mass-segregation parameter**: $S = 1$ reproduces the perfectly ordered
(fully segregated) array, $S = 0$ gives a random (unsegregated) assignment. Crucially, this
shuffle does **not** change the chosen density profile as the segregation degree increases.

### Fractal substructure (§A7, verified)

Two constructions: (i) a homogeneous box-fractal (after Goodwin & Whitworth 2004) where each
parent cell spawns children with probability $2^{\,(D-3)}$, set by the **fractal dimension**
$D$ (option `-D`); $D = 3.0$ gives no fractality, smaller $D$ gives more substructure; and
(ii) fractal substructure **folded into** any smooth density profile (Plummer, King, EFF) by
rescaling radii. The paper is explicit that this is an *ad hoc* (non-physical-origin)
substructure generator giving a smooth spherical→substructured transition.

### Binaries: COM particle then resolve (§A8, verified)

After masses are drawn from the IMF, chosen binaries are **replaced by a single centre-of-mass
(CoM) particle** for the rest of the IC build; **only at the very end**, after the density
profile is established and the velocities virialised, are the CoM particles replaced by their
two component stars with sampled orbital elements. This **COM-then-resolve / scale-separation
convention** is the one progenax follows: a binary is virialised as one CoM body and its
internal binding energy is kept as a separate reservoir. Semi-major axes can be drawn flat
(`adis=0`), from the Kroupa (1995a) period law (`adis=1`, default), or from the Duquennoy &
Mayor (1991) Galactic-field period law (`adis=2`); eccentricities are thermal $f(e) = 2e$
(Duquennoy & Mayor 1991), with pre-main-sequence eigenevolution (Kroupa 1995b) circularising
short-period orbits.

### Fill factor and diagnostics

McLuster's R136-calibrated comparison sample motivates the tidally-filling fill-factor range
$r_h/r_t \approx 0.05$–$0.3$ progenax quotes, and the $\mathcal{Q}$-parameter range used in the
fractal-substructure validation. The $\mathcal{Q}$ diagnostic itself is from
[](cartwright-2004.md).

## Use in progenax

- [](../../10-theory/tidal-and-substructure/mass-segregation.md) — the partial-segregation
  $S$-shuffle {eq}`mcluster-shuffle`.
- [](../../10-theory/tidal-and-substructure/fractal.md) — fractal-substructure conventions and
  the $\mathcal{Q}$ range.
- [](../../10-theory/tidal-and-substructure/tidal.md) — the tidally-filling fill-factor range.
- [](../../10-theory/binaries/index.md) — the §A8 COM-then-resolve binary virialisation
  convention.
- [](../../10-theory/populations/index.md) — layered / multi-component cluster set-up.
- [](../../50-validation/fractal-substructure.md) — McLuster $\mathcal{Q}$-range anchor for the
  substructure validation.

## Notes

The McLuster code paper and the dynamical-diagnostics calibration paper are the **same**
publication (MNRAS 417, 2300). progenax's spatial IC machinery is an independent JAX-native
re-implementation, **cross-validated against** McLuster's conventions and calibrated ranges
rather than ported from it. The fractal/substructure ICs themselves now live in the
experimental `gravoturb_fdf` subsystem; the Cartwright & Whitworth $\mathcal{Q}$ *diagnostic*
survives in `progenax.diagnostics`.
