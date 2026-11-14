---
title: Tidal physics
description: The Jacobi (tidal) radius — its derivation from the restricted three-body problem, the point-mass and isothermal-halo forms, and progenax's differentiable `apply_tidal_truncation` utility.
---

# Tidal physics

For a cluster on a circular orbit, the rotating effective potential has
escape saddles near the **Jacobi radius** $r_J$. The leading tidal estimate
balances cluster gravity against the local host and centrifugal fields.
An individual star's energy and orbit matter for escape; radius alone
does not give its binding status or escape time. progenax provides
two closed-form estimators of $r_J$ — `jacobi_radius` (point-mass
host) and `jacobi_radius_isothermal` (flat-rotation-curve host) — and
a static-shape utility, `apply_tidal_truncation`, that spatially selects
stars with $|\mathbf{r}| > r_t$ from the IC by zeroing their mass.

This chapter derives the Jacobi-radius formula from the restricted
three-body problem, gives the two host-potential forms progenax
implements, and documents the truncation utility.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the tidal (Jacobi) radius and how progenax truncates an IC to it; no prior galactic-dynamics literature assumed.
**Prerequisites:** [tidal physics & substructure](index.md) (the modifier-layer framing); the [King profile](../spatial-profiles/king.md) for the built-in $r_t$ context.
**You'll get:** the Jacobi-radius derivation, the point-mass (factor 3) vs. flat-rotation-curve (factor 2) forms, the differentiable `apply_tidal_truncation` utility, and the fill-factor $r_h/r_J$.
:::

## The Jacobi radius

For a cluster of mass $M_{\mathrm{cl}}$ on a *circular* orbit at
galactocentric radius $R$ in a spherical host with enclosed mass
$M_{\mathrm{gal}}(<R)$, the leading local tidal estimate is

```{math}
:label: jacobi-circular
r_J \;=\; R\,\biggl[\frac{M_{\mathrm{cl}}}{(3 - \mathrm{d}\ln M_{\mathrm{gal}}/\mathrm{d}\ln R)\,M_{\mathrm{gal}}(<R)}\biggr]^{1/3}.
```

[↗ model card](#card-jacobi-circular)

The numerical factor in the denominator carries the host's mass-profile
dependence through $\mathrm{d}\ln M_{\mathrm{gal}}/\mathrm{d}\ln R$:

- **Point-mass host** ($M_{\mathrm{gal}}(<R) = $ const): the
  derivative is 0, giving the factor **3**,
  $r_J = R\,(M_{\mathrm{cl}}/3 M_{\mathrm{gal}})^{1/3}$ — the classical
  Hill/Roche radius {cite:p}`King1962`.
- **Singular isothermal halo / flat rotation curve**
  ($M_{\mathrm{gal}}(<R) \propto R$, so $v_c^2 = G M_{\mathrm{gal}}/R$
  is constant): the derivative is 1, giving the factor **2**,
  $r_J = R\,(M_{\mathrm{cl}}/2 M_{\mathrm{gal}})^{1/3}$. The factor
  of 2 (rather than 3) is the signature of the logarithmic potential.

progenax implements these as two separate functions, each with a fixed
factor — there is no callable-host-profile API:

- `jacobi_radius(M_cluster, M_galaxy, R_galactic)` — the **point-mass**
  factor-3 form $r_J = R\,(M_{\mathrm{cl}}/3 M_{\mathrm{gal}})^{1/3}$
  ({cite:t}`King1962`; Binney & Tremaine 2008 §8.3).
- `jacobi_radius_isothermal(M_cluster, V_circ, R_galactic, G)` — the
  **flat-rotation-curve** factor-2 form, parameterised by the host's
  circular velocity $V_{\mathrm{circ}}$ rather than its enclosed mass.
  Algebraically $r_J = (G\,M_{\mathrm{cl}}/2\Omega^2)^{1/3}$ with
  $\Omega = V_{\mathrm{circ}}/R$, equivalent to the N-body-calibrated
  relation of {cite:t}`Baumgardt2003` Eq. 1.

```{warning}
**Unit trap in `jacobi_radius_isothermal`.** `V_circ` must be in the
*same* length/time units as `G` — pc/Myr for `STELLAR`, **not** km/s.
The ecosystem's display convention quotes velocities in km/s, but
`STELLAR.G` is in $\mathrm{pc}^3\,\Msun^{-1}\,\mathrm{Myr}^{-2}$, so
pass `V_circ` in pc/Myr ($1\,\mathrm{km\,s^{-1}} = 1.0227\,\mathrm{pc\,Myr^{-1}}$).
Mixing the two biases $r_J$ by $\sim 1.5\%$ per the conversion factor.
```

## Derivation sketch

For a point-mass host, consider the line connecting the host and cluster
in the orbital plane, with the cluster at coordinate $R$ and a star at
cluster-centered coordinate $r$ toward the host. The rotating effective
potential along that line is

```{math}
:label: roche-potential
\Phi_{\mathrm{eff}}(r) = -\frac{G M_{\mathrm{cl}}}{|r|}
-\frac{G M_{\mathrm{gal}}}{|R-r|}
-\frac12\Omega^2(R-r)^2,
\qquad \Omega^2=\frac{G M_{\mathrm{gal}}}{R^3}.
```

Expanding the host and centrifugal forces for $|r|/R\ll1$ gives
$G M_{\mathrm{cl}}/r_J^2=3\Omega^2r_J$. The finite-mass-ratio
restricted-three-body L1 distance differs from this leading Hill result.
For a spherical extended host, the corresponding local coefficient is
$\Omega^2-\Phi_{\rm gal}''(R)=(3-\mathrm{d}\ln M/\mathrm{d}\ln R)\Omega^2$,
which gives {eq}`jacobi-circular` when this coefficient is positive.
The point-mass potential above must not be reused as an extended-host
potential merely by replacing its mass with $M(<R)$.

## Eccentric orbits

The utility functions do not accept orbital phase, eccentricity or an orbit,
and do not select a perigalacticon radius. A circular-orbit estimate evaluated
at an eccentric orbit's pericenter is an additional approximation, not a
guarantee against later mass loss. A time-dependent tide and the cluster's
response require a dynamical model beyond these IC utilities.

For comparison, a caller may evaluate the same approximate expression at
two supplied radii:

```python
from progenax.tidal import jacobi_radius

r_peri = jacobi_radius(M_cl, M_gal_peri, R_peri)
r_apo = jacobi_radius(M_cl, M_gal_apo, R_apo)
```

These are conditional point-mass-host estimates. Their arithmetic mean is
not an orbital time average, and neither value by itself determines the
cluster's evolving edge.

## Tidal truncation

`apply_tidal_truncation(positions, velocities, masses, r_t)`
"removes" stars beyond a truncation radius $r_t$ — but rather than
boolean-indexing them out (which would collapse the array shape), it
sets the mass of every star with $r > r_t$ to **zero**. Those zero-mass
"ghost" particles then contribute nothing to any mass-weighted quantity
(energy, virial ratio, center of mass) while keeping the array length
fixed at $N$:

```python
from progenax.tidal import apply_tidal_truncation

positions, velocities, masses_trunc, keep_mask = apply_tidal_truncation(
    positions, velocities, masses, r_t=r_J,
)
# All four outputs have length N (shape-preserving):
#   positions, velocities : unchanged (truncated stars left in place)
#   masses_trunc          : masses with r > r_t entries set to 0
#   keep_mask             : bool (N,), True where r <= r_t
```

**Implementation detail.** The forward pass is an *exact* hard
Heaviside cut. The mass-zeroing is wrapped in a `@jax.custom_jvp`
straight-through surrogate: the backward pass replaces the
membership tangent with a logistic bump (width `grad_width * r_t`, default
5%). A fixed-catalogue hard cut is piecewise constant between crossings
and discontinuous at them; this straight-through tangent is an estimator,
not its ordinary derivative. The static output shape supports JAX transforms; use `keep_mask` to filter *number*-based downstream
quantities that would otherwise still see the zero-mass ghosts.

```{warning}
**A spatial cut does not construct a stationary DF.** The survivors retain
their original velocities while removing outer mass changes the potential.
The resulting configuration generally needs its density–velocity consistency
checked. A global virial rescale sets $T/|V|$, but does not restore a
stationary distribution. A finite lowered model supplies a consistent
continuum cutoff; its numerical domain still needs checking.
```

## Fill-factor: $r_h / r_J$

A useful dimensionless quantity is the **fill-factor**

```{math}
:label: fill-factor
\mathcal{F} \;\equiv\; \frac{r_h}{r_J}
```

[↗ model card](#card-fill-factor)

which measures how "full" the cluster is relative to its tidal limit.
Galactic globular clusters have observed $\mathcal{F}$ in the range
0.05–0.3 {cite:p}`Kuepper2011`. (The fill-factor / "tidally filling"
terminology is later community usage — it is a *definitional* ratio,
not a result tabulated by any single source; {cite:t}`King1966`'s
single-mass models supply the tidal radius $r_t$ but no fill-factor
data.) progenax's `fill_factor_to_r_h(fill_factor, r_J)` utility
computes the half-mass radius corresponding to a target fill factor —
useful when the science specifies the tidal-truncation regime rather
than the absolute size.

```{list-table} Typical fill factors for Galactic clusters.
:header-rows: 1

* - Cluster type
  - $\mathcal{F} = r_h / r_J$
  - Comment
* - Galactic globular (typical)
  - 0.05–0.15
  - Tidally limited; old enough to have lost outer halo
* - Galactic open
  - 0.10–0.30
  - Younger; less tidal stripping
* - Fitted or chosen filling factor
  - Model-dependent
  - No universal maximum of 0.5 follows from the definition
```

## Composition with King profile

The {cite:t}`King1966` profile already has a *built-in* tidal radius
$r_t = \xi_t \cdot r_c$ ([](../spatial-profiles/king.md)). Passing a
King profile through `apply_tidal_truncation` at its own model edge is
redundant. Identifying that edge with an external $r_J$ is a separate
host-orbit and tidal-filling assumption; the spherical King constructor
does not solve the host tidal field. For non-equilibrium configurations — King-like
clusters that are not in tidal equilibrium with their current
galactocentric position — `apply_tidal_truncation` adds the
secondary truncation explicitly.

Plummer is untruncated; imposing a finite spatial cut changes its model.
The released EFF profile already carries a finite imposed `r_t`; that
parameter is not automatically the Jacobi radius of an external orbit.

## Domain of validity

1. **Compact cluster and local expansion.** The leading expression assumes
   the cluster is small compared with the orbital radius and its mass is
   effectively enclosed inside the tidal balance point. Extended mass and
   higher host-field terms require their own error assessment.
2. **Static host potential.** Galactic disc/halo potential is assumed
   constant during the cluster's orbit. For times longer than the
   galaxy's secular evolution timescale ($\sim 1$–$10$ Gyr), this
   assumption breaks down.
3. **Host shape and slope.** The general displayed formula assumes a
   spherical host and needs its radial mass-profile slope. Passing total
   enclosed mass to `jacobi_radius` still selects the point-mass factor 3;
   it does not include the slope or a nonspherical disk tidal tensor.
4. **No cluster-cluster interactions.** Two clusters in orbit around
   each other (e.g. mutually-bound binary clusters) require a
   different treatment beyond the scope of this utility.

## Implementation, validation & references

- **In code:** `src/progenax/tidal.py` (`jacobi_radius`,
  `jacobi_radius_isothermal`, `apply_tidal_truncation`,
  `fill_factor_to_r_h`). See the [tidal API](../../30-api/tidal.md).
- **Validated in:** [tidal truncation](../../50-validation/tidal-truncation.md)
  — the regression suite for $r_J$ and the shape-preserving truncation.
- **Primary sources:** the Jacobi-radius derivation is standard
  textbook material (Murray & Dermott *Solar System Dynamics* §3;
  Binney & Tremaine 2008 §8.3); the point-mass factor-3 form traces to
  {cite:t}`King1962` and the N-body-calibrated isothermal form to
  {cite:t}`Baumgardt2003`. The isothermal-halo approximation and
  fill-factor comparison sample follow {cite:t}`King1966` and
  {cite:t}`Kuepper2011`. Full notes in the
  [bibliography](../../99-bibliography/per-paper/king-1962.md).
