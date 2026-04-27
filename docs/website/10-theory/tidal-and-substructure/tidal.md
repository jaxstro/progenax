---
title: Tidal physics
description: The Jacobi (tidal) radius — its derivation from the restricted three-body problem, the corrections for non-circular orbits and extended halos, and progenax's `apply_tidal_truncation` utility.
---

# Tidal physics

A star cluster orbiting in a host galaxy's potential is *tidally
limited*: stars beyond a critical radius — the **Jacobi radius**
$r_J$, also called the tidal radius — are stripped by the differential
pull between the cluster's gravity and the galaxy's. progenax models
this with a single utility, `apply_tidal_truncation`, that removes
all stars with $|\mathbf{r}| > r_J$ from the IC. The Jacobi radius
itself is computed from the cluster mass and the host galaxy's
enclosed mass profile via `jacobi_radius`.

This chapter derives the Jacobi-radius formula from the restricted
three-body problem, lists the corrections for eccentric orbits and
extended halos, and documents the truncation utility.

## The Jacobi radius

For a cluster of mass $M_{\mathrm{cl}}$ on a *circular* orbit at
galactocentric radius $R$ in a host galaxy with enclosed mass
$M_{\mathrm{gal}}(<R)$, the Jacobi radius is

```{math}
:label: jacobi-circular
r_J \;=\; R\,\biggl[\frac{M_{\mathrm{cl}}}{(2 + \mathrm{d}\ln M_{\mathrm{gal}}/\mathrm{d}\ln R)\,M_{\mathrm{gal}}(<R)}\biggr]^{1/3}.
```

The bracketed denominator carries the host's mass-profile dependence
through $\mathrm{d}\ln M_{\mathrm{gal}}/\mathrm{d}\ln R$:

- **Point-mass host** ($M_{\mathrm{gal}}(<R) = $ const): the
  derivative is 0, recovering $r_J = R\,(M_{\mathrm{cl}}/2 M_{\mathrm{gal}})^{1/3}$
  — the classical Roche lobe.
- **Singular isothermal halo** ($M_{\mathrm{gal}}(<R) \propto R$): the
  derivative is 1, giving $r_J = R\,(M_{\mathrm{cl}}/3 M_{\mathrm{gal}})^{1/3}$.
- **Flat rotation curve**: same as singular isothermal (since
  $v_c^2 = G M / R$ const requires $M \propto R$).

The standard "Galactic-cluster" approximation uses the singular
isothermal halo form:

```{math}
:label: jacobi-iso
r_J \;\approx\; \biggl(\frac{M_{\mathrm{cl}}}{3 M_{\mathrm{gal}}(<R)}\biggr)^{\!1/3}\,R
```

This is the formula `progenax.tidal.jacobi_radius_isothermal` returns
by default; the more general {eq}`jacobi-circular` is available via
`jacobi_radius` with an explicit `M_gal_enclosed_func` callable.

## Derivation sketch

The derivation comes from the **restricted three-body problem** in
the rotating frame co-rotating with the cluster's circular orbit. In
that frame, the effective potential is

```{math}
:label: roche-potential
\Phi_{\mathrm{eff}}(\mathbf{r}) \;=\; -\frac{G\,M_{\mathrm{cl}}}{|\mathbf{r}|}
  - \frac{G\,M_{\mathrm{gal}}(<R)}{|\mathbf{R} - \mathbf{r}|}
  - \tfrac{1}{2}\,\Omega_R^2\,|\mathbf{R} - \mathbf{r}|^2
```

with $\Omega_R^2 = G M_{\mathrm{gal}}(<R)/R^3$ the angular velocity of
the cluster's circular orbit. The Lagrange points $L_1, L_2$ along the
line connecting the cluster centre to the galactic centre define the
distance $r_J$ at which a test particle's effective potential is
saddle-shaped — the boundary beyond which it can be lost to the host.
Solving $\partial \Phi_{\mathrm{eff}}/\partial r = 0$ to lowest order
in $r_J/R$ yields {eq}`jacobi-circular`.

Higher-order corrections in $r_J/R$ are negligible for $r_J/R \lesssim
0.1$ (typical for Galactic clusters at $R \gtrsim 1$ kpc) and become
significant only for very nearby clusters or very massive ones
(globular clusters near the Galactic centre, ultra-compact dwarfs
near their hosts).

## Corrections for eccentric orbits

For a cluster on an eccentric orbit, the Jacobi radius varies with
orbital phase: it is smallest at perigalacticon (where tidal stripping
is most effective) and largest at apogalacticon. progenax's default
treatment uses the *perigalacticon* Jacobi radius — the most
restrictive — to set the IC truncation. This is conservative: the
cluster will not lose any additional mass during evolution from the
initial condition.

For a more sophisticated treatment, `jacobi_radius` accepts a
phase-averaged $\langle r_J \rangle$ via:

```python
r_peri = jacobi_radius(M_cl, M_gal_at(R_peri), R_peri)
r_apo  = jacobi_radius(M_cl, M_gal_at(R_apo),  R_apo)
r_J_avg = 0.5 * (r_peri + r_apo)   # Phase-averaged
```

Whether to use peri, apo, or a phase average depends on the science
target. For long-term evolution studies (Gyr timescales) the
perigalacticon value sets the long-term mass-loss rate. For short-term
("snapshot") studies the apogalacticon value is more representative
of the cluster's instantaneous extent.

## Tidal truncation

`apply_tidal_truncation(positions, velocities, masses, r_t)` is the
utility that filters out stars beyond a specified truncation radius:

```python
from progenax.tidal import apply_tidal_truncation

positions, velocities, masses, keep_mask = apply_tidal_truncation(
    positions, velocities, masses, r_t=r_J,
)
# Returns kept arrays plus an original-length mask
```

Implementation detail: the function computes an original-length boolean
mask and uses it to return the kept rows. The returned particle arrays
therefore have length `M <= N`.

```{warning}
**Shape-collapsed output.** `apply_tidal_truncation` uses boolean
indexing, so it is convenient for preparing IC catalogs but not a
fixed-shape operation for code that must stay inside a single JIT trace.
Keep `keep_mask` if you need to map the retained particles back to the
original catalog.
```

In the current implementation, `apply_tidal_truncation` returns
shape-collapsed kept arrays plus the original-length boolean mask:
`(positions_kept, velocities_kept, masses_kept, keep_mask)`. That
boolean indexing is useful for ordinary IC preparation but is not the
fixed-shape truncation style required inside a fully JIT-compiled
pipeline.

## Fill-factor: $r_h / r_J$

A useful dimensionless quantity is the **fill-factor**

```{math}
:label: fill-factor
\mathcal{F} \;\equiv\; \frac{r_h}{r_J}
```

which measures how "full" the cluster is relative to its tidal limit.
Galactic globular clusters have observed $\mathcal{F}$ in the range
0.05–0.3 {cite:p}`King1966,Kuepper2011`. progenax's
`fill_factor_to_r_h(fill_factor, r_J)` utility computes the
half-mass radius corresponding to a target fill factor — useful when
the science specifies the tidal-truncation regime rather than the
absolute size.

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
* - Tidal-limit-filling
  - $\mathcal{F} \to 0.5$
  - Theoretical maximum; no real cluster reaches this
```

## Composition with King profile

The {cite:t}`King1966` profile already has a *built-in* tidal radius
$r_t = \xi_t \cdot r_c$ ([](../spatial-profiles/king.md)). Passing a
King profile through `apply_tidal_truncation` is therefore redundant
*if* $r_t = r_J$, which is the equilibrium configuration King
implicitly assumes. For non-equilibrium configurations — King-like
clusters that are not in tidal equilibrium with their current
galactocentric position — `apply_tidal_truncation` adds the
secondary truncation explicitly.

For Plummer and EFF (which extend to infinity), tidal truncation
must be applied externally; it is not built in.

## Domain of validity

1. **Point-mass cluster.** {eq}`jacobi-circular` treats the cluster
   as a point mass for the host's tidal field. Corrections for the
   cluster's extended mass distribution are $\mathcal{O}((r_J/R)^2)$
   and become important only for very large clusters.
2. **Static host potential.** Galactic disc/halo potential is assumed
   constant during the cluster's orbit. For times longer than the
   galaxy's secular evolution timescale ($\sim 1$–$10$ Gyr), this
   assumption breaks down.
3. **Single component.** {eq}`jacobi-circular` uses a single $M_{\mathrm{gal}}(<R)$.
   For galaxies with multi-component potentials (disc + halo + bulge),
   compute the *total* enclosed mass; progenax accepts a callable
   `M_gal_enclosed_func` for this.
4. **No cluster-cluster interactions.** Two clusters in orbit around
   each other (e.g. mutually-bound binary clusters) require a
   different treatment beyond the scope of this utility.

## References

The Jacobi-radius derivation is standard textbook material; Murray &
Dermott *Solar System Dynamics* §3 gives a clean treatment. The
isothermal-halo approximation is standard in the cluster literature;
{cite:t}`King1966` and {cite:t}`Kuepper2011` use it in the contexts
relevant here. Fill-factor data for observed clusters comes from
{cite:t}`Kuepper2011`'s McLuster comparison sample.
