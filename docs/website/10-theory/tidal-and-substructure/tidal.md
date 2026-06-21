---
title: Tidal physics
description: The Jacobi (tidal) radius — its derivation from the restricted three-body problem, the point-mass and isothermal-halo forms, and progenax's differentiable `apply_tidal_truncation` utility.
---

# Tidal physics

A star cluster orbiting in a host galaxy's potential is *tidally
limited*: stars beyond a critical radius — the **Jacobi radius**
$r_J$, also called the tidal radius — are stripped by the differential
pull between the cluster's gravity and the galaxy's. progenax provides
two closed-form estimators of $r_J$ — `jacobi_radius` (point-mass
host) and `jacobi_radius_isothermal` (flat-rotation-curve host) — and
a differentiable utility, `apply_tidal_truncation`, that "removes"
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
galactocentric radius $R$ in a host galaxy with enclosed mass
$M_{\mathrm{gal}}(<R)$, the Jacobi radius is

```{math}
:label: jacobi-circular
r_J \;=\; R\,\biggl[\frac{M_{\mathrm{cl}}}{(3 - \mathrm{d}\ln M_{\mathrm{gal}}/\mathrm{d}\ln R)\,M_{\mathrm{gal}}(<R)}\biggr]^{1/3}.
```

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

For a more sophisticated treatment, evaluate `jacobi_radius` at the two
orbital extremes (passing the enclosed galaxy mass at each radius) and
phase-average:

```python
from progenax.tidal import jacobi_radius

r_peri = jacobi_radius(M_cl, M_gal_peri, R_peri)  # M_gal_peri = M_gal(<R_peri)
r_apo  = jacobi_radius(M_cl, M_gal_apo,  R_apo)   # M_gal_apo  = M_gal(<R_apo)
r_J_avg = 0.5 * (r_peri + r_apo)                  # Phase-averaged
```

Whether to use peri, apo, or a phase average depends on the science
target. For long-term evolution studies (Gyr timescales) the
perigalacticon value sets the long-term mass-loss rate. For short-term
("snapshot") studies the apogalacticon value is more representative
of the cluster's instantaneous extent.

## Tidal truncation

`apply_tidal_truncation(positions, velocities, masses, r_t)`
"removes" stars beyond a truncation radius $r_t$ — but rather than
boolean-indexing them out (which would collapse the array shape), it
sets the mass of every star with $r > r_t$ to **zero**. Those zero-mass
"ghost" particles then contribute nothing to any mass-weighted quantity
(energy, virial ratio, centre of mass) while keeping the array length
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
delta-function derivative of the step with a logistic bump (width
`grad_width * r_t`, default 5%), so $r_t$ — and any upstream parameter
feeding it (e.g. via `jacobi_radius`) — stays differentiable. Because
the output shape is static, the function is fully `jit` / `vmap` /
`grad` safe; use `keep_mask` to filter *number*-based downstream
quantities that would otherwise still see the zero-mass ghosts.

```{warning}
**The truncated set is super-virial, not stationary.** The survivors
keep the velocities they were drawn with for the *untruncated*
potential, but they now sit in the shallower potential of the
mass-reduced system. The truncated set is therefore super-virial —
some stars near $r_t$ are formally unbound, and the configuration is
not a stationary equilibrium. If you need a stationary IC, re-virialise
the survivors (`virial_scale`) or use an $r_t$-consistent equilibrium
model (King / LIMEPY) directly.
```

## Fill-factor: $r_h / r_J$

A useful dimensionless quantity is the **fill-factor**

```{math}
:label: fill-factor
\mathcal{F} \;\equiv\; \frac{r_h}{r_J}
```

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
   compute the *total* enclosed mass yourself and pass it to
   `jacobi_radius` (which takes a scalar `M_galaxy`, not a callable).
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
