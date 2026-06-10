---
title: "The Eddington engine (Engine B)"
description: "Density-defined multi-component equilibria: prescribed Plummer/EFF/King densities in one shared potential via a single quadrature pass, per-component Eddington inversion (optionally Osipkov–Merritt), the f ≥ 0 realizability gate, derived domains, and hybrid sampling with predicted truncation-edge offsets."
---

# The Eddington engine (Engine B)

Engine B is the density-defined route into
[`MultiComponentCluster`](index.md): you prescribe each component's
*density shape* (Plummer, EFF, or King density) and its mass-fraction
amplitude, and the engine constructs the one shared self-consistent
potential they jointly generate, recovers each component's
distribution function by **Eddington inversion in that shared
potential**, and samples a true joint equilibrium — or refuses, with
the physics named, when the decomposition does not exist as one.

```python
from progenax import MultiComponentCluster, PlummerProfile, EFFProfile

model = MultiComponentCluster.from_density_profiles(
    profiles=[PlummerProfile(r_h=2.0), EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],
    mass_fractions=jnp.array([0.6, 0.4]),   # M_j / M_total, must sum to 1
    m_j=jnp.array([0.5, 1.0]),              # stellar-mass labels (N_frac_j ∝ f_j/m_j)
    r_a_j=None,                              # optional per-component OM radii
)
```

## One quadrature pass, no ODE

The structural contrast with
[Engine A](../spatial-profiles/lowered-model-family.md) is this:
Engine A's density is a *functional of the potential* (the DF defines
$\rho(\psi)$), so finding the model requires solving a coupled Poisson
ODE. Engine B's total density is **prescribed**,

```{math}
:label: engine-b-rhotot
\rho_{\rm tot}(r) \;=\; \sum_j \frac{M_j}{M_{\rm tot}}\,
\hat\rho_j(r) \quad \text{(each } \hat\rho_j \text{ normalized to unit truncated mass)},
```

so the shared potential follows from **one cumulative-trapezoid
pass** — no ODE, no iteration:

```{math}
:label: engine-b-poisson
M(<r) = 4\pi\!\int_0^r \rho_{\rm tot}\, s^2\, \mathrm{d}s, \qquad
\Phi(r) = -G\left[\frac{M(<r)}{r} + 4\pi\!\int_r^{r_t} \rho_{\rm tot}\, s\, \mathrm{d}s\right],
```

with the relative potential $\Psi = \Phi(r_t) - \Phi$ (so
$\Psi(r_t) = 0$, increasing inward) and
$\mathrm{d}\Psi/\mathrm{d}r = -GM(<r)/r^2$ analytic from the enclosed
mass. The implementation (`profiles/density_poisson.py`) works in
dimensionless units ($G = 1$, total truncated mass 1) on a fixed
$n_r = 6000$ grid, and stores per-component enclosed-mass CDFs
$M_j(<r)$ for the position sampler.

## Per-component Eddington inversion in the shared Ψ

Each component's ergodic DF is the standard Eddington inversion
— but performed in the **shared** relative potential, not the
component's own isolated one:

```{math}
:label: eddington-formula
f_j(E) \;=\; \frac{1}{\sqrt{8}\,\pi^2}\left[
\int_0^{E} \frac{\mathrm{d}^2\rho_j}{\mathrm{d}\Psi^2}\,
\frac{\mathrm{d}\Psi}{\sqrt{E-\Psi}}
\;+\; \frac{1}{\sqrt{E}}\left.\frac{\mathrm{d}\rho_j}{\mathrm{d}\Psi}\right|_{\Psi=0}
\right].
```

Two numerical points are load-bearing:

- **The $u = \sqrt{E - \Psi}$ substitution** turns the integrable
  $1/\sqrt{E-\Psi}$ singularity into a smooth integrand
  ($\mathrm{d}\Psi = -2u\,\mathrm{d}u$), which is what makes the
  quadrature both accurate and gradient-safe.
- **The truncation boundary term**
  $(\mathrm{d}\rho_j/\mathrm{d}\Psi)|_{\Psi=0}/\sqrt{E}$ is *not*
  optional for truncated models: dropping it corrupts $f$ at low
  binding energies.

The inverter (`eddington_invert`, in `kinematics/eddington.py`) was
extracted **bit-identically** from the validated EFF Eddington table,
so the existing EFF physics suite pins its behavior. Its strongest
truth test bypasses all of progenax's own potential numerics: fed the
analytic Plummer $(\rho, \Psi)$ pair it reproduces the closed-form
$f(E) \propto E^{7/2}$ law to $1.06\times 10^{-4}$ — *with the
untruncated zero point*; the truncated case is covered by an exact
closed form including the boundary term, matched to
$8.86\times 10^{-6}$.

### Osipkov–Merritt anisotropy

Per-component radial anisotropy uses the augmented-density device of
{cite:t}`Merritt1985`: replacing $\rho_j$ by

```{math}
:label: om-augmented
\rho_{Q,j}(r) \;=\; \left(1 + \frac{r^2}{r_{a,j}^2}\right)\rho_j(r)
```

in {eq}`eddington-formula` yields $f_j(Q)$ with
$Q = E - J^2/2r_{a,j}^2$, i.e. the Osipkov–Merritt anisotropy profile
$\beta_j(r) = r^2/(r^2 + r_{a,j}^2)$. Each component carries its own
$r_{a,j}$ (`inf` = isotropic). The sampled anisotropy realizes the
target profile: max $|\beta_{\rm sampled} - r^2/(r^2+r_a^2)| = 0.028$
(4 seeds × 20k stars, gate 0.05).

## Derived domains (the model decides, never the code)

A component's radial extent is part of the prescribed model, so the
cluster truncation radius $r_t$ is **derived**, with the rule that set
it stored as provenance:

1. **Component extents:** Plummer is infinite; EFF and King end at
   their own $r_t$.
2. **Cluster edge = max finite extent.** If any component is finite,
   $r_t$ is the largest finite extent (provenance names the winning
   profile and component index).
3. **All-infinite mixes** (e.g. pure Plummer): $r_t$ is the radius
   enclosing $f_{\rm enc} = 0.995$ (configurable) of the summed
   analytic mass, found by a fixed 80-step bisection.
4. **Explicit `r_t=` override wins** — *except* that an override which
   would re-truncate a King component below its natural edge raises a
   `ValueError`: a King model's lowered-Maxwellian edge is prescribed
   physics, never silently cut.

Per-component truncated-mass fractions $M_j(<r_t)/M_j(\infty)$ are
stored as diagnostics, so a "too small $r_t$" cannot hide.

## The realizability gate: $f_j \ge 0$ or refuse

Eddington inversion is the *unique* candidate ergodic (or OM) DF for a
density in a potential. If that candidate is negative anywhere, **the
prescribed component does not exist as an equilibrium in this shared
potential** — typically a too-shallow component in a concentrated
companion's potential. Engine B treats this as physics:

- Every build computes and stores the margin
  $f_{\min,j} = \min f_j / \max|f_j|$ per component.
- **Genuine** negativity ($f_{\min,j} < -10^{-3}$, separating physics
  from grid-level quadrature ringing) on a concrete build raises a
  `ValueError` naming the component and the remedy ("density too
  shallow to be supported in this shared potential — steepen it, raise
  its mass fraction, or raise `r_a_j`").
- Traced builds (under `jax.jit`/`jax.grad`) cannot raise, so they
  always store the `f_min_j` diagnostic instead — the same two-tier
  pattern `EFFVelocityDF` uses.
- The DF is **never clamped silently**: a clamped DF integrates back
  to a *different* density than the one prescribed.

This is not a corner case. The originally drafted halo+core example
($a_{\rm EFF} = 0.4$ inside a Plummer halo) is genuinely unrealizable
($f_{\min} = -0.20$, resolution-independent), with the gate flip
measured between $a = 0.65$ and $0.68$ — see the
[worked example](two-component.md) for the full story.

## What the DF can and cannot represent

The Eddington pair $(\rho_j, f_j)$ represents
$\rho_{Q,j}(\Psi) - \rho_{Q,j}(0)$: any **constant edge offset is
invisible to an ergodic DF**. A hard-truncated prescribed density has
$\rho_j(r_t) > 0$, and no $f(E)$ can carry that offset; with OM
anisotropy the unrepresentable offset is amplified by the augmentation
factor $1 + r_t^2/r_a^2$. The consequence is a small, *predictable*
deficit between the DF-reconstructed density
$\rho_{{\rm DF},j} = 4\pi\int w^2 f_j\,\mathrm{d}w / (1+r^2/r_a^2)$
and the prescribed $\rho_j$ near the edge. Interior fidelity is gated:
$\max|\rho_{\rm DF}/\rho_{\rm presc} - 1| = 1.06\times 10^{-3}$ (OM
build) and $2.4\times 10^{-4}$ (isotropic) inside the component
half-mass radius, against a $5\times 10^{-3}$ gate.

## Hybrid sampling and the predict-the-offset $Q_j$

`sample_cluster` for an Engine B model is a **hybrid**: positions come
from the prescribed $\rho_j$ (per-component inverse-CDF on
$M_j(<r)$), speeds from the component's $f_j$ table at the star's
$\Psi(r_i)$, directions from the OM stretched split at the star's
component $r_{a,j}$, and the velocity scale from the *actual sampled
mass* $\sum_i m_i$ (never an independent input total — energies must
use the realized cluster mass). There is **no external virial
rescale**.

Because of the edge-offset physics above, a hard-truncated component's
sampled $Q_j$ plateaus slightly below 0.5 — and Engine B *predicts*
the plateau with an exact-quadrature hybrid expectation (prescribed-ρ
weights × DF speed moments × prescribed-total Clausius field):
predicted $Q_{\rm halo} = 0.4953$, sampled $0.4947 \pm 0.0014$ over 18
seeds — verified truncation-edge physics, gated against the
prediction. The pure-DF theory oracle (`component_virial_ratios`,
which weights by $\rho_{\rm DF}$ — the density the DF actually
represents) reads $Q_j = 0.5$ to a few $\times 10^{-4}$, as the
steady-state identity demands. Do **not** rescale Engine B output to
"fix" the plateau.

## A numerics lesson: never differentiate interpolated data

The King-density branch initially computed $\mathrm{d}W/\mathrm{d}r$
by `jnp.gradient` of the *interpolated* King $\psi$ grid. Piecewise
linear interpolation makes that derivative a staircase, and the
Eddington $\mathrm{d}^2\rho/\mathrm{d}\Psi^2$ kernel plus the Abel
$1/\sqrt{E-\Psi}$ weight focus the staircase ringing exactly into
$f(E \to \Psi_0)$ — a *single King component* (whose true ergodic DF
is strictly positive) read $f_{\min} = -0.679$, squarely inside the
realizability gate's field of view. The fix integrates King's own
Poisson identity,

```{math}
:label: king-poisson-identity
\frac{\mathrm{d}\psi}{\mathrm{d}\xi} \;=\;
-\frac{9}{\hat\rho_0}\,\xi^{-2}\int_0^{\xi} \hat\rho(\psi(s))\, s^2\, \mathrm{d}s,
```

by cumulative trapezoid of the **closed-form** density — after which
$f_{\min} = +5.1\times 10^{-7}$. The general rule is worth the
emphasis: in any Abel-type inversion, differentiate closed forms or
exact identities, never interpolated data.

## Validation summary

All anchors from `scripts/validate_multicomponent_eddington.py`
(11/11 PASS); the full evidence page is at
[](../../50-validation/engine-b-eddington.md).

```{list-table} Engine B measured anchors (2026-06-10 close-out).
:header-rows: 1

* - Check
  - Measured
  - Gate
* - King A-vs-B radial KS distance (two independent engines, $N=2\times 10^4$)
  - $2\times 10^{-4}$
  - $< 0.02$
* - King A-vs-B max $|\sigma_B/\sigma_A - 1|$ (interior bins)
  - $3\times 10^{-4}$
  - $< 0.02$
* - Plummer $f(E) \propto E^{7/2}$ (untruncated zero point)
  - $1.06\times 10^{-4}$
  - $< 10^{-3}$
* - Plummer $f(E)$ vs exact truncated closed form
  - $8.86\times 10^{-6}$
  - $< 10^{-4}$
* - Halo+core theory $Q_j$ (DF-weighted oracle)
  - $[0.50038,\ 0.50012]$
  - $0.5 \pm 3\times 10^{-3}$
* - Halo+core sampled global $Q$ ($N=3\times 10^4$, unscaled)
  - $0.4976$
  - $0.5 \pm 0.02$
* - Hard-truncated halo $Q_j$ vs hybrid prediction
  - sampled $0.4947 \pm 0.0014$ vs predicted $0.4953$
  - $|\Delta| < 0.012$
* - OM $\beta(r)$ realization (max deviation)
  - $0.028$
  - $< 0.05$
* - DF-density fidelity, interior (OM / isotropic)
  - $1.06\times 10^{-3}$ / $2.4\times 10^{-4}$
  - $< 5\times 10^{-3}$
* - AD-vs-FD gradients ($r_h$, mass-fraction $t$, $r_{a}$)
  - $5.6\times 10^{-9}$ / $7.8\times 10^{-7}$ / $2.0\times 10^{-8}$
  - $< 10^{-3}$
```

The King anchor deserves emphasis: a single King component is the only
configuration both engines describe *identically* (Engine A at $g=1$;
Engine B from the King density), through entirely disjoint numerics
(coupled ODE + lowered-DF sampling vs. quadrature potential +
Eddington inversion). Their agreement at the $10^{-4}$ level is the
cross-engine trust statement for the whole
`MultiComponentCluster` design.

## Differentiability

The build is differentiable in the profile parameters (e.g. halo
$r_h$), the mass fractions (via a reparametrized scalar), and the
$r_{a,j}$ — AD matches finite differences to $\le 7.8\times 10^{-7}$
(table above). Two deliberate exceptions: the **domain choice**
(`derive_r_t`) concretizes — picking which component's extent wins is
a construction-time decision, not differentiated through — and the
King component's *internal* subgraph (its own ODE solution) is
constant with respect to the differentiated parameters.

:::{admonition} Honest limitations (current state, fixes tracked)
:class: warning
- **`total_density()` and `rescale_j` are Engine-A-only accessors.**
  On an Engine B model they return NaN — deliberately: the Engine-A
  fields are NaN tripwires so accidental A-path use poisons results
  visibly rather than silently. Engine-B equivalents are tracked; read
  `engine_b.rho_j_poisson` / the prescribed profiles meanwhile.
- **The `is_aniso` flag reflects the Engine-A sampler switch only**,
  not Engine-B OM state. To check whether an Engine B model is
  anisotropic, read `model.engine_b.r_a_j` (finite entries = OM
  components).
- **Speed-sampler thresholds are absolute**: very large-scale models
  ($r_h \gtrsim 10^4$ pc) hit a known scale-dependence issue in the
  speed sampler. Fix tracked; star-cluster-scale models (the design
  target) are unaffected.
- **No Phase 1.5-style speed tables yet**: Engine B speed draws use
  the per-star differentiable inverse-CDF over the $f_j(E)$ grid —
  deliberately, since that path doubles as the oracle. Precomputed
  speed-CDF tables (the Engine A treatment) will be added only if
  profiling shows Engine B sampling matters at $N \ge 10^5$.
:::

## References

Eddington inversion is standard ergodic-DF machinery; the
augmented-density Osipkov–Merritt construction is
{cite:t}`Merritt1985`. The density components are
[Plummer](../spatial-profiles/plummer.md) {cite:p}`Plummer1911`,
[EFF](../spatial-profiles/eff.md) {cite:p}`ElsonFallFreeman1987`, and
[King](../spatial-profiles/king.md) {cite:p}`King1966`. The DF-defined
counterpart is the {cite:t}`Gieles2015` family —
[Engine A](../spatial-profiles/lowered-model-family.md).
