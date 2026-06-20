---
title: Plummer profile
description: The Plummer (1911) spatial density profile — its closed-form derivations for mass, potential, velocity dispersion, and distribution function, plus the half-mass radius bug that earlier versions of progenax tripped over.
---

# The Plummer profile

The Plummer model {cite:p}`Plummer1911` is the canonical spatial density
profile for star clusters: closed-form everything, smooth, untruncated,
and historically the first density profile derived for which the
distribution function $f(E)$ admits an analytic solution. It is the
default progenax spatial profile for all production ICs and is the
right starting point for understanding every other profile in the
package.

This chapter derives the Plummer profile from first principles, lists
the closed-form expressions for the four observables that every
N-body code consumes (mass, potential, velocity dispersion, isotropic
DF), tabulates the half-mass-radius / scale-radius mapping that
recurs in every progenax test, and documents the
2025-12 transcription bug whose fix is now anchored by 14 regression
tests.

## The density profile

The Plummer profile is

```{math}
:label: plummer-rho
\rho(r) \;=\; \frac{3\,M}{4\pi\,a^3}\,\biggl[1 + \biggl(\frac{r}{a}\biggr)^{\!2}\,\biggr]^{-5/2}
```

where $M$ is the total mass and $a$ is the **Plummer scale radius**.
The profile is centrally smooth ($\rho(0) = 3M/(4\pi a^3) < \infty$),
asymptotically $\rho \propto r^{-5}$ at $r \gg a$, and integrates to
finite total mass even though $r$ extends to infinity. The smoothness
at $r = 0$ is what makes it analytically tractable — there is no
central cusp to integrate around.

The $r^{-5}$ outer fall-off is steeper than observed for real star
clusters (which typically show $\rho \propto r^{-3}$ to $r^{-4}$ at
large radii), so Plummer is in this sense a *toy* profile. Its
analytic tractability is what keeps it in production use as a baseline
against which more realistic profiles (King, EFF) are calibrated.

## Closed-form mass and potential

Integrating {eq}`plummer-rho` against the spherical volume element
gives the cumulative mass:

```{math}
:label: plummer-mcum
M(<r) \;=\; M\,\frac{r^3}{(r^2 + a^2)^{3/2}}
```

The total mass is recovered as $r \to \infty$: $M(<\infty) = M$. Solving
$M(<r_h) = M/2$ for the half-mass radius gives the relation that
recurs throughout progenax,

```{math}
:label: plummer-rh-a
\boxed{\;\;a \;=\; r_h\,\sqrt{2^{2/3} - 1}\;\approx\;0.7664\,r_h\;\;}
```

so that progenax's user-facing `r_h` parameter unambiguously specifies
the radius enclosing half the cluster mass, regardless of the internal
scale-radius convention.

```{warning}
**Misinverting the relation inflates the scale radius by 1.7×.** A previous
version of progenax stored $a = r_h / \sqrt{2^{2/3} - 1}$ instead of $a = r_h
\sqrt{2^{2/3} - 1}$ — the inverse. Since $\sqrt{2^{2/3}-1}\approx 0.766$, the
*scale radius* $a$ comes out $1/0.766^2 \approx 1.70\times$ too large
(equivalently, the cluster's effective $r_h$ is then $1/0.766 \approx 1.305\times$
too large). Because every Plummer length scales with $a$, the realised cluster
is $\sim1.7\times$ over-sized. The bug propagated into the validation suite (where
"50% within $r_h$" failed by ~25%) and was caught only when
`tests/validation/test_plummer_physics.py::test_half_mass_radius`
was added in 2025-12. The lesson: anchor every closed-form constant
on a quantitative test against the *defining condition* (here,
$M(<r_h) = M/2$), not against the constant itself.
```

The gravitational potential is

```{math}
:label: plummer-phi
\Phi(r) \;=\; -\frac{G\,M}{\sqrt{r^2 + a^2}}
```

with central value $\Phi(0) = -GM/a$ and asymptotic $\Phi(r) \to 0$ as
$r \to \infty$. The escape speed at radius $r$ is

```{math}
:label: plummer-vesc
v_{\mathrm{esc}}(r) \;=\; \sqrt{-2\Phi(r)}
\;=\; \sqrt{\frac{2GM}{\sqrt{r^2 + a^2}}}\,.
```

## Inverse-CDF sampling

For Monte Carlo IC generation, progenax uses inverse-CDF sampling on
{eq}`plummer-mcum`. Setting $u = M(<r) / M$ for $u \sim \mathcal{U}(0, 1)$
and inverting:

```{math}
:label: plummer-inverse-cdf
r(u) \;=\; a\,\sqrt{\frac{u^{2/3}}{1 - u^{2/3}}}
```

The expression is exact, has no singularities for $u \in [0, 1)$, and
is differentiable analytically in $u$ and $a$. Each call to
`PlummerProfile.sample_positions(masses, key)` draws $N_\star$
uniform variates, evaluates {eq}`plummer-inverse-cdf` to get radii,
draws isotropic angles, and returns 3D positions — all in a single
JIT-compiled `vmap` over particles, fully differentiable in $r_h$
through the chain $r_h \to a \to r$.

## Velocity dispersion

For the Plummer profile in dynamical equilibrium, the radial-velocity
dispersion has the closed form

```{math}
:label: plummer-sigma-r
\sigma_r^2(r) \;=\; \frac{G\,M}{6\,\sqrt{r^2 + a^2}}
```

Tangential and total velocity dispersions are equal to $\sigma_r$ for
the *isotropic* Plummer DF (the standard equilibrium choice; see
[](../velocity-dfs/plummer-dfs.md)). The central velocity dispersion is

```{math}
\sigma_0^2 \;=\; \frac{G\,M}{6\,a},
\qquad
\sigma_0^2 / |\Phi(0)| \;=\; \tfrac{1}{6}.
```

This last ratio fixes the cluster's virial state: integrating
{eq}`plummer-sigma-r` against the density profile gives $T = -V/2$, i.e.
$Q_{\mathrm{vir}} = T/|V| = 1/2$ — the equilibrium value
([](../ic-philosophy.md)).

## The Plummer distribution function

The isotropic distribution function for the Plummer *density* (the 1911
space-density law, {eq}`plummer-rh-a` context) in energy space is the
Eddington-inversion result {cite:p}`Merritt1985` (his Eq. 42; equivalently
Eddington 1916 / Binney & Tremaine §4) — **not** derived in
{cite:t}`Plummer1911`, which predates the inversion method:

```{math}
:label: plummer-df
f(E) \;=\; \frac{24\sqrt{2}}{7\pi^3}\,\frac{a^2}{G^5\,M^4}\,(-E)^{7/2},
\qquad E < 0
```

with $E = \tfrac{1}{2}v^2 + \Phi(r)$ the specific energy. The
exponent $7/2$ derives from the $(\rho)^{-5/2}$ density and the
Eddington inversion machinery — see [](../velocity-dfs/plummer-dfs.md)
for the full derivation. The DF is finite, monotonic in $E$, and
positive for all $E < 0$, so progenax can sample velocities at any
position via inverse-CDF on $f(E)$ without rejection.

## The five-step closed-form derivation

For reference, the five quantities derived above are the five steps
that recur in every Plummer-related calculation. Collecting them in
one place:

```{list-table}
:header-rows: 1

* - Quantity
  - Closed form
* - Density $\rho(r)$
  - $\dfrac{3M}{4\pi a^3}\,(1 + r^2/a^2)^{-5/2}$
* - Cumulative mass $M(<r)$
  - $M\,r^3/(r^2 + a^2)^{3/2}$
* - Potential $\Phi(r)$
  - $-GM/\sqrt{r^2+a^2}$
* - Radial $\sigma_r^2(r)$
  - $GM/[6\sqrt{r^2+a^2}]$
* - Isotropic DF $f(E)$
  - $\frac{24\sqrt{2}}{7\pi^3}\,\frac{a^2}{G^5 M^4}\,(-E)^{7/2}$
```

The half-mass relation $a = r_h\sqrt{2^{2/3}-1}$ {eq}`plummer-rh-a`
ties this all together. Every progenax Plummer-related test
(`tests/validation/test_plummer_physics.py`) anchors on at least
two of these quantities.

## Implementation in progenax

```python
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF
from jaxstro.units import STELLAR

profile = PlummerProfile(r_h=1.0)              # Half-mass radius in pc
velocity_df = PlummerVelocityDF(r_h=1.0)        # Same r_h for equilibrium

masses = jnp.ones(1000)                         # 1000 M_sun
key = jax.random.PRNGKey(42)

positions = profile.sample_positions(masses, key)
velocities = velocity_df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

Both `PlummerProfile` and `PlummerVelocityDF` are Equinox modules
(immutable PyTrees), `@jax.jit`-compatible, and differentiable in
$r_h$. Sampling positions and velocities under `jax.grad` flows
gradients through $r_h \to a \to r \to \mathbf{x}$ (positions) and
$r_h \to a \to \sigma_r \to \mathbf{v}$ (velocities) — the
expensive-but-tractable chain we accept (~$10\times$ slowdown vs
non-differentiable sampling) in exchange for HMC-friendly inference.

```{admonition} Common pitfall
:class: warning
Use the **same** `r_h` value for the profile and its velocity DF when
you want an equilibrium IC. Mismatched scale radii produce
non-equilibrium starting states, which can be useful for studying
violent relaxation but are not "the Plummer cluster" in the equilibrium
sense. The progenax convenience builders enforce this by accepting one
`r_h` and instantiating both classes from it.
```

See [](../../30-api/profiles.md) for the full `PlummerProfile`
signature, [](../../30-api/kinematics.md) for `PlummerVelocityDF`, and
[](../../50-validation/plummer-equilibrium.md) for the regression
suite that locks every closed-form expression above.

## Domain of validity and limitations

The Plummer profile is mathematically smooth and infinitely extended.
Two consequences for production use:

1. **No tidal truncation built in.** Real Galactic clusters have a
   tidal radius beyond which stars are stripped by the host galaxy's
   tidal field. Plummer ICs need to be truncated post-hoc — see
   [](../tidal-and-substructure/tidal.md) for the Jacobi-radius
   computation and `apply_tidal_truncation` utility.

2. **Outer slope is too steep.** At $r \gg a$, $\rho \propto r^{-5}$,
   steeper than the $r^{-3}$ to $r^{-4}$ observed in real clusters.
   For ICs whose outer-profile shape matters (e.g. fitting LMC young
   massive clusters), use [](eff.md) which exposes the outer slope as
   a free parameter.

For most production star-cluster work — Galactic globular clusters,
Milky Way analogues, dynamical-evolution studies — Plummer is the right
default. It is also the most-tested profile in progenax (see the
[test dashboard](../../50-validation/test-dashboard.md) for the live
counts) and the one against which the two alternatives are calibrated.

## References

The Plummer model is {cite:t}`Plummer1911`. The closed-form derivations
are standard textbook material; {cite:t}`Aarseth1974` is a clean
reference for N-body initialisation specifically. The 2025-12
half-mass-radius bug fix is recorded in
[](../../90-development-log/2025-12-07-imf-stack-fix.md).
