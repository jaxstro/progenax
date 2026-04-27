---
title: Kepler orbital elements
description: The seven Keplerian orbital elements (a, e, i, Ω, ω, M, t_p), their physical meaning, and the conversion from elements to phase-space coordinates that progenax's KeplerElements class implements.
---

# Kepler orbital elements

A binary's relative orbit is fully specified by **seven orbital elements**:
the semi-major axis $a$, eccentricity $e$, inclination $i$, longitude
of the ascending node $\Omega$, argument of periastron $\omega$, mean
anomaly $M$ at epoch, and the orbital period $P$ (or equivalently, the
time of periastron passage $t_p$). progenax stores these in the
`KeplerElements` PyTree and provides closed-form mappings between
elements and phase-space coordinates $(\mathbf{r}, \mathbf{v})$ for
each component star.

This chapter derives the elements-to-phase-space conversion, lists
the conventions progenax uses, and documents the differentiability
properties that make `KeplerElements` HMC-compatible.

## The seven elements

```{list-table}
:header-rows: 1
:widths: 16 28 56

* - Element
  - Symbol
  - Physical meaning
* - Semi-major axis
  - $a$
  - Half the long axis of the orbit ellipse; sets the orbit size
* - Eccentricity
  - $e$
  - Shape: $e = 0$ circle, $e = 1$ parabola, $0 < e < 1$ ellipse
* - Inclination
  - $i$
  - Tilt of orbit plane relative to a chosen reference plane (typically the cluster's $xy$ plane)
* - Longitude of ascending node
  - $\Omega$
  - Where the orbit crosses the reference plane going "up"
* - Argument of periastron
  - $\omega$
  - Angle from ascending node to periastron, in the orbit plane
* - Mean anomaly at epoch
  - $M$
  - Phase along the orbit at $t = 0$; uniform in time
* - Orbital period
  - $P$
  - Time to complete one orbit, $P = 2\pi\sqrt{a^3 / [G(m_1+m_2)]}$
```

The first six are *positional* (they specify where the orbit is in
space); the seventh specifies the *temporal* phase. Together they
fully determine $(\mathbf{r}_{\mathrm{rel}}(t), \mathbf{v}_{\mathrm{rel}}(t))$
for the relative orbit.

## Kepler's third law

The orbital period and semi-major axis are linked by the total mass:

```{math}
:label: kepler-third
P^2 \;=\; \frac{4\pi^2\,a^3}{G\,(m_1 + m_2)}
\quad\Longleftrightarrow\quad
a \;=\; \biggl[\frac{G\,(m_1+m_2)\,P^2}{4\pi^2}\biggr]^{1/3}.
```

progenax exposes both directions:

- `compute_period(a, m_total, *, G)` — given $a$, returns $P$.
- `period_to_semimajor_axis(P, m_total, *, G)` — given $P$, returns $a$.

The choice of $a$ vs $P$ as the parameterised quantity matters in
practice: most observational period distributions
([](period-distributions.md)) parameterise $P$ directly, so progenax's
default `KeplerElements` constructor accepts $P$ and computes $a$
internally via {eq}`kepler-third`.

## From elements to phase-space coordinates

Converting orbital elements to relative position $\mathbf{r}$ and
velocity $\mathbf{v}$ at time $t$ requires four steps:

**Step 1 — solve Kepler's equation** for the eccentric anomaly $E$:

```{math}
:label: kepler-eq
E - e\,\sin E \;=\; M(t),
\qquad M(t) = M_0 + n\,t,\quad n = 2\pi/P.
```

This is a transcendental equation in $E$. progenax uses a
fixed-iteration Newton solver:

```python
@jax.jit
def solve_kepler(M, e, n_iter=10):
    E = M  # Initial guess (good for small e)
    for _ in range(n_iter):
        E = E - (E - e * jnp.sin(E) - M) / (1.0 - e * jnp.cos(E))
    return E
```

10 iterations gives double-precision convergence for $e \le 0.9$;
20 iterations covers $e \le 0.99$. The fixed iteration count is JIT-
and grad-compatible; the convergence rate is quadratic so the cost
is bounded.

**Step 2 — eccentric anomaly to true anomaly** $\nu$:

```{math}
:label: true-anomaly
\tan(\nu/2) \;=\; \sqrt{\frac{1+e}{1-e}}\,\tan(E/2).
```

**Step 3 — orbit-plane coordinates.** Position and velocity in the
orbit plane (with periastron along the $x$-axis):

```{math}
:label: orbit-plane
\begin{aligned}
r &= a\,(1 - e\,\cos E) \\
\mathbf{r}_{\mathrm{op}} &= r\,(\cos\nu,\,\sin\nu,\,0) \\
\mathbf{v}_{\mathrm{op}} &= \frac{n\,a^2}{r}\,(-\sin E,\,\sqrt{1-e^2}\cos E,\,0)
\end{aligned}
```

**Step 4 — rotate into the reference frame** via three Euler-angle
rotations (about $z$ by $\Omega$, then $x$ by $i$, then $z$ by
$\omega$):

```{math}
:label: rotation
\mathbf{r}_{\mathrm{rel}} \;=\; R_z(\Omega)\,R_x(i)\,R_z(\omega)\,\mathbf{r}_{\mathrm{op}},
\qquad
\mathbf{v}_{\mathrm{rel}} \;=\; R_z(\Omega)\,R_x(i)\,R_z(\omega)\,\mathbf{v}_{\mathrm{op}}.
```

The full pipeline is one JIT trace and fully `vmap`-able over a
population of binaries.

## From relative to component coordinates

Given $\mathbf{r}_{\mathrm{rel}} = \mathbf{r}_2 - \mathbf{r}_1$ and
component masses $m_1, m_2$, the individual particle coordinates are

```{math}
:label: components
\begin{aligned}
\mathbf{r}_1 &= -\frac{m_2}{m_1 + m_2}\,\mathbf{r}_{\mathrm{rel}} \\
\mathbf{r}_2 &= +\frac{m_1}{m_1 + m_2}\,\mathbf{r}_{\mathrm{rel}}
\end{aligned}
```

with the corresponding velocity decomposition for $\mathbf{v}_1,
\mathbf{v}_2$. These are then *added to* the binary's centre-of-mass
position and velocity (which come from the spatial profile and
velocity DF — see [](index.md)) to produce the absolute particle
coordinates.

## Sampling: isotropic angles + population distributions

For a Monte Carlo binary population, progenax samples the seven
elements as follows:

1. **$P$** from the population period distribution
   ([](period-distributions.md)) — typically a log-normal or
   {cite:t}`Sana2012` empirical fit.
2. **$a$** from $P$ and $m_{\mathrm{tot}}$ via {eq}`kepler-third`.
3. **$e$** from the population eccentricity distribution
   ([](eccentricity.md)) — thermal, uniform, or
   {cite:t}`Moe2017` period-dependent.
4. **$\cos i \sim \mathcal{U}(-1, 1)$** — isotropic inclination.
5. **$\Omega \sim \mathcal{U}(0, 2\pi)$** — uniform.
6. **$\omega \sim \mathcal{U}(0, 2\pi)$** — uniform.
7. **$M_0 \sim \mathcal{U}(0, 2\pi)$** — uniform orbital phase.

The mass ratio $q$ comes from [](../imfs/mass-ratio-distributions.md);
the binary fraction from [](../imfs/multiplicity-statistics.md). The
seven orbital elements plus $q$ specify the orbit completely.

## Differentiability properties

`KeplerElements` is differentiable in all element values. Specifically:

- $\partial\mathbf{r}/\partial a$, $\partial\mathbf{r}/\partial e$,
  $\partial\mathbf{r}/\partial(i, \Omega, \omega, M_0)$, and
  $\partial\mathbf{r}/\partial P$ all flow analytically through
  {eq}`orbit-plane`–{eq}`rotation`.
- The Newton iterations on Kepler's equation {eq}`kepler-eq` are
  fixed-count `lax.fori_loop` calls, gradient-compatible.

This matters when *fitting* binary orbits — e.g. inferring $(a, e,
i, \Omega, \omega, M_0)$ from radial-velocity or astrometric
observations of a known binary. The full inference loop is one JAX
gradient call.

## Domain of validity

1. **Bound orbits only** ($e < 1$). Hyperbolic orbits ($e \ge 1$)
   represent unbound encounters, not binaries; they are not handled
   by `KeplerElements`.
2. **Two-body approximation.** Higher-order multiples (triples,
   quadruples) are not represented by a single `KeplerElements`
   instance. progenax models them as nested binaries (an outer
   `KeplerElements` whose "secondary" is itself an inner binary).
3. **Newtonian gravity.** Relativistic effects (periastron precession,
   orbital decay via gravitational waves) are not included. For
   binaries with $a \ll 1$ AU and component masses $\gtrsim 10\,\Msun$,
   relativistic corrections become observable on Gyr timescales.

## References

Kepler-element machinery is standard textbook material; Murray &
Dermott *Solar System Dynamics* §2 gives a clean derivation. Modern
N-body codes (NBODY6, COSMIC) use the same conventions and conversion
sequence. progenax's `KeplerElements` is JAX-native via
{cite:t}`Equinox`'s PyTree pattern; the Newton solver follows the
standard fixed-iteration approach.
