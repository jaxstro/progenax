---
title: King velocity distribution functions
description: The King (1966) lowered-Maxwellian DF, its tidal-truncated velocity ellipsoid, and progenax's sampling implementation.
---

# King velocity distribution functions

The King velocity DF is the lowered-Maxwellian that defines the King
spatial profile in the first place. Unlike Plummer — where the DF
follows from the density profile via Eddington inversion — King's DF
*is* the input, and the density profile follows from integrating the
DF over velocity space ([](../spatial-profiles/king.md)). Sampling
King velocities is therefore conceptually simpler than sampling
Plummer velocities: at each particle's position, draw from the
truncated Maxwellian directly.

## The lowered-Maxwellian DF

For a particle of specific energy $E$ at position $\mathbf{r}$, the
King DF is

```{math}
:label: king-df-velocity
f(E) \;=\;
\begin{cases}
  \rho_1\,(2\pi\sigma_0^2)^{-3/2}\,\Bigl[\,e^{(\Phi_t - E)/\sigma_0^2} - 1\,\Bigr], & E < \Phi_t \\
  0, & E \ge \Phi_t
\end{cases}
```

where $\sigma_0$ is the central one-dimensional velocity dispersion,
$\Phi_t \equiv \Phi(r_t)$ is the potential at the tidal radius, and
$\rho_1$ is a normalisation constant. The "$-1$" subtraction is the
*lowering*: it ensures the DF vanishes at $E = \Phi_t$ rather than
extending exponentially to $E \to \infty$ as a pure Maxwellian
would.

In terms of the dimensionless central potential $W_0 \equiv (\Phi_t -
\Phi(0))/\sigma_0^2$ ([](../spatial-profiles/king.md), {eq}`king-W`),
the DF becomes

```{math}
:label: king-df-W
f(W, v) \;\propto\; e^{W - v^2 / (2\sigma_0^2)} - 1,
\qquad v < v_{\mathrm{esc}}(W) = \sigma_0\sqrt{2W}
```

at any radius where the local dimensionless potential is $W$. The
escape speed at the tidal radius is zero ($W_t = 0$, $v_{\mathrm{esc}} =
0$), so no particles populate that boundary — the cluster is sharply
truncated.

## Marginal speed distribution

Integrating the DF over velocity directions gives the speed
distribution at fixed position:

```{math}
:label: king-fv
f(v \mid W) \;\propto\; v^2\,\Bigl[\,e^{W - v^2 / (2\sigma_0^2)} - 1\,\Bigr],
\qquad 0 \le v \le \sigma_0\sqrt{2W}
```

The $v^2$ prefactor is the spherical-shell volume element in velocity
space. The bracketed factor is positive throughout $0 \le v \le
\sigma_0\sqrt{2W}$ and zero at the upper bound — which means
inverse-CDF sampling on $f(v \mid W)$ produces velocities in $[0,
v_{\mathrm{esc}}]$ with no rejection.

## Sampling

progenax samples King velocities via inverse-CDF on {eq}`king-fv`:

```python
# At each particle's radius r, look up W(r) from the King ODE solution
W_per_particle = jnp.interp(r_per_particle, xi_grid * r_c, W_solution)

# Build a 2D inverse-CDF table: u ↔ v at each W
u = jax.random.uniform(key, (N,))
v = lookup_2d_inverse_cdf(u, W_per_particle, table)

# Isotropic angles
v_vec = v[:, None] * isotropic_unit_vector(key2, N)
```

The 2D lookup table is precomputed once at the King-solution stage:
for a grid of $W$ values, evaluate the cumulative integral of
{eq}`king-fv` at a grid of speeds, then store as `(W, u_table, v_table)`.
At sampling time, each particle's $W$ value (set by its radius)
selects a row of the table; bilinear interpolation in `(W, u)` gives
the speed.

The whole pipeline is JIT-compatible and differentiable in $r_h$ via
the chain $r_h \to r_c \to \sigma_0 \to v$.

## Velocity dispersion profile

The King DF's velocity dispersion at radius $r$ is

```{math}
:label: king-sigma
\sigma^2(W) \;=\; \sigma_0^2\,\frac{\int_0^{\sqrt{2W}} v^4\,(e^{W - v^2/2} - 1)\,\mathrm{d}v}{\int_0^{\sqrt{2W}} v^2\,(e^{W - v^2/2} - 1)\,\mathrm{d}v}
```

(in units where $\sigma_0 = 1$). The integrals do not have closed
form in general, but progenax precomputes $\sigma(W)$ on the same
$W$-grid used for the speed-CDF table inside the sampler. The current
public `KingVelocityDF` API exposes `sample_velocities`; it does not
export a separate velocity-dispersion method. The dispersion is
monotonically decreasing in $W$ — i.e. *increasing* with radius — which
is the King DF's defining property and the reason King clusters look
"puffed up" at the tidal radius compared to Plummer.

## Pairing with the King profile

The King DF is in equilibrium with the King density profile *by
construction*: both are derived from the same lowered-isothermal
Maxwellian. Pair them by using the same King concentration and tidal
radius:

```python
from progenax.profiles import KingProfile
from progenax.kinematics import KingVelocityDF
from jaxstro.units import STELLAR

profile = KingProfile.from_W0_rc(W0=7.0, r_c=1.0)
df = KingVelocityDF(W0=7.0, r_c=1.0, r_t=profile.r_t)

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key)
velocities = df.sample_velocities(positions, masses, key, G=STELLAR.G)
```

Using the same `W0`, `r_c`, and `r_t` keeps the profile and DF
consistent. Pairing a King profile at $W_0 = 7$ with a King DF at
$W_0 = 5$ would produce a mismatched non-equilibrium IC.

## Differentiability caveat

Like the King density profile, the King velocity DF is differentiable
in $r_h$ (via $r_c$ and $\sigma_0$) but *not* in $W_0$ — the latter
parameterises the underlying ODE solution and changing it would
require differentiating through the solver. progenax's standard usage
treats $W_0$ as a fixed structural choice (held at e.g. $W_0 = 7$ for
the average Galactic globular cluster) and infers $r_h$ via HMC.

For applications that genuinely need $\partial / \partial W_0$
gradients — e.g. inferring the King concentration of an unresolved
cluster from photometric data — the recommendation is to use the
{cite:t}`Gieles2015` LIMEPY family in its $g = 1$ (King-equivalent)
limit, where the parameter $g$ enters the solution analytically.
LIMEPY support is not currently implemented in progenax.

## Domain of validity

1. **Single-mass equilibrium.** Multi-mass equilibrium is handled by
   LIMEPY, not by the standard King DF here.
2. **Spherical and isotropic** by default. Anisotropic and rotating
   variants live in [](rotation-anisotropy.md).
3. **Tidal cutoff is sharp.** Real clusters have a smooth transition
   from the bound population to the tidal tail; King's mathematical
   cutoff at $E = \Phi_t$ is an idealisation. For studies where the
   tidal-tail kinematics matter, post-evolution analysis is the
   right approach.

## References

The original lowered-isothermal model is {cite:t}`King1966`. The
LIMEPY generalisation is {cite:t}`Gieles2015`. progenax's `diffrax`
integration follows the standard {cite:t}`Aarseth1974` numerical
prescription. The validation suite is at
[](../../50-validation/king-profile.md).
