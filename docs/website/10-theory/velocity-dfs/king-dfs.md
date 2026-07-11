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

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers learning the King lowered-Maxwellian DF and how a tidally truncated cluster gets its velocities; no prior stellar-dynamics literature assumed.
**Prerequisites:** the [King profile](../spatial-profiles/king.md) (the $W_0$, $r_c$, $r_t$ parameters and the King ODE) and the [Plummer DF](plummer-dfs.md) for the contrast (DF-from-density vs density-from-DF).
**You'll get:** the lowered-Maxwellian $f(E)$ and its dimensionless $W$ form, the marginal speed distribution, how progenax samples it per-particle by inverse-CDF, the radius-dependent dispersion, and what is (and isn't) differentiable.
:::

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

[↗ model card](#card-king-df-velocity)

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

[↗ model card](#card-king-fv)

The $v^2$ prefactor is the spherical-shell volume element in velocity
space. The bracketed factor is positive throughout $0 \le v \le
\sigma_0\sqrt{2W}$ and zero at the upper bound — which means
inverse-CDF sampling on $f(v \mid W)$ produces velocities in $[0,
v_{\mathrm{esc}}]$ with no rejection.


```{figure} ../figures/king_lowered_maxwellian.webp
:label: fig-king-lowered-maxwellian
:width: 88%

What "lowering" means. One normalized Maxwellian (grey, dashed) and the King
speed distribution at three local well depths: at $W = 6$ (a cluster core)
the truncation barely bites, at $W = 1$ (near the tidal edge) most of the
Maxwellian is cut away. The lowering matters most where the well is
shallow — the origin of King clusters' cold outskirts. Regenerate:
`python -m laboratory.icviz --only king-lowered-maxwellian`.
```

## Sampling

progenax samples King velocities via a per-particle inverse-CDF on
{eq}`king-fv`:

```python
# At each particle's radius r, look up W(r) from the King ODE solution
W_per_particle = jnp.interp(radii / r_c, xi_grid, psi_grid, left=W0, right=0.0)

# Per particle: build a 1-D speed CDF on [0, sqrt(2W)] and invert it (vmap'd)
def sample_unit_speed(key, W):
    u_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * W), N_SPEED_GRID)   # 256 points
    g = u_grid**2 * (jnp.exp(W - u_grid**2 / 2.0) - 1.0)         # eq:king-fv
    cdf = jnp.cumsum(0.5 * (g[1:] + g[:-1])) * du                # trapezoid
    return jnp.interp(jax.random.uniform(key), cdf / cdf[-1], u_grid)

u = jax.vmap(sample_unit_speed)(keys, W_per_particle)            # speed / sigma_0
v_vec = (sigma_0 * u)[:, None] * isotropic_unit_vector(key2, N)
```

Each particle builds its own fixed-size (256-point) speed grid scaled
to its local escape speed $\sigma_0\sqrt{2W}$ and inverts the trapezoidal
CDF by interpolation — there is no precomputed cross-$W$ table. The grid
size is fixed and the operations are `jnp.interp`/`vmap` only (no
`while_loop`), so the whole pipeline is JIT-compatible and differentiable
in $r_h$ via the chain $r_h \to r_c \to \sigma_0 \to v$.

## Velocity dispersion profile

The King DF's velocity dispersion at radius $r$ is

```{math}
:label: king-sigma
\sigma^2(W) \;=\; \sigma_0^2\,\frac{\int_0^{\sqrt{2W}} v^4\,(e^{W - v^2/2} - 1)\,\mathrm{d}v}{\int_0^{\sqrt{2W}} v^2\,(e^{W - v^2/2} - 1)\,\mathrm{d}v}
```

[↗ model card](#card-king-sigma)

(in units where $\sigma_0 = 1$). The integrals do not have closed
form in general, but progenax precomputes $\sigma(W)$ on the same
$W$-grid used for the speed-CDF table inside the sampler. The current
public `KingVelocityDF` API exposes `sample_velocities`; it does not
export a separate velocity-dispersion method. The dispersion is
monotonically **increasing** in $W$: $\sigma^2 \to 0$ as $W \to 0$ and
$\sigma^2 \to 3\sigma_0^2$ as $W \to \infty$ (the untruncated-Maxwellian
limit). Since $W$ decreases outward — from $W_0$ at the centre to $0$ at
the tidal radius — **$\sigma$ falls with radius and vanishes at the tidal
boundary**, where the escape speed $\to 0$. King clusters therefore have
*cold outskirts*, in contrast to a Plummer sphere whose dispersion also
declines outward but stays finite at all radii.

## Check yourself

:::{dropdown} 1. Which $W$ is most Maxwellian?
Before studying {numref}`fig-king-lowered-maxwellian`: does the King DF
resemble a Maxwellian most at the cluster centre or the edge? (Centre: large
$W$ pushes $v_{\rm esc} = \sigma_0\sqrt{2W}$ far into the Maxwellian tail,
so almost nothing is cut.)
:::

:::{dropdown} 2. Quantify the cut
Compute the fraction of a pure Maxwellian beyond $v_{\rm esc} = \sqrt{2W}$:
$\mathrm{erfc}(\sqrt{W}) + \sqrt{4W/\pi}\,e^{-W}$. At $W = 1$ that is
$0.57$ — the lowering removes the *majority* of the distribution — while at
$W = 6$ it is $0.007$. Check both against the figure by eye, then verify with
a quick quadrature of $v^2 e^{-v^2/2}$.
:::

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
df = KingVelocityDF(W0=7.0, r_c=1.0)   # r_t is derived from W0 internally

masses = jnp.ones(1000)
positions = profile.sample_positions(masses, key_pos)
velocities = df.sample_velocities(positions, masses, key_vel, G=STELLAR.G)
```

`KingVelocityDF` takes only `(W0, r_c)` — it re-solves the King ODE
internally and derives the tidal radius from $W_0$, so there is **no
`r_t` argument**. Using the same `W0` and `r_c` in the profile and DF
keeps them consistent. Pairing a King profile at $W_0 = 7$ with a King DF
at $W_0 = 5$ would produce a mismatched non-equilibrium IC.

## Differentiability

The King velocity DF is differentiable in $r_c$ (via $\sigma_0$) and in the
total mass $M_{\rm tot}$ ($\sigma_0\propto\sqrt{GM}$). It is **also**
differentiable in $W_0$: `diffrax` propagates $\partial\psi/\partial W_0$
through the ODE solve, so the dispersion profile and other shape observables
carry correct $W_0$ gradients (gradient-validated in
[](../../50-validation/king-profile.md)). The lone non-differentiable quantity
is the *scalar* tidal radius $r_t$, whose $W_0$-derivative is zeroed by the
`argmax` zero-crossing in `_find_tidal_radius`; inference should therefore use
the profile *shape* rather than the scalar $r_t$.

Consequently progenax supports joint gradient-based / HMC inference of the King
structural parameters $(W_0, r_c, M_{\rm tot})$ directly — e.g. fitting a
cluster's number-density and velocity-dispersion profiles. The lowered-model
family formalized by {cite:t}`Gieles2015` (where the truncation parameter $g$
enters analytically) remains the natural extension for multi-mass /
anisotropic generalisations; progenax plans to implement this family natively
as its own differentiable generalization (see
[](../spatial-profiles/lowered-model-family.md)), but it is not yet available.

## Domain of validity

1. **Single-mass equilibrium.** Multi-mass equilibrium is described by
   the lowered-model family formalized by {cite:t}`Gieles2015`, not by the
   standard King DF here.
2. **Spherical and isotropic** by default. Anisotropic and rotating
   variants live in [](rotation-anisotropy.md).
3. **Tidal cutoff is sharp.** Real clusters have a smooth transition
   from the bound population to the tidal tail; King's mathematical
   cutoff at $E = \Phi_t$ is an idealisation. For studies where the
   tidal-tail kinematics matter, post-evolution analysis is the
   right approach.

## Implementation, validation & references

- **In code:** `src/progenax/kinematics/king_df.py` (the per-particle
  lowered-Maxwellian speed sampler; the King ODE solve it shares with
  the profile is in `src/progenax/profiles/king.py`) — see the
  [`KingVelocityDF` API](../../30-api/kinematics.md).
- **Validated in:** [King profile](../../50-validation/king-profile.md)
  — the regression suite that locks the true-DF equilibrium
  ($Q_{\mathrm{vir}} \approx 0.5$ unscaled) and the $W_0$ gradients.
- **Primary sources:** the lowered-isothermal model is {cite:t}`King1966`;
  the multi-mass LIMEPY generalisation is {cite:t}`Gieles2015`, and the
  `diffrax` integration follows the standard {cite:t}`Aarseth1974`
  numerical prescription — full notes in the
  [bibliography](../../99-bibliography/per-paper/king-1966.md).
