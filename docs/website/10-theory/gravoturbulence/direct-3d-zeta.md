---
title: Direct 3D ζ measurement — `zeta_fdf_direct`
description: Soft-mask formulation that computes ζ directly from a 3D density field — no power-law assumption, fully differentiable through the soft sigmoid threshold.
---

# Direct 3D ζ measurement: `zeta_fdf_direct`

```{seealso}
This chapter introduces the **parameter-free** ζ computation that
operates on an arbitrary 3D density field. For the *power-law*
analytic form see [](pp20.md). For the *cored profile* numerical
integration see [](cored-profiles.md). For the cloud-PDF formulation
that ζ duals to, see [](pdf-and-fdf.md).
```

The PP20 ([](pp20.md)) and cored ([](cored-profiles.md)) ζ
calculations both *parameterise* the cloud's density profile —
power-law or cored. For inferences using simulation snapshots or
detailed observational density maps, that parameterisation throws
away most of the information in the data. progenax's
`zeta_fdf_direct` instead measures ζ *directly* from a 3D density
field via a soft-mask formulation:

```{math}
:label: zeta-fdf-direct
\zeta_{\mathrm{FDF}} \;=\; \frac{\sum_{\mathbf{x}} w(\mathbf{x})\,\rho(\mathbf{x})^{3/2}}
                                 {M_{\mathrm{tail}}\,\sqrt{\bar\rho_{\mathrm{tail}}}}
```

where $w(\mathbf{x})$ is a *soft tail weight* in $[0, 1]$ identifying
the gravitating tail, $M_{\mathrm{tail}} = \sum w(\mathbf{x})\,\rho(\mathbf{x})$
is the weighted tail mass, and $\bar\rho_{\mathrm{tail}} = M_{\mathrm{tail}}
/ \sum w(\mathbf{x})$ is the weighted mean tail density.

This chapter walks through the soft-mask formulation, explains why a
*soft* (sigmoid) threshold is essential for differentiability, and
documents progenax's `zeta_fdf_direct` implementation.

## Why "soft" rather than hard threshold

The natural definition of "tail" gas is "everything above the
transition density $\rho_t$". The *hard* mask is

```{math}
:label: hard-mask
w_{\mathrm{hard}}(\mathbf{x}) \;=\;
\begin{cases}
  1, & \rho(\mathbf{x}) \ge \rho_t \\
  0, & \rho(\mathbf{x}) < \rho_t
\end{cases}
```

This works for forward Monte Carlo. It is *not* differentiable — the
gradient $\partial w_{\mathrm{hard}}/\partial \rho$ is zero almost
everywhere and a delta function at $\rho = \rho_t$. Inferring $\rho_t$
or any density-PDF parameter through such a mask is impossible with
JAX's autodiff.

The **soft mask** replaces the step function with a sigmoid:

```{math}
:label: soft-mask
w(\mathbf{x}) \;=\; \frac{1}{1 + \exp\bigl[-\kappa\,(s(\mathbf{x}) - s_t)\bigr]}
```

where $s = \ln(\rho/\langle\rho\rangle)$ and $\kappa$ is a width
parameter (large $\kappa$ approaches the hard threshold; small $\kappa$
gives a smooth transition). The gradient
$\partial w/\partial s$ is the sigmoid derivative — non-zero,
continuous, and well-behaved. progenax uses $\kappa = 10$ by default,
giving a transition width $\sim 0.1$ in $s$ ($\sim 10\%$ in $\rho$).

```{admonition} The differentiability dividend
:class: note
The soft mask is the technical innovation that lets progenax
*infer* $s_t$ from observed dense-gas SFR data. With a hard mask, the
inference would require finite-difference gradients (slow, noisy,
not HMC-compatible). With the soft mask, $\partial \zeta/\partial s_t$
is analytically defined, and HMC chains converge in $\sim 1500$ steps
per [](../../50-validation/binary-imf.md)-class validation runs.
```

## Implementation

```python
@jax.jit
def zeta_fdf_direct(rho_grid, tail_weights):
    M_tail = jnp.sum(tail_weights * rho_grid)
    V_tail = jnp.sum(tail_weights)
    rho_tail_mean = M_tail / jnp.maximum(V_tail, 1e-10)

    sfr_weighted = jnp.sum(tail_weights * jnp.power(rho_grid, 1.5))
    tophat_sfr = M_tail * jnp.sqrt(rho_tail_mean)

    return jnp.maximum(sfr_weighted / jnp.maximum(tophat_sfr, 1e-10), 1.0)
```

`rho_grid` is the 3D density field (shape `(Nx, Ny, Nz)`) and
`tail_weights` is the matching soft-mask field (same shape, values in
$[0, 1]$). Both are typically generated upstream by
`compute_tail_pmfs_bm19` which evaluates the BM19 forward model on
the same grid as the density field.

```{note}
**Volume element $\mathrm{d}V$ cancels.** The numerator and
denominator both contain $\mathrm{d}V$ as a factor; for a uniform
voxel grid it cancels exactly, leaving the unitless sums above.
For non-uniform grids (rare in cloud studies but common in cosmological
simulations), pass an explicit `dV` weight to a generalised version.
```

## How `tail_weights` is constructed

The soft tail weights are computed from the same {cite:t}`Burkhart2018`
PDF parameters that drive the analytic $\zeta(p)$ from [](pp20.md):

```python
# Given sigma_s, alpha, the BM19 transition density s_t is closed-form
sigma_s_sq = sigma_s_squared(mach=10.0, b=0.4)
s_t = transition_density(sigma_s_sq, alpha=2.0)

# Soft mask via sigmoid in log-density space
s = jnp.log(rho_grid / jnp.mean(rho_grid))
kappa = 10.0
tail_weights = jax.nn.sigmoid(kappa * (s - s_t))

# Direct ζ measurement
zeta_direct = zeta_fdf_direct(rho_grid, tail_weights)
```

The full pipeline — turbulence parameters → BM19 transition density
→ soft mask → 3D ζ — is implemented in
`progenax.gravoturb.bm19_model.compute_tail_pmfs_bm19`. See [](bm19.md).

## What "uniform tail with weighted mean density" means

The denominator $M_{\mathrm{tail}}\,\sqrt{\bar\rho_{\mathrm{tail}}}$
is the SFR a uniform-density slab would produce *if it had the same
mass and the same weighted-mean density* as the actual tail. This is
the natural top-hat reference for the FDF integral: it asks "how much
extra SFR does the *non-uniformity* of the tail buy you?" rather than
"how much extra SFR does star formation buy you above zero?".

The ratio $\zeta_{\mathrm{FDF}}$ is therefore exactly comparable to the
geometric ζ from [](pp20.md): both measure SFR-magnification over a
uniform-density reference. For a power-law profile, they agree to the
precision of the soft-mask approximation (typically $< 1\%$ for
production $\kappa$ values).

## Limiting behaviours

```{list-table}
:header-rows: 1
:widths: 30 30 40

* - Field shape
  - Tail weights
  - $\zeta_{\mathrm{FDF}}$
* - Uniform $\rho$ everywhere
  - Any non-empty $w$
  - 1 (top-hat)
* - Concentrated central peak
  - $w$ peaks where $\rho$ peaks
  - $> 1$
* - Hard-threshold tail
  - $\kappa \to \infty$, sharp $w$
  - Approaches the hard-mask answer
* - Empty tail
  - All $w = 0$
  - Returns clamped value (1)
```

The fourth case — empty tail — is handled defensively: progenax
clamps $V_{\mathrm{tail}}$ and $M_{\mathrm{tail}}$ to floor values
$\sim 10^{-10}$ to avoid division-by-zero in degenerate inputs.

## When to use the direct 3D approach

```{list-table}
:header-rows: 1
:widths: 32 68

* - Use direct 3D ζ when…
  - …because
* - You have a simulation snapshot
  - The 3D field already exists; no need to fit a parametric profile
* - The density field has substructure
  - Substructure violates the {cite:t}`King1966`/PP20 spherical-symmetry assumption; direct measurement captures it
* - $s_t$ is itself a free parameter
  - Soft-mask sigmoid makes $\partial \zeta/\partial s_t$ analytic
* - Multi-component cloud
  - Multiple density populations along a line of sight; PDF-based approach is less robust
```

```{list-table}
:header-rows: 1
:widths: 32 68

* - Use PP20 / cored instead when…
  - …because
* - You only have observational $r_{\mathrm{eff}}$ + $\rho(r)$
  - PP20 is the right tool for power-law-fit profiles
* - Computational speed matters and field is small
  - Analytic forms are $\mathcal{O}(1)$; direct 3D is $\mathcal{O}(N_x^3)$
* - You want to compare with published PP20 ζ values
  - PP20 is the published reference; direct 3D is a generalisation
```

## Domain of validity

1. **Soft-mask width $\kappa$ trade-off** — large $\kappa$ approaches
   the hard mask but gives small gradients; small $\kappa$ gives strong
   gradients but blurs the threshold. progenax's $\kappa = 10$ default
   is a calibrated compromise for typical $\sigma_s \sim 1.7$ clouds;
   it can be exposed as a free parameter for inference if needed.
2. **No periodic-boundary handling** — direct ζ assumes the input grid
   is the full cloud volume, not a periodic-cube cosmological
   sub-volume. For cosmological simulations, extract the cloud
   sub-volume first.
3. **Uniform voxel size assumed** — for non-uniform grids, the volume
   element does not cancel and an explicit weighting is needed.
   progenax does not currently expose this; for cosmological adaptive-mesh
   refinements, it would need to be added.

## References

The direct-3D measurement of ζ from a soft-masked density field is
original to progenax; the soft-sigmoid mask follows the standard
"reparameterise discrete thresholds for autodiff" technique used
across machine-learning differentiable programming. The
{cite:t}`Burkhart2018` PDF parameters that drive the soft mask are
documented at [](density-pdf-fundamentals.md).
