---
title: Cored profiles — `magnification_factor_with_core`
description: Numerical-integration ζ for profiles with a flat inner core, ρ(r) = ρ_c / [1 + (r/r_c)²]^(p/2). Bridges PP20's pure power-law ζ to realistic molecular-cloud geometries.
---

# Cored profiles: `magnification_factor_with_core`

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent + fractal-density-field (FDF) pipeline was rebuilt **clean-room** (2026-06) as
the standalone **`gravoturb_fdf`** package — a follow-up-paper feature **excluded from the released
progenax wheel**. Import it as `gravoturb_fdf` (repo-only, under `src/experimental/`), **not** as
`progenax.gravoturb` (removed in the 2026-06 rewrite). Fresh validation:
`src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`.
```

```{seealso}
This chapter introduces the **cored** ζ computation for profiles with
a flat inner core. For the *pure power-law* analytic ζ(p) it
generalises, see [](pp20.md). For the *direct 3D measurement* that
makes no parametric assumption, see [](direct-3d-zeta.md).
```

The PP20 analytic ζ(p) ([](pp20.md)) assumes a *pure power-law*
density profile $\rho(r) \propto r^{-p}$. Real molecular clouds have
**flat inner cores** — set by thermal pressure or magnetic support —
that transition to a power-law envelope outside some core radius
$r_c$. progenax's `magnification_factor_with_core` numerically
integrates ζ for the cored profile

```{math}
:label: cored-profile
\rho(r) \;=\; \frac{\rho_c}{\bigl[\,1 + (r/r_c)^2\,\bigr]^{p/2}}
```

This profile is flat at $r \ll r_c$ (with $\rho \to \rho_c$) and
power-law at $r \gg r_c$ (with $\rho \propto r^{-p}$), producing a
finite central density without sacrificing the power-law outer slope
that drives the gravitating-tail SFR.

## Why cored matters

The pure power-law $\rho \propto r^{-p}$ is mathematically convenient
but physically problematic at $r = 0$: it predicts infinite density
at the cloud centre, which is unphysical. Real molecular clouds have
a finite central density set by either thermal pressure (the Jeans
mass at the cloud temperature) or magnetic-field support. The cored
profile {eq}`cored-profile` captures this with a single new parameter
$r_c$ — the radius below which the profile flattens.

This matters for ζ in two ways:

1. **At $p \to 2$** (singular isothermal limit), the pure power-law
   $\zeta$ diverges as $1/(2-p)$ ([](pp20.md)). The cored ζ stays
   *finite* because the flat inner core regularises the
   $\int \rho^{3/2}\,\mathrm{d}V$ integral.
2. **For inference**, the cored form gives stable gradients at all
   $p$ — including $p \to 2$ where the pure form is clipped at
   $P_{\max} = 1.95$. progenax exposes both, with the cored function
   as the right choice when $r_c/R$ is itself a free parameter.

## The cored-profile ζ

Substituting {eq}`cored-profile` into the integral definition

```{math}
\zeta \;=\; \frac{\int \rho^{3/2}\,\mathrm{d}V}{M\,\sqrt{\langle\rho\rangle}}
```

with the cloud volume restricted to $r \le R$ (outer radius), the
result has no closed form for general $(p, r_c/R)$. progenax
evaluates the integral numerically on a fixed log-spaced radial grid
via trapezoidal quadrature:

```python
@jax.jit
def magnification_factor_with_core(p, r_c_over_R, n_radial_points=100):
    x = jnp.linspace(0.01, 1.0, n_radial_points)        # r / R
    x_c = jnp.maximum(r_c_over_R, 1e-4)
    rho = jnp.power(1.0 + (x / x_c) ** 2, -p / 2.0)     # ρ / ρ_c
    dV = 4.0 * jnp.pi * x ** 2 * (x[1] - x[0])

    M = jnp.sum(rho * dV)
    mean_rho = M / (4.0 * jnp.pi / 3.0)
    sfr_int = jnp.sum(jnp.power(rho, 1.5) * dV)
    return jnp.maximum(sfr_int / (M * jnp.sqrt(mean_rho)), 1.0)
```

The output is differentiable in both $p$ and $r_c/R$. $n_{\mathrm{radial\ points}}
= 100$ is the default; tests show $\sim 1\%$ accuracy over the
production parameter range.

## Limiting behaviours

```{list-table} Limits of `magnification_factor_with_core(p, r_c/R)`.
:header-rows: 1

* - Limit
  - Behaviour
  - Recovers
* - $r_c/R \to 0$
  - Pure power-law profile
  - Analytic $\zeta(p)$ from [](pp20.md)
* - $r_c/R \to 1$
  - Flat profile (core larger than cloud)
  - $\zeta = 1$ (uniform density)
* - $p \to 0$
  - $\rho \to$ const everywhere
  - $\zeta = 1$
* - $p \to 2$
  - Singular isothermal at large $r$
  - Large but finite ζ (core regularises)
```

The first limit is the most important: as $r_c/R \to 0$, the cored
profile becomes a pure power-law and the cored ζ should converge to
the analytic PP20 value. `tests/experimental/unit/test_pp20.py` and
the AC4 direct-field check verify ζ convergence at
$p \in \{0.5, 1.0, 1.5\}$.

The fourth limit is what distinguishes cored from pure: a *cored*
profile with $p \to 2$ does *not* diverge the way the pure power-law
does. progenax does not clip $p$ in the cored function — the
numerical safety problem doesn't arise.

## When to use cored vs pure power-law

```{list-table}
:header-rows: 1

* - Use cored when…
  - …because
* - Real cloud with thermal-pressure inner core
  - Pure power-law overpredicts ζ near $p \to 2$
* - $p$ inference where $p$ might approach 2
  - Cored ζ stays finite at all $p$, gives stable HMC
* - Need to vary $r_c/R$ as a free parameter
  - Cored has it; PP20 does not
```

```{list-table}
:header-rows: 1

* - Use pure power-law (PP20) when…
  - …because
* - Idealised analytical analysis
  - Closed form available; no quadrature needed
* - $p < 1.5$ regime
  - Cored and pure-power-law agree to $< 1\%$ here; pure is faster
* - Reproducing {cite:t}`ParmentierPasquali2020` Eq. 6
  - PP20 is the published formula; cored is a generalisation
```

## Implementation in `gravoturb_fdf`

```python
from gravoturb_fdf.theory.pp20 import (
    magnification_factor,
    magnification_factor_with_core,
)

# Pure power-law ζ (analytic)
zeta_pure = magnification_factor(jnp.float64(1.5))            # √2 ≈ 1.414

# Cored ζ with r_c/R = 0.1 (typical core size)
zeta_cored = magnification_factor_with_core(
    jnp.float64(1.5), jnp.float64(0.1)
)                                                              # ≈ 1.4
```

Both functions are `@jax.jit`-compatible. `magnification_factor_with_core`
is differentiable in both $p$ and $r_c/R$.

```{warning}
**`n_nodes` must be a static int, not a traced value.** Under JIT, the
trapezoid grid is constructed at trace time; making it trace-dependent
breaks the grid construction. The API exposes it as a Python `int`
keyword argument with a sensible default (`n_nodes=2048`).
```

## Domain of validity

1. **Inner-edge approximation** — the trapezoid starts at $x = 1/n_{\mathrm{nodes}}$
   ($\approx 5\times10^{-4}$ at the default $n_{\mathrm{nodes}}=2048$), not exactly
   at $r = 0$. Because the integrand $\sim x^2$ near the core, the omitted region
   contributes $\mathcal{O}((1/n_{\mathrm{nodes}})^3) \sim 10^{-10}$.
2. **Trapezoidal accuracy** — fixed-grid trapezoid with the default
   `n_nodes=2048` is accurate to well below a percent; increase `n_nodes`
   for more (the cost scales linearly).
3. **Spherical symmetry assumed** — the cored profile {eq}`cored-profile`
   is spherical. Non-spherical cores require the direct 3D treatment;
   see [](direct-3d-zeta.md).
4. **Single inner-core scale** — real clouds may have hierarchical
   substructure with multiple core radii. `gravoturb_fdf` models the
   single-scale case; multi-scale requires the direct 3D approach
   or a parametric extension not currently implemented.

## References

The cored profile {eq}`cored-profile` is a standard {cite:t}`King1966`-style
generalisation of the pure power-law; it appears throughout the
molecular-cloud literature. `gravoturb_fdf`'s numerical integration follows
the standard trapezoidal-quadrature approach. For the corresponding
analytic limit see [](pp20.md); for the parameter-free 3D approach
see [](direct-3d-zeta.md).
