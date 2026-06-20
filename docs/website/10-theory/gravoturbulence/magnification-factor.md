---
title: The magnification factor ζ — three ways to compute it
description: The Parmentier & Pasquali (2020) ζ(p) for power-law profiles, the numerical cored-profile ζ, and the parameter-free direct-3D ζ measurement — one definition, one "which mode when" table, and the gravoturb_fdf implementations verified against source.
---

# The magnification factor ζ

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent + fractal-density-field (FDF) pipeline was rebuilt **clean-room** (2026-06) as
the standalone **`gravoturb_fdf`** package — a follow-up-paper feature **excluded from the released
progenax wheel**. Import it as `gravoturb_fdf` (repo-only, under `src/experimental/`), **not** as
`progenax.gravoturb` (removed in the 2026-06 rewrite). Fresh validation:
`src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`.
```

The **magnification factor** ζ quantifies how much the cloud-integrated star formation rate (SFR) of
a centrally-concentrated cloud exceeds that of a uniform-density ("top-hat") cloud of the same total
mass and outer radius. {cite:t}`ParmentierPasquali2020` introduced ζ as a compact way to fold the
*geometry* of a molecular cloud into the dense-gas SFR-mass relation, extending earlier
"single-mean-density" frameworks {cite:p}`TanKrumholzMcKee2006,FederrathKlessen2012,Burkhart2018`.

This chapter gives the single integral definition of ζ, then derives and implements the **three
computation modes** that `gravoturb_fdf` exposes — each captures a different physical situation:

- **PP20 analytic ζ(p)** — exact, closed-form, for a *pure power-law* profile.
- **Cored ζ** — numerical, for a profile with a *flat inner core* (realistic molecular clouds).
- **Direct 3D ζ** — parameter-free, measured directly from an *arbitrary 3D density field*.

```{seealso}
The cloud's *density-space* dual of ζ — the PDF-weighted integral
$\int(\rho/\langle\rho\rangle)^{3/2}p_V(\rho)\,\mathrm d\rho$ — and the α↔p mapping that feeds $p$
into ζ(p) are developed in [](density-pdf-and-fdf.md). The {cite:t}`Burkhart2018` framework that
consumes ζ downstream is [](bm19.md).
```

## Why ζ exists, and how it is defined

The local star formation rate per unit volume scales as $\dot\rho_\star \propto \rho /
t_{\mathrm{ff}}(\rho) \propto \rho^{3/2}$, because the local free-fall time $t_{\mathrm{ff}} \propto
\rho^{-1/2}$ (the [freefall-density factor](density-pdf-and-fdf.md)). A cloud whose density field is
*not* uniform therefore fragments faster in its high-density regions than its mean density would
suggest — the cloud-integrated SFR is biased above the SFR a uniform cloud of the same mass would
produce.

ζ is the ratio of these two SFRs, at fixed total mass $M$ and outer radius $R$:

```{math}
:label: zeta-def
\zeta \;\equiv\; \frac{\mathrm{SFR}_{\mathrm{clump}}}{\mathrm{SFR}_{\mathrm{TH}}}
     \;=\; \frac{\int_{V_R} \rho^{3/2}\,\mathrm{d}V}{M\,\sqrt{\langle\rho\rangle}}
     \;=\; \frac{\langle\rho^{1/2}\rangle_{\mathrm{mass}}}{\langle\rho\rangle^{1/2}}
```

where $V_R$ is the cloud volume, $\langle\rho\rangle = M/V_R$ is the volume-averaged density, and the
top-hat reference $\mathrm{SFR}_{\mathrm{TH}} \propto M / t_{\mathrm{ff}}(\langle\rho\rangle) \propto
M \sqrt{\langle\rho\rangle}$ is the SFR the same gas mass would produce if redistributed uniformly.
By construction $\zeta(\rho = \mathrm{const}) = 1$ — a uniform cloud is its own top-hat reference.
For ζ > 1 the cloud is "magnified"; centrally concentrated profiles always give ζ > 1, and the
steeper the concentration, the larger ζ.

## Mode 1 — PP20 analytic ζ(p) for a power-law profile

Consider a sphere of outer radius $R$ filled with a pure power-law profile

```{math}
:label: pp20-rho
\rho(r) \;=\; \rho_R \left(\frac{r}{R}\right)^{-p},
\qquad 0 \le p < 2.
```

The upper bound $p < 2$ is set by the convergence threshold of the SFR integrand
$\int r^{2-3p/2}\,\mathrm{d}r$ — physically, $p = 2$ is the singular isothermal sphere where the
central density runs away. Five steps evaluate {eq}`zeta-def` analytically.

**Step 1 — total mass.**

```{math}
M \;=\; 4\pi\,\rho_R\, R^{p} \int_0^R r^{2-p}\,\mathrm{d}r
  \;=\; \frac{4\pi\,\rho_R\, R^3}{3-p}
\qquad (p < 3).
```

**Step 2 — mean density.**

```{math}
\langle\rho\rangle \;=\; \frac{M}{(4/3)\pi R^3}
  \;=\; \frac{3\,\rho_R}{3-p}.
```

**Step 3 — SFR integral.**

```{math}
\int \rho^{3/2}\,\mathrm{d}V
  \;=\; 4\pi\,\rho_R^{3/2}\, R^{3p/2} \int_0^R r^{2-3p/2}\,\mathrm{d}r
  \;=\; \frac{8\pi\,\rho_R^{3/2}\, R^3}{3\,(2-p)}.
```

The denominator $(2-p)$ in this step is the source of the $\zeta \to \infty$ divergence as
$p \to 2$.

**Step 4 — top-hat reference.**

```{math}
M\,\langle\rho\rangle^{1/2}
  \;=\; \frac{4\pi\sqrt{3}\,\rho_R^{3/2}\, R^3}{(3-p)^{3/2}}.
```

**Step 5 — combine.** Dividing Step 3 by Step 4 collapses every $\rho_R$ and $R$ factor:

```{math}
:label: pp20-zeta-canonical
\boxed{\;\zeta(p) \;=\; \frac{2\,(3-p)^{3/2}}{3^{3/2}\,(2-p)}
       \;=\; \frac{(3-p)^{3/2}}{(3^{3/2}/2)\,(2-p)}\;}
```

This is the canonical analytic form implemented in `gravoturb_fdf.theory.pp20.magnification_factor`.
{cite:t}`ParmentierPasquali2020` quote the same result in Eq. 6 as

```{math}
:label: pp20-eq6
\zeta(p) \;=\; \frac{(3-p)^{3/2}}{2.6\,(2-p)},
```

where "2.6" is a numerical approximation to $3^{3/2}/2 = 2.598\ldots$ Equations
{eq}`pp20-zeta-canonical` and {eq}`pp20-eq6` therefore agree to 0.08% across the entire physical
$0 \le p < 2$ domain. `gravoturb_fdf` uses the unrounded {eq}`pp20-zeta-canonical` form so that
anchor values are exact, and the exact constant is fixed by the physical top-hat lower limit
$\zeta(0) = 1$.

### Spot values

These anchors are exact analytic values; `tests/experimental/unit/test_pp20.py` and AC3 lock them
(the AC suite prints them — see `gravoturb_fdf/VALIDATION_SUMMARY.md`). The ζ column is computed
directly with `magnification_factor`.

```{list-table}
:header-rows: 1

* - $p$
  - $\zeta(p)$ (canonical)
  - PP20 Eq. 6 ("2.6" rounded)
  - Comment
* - 0
  - 1 (exact)
  - 0.999
  - Top-hat; reference value by construction
* - 1
  - $2 \cdot 2^{3/2}/3^{3/2} \approx 1.0887$
  - 1.0879
  - SIS-like inner core
* - 1.5
  - $\sqrt{2} \approx 1.4142$
  - 1.4132
  - Exact analytic anchor
* - 1.67
  - 1.789
  - 1.788
  - {cite:t}`Kainulainen2014` median observational $p$ for 16 nearby clouds
* - 1.9
  - 4.441
  - 4.437
  - Approaching the singular isothermal limit
* - 1.95
  - 8.282
  - 8.276
  - Deep in the near-singular regime
```

The $p = 1.67$ value is observational gold: {cite:t}`Kainulainen2014` derived $p \approx 5/3$ as the
median radial slope inferred from the column-density PDFs of 16 nearby molecular clouds (Pipe, Lupus,
Perseus, Aquila, Polaris, …). The corresponding $\zeta \approx 1.79$ is a useful sanity check for any
PP20 implementation. ζ rises steeply toward $p = 2$ (from $\approx 4.44$ at $p = 1.9$ to $\approx
8.28$ at $p = 1.95$) — the divergence the cored and direct-3D modes below regularise.

### Implementation

The shipping function is exactly {eq}`pp20-zeta-canonical` — **there is no clip and no `P_MAX`**.
The analytic form is well-behaved over the entire physical $0 \le p < 2$ window; the only singularity
is at $p = 2$ (not at $p = 1.3$, despite earlier buggy docstrings — see the Historical note). Where
gradients near $p \to 2$ would destabilise an HMC chain, use the **cored** mode below, which stays
finite at all $p$.

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

# Exact PP20 constant: 3^{3/2}/2 = 2.5980762... (PP20 Eq. 6 prints it rounded as 2.6).
_PP20_CONST = 3.0**1.5 / 2.0

def magnification_factor(p: Float[Array, ""]) -> Float[Array, ""]:
    # zeta(p) = (3-p)^{3/2} / [ (3^{3/2}/2) (2-p) ];  valid 0 <= p < 2, diverges as p -> 2.
    return (3.0 - p) ** 1.5 / (_PP20_CONST * (2.0 - p))
```

The function is JIT-compatible, vectorisable via `jax.vmap`, and differentiable across the physical
domain. See the `gravoturb_fdf.theory.pp20` source for the full signature and
[](../../50-validation/physics-tests.md) for the regression suite that anchors every spot value
above.

## Mode 2 — cored profiles: `magnification_factor_with_core`

The PP20 analytic ζ(p) assumes a *pure power-law* profile, which predicts infinite density at
$r = 0$. Real molecular clouds have **flat inner cores** — set by thermal pressure or magnetic
support — that transition to a power-law envelope outside some core radius $r_c$. `gravoturb_fdf`'s
`magnification_factor_with_core` numerically integrates ζ for the cored profile

```{math}
:label: cored-profile
\rho(r) \;=\; \frac{\rho_c}{\bigl[\,1 + (r/r_c)^2\,\bigr]^{p/2}}
```

This profile is flat at $r \ll r_c$ (with $\rho \to \rho_c$) and power-law at $r \gg r_c$ (with
$\rho \propto r^{-p}$), producing a finite central density without sacrificing the power-law outer
slope that drives the gravitating-tail SFR. This matters for ζ in two ways:

1. **At $p \to 2$** the pure power-law $\zeta$ diverges as $1/(2-p)$. The cored ζ stays *finite*
   because the flat inner core regularises the $\int \rho^{3/2}\,\mathrm{d}V$ integral.
2. **For inference**, the cored form gives stable gradients at all $p$ — including $p \to 2$ where
   the pure form diverges — and exposes $r_c/R$ as a free parameter.

Substituting {eq}`cored-profile` into {eq}`zeta-def`, the result has no closed form for general
$(p, r_c/R)$. `gravoturb_fdf` evaluates the integral numerically on a fixed (grad-safe) radial grid
and delegates the actual ratio to the direct estimator below — the constant $4\pi R^3$ volume factor
cancels in the ratio:

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def magnification_factor_with_core(
    p: Float[Array, ""],
    r_c_over_R: Float[Array, ""],
    n_nodes: int = 2048,
) -> Float[Array, ""]:
    # Cored rho(r) = rho_c [1 + (r/r_c)^2]^{-p/2}, trapezoid over r/R in (0, 1].
    x = jnp.linspace(1.0 / n_nodes, 1.0, n_nodes)        # r/R in (0,1]
    rho = (1.0 + (x / r_c_over_R) ** 2) ** (-p / 2.0)    # rho / rho_c
    w = x**2                                             # dV ~ r^2 dr (4*pi*R^3 cancels)
    return zeta_fdf_direct(rho, w)
```

The output is differentiable in both $p$ and $r_c/R$. As $r_c/R \to 0$ the profile approaches a pure
power law and ζ → the analytic `magnification_factor`; as $r_c/R \to \infty$ (a core larger than the
cloud) it approaches a top-hat and ζ → 1.

```{warning}
**`n_nodes` must be a static int, not a traced value.** Under JIT the trapezoid grid is constructed
at trace time; making it trace-dependent breaks the grid construction. The API exposes it as a Python
`int` keyword with a sensible default (`n_nodes=2048`). The trapezoid starts at $x = 1/n_{\mathrm{nodes}}$
($\approx 5\times10^{-4}$ at the default), not exactly $r = 0$; because the integrand $\sim x^2$ near
the core, the omitted region contributes $\mathcal{O}((1/n_{\mathrm{nodes}})^3) \sim 10^{-10}$.
```

### Limiting behaviours

```{list-table} Limits of `magnification_factor_with_core(p, r_c/R)`.
:header-rows: 1

* - Limit
  - Behaviour
  - Recovers
* - $r_c/R \to 0$
  - Pure power-law profile
  - Analytic $\zeta(p)$ from Mode 1
* - $r_c/R \to 1$
  - Core comparable to cloud
  - $\zeta$ approaching 1 (near-uniform); e.g. $\zeta(1.5, 0.1) \approx 1.19$ vs pure $\sqrt 2 \approx 1.41$
* - $p \to 0$
  - $\rho \to$ const everywhere
  - $\zeta = 1$
* - $p \to 2$
  - Singular isothermal at large $r$
  - Large but finite ζ (core regularises)
```

`tests/experimental/unit/test_pp20.py` and the AC4 direct-field check verify ζ convergence to the
analytic value at $p \in \{0.5, 1.0, 1.5\}$ as $r_c/R \to 0$.

(direct-3d)=
## Mode 3 — direct 3D ζ: `zeta_fdf_direct`

The PP20 and cored modes both *parameterise* the cloud's density profile. For inferences using
simulation snapshots or detailed observational density maps, that parameterisation throws away most
of the information in the data. `zeta_fdf_direct` instead measures ζ *directly* from a sampled
density field and its volume weights:

```{math}
:label: zeta-fdf-direct
\zeta_{\mathrm{FDF}} \;=\; \frac{\sum_{i} \rho_i^{3/2}\,w_i \,\bigl(\sum_i w_i\bigr)^{1/2}}
                                 {\bigl(\sum_i \rho_i\,w_i\bigr)^{3/2}}
```

This is exactly $\langle\rho^{1/2}\rangle_{\mathrm{mass}}/\langle\rho\rangle^{1/2}$ for cells of
volume $w_i$ — the discrete form of {eq}`zeta-def`. The implementation is the shipping source
verbatim:

```python
import jax.numpy as jnp
from jaxtyping import Array, Float

def zeta_fdf_direct(
    rho: Float[Array, " n"], weights: Float[Array, " n"]
) -> Float[Array, ""]:
    # zeta = sum(rho^{3/2} w) * sqrt(sum w) / (sum(rho w))^{3/2}.  rho need not be normalized.
    num = jnp.sum(rho**1.5 * weights) * jnp.sqrt(jnp.sum(weights))
    den = jnp.sum(rho * weights) ** 1.5
    return num / den
```

`rho` is the flattened density field (any shape; need not be normalized) and `weights` the matching
per-cell volume element. For a uniform voxel grid the volume element cancels exactly between numerator
and denominator, so passing `weights = ones_like(rho)` gives the correct unitless ζ. This is the
estimator `magnification_factor_with_core` (Mode 2) delegates to, and the right tool for measuring ζ
from a 3D field with no parametric assumption.

```{note}
**No clamps, no floors.** The shipping `zeta_fdf_direct` and `magnification_factor_with_core`
contain no `jnp.maximum(..., 1.0)` clip and no `1e-10` division guard — earlier doc snippets that
showed those were stale. ζ is the bare ratio {eq}`zeta-fdf-direct`; for a degenerate all-zero field
the ratio is naturally undefined and the caller is responsible for supplying a non-empty field.
```

### Soft-mask weights for differentiable tail inference

A common use is to weight the field by a *soft* identification of the gravitating tail rather than
its full volume. The natural "tail" definition — "everything above the transition density $\rho_t$"
— is a hard step function, whose gradient is zero almost everywhere; inferring $\rho_t$ or any
density-PDF parameter through it is impossible with autodiff. Replacing the step with a sigmoid in
log-density,

```{math}
:label: soft-mask
w(\mathbf{x}) \;=\; \frac{1}{1 + \exp\bigl[-\kappa\,(s(\mathbf{x}) - s_t)\bigr]},
\qquad s = \ln(\rho/\langle\rho\rangle),
```

gives a continuous, well-behaved gradient $\partial w/\partial s$ (the sigmoid derivative).
Large $\kappa$ approaches the hard threshold; small $\kappa$ gives a smooth transition (a typical
$\kappa \sim 10$ gives a transition width $\sim 0.1$ in $s$). This is the technical step that lets
the inference layer *infer* $s_t$ and the PDF parameters from observed dense-gas structure:

```python
import jax
import jax.numpy as jnp
from gravoturb_fdf.theory.bm19 import sigma_s_squared, transition_density
from gravoturb_fdf.theory.pp20 import zeta_fdf_direct

# BM19 transition density is closed-form in (sigma_s, alpha)
sigma_s_sq = sigma_s_squared(10.0, 0.4)
s_t = transition_density(2.0, sigma_s_sq)            # args: alpha, σ_s²

# soft tail weights via a sigmoid in log-density space
s = jnp.log(rho_grid.ravel() / jnp.mean(rho_grid))
tail_weights = jax.nn.sigmoid(10.0 * (s - s_t))

# direct ζ over the soft-masked field
zeta_direct = zeta_fdf_direct(rho_grid.ravel(), tail_weights)
```

In the full subsystem the density field and soft mask come from
`gravoturb_fdf.field.pipeline.build_fdf_field` (the AC6 cornerstone); the realized dense fraction is
`gravoturb_fdf.field.tail.f_tail_actual`. The soft-mask reparameterisation of a discrete threshold is
the standard differentiable-programming technique; here it makes $\partial\zeta/\partial s_t$
analytic and HMC-compatible (see [](inference.md)).

## Which ζ-mode when

```{list-table} Choosing among the three ζ computations.
:header-rows: 1

* - Mode
  - Use when…
  - Cost / differentiability
  - Caveat
* - **PP20 analytic** `magnification_factor(p)`
  - You have a fitted power-law slope $p$ (most observational papers); idealised analysis; reproducing PP20 Eq. 6
  - $\mathcal{O}(1)$, analytic gradient in $p$
  - Pure power-law only; diverges as $p \to 2$
* - **Cored** `magnification_factor_with_core(p, r_c/R)`
  - Real cloud with a thermal/magnetic inner core; $p$ inference that may approach 2; $r_c/R$ a free parameter
  - $\mathcal{O}(n_{\mathrm{nodes}})$ trapezoid, grad-safe in $p$ and $r_c/R$
  - Spherical, single core scale; stays finite at $p \to 2$
* - **Direct 3D** `zeta_fdf_direct(rho, w)`
  - You have a simulation snapshot or detailed map; substructure; $s_t$ itself a free parameter
  - $\mathcal{O}(N_{\mathrm{cells}})$ sum, grad-safe through soft-mask weights
  - Assumes uniform voxels unless explicit `w` given; needs a non-empty field
```

For HMC-based inference of cloud parameters, all three are differentiable. The choice depends on the
level of cloud parameterisation in the inference target: power-law fit → PP20; cored fit → cored;
gridded field → direct 3D.

## Historical note — the 2026-04-28 transcription fix

```{admonition} Historical
:class: note

A prior implementation of `magnification_factor` read

`zeta = (3 - p) / (2.6 - 2 * p) ** 1.5`   ← WRONG

This is a transcription error of {cite:t}`ParmentierPasquali2020` Eq. 6: the constant 2.6 was moved
*inside* the 3/2 power, the $(3-p)$ factor lost its 3/2 exponent, and the multiplicative $(2-p)$ was
garbled into linear $(2.6 - 2p)$. The buggy denominator vanishes at $p = 1.3$, which earlier
docstrings rationalised as a "domain limit." The actual PP20 Eq. 6 has no singularity at $p = 1.3$;
the only singularity is at $p = 2$. Numerical impact at the $p = 0.5$ canonical anchor: buggy form
gave $\zeta = 1.235$, true value is 1.014.

The fix landed 2026-04-28 along with regression tests anchoring ζ on the values in the spot-value
table above. Full history: [](../../90-development-log/2026-04-28-pp20-fix.md).
```

## References

ζ in this chapter follows {cite:t}`ParmentierPasquali2020` directly; the integral definition
{eq}`zeta-def` and the α↔p correspondence come from {cite:t}`TanKrumholzMcKee2006`,
{cite:t}`FederrathKlessen2012`, and {cite:t}`Kritsuk2011`. The {cite:t}`Kainulainen2014`
observational anchor provides the $p = 1.67$ check value used in regression tests. The cored profile
{eq}`cored-profile` is a standard {cite:t}`King1966`-style generalisation; the direct-3D soft-mask
measurement is original to `gravoturb_fdf`, with the soft-sigmoid following the standard
differentiable-programming "reparameterise discrete thresholds for autodiff" technique. The
{cite:t}`Burkhart2018,BurkhartMocz2019` framework that consumes ζ downstream is [](bm19.md).
```
