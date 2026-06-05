---
title: Density PDF fundamentals
description: The Federrath & Klessen (2012) lognormal + power-law density PDF for self-gravitating supersonic turbulence — derivation, Mach-number scaling, forcing parameter b, and the transition to the dense power-law tail.
---

# Density PDF fundamentals

A turbulent molecular cloud has a *distribution* of densities, not a
single density. The shape of that distribution — the **volume-density
PDF** $p_V(\rho)$ — is the foundational object that every subsequent
chapter in the gravoturbulence section consumes. progenax adopts the
{cite:t}`FederrathKlessen2012` lognormal + power-law parameterisation:
a lognormal core (set by the cloud's turbulence) joined to a
high-density power-law tail (set by self-gravity-driven collapse).

This chapter derives both pieces, lists the parameter dependences on
Mach number and forcing geometry, and connects to the
[](freefall-density-factor.md) that turns the PDF into a star-formation
rate.

## The lognormal core

In a *purely turbulent* (non-self-gravitating) supersonic medium, the
density PDF is well-described by a lognormal:

```{math}
:label: pdf-lognormal
p_V(s) \;=\; \frac{1}{\sqrt{2\pi\sigma_s^2}}\,\exp\!\biggl[-\frac{(s - \langle s\rangle)^2}{2\sigma_s^2}\biggr]
```

where $s \equiv \ln(\rho/\langle\rho\rangle)$ is the log-density,
$\langle s \rangle = -\sigma_s^2/2$ (so that $\langle\rho\rangle$ is
the mean), and $\sigma_s^2$ is the log-density variance. The lognormal
form follows from the central limit theorem applied to the cumulative
product of compressions and rarefactions a fluid element experiences
in a turbulent cascade.

The variance scales with Mach number as

```{math}
:label: sigma-s
\sigma_s^2 \;=\; \ln\bigl[1 + b^2\,\mathcal{M}^2\bigr]
```

with $b$ the **turbulence-driving parameter** ($b \in [1/3, 1]$) and
$\mathcal{M} = v_{\mathrm{turb}} / c_s$ the sonic Mach number. The
forcing parameter has well-defined physical limits:

```{list-table}
:header-rows: 1

* - $b$
  - Forcing
  - Physical interpretation
* - $1/3$
  - Solenoidal (divergence-free)
  - Turbulence stirred by shear; rotational modes only
* - $0.4$
  - Natural mix
  - Default for ISM turbulence in observations
* - $1$
  - Compressive (curl-free)
  - Turbulence stirred by compression; e.g. expanding HII regions, supernova shells
```

For a typical Mach $\mathcal{M} = 10$ molecular cloud and $b = 0.4$,
$\sigma_s^2 = \ln[1 + 16] \approx 2.83$, so $\sigma_s \approx 1.7$.
The lognormal therefore spans about 4 e-folds in $s$ at the $\pm 2\sigma$
level — orders of magnitude in $\rho$.

## The power-law tail

When self-gravity becomes important — at densities high enough that
the local freefall time is shorter than the turbulent crossing time —
the density PDF develops a **power-law tail** at high $s$:

```{math}
:label: pdf-power-law
p_V(s) \;\propto\; e^{-\alpha\,s}\quad\text{for}\quad s \gtrsim s_t
```

where $\alpha$ is the tail slope and $s_t$ is the **transition density**
where the PDF crosses from lognormal to power-law. Physically, the
power-law tail represents *self-gravitating gas* in approximate
isothermal collapse — the {cite:t}`Kritsuk2011` derivation maps
$\rho \propto r^{-p}$ to a power-law in $s$ with slope $\alpha = 3/p$.

For the canonical {cite:t}`Burkhart2021` α window
$\alpha \in [\alpha_{\mathrm{sat}}, \alpha_0] = [1.5, 3.0]$, the
corresponding radial profile slope is $p = 3/\alpha \in [1, 2]$ —
from "marginally collapsing" to "singular isothermal." This is the
mapping that connects the gravoturbulent framework to the
[](pp20.md) magnification factor.

## The transition density $s_t$

The matching point $s_t$ between lognormal and power-law is
constrained by *continuity* of $p_V(s)$ and its derivative
{cite:p}`Burkhart2018`:

```{math}
:label: s-t
s_t \;=\; \Bigl(\alpha - \tfrac{1}{2}\Bigr)\,\sigma_s^2
```

This is a closed-form expression: given $\sigma_s^2$ (set by Mach +
forcing) and $\alpha$ (a free parameter), $s_t$ is determined.
progenax's `transition_density(sigma_s_sq, alpha)` returns this value
directly; it is differentiable in both arguments.

For typical molecular-cloud parameters ($\sigma_s^2 \approx 2.8$,
$\alpha \approx 2$), $s_t = (\alpha - \tfrac{1}{2})\sigma_s^2 \approx 4.25$,
meaning the lognormal-to-power-law transition occurs around
$\rho/\langle\rho\rangle \approx e^{4.25} \approx 70$. This matches the
{cite:t}`Kainulainen2014` observational dense-gas threshold
$s_{\mathrm{th}} \approx 4.2$ (as adopted by {cite:t}`ParmentierPasquali2020`).

## Implementation in progenax

```python
from progenax.gravoturb.bm19_model import (
    sigma_s_squared, transition_density,
)
from progenax.gravoturb.bm19_pdf import bm19_volume_pdf

mach = 10.0          # Sonic Mach number
b = 0.4              # Forcing parameter
alpha = 2.0          # Power-law tail slope

sigma_s_sq = sigma_s_squared(mach, b)             # ≈ 2.83
sigma_s = jnp.sqrt(sigma_s_sq)
s_t = transition_density(sigma_s_sq, alpha)        # ≈ 4.25

# Evaluate the full PDF
s_grid = jnp.linspace(-5, 10, 200)
pdf_values = bm19_volume_pdf(s_grid, sigma_s_sq, s_t, alpha)
```

`bm19_volume_pdf` evaluates the lognormal piece for $s < s_t$ and
the power-law piece for $s \ge s_t$, with continuous matching at $s_t$
enforced by the closed-form {eq}`s-t`. The function is JIT-compatible
and differentiable in $\mathcal{M}$, $b$, and $\alpha$ — useful for
inferring all three from observed cloud properties.

## Domain of validity

1. **Supersonic turbulence required** — at low Mach the lognormal
   variance shrinks below $\sim 1$ and the lognormal approximation
   breaks down. progenax's parameterisation works for $\mathcal{M}
   \gtrsim 3$.
2. **Self-gravity required for the power-law tail** — at very low
   density (diffuse warm neutral medium) there is no gravitating
   tail and the PDF is purely lognormal. The {cite:t}`Burkhart2018`
   framework assumes a power-law tail exists.
3. **Single-component clouds** — the PDF describes one cloud; for
   multi-component cloud complexes (multiple clouds along the line of
   sight) the observed PDF is a convolution. progenax does not
   currently model the multi-cloud case.
4. **Static description** — the PDF is the *time-averaged* density
   distribution. Real clouds evolve; the PDF parameters change on
   the cloud's free-fall timescale. For inference, the PDF is
   typically evaluated at a single instantaneous snapshot.

## References

The lognormal core is the standard turbulence prediction; see
{cite:t}`FederrathKlessen2012` for the canonical reference and the
$\sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)$ derivation. The power-law
tail is {cite:t}`Kritsuk2011`'s derivation from collapsing self-gravitating
gas. The transition-density matching is {cite:t}`Burkhart2018`. For
observational verification of the lognormal+power-law form see
{cite:t}`Kainulainen2014`.
