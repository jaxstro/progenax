---
title: King profile validation
description: "Validation suite for the King profile + matched velocity DF: ODE integration vs LIMEPY reference, tidal truncation, W₀ sweep."
---
# King profile validation

The King suite verifies the lowered-isothermal ODE integration and
the matched velocity DF. Test file: `tests/validation/test_king_physics.py`.

## What is verified

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - ODE solution at $W_0 = 5, 7, 9$
  - 0.1% rel
  - {cite:t}`Gieles2015` LIMEPY at $g = 1$
* - Tidal radius $\xi_t$
  - 0.5% rel
  - LIMEPY $\xi_t(W_0)$
* - Concentration $c = \log_{10}(\xi_t)$
  - 0.5% rel
  - $c(W_0)$ from ODE
* - $r_h / r_c$ vs $W_0$
  - 1% rel
  - LIMEPY interpolation table
* - Density profile sampling
  - KS p-value $> 0.05$ at $N = 10^4$
  - Inverse-CDF on integrated $M(<\xi)$
* - Velocity dispersion profile
  - 2% rel
  - Numerical from $f(W, v)$ at $N = 10^4$
* - Virial ratio $Q_{\mathrm{vir}}$
  - $5\!\times\!10^{-3}$ abs
  - $2T + V = 0$
* - Tidal-radius cutoff
  - 100% bound
  - All particles have $r < r_t$
```

## Spot results at $W_0 = 7$

```{list-table}
:header-rows: 1

* - Quantity
  - LIMEPY
  - progenax
  - Match
* - $\xi_t = r_t/r_c$
  - $30.95$
  - $30.94$
  - 0.03%
* - $c$
  - $1.491$
  - $1.491$
  - $< 0.001$
* - $r_h/r_c$
  - $3.49$
  - $3.49$
  - 0.1%
* - $\sigma_0/\sqrt{GM/r_c}$
  - $0.667$
  - $0.667$
  - 0.05%
```

## How to run

```bash
pytest tests/validation/test_king_physics.py -v
```

$\sim 60$ seconds on CPU. The ODE integration is the dominant cost.

## What this suite does *not* test

- **$W_0$ gradients** — King's $W_0$ is treated as a static
  parameter; partial differentiability is documented at
  [](../20-architecture/differentiability.md).
- **Multi-mass King** — LIMEPY's strength; not yet in progenax.
- **Rotating King** — out of scope.

## References

{cite:t}`King1966` is the original; {cite:t}`Gieles2015` LIMEPY is
the cross-validation reference. Theory at
[](../10-theory/spatial-profiles/king.md).
