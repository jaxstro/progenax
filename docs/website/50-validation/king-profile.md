---
title: King profile validation
description: "Validation suite for the King profile + matched velocity DF: ODE integration vs King 1966 Table II, tidal truncation, W₀ sweep."
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
* - Tidal radius $\xi_t = r_t/r_c$ from the ODE
  - $\le 0.03$ in $\log_{10}$
  - {cite:t}`King1966` Table II ($c$ column)
* - Concentration $c = \log_{10}(\xi_t)$ at $W_0 = 3,5,7,9$
  - $\le 0.02$–$0.03$ abs
  - {cite:t}`King1966` Table II ($\log c$)
* - $r_h / r_c$ varies with $W_0$
  - positivity + monotone trend
  - Integrated mass profile (code)
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

```{list-table} progenax vs King (1966) Table II at $W_0 = 7$.
:header-rows: 1

* - Quantity
  - King 1966 Table II
  - progenax
  - Match
* - $\xi_t = r_t/r_c$ (Table II $c$)
  - $33.71$
  - $33.75$
  - 0.1%
* - concentration $\log_{10}\xi_t$
  - $1.528$
  - $1.528$
  - exact
* - $r_h/r_c$ (computed; not in Table II)
  - —
  - $3.92$
  - —
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
