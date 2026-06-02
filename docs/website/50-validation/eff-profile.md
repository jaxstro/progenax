---
title: EFF profile validation
description: "Validation suite for the Elson-Fall-Freeman (1987) profile: density sampling, asymptotic-slope verification, half-mass-radius mapping."
---
# EFF profile validation

EFF tests verify the closed-form expressions and the inverse-CDF
sampling. Test file: `tests/validation/test_eff_physics.py`.

## What is verified

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - Half-mass radius
  - 1% rel
  - $r_h = a\sqrt{2^{2/(\gamma-3)} - 1}$
* - Total mass at $\gamma > 3$
  - $10^{-12}$ rel
  - Closed-form integral
* - Outer slope at $r \gg a$
  - Recovers $\rho \propto r^{-\gamma}$
  - Asymptotic limit
* - $\gamma > 3$ check at construction
  - Raises `ValueError`
  - Mass would diverge otherwise
* - Inverse-CDF sampling KS test
  - p-value $> 0.05$ at $N = 10^4$
  - Density-profile match
* - Differentiability in $\gamma$
  - Finite gradient
  - $\gamma$ is differentiable (unlike King's $W_0$)
```

## Spot results at $\gamma = 4$, $r_h = 1$

```{list-table}
:header-rows: 1

* - Quantity
  - Expected
  - Measured
  - Status
* - $a$ from $r_h$
  - $1.0$ exactly
  - $1.0$
  - Pass
* - $M(<r)/M$ at $r = a$
  - $0.354$
  - $0.354 \pm 0.005$
  - Pass
* - Outer slope at $r = 5 a$
  - $-4.0$
  - $-3.97$
  - Pass
```

## How to run

```bash
pytest tests/validation/test_eff_physics.py -v
```

$\sim 20$ seconds on CPU.

## References

{cite:t}`ElsonFallFreeman1987` is the original. Theory at
[](../10-theory/spatial-profiles/eff.md).
