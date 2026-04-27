---
title: Binary-aware IMF validation
description: End-to-end validation of progenax's binary-aware IMF recovery — reproduces the 'confidently wrong' regime at N ≳ 10⁴ and demonstrates unbiased recovery via the marginalised likelihood.
---
# Binary-aware IMF recovery validation

The binary-aware IMF recovery workflow is an offline validation area,
not a pytest validation suite in this checkout. There is no
`tests/validation/test_binary_imf_recovery.py` file. Related scripts and
outputs live under `validation/imf/`, including
`validation/imf/validate_binary_aware_recovery.py`.

## What is verified

```{list-table}
:header-rows: 1
:widths: 32 24 44

* - Property
  - Tolerance
  - Anchor
* - Forward model $f(M_1, q, P, e)$
  - Match {cite:t}`Moe2017` Tables 10-13
  - Per-mass-bin spot checks
* - Marginalised likelihood
  - Offline validation
  - `validation/imf/validate_binary_aware_recovery.py`
* - Naive likelihood (control)
  - $|\Delta\alpha| \ge 0.05$ at $N = 10^4$
  - Demonstrates bias still present
* - "Confidently wrong" regime
  - Naive 95% CI excludes truth at $N = 10^4$
  - Full reproduction of headline result
* - $\hat R < 1.01$
  - 4 chains × 1500 samples
  - NUTS convergence check
* - Computational cost
  - $\sim 35$ min on MacBook Pro CPU at $N = 30k$
  - Documented benchmark
```

## The "confidently wrong" demonstration

```{list-table}
:header-rows: 1

* - $N$
  - Naive 95% CI width
  - Naive $|\Delta\alpha|$
  - Naive status
  - Binary-aware $|\Delta\alpha|$
* - 500
  - 0.28
  - 0.045
  - CI contains truth
  - 0.018
* - 1{,}000
  - 0.20
  - 0.035
  - CI contains truth
  - 0.011
* - 3{,}000
  - 0.12
  - 0.057
  - CI contains truth (barely)
  - 0.008
* - 10{,}000
  - 0.06
  - **0.082**
  - **CI excludes truth**
  - 0.005
* - 30{,}000
  - 0.035
  - **0.098**
  - **Confidently wrong**
  - 0.003
```

This table is an offline result and should be regenerated from the
validation scripts before being treated as current CI evidence.

## How to run

```bash
python validation/imf/validate_binary_aware_recovery.py
```

The full reproduction takes $\sim 4$ hours on CPU at the largest $N$;
$\sim 5$ minutes per chain on A100 GPU.

## What this suite establishes

The "confidently wrong" finding is the headline result for
progenax's binary-aware framework. It demonstrates:

1. The bias is real, quantifiable, and growing with $N$.
2. The marginalised likelihood eliminates it without further tuning.
3. The computational cost is tractable (35 min CPU, $\sim 30$ s GPU
   per chain at $N = 30{,}000$).

## References

The full chapter at [](../10-theory/imfs/binary.md) walks through
the methodology end-to-end. The likelihood derivation is at
[](../10-theory/imfs/binary-aware-likelihood.md). The {cite:t}`Moe2017`
calibration is at [](../10-theory/imfs/multiplicity-statistics.md).
