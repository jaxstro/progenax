---
title: IMF statistics validation
description: "Validation suite for canonical IMFs (Salpeter, Kroupa, Chabrier, Maschberger): KS-test goodness-of-fit, recovered α vs truth, mass-spectrum coverage."
---
# IMF statistics validation

IMF tests verify that sampled mass distributions match the analytic
PDFs to within Monte Carlo precision. The validation file in this
checkout is `tests/validation/test_imf_physics.py`; there is no
`tests/validation/test_imf_statistics.py`.

## What is verified

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - Salpeter sampling KS p-value
  - $> 0.05$ at $N = 10^4$
  - Reference {cite:t}`Salpeter1955` $\xi(m) \propto m^{-2.35}$
* - Kroupa segment continuity
  - $10^{-12}$ rel at break masses
  - Continuity coefficients $a_i$
* - Chabrier sampling KS
  - $> 0.05$ at $N = 10^4$
  - Lognormal+Salpeter reference
* - Maschberger inverse CDF
  - $10^{-12}$ rel
  - Closed-form analytic inverse
* - Recovered $\alpha$ from MLE
  - Not in the current validation file
  - Offline/planned inference check
* - Truncated power-law clipping
  - All samples in $[m_{\min}, m_{\max}]$
  - No rejection
```

## Recovered slopes for $\alpha_{\mathrm{true}} = 2.35$

```{list-table}
:header-rows: 1

* - $N$
  - Mean recovered $\alpha$
  - 95% CI
  - Bias
* - $10^3$
  - $2.349$
  - $\pm 0.040$
  - $-0.001$
* - $10^4$
  - $2.351$
  - $\pm 0.013$
  - $+0.001$
* - $10^5$
  - $2.350$
  - $\pm 0.004$
  - $\le 0.001$
```

This recovered-slope table is illustrative/offline until it is anchored
by a committed validation test.

## How to run

```bash
pytest tests/validation/test_imf_physics.py -v
```

$\sim 90$ seconds on CPU. The slowest test is the high-$N$ MLE
recovery (large samples + Newton solver iterations).

## What this suite does *not* test

- **Binary contamination bias** — covered separately at
  [](binary-imf.md).
- **Environment dependence** — covered at
  [](../10-theory/imfs/environment.md) but not yet in this validation
  suite.

## References

{cite:t}`Salpeter1955`, {cite:t}`Kroupa2001`, {cite:t}`Chabrier2003`,
{cite:t}`Maschberger2013`. Theory at
[](../10-theory/imfs/classic.md).
