---
title: Fractal substructure validation
description: Current validation status for the FDF fractal-substructure implementation.
---
# Fractal substructure validation

```{important}
Status: **unit/offline-backed, not a dedicated validation suite yet**.

There is no `tests/validation/test_fractal_substructure.py` file in
this checkout. The current implementation is covered by unit tests under
`tests/unit/cluster/` and `tests/unit/substructure/`.
```

## What is currently verified

```{list-table}
:header-rows: 1

* - Property
  - Status
  - Anchor
* - FDF displacement helpers and generated IC shapes
  - Unit-tested
  - `tests/unit/cluster/test_fdf.py`
* - Legacy/fractal cluster behavior
  - Unit-tested
  - `tests/unit/cluster/test_fractal.py`
* - CW04-style Q diagnostics
  - Unit-tested
  - `tests/unit/substructure/test_q_parameter.py`
* - Q baseline comparisons
  - Unit-tested
  - `tests/unit/substructure/test_q_baselines.py`
* - Published calibration table for GW04 `D` vs FDF `chi`
  - Offline/illustrative
  - Needs a committed validation script before being treated as CI evidence
```

## Calibration table status

The table in the theory chapter is an approximate calibration guide.
It should not be read as fresh output from a
`tests/validation/test_fractal_substructure.py` file until that file is
added and the outputs are regenerated.

## How to run current checks

```bash
pytest tests/unit/cluster/test_fdf.py -v
pytest tests/unit/cluster/test_fractal.py -v
pytest tests/unit/substructure/test_q_parameter.py -v
pytest tests/unit/substructure/test_q_baselines.py -v
```

## References

Theory at [](../10-theory/tidal-and-substructure/fractal.md). The GW04
baseline is {cite:t}`Goodwin2004`; the Q diagnostic is
{cite:t}`Cartwright2004`; the dynamical context is {cite:t}`Allison2009`.
