---
title: Fractal substructure validation
description: Validation status for substructure tooling — the CW04 Q diagnostic (survives, unit-tested) and the experimental gravoturb_fdf turbulent-density method. The fractal-displacement-field generator was removed in the 2026-06 rewrite.
---
# Fractal substructure validation

```{admonition} Generator removed — diagnostic survives
:class: warning
The differentiable **Fractal Displacement Field generator** was removed in the 2026-06
clean-room rewrite and has **no released successor** (released `generate_cluster_ic` no
longer takes a fractal layer). Its old tests (`tests/unit/cluster/test_fdf.py`,
`tests/unit/cluster/test_fractal.py`) were deleted with it. What remains validated is the
substructure **diagnostic** and a *different* turbulent-IC method (experimental
`gravoturb_fdf`). See the theory chapter [](../10-theory/tidal-and-substructure/fractal.md).
```

## What is currently verified

```{list-table}
:header-rows: 1

* - Property
  - Status
  - Anchor
* - CW04 $Q$ estimator vs Cartwright & Whitworth (2004) Table 1
  - Unit-tested
  - `tests/unit/substructure/test_q_parameter.py`, `test_q_baselines.py`
* - Differentiable kNN $Q$ approximation
  - Unit-tested
  - `tests/unit/substructure/test_q_approx.py`
* - Turbulent-density ICs: BM19 cornerstone + $Q(f_{\mathrm{sub}})$ calibration
  - Experimental, AC-backed
  - `src/experimental/gravoturb_fdf/` (AC6, AC7 — see its `VALIDATION_SUMMARY.md`)
```

The released CW04 $Q$ estimator uses the CW04 area convention
$A = \pi R^2$ (reproduces Table 1 to $<0.01$; convex-hull area biases $Q$
high by $\sim+0.1$). The experimental `gravoturb_fdf` ships an
equivalence-pinned copy of the same estimator. The $Q(f_{\mathrm{sub}})$
substructure trend (more dense-tail stars → lower $Q$, in the direction of
the GW04 fractal-dimension ladder) is measured with realization bands in
that package's AC7, not tuned to a target.

## How to run current checks

```bash
pytest tests/unit/substructure/test_q_parameter.py -v
pytest tests/unit/substructure/test_q_baselines.py -v
pytest tests/unit/substructure/test_q_approx.py -v

# Experimental gravoturb_fdf acceptance (repo-only; needs src/experimental on the path):
PYTHONPATH=src:src/experimental python -m gravoturb_fdf.validation.acceptance
```

## References

Theory at [](../10-theory/tidal-and-substructure/fractal.md) and
[](../20-architecture/jax-native-substructure-q.md). The GW04 baseline is
{cite:t}`Goodwin2004`; the Q diagnostic is {cite:t}`Cartwright2004`; the
dynamical context is {cite:t}`Allison2009`.
