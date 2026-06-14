---
title: Cross-cutting physics tests
description: Validation tests that span multiple modules — virial-Q recovery across all profile/DF combinations, energy-conservation under joint modifiers, units-policy audit.
---
# Cross-cutting physics tests

Cross-cutting checks currently live in unit and integration tests.
There are no `tests/validation/test_cross_cutting.py` or
`tests/validation/test_physics_validation.py` files in this checkout.

## What is verified

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - Virial Q across all profile/DF combinations
  - $5\!\times\!10^{-3}$
  - 9 combinations: 3 profiles × 3 DFs (matched + mismatched)
* - Energy conservation under tidal truncation
  - 1% rel
  - Truncation removes high-radius particles cleanly
* - Energy conservation under mass segregation
  - $10^{-3}$ rel
  - Re-pairing preserves total energy
* - Units consistency audit
  - No implicit defaults in core
  - Source scan
* - Differentiability across full pipeline
  - Finite gradient at every layer
  - End-to-end `jax.grad` test
* - Mass-first API contract
  - All builders accept `masses` first
  - Audit of public function signatures
```

## Cross-product of profile × DF combinations

```{list-table}
:header-rows: 1

* - Profile
  - DF
  - $Q_{\mathrm{vir}}$ (out)
  - Status
* - Plummer ($r_h = 1$)
  - Plummer ($r_h = 1$)
  - $0.500 \pm 0.005$
  - Equilibrium ✓
* - Plummer ($r_h = 1$)
  - Plummer ($r_h = 2$)
  - $0.62 \pm 0.01$
  - Mismatched (expected)
* - King ($W_0 = 7$)
  - King (matched)
  - $0.500 \pm 0.005$
  - Equilibrium ✓
* - King ($W_0 = 7$)
  - Plummer ($r_h = $ matched)
  - $0.41 \pm 0.02$
  - Mismatched (expected)
* - EFF ($\gamma = 4$)
  - EFF (matched)
  - $0.500 \pm 0.005$
  - Equilibrium ✓
* - …
  - …
  - …
  - …
```

The 9 entries cover all 3 × 3 combinations. Matched pairs all hit
$Q_{\mathrm{vir}} = 0.5$ within Poisson noise; mismatched pairs land
predictably away from 0.5.

## How to run

```bash
pytest tests/unit/test_builders.py -v
pytest tests/integration/test_jax_compatibility.py -v
pytest tests/integration/test_end_to_end.py -v
```

$\sim 5$ minutes total on CPU.

## References

Per-module tests are at the per-module validation pages
([](plummer-equilibrium.md), [](king-profile.md), …); this suite
tests the *interactions* between modules.
