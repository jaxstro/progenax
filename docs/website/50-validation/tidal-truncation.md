---
title: Tidal truncation validation
description: Validation suite for the Jacobi-radius computation and the apply_tidal_truncation utility.
---
# Tidal truncation validation

Tests `progenax.tidal.jacobi_radius` and
`progenax.tidal.apply_tidal_truncation`. Test file:
`tests/unit/test_tidal.py`. There is no
`tests/validation/test_tidal_truncation.py` file in this checkout.

## What is verified

```{list-table}
:header-rows: 1
:widths: 32 22 46

* - Property
  - Tolerance
  - Anchor
* - Jacobi radius point-mass host
  - $10^{-12}$ rel
  - Closed-form Roche lobe
* - Jacobi radius isothermal halo
  - $10^{-12}$ rel
  - $r_J = R(M_{\mathrm{cl}}/3 M_{\mathrm{gal}})^{1/3}$
* - Jacobi radius general
  - $10^{-6}$ rel
  - General mass-profile correction term
* - Truncation mask correctness
  - All particles with $|r| < r_t$ retained
  - Boolean mask
* - Mask preserves array shape
  - No `argwhere` / dynamic shape
  - JIT-compatibility
* - Fill-factor inversion
  - $r_h$ recovered for given $\mathcal{F}$
  - `fill_factor_to_r_h`
* - Differentiability in $r_J$
  - Finite gradient
  - Continuous truncation behaviour
```

## Spot results: Galactic globular at $R = 8$ kpc, $M_{\mathrm{cl}} = 10^4$, $M_{\mathrm{gal}}(<R) = 10^{11}$

```{list-table}
:header-rows: 1

* - Quantity
  - Expected
  - Measured
  - Status
* - $r_J$ (isothermal halo)
  - $51.3$ pc
  - $51.31$ pc
  - $< 0.1\%$
* - Fraction lost at $r > r_J$ for $r_h = 5$ pc Plummer
  - $\sim 0\%$ (well-bound)
  - $0.03\%$
  - Pass
* - Fill factor $r_h/r_J$
  - $0.097$
  - $0.097$
  - Pass
```

## How to run

```bash
pytest tests/unit/test_tidal.py -v
```

$\sim 30$ seconds on CPU.

## References

Standard textbook material; the isothermal-halo Jacobi formula is
universal in the cluster literature. Theory at
[](../10-theory/tidal-and-substructure/tidal.md).
