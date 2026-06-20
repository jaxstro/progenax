---
title: Gravoturbulent + PP20 validation
description: The PP20 ζ(p) regression suite, BM19 unit coverage, and the historical record of the 2026-04-28 transcription bug fix.
---
# Gravoturbulent + PP20 validation

```{admonition} Experimental — not in the released wheel
:class: warning
The gravoturbulent ζ/BM19 code now lives in the experimental **`gravoturb_fdf`** package
(repo-only, **not** in the released wheel; the old `progenax.gravoturb` and
`tests/unit/physics/` were removed in the 2026-06 rewrite). The anchors below are now
backed by the committed **AC suite** (AC1–AC4) and `tests/experimental/`, not the deleted
`tests/unit/physics/` files.
```

```{seealso}
The 2026-04-28 PP20 transcription bug history is at
[](../90-development-log/2026-04-28-pp20-fix.md). The full theory is
at [](../10-theory/gravoturbulence/magnification-factor.md).
```

Current test + acceptance files:

- `gravoturb_fdf/validation/acceptance.py` — **AC3/AC4** anchor ζ(p) on
  PP20 Eq. 6 + analytic + the direct-field estimator; **AC1/AC2** anchor the
  BM19 scalars + mass conservation. These *print* their numbers (see
  `gravoturb_fdf/VALIDATION_SUMMARY.md`).
- `tests/experimental/unit/test_pp20.py` — PP20 ζ(p) unit tests (anchors,
  no-spurious-pole regression, divergence at $p=2$).
- `tests/experimental/unit/test_bm19.py`, `tests/experimental/unit/test_pdf.py`
  — BM19 scalar + density-PDF/iCDF coverage.
- `tests/experimental/validation/test_acceptance.py` — asserts the AC
  PASS verdicts so "validated" is backed by fresh output.

## What is verified (PP20 ζ(p))

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - $\zeta(p=0) = 1$
  - $10^{-10}$ abs
  - Top-hat reference
* - $\zeta(p=1) = 2 \cdot 2^{3/2}/3^{3/2}$
  - $10^{-12}$ rel
  - Exact analytic
* - $\zeta(p=1.5) = \sqrt{2}$
  - $10^{-12}$ rel
  - Exact analytic
* - $\zeta(p=1.67) \approx 1.789$
  - 0.02 abs
  - {cite:t}`Kainulainen2014` median
* - PP20 Eq. 6 vs analytic at 10 $p$ values
  - 0.2% rel
  - 2.6 ≈ $3^{3/2}/2$ rounding
* - Monotonic in $p$ over $[0, 1.9]$
  - Strict
  - No spurious singularity at $p = 1.3$
* - Smooth across $p = 1.3$
  - $< 1\%$ change
  - Refutes the bug-induced "domain limit"
* - Diverges as $p \to 2$
  - $\zeta(1.95) > 5\,\zeta(1)$
  - Singular isothermal limit
* - $P_{\max} = 1.95$ clipping
  - Identity above $P_{\max}$
  - Numerical safety
* - JIT, vmap, grad compatibility
  - Round-trips
  - JAX-native
* - No buggy form in source tree
  - Source-text scan
  - Regression trap
```

These regression anchors are exercised by `tests/experimental/unit/test_pp20.py`
and printed by AC3/AC4; run the commands below for fresh status.

## What is verified (BM19 unit coverage)

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - $\sigma_s^2 = \ln(1 + b^2 \mathcal{M}^2)$
  - $10^{-12}$ rel
  - Closed form
* - Transition density $s_t$ continuity
  - $10^{-12}$ rel at lognormal-power-law match
  - Analytic matching condition
* - $f_{\mathrm{dense}}$ at canonical params
  - 1% rel
  - Numerical integration cross-check
* - α↔p mapping
  - $p = 3/\alpha$ exactly
  - Definition
* - End-to-end forward chain (1D → 3D realization)
  - AC6 cornerstone: realized vs BM19 $f_{\mathrm{dense}}$, ensemble bias $<1\%$
  - `gravoturb_fdf` AC6 (128³); see its `VALIDATION_SUMMARY.md`
```

## Spot ζ values (canonical anchors)

```{list-table}
:header-rows: 1

* - $p$
  - $\zeta(p)$
  - Comment
* - 0
  - 1.0 (exact)
  - Top-hat
* - 0.5
  - 1.014
  - Pre-2026-04-28 buggy formula returned 1.235
* - 1.0
  - 1.0887
  - Exact analytic anchor
* - 1.5
  - $\sqrt{2}$ ≈ 1.4142
  - Exact analytic anchor
* - 1.67
  - 1.789
  - Kainulainen+14 observational
* - 1.95
  - 8.28
  - Approaching $P_{\max}$
```

## How to run

```bash
# Unit tests (repo-only; needs src/experimental on the path)
PYTHONPATH=src:src/experimental pytest tests/experimental/unit/test_pp20.py -v
PYTHONPATH=src:src/experimental pytest tests/experimental/unit/test_bm19.py tests/experimental/unit/test_pdf.py -v

# Acceptance suite — prints the ζ/BM19 anchor numbers (AC1–AC4)
PYTHONPATH=src:src/experimental python -m gravoturb_fdf.validation.acceptance
```

A few seconds for the unit tests; the full AC suite (incl. the 128³ AC6
cornerstone) takes a couple of minutes.

## The 2026-04-28 fix

The 35 tests were added simultaneously with the bug fix on 2026-04-28.
Pre-fix, the buggy `magnification_factor` returned values like
$\zeta(0.5) = 1.235$ that all *prior* tests accepted (because the
prior tests anchored on the buggy formula's output). The fix
re-anchored every test on either PP20 Eq. 6, the integral-derived
analytic form, or the Kainulainen+14 observation. Full history at
[](../90-development-log/2026-04-28-pp20-fix.md).

## References

{cite:t}`ParmentierPasquali2020` for ζ(p); {cite:t}`Kainulainen2014`
for the observational anchor; {cite:t}`Burkhart2018,BurkhartMocz2019` for
the BM19 framework. Theory at
[](../10-theory/gravoturbulence/magnification-factor.md) and
[](../10-theory/gravoturbulence/bm19.md).
