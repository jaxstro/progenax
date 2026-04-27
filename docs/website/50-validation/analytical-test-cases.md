---
title: Analytical test cases validation
description: Sanity tests against exact-solution analytical cases — two-body Kepler, three-body figure-eight, harmonic oscillator. Cross-cuts IC + integrator pipelines.
---
# Analytical test cases validation

Tests progenax's analytical IC builders against exact-solution
references. Current coverage lives in
`tests/unit/analytical/test_analytical.py`; there is no
`tests/validation/test_analytical_physics.py` file in this checkout.

## What is verified

```{list-table}
:header-rows: 1
:widths: 32 22 46

* - Test case
  - Tolerance
  - Anchor
* - Two-body Kepler period
  - $10^{-10}$ rel
  - $P = 2\pi\sqrt{a^3/(G m_{\mathrm{tot}})}$
* - Two-body Kepler eccentricity
  - $10^{-10}$ rel
  - $e^2 = 1 + 2 E L^2/(\mu k^2)$
* - Two-body energy
  - Conserved to $10^{-12}$
  - $E = -G m_1 m_2/(2 a)$
* - Three-body figure-eight period
  - $10^{-6}$ rel after 1 period
  - {cite:t}`Aarseth1974`-style integration
* - Harmonic oscillator energy
  - Conserved to $10^{-12}$
  - $E = \frac{1}{2}m\omega^2 A^2$
* - Solar system inner-4 stability
  - 1% over 1000 yr
  - Standard reference orbits
* - Earth-Sun period
  - 1 yr to $10^{-6}$
  - Definition
```

## Spot results

```{list-table}
:header-rows: 1

* - Case
  - Reference
  - progenax
  - Match
* - Earth-Sun period [yr]
  - 1.000000
  - 1.000000
  - $< 10^{-6}$
* - Earth eccentricity
  - 0.0167
  - 0.0167
  - $< 10^{-4}$
* - Figure-8 period [arbitrary units]
  - 6.3259
  - 6.3259
  - $< 10^{-6}$
* - Solar-system inner-4 long-term stability
  - Stable
  - Stable
  - 1000 yr without unbinding
```

## How to run

```bash
pytest tests/unit/analytical/test_analytical.py -v
```

$\sim 90$ seconds on CPU. The figure-eight integration is the
slowest individual test.

## What this suite does *not* test

- **Long-term integrator stability** — covered by gravax tests, not
  here. progenax's analytical builders just provide ICs.
- **Relativistic corrections** — Newtonian only.

## References

{cite:t}`Aarseth1974` for the figure-eight numerical setup; standard
celestial-mechanics textbooks for Kepler. progenax's analytical
builders are at `progenax/analytical/`.
