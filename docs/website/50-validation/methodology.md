---
title: Validation methodology
description: progenax's three-tier test architecture, tolerance conventions, and how to add new physics-anchored validation tests.
---
# Validation methodology

```{seealso}
The [validation index](index.md) gives the per-suite roadmap. This
page documents the methodology — the three-tier architecture, the
pass/fail conventions, and the rules for adding new tests.
```

progenax adopts a **three-tier test architecture** that separates
mechanical correctness (unit), end-to-end behaviour (integration),
and physical correctness (validation). Each tier has its own
purpose, frequency, and tolerance convention.

## The three tiers

```{list-table}
:header-rows: 1

* - Tier
  - Purpose
  - Example
* - **Unit** (`tests/unit/`)
  - Per-function mechanical correctness
  - "`PlummerProfile.density(r=1.0)` returns a positive scalar"
* - **Integration** (`tests/integration/`)
  - End-to-end builder/pipeline behaviour
  - "`build_plummer_cluster` produces an `ICResult` with N matching particles, finite kinetic energy, COM at origin"
* - **Validation** (`tests/validation/`)
  - Quantitative match to analytic or published physics
  - "$\zeta(p=1.67) = 1.789$ to $\pm 0.02$ per {cite:t}`Kainulainen2014`"
```

Unit tests run on every push and must complete in $< 5$ minutes total.
Integration tests run on PR merge and must complete in $< 30$ minutes.
Validation tests run nightly (or on PR merge to `main`) and may run
$\sim 4$ hours.

## Tolerance conventions

```{list-table}
:header-rows: 1

* - Test type
  - Default tolerance
  - Rationale
* - Closed-form analytic
  - $1\!\times\!10^{-12}$ relative
  - Float64 precision
* - Finite-$N$ statistical
  - $5\!\times\!10^{-3}$ relative, $N = 10^4$
  - Monte Carlo Poisson noise $\sim 1/\sqrt{N}$
* - Approximation match
  - $5\!\times\!10^{-2}$ relative
  - e.g. cored ζ vs power-law ζ
* - Observational anchor
  - $1\!\times\!10^{-2}$ absolute
  - e.g. Kainulainen+14 ζ ≈ 1.79
```

Tests express these via `pytest.approx(expected, rel=tol)` or
`pytest.approx(expected, abs=tol)`. A regression that exceeds the
declared tolerance blocks merging.

## Anchor patterns

The single most important methodology lesson from progenax's
validation history is **anchor on the defining condition, not the
derived constant**:

```{list-table}
:header-rows: 1

* - Bad anchor
  - Good anchor
* - `assert a == pytest.approx(0.7664)`
  - `assert M_at_r_h / M_total == pytest.approx(0.5, rel=0.01)`
* - `assert zeta_p_one_half == pytest.approx(1.235)`
  - `assert zeta_p_one_half == pytest.approx(_zeta_analytic(0.5), rel=1e-10)`
* - `assert q_vir == 0.5`
  - `assert (T - 0.5 * abs(V)) / abs(V) == pytest.approx(0.0, abs=5e-3)`
```

Bad anchors test internal constants that can be silently mis-derived;
good anchors test the *physical condition* the constant should
satisfy. The PP20 ζ(p) transcription bug shipped past *internal* tests
because those tests asserted the buggy values; it would have been
caught immediately by anchoring on $\zeta(1.5) = \sqrt{2}$.

## Adding a new validation test

The four-step pattern:

1. **Identify the defining condition** — what is the physical or
   mathematical statement that the function being tested is
   *supposed* to satisfy? (E.g. virial theorem $2T + V = 0$.)
2. **Compute the expected value** — analytical (closed form) or
   published (with citation).
3. **Choose a tolerance** based on the convention table above.
4. **Write the test** with `pytest.approx` and a descriptive
   docstring referencing the source.

Example:

```python
def test_plummer_virial_at_equilibrium():
    """Plummer with matched DF must satisfy 2T + V = 0
    (virial theorem) to within Poisson noise.

    Reference: any standard textbook; the closed-form Plummer DF
    in [](../10-theory/velocity-dfs/plummer-dfs.md) is the source of
    truth.
    """
    # Q=None -> the faithful UNSCALED Plummer equilibrium (no virial rescale), so
    # 2T + V = 0 is a genuine test of the matched DF, not a value imposed by the
    # builder. (The default Q=0.5 would force 2T + V = 0 by construction.)
    ic = build_plummer_cluster(n=10_000, r_h=1.0, key=jax.random.PRNGKey(0), Q=None)
    T = compute_kinetic_energy(ic.velocities, ic.masses)
    V = compute_potential_energy(ic.positions, ic.masses, G=STELLAR.G)
    assert (2 * T + V) / abs(V) == pytest.approx(0.0, abs=5e-3)
```

The docstring cites the source of truth; the assertion uses the
defining condition; the tolerance follows the finite-$N$ statistical
convention.

## When validation tests fail

A failing validation test is a *physics* claim that is no longer
true. Three responses, in priority order:

1. **Investigate the change.** Most validation failures are signals
   of real regressions — buggy code, broken refactor, unintended
   parameter change. Find the root cause.
2. **If the test was wrong**, fix the test (with a clear commit
   message explaining the prior error). progenax's PP20 ζ(p)
   transcription bug is the canonical example.
3. **If the physics has changed**, update the test and document the
   change in the dev log.

What is *not* OK: weakening a tolerance to make the test pass without
explaining why. The test exists because the original physics was
correct to that tolerance; if the new code produces a value
$5\times$ further away, the new code is suspect.

## Reproducibility

Every validation test uses a fixed PRNG key and fixed numerical
tolerances. Re-running on the same hardware reproduces results
bit-for-bit. Re-running on different hardware (CPU vs GPU) reproduces
results to float-precision tolerance.

The fixed PRNG key is *deliberate*: it makes regressions reproducible
and bisectable. progenax does *not* run tests with random seeds in CI
because the resulting flakiness obscures real regressions.

## References

The three-tier methodology is standard practice in scientific
software; the specific tolerance conventions follow common practice
in the JAX ecosystem (NumPyro, BlackJAX). The "anchor on defining
condition" lesson came from the 2026-04-28 PP20 ζ(p) bug fix
([](../90-development-log/2026-04-28-pp20-fix.md)), which the
validation suite caught only after the analytic anchor was added.
