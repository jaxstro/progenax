---
title: Validation methodology
description: progenax's three-tier test architecture, tolerance conventions, and how to add new physics-anchored validation tests.
---
# Validation methodology

```{seealso}
The [validation index](index.md) gives the per-suite roadmap. This
page documents the methodology — the three-tier architecture, the
error-budget rules, and the rules for adding new tests.
```

progenax adopts a **three-tier test architecture** that separates
mechanical correctness (unit), end-to-end behavior (integration),
and physical correctness (validation). Each tier has its own
purpose and evidentiary scope.

## The three tiers

```{list-table}
:header-rows: 1

* - Tier
  - Purpose
  - Example
* - **Unit** (`tests/unit/`)
  - Per-function mechanical correctness
  - "The Plummer density shape at $r=a$ equals $2^{-5/2}$"
* - **Integration** (`tests/integration/`)
  - End-to-end builder/pipeline behaviour
  - "`build_plummer_cluster` produces an `ICResult` with N matching particles, finite kinetic energy, COM at origin"
* - **Validation** (`tests/validation/`)
  - Quantitative match to analytic or published physics
  - "$\zeta(p=1.67) = 1.789$ to $\pm 0.02$ per {cite:t}`Kainulainen2014`"
```

Run focused local checks during development and the required integrated checks
before a release claim. The FAST core selector excludes `slow`; the FULL core
selector includes it. Runtime depends on hardware, precision, compilation cache,
static configurations, and test exposure, so a directory name implies no fixed
time budget.

The configured `tests.yml` events are pull requests and manual dispatch;
`physics-validation.yml` uses release tags and manual dispatch. These are
configured triggers, not proof of remote enablement or successful runs.
Private-repository CI requires Anna's explicit permission for the repository
and intended scope before an action triggers it. See the current workflow
owners rather than assuming a nightly or per-push schedule.

## Tolerance conventions

A test's tolerance must follow the quantity and its error source, rather than a
blanket percentage:

```{list-table}
:header-rows: 1

* - Comparison
  - Error budget to establish
* - Analytic expression
  - Floating-point scale, conditioning, cancellation, and any approximation error
* - Finite-sample statistic
  - Observable variance, component/bin occupancy, estimator bias, and seed uncertainty
* - Numerical method
  - Resolution or tolerance refinement against an independent reference
* - Empirical anchor
  - Source uncertainty, units, calibration domain, and the intended discrepancy
```

State the reference and budget before qualification, and retain failures. A
near-zero residual needs the scale that generated its roundoff, not a relative
error divided by that residual. A fixed random key aids reproducibility but
does not replace an ensemble uncertainty or rare-population occupancy check.

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

### Three failure modes the anchor pattern has caught

The defining-condition anchor is not abstract — it has empirically
caught three distinct classes of regression in progenax's history:

1. **Closed-form transcription bugs** — the 2026-04-28 PP20 ζ(p) bug
   ([](../90-development-log/2026-04-28-pp20-fix.md)) was caught by
   anchoring on the analytic value $\zeta(1.5) = \sqrt{2}$; the buggy
   formula gave $\infty$. Without the anchor the bug would have shipped
   unnoticed, because every other test was self-consistent with the
   buggy formula.
2. **Inversion bugs** — the Plummer half-mass-radius bug (see the
   historical note in [](../10-theory/spatial-profiles/plummer.md)),
   where $a = r_h/\sqrt{2^{2/3}-1}$ was used instead of
   $a = r_h\sqrt{2^{2/3}-1}$, was caught by `test_half_mass_radius`
   checking $M(<r_h) = M/2$ *directly* rather than asserting
   $a = $ some constant.
3. **Derivative regressions** — a finite-gradient check can pass a silently
   blocked zero. Compare the intended derivative with an analytic result or
   converged finite differences. A sigmoid changes the model and is not an
   automatic repair for a physical support boundary; see
   [](../20-architecture/differentiability.md).

Each is now anchored by a regression test on the defining condition,
not the derived constant.

## Adding a new validation test

The four-step pattern:

1. **Identify the defining condition** — what is the physical or
   mathematical statement that the function being tested is
   *supposed* to satisfy? (E.g. virial theorem $2T + V = 0$.)
2. **Compute the expected value** — analytical (closed form) or
   published (with citation).
3. **Derive an error budget** from the independent reference, numerical
   conditioning/convergence, and sampling uncertainty.
4. **Write the test** with `pytest.approx` and a descriptive
   docstring referencing the source.

The existing Plummer virial test illustrates an independent continuum anchor:
`TestPlummerVirialEquilibrium.test_virial_ratio` compares raw sampled kinetic
energy with $V=-3\pi GM^2/(32a)$. It does not rescale the sample to force the
answer. Its particular finite-sample tolerance belongs to that test's exposure;
copying the number into a new model would not establish an error budget.

## When validation tests fail

A failed check can indicate a physical disagreement, numerical error, software
defect, statistical fluctuation, or an incorrect test premise. Distinguish these
causes before interpreting the result. Three responses are:

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

Record PRNG keys, configuration, source revision or file hashes, dependency
versions, backend, hardware, precision, and artifacts. A key alone does not
guarantee bitwise reproducibility across compiler/library versions, devices,
or reduction schedules. Recheck numerical and statistical budgets on a new
backend rather than assuming float-precision agreement.

The fixed PRNG key is *deliberate*: it makes regressions reproducible
and bisectable. progenax does *not* run tests with random seeds in CI
because the resulting flakiness obscures real regressions.

## References

The three-tier methodology is standard practice in scientific
software; repository instructions require independent references and
quantity-specific, scale-aware tolerances. The "anchor on defining
condition" lesson came from the 2026-04-28 PP20 ζ(p) bug fix
([](../90-development-log/2026-04-28-pp20-fix.md)), which the
validation suite caught only after the analytic anchor was added.
