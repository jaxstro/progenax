# Ticket: implement the full Moe & Di Stefano (2017) two-slope, period-dependent q-distribution

**Opened:** 2026-06-04 (Batch 3b-binary) · **Severity:** Enhancement · **Owner:** Anna ·
**Requested by:** Anna ("ticket the complete Moe model for us to add later").

## What

`progenax.imf.binary.MoeDiStefano2017` currently approximates the mass-ratio distribution with a
**single** power-law slope γ(M₁) (0.4 / 0.3 / 0.0 / −0.5 by mass range) plus a period-independent
twin excess F_twin(M₁). This is a *period-averaged single-slope reduction* of the actual model in
Moe & Di Stefano (2017), ApJS 230, 15 (now documented as such in the class docstring — Batch 3b-binary).

The full model (Moe & Di Stefano 2017, §9, Table 13) is a **three-parameter, period-dependent**
mass-ratio distribution:

```
p_q(q | M1, P) ∝ q^{γ_smallq}              for 0.1 < q < 0.3   (Eq. 2)
              ∝ q^{γ_largeq}              for 0.3 < q < 1.0
              + F_twin excess             for 0.95 < q < 1.0
```

with **all three** of γ_smallq(M₁,P), γ_largeq(M₁,P), F_twin(M₁,P) tabulated as functions of
primary mass AND orbital period (Table 13, at log P = 1, 3, 5, 7). Key verified anchors:

- γ_largeq(log P=1) = −0.5 for all masses; → −2.0 at long P for massive stars.
- γ_smallq(log P=1) ≈ 0.1–0.3; F_twin(log P=1) = {solar 0.30, A 0.22, mid-B 0.17, early-B 0.14, O 0.08}, → <0.03 at long P.

## Why a faithful model matters

The single-slope reduction loses the **P–q interrelation** that is the paper's central result
("Mind your Ps and Qs"): short-period binaries favour twins/equal-q, intermediate-period favour
small q (q≈0.2–0.3), long-period approach random IMF pairings. A binary-population study that
needs realistic (P, q) joint sampling — e.g. compact-binary progenitor rates, RLOF fractions —
requires the full model, not the period-averaged single slope.

## Scope of work (later)

1. **Brainstorm first** (superpowers:brainstorming): the (P, q) joint sampler design — how to
   sample P (from the period distribution) then q | (M₁, P) from the two-slope + twin form, kept
   differentiable (no argmax/argsort; smooth/closed-form inverse where possible).
2. Tabulate γ_smallq / γ_largeq / F_twin from Table 13 (interpolate in log P and M₁); cite Table 13
   precisely (per-paper note moe-distefano-2017.md already records the values).
3. TDD RED→GREEN: recover Moe's marginal q-distribution + the period-dependence in tests; FD-vs-
   autodiff grad-checks on the new sampler entry points.
4. Decide API: a new `MoeDiStefano2017Full` (P-dependent) alongside the current period-averaged
   `MoeDiStefano2017`, or replace it (HITL decision).

## Status

Deferred (enhancement). The current single-slope `MoeDiStefano2017` is documented as an
approximation (Batch 3b-binary); this ticket tracks the faithful replacement.
