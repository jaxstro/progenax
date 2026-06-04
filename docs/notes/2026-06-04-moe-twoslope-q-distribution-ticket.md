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

## Table 13 — transcribed VERBATIM from the PDF (p. 52), verified 2026-06-04 (Batch 4i Gate-1)

Mass bins: Solar 0.8–1.2 · A/late-B 2–5 · Mid-B 5–9 · Early-B 9–16 · O >16 M☉.
Columns below in that order. "<0.03" = published upper limit.

**γ_largeq (q=0.3–1.0):**
```
logP=1:  -0.5  -0.5  -0.5  -0.5  -0.5
logP=3:  -0.5  -0.9  -1.7  -1.7  -1.7
logP=5:  -0.5  -1.4  -2.0  -2.0  -2.0
logP=7:  -1.1  -2.0  -2.0  -2.0  -2.0
```
**γ_smallq (q=0.1–0.3):**
```
logP=1:   0.3   0.2   0.1   0.1   0.1
logP=3:   0.3   0.1  -0.2  -0.2  -0.2
logP=5:   0.3  -0.5  -1.2  -1.2  -1.2
logP=7:   0.3  -1.0  -1.5  -1.5  -1.5
```
**F_twin (q>0.95):**
```
logP=1:   0.30  0.22  0.17  0.14  0.08
logP=3:   0.20  0.10  <0.03 <0.03 <0.03
logP=5:   0.10  <0.03 <0.03 <0.03 <0.03
logP=7:   <0.03 <0.03 <0.03 <0.03 <0.03
```
**Companion frequency f_logP;q>0.1 (per dex of logP) — the M1-dependent period dist shape:**
```
logP=1 (0.5–1.5):  0.027  0.07  0.14  0.19  0.29
logP=3 (2.5–3.5):  0.057  0.12  0.22  0.26  0.32
logP=5 (4.5–5.5):  0.095  0.13  0.20  0.23  0.30
logP=7 (6.5–7.5):  0.075  0.09  0.11  0.13  0.18
```
**η (e-slope, e=0–0.8 e_max):** logP=2: 0.1 0.3 0.6 0.7 0.7 ; logP=4: 0.4 0.5 0.7 0.8 0.8.

**Cross-checks (pass):** (1) the pre-existing single-slope-reduction anchors match the real
table (NOT fabricated). (2) The faithful `MoeEccentricity` (Eqs 17–18, Batch 4d) reproduces
Table 13's η: Eq17@logP=2→0.13≈0.1, Eq18(O)@logP=4→0.84≈0.8 — so the **e-axis is DONE**; 4i
needs only the q-axis + period distribution + joint sampling. The model form is Eq. 2:
p_q ∝ q^γsmallq (0.1<q<0.3) then q^γlargeq (0.3<q<1.0) + F_twin excess (0.95<q<1.0),
continuous at q=0.3, normalized on [0.1,1].

## Locked design decisions (Anna, 2026-06-04 Gate-1)

- **Interpolation:** bilinear in (log10 M1, logP), clamped outside the grid.
- **Mass-bin nodes:** Solar→1.0, A/late-B→3.2, Mid-B→6.7, Early-B→12, O→20 Msun.
- **logP nodes:** {1,3,5,7} for γ/F_twin/freq; clamp at edges.
- **"<0.03" F_twin cells → 0** (twins vanish at long P; using the 0.03 upper limit would overstate).
- **Period distribution:** piecewise-linear interp of the f_logP;q>0.1 anchors in logP, normalized
  over logP ∈ [0.2, 8.0], clamped at edges (the M1-dependent period sampler).
- **API: ADD alongside (do NOT replace, do NOT change BinaryIMF default):**
  `MoeDiStefano2017Full` (two-slope+twin q | M1,logP), `MoePeriod(M1)` (period dist),
  `MoeJointOrbit(M1,key) -> (P,q,e)` (joint sampler; e via the existing MoeEccentricity, 4d).
  Keep the period-averaged `MoeDiStefano2017` as the documented fast approximation.

## Implementation plan (TDD, when Anna gives the go)

1. `MoeDiStefano2017Full`: Table-13 lookup (bilinear) -> γsmallq, γlargeq, F_twin; sample the
   continuous piecewise power law (q^γsmallq on [0.1,0.3], q^γlargeq on [0.3,1.0], matched at
   q=0.3, normalized on [0.1,1]) + F_twin twin excess on [0.95,1.0]. Differentiable inverse-CDF
   (double-where at γ=−1, mirroring SanaOBPeriod). pdf/cdf for protocol conformance.
2. `MoePeriod(M1)`: normalized piecewise-linear f(logP) -> sample logP -> P[days].
3. `MoeJointOrbit`: logP~MoePeriod; q~Full(M1,logP); e~MoeEccentricity(P,M1).
4. RED tests: recover each Table-13 cell (γ/F_twin/freq) at the grid points; q-distribution
   moments vs the slopes; the P–q correlation (short-P -> twin/equal-q excess; long-P -> small q
   approaching random IMF pairings); FD-vs-autodiff grad-checks on every sampler entry point;
   protocol conformance. Full suite green both jax envs; commit + push.

## Status

**Gate-1 COMPLETE** (2026-06-04): Table 13 transcribed verbatim + verified against PDF p.52;
all design decisions locked above. **Implementation PENDING Anna's verification of the grid +
explicit go.** The single-slope `MoeDiStefano2017` remains the documented approximation until then.
