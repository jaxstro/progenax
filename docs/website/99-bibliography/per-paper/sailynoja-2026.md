---
title: Säilynoja et al. (2026)
description: Annotated reference for Säilynoja, Schmitt, Bürkner & Vehtari — "Posterior SBC", calibration checking conditional on the observed data; a future-direction extension of the prior-SBC check gravoturb currently uses.
---

# Säilynoja et al. (2026)

```{admonition} Posterior SBC: simulation-based calibration checking conditional on data
:class: note

**Authors.** Teemu Säilynoja, Marvin Schmitt, Paul-Christian Bürkner, Aki Vehtari

**Reference.** Statistics and Computing **36**, 78 (2026).

**DOI.** [10.1007/s11222-026-10825-9](https://doi.org/10.1007/s11222-026-10825-9)

**Verified.** Abstract, §1–§3 (Eqs. 1–3, Algorithms 1–2), Figs. 1–2 checked against the held
PDF (2026-06). The fact `gravoturb` cares about: ordinary ("prior") SBC validates
calibration *averaged over the prior* and can miss miscalibration confined to the small region
of parameter space relevant for a *given* dataset (their Fig. 1); **posterior SBC** instead
draws `θ' ~ π(θ|y_obs)`, simulates `y_i ~ π(y|θ'_i)`, refits, and tests PIT uniformity — a
calibration check focused on the data at hand, valid even under misspecification (their Eq. 3).
```

## The big idea

Prior SBC ({cite:t}`Talts2018`) checks that an inference algorithm is calibrated *across the
whole prior*. But the prior can place most of its mass where inference is easy while a small
problematic region — the one that actually matters for a specific observed dataset — is washed
out in the prior average (their Fig. 1: two canceling biases that prior SBC cannot see).
**Posterior SBC** conditions the whole calibration check on the observed data `y_obs`: it draws
ground truths from the *posterior* `π(θ|y_obs)`, generates data from them, refits, and tests the
PIT values for uniformity. Using the sequential-Bayesian-update view (their Eq. 3), this is the
right check when you care about trusting the inference *on the dataset you actually have*, and it
remains valid under model misspecification. It is operationalized for both MCMC and amortized
(neural) Bayesian inference, where it serves as a cheap default trustworthiness diagnostic.

## Use in progenax

Not yet used — recorded as a **future direction**. The current `gravoturb` AC18 is *prior*
SBC (calibration over the BM19 prior); its documented honest-scope caveat ("validates the engine
under the model, averaged over the prior") is exactly the limitation posterior SBC addresses. A
natural follow-up beyond workstream ① would add a posterior-SBC check conditioned on a specific
mock cluster's data — more honest for "trust the inferred (ℳ, α, β) for *this* cluster." This
paper *uses* the {cite:t}`Sailynoja2022` ECDF-difference graphical test (via `bayesplot`); it does
**not** re-derive the simultaneous-band construction, so the F2.5 figure is grounded in the 2022
paper, not this one.

## Notes

- Code for the paper's experiments: `github.com/TeemuSailynoja/posterior-sbc` (R / `bayesplot`;
  cross-check only — the `gravoturb` figures use the `arviz` array layer).
- Posterior SBC is still **embarrassingly parallel** (independent refits) and inherits SBC's
  honest scope: it certifies *self-consistency of the inference*, not that the generative model
  matches reality.
