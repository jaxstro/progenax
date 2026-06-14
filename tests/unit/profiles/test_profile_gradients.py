"""FD-vs-autodiff gradient checks on profile samplers — MIGRATED to the grad-audit registry.

Every AD-vs-FD sampler-gradient assertion that used to live here is now owned by the
grad-audit registry (the single source of truth for gradient correctness):

    tests/validation/grad_audit/registry.py ::
        PlummerProfile.sample_positions [r_h]
        EFFProfile.sample_positions      [gamma, r_t, a]
        KingProfile.sample_positions     [r_c, W0]

The registry cases are FD-matched with measured AD/FD ratios and explicit tolerances
(equal or stronger than the checks removed here). See
docs/website/50-validation/differentiability-audit.md. (audit T6 consolidation; registry is SoT)

The final holdout — EFFProfile.sample_positions(a) — was migrated once the registry gained the
matching `EFFProfile.sample_positions` `a` case (Task 4.2b review-fix); the safety interlock had
kept it here until then.
"""
