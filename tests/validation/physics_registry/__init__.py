"""Physics-validation registry (Phase 4 / Task 4.1).

A ratcheting source of truth: every public MODEL (profile / velocity-DF / IMF /
cluster builder / equilibrium engine in ``progenax.__all__``) has its required
physics invariants enumerated and each mapped to the validation test that
asserts it. A new model with no entry reds CI.
"""
