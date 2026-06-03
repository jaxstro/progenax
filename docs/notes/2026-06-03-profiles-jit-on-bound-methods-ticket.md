# Ticket: `@jax.jit` on bound `_sample_radii` methods (profiles)

**Opened:** 2026-06-03 · **Status:** 🔵 OPEN (Minor, maintainability) · **Source:** Batch 1
profiles SoTA review (finding P5c)

## Issue

`PlummerProfile._sample_radii` ([plummer.py:92](../../src/progenax/profiles/plummer.py)) and
`UniformSphereProfile._sample_radii` ([uniform.py:97](../../src/progenax/profiles/uniform.py))
are decorated with `@jax.jit` on the **bound method**. `self` (an Equinox PyTree) then becomes
a traced argument: the method re-traces per instance, and it would break if a profile ever held
a non-array static field. King and EFF correctly do **not** decorate their `_sample_radii`.

This is not a correctness bug (Plummer/Uniform are pure-array PyTrees, so it works today) — it
is an inconsistent, slightly fragile pattern.

## Fix (when scheduled)

Drop the `@jax.jit` decorator from the bound `_sample_radii` methods (let the caller jit the
public `sample_positions` if desired), making all four profiles consistent. Verify the
profile sampling tests + the FD-vs-autodiff grad-checks
([test_profile_gradients.py](../../tests/unit/profiles/test_profile_gradients.py)) still pass.

## Also deferred from the same review (Minor, P5d)

- `test_king.py` / `test_eff.py` `test_jit_compatible` assert only `isfinite` — tighten to
  assert jit-matches-eager (the pattern already used in `test_king_grad.py`).
- Two near-duplicate King Table-II concentration tests (`test_king_physics.py` in-class +
  module-level) — consolidate.
- King CDF-quadrature tolerance in `test_cdf_quadrature.py` could tighten from 1e-4 toward the
  achievable ~1e-6 (King).
