# Ticket: accelerated neighbour-list bound-pair / multiple finder → gravax

**Opened:** 2026-06-04 (Batch 4j) · **Severity:** Enhancement · **Owner:** Anna ·
**Home package:** **gravax** (not progenax).

## What

`progenax.binaries.diagnostics.find_bound_pairs` / `find_bound_multiples` materialize the
full N×N separation matrix (`positions[:,None,:] - positions[None,:,:]`) to find
mutual-nearest-neighbour bound pairs. This is **O(N²)** in time and memory — fine for the
N ≲ few×10³ snapshots progenax IC generation produces, but it does not scale to the
N = 10⁴–10⁶ dynamical snapshots a gravax run produces.

## Why it belongs in gravax (not progenax)

1. The accelerated version needs a **spatial neighbour structure** (cell list / kd-tree /
   octree). gravax **already owns** this machinery — the Hermite Ahmad–Cohen integrator
   maintains neighbour lists. Reusing it avoids duplicating a non-trivial data structure.
2. The acceleration is inherently **non-differentiable** (argsort / spatial binning), so it
   does not belong on progenax's differentiable IC-generation path. It is a *measurement*
   on an evolved snapshot — gravax's domain.
3. progenax's eager O(N²) finder is correct and adequate for *initial conditions* (the
   primordial pairing is known at t=0 via `primordial_system_id`; the finder is mainly for
   validation and small clusters). The scaling need arises only after dynamical evolution.

## Scope of work (in gravax, later)

1. Reuse the Hermite-AC neighbour list (or add a standalone cell list) to restrict
   candidate pairs to spatial neighbours → O(N) / O(N log N).
2. Keep the **same physics**: mutual-nearest-neighbour + `E_rel < 0` (NBODY/kira / Aarseth
   2003 criterion). `relative_energy` (the differentiable two-body primitive) is reusable
   as-is.
3. Hierarchical multiples: the fixed-depth `lax.scan` COM-collapse pattern from
   `find_bound_multiples` carries over; only the per-level neighbour search changes.
4. Bench against the progenax O(N²) finder on small N (must agree exactly) before scaling up.

## Related

- `binary_energy_budget` (Batch 4j) is **differentiable** and O(N) (system_id segment sums
  + structural pairing) — it does *not* need this; it is the t=0 internal-energy budget,
  not a neighbour search.
- gravax hardening survey (2026-06-04): gravax is under-hardened (45–68%); this is one of
  several deferred gravax items (also: `from_ic` missing, softening_policy ignored by
  integrators, kepler.py duplicated/diverged, `while_loop` in chunked_scan, under-testing).

## Status

**OPEN** — deferred to the gravax hardening pass ("fix gravax later", Anna 2026-06-04).
