# Ticket: analytical integrator-validation oracles → gravax hardening pass

**Opened:** 2026-06-04 (Batch 5 follow-up) · **Severity:** Enhancement · **Owner:** Anna ·
**Home:** oracles + IC-level correctness in **progenax** `analytical/`; the integrate-and-check
tests in **gravax** (where the integrators live). Anna 2026-06-04: "those were initially
designed for gravax, so lean towards that" → ticket all of it for the gravax pass.

## Context

`progenax.analytical` supplies *exact-solution initial conditions*; gravax integrates them and
checks conservation / closure / symplecticity. Batch 5 fixed the figure-eight (it was wrong)
and added IC-level self-validation. The oracles below each exercise an integrator behaviour the
current set does **not** cover. Per the split: progenax adds the oracle + an IC-level correctness
test (no integrator needed); gravax adds the integration test in its hardening pass.

**Grounding note:** the exact constants/benchmark digits below must be derived or read from the
primary source at implementation time (no-assumptions rule) — they are stated here as the design
target, not verified values. Reference PDFs to fetch: Szebehely & Peters (1967) for Pythagorean;
Murray & Dermott (1999) / Goldstein for the LRL vector and Lagrange central configuration.

## Recommended additions (priority order)

### 1. Eccentricity-vector / Laplace–Runge–Lenz conservation (two-body) — HIGHEST VALUE

The classic discriminator of **symplectic vs non-symplectic** integration. The eccentricity
(LRL) vector `e_vec = (v × L)/(G M_tot) − r̂` is conserved for an exact 1/r² force; non-symplectic
schemes show spurious **perihelion precession** (the apsidal angle drifts).
- **progenax (IC-level):** assert `two_body_kepler`'s eccentricity vector has magnitude `e` and
  points toward perihelion (along the apsidal line set by `omega`).
- **gravax (integration):** integrate an eccentric orbit for many periods; confirm
  `|Δe_vec|`/`e` and the apsidal-angle drift stay below tolerance for symplectic integrators
  (PEFRL/Yoshida/IAS15) and visibly precess for a non-symplectic baseline (RK4) — a regression
  that proves the symplectic property.

### 2. Lagrange equilateral-triangle 3-body (`three_body_lagrange_triangle`)

Exact central configuration: three equal masses `m` at the vertices of an equilateral triangle
(side `s`), **rigidly rotating** about the centroid. Design target (verify at impl):
`ω² = G·M_tot / s³` (M_tot = 3m), period `T = 2π√(s³/(G M_tot))`; circular speed `v = ω·s/√3`.
- **progenax (IC-level):** equal masses; centroid at origin; mutual distances all `s`; total
  angular momentum L ≠ 0 (complements the figure-eight's L=0).
- **gravax (integration):** integrate one rotation period; mutual distances stay constant (rigid
  rotation) and the configuration returns to start — a clean rigid-rotation integrator check.

### 3. Pythagorean / Burrau 3-body (`pythagorean_three_body`)

Masses 3, 4, 5 at the vertices of a 3-4-5 right triangle, released from **rest** (G=1) — the
canonical **chaotic close-encounter stress test** (Szebehely & Peters 1967, AJ 72, 876; documented
series of close approaches and a near-ejection). No closed form.
- **progenax (IC-level):** the documented masses/positions; total momentum = 0; total energy
  finite and negative.
- **gravax (integration):** stress the adaptive-timestep / regularization machinery (Hermite,
  IAS15) through the close encounters; require energy conservation `|ΔE/E|` below tolerance and
  qualitative agreement with the published evolution. The best torture test for close-encounter
  handling.

### 4. Two-body vis-viva + apsides + analytic L (cheap IC-level, progenax)

Pin the `two_body_kepler` construction directly (no integrator):
- perihelion (ν=0): `v = √(G M_tot/a · (1+e)/(1−e))`, `r = a(1−e)`; aphelion: `r = a(1+e)`.
- angular-momentum **value** `L = μ √(G M_tot a (1−e²))` (we currently test only that L is
  *conserved*, not that it equals the analytic value).

### 5. Hyperbolic / parabolic two-body (e ≥ 1)

Generalize `two_body_kepler` for unbound orbits (`p = a(1−e²)`, `a < 0` for hyperbolae) →
scattering / flyby integrator tests for gravax close-encounter and unbound-orbit validation.

## Lower priority / out of scope

- Circular restricted 3-body (CR3BP) + Jacobi-constant conservation (more involved).
- Euler collinear 3-body central configuration (niche).
- `harmonic_oscillator`: needs external-force support most N-body integrators lack (already a
  documented placeholder in `analytical/few_body.py`).

## Status

**OPEN** — deferred to the gravax hardening pass (sits alongside the other gravax items:
`from_ic`, softening-policy wiring, kepler dedup, the accelerated bound-finder
[2026-06-04-accelerated-bound-finder-gravax-ticket.md], and under-testing). progenax `analytical/`
is left as-is (Batch 5 complete @ 49a897f).
