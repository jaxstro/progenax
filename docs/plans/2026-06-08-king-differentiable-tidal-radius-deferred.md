# Differentiable King tidal radius (∂r_t/∂W₀) — RESOLVED (Approach B, IFT)

**Status:** RESOLVED (2026-06-13, audit Task 1.2b). The unclamped-ψ root-find
recommended below (the "fix must keep the density safe while recovering the
crossing slope" line) is now implemented for **both** King and Michie.
`solve_king_profile` and `solve_michie_profile` now unconditionally return the
3-tuple `(xi_grid, psi_clamped, psi_raw)`, where `psi_raw` is the UNCLAMPED ODE
solution (`solution.ys[:,0]`, negative past the crossing). `KingProfile.from_W0_rc`,
`MichieProfile.from_W0_rc`, and `MichieVelocityDF` feed that raw ψ to
`_find_tidal_radius`, so ∂r_t/∂W₀ now flows through the diffrax solve (the IFT
result to grid accuracy). The density/CDF/mu/virial paths and `KingVelocityDF`
keep using `psi_clamped`. There is **no backward-compat shim/flag** (house rule);
all callsites were updated directly to unpack the 3-tuple (those that don't need
`psi_raw` discard it with `_`). Validated AD-vs-FD: King W₀=8 AD≈48,
FD≈48 (ratio within ~1e-3), Michie W₀=8/r_a=10 ratio ≈1.00002; grad-audit case
`KingProfile.r_t` (theta0=8.0) now classifies `consistent`.

**Forward-value caveat (the one place this doc was wrong):** the original header
claimed the forward `r_t` value is *unchanged* by Approach B — it is NOT. The
pre-fix clamped path SNAPPED ξ_t to the grid node (ψ₁ clamped to 0 → t=1); the fix
interpolates the true crossing (ψ₁<0 → t<1). At W₀=8 (xi_max=400,n=8000) r_t
shifts 68.15852 → 68.14678 (~6e-4 relative, ~23% of one grid cell) — far below
ODE/grid accuracy, and the unclamped interpolation is the *more accurate* crossing,
not a regression. The unit regression test pins the new (interpolated) values:
`tests/unit/profiles/test_king.py::TestDifferentiableTidalRadius`.

---

_Original deferral note (2026-06-08) preserved below for context._

**Status (original):** DEFERRED (2026-06-08). Decision: keep the current
`argmax`-based tidal-radius finder; implement the differentiable version (Approach
B below) **when a scalar-`r_t` use case becomes concrete** — most likely the
tidal-field / Jacobi-radius coupling, which Anna plans to integrate (progenax
already has the tidal machinery: `progenax.tidal.jacobi_radius`,
`apply_tidal_truncation`).

This was framed as a **non-breaking add-later**; in practice the forward value
shifts marginally (see caveat above), but no caller or test relied on the snapped
value, so the change is safe.

## What is and isn't differentiable today (verified 2026-06-08)

- **Differentiable (machine precision):** `r_c`, `M_tot`, and **`W₀` for any
  density/velocity *shape* observable** — `diffrax` propagates ∂ψ/∂W₀ through the
  ODE solve (validated AD-vs-FD: r_c 2e-10, W₀ 2e-7, M 1.6e-6; Fig 5).
- **Blocked:** the **scalar tidal radius `r_t`** w.r.t. W₀ — `∂r_t/∂W₀ = 0`
  identically.

### Why `r_t` is blocked (mechanism, verified)

`solve_king_profile` ends with `psi_grid = jnp.maximum(psi_grid, 0.0)` (the clamp
that keeps the lowered-Maxwellian density gradient-safe at ψ=0). At the first
crossing node this forces `ψ₁ = 0` exactly, so the linear-interpolation weight
`t = ψ₀/(ψ₀ − ψ₁) = 1`, and `ξ_t` snaps to the fixed grid node
`xi_grid[argmax(ψ≤0)]`. That node is (i) a constant `linspace` value and (ii)
selected by `argmax` (integer, zero gradient) → `∂ξ_t/∂W₀ = 0`, stair-stepping as
W₀ moves the crossing across a cell. (Confirmed: ψ₁ = 0.0, t = 1.0, ξ_t = grid node.)

Note the tension: the *same* clamp that makes the **density** gradient-safe is what
kills the **tidal-radius** gradient. The fix must keep the density safe while
recovering the crossing slope (use a raw, unclamped ψ for the root-find only).

## Why deferred

The stated near-term goal — *fit King profiles, infer structural parameters from
data* — is **already covered without ∂r_t/∂W₀**: you fit the profile *shape*
(Σ(r), σ(r)), which is differentiable in W₀; the concentration
`c = log₁₀(r_t/r_c)` and `r_t` are then deterministic functions of the inferred
W₀ that you read off the posterior. `∂r_t/∂W₀` is only needed when `r_t` (or `c`)
appears on the **left** of a likelihood/prior — i.e. as an *input* to the loss,
not an *output* you report.

## Implementation plan (Approach B — implicit function theorem)

`ξ_t` is implicitly defined by `ψ(ξ_t, W₀) = 0`. By the IFT:

$$\frac{\partial \xi_t}{\partial W_0} = -\frac{\partial\psi/\partial W_0}{\partial\psi/\partial\xi}\Bigg|_{\xi_t}.$$

- The denominator `∂ψ/∂ξ` at ξ_t is the ODE's **second state `y[1]`** (already
  produced by the solve) — no extra work.
- The numerator `∂ψ/∂W₀` at ξ_t comes from differentiating the solve (forward-mode).
- Wrap `_find_tidal_radius` (or a thin `tidal_radius(W₀)`) in `jax.custom_jvp`:
  compute the forward `ξ_t` however is robust (current argmax+interp is fine for
  the *value*), and supply the IFT expression as the JVP. Result: exact, **smooth**
  `∂r_t/∂W₀` with no grid-jump artifacts, forward value unchanged.

**Alternatives considered:**
- **A (cheap):** unclamped bracketed linear interpolation — keep argmax to locate
  the cell, interpolate with raw ψ straddling zero (ψ₀>0, ψ₁<0) so `t∈(0,1)`.
  Correct gradient *almost everywhere*; tiny kink where the bracket index jumps.
  ~5 lines. Good enough for inference; less clean than B for a methods paper.
- **C (most JAX-native):** replace the post-hoc search with a `diffrax.Event` at
  ψ=0; diffrax returns ξ_t with correct smooth gradients (does the IFT internally).
  More solver plumbing; attractive if we revisit the solve anyway.

**Recommendation when triggered:** B (custom_jvp + IFT) for the exact smooth
gradient; it's ~15–20 lines and reuses `y[1]`. Validate `∂r_t/∂W₀` against central
finite differences and add a 6th panel to the gradient-validation figure.

## Science cases that would trigger implementation (ranked)

**Compelling (justify B):**
1. **Galactic potential / orbits from GC limiting radii.** King `r_t ≈ r_J =
   (M_cl / 3 M_gal(<R))^{1/3} R`. A differentiable `r_t` lets gradients flow from
   observed cluster truncation → `M_gal(<R)` and pericenter. With Gaia orbits, do
   HMC over (cluster structure, orbit, Milky-Way mass) jointly, using a population
   of clusters' limiting radii as constraints. On-brand for the differentiable
   jaxstro thesis and the Cottrell census.
2. **Roche-filling / tidal-coupling priors.** Impose `r_t = r_J(orbit, M_gal)` as
   a physical prior linking structure to orbit (vs. r_t free) — needs
   ∂r_t/∂(W₀, orbit). This is the direct hook for integrating
   `progenax.tidal.jacobi_radius` / `apply_tidal_truncation` into a differentiable
   fit (Anna's planned tidal-profile integration).

**Marginal (B is convenience, not necessity):**
3. **Catalog-concentration fitting.** A direct `(c_model − c_catalog)²/σ_c²` term
   (Harris-style) — can instead refit the underlying profile.
4. **Outer-edge-sensitive end-to-end observables** — tidal-tail onset, count beyond
   ~0.9 r_t in a mock image; real but downstream of gravax/rendering.

**Not a case:** ordinary concentration inference — already covered via W₀.

## Trigger condition (when to pick this up)

Implement B when **any planned analysis puts `r_t` or `c` on the left of a
likelihood/prior**, in priority order: tidal-field coupling (case 1/2) → catalog-c
(case 3) → outer-edge observables (case 4). At that point: implement B, add the FD
grad-check + the 6th gradient-validation panel, and wire `r_t ≈ r_J` through
`progenax.tidal`.
