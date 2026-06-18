# W₀-OED validation demo (concentration) — design

**Date:** 2026-06-18
**Branch:** `feat/oed-concentration-w0`
**Status:** design ratified (brainstorming, HITL); plan + implementation pending
**Arc:** closes the end-to-end loop on the W₀-differentiability work (ADR-0016 C¹ PCHIP,
ADR-0017 df_moment W₀ lock) by exercising it through an actual Fisher/OED inference with
W₀ as a parameter.

## Motivation

The Michie-W₀ arc (ADR-0016) and the df_moment W₀ lock (ADR-0017) made the dispersion
forward models W₀-differentiable. That is validated **only at the gradient-audit level**
(AD-vs-FD, 98/0 hazards) — never through a Fisher/OED inference that treats W₀ as a
parameter. The stated goal of the whole line of work, "W₀-differentiable σ for OED/Fisher,"
has an open loop. This arc closes it: a c-/D-optimal optimal-experimental-design demo that
asks **where to spend a fixed star budget (radial bins × {RV, PM_R, PM_T}) to best constrain
a cluster's concentration W₀**.

### Two honest caveats (load-bearing — do not oversell)

1. This validates the **`project_dispersion` (Jeans) W₀ path**, not `df_moment_dispersion`
   (a separate path, already grad-audited; the OED Fisher does not ride it).
2. It is **informax-bound and held OUT of v0.1.0** (the OED hold-out decision): it ships as a
   **scripts + `60-science-demos`** validation/demo, **not** released-core — no
   `src/progenax/` API surface, no released-core registry burden.

## Decisions (ratified in brainstorming, one at a time)

| # | Question | Decision |
|---|----------|----------|
| 1 | Forward model | **King + Michie both** — headline OM-King; Michie added to exercise the ADR-0016 native r_t(W₀) path explicitly |
| 2 | Parameter set θ | **(W₀, r_a, M)** joint (r_c ≡ 1 length unit); headline target = W₀ |
| 3 | Criterion | **c-optimal on σ(W₀)** headline; D and A computed alongside (ADR-0001, Stage-1 precedent) |
| 4 | Gate | **Real-star, both models** — eddington_invert OM samplers (sampler ≡ Fisher-model); @slow calibration + AD-vs-FD on ∂σ/∂lnW₀ |
| 5 | Thresholds | **Stage-1 precedent** — AD-vs-FD rel < 1e-3; calibration within 2√(2/n_draws), no significant bias |
| 6 | Hypothesis (pre-registered) | **W₀ differs from r_a** — more core/intermediate weight, more channel-balanced, modest outward pull to break the W₀×r_a degeneracy |

## Adversarial verification against the live repo (done before ratifying)

- **Stage-1 backbone API** (`scripts/_demo_oed.py`): confirmed — `predict_sigma`,
  `jacobian_and_sigma` (the ONE jacrev + ln-θ scaling `J*theta`), `blocks_from_eps`,
  `fisher` (additive `F = Σ n·c·M`), `c/d/a_criterion`, `optimize_design`,
  `calibrate_fisher`. The new arc reuses this structure verbatim, swapping the profile.
- **`project_dispersion` supports OM anisotropy only** (no `beta_fn` arg) — so any model it
  projects is "*that profile's density under OM* β(r)=r²/(r²+r_a²)". This is **why** Michie
  through `project_dispersion` is OM-on-Michie-density, not native Michie.
- **W₀ lives only on King/Michie** (`KingProfile.from_W0_rc(W0, r_c)`,
  `MichieProfile.from_W0_rc(W0, r_c, r_a)`); Plummer (Stage-1) has none. Both profiles expose
  `sample_positions` + `density`.
- **No OM sampler exists for King**, and `MichieVelocityDF` samples the *native* β. The
  matching samplers for the OM projection are assembled from the **generic**
  `eddington_invert(ρ, dρ/dr, Ψ, dΨ/dr, r_a)` + `sample_speed_from_f_table` +
  `assign_om_directions` (Merritt 1985 augmented density ρ_Q = (1+r²/r_a²)ρ). This restores a
  Stage-1-style real-star gate for **both** models.
- **∂σ/∂W₀ through `project_dispersion` runs and is FD-consistent** (verified directly, not
  trusted from ADR-0016):

  | Model | ∂σ_los/∂W₀ AD-vs-FD rel-err (R = 0.5, 2, 8 r_c) at W₀=6 | Verdict |
  |---|---|---|
  | OM-King | 1.5e-6, 3.2e-4, 1.7e-4 | clean at these 3 radii, < 1e-3 |
  | OM-Michie | 3.3e-5, 1.1e-5, **8.0e-3** | clean inner; ~8e-3 at R=8 |

  **CORRECTION (Task 3 full 12-bin sweep — this 3-radius probe was under-sampled).** The probe
  above used only R=0.5, 2, 8 r_c and so MISSED the **mid-radius high-curvature bins**. A full
  sweep over all 12 `R_BINS` shows a *fixed-step* h=1e-4 central FD breaks 1e-3 at **R≈2.2–4.4
  r_c for BOTH King (bins 6,7) and Michie-inner (bins 7,8)** — and, after the C¹ PCHIP fix, the
  Michie **outermost** bin (R=12) is itself FD-clean (~5e-5), i.e. the FD-unreliable region is
  the mid-radius bend, not the outskirts. An h-sweep to 1e-6 confirms **FD→AD (rel ~1e-6–1e-8)
  at every bin**, so the AD gradient is correct everywhere; the mid-radius >1e-3 values are pure
  O(h²σ‴) FD truncation — the ADR-0016 signature, not a code defect. **Consequence for the
  gate:** every bin is gated by the repo's ratified Richardson idiom (AD finite + AD==converged
  fine-step FD + fixed-step FD converges toward AD as h↓), with the strict `rel<1e-3` fixed-step
  floor applied only to the FD-reliable bins (≥ K−2 of them). This is *stronger* than a single
  fixed-h floor and weakens no verified claim (the 3-radius "clean King" was just narrowly
  scoped). Independently re-derived and confirmed in the Task-3 code review.

## Architecture

Scripts-only, mirroring the Stage-1 / Stage-2 layout:

- `scripts/_demo_oed_concentration.py` — core (forward model + Fisher + criteria + optimizer +
  OM samplers + calibration). Consumer of `progenax.project_dispersion`,
  `progenax.KingProfile`/`MichieProfile`, and the `progenax.kinematics.eddington` helpers.
- `scripts/demo_oed_concentration.py` — gated CLI (exit 0), produces the figures.
- `tests/` — scripts-level unit tests + the @slow calibration gate + the AD-vs-FD gate.
- `docs/website/60-science-demos/optimal-design/concentration.md` — MyST page, "Inputs and
  assumptions" standard, 0 warnings.

No `src/progenax/` change → coverage/dashboard staleness gate is unaffected. (If any `src/`
edit becomes necessary, a FULL `--cov` re-stamp is required.)

### Forward / Fisher backbone (reused from Stage-1)

```text
theta = (W0, r_a, M)            # r_c ≡ 1 is the length unit
predict_sigma(theta, R, G):
    prof = KingProfile.from_W0_rc(W0, 1.0)           # headline
         | MichieProfile.from_W0_rc(W0, 1.0, r_a)    # second model
    return project_dispersion(prof, r_a, R, M, G) -> (sigma_los, sigma_pm_r, sigma_pm_t)
```

- The **same** `r_a` is both a θ-component (we constrain it) and the OM anisotropy radius
  passed to `project_dispersion` — consistent, single source of truth.
- ONE `jax.jacrev(predict_sigma)` → `J = ∂σ/∂lnθ` (3 channels × K bins × 3 params), reverse-mode
  by policy (the King/Michie custom_vjp ODEs have no jvp rule, so forward-mode would crash).
- Additive Fisher `F = Σ_{c,b} n_eff,{c,b} · M_{c,b}` with `M_{c,b} = 2 J Jᵀ / (σ² + ε_c²)` in
  the dimensionless ln-θ metric (ADR-0011) → every `F⁻¹` entry is a fractional variance.
- c-optimality on the W₀ index (headline), D and A alongside; optax multi-start optimizer over
  the softmax design vector. All lifted from `_demo_oed.py`.

### The two OM samplers (the main implementation risk → de-risked in Task 1)

For each model, assemble a particle sampler that draws from the **exact** model
`project_dispersion` projects (OM-on-that-density), so sampler ≡ Fisher-model:

1. positions from `prof.sample_positions(masses, key)`;
2. ρ(r), dρ/dr, Ψ(r), dΨ/dr from the profile (follow the `PlummerVelocityDF` /
   `EFFVelocityDF` OM pattern — they do exactly this internally);
3. `E_grid, f_grid = eddington_invert(r, ρ, dρ/dr, Ψ, dΨ/dr, r_a)`;
4. speeds from `sample_speed_from_f_table`, directions from `assign_om_directions(r_a=r_a)`.

Task 1 must **verify the OM DF is non-negative** at the chosen (W₀, r_a) before any gate is
built on it (Merritt's r_a lower bound; `eddington_invert` returns raw f so negativity is
detectable). Deliberately do **not** use `MichieVelocityDF` (native β) — it would mismatch the
OM projection and bias the gate.

## Validation gate (thresholds locked; never weakened — fix the root cause)

1. **AD-vs-FD on ∂σ/∂lnW₀** (the W₀ column of J — the load-bearing differentiability check):
   - OM-King: `rel < 1e-3` across all K radii (verified ≤ 3.2e-4).
   - OM-Michie: `rel < 1e-3` at R ≲ r_a (fixed-step FD); outer radii via **Richardson-FD**
     (`test_grad_michie_W0_richardson` asserts AD→FD convergence as h↓).
2. **Real-star @slow calibration** (sampler ≡ Fisher-model, end-to-end):
   per draw → sample OM particles → project to sky → bin by R → subsample design counts → add
   per-star ε → σ_hat + SE → MAP-fit θ in the **ln-θ GN metric** (Stage-2's `_fit_theta_gn`,
   not physical-Adam) → collect lnŴ₀. Assert realized `Var(lnŴ₀)` ≈ Fisher `(F⁻¹)_{W₀W₀}`
   within `2√(2/n_draws)`, no significant bias.

## Science deliverable

Pre-registered hypothesis (**H1**): W₀'s c-optimal allocation **differs** from r_a's — more
weight at core/intermediate radii and more channel-balanced (concentration is isotropic,
pinned by the core↔truncation σ contrast), with only a *modest* outward pull to break the
W₀×r_a degeneracy. Null/alternative (**H0**): W₀ resembles r_a (outskirts-PM-dominated), which
would mean concentration cannot be pinned independently of anisotropy in this design space. A
wrong prediction is a finding, not a failure (null-result integrity).

Outputs: the W₀ optimal allocation map (radius × channel) side-by-side with Stage-1's r_a map;
c/D/A comparison; the equal-precision star-factor; the `F_{W₀,r_a}` off-diagonal quantifying
the degeneracy. Publication-quality figures.

## ADR

Record the OM-King + OM-on-Michie-density modeling choice and the eddington real-star gate as
**ADR-0018**.

## Out of scope (YAGNI)

- No magnitude/depth knob (that is Stage-2 machinery; this rides the Stage-1 fixed-completeness
  backbone).
- No 4th parameter (r_c); no native-Michie projection (impossible through `project_dispersion`).
- No `src/progenax/` API; no released-core promotion (informax-bound).
