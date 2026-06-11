# Multimass LIMEPY Hardening — Design (reference parity + meq/zeta)

**Date:** 2026-06-11 (brainstormed with Anna; equations verified against held PDFs)
**Status:** APPROVED design. Anna decisions recorded inline.

## Purpose

Establish progenax's Engine A multimass equilibrium (`MultiComponentCluster` /
`find_alpha_for_masses` / the LIMEPY DF) as a **faithful, differentiable,
validated reimplementation of the canonical LIMEPY** — by (1) a direct
ours-vs-reference-LIMEPY cross-validation harness, (2) validating that our model
reproduces Bianchini2016's equipartition-saturation σ(m) with the *derived*
m_eq (no new parameter), and (3) documenting the methods + theory in per-paper
notes and a theory page. The `meq`/`zeta` fitting knobs are **deferred** (Anna
2026-06-11) — verified to be extra freedom, not the saturation mechanism.
**This arc is validation + documentation: no released-core behavior change.**

## Equation verification (DONE before any code — Anna's instruction)

Read from the held PDFs: `docs/core-papers/gieles-2015-LIMEPY.pdf` (GZ15),
`docs/core-papers/Peuten2017.pdf` (Peuten, Zocchi, Gieles+ 2017, MNRAS 470, 2736
— "Testing isothermal models II: Multimass", the definitive LIMEPY-multimass
methods paper), `docs/core-papers/Bianchini2016.pdf` (MNRAS 458, 3644).

**Our current code is a faithful match to the PUBLISHED LIMEPY multimass model:**

| Quantity | Published equation | Our code | Status |
|---|---|---|---|
| velocity scale | `s_j = s·μ_j^{−δ}` (GZ15 eq 24; Peuten eq 5) | `w_j = μ_j^{−δ}`, `s_j = s·w_j` (`multicomponent.py`) | ✓ |
| anisotropy radius | `r̂_{a,j} = r̂_a·μ_j^{η}` (GZ15 eq 25; Peuten eq 3) | `ra_hat_j = ra_hat·μ_j^η` (`limepy_multimass.py:336`) | ✓ |
| mass ratio | `μ_j = m_j/m̄` (GZ15; Peuten eq 4) | `mu_j = m_j/bar_m` (`:334`) | ✓ |
| potential rescale | density uses `μ_j^{2δ}·φ̂` (GZ15 eq 29) | `rescale_j = μ_j^{2δ}` (`:335`) | ✓ |
| mean mass | `m̄ = (Σ m_j ρ_{0j})/(Σ ρ_{0j})` central-density-weighted (GZ15 eq 26) | `bar_m = Σ m_j α_j` (α_j = ρ_{0j}/ρ_0, Σα_j=1 ⇒ identical) | ✓ |
| MF iteration | "eigenvalue problem ... solved by iteration", first guess `α_j = M_j/ΣM_j` (GZ15 §2.2, §4) | `find_alpha` √-update `α ← normalize(α·√(f_target/f_real))` | ✓ |

**The m̄-convention reconciliation (Peuten2017 §2, eqs 8–9) — REQUIRED for the
harness:** GZ15/ours use the **central-density-weighted** m̄ (eq 26). The Peuten
era LIMEPY *code* switched to the **global** mean mass for speed (esp. with BHs),
which rescales the meaning of `W₀` and `r_a` (they then refer to the m̄ group).
The exact translation between an m̄ and an m̄* convention:
- `W₀* = W₀·(m̄/m̄*)^{2δ}` (Peuten eq 8)
- `r̂_a* = r̂_a·(m̄/m̄*)^{η+δ}` (Peuten eq 9)
They describe the **same physical model** — the harness must apply eqs 8–9 (or
configure the reference code's m̄ mode) so ours-vs-reference is like-for-like.

**The meq finding (Anna flagged "verify all equations"):**
- **`meq` is NOT in GZ15 or Peuten2017.** Both use the pure power law `μ_j=m_j/m̄`.
  The LIMEPY *code*'s `μ_j=(m_j+meq)/m̄` (its "Feb 2016" addition) cites "eq 24
  GZ15" in its docstring — that citation is **wrong** (eq 24 has no meq).
- Its physical motivation is **Bianchini2016 eq 3**: `σ(m)=σ₀exp(−m/2m_eq)` for
  m≤m_eq, `σ_eq·(m/m_eq)^{−1/2}` for m>m_eq (σ_eq=σ₀e^{−1/2}); slope
  `η(m)=−dlnσ/dlnm = ½(m/m_eq)` below, `½` above (Bianchini eq 4). This is an
  **exponential, observational σ(m) FITTING function** (a *different* functional
  form from the code's power-law softening), used to *measure* the degree of
  equipartition from data — not a self-consistent DF parametrization.

**The Appendix-A finding (Bianchini2016 App. A — the bridge; read in full):**
Bianchini *derives* the exponential σ(m) **from the GZ15 multimass DF** (our
model). The component central dispersion (eq A1)
`σ̂_{1d,j0} = (1/μ_j²)·E_γ(g+5/2; μ_j^{2δ}φ̂₀)/E_γ(g+3/2; μ_j^{2δ}φ̂₀)`,
Taylor-expanded for low mass (μ_j≪1, δ=½), gives (eq A3)
`σ ~ σ₀[1 − ½(m/m_eq) + ⅛(m/m_eq)²]` = the Taylor series of `σ₀exp(−m/2m_eq)`;
the high-mass limit gives `σ ∝ m^{−1/2}`. **m_eq is therefore NOT an input — it
EMERGES from the model:**
> **`m_eq = m̄·(g+5/2)(g+7/2)/φ̂₀`** (matching App. A2↔A3 linear terms).

So **our current code (μ_j=m_j/m̄, δ=½, no meq) ALREADY reproduces Bianchini's
σ(m) saturation** with this derived m_eq, set by the mean mass, truncation g, and
central concentration Ŵ₀=φ̂₀. The code's `μ_j=(m_j+meq)/m̄` is a **separate
phenomenological knob** that adds *extra* low-mass softening to **decouple the
equipartition degree from g/Ŵ₀** when fitting clusters — NOT "the saturation"
(the model already has that). m_eq↔relaxation: Bianchini's headline result is
`m_eq` correlates tightly with dynamical age `n_eq=T_age/T_rc` (clusters >~20 core
relaxation times reach max equipartition) — a kinematic proxy for relaxation
state. The Spitzer instability (heavy stars decoupling into a self-gravitating
subsystem) is the physical origin of the `zeta` knob.

- **Decision (Anna 2026-06-11): DEFER `meq` and `zeta`** — they are *knobs*
  (extra fitting freedom), not needed to capture the saturation physics (the
  model already does, per App. A). Revisit later if a fit demands decoupling
  equipartition from g/Ŵ₀ (meq) or a BH/remnant subsystem (zeta). When added,
  document honestly as code-heuristics (NOT GZ15/Peuten equations).
- **Promoted instead (zero new released-core code):** *validate* that our model
  reproduces Bianchini's σ(m) with the **derived** `m_eq=m̄(g+5/2)(g+7/2)/φ̂₀` —
  proving progenax captures the equipartition-saturation physics with no new
  parameters. High grounding payoff; a methods-paper figure.

**Published realism lever we ALREADY have:** Peuten2017's actual method is to
**fit δ freely** (not fix δ=0.5) against N-body — and progenax already exposes δ
as a differentiable free parameter. So the headline "do it better the published
way" is largely already in hand; meq/zeta are the *code-parity* extras.

## Approach

### Phase 1a — reference cross-validation harness (fully grounded, build first)

`scripts/validate_limepy_reference.py` (gated CLI, expected-vs-measured table,
exit-nonzero) + `tests/validation/test_limepy_reference_parity.py` (skip if the
reference is not importable) + a `ρ_j(r)`/`σ_j(r)` ours-vs-ref overlay+residual
figure via `_plotstyle`.

- **Reference import (not a dependency):** test/script-local
  `sys.path.insert(0, ".../ref-repos/limepy")` (pure numpy/scipy). Released gate
  never hard-depends on it (skip-if-absent).
- **m̄ reconciliation:** detect/which m̄ mode the reference code uses; apply
  Peuten eqs 8–9 to translate W₀/r̂_a, OR configure the reference to the
  central-density-weighted mode, so the comparison is like-for-like.
- **Compare scale-invariant quantities** (convention-light): converged
  per-component central-density ratios `α_j`; density *shape* `ρ_j(r)/ρ_j(0)` on
  a shared `r/r_scale` grid; dispersion shape `σ_j(r)/σ_j(0)` (or `v²_j`);
  per-component mass fractions at convergence; concentration `r_t/r_c`, `r_h`.
- **Configs:** single-mass (nmbin=1 ≡ King) sanity; a 2-component; the exact B2
  4-component (binned Maschberger); iso + anisotropic (η=0 and η≠0); g∈{1,1.5}.
  Gates set AFTER measuring (honest); target ~1% shape agreement (same ODE).

### Phase 1a+ — derived-m_eq saturation validation (zero new released-core code)

Promoted from the deferred meq work (Anna 2026-06-11). With NO new model
parameter, validate that our standard model (μ_j=m_j/m̄, δ=½) reproduces
Bianchini's equipartition relation:
- Compute the per-component central dispersion `σ̂_{1d,j0}` from our model across
  a mass spectrum; fit / compare its `m`-dependence to Bianchini eq 3
  `σ₀exp(−m/2m_eq)` (low-mass) + `m^{−1/2}` (high-mass).
- Check the **derived** `m_eq = m̄(g+5/2)(g+7/2)/φ̂₀` (Bianchini App. A2↔A3)
  against the m_eq recovered by fitting our model's σ(m) — they must agree (this
  is the headline: progenax captures the saturation physics analytically).
- Lives in the same `validate_limepy_reference.py` CLI + a σ(m)-vs-Bianchini
  figure. Gate: derived vs fitted m_eq agreement (tolerance set after measuring).

### DEFERRED (Anna 2026-06-11) — `meq` + `zeta` as fitting knobs

Not built now (they are extra fitting freedom, NOT the saturation mechanism —
the model already saturates, per App. A). Recorded for a future arc:
- **meq:** `mu_j=(m_j+meq)/bar_m` (one line at `limepy_multimass.py:334`),
  default 0 ⇒ no-op; decouples the equipartition degree from g/Ŵ₀.
- **zeta/zeta_lim:** `s²_j*=zeta` for `m_j>zeta_lim` via a static
  `jnp.where(m_j>zeta_lim, zeta, 1.0)` mask, default 1 ⇒ no-op; BH/remnant
  decoupling (Spitzer instability).
- Both added as optional kwargs to `find_alpha_for_masses`/`from_imf`/
  `from_mass_segregation` (defaults = current behavior ⇒ zero blast radius;
  Engine A only; document as code-heuristics, NOT GZ15/Peuten equations).

### Documentation deliverables (this arc)

- New per-paper notes `peuten-2017.md`, `bianchini-2016.md` (verified equations,
  page cites, incl. Bianchini App. A derivation + the derived-m_eq formula);
  update `gieles-zocchi-2015.md` with eqs 24–26/29 + the m̄-mode note (Peuten
  eqs 8–9). Fix our `find_alpha_for_masses` docstring's "eigenvalue" wording to
  also name the GZ15 √-update MF iteration; ensure NO progenax docstring repeats
  the code's wrong "meq = eq 24 GZ15" citation. A theory/methods page documenting
  the multimass equilibrium + the equipartition-saturation result.

## Validation gates (never weakened)

1. Reference parity: ours-vs-reference-LIMEPY shape agreement <~1% (set after
   measuring) across the config set; the 1163 released-core suite green; a stored
   regression pin.
2. Derived-m_eq saturation: our model's fitted σ(m) m_eq agrees with the derived
   `m̄(g+5/2)(g+7/2)/φ̂₀` (tolerance set after measuring); σ(m) flattens at low
   mass and → m^{−1/2} at high mass (Bianchini eqs 3, A3).

## Out of scope

Engine B; continuous-mass DF f(E,m) (the genuine "beyond LIMEPY" research —
deferred to a future N-body-validated arc); changing the √-update or N_COMP;
the B2 demo science (Tasks 6–9 resume after this or interleave — demo-only, no
conflict with this released-core arc).
