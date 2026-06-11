# Multimass LIMEPY Hardening — Design (reference parity + meq/zeta)

**Date:** 2026-06-11 (brainstormed with Anna; equations verified against held PDFs)
**Status:** APPROVED design. Anna decisions recorded inline.

## Purpose

Harden progenax's Engine A multimass equilibrium (`MultiComponentCluster` /
`find_alpha_for_masses` / the LIMEPY DF) into a **faithful, differentiable,
validated superset of the canonical LIMEPY code** — by (1) building a direct
ours-vs-reference-LIMEPY cross-validation harness, then (2) adding the two
beyond-GZ15 code parameters we omitted (`meq`, `zeta`), with honest provenance.

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
- **Decision (Anna):** implement the reference *code*'s `meq` form
  (`μ_j=(m_j+meq)/m̄`) for full reference-code parity, **documented honestly** as
  a code heuristic physically motivated by Bianchini2016's observed saturation,
  **NOT** a GZ15/Peuten equation. Validate it reproduces the reference code's meq
  behavior (and, qualitatively, the Bianchini saturation direction).
- **`zeta`/`zeta_lim`** (the code's `s²_j *= zeta` for `m_j>zeta_lim`,
  massive-object decoupling): also code-only (not in these papers; conceptually
  the BH/remnant decoupling Peuten2017 studies). Same honest-provenance treatment.

**Published realism lever we ALREADY have:** Peuten2017's actual method is to
**fit δ freely** (not fix δ=0.5) against N-body — and progenax already exposes δ
as a differentiable free parameter. So the headline "do it better the published
way" is largely already in hand; meq/zeta are the *code-parity* extras.

## Approach — two phases

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

### Phase 1b — add `meq` + `zeta` (code parity, honest provenance)

- **meq:** `mu_j = (m_j + meq)/bar_m` (one-line at `limepy_multimass.py:334`);
  flows into `w_j = mu_j^{−δ}` and `ra_hat_j = ra_hat·mu_j^η` automatically.
  Differentiable in `meq` (smooth) → a new inferrable parameter. Default
  `meq=0.0` ⇒ bit-identical to today.
- **zeta/zeta_lim:** `s²_j *= zeta` for `m_j>zeta_lim` ⇒ in our terms
  `w_j *= √zeta` (equivalently `rescale_j /= zeta`) on heavy bins, via a static
  mask `jnp.where(m_j>zeta_lim, zeta, 1.0)` (m_j are fixed bin reps ⇒ jit/grad
  safe; differentiable in `zeta`, not the hard `zeta_lim`). Default `zeta=1.0`.
- **API:** add `meq=0.0, zeta=1.0, zeta_lim=3.0` to `find_alpha_for_masses`,
  `MultiComponentCluster.from_imf`, `from_mass_segregation` (defaults = current
  behavior ⇒ zero blast radius; the 1163 invariant stays green). `from_components`
  (direct `w_j`) untouched. **Engine B untouched** (these are Engine-A
  equipartition-ansatz params only). Do NOT change the √-update solver — only the
  `mu_j`/`s²_j` it consumes.
- **Validation:** extend the harness with `meq>0` and `zeta>1` configs cross-
  checked ours-vs-reference (α_j, ρ_j, σ_j ~1%); grad wrt `meq`/`zeta` finite +
  AD-vs-FD; physics check that `meq>0` flattens σ(m) at the heavy end (the
  saturation direction, qualitatively consistent with Bianchini eq 3).

### Documentation deliverables

- New per-paper notes `peuten-2017.md`, `bianchini-2016.md` (verified equations,
  page cites); update `gieles-zocchi-2015.md` with eqs 24–26/29 + the m̄-mode
  note. Fix our `find_alpha_for_masses` docstring's "eigenvalue" wording to also
  name the GZ15 √-update MF iteration; and ensure NO progenax docstring repeats
  the code's wrong "meq = eq 24 GZ15" citation.

## Validation gates (never weakened)

1. Reference parity at meq=0: shape agreement <~1% (set after measuring), the
   1163 released-core suite green, a stored regression pin.
2. meq/zeta correctness: ours-vs-reference at meq>0/zeta>1 within the same
   tolerance; grad wrt meq/zeta matches central FD <1e-5; meq=0/zeta=1 ≡ current
   (regression pin).
3. Physics: σ(m) saturates (flattens at heavy end) for meq>0, qualitatively
   matching Bianchini eq 3.

## Out of scope

Engine B; continuous-mass DF f(E,m) (the genuine "beyond LIMEPY" research —
deferred to a future N-body-validated arc); changing the √-update or N_COMP;
the B2 demo science (Tasks 6–9 resume after this or interleave — demo-only, no
conflict with this released-core arc).
