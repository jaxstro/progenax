---
title: Peuten et al. (2017)
description: Annotated reference for Peuten, Zocchi, Gieles, Gualandris & Hénault-Brunet — "Testing isothermal models II: Multimass" / the LIMEPY multimass methods paper, with the verified velocity-scale, anisotropy, and mean-mass-convention equations.
---

# Peuten et al. (2017)

```{admonition} Testing lowered isothermal models II: Multimass models (LIMEPY)
:class: note

**Authors.** M. Peuten, A. Zocchi, M. Gieles, A. Gualandris, V. Hénault-Brunet.

**Reference.** *Monthly Notices of the Royal Astronomical Society* **470**,
2736–2761 (2017). **DOI.** [10.1093/mnras/stx1311](https://doi.org/10.1093/mnras/stx1311) ·
**arXiv.** [1609.01720](https://arxiv.org/abs/1609.01720) ·
**ADS.** [2017MNRAS.470.2736P](https://ui.adsabs.harvard.edu/abs/2017MNRAS.470.2736P).

**Held PDF.** `docs/core-papers/Peuten2017.pdf`. Equations below verified
page-by-page (2026-06-11, §2, pp 2737–2739).
```

This is the **definitive LIMEPY *multimass* methods paper** — the "follow-up
study (Peuten et al., in preparation)" promised by
[Gieles & Zocchi 2015](gieles-zocchi-2015.md) §3.2. It tests the multimass
lowered-isothermal models against N-body simulations of clusters with different
stellar-mass black-hole (BH) and neutron-star (NS) retention fractions, fitting
the LIMEPY model to mock observables. progenax holds it because it pins down the
exact multimass parametrization (and the mean-mass-convention subtlety) that our
Engine A reimplements.

## The multimass parametrization (§2, eqs 3–5 — verified, matches progenax)

A multimass LIMEPY model adds, to the single-mass `(g, W₀, r_a)`, the per-bin
masses `(M_j, m_j)` and **two** parameters `δ, η` that set the mass dependence of
the velocity scale and anisotropy radius via power laws:

```{math}
\hat r_{a,j} = \hat r_a\, \mu_j^{\eta} \quad (\text{eq 3}),
\qquad
s_j = s\, \mu_j^{-\delta} \quad (\text{eq 5}),
\qquad
\mu_j = \frac{m_j}{\bar m} \quad (\text{eq 4}).
```

`η=0` ⇒ mass-independent anisotropy (as Gunn & Griffin 1979); `δ=½` is the
traditional equipartition assumption (`m_j s_j² = const`), **but Peuten et al.
determine `δ` from fits to the N-body models rather than fixing it** — i.e. the
published realism lever is a *free δ*, which progenax already exposes as a
differentiable parameter. (These are the same `s_j = s μ_j^{−δ}`,
`r̂_{a,j} = r̂_a μ_j^η`, `μ_j^{2δ}` density rescaling that progenax's
`MultiComponentCluster` / `find_alpha_for_masses` implement.)

The dimensionless Poisson equation solved is `∇̂²φ̂ = −9 Σ_j α_j ρ̂_j` (eq 7),
`α_j = ρ_{0j}/ρ_0`, iterated by varying `α_j` until the computed `M_j` match the
input — the eigenvalue-by-iteration scheme also stated in
[GZ15](gieles-zocchi-2015.md) §2.2/§4 (progenax's √-update `find_alpha_for_masses`).

## The mean-mass convention — central vs global (§2, eqs 8–9 — verified, important)

Peuten et al. document a subtlety that **the ours-vs-reference validation must
handle**:

- In the formulation of Da Costa & Freeman (1976), Gunn & Griffin (1979) and
  **GZ15, `m̄` is the central-density-weighted mean mass**
  `m̄ = (Σ_j m_j ρ_{0j})/(Σ_j ρ_{0j})` (GZ15 eq 26) — **this is what progenax
  uses** (`bar_m = Σ m_j α_j`).
- The Peuten-era LIMEPY *code* instead uses the **global mean mass** for speed
  ("especially for models with BHs"); the two give the same results within
  numerical uncertainty, but switching convention **rescales the meaning of `W₀`
  and `r_a`** (they then refer to the `m̄` group). The exact translation between
  an `m̄` convention and an `m̄*` convention is (eqs 8–9):

```{math}
W_0^{*} = W_0\left(\frac{\bar m^{*}}{\bar m}\right)^{2\delta} \quad(\text{eq 8}),
\qquad
\hat r_a^{*} = \hat r_a\left(\frac{\bar m^{*}}{\bar m}\right)^{\eta+\delta} \quad(\text{eq 9}).
```

The direction makes physical sense: a *heavier* reference mass
(`m̄* > m̄`) means a *colder* reference velocity scale
(`s* = s (m̄*/m̄)^{−δ}`), so the same physical central potential is a *deeper*
`W₀* = φ₀/s*²` in the starred convention. A direct progenax-vs-reference-code
comparison must apply eqs 8–9 (or set the reference code to the
central-density-weighted mode, `meanmassdef='central'` — the route progenax's
parity harness takes) so the two describe the *same* physical model. (The `δ`
in eq 9 comes from the `r_0` dependence of `r̂_a`.)

```{admonition} Transcription correction (2026-06-11)
:class: warning
An earlier revision of this note (and of the hardening design doc) had the
eq 8–9 ratio inverted, `(m̄/m̄*)`. The paper (p. 2738) reads `(m̄*/m̄)` as
written above — re-verified against the held PDF during Task 3 of the
multimass validation arc.
```

## N-body test setup (§3 — context, verified)

Four `N=10⁵` Hénon-isochrone N-body models (Kroupa IMF 0.1–100 M_⊙, [Fe/H]=−2,
evolved to ~12 Gyr with NBODY6) with different BH/NS retention fractions (100%,
33%, 10%, 0%). Supernova kicks are mimicked by removing a fraction of remnants.
The multimass LIMEPY models are fit to snapshots every Gyr; bound objects are
selected via the Jacobi radius `r_J = (G M_cl / 2Ω²)^{1/3}` (eq 10) and the
critical energy `E_crit = φ(r_J) = −G M_cl / r_J` (eq 11). Headline science:
constraining the present-day BH population of clusters (the paper is motivated by
the NGC 6101 BH-population question).

## Use in progenax

- **Velocity-scale / anisotropy / Poisson equations** (eqs 3–5, 7) are the
  published basis our Engine A reimplements faithfully (verified match — see the
  hardening design note).
- **The m̄-convention translation (eqs 8–9)** is required by the
  ours-vs-reference-LIMEPY cross-validation harness.
- **Free-δ fitting** is the published "more realistic" lever — already supported
  differentiably in progenax. The code's extra `meq`/`zeta` knobs are *not* in
  this paper's equations (deferred in progenax; documented as code-heuristics).
