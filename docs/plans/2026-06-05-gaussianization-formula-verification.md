# Gaussianization predicted-statistics — formula-verification memo (Phase 0)

> **Status:** Phase-0 grounding for the differentiable predicted-statistics plan
> (`/Users/anna/.claude/plans/continue-the-gravoturb-fdf-differentiabl-dreamy-firefly.md`).
> **Rule:** every formula below is cited to a *held PDF* in `docs/core-papers/`, read directly
> this session ([[no-assumptions-verify-against-pdfs]]). **GATE: Anna reviews this memo before
> any Phase-1 statistics code.**

## 0. Provenance correction (why this memo exists)

The design doc cited *"Szapudi & Pan 2004, ApJ 601, 697"* for the Gaussianization 2-pt formula.
That citation (prior-session provenance) is **wrong on volume and page**. The actual paper is:

- **Szapudi & Pan 2004, ApJ *602*, 26–37** (= **arXiv:astro-ph/0308525**, ADS
  `2004ApJ...602...26S`) — "On Recovering the Nonlinear Bias Function from Counts-in-Cells
  Measurements." (verified from the held PDF title page + arXiv).

This is exactly why we re-derive from the PDFs rather than trusting the inherited citation.

## 1. Held source PDFs (verified identities)

| File in `docs/core-papers/` | Verified bibliographic identity | Role here |
|---|---|---|
| `Coles-Jones-1991.pdf` *(was mis-named `-1981`; renamed)* | Coles & Jones 1991, **MNRAS 248, 1** | lognormal field; 2-pt exp-case; mean-1 norm; compound-Poisson counts; fat-tail caution; hierarchical higher-order |
| `Szapudi_2004_ApJ_602_26.pdf` | Szapudi & Pan 2004, **ApJ 602, 26** | CIC Poisson integral; CIC Poisson *likelihood*; SLN3 log-density PDF; σ_Φ(R) smoothing-scale dependence |
| `Carron_Szapudi_2013_MNRAS_434_2961.pdf` | Carron & Szapudi 2013, **MNRAS 434, 2961** | log-transform sufficiency; power-transform exponent ↔ spectral slope |
| `Carron_Szapudi_2014_MNRASL_439_L11.pdf` | Carron & Szapudi 2014, **MNRASL 439, L11** | sufficient observables for Poisson-sampled lognormal (discrete) |
| `Carron_Wolk_Szapudi_2014_MNRAS_444_994.pdf` | Carron, Wolk & Szapudi 2014, **MNRAS 444, 994** | lognormal-Poisson field generation + power spectra + **CIC + covariances + survey window** |
| `Federrath-2010.pdf` | Federrath et al. 2010, A&A 512, A81 | projected (column) PDF / Limber — **deferred to Phase 2** (see §11) |

## 2. The copula map T and the mean-1 convention

Our field: `g` = unit-variance Gaussian random field with `P(k) ∝ k^{−β}` (the simulator's
`gaussian_random_field`). The Gaussianized log-density is `s = T(g)`, with the BM19 copula map

```
T(g) = bm19_icdf(Φ(g); ℳ,b,α) − log⟨e^{s_raw}⟩,   Φ(g) = ½[1 + erf(g/√2)].
```

The subtractive `log⟨e^s⟩` enforces `⟨e^s⟩ = 1` (the ρ₀ convention).
**Grounded:** Coles & Jones 1991 **Eq (21)**: for `χ = ρ/ρ₀`, `⟨χ⟩ = exp(σ²/2)`, and they
renormalize `χ → χ·exp(−σ²/2)` to keep the mean density fixed — identical to our `shift`.
(Existing code: `field/field.py:rank_copula_field` already does `s = s_raw − log⟨e^{s_raw}⟩`.)

## 3. The 2-point series (the central Phase-1 formula)

**Claim to implement:** for a local monotone transform `s = T(g)` of a unit Gaussian field with
normalized correlation `ρ_g(r) = ⟨g(x)g(x+r)⟩` (so `ρ_g(0)=1`),

```
ξ_s(r) = Σ_{n≥1} (c_n² / n!) · ρ_g(r)^n ,     c_n = ⟨ T(g) · He_n(g) ⟩ ,
```

with `He_n` the **probabilists'** Hermite polynomials (weight `φ(g)=e^{−g²/2}/√(2π)`,
`⟨He_n He_m⟩ = n! δ_{nm}`).

**Derivation (Mehler bivariate-Hermite expansion — classical, mathematically verifiable):**
For `(g,g')` bivariate standard normal with correlation `ρ`, Mehler's formula gives the joint
density `φ_ρ(g,g') = φ(g)φ(g') Σ_{n≥0} (ρ^n/n!) He_n(g)He_n(g')`. Expanding
`T(g)=Σ_n (c_n/n!)He_n(g)` (with `c_n=⟨T He_n⟩` by orthogonality) and integrating,
`⟨T(g)T(g')⟩ = Σ_{n≥0}(c_n²/n!)ρ^n`. Subtracting the `n=0` term (`c_0=⟨T⟩`) yields the series
above. ∎

**Held-PDF grounding via the exp special case (Coles & Jones 1991 Eq 30):**
Take `T(g)=e^{σg}` ⇒ (generating function, §4 below) `c_n = σ^n e^{σ²/2}`. Then
`Σ_{n≥1}(c_n²/n!)ρ^n = e^{σ²}(e^{σ²ρ}−1)`, so for the mean-1 field `χ=e^{σg−σ²/2}`:
`ξ_χ(r) = e^{σ²ρ_g(r)} − 1 = exp[Ξ(r)] − 1` with `Ξ(r)=σ²ρ_g(r)` the Gaussian covariance.
**This is exactly Coles & Jones 1991 Eq (30)** `1 + ξ(r) = exp[Ξ(r)]` (read from the PDF,
§5.1) — the general series reduces to their verified result. The cosmology pedigree of the
general (arbitrary-transform) version is Carron & Szapudi 2013 (transforms as order-by-order
polynomials).

**For our log-density `s`:** in the lognormal *body* `s` is linear in `g`
(`s = σ_s g − σ_s²/2`) ⇒ `c_1 = σ_s`, `c_{n≥2}=0`, `ξ_s = σ_s² ρ_g`. The BM19 power-law
*tail* makes `T` non-linear ⇒ `c_{n≥2} ≠ 0`. We compute `c_n` by Gauss–Hermite quadrature of
the BM19 map (Task 1.2).

## 4. Variance identity and the exp-map analytic test anchor

**Variance identity:** `ξ_s(0) = Σ_{n≥1} c_n²/n! = Var[s]` (= `σ_s² = ln(1+b²ℳ²)` in the
lognormal-body limit; slightly larger with the tail). Test `test_xi_s_at_rho1_equals_variance`.

**exp-map anchor (pins the Hermite convention before any oracle work):**
generating function `⟨e^{tg} He_n(g)⟩ = t^n e^{t²/2}` ⇒ for `T=e^{σg}`, `c_n = σ^n e^{σ²/2}`,
and `Var[e^{σg}] = Σ_{n≥1} c_n²/n! = e^{σ²}(e^{σ²}−1)`.
**Grounded:** Coles & Jones 1991 **Eq (17)** `μ'_n = exp(nμ + n²σ²/2)` (lognormal moments) —
the same generating-function structure. Tests: `test_cn_exp_map_matches_generating_function`,
`test_cn_linear_map_only_c1`, `test_cn_bm19_lognormal_limit_variance`.

## 5. Counts-in-cells: compound-Poisson distribution and likelihood (Phase 3 / Milestone 2)

**Compound-Poisson CIC count distribution:**
```
P(N) = ∫ Poisson(N | N̄·(1+δ)) · p(δ) dδ                    (continuous PDF → discrete counts)
```
**Grounded twice:**
- Szapudi & Pan 2004 **Eq (3)**: `P_N = ∫ p(δ) [⟨N⟩(1+δ)]^N e^{−⟨N⟩(1+δ)}/N! dδ`, with the
  Poisson kernel **Eq (4)** `K(N,δ)=[⟨N⟩(1+δ)]^N e^{−⟨N⟩(1+δ)}/N!` ("locally Poisson" sampling).
- Coles & Jones 1991 **Eqs (50)–(51)**: `λ=βρ` (Cox rate ∝ density) and
  `Pr(N=n)=∫ p(λ) P(n|λ) dλ`.

**CIC count likelihood (Milestone-2 1-pt block):** Szapudi & Pan 2004 **Eq (8)**
`L = Π_N (M P_N)^{M P̃_N} e^{−M P_N} / (M P̃_N)!` (M = number of cells; P̃_N = measured CPDF) —
a proper Poisson likelihood on the CIC histogram. This is the form Task 6.1 implements.

## 6. Smoothed log-density PDF p_R and the smoothing-scale framing (Task 3.2; Decision #1)

Szapudi & Pan 2004 **§2.2 ("Skewed Lognormal," SLN3)** model the *log-density* PDF as an
Edgeworth/Hermite series, **Eq (6)**:
```
p3(δ)dδ = [1 + (1/3!)T₃σ_Φ H₃(ν) + (1/4!)T₄σ_Φ² H₄(ν) + (10/6!)T₃²σ_Φ² H₆(ν)] G(ν) dν,
ν ≡ Φ/σ_Φ,  Φ = log ρ − ⟨log ρ⟩,
```
with **Eq (7)** `T₃ = ⟨Φ³⟩/σ_Φ³`, `T₄ = (⟨Φ⁴⟩−3σ_Φ⁴)/σ_Φ⁴`. This is the template for `p_R`
(a smoothed, skewed log-normal). **Smoothing-scale validation (Decision #1):** their **Table 1**
shows `σ_Φ` *decreasing with cell size R* (σ_Φ: 1.185 → 1.108 → 0.906 → 0.633 as
R: 2.21 → 4.42 → 8.83 → 17.66 h⁻¹Mpc) — empirical backing that the log-density variance is a
function of the smoothing scale R, exactly our regularization. (Our `σ_s²(R)` from Task 2.2.)

## 7. Why log-density, and the β connection (Decision: log-space 2-pt)

Carron & Szapudi 2013 (MNRAS 434, 2961) abstract + §1 (read): the information lost to the 2-pt
in the non-linear/large-variance regime is recovered by a **local non-linear transform**, and
"the optimal transform is essentially the simple power transform with an exponent related to the
**slope of the power spectrum**; when this is −1 it is indistinguishable from the **logarithmic
transform**." Carron & Szapudi 2014 (MNRASL 439, L11) extends sufficiency to **Poisson-sampled
lognormal (discrete)** fields. → grounds (a) using log-density as the near-sufficient statistic,
(b) the principled link between the transform and the spectral slope β.

## 8. Covariance and the survey window (Open Q #4 — Phase 5)

Carron, Wolk & Szapudi 2014 (MNRAS 444, 994): fast *exact* generation of lognormal-Poisson
fields producing "power spectra, counts-in-cells probability distributions as well as
**covariances** perfectly consistent with the data," and treats **supersurvey / beat-coupling
('super-sample') covariance** from the finite survey window via the non-linear transform. → the
reference for the analytic-covariance + mock cross-check (Task 5.1) and the survey-window caveat.

## 9. Fat-tail / moment caution (Decision #1 rationale)

Coles & Jones 1991 **Eqs (25)–(27)** + surrounding text (read): the lognormal "is not completely
determined by its moments"; the MGF integral **diverges** for the series expansion, and the
one-point PDF "should not decay any slower than an exponential at large χ" for moments to exist.
This is the foundational statement of the fat-tail/moment-divergence problem that our
smoothing-scale regularization (working in log-space + finite scale R) sidesteps.

## 10. Marginal-induced higher-order correlations (deferred 3-pt, Phase 8)

Coles & Jones 1991 **Eq (43)** `ξ_n(r_1,…,r_n) = Π_{i>j}[ξ(r_ij)+1] − 1` (Kirkwood/Groth–Peebles
hierarchical scaling) — the higher-order correlations of a lognormal/transformed-Gaussian field
are **derived** from its 1-pt + 2-pt. This grounds the §6b claim that the model's 3-pt is purely
*marginal-induced*, and is the analytic prediction for the deferred 3-pt null test.

## 11. [VERIFIED 2026-06-05] Limber projection + projected PDF (Task 2.3 grounding)

- **Projected (column) PDF narrowing — FK10 §3.5 CONFIRMED** (read `Federrath-2010.pdf` p9,
  "3.5. The column density PDFs and comparison with observations"). Verbatim: *"by computing
  projections of the volumetric density fields, density fluctuations are effectively averaged
  out by integration along the line-of-sight, and as a consequence, the column density
  dispersions become smaller compared to the corresponding volumetric density dispersions."*
  Quantified (Table 3 / Fig 7): the **column** log-density dispersion σ_η ≈ 0.46 (solenoidal) /
  1.51 (compressive) is smaller than the volumetric σ_s. So projection narrows the 1-pt PDF —
  the 1-pt effect for the projected CIC. (This is a **1-point** statement; FK10 gives no 2-pt
  kernel.) §3.4 (p8) also notes the high-density tail is resolution-dependent (un-converged even
  at 1024³) — supports the smoothing-scale regularization (Decision #1).
- **Limber 2-pt kernel** `w(r_⊥) = ∫ ξ_3D(√(r_⊥²+ℓ²)) dℓ` — classical (Limber 1953; Peebles
  1980). On the periodic grid this is the EXACT discrete identity: for a column field
  `Σ(x,y)=Σ_z f(x,y,z)`, the 2-D autocovariance `ξ_Σ(r_⊥)=Σ_{Δz} ξ_f(r_⊥,Δz)` (LOS-lag sum) —
  validated to machine precision against the realization oracle in Task 2.3.
- **[OPEN modeling choice, Task 2.3]** which field is projected: column density `Σ=∫ρ dℓ` is a
  **linear-ρ** integral (tail-sensitive ξ_ρ), whereas the β-carrier we measure cleanly is the
  **log-density** ξ_s. The CIC counts ∝ projected mass (Σ), so the projected *linear*-density
  2-pt is what sources the 2-D CIC; the smoothing scale R regularizes its tail (Decision #1).
  The Limber operator itself is field-agnostic; record the chosen sourced field in the design doc.

## 12. Net status

Everything Phase 1 (gaussianization ξ_s) and Phase 3 (CIC moments + compound-Poisson) needs is
grounded in held PDFs with equation-level citations above. Limber/projection (Phase 2) is the
only deferred item. The general 2-pt series is the classical Mehler expansion, **reducing to the
PDF-verified Coles & Jones Eq (30) in the exp case** and independently pinned by the exp-map test
anchor (§4) — so Phase 1 can proceed on solid, verified footing once Anna approves this memo.

### Design-doc corrections to apply (provenance)
- Replace "Szapudi & Pan 2004, ApJ 601, 697" → **"Szapudi & Pan 2004, ApJ 602, 26"** wherever it
  appears (design doc §5, References).
- Add the three Carron papers + their roles to the References block.
