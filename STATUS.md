# progenax — status

**Gravoturb Aim-2/3 extensions arc (2026-07-18, branch `feat/gravoturb-extinction-multiplicity`; awaiting Anna merge):**
**A (gas→extinction→fluxax, Aim 3) COMPLETE** — `GravoturbDustModel` (differentiable star-embedded LOS gas
column + Rémy-Ruyer+2014 metallicity-keyed dust-to-gas via `BirthEnvironment`), duck-typed into fluxax's
`DustModel.column` slot. fluxax got **2MASS J/H/Ks bands** (per-curve coverage: G23 excluded at H/Ks — a real
Gordon+2023 low-R_V NIR pathology) + **A0 axis extended 20→150 mag** (byte-identical below 20) — both on
fluxax branch `feat/add-2mass-ks-band` (full suite 1239 passed). Aim-3 g→K reveal figure
(`scripts/demo_extinction_bands.py`; median A_g=18→A_K=2.2, recovery 19%→79%, natal t=0 pre-feedback). Extinction
verified physically correct (standard A_V=1 at 21 M⊙/pc²). ADR-0055.
**C (λ_mult env-coupled multiplicity, Aims 1-2) COMPLETE** — `blended_system_placement` (Gaussian-copula: mass
strength λ_corr × multiplicity strength λ_mult → density), `CompositionSpec.lambda_mult` (None = byte-identical
legacy path). Measured emergent mass-channel baseline first; whole-system placement preserves marginals + Moe
joint, no mass-conditioning violation. Byte-gate 60✓, experimental unit 439 passed. ADR-0056.
B/Quokka are much-later funded-work (NOT this arc). Next: Anna's merge decision + optional Aim-1 binary figure.

next: **REPO IS PUBLIC (flipped 2026-07-11). Slice C merged + pushed (release gate PASSED at merge: 1651/2skip, cov 96.18%, 24/24 validation scripts). Anna executes the Slice-D punch list (`docs/website/95-release/checklist.md` §Slice D): CI re-enable + 3.10-matrix fix, CONTRIBUTING.md, then at the tag CITATION.cff + Zenodo DOI, sdist excludes, check.sh path — and after the tag, PyPI (jaxstro first).**

**M1 public-flip remediation** COMPLETE + merged to local `main` (unpushed): green gates, two theory-doc equation corrections (EFF enclosed mass, King dispersion direction), the LIMEPY silent-r_t-pinning guard, and internal-docs pruning. See the release audit (`audits/PROGENAX_PUBLIC_RELEASE_AUDIT.md`, maintainer-local).

**Slice A COMPLETE** (branch `feat/slice-a-shipped-code-fixes`, 12 commits; trimmed fast gate 1580 passed / 0 failed in 6.5 min): S2 sqrt-stretched mass-CDF grids for `LIMEPYProfile` + Engine-A (+49% → <2%) + Engine-B (`density_poisson` Poisson grid AND the position-CDF resampling grid: +2.5%/+17% → +0.004%/+0.12%; anchors re-validated 12/12); S3 `q_approx` recalibration; S4 α=1 IMF gradient fixed outright via expm1-stable kernels in `progenax.numerics` (AD/FD ratio 1.0000000 at exactly α=1; grad-audit 98 cases, 97 clean, 0 hazards; 3 formerly-known_blocked edges now `consistent`); S7 constructor validation; D1 binary-fraction honesty. **2026-07-10 test retirement (Anna-approved, quality over quantity):** OED demo suites deleted (74 → 5; the tooling migrated to informax; progenax keeps ONE basic demo — binary-robustness, "binaries RVs vs. inferred cluster mass" — with 5 load-bearing pins), ZAMS finite-only grad smokes deleted (registry-duplicated), dashboard restamped; fast gate 13.4 → 6.5 min. OED docs pages carry deprecation admonitions; Anna verifies the informax port before the demos/pages themselves are removed. **Merge precondition:** one `scripts/release_gate.sh` run (coverage restamp) at merge time.

**Slice B COMPLETE (2026-07-11, branch `feat/slice-b-provenance-registry`, 10 commits):** the ADR-0034
provenance registry is built, enforced, and fully populated. 7 family YAMLs → 28 model cards → generated
glossary (`docs/website/15-model-reference/`, in the site TOC) + the 5th enforcement registry
(`tests/validation/provenance_cards/`: bibkey→DOI/eprint/adsurl, code_refs import, validation node ids
collect AND assert, verified⇒code+validation, coverage ratchet vs MODEL_INVARIANTS with REGISTRY_FULL=True
= 21/21 locked forever, glossary freshness; teeth proven by mutation). references.bib: 100% public
pointers (all verified live). Theory-docs sweep findings, all fixed: rotation-anisotropy.md documented a
FABRICATED differential-rotation API (real curve: v_peak(R/R_peak)exp(1−R/R_peak)); lowered-model-family.md's
GZ15 "misprint" admonition was itself fabricated (main-text Eq. 8 prints E_γ(g+3/2) CORRECTLY; the 2018
erratum fixes Eqs. 20-21+41); plummer-dfs.md Eddington Φ/Ψ notation; PRNG key-reuse on 5 snippets. New
source verifications: Dejonghe 1987 Eq. 43 (the 3π/64 σ_los oracle), Michie 1963 Eq. 5.8 (scanned, visual),
Maschberger Table 1 + Parravano Eq. 1 (+ 2 new per-paper notes, brain-named PDFs in docs/core-papers/).
House guidance encoded: Maschberger preferred over Kroupa/Chabrier (smooth C^∞ + closed-form quantile).

**Provenance architecture decided (ADR-0034):** a machine-readable model-card registry becomes the single source of truth → generated glossary + enforcement test + Brain equation-digest drafts; public in-repo (glass-box), PDFs gitignored (DOI + arXiv as the public pointer); prove in progenax, then hoist to `jaxstro`. Design: `docs/plans/2026-07-10-provenance-registry-design.md`. Slice B = build it + populate via the theory-docs derivation sweep.

**Slice C COMPLETE (2026-07-11, branch `feat/slice-c-docs-pedagogy`):** C1 mechanical docs fixes; C2
pedagogy pass (Anna-approved page by page): 13 publication-quality figures from the new modular **ICViz**
library (`laboratory/icviz/` — FigureSpec registry + CLI, StarViz-derived seaborn theme, triple export
PDF/PNG gitignored + WebP embedded), figures double as correctness proofs (residual panels, escape-envelope
oracles, binary resolved-vs-unresolved σ-inflation), worked Eddington derivation, exercises on ~8 theory
pages, ONE canonical reading ramp; C3 site aesthetics: API pages regenerated as structured cards (Google-
docstring parameter tables + model-card backlinks on 37 symbols + gradient-verified badges from the
grad-audit JSON), glossary cards carry meta rows (status · counts · API backlink), landing-page hero
figure, and a MyST plugin (`docs/website/plugins/card-links.mjs`) opening `#card-*` glossary links in a
new tab (verified in built AST, 17 pages). Docs gate PASS 212 pages / 0 warnings; fast gate 1586 passed;
provenance suite 6/6.

blocker: none. GitHub Actions CI is dormant and must be re-enabled before the first `v*` tag (tracked in the Slice-D release checklist).

due: none.

next: Anna executes Slice D (checklist on the site, `95-release/checklist.md`) → tag `v0.1.0` → PyPI (after jaxstro publishes). Still on Anna's desk: review/commit `.brain-drafts/` digests to ~/brain; verify the informax OED port before removing the deprecated demo pages. Two source PDFs (Dejonghe 1987, Parravano 2011) are in `~/brain/_inbox/` for the sweep.

**Magnetized-turbulence arc COMPLETE + MERGED to local main (2026-07-19, branch
`feat/gravoturb-magnetic`, 15 commits).** `MagneticSpec` on `build_cluster_ic` — μ_Φ (mass-to-flux)
is the single primary knob; B₀/ℳ_A/β₀/r_A are derived and logged (ADR-0060). Layers: **L1** magnetic
σ_s² (Molina 2012 / F&K12, exact via b_eff = b√(β₀/(β₀+1))); **L2** velocity anisotropy with
*pluggable* r_A(ℳ_A) closures — theory (Hu & Lazarian 2021, ℳ_A^{-4/3}, sourced), phenomenological,
or empirical override (ADR-0061); **L3** divergence-free vector **B** grid for RMHD seeding (Nyquist-
plane fix ⇒ ∇·B ~ 4e-16). **s_crit collapse-threshold channel un-deferred** (ADR-0063): the α_vir-free
magnetothermal-Jeans shift Δs=ln(1+1/β₀) (F&K12 Eq.21) — magnetic support reduces the collapse-eligible
fraction, and a strongly sub-critical cloud drives it below the requested SFE so the gas solver refuses
("too much flux ⇒ SF ceases", emergent). **Ambipolar** flux-loss closure (static, not a solver) rescues
SF. Constants sourced, not memorized (c_Φ=0.17/√G PN11 Eq.16; κ=½ Molina). `magnetic=None` is
byte-identical (GRAVOTURB_BYTE_GATE=1: 490 passed). AC-MAG1..8 acceptance suite 8/8 PASS + 3-panel
publication money figure (`validation/magnetic_money_figure.py`). Ohmic/Hall scoped out permanently.
NEXT ARC: Bonnor-Ebert/polytropic **gas** envelope replacing the stellar Plummer (ADR-0062).

**Gravoturb finalization arc (started 2026-07-16, branch `feat/gravoturb-finalization`):** CAREER
Aim 1 science audit DONE (verdict: preliminary-figure only until gaps close; note in `audits/`,
maintainer-local) → six-phase finalization design RATIFIED + adversarially reviewed (A1–A5
amendments applied; `docs/plans/2026-07-16-…`, maintainer-local). **Phase 0 hardening COMPLETE**
(5 commits: AC-IC0–IC6 committed record incl. new envelope-fidelity map, Gravax seam test 4/4,
guide deprecation banner, byte-identity pins; baseline 350 passed/1 xfail). **Phase 0.5 COMPLETE**
(4 commits: package renamed `gravoturb_fdf`→`gravoturb` w/ physics-descriptive module+symbol
names, field.py split, spec objects CloudSpec/GeometrySpec/VelocitySpec/CompositionSpec, curated
top-level API; byte-identity 9-pin gate PASSES through the new API — zero behavior change proven;
full experimental suite 368 passed/1 xfail; docs 213 pages/0 warnings incl. a pre-existing
model-card backlink fix + new builders_cluster API page; released core untouched, `git diff main
-- src/progenax` empty). **Phase 1 COMPLETE** (multi-freefall placement default: p⋆ ∝ w·ρ^{3/2}
per FK12 Eq. 7/8 verified vs held PDF; f_sub → derived differentiable `f_sub_derived`; gate
AC-IC7 PASS incl. independent-numpy-oracle KS=0.006 + A4 AC-IC0 re-run vs placement-consistent
reference; caveat recorded: ≥64³ at ℳ≥8; unit tier 357 passed). **Comprehensive review + remediation COMPLETE (2026-07-16, Anna-directed "everything now"):**
8-angle adversarial review + architecture assessment → 10 verified findings, all fixed in 4
commits: honest placement fractions (tail_star_fraction + collapse_eligible_fraction replace the
conflated f_sub_derived — >2× semantic bug), spec regressions (Q_target≥0, traced construction),
byte-identity gate hardened (exact fingerprint + GRAVOTURB_BYTE_GATE strict mode), docs drift +
FDF de-jargoning, layering fixed (measure→diagnostics; model.py factory; 2-D β = ACTIVE headline
per Anna), tests renamed to module mirrors, both monoliths split, AC-IC1/IC4 now validate BOTH
placement modes (AC-IC4 multi-freefall initially FAILED at 32³ — root-caused to the ≥64³-at-ℳ≥8
caveat, fixed by resolution not thresholds; placement_n_eff diagnostic added). Gates: 380 passed
+1 xfail strict; acceptance 11/11; docs 213pp/0 warnings. **Phase 2 COMPLETE (2026-07-16, TDD):**
physical velocity mode `VelocitySpec(mode='physical', c_s=…[km/s], eta_v=1)` — σ_⋆ = η_v·ℳ·c_s
via `scale_to_dispersion` after COM removal, **Q_virial emergent**, BM92 `alpha_vir` diagnostic
on-path (both modes; 1-D literature convention per the 2026-07-16 audit — fiducial α_vir≈0.18),
builder takes `units=` for the km/s→pc/Myr conversion
(G-consistency checked, no silent precedence), `cloud_spec_from_larson` closes (M_ecl, SFE,
ρ_cl) → (ℳ, β, b, box=2R_cloud) through the released Larson chain. Gate **AC-IC8 PASS**
(σ round trip ≤2e-16; emergent-Q grid monotone in ℳ and r_h, fiducial Q≈0.10–0.16 strongly
SUBVIRIAL — the physical cold-birth regime; Q∝η_v² exact; units pin 0.97779; physical-mode
gravax seam 6/6); acceptance now 12/12; `virial_target` byte-identical (rename pins re-passed).
**Post-Phase-2 audit (2026-07-16, Anna-directed):** 3-angle review (plan-adherence + physics + architecture),
zero critical; fixes committed (α_vir 1-D convention, traced-η_v guard, units-any-mode, committed IMF
artifact, layering policy, Larson σ_v0 convention; gates 403+1xfail strict / 13/13 acceptance / released
1587). **FrameTransform ledger** added (star↔grid COM/velocity map recorded — the frames were silently
unreconcilable); inference diagnostics/SBC split at seams + hardened (loud guards, RED-first); strict
test mirrors. **Aim 2 handoff RATIFIED into the plan (Phase 4a,** design addendum in the maintainer-local
plan doc): Helmholtz first → single `TurbulentCloudIC` replacing ClusterIC (nested blocks, gas=None
star-only path, no shim), field-first velocity normalization (volume-weighted σ_g=ℳc_s exact; stellar
σ emergent-with-scatter), ε⋆=1−exp(−τ⋆w/t_ff) partition (IFT-differentiated bisection), gates AC-G1–G8
in a new gas_acceptance.py. **Phase 3 Helmholtz coupling COMPLETE (2026-07-16, TDD):** `theory/driving.py` (χ_F10 = b/√3,
PDF-verified — F10 Eq. 22's radical is over D only; Eq. 23 is the forcing-side cubic, NOT the
inversion; χ never reaches 1 for forced turbulence) + `realization/helmholtz.py` (one white field →
P∥/P⊥ projectors with per-mode compressive fraction exactly χ → ĝ ∝ −∇·v, **β derived = β_v−2**,
corr(g,−∇·v)=1 by construction) + `CloudSpec(coupling='helmholtz', beta=None)` with builder-entry
`validate_spec_bundle` (ADR-0041 Option A). Gate **AC-IC9 5/5 PASS** (new `coupling_acceptance.py`;
derived slope |err|<0.05; C=corr·√(E_long/E_tot) tracks √χ; AC6/σ_⋆/AC-IC4 re-pass coupled at
unweakened thresholds; carrier→2.0 + coupled≡independent convergence; infall signature +1.74 vs
−0.006 ablation). TDD caught 3 instrument defects (mixed-Nyquist transversality leak → planes
zeroed; scale-invariant draft gate statistic → amplitude-weighted C, correction surfaced; integer-|k|
binning slope bias → mode-level regression). `independent` mode byte-identical (pins re-passed).
**Phase 4a stars+gas handoff COMPLETE (2026-07-16, TDD; gates pending final suite confirmation):**
`realization/gas.py` (normalization ρ_cl=M_cl·ρ̃/∫ρ̃dV exact; ε⋆=1−exp(−τ⋆w/t_ff) partition; τ⋆ by
scan-bisection + IFT derivative, AD/FD 6e-11) + `GasSpec` + **`TurbulentCloudIC` replaces ClusterIC
outright** (nested stars/gas/fields/geometry/physics/ledger; gas=None star-only; no shim; all callers
+tests cut over, byte pins re-passed) + **field-first physical velocities** (gas grid σ_g=ℳc_s exact
<1e-10; stellar σ emergent, characterized 0.75–0.89±0.07–0.14 — COM removal strips coherent bulk;
AC-IC8a re-scoped from measurement) + joint stars+gas COM/momentum frame. Gate **AC-G1–G8 8/8**
(new gas_acceptance.py): closure ≤3e-15, pointwise conservation 2e-16, SFE reproduced ≤8e-16,
w·ρ^{3/2} limit CV 4e-4, refusals loud. Physics found: the freefall partition CAPS reachable SFE at
the collapse-eligible share (falls with ℳ: ~0.79/0.36/0.16 at ℳ=4/8/12), over-ceiling RAISES;
fully-consumed cells (ρ_g=0) counted. **Phase 4b composition COMPLETE (2026-07-16, TDD; gate AC-IC10/IC11 2/2** in new
composition_acceptance.py): λ_corr primordial segregation (Spearman +0.03→+0.997 across the sweep,
Λ_MSR 1.20→10.4, off-path byte-identical, fold_in key), binaries barycenter-first
(CompositionSpec.companions = any released CompanionModel; system masses drive dynamics + gas
contract; amplitude before resolve_binary_components; stars.system_id provenance;
binary_energy_budget printed — Q_com 0.234 vs Q_resolved 0.445 scale separation), per-cell local
IMF **defensibility-REFUSED** vs the held Marks+2012 PDF (α₃ is a GLOBAL cloud-core relation; the
cluster-level env_to_imf_params + masses-first route documented instead). **Phase 5 A1 windowing COMPLETE (2026-07-16, 9720265):** envelope distortion characterized per
channel (β immune ≤2%; ℳ count-channel +0.06–0.14; α tail flooded 44→150 cells) → survey-style
treatment: known-intensity detrend + DECLARED effective-volume mask (n̄≥n_min; unmasked detrending
diverges 0.31-vs-0.02 in the wings) + `envelope_cell_intensity` forward-model side; transfer
statement verified (turbulence excess enveloped ≡ periodic within scatter). **Phase 5 CLOSED for the CAREER deliverable (2026-07-17):** scope reset to FEASIBILITY
(demonstrate a validated WIP instrument, NOT complete the science). Shipped
`feasibility_figure.py` (1ce26c1) — a 6-panel publication figure: cloud+stars / residual gas /
primordial segregation (real IMF) / BM19 PDF / χ_F10 coupling / validation scorecard with the WIP
status line. The two heavy Phase-5 jobs (symmetric-Hermite production run + AC-IC12 NUTS coverage
campaign) were KILLED mid-run and are **committed drivers + a DEFERRED TODO** (Anna: optimize/
refactor later — gravax must be perfected first, and inference should route through informax's
pipeline). This makes gravoturb a **cross-package pipeline-validation strategy**: gravoturb IC →
gravax dynamics → informax inference. Pilot findings recorded for the resume: production IC is
near-cold (Q≈0.01) hyper-clumped; unsoftened adaptive Hermite dt→1e-9 (impossible), unsoftened
fixed-dt symmetric explodes, order-4 symmetric+ε=Δx/4 is stable (~1e-5). AC-IC12 needs the α-tail
grid + informax port before it gates.

**CAREER "money figure" FINALIZED (2026-07-17) — proposal Fig. 4.** `feasibility_figure.py` reworked
into the controlled 3-panel deliverable: (a) parent cloud + stars, (b) λ_corr=0 vs (c) λ_corr=0.6 at
IDENTICAL cloud/positions/IMF (only the mass–gas coupling differs) — the matched segregated/unsegregated
control. Star field renders five physical ZAMS observables (Tout+1996): colour=spectral type,
size=radius, α=depth, two-tier M-dwarf-haze + resolved composite, dark separators. Proposal styling:
no axes (each panel (6 pc)², stated in caption), no figure title, enlarged panel/colorbar text,
**rasterized dense layers + 200 dpi → PDF 3.2 MB → 0.6 MB, loads instantly**. Caption drafted (throughline
+ cite-keywords for Codex; ρ_S/Q_0 dropped for clarity; α_ρ shown as p(s)∝e^{−α_ρ s}). Diagnostic:
ρ_S(mass, local gas density) ≈ λ_corr (0.00→0.60), logged not annotated.

**gravoturb impact/extension planning (2026-07-17, brainstorm w/ Anna).** Framing: gravoturb = a
**controlled natal-imprint generator** (each channel a testable "memory" of birth). Two design docs
(on-disk, docs/plans gitignored; durable record in memory + brain xref): (1) standalone-package vision
(extract the differentiable field engine + multi-format exporters — Quokka/Athena++/FLASH — for MHD-sim
ICs; POST-CAREER; already named in the proposal Broader Impacts); (2) Aim-2/Aim-3 extensions spec
grounded in the project-description PDF. **RATIFIED next arc (FRESH session, new branch): implement A + C.**
A = residual gas → differential extinction → fluxax (Aim 3): physical star-embedded LOS geometry ×
metallicity-keyed dust-to-gas (δ_dg∝Z tied to `BirthEnvironment`/env-IMF; low-Z ⇒ top-heavy IMF AND
less extinction; g-vs-K reveal), couples to fluxax's existing `DustModel.column` slot. C =
environment-coupled multiplicity (Aims 1–2): `λ_mult` over {f_bin, period, q}, Moe baseline
byte-identical, measure emergent (mass-channel×λ_corr) coupling first; reuses the binary stack. B
(feedback η_p, Aim 2) + Quokka comparison are MUCH later, not in the A/C arc.

**Gravoturb-finalization branch MERGED to local main (2026-07-17).** 55 commits; released-core delta is
docstring-only (gravoturb_fdf→gravoturb rename + σ_v convention note in cluster/turbulence.py). Fast gate
re-run green at merge. main unpushed (Anna's call). A/C proceed on a NEW branch off main in a fresh session.

(Detailed arc-by-arc development history prior to 2026-07 lives in git history and maintainer-local notes.)
