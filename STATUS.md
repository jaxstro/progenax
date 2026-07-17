# progenax — status

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
Next: **Phase 4a stars+gas handoff** (ratified), 4b composition, 5 identifiability + production run.

(Detailed arc-by-arc development history prior to 2026-07 lives in git history and maintainer-local notes.)
