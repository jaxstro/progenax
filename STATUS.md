# progenax — status

next: **Slice B (ADR-0034 provenance registry) POPULATED — 28 model cards, 21/21 registry coverage locked (REGISTRY_FULL), glossary live, 24 Brain digest drafts staged in .brain-drafts/ for Anna. Branch feat/slice-b-provenance-registry awaiting merge go. Then Slice C (docs pedagogy + aesthetics); repo flips public after Slice C.**

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

blocker: none. GitHub Actions CI is dormant and must be re-enabled before the first `v*` tag (tracked in the Slice-D release checklist).

due: none.

next: Anna reviews the glossary + commits .brain-drafts/ digests to ~/brain → merge Slice B → Slice C (docs pedagogy + site aesthetics/MyST plugins) → write the Slice-D release checklist. Two source PDFs (Dejonghe 1987, Parravano 2011) are in `~/brain/_inbox/` for the sweep.

(Detailed arc-by-arc development history prior to 2026-07 lives in git history and maintainer-local notes.)
