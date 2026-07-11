# progenax — status

next: **Post-M1 improvement program in progress — Slice A (shipped-code fixes) 5 of 7 done; Slices B, C continue in a fresh session; repo flips public after Slice C.**

**M1 public-flip remediation** COMPLETE + merged to local `main` (unpushed): green gates, two theory-doc equation corrections (EFF enclosed mass, King dispersion direction), the LIMEPY silent-r_t-pinning guard, and internal-docs pruning. See the release audit (`audits/PROGENAX_PUBLIC_RELEASE_AUDIT.md`, maintainer-local).

**Slice A** (branch `feat/slice-a-shipped-code-fixes`, 6 commits, fast gate 1566 passed / 0 failed — ready to merge to `main` on the maintainer's go): S2 sqrt-stretched mass-CDF grid for `LIMEPYProfile` + Engine-A (core-mass error +49% → <2%); S3 re-fit stale `q_approx` calibration (+7% bias → ~0); S7 eager positive-input validation (`progenax.numerics.require_positive`, on Plummer/King); D1 binary-fraction gradient honesty (no cryptic crash + zero-gradient docstrings). **Remaining (delicate core-numerics, deferred to the fresh session, fully derived):** S4 α=1 IMF gradient (expm1-stable segment-integral rewrite) and Engine-B `density_poisson` grid (needs anchor re-validation).

**Provenance architecture decided (ADR-0034):** a machine-readable model-card registry becomes the single source of truth → generated glossary + enforcement test + Brain equation-digest drafts; public in-repo (glass-box), PDFs gitignored (DOI + arXiv as the public pointer); prove in progenax, then hoist to `jaxstro`. Design: `docs/plans/2026-07-10-provenance-registry-design.md`. Slice B = build it + populate via the theory-docs derivation sweep.

blocker: none. GitHub Actions CI is dormant and must be re-enabled before the first `v*` tag (tracked in the Slice-D release checklist).

due: none.

next (fresh session): `docs/plans/2026-07-11-next-session-prompt.md` — finish Slice A (S4, Engine-B) → Slice B (provenance registry + theory sweep) → Slice C (docs pedagogy + site aesthetics/MyST plugins) → write the Slice-D release checklist. Two source PDFs (Dejonghe 1987, Parravano 2011) are in `~/brain/_inbox/` for the sweep.

(Detailed arc-by-arc development history prior to 2026-07 lives in git history and maintainer-local notes.)
