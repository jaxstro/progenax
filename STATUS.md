# progenax — status

next: **M1 public-flip remediation COMPLETE — ready to change the GitHub repo private → public.** M1 cleared the four release blockers from the 2026-07 pre-release audit:
1. **Green gates** — re-froze the stale test-registry characterization goldens (line-number pins that drifted when an earlier docstring cleanup shifted a source file); value-multiset parity proven, only line numbers moved.
2. **Docs-science corrections** — the EFF theory page's enclosed-mass block was mathematically wrong; replaced with the exact hypergeometric $M(<r)=\tfrac{4\pi}{3}\rho_0 r^3\,{}_2F_1(\tfrac32,\tfrac\gamma2;\tfrac52;-r^2/a^2)$ and Gamma-function $M_{\rm total}$ (verified to 1e-13), plus the King velocity-dispersion direction (increases with $W$, so $\sigma$ *falls* with radius — cold outskirts). The `EFFProfile`/`KingVelocityDF` code was already correct; only the docs were wrong.
3. **LIMEPY truncation guard** — general-g isotropic LIMEPY and Engine-A (`MultiComponentCluster`) models whose true tidal radius exceeds the ODE domain (e.g. $W_0=9,\,g=2$, true $\xi_t\approx2132$) were silently built with $r_t$ pinned to the domain edge — a wrong tidal radius, mass CDF, and DF scale. Now they refuse eagerly (King-parity guard + `r_t_is_pinned` diagnostic).
4. **Internal-docs pruning** — untracked the maintainer-local working documents (`docs/plans`, `docs/notes`, `docs/archive`, `.adr`, `.claude-work`) from the public tree (kept on disk + in history) and scrubbed ~100 dangling references from live code/docs; trimmed CLAUDE.md and this file to public form.

Follow-on fixes: restored the OED demo's `d_`/`a_criterion` re-exports (a latent bug the plot-free dev env had masked), declared a `[viz]` extra (`matplotlib`) for the standalone `scripts/validate_*.py` figure scripts, and marked the OED demo CLI smoke tests `@slow`. **Local release gate PASSES** (registries full · line-cov 96.13% ≥ 90 · dashboard fresh · full suite green · 24/24 validation scripts); docs gate 178 pages / 0 warnings.

blocker: none. GitHub Actions CI is dormant and must be re-enabled before the first `v*` tag (see M2).

due: none.

next (M2, before tagging v0.1.0): re-enable the three CI workflows and fix the physics-validation matrix's stale Python-3.10 leg (`requires-python` is `>=3.11`); add `CITATION.cff` + mint a Zenodo DOI on the tag; add an sdist include/exclude (the sdist currently bundles docs + internal files); fix the hardcoded local path in `scripts/check.sh`; add `CONTRIBUTING.md`. Deferred: optimal-experimental-design work lives in the sibling **informax** package (the OED machinery + a `progenax_kinematic_oed` worked example + a progenax-parity test); progenax keeps only a docs-overview pointer. A full removal of progenax's OED demo scripts/tests (deferring entirely to informax) is a candidate future cleanup.

(Detailed arc-by-arc development history prior to 2026-07 lives in git history and maintainer-local notes.)
