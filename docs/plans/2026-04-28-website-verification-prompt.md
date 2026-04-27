# Prompt: progenax website verification session

**Use this prompt** to kick off a new Claude Code session whose mission
is to verify the freshly-authored progenax website against the actual
codebase, identify gaps and aspirational claims, and produce a
prioritised punch list of fixes. The website was authored largely by
Claude in a single session on 2026-04-28; this verification session
should be done by a fresh agent with no prior bias from the authoring.

---

## Prompt to paste

> You are verifying the progenax documentation website at
> `docs/website/` against the actual progenax codebase at `src/progenax/`.
> The website was authored on 2026-04-28 in a multi-hour session that
> migrated existing material and authored new chapters covering theory,
> architecture, validation, bibliography, getting-started, and
> auto-generated API reference. The authoring agent (Claude Opus) self-
> reported high confidence on physics + architecture but flagged that
> *specific claims* — function names, numerical values, "planned"
> features described as if real — need independent verification.
>
> **Read these orientation artefacts first:**
>
> 1. `docs/plans/2026-04-28-progenax-website-design.md` — the rollout plan
>    with phase definitions and the per-source-doc migration mapping.
> 2. `docs/notes/2026-04-28-pp20-fix.md` — the canonical example of a
>    bug that survived for months because tests were anchored on the
>    buggy formula's output rather than the published reference. The
>    same anchor-pattern problem may exist in other modules.
> 3. `progenax/CLAUDE.md` and `CLAUDE.md` — the package-specific and
>    ecosystem-wide development conventions.
> 4. `docs/website/myst.yml` — the website TOC; gives the full chapter
>    list.
> 5. `docs/website/_build/html/index.html` (after `cd docs/website && myst build --html`) — preview the rendered site.
>
> **The verification mission has four parts:**
>
> ### Part 1 — code-vs-docs audit per module
>
> For each public progenax module — `profiles`, `kinematics`, `imf`,
> `binaries`, `analytical`, `builders`, `tidal`, `populations`,
> `gravoturb`, `protocols` — produce a status table answering:
>
> 1. Every Python identifier mentioned in the corresponding website
>    chapter — does it actually exist in the code? Run a grep audit:
>    extract all backticked symbols (e.g. `BinaryIMF.with_period_conditional`,
>    `apply_mass_segregation_baumgardt`, `progenax.cluster.fdf`) from
>    the relevant theory and architecture chapters and check each via
>    `grep -r` in `src/progenax/`. Flag aspirational names.
> 2. Every numerical claim in the docs — does it match the actual test
>    output? Cross-reference against `tests/validation/test_*.py` and
>    `validation/imf/*` and `validation/turbulence/*`. Re-run validation
>    tests where needed (be patient — some take minutes).
> 3. Every code snippet in the docs — does it import + run? Spot-check
>    the longer examples (like the binary-IMF NumPyro snippet, the
>    two-component-cluster pipeline, the FDF positional displacement).
>
> Produce one **module status table** per module, with columns:
> - **Symbol / claim** — the thing being checked.
> - **Status** — ✓ correct / ✗ wrong / ⚠ aspirational / ❓ ambiguous.
> - **Evidence** — file path + line number that supports the status,
>   or the actual code if it differs from docs.
> - **Action needed** — fix docs, fix code, file an issue, or no action.
>
> The PP20 ζ(p) bug fix at `progenax/gravoturb/pp20_magnification.py`
> is the exemplar of "✗ wrong, action: anchor on published value."
> The mass-segregation `λ_seg` blending described in the docs may be
> aspirational — verify.
>
> ### Part 2 — validation suite completeness audit
>
> The website at `50-validation/` claims a three-tier test architecture
> with specific tolerances and pass/fail criteria. For each per-suite
> validation page (`plummer-equilibrium.md`, `king-profile.md`, …,
> `analytical-test-cases.md`):
>
> 1. Does the corresponding test file actually exist at
>    `tests/validation/test_<name>.py`? List the missing ones.
> 2. Do the listed properties have actual tests? E.g.
>    `test_plummer_physics.py::test_virial_ratio` — confirm.
> 3. Are the pass/fail tolerances in the docs the same as what the
>    tests actually use?
> 4. Are the spot-result tables (e.g. "Λ_MSR = 1.92 ± 0.13 at
>    λ_seg = 1.0") consistent with running the test?
>
> Produce a **validation status table** with columns:
> - **Page** — `50-validation/<name>.md`.
> - **Test file present?**
> - **Properties claimed → actually tested?**
> - **Numbers match?**
> - **Action needed.**
>
> ### Part 3 — bibliography accuracy audit
>
> For each entry in `docs/website/references.bib` (27 entries) and the
> 20 per-paper detail pages under `docs/website/99-bibliography/per-paper/`:
>
> 1. Verify the DOI resolves to the right paper.
> 2. Verify volume / page numbers match ADS.
> 3. Verify the paraphrased abstract is faithful (not distorted).
> 4. Verify the "Use in progenax" cross-refs point at real chapters
>    that actually cite the paper.
>
> This is the lightest part — most BibTeX entries are canonical and
> stable. Focus the time on the per-paper abstracts.
>
> ### Part 4 — gap analysis (open-ended brainstorming)
>
> Some things are likely missing from the docs that *should* be there.
> Brainstorm with the user (Anna, alrosen@sdsu.edu) about:
>
> 1. **Topics covered in the code but not the docs.** Walk
>    `src/progenax/` and surface any module / function / class that
>    has substantial logic but no chapter discusses it.
> 2. **Topics covered in the docs but the implementation status is
>    unclear.** The docs describe several "planned" features (LIMEPY
>    backend, stellax integration, period-conditional BinaryIMF, etc.) —
>    establish a clear "implemented vs planned vs aspirational"
>    classification for each.
> 3. **Per-paper bibliography gaps.** Are there 5–10 papers that the
>    code clearly cites in docstrings but don't have a per-paper
>    detail page yet?
> 4. **Cross-link density.** Are any chapters orphaned (no inbound
>    links from anywhere)? Use a graph-build to surface them.
> 5. **The Phase E tutorial gap.** The three "executable notebook
>    coming later" tutorials still need to be promoted to `.ipynb` form;
>    is the prose actually correct enough to convert as-is, or does
>    it need rewriting first?
> 6. **Validation regenerations.** The PP20 fix flagged
>    `b5_zeta_comparison.png`, `b6_pp20_diagram.png`, and
>    `e5_pp20_diagram.png` as needing regeneration. Are there other
>    plots / committed JSON outputs that need similar refresh?
>
> Use the `superpowers:brainstorming` skill for this part — present
> findings and proposed actions one topic at a time, get Anna's
> sign-off, and produce a prioritised punch list.
>
> ### Deliverables
>
> By the end of the session:
>
> 1. **A status report** at `docs/reviews/2026-XX-XX-website-verification.md`
>    containing the four status tables (one per module audit, one
>    validation table, one bibliography table) and the gap-analysis
>    discussion.
> 2. **A punch list** at `docs/plans/2026-XX-XX-website-fix-plan.md`
>    with prioritised fixes (high / medium / low) and estimated effort
>    per item.
> 3. **An honest summary paragraph** at the top of the status report:
>    "The website is X% accurate; the most important fixes are
>    A, B, C; Phase E (tutorials) is/is-not safe to start now."
>
> Do not silently fix things during the audit — the goal is to *catalogue*
> first, fix later. Anna will triage the punch list herself.
>
> Use the `superpowers:requesting-code-review` skill if you want a
> second-agent review of any specific module's audit. Use the
> `manuscript-workflow:numbers` skill for the bibliography accuracy
> sub-audit if it helps systematise the cross-checking.

---

## What this prompt is *not*

The verification session **should not silently fix things** during the
audit. The goal is to produce a clear, prioritised punch list of work
that Anna can triage. Fixing should happen in a separate session
(possibly with `superpowers:execute-plan` consuming the punch list).

The verification session **should not re-author chapters** — even if
a chapter's wording is awkward, the audit's job is to catalogue
whether it's *factually correct*, not whether the prose could be
better. Style is a separate concern.

The verification session **should brainstorm** in Part 4 with Anna
present. Parts 1–3 are mechanical and don't need conversation. Part 4
explicitly does.

## Suggested session length

A thorough audit takes 4–8 hours of agent time:
- Part 1 (per-module audit, 10 modules): 2–3 hours.
- Part 2 (validation completeness, 12 pages): 1–2 hours, includes
  re-running tests.
- Part 3 (bibliography accuracy, 27 entries): 1 hour.
- Part 4 (gap-analysis brainstorming): 1–2 hours, depends on how much
  Anna pushes back on findings.

Run it across one or two sessions; it does not need to be done in a
single sitting.

## Reference: today's authoring footprint

For context, the 2026-04-28 authoring session produced:

```
docs/website/
├── 00-getting-started/  (7 pages)
├── 10-theory/           (32 pages)
├── 20-architecture/     (9 pages)
├── 30-api/              (12 auto-generated pages, 135 symbols)
├── 40-howto/            (6 stubs, not authored)
├── 50-validation/       (14 pages)
├── 90-development-log/  (13 absorbed entries)
├── 99-bibliography/     (22 pages including 20 per-paper)
├── myst.yml             (TOC + bib config)
├── references.bib       (27 entries)
├── scripts/build_api_reference.py  (stdlib inspect-based)
└── _build/html/         (376 rendered pages)
```

The 376-page rendered site is the audit target. The verification
session should produce a status report covering the entire surface.
