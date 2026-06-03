# WS2 doc-accuracy batch — gate packet

**Date:** 2026-06-03 · **Branch:** `docs/ws2-accuracy-batch` · **Base:** `main @ 8941bd6` ·
**Reviewer:** Claude Opus 4.8 · **Source:** D0 read-only doc-audit fan-out (4 section agents) ·
**Scope:** accuracy-harden existing `docs/website/` content (NOT authoring stubs) ·
**Status:** 🚦 **AT GATE 2** (diff + clean MyST build below)

## Decisions (Gate 1)

- **Counts:** de-hardcode — replace brittle exact counts in evergreen pages with qualitative /
  file references; keep exact numbers only in dated changelog entries.
- **Bib naming:** rename the cite key `Moe2017` → `MoeDiStefano2017` (matches the
  `moe-distefano-2017.md` per-paper page; touches the bib + all citations).
- **Scope:** do all of W1–W10 as one PR.

## Findings ledger + resolution

| id | finding | sev | resolution |
|----|---------|-----|------------|
| W1 | `30-api/` not regenerated after the monolith split → 14 broken `analytical/core.py` source links | High | **Fixed** — regenerated via `build_api_reference.py` (run from `docs/website/`); `analytical.md` stale links 14→0; also refreshed signatures incl. Batch 0 (`tidal.md`, `builders.md`, `imf.md`). |
| W2 | `whats-new.md` test-count contradiction (949 vs 874) | Major | **Fixed** — de-hardcoded the conflicting raw suite totals; kept the `86 % → 91 %` coverage narrative. |
| W3 | `code-reviews.md` present-tense framing of RESOLVED bugs | Major | **Fixed** — added a dated `Status: every finding below was RESOLVED` banner atop the follow-up-audit section. |
| W4 | `plummer-equilibrium.md` brittle "12 validation tests" | Major | **Fixed** — de-hardcoded (kept the test-file reference). |
| W5 | `moe-distefano-2017.md` ↔ `Moe2017` cite-key mismatch | Major | **Fixed** — renamed cite key to `MoeDiStefano2017` across `references.bib` + ~17 doc files; `Moe2019` untouched; build resolves all. |
| W6 | "135 symbols" (per-module) vs 48 (top-level) unexplained | Minor | **Fixed** — de-hardcoded `30-api/index.md` + added a clarifying note to the generated full-symbol-index. |
| W7 | "Phase E notebook coming later" admonitions | Minor | **No change** — accurate (notebooks intentionally deferred; matches the Phase-E memory). |
| W8 | glossary cross-ref anchor | Minor | **Deferred** — low value; noted. |
| W9 | `CLAUDE.md` "57+ exports" | Minor | **Fixed** — de-hardcoded heading to "Public API". |
| W10 | `code-reviews.md` "874 collected" historical count | Minor | **Covered** by the W3 banner (the section is a dated audit snapshot). |

## Verification (Gate 2 evidence)

- **MyST build:** `myst build` (v1.9.0) → **clean, 0 warnings/errors**; no unresolved cite keys
  or broken cross-references (validates the W5 rename + W1 regen end-to-end).
- **W1:** `grep -c analytical/core.py 30-api/analytical.md` → **0** (was 14).
- **W5:** `references.bib` key `MoeDiStefano2017`; **0** lingering bare `Moe2017`; `Moe2019` intact (9).
- **Diff:** 31 files, +199 / −162 (mostly the cite-key rename + API regen).

## Gate status

- **G1 — punch-list + decisions:** ✅ approved (de-hardcode; rename cite key; all W1–W10).
- **G2 — diff + clean build before commit:** 🚦 **awaiting Anna** (this packet).
- **G3 — branch + green CI before PR/merge:** ☐
