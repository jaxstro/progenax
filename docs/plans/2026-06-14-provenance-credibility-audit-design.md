# Design — Provenance & credibility audit (consolidation rides along)

> **Status:** design (brainstorm complete + ratified 2026-06-14). The inventory → fetch →
> verify → ledger work happens in a **fresh session** against this design.
> Predecessor arc: the gradient-gate lock-in (merged to `main`, `1ee9d89`).

## Why this arc

progenax is being prepared for release — **paper + collaborators now, public PyPI soon**
("both, sequenced"; Anna, 2026-06-14 Q1). For a *scientific* release the highest-stakes
integrity work is credibility: a fabricated citation or a constant that does not trace to its
source is a landmine in a methods paper. There is a **known pattern of hallucinated content from
older AI-authored sessions** — fake LIMEPY tables, a debunked King Table II, phantom
`IGIMF`/`EnvironmentIMF` classes — so a systematic provenance audit is expected to catch live
problems, not just produce a clean bill.

The packaging blocker (R2 — `jaxstro` is not on PyPI, so `pip install progenax` cannot resolve
today) is **deferred to a later dedicated arc** (public-PyPI step). This arc is paper-facing
credibility, with maintainability/consolidation riding along.

## Locked decisions (do not relitigate)

- **D1 — Arc = credibility + consolidation sweep** (chosen over R2 packaging-strategy spike and the
  methods-paper figures as the next arc; Anna Q2).
- **D2 — Provenance-led; consolidation rides along** (Anna Q3). The spine is verifying scientific
  claims against held primary sources; dead-code/stale-doc retirement happens opportunistically in
  the same module pass, not as a co-equal phase.
- **D3 — Triage-first, deep-verify the high-risk set** (Anna Q4). Inventory → risk-score →
  deep-verify high-risk against PDFs + spot-sample the rest → fix/flag + ledger. Escalate to
  comprehensive only if triage reveals systemic rot.
- **D4 — `needs-fetch` is a first-class verdict** (Anna, 2026-06-14). When verification needs a
  paper not held, STOP and ask Anna to fetch it; never verify a primary-source fact from memory or
  a review agent. The audit surfaces a consolidated "papers to fetch" list early.

## Method — four stages

1. **Inventory (broad, parallelizable read-only agents).** Three lists:
   (a) every cited paper across docs + docstrings, tagged *held PDF* vs *cited-but-not-held*;
   (b) every paper-traceable constant/coefficient in `src/` (Moe Table 13, Marks α₃ coefficients,
       King/LIMEPY/Plummer/EFF indices, IMF breakpoints/exponents) + its current source comment
       (or "none");
   (c) every numeric / "Measured" / "Table" claim in the theory + validation docs.
2. **Risk-score.** Jump to deep-verify: cited-but-not-held papers; constants with no source
   comment; numbers more precise than their source could justify; content introduced in an
   AI-authored commit (`git blame`/`git log`); anything matching the known fabrication fingerprints
   (fake LIMEPY table, phantom classes, debunked Table II).
3. **Deep-verify the high-risk set against the actual PDFs** (read the paper, confirm the
   number/equation/claim). Spot-sample the low-risk set to estimate the residual error rate.
4. **Fix-or-flag + ledger.** *Fix in place* when the PDF gives an unambiguous correction (the
   commit cites the PDF). *Flag for Anna* anything needing scientific judgment or where the right
   value is ambiguous. *needs-fetch* when the PDF is not held. Record every claim in the ledger.

## Three audit surfaces (highest-risk first)

1. **Citations ↔ per-paper notes** — 50 per-paper notes / 74 held core-paper PDFs (134 PDFs under
   `docs/`). Is each cited paper's PDF held? Do the note's claimed equations/tables/values match
   the PDF? (Historical fabrication has hidden here.)
2. **Hardcoded scientific constants in `src/`** — the curated paper-traceable set; each must trace
   to a specific paper equation/table, verified against the PDF.
3. **Numeric / "Measured" / "Table" claims in theory + validation docs.**

## Deliverables

- **`provenance-ledger.md`** (location TBD: `.claude-work/` or `docs/`) — one row per audited item:
  `location (file:line)` → `source (paper + eq/table/page)` → `verdict` ∈
  {✅ verified, 🔧 fixed, ⚠ flagged-for-Anna, 📄 needs-fetch, 🗑 dead-removed}.
- **In-place fixes**, each commit naming the PDF that justifies it.
- A **consolidation changelog** of retired dead/stale code + docs.
- A consolidated **"papers to fetch"** list, surfaced early.

## Phasing & HITL checkpoints

1. **Inventory + risk-score** (broad; fan out read-only agents over the three surfaces) → present
   the scored list + the `needs-fetch` batch. **CHECKPOINT:** Anna fetches the missing PDFs.
2. **Deep-verify the high-risk set** against held PDFs, module by module → fix the unambiguous,
   flag the judgment calls. **CHECKPOINT per batch:** Anna adjudicates every ⚠ flag and approves
   every 🗑 deletion before it lands.
3. **Spot-sample** the low-risk set; **ledger + close-out.**

## Verification (non-negotiable)

- No scientific value changes unless a **held PDF** justifies it (never from memory or a review
  agent). Every fix commit names its source.
- `make build` (website) stays **0 warnings**.
- The **full released-core gate stays green** after any consolidation deletion.
- HITL: Anna adjudicates every flag and approves every deletion; nothing science-facing or
  destructive lands without her sign-off.

## Done when

Every high-risk claim is ✅/🔧/⚠/📄-resolved; the ⚠ flags are adjudicated by Anna; the fetch list
is closed (or explicitly deferred); the ledger is committed; dead code retired with Anna's
approval; build + released-core gate green.

## Skills / discipline

`research-workflow:provenance-of-constants` (the constants backbone), the paper-grounding workflow
(real PDF → verify the per-paper note → verify the code), `astro-code-dev` (astrophysics
correctness), and the no-assumptions rule (read the actual PDF; never assert a primary-source fact
from memory or a review agent). Branch off `main` (`feat/provenance-credibility-audit`); commit per
verified batch; HITL at every checkpoint.
