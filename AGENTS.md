# AGENTS.md (Codex) - progenax

Guidance for Codex when working in `progenax`.

## Read First
- `CLAUDE.md`
- `README.md`
- `pyproject.toml`

## Units Policy
- `DEFAULT_UNITS`: `STELLAR` (Msun, pc, Myr)
- Core APIs require explicit `G` or `units`.
- Convenience wrappers may accept `units=None` and resolve to `DEFAULT_UNITS`.

## JAX Rules
- Use `jax.numpy` only in core code.
- Ensure `jax.grad` works through IC generation.

## Testing
- `pytest tests/ -v`
- `pytest tests/unit/ -v`
- `pytest tests/integration/ -v`
- `pytest tests/validation/ -v`

## Brain hub - this repo is a spoke of ~/brain (read-only from here)

- **Never edit `~/brain` from this session** - not hat homes, ADRs, configs, knowledge, or `_generated/`.
- **One write path home - the inbox, via capture** (works from any directory):
      brain "what happened - short, factual"
- **Cross-cutting insight** (something here that's also relevant to another project/paper)?
      brain "xref: <insight> - touches <other project / paper>"
  It becomes a brain concept and resurfaces there via `/brain-pack` (ADR-0019).
- **Full protocol + conventions:** read `~/brain/AGENTS.md` and `~/brain/guide/` before cross-session work
  (pull-only hub; spec -> session -> log handoffs, ADR-0018; modern mystmd if this is a MyST site).
- **Starting focused work here?** Pull a context pack from the hub: `/brain-pack <this-project>`.
- **Need papers/equations?** Start with that pack's Relevant literature and Equation-critical sources. Read
  source notes in `~/brain/knowledge/sources/`; verify exact equations/tables against
  `~/brain/knowledge/library/<bibkey>.pdf`; use `~/brain/knowledge/derived/equation-digests/` only when rows
  are verified; treat `~/brain/knowledge/raw/` as search-only. Capture needed source-note expansions back to the hub:
      brain "source-note update: <bibkey> - <what this package needs>"

<!-- brain-handshake: keep in sync with ~/brain/guide/how-to/set-up-a-project.md#spoke-stanza -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.
