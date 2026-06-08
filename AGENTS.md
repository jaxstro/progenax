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


## Brain hub

This repo is a spoke of `~/brain`. Full handshake in `CLAUDE.md` (Brain hub section) and
`~/brain/AGENTS.md`: pull-only hub, never edit `~/brain` from here, send notes home via
`brain "…"` (and `brain "xref: …"` for cross-cutting insights).

<!-- brain-handshake -->

<!-- brain-status-convention -->
## Brain status updates
When you make notable progress, hit a blocker, or set the next action, update this repo's `STATUS.md` (`next:` / `blocker:` / `due:` lines) — the brain pulls it into the portfolio dashboard + standup via `federate.py` (see `~/brain/work/meta/status-convention.md`). Brain stays pull-only: never hand-edit `~/brain`; capture events with `brain "…"`.
