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
