# Note: `pytest --cov=<submodule>` aborts with the JAX stack — use `--cov=progenax`

**Opened:** 2026-06-03 · **Status:** ✅ Understood / documented (use the canonical invocation) ·
**Source:** 2026-06 follow-up environment hardening

## Symptom

A narrow coverage source — `pytest ... --cov=progenax.profiles.api` (or any
`--cov=<submodule>` / `--cov=<subpackage>`) — aborts with **exit code 134 (SIGABRT)** and no
pytest output:

```text
[globals.cc : 105] RAW: absl::log_internal::SetTimeZone() has already been called
Fatal Python error: Aborted
```

It is **not** a numpy "cannot load module more than once" double-load (an earlier guess during
the audit); the faulthandler stack shows the abort inside **jaxlib** (`jaxlib/xla_client.py`),
in **abseil**'s global logging init.

## Root cause

`coverage` resolves a **module/submodule** `source` spec by *importing* it (to map the module
to a file). With a narrow source like `progenax.profiles.api`, that import pulls in
`jax → jaxlib → xla_client`, which calls abseil's **one-time** global logging initialiser.
coverage's source handling causes the jaxlib import chain to run a **second** time in the same
process, and abseil's `SetTimeZone()` guard (`globals.cc:105`) calls `abort()` on the second
init → SIGABRT.

Minimal reproducer / control:

```python
import coverage
# ABORTS (exit 134):
coverage.Coverage(source=['progenax.profiles.api']).start(); import progenax.profiles.api
# CLEAN (exit 0):
coverage.Coverage(source=['progenax']).start();            import progenax.profiles.api
```

A **package** source (`progenax`) is resolved by **directory path**, not by importing a
submodule, so jaxlib is imported only once → no abort.

## Not macOS-specific

abseil's global-init guard is platform-independent C++, and coverage's module-source import is
platform-independent Python. So this is **not a macOS incompatibility** — it is specific to
narrow `--cov=<module>` scoping with the JAX stack and would occur on any OS. Reproduced here on
macOS 15 / Python 3.13.7 / jaxlib 0.10.1 / coverage 7.14.1. (CI runs on Linux and is unaffected
because it uses `--cov=progenax`, not because Linux is immune.)

## What to use (all work; CI is unaffected)

- ✅ `pytest --cov` — uses the configured `[tool.coverage.run] source = ["progenax"]` (pyproject).
- ✅ `pytest --cov=progenax` — explicit package scope (exactly what `.github/workflows/tests.yml` runs).
- ❌ `pytest --cov=progenax.<submodule>` — narrow scope, aborts. Do **not** use with the JAX stack.

For per-file/per-module coverage, run `pytest --cov=progenax --cov-report=term-missing` and read
the module's row in the report — this is how `profiles/api.py` was measured (37% → **100%** after
the 2026-06-03 follow-up tests) and the suite total at **86%**.

## Upgrade check (same investigation)

`uv lock --upgrade --dry-run` resolved all 33 packages with **no changes**: numpy **2.4.6**,
jax/jaxlib **0.10.1**, equinox **0.13.8**, diffrax **0.7.2**, jaxtyping **0.3.10** are already
the latest compatible on PyPI. Upgrading would neither change the pinned versions nor fix this
abort — it is a structural coverage×jaxlib interaction, not a version bug. The lockfile is
consistent (`uv lock --check` clean) and the local venv matches it (local == CI).
