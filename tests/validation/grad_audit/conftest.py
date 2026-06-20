"""Shared grad-audit regeneration fixture (Task 3.1 consolidation).

Both ``test_json_fresh`` (staleness diff vs the committed JSON) and ``test_audit_script``
(required-keys + valid-JSON) need a fresh ``run_audit()`` regeneration. A full regeneration is
~400 s, so running it twice (once per test) wastes ~one full regeneration in the FULL gate. This
session-scoped fixture regenerates the grad-audit JSON **once** into a session tmp path and hands
both tests the same ``(rows, json_path)`` — the rows the engine returned and the file it wrote.
Neither test's assertions change; only the duplicate ``run_audit()`` call is removed.

Both consumers remain ``@pytest.mark.slow`` (this single regeneration is itself heavy and
FULL-gate-only).
"""

from pathlib import Path

import pytest
from scripts.audit_gradients import (
    run_audit,  # run_audit(out_json) -> rows; also writes out_json
)


@pytest.fixture(scope="session")
def fresh_audit(tmp_path_factory) -> tuple[list[dict], Path]:
    """Regenerate the grad-audit JSON exactly ONCE for the whole session.

    Returns ``(rows, json_path)``:
    - ``rows``: the list of audited-entry dicts returned by ``run_audit`` (the in-memory result
      both tests assert on).
    - ``json_path``: the file ``run_audit`` wrote, so the valid-JSON check can read it back.
    """
    json_path = tmp_path_factory.mktemp("grad_audit") / "fresh.json"
    rows = run_audit(out_json=json_path)
    return rows, json_path
