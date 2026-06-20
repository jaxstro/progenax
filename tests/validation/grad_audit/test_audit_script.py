"""Task 0.4: scripts/audit_gradients.py emits required-key rows + valid JSON."""

import json

import pytest


@pytest.mark.slow
def test_run_audit_emits_required_keys(fresh_audit):
    rows, json_path = (
        fresh_audit  # rows + written file from the session's single regeneration
    )
    assert rows and {"id", "direction", "param", "ratio", "status", "ad", "fd"} <= set(
        rows[0]
    )
    json.loads(json_path.read_text())  # valid JSON
