"""Task 0.4: scripts/audit_gradients.py emits required-key rows + valid JSON."""


def test_run_audit_emits_required_keys(tmp_path):
    from scripts.audit_gradients import run_audit
    rows = run_audit(out_json=tmp_path / "r.json")
    assert rows and {"id", "direction", "param", "ratio", "status", "ad", "fd"} <= set(rows[0])
    import json
    json.loads((tmp_path / "r.json").read_text())  # valid JSON
