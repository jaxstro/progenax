"""Check the release gate's recorded-exit predicate without running its workloads."""

import ast
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def status_program():
    """Extract only the status predicate from its existing shell-heredoc owner."""
    path = Path(__file__).resolve().parents[2] / "scripts/release_gate.sh"
    script = path.read_text()
    marker = "$RUN python - \"$FULL_SUITE_GREEN\" <<'PY'\n"
    program = script.split(marker, 1)[1].split("\nPY", 1)[0]
    tree = ast.parse(program)
    body = []
    for item in tree.body:
        if isinstance(item, ast.FunctionDef) and item.name == "_code":
            body.append(item)
        elif isinstance(item, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in {"val_fails", "validation_scripts_pass"}
            for target in item.targets
        ):
            body.append(item)
    assert len(body) == 3, "Release status owner moved; update the focused extractor."
    return compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")


@pytest.mark.parametrize(
    "statuses, expected_pass, expected_failures",
    [
        ({"zero": 0}, True, []),
        ({"nested": {"exit": 0}}, True, []),
        ({"false": False}, False, ["false"]),
        ({"true": True}, False, ["true"]),
        ({"nested_false": {"exit": False}}, False, ["nested_false"]),
        ({"timeout": "timeout"}, False, ["timeout"]),
        ({"unknown": "unknown"}, False, ["unknown"]),
        ({"missing": None}, False, ["missing"]),
        ({"float": 0.0}, False, ["float"]),
        ({"signal": -15}, False, ["signal"]),
        ({"no_exit": {}}, False, ["no_exit"]),
        ({"zero": 0, "timeout": "timeout"}, False, ["timeout"]),
        ({}, False, []),
    ],
)
def test_release_requires_recorded_integer_zero(
    status_program, statuses, expected_pass, expected_failures
):
    """Reject timeout/unknown/bool/missing/empty evidence instead of false PASS."""
    state = {"val_scripts": statuses}
    exec(status_program, state)
    assert state["validation_scripts_pass"] is expected_pass
    assert state["val_fails"] == expected_failures
