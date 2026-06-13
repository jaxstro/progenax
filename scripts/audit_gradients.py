"""Run the grad-audit REGISTRY and emit validation/data/grad_audit_results.json + a markdown
table. The website doc cites this JSON; the pytest gate asserts on the same engine (design D1)."""
import json
import sys
from dataclasses import asdict
from pathlib import Path

# Direct invocation (`python scripts/audit_gradients.py`) puts only scripts/ on sys.path;
# the repo root must be present so `tests.validation.grad_audit.*` resolves. (Harmless under
# pytest / `-m`, where the repo root is already on the path.)
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import progenax  # noqa: E402,F401  (float64)
from tests.validation.grad_audit.core import audit_entry_point  # noqa: E402
from tests.validation.grad_audit.registry import REGISTRY  # noqa: E402

_DEFAULT_JSON = Path(__file__).resolve().parents[1] / "validation" / "data" / "grad_audit_results.json"


def run_audit(out_json: Path = _DEFAULT_JSON) -> list[dict]:
    rows = []
    for c in REGISTRY:
        rows.append(asdict(audit_entry_point(c)))
        for e in c.edges:
            rows.append(asdict(audit_entry_point(
                c, theta=e.theta0, tol=e.tol or c.tol, expect=e.expect or c.expect)))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = "| id | dir | param | ratio | finite | status |\n|---|---|---|---|---|---|\n"
    body = "".join(
        f"| `{r['id']}` | {r['direction']} | {r['param']} | {r['ratio']:.6f} | "
        f"{r['finite']} | {r['status']} |\n" for r in rows)
    return head + body


if __name__ == "__main__":
    rows = run_audit()
    print(to_markdown(rows))
    n_haz = sum(r["status"] == "hazard" for r in rows)
    print(f"\n{len(rows)} cases; {n_haz} hazard(s).")
    raise SystemExit(1 if n_haz else 0)
