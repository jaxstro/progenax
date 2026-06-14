"""Layer 2 staleness guard (design D2/Q3): the committed grad_audit_results.json must match a
fresh regeneration. EXACT on the discrete/structural projection (id/direction/param/theta/expect/
tol/finite/status); rtol-tolerant on the floats (ad/fd/ratio/abs_ad) because the committed JSON is
generated on macOS arm64 and CI regenerates on Ubuntu x86 — a literal byte-diff is infeasible.

_RTOL is PROVISIONAL. It is set to 2e-3 pending the Task-A5 cross-arch CI calibration: the first
`gradient-gate` CI run (Ubuntu x86) regenerates the JSON and its float deltas vs the committed
arm64 artifact are the real measured basis. Same-arch regeneration is ~bit-identical (max reldiff
observed ≪ 2e-3 — effectively the float64 ODE/AD noise floor), so 2e-3 carries ample same-arch
margin; cross-arch the noisiest cases are the ODE-solve derivatives (King/Michie r_t, Engine-A/B
sample_cluster). If A5's CI evidence shows a well-conditioned case exceeding 2e-3 purely from arch,
widen _RTOL to the measured max ×3 and record the measured deltas here. Do NOT loosen blindly — a
large reldiff on a closed-form case is a real drift, not arch noise.
"""
import json
import math
from pathlib import Path

from scripts.audit_gradients import run_audit, _DEFAULT_JSON  # run_audit(out_json) -> rows

# Exact-compared, arch-invariant structural fields. `tol` is the per-case tolerance from the
# registry (a discrete input, not a measured float) — a hand-edit loosening a gate that does NOT
# flip `status` would otherwise slip through, so it is compared exactly here.
_DISCRETE = ("id", "direction", "param", "expect", "tol", "status")
_FLOAT = ("ad", "fd", "ratio", "abs_ad")
_RTOL = 2e-3   # PROVISIONAL — see module docstring; calibrated cross-arch in Task A5.
# Non-vacuous floor: the registry currently has 66 rows; guard against a silently-emptied
# registry/JSON (an emptied set passes cset==fset trivially — the classic staleness blind spot).
_MIN_ROWS = 60


def _key(row):
    return (row["id"], row["param"], round(float(row["theta"]), 12))


def test_committed_json_matches_fresh_regeneration(tmp_path):
    committed = json.loads(Path(_DEFAULT_JSON).read_text())
    fresh = run_audit(out_json=tmp_path / "fresh.json")
    # Floor first: a comparator over two empty sets passes vacuously; pin that neither side
    # collapsed (a zeroed registry is itself a drift the gate must catch).
    assert len(committed) >= _MIN_ROWS and len(fresh) >= _MIN_ROWS, (
        f"too few rows (committed={len(committed)}, fresh={len(fresh)}; floor={_MIN_ROWS}) — "
        f"the registry/JSON looks emptied, not merely drifted.")
    cset, fset = {_key(r) for r in committed}, {_key(r) for r in fresh}
    assert cset == fset, (
        f"row-set drift (cases added/removed/retheta'd):\n  only committed: {sorted(cset - fset)}"
        f"\n  only fresh: {sorted(fset - cset)}\n  -> regenerate + recommit the JSON.")
    cby, fby = {_key(r): r for r in committed}, {_key(r): r for r in fresh}
    drift = []
    for k in cby:
        c, f = cby[k], fby[k]
        for field in _DISCRETE:
            if c[field] != f[field]:
                drift.append(f"{k} {field}: committed={c[field]!r} fresh={f[field]!r}")
        if bool(c["finite"]) != bool(f["finite"]):
            drift.append(f"{k} finite: committed={c['finite']} fresh={f['finite']}")
        for field in _FLOAT:
            cv, fv = float(c[field]), float(f[field])
            c_fin, f_fin = math.isfinite(cv), math.isfinite(fv)
            if c_fin != f_fin:
                # A clean<->inf/nan transition is exactly the regression a staleness gate exists
                # to catch (core.py emits ratio=inf when fd==0, ad!=0; nan if a grad blows up).
                drift.append(f"{k} {field}: finiteness changed committed={cv!r} fresh={fv!r}")
            elif not c_fin:  # both non-finite -> require the same kind (inf / -inf / nan)
                if repr(cv) != repr(fv):
                    drift.append(f"{k} {field}: non-finite kind changed "
                                 f"committed={cv!r} fresh={fv!r}")
            else:
                denom = max(abs(cv), abs(fv), 1e-30)
                if abs(cv - fv) / denom > _RTOL:
                    drift.append(f"{k} {field}: committed={cv:.6e} fresh={fv:.6e} "
                                 f"(reldiff={abs(cv - fv)/denom:.2e} > rtol={_RTOL:.0e})")
    assert not drift, "staleness drift (regenerate + recommit JSON if intended):\n  " + \
        "\n  ".join(drift)
