"""Task 1.6 staleness gate (the ratchet): the committed ``validation/data/
test_dashboard.json`` must match a fresh IN-PROCESS regeneration. This is what
makes the dashboard non-drifting by construction — edit a test, change a registry,
or move a validate-script, and this gate goes RED until the JSON is regenerated.

Mirrors ``grad_audit/test_json_fresh.py``: regenerate via ``build_dashboard()``,
semantic-diff vs the committed JSON — EXACT on the discrete/structural fields
(module test counts, registry status, gate flags, validation-script exit codes,
durations/line-coverage status), rtol-tolerant on the only floats (per-module
``line_cov`` percentages). The ``generated_utc`` timestamp is IGNORED (it always
differs; we pass a fixed sentinel so even a literal compare of that field would be
benign).

INTROSPECTION-ONLY / COST: ``build_dashboard`` calls ``collect_test_inventory()``,
which shells out to ``pytest --collect-only`` (~10-35s). That is the ONLY heavy
part — it does NOT run the suite or ``--cov``. Measured wall-time here is >15s, so
this test is ``@pytest.mark.slow``: it re-collects the whole suite, which is full-
gate work, not fast-inner-loop work. The FAST gate stays fast; the FULL gate
catches dashboard drift.

_RTOL is for the line-coverage floats. As of Task 2.2 ``line_coverage`` is
``"measured"`` (the committed full-suite ``coverage.json`` exists) and the
src-bearing module rows carry a real per-directory ``line_cov`` percentage. Same-
process regeneration re-parses the SAME committed ``coverage.json`` byte-for-byte,
so the deltas are exactly zero; 1e-6 is an ample float64 round-trip margin. If CI
ever regenerates coverage on a different arch, widen to the measured max (do NOT
loosen blindly — a large reldiff on a same-suite parse is a real drift, not noise).
"""
import json
import math
import subprocess
import time

import pytest

from scripts._dashboard_render import render_dashboard_page
from scripts.build_test_dashboard import (
    _DASHBOARD_JSON,
    _DASHBOARD_PAGE,
    _REPO_ROOT,
    build_dashboard,
)

# Stamped verbatim into generated_utc by build_dashboard; the gate IGNORES this
# field, so any fixed value works. A literal sentinel makes the intent explicit.
_FIXED_TS = "1970-01-01T00:00:00+00:00"

# rtol for the line-coverage floats (per-module line_cov %). See module docstring.
_RTOL = 1e-6

# Anti-vacuous floor (mirrors grad-audit's _MIN_ROWS): a semantic diff over two
# silently-emptied censuses passes trivially. Pin a sane minimum so a collapsed
# inventory is caught as drift, not waved through. The live census has 40+ module
# buckets and ~1500 collected tests; these floors carry ample margin while staying
# count-agnostic (they needn't track the exact totals).
_MIN_MODULES = 30
_MIN_TESTS = 1000

_TIER_KEYS = ("unit", "integration", "validation")


def _total_tests(dashboard: dict) -> int:
    return sum(
        m[tier] for m in dashboard["modules"].values() for tier in _TIER_KEYS
    )


def _diff_floats(label: str, cv, fv, drift: list) -> None:
    """Compare two line_cov values: None==None exact; floats rtol-tolerant."""
    if cv is None or fv is None:
        if cv != fv:
            drift.append(f"{label}: committed={cv!r} fresh={fv!r}")
        return
    cv, fv = float(cv), float(fv)
    c_fin, f_fin = math.isfinite(cv), math.isfinite(fv)
    if c_fin != f_fin:
        drift.append(f"{label}: finiteness changed committed={cv!r} fresh={fv!r}")
    elif not c_fin:
        if repr(cv) != repr(fv):
            drift.append(f"{label}: non-finite kind changed committed={cv!r} fresh={fv!r}")
    else:
        denom = max(abs(cv), abs(fv), 1e-30)
        if abs(cv - fv) / denom > _RTOL:
            drift.append(
                f"{label}: committed={cv:.6e} fresh={fv:.6e} "
                f"(reldiff={abs(cv - fv) / denom:.2e} > rtol={_RTOL:.0e})"
            )


def _diff_modules(committed: dict, fresh: dict, drift: list) -> None:
    """Per-module: EXACT on tier counts, rtol on line_cov, exact module-set."""
    cmods, fmods = committed["modules"], fresh["modules"]
    cset, fset = set(cmods), set(fmods)
    if cset != fset:
        drift.append(
            f"module set drift: only committed={sorted(cset - fset)} "
            f"only fresh={sorted(fset - cset)}"
        )
    for module in sorted(cset & fset):
        c, f = cmods[module], fmods[module]
        for tier in _TIER_KEYS:
            if c[tier] != f[tier]:
                drift.append(
                    f"modules[{module!r}][{tier!r}]: committed={c[tier]!r} fresh={f[tier]!r}"
                )
        _diff_floats(f"modules[{module!r}]['line_cov']", c["line_cov"], f["line_cov"], drift)


def _head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# The measured package prefix. Coverage measures ONLY src/progenax (the released
# wheel); a change anywhere else (tests, scripts, docs, experimental) cannot invalidate
# the line-coverage of unchanged source. So the freshness diff is scoped to this path.
_SRC_PREFIX = "src/progenax/"

# The full-suite selector the floor/freshness gate requires (mirrors the api-coverage
# manifest's FULL_SELECTOR — a `-m "not slow"` pass understates coverage).
_FULL_SELECTOR = "full"


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command at the repo root (NOT check=True — callers read returncode)."""
    return subprocess.run(
        ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
    )


def assert_coverage_provenance_fresh(line_coverage: dict, head_sha: str) -> None:
    """Phase-2 provenance teeth: a MEASURED coverage block must be SRC-FRESH at HEAD.

    RATIONALE (Task 2.2 correctness fix). The naive check ``git_sha == HEAD`` is
    unworkable: it forces a ~14-min full-suite ``--cov`` re-measure on EVERY commit,
    including test-only / docs-only commits that cannot change the line-coverage of
    ``src/progenax``. A test-only commit only ever RAISES coverage (more tests exercise
    the same unchanged source), so the committed floor stays a valid LOWER BOUND — there
    is nothing to invalidate.

    The right invariant is SOURCE-based: the committed coverage is STALE iff the measured
    SOURCE changed since it was measured, i.e.::

        git diff --name-only <coverage_provenance.git_sha> HEAD -- src/progenax/

    is NON-EMPTY. Adding/removing TESTS does not invalidate it. We never re-run ``--cov``;
    we only diff the source tree, which is cheap.

    Two stale conditions raise (a measured block must NOT pass silently if either holds):

    1. ``selector != "full"`` — a partial run that understates coverage.
    2. The measured ``git_sha`` is not an ANCESTOR of HEAD (e.g. a rebased/orphaned sha,
       or a sha from a sibling branch). We cannot compute a meaningful src diff against a
       commit HEAD did not descend from, so we conservatively treat it as stale with a
       clear message rather than silently passing.
    3. ``git diff <git_sha> HEAD -- src/progenax/`` is non-empty — the measured source
       moved.

    When not-measured there is no stamp -> no-op. Extracted so the regression test can
    exercise the teeth with a sha after which a ``src/progenax`` file changed.
    """
    if line_coverage.get("status") != "measured":
        return
    provenance = line_coverage["coverage_provenance"]

    selector = provenance.get("selector")
    assert selector == _FULL_SELECTOR, (
        f"stale coverage.json: selector={selector!r} (not {_FULL_SELECTOR!r}) — the "
        f"committed coverage must come from the FULL suite, not a partial run."
    )

    git_sha = provenance["git_sha"]
    # git_sha must be an ancestor of HEAD for the src diff to be meaningful. `git
    # merge-base --is-ancestor A B` exits 0 iff A is an ancestor of B. A non-zero exit
    # (1 = not ancestor; 128 = bad/unknown sha) => we cannot trust the diff => stale.
    ancestor = _git("merge-base", "--is-ancestor", git_sha, head_sha)
    assert ancestor.returncode == 0, (
        f"stale coverage.json: provenance git_sha={git_sha} is not an ancestor of "
        f"HEAD={head_sha} (git exit {ancestor.returncode}) — the measured commit is not "
        f"on HEAD's history; re-measure FULL-suite coverage at the current commit."
    )

    diff = _git("diff", "--name-only", git_sha, head_sha, "--", _SRC_PREFIX)
    changed = [ln for ln in diff.stdout.splitlines() if ln.strip()]
    assert not changed, (
        f"stale coverage.json: {len(changed)} {_SRC_PREFIX} file(s) changed since the "
        f"coverage was measured at {git_sha} (e.g. {changed[:3]}) — re-run the FULL-suite "
        f"--cov and re-stamp. (Test-only changes do NOT invalidate coverage; source ones do.)"
    )


def _page_minus_timestamp(page_text: str) -> list[str]:
    """The page lines with the volatile ``Generated at ...`` line dropped.

    The rendered page embeds ``Generated at `<utc>`.`` which always differs (like
    the JSON's ``generated_utc``). The page-freshness gate ignores exactly that one
    line so it compares structural content, not the timestamp.
    """
    return [
        line for line in page_text.splitlines()
        if not line.startswith("Generated at `")
    ]


def _last_src_changing_parent() -> str | None:
    """The PARENT of the most recent commit that touched ``src/progenax/``.

    Coverage stamped at THIS sha is, by construction, STALE: a ``src/progenax`` file
    changed in the very next commit, so ``git diff <parent> HEAD -- src/progenax/`` is
    non-empty. Returns ``None`` if the history is too shallow to have such a parent
    (then the stale-on-src-change leg is skipped — the synthetic-sha legs still run).
    """
    last_src = _git("log", "-1", "--format=%H", "--", _SRC_PREFIX).stdout.strip()
    if not last_src:
        return None
    parent = _git("rev-parse", f"{last_src}^").stdout.strip()
    if not parent:
        return None
    # Sanity: confirm this really is a stale anchor (src changed between parent..HEAD).
    diff = _git("diff", "--name-only", parent, _head_sha(), "--", _SRC_PREFIX)
    return parent if diff.stdout.strip() else None


def _measured_block(git_sha: str, selector: str = "full") -> dict:
    """A committed-shaped MEASURED ``line_coverage`` block with the given stamp."""
    return {
        "status": "measured",
        "total_percent": 92.3,
        "per_module": {"builders": 90.0},
        "per_file_lines": {"builders": {"covered_lines": 90, "num_statements": 100}},
        "coverage_provenance": {
            "selector": selector,
            "git_sha": git_sha,
            "total_percent": 92.3,
            "measured_utc": "2026-06-14T00:00:00+00:00",
        },
    }


def test_coverage_provenance_teeth_fire_on_stale_src():
    """C1 (src-based): prove the provenance teeth bite on a SOURCE change, not on tests.

    The freshness invariant is src-based (Task 2.2): coverage is stale iff a
    ``src/progenax`` file changed since it was measured. We assert:

    - FRESH: stamped at HEAD (no src diff HEAD..HEAD) -> passes.
    - STALE on src change: stamped at the parent of the last src-touching commit (a real
      ``src/progenax`` file changed since) -> RAISES.
    - STALE on non-ancestor sha: an all-zero sha is not on HEAD's history -> RAISES.
    - STALE on partial selector: a non-"full" selector -> RAISES (understates coverage).
    - not-measured -> no-op (no stamp to check).
    """
    head = _head_sha()

    # FRESH: stamped at HEAD; diff HEAD..HEAD over src/progenax is empty -> no raise.
    assert_coverage_provenance_fresh(_measured_block(head), head_sha=head)

    # STALE on a real src change: parent of the last src-touching commit.
    stale_src_sha = _last_src_changing_parent()
    if stale_src_sha is not None:
        with pytest.raises(AssertionError, match=r"file\(s\) changed since"):
            assert_coverage_provenance_fresh(_measured_block(stale_src_sha), head_sha=head)

    # STALE on a non-ancestor (orphaned/unknown) sha -> the merge-base leg raises.
    with pytest.raises(AssertionError, match="not an ancestor"):
        assert_coverage_provenance_fresh(
            _measured_block("0" * 40), head_sha=head
        )

    # STALE on a partial selector -> the selector leg raises (even at a fresh sha).
    with pytest.raises(AssertionError, match="not 'full'"):
        assert_coverage_provenance_fresh(
            _measured_block(head, selector="tests/unit -m 'not slow'"), head_sha=head
        )

    # not-measured is always a no-op (no stamp to check).
    assert_coverage_provenance_fresh({"status": "not-measured"}, head_sha="whatever")


def test_committed_page_matches_render_of_committed_json():
    """I1: the rendered page must match a fresh in-process render of the committed
    JSON (ignoring the volatile ``Generated at ...`` line). A drifted/stale page —
    e.g. one missing a module row — fails this gate.
    """
    committed_json = json.loads((_REPO_ROOT / _DASHBOARD_JSON).read_text())
    committed_page = (_REPO_ROOT / _DASHBOARD_PAGE).read_text()
    fresh_page = render_dashboard_page(committed_json)
    assert _page_minus_timestamp(committed_page) == _page_minus_timestamp(fresh_page), (
        "rendered test-dashboard.md is stale vs the committed JSON — re-render with "
        "`build_test_dashboard.py --render` and recommit the page."
    )


@pytest.mark.slow  # re-collects the whole suite (~10-35s); FULL-gate work, see docstring.
def test_committed_dashboard_matches_fresh_regeneration():
    committed = json.loads((_REPO_ROOT / _DASHBOARD_JSON).read_text())
    t0 = time.perf_counter()
    fresh = build_dashboard(_FIXED_TS)
    elapsed = time.perf_counter() - t0
    print(f"\n[dashboard staleness gate] in-process regeneration took {elapsed:.1f}s")

    # Anti-vacuous floor FIRST: a semantic diff over two emptied censuses passes
    # vacuously, so pin that neither side collapsed before trusting the diff.
    for side, dash in (("committed", committed), ("fresh", fresh)):
        n_mod, n_test = len(dash["modules"]), _total_tests(dash)
        assert n_mod >= _MIN_MODULES and n_test >= _MIN_TESTS, (
            f"{side} dashboard too small (modules={n_mod}, tests={n_test}; "
            f"floors modules>={_MIN_MODULES}, tests>={_MIN_TESTS}) — the census "
            f"looks emptied, not merely drifted."
        )

    drift: list[str] = []

    # IGNORE generated_utc (always differs); EXACT-compare every other top-level
    # block except `modules`, which has its own float-aware comparator. registries,
    # gate, line_coverage, durations, validation_scripts are all discrete/structural
    # JSON (status strings, ints, bools, exit codes) -> a plain == is the gate.
    _diff_modules(committed, fresh, drift)
    for block in ("registries", "gate", "line_coverage", "durations", "validation_scripts"):
        if committed.get(block) != fresh.get(block):
            drift.append(
                f"{block} drift:\n    committed={committed.get(block)!r}"
                f"\n    fresh={fresh.get(block)!r}"
            )

    # A new/removed top-level key (other than the ignored generated_utc) is drift.
    cdash_keys = set(committed) - {"generated_utc"}
    fdash_keys = set(fresh) - {"generated_utc"}
    if cdash_keys != fdash_keys:
        drift.append(
            f"top-level key drift: only committed={sorted(cdash_keys - fdash_keys)} "
            f"only fresh={sorted(fdash_keys - cdash_keys)}"
        )

    # Page freshness (I1): the rendered MyST page must match a fresh in-process
    # render of the committed JSON, ignoring the volatile `Generated at ...` line
    # (mirrors how the JSON gate ignores generated_utc). A stale page — e.g. missing
    # a module row — REDs here until re-rendered.
    committed_page = (_REPO_ROOT / _DASHBOARD_PAGE).read_text()
    fresh_page = render_dashboard_page(committed)
    if _page_minus_timestamp(committed_page) != _page_minus_timestamp(fresh_page):
        drift.append(
            "rendered test-dashboard.md is stale vs the committed JSON — re-render "
            "with `build_test_dashboard.py --render` and recommit the page."
        )

    assert not drift, (
        "dashboard staleness drift (regenerate + recommit "
        f"{_DASHBOARD_JSON} (and re-render the page) if intended):\n  "
        + "\n  ".join(drift)
    )

    # Provenance check (conditional, Phase 2+): once line coverage is MEASURED, the
    # committed coverage carries a git_sha; a stale coverage.json (measured on an old
    # tree) must not pass silently. We do NOT re-run --cov — we only read the stamp.
    # When not-measured (now), there is no stamp to check -> no-op. See C1.
    assert_coverage_provenance_fresh(committed["line_coverage"], _head_sha())
