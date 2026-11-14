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
from pathlib import Path

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
    return sum(m[tier] for m in dashboard["modules"].values() for tier in _TIER_KEYS)


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
            drift.append(
                f"{label}: non-finite kind changed committed={cv!r} fresh={fv!r}"
            )
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
        _diff_floats(
            f"modules[{module!r}]['line_cov']", c["line_cov"], f["line_cov"], drift
        )


def _head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# A current-suite coverage claim depends on both the measured source and the tests
# selected by the released-core full-suite command. Tests can be removed, skip, or
# exercise fewer paths without changing source, so test-only changes can lower coverage.
_SRC_PREFIX = "src/progenax/"
_COVERAGE_INPUT_PATHS = (
    _SRC_PREFIX,
    "tests/unit/",
    "tests/integration/",
    "tests/validation/",
    "tests/conftest.py",
    "tests/__init__.py",
    "conftest.py",
    "pyproject.toml",
    "pytest.ini",
    ".coveragerc",
    "setup.cfg",
    "tox.ini",
)

# The full-suite selector the floor/freshness gate requires (mirrors the api-coverage
# manifest's FULL_SELECTOR — a `-m "not slow"` pass understates coverage).
_FULL_SELECTOR = "full"


def _git(*args: str) -> subprocess.CompletedProcess:
    """Run a git command at the repo root (NOT check=True — callers read returncode)."""
    return subprocess.run(
        ["git", *args], cwd=str(_REPO_ROOT), capture_output=True, text=True
    )


def assert_coverage_provenance_fresh(line_coverage: dict, head_sha: str) -> None:
    """Reject a current-suite claim when source, selected tests, or config changed.

    Keep the full-selector and ancestor checks. Compare measured inputs with HEAD,
    the index, and the working tree, and inspect untracked inputs. A source/test
    change at any of those layers makes the recorded coverage historical. Unrelated
    documentation and experimental-only files are outside this dependency set.

    A commit-only stamp cannot identify a measurement made on different dirty-tree
    inputs. Such a run is not accepted by restamping HEAD; its input-identity contract
    remains unresolved. This check neither runs coverage nor rewrites its artifact.
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
    # git_sha must be an ancestor of HEAD for the input comparison. `git
    # merge-base --is-ancestor A B` exits 0 iff A is an ancestor of B. A non-zero exit
    # (1 = not ancestor; 128 = bad/unknown sha) => we cannot trust the diff => stale.
    ancestor = _git("merge-base", "--is-ancestor", git_sha, head_sha)
    assert ancestor.returncode == 0, (
        f"stale coverage.json: provenance git_sha={git_sha} is not an ancestor of "
        f"HEAD={head_sha} (git exit {ancestor.returncode}) — the measured commit is not "
        f"on HEAD's history; re-measure FULL-suite coverage at the current commit."
    )

    changes = {}
    commands = (
        ("HEAD", ("diff", "--name-only", "-z", git_sha, head_sha)),
        ("index", ("diff", "--cached", "--name-only", "-z", git_sha)),
        ("worktree", ("diff", "--name-only", "-z", git_sha)),
        ("untracked", ("ls-files", "--others", "--exclude-standard", "-z")),
    )
    for layer, args in commands:
        result = _git(*args, "--", *_COVERAGE_INPUT_PATHS)
        assert result.returncode == 0, (
            f"stale coverage.json: cannot inspect {layer} coverage inputs "
            f"(git exit {result.returncode}): {result.stderr.strip()}"
        )
        paths = sorted(path for path in result.stdout.split("\0") if path)
        if paths:
            changes[layer] = paths
    examples = {layer: paths[:3] for layer, paths in changes.items()}
    assert not changes, (
        f"stale coverage.json: source/test/config inputs differ from measured commit "
        f"{git_sha} ({examples}). The recorded coverage describes the historical suite, "
        "not the current one. Re-measure FULL-suite coverage with identified inputs; "
        "a commit-only stamp cannot certify a different dirty tree."
    )


def _page_minus_timestamp(page_text: str) -> list[str]:
    """The page lines with the volatile ``Generated at ...`` line dropped.

    The rendered page embeds ``Generated at `<utc>`.`` which always differs (like
    the JSON's ``generated_utc``). The page-freshness gate ignores exactly that one
    line so it compares structural content, not the timestamp.
    """
    return [
        line for line in page_text.splitlines() if not line.startswith("Generated at `")
    ]


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


def _temp_git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    """Use an isolated Git repository without hooks, signing, or remote operations."""
    return subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Coverage regression",
            "-c",
            "user.email=coverage-regression@example.invalid",
            *args,
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture
def coverage_repo(tmp_path, monkeypatch):
    """Build actual commit/index/worktree states without touching this checkout's Git."""
    repo = tmp_path / "coverage-repo"
    repo.mkdir()
    for variable in (
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_COMMON_DIR",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    ):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    files = {
        "src/progenax/core.py": "def twice(value):\n    return 2 * value\n",
        "tests/unit/test_core.py": "def test_twice():\n    assert 2 * 2 == 4\n",
        "tests/integration/test_core.py": "def test_sum():\n    assert sum([1, 2]) == 3\n",
        "tests/validation/test_core.py": "def test_square():\n    assert 3**2 == 9\n",
        "tests/conftest.py": "REFERENCE_SEED = 7\n",
        "tests/__init__.py": "",
        "pyproject.toml": '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
        "docs/index.md": "Coverage review documentation.\n",
    }
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    _temp_git(repo, "init", "--quiet")
    _temp_git(repo, "add", ".")
    _temp_git(repo, "commit", "--quiet", "-m", "Measured input fixture")
    measured = _temp_git(repo, "rev-parse", "HEAD").stdout.strip()
    monkeypatch.setitem(globals(), "_REPO_ROOT", repo)
    return repo, measured


def test_coverage_provenance_teeth_fire_on_stale_src(coverage_repo):
    """Reject partial/nonancestor stamps while accepting an unchanged actual Git tree."""
    _repo, head = coverage_repo
    assert_coverage_provenance_fresh(_measured_block(head), head_sha=head)

    # STALE on a non-ancestor (orphaned/unknown) sha -> the merge-base leg raises.
    with pytest.raises(AssertionError, match="not an ancestor"):
        assert_coverage_provenance_fresh(_measured_block("0" * 40), head_sha=head)

    # STALE on a partial selector -> the selector leg raises (even at a fresh sha).
    with pytest.raises(AssertionError, match="not 'full'"):
        assert_coverage_provenance_fresh(
            _measured_block(head, selector="tests/unit -m 'not slow'"), head_sha=head
        )

    # not-measured is always a no-op (no stamp to check).
    assert_coverage_provenance_fresh({"status": "not-measured"}, head_sha="whatever")


@pytest.mark.parametrize(
    "path, state",
    [
        ("src/progenax/core.py", "dirty"),
        ("src/progenax/new.py", "untracked"),
        ("src/progenax/core.py", "index-only"),
        ("src/progenax/core.py", "committed"),
        ("tests/unit/test_core.py", "dirty"),
        ("tests/integration/test_core.py", "deleted"),
        ("tests/validation/test_core.py", "index-only"),
        ("tests/unit/test_core.py", "committed"),
        ("tests/integration/test_new.py", "untracked"),
        ("tests/conftest.py", "dirty"),
        ("pyproject.toml", "dirty"),
        ("pytest.ini", "untracked"),
    ],
)
def test_coverage_provenance_rejects_changed_inputs(coverage_repo, path, state):
    """Catch stale source/tests/config in HEAD, index, worktree, and untracked files."""
    repo, measured = coverage_repo
    target = repo / path
    original = target.read_text() if target.exists() else ""
    if state == "deleted":
        target.unlink()
    else:
        target.write_text(original + "# Changed coverage input.\n")
    if state in {"index-only", "committed"}:
        _temp_git(repo, "add", path)
        if state == "index-only":
            target.write_text(original)
        else:
            _temp_git(repo, "commit", "--quiet", "-m", "Changed measured input")
    head = _temp_git(repo, "rev-parse", "HEAD").stdout.strip()
    with pytest.raises(AssertionError, match="source/test/config inputs differ"):
        assert_coverage_provenance_fresh(_measured_block(measured), head_sha=head)


def test_coverage_provenance_allows_unrelated_docs_and_experimental(coverage_repo):
    """Do not invalidate released-core coverage for unrelated docs/experimental edits."""
    repo, measured = coverage_repo
    (repo / "docs/index.md").write_text("Updated unrelated documentation.\n")
    _temp_git(repo, "add", "docs/index.md")
    _temp_git(repo, "commit", "--quiet", "-m", "Documentation-only change")
    (repo / "docs/local-note.md").write_text("Untracked unrelated documentation.\n")
    excluded = repo / "tests/experimental/test_held.py"
    excluded.parent.mkdir(parents=True)
    excluded.write_text("# Outside the selected core suite.\n")
    head = _temp_git(repo, "rev-parse", "HEAD").stdout.strip()
    assert_coverage_provenance_fresh(_measured_block(measured), head_sha=head)


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
    for block in (
        "registries",
        "gate",
        "line_coverage",
        "durations",
        "validation_scripts",
    ):
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
    # A not-measured block has no stamp to check; a measured block is checked
    # against the current source, selected tests, and configuration above.
    assert_coverage_provenance_fresh(committed["line_coverage"], _head_sha())
