"""Render the generated test-dashboard JSON to a MyST matrix page.

Split out of ``scripts/build_test_dashboard.py`` so the generator stays under the
500-LOC file cap (the table-building logic is the bulk of ``--render``). This is a
plain SCRIPT module — pure string assembly, no JAX, no side effects beyond
returning the page text. ``build_test_dashboard.py --render`` imports
:func:`render_dashboard_page` and writes its output to
``docs/website/50-validation/test-dashboard.md``.

The page is GENERATED — never hand-edited. It opens with frontmatter + an intro
paragraph that says so and names the regenerating command, then a MyST
``{list-table}`` colon-fence matrix: one row per module, columns

    module | unit | integration | validation | line-cov % | grad-audit fill |
    slowest test | validation PASS
"""

from __future__ import annotations

_GENERATING_CMD = "uv run --no-sync python scripts/build_test_dashboard.py --render"


def _line_cov_cell(line_cov: float | None) -> str:
    """A stored per-module percentage, or an explicit absence of a recorded value."""
    if line_cov is None:
        return "not recorded"
    return f"{line_cov:.1f}"


def _grad_audit_fill(registries: dict) -> str:
    """One-cell summary of the (repo-wide) grad-audit registry fill.

    Show ``AUDITED/(AUDITED+EXEMPT)`` and the recorded hazard count when built,
    else ``not built``. This is a repo-wide stored figure, not a per-module
    gradient verdict or a claim that the current source is hazard-free.
    """
    block = registries.get("differentiability", {})
    if block.get("status") != "built":
        return "not built"
    audited = block.get("audited", 0)
    exempt = sum(block.get("exempt", {}).values())
    hazards = block.get("hazards", 0)
    return f"{audited}/{audited + exempt} audited, {hazards} haz"


def _slowest_cell(durations: dict, module: str) -> str:
    """The slowest test for ``module`` from the durations block, or a dash.

    ``durations`` is either ``{"status": "not-measured"}`` (no committed artifact)
    or ``{module: {"slowest_test": str, "seconds": float}}``. We surface only the
    test node tail + seconds to keep the cell short.
    """
    entry = durations.get(module)
    if not isinstance(entry, dict):
        return "—"
    node = entry.get("slowest_test", "")
    tail = node.rsplit("::", 1)[-1] if "::" in node else node
    seconds = entry.get("seconds")
    if seconds is None:
        return tail or "—"
    return f"{tail} ({seconds:.1f}s)"


def _validation_pass_cell(validation_scripts: dict, module: str) -> str:
    """PASS/FAIL/unknown for a module's ``validate_<module>.py`` script, if any.

    ``validation_scripts`` maps ``validate_*.py`` -> exit code (or ``"unknown"``).
    We match a script named ``validate_<module>.py`` (exact stem). Exit ``0`` ->
    PASS, a nonzero int -> FAIL, ``"unknown"`` -> "—". Modules with no matching
    validate script also show "—".
    """
    code = validation_scripts.get(f"validate_{module}.py")
    if code == 0:
        return "PASS"
    if isinstance(code, int):
        return "FAIL"
    return "—"


def _matrix_rows(dashboard: dict) -> list[str]:
    """Build the ``{list-table}`` body rows (one per module, sorted)."""
    registries = dashboard.get("registries", {})
    durations = dashboard.get("durations", {})
    validation_scripts = dashboard.get("validation_scripts", {})
    grad_fill = _grad_audit_fill(registries)  # repo-wide; same on every row

    rows: list[str] = []
    for module, counts in sorted(dashboard.get("modules", {}).items()):
        cells = [
            f"`{module}`",
            str(counts.get("unit", 0)),
            str(counts.get("integration", 0)),
            str(counts.get("validation", 0)),
            _line_cov_cell(counts.get("line_cov")),
            grad_fill,
            _slowest_cell(durations, module),
            _validation_pass_cell(validation_scripts, module),
        ]
        # MyST list-table row: a top-level bullet, then one nested bullet per cell.
        block = [f"* - {cells[0]}"]
        block += [f"  - {cell}" for cell in cells[1:]]
        rows.append("\n".join(block))
    return rows


# Human labels for the four registries, in dashboard/key order. The prose sentence
# derives BUILT/not-built from the JSON so it can never drift from the actual blocks.
_REGISTRY_LABELS = {
    "differentiability": "differentiability",
    "api_coverage": "API-coverage",
    "physics_validation": "physics-validation",
    "provenance": "provenance",
}


def _registry_status_phrase(registries: dict) -> str:
    """A prose sentence naming which registries are built vs not (derived from JSON).

    Partitions the four registry blocks by their ``status`` so the intro paragraph
    cannot claim a registry is "not built" once it lands. An ``api_coverage`` block
    additionally reports its ``untested`` hole count when built-but-not-full.
    """
    built, not_built = [], []
    for key, label in _REGISTRY_LABELS.items():
        block = registries.get(key, {})
        if block.get("status") == "built":
            tag = label
            if block.get("full") is False and "untested" in block:
                tag += f" ({block['untested']} hole(s))"
            built.append(tag)
        else:
            not_built.append(label)
    parts = []
    if built:
        parts.append(f"Built registries: {', '.join(built)}.")
    if not_built:
        parts.append(f"Not built yet: {', '.join(not_built)}.")
    return " ".join(parts)


def _validation_scripts_section(validation_scripts: dict) -> list[str]:
    """A standalone table of EVERY ``validate_*.py`` -> PASS/FAIL/—.

    The per-module matrix only surfaces scripts whose stem matches a census module
    (``validate_<module>.py``), so 22 of 24 never appear there. This section lists ALL
    of them straight from ``validation/data/validation_runs.json`` so a failing
    validation script is visible on the page, not just in the JSON.
    """
    if not validation_scripts:
        return []
    rows: list[str] = []
    for name, code in sorted(validation_scripts.items()):
        if code == 0:
            verdict = "PASS"
        elif isinstance(code, int):
            verdict = f"FAIL (exit {code})"
        else:
            verdict = "—"
        rows.append(f"* - `{name}`\n  - {verdict}")
    n_fail = sum(
        1 for c in validation_scripts.values() if isinstance(c, int) and c != 0
    )
    return [
        "## Validation scripts",
        "",
        f"Recorded exit status of `scripts/validate_*.py` "
        f"(from `validation/data/validation_runs.json`): "
        f"**{len(validation_scripts)} scripts, {n_fail} failing**.",
        "These are stored results, not executions or freshness checks performed by this page.",
        "",
        "```{list-table} Validation script runs",
        ":header-rows: 1",
        ":align: left",
        "",
        "* - Script\n  - Status",
        "\n".join(rows),
        "```",
        "",
    ]


def render_dashboard_page(dashboard: dict) -> str:
    """Return the full MyST page text for the dashboard JSON.

    Frontmatter (title/description) + a GENERATED-banner intro naming the
    regenerating command + the ``{list-table}`` matrix.
    """
    generated_utc = dashboard.get("generated_utc", "unknown")
    gate = dashboard.get("gate", {})
    registries = dashboard.get("registries", {})
    line_cov = gate.get("line_cov_measured")
    line_cov_str = (
        "not recorded (requires a full-suite `--cov` measurement)"
        if line_cov is None
        else f"{line_cov:.1f}% (floor {gate.get('line_cov_floor')}%)"
    )
    registry_phrase = _registry_status_phrase(registries)
    provenance = dashboard.get("line_coverage", {}).get("coverage_provenance") or {}
    measured_utc = provenance.get("measured_utc", "not recorded")
    measured_sha = provenance.get("git_sha", "not recorded")
    full_suite = gate.get("full_suite_green")
    full_suite_text = (
        "No full-suite pass/fail verdict is recorded in this dashboard."
        if full_suite is None
        else f"Stored full-suite verdict: {'PASS' if full_suite else 'FAIL'}; "
        "current input freshness still requires a separate check."
    )

    header = [
        "* - Module",
        "  - Unit",
        "  - Integration",
        "  - Validation",
        "  - Line-cov %",
        "  - Recorded grad-audit fill",
        "  - Slowest test",
        "  - Recorded validation",
    ]

    lines = [
        "---",
        "title: Test Dashboard",
        "description: Generated display of stored progenax test, coverage, registry, "
        "duration, and validation artifacts; not a current release verdict.",
        "---",
        "# Test Dashboard",
        "",
        "```{warning}",
        "**Stored snapshot, not a current release verdict.** PASS cells, coverage "
        "percentages, and hazard counts describe recorded artifacts. Treat them as "
        "historical until source/test/configuration freshness and the full test "
        "gate pass for the intended revision.",
        full_suite_text,
        "",
        "This page is generated. Display-only regeneration from the unchanged "
        "stored JSON is:",
        "",
        f"    {_GENERATING_CMD}",
        "",
        "Updating the census with `--emit` is separate. Neither operation "
        "reruns coverage, gradient checks, or validation scripts.",
        "```",
        "",
        f"Census generated at `{generated_utc}`. Recorded line coverage: {line_cov_str}. "
        "Coverage, durations and validation PASS entries come from stored artifacts; "
        "regenerating this page does not rerun those checks or establish their validity "
        "for the current revision. "
        "`Line-cov %` cells show the stored statement-weighted per-directory "
        "percentage, or **not recorded** when no value is present. A missing value "
        "does not imply a pending measurement for a test-only module. "
        "The gradient column reports the stored repo-wide differentiability registry "
        f"(audited / audited+exempt, hazard count). {registry_phrase}",
        "",
        f"Coverage measurement recorded at `{measured_utc}` for commit `{measured_sha}`. "
        "A commit stamp does not identify different dirty-tree inputs. See "
        "[testing architecture](testing-architecture.md) for current input-freshness "
        "requirements. The stored census and recorded measurements can have different dates.",
        "",
        "```{list-table} Per-module test + coverage matrix",
        ":header-rows: 1",
        ":align: left",
        "",
        "\n".join(header),
        "\n".join(_matrix_rows(dashboard)),
        "```",
        "",
        *_validation_scripts_section(dashboard.get("validation_scripts", {})),
    ]
    return "\n".join(lines)
