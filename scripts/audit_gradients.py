"""Run the grad-audit REGISTRY and emit validation/data/grad_audit_results.json + a markdown
table. The website doc cites this JSON; the pytest gate asserts on the same engine (design D1).

With ``--plots`` it also renders two publication-quality PNGs into validation/plots/ (and a
committed copy into the website figures dir) from the JUST-WRITTEN JSON — never recomputing,
so the figures and the cited table are the same single source of truth."""
import argparse
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

_DEFAULT_JSON = _REPO_ROOT / "validation" / "data" / "grad_audit_results.json"
_PLOTS_DIR = _REPO_ROOT / "validation" / "plots"
# Committed, build-clean copies the website embeds (the *.png/*.pdf gitignore exception):
_SITE_FIG_DIR = _REPO_ROOT / "docs" / "website" / "50-validation" / "figures"


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


def _make_plots(rows: list[dict]) -> list[Path]:
    """Render the two audit figures from `rows` (the in-memory JSON). Guarded import:
    matplotlib is the [viz] extra and this is a script, so a clean skip is acceptable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print(
            "[--plots] matplotlib not installed — skipping figures. "
            "Install the viz extra:  env -u VIRTUAL_ENV uv pip install -e \".[viz]\"",
            file=sys.stderr,
        )
        return []

    import numpy as np

    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _SITE_FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Okabe-Ito-ish palette consistent with the rest of the validation gallery.
    C_CLEAN = "#0072B2"
    C_LIMIT = "#D55E00"
    EPS = 1e-12

    # ---- Figure 1: per-case |AD/FD - 1| residual on a log axis -------------------
    # Clean/consistent cases carry a meaningful residual; known-limitation cases have an
    # undefined or zero/blocked ratio (annotated, not plotted on the residual axis).
    clean_idx, clean_res = [], []
    limit_idx = []
    labels = []
    for i, r in enumerate(rows):
        labels.append(r["id"])
        is_limit = r["status"] == "known-limitation"
        ratio = r["ratio"]
        # A residual is only meaningful when the case expects AD==FD (consistent) and the
        # ratio is a finite, defined number. known_blocked / blocked-zero ratios are not.
        residual_defined = (
            not is_limit
            and r["expect"] == "consistent"
            and abs(r["abs_ad"]) > EPS
            and np.isfinite(ratio)
        )
        if residual_defined:
            clean_idx.append(i)
            clean_res.append(max(abs(ratio - 1.0), EPS))  # floor so log axis is happy
        else:
            limit_idx.append(i)

    fig, ax = plt.subplots(figsize=(11, 5.0))
    if clean_idx:
        ax.scatter(
            clean_idx, clean_res, c=C_CLEAN, s=42, zorder=3,
            edgecolors="white", linewidths=0.5, label="clean (consistent)",
        )
    # Per-class tolerance band is heterogeneous; show the loosest gate that any clean case
    # must clear (the linear-interp r_t derivative, 1e-2) and the tightest (closed-form,1e-5).
    ax.axhspan(0, 1e-2, color=C_CLEAN, alpha=0.06, zorder=0)
    ax.axhline(1e-2, color=C_CLEAN, ls="--", lw=1.0, alpha=0.7,
               label=r"loosest tol band ($10^{-2}$, $r_t$ interp)")
    ax.axhline(1e-5, color="#444444", ls=":", lw=1.0, alpha=0.7,
               label=r"tightest tol ($10^{-5}$, closed-form)")
    # Annotate the known-limitation cases at the bottom of the axis.
    for j, i in enumerate(limit_idx):
        ax.scatter([i], [EPS * 3], marker="x", c=C_LIMIT, s=70, zorder=4)
    if limit_idx:
        ax.scatter([], [], marker="x", c=C_LIMIT, s=70,
                   label="known-limitation (ratio undefined / blocked)")

    ax.set_yscale("log")
    ax.set_ylim(EPS, 1.0)
    ax.set_xlim(-1, len(rows))
    ax.set_xlabel("registry case index")
    ax.set_ylabel(r"$|\mathrm{AD}/\mathrm{FD} - 1|$  (residual)")
    ax.set_title(
        f"Gradient audit — per-case AD-vs-FD residual ({len(rows)} cases, "
        f"{len(clean_idx)} clean / {len(limit_idx)} known-limitation, 0 hazards)"
    )
    ax.legend(loc="lower right", framealpha=0.95, fontsize=9)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    f1 = _PLOTS_DIR / "grad_audit_ratio.png"
    fig.savefig(f1, dpi=150)
    fig.savefig(_SITE_FIG_DIR / "grad_audit_ratio.png", dpi=150)
    plt.close(fig)

    # ---- Figure 2: summary bar chart — counts by direction × status --------------
    directions = ["params->IC", "params->summary"]
    statuses = ["clean", "known-limitation"]
    counts = {d: {s: 0 for s in statuses} for d in directions}
    for r in rows:
        counts.setdefault(r["direction"], {s: 0 for s in statuses})
        counts[r["direction"]][r["status"]] += 1

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    x = np.arange(len(directions))
    w = 0.38
    clean_vals = [counts[d]["clean"] for d in directions]
    limit_vals = [counts[d]["known-limitation"] for d in directions]
    b1 = ax.bar(x - w / 2, clean_vals, w, color=C_CLEAN, label="clean", zorder=3)
    b2 = ax.bar(x + w / 2, limit_vals, w, color=C_LIMIT,
                label="known-limitation", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(str(int(h)), (bar.get_x() + bar.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([r"params $\rightarrow$ IC", r"params $\rightarrow$ summary"])
    ax.set_ylabel("number of registry cases")
    n_clean = sum(clean_vals)
    n_limit = sum(limit_vals)
    ax.set_title(
        f"Gradient-gate summary — {len(rows)} cases "
        f"({n_clean} clean, {n_limit} known-limitation, 0 hazards)"
    )
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)
    ax.set_ylim(0, max(clean_vals) * 1.18)
    fig.tight_layout()
    f2 = _PLOTS_DIR / "grad_audit_summary.png"
    fig.savefig(f2, dpi=150)
    fig.savefig(_SITE_FIG_DIR / "grad_audit_summary.png", dpi=150)
    plt.close(fig)

    print(f"[--plots] wrote {f1} and {f2}")
    print(f"[--plots] committed website copies into {_SITE_FIG_DIR}")
    return [f1, f2]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plots", action="store_true",
        help="after writing the JSON, render grad_audit_ratio.png + grad_audit_summary.png",
    )
    args = parser.parse_args()

    rows = run_audit()
    print(to_markdown(rows))
    n_haz = sum(r["status"] == "hazard" for r in rows)
    print(f"\n{len(rows)} cases; {n_haz} hazard(s).")
    if args.plots:
        _make_plots(rows)
    raise SystemExit(1 if n_haz else 0)
