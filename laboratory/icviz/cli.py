"""ICViz CLI: render registered figures to PDF + PNG + WebP.

Usage (repo root):
    env -u VIRTUAL_ENV uv run --no-sync python -m laboratory.icviz --list
    env -u VIRTUAL_ENV uv run --no-sync python -m laboratory.icviz --only imf-classic-slopes
    env -u VIRTUAL_ENV uv run --no-sync python -m laboratory.icviz --all
"""

from __future__ import annotations

import argparse
import shutil

from .registry import FIGURES
from .style import save_figure_formats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="list registered figures")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="render these figures")
    ap.add_argument("--all", action="store_true", help="render every figure")
    args = ap.parse_args(argv)

    if args.list or not (args.only or args.all):
        for name, spec in FIGURES.items():
            print(f"  {name:<28} -> {spec.stem}.{{pdf,png,webp}}  [{spec.page}]")
            if spec.caption:
                print(f"    {spec.caption}")
        return 0

    names = list(FIGURES) if args.all else args.only
    for name in names:
        if name not in FIGURES:
            ap.error(f"unknown figure {name!r} (see --list)")
        spec = FIGURES[name]
        fig = spec.builder()
        for path in save_figure_formats(fig, spec.output_stem):
            print(f"wrote {path}")
        # Only the WebP is committed/embedded; masters stay in icviz/plots/.
        spec.site_webp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(spec.output_stem.with_suffix(".webp"), spec.site_webp)
        print(f"wrote {spec.site_webp}  (site embed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
