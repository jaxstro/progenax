#!/usr/bin/env bash
#
# Docs gate — the release-clean invariant for the progenax MyST site.
#
# Enforces, in one command, the two things a clean `myst build` does NOT:
#   1. Zero broken relative `.md` links. `myst build` silently passes a link to a
#      nonexistent `.md` target (it treats it as an opaque URL), so a "0 content
#      warnings" build can still ship dead links. `check_links_and_counts.py` is
#      the authority for this class and exits nonzero on any broken link.
#   2. Zero MyST content warnings. `myst build` exits 0 even when it prints
#      `⚠️ ...` content warnings (broken cross-refs, empty link/eq targets, etc.).
#      `myst build --strict` does NOT help: it escalates only ERRORS, not
#      warnings (verified against mystmd v1.10.1). So we capture the build output
#      and fail if any `⚠` warning line survives the node/dev-server noise filter.
#
#   3. Card-link resolution. The card-links plugin (plugins/card-links.mjs)
#      computes glossary URLs from slug assumptions `myst build` does not
#      enforce (root-flat slugs + build-order-dependent dedup); every emitted
#      href is verified against the build's own myst.xref.json.
#
# Usage:  bash scripts/check_docs.sh   (or `make gate`)   from docs/website/.
# Exit:   0 = all gates pass; 1 = any broken link, warning, or dead card link.

set -uo pipefail
cd "$(dirname "$0")/.."   # -> docs/website/

fail=0

echo "== [1/3] link / count gate =================================="
if python3 scripts/check_links_and_counts.py; then
  echo "  link gate: PASS"
else
  echo "  link gate: FAIL (broken .md links above)"
  fail=1
fi

echo ""
echo "== [2/3] build content-warning gate ========================="
log="$(mktemp)"
NODE_OPTIONS=--no-deprecation myst build --html >"$log" 2>&1
build_rc=$?

# MyST content warnings are emitted as lines starting with the ⚠️ glyph. Exclude
# the upstream node DEP0169 deprecation (documented, suppressed in the Makefile)
# and any stray `myst start` dev-server HTTP log lines.
warns="$(grep -E '⚠' "$log" | grep -viE 'DeprecationWarning|DEP0169|GET |POST |HTTP' || true)"

if [ "$build_rc" -ne 0 ]; then
  echo "  myst build FAILED (exit $build_rc):"
  tail -5 "$log" | sed 's/^/    /'
  fail=1
fi
if [ -n "$warns" ]; then
  echo "  build gate: FAIL — content warnings:"
  echo "$warns" | sed 's/^/    /'
  fail=1
elif [ "$build_rc" -eq 0 ]; then
  pages="$(grep -E '📚 Built' "$log" | tail -1 | sed 's/^/  /')"
  echo "  build gate: PASS — 0 content warnings"
  [ -n "$pages" ] && echo "$pages"
fi
rm -f "$log"

echo ""
echo "== [3/3] card-link resolution gate =========================="
if [ "$build_rc" -eq 0 ]; then
  if ! python3 scripts/check_card_links.py; then
    fail=1
  fi
else
  echo "  card-link gate: SKIP (build failed above)"
fi

echo ""
if [ "$fail" -eq 0 ]; then
  echo "DOCS GATE: PASS"
else
  echo "DOCS GATE: FAIL"
fi
exit "$fail"
