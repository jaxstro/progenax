// Card-link new-tab plugin (ADR-0034 optional follow-up).
//
// Theory pages link each carded equation to the provenance glossary via
// plain cross-references: [↗ model card](#card-<label>). Those resolve
// same-tab, so a reader mid-derivation loses their place. This transform
// rewrites exactly those links into raw <a target="_blank"> anchors so the
// glossary opens beside the derivation.
//
// Resolution: the glossary pages are GENERATED, so the anchor -> page map is
// recovered by scanning docs/website/15-model-reference/*.md for
// `(card-...)=` targets and `:label: card-...` equation labels — no YAML
// dependency, no hardcoded map to drift. Links whose anchor is not found are
// left untouched (the normal cross-ref pipeline still handles them).
//
// URL scheme: MyST serves every page at a ROOT-LEVEL slug (the filename stem,
// html-normalized: `spatial_profiles.md` -> /spatial-profiles), NOT its
// directory path. Slugs are also deduplicated build-order-dependently (the
// glossary `binaries.md` currently wins `/binaries`; the API page gets
// `/binaries-1`) — so every href this plugin emits is verified against the
// built `myst.xref.json` by scripts/check_card_links.py in the docs gate.
//
// Constraint notes (mystmd): only the `html` node carries raw markup, and
// plain anchors survive sanitization (no <script> involved). Built-site URLs
// are root-absolute (/15-model-reference/<stem>#<anchor>), so a project Pages
// site needs its subpath prefix: BASE follows the same BASE_URL env var the
// Pages workflow already passes to `myst build` (e.g. /progenax).

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE = (process.env.BASE_URL ?? '').replace(/\/+$/, '');
const GLOSSARY_DIR = path.join(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '15-model-reference',
);

let anchorMap = null;

function buildAnchorMap() {
  const map = new Map();
  let files = [];
  try {
    files = fs.readdirSync(GLOSSARY_DIR).filter((f) => f.endsWith('.md'));
  } catch {
    return map; // glossary absent: transform becomes a no-op
  }
  for (const file of files) {
    const stem = file.replace(/\.md$/, '');
    const text = fs.readFileSync(path.join(GLOSSARY_DIR, file), 'utf8');
    for (const m of text.matchAll(/^\((card-[\w-]+)\)=/gm)) {
      map.set(m[1], stem);
    }
    for (const m of text.matchAll(/^:label:\s+(card-[\w-]+)\s*$/gm)) {
      map.set(m[1], stem);
    }
  }
  return map;
}

// MyST's html-id normalization (lowercase, non-alphanumerics -> '-'): applied
// to both the page slug (filename stem) and the anchor fragment.
function htmlNormalize(s) {
  return s.toLowerCase().replace(/[^a-z0-9-]+/g, '-').replace(/-+/g, '-');
}

function textOf(node) {
  if (node.value) return node.value;
  return (node.children ?? []).map(textOf).join('');
}

function walk(node, fn, parent = null, index = null) {
  fn(node, parent, index);
  (node.children ?? []).forEach((child, i) => walk(child, fn, node, i));
}

const cardLinkTransform = {
  name: 'card-links-new-tab',
  doc: 'Open [↗ model card](#card-...) glossary links in a new tab.',
  stage: 'document',
  plugin: () => (tree) => {
    anchorMap ??= buildAnchorMap();
    if (anchorMap.size === 0) return;
    walk(tree, (node, parent, index) => {
      if (node.type !== 'link' || parent == null || index == null) return;
      const url = node.url ?? '';
      const m = url.match(/^#(card-[\w-]+)$/);
      if (!m) return;
      const stem = anchorMap.get(m[1]);
      if (!stem) return;
      const label = textOf(node) || '↗ model card';
      parent.children[index] = {
        type: 'html',
        value:
          `<a href="${BASE}/${htmlNormalize(stem)}#${htmlNormalize(m[1])}" ` +
          'target="_blank" rel="noopener" ' +
          'title="opens the model-card glossary in a new tab">' +
          `${label}</a>`,
      };
    });
  },
};

export default {
  name: 'Card links open in a new tab',
  transforms: [cardLinkTransform],
};
