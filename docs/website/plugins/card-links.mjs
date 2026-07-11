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
// Constraint notes (mystmd): only the `html` node carries raw markup, and
// plain anchors survive sanitization (no <script> involved). Built-site URLs
// are root-absolute (/15-model-reference/<stem>#<anchor>) — adjust BASE if
// the site ever deploys under a subpath.

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const BASE = '';
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
          `<a href="${BASE}/15-model-reference/${stem}#${m[1]}" ` +
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
