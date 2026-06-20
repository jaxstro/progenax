"""One-shot seed generator (committed for reproducibility, NOT imported by the gate).
Prints the (id, param) coverage units in REGISTRY and the full __all__ list, so the
manifest literals can be seeded without transcription error. The manifest is then an
INDEPENDENT frozen literal — deleting a registry case must trip the coverage ratchet."""

import progenax  # noqa: F401 (float64 + __all__)
from tests.validation.grad_audit.registry import REGISTRY


def main():
    pairs = sorted({(c.id, c.param) for c in REGISTRY})
    print(f"# {len(pairs)} (id, param) coverage units:")
    for cid, p in pairs:
        print(f'    ("{cid}", "{p}"),')
    print(f"\n# {len(progenax.__all__)} __all__ symbols:")
    for s in sorted(progenax.__all__):
        print(f'    "{s}": ...,')


if __name__ == "__main__":
    main()
