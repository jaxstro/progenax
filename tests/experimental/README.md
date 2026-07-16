# tests/experimental — gravoturb test tree

Repo-only experimental suite (`PYTHONPATH=src:src/experimental`); mirrors the
`src/experimental/gravoturb` package.

**Naming rule: test files mirror the module they test.** `test_<module>.py` covers
`gravoturb.<subpackage>.<module>` — e.g. `test_gaussian_field.py` ↔
`realization/gaussian_field.py`, `test_density_pdf.py` ↔ `theory/density_pdf.py`,
`test_counts_in_cells.py` ↔ `theory/counts_in_cells.py`, `test_log_correlations.py` ↔
`theory/log_correlations.py`, `test_measure.py` ↔ `diagnostics/measure.py`. Test *function*
names keep their physics names.

Placement (realization/placement.py) is covered by three files:
`test_placement_sampling.py` (tail/smooth categorical star sampling),
`test_placement_mask.py` (the soft-sigmoid dense-tail mask), and
`test_multi_freefall.py` (FK12 multi-freefall placement with derived f_sub).

Byte-identity pins: `test_rename_byte_identity.py` gates that `gravoturb` reproduces
pre-rename (`gravoturb_fdf`, commit 66f627d) realizations bit-exactly against the SHA-256
pins in `fixtures/rename_pins/pre_rename_sha256.json` (same-machine contract; skips on a
different environment fingerprint).

Layout: `unit/` (fast, per-module), `validation/` (AC1–AC17 acceptance assertions),
`integration/`, `fixtures/`.
