# tests/experimental — gravoturb test tree

Repo-only experimental suite (`PYTHONPATH=src:src/experimental`); mirrors the
`src/experimental/gravoturb` package.

**Naming rule: test files mirror the module they test.** `test_<module>.py` covers
`gravoturb.<subpackage>.<module>` — e.g. `test_gaussian_field.py` ↔
`realization/gaussian_field.py`, `test_density_pdf.py` ↔ `theory/density_pdf.py`,
`test_counts_in_cells.py` ↔ `theory/counts_in_cells.py`, `test_log_correlations.py` ↔
`theory/log_correlations.py`, `test_measure.py` ↔ `diagnostics/measure.py`. Test *function*
names keep their physics names.

A module MAY be covered by several `test_<module>_<concern>.py`-style files when one
file per concern is clearer. Current multi-file modules (2026-07-16 convention pass —
`test_velocity.py`/`test_masses.py` were strict-renamed to their modules):

- `realization/placement.py` ↔ `test_placement_sampling.py` (tail/smooth categorical
  star sampling), `test_placement_mask.py` (the soft-sigmoid dense-tail mask), and
  `test_multi_freefall.py` (FK12 multi-freefall placement with derived fractions);
- `theory/log_correlations.py` ↔ `test_log_correlations.py` + `test_density_hermite.py`
  (the Hermite/Mehler expansion concern);
- the `inference/` gradient path is pooled in `test_inference.py` (covariance,
  likelihood, fisher — one differentiable-chain suite), with `test_sbc.py`,
  `test_diagnostics.py` (↔ `inference/diagnostics.py`), `test_priors.py`,
  `test_projected_logp.py`, and `test_flow_npe.py` per-module.

Byte-identity pins: `test_rename_byte_identity.py` gates that `gravoturb` reproduces
pre-rename (`gravoturb_fdf`, commit 66f627d) realizations bit-exactly against the SHA-256
pins in `fixtures/rename_pins/pre_rename_sha256.json` (same-machine contract; skips on a
different environment fingerprint).

Layout: `unit/` (fast, per-module), `validation/` (AC1–AC17 acceptance assertions),
`integration/`, `fixtures/`.
