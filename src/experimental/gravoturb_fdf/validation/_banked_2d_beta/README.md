# Banked 2D-β inference — archived scratch drivers

These are **historical, one-off investigation drivers** from the 2-D-projected spectral-slope (β)
inference arc, which was **banked as a methods result, not optimized further** (2026-06-07): the
observable is cosmic-variance + dynamics limited and gas measures β more directly. See the decision
record and the full numbers in:

- `docs/plans/2026-06-07-gravoturb-fdf-2d-inference-retrospective.md`
- `docs/plans/2026-06-07-gravoturb-fdf-2d-projection-native-inference-design.md`

**Status: not maintained, not run in CI, not imported by any test or live module.** They are kept for
reproducibility of the banked result. The *live* validation lives one directory up:
`acceptance.py` (AC1–AC17), `cluster_acceptance.py` (FDF cluster-IC forward tool), `calibration.py`,
`measure.py`.

Naming: `_d0*` = discriminating diagnostics (gate "is analytic possible?"); `_v1*`–`_v5*` = the
inference attempts (grid/HMC/emulator/SBC/rank-G/log₊/shot/flow); `_m1/_m2` = milestone bake-off +
the old IC gallery (superseded by `cluster_acceptance.py`); `projection_fisher_spike.py` = a Fisher
forecast spike. To run one, restore it next to the live modules (they use absolute
`gravoturb_fdf.validation.*` imports) and execute with `PYTHONPATH=src:src/experimental`.
