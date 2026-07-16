# Banked 2D-β inference — archived scratch drivers

> **FROZEN (2026-07-16 rename, amendment A2):** these scripts import the **pre-rename package
> name `gravoturb_fdf`** and were deliberately left untouched by the `gravoturb` rename — they
> are banked evidence, not live code. Do not run them against the current tree; the live
> package is `gravoturb` (see `../../README.md`).

These are **historical, one-off investigation drivers** from the 2-D-projected spectral-slope (β)
inference arc. **Only these old scratch drivers are banked** — the 2-D projected-β forward model
itself is the **ACTIVE headline** (per Anna's decision), living in the package's inference layer:
`inference/projected_logp.py` (the 2-D projected forward model + likelihood) and
`inference/flow_npe.py` (the NPE baseline). What is banked here is the evidence trail of the
scratch investigation (2026-06-07 numbers, grid/HMC/emulator/SBC attempts), not the method. See:

- an internal 2D-inference retrospective
- an internal 2D-projection design note

**Status: not maintained, not run in CI, not imported by any test or live module.** They are kept for
reproducibility of the banked evidence. The *live* validation lives one directory up:
`acceptance.py` (AC1–AC17), `cluster_acceptance.py` (gravoturbulent cluster-IC forward tool),
`calibration.py`, `measure.py`; the live 2-D inference path is `../../inference/projected_logp.py`
+ `../../inference/flow_npe.py`.

Naming: `_d0*` = discriminating diagnostics (gate "is analytic possible?"); `_v1*`–`_v5*` = the
inference attempts (grid/HMC/emulator/SBC/rank-G/log₊/shot/flow); `_m1/_m2` = milestone bake-off +
the old IC gallery (superseded by `cluster_acceptance.py`); `projection_fisher_spike.py` = a Fisher
forecast spike. To run one, restore it next to the live modules (they use absolute
`gravoturb_fdf.validation.*` imports) and execute with `PYTHONPATH=src:src/experimental`.
