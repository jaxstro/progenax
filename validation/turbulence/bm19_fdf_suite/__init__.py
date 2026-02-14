# progenax/validation/bm19_fdf_suite/__init__.py
"""BM19+FDF+PP20 Comprehensive Validation Suite.

This package contains a curated set of validation scripts for the
BM19-consistent gravoturbulent framework implementation.

Tier 1 (Core Validation):
- A1: f_tail_actual vs f_dense consistency
- A2: Resolution convergence
- A3: PDF shape with s_t visualization
- A4: PN11 vs BM19 comparison
- B5: zeta_FDF vs zeta_analytic
- B6: PP20 diagram placement
- C7: Column density threshold
- C8: eta_survive sensitivity

Tier 2 (Extended Validation):
- D9: Parameter sensitivity tornado
- D10: Monte Carlo uncertainty propagation
- D11: SFR-Sigma contours with Larson track

Paper A additions:
- Q calibration: BM19+FDF -> ICs -> Q measurement

Run with:
    python -m progenax.validation.bm19_fdf_suite.run_tier1
    python -m progenax.validation.bm19_fdf_suite.run_all
"""
