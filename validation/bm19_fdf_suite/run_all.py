#!/usr/bin/env python
"""Run ALL validation scripts (Tier 1 + Tier 2 + Paper A).

Complete validation suite for BM19+FDF+PP20 implementation.
"""

from __future__ import annotations

import sys
import time


def run_all():
    """Execute all validation scripts."""
    print("=" * 80)
    print("BM19+FDF+PP20 COMPREHENSIVE VALIDATION SUITE")
    print("=" * 80)
    print()

    start_time = time.time()
    results = {}

    # =========================================================================
    # TIER 1: Core Validation
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TIER 1: CORE VALIDATION")
    print("#" * 80)

    # A1: f_tail vs f_dense
    print("\n" + "=" * 60)
    print("Running A1: f_tail_actual vs f_dense")
    print("=" * 60)
    try:
        from . import tier1_a1_ftail_fdense
        results["A1"] = tier1_a1_ftail_fdense.main()
        print("A1: SUCCESS")
    except Exception as e:
        print(f"A1: FAILED - {e}")
        results["A1"] = None

    # A2: Resolution convergence
    print("\n" + "=" * 60)
    print("Running A2: Resolution Convergence")
    print("=" * 60)
    try:
        from . import tier1_a2_resolution
        results["A2"] = tier1_a2_resolution.main()
        print("A2: SUCCESS")
    except Exception as e:
        print(f"A2: FAILED - {e}")
        results["A2"] = None

    # A3: PDF shape
    print("\n" + "=" * 60)
    print("Running A3: PDF Shape Visualization")
    print("=" * 60)
    try:
        from . import tier1_a3_pdf_shape
        results["A3"] = tier1_a3_pdf_shape.main()
        print("A3: SUCCESS")
    except Exception as e:
        print(f"A3: FAILED - {e}")
        results["A3"] = None

    # A4: PN11 vs BM19
    print("\n" + "=" * 60)
    print("Running A4: PN11 vs BM19 Comparison")
    print("=" * 60)
    try:
        from . import tier1_a4_pn11_vs_bm19
        results["A4"] = tier1_a4_pn11_vs_bm19.main()
        print("A4: SUCCESS")
    except Exception as e:
        print(f"A4: FAILED - {e}")
        results["A4"] = None

    # B5: zeta comparison
    print("\n" + "=" * 60)
    print("Running B5: Zeta FDF vs Analytic")
    print("=" * 60)
    try:
        from . import tier1_b5_zeta_comparison
        results["B5"] = tier1_b5_zeta_comparison.main()
        print("B5: SUCCESS")
    except Exception as e:
        print(f"B5: FAILED - {e}")
        results["B5"] = None

    # B6: PP20 diagram
    print("\n" + "=" * 60)
    print("Running B6: PP20 Diagram")
    print("=" * 60)
    try:
        from . import tier1_b6_pp20_diagram
        results["B6"] = tier1_b6_pp20_diagram.main()
        print("B6: SUCCESS")
    except Exception as e:
        print(f"B6: FAILED - {e}")
        results["B6"] = None

    # C7: Column density threshold
    print("\n" + "=" * 60)
    print("Running C7: Column Density Threshold")
    print("=" * 60)
    try:
        from . import tier1_c7_column_density
        results["C7"] = tier1_c7_column_density.main()
        print("C7: SUCCESS")
    except Exception as e:
        print(f"C7: FAILED - {e}")
        results["C7"] = None

    # C8: eta sensitivity
    print("\n" + "=" * 60)
    print("Running C8: Eta Sensitivity")
    print("=" * 60)
    try:
        from . import tier1_c8_eta_sensitivity
        results["C8"] = tier1_c8_eta_sensitivity.main()
        print("C8: SUCCESS")
    except Exception as e:
        print(f"C8: FAILED - {e}")
        results["C8"] = None

    # =========================================================================
    # TIER 2: Extended Validation
    # =========================================================================
    print("\n" + "#" * 80)
    print("# TIER 2: EXTENDED VALIDATION")
    print("#" * 80)

    # D9: Parameter sensitivity
    print("\n" + "=" * 60)
    print("Running D9: Parameter Sensitivity")
    print("=" * 60)
    try:
        from . import tier2_d9_sensitivity
        results["D9"] = tier2_d9_sensitivity.main()
        print("D9: SUCCESS")
    except Exception as e:
        print(f"D9: FAILED - {e}")
        results["D9"] = None

    # D10: Monte Carlo
    print("\n" + "=" * 60)
    print("Running D10: Monte Carlo Uncertainty")
    print("=" * 60)
    try:
        from . import tier2_d10_monte_carlo
        results["D10"] = tier2_d10_monte_carlo.main()
        print("D10: SUCCESS")
    except Exception as e:
        print(f"D10: FAILED - {e}")
        results["D10"] = None

    # D11: SFR contours
    print("\n" + "=" * 60)
    print("Running D11: SFR-Sigma Contours")
    print("=" * 60)
    try:
        from . import tier2_d11_sfr_contours
        results["D11"] = tier2_d11_sfr_contours.main()
        print("D11: SUCCESS")
    except Exception as e:
        print(f"D11: FAILED - {e}")
        results["D11"] = None

    # =========================================================================
    # PAPER A: Q Calibration
    # =========================================================================
    print("\n" + "#" * 80)
    print("# PAPER A: Q CALIBRATION")
    print("#" * 80)

    print("\n" + "=" * 60)
    print("Running Paper A: Q Calibration")
    print("=" * 60)
    try:
        from . import paper_a_q_calibration
        results["Q_cal"] = paper_a_q_calibration.main()
        print("Q Calibration: SUCCESS")
    except Exception as e:
        print(f"Q Calibration: FAILED - {e}")
        results["Q_cal"] = None

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - start_time
    n_success = sum(1 for v in results.values() if v is not None)
    n_total = len(results)

    print("\n" + "=" * 80)
    print("COMPREHENSIVE VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nResults: {n_success}/{n_total} scripts completed successfully")
    print(f"Time elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

    # Group by tier
    tier1_keys = ["A1", "A2", "A3", "A4", "B5", "B6", "C7", "C8"]
    tier2_keys = ["D9", "D10", "D11"]
    paper_keys = ["Q_cal"]

    print("\nTier 1 (Core):")
    for name in tier1_keys:
        status = "SUCCESS" if results.get(name) is not None else "FAILED"
        print(f"  {name}: {status}")

    print("\nTier 2 (Extended):")
    for name in tier2_keys:
        status = "SUCCESS" if results.get(name) is not None else "FAILED"
        print(f"  {name}: {status}")

    print("\nPaper A:")
    for name in paper_keys:
        status = "SUCCESS" if results.get(name) is not None else "FAILED"
        print(f"  {name}: {status}")

    print("\nPlots saved to: progenax/validation/plots/bm19_fdf_suite/")
    print("\nNext step: Generate VALIDATION_SUMMARY.md with analysis")

    return results


if __name__ == "__main__":
    run_all()
