#!/usr/bin/env python
"""Run all Tier 1 validation scripts.

Tier 1 includes:
- A1-A4: BM19+FDF core validation
- B5-B6: PP20/zeta validation
- C7-C8: Observational anchors
"""

from __future__ import annotations

import sys
import time


def run_all_tier1():
    """Execute all Tier 1 validation scripts."""
    print("=" * 80)
    print("BM19+FDF+PP20 VALIDATION SUITE - TIER 1")
    print("=" * 80)
    print()

    start_time = time.time()
    results = {}

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

    # Summary
    elapsed = time.time() - start_time
    n_success = sum(1 for v in results.values() if v is not None)
    n_total = len(results)

    print("\n" + "=" * 80)
    print("TIER 1 VALIDATION COMPLETE")
    print("=" * 80)
    print(f"\nResults: {n_success}/{n_total} scripts completed successfully")
    print(f"Time elapsed: {elapsed:.1f} seconds")

    for name, result in results.items():
        status = "SUCCESS" if result is not None else "FAILED"
        print(f"  {name}: {status}")

    print("\nPlots saved to: progenax/validation/plots/bm19_fdf_suite/")

    return results


if __name__ == "__main__":
    # Allow running as both `python run_tier1.py` and `python -m validation.bm19_fdf_suite.run_tier1`
    import sys
    import os

    # Add progenax to path if running as standalone script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    progenax_root = os.path.dirname(os.path.dirname(script_dir))
    if progenax_root not in sys.path:
        sys.path.insert(0, progenax_root)

    run_all_tier1()
