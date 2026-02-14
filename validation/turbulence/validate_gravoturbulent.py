#!/usr/bin/env python3
"""
Validation script for gravoturbulent f_sub derivation (Burkhart 2018).

This script validates the physics-based f_sub calculation against:
1. Worked examples from the theory document (Orion, YMC, Taurus)
2. Monotonicity requirements (physical behavior)
3. Preset coverage across f_sub dynamic range

Run:
    python validation/validate_gravoturbulent.py

Generates plots:
    validation/plots/gravoturbulent/fsub_vs_sigma.png
    validation/plots/gravoturbulent/presets.png
    validation/plots/gravoturbulent/derivation_chain.png
    validation/plots/gravoturbulent/sensitivity.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add progenax to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from progenax.cluster.fdf_config import (
    GRAVOTURBULENT_PRESETS,
    GravoturbulentEnv,
    env_from_preset,
    gravoturbulent_summary,
    tail_layer_from_env,
)

# Plot settings with LaTeX support
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.figsize": (10, 6),
    "text.usetex": False,  # Use mathtext instead of full LaTeX
    "mathtext.fontset": "cm",  # Computer Modern for math
    "font.family": "serif",
})

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)


def validate_worked_examples():
    """Validate against theory document worked examples (Sections 11.1-11.3)."""
    print("\n" + "=" * 70)
    print("VALIDATION 1: Worked Examples from Theory Document")
    print("=" * 70)

    examples = [
        {
            "name": "Orion (Section 11.1)",
            "env": GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6),
            "expected": {
                "sigma_s": 1.78,
                "alpha_vir": 1.33,
                "s_crit": 3.84,
                "f_tail": 0.107,
                "f_sub": 0.064,
            },
            "tolerances": {
                "sigma_s": 0.02,
                "alpha_vir": 0.02,
                "s_crit": 0.1,
                "f_tail": 0.02,
                "f_sub": 0.01,
            },
        },
        {
            "name": "YMC (Section 11.2)",
            "env": GravoturbulentEnv(Sigma=1500, Mach=25, eta_survive=0.85),
            "expected": {
                "sigma_s": 2.15,
                "alpha_vir": 0.133,
                "s_crit": 3.00,
                "f_tail": 0.372,
                "f_sub": 0.316,
            },
            "tolerances": {
                "sigma_s": 0.02,
                "alpha_vir": 0.02,
                "s_crit": 0.15,
                "f_tail": 0.03,
                "f_sub": 0.03,
            },
        },
        {
            "name": "Taurus (Section 11.3)",
            "env": GravoturbulentEnv(Sigma=40, Mach=6, eta_survive=0.4),
            "expected": {
                "sigma_s": 1.38,
                "alpha_vir": 5.0,
                "s_crit": 3.77,
                "f_tail": 0.021,
                "f_sub": 0.008,
            },
            "tolerances": {
                "sigma_s": 0.02,
                "alpha_vir": 0.1,
                "s_crit": 0.1,
                "f_tail": 0.005,
                "f_sub": 0.002,
            },
        },
    ]

    all_passed = True
    results_table = []

    for example in examples:
        result = gravoturbulent_summary(example["env"])
        print(f"\n{example['name']}:")
        print(f"  Environment: Σ={example['env'].Sigma}, M={example['env'].Mach}, η={example['env'].eta_survive}")
        print("-" * 50)
        print(f"  {'Property':<12} {'Expected':<10} {'Computed':<10} {'Error':<10} {'Status'}")
        print("  " + "-" * 48)

        for prop in ["sigma_s", "alpha_vir", "s_crit", "f_tail", "f_sub"]:
            expected = example["expected"][prop]
            computed = getattr(result, prop)
            tol = example["tolerances"][prop]
            error = abs(computed - expected)
            passed = error < tol
            status = "✅" if passed else "❌"
            if not passed:
                all_passed = False
            print(f"  {prop:<12} {expected:<10.4f} {computed:<10.4f} {error:<10.4f} {status}")
            results_table.append({
                "example": example["name"],
                "property": prop,
                "expected": expected,
                "computed": computed,
                "error": error,
                "tolerance": tol,
                "passed": passed,
            })

    return all_passed, results_table


def validate_monotonicity():
    """Validate physical monotonicity: higher Σ → higher f_sub."""
    print("\n" + "=" * 70)
    print("VALIDATION 2: Physical Monotonicity")
    print("=" * 70)

    all_passed = True

    # Test f_sub vs Σ monotonicity
    print("\nTest: f_sub increases with Σ (fixed M=15, η=0.7)")
    print("-" * 50)
    Sigmas = [50, 100, 300, 1000, 3000]
    f_subs = []
    for Sigma in Sigmas:
        env = GravoturbulentEnv(Sigma=Sigma, Mach=15, eta_survive=0.7)
        result = gravoturbulent_summary(env)
        f_subs.append(result.f_sub)
        print(f"  Σ={Sigma:>4} M☉/pc² → f_sub = {result.f_sub:.4f}")

    # Check monotonicity
    for i in range(len(f_subs) - 1):
        if f_subs[i] >= f_subs[i + 1]:
            print(f"  ❌ FAILED: f_sub({Sigmas[i]})={f_subs[i]:.4f} >= f_sub({Sigmas[i+1]})={f_subs[i+1]:.4f}")
            all_passed = False

    if all(f_subs[i] < f_subs[i + 1] for i in range(len(f_subs) - 1)):
        print("  ✅ Monotonicity verified: f_sub strictly increases with Σ")

    # Test α_vir vs Σ monotonicity
    print("\nTest: α_vir decreases with Σ (inverse relationship)")
    print("-" * 50)
    alpha_virs = []
    for Sigma in Sigmas:
        env = GravoturbulentEnv(Sigma=Sigma, Mach=15, eta_survive=0.7)
        result = gravoturbulent_summary(env)
        alpha_virs.append(result.alpha_vir)
        print(f"  Σ={Sigma:>4} M☉/pc² → α_vir = {result.alpha_vir:.4f}")

    # Verify α_vir ∝ 1/Σ
    ratios = [Sigmas[i + 1] / Sigmas[i] for i in range(len(Sigmas) - 1)]
    alpha_ratios = [alpha_virs[i] / alpha_virs[i + 1] for i in range(len(alpha_virs) - 1)]
    print(f"\n  α_vir ratio should match Σ ratio (inverse relationship):")
    for i, (sigma_r, alpha_r) in enumerate(zip(ratios, alpha_ratios)):
        match = abs(sigma_r - alpha_r) < 0.01
        status = "✅" if match else "❌"
        print(f"    Σ ratio={sigma_r:.2f}, α_vir ratio={alpha_r:.2f} {status}")
        if not match:
            all_passed = False

    # Test η_survive linear scaling
    print("\nTest: f_sub scales linearly with η_survive")
    print("-" * 50)
    env_base = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.3)
    env_high = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.9)
    result_base = gravoturbulent_summary(env_base)
    result_high = gravoturbulent_summary(env_high)

    print(f"  η=0.3 → f_sub = {result_base.f_sub:.4f}, f_tail = {result_base.f_tail:.4f}")
    print(f"  η=0.9 → f_sub = {result_high.f_sub:.4f}, f_tail = {result_high.f_tail:.4f}")

    f_tail_match = abs(result_base.f_tail - result_high.f_tail) < 1e-6
    ratio = result_high.f_sub / result_base.f_sub
    expected_ratio = 0.9 / 0.3  # = 3.0
    ratio_match = abs(ratio - expected_ratio) < 0.01

    print(f"\n  f_tail should be identical: {f_tail_match}")
    print(f"  f_sub ratio = {ratio:.3f} (expected 3.0): {ratio_match}")

    if f_tail_match:
        print("  ✅ f_tail unchanged (depends only on Σ, M)")
    else:
        print("  ❌ f_tail changed unexpectedly")
        all_passed = False

    if ratio_match:
        print("  ✅ f_sub scales linearly with η_survive")
    else:
        print("  ❌ f_sub ratio incorrect")
        all_passed = False

    return all_passed


def validate_presets():
    """Validate preset environments cover expected f_sub range."""
    print("\n" + "=" * 70)
    print("VALIDATION 3: Preset Coverage")
    print("=" * 70)

    print("\nPreset f_sub values (sorted):")
    print("-" * 70)
    print(f"{'Preset':<15} {'Σ':<8} {'M':<6} {'η':<6} {'f_tail':<10} {'f_sub':<10} {'α_vir':<10}")
    print("-" * 70)

    f_subs = {}
    for name in GRAVOTURBULENT_PRESETS:
        env = GRAVOTURBULENT_PRESETS[name]
        result = gravoturbulent_summary(env)
        f_subs[name] = result.f_sub

    # Sort by f_sub
    for name in sorted(f_subs.keys(), key=lambda x: f_subs[x]):
        env = GRAVOTURBULENT_PRESETS[name]
        result = gravoturbulent_summary(env)
        print(f"{name:<15} {env.Sigma:<8.0f} {env.Mach:<6.0f} {env.eta_survive:<6.2f} {result.f_tail:<10.4f} {result.f_sub:<10.4f} {result.alpha_vir:<10.4f}")

    # Check dynamic range
    ratio = max(f_subs.values()) / min(f_subs.values())
    print(f"\nDynamic range: max(f_sub)/min(f_sub) = {ratio:.1f}×")

    if ratio > 10:
        print("  ✅ Presets span >10× dynamic range (as expected from theory)")
        return True
    else:
        print(f"  ❌ Dynamic range too narrow ({ratio:.1f}× < 10×)")
        return False


def validate_physical_constraints():
    """Validate physical bounds on f_tail and f_sub."""
    print("\n" + "=" * 70)
    print("VALIDATION 4: Physical Constraints")
    print("=" * 70)

    all_passed = True

    # Test extreme environments
    print("\nExtreme environment tests:")
    print("-" * 50)

    # Starburst (very high Σ, M)
    env_extreme = GravoturbulentEnv(Sigma=10000, Mach=50, eta_survive=1.0)
    result = gravoturbulent_summary(env_extreme)
    print(f"  Extreme starburst (Σ=10000, M=50, η=1.0):")
    print(f"    f_tail = {result.f_tail:.4f} (must be ≤1.0)")
    print(f"    f_sub = {result.f_sub:.4f} (must be ≤1.0)")

    if result.f_tail <= 1.0 and result.f_sub <= 1.0:
        print("    ✅ Physical bounds satisfied")
    else:
        print("    ❌ f_tail or f_sub exceeds 1.0!")
        all_passed = False

    # Diffuse cloud (low Σ, M)
    env_diffuse = GravoturbulentEnv(Sigma=10, Mach=3, eta_survive=0.1)
    result = gravoturbulent_summary(env_diffuse)
    print(f"\n  Very diffuse cloud (Σ=10, M=3, η=0.1):")
    print(f"    f_tail = {result.f_tail:.6f} (must be ≥0)")
    print(f"    f_sub = {result.f_sub:.6f} (must be ≥0)")

    if result.f_tail >= 0 and result.f_sub >= 0:
        print("    ✅ Physical bounds satisfied")
    else:
        print("    ❌ f_tail or f_sub is negative!")
        all_passed = False

    # η_survive = 0 test
    env_no_survive = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.0)
    result = gravoturbulent_summary(env_no_survive)
    print(f"\n  No survival (η=0):")
    print(f"    f_tail = {result.f_tail:.4f} (should be positive)")
    print(f"    f_sub = {result.f_sub:.6f} (must be 0)")

    if result.f_tail > 0 and result.f_sub == 0.0:
        print("    ✅ f_sub correctly zero when η=0")
    else:
        print("    ❌ f_sub should be exactly 0 when η=0")
        all_passed = False

    return all_passed


def generate_plots():
    """Generate validation plots."""
    print("\n" + "=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)

    # Plot 1: f_sub vs Σ for different Mach numbers
    print("\nPlot 1: f_sub vs Σ for different Mach numbers...")
    fig, ax = plt.subplots(figsize=(10, 6))

    Sigmas = np.logspace(1, 4, 100)
    Machs = [6, 10, 15, 20, 30]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(Machs)))

    for M, color in zip(Machs, colors):
        f_subs = []
        for Sigma in Sigmas:
            env = GravoturbulentEnv(Sigma=Sigma, Mach=M, eta_survive=0.7)
            result = gravoturbulent_summary(env)
            f_subs.append(result.f_sub)
        ax.plot(Sigmas, f_subs, color=color, lw=2, label=rf"$\mathcal{{M}} = {M}$")

    # Add preset markers
    preset_x = []
    preset_y = []
    preset_names = []
    for name in GRAVOTURBULENT_PRESETS:
        env = GRAVOTURBULENT_PRESETS[name]
        result = gravoturbulent_summary(env)
        preset_x.append(env.Sigma)
        preset_y.append(result.f_sub)
        preset_names.append(name)

    ax.scatter(preset_x, preset_y, c='red', s=100, marker='*', zorder=5, label='Presets')
    for x, y, name in zip(preset_x, preset_y, preset_names):
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xscale('log')
    ax.set_xlabel(r'Surface Density $\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]')
    ax.set_ylabel(r'Substructure Fraction $f_\mathrm{sub}$')
    ax.set_title(r'Gravoturbulent $f_\mathrm{sub}$ vs Cloud Surface Density' + '\n' + r'($\eta_\mathrm{survive} = 0.7$ fixed)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(10, 10000)
    ax.set_ylim(0, 0.6)

    plot_path = PLOT_DIR / "fsub_vs_sigma.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    # Plot 2: Presets comparison bar chart
    print("\nPlot 2: Presets comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sort presets by f_sub
    sorted_presets = sorted(GRAVOTURBULENT_PRESETS.keys(),
                           key=lambda x: gravoturbulent_summary(GRAVOTURBULENT_PRESETS[x]).f_sub)

    f_tails = []
    f_subs = []
    for name in sorted_presets:
        env = GRAVOTURBULENT_PRESETS[name]
        result = gravoturbulent_summary(env)
        f_tails.append(result.f_tail)
        f_subs.append(result.f_sub)

    x = np.arange(len(sorted_presets))
    width = 0.35

    # Left: f_tail (physics) vs f_sub (with survival)
    axes[0].bar(x - width/2, f_tails, width, label=r'$f_\mathrm{tail}$ (gas)', color='steelblue', alpha=0.8)
    axes[0].bar(x + width/2, f_subs, width, label=r'$f_\mathrm{sub}$ (stellar)', color='coral', alpha=0.8)
    axes[0].set_ylabel('Fraction')
    axes[0].set_title(r'Gas Tail Fraction vs Stellar Substructure')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sorted_presets, rotation=45, ha='right')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Right: Environment parameters
    Sigmas = [GRAVOTURBULENT_PRESETS[n].Sigma for n in sorted_presets]
    Machs = [GRAVOTURBULENT_PRESETS[n].Mach for n in sorted_presets]
    etas = [GRAVOTURBULENT_PRESETS[n].eta_survive for n in sorted_presets]

    ax2 = axes[1]
    ax2.bar(x - width, np.log10(Sigmas), width, label=r'$\log_{10}(\Sigma)$', color='forestgreen', alpha=0.8)
    ax2.bar(x, np.log10(Machs), width, label=r'$\log_{10}(\mathcal{M})$', color='purple', alpha=0.8)
    ax2.bar(x + width, etas, width, label=r'$\eta_\mathrm{survive}$', color='gold', alpha=0.8)
    ax2.set_ylabel('Value')
    ax2.set_title('Environment Parameters by Preset')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sorted_presets, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plot_path = PLOT_DIR / "presets.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    # Plot 3: Derivation chain visualization
    print("\nPlot 3: Derivation chain for Orion example...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Use Orion as example
    env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6)
    result = gravoturbulent_summary(env)

    # Panel 1: Lognormal PDF
    ax = axes[0, 0]
    s = np.linspace(-5, 10, 500)
    sigma_s = result.sigma_s
    sigma_s_sq = sigma_s**2
    pdf = np.exp(-(s + sigma_s_sq/2)**2 / (2 * sigma_s_sq)) / (np.sqrt(2 * np.pi) * sigma_s)
    ax.fill_between(s, pdf, where=(s > result.s_crit), alpha=0.5, color='coral', label=rf'$f_\mathrm{{tail}} = {result.f_tail:.3f}$')
    ax.plot(s, pdf, 'k-', lw=2)
    ax.axvline(result.s_crit, color='red', ls='--', lw=2, label=rf'$s_\mathrm{{crit}} = {result.s_crit:.2f}$')
    ax.axvline(0, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(r'$s = \ln(\rho/\rho_0)$')
    ax.set_ylabel(r'$p(s)$')
    ax.set_title(r'Lognormal Density PDF')
    ax.legend(loc='upper right')
    ax.set_xlim(-5, 10)

    # Panel 2: α_vir vs Σ
    ax = axes[0, 1]
    Sigmas_plot = np.logspace(1, 4, 100)
    alpha_virs = [env.alpha_0 * (env.Sigma_0 / S) for S in Sigmas_plot]
    ax.loglog(Sigmas_plot, alpha_virs, 'b-', lw=2)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.5, label=r'$\alpha_\mathrm{vir} = 1$ (bound)')
    ax.scatter([env.Sigma], [result.alpha_vir], c='red', s=100, zorder=5)
    ax.annotate(rf'Orion' + '\n' + rf'$\alpha={result.alpha_vir:.2f}$', (env.Sigma, result.alpha_vir),
                xytext=(10, 10), textcoords='offset points')
    ax.set_xlabel(r'Surface Density $\Sigma$ [$M_\odot\,\mathrm{pc}^{-2}$]')
    ax.set_ylabel(r'Virial Parameter $\alpha_\mathrm{vir}$')
    ax.set_title(r'$\alpha_\mathrm{vir} = \alpha_0 \times (\Sigma_0/\Sigma)$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: σ_s vs Mach
    ax = axes[0, 2]
    Machs_plot = np.linspace(2, 40, 100)
    sigma_s_values = [np.sqrt(np.log(1 + 0.4**2 * M**2)) for M in Machs_plot]
    ax.plot(Machs_plot, sigma_s_values, 'b-', lw=2)
    ax.scatter([env.Mach], [result.sigma_s], c='red', s=100, zorder=5)
    ax.annotate(rf'Orion' + '\n' + rf'$\sigma_s={result.sigma_s:.2f}$', (env.Mach, result.sigma_s),
                xytext=(10, -20), textcoords='offset points')
    ax.set_xlabel(r'Mach Number $\mathcal{M}$')
    ax.set_ylabel(r'PDF Width $\sigma_s$')
    ax.set_title(r'$\sigma_s = \sqrt{\ln(1 + b^2\mathcal{M}^2)}$')
    ax.grid(True, alpha=0.3)

    # Panel 4: s_crit vs α_vir×M²
    ax = axes[1, 0]
    alpha_M2 = np.logspace(-1, 4, 100)
    prefactor = np.pi**2 * 0.35**2 / 5.0
    s_crit_values = np.log(prefactor * alpha_M2)
    ax.semilogx(alpha_M2, s_crit_values, 'b-', lw=2)
    ax.scatter([result.alpha_vir * env.Mach**2], [result.s_crit], c='red', s=100, zorder=5)
    ax.annotate(rf'Orion' + '\n' + rf'$s_\mathrm{{crit}}={result.s_crit:.2f}$',
                (result.alpha_vir * env.Mach**2, result.s_crit),
                xytext=(10, 10), textcoords='offset points')
    ax.set_xlabel(r'$\alpha_\mathrm{vir} \times \mathcal{M}^2$')
    ax.set_ylabel(r'Critical Density $s_\mathrm{crit}$')
    ax.set_title(r'$s_\mathrm{crit} = \ln\left(\frac{\pi^2 \phi_x^2}{5} \alpha_\mathrm{vir} \mathcal{M}^2\right)$')
    ax.grid(True, alpha=0.3)

    # Panel 5: f_tail vs u_crit
    ax = axes[1, 1]
    u_values = np.linspace(-2, 4, 100)
    from scipy.special import erfc
    f_tail_values = 0.5 * erfc(u_values)
    ax.plot(u_values, f_tail_values, 'b-', lw=2)
    ax.scatter([result.u_crit], [result.f_tail], c='red', s=100, zorder=5)
    ax.annotate(rf'Orion' + '\n' + rf'$f_\mathrm{{tail}}={result.f_tail:.3f}$',
                (result.u_crit, result.f_tail),
                xytext=(10, 10), textcoords='offset points')
    ax.set_xlabel(r'$u_\mathrm{crit} = \frac{s_\mathrm{crit} - \sigma_s^2/2}{\sqrt{2}\,\sigma_s}$')
    ax.set_ylabel(r'Tail Fraction $f_\mathrm{tail}$')
    ax.set_title(r'$f_\mathrm{tail} = \frac{1}{2}\,\mathrm{erfc}(u_\mathrm{crit})$')
    ax.grid(True, alpha=0.3)

    # Panel 6: f_sub vs f_tail for different η
    ax = axes[1, 2]
    f_tail_range = np.linspace(0, 0.6, 100)
    etas_plot = [0.3, 0.5, 0.7, 0.9]
    for eta in etas_plot:
        ax.plot(f_tail_range, eta * f_tail_range, lw=2, label=rf'$\eta = {eta}$')
    ax.scatter([result.f_tail], [result.f_sub], c='red', s=100, zorder=5)
    ax.annotate(rf'Orion' + '\n' + rf'$f_\mathrm{{sub}}={result.f_sub:.3f}$',
                (result.f_tail, result.f_sub),
                xytext=(10, -10), textcoords='offset points')
    ax.set_xlabel(r'Gas Tail Fraction $f_\mathrm{tail}$')
    ax.set_ylabel(r'Stellar Fraction $f_\mathrm{sub}$')
    ax.set_title(r'$f_\mathrm{sub} = \eta_\mathrm{survive} \times f_\mathrm{tail}$')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.6)
    ax.set_ylim(0, 0.6)

    fig.suptitle(r'Gravoturbulent Derivation Chain: Orion Example' + '\n' + r'($\Sigma=150\,M_\odot\,\mathrm{pc}^{-2}$, $\mathcal{M}=12$, $\eta=0.6$)', fontsize=14)
    plot_path = PLOT_DIR / "derivation_chain.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {plot_path}")

    # Plot 4: Sensitivity analysis
    print("\nPlot 4: Sensitivity analysis...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Base environment (Orion-like)
    base_env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6)
    base_result = gravoturbulent_summary(base_env)

    # Vary b (turbulence driving)
    ax = axes[0]
    b_values = np.linspace(0.25, 1.0, 50)
    f_subs_b = []
    for b in b_values:
        env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6, b=b)
        result = gravoturbulent_summary(env)
        f_subs_b.append(result.f_sub)
    ax.plot(b_values, f_subs_b, 'b-', lw=2)
    ax.axvline(0.33, color='green', ls='--', alpha=0.7, label='Solenoidal (0.33)')
    ax.axvline(0.4, color='red', ls='--', alpha=0.7, label='Default (0.4)')
    ax.axvline(1.0, color='orange', ls='--', alpha=0.7, label='Compressive (1.0)')
    ax.set_xlabel(r'Turbulence Driving $b$')
    ax.set_ylabel(r'$f_\mathrm{sub}$')
    ax.set_title(r'Sensitivity to Turbulence Driving')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Vary φ_x (sonic scale)
    ax = axes[1]
    phi_x_values = np.linspace(0.1, 0.6, 50)
    f_subs_phi = []
    for phi_x in phi_x_values:
        env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6, phi_x=phi_x)
        result = gravoturbulent_summary(env)
        f_subs_phi.append(result.f_sub)
    ax.plot(phi_x_values, f_subs_phi, 'b-', lw=2)
    ax.axvline(0.17, color='green', ls='--', alpha=0.7, label=r'Strong $B$ (0.17)')
    ax.axvline(0.35, color='red', ls='--', alpha=0.7, label='Default (0.35)')
    ax.axvline(0.5, color='orange', ls='--', alpha=0.7, label=r'Weak $B$ (0.5)')
    ax.set_xlabel(r'Sonic Scale Factor $\phi_x$')
    ax.set_ylabel(r'$f_\mathrm{sub}$')
    ax.set_title(r'Sensitivity to Sonic Scale')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # Vary α₀ (reference virial)
    ax = axes[2]
    alpha_0_values = np.linspace(1.0, 4.0, 50)
    f_subs_alpha = []
    for alpha_0 in alpha_0_values:
        env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6, alpha_0=alpha_0)
        result = gravoturbulent_summary(env)
        f_subs_alpha.append(result.f_sub)
    ax.plot(alpha_0_values, f_subs_alpha, 'b-', lw=2)
    ax.axvline(2.0, color='red', ls='--', alpha=0.7, label='Default (2.0)')
    ax.axvspan(1.5, 2.5, alpha=0.2, color='gray', label=r'$\pm 25\%$ scatter')
    ax.set_xlabel(r'Reference Virial $\alpha_0$')
    ax.set_ylabel(r'$f_\mathrm{sub}$')
    ax.set_title(r'Sensitivity to Virial Reference')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.suptitle(r'Parameter Sensitivity Analysis (Orion base: $\Sigma=150$, $\mathcal{M}=12$, $\eta=0.6$)', fontsize=12)
    plot_path = PLOT_DIR / "sensitivity.png"
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"  Saved: {plot_path}")


def run_validation():
    """Run all validations and print summary."""
    print("\n" + "=" * 70)
    print("GRAVOTURBULENT f_sub VALIDATION")
    print("Burkhart (2018) Physics-Based Derivation")
    print("=" * 70)

    results = {}

    # Run validations
    results["worked_examples"], examples_data = validate_worked_examples()
    results["monotonicity"] = validate_monotonicity()
    results["presets"] = validate_presets()
    results["physical_constraints"] = validate_physical_constraints()

    # Generate plots
    generate_plots()

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = all(results.values())

    for name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name:<25} {status}")

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VALIDATIONS PASSED")
    else:
        print("❌ SOME VALIDATIONS FAILED")
    print("=" * 70)

    print("\nPlots saved to:")
    for plot in sorted(PLOT_DIR.glob("*.png")):
        print(f"  - {plot}")

    return all_passed


if __name__ == "__main__":
    success = run_validation()
    sys.exit(0 if success else 1)
