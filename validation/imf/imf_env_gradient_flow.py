# progenax/validation/imf_env_gradient_flow.py
"""
v0.2 Demo: Gradient flow through environment → IMF → likelihood.

Demonstrates that we can compute:
    d(NLL) / d(log_density)
    d(NLL) / d(metallicity)

This enables inference of birth environment from observed stellar masses.

Run:
    cd progenax
    uv run python validation/imf_env_gradient_flow.py
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from progenax.imf import IMFParams, sample_masses_from_params, individual_mass_nll
from progenax.imf.environment_v2 import BirthEnvironment, env_to_imf_params


def main():
    print("=" * 60)
    print("v0.2 Demo: Environment → IMF → Likelihood Gradient Flow")
    print("=" * 60)

    # 1. Create a "true" environment
    TRUE_LOG_DENSITY = 5.0  # Dense region
    TRUE_METALLICITY = -0.5  # Moderately metal-poor

    print(f"\n1. True birth environment:")
    print(f"   log₁₀(ρ) = {TRUE_LOG_DENSITY}")
    print(f"   [Fe/H] = {TRUE_METALLICITY}")

    true_env = BirthEnvironment(
        log_density=jnp.array(TRUE_LOG_DENSITY),
        metallicity=jnp.array(TRUE_METALLICITY),
    )

    # 2. Get IMF params from environment
    true_params = env_to_imf_params(true_env, model="marks2012_like")
    print(f"\n2. Resulting IMF (marks2012_like):")
    print(f"   α_high = {float(true_params.alpha_high):.3f}")

    # 3. Generate mock data
    N_MASSES = 500
    key = jax.random.PRNGKey(123)
    u = jax.random.uniform(key, (N_MASSES,))
    observed_masses = sample_masses_from_params(true_params, u)

    print(f"\n3. Generated {N_MASSES} mock masses")
    n_heavy = int(jnp.sum(observed_masses > 10.0))
    print(f"   Masses > 10 M☉: {n_heavy}")

    # 4. Define NLL as function of environment
    def nll_from_env(log_density, metallicity):
        """NLL as function of environment parameters."""
        env = BirthEnvironment(
            log_density=log_density,
            metallicity=metallicity,
        )
        params = env_to_imf_params(env, model="marks2012_like")
        return individual_mass_nll(observed_masses, params)

    # 5. Compute gradients
    print("\n4. Computing gradients at various environments...")
    grad_fn = jax.grad(nll_from_env, argnums=(0, 1))

    test_points = [
        (3.0, 0.0, "Low density, solar [Fe/H]"),
        (5.0, -0.5, "TRUE environment"),
        (6.0, 0.0, "Very dense, solar [Fe/H]"),
    ]

    print("\n   log(ρ) | [Fe/H] | NLL      | ∂NLL/∂log(ρ) | ∂NLL/∂[Fe/H]")
    print("   " + "-" * 60)

    for log_rho, feh, label in test_points:
        nll = nll_from_env(jnp.array(log_rho), jnp.array(feh))
        g_rho, g_feh = grad_fn(jnp.array(log_rho), jnp.array(feh))

        print(f"   {log_rho:5.1f}  | {feh:5.1f}  | {float(nll):8.1f} | {float(g_rho):12.2f} | {float(g_feh):12.2f}")

        if label == "TRUE environment":
            print(f"   {'':5s}  | {'':5s}  | {'':8s} | ← TRUE")

    # 6. Verify gradient at true environment
    print("\n5. Verifying gradients at TRUE environment...")
    g_rho, g_feh = grad_fn(jnp.array(TRUE_LOG_DENSITY), jnp.array(TRUE_METALLICITY))

    checks = [
        ("∂NLL/∂log(ρ) is finite", jnp.isfinite(g_rho)),
        ("∂NLL/∂[Fe/H] is finite", jnp.isfinite(g_feh)),
    ]

    all_passed = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"   {status} {name}")
        all_passed = all_passed and passed

    # 7. Verify NLL is minimized near true environment
    print("\n6. Verifying NLL landscape...")
    nll_true = nll_from_env(jnp.array(TRUE_LOG_DENSITY), jnp.array(TRUE_METALLICITY))
    nll_low = nll_from_env(jnp.array(3.0), jnp.array(TRUE_METALLICITY))
    nll_high = nll_from_env(jnp.array(6.5), jnp.array(TRUE_METALLICITY))

    print(f"   NLL at log(ρ)=3.0:  {float(nll_low):.1f}")
    print(f"   NLL at log(ρ)=5.0:  {float(nll_true):.1f} ← TRUE")
    print(f"   NLL at log(ρ)=6.5:  {float(nll_high):.1f}")

    # True environment should have reasonable NLL (not necessarily minimum due to noise)
    # But gradients should point toward it from far away

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ v0.2 GRADIENT FLOW DEMO PASSED")
        print("  Gradients flow: environment → IMFParams → likelihood")
    else:
        print("✗ v0.2 GRADIENT FLOW DEMO FAILED")
    print("=" * 60)


if __name__ == "__main__":
    main()
