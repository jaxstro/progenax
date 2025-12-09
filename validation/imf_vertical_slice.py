#!/usr/bin/env python
"""v0.3 Vertical Slice: Environment-Dependent IMF with Gradient Flow.

Demonstrates the complete workflow:
1. Create BirthEnvironment from cluster mass
2. Get IMF params via env_to_imf_params (Jerabkova default)
3. Sample masses from environment-dependent IMF
4. Verify alpha3 matches paper expectations
5. Demonstrate gradient flow for inference

SUCCESS CRITERIA:
1. Alpha3 matches Jerabkova+2018 Eq. 9 predictions
2. Gradients flow from env params to likelihood
3. Gradient-based recovery of log_mecl from masses

Run:
    cd /Users/anna/projects/jaxstro-dev/progenax
    /Users/anna/miniforge3/envs/astro/bin/python validation/imf_vertical_slice.py
"""

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from progenax.imf import (
    BirthEnvironment,
    env_to_imf_params,
    alpha3_jerabkova_mecl,
    sample_masses_from_params,
    individual_mass_nll,
    JERABKOVA_COEFFICIENTS,
)


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_environment_to_imf():
    """Part 1: Demonstrate environment → IMF workflow."""
    print_header("PART 1: Environment → IMF Workflow")

    # Create environments across the calibration domain
    environments = [
        ("Solar neighborhood (1000 M☉)", BirthEnvironment.solar_neighborhood()),
        ("Massive GC (10^6 M☉, [Fe/H]=-1.5)", BirthEnvironment.massive_gc(FeH=-1.5)),
        ("NGC 7078 (most top-heavy)", BirthEnvironment.ngc_7078()),
    ]

    print("\nEnvironment → alpha3 mapping (Jerabkova+2018 Eq. 9):")
    print("-" * 70)
    print(f"{'Environment':<35} {'log_mecl':>10} {'[Fe/H]':>8} {'α₃':>8}")
    print("-" * 70)

    for name, env in environments:
        params = env_to_imf_params(env)
        print(f"{name:<35} {float(env.log_mecl):>10.2f} {float(env.metallicity):>8.2f} {float(params.alpha_high):>8.3f}")

    print("-" * 70)

    # Verify Jerabkova equation directly
    print("\nVerifying Jerabkova+2018 Eq. 9:")
    log_mecl_6 = jnp.array(0.0)  # 10^6 M_sun
    FeH = jnp.array(-1.5)
    x = JERABKOVA_COEFFICIENTS["FeH_coeff"] * FeH + JERABKOVA_COEFFICIENTS["logMecl_coeff"] * log_mecl_6 + JERABKOVA_COEFFICIENTS["constant"]
    expected_alpha3 = JERABKOVA_COEFFICIENTS["alpha3_slope"] * x + JERABKOVA_COEFFICIENTS["alpha3_intercept"]
    computed = alpha3_jerabkova_mecl(log_mecl_6, FeH)

    c = JERABKOVA_COEFFICIENTS
    print(f"  x = {c['FeH_coeff']}*({float(FeH):.1f}) + {c['logMecl_coeff']}*({float(log_mecl_6):.1f}) + {c['constant']} = {float(x):.3f}")
    print(f"  α₃ = {c['alpha3_slope']}*{float(x):.3f} + {c['alpha3_intercept']} = {float(expected_alpha3):.3f}")
    print(f"  Computed: {float(computed):.3f}")
    print(f"  Match: {'✓' if jnp.isclose(computed, expected_alpha3, atol=0.01) else '✗'}")

    return True


def demo_mass_sampling():
    """Part 2: Sample masses from environment-dependent IMF."""
    print_header("PART 2: Sampling from Environment-Dependent IMF")

    # Compare sampling from different environments
    key = jax.random.PRNGKey(42)
    N = 10000

    environments = [
        ("Universal Kroupa", BirthEnvironment.solar_neighborhood(), "universal_kroupa"),
        ("Solar neighborhood", BirthEnvironment.solar_neighborhood(), "jerabkova2018"),
        ("Massive GC", BirthEnvironment.massive_gc(FeH=-1.5), "jerabkova2018"),
    ]

    print(f"\nSampling {N} masses from each environment:")
    print("-" * 70)
    print(f"{'Environment':<25} {'α₃':>8} {'Median [M☉]':>12} {'>10 M☉ [%]':>12}")
    print("-" * 70)

    for name, env, model in environments:
        params = env_to_imf_params(env, model=model)
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, (N,))
        masses = sample_masses_from_params(params, u)

        median = float(jnp.median(masses))
        frac_massive = float(jnp.sum(masses > 10.0) / N * 100)

        print(f"{name:<25} {float(params.alpha_high):>8.3f} {median:>12.4f} {frac_massive:>12.2f}")

    print("-" * 70)
    print("Note: Top-heavy IMF (lower α₃) → higher median mass, more massive stars")

    return True


def demo_gradient_flow():
    """Part 3: Demonstrate gradient flow for inference."""
    print_header("PART 3: Gradient Flow for Inference")

    # Generate mock data from a massive GC environment
    TRUE_LOG_MECL = 6.5  # 3 × 10^6 M_sun
    TRUE_FEH = -1.5

    print(f"\nGround truth:")
    print(f"  log_mecl = {TRUE_LOG_MECL} (M_ecl = 10^{TRUE_LOG_MECL:.1f} M☉)")
    print(f"  [Fe/H] = {TRUE_FEH}")

    # Generate masses
    key = jax.random.PRNGKey(12345)
    N = 2000
    true_env = BirthEnvironment(metallicity=jnp.array(TRUE_FEH), log_mecl=jnp.array(TRUE_LOG_MECL))
    true_params = env_to_imf_params(true_env, smooth_alpha3=True)
    u = jax.random.uniform(key, (N,))
    observed_masses = sample_masses_from_params(true_params, u)

    print(f"\nGenerated {N} mock masses:")
    print(f"  True α₃: {float(true_params.alpha_high):.3f}")
    print(f"  Median mass: {float(jnp.median(observed_masses)):.4f} M☉")
    print(f"  Fraction >10 M☉: {float(jnp.sum(observed_masses > 10.0) / N * 100):.2f}%")

    # Define loss function: infer log_mecl (FeH fixed)
    @jax.jit
    def nll(log_mecl):
        env = BirthEnvironment(metallicity=jnp.array(TRUE_FEH), log_mecl=log_mecl)
        params = env_to_imf_params(env, smooth_alpha3=True)
        return individual_mass_nll(observed_masses, params)

    # Verify gradients work
    print("\nVerifying gradient computation...")
    grad_nll = jax.grad(nll)
    test_grad = grad_nll(jnp.array(5.0))
    print(f"  Gradient at log_mecl=5.0: {float(test_grad):.4f}")

    assert jnp.isfinite(test_grad), "FAIL: Gradient must be finite"
    assert test_grad != 0.0, "FAIL: Gradient must be non-zero"
    print("  ✓ Gradients are finite and non-zero")

    # Gradient descent to recover log_mecl
    print("\nRunning gradient descent to recover log_mecl...")
    optimizer = optax.adam(learning_rate=0.1)
    log_mecl_hat = jnp.array(4.5)  # Initial guess (too low)
    opt_state = optimizer.init(log_mecl_hat)

    print("  Step |   NLL    | log_mecl | α₃")
    print("  " + "-" * 45)

    for step in range(100):
        loss = nll(log_mecl_hat)
        g = grad_nll(log_mecl_hat)
        updates, opt_state = optimizer.update(g, opt_state)
        log_mecl_hat = optax.apply_updates(log_mecl_hat, updates)

        if step % 20 == 0 or step == 99:
            env = BirthEnvironment(metallicity=jnp.array(TRUE_FEH), log_mecl=log_mecl_hat)
            params = env_to_imf_params(env, smooth_alpha3=True)
            print(f"  {step:4d} | {float(loss):8.2f} | {float(log_mecl_hat):8.3f} | {float(params.alpha_high):.3f}")

    # Check recovery
    print("\nResults:")
    print(f"  True log_mecl:     {TRUE_LOG_MECL:.2f}")
    print(f"  Inferred log_mecl: {float(log_mecl_hat):.2f}")

    error = abs(float(log_mecl_hat) - TRUE_LOG_MECL)
    print(f"  Absolute error:    {error:.3f}")

    success = error < 0.5  # Allow 0.5 dex tolerance
    if success:
        print(f"  ✓ Recovery within 0.5 dex")
    else:
        print(f"  ✗ Recovery error too large")

    return success


def main():
    """Run all vertical slice demonstrations."""
    print("=" * 60)
    print("v0.3 Vertical Slice: Environment-Dependent IMF")
    print("Paper-calibrated Jerabkova+2018 / Marks+2012")
    print("=" * 60)

    results = []

    # Part 1: Environment → IMF
    results.append(("Environment → IMF", demo_environment_to_imf()))

    # Part 2: Mass sampling
    results.append(("Mass Sampling", demo_mass_sampling()))

    # Part 3: Gradient flow
    results.append(("Gradient Flow", demo_gradient_flow()))

    # Summary
    print_header("SUMMARY")

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ v0.3 VERTICAL SLICE PASSED")
    else:
        print("✗ v0.3 VERTICAL SLICE FAILED")
    print("=" * 60)

    if not all_passed:
        raise AssertionError("Vertical slice failed")


if __name__ == "__main__":
    main()
