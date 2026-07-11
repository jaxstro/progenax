"""
v0.1 Vertical Slice: Recover alpha_high from mock masses via gradient descent.

SUCCESS CRITERIA:
1. Gradients are finite and non-zero
2. JIT compilation works
3. |α̂_high - α_true| < 0.1 for N=1000 masses

Run:
    cd progenax
    uv run python validation/imf_inference_vertical_slice.py
"""

import jax
import jax.numpy as jnp
import optax

jax.config.update("jax_enable_x64", True)

from progenax.imf import (
    IMFParams,
    sample_masses_from_params,
    individual_mass_nll,
)


def main():
    print("=" * 60)
    print("v0.1 Vertical Slice: Differentiable IMF Inference")
    print("=" * 60)

    # 1. Ground truth: slightly top-heavy IMF
    TRUE_ALPHA_HIGH = 2.5
    print(f"\n1. Ground truth: α_high = {TRUE_ALPHA_HIGH}")

    true_params = IMFParams(
        alpha_low=jnp.array(0.3),
        alpha_mid=jnp.array(1.3),
        alpha_high=jnp.array(TRUE_ALPHA_HIGH),
    )

    # 2. Generate mock data
    N_MASSES = 1000
    key = jax.random.PRNGKey(42)
    u = jax.random.uniform(key, (N_MASSES,))
    observed_masses = sample_masses_from_params(true_params, u)

    print(f"\n2. Generated {N_MASSES} mock masses")
    print(f"   Mass range: [{float(observed_masses.min()):.3f}, {float(observed_masses.max()):.2f}] M☉")
    print(f"   Median: {float(jnp.median(observed_masses)):.3f} M☉")

    # 3. Define loss function (only alpha_high is free)
    @jax.jit
    def nll(alpha_high):
        params = IMFParams(
            alpha_low=jnp.array(0.3),   # Fixed
            alpha_mid=jnp.array(1.3),   # Fixed
            alpha_high=alpha_high,
        )
        return individual_mass_nll(observed_masses, params)

    # 4. Verify gradients work
    print("\n3. Verifying gradient computation...")
    grad_nll = jax.grad(nll)
    test_grad = grad_nll(jnp.array(2.3))
    print(f"   Gradient at α=2.3: {float(test_grad):.6f}")

    assert jnp.isfinite(test_grad), "FAIL: Gradient must be finite"
    assert test_grad != 0.0, "FAIL: Gradient must be non-zero"
    print("   ✓ Gradients are finite and non-zero")

    # 5. Optimize
    print("\n4. Running gradient descent...")
    optimizer = optax.adam(learning_rate=0.05)
    alpha_hat = jnp.array(2.3)  # Initial guess (canonical Kroupa)
    opt_state = optimizer.init(alpha_hat)

    print("   Step |   NLL    | α_high")
    print("   " + "-" * 30)

    for step in range(150):
        loss = nll(alpha_hat)
        g = grad_nll(alpha_hat)
        updates, opt_state = optimizer.update(g, opt_state)
        alpha_hat = optax.apply_updates(alpha_hat, updates)

        if step % 25 == 0 or step == 149:
            print(f"   {step:4d} | {float(loss):8.2f} | {float(alpha_hat):.4f}")

    # 6. Verify recovery
    print("\n5. Results:")
    print(f"   True α_high:     {TRUE_ALPHA_HIGH:.3f}")
    print(f"   Inferred α̂_high: {float(alpha_hat):.3f}")

    error = abs(float(alpha_hat) - TRUE_ALPHA_HIGH)
    print(f"   Absolute error:  {error:.4f}")

    # Success criteria
    print("\n6. Success criteria:")
    if error < 0.1:
        print(f"   ✓ |α̂ - α_true| = {error:.4f} < 0.1")
        print("\n" + "=" * 60)
        print("✓ v0.1 VERTICAL SLICE PASSED")
        print("=" * 60)
    else:
        print(f"   ✗ |α̂ - α_true| = {error:.4f} >= 0.1")
        print("\n" + "=" * 60)
        print("✗ v0.1 VERTICAL SLICE FAILED")
        print("=" * 60)
        raise AssertionError(f"Recovery failed: error={error:.4f} >= 0.1")


if __name__ == "__main__":
    main()
