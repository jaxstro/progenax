"""Tests for DifferentiableBinaryFraction and DifferentiableBinaryModel.

TDD: These tests are written BEFORE the implementation.
"""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class TestDifferentiableBinaryFraction:
    """Test the smooth mass-dependent binary fraction model."""

    def test_import(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        assert DifferentiableBinaryFraction is not None

    def test_from_moe2017_factory(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        assert hasattr(dbf, "a")
        assert hasattr(dbf, "b")

    def test_returns_fraction_in_0_1(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        for m in [0.1, 0.5, 1.0, 5.0, 20.0]:
            fb = dbf(jnp.float64(m))
            assert 0.0 < float(fb) < 1.0, f"f_b({m}) = {float(fb)} not in (0, 1)"

    def test_increases_with_mass(self):
        """Moe+2017: binary fraction increases from M-dwarfs to O-stars."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        fb_low = float(dbf(jnp.float64(0.3)))
        fb_high = float(dbf(jnp.float64(10.0)))
        assert fb_high > fb_low, f"f_b should increase with mass: {fb_low} vs {fb_high}"

    def test_matches_moe2017_within_10pct(self):
        """Smooth fit should be within ~10% of step function at representative masses."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        # Moe+2017 Table 13 values at representative masses
        expected = [(0.75, 0.44), (1.5, 0.50), (7.5, 0.80)]
        for m, fb_expected in expected:
            fb_actual = float(dbf(jnp.float64(m)))
            assert abs(fb_actual - fb_expected) < 0.10, (
                f"m={m}: expected {fb_expected}, got {fb_actual}"
            )

    def test_differentiable_wrt_a(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        grad = jax.grad(
            lambda a: DifferentiableBinaryFraction(a=a, b=dbf.b, c=dbf.c)(1.0)
        )(dbf.a)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0

    def test_differentiable_wrt_b(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        grad = jax.grad(
            lambda b: DifferentiableBinaryFraction(a=dbf.a, b=b, c=dbf.c)(2.0)
        )(dbf.b)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0

    def test_vmap_compatible(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        masses = jnp.array([0.3, 1.0, 5.0, 20.0])
        fbs = jax.vmap(dbf)(masses)
        assert fbs.shape == (4,)
        assert jnp.all(jnp.isfinite(fbs))

    def test_is_equinox_module(self):
        import equinox as eqx

        from progenax.imf.differentiable_binary import DifferentiableBinaryFraction

        dbf = DifferentiableBinaryFraction.from_moe2017()
        assert isinstance(dbf, eqx.Module)


class TestDifferentiableBinaryModel:
    """Test the full differentiable binary population model."""

    def test_import(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        assert DifferentiableBinaryModel is not None

    def test_moe2017_factory(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        model = DifferentiableBinaryModel.moe2017()
        assert hasattr(model, "binary_fraction")
        assert hasattr(model, "temperature")

    def test_sample_systems_returns_correct_shapes(self):
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        model = DifferentiableBinaryModel.moe2017()
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        N = 100
        m1 = jnp.ones(N) * 1.0  # 1 Msun primaries
        u_binary = jax.random.uniform(k1, (N,))
        u_q = jax.random.uniform(k2, (N,))

        m2, soft_weights = model.sample_systems(m1, u_binary, u_q)
        assert m2.shape == (N,)
        assert soft_weights.shape == (N,)

    def test_soft_weights_near_binary(self):
        """At low temperature, most weights should be near 0 or 1."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        model = DifferentiableBinaryModel.moe2017(temperature=0.001)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        N = 1000
        m1 = jnp.ones(N)
        u_binary = jax.random.uniform(k1, (N,))
        u_q = jax.random.uniform(k2, (N,))

        _, soft_weights = model.sample_systems(m1, u_binary, u_q)

        # At T=0.001, <1% of stars should have weights between 0.1 and 0.9
        marginal = jnp.sum((soft_weights > 0.1) & (soft_weights < 0.9))
        assert int(marginal) < N * 0.02, (
            f"{int(marginal)} stars are marginal (expected < {N * 0.02})"
        )

    def test_m2_positive(self):
        """Secondary masses should be positive."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        model = DifferentiableBinaryModel.moe2017()
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        m1 = jnp.array([0.5, 1.0, 5.0, 20.0])
        u_binary = jax.random.uniform(k1, (4,))
        u_q = jax.random.uniform(k2, (4,))

        m2, _ = model.sample_systems(m1, u_binary, u_q)
        assert jnp.all(m2 > 0), f"m2 has non-positive values: {m2}"

    def test_m2_less_than_m1(self):
        """Secondary mass should be <= primary mass (q <= 1)."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        model = DifferentiableBinaryModel.moe2017()
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        m1 = jnp.array([0.5, 1.0, 5.0, 20.0])
        u_binary = jax.random.uniform(k1, (4,))
        u_q = jax.random.uniform(k2, (4,))

        m2, _ = model.sample_systems(m1, u_binary, u_q)
        assert jnp.all(m2 <= m1 * 1.001), f"m2 > m1: {m2} vs {m1}"

    def test_gradient_wrt_temperature(self):
        """Gradient flows through temperature parameter."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        m1 = jnp.ones(100)
        u_b = jax.random.uniform(k1, (100,))
        u_q = jax.random.uniform(k2, (100,))

        def total_flux(temp):
            model = DifferentiableBinaryModel.moe2017(temperature=temp)
            m2, w = model.sample_systems(m1, u_b, u_q)
            return jnp.sum(w * m2)

        grad = jax.grad(total_flux)(0.01)
        assert jnp.isfinite(grad)

    def test_gradient_wrt_binary_fraction_params(self):
        """Gradient flows through f_b parameters (a, b)."""
        from progenax.imf.differentiable_binary import (
            DifferentiableBinaryFraction,
            DifferentiableBinaryModel,
        )

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        m1 = jnp.ones(100)
        u_b = jax.random.uniform(k1, (100,))
        u_q = jax.random.uniform(k2, (100,))

        def total_weighted_m2(a_fb):
            bf = DifferentiableBinaryFraction(a=a_fb, b=1.4170, c=0.4755)
            model = DifferentiableBinaryModel(
                binary_fraction=bf,
                gamma_intercept=0.1907,
                gamma_slope=-0.7521,
                temperature=0.01,
            )
            m2, w = model.sample_systems(m1, u_b, u_q)
            return jnp.sum(w * m2)

        grad = jax.grad(total_weighted_m2)(0.0416)
        assert jnp.isfinite(grad)
        assert float(grad) != 0.0

    def test_binary_fraction_statistics(self):
        """At low T, the realized binary fraction should match f_b(m)."""
        from progenax.imf.differentiable_binary import DifferentiableBinaryModel

        model = DifferentiableBinaryModel.moe2017(temperature=0.001)
        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        N = 5000
        m1 = jnp.ones(N)  # 1 Msun: f_b should be ~0.45-0.55
        u_b = jax.random.uniform(k1, (N,))
        u_q = jax.random.uniform(k2, (N,))

        _, soft_weights = model.sample_systems(m1, u_b, u_q)
        realized_fb = float(jnp.mean(soft_weights))

        # Should be near the Moe+2017 value for 1 Msun (~0.44-0.55)
        assert 0.35 < realized_fb < 0.65, (
            f"Realized f_b = {realized_fb}, expected ~0.5 for 1 Msun"
        )
