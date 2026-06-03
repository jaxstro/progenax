"""Energy/virial single-source-of-truth + gradient-safety (Batch 0, F1+F2+F10).

Guards against the duplicated-physics regression where the softening=0 gradient
fix lived in builders.py but not in the dynamics/virial.py copy used by
cluster/ and kinematics/api.py.
"""
import jax
import jax.numpy as jnp


class TestEnergyGradientSafety:
    """compute_potential_energy must have a finite gradient even at softening=0."""

    def test_potential_energy_softening_zero_grad_is_finite(self):
        from progenax.dynamics.virial import compute_potential_energy

        key = jax.random.PRNGKey(3)
        pos = jax.random.normal(key, (10, 3))
        masses = jnp.ones(10)
        G = 0.00450
        grad = jax.grad(lambda p: compute_potential_energy(p, masses, G, 0.0))(pos)
        assert jnp.all(jnp.isfinite(grad)), (
            "softening=0 gradient of dynamics.compute_potential_energy must be finite "
            "(double-where diagonal guard), not NaN"
        )

    def test_potential_energy_softening_zero_value_matches_softened_limit(self):
        """Forward value at softening=0 stays correct (sharp, no spurious floor)."""
        from progenax.dynamics.virial import compute_potential_energy

        # Two unit masses at separation 2 -> V = -G * 1 / 2
        pos = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        masses = jnp.ones(2)
        G = 1.0
        V = compute_potential_energy(pos, masses, G, 0.0)
        assert jnp.isclose(V, -0.5, atol=1e-12)


class TestSingleSourceOfTruth:
    """builders.py must re-export the dynamics energy functions (one implementation)."""

    def test_builders_energy_is_dynamics_energy(self):
        from progenax.builders import (
            compute_potential_energy as pe_b,
            compute_kinetic_energy as ke_b,
        )
        from progenax.dynamics.virial import (
            compute_potential_energy as pe_d,
            compute_kinetic_energy as ke_d,
        )
        assert pe_b is pe_d, "builders.compute_potential_energy must be the dynamics one"
        assert ke_b is ke_d, "builders.compute_kinetic_energy must be the dynamics one"
