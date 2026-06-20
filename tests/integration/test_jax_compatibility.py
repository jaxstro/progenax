"""
JAX compatibility tests for progenax modules.

Consolidated tests ensuring core modules work with JIT, grad, and vmap.
One test per category per module type.
"""

import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR

from progenax.imf import PowerLawIMF
from progenax.kinematics import PlummerVelocityDF
from progenax.profiles import PlummerProfile

G = STELLAR.G  # ≈ 0.00450 [pc³ Msun⁻¹ Myr⁻²]


class TestProfileJAXCompatibility:
    """Test spatial profiles work with JAX transformations."""

    def test_plummer_jit(self):
        """Plummer sample_positions works under JIT."""
        profile = PlummerProfile(r_h=1.0)
        masses = jnp.ones(100)
        key = jax.random.PRNGKey(42)

        @jax.jit
        def sample(key):
            return profile.sample_positions(masses, key)

        positions = sample(key)
        assert positions.shape == (100, 3)
        assert jnp.all(jnp.isfinite(positions))

    # The gradient of PlummerProfile.sample_positions wrt r_h is FD-audited by the
    # grad-audit registry (tests/validation/grad_audit/registry.py ::
    # PlummerProfile.sample_positions [r_h]); see
    # docs/website/50-validation/differentiability-audit.md. The former finite-only
    # test_plummer_grad smoke was removed (audit T6: a silently-zeroed grad would PASS
    # isfinite; the registry FD case is strictly stronger; registry is SoT).


class TestVelocityDFJAXCompatibility:
    """Test velocity DFs work with JAX transformations."""

    def test_plummer_df_jit(self):
        """Plummer velocity sampling works under JIT."""
        df = PlummerVelocityDF(r_h=1.0)
        N = 100
        positions = jax.random.normal(jax.random.PRNGKey(0), (N, 3))
        masses = jnp.ones(N)

        @jax.jit
        def sample(key):
            return df.sample_velocities(positions, masses, key, G=G)

        velocities = sample(jax.random.PRNGKey(42))
        assert velocities.shape == (N, 3)
        assert jnp.all(jnp.isfinite(velocities))

    # The gradient of PlummerVelocityDF.sample_velocities wrt r_h is FD-audited by the
    # grad-audit registry (tests/validation/grad_audit/registry.py ::
    # PlummerVelocityDF.sample_velocities [r_h]); see
    # docs/website/50-validation/differentiability-audit.md. The former finite-only
    # test_plummer_df_grad smoke was removed (audit T6: isfinite passes a silently-zeroed
    # grad; the registry FD case is strictly stronger; registry is SoT).


class TestIMFJAXCompatibility:
    """Test IMFs work with JAX transformations."""

    def test_powerlaw_jit(self):
        """Power-law IMF sampling works under JIT."""
        imf = PowerLawIMF.kroupa()

        @jax.jit
        def sample(key):
            return imf.sample(key, 100)

        masses = sample(jax.random.PRNGKey(42))
        assert masses.shape == (100,)
        assert jnp.all(jnp.isfinite(masses))
        assert jnp.all(masses >= imf.m_min)
        assert jnp.all(masses <= imf.m_max)

    def test_powerlaw_grad(self):
        """Gradient flows through power-law PPF."""
        imf = PowerLawIMF.kroupa()

        def total_mass(u):
            return jnp.sum(imf.ppf(u))

        grad_fn = jax.grad(total_mass)
        u = jnp.array([0.3, 0.5, 0.7])
        grads = grad_fn(u)

        assert jnp.all(jnp.isfinite(grads))
        # PPF is monotonic, so dm/du > 0
        assert jnp.all(grads > 0)

    def test_powerlaw_vmap(self):
        """Power-law PPF works with vmap over batches."""
        imf = PowerLawIMF.kroupa()

        # Batch of uniform samples
        u_batch = jnp.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
            ]
        )

        # vmap over first axis
        batched_ppf = jax.vmap(imf.ppf)
        masses = batched_ppf(u_batch)

        assert masses.shape == (3, 3)
        assert jnp.all(jnp.isfinite(masses))


# The full Plummer IC pipeline gradient wrt r_h (profile x DF) is FD-audited by the
# grad-audit registry (tests/validation/grad_audit/registry.py ::
# build_spatial_ic[Plummer] [r_h positions] + build_spatial_ic[Plummer].velocities
# [r_h speeds], which run the SAME params->IC path end-to-end); see
# docs/website/50-validation/differentiability-audit.md. The former finite-only
# TestPipelineDifferentiability::test_plummer_ic_grad_wrt_r_h smoke was removed (audit
# T6: isfinite passes a silently-zeroed grad; the registry FD cases are strictly
# stronger; registry is SoT).


# AD-vs-FD through the public build_spatial_ic (the CLAUDE.md headline 'fully
# differentiable' path, audit CR-FU-2) is owned by the grad-audit registry
# (tests/validation/grad_audit/registry.py :: build_spatial_ic[Plummer] [r_h positions]
# + build_spatial_ic[Plummer].velocities [r_h speeds]); see
# docs/website/50-validation/differentiability-audit.md. The former
# test_build_spatial_ic_differentiable_wrt_r_h was removed here (audit T6 consolidation;
# registry is SoT). (The finite-only smoke tests in this file are a separate batch.)


def test_compute_potential_energy_grad_finite_at_default_softening():
    """grad of the public compute_potential_energy at the default softening=0 (the
    CLAUDE.md C1 example form) must be finite and FD-correct (double-where; audit 🟠)."""
    import jax
    import jax.numpy as jnp
    from jaxstro.units import STELLAR

    from progenax import compute_potential_energy

    pos = jax.random.normal(jax.random.PRNGKey(1), (16, 3))
    m = jnp.ones(16)
    f = lambda p: compute_potential_energy(p, m, G=STELLAR.G)  # softening=0 default
    g = jax.grad(f)(pos)
    assert jnp.all(jnp.isfinite(g)), "grad not finite at softening=0"
    # FD check on a random direction
    v = jax.random.normal(jax.random.PRNGKey(2), pos.shape)
    v = v / jnp.linalg.norm(v)
    fd = (f(pos + 1e-5 * v) - f(pos - 1e-5 * v)) / 2e-5
    ad = jnp.sum(g * v)
    assert abs(ad - fd) / (abs(ad) + abs(fd) + 1e-30) < 1e-5, f"ad {ad} vs fd {fd}"


# NOTE: the FDF density-field differentiability test was retired in P5 with the legacy
# cluster.fdf_density subsystem. Its replacement — grad through the rank-copula CDF table —
# is covered by tests/experimental/unit/test_copula.py
# (test_rank_copula_differentiable_in_alpha, test_mass_conserving_differentiable_in_alpha).
