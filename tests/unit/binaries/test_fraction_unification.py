"""Unified binary-fraction protocol (Batch 4h).

All binary-fraction models (mass-based in imf/binary, radial in binaries/) conform
to a single `BinaryFractionModel` protocol: `probability(masses, radii=None) ->
f_bin`. Mass models ignore radii; radial ignores masses; CombinedBinaryFraction
modulates one by the other.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.binary import ConstantBinaryFraction, MassDependentBinaryFraction
from progenax.imf.differentiable_binary import DifferentiableBinaryFraction
from progenax.binaries import RadialBinaryFraction

MASSES = jnp.array([0.3, 1.0, 5.0, 20.0])
RADII = jnp.array([0.1, 1.0, 3.0, 8.0])


class TestBinaryFractionProtocol:
    def test_all_models_conform(self):
        from progenax.protocols import BinaryFractionModel
        for model in (
            ConstantBinaryFraction(0.5),
            MassDependentBinaryFraction(),
            DifferentiableBinaryFraction(a=0.0, b=0.5, c=0.0),
            RadialBinaryFraction(),
        ):
            assert isinstance(model, BinaryFractionModel), type(model)

    def test_mass_probability_matches_call(self):
        c = ConstantBinaryFraction(0.42)
        assert jnp.allclose(c.probability(MASSES), 0.42)
        md = MassDependentBinaryFraction()
        assert jnp.allclose(md.probability(MASSES), md(MASSES))

    def test_radial_probability_matches_compute(self):
        r = RadialBinaryFraction()
        assert jnp.allclose(r.probability(MASSES, RADII), r.compute(RADII))

    def test_probability_ignores_unused_covariate(self):
        # mass model: radii ignored
        c = ConstantBinaryFraction(0.3)
        assert jnp.allclose(c.probability(MASSES, RADII), 0.3)


class TestCombinedBinaryFraction:
    def test_product_clipped(self):
        from progenax.binaries import CombinedBinaryFraction
        mass = MassDependentBinaryFraction()
        radial = RadialBinaryFraction(fb0=0.5, A=0.5, alpha=1.0, r_scale=1.0)
        comb = CombinedBinaryFraction(mass_model=mass, radial_model=radial)
        expected = jnp.clip(mass.probability(MASSES) * radial.probability(MASSES, RADII), 0.0, 1.0)
        assert jnp.allclose(comb.probability(MASSES, RADII), expected)
        assert jnp.all(comb.probability(MASSES, RADII) <= 1.0)

    def test_combined_conforms(self):
        from progenax.protocols import BinaryFractionModel
        from progenax.binaries import CombinedBinaryFraction
        comb = CombinedBinaryFraction(
            mass_model=ConstantBinaryFraction(0.8), radial_model=RadialBinaryFraction()
        )
        assert isinstance(comb, BinaryFractionModel)

    def test_jit_and_grad(self):
        from progenax.binaries import CombinedBinaryFraction
        comb = CombinedBinaryFraction(
            mass_model=ConstantBinaryFraction(0.5), radial_model=RadialBinaryFraction()
        )
        jitted = jax.jit(comb.probability)
        assert jnp.all(jnp.isfinite(jitted(MASSES, RADII)))
        # differentiable wrt the constant mass fraction
        g = jax.grad(
            lambda f: jnp.mean(
                CombinedBinaryFraction(
                    mass_model=ConstantBinaryFraction(f), radial_model=RadialBinaryFraction()
                ).probability(MASSES, RADII)
            )
        )(0.5)
        assert jnp.isfinite(g) and g > 0.0
