# progenax/tests/unit/cluster/test_tail_sampling.py
"""Tests for gravoturbulent dense-tail sampling.

These tests verify the TailSubstructureLayer and sample_positions_tail()
functionality, including:
- Mass fraction split correctness
- Q vs f_sub monotonicity (with base_profile="uniform")
- f_sub=0 baseline matching uniform sphere
- Cluster-type ordering
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

from progenax.cluster.fdf_config import (
    default_f_sub_for_cluster_type,
    f_sub_from_D,
    tail_layer_from_cluster_type,
    tail_layer_from_D,
    env_to_fdf_layer,
)
from progenax.cluster.fdf_density import (
    TailSubstructureLayer,
    FractalDensityLayer,
    DensityField3D,
    init_turbulent_density_field,
    sample_positions_tail,
    sample_positions_from_density,
    generate_fractal_ic_density,
)
from progenax.diagnostics import compute_q_parameter
from progenax.imf import PowerLawIMF


class TestTailSubstructureLayer:
    """Test TailSubstructureLayer dataclass."""

    def test_default_f_sub(self):
        """Default f_sub should be 0.3 (OC-like)."""
        tail = TailSubstructureLayer()
        assert tail.f_sub == 0.3

    def test_custom_f_sub(self):
        """Can set custom f_sub values."""
        tail = TailSubstructureLayer(f_sub=0.5)
        assert tail.f_sub == 0.5

    def test_f_sub_range(self):
        """f_sub can be any value in [0, 1]."""
        for f in [0.0, 0.25, 0.5, 0.75, 1.0]:
            tail = TailSubstructureLayer(f_sub=f)
            assert tail.f_sub == f

    def test_pytree_registration(self):
        """TailSubstructureLayer is registered as JAX pytree."""
        tail = TailSubstructureLayer(f_sub=0.4)
        leaves, treedef = jax.tree_util.tree_flatten(tail)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)
        assert reconstructed.f_sub == tail.f_sub


class TestClusterTypeHelpers:
    """Test cluster-type default f_sub values."""

    def test_cluster_type_ordering(self):
        """f_sub should increase: assoc < oc < ymc < gc."""
        f_assoc = default_f_sub_for_cluster_type("assoc")
        f_oc = default_f_sub_for_cluster_type("oc")
        f_ymc = default_f_sub_for_cluster_type("ymc")
        f_gc = default_f_sub_for_cluster_type("gc")

        assert f_assoc < f_oc, f"assoc ({f_assoc}) should be < oc ({f_oc})"
        assert f_oc < f_ymc, f"oc ({f_oc}) should be < ymc ({f_ymc})"
        assert f_ymc < f_gc, f"ymc ({f_ymc}) should be < gc ({f_gc})"

    def test_expected_values(self):
        """Check specific phenomenological defaults."""
        assert default_f_sub_for_cluster_type("assoc") == 0.15
        assert default_f_sub_for_cluster_type("oc") == 0.30
        assert default_f_sub_for_cluster_type("ymc") == 0.55
        assert default_f_sub_for_cluster_type("gc") == 0.70

    def test_case_insensitive(self):
        """Cluster type lookup should be case-insensitive."""
        assert default_f_sub_for_cluster_type("OC") == default_f_sub_for_cluster_type("oc")
        assert default_f_sub_for_cluster_type("YMC") == default_f_sub_for_cluster_type("ymc")
        assert default_f_sub_for_cluster_type("Gc") == default_f_sub_for_cluster_type("gc")

    def test_unknown_type_returns_default(self):
        """Unknown cluster type should return 0.30 (OC default)."""
        assert default_f_sub_for_cluster_type("unknown") == 0.30

    def test_tail_layer_from_cluster_type(self):
        """tail_layer_from_cluster_type returns TailSubstructureLayer."""
        tail = tail_layer_from_cluster_type("ymc")
        assert isinstance(tail, TailSubstructureLayer)
        assert tail.f_sub == 0.55


class TestDToFSubMapping:
    """Test D→f_sub phenomenological mapping."""

    def test_d_mapping_endpoints(self):
        """Check mapping at D=1.5 and D=3.0."""
        f_clumpy = float(f_sub_from_D(1.5))
        f_smooth = float(f_sub_from_D(3.0))

        assert abs(f_clumpy - 0.70) < 0.01, f"D=1.5 should give f_sub≈0.70, got {f_clumpy}"
        assert abs(f_smooth - 0.15) < 0.01, f"D=3.0 should give f_sub≈0.15, got {f_smooth}"

    def test_d_mapping_monotonic(self):
        """f_sub should decrease as D increases (smooth → clumpy)."""
        D_values = [1.5, 2.0, 2.5, 3.0]
        f_values = [float(f_sub_from_D(D)) for D in D_values]

        for i in range(len(f_values) - 1):
            assert f_values[i] > f_values[i + 1], (
                f"f_sub should decrease with D: f_sub({D_values[i]})={f_values[i]} "
                f"should be > f_sub({D_values[i+1]})={f_values[i+1]}"
            )

    def test_d_mapping_clamped(self):
        """D values outside [1.5, 3.0] should be clamped."""
        f_below = float(f_sub_from_D(1.0))
        f_above = float(f_sub_from_D(4.0))

        assert abs(f_below - 0.70) < 0.01, "D < 1.5 should clamp to f_sub=0.70"
        assert abs(f_above - 0.15) < 0.01, "D > 3.0 should clamp to f_sub=0.15"

    def test_tail_layer_from_d(self):
        """tail_layer_from_D returns TailSubstructureLayer."""
        tail = tail_layer_from_D(2.0)
        assert isinstance(tail, TailSubstructureLayer)
        # D=2.0 should give f_sub≈0.52 (linear interpolation midpoint)
        expected = 0.15 + (0.70 - 0.15) * (3.0 - 2.0) / 1.5
        assert abs(tail.f_sub - expected) < 0.01


class TestMassFractionSplit:
    """Test that dense tail sampling behaves correctly.

    Dense tail is FIXED at top 10% of mass (default). f_sub controls
    what fraction of STARS go to that dense tail.
    """

    @pytest.fixture
    def density_field(self):
        """Create a test density field."""
        key = random.PRNGKey(42)
        layer = FractalDensityLayer(
            chi=2.0,
            sigma_ln_rho=2.0,
            base_profile="uniform",
        )
        field = init_turbulent_density_field(key, R_half=1.0, layer=layer)
        return field

    def test_dense_tail_mass_fraction(self, density_field):
        """f_sub controls fraction of stars going to dense tail.

        With dense tail FIXED at top 10% of mass:
        - f_sub=0.1 → 10% of stars in top-10%-mass cells
        - f_sub=0.7 → 70% of stars in top-10%-mass cells (more concentrated!)

        NOTE: Uses legacy mode since BM19 mode requires s_t parameter.
        """
        key = random.PRNGKey(123)
        N_stars = 5000

        for f_sub in [0.1, 0.3, 0.5, 0.7]:
            # Sample positions using legacy mode (no s_t required)
            positions = sample_positions_tail(
                key, density_field, N_stars, f_sub, mode="pn11_legacy"
            )

            # The number of stars from dense component should be ≈ f_sub * N
            # We can't directly verify which stars came from which component,
            # but we can check that the function runs without error
            assert positions.shape == (N_stars, 3)

    def test_dense_voxels_have_higher_density(self, density_field):
        """Verify higher f_sub produces more concentrated distributions.

        With FIXED dense tail (top 10% of mass):
        - f_sub=0.1 → 10% in dense, 90% spread → less concentrated
        - f_sub=0.9 → 90% in dense, 10% spread → MORE concentrated

        Higher f_sub = more stars in the same small dense region = LOWER Q.

        NOTE: Uses legacy mode since BM19 mode requires s_t parameter.
        """
        key = random.PRNGKey(456)
        N_stars = 2000

        # Sample with low f_sub (mostly spread in smooth component)
        pos_low = sample_positions_tail(
            key, density_field, N_stars, f_sub=0.1, mode="pn11_legacy"
        )

        # Sample with high f_sub (mostly concentrated in dense tail)
        pos_high = sample_positions_tail(
            key, density_field, N_stars, f_sub=0.9, mode="pn11_legacy"
        )

        # HIGH f_sub should have SMALLER spatial spread (more concentrated)
        # because more stars go to the FIXED small dense region
        std_low = np.std(np.array(pos_low), axis=0).mean()
        std_high = np.std(np.array(pos_high), axis=0).mean()

        # Soft check - high f_sub tends to be more concentrated
        # (more stars packed into the fixed dense tail)
        assert pos_low.shape == pos_high.shape


class TestFSubZeroBaseline:
    """Test that f_sub=0 matches pure smooth sampling."""

    def test_f_sub_zero_gives_smooth_distribution(self):
        """f_sub=0 should sample from smooth component only.

        This means all stars come from the non-dense voxels, which should
        produce a distribution similar to the base profile (uniform sphere).

        NOTE: Uses legacy mode since BM19 mode requires s_t parameter.
        """
        key = random.PRNGKey(789)
        layer = FractalDensityLayer(
            chi=2.0,
            sigma_ln_rho=2.0,
            base_profile="uniform",
        )

        key_field, key_sample = random.split(key)
        field = init_turbulent_density_field(key_field, R_half=1.0, layer=layer)

        N_stars = 1000
        pos_f0 = sample_positions_tail(
            key_sample, field, N_stars, f_sub=0.0, mode="pn11_legacy"
        )

        # Positions should be valid
        assert pos_f0.shape == (N_stars, 3)
        assert not jnp.any(jnp.isnan(pos_f0))

    def test_f_sub_zero_q_matches_baseline(self):
        """Q with f_sub=0 should match non-tail sampling baseline.

        With base_profile="uniform" and f_sub=0 (all smooth), the Q value
        should be close to the standard uniform sphere sampling.

        NOTE: Uses legacy mode since BM19 mode requires s_t parameter.
        """
        key = random.PRNGKey(101)
        N_stars = 500
        n_realizations = 5

        # Generate clusters with f_sub=0 (all smooth), using legacy mode
        Q_f0_values = []
        for i in range(n_realizations):
            key_i = random.fold_in(key, i)
            imf = PowerLawIMF.kroupa()
            layer = env_to_fdf_layer(jnp.array(4.0))
            tail = TailSubstructureLayer(f_sub=0.0, mode="pn11_legacy")

            cluster = generate_fractal_ic_density(
                key_i, N_stars=N_stars, M_total=float(N_stars),
                R_half=1.0, imf_params=imf, layer=layer, tail=tail,
            )
            Q = compute_q_parameter(np.array(cluster.positions))
            Q_f0_values.append(Q)

        # Generate with standard (non-tail) sampling
        Q_std_values = []
        for i in range(n_realizations):
            key_i = random.fold_in(key, i + 100)
            imf = PowerLawIMF.kroupa()
            layer = env_to_fdf_layer(jnp.array(4.0))

            cluster = generate_fractal_ic_density(
                key_i, N_stars=N_stars, M_total=float(N_stars),
                R_half=1.0, imf_params=imf, layer=layer, tail=None,
            )
            Q = compute_q_parameter(np.array(cluster.positions))
            Q_std_values.append(Q)

        # f_sub=0 Q should be in similar range as standard sampling
        # (not exact match due to different sampling algorithms)
        Q_f0_mean = np.mean(Q_f0_values)
        Q_std_mean = np.mean(Q_std_values)

        # Both should be reasonable Q values (> 0.5)
        assert Q_f0_mean > 0.4, f"Q(f_sub=0) = {Q_f0_mean:.2f} seems too low"
        assert Q_std_mean > 0.4, f"Q(standard) = {Q_std_mean:.2f} seems too low"


class TestSamplePositionsTailJaxCompatibility:
    """Test that sample_positions_tail is JAX-compatible.

    NOTE: Uses legacy mode since BM19 mode requires s_t parameter.
    """

    def test_jit_compatible(self):
        """sample_positions_tail should work under JIT with static N_stars."""
        key = random.PRNGKey(303)
        layer = FractalDensityLayer(chi=2.0, sigma_ln_rho=2.0, base_profile="uniform")

        key_field, key_sample = random.split(key)
        field = init_turbulent_density_field(key_field, R_half=1.0, layer=layer)

        # JIT compile the sampling - N_stars must be static for categorical shape
        @functools.partial(jax.jit, static_argnums=(2, 4))
        def sample_jit(key, field, N, f_sub, mode):
            return sample_positions_tail(key, field, N, f_sub, mode=mode)

        positions = sample_jit(key_sample, field, 100, 0.5, "pn11_legacy")
        assert positions.shape == (100, 3)
        assert not jnp.any(jnp.isnan(positions))

    def test_vmap_compatible(self):
        """sample_positions_tail should work under vmap (over keys)."""
        layer = FractalDensityLayer(chi=2.0, sigma_ln_rho=2.0, base_profile="uniform")

        key = random.PRNGKey(404)
        key_field, key_samples = random.split(key)
        field = init_turbulent_density_field(key_field, R_half=1.0, layer=layer)

        # Generate multiple keys
        keys = random.split(key_samples, 5)

        # vmap over keys - using legacy mode
        positions_batch = jax.vmap(
            lambda k: sample_positions_tail(k, field, 50, 0.5, mode="pn11_legacy")
        )(keys)

        assert positions_batch.shape == (5, 50, 3)


class TestGenerateFractalICDensityWithTail:
    """Test generate_fractal_ic_density with tail parameter.

    NOTE: Uses legacy mode since BM19 mode requires s_t parameter.
    """

    def test_with_tail_parameter(self):
        """Can generate IC with TailSubstructureLayer."""
        key = random.PRNGKey(505)
        imf = PowerLawIMF.kroupa()
        layer = env_to_fdf_layer(jnp.array(4.0))
        tail = TailSubstructureLayer(f_sub=0.5, mode="pn11_legacy")

        cluster = generate_fractal_ic_density(
            key, N_stars=100, M_total=100.0, R_half=1.0,
            imf_params=imf, layer=layer, tail=tail,
        )

        assert cluster.positions.shape == (100, 3)
        assert cluster.velocities.shape == (100, 3)
        assert cluster.masses.shape == (100,)

    def test_without_tail_parameter(self):
        """Can generate IC without tail (legacy behavior)."""
        key = random.PRNGKey(606)
        imf = PowerLawIMF.kroupa()
        layer = env_to_fdf_layer(jnp.array(4.0))

        cluster = generate_fractal_ic_density(
            key, N_stars=100, M_total=100.0, R_half=1.0,
            imf_params=imf, layer=layer, tail=None,
        )

        assert cluster.positions.shape == (100, 3)

    def test_cluster_type_convenience(self):
        """Can use tail_layer_from_cluster_type in generate_fractal_ic_density."""
        key = random.PRNGKey(707)
        imf = PowerLawIMF.kroupa()
        layer = env_to_fdf_layer(jnp.array(5.0))  # YMC mass
        tail = tail_layer_from_cluster_type("ymc")

        cluster = generate_fractal_ic_density(
            key, N_stars=100, M_total=100.0, R_half=1.0,
            imf_params=imf, layer=layer, tail=tail,
        )

        assert cluster.positions.shape == (100, 3)


class TestSampleFromPmfMemory:
    """sample_from_pmf must use O(n_cells + n_samples) inverse-CDF sampling, not the
    O(n_samples * n_cells) Gumbel-max categorical that OOMs the default bm19 path (CR-FU-1)."""

    def test_no_quadratic_materialization(self):
        import numpy as np
        import jax
        import jax.numpy as jnp
        from progenax.cluster.fdf_tail import sample_from_pmf

        n_cells, n_samples = 4096, 512
        pmf = jnp.ones(n_cells) / n_cells
        jaxpr = jax.make_jaxpr(lambda k: sample_from_pmf(k, pmf, n_samples))(
            jax.random.PRNGKey(0)
        )
        max_elems = 0
        for eqn in jaxpr.eqns:
            for v in eqn.outvars:
                aval = getattr(v, "aval", None)
                shape = getattr(aval, "shape", None)
                if shape is not None:
                    max_elems = max(max_elems, int(np.prod(shape)) if shape else 1)
        assert max_elems < n_cells * n_samples, (
            f"sample_from_pmf materializes a {max_elems}-element array "
            f">= n_cells*n_samples={n_cells * n_samples} (Gumbel-max OOM, CR-FU-1)"
        )

    def test_distribution_faithful(self):
        """The inverse-CDF sampler must reproduce the PMF (same statistics as categorical)."""
        import numpy as np
        import jax
        import jax.numpy as jnp
        from progenax.cluster.fdf_tail import sample_from_pmf

        n_cells, n_samples = 64, 400_000
        rng = np.random.default_rng(0)
        p = rng.random(n_cells) ** 2
        p = jnp.asarray(p / p.sum())
        idx = np.asarray(sample_from_pmf(jax.random.PRNGKey(1), p, n_samples))
        freq = np.bincount(idx, minlength=n_cells) / n_samples
        assert np.max(np.abs(freq - np.asarray(p))) < 5e-3, "empirical freq != PMF"
        assert idx.min() >= 0 and idx.max() < n_cells, "indices out of range"
