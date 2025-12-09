"""Tests for Fractal Density Field (FDF-D) implementation."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from jax import random


class TestFractalDensityLayer:
    """Tests for FractalDensityLayer parameter bundle."""

    def test_default_construction(self):
        """FractalDensityLayer has sensible defaults."""
        from progenax.cluster.fdf_density import FractalDensityLayer

        layer = FractalDensityLayer()

        assert layer.chi == 2.0
        assert layer.sigma_ln_rho == 2.0
        assert layer.lambda_frac == 1.0
        assert layer.grid_size == 64
        assert layer.box_size_factor == 4.0
        assert layer.use_log_normal is True
        assert layer.virial_ratio == 0.5
        assert layer.base_profile == "uniform"
        assert layer.sphere_radius_factor == 2.5

    def test_custom_construction(self):
        """FractalDensityLayer accepts custom parameters."""
        from progenax.cluster.fdf_density import FractalDensityLayer

        layer = FractalDensityLayer(
            chi=1.6,
            sigma_ln_rho=2.5,
            lambda_frac=0.5,
            grid_size=32,
            base_profile="plummer",
            virial_ratio=0.3,
        )

        assert layer.chi == 1.6
        assert layer.sigma_ln_rho == 2.5
        assert layer.lambda_frac == 0.5
        assert layer.grid_size == 32
        assert layer.base_profile == "plummer"
        assert layer.virial_ratio == 0.3

    def test_layer_is_pytree(self):
        """FractalDensityLayer is a valid JAX pytree."""
        from progenax.cluster.fdf_density import FractalDensityLayer

        layer = FractalDensityLayer(chi=2.0, sigma_ln_rho=1.5)

        leaves, treedef = jax.tree_util.tree_flatten(layer)
        layer2 = jax.tree_util.tree_unflatten(treedef, leaves)

        assert layer.chi == layer2.chi
        assert layer.sigma_ln_rho == layer2.sigma_ln_rho


class TestDensityField3D:
    """Tests for DensityField3D dataclass."""

    def test_density_field_construction(self):
        """DensityField3D can be constructed with correct shapes."""
        from progenax.cluster.fdf_density import DensityField3D

        Nx = Ny = Nz = 32
        rho_grid = jnp.ones((Nx, Ny, Nz))
        x_grid = jnp.linspace(-2, 2, Nx)
        y_grid = jnp.linspace(-2, 2, Ny)
        z_grid = jnp.linspace(-2, 2, Nz)

        field = DensityField3D(
            rho_grid=rho_grid,
            x_grid=x_grid,
            y_grid=y_grid,
            z_grid=z_grid,
            box_half_size=2.0,
        )

        assert field.rho_grid.shape == (Nx, Ny, Nz)
        assert field.x_grid.shape == (Nx,)
        assert field.box_half_size == 2.0

    def test_density_field_is_pytree(self):
        """DensityField3D is a valid JAX pytree."""
        from progenax.cluster.fdf_density import DensityField3D

        N = 16
        field = DensityField3D(
            rho_grid=jnp.ones((N, N, N)),
            x_grid=jnp.linspace(-1, 1, N),
            y_grid=jnp.linspace(-1, 1, N),
            z_grid=jnp.linspace(-1, 1, N),
            box_half_size=1.0,
        )

        leaves, treedef = jax.tree_util.tree_flatten(field)
        field2 = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(field.rho_grid, field2.rho_grid)


class TestInitTurbulentDensityField:
    """Tests for init_turbulent_density_field function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    def test_output_shapes(self, key):
        """init_turbulent_density_field produces correct shapes."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        layer = FractalDensityLayer(grid_size=32)
        field = init_turbulent_density_field(key, R_half=1.0, layer=layer)

        assert field.rho_grid.shape == (32, 32, 32)
        assert field.x_grid.shape == (32,)
        assert field.y_grid.shape == (32,)
        assert field.z_grid.shape == (32,)

    def test_density_is_normalized(self, key):
        """Density field integrates approximately to 1."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        layer = FractalDensityLayer(grid_size=64)
        field = init_turbulent_density_field(key, R_half=1.0, layer=layer)

        dx = field.x_grid[1] - field.x_grid[0]
        dV = dx**3
        total_mass = jnp.sum(field.rho_grid) * dV

        # Allow 10% tolerance due to grid discretization and turbulent modulation
        assert jnp.isclose(total_mass, 1.0, rtol=0.10)

    def test_density_is_positive(self, key):
        """Lognormal density is positive everywhere."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        layer = FractalDensityLayer(use_log_normal=True, grid_size=32)
        field = init_turbulent_density_field(key, R_half=1.0, layer=layer)

        assert jnp.all(field.rho_grid >= 0)

    def test_different_keys_produce_different_fields(self):
        """Different random keys produce different density fields."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        key1 = random.PRNGKey(42)
        key2 = random.PRNGKey(123)
        layer = FractalDensityLayer(grid_size=32)

        field1 = init_turbulent_density_field(key1, R_half=1.0, layer=layer)
        field2 = init_turbulent_density_field(key2, R_half=1.0, layer=layer)

        # Density grids should differ
        assert not jnp.allclose(field1.rho_grid, field2.rho_grid)

    def test_uniform_base_profile(self, key):
        """Uniform base profile creates density in spherical region."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        layer = FractalDensityLayer(
            base_profile="uniform",
            sphere_radius_factor=2.0,
            grid_size=32,
        )
        field = init_turbulent_density_field(key, R_half=1.0, layer=layer)

        # Density should be zero outside the sphere
        X, Y, Z = jnp.meshgrid(field.x_grid, field.y_grid, field.z_grid, indexing="ij")
        R2 = X**2 + Y**2 + Z**2
        R_sphere = 2.0 * 1.0  # sphere_radius_factor * R_half
        outside_mask = R2 > (R_sphere * 1.1) ** 2

        # Most density outside should be near zero
        assert jnp.sum(field.rho_grid * outside_mask) < 0.01

    def test_plummer_base_profile(self, key):
        """Plummer base profile creates centrally concentrated density."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        layer = FractalDensityLayer(base_profile="plummer", grid_size=32)
        field = init_turbulent_density_field(key, R_half=1.0, layer=layer)

        # Central density should be higher than edge density
        center_idx = 16
        center_density = field.rho_grid[center_idx, center_idx, center_idx]
        edge_density = field.rho_grid[0, 0, 0]

        assert center_density > edge_density


class TestSamplePositionsFromDensity:
    """Tests for sample_positions_from_density function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    @pytest.fixture
    def field(self, key):
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            init_turbulent_density_field,
        )

        layer = FractalDensityLayer(grid_size=64)
        return init_turbulent_density_field(key, R_half=1.0, layer=layer)

    def test_output_shape(self, key, field):
        """sample_positions_from_density returns correct shape."""
        from progenax.cluster.fdf_density import sample_positions_from_density

        N_stars = 500
        positions = sample_positions_from_density(key, field, N_stars)

        assert positions.shape == (N_stars, 3)

    def test_positions_within_box(self, key, field):
        """Sampled positions are within the box."""
        from progenax.cluster.fdf_density import sample_positions_from_density

        positions = sample_positions_from_density(key, field, N_stars=1000)

        L = field.box_half_size
        assert jnp.all(positions >= -L)
        assert jnp.all(positions <= L)

    def test_positions_follow_density(self, key, field):
        """Sampled positions follow the density distribution (rough check)."""
        from progenax.cluster.fdf_density import sample_positions_from_density

        positions = sample_positions_from_density(key, field, N_stars=2000)

        # For uniform base profile, most stars should be within the sphere
        radii = jnp.linalg.norm(positions, axis=1)
        R_sphere = 2.5 * 1.0  # default sphere_radius_factor * R_half

        fraction_inside = jnp.mean(radii < R_sphere)
        assert fraction_inside > 0.9  # Most should be inside

    def test_jit_compatible(self, key, field):
        """sample_positions_from_density can be JIT compiled."""
        from progenax.cluster.fdf_density import sample_positions_from_density

        @jax.jit
        def sample(key):
            return sample_positions_from_density(key, field, N_stars=100)

        positions = sample(key)
        assert positions.shape == (100, 3)


class TestGenerateFractalICDensity:
    """Tests for generate_fractal_ic_density function."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    @pytest.fixture
    def imf(self):
        from progenax.imf import PowerLawIMF

        return PowerLawIMF.kroupa()

    @pytest.fixture
    def layer(self):
        from progenax.cluster.fdf_density import FractalDensityLayer

        return FractalDensityLayer(chi=2.0, sigma_ln_rho=2.0)

    def test_output_is_cluster_state(self, key, imf, layer):
        """generate_fractal_ic_density returns a ClusterState."""
        from progenax.cluster.fdf_density import generate_fractal_ic_density
        from progenax.cluster import ClusterState

        cluster = generate_fractal_ic_density(
            key,
            N_stars=100,
            M_total=100.0,
            R_half=1.0,
            imf_params=imf,
            layer=layer,
        )

        assert isinstance(cluster, ClusterState)
        assert cluster.masses.shape == (100,)
        assert cluster.positions.shape == (100, 3)
        assert cluster.velocities.shape == (100, 3)

    def test_total_mass_is_correct(self, key, imf, layer):
        """Total mass matches M_total."""
        from progenax.cluster.fdf_density import generate_fractal_ic_density

        M_total = 500.0
        cluster = generate_fractal_ic_density(
            key,
            N_stars=200,
            M_total=M_total,
            R_half=1.0,
            imf_params=imf,
            layer=layer,
        )

        assert jnp.isclose(jnp.sum(cluster.masses), M_total, rtol=1e-5)

    def test_com_is_centered(self, key, imf, layer):
        """Center of mass is near origin."""
        from progenax.cluster.fdf_density import generate_fractal_ic_density

        cluster = generate_fractal_ic_density(
            key,
            N_stars=300,
            M_total=300.0,
            R_half=1.0,
            imf_params=imf,
            layer=layer,
        )

        M_total = jnp.sum(cluster.masses)
        x_com = jnp.sum(cluster.masses[:, None] * cluster.positions, axis=0) / M_total

        assert jnp.allclose(x_com, 0.0, atol=1e-10)

    def test_virial_ratio_achieved(self, key, imf):
        """Generated cluster achieves target virial ratio."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            generate_fractal_ic_density,
        )
        from progenax.dynamics.virial import compute_virial_ratio
        from jaxstro.units import STELLAR

        for Q_target in [0.3, 0.5]:
            layer = FractalDensityLayer(virial_ratio=Q_target)
            key, subkey = random.split(key)

            cluster = generate_fractal_ic_density(
                subkey,
                N_stars=300,
                M_total=300.0,
                R_half=1.0,
                imf_params=imf,
                layer=layer,
            )

            Q_actual = compute_virial_ratio(
                cluster.positions, cluster.velocities, cluster.masses, G=STELLAR.G
            )

            assert jnp.isclose(Q_actual, Q_target, rtol=0.05), (
                f"Q_target={Q_target}, Q_actual={Q_actual}"
            )


class TestDensityLayerFromD:
    """Tests for density_layer_from_D calibration helper."""

    def test_creates_valid_layer(self):
        """density_layer_from_D creates valid FractalDensityLayer."""
        from progenax.cluster.fdf_density import (
            FractalDensityLayer,
            density_layer_from_D,
        )

        layer = density_layer_from_D(D=2.0, virial_ratio=0.3)

        assert isinstance(layer, FractalDensityLayer)
        assert layer.virial_ratio == 0.3
        assert layer.chi >= 1.6
        assert layer.chi <= 3.0

    def test_clamps_D_to_valid_range(self):
        """density_layer_from_D clamps D to valid range."""
        from progenax.cluster.fdf_density import density_layer_from_D

        # D below range
        layer_low = density_layer_from_D(D=1.0)
        assert layer_low.chi >= 1.6

        # D above range
        layer_high = density_layer_from_D(D=4.0)
        assert layer_high.chi <= 3.0

    def test_default_base_profile_is_uniform(self):
        """Default base profile is 'uniform'."""
        from progenax.cluster.fdf_density import density_layer_from_D

        layer = density_layer_from_D(D=2.0)
        assert layer.base_profile == "uniform"

    def test_accepts_base_profile_parameter(self):
        """Can specify base profile."""
        from progenax.cluster.fdf_density import density_layer_from_D

        layer = density_layer_from_D(D=2.0, base_profile="plummer")
        assert layer.base_profile == "plummer"


class TestQParameterTrend:
    """Tests for Q(D) trend - the main validation requirement."""

    @pytest.fixture
    def key(self):
        return random.PRNGKey(42)

    @pytest.fixture
    def imf(self):
        from progenax.imf import PowerLawIMF

        return PowerLawIMF.kroupa()

    def test_q_increases_with_d(self, key, imf):
        """Q parameter increases as D increases (clumpy -> smooth)."""
        from progenax.cluster.fdf_density import (
            generate_fractal_ic_density,
            density_layer_from_D,
        )
        from progenax.diagnostics import compute_q_parameter

        N_stars = 1000
        n_realizations = 5

        D_values = [1.6, 3.0]
        Q_means = []

        for D in D_values:
            Q_list = []
            for i in range(n_realizations):
                key_i = random.PRNGKey(i * 100 + int(D * 10))
                layer = density_layer_from_D(D=D, sigma_ln_rho=2.0)
                cluster = generate_fractal_ic_density(
                    key_i,
                    N_stars=N_stars,
                    M_total=float(N_stars),
                    R_half=1.0,
                    imf_params=imf,
                    layer=layer,
                )
                Q = compute_q_parameter(np.array(cluster.positions))
                Q_list.append(Q)
            Q_means.append(np.mean(Q_list))

        # Q(D=3.0) should be greater than Q(D=1.6)
        assert Q_means[1] > Q_means[0], (
            f"Q(D=1.6)={Q_means[0]:.3f} should be < Q(D=3.0)={Q_means[1]:.3f}"
        )


class TestModuleExports:
    """Tests for public API exports."""

    def test_density_fdf_exports_from_cluster(self):
        """Density FDF classes are exported from progenax.cluster."""
        from progenax.cluster import (
            FractalDensityLayer,
            DensityField3D,
            generate_fractal_ic_density,
            init_turbulent_density_field,
            sample_positions_from_density,
            density_layer_from_D,
        )

        assert FractalDensityLayer is not None
        assert DensityField3D is not None
        assert generate_fractal_ic_density is not None
        assert init_turbulent_density_field is not None
        assert sample_positions_from_density is not None
        assert density_layer_from_D is not None
