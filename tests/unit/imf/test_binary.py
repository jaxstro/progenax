"""Tests for binary star mass functions.

Physics tests only - distribution properties and literature comparisons.
"""

import jax
import jax.numpy as jnp
import pytest

from progenax.imf.binary import (
    BinaryIMF,
    ConstantBinaryFraction,
    FlatMassRatio,
    MassDependentBinaryFraction,
    MoeDiStefano2017,
    PowerLawMassRatio,
    TwinPeakedMassRatio,
)
from progenax.imf.power_law import PowerLawIMF


class TestMassRatioDistributions:
    """Test mass-ratio distributions (Flat, PowerLaw, TwinPeaked)."""

    def test_flat_pdf_normalization(self):
        """Flat mass ratio: PDF integrates to 1."""
        q_dist = FlatMassRatio(q_min=0.1)
        q_grid = jnp.linspace(0.1, 1.0, 1000)
        integral = jnp.trapezoid(q_dist.pdf(q_grid), q_grid)
        assert jnp.abs(integral - 1.0) < 1e-4

    def test_flat_ppf_inverse_cdf(self):
        """Flat: PPF is inverse of CDF."""
        q_dist = FlatMassRatio(q_min=0.1)
        u = jnp.array([0.0, 0.25, 0.5, 0.75, 1.0])
        q = q_dist.ppf(u)
        u_reconstructed = q_dist.cdf(q)
        assert jnp.allclose(u, u_reconstructed, atol=1e-6)

    def test_powerlaw_pdf_normalization(self):
        """PowerLaw: PDF integrates to 1 for various gamma."""
        for gamma in [-0.5, 0.0, 0.3]:
            q_dist = PowerLawMassRatio(gamma=gamma, q_min=0.1)
            q_grid = jnp.linspace(0.1, 1.0, 1000)
            integral = jnp.trapezoid(q_dist.pdf(q_grid), q_grid)
            assert jnp.abs(integral - 1.0) < 1e-3, f"Failed for gamma={gamma}"

    def test_powerlaw_gamma_zero_is_flat(self):
        """PowerLaw with gamma=0 equals FlatMassRatio."""
        q_dist_pl = PowerLawMassRatio(gamma=0.0, q_min=0.1)
        q_dist_flat = FlatMassRatio(q_min=0.1)

        q = jnp.array([0.2, 0.5, 0.8])
        assert jnp.allclose(q_dist_pl.pdf(q), q_dist_flat.pdf(q), rtol=1e-6)

    def test_twinpeaked_pdf_normalization(self):
        """TwinPeaked: PDF integrates to 1."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.1, q_min=0.1)
        q_grid = jnp.linspace(0.1, 1.0, 2000)
        integral = jnp.trapezoid(q_dist.pdf(q_grid), q_grid)
        assert jnp.abs(integral - 1.0) < 1e-3

    def test_twinpeaked_peak_at_q_one(self):
        """TwinPeaked: PDF higher near q=1 than mid-range."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.2, sigma_twin=0.03)
        pdf_near_one = q_dist.pdf(jnp.array(0.98))
        pdf_mid = q_dist.pdf(jnp.array(0.5))
        assert pdf_near_one > pdf_mid


class TestPowerLawMassRatioArrayBranches:
    """Cover PowerLawMassRatio cdf/ppf array branches and sample statistics.

    The existing tests only exercise the PDF and gamma=0 equivalence. These
    target the vmapped cdf/ppf array paths (lines ~177, ~207) and the
    inverse-CDF round-trip, plus sampling statistics.
    """

    @pytest.mark.parametrize("gamma", [-0.5, 0.0, 0.3])
    def test_cdf_array_monotonic_and_boundaries(self, gamma):
        """cdf(array) is monotonic increasing with cdf(q_min)=0, cdf(1)=1."""
        q_min = 0.1
        q_dist = PowerLawMassRatio(gamma=gamma, q_min=q_min)
        q = jnp.linspace(q_min, 1.0, 50)
        F = q_dist.cdf(q)

        # Shape preserved by the array branch
        assert F.shape == q.shape
        # Boundary values
        assert jnp.abs(F[0] - 0.0) < 1e-6, f"cdf(q_min) != 0 for gamma={gamma}"
        assert jnp.abs(F[-1] - 1.0) < 1e-6, f"cdf(1) != 1 for gamma={gamma}"
        # Strictly increasing (discriminating: a constant/garbage CDF fails)
        assert jnp.all(jnp.diff(F) > 0), f"cdf not monotonic for gamma={gamma}"

    @pytest.mark.parametrize("gamma", [-0.5, 0.0, 0.3])
    def test_ppf_array_inverse_of_cdf(self, gamma):
        """ppf(array) inverts cdf: cdf(ppf(u)) ~= u, with q in [q_min, 1]."""
        q_min = 0.1
        q_dist = PowerLawMassRatio(gamma=gamma, q_min=q_min)
        u = jnp.array([0.01, 0.2, 0.4, 0.6, 0.8, 0.99])
        q = q_dist.ppf(u)

        # Shape preserved + physical range
        assert q.shape == u.shape
        assert jnp.all(q >= q_min - 1e-9), f"q < q_min for gamma={gamma}"
        assert jnp.all(q <= 1.0 + 1e-9), f"q > 1 for gamma={gamma}"
        # Round-trip cdf(ppf(u)) == u (analytic inverse, tight tolerance)
        u_round = q_dist.cdf(q)
        assert jnp.allclose(u_round, u, atol=1e-6), (
            f"cdf(ppf(u)) != u for gamma={gamma}"
        )

    @pytest.mark.parametrize("gamma", [-0.5, 0.0, 0.5])
    def test_sample_matches_distribution(self, gamma):
        """sample() statistics match the analytic PDF mean and KS-fit the CDF."""
        q_min = 0.1
        q_dist = PowerLawMassRatio(gamma=gamma, q_min=q_min)
        key = jax.random.PRNGKey(7)
        samples = q_dist.sample(key, 5000)

        # All samples physical
        assert jnp.all(samples >= q_min - 1e-9)
        assert jnp.all(samples <= 1.0 + 1e-9)

        # Analytic mean E[q] = integral q*pdf(q) dq on a fine grid
        q_grid = jnp.linspace(q_min, 1.0, 4000)
        pdf = q_dist.pdf(q_grid)
        mean_analytic = jnp.trapezoid(q_grid * pdf, q_grid)
        mean_sample = jnp.mean(samples)
        # 5000 draws: sample mean within ~0.02 of analytic mean
        assert jnp.abs(mean_sample - mean_analytic) < 0.02, (
            f"mean {mean_sample:.3f} vs analytic {mean_analytic:.3f} (gamma={gamma})"
        )

        # KS statistic: max |empirical CDF - analytic CDF| should be small
        sorted_s = jnp.sort(samples)
        emp_cdf = (jnp.arange(1, sorted_s.shape[0] + 1)) / sorted_s.shape[0]
        ana_cdf = q_dist.cdf(sorted_s)
        ks = jnp.max(jnp.abs(emp_cdf - ana_cdf))
        assert ks < 0.05, f"KS={ks:.3f} too large for gamma={gamma}"


class TestTwinPeakedMassRatioArrayBranches:
    """Cover TwinPeakedMassRatio cdf array branch and ppf Newton convergence.

    Targets the vmapped cdf (line ~320) and the fori_loop Newton ppf
    (lines ~325-340) for both scalar and array inputs.
    """

    def test_cdf_array_monotonic_and_boundaries(self):
        """cdf(array): monotonic, cdf(q_min)~=0, cdf(1)~=1."""
        q_min = 0.1
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.2, sigma_twin=0.03, q_min=q_min)
        q = jnp.linspace(q_min, 1.0, 80)
        F = q_dist.cdf(q)

        assert F.shape == q.shape
        assert jnp.abs(F[0] - 0.0) < 1e-5, "cdf(q_min) should be ~0"
        assert jnp.abs(F[-1] - 1.0) < 1e-5, "cdf(1) should be ~1"
        # Non-decreasing (twin peak makes CDF steep near q=1, but never decreasing)
        assert jnp.all(jnp.diff(F) >= -1e-9), "twin-peaked cdf must be non-decreasing"

    def test_ppf_scalar_newton_converges(self):
        """ppf scalar: Newton iteration converges so |cdf(ppf(u)) - u| < 1e-5."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.2, sigma_twin=0.03, q_min=0.1)
        for u_val in [0.1, 0.3, 0.5, 0.7, 0.85]:
            u = jnp.array(u_val)
            q = q_dist.ppf(u)
            assert jnp.ndim(q) == 0, "scalar ppf should return scalar"
            assert 0.1 - 1e-6 <= float(q) <= 1.0 + 1e-6
            residual = jnp.abs(q_dist.cdf(q) - u)
            assert residual < 1e-5, f"Newton ppf residual {residual:.2e} at u={u_val}"

    def test_ppf_array_newton_converges(self):
        """ppf array: vmapped Newton converges for all u, shape preserved."""
        q_dist = TwinPeakedMassRatio(gamma=0.0, f_twin=0.2, sigma_twin=0.03, q_min=0.1)
        u = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
        q = q_dist.ppf(u)

        assert q.shape == u.shape
        assert jnp.all(q >= 0.1 - 1e-6)
        assert jnp.all(q <= 1.0 + 1e-6)
        residuals = jnp.abs(q_dist.cdf(q) - u)
        assert jnp.all(residuals < 1e-5), f"max residual {jnp.max(residuals):.2e}"


class TestMoeDiStefano2017:
    """Test mass-dependent q-distribution from Moe+17."""

    def test_gamma_varies_with_mass(self):
        """Power-law exponent varies with primary mass (Moe+17)."""
        q_dist = MoeDiStefano2017()
        gamma_low = q_dist._gamma_of_mass(jnp.array(0.5))
        gamma_solar = q_dist._gamma_of_mass(jnp.array(1.0))
        gamma_massive = q_dist._gamma_of_mass(jnp.array(10.0))

        # γ decreases with mass
        assert gamma_low > gamma_solar > gamma_massive

    def test_ftwin_varies_with_mass(self):
        """Twin fraction varies with primary mass (solar-type peak)."""
        q_dist = MoeDiStefano2017()
        f_low = q_dist._ftwin_of_mass(jnp.array(0.5))
        f_solar = q_dist._ftwin_of_mass(jnp.array(1.0))
        f_massive = q_dist._ftwin_of_mass(jnp.array(10.0))

        # Solar-type has highest twin excess
        assert f_solar > f_low and f_solar > f_massive


class TestBinaryFractionModels:
    """Test binary fraction models."""

    def test_constant_returns_constant(self):
        """ConstantBinaryFraction returns same value for all masses."""
        model = ConstantBinaryFraction(f_bin=0.6)
        masses = jnp.array([0.1, 0.5, 1.0, 5.0, 10.0])
        assert jnp.allclose(model(masses), 0.6)

    def test_mass_dependent_increases_with_mass(self):
        """MassDependentBinaryFraction increases with mass."""
        model = MassDependentBinaryFraction()
        f_low = model(jnp.array(0.3))
        f_solar = model(jnp.array(1.0))
        f_massive = model(jnp.array(15.0))
        assert f_low < f_solar < f_massive

    def test_mass_dependent_matches_moe2017(self):
        """Values match Moe+17 Table 13."""
        model = MassDependentBinaryFraction()
        # Key mass bins from literature
        assert jnp.abs(model(jnp.array(0.3)) - 0.26) < 1e-6   # M-dwarf
        assert jnp.abs(model(jnp.array(0.8)) - 0.44) < 1e-6   # K-dwarf
        assert jnp.abs(model(jnp.array(1.5)) - 0.50) < 1e-6   # A-star
        assert jnp.abs(model(jnp.array(15.0)) - 0.90) < 1e-6  # O-star


class TestBinaryIMF:
    """Test binary IMF composition."""

    def test_binary_fraction_matches_target(self):
        """Binary fraction matches target within statistics."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        key = jax.random.PRNGKey(123)
        _, _, is_binary = binary_imf.sample_systems(key, 10000)

        frac = jnp.mean(is_binary.astype(float))
        assert jnp.abs(frac - 0.5) < 0.02

    def test_singles_have_zero_m2(self):
        """Single stars have m2=0."""
        primary_imf = PowerLawIMF.kroupa()
        binary_imf = BinaryIMF.simple(primary_imf, binary_fraction=0.5)

        key = jax.random.PRNGKey(42)
        _, m2, is_binary = binary_imf.sample_systems(key, 1000)

        singles_m2 = m2[~is_binary]
        assert jnp.allclose(singles_m2, 0.0)

    def test_binaries_satisfy_q_constraint(self):
        """Binary secondaries satisfy q = m2/m1 in [q_min, 1]."""
        primary_imf = PowerLawIMF.kroupa()
        q_min = 0.15
        binary_imf = BinaryIMF(
            primary_imf=primary_imf,
            q_distribution=FlatMassRatio(q_min=q_min),
            binary_fraction=1.0,
        )

        key = jax.random.PRNGKey(42)
        m1, m2, is_binary = binary_imf.sample_systems(key, 1000)

        q = m2 / m1
        assert jnp.all(q >= q_min - 1e-6)
        assert jnp.all(q <= 1.0 + 1e-6)


class TestBinaryIMFHelpers:
    """Cover BinaryIMF helper/aggregate methods and callable branches.

    Targets uncovered lines in binary/imf.py: _get_binary_fraction_model
    default (None path), _get_binary_fraction callable branch,
    sample_mass_ratios custom callable, sample_all_masses, mean_system_mass,
    binary_fraction_overall, and the moe2017/massive_stars factories.
    """

    def test_binary_fraction_model_default_is_mass_dependent(self):
        """binary_fraction=None resolves to MassDependentBinaryFraction."""
        imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa())  # both q & f_bin None
        model = imf._get_binary_fraction_model()
        assert isinstance(model, MassDependentBinaryFraction)
        # And the default q distribution is the Moe+17 model
        assert isinstance(imf._get_q_distribution(), MoeDiStefano2017)

    def test_get_binary_fraction_callable_branch(self):
        """A custom callable f_bin(m) is invoked element-wise."""
        def my_f_bin(m):
            return jnp.where(m < 1.0, 0.4, 0.8)

        imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa(), binary_fraction=my_f_bin)
        masses = jnp.array([0.2, 0.5, 1.5, 5.0])
        f = imf._get_binary_fraction(masses)
        expected = jnp.array([0.4, 0.4, 0.8, 0.8])
        assert jnp.allclose(f, expected), f"callable f_bin not applied: {f}"

    def test_get_binary_fraction_float_branch(self):
        """A float binary_fraction broadcasts to a full array."""
        imf = BinaryIMF(primary_imf=PowerLawIMF.kroupa(), binary_fraction=0.55)
        masses = jnp.array([0.3, 1.0, 10.0])
        f = imf._get_binary_fraction(masses)
        assert f.shape == masses.shape
        assert jnp.allclose(f, 0.55)

    def test_sample_mass_ratios_custom_callable(self):
        """A custom q_sampler(key, m1) is called directly (callable branch)."""
        def my_q_sampler(key, m1):
            # Deterministic-ish: q depends on m1 only, in (0.3, 1.0)
            return jnp.full_like(m1, 0.42)

        imf = BinaryIMF(
            primary_imf=PowerLawIMF.kroupa(),
            q_distribution=my_q_sampler,
        )
        key = jax.random.PRNGKey(0)
        m1 = jnp.array([0.5, 1.0, 5.0, 20.0])
        q = imf.sample_mass_ratios(key, m1)
        assert q.shape == m1.shape
        assert jnp.allclose(q, 0.42), "custom q_sampler branch not taken"

    def test_sample_mass_ratios_flat_distribution_branch(self):
        """Non-callable, non-Moe distribution uses q_dist.sample(key, n)."""
        imf = BinaryIMF(
            primary_imf=PowerLawIMF.kroupa(),
            q_distribution=FlatMassRatio(q_min=0.2),
        )
        key = jax.random.PRNGKey(1)
        m1 = jnp.ones(500)
        q = imf.sample_mass_ratios(key, m1)
        assert q.shape == (500,)
        assert jnp.all(q >= 0.2 - 1e-6) and jnp.all(q <= 1.0 + 1e-6)

    def test_sample_all_masses_shapes_and_content(self):
        """sample_all_masses returns flattened masses (n + n_binary) and mask."""
        n = 1000
        imf = BinaryIMF.simple(PowerLawIMF.kroupa(), binary_fraction=0.5)
        key = jax.random.PRNGKey(99)
        all_masses, is_binary = imf.sample_all_masses(key, n)

        n_binary = int(jnp.sum(is_binary))
        assert is_binary.shape == (n,)
        # All n primaries + the n_binary secondaries
        assert all_masses.shape == (n + n_binary,), (
            f"expected {n + n_binary} masses, got {all_masses.shape[0]}"
        )
        # All masses strictly positive (secondaries with m2=0 were filtered out)
        assert jnp.all(all_masses > 0.0)

    def test_mean_system_mass_constant_fbin(self):
        """mean_system_mass = E[M1]*(1 + f_bin*E[q]) for constant f_bin + flat q."""
        primary = PowerLawIMF.kroupa()
        f_bin = 0.5
        q_min = 0.1
        imf = BinaryIMF.simple(primary, binary_fraction=f_bin, q_min=q_min)

        mean_m1 = primary.mean_mass()
        avg_q = (1.0 + q_min) / 2.0  # flat q -> mean is midpoint
        expected = mean_m1 * (1.0 + f_bin * avg_q)

        result = imf.mean_system_mass()
        assert jnp.isclose(result, expected, rtol=1e-5), (
            f"mean_system_mass {float(result):.4f} vs expected {float(expected):.4f}"
        )
        # Sanity: a binary population has more mass per system than singles only
        assert float(result) > float(mean_m1)

    def test_binary_fraction_overall_constant(self):
        """binary_fraction_overall returns the constant fraction exactly."""
        imf = BinaryIMF.simple(PowerLawIMF.kroupa(), binary_fraction=0.37)
        assert jnp.isclose(imf.binary_fraction_overall(), 0.37)

    def test_binary_fraction_overall_mass_dependent_in_range(self):
        """For a mass-dependent model, overall fraction is an IMF-weighted mean in (0,1)."""
        imf = BinaryIMF.moe2017(PowerLawIMF.kroupa())
        f_overall = imf.binary_fraction_overall()
        # Kroupa is bottom-heavy -> dominated by low-mass (low f_bin) stars
        assert 0.0 < f_overall < 1.0
        # M-dwarf floor is ~0.22, O-star ceiling ~0.90; weighted mean must lie between
        assert 0.2 < f_overall < 0.9

    def test_factory_moe2017_component_types(self):
        """moe2017() factory wires MoeDiStefano2017 + MassDependentBinaryFraction."""
        imf = BinaryIMF.moe2017(PowerLawIMF.kroupa())
        assert isinstance(imf.q_distribution, MoeDiStefano2017)
        assert isinstance(imf.binary_fraction, MassDependentBinaryFraction)

    def test_factory_massive_stars_component_types(self):
        """massive_stars() factory wires PowerLawMassRatio(gamma<0) + float f_bin."""
        imf = BinaryIMF.massive_stars(PowerLawIMF.kroupa(), gamma=-0.1, binary_fraction=0.7)
        assert isinstance(imf.q_distribution, PowerLawMassRatio)
        assert jnp.isclose(imf.q_distribution.gamma, -0.1)
        assert float(imf.binary_fraction) == 0.7
        # And the overall fraction equals that constant
        assert jnp.isclose(imf.binary_fraction_overall(), 0.7)

    def test_factory_massive_stars_default_fbin_sana(self):
        """massive_stars() DEFAULT f_bin = 0.69 (Sana et al. 2012, intrinsic f_bin=0.69±0.09)."""
        imf = BinaryIMF.massive_stars(PowerLawIMF.kroupa())
        assert float(imf.binary_fraction) == 0.69
        assert jnp.isclose(imf.q_distribution.gamma, -0.1)  # Sana kappa = -0.1 (uniform q)


# =============================================================================
# FD-vs-autodiff grad-checks + gamma=-1 edge + pdf_given_primary
# =============================================================================

from progenax.imf.differentiable_binary import (
    DifferentiableBinaryFraction,
    DifferentiableBinaryModel,
)

_U = jnp.array([0.2, 0.4, 0.6, 0.8])


def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _assert_grad_matches_fd(f, x0, h=1e-5, rtol=1e-4, atol=1e-9):
    g = jax.grad(f)(x0)
    g_fd = _central_fd(f, x0, h)
    assert jnp.isfinite(g), f"autodiff grad is {g}"
    assert jnp.abs(g) > 1e-6, f"grad effectively zero ({g}); FD says {g_fd}"
    assert jnp.abs(g - g_fd) <= rtol * jnp.abs(g_fd) + atol, (
        f"autodiff {float(g):.6e} vs FD {float(g_fd):.6e} "
        f"(rel {float(jnp.abs(g - g_fd) / (jnp.abs(g_fd) + 1e-12)):.2e})"
    )


class TestBinaryGradients:
    """Autodiff gradients of the binary samplers match central finite differences."""

    def test_powerlaw_ppf_grad_gamma(self):
        _assert_grad_matches_fd(
            lambda g: jnp.sum(PowerLawMassRatio(gamma=g, q_min=0.1).ppf(_U)), -0.1
        )

    def test_twinpeaked_ppf_grad_ftwin(self):
        _assert_grad_matches_fd(
            lambda ft: jnp.sum(TwinPeakedMassRatio(f_twin=ft, sigma_twin=0.05, q_min=0.1).ppf(_U)),
            0.1,
        )

    def test_twinpeaked_ppf_grad_sigma(self):
        _assert_grad_matches_fd(
            lambda s: jnp.sum(TwinPeakedMassRatio(f_twin=0.1, sigma_twin=s, q_min=0.1).ppf(_U)),
            0.05,
        )

    def test_diffmodel_grad_gamma_intercept(self):
        m1 = jnp.array([1.0, 5.0, 20.0, 50.0])
        ub = jnp.array([0.1, 0.3, 0.5, 0.7])
        uq = jnp.array([0.2, 0.4, 0.6, 0.8])

        def loss(gi):
            mdl = DifferentiableBinaryModel(
                binary_fraction=DifferentiableBinaryFraction.from_moe2017(),
                gamma_intercept=gi, gamma_slope=-0.7521, temperature=0.01,
            )
            m2, _ = mdl.sample_systems(m1, ub, uq)
            return jnp.sum(m2)

        _assert_grad_matches_fd(loss, 0.1907)


class TestPowerLawGammaMinusOne:
    """gamma = -1 (thermal q-distribution) must not crash and must be correct.

    lax.cond traces BOTH branches; the neq-branch's 1/(gamma+1) must be divide-safe so
    exactly gamma=-1 routes to the log-branch without a ZeroDivisionError at trace time.
    """

    def test_ppf_gamma_minus_one_no_crash(self):
        q = PowerLawMassRatio(gamma=-1.0, q_min=0.1).ppf(_U)
        assert jnp.all(jnp.isfinite(q)), f"non-finite ppf at gamma=-1: {q}"
        assert jnp.all((q >= 0.1 - 1e-6) & (q <= 1.0 + 1e-6)), f"q out of [q_min,1]: {q}"

    def test_cdf_gamma_minus_one_no_crash(self):
        c = PowerLawMassRatio(gamma=-1.0, q_min=0.1).cdf(jnp.array([0.1, 0.3, 0.6, 1.0]))
        assert jnp.all(jnp.isfinite(c)), f"non-finite cdf at gamma=-1: {c}"
        assert float(c[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(c[-1]) == pytest.approx(1.0, abs=1e-6)

    def test_gamma_minus_one_matches_limit(self):
        # gamma=-1 (log branch) should agree with gamma=-1+-eps (power branch) ppf.
        u = jnp.array([0.3, 0.5, 0.7])
        q_exact = PowerLawMassRatio(gamma=-1.0, q_min=0.1).ppf(u)
        q_near = PowerLawMassRatio(gamma=-1.0 + 1e-4, q_min=0.1).ppf(u)
        assert jnp.allclose(q_exact, q_near, atol=1e-3), f"{q_exact} vs {q_near}"


class TestMoeDiStefanoPDF:
    """MoeDiStefano2017.pdf_given_primary is a normalized density over [q_min, 1]."""

    @pytest.mark.parametrize("m1", [0.5, 1.0, 2.0, 10.0])
    def test_pdf_normalized(self, m1):
        moe = MoeDiStefano2017(q_min=0.1, sigma_twin=0.05)
        qg = jnp.linspace(0.1, 1.0, 5000)
        integral = float(jnp.trapezoid(moe.pdf_given_primary(qg, m1), qg))
        assert integral == pytest.approx(1.0, abs=0.02), f"integral={integral} at m1={m1}"
