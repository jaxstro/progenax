"""Unit tests for the gravoturb → differential-extinction adapter (CAREER Aim 3, extension A).

A young cluster is reddened *by its own natal gas*. gravoturb emits the residual-gas 3-D density
grid (``TurbulentCloudIC.gas.rho_residual`` [M⊙/pc³]); this adapter turns it into a physical,
spatially-correlated differential-extinction screen that plugs into fluxax's duck-typed
``dust_model.column(positions) -> A_V`` slot.

Convention (matches fluxax exactly): positions are cluster-frame (x, y, z) in pc; **z (axis 2) is the
line of sight**, the observer sits at small z, the near face is grid index 0, and the reddening column
accumulates from the near face to the star (star-embedded depth — spec decision A(a)).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.experimental

# --- Reference constants for the solar-anchor amplitude (locked in the test to pin the physics) ---
# 1 M⊙/pc² → g/cm² (M⊙ = 1.98892e33 g over pc² = (3.0857e18 cm)²), mirrors feasibility_figure.py:60.
_MSUN_PC2_TO_G_CM2 = 1.98892e33 / (3.0857e18) ** 2
_M_H = 1.6726219e-24          # hydrogen mass [g]
_MU = 1.4                     # mean molecular weight per H (atomic H + He)
_N_H_PER_A_V_SOLAR = 1.9e21   # N_H/A_V [cm⁻² mag⁻¹], MW/solar anchor (Bohlin+1978, rounded)


def _av_from_sigma(sigma_msun_pc2):
    """Expected solar A_V from a LOS gas surface density [M⊙/pc²], from first principles."""
    sigma_g_cm2 = sigma_msun_pc2 * _MSUN_PC2_TO_G_CM2
    n_h = sigma_g_cm2 / (_MU * _M_H)          # [cm⁻²]
    return n_h / _N_H_PER_A_V_SOLAR           # [mag]


def _uniform_gas(n=16, box=4.0, rho0=1.0):
    """A uniform residual-gas cube: shape (n,n,n) at density rho0 [M⊙/pc³], box side ``box`` [pc]."""
    rho = jnp.full((n, n, n), rho0)
    origin = jnp.array([box / 2, box / 2, box / 2])  # cluster COM at box centre → cluster frame in [-box/2, box/2)
    return rho, box, origin


def test_column_shape_and_positive():
    """.column(positions) returns one A_V per star, all strictly positive inside the gas."""
    from gravoturb.extinction import GravoturbDustModel

    rho, box, origin = _uniform_gas()
    dust = GravoturbDustModel.from_grid(rho, box_size=box, origin=origin)
    pos = jnp.array([[0.0, 0.0, 0.0], [0.5, -0.5, 1.0], [-1.0, 0.3, -1.0]])
    av = dust.column(pos)
    assert av.shape == (3,)
    assert jnp.all(av >= 0.0)


def test_uniform_slab_oriented_and_monotone():
    """Uniform slab: A_V rises monotonically along +z (LOS), ~0 at the near face, max at the far face."""
    from gravoturb.extinction import GravoturbDustModel

    rho, box, origin = _uniform_gas()
    dust = GravoturbDustModel.from_grid(rho, box_size=box, origin=origin)
    # Stars on the LOS axis from near face (z=-box/2) to far face (z≈+box/2), transverse-centred.
    z = jnp.linspace(-box / 2 + 1e-3, box / 2 - 1e-3, 9)
    pos = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=-1)
    av = dust.column(pos)
    assert jnp.all(jnp.diff(av) > 0)                 # monotone increasing with LOS depth
    assert float(av[0]) < 0.05 * float(av[-1])       # near-face star barely reddened


def test_uniform_slab_linear_in_depth():
    """Uniform slab → column linear in depth → A_V linear in z (analytic limit, spec validation (ii))."""
    from gravoturb.extinction import GravoturbDustModel

    rho, box, origin = _uniform_gas()
    dust = GravoturbDustModel.from_grid(rho, box_size=box, origin=origin)
    z = jnp.linspace(-box / 2 + 0.5, box / 2 - 0.5, 12)  # interior only (avoid edge clamp)
    pos = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=-1)
    av = np.asarray(dust.column(pos))
    r = np.corrcoef(np.asarray(z), av)[0, 1]
    assert r > 0.9999                                # linear in depth


def test_uniform_slab_solar_amplitude():
    """A_V magnitude matches the N_H/A_V=1.9e21 physics for the known LOS column (spec (i) amplitude)."""
    from gravoturb.extinction import GravoturbDustModel

    n, box, rho0 = 16, 4.0, 0.7
    rho, box, origin = _uniform_gas(n=n, box=box, rho0=rho0)
    dust = GravoturbDustModel.from_grid(rho, box_size=box, origin=origin)
    # Star at the far face: full-slab LOS column Σ = rho0 * box  [M⊙/pc²] (minus half a cell at centre).
    cell = box / n
    z_far = box / 2 - cell / 2          # centre of the last cell
    pos = jnp.array([[0.0, 0.0, z_far]])
    # column from near face to the last cell CENTRE = (n-0.5)·rho·cell = rho·(box - cell/2)
    sigma = rho0 * (box - cell / 2)
    expected = _av_from_sigma(sigma)
    got = float(dust.column(pos)[0])
    assert got == pytest.approx(expected, rel=1e-3)


def test_column_gradient_ad_vs_fd():
    """Gradient of mean A_V w.r.t. a gas-density scale is AD-correct (the Aim-3 Fisher backbone)."""
    from gravoturb.extinction import GravoturbDustModel

    n, box = 12, 4.0
    base = jax.random.uniform(jax.random.PRNGKey(0), (n, n, n)) + 0.2  # positive lognormal-ish gas
    origin = jnp.array([box / 2, box / 2, box / 2])
    pos = jax.random.uniform(jax.random.PRNGKey(1), (20, 3)) * (box * 0.8) - box * 0.4  # interior

    def mean_av(scale):
        dust = GravoturbDustModel.from_grid(scale * base, box_size=box, origin=origin)
        return jnp.mean(dust.column(pos))

    g_ad = float(jax.grad(mean_av)(1.0))
    eps = 1e-4
    g_fd = float((mean_av(1.0 + eps) - mean_av(1.0 - eps)) / (2 * eps))
    assert g_ad == pytest.approx(g_fd, rel=1e-5)
    assert jnp.isfinite(g_ad)


# ---------------------------------------------------------------------------
# Metallicity-keyed dust-to-gas (Rémy-Ruyer+2014 broken power law, X_CO,Z case)
# ---------------------------------------------------------------------------

def test_remy_ruyer_solar_is_anchor():
    """At [Fe/H]=0 (solar) the Rémy-Ruyer scaling returns exactly the MW anchor N_H/A_V=1.9e21."""
    from gravoturb.extinction import remy_ruyer_n_h_per_a_v

    assert float(remy_ruyer_n_h_per_a_v(0.0)) == pytest.approx(1.9e21, rel=1e-6)


def test_remy_ruyer_high_metallicity_branch_is_inverse_Z():
    """Above the break (x>x_t): N_H/A_V ∝ 10^(-[Fe/H]) = Z_sun/Z (near-solar linear regime, slope 1)."""
    from gravoturb.extinction import remy_ruyer_n_h_per_a_v

    # [Fe/H]=-0.3 → x=8.39 > x_t=8.10 → factor 10^0.3
    got = float(remy_ruyer_n_h_per_a_v(-0.3))
    assert got == pytest.approx(1.9e21 * 10 ** 0.3, rel=1e-4)


def test_remy_ruyer_low_metallicity_branch_is_steep():
    """Below the break (x<=x_t): steep slope alpha_L=3.10 → strongly reduced dust (higher N_H/A_V)."""
    from gravoturb.extinction import remy_ruyer_n_h_per_a_v

    # [Fe/H]=-1.0 → x=7.69 < x_t → log-factor = (0.96-2.21) - 3.10*(-1.0) = 1.85
    got = float(remy_ruyer_n_h_per_a_v(-1.0))
    assert got == pytest.approx(1.9e21 * 10 ** 1.85, rel=1e-3)


def test_remy_ruyer_continuous_at_break():
    """The broken power law is C0-continuous at the transition [Fe/H]_t = 8.10-8.69 = -0.59."""
    from gravoturb.extinction import remy_ruyer_n_h_per_a_v

    feh_t = 8.10 - 8.69
    below = float(remy_ruyer_n_h_per_a_v(feh_t - 1e-4))
    above = float(remy_ruyer_n_h_per_a_v(feh_t + 1e-4))
    assert below == pytest.approx(above, rel=0.03)   # continuous within the paper's 2-dec rounding


def test_remy_ruyer_monotone_thins_dust_at_low_Z():
    """Lower metallicity → more gas per dust → higher N_H/A_V (monotone), so A_V per column drops."""
    from gravoturb.extinction import remy_ruyer_n_h_per_a_v

    fehs = jnp.array([0.3, 0.0, -0.5, -1.0, -2.0])
    vals = jnp.array([float(remy_ruyer_n_h_per_a_v(f)) for f in fehs])
    assert jnp.all(jnp.diff(vals) > 0)               # increasing as [Fe/H] decreases


def test_low_metallicity_reduces_extinction_via_from_grid():
    """A low-Z birth yields less extinction for the SAME gas column (spec decision A(b))."""
    from gravoturb.extinction import GravoturbDustModel

    rho, box, origin = _uniform_gas(rho0=1.0)
    pos = jnp.array([[0.0, 0.0, 1.5]])
    solar = GravoturbDustModel.from_grid(rho, box_size=box, origin=origin, feh=0.0)
    poor = GravoturbDustModel.from_grid(rho, box_size=box, origin=origin, feh=-1.0)
    assert float(poor.column(pos)[0]) < float(solar.column(pos)[0])
    # ratio set purely by the dust-to-gas scaling (same geometry): A_V_poor/A_V_solar = 1.9e21/N_H/A_V(-1)
    ratio = float(poor.column(pos)[0]) / float(solar.column(pos)[0])
    assert ratio == pytest.approx(10 ** (-1.85), rel=1e-3)


def test_from_ic_reads_birth_environment_metallicity():
    """from_ic keys the dust amplitude to BirthEnvironment.metallicity (joint env-memory imprint)."""
    from gravoturb.extinction import GravoturbDustModel
    from gravoturb.cluster import GasState, Geometry, TurbulentCloudIC
    from progenax.imf import BirthEnvironment

    n, box = 12, 4.0
    rho = jnp.full((n, n, n), 1.0)
    cell_vol = (box / n) ** 3
    gas = GasState(rho_cloud=rho, rho_residual=rho, velocity=jnp.zeros((n, n, n, 3)),
                   pressure=rho, cell_volume=jnp.asarray(cell_vol))
    geom = Geometry(shape=(n, n, n), box_size=jnp.asarray(box),
                    origin=jnp.array([box / 2, box / 2, box / 2]))
    ic = TurbulentCloudIC(stars=None, gas=gas, fields=None, geometry=geom,
                          physics=None, ledger=None)
    pos = jnp.array([[0.0, 0.0, 1.5]])

    solar_env = BirthEnvironment.solar()
    poor_env = BirthEnvironment.from_cluster_mass(1e4, FeH=-1.0)
    av_solar = float(GravoturbDustModel.from_ic(ic, env=solar_env).column(pos)[0])
    av_poor = float(GravoturbDustModel.from_ic(ic, env=poor_env).column(pos)[0])
    assert av_poor < av_solar
    # from_ic with no env defaults to solar
    av_default = float(GravoturbDustModel.from_ic(ic).column(pos)[0])
    assert av_default == pytest.approx(av_solar, rel=1e-6)
