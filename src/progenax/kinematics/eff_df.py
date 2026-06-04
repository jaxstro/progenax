"""
EFF (Elson-Fall-Freeman 1987) isotropic velocity DF via Eddington inversion.

The EFF profile rho(r) = rho_0 (1 + (r/a)^2)^(-gamma/2) has no closed-form DF, so we
build the exact isotropic ergodic DF f(E) from the density via the Eddington formula:

    f(E) = 1/(sqrt(8) pi^2) [ int_0^E (d^2 rho/d Psi^2)/sqrt(E - Psi) dPsi
                              + (d rho/d Psi)|_{Psi=0} / sqrt(E) ],

where Psi(r) is the relative potential (Psi(r_t)=0). The speed at radius r is then drawn
from g(v) ∝ v^2 f(Psi(r) - v^2/2) on [0, v_esc = sqrt(2 Psi(r))], isotropic directions.
The singular Eddington integral is evaluated with the substitution u = sqrt(E - Psi),
which removes the integrable 1/sqrt singularity.

Equilibrium fidelity: the EFF is an *empirical* density (not a DF model), so a sharply
truncated EFF is only approximately stationary. For mild truncation (e.g. gamma=5) the
sampled cluster is virial to ~1% (Q ~ 0.495); for the steep gamma=3 default, whose mass
diverges logarithmically, severe truncation leaves it ~5-8% sub-virial. This is intrinsic
to truncating an empirical profile, not a DF error. (For a strict lowered-DF equilibrium,
use the King model.)
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from progenax import defaults

_N_R = 6000      # radial grid for potential / density tabulation
_N_E = 1000      # energy grid for f(E)
_N_U = 2000      # quadrature points for the (substituted) Eddington integral
_N_SPEED = 256   # per-particle speed inverse-CDF resolution


def _eff_eddington_table(a, gamma, r_t, n_r=_N_R, n_e=_N_E):
    """Tabulate the dimensionless (G=1, rho_0=1) EFF potential and Eddington f(E)."""
    r = jnp.linspace(1e-5, r_t, n_r)
    rho = (1.0 + (r / a) ** 2) ** (-gamma / 2.0)
    drho_dr = -gamma * (r / a**2) * (1.0 + (r / a) ** 2) ** (-gamma / 2.0 - 1.0)

    dr = jnp.diff(r)

    def cumtrap(y):
        return jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (y[1:] + y[:-1]) * dr)])

    inner = cumtrap(rho * r**2)        # int_0^r rho s^2 ds
    tail = cumtrap(rho * r)            # int_0^r rho s ds
    outer = tail[-1] - tail            # int_r^{r_t} rho s ds
    Phi = -4.0 * jnp.pi * (inner / r + outer)   # gravitational potential (G=1, rho_0=1)
    Psi = Phi[-1] - Phi                # relative potential, Psi(r_t)=0, increases inward
    mu = inner[-1]                     # int_0^{r_t} rho s^2 ds (dimensionless mass / 4pi)

    Mr = 4.0 * jnp.pi * inner
    dPsi_dr = -Mr / r**2               # analytic dPsi/dr
    # drho/dPsi = (drho/dr)/(dPsi/dr). At r->0 the enclosed mass ->0, so dPsi/dr->0
    # (exactly at index 0, where inner[0]=0). A bare 0-denominator divide is finite
    # in the forward pass after the center fix below, but its BACKWARD pass is NaN
    # (0 * inf in the VJP), which kills grad w.r.t. (a, gamma). Guard with the
    # double-where pattern so no inf/NaN ever enters the graph, then set the center
    # point from its neighbor (the ratio has a finite limit 3 gamma / 4 pi a^2).
    safe_dPsi_dr = jnp.where(dPsi_dr == 0.0, 1.0, dPsi_dr)
    drho_dPsi = jnp.where(dPsi_dr == 0.0, 0.0, drho_dr / safe_dPsi_dr)
    drho_dPsi = drho_dPsi.at[0].set(drho_dPsi[1])
    d2rho_dPsi2 = jnp.gradient(drho_dPsi, Psi)

    Psi0 = Psi[0]
    Psi_asc = Psi[::-1]
    d2_asc = d2rho_dPsi2[::-1]
    bnd = drho_dPsi[-1]                # d rho/d Psi at Psi=0 (truncation boundary term)

    # End just below Psi0: E=Psi0 is the singular central energy (the Eddington
    # integrand reaches r->0); central lookups clamp to f_grid[-1], and the w^2 factor
    # makes the w->0 (E->Psi0) contribution negligible for sampling.
    E_grid = jnp.linspace(1e-4 * Psi0, 0.999 * Psi0, n_e)

    def f_one(E):
        # u = sqrt(E - Psi): int_0^E g/sqrt(E-Psi) dPsi = 2 int_0^sqrt(E) g(E-u^2) du
        u = jnp.linspace(0.0, jnp.sqrt(E), _N_U)
        g = jnp.interp(E - u**2, Psi_asc, d2_asc)
        return (2.0 * jnp.trapezoid(g, u) + bnd / jnp.sqrt(E)) / (jnp.sqrt(8.0) * jnp.pi**2)

    f_grid = jnp.maximum(jax.vmap(f_one)(E_grid), 0.0)
    return r, Psi, E_grid, f_grid, mu


def _sample_unit_speed(key, Psi_r, E_grid, f_grid, n_w):
    """Sample one speed w ~ w^2 f(Psi_r - w^2/2) on [0, sqrt(2 Psi_r)] (units of sqrt(kappa))."""
    Psi_safe = jnp.maximum(Psi_r, 1e-12)
    w_grid = jnp.linspace(0.0, jnp.sqrt(2.0 * Psi_safe), n_w)
    f_at = jnp.interp(Psi_r - w_grid**2 / 2.0, E_grid, f_grid)
    p = jnp.maximum(w_grid**2 * f_at, 0.0)
    dw = w_grid[1] - w_grid[0]
    cdf = jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (p[1:] + p[:-1])) * dw])
    cdf = cdf / (cdf[-1] + 1e-30)
    w = jnp.interp(jax.random.uniform(key), cdf, w_grid)
    return jnp.where(Psi_r > 1e-6, w, 0.0)


class EFFVelocityDF(eqx.Module):
    """
    EFF (Elson-Fall-Freeman 1987) isotropic velocity DF via Eddington inversion.

    Samples the exact isotropic ergodic DF f(E) of the (truncated) EFF density. The
    central velocity scale is fixed self-consistently from (G, M_total, a, gamma, r_t),
    so no external virial rescale is needed.

    Attributes:
        a: Scale radius [length units], must match the spatial profile
        gamma: Power-law index
        r_t: Tidal/truncation radius [length units]
        r_grid, Psi_grid: relative potential Psi(r) (dimensionless: G=1, rho_0=1)
        E_grid, f_grid: tabulated Eddington DF f(E) >= 0
        mu: int_0^{r_t} rho_tilde r^2 dr (sets rho_0 = M_total / (4 pi mu))

    References:
        Elson, Fall & Freeman (1987), ApJ, 323, 54
        Binney & Tremaine (2008), "Galactic Dynamics", 2nd ed., Eq. 4.46 (Eddington)
    """

    a: Float[Array, ""]
    gamma: Float[Array, ""]
    r_t: Float[Array, ""]
    r_grid: Float[Array, "n_r"]
    Psi_grid: Float[Array, "n_r"]
    E_grid: Float[Array, "n_e"]
    f_grid: Float[Array, "n_e"]
    mu: Float[Array, ""]

    def __init__(self, a: float = 1.0, gamma: float = 3.0, r_t: float = 10.0):
        self.a = jnp.asarray(a)
        self.gamma = jnp.asarray(gamma)
        self.r_t = jnp.asarray(r_t)
        r, Psi, E_grid, f_grid, mu = _eff_eddington_table(a, gamma, r_t)
        self.r_grid = r
        self.Psi_grid = Psi
        self.E_grid = E_grid
        self.f_grid = f_grid
        self.mu = mu

    def sample_velocities(
        self,
        positions: Float[Array, "N 3"],
        masses: Float[Array, "N"],
        key: PRNGKeyArray,
        G: float | None = None,
    ) -> Float[Array, "N 3"]:
        """
        Sample velocities from the EFF Eddington DF.

        At radius r, the speed is drawn from g(v) ∝ v^2 f(Psi(r) - v^2/2sigma^2) on
        [0, v_esc(r)] via a differentiable tabulated inverse-CDF; directions are isotropic.
        The physical velocity scale is sqrt(kappa), kappa = G rho_0 = G M_total / (4 pi mu).
        """
        if G is None:
            G = defaults.DEFAULT_UNITS.G

        N = positions.shape[0]
        M_total = jnp.sum(masses)
        radii = jnp.linalg.norm(positions, axis=1)

        Psi_r = jnp.interp(radii, self.r_grid, self.Psi_grid, left=self.Psi_grid[0], right=0.0)
        kappa = G * M_total / (4.0 * jnp.pi * self.mu)

        key_speed, key_dir = jax.random.split(key)
        speed_keys = jax.random.split(key_speed, N)
        w = jax.vmap(
            lambda k, p: _sample_unit_speed(k, p, self.E_grid, self.f_grid, _N_SPEED)
        )(speed_keys, Psi_r)
        speeds = jnp.sqrt(kappa) * w

        dirs = jax.random.normal(key_dir, shape=(N, 3))
        dirs = dirs / (jnp.linalg.norm(dirs, axis=1, keepdims=True) + 1e-30)

        return speeds[:, None] * dirs


__all__ = ["EFFVelocityDF"]
