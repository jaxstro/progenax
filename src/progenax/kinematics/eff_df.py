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
from progenax.kinematics.eddington import (
    assign_om_directions,
    sample_speed_from_f_table,
)

_N_R = 6000      # radial grid for potential / density tabulation
_N_E = 1000      # energy grid for f(E)
_N_U = 2000      # quadrature points for the (substituted) Eddington integral
_N_SPEED = 256   # per-particle speed inverse-CDF resolution


def _eff_eddington_table(a, gamma, r_t, r_a=None, n_r=_N_R, n_e=_N_E):
    """Tabulate the dimensionless (G=1, rho_0=1) EFF potential and Eddington f(E).

    With anisotropy_radius r_a set, the *potential* Psi(r) still comes from the true
    EFF density rho, but the DF is the Eddington inversion of the Osipkov-Merritt
    *augmented* density rho_Q = (1 + r^2/r_a^2) rho (Merritt 1985, Eqs. 9-11). r_a=None
    recovers the isotropic Eddington DF exactly (rho_Q = rho).
    """
    r = jnp.linspace(1e-5, r_t, n_r)
    rho = (1.0 + (r / a) ** 2) ** (-gamma / 2.0)
    drho_dr = -gamma * (r / a**2) * (1.0 + (r / a) ** 2) ** (-gamma / 2.0 - 1.0)

    # Osipkov-Merritt augmented density rho_Q = (1 + r^2/r_a^2) rho (and its r-deriv).
    # The DF inversion below uses rho_Q in place of rho; the potential Psi is unchanged
    # (it is set by the true mass density rho). r_a=None -> weight 1 -> isotropic.
    if r_a is None:
        rho_df, drho_df_dr = rho, drho_dr
    else:
        w = 1.0 + (r / r_a) ** 2
        rho_df = w * rho
        drho_df_dr = (2.0 * r / r_a**2) * rho + w * drho_dr

    dr = jnp.diff(r)

    def cumtrap(y):
        return jnp.concatenate([jnp.zeros(1), jnp.cumsum(0.5 * (y[1:] + y[:-1]) * dr)])

    inner = cumtrap(rho * r**2)        # int_0^r rho s^2 ds  (true density -> potential)
    tail = cumtrap(rho * r)            # int_0^r rho s ds
    outer = tail[-1] - tail            # int_r^{r_t} rho s ds
    Phi = -4.0 * jnp.pi * (inner / r + outer)   # gravitational potential (G=1, rho_0=1)
    Psi = Phi[-1] - Phi                # relative potential, Psi(r_t)=0, increases inward
    mu = inner[-1]                     # int_0^{r_t} rho s^2 ds (dimensionless mass / 4pi)

    # DF inversion variables use the augmented density rho_df (= rho if isotropic).
    drho_dr = drho_df_dr
    Mr = 4.0 * jnp.pi * inner
    dPsi_dr = -Mr / r**2               # analytic dPsi/dr (from true density)
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

    # Raw (unclamped) f(E): the speed sampler clamps the speed pdf at use, and the raw
    # values let the DF constructor detect a genuinely negative (unphysical) OM DF.
    f_grid = jax.vmap(f_one)(E_grid)
    return r, Psi, E_grid, f_grid, mu


class EFFVelocityDF(eqx.Module):
    """
    EFF (Elson-Fall-Freeman 1987) velocity DF via Eddington inversion.

    Samples the exact ergodic DF f(E) of the (truncated) EFF density. The central
    velocity scale is fixed self-consistently from (G, M_total, a, gamma, r_t), so no
    external virial rescale is needed.

    With ``anisotropy_radius`` (r_a) set, the DF is the Osipkov-Merritt radially
    anisotropic model for the same EFF density: f = f(Q), Q = E + J^2/2r_a^2, built by
    Eddington inversion of the augmented density rho_Q = (1 + r^2/r_a^2) rho (Merritt
    1985). The realised anisotropy is beta(r) = r^2/(r^2 + r_a^2). r_a=None is isotropic.

    Attributes:
        a: Scale radius [length units], must match the spatial profile
        gamma: Power-law index
        r_t: Tidal/truncation radius [length units]
        anisotropy_radius: Osipkov-Merritt radius r_a, or None for isotropic
        r_grid, Psi_grid: relative potential Psi(r) (dimensionless: G=1, rho_0=1)
        E_grid, f_grid: tabulated ergodic DF f(E) (augmented density if anisotropic)
        mu: int_0^{r_t} rho_tilde r^2 dr (sets rho_0 = M_total / (4 pi mu))

    References:
        Elson, Fall & Freeman (1987), ApJ, 323, 54
        Binney & Tremaine (2008), "Galactic Dynamics", 2nd ed., Eq. 4.46 (Eddington)
        Merritt (1985), AJ, 90, 1027 (Osipkov-Merritt anisotropy)
    """

    a: Float[Array, ""]
    gamma: Float[Array, ""]
    r_t: Float[Array, ""]
    anisotropy_radius: Float[Array, ""] | None
    r_grid: Float[Array, "n_r"]
    Psi_grid: Float[Array, "n_r"]
    E_grid: Float[Array, "n_e"]
    f_grid: Float[Array, "n_e"]
    mu: Float[Array, ""]

    def __init__(
        self,
        a: float = 1.0,
        gamma: float = 3.0,
        r_t: float = 10.0,
        anisotropy_radius: float | None = None,
    ):
        self.a = jnp.asarray(a)
        self.gamma = jnp.asarray(gamma)
        self.r_t = jnp.asarray(r_t)
        self.anisotropy_radius = (
            None if anisotropy_radius is None else jnp.asarray(anisotropy_radius)
        )
        r, Psi, E_grid, f_grid, mu = _eff_eddington_table(a, gamma, r_t, anisotropy_radius)
        # Refuse a genuinely negative (unphysical) Osipkov-Merritt DF rather than
        # silently clamping it: too small an r_a asks for more radial anisotropy than
        # the density can support with f >= 0 (Merritt 1985, Eq. 46). Concrete-r_a only;
        # under tracing (grad w.r.t. r_a) the table stays traced-safe and the check skips.
        if isinstance(anisotropy_radius, (int, float)):
            f_min = float(jnp.min(f_grid))
            f_max = float(jnp.max(jnp.abs(f_grid)))
            if f_min < -1e-3 * f_max:
                raise ValueError(
                    f"EFF Osipkov-Merritt DF is negative (min f / max|f| = "
                    f"{f_min / (f_max + 1e-300):.2e}) for anisotropy_radius="
                    f"{anisotropy_radius}: too radially anisotropic for gamma={gamma}, "
                    f"r_t={r_t}. Increase anisotropy_radius."
                )
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

        At radius r the speed is drawn from g(s) ∝ s^2 f(Psi(r) - s^2/2) via a
        differentiable tabulated inverse-CDF; the physical velocity scale is sqrt(kappa),
        kappa = G rho_0 = G M_total / (4 pi mu). Directions are isotropic when
        anisotropy_radius is None, else the Osipkov-Merritt stretched split (Merritt 1985).
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
        s = jax.vmap(
            lambda k, p: sample_speed_from_f_table(k, p, self.E_grid, self.f_grid, _N_SPEED)
        )(speed_keys, Psi_r)
        speeds = jnp.sqrt(kappa) * s

        r_a = None if self.anisotropy_radius is None else self.anisotropy_radius
        return assign_om_directions(key_dir, positions, speeds, r_a)


__all__ = ["EFFVelocityDF"]
