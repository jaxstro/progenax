"""Moe & Di Stefano (2017) full mass-dependent mass-ratio model.

Split from ``binary.py`` (file-length limit). Reference: Moe & Di Stefano (2017)
ApJS 230, 15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

if TYPE_CHECKING:
    from ...binaries import MoeEccentricity


class MoeDiStefano2017(eqx.Module):
    """Mass-dependent mass-ratio distribution from Moe & Di Stefano (2017).

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15, Table 13.

    APPROXIMATION — single-slope, period-averaged. Moe & Di Stefano's actual mass-ratio
    distribution is a THREE-parameter, PERIOD-DEPENDENT form (Table 13): two power-law
    slopes γ_smallq (0.1<q<0.3) and γ_largeq (0.3<q<1.0) plus a twin excess F_twin, with
    all three tabulated as functions of BOTH primary mass AND orbital period (at logP=1,3,5,7).
    This class collapses that to a SINGLE period-averaged slope γ(M1) and a period-averaged
    F_twin(M1) — it captures the qualitative trend (low-mass companions favour equal q, OB
    companions favour small q) but is not a verbatim Table row. A faithful two-slope,
    period-dependent implementation is tracked for a future release.

    Period-averaged single-slope reduction (γ from the qualitative γ_smallq/γ_largeq trend;
    f_twin period-averaged from Table 13's F_twin, which falls from ~0.1–0.3 at logP=1 to
    <0.03 at long P):
        - M1 < 0.8 Msun: γ ≈ 0.4, f_twin ≈ 0.05 (M-dwarfs)
        - 0.8 < M1 < 1.2 Msun: γ ≈ 0.3, f_twin ≈ 0.10 (Solar-type; Table 13 F_twin period-avg)
        - 1.2 < M1 < 3.5 Msun: γ ≈ 0.0, f_twin ≈ 0.08 (A/F stars)
        - M1 > 3.5 Msun: γ ≈ -0.5, f_twin ≈ 0.03 (OB stars; Table 13 γ_largeq(logP=1)=-0.5)

    Parameters:
        q_min: Minimum mass ratio (default: 0.1)
        sigma_twin: Width of twin peak (default: 0.03)
    """

    q_min: float = 0.1
    sigma_twin: float = 0.03

    def _gamma_of_mass(self, m1: Float[Array, "..."]) -> Float[Array, "..."]:
        """Period-averaged single-slope γ(M1) (reduction of Moe+17 Table 13 γ_smallq/γ_largeq)."""
        # Piecewise linear interpolation
        gamma = jnp.where(
            m1 < 0.8,
            0.4,  # M-dwarfs
            jnp.where(
                m1 < 1.2,
                0.3,  # Solar-type
                jnp.where(
                    m1 < 3.5,
                    0.0,  # A/F stars
                    -0.5,  # OB stars
                ),
            ),
        )
        return gamma

    def _ftwin_of_mass(self, m1: Float[Array, "..."]) -> Float[Array, "..."]:
        """Period-averaged twin excess F_twin(M1) (from Moe+17 Table 13, averaged over logP)."""
        f_twin = jnp.where(
            m1 < 0.8,
            0.05,  # M-dwarfs
            jnp.where(
                m1 < 1.2,
                0.10,  # Solar-type (highest twin excess)
                jnp.where(
                    m1 < 3.5,
                    0.08,  # A/F stars
                    0.03,  # OB stars (lowest twin excess)
                ),
            ),
        )
        return f_twin

    def sample_given_primary(
        self, key: PRNGKeyArray, m1: Float[Array, "n"]
    ) -> Float[Array, "n"]:
        """Sample mass ratios given primary masses.

        This is the key method: q distribution depends on M1.

        Args:
            key: JAX random key
            m1: Primary masses (n,)

        Returns:
            Mass ratios q ∈ [q_min, 1] with shape (n,)
        """
        n = m1.shape[0]
        key1, key2, key3 = jax.random.split(key, 3)

        # Get mass-dependent parameters
        gamma = self._gamma_of_mass(m1)
        f_twin = self._ftwin_of_mass(m1)

        # Decide which component: power-law or twin
        is_twin = jax.random.uniform(key1, (n,)) < f_twin

        # Sample power-law component
        u_pl = jax.random.uniform(key2, (n,))

        def sample_powerlaw(gamma_val, u_val):
            """Sample from power-law q^gamma."""
            q0 = self.q_min

            def neq_m1():
                g = gamma_val
                norm = (1.0 - q0 ** (g + 1.0)) / (g + 1.0)
                inner = u_val * (g + 1.0) * norm + q0 ** (g + 1.0)
                return inner ** (1.0 / (g + 1.0))

            def eq_m1():
                norm = -jnp.log(q0)
                return q0 * jnp.exp(u_val * norm)

            return jax.lax.cond(jnp.abs(gamma_val + 1.0) < 1e-10, eq_m1, neq_m1)

        q_powerlaw = jax.vmap(sample_powerlaw)(gamma, u_pl)

        # Sample twin component (truncated Gaussian centered at q=1, truncated to [q_min, 1])
        # Use inverse CDF sampling for correct distribution
        z_min = (self.q_min - 1.0) / self.sigma_twin
        _z_max = 0.0  # (1.0 - 1.0) / sigma = 0  (documents upper truncation bound)
        # CDF values at boundaries
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        cdf_max = 0.5  # Φ(0) = 0.5
        # Sample uniform in [cdf_min, cdf_max], then inverse CDF
        u_twin = jax.random.uniform(key3, (n,))
        u_scaled = cdf_min + u_twin * (cdf_max - cdf_min)
        # Inverse CDF: Φ^(-1)(u) = √2 * erfinv(2u - 1)
        z_samples = jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u_scaled - 1.0)
        q_twin = 1.0 + z_samples * self.sigma_twin

        # Select based on component
        q = jnp.where(is_twin, q_twin, q_powerlaw)
        return q

    def pdf_given_primary(
        self, q: Float[Array, "..."], m1: float
    ) -> Float[Array, "..."]:
        """PDF of mass ratio given primary mass."""
        gamma = self._gamma_of_mass(jnp.asarray(m1))
        f_twin = self._ftwin_of_mass(jnp.asarray(m1))

        # Power-law normalization
        g = gamma
        q0 = self.q_min
        pl_norm = jax.lax.cond(
            jnp.abs(g + 1.0) < 1e-10,
            lambda: -jnp.log(q0),
            lambda: (1.0 - q0 ** (g + 1.0)) / (g + 1.0),
        )
        pl_pdf = q**g / pl_norm

        # Gaussian twin peak
        z_min = (q0 - 1.0) / self.sigma_twin
        cdf_min = 0.5 * (1.0 + jax.scipy.special.erf(z_min / jnp.sqrt(2.0)))
        gauss_norm = 0.5 - cdf_min
        gauss_pdf = (
            jnp.exp(-0.5 * ((q - 1.0) / self.sigma_twin) ** 2)
            / (self.sigma_twin * jnp.sqrt(2.0 * jnp.pi))
            / gauss_norm
        )

        combined = (1.0 - f_twin) * pl_pdf + f_twin * gauss_pdf
        in_range = (q >= q0) & (q <= 1.0)
        return jnp.where(in_range, combined, 0.0)


# =============================================================================
# Faithful two-slope, period-dependent model (Batch 4i)
# =============================================================================

# Table 13 (Moe & Di Stefano 2017, p.52), transcribed + verified 2026-06-04.
# Rows = log P {1,3,5,7}; cols = mass bins {Solar, A/late-B, Mid-B, Early-B, O}
# at representative masses 1.0/3.2/6.7/12/20 Msun. "<0.03" twin cells -> 0.
_MASS_NODES_LOG = jnp.log10(jnp.array([1.0, 3.2, 6.7, 12.0, 20.0]))
_LOGP_NODES = jnp.array([1.0, 3.0, 5.0, 7.0])
_GAMMA_LARGEQ = jnp.array(
    [  # Moe & Di Stefano (2017) ApJS 230, 15, Table 13 (p.52)
        [-0.5, -0.5, -0.5, -0.5, -0.5],
        [-0.5, -0.9, -1.7, -1.7, -1.7],
        [-0.5, -1.4, -2.0, -2.0, -2.0],
        [-1.1, -2.0, -2.0, -2.0, -2.0],
    ]
)
_GAMMA_SMALLQ = jnp.array(
    [  # Moe & Di Stefano (2017) ApJS 230, 15, Table 13 (p.52)
        [0.3, 0.2, 0.1, 0.1, 0.1],
        [0.3, 0.1, -0.2, -0.2, -0.2],
        [0.3, -0.5, -1.2, -1.2, -1.2],
        [0.3, -1.0, -1.5, -1.5, -1.5],
    ]
)
_F_TWIN = jnp.array(
    [  # Moe & Di Stefano (2017) ApJS 230, 15, Table 13 (p.52), F_twin(logP=1)
        [0.30, 0.22, 0.17, 0.14, 0.08],
        [0.20, 0.10, 0.0, 0.0, 0.0],
        [0.10, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def _bilinear(grid: Float[Array, "4 5"], logP, logm1):
    """Bilinear interpolation of a (logP, mass) grid, clamped outside (jnp.interp)."""
    logP, logm1 = jnp.broadcast_arrays(
        jnp.asarray(logP, dtype=jnp.float64), jnp.asarray(logm1, dtype=jnp.float64)
    )
    shape = logP.shape
    fp = logP.reshape(-1)
    fm = logm1.reshape(-1)
    rows = jnp.stack(
        [jnp.interp(fm, _MASS_NODES_LOG, grid[i]) for i in range(4)], axis=0
    )
    out = jax.vmap(lambda lp, col: jnp.interp(lp, _LOGP_NODES, col))(fp, rows.T)
    return out.reshape(shape)


def _pl_integral(a, b, gamma):
    """∫_a^b q^gamma dq, divide-safe at gamma=-1 (double-where)."""
    gp1 = gamma + 1.0
    is_m1 = jnp.abs(gp1) < 1e-10
    gp1s = jnp.where(is_m1, 1.0, gp1)
    general = (b**gp1s - a**gp1s) / gp1s
    return jnp.where(is_m1, jnp.log(b / a), general)


class MoeDiStefano2017Full(eqx.Module):
    """Faithful two-slope, period-dependent Moe & Di Stefano (2017) mass-ratio model.

    The mass-ratio pdf is a two-slope power law plus a twin excess block,
    jointly normalized over [q_min, 1.0]:

        p_q(q | M1, P) ∝ p_2slope(q) + [twin block on [0.95, 1.0]]

    where p_2slope ∝ q^γsmallq on [q_min, 0.3] then q^γlargeq on [0.3, 1.0]
    (continuous at q=0.3). F_twin is the EXCESS-twin fraction of the q > 0.3
    population (MD17 p.5, Fig. 2 — NOT the fraction of all q > q_min companions,
    and NOT the total q > 0.95 fraction). To realize that convention the twin
    block carries unnormalized mass ft/(1-ft)·I_B (I_B = power-law mass on
    [0.3, 1]), so twin/(twin + I_B) = ft exactly (see `_components`). The pre-fix
    code mixed ft against the whole [q_min, 1] population, over-weighting twins
    by ~22% at solar logP=1 (realized 0.367 vs Table-13 0.300) — audit R3.
    γsmallq, γlargeq, F_twin are bilinearly interpolated (clamped) over
    Table 13 (verified against the PDF p.52). η(P,M1) for eccentricity is handled by
    `progenax.binaries.MoeEccentricity` (which reproduces Table 13's η).

    This is period-dependent (sample takes periods AND masses) — it does NOT fit the
    unconditional MassRatioProtocol; use it via `MoeJointOrbit`. The single-slope
    period-averaged `MoeDiStefano2017` remains the fast approximation / BinaryIMF default.

    Reference:
        Moe & Di Stefano (2017) ApJS 230, 15, §9.1 Eqs. 2-3, Table 13 (p.52), Fig. 2.

    Parameters:
        q_min: Minimum mass ratio for the normalized range (default: 0.1).
    """

    q_min: float = 0.1
    q_break: float = 0.3
    n_grid: int = 512

    def gamma_smallq(self, periods, masses):
        """Power-law slope across 0.1<q<0.3 (Table 13, bilinear in logM1, logP)."""
        return _bilinear(_GAMMA_SMALLQ, jnp.log10(periods), jnp.log10(masses))

    def gamma_largeq(self, periods, masses):
        """Power-law slope across 0.3<q<1.0 (Table 13, bilinear in logM1, logP)."""
        return _bilinear(_GAMMA_LARGEQ, jnp.log10(periods), jnp.log10(masses))

    def f_twin(self, periods, masses):
        """Excess twin fraction at q>0.95 (Table 13, bilinear; <0.03 cells -> 0)."""
        return _bilinear(_F_TWIN, jnp.log10(periods), jnp.log10(masses))

    def _two_slope(self, periods, masses):
        gs = self.gamma_smallq(periods, masses)
        gl = self.gamma_largeq(periods, masses)
        C = jnp.power(self.q_break, gs - gl)  # continuity at q_break
        I_A = _pl_integral(self.q_min, self.q_break, gs)
        I_B = C * _pl_integral(self.q_break, 1.0, gl)
        return gs, gl, C, I_A, I_B

    def _components(self, q, masses, periods):
        """Normalized (power-law, twin-excess) mixture components.

        Paper convention (MD17 p.5, Fig. 2): F_twin is the excess-twin fraction
        of q > 0.3 companions. The unnormalized twin block therefore carries
        mass ft/(1-ft) * I_B (I_B = power-law mass on [q_break, 1]), so that
        twin/(twin + I_B) = ft exactly after joint normalization.
        """
        gs, gl, C, I_A, I_B = self._two_slope(periods, masses)
        ft = self.f_twin(periods, masses)
        p_lo = jnp.power(q, gs)
        p_hi = C * jnp.power(q, gl)
        p_pl_unnorm = jnp.where(q < self.q_break, p_lo, p_hi)
        ft_safe = jnp.minimum(ft, 0.95)  # Table 13 max is ~0.3; guard 1/(1-ft)
        twin_mass = ft_safe / (1.0 - ft_safe) * I_B
        twin_unnorm = twin_mass * jnp.where((q >= 0.95) & (q <= 1.0), 1.0 / 0.05, 0.0)
        Z_tot = I_A + I_B + twin_mass
        in_range = (q >= self.q_min) & (q <= 1.0)
        p_pl = jnp.where(in_range, p_pl_unnorm / Z_tot, 0.0)
        p_twin = jnp.where(in_range, twin_unnorm / Z_tot, 0.0)
        return p_pl, p_twin

    def pdf(self, q, masses, periods):
        """Conditional mass-ratio pdf p(q | M1, P), normalized on [q_min, 1].

        F_twin follows the paper convention: excess-twin fraction of the
        q > 0.3 population (NOT of all q > q_min companions — audit R3).
        """
        p_pl, p_twin = self._components(q, masses, periods)
        return p_pl + p_twin

    def sample(self, key: PRNGKeyArray, masses, periods):
        """Sample q | (M1 [Msun], P [days]) from the two-slope + twin mixture.

        Uses a grid-based inverse-CDF of the analytic `pdf` (like `MoePeriod`): this is
        smooth and **properly reparameterized**, so gradients wrt the mixture weights
        (γ, F_twin -> M1, P) are FD-accurate — unlike a multi-uniform segment/twin
        decision, which would block the reassignment gradient.

        Args:
            masses: primary masses (n,) [Msun]; periods: orbital periods (n,) [days].
        """
        n = masses.shape[0]
        q_grid = jnp.linspace(self.q_min, 1.0, self.n_grid)
        p = self.pdf(q_grid, masses[:, None], periods[:, None])  # (n, G)
        dq = q_grid[1] - q_grid[0]
        cdf = jnp.cumsum(p, axis=1) * dq
        cdf = cdf - cdf[:, :1]
        cdf = cdf / cdf[:, -1:]
        u = jax.random.uniform(key, (n,))
        return jax.vmap(lambda c, uu: jnp.interp(uu, c, q_grid))(cdf, u)


# Companion frequency f_logP;q>0.1 per dex (Table 13) — the M1-dependent period
# distribution shape. Rows = logP {1,3,5,7}; cols = mass bins.
_COMPANION_FREQ = jnp.array(
    [  # Moe & Di Stefano (2017) ApJS 230, 15, Table 13 (p.52) f_logP;q>0.1
        [0.027, 0.07, 0.14, 0.19, 0.29],
        [0.057, 0.12, 0.22, 0.26, 0.32],
        [0.095, 0.13, 0.20, 0.23, 0.30],
        [0.075, 0.09, 0.11, 0.13, 0.18],
    ]
)


class MoePeriod(eqx.Module):
    """Mass-dependent orbital-period distribution from Moe & Di Stefano (2017) Table 13.

    The companion-frequency anchors f_logP;q>0.1(M1) at logP={1,3,5,7} (bilinear in
    M1) define the period-distribution *shape*; we sample logP by inverting the
    normalized cumulative of the piecewise-linear density over [logP_min, logP_max]
    (clamped at the edges). Differentiable wrt M1 via jnp.interp / cumsum.

    Solar-type companion frequency peaks at long periods (logP~5); early-type is
    flatter and weighted to shorter periods — reproducing Moe's period trend.

    Reference: Moe & Di Stefano (2017) ApJS 230, 15, Table 13, Figure 37.
    """

    logP_min: float = 0.2
    logP_max: float = 8.0
    n_grid: int = 257

    def _density_grid(self, masses):
        logm1 = jnp.log10(masses)
        anchors = jnp.stack(
            [jnp.interp(logm1, _MASS_NODES_LOG, _COMPANION_FREQ[i]) for i in range(4)],
            axis=-1,
        )  # (n, 4)
        grid = jnp.linspace(self.logP_min, self.logP_max, self.n_grid)
        f = jax.vmap(lambda a: jnp.interp(grid, _LOGP_NODES, a))(anchors)  # (n, G)
        return grid, f

    def sample(self, key: PRNGKeyArray, masses) -> Float[Array, "n"]:
        """Sample one period [days] per primary mass [Msun] (shape n,)."""
        grid, f = self._density_grid(masses)
        dx = grid[1] - grid[0]
        cdf = jnp.cumsum(f, axis=1) * dx
        cdf = cdf - cdf[:, :1]
        cdf = cdf / cdf[:, -1:]
        u = jax.random.uniform(key, (masses.shape[0],))
        log_P = jax.vmap(lambda c, uu: jnp.interp(uu, c, grid))(cdf, u)
        return 10.0**log_P


class MoeJointOrbit(eqx.Module):
    """Faithful joint (P, q, e) sampler — the Moe & Di Stefano (2017) interrelation.

    Given a primary mass M1, samples the *correlated* orbital parameters:
        logP ~ MoePeriod(M1);  q ~ MoeDiStefano2017Full(M1, P);  e ~ MoeEccentricity(P, M1).
    The P–q–e coupling (short-P -> larger q / twins / circular; long-P -> small q
    approaching random IMF pairings) is the paper's central result ("Mind your Ps and Qs").

    Construct via `MoeJointOrbit.default()` for the standard components, or pass your
    own. The eccentricity sampler is duck-typed (any `.sample(key, periods, masses)`)
    so `imf` need not hard-import `binaries`.

    Reference: Moe & Di Stefano (2017) ApJS 230, 15 (full joint distribution).
    """

    period: MoePeriod
    massratio: MoeDiStefano2017Full
    eccentricity: eqx.Module

    @classmethod
    def default(cls) -> "MoeJointOrbit":
        """Standard Moe joint sampler (lazy-imports the Roche-capped MoeEccentricity)."""
        from ...binaries import MoeEccentricity  # lazy: imf -> binaries (no cycle)

        return cls(
            period=MoePeriod(),
            massratio=MoeDiStefano2017Full(),
            eccentricity=MoeEccentricity(),
        )

    def sample(self, key: PRNGKeyArray, masses):
        """Sample (P [days], q, e) jointly given primary masses [Msun] (shape n,)."""
        k_P, k_q, k_e = jax.random.split(key, 3)
        P = self.period.sample(k_P, masses)
        q = self.massratio.sample(k_q, masses, P)
        # eccentricity is declared as the generic eqx.Module base but is always a
        # MoeEccentricity (set in from_defaults); narrow for the type checker.
        e = cast("MoeEccentricity", self.eccentricity).sample(k_e, P, masses)
        return P, q, e
