r"""gravoturb → differential-extinction adapter (CAREER Aim 3, extension A).

A young cluster is reddened *by its own natal gas*. gravoturb emits the residual-gas 3-D density grid
``TurbulentCloudIC.gas.rho_residual`` [M⊙/pc³]; :class:`GravoturbDustModel` turns it into a physical,
spatially-correlated **differential**-extinction screen and satisfies fluxax's duck-typed
``dust_model.column(positions) -> A_V`` slot (``fluxax.populations.population_to_catalog`` calls
``.column`` and ``.rv`` once and records the column on ``CatalogState.extinctions``).

**Geometry — star-embedded depth (spec decision A(a)).** Each star is reddened only by the residual-gas
column *between it and the observer*. Convention matches fluxax exactly: positions are cluster-frame
(x, y, z) in pc; **z (axis 2) is the line of sight**, the observer sits at small z, the near face is
grid index 0, and the column accumulates from the near face to the star. Front stars are light,
back-of-core stars are heavy.

**Amplitude — solar anchor here; metallicity keying added by** :func:`from_ic`/a later cycle
(spec decision A(b)): ``A_V`` per gas column scales with the dust-to-gas ratio δ_dg ∝ Z, so
``N_H/A_V ∝ 1/Z``.

The cumulative LOS column grid is precomputed at construction, so the whole object is a differentiable
pytree: gradients flow gas parameters → ``rho_residual`` → ``column_grid`` → ``A_V`` (the Aim-3 Fisher
backbone). ``.column`` itself is one trilinear sample + a linear scaling.
"""

import equinox as eqx
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
from jaxtyping import Array, Float

# --- Unit bridge + microphysics constants (provenance inline) ---
# 1 M⊙/pc² → g/cm²: M⊙ = 1.98892e33 g over pc² = (3.0857e18 cm)²; ≈ 2.089e-4. Mirrors the figure
# code's ``MSUN_PC2_TO_G_CM2`` (validation/feasibility_figure.py).
_MSUN_PC2_TO_G_CM2 = 1.98892e33 / (3.0857e18) ** 2
_M_H = 1.6726219e-24          # hydrogen mass [g] (CODATA 2018)
# MW/solar gas-to-extinction anchor N_H/A_V [cm⁻² mag⁻¹]: Bohlin, Savage & Drake 1978 (ApJ 224, 132)
# report N_H/E(B-V) = 5.8e21 → N_H/A_V = 5.8e21/3.1 ≈ 1.87e21; we round to 1.9e21 (spec default).
_N_H_PER_A_V_SOLAR = 1.9e21

# --- Metallicity-keyed dust-to-gas (Rémy-Ruyer et al. 2014, A&A 563, A31; arXiv:1312.3442) ---
# Table 1, broken power law, **X_CO,Z case**, with y=log₁₀(G/D), x=12+log(O/H):
#     x >  x_t :  y = A_HI  + ALPHA_H·(x_⊙ − x)      (near-solar linear regime, slope 1)
#     x <= x_t :  y = B_LO  + ALPHA_L·(x_⊙ − x)      (steep low-Z regime)
# a=2.21 gives the solar G/D = 10^2.21 = 162 (Zubko et al. 2004); x_⊙ = 8.69 (Asplund et al. 2009).
_RR_A_HI = 2.21       # high-Z intercept (= log₁₀ G/D_⊙); fixed in the fit
_RR_ALPHA_H = 1.00    # high-Z slope; fixed in the fit
_RR_B_LO = 0.96       # low-Z intercept (X_CO,Z)
_RR_ALPHA_L = 3.10    # low-Z slope (X_CO,Z)
_RR_X_T = 8.10        # break metallicity 12+log(O/H) (X_CO,Z)
_RR_X_SUN = 8.69      # solar 12+log(O/H) (Asplund et al. 2009)


def remy_ruyer_n_h_per_a_v(
    feh, *, n_h_per_a_v_solar: float = _N_H_PER_A_V_SOLAR
) -> Float[Array, ""]:
    r"""Metallicity-scaled gas-to-extinction ratio N_H/A_V [cm⁻² mag⁻¹] (spec decision A(b)).

    Since ``A_V ∝ (gas column)/(G/D)``, the gas-to-extinction ratio scales linearly with the
    gas-to-dust mass ratio: ``N_H/A_V(Z) = (N_H/A_V)_⊙ · (G/D)(Z)/(G/D)_⊙``. G/D(Z) is the
    Rémy-Ruyer et al. (2014) broken power law (X_CO,Z case); at solar metallicity the factor is 1.

    Metallicity is taken as ``[Fe/H]`` (``BirthEnvironment.metallicity``) and mapped to oxygen
    abundance by ``12+log(O/H) = x_⊙ + [Fe/H]`` — i.e. assuming solar abundance ratios ([O/Fe]=0.
    A first-order simplification: α-enhancement at low [Fe/H] would raise the true O/H, so this
    slightly *over*-thins the dust in the metal-poor regime). Differentiable; the broken power law
    is C⁰ (a slope kink at the break), which is the intended physics.
    """
    feh = jnp.asarray(feh, dtype=float)
    x_minus_sun = -feh                                  # x_⊙ − x = −[Fe/H]
    feh_t = _RR_X_T - _RR_X_SUN                          # transition in [Fe/H] (= −0.59)
    log_gd_hi = _RR_A_HI + _RR_ALPHA_H * x_minus_sun     # x > x_t branch
    log_gd_lo = _RR_B_LO + _RR_ALPHA_L * x_minus_sun     # x <= x_t branch
    log_gd = jnp.where(feh >= feh_t, log_gd_hi, log_gd_lo)
    return n_h_per_a_v_solar * 10.0 ** (log_gd - _RR_A_HI)


class GravoturbDustModel(eqx.Module):
    """Differential-extinction screen from a gravoturb residual-gas grid (duck-typed fluxax DustModel).

    Attributes
    ----------
    column_grid : Float[Array, "nx ny nz"]
        Cumulative LOS gas surface density Σ_gas from the near face to each **cell centre**
        [M⊙/pc²], precomputed along ``los_axis``.
    origin : Float[Array, "3"]
        Cluster→grid coordinate shift [pc] (= ``TurbulentCloudIC.geometry.origin``); grid
        coordinate of a cluster-frame position is ``position + origin`` (∈ [0, box)).
    box_size : Float[Array, ""]
        Physical box side [pc].
    a_v_per_sigma : Float[Array, ""]
        Conversion A_V per unit Σ_gas [mag / (M⊙/pc²)], = MSUN_PC2_TO_G_CM2 / (μ m_H) / (N_H/A_V).
        Carries the (metallicity-scaled) dust-to-gas amplitude.
    los_axis : int
        Line-of-sight axis (static). fluxax uses 2 (z).
    """

    column_grid: Float[Array, "nx ny nz"]
    origin: Float[Array, "3"]
    box_size: Float[Array, ""]
    a_v_per_sigma: Float[Array, ""]
    los_axis: int = eqx.field(static=True)

    @classmethod
    def from_grid(
        cls,
        rho_residual: Float[Array, "nx ny nz"],
        *,
        box_size,
        origin,
        feh=0.0,
        mu: float = 1.4,
        n_h_per_a_v_solar: float = _N_H_PER_A_V_SOLAR,
        los_axis: int = 2,
    ) -> "GravoturbDustModel":
        """Build from a raw residual-gas density grid [M⊙/pc³] and its grid geometry.

        ``mu`` is the mean molecular weight per hydrogen (1.4 for atomic H + He). ``feh`` is the
        birth metallicity [Fe/H]; the gas-to-extinction ratio is scaled from the MW/solar anchor by
        the Rémy-Ruyer+2014 dust-to-gas law (a low-Z birth thins the dust — spec decision A(b)).
        """
        n_h_per_a_v = remy_ruyer_n_h_per_a_v(feh, n_h_per_a_v_solar=n_h_per_a_v_solar)
        n_los = rho_residual.shape[los_axis]
        cell = box_size / n_los
        # Column to each cell CENTRE from the near face (index 0): cells fully in front + half of
        # the local cell → (cumsum - ½·local) · dz. Exact at cell centres, symmetric, differentiable.
        cum = (jnp.cumsum(rho_residual, axis=los_axis) - 0.5 * rho_residual) * cell
        a_v_per_sigma = _MSUN_PC2_TO_G_CM2 / (mu * _M_H) / n_h_per_a_v
        return cls(
            column_grid=cum,
            origin=jnp.asarray(origin, dtype=float),
            box_size=jnp.asarray(box_size, dtype=float),
            a_v_per_sigma=jnp.asarray(a_v_per_sigma, dtype=float),
            los_axis=los_axis,
        )

    @classmethod
    def from_ic(cls, ic, *, env=None, mu: float = 1.4, los_axis: int = 2) -> "GravoturbDustModel":
        """Build directly from a ``TurbulentCloudIC``, keying dust amplitude to its birth metallicity.

        Reads the residual-gas grid and geometry off ``ic``; ``env`` is a
        ``progenax.imf.BirthEnvironment`` whose ``metallicity`` ([Fe/H]) sets the Rémy-Ruyer
        dust-to-gas scaling. Passing the *same* ``BirthEnvironment`` that sets the environment-
        dependent IMF makes A a coherent joint environment-memory imprint (low-Z → top-heavier IMF
        AND thinner reddening). ``env=None`` defaults to solar ([Fe/H]=0). Raises if ``ic`` is a
        star-only build (no gas).
        """
        if ic.gas is None:
            raise ValueError(
                "GravoturbDustModel.from_ic requires a gas build; this IC is star-only "
                "(ic.gas is None / ledger.gas_included=False)."
            )
        feh = 0.0 if env is None else env.metallicity
        return cls.from_grid(
            ic.gas.rho_residual,
            box_size=ic.geometry.box_size,
            origin=ic.geometry.origin,
            feh=feh,
            mu=mu,
            los_axis=los_axis,
        )

    def column(self, positions: Float[Array, "N 3"]) -> Float[Array, "N"]:
        """Per-star V-band extinction A_V [mag] from the LOS residual-gas column (cluster-frame pc)."""
        n_axes = jnp.asarray(self.column_grid.shape)
        # cluster frame → grid coords; cell centres at (i+0.5)/n·box → coord = p/box·n − 0.5.
        coords = ((positions + self.origin) / self.box_size) * n_axes - 0.5
        # ``nearest`` clamps at faces (no periodic wrap for a cumulative column).
        sigma = map_coordinates(self.column_grid, coords.T, order=1, mode="nearest")
        return jnp.clip(sigma, 0.0) * self.a_v_per_sigma

    def rv(self, positions: Float[Array, "N 3"]) -> None:
        """No spatial R_V channel — callers fall back to the scalar ``ExtinctionModel.R_V``."""
        return None
