# progenax/src/progenax/cluster/gravoturbulent.py
"""Gravoturbulent environment and tail selection.

This module provides the physics-based interface for deriving stellar
substructure from gravoturbulent star formation theory.

Main Entry Points
-----------------
- GravoturbulentEnv: Birth cloud environment parameters
- TailSelectionConfig: Tail selection algorithm configuration
- tail_layer_from_env(): Create TailSubstructureLayer from physics
- env_from_preset(): Get preset environments (Taurus, Orion, etc.)

Physics Models
--------------
- BM19 (default): Burkhart & Mocz 2019 framework
- PN11 (alternative): Padoan-Nordlund 2011 framework

References
----------
- Burkhart & Mocz (2019) ApJ 879, 129 (BM19)
- Padoan & Nordlund (2011) ApJ 730, 40 (PN11)
- Federrath & Klessen (2012) ApJ 761, 156
- Heyer & Dame (2015) ARA&A 53, 583
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from progenax.cluster.fdf_density import TailSubstructureLayer
    from progenax.gravoturb import BM19Result, PN11Result


# =============================================================================
# Environment Dataclass
# =============================================================================


@dataclass(frozen=True)
class GravoturbulentEnv:
    """Birth cloud environment for gravoturbulent f_sub derivation.

    This encapsulates the ISM properties that determine what fraction
    of gas mass ends up in gravitationally collapsing dense regions.

    BM19 Parameters (Primary)
    -------------------------
    Sigma : float
        Cloud surface density [M☉/pc²]. Primary driver of physics.
        Typical ranges: 50 (diffuse GMC) to 3000+ (starburst).
    Mach : float
        Turbulent Mach number. Sets PDF width σ_s.
        Typical ranges: 6 (Taurus-like) to 30+ (starburst).
    eta_survive : float
        Feedback survival fraction [0-1]. POORLY CONSTRAINED.
        Fraction of f_tail that survives as stellar substructure.
    b : float
        Turbulence driving parameter (default 0.4).
        0.33 = solenoidal, 1.0 = compressive.
    alpha : float
        BM19 powerlaw slope (default 2.0).
        Controls PDF tail steepness and transition density s_t.

    PN11 Parameters (Alternative Model)
    ------------------------------------
    theta : float
        Turbulence integral-scale factor (default 0.35; PN11 Eq. 8).
        Only used when model="pn11".

    Examples
    --------
    >>> # Orion-like GMC
    >>> env = GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6)
    >>>
    >>> # YMC-forming clump
    >>> env = GravoturbulentEnv(Sigma=1000, Mach=20, eta_survive=0.85, alpha=2.5)
    """

    # Primary BM19 parameters
    Sigma: float
    Mach: float
    eta_survive: float
    b: float = 0.4
    alpha: float = 2.0

    # PN11 parameters (alternative model)
    theta: float = 0.35


# =============================================================================
# Tail Selection Configuration
# =============================================================================


@dataclass(frozen=True)
class TailSelectionConfig:
    """Configuration for FDF tail selection method.

    Controls how stars are assigned to the "dense tail" vs "smooth"
    components during gravoturbulent IC generation.

    Attributes
    ----------
    mode : str
        Selection algorithm:
        - "bm19": s > s_t threshold (DEFAULT, physics-consistent)
        - "pn11_legacy": Local overdensity ranking (for comparison)
    kappa : float
        Sigmoid sharpness for BM19 mode (default 10.0).
        Higher = sharper threshold.
    dense_tail_mass_frac : float
        Fixed mass fraction for legacy mode (default 0.10).

    Examples
    --------
    >>> # Default BM19 mode (recommended)
    >>> config = TailSelectionConfig()
    >>>
    >>> # Sharper BM19 threshold
    >>> config = TailSelectionConfig(mode="bm19", kappa=20.0)
    """

    mode: str = "bm19"
    kappa: float = 10.0
    dense_tail_mass_frac: float = 0.10

    def __post_init__(self):
        """Validate configuration."""
        valid_modes = ("bm19", "pn11_legacy")
        if self.mode not in valid_modes:
            raise ValueError(
                f"Invalid mode '{self.mode}'. Must be one of {valid_modes}."
            )
        if self.kappa <= 0:
            raise ValueError(f"kappa must be positive, got {self.kappa}")
        if not 0 < self.dense_tail_mass_frac < 1:
            raise ValueError(
                f"dense_tail_mass_frac must be in (0, 1), got {self.dense_tail_mass_frac}"
            )


# =============================================================================
# Presets (Common Environments)
# =============================================================================


GRAVOTURBULENT_PRESETS: dict[str, GravoturbulentEnv] = {
    "taurus": GravoturbulentEnv(Sigma=40, Mach=6, eta_survive=0.4),
    "orion": GravoturbulentEnv(Sigma=150, Mach=12, eta_survive=0.6),
    "typical_gmc": GravoturbulentEnv(Sigma=100, Mach=10, eta_survive=0.5),
    "dense_gmc": GravoturbulentEnv(Sigma=300, Mach=15, eta_survive=0.65),
    "ymc_precursor": GravoturbulentEnv(Sigma=1000, Mach=20, eta_survive=0.85),
    "starburst": GravoturbulentEnv(Sigma=3000, Mach=30, eta_survive=0.9),
}


def env_from_preset(preset: str) -> GravoturbulentEnv:
    """Get gravoturbulent environment from named preset.

    Available presets:
        - "taurus": Diffuse cloud (Σ=40, M=6, η=0.4) → f_sub~0.008
        - "orion": Typical GMC (Σ=150, M=12, η=0.6) → f_sub~0.06
        - "typical_gmc": Moderate GMC (Σ=100, M=10, η=0.5)
        - "dense_gmc": Dense GMC (Σ=300, M=15, η=0.65)
        - "ymc_precursor": YMC-forming clump (Σ=1000, M=20, η=0.85)
        - "starburst": Extreme environment (Σ=3000, M=30, η=0.9)

    Parameters
    ----------
    preset : str
        Preset name (case-insensitive).

    Returns
    -------
    GravoturbulentEnv
        Pre-configured environment.

    Raises
    ------
    KeyError
        If preset name not found.

    Examples
    --------
    >>> env = env_from_preset("orion")
    >>> env.Sigma
    150
    """
    key = preset.lower()
    if key not in GRAVOTURBULENT_PRESETS:
        available = ", ".join(GRAVOTURBULENT_PRESETS.keys())
        raise KeyError(f"Unknown preset '{preset}'. Available: {available}")
    return GRAVOTURBULENT_PRESETS[key]


# =============================================================================
# Physics-Based Tail Layer Creation
# =============================================================================


def tail_layer_from_env(
    env: GravoturbulentEnv,
    model: Literal["bm19", "pn11"] = "bm19",
) -> "TailSubstructureLayer":
    """Create TailSubstructureLayer from gravoturbulent environment.

    This is the RECOMMENDED way to create a TailSubstructureLayer when
    you have physical knowledge of the birth cloud environment.

    Parameters
    ----------
    env : GravoturbulentEnv
        Cloud environment parameters (Σ, M, η_survive, etc.)
    model : {"bm19", "pn11"}, default "bm19"
        Physics framework to use:
        - "bm19": Burkhart & Mocz 2019 (RECOMMENDED)
          Uses env.alpha for transition density.
        - "pn11": Padoan-Nordlund 2011 (alternative)
          Uses env.theta (turbulence integral-scale factor).

    Returns
    -------
    TailSubstructureLayer
        With f_sub derived from physics and full provenance.

    Examples
    --------
    >>> # BM19 (default)
    >>> env = GravoturbulentEnv(Sigma=1000, Mach=20, eta_survive=0.85)
    >>> tail = tail_layer_from_env(env)
    >>> print(f"f_sub = {tail.f_sub:.3f}")

    >>> # PN11 (when magnetic support matters)
    >>> tail = tail_layer_from_env(env, model="pn11")
    """
    from progenax.cluster.fdf_density import TailSubstructureLayer

    if model == "bm19":
        from progenax.gravoturb import bm19_pipeline

        result = bm19_pipeline(
            mach=env.Mach,
            b=env.b,
            alpha=env.alpha,
            eta_survive=env.eta_survive,
        )
        return TailSubstructureLayer(
            f_sub=float(result.f_sub),
            mode="bm19",
            env=env,
            result=result,
        )
    elif model == "pn11":
        from progenax.gravoturb import pn11_pipeline

        result = pn11_pipeline(
            mach=env.Mach,
            Sigma=env.Sigma,
            eta_survive=env.eta_survive,
            b=env.b,
            theta=env.theta,
        )
        return TailSubstructureLayer(
            f_sub=float(result.f_sub),
            mode="pn11",
            env=env,
            result=result,
        )
    else:
        raise ValueError(f"Invalid model '{model}'. Use 'bm19' or 'pn11'.")


__all__ = [
    "GravoturbulentEnv",
    "TailSelectionConfig",
    "GRAVOTURBULENT_PRESETS",
    "env_from_preset",
    "tail_layer_from_env",
]
