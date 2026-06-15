r"""B12 -- The binary-inflated dynamical mass: a "confidently wrong" virial mass.

Unresolved binaries inflate a stellar system's line-of-sight velocity dispersion,
biasing the **virial / dynamical mass high**. In low-dispersion systems
(ultra-faint dwarfs, low-mass GCs) this is a *large fractional* effect that has
driven real debates about M/L ratios and dark-matter content. This demo shows:

  (1) the bias is real -- ``sigma_obs > sigma_true`` and ``M_naive`` is inflated;
  (2) a **dispersion-only** analysis cannot remove it -- the ``(sigma_true, f_b)``
      problem is rank-1 degenerate (one number can't separate two parameters);
  (3) a **differentiable joint recovery** from the non-Gaussian *wings* of the
      velocity distribution returns an unbiased dynamical mass, with a
      Fisher/CRLB forecast vs sample size ``N`` and RV precision ``eps``.

Forward model: one isotropic single-population cluster gives each star a LOS COM
velocity ``v_COM ~ N(0, sigma_true^2)``. A fraction ``f_b`` are unresolved Moe &
Di Stefano (2017) binaries whose observed velocity is the ZAMS-flux-weighted SB2
blend ``v_obs = v_COM + Delta`` (``Delta`` from the sigma-independent kernel
K_orb). Per-star RV noise ``N(0, eps^2)`` is added to every star.

This is the kinematic companion to the B4 unresolved-binary mass-function demo
(B4 measures f_b photometrically; B12 measures the dynamical mass kinematically).

Gates (CLI exits 0 iff all pass):
  1. Bias       -- sigma_obs > sigma_true; M_naive/M_true > 1.10 at f_b=0.5.
  2. Degeneracy -- dispersion-only Fisher is rank-1 (near-singular).
  3. Recovery   -- joint (sigma_true, f_b) MLE within 3sigma; M_dyn unbiased;
                   full-distribution Fisher well-conditioned.
  4. eps-floor  -- bias-removal degrades monotonically as RV precision worsens.
  5. Null       -- f_b=0 gives no bias and recovered f_b ~ 0.
  6. AD-vs-FD   -- the mixture Jacobian d mu / d z matches finite differences.

Demo only: scripts/ + docs/ (no src/progenax change). Units: ALL velocities km/s.

References:
  Moe & Di Stefano (2017) ApJS 230, 15  -- the P-q-e binary statistics.
  Tout et al. (1996) MNRAS 281, 257     -- the ZAMS mass-luminosity relation.
"""
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np

import progenax  # noqa: F401  -- enables float64 at import

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _demo_binaries import sample_blend_velocities  # noqa: E402
from _plotstyle import apply_pub_style  # noqa: E402

apply_pub_style()

OUTPUT_DIR = "validation/plots"

# --- configuration ---------------------------------------------------------- #
# UFD-like regime: a low-dispersion, metal-poor system where unresolved binaries
# are a large fractional contaminant of the velocity dispersion.
SIGMA_TRUE = 5.0          # true cluster LOS velocity dispersion [km/s]
Z_MET = 1e-3              # metallicity for the Tout ZAMS photometry (metal-poor)
F_B_TRUE = 0.5            # unresolved binary fraction
N_STARS = 1500            # RV stars in the mock survey
EPS_KMS = 1.0             # per-star RV measurement precision [km/s]
R_H_PC = 30.0             # half-mass radius [pc] (sets the virial-mass scale)

# Binned-likelihood velocity grid: wide enough (~+/-7 sigma_obs) to hold the
# non-Gaussian binary wings that break the degeneracy.
V_EDGES = np.linspace(-40.0, 40.0, 81)   # 80 bins of 1 km/s
N_POOL = 200_000          # K_orb template pool (low template noise)
KORB_GRID_MAX = 150.0     # K_orb grid half-width [km/s] (must span the wings)
KORB_N_GRID = 601

# Parameter boxes for the bounded MLE (logit/expit reparametrization).
SIGMA_BOX = (0.5, 30.0)
FB_BOX = (0.0, 0.95)
N_ADAM = 600
ADAM_LR = 3e-2
SEED = 0


def build_mock_vlos(key, f_b=F_B_TRUE, sigma_true=SIGMA_TRUE, eps=EPS_KMS,
                    n_stars=N_STARS, Z=Z_MET):
    r"""Mock observed LOS velocities of a binary-contaminated cluster [km/s].

    ``n_b = round(f_b * n_stars)`` stars are unresolved binaries: their observed
    velocity is the cluster COM draw plus the flux-weighted blend ``Delta`` (drawn
    fresh from the Moe+ZAMS machinery, NOT from the histogram kernel). Every star
    then gets independent ``N(0, eps^2)`` RV measurement noise.

    Returns ``v_obs`` of shape ``(n_stars,)``.
    """
    k_com, k_binary, k_noise = jax.random.split(key, 3)
    v_com = sigma_true * jax.random.normal(k_com, (n_stars,))

    n_b = int(round(f_b * n_stars))
    if n_b > 0:
        delta = sample_blend_velocities(k_binary, n_b, Z=Z)
        delta = jnp.concatenate([jnp.asarray(delta), jnp.zeros(n_stars - n_b)])
    else:
        delta = jnp.zeros(n_stars)

    noise = eps * jax.random.normal(k_noise, (n_stars,))
    return v_com + delta + noise
