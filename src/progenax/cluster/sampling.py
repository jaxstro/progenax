# progenax/src/progenax/cluster/sampling.py
"""JIT-compiled sampling core for MultiComponentCluster (Engines A and B).

Extracted from cluster/multicomponent.py (Task 4, file-length discipline): the
model module owns construction + theory oracles; this module owns the one
jitted per-star sampling kernel. `MultiComponentCluster.sample_cluster` is the
public entry point -- it calls `_sample_cluster_arrays` and assembles the
ICResult outside the JIT boundary.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from progenax.builders import compute_stellar_radii
from progenax.cluster.eddington_engine import engine_b_star_velocities
from progenax.kinematics._speed_kernels import _N_C, _sample_costheta_given_u
from progenax.profiles.limepy_multimass import _isotropic_dirs
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable, SpeedCDFTable


@eqx.filter_jit
def _sample_cluster_arrays(model, key: PRNGKeyArray, n_stars: int, G: float):
    """JIT-compiled sampler core -> (pos, vel, m_i, stellar_radii, component_id).

    `model` enters as a PyTree (engine and is_aniso are static fields, so the
    Engine A/B and iso/aniso branches are resolved at trace time); n_stars and
    G are static arguments (one compilation per distinct value). Key splits
    match the original eager path; the isotropic speed VALUES come from the
    SpeedCDFTable inverse (one batched E_gamma build per call, distributionally
    identical to the exact per-star sampler -- statistical oracles in
    test_limepy_tables.py).
    """
    k_assign, k_pos, k_pdir, k_speed, k_vdir = jax.random.split(key, 5)

    c = jax.random.categorical(k_assign, jnp.log(model.N_frac_j + 1e-30),
                               shape=(n_stars,))
    m_i = model.m_j[c]
    M_total = jnp.sum(m_i)  # the cluster mass IS the sum of its stars

    # Positions: per-star inverse-CDF on its component's mass CDF + isotropic dirs
    # (shared by Engine A and Engine B -- N_frac_j and _cdf_j are engine-agnostic).
    u = jax.random.uniform(k_pos, (n_stars,))
    radii = jax.vmap(lambda uu, cc: jnp.interp(uu, model._cdf_j[cc], model._r_grid))(u, c)
    pos = radii[:, None] * _isotropic_dirs(k_pdir, n_stars)
    speed_keys = jax.random.split(k_speed, n_stars)

    if model.engine == "B":
        # Engine B (static Python branch -- compiled per engine, zero runtime
        # dispatch): per-star Psi on the shared sampler grid, dimensionless
        # speed from the star's component Eddington row, physical scale
        # sqrt(G M_sampled / (4 pi mu)) from the ACTUAL sampled mass, OM
        # stretched directions with per-star r_a_j[c] (inf -> isotropic).
        vel = engine_b_star_velocities(model.engine_b, speed_keys, k_vdir, pos,
                                       radii, model._r_grid, c, G, M_total)
        return pos, vel, m_i, compute_stellar_radii(m_i), c

    # Engine A: per-star rescaled potential W_j(r) = psi(r)/w_j^2 and velocity
    # scale s_i = s w_j[c] (shared by the iso/aniso paths below).
    rescale_i = model.rescale_j[c]
    s = jnp.sqrt(G * M_total / (9.0 * model.r_c * model.mu_tot))
    s_i = s * model.w_j[c]
    W_i = rescale_i * jnp.maximum(
        jnp.interp(radii / model.r_c, model.xi_grid, model.psi_grid,
                   left=model.W0, right=0.0), 0.0)

    if model.is_aniso:
        # Anisotropic: the speed u comes from ONE precomputed 3-D speed-MARGINAL
        # CDF table (Task 6) replacing the per-star 256-point quadrature of
        # _sample_speed_angle; the angular conditional cos(theta)|u stays EXACT
        # (_sample_costheta_given_u -- the same code _sample_speed_angle calls).
        # The box covers every star exactly: W_i = rescale_i*psi <=
        # max(rescale)*W0 and p_i = (r/r_c)/ra_hat_j[c] <= (r_t/r_c)/min(ra_hat)
        # (radii never exceed r_t by construction of the mass-CDF draw); the
        # 1e-3 p floor guards the degenerate all-isotropic corner like
        # _solver_table. Built per call, differentiable in (W0, g, w_j, ra_hat_j).
        p_i = (radii / model.r_c) / model.ra_hat_j[c]
        p_box = jnp.maximum((model.r_t / model.r_c) / jnp.min(model.ra_hat_j), 1e-3)
        table = AnisoSpeedCDFTable.build(jnp.max(model.rescale_j) * model.W0,
                                         p_box, model.g)
        ku_kc = jax.vmap(jax.random.split)(speed_keys)
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(ku_kc[:, 0])
        u_sp = jax.vmap(table.inverse)(W_i, p_i, unif)
        cos_t = jax.vmap(
            lambda kk, uu, pp: _sample_costheta_given_u(kk, uu, pp, _N_C)
        )(ku_kc[:, 1], u_sp, p_i)
        u_r = u_sp * cos_t
        u_t = u_sp * jnp.sqrt(jnp.maximum(1.0 - cos_t**2, 0.0))
        v_r, v_t = s_i * u_r, s_i * u_t
        # v_r along r_hat (signed); v_t in a random azimuth perpendicular to r_hat.
        r_hat = pos / (radii[:, None] + 1e-30)
        rand = jax.random.normal(k_vdir, (n_stars, 3))
        rand = rand - jnp.sum(rand * r_hat, axis=1, keepdims=True) * r_hat
        t_hat = rand / (jnp.linalg.norm(rand, axis=1, keepdims=True) + 1e-30)
        vel = v_r[:, None] * r_hat + v_t[:, None] * t_hat
    else:
        # Isotropic: ONE precomputed speed-CDF table (Task 5) replaces the
        # per-star 256-point E_gamma quadrature: a star's draw is a sqrt(W)-row
        # lookup + inverse-CDF interp. The box W_max = max(rescale)*W0 covers
        # every star exactly (W_i = rescale_i * psi <= max(rescale) * W0); the
        # build is traced per call (65k E_gamma points, amortized over n_stars)
        # and stays differentiable in (W0, g, w_j) -- the model is unchanged.
        table = SpeedCDFTable.build(jnp.max(model.rescale_j) * model.W0, model.g)
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(speed_keys)
        u_speed = jax.vmap(table.inverse)(W_i, unif)
        vel = (s_i * u_speed)[:, None] * _isotropic_dirs(k_vdir, n_stars)

    return pos, vel, m_i, compute_stellar_radii(m_i), c


__all__ = ["_sample_cluster_arrays"]
