# progenax/src/progenax/cluster/sampling.py
"""Cluster sampling and first-order population-expectation estimation.

Construction and equilibrium oracles live in ``multicomponent``. This module
owns categorical assignment, conditional phase-space draws, and score correction.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.numerics import gauss_legendre_nodes
from jaxtyping import PRNGKeyArray

from progenax.builders import ICResult
from progenax.cluster._radial_cdf import (
    component_radial_quantile,
    inverse_cdf_second_moments,
    potential_interp,
    radial_second_moments,
)
from progenax.cluster.eddington_engine import engine_b_star_velocities
from progenax.kinematics._speed_kernels import _N_C, _sample_costheta_given_u
from progenax.kinematics.eddington import _df_speed_second_moment
from progenax.profiles.limepy_multimass import _isotropic_dirs
from progenax.profiles.limepy_tables import AnisoSpeedCDFTable, SpeedCDFTable
from progenax.stellar import zams_radius


@eqx.filter_jit
def _sample_cluster_arrays(model, key: PRNGKeyArray, n_stars: int, G: float):
    """Return positions, velocities, masses, stellar radii, and component labels.

    Engine and anisotropy branches are static under JIT. The key sequence is
    shared with the conditional sampler used for discrete-mixture validation.
    """
    k_assign = jax.random.split(key, 5)[0]
    c = jax.random.categorical(
        k_assign, jnp.log(model.N_frac_j + 1e-30), shape=(n_stars,)
    )
    return _sample_cluster_arrays_given_components(model, key, c, G)


@eqx.filter_jit
def _sample_cluster_arrays_given_components(
    model, key, c, G, *, radial_probabilities=None
):
    """Conditional sampler with the public sampler's unchanged key splitting.

    Optional radial probabilities expose the existing inverse for stratified
    validation; ordinary cluster draws use the original uniform key sequence.
    """
    n_stars = c.shape[0]
    _, k_pos, k_pdir, k_speed, k_vdir = jax.random.split(key, 5)
    m_i = model.m_j[c]
    M_total = jnp.sum(m_i)  # the cluster mass IS the sum of its stars

    # Engine A inverts integrated linear radial density; Engine B retains its CDF.
    u = (
        jax.random.uniform(k_pos, (n_stars,))
        if radial_probabilities is None
        else radial_probabilities
    )
    if model.engine == "A":
        radii = component_radial_quantile(
            u, c, model._r_grid, model.engine_a.radial_pdf_j, model._cdf_j
        )
    else:
        radii = jax.vmap(
            lambda uu, cc: jnp.interp(uu, model._cdf_j[cc], model._r_grid)
        )(u, c)
    pos = radii[:, None] * _isotropic_dirs(k_pdir, n_stars)
    speed_keys = jax.random.split(k_speed, n_stars)

    if model.engine == "B":
        # Engine B (static Python branch -- compiled per engine, zero runtime
        # dispatch): per-star Psi on the shared sampler grid, dimensionless
        # speed from the star's component Eddington row, physical scale
        # sqrt(G M_sampled / (4 pi mu)) from the ACTUAL sampled mass, OM
        # stretched directions with per-star r_a_j[c] (inf -> isotropic).
        vel = engine_b_star_velocities(
            model.engine_b, speed_keys, k_vdir, pos, radii, model._r_grid, c, G, M_total
        )
        return pos, vel, m_i, zams_radius(m_i), c

    # Engine A: per-star rescaled potential W_j(r) = psi(r)/w_j^2 and velocity
    # scale s_i = s w_j[c] (shared by the iso/aniso paths below).
    rescale_i = model.rescale_j[c]
    s = jnp.sqrt(G * M_total / (9.0 * model.r_c * model.mu_tot))
    s_i = s * model.w_j[c]
    W_i = rescale_i * jnp.maximum(
        potential_interp(
            radii / model.r_c,
            model.xi_grid,
            model.engine_a.psi_raw,
            left=model.W0,
            right=0.0,
        ),
        0.0,
    )

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
        table = AnisoSpeedCDFTable.build(
            jnp.max(model.rescale_j) * model.W0, p_box, model.g
        )
        ku_kc = jax.vmap(jax.random.split)(speed_keys)
        unif = jax.vmap(lambda kk: jax.random.uniform(kk))(ku_kc[:, 0])
        u_sp = jax.vmap(table.inverse)(W_i, p_i, unif)
        cos_t = jax.vmap(lambda kk, uu, pp: _sample_costheta_given_u(kk, uu, pp, _N_C))(
            ku_kc[:, 1], u_sp, p_i
        )
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

    return pos, vel, m_i, zams_radius(m_i), c


def _expectation_with_score(values, log_probabilities):
    """Sample mean with a first-order categorical score and leave-one-out baseline.

    Each row must be an independent realization. The stopped baseline excludes
    that row, so its product with the row's probability score has zero expectation.
    The zero-valued correction preserves the ordinary Monte Carlo estimate.
    Its AD rule is intended for first derivatives only.
    """
    baseline = (jnp.sum(values) - values) / (values.shape[0] - 1)
    advantage = jax.lax.stop_gradient(values - baseline)
    score = log_probabilities - jax.lax.stop_gradient(log_probabilities)
    return jnp.mean(values + advantage * score)


@eqx.filter_jit
def _estimate_expectation(model, observable, key, n_stars, G, n_realizations):
    """Apply the scalar observable to independent categorical cluster draws."""
    log_pi = jax.nn.log_softmax(jnp.log(model.N_frac_j + 1e-30))

    def draw(draw_key):
        pos, vel, masses, radii, labels = _sample_cluster_arrays(
            model, draw_key, n_stars, G
        )
        value = jnp.asarray(
            observable(
                ICResult(
                    positions=pos,
                    velocities=vel,
                    masses=masses,
                    stellar_radii=radii,
                    component_id=labels,
                )
            )
        )
        if value.ndim != 0:
            raise ValueError("observable must return a scalar")
        return value, jnp.sum(log_pi[labels])

    values, log_probabilities = jax.vmap(draw)(jax.random.split(key, n_realizations))
    return _expectation_with_score(values, log_probabilities)


_RADIAL_MOMENT_NODES, _RADIAL_MOMENT_WEIGHTS = gauss_legendre_nodes(4)


def _engine_b_population_speed_moment(model, n_stars, G):
    """Integrate conditional DF moments over the sampler's radial distribution."""
    state = model.engine_b
    r = model._r_grid
    quadrature_radii = (
        (r[1:] + r[:-1])[:, None] / 2
        + (r[1:] - r[:-1])[:, None] / 2 * _RADIAL_MOMENT_NODES
    ).ravel()
    radii = jnp.concatenate((quadrature_radii, r[:1], r[-1:]))
    potential = jnp.interp(radii, r, state.Psi_grid)
    components = []
    for j in range(model.m_j.size):

        @jax.checkpoint
        def local(psi):
            return _df_speed_second_moment(psi, state.E_grid, state.f_j_grid[j])

        second = jax.lax.map(local, potential, batch_size=32)
        finite = jnp.isfinite(state.r_a_j[j])
        radius = jnp.where(finite, state.r_a_j[j], 1.0)
        angular = 1 / 3 + 2 / 3 * jnp.where(
            finite, 1 / (1 + (radii / radius) ** 2), 1.0
        )
        moment = second * angular
        interval_mean = moment[:-2].reshape((-1, 4)) @ _RADIAL_MOMENT_WEIGHTS / 2
        cdf = model._cdf_j[j]
        components.append(
            jnp.sum(jnp.diff(cdf) * interval_mean)
            + cdf[0] * moment[-2]
            + (1 - cdf[-1]) * moment[-1]
        )
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    total_mass_given_component = model.m_j + (n_stars - 1) * (pi @ model.m_j)
    return (
        G
        * jnp.sum(pi * jnp.stack(components) * total_mass_given_component)
        / (4 * jnp.pi * state.mu)
    )


def _estimate_population_moments(
    model, key, n_stars, G, n_realizations, draws_per_component
):
    """Analytic mass/radius moments and conditionally averaged speed moment."""
    n_components = model.m_j.shape[0]
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    mean_mass = pi @ model.m_j
    component_radius_squared = (
        radial_second_moments(model._r_grid, model.engine_a.radial_pdf_j)
        if model.engine == "A"
        else inverse_cdf_second_moments(model._r_grid, model._cdf_j)
    )
    mean_radius_squared = pi @ component_radius_squared
    if model.engine == "B":
        return {
            "mean_mass": mean_mass,
            "mean_radius_squared": mean_radius_squared,
            "mean_speed_squared": _engine_b_population_speed_moment(model, n_stars, G),
        }
    labels = jnp.repeat(jnp.arange(n_components), draws_per_component)
    integration_mass = draws_per_component * jnp.sum(model.m_j)
    correction = (model.m_j + (n_stars - 1) * mean_mass) / integration_mass

    def draw(draw_key):
        _, velocities, _, _, _ = _sample_cluster_arrays_given_components(
            model, draw_key, labels, G
        )
        component_speed_squared = jnp.mean(
            jnp.sum(velocities**2, axis=1).reshape(n_components, draws_per_component),
            axis=1,
        )
        return pi @ (component_speed_squared * correction)

    speed_squared = jax.lax.map(draw, jax.random.split(key, n_realizations))
    return {
        "mean_mass": mean_mass,
        "mean_radius_squared": mean_radius_squared,
        "mean_speed_squared": jnp.mean(speed_squared),
    }


def _estimate_mass_squared_and_kinetic(
    model, key, n_stars, G, n_realizations, draws_per_component
):
    """Conditionally average two finite-population mass/kinetic observables.

    Integration draws have equal component occupancy; they are not a physical
    catalogue. The exact label weights retain the correlation between the mass
    of a star and the catalogue mass that sets its velocity normalization.
    """
    pi = jax.nn.softmax(jnp.log(model.N_frac_j + 1e-30))
    mean_mass = pi @ model.m_j
    mass_squared = (
        n_stars * (pi @ model.m_j**2) + n_stars * (n_stars - 1) * mean_mass**2
    )
    labels = jnp.repeat(jnp.arange(model.m_j.size), draws_per_component)
    integration_mass = draws_per_component * jnp.sum(model.m_j)
    correction = (model.m_j + (n_stars - 1) * mean_mass) / integration_mass

    def draw(draw_key):
        _, velocities, _, _, _ = _sample_cluster_arrays_given_components(
            model, draw_key, labels, G
        )
        speed = jnp.mean(
            jnp.sum(velocities**2, axis=1).reshape(model.m_j.size, -1), axis=1
        )
        return jnp.sum(pi * model.m_j * speed * correction)

    kinetic = jnp.mean(jax.lax.map(draw, jax.random.split(key, n_realizations)))
    return {
        "total_mass_squared": mass_squared,
        "twice_kinetic_energy_per_star": kinetic,
    }


__all__ = ["_sample_cluster_arrays"]
