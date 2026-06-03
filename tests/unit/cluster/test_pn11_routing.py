"""PN11 tail-layer routing in generate_fractal_ic_density (audit minor: latent crash).

tail_layer_from_env(env, model="pn11") yields a TailSubstructureLayer with mode="pn11".
generate_fractal_ic_density only routed "bm19"/"gravoturbulent" (extract s_t) and
"direct"/"cluster_type"/"D_mapping" (-> pn11_legacy), so a "pn11" layer fell through to
sample_positions_tail(mode="pn11") and raised ValueError("Invalid mode 'pn11'").

Fix: route "pn11" through the BM19 threshold sampler using the PN11 critical log-density
s_crit as the threshold (PN11Result carries s_crit, not s_t).
"""

import jax
import jax.numpy as jnp

import progenax  # noqa: F401  (enables float64)
from progenax.cluster.fdf_density import (
    FractalDensityLayer,
    generate_fractal_ic_density,
)
from progenax.cluster.gravoturbulent import GravoturbulentEnv, tail_layer_from_env
from progenax.imf import PowerLawIMF


def test_pn11_tail_layer_generates_ic_without_error():
    env = GravoturbulentEnv(Sigma=500, Mach=15, eta_survive=0.7)
    tail = tail_layer_from_env(env, model="pn11")
    assert tail.mode == "pn11"

    cluster = generate_fractal_ic_density(
        jax.random.PRNGKey(0),
        N_stars=100,
        M_total=100.0,
        R_half=1.0,
        imf_params=PowerLawIMF.kroupa(),
        layer=FractalDensityLayer(grid_size=16),
        tail=tail,
    )

    assert cluster.positions.shape == (100, 3)
    assert jnp.all(jnp.isfinite(cluster.positions))
