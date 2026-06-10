# progenax/src/progenax/profiles/limepy_tables.py
"""Tabulated LIMEPY DF moments (Phase 1.5 performance layer).

AnisoDensityTable tabulates the anisotropic lowered-isothermal density
rho_hat(W, p; g) -- the 86% hotspot of the multi-component coupled solve --
on a (sqrt(W), asinh(p)) grid, built in one batched call with the EXACT
quadrature (_aniso_density_scalar). The quadrature path remains available
everywhere as the oracle; this module must reproduce it to <= 1e-5 relative
(asserted in tests/unit/profiles/test_limepy_tables.py).

Coordinates: rho_hat ~ W^(g+3/2) as W->0 (sqrt stretches the power-law region);
rho_hat decays ~ 1/p at large p via T(beta) ~ sqrt(pi/4 beta) (asinh gives
log-like large-p resolution). Queries clamp to the box. Differentiable in the
build inputs (g) and the queries (W, p).

Interpolation scheme (measured, 2026-06): tensor-product 4-point cubic
Lagrange on the uniform (sqrt W, asinh p) grid, O(h^4). Bilinear interpolation
(the original design) converges only O(h^2) with a large constant here -- the
e^W growth and the W^(g+3/2) corner dominate the second derivative -- and
misses the 1e-5 budget even at 3072x512 nodes (5.7e-5 measured, ~39 s build).
Cubic meets it at the default 512x96 (6.1e-6 measured, ~2 s build). The build
maps over sqrt(W) rows with `jax.lax.map` (scan-based, differentiable): a full
double-vmap would materialize an (n_W, n_p, 256, 91) quadrature intermediate
and exhaust memory.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from progenax.profiles.limepy import _aniso_density_scalar


def _cubic_lagrange_weights(t: Float[Array, ""]) -> Float[Array, "4"]:
    """Lagrange basis on the uniform 4-point stencil {0, 1, 2, 3} at t in [0, 3].

    Exact for cubics -> O(h^4) interpolation error on smooth data; the stencil
    is shifted inward at the box edges (one-sided cubic, same order).
    Differentiable in t.
    """
    w0 = -(t - 1.0) * (t - 2.0) * (t - 3.0) / 6.0
    w1 = t * (t - 2.0) * (t - 3.0) / 2.0
    w2 = -t * (t - 1.0) * (t - 3.0) / 2.0
    w3 = t * (t - 1.0) * (t - 2.0) / 6.0
    return jnp.stack([w0, w1, w2, w3])


class AnisoDensityTable(eqx.Module):
    """Cubic-interpolation table of rho_hat(W, p) for fixed g (one table serves
    all components of a coupled solve: component j queries
    (rescale_j*psi, xi/ra_j))."""

    s_nodes: Float[Array, "n_W"]   # sqrt(W) nodes, uniform on [0, sqrt(W_max)]
    q_nodes: Float[Array, "n_p"]   # asinh(p) nodes, uniform on [0, asinh(p_max)]
    values: Float[Array, "n_W n_p"]

    @classmethod
    def build(cls, W_max, p_max, g, n_W: int = 512, n_p: int = 96):
        """Tabulate the exact quadrature on the (sqrt W, asinh p) grid.

        W_max gets a x1.001 safety factor so in-domain queries never clamp.
        Differentiable in (g) and the queries (W, p): node values carry d/dg
        through `jax.lax.map` (scan-based, unlike a double-vmap it never
        materializes the full (n_W, n_p, n_quad) quadrature intermediate).
        """
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        q = jnp.linspace(0.0, jnp.arcsinh(jnp.asarray(p_max)), n_p)
        W = s**2
        p = jnp.sinh(q)
        row = lambda w: jax.vmap(lambda pp: _aniso_density_scalar(w, pp, g))(p)
        vals = jax.lax.map(row, W)
        return cls(s_nodes=s, q_nodes=q, values=vals)

    def evaluate(self, W, p):
        """Bicubic (4-point Lagrange) interp at (W, p), clamped to the table
        box. W <= 0 -> 0 (no stars above the escape energy).

        Cubic Lagrange can overshoot by O(1e-15 * central) near the W->0
        power-law corner; the result is clamped to preserve the rho_hat >= 0
        contract (gradient is zero only on that measure-tiny set).
        Gradients are exactly 0 in the clamp region (edge saturation) --
        relevant if optimizing ra_hat_j near the p edge.
        """
        # 1e-12 floor mirrors _aniso_density_scalar's oracle guard: sqrt(0)
        # has a NaN cotangent under jax.grad and the final where() masks only
        # the primal. s = 1e-6 is deep inside the first cell.
        s = jnp.sqrt(jnp.maximum(W, 1e-12))
        q = jnp.arcsinh(jnp.maximum(p, 0.0))
        s = jnp.clip(s, self.s_nodes[0], self.s_nodes[-1])
        q = jnp.clip(q, self.q_nodes[0], self.q_nodes[-1])
        # Cell widths: the grid is uniform by construction (linspace in build).
        hs = self.s_nodes[1] - self.s_nodes[0]
        hq = self.q_nodes[1] - self.q_nodes[0]
        i = jnp.clip(jnp.searchsorted(self.s_nodes, s) - 1, 0, self.s_nodes.size - 2)
        j = jnp.clip(jnp.searchsorted(self.q_nodes, q) - 1, 0, self.q_nodes.size - 2)
        # Center the 4-point stencil on the query cell; shift inward at edges.
        i0 = jnp.clip(i - 1, 0, self.s_nodes.size - 4)
        j0 = jnp.clip(j - 1, 0, self.q_nodes.size - 4)
        ws = _cubic_lagrange_weights((s - self.s_nodes[i0]) / hs)
        wq = _cubic_lagrange_weights((q - self.q_nodes[j0]) / hq)
        sub = jax.lax.dynamic_slice(self.values, (i0, j0), (4, 4))
        v = ws @ sub @ wq
        v = jnp.maximum(v, 0.0)  # clamp cubic overshoot (rho_hat >= 0 contract)
        return jnp.where(W > 0.0, v, 0.0)


__all__ = ["AnisoDensityTable"]
