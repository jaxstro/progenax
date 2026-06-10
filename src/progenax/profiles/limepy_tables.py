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

from progenax.profiles.limepy import (
    _angle_integral_T,
    _aniso_density_scalar,
    lowered_exponential,
)


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


class SpeedCDFTable(eqx.Module):
    """Isotropic speed inverse-CDF table u(W, unif) for fixed g (Tranche B).

    Replaces the per-star 256-point E_gamma quadrature of
    `_sample_unit_speed` with ONE precomputed table in NORMALIZED speed
    coordinates x = u / sqrt(2W) in [0, 1]: one rectangular (sqrt W, x) grid
    serves every W, and a star's draw is a row lookup + two `jnp.interp`
    inverse-CDF evaluations + one lerp. The speed weight

        u^2 E_gamma(g, W - u^2/2) du = (2W)^{3/2} x^2 E_gamma(g, W(1-x^2)) dx

    has a W-only prefactor that cancels in the normalized CDF, so each row
    stores the CDF of x^2 E_gamma(g, W(1 - x^2)) alone. The weights are >= 0
    (E_gamma >= 0), so every CDF row is monotone by construction, and LINEAR
    interpolation of monotone rows stays monotone -- the cubic-overshoot
    clamp of AnisoDensityTable does not arise here.
    """

    s_nodes: Float[Array, "n_W"]  # sqrt(W) nodes, uniform on [0, sqrt(W_max)]
    x_nodes: Float[Array, "n_x"]  # x = u/sqrt(2W) nodes, uniform on [0, 1]
    cdf: Float[Array, "n_W n_x"]  # normalized speed CDF per sqrt(W) row

    @classmethod
    def build(cls, W_max, g, n_W: int = 256, n_x: int = 256):
        """Tabulate normalized speed CDFs on the (sqrt W, x) grid.

        One batched E_gamma evaluation of n_W x n_x points (65k at default
        resolution, vs 256 per star in the exact sampler). W_max gets the
        x1.001 safety factor so in-domain queries never clamp. Differentiable
        in g (through the node CDFs) and W_max (through the nodes). The W = 0
        row is floored to W = 1e-6 -- the `inverse` draw-guard threshold, so
        the floored row is never drawn from directly (W <= 1e-6 returns u = 0)
        and serves only as the lower lerp anchor for stars just above the
        guard; its shape is the W = 1e-6 CDF, indistinguishable from the
        analytic W -> 0 limit x^2 (1-x^2)^g at that W. The floor also keeps
        the raw row total (which scales as W^g) far above float64 underflow
        for g in [0, 3.5], so the plain relative normalization below ends
        every row's CDF at exactly 1.0 (an absolute +1e-30 regularizer would
        swamp a W = 1e-12 row total ~W^g at g >= 2.5 and corrupt row 0).
        Per-component row resolution: a component with velocity-scale ratio w
        (w_j / max_j w_j after rescaling) only ever queries the first w_min/w
        fraction of the n_W = 256 sqrt(W) rows, so its effective resolution
        shrinks proportionally for extreme scale ratios.
        """
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        x = jnp.linspace(0.0, 1.0, n_x)
        W = jnp.maximum(s**2, 1e-6)  # W=0-row floor at the draw guard (see above)
        wgt = jnp.maximum(
            x[None, :] ** 2 * lowered_exponential(g, W[:, None] * (1.0 - x[None, :] ** 2)),
            0.0,
        )
        dx = x[1] - x[0]
        cdf = jnp.concatenate(
            [jnp.zeros((n_W, 1)),
             jnp.cumsum(0.5 * (wgt[:, 1:] + wgt[:, :-1]), axis=1) * dx],
            axis=1,
        )
        # Relative normalization: every row total is strictly positive (the
        # 1e-6 W floor keeps it >= O(1e-23) even at g = 3.5, far above float64
        # underflow), so plain division ends each row's CDF at exactly 1.0
        # and stays differentiable.
        cdf = cdf / cdf[:, -1:]
        return cls(s_nodes=s, x_nodes=x, cdf=cdf)

    def inverse(self, W, unif):
        """Inverse-CDF draw: unit speed u in [0, sqrt(2W)] at potential W from
        the uniform variate unif in [0, 1]. Scalar in, scalar out; vmap over
        stars. Differentiable in W (and in g through the stored CDF rows).

        Locates the two neighboring sqrt(W) rows, inverse-interpolates each
        row's CDF at unif, and lerps the two x results by the cell fraction;
        u = x sqrt(2W). W clamps to the table box; W <= 1e-6 draws exactly 0
        (matching _sample_unit_speed's bound guard).
        """
        # 1e-12 floor mirrors _sample_unit_speed: sqrt(0) has a NaN cotangent
        # under jax.grad and the final where() masks only the primal.
        W_safe = jnp.maximum(W, 1e-12)
        s = jnp.clip(jnp.sqrt(W_safe), self.s_nodes[0], self.s_nodes[-1])
        hs = self.s_nodes[1] - self.s_nodes[0]  # uniform grid by construction
        i = jnp.clip(jnp.searchsorted(self.s_nodes, s) - 1, 0, self.s_nodes.size - 2)
        t = jnp.clip((s - self.s_nodes[i]) / hs, 0.0, 1.0)
        x_lo = jnp.interp(unif, self.cdf[i], self.x_nodes)
        x_hi = jnp.interp(unif, self.cdf[i + 1], self.x_nodes)
        u = ((1.0 - t) * x_lo + t * x_hi) * jnp.sqrt(2.0 * W_safe)
        return jnp.where(W > 1e-6, u, 0.0)


class AnisoSpeedCDFTable(eqx.Module):
    """Anisotropic speed-MARGINAL inverse-CDF table u(W, p, unif) for fixed g
    (Tranche B, Task 6).

    Replaces the per-star 256-point quadrature of the speed-marginal step of
    `_sample_speed_angle` (u^2 E_gamma(g, W - u^2/2) T(p^2 u^2 / 2) on
    [0, sqrt(2W)]) with ONE precomputed 3-D table in normalized coordinates
    x = u / sqrt(2W) in [0, 1] on a (sqrt W, asinh p) grid:

        u^2 E T du = (2W)^(3/2) x^2 E_gamma(g, W(1-x^2)) T(p^2 W x^2) dx,

    so the W-only prefactor cancels in the relative row normalization and
    each (sqrt W, asinh p) row stores the CDF of
    x^2 E_gamma(g, W(1-x^2)) T(p^2 W x^2) alone (T's argument: beta =
    p^2 u^2 / 2 = p^2 W x^2). The angular conditional cos(theta)|u is NOT
    tabulated -- it stays exact (`_sample_costheta_given_u`, cheap exp
    arithmetic). Axis choices mirror AnisoDensityTable (sqrt W stretches the
    W -> 0 power law; asinh p gives log-like large-p resolution) and the
    SpeedCDFTable normalization lesson applies: a 1e-6 row-W floor keeps raw
    row totals (~W^g, further T-suppressed at large p) far above float64
    underflow so plain relative division ends every row's CDF at exactly 1.0.
    """

    s_nodes: Float[Array, "n_W"]  # sqrt(W) nodes, uniform on [0, sqrt(W_max)]
    q_nodes: Float[Array, "n_p"]  # asinh(p) nodes, uniform on [0, asinh(p_max)]
    x_nodes: Float[Array, "n_x"]  # x = u/sqrt(2W) nodes, uniform on [0, 1]
    cdf: Float[Array, "n_W n_p n_x"]  # normalized speed-marginal CDF per row

    @classmethod
    def build(cls, W_max, p_max, g, n_W: int = 192, n_p: int = 48,
              n_x: int = 192):
        """Tabulate normalized speed-marginal CDFs on the (sqrt W, asinh p, x)
        grid (1.8M float64 = 14 MB at the defaults).

        Maps over sqrt(W) rows with `jax.lax.map` (scan-based, differentiable):
        a fully batched build would materialize the (n_W, n_p, n_x, 91)
        Poisson-sum intermediate of `_angle_integral_T` (~1.3 GB at the
        defaults) -- the AnisoDensityTable build lesson. W_max gets the x1.001
        safety factor so in-domain queries never clamp. Differentiable in g
        and (W_max, p_max) through the nodes; the W = 0 row is floored to
        W = 1e-6 (the `inverse` draw-guard threshold, see SpeedCDFTable).
        """
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        q = jnp.linspace(0.0, jnp.arcsinh(jnp.asarray(p_max)), n_p)
        x = jnp.linspace(0.0, 1.0, n_x)
        W = jnp.maximum(s**2, 1e-6)  # W=0-row floor at the draw guard
        p = jnp.sinh(q)

        def row(w):
            E = lowered_exponential(g, w * (1.0 - x**2))  # (n_x,)
            T = _angle_integral_T(p[:, None] ** 2 * w * x[None, :] ** 2)
            wgt = jnp.maximum(x[None, :] ** 2 * E[None, :] * T, 0.0)
            dx = x[1] - x[0]
            c = jnp.concatenate(
                [jnp.zeros((n_p, 1)),
                 jnp.cumsum(0.5 * (wgt[:, 1:] + wgt[:, :-1]), axis=1) * dx],
                axis=1,
            )
            # Relative normalization: the 1e-6 W floor keeps every row total
            # strictly positive (>= O(1e-23) at g = 3.5, further suppressed
            # only polynomially by T at large p), so the division is exact to
            # an ulp and differentiable. Inside lax.map XLA rewrites x/x as
            # x * (1/x) (1 ulp short of 1.0), so pin the last column to its
            # mathematically identical value 1.0 (true gradient there is 0).
            c = c / c[:, -1:]
            return c.at[:, -1].set(1.0)

        cdf = jax.lax.map(row, W)
        return cls(s_nodes=s, q_nodes=q, x_nodes=x, cdf=cdf)

    def inverse(self, W, p, unif):
        """Inverse-CDF draw: unit speed u in [0, sqrt(2W)] at potential W and
        anisotropy parameter p = r/r_a from the uniform variate unif in [0, 1].
        Scalar in, scalar out; vmap over stars. Differentiable in (W, p) and
        in g through the stored CDF rows.

        Locates the (sqrt W, asinh p) cell, inverse-interpolates the FOUR
        neighboring rows' CDFs at unif, bilinearly lerps the four x results,
        and rescales u = x sqrt(2W). (W, p) clamp to the table box; W <= 1e-6
        draws exactly 0 (matching _sample_speed_angle's bound guard).
        """
        # 1e-12 floor mirrors _sample_speed_angle: sqrt(0) has a NaN cotangent
        # under jax.grad and the final where() masks only the primal.
        W_safe = jnp.maximum(W, 1e-12)
        s = jnp.clip(jnp.sqrt(W_safe), self.s_nodes[0], self.s_nodes[-1])
        q = jnp.clip(jnp.arcsinh(jnp.maximum(p, 0.0)),
                     self.q_nodes[0], self.q_nodes[-1])
        hs = self.s_nodes[1] - self.s_nodes[0]  # uniform grids by construction
        hq = self.q_nodes[1] - self.q_nodes[0]
        i = jnp.clip(jnp.searchsorted(self.s_nodes, s) - 1, 0, self.s_nodes.size - 2)
        j = jnp.clip(jnp.searchsorted(self.q_nodes, q) - 1, 0, self.q_nodes.size - 2)
        ts = jnp.clip((s - self.s_nodes[i]) / hs, 0.0, 1.0)
        tq = jnp.clip((q - self.q_nodes[j]) / hq, 0.0, 1.0)
        rows = jax.lax.dynamic_slice(self.cdf, (i, j, jnp.asarray(0, i.dtype)),
                                     (2, 2, self.x_nodes.size))
        x4 = jax.vmap(lambda c: jnp.interp(unif, c, self.x_nodes))(
            rows.reshape(4, -1))
        x = ((1.0 - ts) * (1.0 - tq) * x4[0] + (1.0 - ts) * tq * x4[1]
             + ts * (1.0 - tq) * x4[2] + ts * tq * x4[3])
        u = x * jnp.sqrt(2.0 * W_safe)
        return jnp.where(W > 1e-6, u, 0.0)


__all__ = ["AnisoDensityTable", "AnisoSpeedCDFTable", "SpeedCDFTable"]
