# progenax/src/progenax/profiles/limepy_tables.py
"""Density and speed-CDF tables for lowered-isothermal models.

The density table interpolates density divided by its analytic isotropic
value in sqrt(W) and log(1+p^2 W), where W is the relative potential and
p=r/r_a. The analytic normalization resolves both the escape boundary and
the deep-potential limit; the even coordinate preserves the zero p derivative.
Speed-CDF tables use their separately documented grids and quadrature rules.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxstro.numerics import gauss_legendre_nodes
from jaxtyping import Array, Float

from progenax.profiles.limepy import (
    _SPEED_T,
    _SPEED_WEIGHTS,
    _angle_integral_T,
    limepy_density_hat,
    lowered_exponential,
)


def _speed_weight(W, g, x):
    """Unnormalized density in x=u/sqrt(2W), with the common W**g removed.

    At W=0 the limiting shape is x**2 (1-x**2)**g / Gamma(g+1).
    The g=0 escape endpoint uses its interior limit for integration.
    """
    safe = jnp.where(W > 0.0, W, 1.0)
    z = jnp.maximum(1.0 - x**2, 0.0)
    residual = lowered_exponential(g, safe * z) / safe**g
    residual = jnp.where((g == 0.0) & (z == 0.0), 1.0, residual)
    # Avoid an inactive log(0) in derivatives with respect to g.
    power = jnp.exp(g * jnp.log(jnp.where(z > 0.0, z, 1.0)))
    power = jnp.where(z > 0.0, power, jnp.where(g == 0.0, 1.0, 0.0))
    limit = power * jnp.exp(-jax.scipy.special.gammaln(g + 1.0))
    return x**2 * jnp.where(W > 0.0, residual, limit)


_CDF_NODES, _CDF_WEIGHTS = gauss_legendre_nodes(8)


def _speed_cdf(W, g, x, beta=0.0, *, angular=None):
    """Integrate normalized speed weights with eight interior nodes per cell.

    beta=p**2*W supplies the angular scale. Leading dimensions of W and
    beta broadcast over the final (interval, quadrature-node) dimensions.
    """
    width = jnp.diff(x)
    nodes = x[:-1, None] + width[:, None] * (_CDF_NODES + 1.0) / 2.0
    if angular is None:
        angular = _angle_integral_T(beta * nodes**2)
    weight = _speed_weight(W, g, nodes) * angular
    area = width * jnp.sum(weight * _CDF_WEIGHTS / 2.0, axis=-1)
    cumulative = jnp.concatenate(
        (jnp.zeros_like(area[..., :1]), jnp.cumsum(area, axis=-1)), axis=-1
    )
    cumulative = cumulative / cumulative[..., -1:]
    return cumulative.at[..., -1].set(1.0)


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

    s_nodes: Float[Array, "n_W"]  # sqrt(W) nodes, uniform on [0, sqrt(W_max)]
    q_nodes: Float[Array, "n_p"]  # log(1+p^2 W) nodes
    values: Float[Array, "n_W n_p"]  # rho_hat / (sqrt(2*pi) * E_gamma(g+3/2,W))
    g: Float[Array, ""]

    @classmethod
    def build(cls, W_max, p_max, g, n_W: int = 160, n_p: int = 160):
        """Build the factored density on a uniform coupled-coordinate grid.

        W_max and p_max must be positive; both grid sizes must be at least four.
        The upper potential has the existing 1.001 safety factor. A 128-node
        Gauss-Legendre speed rule excludes the escape endpoint, including g=0.
        Rows at W=0 use the analytic limiting ratio. Interior queries and
        build inputs carry derivatives, including the one-sided g=0 tangent.
        """
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        q = jnp.linspace(0.0, jnp.log1p((p_max * s[-1]) ** 2), n_p)
        t = _SPEED_T
        angular = _angle_integral_T(jnp.expm1(q)[:, None] * t**2)
        limit = (1.0 - t**2) ** g * jnp.exp(-jax.scipy.special.gammaln(g + 1))

        def row(w):
            safe = jnp.where(w > 0, w, 1.0)
            ratio = lowered_exponential(g, safe * (1.0 - t**2)) / safe**g
            ratio = jnp.where(w > 0, ratio, limit)
            residual = 2.0**1.5 * jnp.sum(
                _SPEED_WEIGHTS * t**2 * ratio * angular, axis=1
            )
            isotropic = jnp.sqrt(2.0 * jnp.pi) * limepy_density_hat(safe, g)
            normalization = jnp.where(
                w > 0,
                isotropic / safe ** (g + 1.5),
                jnp.sqrt(2.0 * jnp.pi) * jnp.exp(-jax.scipy.special.gammaln(g + 2.5)),
            )
            return residual / normalization

        return cls(s, q, jax.lax.map(row, s**2), jnp.asarray(g))

    def evaluate(self, W, p):
        """Interpolate density; W<=0 returns zero, and coordinates saturate at edges.

        Four-point tensor-product Lagrange interpolation acts on the anisotropic
        density ratio. The analytic isotropic density is restored afterward.
        The differentiable accuracy contract covers interior physical queries.
        """
        safe = jnp.where(W > 0, W, 1.0)
        s = jnp.sqrt(jnp.maximum(W, 1e-12))
        q = jnp.log1p(p * p * safe)
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
        return jnp.where(
            W > 0.0, v * jnp.sqrt(2.0 * jnp.pi) * limepy_density_hat(safe, self.g), 0.0
        )


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
    x_nodes: Float[Array, "n_x"]  # x = u/sqrt(2W) nodes, cosine spacing on [0, 1]
    cdf: Float[Array, "n_W n_x"]  # normalized speed CDF per sqrt(W) row

    @classmethod
    def build(cls, W_max, g, n_W: int = 256, n_x: int = 384):
        """Tabulate normalized speed CDFs on the (sqrt W, x) grid.

        Eight-point Gauss integration in each speed interval excludes escape
        endpoints and retains the one-sided g=0 derivative. W_max gets the
        x1.001 safety factor so in-domain queries never clamp. Differentiable
        in g (through the node CDFs) and W_max (through the nodes). The W = 0
        row uses the analytic W -> 0 normalized shape. Dividing out W**g
        before normalization keeps weights finite at small positive W.
        Per-component row resolution: a component with velocity-scale ratio w
        (w_j / max_j w_j after rescaling) only ever queries the first w_min/w
        fraction of the n_W = 256 sqrt(W) rows, so its effective resolution
        shrinks proportionally for extreme scale ratios.
        """
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        x = (1.0 - jnp.cos(jnp.pi * jnp.linspace(0.0, 1.0, n_x))) / 2.0
        W = s**2
        cdf = _speed_cdf(W[:, None, None], g, x)
        return cls(s_nodes=s, x_nodes=x, cdf=cdf)

    def inverse(self, W, unif):
        """Inverse-CDF draw: unit speed u in [0, sqrt(2W)] at potential W from
        the uniform variate unif in [0, 1]. Scalar in, scalar out; vmap over
        stars. Differentiable in W (and in g through the stored CDF rows).

        Locates the two neighboring sqrt(W) rows, inverse-interpolates each
        row's CDF at unif, and lerps the two x results by the cell fraction;
        u = x sqrt(2W). W clamps to the table box; W <= 0 draws exactly 0.
        Potential derivatives are qualified only for W > 0.
        """
        W_safe = jnp.where(W > 0.0, W, 1.0)
        s = jnp.clip(jnp.sqrt(W_safe), self.s_nodes[0], self.s_nodes[-1])
        hs = self.s_nodes[1] - self.s_nodes[0]  # uniform grid by construction
        i = jnp.clip(jnp.searchsorted(self.s_nodes, s) - 1, 0, self.s_nodes.size - 2)
        t = jnp.clip((s - self.s_nodes[i]) / hs, 0.0, 1.0)
        x_lo = jnp.interp(unif, self.cdf[i], self.x_nodes)
        x_hi = jnp.interp(unif, self.cdf[i + 1], self.x_nodes)
        u = ((1.0 - t) * x_lo + t * x_hi) * jnp.sqrt(2.0 * W_safe)
        return jnp.where(W > 0.0, u, 0.0)


class AnisoSpeedCDFTable(eqx.Module):
    """Anisotropic speed-MARGINAL inverse-CDF table u(W, p, unif) for fixed g
    (Tranche B, Task 6).

    Replaces per-star speed-marginal integration in
    `_sample_speed_angle` (u^2 E_gamma(g, W - u^2/2) T(p^2 u^2 / 2) on
    [0, sqrt(2W)]) with ONE precomputed 3-D table in normalized coordinates
    x = u / sqrt(2W) in [0, 1] on a (sqrt W, log(1+p^2 W)) grid:

        u^2 E T du = (2W)^(3/2) x^2 E_gamma(g, W(1-x^2)) T(p^2 W x^2) dx,

    so the W-only prefactor cancels in the relative row normalization and
    each (sqrt W, log(1+p^2 W)) row stores the CDF of
    x^2 E_gamma(g, W(1-x^2)) T(p^2 W x^2) alone (T's argument: beta =
    p^2 u^2 / 2 = p^2 W x^2). The angular conditional cos(theta)|u is NOT
    tabulated -- it uses quadrature (`_sample_costheta_given_u`, cheap exp
    arithmetic). Here sqrt(W) resolves low potentials; log(1+p^2 W) resolves the coupled
    angular scale and preserves the even dependence on p.
    The common W**g factor is removed before row normalization. At W=0,
    the analytic limiting shape supplies the interpolation anchor.

    Build cost is amortized over the draws that reuse a table.
    """

    s_nodes: Float[Array, "n_W"]  # sqrt(W) nodes, uniform on [0, sqrt(W_max)]
    q_nodes: Float[Array, "n_p"]  # log(1+p^2 W) nodes, uniform over the table box
    x_nodes: Float[Array, "n_x"]  # x = u/sqrt(2W), cosine spacing on [0, 1]
    cdf: Float[Array, "n_W n_p n_x"]  # normalized speed-marginal CDF per row

    @classmethod
    def build(cls, W_max, p_max, g, n_W: int = 192, n_p: int = 96, n_x: int = 384):
        """Tabulate normalized speed-marginal CDFs on the (sqrt W, log(1+p^2 W), x)
        grid (7.1M float64 = 57 MB at the defaults). Eight-point Gauss
        integration in each speed interval excludes escape endpoints.

        Maps over sqrt(W) rows with `jax.lax.map` (scan-based, differentiable).
        Angular weights depend only on the coupled coordinate and are computed
        once before the scan. W_max gets the x1.001
        safety factor so in-domain queries never clamp. Differentiable in g
        and (W_max, p_max) through the nodes. The W=0 row uses its analytic limit.
        """
        s = jnp.linspace(0.0, jnp.sqrt(jnp.asarray(W_max) * 1.001), n_W)
        q = jnp.linspace(0.0, jnp.log1p(jnp.asarray(p_max) ** 2 * s[-1] ** 2), n_p)
        x = (1.0 - jnp.cos(jnp.pi * jnp.linspace(0.0, 1.0, n_x))) / 2.0
        W = s**2
        beta = jnp.expm1(q)
        nodes = x[:-1, None] + jnp.diff(x)[:, None] * (_CDF_NODES + 1.0) / 2.0
        angular = _angle_integral_T(beta[:, None, None] * nodes**2)

        def row(w):
            return _speed_cdf(w, g, x, angular=angular)

        cdf = jax.lax.map(row, W)
        return cls(s_nodes=s, q_nodes=q, x_nodes=x, cdf=cdf)

    def inverse(self, W, p, unif):
        """Inverse-CDF draw: unit speed u in [0, sqrt(2W)] at potential W and
        anisotropy parameter p = r/r_a from the uniform variate unif in [0, 1].
        Scalar in, scalar out; vmap over stars. Differentiable in (W, p) and
        in g through the stored CDF rows.

        Locates the (sqrt W, log(1+p^2 W)) cell, inverse-interpolates the FOUR
        neighboring rows' CDFs at unif, bilinearly lerps the four x results,
        and rescales u = x sqrt(2W). (W, p) clamp to the table box; W <= 0
        draws exactly 0. Potential derivatives are qualified only for W > 0.
        """
        W_safe = jnp.where(W > 0.0, W, 1.0)
        s = jnp.clip(jnp.sqrt(W_safe), self.s_nodes[0], self.s_nodes[-1])
        q = jnp.clip(jnp.log1p(p**2 * W_safe), self.q_nodes[0], self.q_nodes[-1])
        hs = self.s_nodes[1] - self.s_nodes[0]  # uniform grids by construction
        hq = self.q_nodes[1] - self.q_nodes[0]
        i = jnp.clip(jnp.searchsorted(self.s_nodes, s) - 1, 0, self.s_nodes.size - 2)
        j = jnp.clip(jnp.searchsorted(self.q_nodes, q) - 1, 0, self.q_nodes.size - 2)
        ts = jnp.clip((s - self.s_nodes[i]) / hs, 0.0, 1.0)
        tq = jnp.clip((q - self.q_nodes[j]) / hq, 0.0, 1.0)
        rows = jax.lax.dynamic_slice(
            self.cdf, (i, j, jnp.asarray(0, i.dtype)), (2, 2, self.x_nodes.size)
        )
        x4 = jax.vmap(lambda c: jnp.interp(unif, c, self.x_nodes))(rows.reshape(4, -1))
        x = (
            (1.0 - ts) * (1.0 - tq) * x4[0]
            + (1.0 - ts) * tq * x4[1]
            + ts * (1.0 - tq) * x4[2]
            + ts * tq * x4[3]
        )
        u = x * jnp.sqrt(2.0 * W_safe)
        return jnp.where(W > 0.0, u, 0.0)


__all__ = ["AnisoDensityTable", "AnisoSpeedCDFTable", "SpeedCDFTable"]
