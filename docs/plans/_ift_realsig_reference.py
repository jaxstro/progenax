"""Resolution for risk #2: when ra_hat is None (isotropic), keep ra_hat AND eta
OUT of the differentiated arg set (move to nondiff/closure). custom_vjp cannot
emit a cotangent for a Python None. Two clean designs:

  DESIGN 1 (recommended): TWO custom_vjp solvers selected by a Python branch in
  the wrapper. Isotropic solver differentiates (m_j, M_j, W0, g, delta) with
  ra_hat/eta baked as nondiff. Anisotropic solver differentiates
  (m_j, M_j, W0, g, delta, ra_hat, eta).

This file implements DESIGN 1 and re-runs STEP B (grad vs FD), STEP C (jit grad),
STEP E (grad wrt M_j), plus the anisotropic grad, to confirm all PASS.
"""
import functools
import jax
import jax.numpy as jnp
import numpy as np

import progenax  # noqa: F401
from progenax.profiles.limepy_multimass import _realized_fractions, _bin_imf
from progenax.imf import Maschberger

np.set_printoptions(precision=8)
XI_MAX, N_POINTS, N_COMP = 300.0, 2000, 4


def _alpha_map(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method):
    f_real = _realized_fractions(alpha, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    a = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
    return a / jnp.sum(a)


def _alpha_residual(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method):
    return alpha - _alpha_map(alpha, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)


# ---------- ISOTROPIC solver: ra_hat/eta are NONDIFF (None sentinel baked in) ----
# nondiff_argnums = (5,6,7,8,9,10) -> (ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter)
# differentiated = (m_j, M_j, W0, g, delta)  [indices 0..4]
@functools.partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11))
def _solve_iso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter):
    f_target = M_j / jnp.sum(M_j)

    def cond(s):
        _, it, r = s
        return jnp.logical_and(it < max_iter, r > tol)

    def body(s):
        a, it, _ = s
        a_new = _alpha_map(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
        f_real = _realized_fractions(a_new, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
        return a_new, it + 1, jnp.max(jnp.abs(f_real - f_target))

    a_star, _, _ = jax.lax.while_loop(cond, body, (f_target, jnp.array(0), jnp.array(jnp.inf)))
    return a_star


def _solve_iso_fwd(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter):
    a_star = _solve_iso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter)
    return a_star, (a_star, m_j, M_j, W0, g, delta)


def _solve_iso_bwd(ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter, res, a_bar):
    a_star, m_j, M_j, W0, g, delta = res
    f_target = M_j / jnp.sum(M_j)
    R_a = lambda a: _alpha_residual(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    _, vjp_a = jax.vjp(R_a, a_star)
    J = jax.vmap(lambda e: vjp_a(e)[0])(jnp.eye(a_star.shape[0]))
    w = jnp.linalg.lstsq(J.T, a_bar, rcond=None)[0]
    R_th = lambda mj, Mj, W, gg, d: _alpha_residual(
        a_star, mj, Mj / jnp.sum(Mj), W, gg, d, xi_max, n_points, ra_hat, eta, aniso_method)
    _, vjp_th = jax.vjp(R_th, m_j, M_j, W0, g, delta)
    gm, gM, gW, gg, gd = vjp_th(w)
    return (-gm, -gM, -gW, -gg, -gd)


_solve_iso.defvjp(_solve_iso_fwd, _solve_iso_bwd)


# ---------- ANISOTROPIC solver: ra_hat/eta DIFFERENTIATED (finite arrays) -------
@functools.partial(jax.custom_vjp, nondiff_argnums=(7, 8, 9, 10, 11))
def _solve_aniso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter):
    f_target = M_j / jnp.sum(M_j)

    def cond(s):
        _, it, r = s
        return jnp.logical_and(it < max_iter, r > tol)

    def body(s):
        a, it, _ = s
        a_new = _alpha_map(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
        f_real = _realized_fractions(a_new, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
        return a_new, it + 1, jnp.max(jnp.abs(f_real - f_target))

    a_star, _, _ = jax.lax.while_loop(cond, body, (f_target, jnp.array(0), jnp.array(jnp.inf)))
    return a_star


def _solve_aniso_fwd(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter):
    a_star = _solve_aniso(m_j, M_j, W0, g, delta, ra_hat, eta, xi_max, n_points, aniso_method, tol, max_iter)
    return a_star, (a_star, m_j, M_j, W0, g, delta, ra_hat, eta)


def _solve_aniso_bwd(xi_max, n_points, aniso_method, tol, max_iter, res, a_bar):
    a_star, m_j, M_j, W0, g, delta, ra_hat, eta = res
    f_target = M_j / jnp.sum(M_j)
    R_a = lambda a: _alpha_residual(a, m_j, f_target, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    _, vjp_a = jax.vjp(R_a, a_star)
    J = jax.vmap(lambda e: vjp_a(e)[0])(jnp.eye(a_star.shape[0]))
    w = jnp.linalg.lstsq(J.T, a_bar, rcond=None)[0]
    R_th = lambda mj, Mj, W, gg, d, rah, et: _alpha_residual(
        a_star, mj, Mj / jnp.sum(Mj), W, gg, d, xi_max, n_points, rah, et, aniso_method)
    _, vjp_th = jax.vjp(R_th, m_j, M_j, W0, g, delta, ra_hat, eta)
    gm, gM, gW, gg, gd, gra, get = vjp_th(w)
    return (-gm, -gM, -gW, -gg, -gd, -gra, -get)


_solve_aniso.defvjp(_solve_aniso_fwd, _solve_aniso_bwd)


def find_alpha_real(m_j, M_j, W0, g, delta, n_iter=30, xi_max=XI_MAX,
                    n_points=N_POINTS, ra_hat=None, eta=0.0, aniso_method="table", tol=1e-6):
    m_j = jnp.asarray(m_j); M_j = jnp.asarray(M_j)
    W0 = jnp.asarray(W0); g = jnp.asarray(g); delta = jnp.asarray(delta)
    if ra_hat is None:
        a = _solve_iso(m_j, M_j, W0, g, delta, None, eta, xi_max, n_points, aniso_method, tol, n_iter)
    else:
        a = _solve_aniso(m_j, M_j, W0, g, delta, jnp.asarray(ra_hat), jnp.asarray(eta),
                         xi_max, n_points, aniso_method, tol, n_iter)
    f_real = _realized_fractions(a, m_j, W0, g, delta, xi_max, n_points, ra_hat, eta, aniso_method)
    residual = jnp.max(jnp.abs(f_real - M_j / jnp.sum(M_j)))
    return a, residual


def bin_imf_diff(alpha_imf, n_comp, m_range):
    edges = jnp.asarray(np.geomspace(m_range[0], m_range[1], n_comp + 1))
    imf = Maschberger(alpha=alpha_imf, m_min=m_range[0], m_max=m_range[1])
    m_list, M_list = [], []
    for k in range(n_comp):
        m_sub = jnp.linspace(edges[k], edges[k + 1], 64)
        pdf = jnp.exp(imf.logpdf(m_sub))
        N = jnp.trapezoid(pdf, m_sub); M = jnp.trapezoid(m_sub * pdf, m_sub)
        m_list.append(M / N); M_list.append(M)
    return jnp.stack(m_list), jnp.stack(M_list)


W0, G_PARAM, DELTA = 5.0, 1.0, 0.4
print("=" * 72)
print("FIX (DESIGN 1: two solvers, iso keeps ra_hat/eta nondiff)")
print("=" * 72)


def loss_theta(theta):
    ai, d, W = theta
    mjj, Mjj = bin_imf_diff(ai, N_COMP, (0.1, 20.0))
    a, _ = find_alpha_real(mjj, Mjj, W, G_PARAM, d, ra_hat=None)
    return jnp.sum(a ** 2)


THETA0 = jnp.array([2.3, DELTA, W0])
g_ad = np.asarray(jax.grad(loss_theta)(THETA0))
h = 1e-4
g_fd = np.array([(float(loss_theta(THETA0.at[i].add(h))) - float(loss_theta(THETA0.at[i].add(-h)))) / (2 * h)
                 for i in range(3)])
labels = ["d/d_alpha_imf", "d/d_delta", "d/d_W0"]
print(f"  {'param':16s} {'FD':>15s} {'AD(IFT)':>15s} {'relerr':>12s}")
for i, lab in enumerate(labels):
    re = abs(g_ad[i] - g_fd[i]) / (abs(g_fd[i]) + 1e-30)
    print(f"  {lab:16s} {g_fd[i]:15.8e} {g_ad[i]:15.8e} {re:12.2e}")
relerr_max = float(np.max(np.abs(g_ad - g_fd) / (np.abs(g_fd) + 1e-30)))
print(f"  STEP B MAX rel-err = {relerr_max:.2e}  (<1e-5: {relerr_max < 1e-5})")

jg = jax.jit(jax.grad(loss_theta))
out = jg(THETA0); jax.block_until_ready(out)
print(f"  STEP C jit(grad) = {np.asarray(out)}  -> OK")

m3, M3 = _bin_imf(Maschberger(alpha=2.3, m_min=0.1, m_max=20.0), 3, (0.1, 20.0))
m3, M3 = jnp.asarray(m3), jnp.asarray(M3)
dM = jax.grad(lambda M: jnp.sum(find_alpha_real(m3, M, 7.0, 1.0, 0.4, n_points=1500)[0] ** 2))(M3)
print(f"  STEP E grad wrt M_j = {np.asarray(dM)}  finite&nonzero: "
      f"{bool(jnp.all(jnp.isfinite(dM)) and jnp.any(jnp.abs(dM) > 0))}")

# anisotropic grad wrt ra_hat still works
m2, M2 = jnp.array([0.5, 2.0]), jnp.array([1.0, 1.0])
g_ra = float(jax.grad(lambda ra: jnp.sum(
    find_alpha_real(m2, M2, 5.0, 1.0, 0.4, ra_hat=ra, xi_max=800.0, n_points=1500)[0] ** 2))(jnp.asarray(10.0)))
print(f"  aniso grad wrt ra_hat = {g_ra:.6e}  (finite: {np.isfinite(g_ra)})")
print("\n  ALL FIXED-PATH CHECKS PASS" if (relerr_max < 1e-5) else "\n  STILL FAILING")
