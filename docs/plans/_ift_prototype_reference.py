"""Empirical IFT-vs-unrolled benchmark for find_alpha_for_masses.

Scratch only. Does NOT modify tracked files.

Compares:
  - Unrolled (current) find_alpha_for_masses  [lax.scan n_iter, reverse custom_vjp ODE]
  - Prototype 1: hand-rolled jax.custom_vjp, reverse-mode implicit backward, fixed-scan fwd
  - Prototype 2: same custom_vjp but adaptive while_loop forward to tolerance

Loss L(theta) = sum_j alpha_j(theta)^2.
"""
import time
import functools

import jax
import jax.numpy as jnp
import numpy as np

from progenax.imf import Maschberger
from progenax.profiles.limepy_multimass import (
    find_alpha_for_masses,
    _realized_fractions,
    _bin_imf,
)

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------------
# Representative case
# ----------------------------------------------------------------------------
N_COMP = 4
W0 = 5.0
G_PARAM = 1.0
DELTA = 0.4
XI_MAX = 300.0
N_POINTS = 2000

imf = Maschberger(alpha=2.3, m_min=0.1, m_max=20.0)
m_j, M_j = _bin_imf(imf, N_COMP, (0.1, 20.0))
m_j = jnp.asarray(m_j)
M_j = jnp.asarray(M_j)
print("m_j =", np.asarray(m_j))
print("M_j =", np.asarray(M_j))
f_target_global = M_j / jnp.sum(M_j)
print("f_target =", np.asarray(f_target_global))


# ----------------------------------------------------------------------------
# Theta packing. theta = (alpha_imf, delta, W0)  -> rebuild m_j, M_j, params
# We differentiate L wrt these scalars. m_j/M_j depend on alpha_imf via the IMF
# binning (done in numpy/jax trapezoid -> differentiable wrt alpha).
# ----------------------------------------------------------------------------
def bin_imf_diff(alpha_imf, n_comp, m_range):
    """Differentiable _bin_imf wrt alpha_imf (fixed log-spaced edges)."""
    edges = jnp.asarray(np.geomspace(m_range[0], m_range[1], n_comp + 1))
    m_list, M_list = [], []
    imf_local = Maschberger(alpha=alpha_imf, m_min=m_range[0], m_max=m_range[1])
    for k in range(n_comp):
        e0, e1 = edges[k], edges[k + 1]
        m_sub = jnp.linspace(e0, e1, 64)
        pdf = jnp.exp(imf_local.logpdf(m_sub))
        N = jnp.trapezoid(pdf, m_sub)
        M = jnp.trapezoid(m_sub * pdf, m_sub)
        m_list.append(M / N)
        M_list.append(M)
    return jnp.stack(m_list), jnp.stack(M_list)


# ============================================================================
# UNROLLED baseline loss
# ============================================================================
def make_unrolled_loss(n_iter):
    def L(theta):
        alpha_imf, delta, W0_ = theta
        m_jj, M_jj = bin_imf_diff(alpha_imf, N_COMP, (0.1, 20.0))
        alpha, _ = find_alpha_for_masses(
            m_jj, M_jj, W0_, G_PARAM, delta,
            n_iter=n_iter, xi_max=XI_MAX, n_points=N_POINTS,
        )
        return jnp.sum(alpha ** 2)
    return L


# ============================================================================
# Prototype core: f_real as explicit function of (alpha, m_j, M_j, W0, g, delta)
# ============================================================================
def f_real_fn(alpha, m_jj, M_jj, W0_, g_, delta_):
    """Realized fractions; explicit args (no closure) so jacrev/vjp are clean."""
    return _realized_fractions(
        alpha, m_jj, W0_, g_, delta_, XI_MAX, N_POINTS, None, 0.0, "table"
    )


def residual_R(alpha, m_jj, M_jj, W0_, g_, delta_):
    """Fixed-point residual R(alpha,theta) = alpha - sqrt-map(alpha).
    R(alpha*, theta) = 0 at the fixed point."""
    f_target = M_jj / jnp.sum(M_jj)
    f_real = f_real_fn(alpha, m_jj, M_jj, W0_, g_, delta_)
    alpha_new = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
    alpha_new = alpha_new / jnp.sum(alpha_new)
    return alpha - alpha_new


def residual_Rf(alpha, m_jj, M_jj, W0_, g_, delta_):
    """Alt residual Rf = f_real - f_target (also zero at fixed point)."""
    f_target = M_jj / jnp.sum(M_jj)
    return f_real_fn(alpha, m_jj, M_jj, W0_, g_, delta_) - f_target


# Linear-solve handlers -------------------------------------------------------
def solve_lstsq(JT, b):
    return jnp.linalg.lstsq(JT, b, rcond=None)[0]


def solve_pinv(JT, b):
    return jnp.linalg.pinv(JT) @ b


def solve_tikhonov(JT, b, lam=1e-8):
    n = JT.shape[0]
    A = JT.T @ JT + lam * jnp.eye(n)
    return jnp.linalg.solve(A, JT.T @ b)


# ============================================================================
# Prototype 1: hand-rolled custom_vjp, reverse-mode implicit backward, fixed fwd
# ============================================================================
def make_proto1(n_iter, residual_fn, linsolve):

    @jax.custom_vjp
    def find_alpha_hr(alpha_imf, delta, W0_):
        m_jj, M_jj = bin_imf_diff(alpha_imf, N_COMP, (0.1, 20.0))
        f_target = M_jj / jnp.sum(M_jj)

        def step(alpha, _):
            f_real = f_real_fn(alpha, m_jj, M_jj, W0_, G_PARAM, delta)
            alpha_new = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
            alpha_new = alpha_new / jnp.sum(alpha_new)
            return alpha_new, None

        alpha_star, _ = jax.lax.scan(step, f_target, None, length=n_iter)
        return alpha_star

    def fwd(alpha_imf, delta, W0_):
        alpha_star = find_alpha_hr(alpha_imf, delta, W0_)
        return alpha_star, (alpha_star, alpha_imf, delta, W0_)

    def bwd(res, alpha_bar):
        alpha_star, alpha_imf, delta, W0_ = res
        m_jj, M_jj = bin_imf_diff(alpha_imf, N_COMP, (0.1, 20.0))

        # J_alpha = d R / d alpha  (n x n), reverse-mode via vmapped vjp over unit cotangents
        R_of_alpha = lambda a: residual_fn(a, m_jj, M_jj, W0_, G_PARAM, delta)
        _, vjp_alpha = jax.vjp(R_of_alpha, alpha_star)
        eye = jnp.eye(alpha_star.shape[0])
        # rows of J: vjp gives (e_i^T J) -> so stacking gives J^T? Build J explicitly.
        # vjp_alpha(e_i) = e_i^T @ J = i-th ROW of J. So stack -> J.
        J_rows = jax.vmap(lambda e: vjp_alpha(e)[0])(eye)  # shape (n,n), row i = i-th row of J
        J_alpha = J_rows
        # Solve J_alpha^T w = alpha_bar
        w = linsolve(J_alpha.T, alpha_bar)

        # theta_bar = - vjp of R wrt theta at cotangent w
        R_of_theta = lambda ai, d, W: residual_fn(
            alpha_star,
            *bin_imf_diff(ai, N_COMP, (0.1, 20.0)),
            W, G_PARAM, d,
        )
        _, vjp_theta = jax.vjp(R_of_theta, alpha_imf, delta, W0_)
        g_ai, g_d, g_W = vjp_theta(w)
        return (-g_ai, -g_d, -g_W)

    find_alpha_hr.defvjp(fwd, bwd)
    return find_alpha_hr


def make_proto1_loss(n_iter, residual_fn, linsolve):
    fa = make_proto1(n_iter, residual_fn, linsolve)
    def L(theta):
        alpha_imf, delta, W0_ = theta
        alpha = fa(alpha_imf, delta, W0_)
        return jnp.sum(alpha ** 2)
    return L, fa


# ============================================================================
# Prototype 2: adaptive while_loop forward to tolerance
# ============================================================================
def make_proto2(tol, max_iter, residual_fn, linsolve):

    @jax.custom_vjp
    def find_alpha_ad(alpha_imf, delta, W0_):
        m_jj, M_jj = bin_imf_diff(alpha_imf, N_COMP, (0.1, 20.0))
        f_target = M_jj / jnp.sum(M_jj)

        def cond(state):
            alpha, it, resid = state
            return jnp.logical_and(it < max_iter, resid > tol)

        def body(state):
            alpha, it, _ = state
            f_real = f_real_fn(alpha, m_jj, M_jj, W0_, G_PARAM, delta)
            alpha_new = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
            alpha_new = alpha_new / jnp.sum(alpha_new)
            resid = jnp.max(jnp.abs(f_real - f_target))
            return (alpha_new, it + 1, resid)

        init = (f_target, jnp.array(0), jnp.array(jnp.inf))
        alpha_star, _, _ = jax.lax.while_loop(cond, body, init)
        return alpha_star

    def fwd(alpha_imf, delta, W0_):
        alpha_star = find_alpha_ad(alpha_imf, delta, W0_)
        return alpha_star, (alpha_star, alpha_imf, delta, W0_)

    def bwd(res, alpha_bar):
        alpha_star, alpha_imf, delta, W0_ = res
        m_jj, M_jj = bin_imf_diff(alpha_imf, N_COMP, (0.1, 20.0))
        R_of_alpha = lambda a: residual_fn(a, m_jj, M_jj, W0_, G_PARAM, delta)
        _, vjp_alpha = jax.vjp(R_of_alpha, alpha_star)
        eye = jnp.eye(alpha_star.shape[0])
        J_alpha = jax.vmap(lambda e: vjp_alpha(e)[0])(eye)
        w = linsolve(J_alpha.T, alpha_bar)
        R_of_theta = lambda ai, d, W: residual_fn(
            alpha_star, *bin_imf_diff(ai, N_COMP, (0.1, 20.0)), W, G_PARAM, d
        )
        _, vjp_theta = jax.vjp(R_of_theta, alpha_imf, delta, W0_)
        g_ai, g_d, g_W = vjp_theta(w)
        return (-g_ai, -g_d, -g_W)

    find_alpha_ad.defvjp(fwd, bwd)
    return find_alpha_ad


# Adaptive iter-count probe (NON-differentiated python loop, just to report counts)
def adaptive_iter_count(alpha_imf, delta, W0_, tol, max_iter):
    m_jj, M_jj = bin_imf_diff(alpha_imf, N_COMP, (0.1, 20.0))
    f_target = M_jj / jnp.sum(M_jj)
    alpha = f_target
    last_resid = np.inf
    for it in range(max_iter):
        f_real = f_real_fn(alpha, m_jj, M_jj, W0_, G_PARAM, delta)
        resid = float(jnp.max(jnp.abs(f_real - f_target)))
        last_resid = resid
        if resid <= tol:
            return it, resid
        alpha = alpha * jnp.sqrt(f_target / (f_real + 1e-300))
        alpha = alpha / jnp.sum(alpha)
    # one more residual eval after last update
    f_real = f_real_fn(alpha, m_jj, M_jj, W0_, G_PARAM, delta)
    last_resid = float(jnp.max(jnp.abs(f_real - f_target)))
    return max_iter, last_resid


# ============================================================================
# Timing helper
# ============================================================================
def time_fn(fn, arg, n_warm=5):
    # compile
    out = fn(arg)
    jax.block_until_ready(out)
    ts = []
    for _ in range(n_warm):
        t0 = time.perf_counter()
        out = fn(arg)
        jax.block_until_ready(out)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(ts)), out


THETA0 = jnp.array([2.3, DELTA, W0])


def run():
    print("\n" + "=" * 70)
    print("BASELINE: unrolled find_alpha_for_masses, value_and_grad(L)")
    print("=" * 70)
    results = {}
    for n_iter in (15, 30):
        L = make_unrolled_loss(n_iter)
        vg = jax.jit(jax.value_and_grad(L))
        v_only = jax.jit(L)
        t_vg, (val, grad) = time_fn(vg, THETA0)
        t_v, _ = time_fn(v_only, THETA0)
        results[("unrolled", n_iter)] = (t_vg, t_v, val, grad)
        print(f"  n_iter={n_iter:2d}: value_and_grad={t_vg:7.2f} ms | value={t_v:7.2f} ms"
              f" | L={float(val):.6e} | grad={np.asarray(grad)}")

    print("\n" + "=" * 70)
    print("PROTOTYPE 1: hand-rolled custom_vjp, reverse implicit bwd, fixed fwd")
    print("  residual=R (sqrt-map), linsolve=lstsq")
    print("=" * 70)
    for n_iter in (15, 30):
        L1, fa1 = make_proto1_loss(n_iter, residual_R, solve_lstsq)
        vg = jax.jit(jax.value_and_grad(L1))
        v_only = jax.jit(L1)
        t_vg, (val, grad) = time_fn(vg, THETA0)
        t_v, _ = time_fn(v_only, THETA0)
        results[("proto1", n_iter)] = (t_vg, t_v, val, grad)
        print(f"  n_iter={n_iter:2d}: value_and_grad={t_vg:7.2f} ms | value={t_v:7.2f} ms"
              f" | L={float(val):.6e} | grad={np.asarray(grad)}")

    print("\n" + "=" * 70)
    print("PROTOTYPE 2: adaptive while_loop fwd (tol=1e-4, cap=60), reverse implicit bwd")
    print("=" * 70)
    L2, fa2 = None, None
    L2_fn = lambda theta: jnp.sum(
        make_proto2(1e-4, 60, residual_R, solve_lstsq)(theta[0], theta[1], theta[2]) ** 2
    )
    # build once outside lambda to avoid re-tracing custom_vjp def
    fa2 = make_proto2(1e-4, 60, residual_R, solve_lstsq)
    def L2(theta):
        return jnp.sum(fa2(theta[0], theta[1], theta[2]) ** 2)
    vg = jax.jit(jax.value_and_grad(L2))
    v_only = jax.jit(L2)
    t_vg, (val, grad) = time_fn(vg, THETA0)
    t_v, _ = time_fn(v_only, THETA0)
    results[("proto2", "adaptive")] = (t_vg, t_v, val, grad)
    print(f"  adaptive: value_and_grad={t_vg:7.2f} ms | value={t_v:7.2f} ms"
          f" | L={float(val):.6e} | grad={np.asarray(grad)}")

    # ----- Forward alpha match -----
    print("\n" + "=" * 70)
    print("FORWARD alpha_star match (max|Delta alpha| vs unrolled, matched n_iter)")
    print("=" * 70)
    for n_iter in (15, 30):
        alpha_unroll, _ = find_alpha_for_masses(
            m_j, M_j, W0, G_PARAM, DELTA, n_iter=n_iter, xi_max=XI_MAX, n_points=N_POINTS
        )
        fa1 = make_proto1(n_iter, residual_R, solve_lstsq)
        alpha_p1 = fa1(2.3, DELTA, W0)
        # note: unrolled here uses fixed m_j/M_j; proto1 rebuilds via bin_imf_diff(2.3)
        # rebuild m_j/M_j consistently for proto via bin_imf_diff to compare fairly
        m_chk, M_chk = bin_imf_diff(2.3, N_COMP, (0.1, 20.0))
        alpha_unroll2, _ = find_alpha_for_masses(
            m_chk, M_chk, W0, G_PARAM, DELTA, n_iter=n_iter, xi_max=XI_MAX, n_points=N_POINTS
        )
        d = float(jnp.max(jnp.abs(alpha_p1 - alpha_unroll2)))
        print(f"  n_iter={n_iter:2d}: max|Delta alpha| = {d:.3e}  (alpha_star={np.asarray(alpha_p1)})")

    # ----- Gradient correctness vs central FD -----
    print("\n" + "=" * 70)
    print("GRADIENT CORRECTNESS: central FD (h=1e-4) vs AD/IFT grads")
    print("=" * 70)
    h = 1e-4

    def fd_grad(Lfn, theta):
        g = []
        for i in range(3):
            tp = theta.at[i].add(h)
            tm = theta.at[i].add(-h)
            g.append((float(Lfn(tp)) - float(Lfn(tm))) / (2 * h))
        return np.array(g)

    # unrolled n_iter=30 as the AD reference, FD on the same
    L_un = make_unrolled_loss(30)
    fd = fd_grad(L_un, THETA0)
    ad_un = np.asarray(jax.grad(L_un)(THETA0))
    L1, _ = make_proto1_loss(30, residual_R, solve_lstsq)
    g_p1 = np.asarray(jax.grad(L1)(THETA0))
    g_p1_Rf, _ = make_proto1_loss(30, residual_Rf, solve_lstsq)
    g_p1_Rf = np.asarray(jax.grad(g_p1_Rf)(THETA0))
    g_p2 = np.asarray(jax.grad(L2)(THETA0))

    labels = ["d/d_alpha_imf", "d/d_delta", "d/d_W0"]
    print(f"  {'param':16s} {'FD':>14s} {'unrolled-AD':>14s} {'P1(R)':>14s} {'P1(Rf)':>14s} {'P2':>14s}")
    for i, lab in enumerate(labels):
        print(f"  {lab:16s} {fd[i]:14.6e} {ad_un[i]:14.6e} {g_p1[i]:14.6e} {g_p1_Rf[i]:14.6e} {g_p2[i]:14.6e}")

    def relerr(a, b):
        return np.max(np.abs(a - b) / (np.abs(b) + 1e-30))
    print(f"\n  rel-err vs FD:  unrolled-AD={relerr(ad_un, fd):.2e}  "
          f"P1(R)={relerr(g_p1, fd):.2e}  P1(Rf)={relerr(g_p1_Rf, fd):.2e}  P2={relerr(g_p2, fd):.2e}")

    # ----- Adaptive forward iter counts across box corners -----
    print("\n" + "=" * 70)
    print("ADAPTIVE FORWARD iter-counts across box corners (tol=1e-4, cap=60)")
    print("=" * 70)
    corners = [
        ("representative", 2.3, 0.4, 5.0),
        ("hard A (2.7,0.6,7)", 2.7, 0.6, 7.0),
        ("hard B (1.9,0.1,4)", 1.9, 0.1, 4.0),
    ]
    for name, ai, d, W in corners:
        it, resid = adaptive_iter_count(ai, d, W, 1e-4, 60)
        below_2e3 = resid < 2e-3
        print(f"  {name:22s}: iters={it:3d}  final_resid={resid:.3e}  (<2e-3: {below_2e3})")

    # ----- Linear-solve handler comparison (correctness + stability) -----
    print("\n" + "=" * 70)
    print("LINEAR-SOLVE HANDLER: grad rel-err vs FD + NaN check, hard corner B")
    print("=" * 70)
    theta_hardB = jnp.array([1.9, 0.1, 4.0])
    fd_hB = fd_grad(make_unrolled_loss(40), theta_hardB)
    for name, solver in [("lstsq", solve_lstsq), ("pinv", solve_pinv),
                         ("tikhonov", solve_tikhonov)]:
        L1s, _ = make_proto1_loss(40, residual_R, solver)
        try:
            g = np.asarray(jax.grad(L1s)(theta_hardB))
            re = np.max(np.abs(g - fd_hB) / (np.abs(fd_hB) + 1e-30))
            nan = (not np.all(np.isfinite(g)))
            print(f"  {name:10s}: grad={g}  relerr_vs_FD={re:.2e}  NaN/Inf={nan}")
        except Exception as e:
            print(f"  {name:10s}: FAILED {e}")

    print("\nFD ref (hardB, unrolled n_iter=40):", fd_hB)


if __name__ == "__main__":
    run()
