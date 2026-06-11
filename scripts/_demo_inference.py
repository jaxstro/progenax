r"""Shared physics-direct inference helper for the progenax Batch-B science demos.

A scripts-local (NOT a packaged API) likelihood layer used by the ``scripts/``
demos that fit a forward kinematic model to mock cluster snapshots. It provides:

* binned summary statistics with honest standard errors:
  - :func:`binned_sigma1d`  -- per-group 1-D velocity dispersion sigma_1d(r),
  - :func:`binned_sigma_beta` -- sigma_1d(r) plus the Binney anisotropy beta(r);
* a Gaussian (chi^2) log-likelihood factory :func:`gaussian_loglike`;
* a fixed-step Adam MLE :func:`mle_adam` and a Fisher covariance
  :func:`fisher_cov`;
* bounded logit/expit reparametrizations for box-constrained parameters.

JAX-native throughout (``jax.numpy``, ``jax.lax``, ``optax``); every public
function is jit/grad-safe so the demos can differentiate through the forward
model. float64 is the demos' responsibility (they call
``jax.config.update("jax_enable_x64", True)`` before importing this).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


# --------------------------------------------------------------------------- #
# Binned summary statistics
# --------------------------------------------------------------------------- #
def _bin_index(radii, r_edges):
    """Radial bin index in [0, K) for each radius, or -1 if outside [r0, rK].

    ``jnp.searchsorted`` with ``side='right'`` gives 1..K for in-range radii; we
    shift to 0..K-1 and mark out-of-range radii (index 0 or K+1 after shift) -1.
    """
    k = jnp.searchsorted(r_edges, radii, side="right") - 1
    n_bins = r_edges.shape[0] - 1
    in_range = (radii >= r_edges[0]) & (radii <= r_edges[-1])
    return jnp.where(in_range, jnp.clip(k, 0, n_bins - 1), -1)


def _grouped_bin_sums(values, group_ids, bin_ids, n_groups, n_bins):
    """Sum ``values`` and count members into a (n_groups, n_bins) grid.

    Valid samples have ``bin_ids >= 0``; invalid ones are routed to a throwaway
    overflow cell so no Python control flow is needed (fully vmappable)."""
    valid = bin_ids >= 0
    g = jnp.where(valid, group_ids, n_groups)  # overflow row
    b = jnp.where(valid, bin_ids, 0)
    flat = g * n_bins + b
    size = (n_groups + 1) * n_bins
    sums = jnp.zeros(size).at[flat].add(jnp.where(valid, values, 0.0))
    counts = jnp.zeros(size).at[flat].add(jnp.where(valid, 1.0, 0.0))
    sums = sums[: n_groups * n_bins].reshape(n_groups, n_bins)
    counts = counts[: n_groups * n_bins].reshape(n_groups, n_bins)
    return sums, counts


def binned_sigma1d(pos, vel, group_ids, n_groups, r_edges, n_min=30):
    r"""Per-group binned 1-D velocity dispersion sigma_1d(r).

    Isotropic estimator ``sigma_1d^2 = <|v|^2> / 3`` within each (group j,
    radial bin k). Standard error of the dispersion estimate is
    ``se = sig_hat / sqrt(6 n)``. The estimator pools all 3 velocity
    components, so the inner sum runs over ``3n`` iid one-D squared normals:
    ``sum |v|^2 / sigma^2 ~ chi^2(3n)``, giving ``Var(sigma_hat^2) = 2 sigma^4
    / (3n)`` and (delta method) ``Var(sigma_hat) ~ sigma^2 / (6n)``. The
    effective sample size is ``3n``, not ``n`` -- hence ``sqrt(6 n)`` rather
    than the single-component ``sqrt(2 n)``.

    Parameters
    ----------
    pos, vel : (N, 3) arrays
    group_ids : (N,) int in [0, n_groups)
    r_edges : (K+1,) monotone radial bin edges
    n_min : bins with fewer than ``n_min`` members get weight 0 (not NaN).

    Returns
    -------
    sig_hat, se, weight, n : each (n_groups, K). ``weight`` is 1.0 for populated
    bins (n >= n_min) and 0.0 otherwise; outputs are NaN-free everywhere.
    """
    n_bins = r_edges.shape[0] - 1
    radii = jnp.linalg.norm(pos, axis=1)
    speed_sq = jnp.sum(vel * vel, axis=1)
    bin_ids = _bin_index(radii, r_edges)

    sum_v2, counts = _grouped_bin_sums(speed_sq, group_ids, bin_ids, n_groups, n_bins)
    populated = counts >= n_min
    safe_counts = jnp.where(counts > 0, counts, 1.0)  # avoid /0 in empty bins

    sigma_sq = jnp.where(populated, sum_v2 / safe_counts / 3.0, 0.0)
    sig_hat = jnp.sqrt(sigma_sq)
    se = jnp.where(populated, sig_hat / jnp.sqrt(6.0 * safe_counts), 0.0)
    weight = jnp.where(populated, 1.0, 0.0)
    return sig_hat, se, weight, counts


class SigmaBetaResult(NamedTuple):
    """Return of :func:`binned_sigma_beta`; all arrays shaped (C, K)."""

    sig_hat: jax.Array   # 1-D dispersion sqrt((sigma_r^2 + sigma_t^2)/3)
    se: jax.Array        # SE of sig_hat = sig_hat / sqrt(6 n) (3 pooled components)
    beta_hat: jax.Array  # Binney anisotropy 1 - sigma_t^2 / (2 sigma_r^2)
    weight: jax.Array    # 1.0 if populated (n >= n_min) else 0.0
    n: jax.Array         # member counts


def binned_sigma_beta(pos, vel, r_edges, component_id=None, n_min=50):
    r"""sigma_1d(r) and Binney anisotropy beta(r) per component.

    Radial/tangential split: ``v_r = v . r_hat``; ``v_t^2 = |v|^2 - v_r^2``
    (two tangential dof). With ``sigma_r^2 = <v_r^2>`` and ``sigma_t^2 = <v_t^2>``
    (summing BOTH tangential components),
    ``beta = 1 - sigma_t^2 / (2 sigma_r^2)`` and
    ``sig_hat = sqrt((sigma_r^2 + sigma_t^2) / 3)``.

    The standard error of ``sig_hat`` is ``se = sig_hat / sqrt(6 n)``: all 3
    velocity components are pooled into ``sig_hat^2``, so the inner sum is over
    ``3n`` iid one-D squared normals (``chi^2(3n)``), giving effective sample
    size ``3n`` (delta method ``Var(sig_hat) ~ sigma^2 / (6n)``), not ``n``.

    ``component_id=None`` treats all stars as a single component (C = 1).
    Otherwise ``component_id`` must be a CONCRETE (static) array: ``n_comp`` is
    read host-side via ``int(component_id.max()) + 1``, so this function is NOT
    traceable over a traced ``component_id`` (unlike :func:`binned_sigma1d`,
    which takes ``n_groups`` explicitly and stays fully traceable).

    Returns a :class:`SigmaBetaResult` namedtuple (``sig_hat, se, beta_hat,
    weight, n``), each (C, K), NaN-free.
    """
    n = pos.shape[0]
    if component_id is None:
        component_id = jnp.zeros((n,), dtype=jnp.int32)
        n_comp = 1
    else:
        n_comp = int(component_id.max()) + 1

    n_bins = r_edges.shape[0] - 1
    radii = jnp.linalg.norm(pos, axis=1)
    r_hat = pos / jnp.maximum(radii, 1e-30)[:, None]
    v_r = jnp.sum(vel * r_hat, axis=1)
    v_r_sq = v_r * v_r
    v_t_sq = jnp.sum(vel * vel, axis=1) - v_r_sq

    bin_ids = _bin_index(radii, r_edges)
    sum_vr2, counts = _grouped_bin_sums(v_r_sq, component_id, bin_ids, n_comp, n_bins)
    sum_vt2, _ = _grouped_bin_sums(v_t_sq, component_id, bin_ids, n_comp, n_bins)

    populated = counts >= n_min
    safe_counts = jnp.where(counts > 0, counts, 1.0)
    sigma_r_sq = jnp.where(populated, sum_vr2 / safe_counts, 0.0)
    sigma_t_sq = jnp.where(populated, sum_vt2 / safe_counts, 0.0)

    sig_hat = jnp.sqrt(jnp.where(populated, (sigma_r_sq + sigma_t_sq) / 3.0, 0.0))
    safe_sigma_r_sq = jnp.where(sigma_r_sq > 0, sigma_r_sq, 1.0)
    beta_hat = jnp.where(populated, 1.0 - sigma_t_sq / (2.0 * safe_sigma_r_sq), 0.0)
    se = jnp.where(populated, sig_hat / jnp.sqrt(6.0 * safe_counts), 0.0)
    weight = jnp.where(populated, 1.0, 0.0)
    return SigmaBetaResult(sig_hat, se, beta_hat, weight, counts)


# --------------------------------------------------------------------------- #
# Gaussian (chi^2) likelihood
# --------------------------------------------------------------------------- #
def gaussian_loglike(data, predict_fn):
    r"""Build a weighted-Gaussian log-likelihood closure.

    ``data = (sig_hat, se, weight)`` of matching shape. The returned
    ``loglike(theta) = -0.5 * sum( weight * ((sig_hat - predict_fn(theta)) / se)^2 )``,
    where ``predict_fn(theta)`` returns model sigma on the same shape as
    ``sig_hat``. Differentiable in ``theta`` (gradient flows through
    ``predict_fn`` only). ``se`` is guarded against 0 on zero-weight bins.
    """
    sig_hat, se, weight = data
    safe_se = jnp.where(se > 0, se, 1.0)

    def loglike(theta):
        resid = (sig_hat - predict_fn(theta)) / safe_se
        return -0.5 * jnp.sum(weight * resid * resid)

    return loglike


# --------------------------------------------------------------------------- #
# Optimization + Fisher
# --------------------------------------------------------------------------- #
def mle_adam(negloglike, z0, n_steps=400, lr=3e-2):
    r"""Minimize ``negloglike(z)`` with optax Adam over a fixed ``jax.lax.scan``.

    Deterministic (fixed ``n_steps``), jit/grad-safe. The ``value_and_grad`` +
    update step is the ``scan`` body, so it is traced and fused by XLA; the
    caller is expected to ``jax.jit`` the outer loss/driver (this function adds
    no explicit ``jax.jit`` of its own). ``z0`` is the unconstrained init.

    Returns ``(z_hat, loss_trace)`` where ``loss_trace`` has shape
    ``(n_steps,)``. ``loss_trace[k]`` is the loss BEFORE update step ``k`` (the
    loss at the iterate that step ``k`` then updates). Consequently the loss of
    the FINAL returned ``z_hat`` (after the last update) is NOT recorded -- a
    caller's plateau / convergence check should account for that one-step lag.
    """
    optimizer = optax.adam(lr)
    value_and_grad = jax.value_and_grad(negloglike)

    def step(carry, _):
        z, opt_state = carry
        loss, grads = value_and_grad(z)
        updates, opt_state = optimizer.update(grads, opt_state, z)
        z = optax.apply_updates(z, updates)
        return (z, opt_state), loss

    (z_hat, _), trace = jax.lax.scan(
        step, (z0, optimizer.init(z0)), None, length=n_steps
    )
    return z_hat, trace


def fisher_cov(negloglike, z_hat):
    r"""Parameter covariance from the inverse Hessian of ``negloglike`` at ``z_hat``.

    The Hessian (observed Fisher information for a negative-log-likelihood) is
    symmetrized, then inverted. Raises :class:`ValueError` if it is not positive
    definite (checked via the eigenvalues) -- we report rather than mask a
    degenerate / saddle fit.
    """
    hess = jax.hessian(negloglike)(z_hat)
    hess = jnp.atleast_2d(hess)
    hess_sym = 0.5 * (hess + hess.T)
    eigvals = jnp.linalg.eigvalsh(hess_sym)
    if not bool(jnp.all(eigvals > 0)):
        raise ValueError(
            f"Hessian of negloglike at z_hat is not positive definite "
            f"(eigenvalues {jnp.asarray(eigvals)}); Fisher covariance is undefined."
        )
    return jnp.linalg.inv(hess_sym)


def fisher_information_gn(residual_fn, z_hat, extra_negloglike=None):
    r"""Gauss-Newton Fisher information at ``z_hat`` (reverse-mode only).

    For a Gaussian negative-log-likelihood ``-1/2 sum r_i(z)^2`` with a
    standardized residual vector ``r`` (already divided by the per-cell SE and
    with masked cells zeroed), the EXPECTED Fisher information is the
    Gauss-Newton form ``F = J^T J`` with ``J = d r / d z``. The dropped term
    ``sum r_i d^2 r_i`` vanishes in expectation (E[r]=0 at the truth) and is pure
    noise at the optimum, so JtJ is the correct (expected-information) Fisher.

    ``J`` is computed with :func:`jax.jacrev` (REVERSE mode): the B2 residual
    runs the diffrax ODE inside ``from_imf``, whose ``custom_vjp`` makes
    ``jax.hessian`` / ``jax.jacfwd`` crash ("can't apply forward-mode autodiff
    to a custom_vjp"). Reverse-mode is safe through that ``custom_vjp``.

    Parameters
    ----------
    residual_fn : callable z -> (M,) array
        Standardized residual vector (divided by SE; masked cells zeroed so they
        contribute 0 to ``J^T J``).
    z_hat : (P,) array
        Point at which to evaluate the information (the MLE).
    extra_negloglike : callable z -> scalar, optional
        An ADDITIONAL negative-log-likelihood term that is ODE-free (hence
        ``jax.hessian``-safe). Its (P, P) Hessian at ``z_hat`` is added to
        ``J^T J`` -- used for the B2 mass channel
        ``-sum Maschberger(alpha(z)).logpdf(m_obs)``.

    Returns
    -------
    F : (P, P) array
        ``J^T J`` (+ ``Hess(extra_negloglike)`` if given), symmetrized.
    """
    J = jax.jacrev(residual_fn)(z_hat)          # (M, P), reverse-mode (ODE-safe)
    F = J.T @ J
    if extra_negloglike is not None:
        F = F + jax.hessian(extra_negloglike)(z_hat)  # ODE-free -> hessian OK
    return 0.5 * (F + F.T)


def constrained_cov(F_z, dtheta_dz):
    r"""Delta-method map of an unconstrained covariance to constrained space.

    Given the unconstrained Fisher information ``F_z`` (P, P) and the per-component
    Jacobian diagonal ``dtheta_dz`` (P,) of the box reparametrization
    ``theta_i = expit(z_i, lo_i, hi_i)`` (so ``G = diag(dtheta_dz)``), returns
    ``cov_theta = G cov_z G^T`` where ``cov_z = F_z^{-1}``.

    Raises :class:`ValueError` if ``F_z`` is not positive definite (reported, not
    masked -- a degenerate / saddle fit must surface).
    """
    F_z = 0.5 * (F_z + F_z.T)
    eigvals = jnp.linalg.eigvalsh(F_z)
    if not bool(jnp.all(eigvals > 0)):
        raise ValueError(
            f"Fisher information F_z is not positive definite "
            f"(eigenvalues {jnp.asarray(eigvals)}); covariance is undefined."
        )
    cov_z = jnp.linalg.inv(F_z)
    G = jnp.diag(dtheta_dz)
    return G @ cov_z @ G.T


# --------------------------------------------------------------------------- #
# Bounded reparametrizations (box [lo, hi])
# --------------------------------------------------------------------------- #
def logit(x, lo, hi):
    r"""Constrained ``x in (lo, hi)`` -> unconstrained ``z = log((x-lo)/(hi-x))``."""
    return jnp.log((x - lo) / (hi - x))


def expit(z, lo, hi):
    r"""Unconstrained ``z`` -> constrained ``x = lo + (hi - lo) * sigmoid(z)`` in (lo, hi).

    The sigmoid is clamped a tiny epsilon away from {0, 1} so that even at
    saturating ``|z|`` (where float ``sigmoid`` rounds to exactly 0 or 1) the
    result stays strictly inside the OPEN interval ``(lo, hi)``.
    """
    eps = jnp.finfo(jnp.result_type(float)).eps
    s = jnp.clip(jax.nn.sigmoid(z), eps, 1.0 - eps)
    return lo + (hi - lo) * s
