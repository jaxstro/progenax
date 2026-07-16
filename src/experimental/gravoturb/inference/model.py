r"""The single-source-of-truth inference model factory: :func:`build_logdensity`.

:func:`build_logdensity` is the shared, **prior-aware** unconstrained log-density factory used
by BOTH the SBC driver (:mod:`gravoturb.inference.sbc`) and AC16
(:func:`gravoturb.validation.acceptance.ac16_hmc_recovery`). It composes (over a constrained
``z`` reparametrization)::

    logdensity(z) = tail_exceedance_loglike(...)               # POT tail -> alpha
                  + sum_c log_count_variance_loglike(meas_v_c, # tail-robust CIC sigma_s^2
                        ..., var_v_c)                          #   -> mach (scalar; weak beta)
                  + bandpower_gaussian_loglike(P_pred-bp, ...) # 2-pt band powers -> beta
                  + prior.logpdf([M, alpha, beta])             # proper BM19 prior
                  + log_jacobian(z)                            # reparam Jacobian

The **2-pt band-power block** is the beta carrier. The single-scalar log-count variance (even
at multiple cell scales) barely constrains the GRF slope beta -- it integrates the spectrum to
one number, so SBC found beta UNDER-constrained (rank-uniformity fails). The design-intended
fix is the field-level log-density power-spectrum band-powers (à la AC15): predicted analytic
band-powers ``P_pred(theta)`` (:func:`power_spectrum_bandpowers`, differentiable, the beta
carrier) fit to the measured periodogram ``band_powers`` of the latent log-density field via a
Gaussian ``-1/2 r^T Cinv r`` with a FIXED-FIDUCIAL Hartlap precision ``bp_precision`` (no
log|C| term, exactly like :func:`gaussian_loglike`). This is a FIELD-LEVEL UPPER BOUND on beta
information: the band-powers are measured from the continuous log-density field, with NO star
shot noise on the 2-pt (cf. the AC15 scoping caveat). ``bp_precision`` is computed ONCE at a
FIXED fiducial theta (NOT the trial truth), so -- like ``var_v`` -- it is a truth-independent
constant and does not break SBC.

The stellar CIC block is the **tail-robust log-count-variance** statistic
(:func:`log_count_variance_loglike`, Task 7) -- a Gaussian fit of the measured
``Var_cells[log_plus(N_cell)]`` to its analytic prediction with a FIXED-FIDUCIAL estimator
variance ``var_v``. It REPLACES the per-cell ``count_loglike`` count-histogram block, which
was tail-sensitive and biased mach high (the AC18 ℳ-bias). Crucially, ``var_v`` is computed
ONCE per inference at a fixed fiducial theta (NOT at the trial truth ``theta*``) so it is a
truth-independent constant -- a truth-keyed var_v would be exactly the kind of SBC artifact
the old POT validity barrier was.

AC16 originally carried a *flat-in-theta* (improper) prior; SBC REQUIRES a proper prior, so
the prior term is part of this shared factory. AC16 is refactored to call this factory with
its own :class:`BM19Prior` so there is a single source of truth for the composition.

**No POT-validity barrier.** Earlier versions added ``pot_validity_barrier(theta, s_thr)`` to
keep ``s_t(theta) <= s_thr``. With the per-trial threshold ``s_thr = s_t(theta*) + margin``
that barrier is a *trial-dependent prior keyed to the truth* (it is absent from the SBC
sampling prior and from mock generation), which chops the upper-alpha posterior asymmetrically
around the truth and skews SBC ranks high at high Mach -- a driver artifact, not an engine
fault. Because ``tail_exceedance_loglike`` is **shift-immune in s_thr** (the lognormal norm
cancels, so the truth-keyed threshold does NOT bias alpha through the likelihood) and the
proper :class:`BM19Prior` already bounds ``(M, alpha, beta)``, the barrier is dropped. SBC
validity is then verified empirically by AC18's rank-uniformity test.
"""

from typing import Callable

import jax.numpy as jnp

from gravoturb.inference.covariance import power_spectrum_bandpowers
from gravoturb.inference.hmc import log_jacobian, to_constrained
from gravoturb.inference.likelihood import (
    log_count_variance_loglike,
    tail_exceedance_loglike,
)
from gravoturb.inference.priors import BM19Prior

# Fixed |k| band-power edges for the field-level 2-pt block (the beta carrier). 5 bins on the
# 24^3 stellar grid the count model FFTs on. NB: AC15/data_vector use jnp.linspace(2.0, 11.0, 5)
# (4 bins); we use one more bin (6 edges -> 5 bins) to give beta a touch more 2-pt leverage on
# the small grid -- harmless, since bp_precision is the matched mock covariance of the SAME edges.
K_EDGES = jnp.linspace(2.0, 11.0, 6)


def build_logdensity(
    prior: BM19Prior,
    data: dict,
    b: float,
    s_thr: float,
    s_max: float,
    shape: tuple[int, int, int],
    cell_sizes: tuple[int, ...],
    bp_precision,
    n_max: int = 10,
    n_s: int = 400,
) -> Callable:
    r"""Build the shared prior-aware unconstrained log-density closure (see module docstring).

    Parameters
    ----------
    prior : BM19Prior
        Proper prior over the 3 free params ``[M, alpha, beta]`` (``b`` is fixed and outside it).
    data : dict
        Per-trial mock data bundle with keys:
        ``"exc_counts"`` (nb,), ``"exc_edges"`` (nb+1,) -- the gas-tail POT histogram from
        :func:`measure_exceedances`; ``"log_count_vars"`` -- tuple of per-cell measured
        ``Var_cells[log_plus(N)]`` scalars (:func:`measure_log_count_variance`, Task 7);
        ``"var_vs"`` -- tuple of per-cell FIXED-FIDUCIAL estimator variances (threaded in from
        :func:`gravoturb.inference.sbc.sbc_ranks`, computed ONCE at a fixed fiducial theta, NOT
        at the trial truth -> SBC-valid); ``"n_bars"`` -- tuple of per-cell ``n_bar`` (matched
        to ``cell_sizes``); ``"band_powers"`` -- (k,) measured periodogram band-powers of the
        latent log-density field on ``K_EDGES`` (the 2-pt beta channel;
        :func:`measured_bandpowers`).
    b : float
        Fixed turbulence driving parameter (the likelihood constant in ``theta4``).
    s_thr, s_max : float
        POT threshold and realized field maximum (data-derived constants; same field as
        ``exc_counts``, which makes the POT block shift-immune).
    shape : (n, n, n)
        Stellar CIC grid the count model FFTs on (forward-bias-matched). The band-power block
        FFTs on this SAME grid (``K_EDGES``).
    cell_sizes : tuple[int, ...]
        CIC cell sizes; ``data["log_count_vars"]`` / ``data["var_vs"]`` / ``data["n_bars"]``
        align with this.
    bp_precision : (k, k) array
        FIXED-FIDUCIAL Hartlap-corrected band-power precision (:func:`mock_precision` over an
        ensemble at the fixed fiducial theta; threaded in from
        :func:`gravoturb.inference.sbc.sbc_ranks`). Truth-independent constant (like ``var_v``)
        -> SBC-valid. numpy/jnp both ok (fixed data).
    n_max, n_s : int
        ``log_count_variance_loglike`` quadrature controls (mirror AC16 defaults). ``n_max`` also
        sets the Gaussianization order of the predicted band-powers.

    Returns
    -------
    logdensity_fn : z (3,) -> scalar log-density (differentiable in z).
    """
    exc_counts = jnp.asarray(data["exc_counts"])
    exc_edges = jnp.asarray(data["exc_edges"])
    log_count_vars = tuple(jnp.asarray(mv) for mv in data["log_count_vars"])
    var_vs = tuple(jnp.asarray(vv) for vv in data["var_vs"])
    n_bars = tuple(float(nb) for nb in data["n_bars"])
    band_powers = jnp.asarray(data["band_powers"])  # measured 2-pt band powers (beta channel)
    bp_precision = jnp.asarray(bp_precision)         # fixed-fiducial Hartlap precision (k, k)

    def logdensity(z):
        m_, a_, be_ = to_constrained(z)
        theta3 = jnp.array([m_, a_, be_])
        theta4 = jnp.array([m_, b, a_, be_])
        # POT tail block -> alpha. SHIFT-IMMUNE in s_thr (the lognormal norm
        # cancels), so a per-trial s_thr keyed to the truth does NOT bias alpha.
        ll = tail_exceedance_loglike(exc_counts, exc_edges, theta4, s_thr, s_max)
        # tail-robust stellar CIC blocks (log-count variance) -> mach (scalar sigma_s^2; only a
        # weak beta dependence). var_v is the FIXED-FIDUCIAL estimator variance (truth-
        # independent; see build_logdensity docstring).
        for c, mv, vv, nb in zip(cell_sizes, log_count_vars, var_vs, n_bars):
            ll = ll + log_count_variance_loglike(
                mv, theta4, shape, c, nb, vv, n_max=n_max, n_s=n_s
            )
        # field-level 2-pt band-power block -> beta. Predicted analytic band-powers (the beta
        # carrier; differentiable) vs the measured periodogram, Gaussian with the fixed-fiducial
        # precision -- same -1/2 r^T Cinv r pattern as gaussian_loglike (no log|C| term). This is
        # the channel the single-scalar log-count variance lacked; it is a FIELD-LEVEL UPPER
        # BOUND (no star shot noise on the 2-pt; see build_logdensity docstring).
        _, P_pred, _ = power_spectrum_bandpowers(
            shape, be_, m_, b, a_, K_EDGES, n_max=n_max
        )
        r = P_pred - band_powers
        ll = ll + (-0.5 * r @ (bp_precision @ r))
        # proper prior (replaces AC16's flat-in-theta improper prior)
        ll = ll + prior.logpdf(theta3)
        # NB: NO pot_validity_barrier here. The barrier penalizes draws with
        # s_t(theta) > s_thr, and with the per-trial s_thr = s_t(theta*)+margin it
        # is a *trial-dependent prior* (keyed to the truth) absent from the SBC
        # sampling prior -- it would chop the upper-alpha posterior asymmetrically
        # around the truth and skew SBC ranks high at high Mach (a driver artifact,
        # not an engine fault). The proper BM19Prior already bounds (M, alpha, beta),
        # and the POT likelihood is well-defined without it, so it is dropped. See
        # the SBC-validity note in the per-paper Talts/Sailynoja docs.
        # reparametrization Jacobian (sampling in unconstrained z)
        return ll + log_jacobian(z)

    return logdensity
