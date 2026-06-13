"""Shared AD-vs-FD gradient-audit engine (single source of truth for the gate + the website).

audit_entry_point(case) returns an AuditResult whose `status` is COMPUTED from
(expect, finite, |ratio-1|<tol, |ad|>eps) -- never hand-set. The same results feed the pytest
gate (tests/validation/test_grad_audit.py) and scripts/audit_gradients.py -> results.json.
Reverse-mode jax.grad only (ODE custom_vjp-safe, mirrors _demo_inference.fisher_information_gn).
"""
from dataclasses import dataclass
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp

Direction = Literal["params->IC", "params->summary"]
Expect = Literal["consistent", "known_zero", "known_blocked"]


@dataclass(frozen=True)
class EdgeConfig:
    """A curated boundary probe for a Case (e.g. W0=12, alpha=1.0)."""
    label: str                       # appears in the case id, e.g. "W0=12"
    theta0: float                    # the edge parameter value
    hazard_id: str | None = None     # links to the hazard map; set if it probes a suspect
    tol: float | None = None         # per-edge tolerance override
    expect: Expect | None = None     # per-edge expect override (e.g. alpha=1.0 -> known_blocked)


@dataclass(frozen=True)
class Case:
    id: str
    direction: Direction
    fn: Callable[[jax.Array], jax.Array]   # theta (scalar) -> output array
    param: str
    theta0: float
    reduce: Callable[[jax.Array], jax.Array] = jnp.sum   # output -> scalar
    expect: Expect = "consistent"
    tol: float = 1e-3
    h_rel: float = 1e-4
    eps: float = 1e-9                        # |AD| silent-zero threshold
    edges: Tuple[EdgeConfig, ...] = ()


@dataclass(frozen=True)
class AuditResult:
    id: str
    direction: str
    param: str
    theta: float
    finite: bool
    ad: float
    fd: float
    ratio: float
    abs_ad: float
    expect: str
    tol: float
    status: str          # clean | known-limitation | hazard


def _scalar(case: Case, theta: jax.Array) -> jax.Array:
    return case.reduce(case.fn(theta))


def _classify(expect, finite, ad, fd, ratio, tol, eps) -> str:
    if expect == "known_zero":
        # Design D2: BOTH |AD|~0 AND |FD|~0 for a known-limitation. AD~0 alone is not
        # enough -- a live FD with blocked AD means the value genuinely moves with the
        # param while the gradient is silently zero, the unannounced-change detector's
        # headline catch, so it is a hazard. (known_zero cases must pick theta0 off any
        # grid-node crossing so FD~0 holds for the genuinely-constant quantity.)
        return "known-limitation" if (abs(ad) < eps and abs(fd) < eps) else "hazard"
    if expect == "known_blocked":
        return "known-limitation" if finite else "hazard"
    # consistent
    if finite and abs(ad) > eps and abs(ratio - 1.0) < tol:
        return "clean"
    return "hazard"


def audit_entry_point(case: Case, theta: float | None = None,
                      tol: float | None = None, expect: str | None = None) -> AuditResult:
    theta = case.theta0 if theta is None else theta
    tol = case.tol if tol is None else tol
    expect = case.expect if expect is None else expect

    from typing import get_args
    assert (expect if expect is not None else case.expect) in get_args(Expect), \
        f"unknown expect class: {expect!r}"

    t = jnp.asarray(theta, dtype=jnp.float64)
    ad = float(jax.grad(lambda x: _scalar(case, x))(t))

    h = case.h_rel * max(abs(float(theta)), 1.0)
    g = lambda x: float(_scalar(case, jnp.asarray(x, dtype=jnp.float64)))
    fd = (g(float(theta) + h) - g(float(theta) - h)) / (2.0 * h)

    finite = bool(jnp.isfinite(jnp.asarray(ad)))
    if fd != 0.0:
        ratio = ad / fd
    else:
        ratio = 1.0 if ad == 0.0 else float("inf")
    status = _classify(expect, finite, ad, fd, ratio, tol, case.eps)
    return AuditResult(case.id, case.direction, case.param, float(theta), finite,
                       ad, fd, ratio, abs(ad), expect, tol, status)
