"""Registered source/derived relations versus unchanged environment kernels.

These are transcription, algebra and local implementation checks. They do not
qualify empirical extrapolation or derivatives at hard/clipping boundaries.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp
from jaxstro.registry import load_source
from jaxstro.registry.symbolic import build_equation

from progenax.imf.environment.density import (
    compute_log_rho_cl_6,
    compute_r_half,
    compute_rho_ecl,
)
from progenax.imf.environment.mapping import (
    alpha3_jerabkova_mecl,
    alpha3_jerabkova_rho,
    alpha3_marks_plane,
    alpha3_marks_table3,
    lowmass_slopes_metallicity,
    x_hat_marks_plane,
    x_jerabkova_generalized,
    x_jerabkova_rho,
)
from progenax.registry import registry_root


@pytest.fixture(scope="module")
def equations():
    records = {}
    for source in (
        "marks2012topheavy",
        "marks2014erratum",
        "marks2012inverse",
        "jerabkova2018igimf",
        "progenaxderivations",
    ):
        bundle = load_source(registry_root(), source)
        for name in bundle.equations:
            if name.startswith(("EQ-MARKS-", "EQ-MK-", "EQ-JERABKOVA-", "EQ-ENV-")):
                records[name] = build_equation(bundle, name)
    return records


def test_density_substitution_and_mass_rounding(equations):
    """Independent substitution derives .61/.6039 and isolates the constant gap."""
    radius = equations["EQ-MK-RADIUS-NOMINAL"]
    density = equations["EQ-MK-DENSITY"]
    log_density = equations["EQ-ENV-LOG-DENSITY"]
    exact = equations["EQ-ENV-X-MASS-EXACT"]
    rounded = equations["EQ-ENV-X-MASS-ROUNDED"]
    mecl = equations["EQ-ENV-X-MECL"]
    rho_coordinate = equations["EQ-JERABKOVA-X-RHO"]
    mass, height = density.symbols["m_env"], density.symbols["h_env"]
    L, eps = log_density.symbols["L_env"], log_density.symbols["eps_env"]
    substituted_density = density.expression.subs(height, radius.expression)
    printed_density = equations["EQ-JERABKOVA-DENSITY-PRINTED"]
    assert (
        sp.simplify(
            printed_density.expression.subs(
                printed_density.symbols["mcl_env"], mass / eps
            )
            - 2 * density.expression / eps
        )
        == 0
    )
    derived_log = (
        sp.log(substituted_density.subs(mass, 10 ** (L + 6)) / eps) / sp.log(10) - 6
    )
    assert (
        sp.simplify(sp.expand_log(derived_log, force=True) - log_density.expression)
        == 0
    )
    assert (
        sp.simplify(
            rho_coordinate.expression.subs(
                rho_coordinate.symbols["R_env"], log_density.expression
            )
            - exact.expression
        )
        == 0
    )
    assert (
        sp.simplify(
            rounded.expression.subs(eps, sp.Rational(33, 100)) - mecl.expression
        )
        == 0
    )
    # Expand logarithms before collecting terms; generic simplify may combine
    # rational log coefficients into huge integer powers unnecessarily.
    difference = sp.expand(
        sp.expand_log(rounded.expression - exact.expression, force=True)
    )
    assert not difference.free_symbols
    assert difference != 0  # Do not call the rounded mass and density routes identical.
    for name in ("L_env", "z_env", "eps_env"):
        assert sp.simplify(sp.diff(difference, rounded.symbols[name])) == 0
    printed = equations["EQ-JERABKOVA-X-MASS-PRINTED"]
    # At reference mass, zero metallicity and reference SFE, a 4pi conversion
    # adds .99 log10(2); it does not produce the printed Eq.9 constant.
    C8 = exact.expression.subs(
        {
            exact.symbols["L_env"]: 0,
            exact.symbols["z_env"]: 0,
            eps: sp.Rational(33, 100),
        }
    )
    printed_constant = printed.expression.subs(
        {printed.symbols["L_env"]: 0, printed.symbols["z_env"]: 0}
    )
    assert (
        sp.expand(
            sp.expand_log(
                printed_constant - C8 - sp.Rational(99, 100) * sp.log(2, 10), force=True
            )
        )
        != 0
    )


def test_density_kernels_and_derivatives(equations, tolerances):
    """Solar/parsec numerical units match the normalized registered quantities."""
    assert jax.config.jax_enable_x64
    mass = jnp.array([1e4, 1e5, 1e6], dtype=jnp.float64)
    radius = equations["EQ-MK-RADIUS-NOMINAL"]
    density = equations["EQ-MK-DENSITY"]
    expr = density.expression.subs(density.symbols["h_env"], radius.expression)
    m = radius.symbols["m_env"]
    for kernel, expression in (
        (compute_r_half, radius.expression),
        (compute_rho_ecl, expr),
    ):
        oracle = sp.lambdify(m, expression, "numpy")
        derivative = sp.lambdify(m, sp.diff(expression, m), "numpy")
        for evaluated in (kernel, jax.jit(kernel)):
            np.testing.assert_allclose(
                evaluated(mass), oracle(np.asarray(mass)), rtol=tolerances.EXACT, atol=0
            )
            for ad in (jax.jacfwd, jax.jacrev):
                np.testing.assert_allclose(
                    ad(evaluated)(mass),
                    np.diag(derivative(np.asarray(mass))),
                    rtol=tolerances.EXACT,
                    atol=tolerances.EXACT,
                )


@pytest.mark.parametrize("route", ["density", "marks", "generalized", "mecl"])
def test_coordinate_kernels_and_first_derivatives(equations, tolerances, route):
    """Retained inputs; derivative oracles are symbolic host expressions."""
    point = jnp.array([-1.5, 0.0, 0.33], dtype=jnp.float64)
    if route == "density":
        record = equations["EQ-JERABKOVA-X-RHO"]
        names = ("z_env", "R_env")
        point = point[:2]
        kernel = lambda t: x_jerabkova_rho(t[0], t[1])
        expr = record.expression
    elif route == "marks":
        record = equations["EQ-ENV-MARKS-ROTATION-ROUNDED"]
        names = ("z_env", "R_env")
        point = point[:2]
        kernel = lambda t: x_hat_marks_plane(t[0], t[1])
        expr = record.expression
    elif route == "generalized":
        record = equations["EQ-ENV-X-MASS-ROUNDED"]
        names = ("z_env", "L_env", "eps_env")
        kernel = lambda t: x_jerabkova_generalized(t[0], 10 ** (t[1] + 6), t[2])
        expr = record.expression
    else:
        record = equations["EQ-ENV-X-MECL"]
        names = ("z_env", "L_env")
        point = point[:2]
        kernel = lambda t: alpha3_jerabkova_mecl(t[1], t[0])
        law = equations["EQ-JERABKOVA-ALPHA3"]
        expr = law.expression.subs(law.symbols["x_env"], record.expression)
    symbols = [record.symbols[name] for name in names]
    oracle = sp.lambdify(symbols, expr, "numpy")
    derivative = sp.lambdify(symbols, [sp.diff(expr, s) for s in symbols], "numpy")
    for evaluated in (kernel, jax.jit(kernel)):
        np.testing.assert_allclose(
            evaluated(point), oracle(*np.asarray(point)), rtol=0, atol=tolerances.EXACT
        )
        for ad in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                ad(evaluated)(point),
                derivative(*np.asarray(point)),
                rtol=0,
                atol=tolerances.EXACT,
            )
    if route == "generalized":
        log_density = equations["EQ-ENV-LOG-DENSITY"]
        rho_kernel = lambda t: compute_log_rho_cl_6(10 ** (t[0] + 6), t[1])
        rho_symbols = [log_density.symbols[n] for n in ("L_env", "eps_env")]
        oracle = sp.lambdify(rho_symbols, log_density.expression, "numpy")
        derivative = sp.lambdify(
            rho_symbols,
            [sp.diff(log_density.expression, s) for s in rho_symbols],
            "numpy",
        )
        for evaluated in (rho_kernel, jax.jit(rho_kernel)):
            np.testing.assert_allclose(
                evaluated(point[1:]),
                oracle(*np.asarray(point[1:])),
                rtol=0,
                atol=tolerances.EXACT,
            )
            for ad in (jax.jacfwd, jax.jacrev):
                np.testing.assert_allclose(
                    ad(evaluated)(point[1:]),
                    derivative(*np.asarray(point[1:])),
                    rtol=0,
                    atol=tolerances.EXACT,
                )


@pytest.mark.parametrize(
    "route,point", [("mcl", 1.0), ("mecl", 1.0), ("rho", 1.0), ("feh", -2.0)]
)
def test_table3_source_branches_and_derivative(equations, tolerances, route, point):
    record = equations["EQ-MARKS-TABLE3-" + route.upper()]
    coordinate = record.symbols["lambda_env"]
    oracle = sp.lambdify(coordinate, record.expression, "numpy")
    derivative = sp.lambdify(
        coordinate, sp.diff(record.expression, coordinate), "numpy"
    )
    kernel = lambda t: alpha3_marks_table3(t, route)
    # Plateau, equality and saturation are value contracts, not boundary derivatives.
    limit = {"mcl": 0.68, "mecl": 0.27, "rho": 0.095, "feh": -0.5}[route]
    values = jnp.array(
        [point, limit, limit - 0.01, limit + 0.01, -10.0, 10.0], dtype=jnp.float64
    )
    for evaluated in (kernel, jax.jit(kernel)):
        np.testing.assert_allclose(
            evaluated(values),
            np.clip(oracle(np.asarray(values)), 0.5, 2.3),
            rtol=0,
            atol=tolerances.EXACT,
        )
        for ad in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                ad(evaluated)(jnp.asarray(point)),
                derivative(point),
                rtol=0,
                atol=tolerances.EXACT,
            )


def test_lowmass_source_and_unclipped_derivative(equations, tolerances):
    record = equations["EQ-MARKS-LOWMASS"]
    z, canonical = record.symbols["z_env"], record.symbols["alpha_c_env"]
    point = jnp.array(-1.5, dtype=jnp.float64)
    expressions = [
        record.expression.subs(canonical, c)
        for c in (sp.Rational(13, 10), sp.Rational(23, 10))
    ]
    oracle = sp.lambdify(z, expressions, "numpy")
    derivative = sp.lambdify(z, [sp.diff(e, z) for e in expressions], "numpy")
    kernel = lambda t: jnp.stack(lowmass_slopes_metallicity(t))
    for evaluated in (kernel, jax.jit(kernel)):
        np.testing.assert_allclose(
            evaluated(point), oracle(float(point)), rtol=0, atol=tolerances.EXACT
        )
        for ad in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                ad(evaluated)(point),
                derivative(float(point)),
                rtol=0,
                atol=tolerances.EXACT,
            )


def test_plane_source_adoption_and_threshold_values(equations, tolerances):
    adopted = equations["EQ-ENV-MARKS-ALPHA3"]
    rotation = equations["EQ-ENV-MARKS-ROTATION-ROUNDED"]
    x = adopted.symbols["x_env"]
    assert adopted.expression.subs(x, -sp.Rational(87, 100)) == sp.Rational(
        573141, 250000
    )
    # The erratum uses rounded coefficients; keep it distinct from adopted Marks.
    erratum = equations["EQ-MARKS-ERRATUM"]
    jerabkova = equations["EQ-JERABKOVA-ALPHA3"]
    assert sp.simplify(erratum.expression - jerabkova.expression) == 0
    printed = equations["EQ-MARKS-EQ15"]
    expr = adopted.expression.subs(x, rotation.expression)
    symbols = [rotation.symbols[n] for n in ("z_env", "R_env")]
    active = -sp.Rational(4072, 10000) * rotation.expression + sp.Rational(19383, 10000)
    assert sp.expand(active - printed.expression) != 0
    source_rotation = equations["EQ-MARKS-ROTATION"]
    sine = source_rotation.symbols["sin_env"]
    cosine = source_rotation.symbols["cos_env"]
    # Fig.6 names x'/sin(theta). Its density coefficient is exactly one,
    # accounting for Eq.15's -0.4072, before its metallicity rounding.
    normalized = source_rotation.expression / sine
    assert sp.simplify(sp.diff(normalized, source_rotation.symbols["R_env"])) == 1
    assert (
        sp.simplify(
            sp.diff(normalized, source_rotation.symbols["z_env"]) - cosine / sine
        )
        == 0
    )
    point = jnp.array([-1.5, 0.0], dtype=jnp.float64)
    oracle = sp.lambdify(symbols, expr, "numpy")
    derivative = sp.lambdify(symbols, [sp.diff(expr, s) for s in symbols], "numpy")
    kernel = lambda t: alpha3_marks_plane(t[1], t[0])
    for evaluated in (kernel, jax.jit(kernel)):
        np.testing.assert_allclose(
            evaluated(point), oracle(*np.asarray(point)), rtol=0, atol=tolerances.EXACT
        )
        for ad in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                ad(evaluated)(point),
                derivative(*np.asarray(point)),
                rtol=0,
                atol=tolerances.EXACT,
            )
    # z=0 removes rotation, making the corrected hard-threshold value exact
    # in the registered coordinate. Check values on each side, not AD at the jump.
    rho = jnp.array([-0.88 / 0.99, -0.86 / 0.99], dtype=jnp.float64)
    jerk = sp.lambdify(x, jerabkova.expression, "numpy")
    np.testing.assert_allclose(
        alpha3_jerabkova_rho(rho, 0.0),
        jerk(0.99 * np.asarray(rho)),
        rtol=0,
        atol=tolerances.EXACT,
    )
