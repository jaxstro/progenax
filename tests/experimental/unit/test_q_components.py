"""Unit tests for the (Q, m_bar, s_bar) substructure decomposition (FDF pivot, build 3).

CW04 Q = m_bar/s_bar conflates substructure with central concentration. Reporting the components
separately decouples them: m_bar (normalized MST length) tracks CLUMPINESS; s_bar (normalized mean
separation) tracks CONCENTRATION. See docs/plans/2026-06-07-gravoturb-fdf-spherical-ic-design.md.
"""

import numpy as np
import pytest

pytestmark = pytest.mark.experimental


def _uniform_sphere(n, rng):
    u = rng.uniform(0, 1, n)
    r = u ** (1 / 3)
    ct = 2 * rng.uniform(0, 1, n) - 1
    st = np.sqrt(1 - ct**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct])


def _concentrated_smooth(n, rng, a=0.2):
    u = rng.uniform(0, 1, n)
    r = a * np.sqrt(u ** (2 / 3) / (1 - u ** (2 / 3)))  # Plummer radii (centrally concentrated)
    ct = 2 * rng.uniform(0, 1, n) - 1
    st = np.sqrt(1 - ct**2)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct])


def _clumpy(n, rng, n_blobs=6):
    centers = rng.uniform(-0.6, 0.6, (n_blobs, 3))
    idx = rng.integers(0, n_blobs, n)
    return centers[idx] + 0.04 * rng.normal(size=(n, 3))


def test_q_components_consistent_with_Q():
    """q_components returns (Q, m_bar, s_bar) with Q == m_bar/s_bar == compute_q_parameter."""
    from gravoturb_fdf.diagnostics.q import compute_q_parameter, q_components

    rng = np.random.default_rng(0)
    pos = _uniform_sphere(300, rng)
    Q, m_bar, s_bar = q_components(pos)
    assert np.isclose(Q, m_bar / s_bar)
    assert np.isclose(Q, compute_q_parameter(pos))
    assert m_bar > 0 and s_bar > 0


def test_components_decouple_concentration_from_substructure():
    """The (m_bar, s_bar) PLANE separates concentration from substructure (the honest decoupling).

    s_bar is the concentration axis; at FIXED concentration (matched s_bar), m_bar isolates
    substructure. The three regimes occupy distinct regions.
    """
    from gravoturb_fdf.diagnostics.q import q_components

    rng = np.random.default_rng(1)
    _Qu, m_u, s_u = q_components(_uniform_sphere(500, rng))           # smooth, unconcentrated
    _Qc, m_c, s_c = q_components(_concentrated_smooth(500, rng))      # smooth, concentrated
    _Qk, m_k, s_k = q_components(_clumpy(500, rng))                   # substructured

    # s_bar is the CONCENTRATION axis: concentrated has much smaller s_bar; clumpy ~ uniform
    assert s_c < 0.5 * s_u
    assert 0.7 * s_u < s_k < 1.3 * s_u            # clumpy at ~the same (low) concentration as uniform
    # at matched concentration (uniform vs clumpy), m_bar isolates SUBSTRUCTURE
    assert m_k < 0.6 * m_u
    # the three regimes are distinct points in the (m_bar, s_bar) plane
    assert s_c < 0.5 * s_k                        # concentrated separated from clumpy by s_bar
