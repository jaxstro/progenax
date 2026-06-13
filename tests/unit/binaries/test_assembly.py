"""Tests for the binary -> spatial-IC connector primitive (Batch 4f).

resolve_binary_components places resolved binary components around their system
COMs in a masked, fixed-shape (2N slots + is_real) representation that is
jit/vmap/grad-safe. Each binary's COM is preserved exactly (m1 r1 + m2 r2 =
(m1+m2) X_com), so the cluster phase space is untouched; only internal structure
is resolved. Singles pass through (primary = the single, secondary = ghost).
"""

import jax
import jax.numpy as jnp
import pytest

from jaxstro.units import STELLAR

G = STELLAR.G


def _example():
    """3 systems: a single + 2 binaries (one inclined-eccentric, one equal-mass circular)."""
    com_pos = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    com_vel = jnp.array([[0.0, 0.5, 0.0], [0.0, 0.0, 0.3], [0.2, 0.0, 0.0]])
    m1 = jnp.array([1.0, 2.0, 1.5])
    m2 = jnp.array([0.0, 1.0, 1.5])
    is_binary = jnp.array([False, True, True])
    a = jnp.array([0.0, 0.5, 1.0])
    e = jnp.array([0.0, 0.3, 0.0])
    inc = jnp.array([0.0, 0.1, 0.0])
    Omega = jnp.array([0.0, 0.2, 0.0])
    omega = jnp.array([0.0, 0.4, 0.0])
    M_anom = jnp.array([0.0, 1.0, 2.0])
    return com_pos, com_vel, m1, m2, is_binary, a, e, inc, Omega, omega, M_anom


def _resolve(*args):
    from progenax.binaries import resolve_binary_components
    com_pos, com_vel, m1, m2, is_binary, a, e, inc, Omega, omega, M_anom = args
    return resolve_binary_components(
        com_pos, com_vel, m1, m2, is_binary, a, e, inc, Omega, omega, M_anom, G=G
    )


class TestResolveBinaryComponents:
    def test_output_shape_2N(self):
        rb = _resolve(*_example())
        assert rb.positions.shape == (6, 3)
        assert rb.velocities.shape == (6, 3)
        assert rb.masses.shape == (6,)
        assert rb.is_real.shape == (6,)

    def test_is_real_mask(self):
        """Primaries always real; secondaries real iff binary. 3 primaries + 2 secondaries = 5."""
        rb = _resolve(*_example())
        # interleaved [p0,s0,p1,s1,p2,s2]; s0 is the single's ghost
        assert bool(rb.is_real[0]) and not bool(rb.is_real[1])  # single + ghost
        assert bool(rb.is_real[2]) and bool(rb.is_real[3])      # binary 1
        assert bool(rb.is_real[4]) and bool(rb.is_real[5])      # binary 2
        assert int(jnp.sum(rb.is_real)) == 5

    def test_primordial_bookkeeping(self):
        rb = _resolve(*_example())
        assert jnp.array_equal(rb.primordial_system_id, jnp.array([0, 0, 1, 1, 2, 2]))
        assert jnp.array_equal(
            rb.is_primordial_secondary, jnp.array([False, True, False, True, False, True])
        )

    def test_single_passthrough(self):
        """A single's primary slot = the single (com pos/vel, mass m1); secondary = ghost m=0."""
        rb = _resolve(*_example())
        assert jnp.allclose(rb.positions[0], jnp.array([1.0, 0.0, 0.0]))
        assert jnp.allclose(rb.velocities[0], jnp.array([0.0, 0.5, 0.0]))
        assert float(rb.masses[0]) == 1.0
        assert float(rb.masses[1]) == 0.0  # ghost secondary

    def test_com_conserved_per_binary(self):
        """m1 r1 + m2 r2 = (m1+m2) X_com and m1 v1 + m2 v2 = (m1+m2) V_com, per binary."""
        com_pos, com_vel, m1, m2, *_ = _example()
        rb = _resolve(com_pos, com_vel, m1, m2, *_example()[4:])
        for i, (slot_p, slot_s) in enumerate([(2, 3), (4, 5)], start=1):
            mp, ms = float(rb.masses[slot_p]), float(rb.masses[slot_s])
            M = mp + ms
            r_com = (mp * rb.positions[slot_p] + ms * rb.positions[slot_s]) / M
            v_com = (mp * rb.velocities[slot_p] + ms * rb.velocities[slot_s]) / M
            assert jnp.allclose(r_com, com_pos[i], atol=1e-12), f"binary {i} r_com"
            assert jnp.allclose(v_com, com_vel[i], atol=1e-12), f"binary {i} v_com"

    def test_masses_preserved(self):
        rb = _resolve(*_example())
        # primaries carry m1; real secondaries carry m2
        assert jnp.allclose(rb.masses[jnp.array([0, 2, 4])], jnp.array([1.0, 2.0, 1.5]))
        assert jnp.allclose(rb.masses[jnp.array([3, 5])], jnp.array([1.0, 1.5]))

    def test_jit_safe(self):
        """jit compiles and returns finite positions.

        AD-vs-FD for resolve_binary_components(a) (the d|sep|/da gradient that this
        test formerly also smoke-checked) is owned by the grad-audit registry
        (tests/validation/grad_audit/registry.py :: resolve_binary_components [+ the
        mixed-is_binary Fisher-integrity variant]); see
        docs/website/50-validation/differentiability-audit.md.
        (audit T6 consolidation; registry is SoT)
        """
        args = _example()
        from progenax.binaries import resolve_binary_components
        jitted = jax.jit(lambda *a: resolve_binary_components(*a, G=G).positions)
        pos = jitted(*args)
        assert jnp.all(jnp.isfinite(pos))
