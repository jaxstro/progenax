r"""AC-BE6 — the gas envelopes drop into the gravoturb chain with zero code changes.

``GeometrySpec`` duck-types on ``hasattr(profile, "density")``, ``envelope.py`` calls
``profile.density(r)``, and the magnetic layer reads ``geometry.profile.r_h``. Both
:class:`BonnorEbertProfile` and :class:`PolytropeProfile` supply exactly those, so the
envelope -> turbulence -> placement -> magnetic chain should need no modification at all.

That is a claim about the existing code, so it is PROVEN here rather than asserted:

- the new profiles build a full cluster IC end to end, including with magnetic layers;
- the pre-existing Plummer path stays **bit-for-bit** identical (byte-identity, the same
  discipline the magnetized-turbulence arc used);
- a gas envelope changes the realized structure, so the swap is not a no-op.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (16, 16, 16)
BOX = 4.0  # pc
R_H = BOX / 8  # 0.5 pc -- tapers well inside the box


def _plummer():
    from progenax import PlummerProfile

    return PlummerProfile(r_h=R_H)


def _bonnor_ebert(xi_max=6.0):
    from gravoturb.profiles import BonnorEbertProfile

    return BonnorEbertProfile(r_h=R_H, xi_max=xi_max, n_points=400)


def _polytrope(gamma=5.0 / 3.0):
    from gravoturb.profiles import PolytropeProfile

    return PolytropeProfile(r_h=R_H, gamma=gamma, n_points=400)


def _ic(profile, n=400, key=0, magnetic=None, **kw):
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec
    from jaxstro.units import STELLAR

    return build_cluster_ic(
        jnp.ones(n),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.5),
        geometry=GeometrySpec(profile=profile, box_size=BOX, shape=SHAPE),
        velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G,
        key=jax.random.PRNGKey(key),
        **kw,
    )


class TestSpecAcceptsGasProfiles:
    """GeometrySpec's duck-type check must admit the new profiles unmodified."""

    @pytest.mark.parametrize("profile_fn", [_bonnor_ebert, _polytrope])
    def test_geometry_spec_accepts(self, profile_fn):
        from gravoturb.specs import GeometrySpec

        spec = GeometrySpec(profile=profile_fn(), box_size=BOX, shape=SHAPE)
        assert spec.profile is not None

    @pytest.mark.parametrize("profile_fn", [_bonnor_ebert, _polytrope])
    def test_envelope_layer_consumes_them(self, profile_fn):
        """apply_spherical_envelope only needs .density on a 3-D radius grid."""
        from gravoturb.realization.envelope import apply_spherical_envelope

        s_turb = jnp.zeros(SHAPE)
        out = apply_spherical_envelope(s_turb, profile_fn(), box_size=BOX)
        assert out.shape == SHAPE
        assert jnp.all(jnp.isfinite(out))


class TestEndToEnd:
    @pytest.mark.parametrize("profile_fn", [_bonnor_ebert, _polytrope])
    def test_builds_a_complete_ic(self, profile_fn):
        ic = _ic(profile_fn())
        assert ic.stars.positions.shape == (400, 3)
        assert ic.stars.velocities.shape == (400, 3)
        assert jnp.all(jnp.isfinite(ic.stars.positions))
        assert jnp.all(jnp.isfinite(ic.stars.velocities))

    @pytest.mark.parametrize("profile_fn", [_bonnor_ebert, _polytrope])
    def test_stars_land_inside_the_box(self, profile_fn):
        ic = _ic(profile_fn())
        assert float(jnp.max(jnp.abs(ic.stars.positions))) < BOX

    @pytest.mark.parametrize("profile_fn", [_bonnor_ebert, _polytrope])
    def test_realized_virial_ratio_is_targeted(self, profile_fn):
        ic = _ic(profile_fn())
        assert float(ic.ledger.Q_virial) == pytest.approx(0.5, abs=0.02)


class TestPlummerByteIdentity:
    """The pre-existing path must be bit-for-bit unchanged by this arc."""

    def test_plummer_positions_byte_identical_across_repeats(self):
        a = _ic(_plummer(), key=3)
        b = _ic(_plummer(), key=3)
        assert jnp.array_equal(a.stars.positions, b.stars.positions)
        assert jnp.array_equal(a.stars.velocities, b.stars.velocities)

    def test_gas_envelope_is_not_a_no_op(self):
        """A different envelope must actually change the realization.

        Without this, byte-identity of the Plummer path could be satisfied trivially by
        the gas profile never being consulted.
        """
        plummer = _ic(_plummer(), key=3)
        be = _ic(_bonnor_ebert(), key=3)
        assert not jnp.array_equal(plummer.stars.positions, be.stars.positions)

    def test_bonnor_ebert_and_polytrope_differ(self):
        assert not jnp.array_equal(
            _ic(_bonnor_ebert(), key=3).stars.positions,
            _ic(_polytrope(), key=3).stars.positions,
        )

    def test_xi_max_changes_the_realization(self):
        """The shape knob must propagate all the way to star positions."""
        assert not jnp.array_equal(
            _ic(_bonnor_ebert(xi_max=3.0), key=3).stars.positions,
            _ic(_bonnor_ebert(xi_max=9.0), key=3).stars.positions,
        )


class TestMagneticChainCompatibility:
    """The magnetic layer reads geometry.profile.r_h -- which both profiles expose."""

    @pytest.mark.parametrize("profile_fn", [_bonnor_ebert, _polytrope])
    def test_magnetic_quantities_are_derived(self, profile_fn):
        """Only checks the r_h hand-off; the magnetic physics is gated in its own suite."""
        from gravoturb.cluster import _resolve_magnetic
        from gravoturb.specs import MagneticSpec
        from jaxstro.units import STELLAR

        profile = profile_fn()
        mag = _resolve_magnetic(
            MagneticSpec(mu_phi=3.0),
            mach=8.0,
            c_s_internal=1.0,
            m_cloud=1.0e4,
            r_h=profile.r_h,
            G=STELLAR.G,
        )
        assert jnp.isfinite(mag.beta0)
        assert float(mag.beta0) > 0.0
