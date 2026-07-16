"""Phase 0.5 rename gate: the `gravoturb` package reproduces pre-rename realizations exactly.

Reference hashes were captured on the pre-rename `gravoturb_fdf` tree (commit 66f627d) at
pinned PRNG keys (tests/experimental/fixtures/rename_pins/pre_rename_sha256.json). A mismatch
means a split/merge changed `jax.random` call order or a computation — the refactor was
required to be zero-behavior-change.

Same-machine/env reference: float64 bit-exactness across the rename is the contract on the
machine that captured the pins; cross-platform FFT/libm differences may legitimately break
these hashes, so skip (don't fail) if the environment fingerprint differs — the physics
suites cover correctness there.
"""

import hashlib
import json
import pathlib
import platform

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from gravoturb.cluster import build_cluster_ic
from gravoturb.realization.envelope import apply_spherical_envelope
from gravoturb.realization.pipeline import build_turbulent_field
from gravoturb.realization.placement import sample_positions
from gravoturb.realization.turbulent_velocity import turbulent_velocity_field
from jaxstro.units import STELLAR

from progenax import PlummerProfile

pytestmark = [pytest.mark.experimental, pytest.mark.unit]

_PINS = json.loads(
    (pathlib.Path(__file__).parents[1] / "fixtures" / "rename_pins" /
     "pre_rename_sha256.json").read_text()
)

# Raw-GRF outputs (velocity field, cluster velocities) are bit-sensitive to the XLA
# threading config: multi- vs single-threaded FFTs reduce in different orders. The pins
# were captured at pre-rename commit 66f627d under the canonical gate env (XLA_FLAGS
# thread-capped, per CLAUDE.md); rank-copula-derived pins are threading-insensitive
# (the copula consumes only the GRF's ranks).
#
# Fingerprint (review-hardened): EXACT XLA_FLAGS equality (extra flags could
# legitimately change FFT reduction order) + Darwin/arm64 (the capture machine class —
# a linux-aarch64 runner would spuriously fail on libm differences). Off-env the tests
# skip — EXCEPT under GRAVOTURB_BYTE_GATE=1 (the documented gate command / CI job),
# where a skip becomes a hard failure so the zero-behavior-change contract can never
# pass vacuously on the machine that is supposed to enforce it.
import os

_env_ok = (
    platform.machine() == _PINS["_env"]["machine"]
    and platform.system() == "Darwin"
    and os.environ.get("XLA_FLAGS", "").strip() == _PINS["_env"]["xla_flags"]
)
_strict = os.environ.get("GRAVOTURB_BYTE_GATE") == "1"
if not _env_ok and _strict:
    raise RuntimeError(
        "GRAVOTURB_BYTE_GATE=1 but the environment fingerprint does not match the "
        f"pin-capture env: need Darwin/arm64 with XLA_FLAGS exactly "
        f"{_PINS['_env']['xla_flags']!r}, got system={platform.system()!r}, "
        f"machine={platform.machine()!r}, XLA_FLAGS={os.environ.get('XLA_FLAGS')!r}"
    )
skip_foreign = pytest.mark.skipif(
    not _env_ok,
    reason="byte-identity pins are same-machine, canonical-XLA-env references "
    "(set GRAVOTURB_BYTE_GATE=1 in the gate command to make this a hard failure)",
)


def _h(x):
    arr = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


@skip_foreign
def test_field_realizations_match_pins():
    for i, (mach, b, alpha, beta) in enumerate(
        [(8.0, 0.5, 1.8, 3.0), (12.0, 0.33, 1.6, 3.5)]
    ):
        fld = build_turbulent_field(mach, b, alpha, beta, (32, 32, 32), jax.random.PRNGKey(7 + i))
        assert _h(fld.s) == _PINS[f"field_s_{i}"]
        scalars = jnp.stack([fld.s_t, fld.f_dense, fld.f_dense_realized])
        assert _h(scalars) == _PINS[f"field_scalars_{i}"]


@skip_foreign
def test_positions_and_velocity_field_match_pins():
    fld = build_turbulent_field(8.0, 0.5, 1.8, 3.0, (32, 32, 32), jax.random.PRNGKey(7))
    s_tot = apply_spherical_envelope(fld.s, PlummerProfile(r_h=0.5), 4.0)
    pos = sample_positions(fld.s, fld.s_t, 8.0, 0.3, 500, jax.random.PRNGKey(21),
                           box_size=4.0, s_density=s_tot)
    assert _h(pos) == _PINS["positions"]
    vf = turbulent_velocity_field((32, 32, 32), 4.0, jax.random.PRNGKey(33))
    assert _h(vf) == _PINS["velocity_field"]


@skip_foreign
def test_cluster_ic_matches_pins():
    from gravoturb.specs import CloudSpec, CompositionSpec, GeometrySpec, VelocitySpec

    ic = build_cluster_ic(
        jnp.ones(400),
        cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
        geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0,
                              shape=(32, 32, 32)),
        velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
        composition=CompositionSpec(placement="two_population", f_sub=0.3),
        G=STELLAR.G, key=jax.random.PRNGKey(42),
    )
    assert _h(ic.positions) == _PINS["cluster_positions"]
    assert _h(ic.velocities) == _PINS["cluster_velocities"]
    assert _h(ic.Q_virial) == _PINS["cluster_Q"]


def test_sigma_s_squared_parity_with_released_core():
    """The duplicate σ_s² across the package boundary stays pinned together.

    gravoturb.theory.density_pdf.sigma_s_squared returns the VARIANCE σ_s²;
    progenax.cluster.turbulence.sigma_ln_rho_from_mach returns the STD σ_s.
    Both implement FK10 Eq. 19.
    """
    from gravoturb.theory.density_pdf import sigma_s_squared

    from progenax.cluster.turbulence import sigma_ln_rho_from_mach

    for mach, b in [(5.0, 0.4), (8.0, 0.5), (12.0, 1.0 / 3.0), (25.0, 1.0)]:
        var = float(sigma_s_squared(mach, b))
        std = float(sigma_ln_rho_from_mach(jnp.asarray(mach), b=b))
        np.testing.assert_allclose(var, std**2, rtol=1e-12)
