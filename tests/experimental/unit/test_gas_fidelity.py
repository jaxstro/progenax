r"""AC-BE7 — the realized gas follows its prescribed envelope (ADR-0069).

ADR-0069 moved the validation target: the Bonnor-Ebert / polytrope models describe the
**gas**, not the stars, so this gates gas-profile fidelity rather than an emergent stellar
EFF slope. (The stellar version is not gated because it cannot be: at t=0 a single
realization's shell density scatters by 130-620%, and EFF fits returned gamma = 101 +/- 132
with a fit rms of 0.44 dex against a 0.03 dex fitter noise floor -- including for the
Plummer control.)

Measurement rules, both forced by the physics and recorded in ADR-0069:

- **Ensemble.** Single realizations are far too noisy; the ensemble mean is the estimand.
- **No catch-all bin.** ``gas_fidelity.shell_mean`` drops out-of-range cells rather than
  clipping them into the end shells, which would dump ~99% of the box (empty space) into
  the outermost shell.
- **Only resolved shells.** Shells are kept if they hold at least ``MIN_CELLS_PER_SHELL``
  cells. That criterion is stated up front, not chosen after seeing which range passes.

Measured at 48 seeds, 64^3, sfe=0.05 (shape = ratio-to-envelope normalised by its own mean,
so flat 1.0 is perfect):

    BE xi_max=6.45  cloud     0.944, 0.969, 1.123   max dev 12.3%
                    residual  0.938, 0.975, 1.108   max dev 10.8%
    Polytrope 5/3   cloud     0.926, 0.989, 1.173   max dev 17.3%
                    residual  0.924, 0.999, 1.152   max dev 15.2%
    standard error            6.7, 3.5, 2.5 %

The SFE ceiling is REALIZATION-dependent, not universal -- see
:class:`TestStarFormationCeilingIsRealizationDependent`.

KNOWN SYSTEMATIC, deliberately not tuned away: the outermost qualifying shell runs 12-17%
HIGH in both profiles, well above its 2.5% standard error. Inner and middle shells agree to
~5%. The threshold below covers that systematic with margin rather than excluding the shell
that exhibits it.

**The systematic is characterised but NOT explained, and the gate is REGIME-LIMITED.**
Three diagnostics localised it:

    raw s_total field, no gas partition   12.4%  (vs 12.3% with gas -> not the gas solver)
    vs Mach   0.5 -> 3.0%,  2 -> 4.0%,  4 -> 6.7%,  8 -> 12.4%     (turbulence-driven)
    vs xi_max 3 (contrast 2.9) -> 9.6%,  6.45 (14) -> 12.4%,  12 (66) -> 26.2%

It is an interaction between the multiplicative turbulent field and the envelope's radial
gradient: inner shells low, outer shells high, growing monotonically with BOTH mach and
contrast. The mechanism is not yet established.

Because it grows with contrast, MAX_SHAPE_DEVIATION = 0.25 is only valid in the tested
regime (xi_max <~ 6.45 at mach=8). At xi_max=12 the measured deviation is 26.2% and this
gate would FAIL -- correctly, since the model's fidelity genuinely degrades there. Do not
raise the threshold to accommodate a steeper envelope; explain the systematic first.
"""

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.experimental

SHAPE = (64, 64, 64)
BOX = 4.0  # pc
R_H = 0.5  # pc
N_SEEDS = 16  # ensemble size for the gate; the reference numbers above used 48

# Covers the measured 12-17% systematic plus the standard error at this ensemble size.
MAX_SHAPE_DEVIATION = 0.25


def _profiles():
    from gravoturb.profiles import BonnorEbertProfile, PolytropeProfile

    return {
        "bonnor_ebert": BonnorEbertProfile(r_h=R_H, xi_max=6.45, n_points=800),
        "polytrope": PolytropeProfile(r_h=R_H, gamma=5.0 / 3.0, n_points=800),
    }


def _ensemble_ratio(profile, n_seeds=N_SEEDS, sfe=0.05, seed0=0):
    """Mean ratio-to-envelope over ``n_seeds`` realizations, for cloud and residual gas."""
    from gravoturb.cluster import build_cluster_ic
    from gravoturb.diagnostics.gas_fidelity import envelope_fidelity
    from gravoturb.realization.envelope import radius_grid
    from gravoturb.specs import (
        CloudSpec,
        CompositionSpec,
        GasSpec,
        GeometrySpec,
        VelocitySpec,
    )
    from jaxstro.units import STELLAR

    radii = radius_grid(SHAPE, BOX)
    cloud, residual = [], []
    for seed in range(seed0, seed0 + n_seeds):
        ic = build_cluster_ic(
            jnp.ones(2000),
            cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.5),
            geometry=GeometrySpec(profile=profile, box_size=BOX, shape=SHAPE),
            velocity=VelocitySpec(beta_v=4.0, mode="physical", c_s=0.2),
            composition=CompositionSpec(placement="two_population", f_sub=0.3),
            gas=GasSpec(sfe=sfe),
            units=STELLAR,
            G=STELLAR.G,
            key=jax.random.PRNGKey(seed),
        )
        cloud.append(envelope_fidelity(ic.gas.rho_cloud, radii, profile).ratio)
        residual.append(envelope_fidelity(ic.gas.rho_residual, radii, profile).ratio)
    return jnp.mean(jnp.stack(cloud), axis=0), jnp.mean(jnp.stack(residual), axis=0)


def _ensemble_seed(profile, seed, sfe):
    """Build a single realization; raises if the requested SFE is unreachable."""
    return _ensemble_ratio(profile, n_seeds=1, sfe=sfe, seed0=seed)


def _resolved_shape(ratio, profile):
    """Normalised shape over shells meeting the cell-count criterion."""
    from gravoturb.diagnostics.gas_fidelity import (
        MIN_CELLS_PER_SHELL,
        shell_edges,
        shell_mean,
    )
    from gravoturb.realization.envelope import radius_grid

    radii = radius_grid(SHAPE, BOX)
    edges = shell_edges(profile.r_edge)
    _, _, counts = shell_mean(jnp.ones(SHAPE), radii, edges)
    keep = counts >= MIN_CELLS_PER_SHELL
    resolved = ratio[keep]
    return resolved / jnp.mean(resolved)


class TestShellBinning:
    """The binning itself, since a catch-all bin previously faked a 97% disagreement."""

    def test_out_of_range_cells_are_dropped_not_clipped(self):
        from gravoturb.diagnostics.gas_fidelity import shell_mean

        edges = jnp.array([1.0, 2.0, 3.0])
        radii = jnp.array([0.1, 1.5, 2.5, 99.0])  # one below, two inside, one far outside
        _, mean, counts = shell_mean(jnp.array([5.0, 1.0, 1.0, 5.0]), radii, edges)
        assert list(counts) == [1.0, 1.0], "end bins must not absorb out-of-range cells"
        assert jnp.allclose(mean, jnp.array([1.0, 1.0]))

    def test_counts_sum_to_in_range_cells_only(self):
        from gravoturb.diagnostics.gas_fidelity import shell_mean

        edges = jnp.linspace(1.0, 2.0, 4)
        radii = jnp.linspace(0.0, 5.0, 500)
        _, _, counts = shell_mean(jnp.ones(500), radii, edges)
        expected = jnp.sum(((radii >= edges[0]) & (radii < edges[-1])).astype(float))
        assert float(jnp.sum(counts)) == pytest.approx(float(expected), abs=1.0)


class TestEnvelopeFidelity:
    """AC-BE7 proper: the ensemble gas shape must track the prescribed envelope."""

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["bonnor_ebert", "polytrope"])
    def test_cloud_shape_tracks_envelope(self, name):
        profile = _profiles()[name]
        cloud, _ = _ensemble_ratio(profile)
        shape = _resolved_shape(cloud, profile)
        dev = float(jnp.max(jnp.abs(shape - 1.0)))
        assert dev < MAX_SHAPE_DEVIATION, f"{name} cloud shape {shape}, max dev {dev:.3f}"

    @pytest.mark.slow
    @pytest.mark.parametrize("name", ["bonnor_ebert", "polytrope"])
    def test_residual_gas_shape_tracks_envelope(self, name):
        """The leftover gas is the component BE actually describes (ADR-0069)."""
        profile = _profiles()[name]
        _, residual = _ensemble_ratio(profile)
        shape = _resolved_shape(residual, profile)
        dev = float(jnp.max(jnp.abs(shape - 1.0)))
        assert dev < MAX_SHAPE_DEVIATION, f"{name} residual shape {shape}, max dev {dev:.3f}"

    @pytest.mark.slow
    def test_residual_is_no_worse_than_cloud(self):
        """Measured: residual tracks slightly BETTER (10.8% vs 12.3% at 48 seeds).

        Asserted only as 'no worse' plus a tolerance -- the margin is small compared to the
        standard error, so a strict inequality would be a coin-flip gate.
        """
        profile = _profiles()["bonnor_ebert"]
        cloud, residual = _ensemble_ratio(profile)
        dev_c = float(jnp.max(jnp.abs(_resolved_shape(cloud, profile) - 1.0)))
        dev_r = float(jnp.max(jnp.abs(_resolved_shape(residual, profile) - 1.0)))
        assert dev_r < dev_c + 0.05, f"cloud {dev_c:.3f} vs residual {dev_r:.3f}"


class TestStarFormationCeilingIsRealizationDependent:
    """Whether a requested SFE is achievable depends on the REALIZATION, not just the mach.

    Measured across seeds 0-5 (mach=8, b=0.5, alpha=1.8):

        sfe    seed 0  1      2      3      4      5
        0.20   ok      ok     ok     ok     ok     ok
        0.30   ok      ok     ok     RAISE  RAISE  ok
        0.60   ok      ok     ok     RAISE  RAISE  ok

    So there is no universal ceiling: some realizations supply enough collapse-eligible
    dense gas to reach sfe=0.6, others refuse by 0.3. That is a consequence of the same
    large-scale-mode variance that dominates everything else here (ADR-0069) -- the dense
    tail available for star formation is itself a realization property.

    A single-seed refusal gate would therefore be a coin flip, so these tests assert the
    seed-DEPENDENCE, which is deterministic for fixed seeds.
    """

    @pytest.mark.slow
    def test_high_sfe_is_refused_for_some_realizations(self):
        profile = _profiles()["bonnor_ebert"]
        outcomes = []
        for seed in (3, 4):
            try:
                _ensemble_seed(profile, seed, sfe=0.6)
                outcomes.append("ok")
            except Exception:
                outcomes.append("raise")
        assert "raise" in outcomes, f"expected refusal for seeds 3/4, got {outcomes}"

    @pytest.mark.slow
    def test_high_sfe_is_achievable_for_others(self):
        """Refusal must not be universal, or the guard would just be a hard cap."""
        profile = _profiles()["bonnor_ebert"]
        _ensemble_seed(profile, 0, sfe=0.6)  # must not raise

    @pytest.mark.slow
    def test_modest_sfe_is_accepted_everywhere(self):
        profile = _profiles()["bonnor_ebert"]
        for seed in range(4):
            _ensemble_seed(profile, seed, sfe=0.05)
