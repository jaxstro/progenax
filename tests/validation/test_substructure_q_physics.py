"""
Physics validation for the Cartwright & Whitworth (2004) Q substructure diagnostic.

Q = m_bar / s_bar distinguishes centrally-concentrated clusters (Q > 0.8) from
fractal/substructured ones (Q < 0.8). CW04 Table 1 (3D projected to 2D) anchors:
  uniform sphere (3D0) Q ~ 0.79; r^-1 (3D1) ~ 0.84; r^-2 (3D2) ~ 0.93;
  fractal D=1.5/2.0/2.5 -> Q ~ 0.47/0.58/0.70.

Two implementations are validated: the exact scipy `compute_q_parameter` and the
JAX-differentiable `q_approx` (used for substructure inference).

Reference: Cartwright & Whitworth (2004), MNRAS 348, 589.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from progenax.diagnostics.substructure import compute_q_parameter
from progenax.diagnostics.q_approx import q_approx


def _uniform_sphere(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.uniform(0, 1, n)
    r = u ** (1 / 3)
    cos_t = rng.uniform(-1, 1, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t])


def _plummer(n, seed, a=0.5):
    rng = np.random.default_rng(seed)
    u = np.clip(rng.uniform(0, 1, n), 1e-9, 1 - 1e-9)
    r = a * u ** (1 / 3) / np.sqrt(1 - u ** (2 / 3))
    cos_t = rng.uniform(-1, 1, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t])


def _concentrated(n, seed):
    """r^-2 number density (radial CDF ~ r, i.e. r = u*R): CW04 '3D2', Q ~ 0.93,
    inside the CW04-calibrated range (unlike a steep Plummer, whose Q > 1)."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(0, 1, n)
    cos_t = rng.uniform(-1, 1, n)
    sin_t = np.sqrt(1 - cos_t ** 2)
    phi = rng.uniform(0, 2 * np.pi, n)
    return np.column_stack([r * sin_t * np.cos(phi), r * sin_t * np.sin(phi), r * cos_t])


def _clumpy(n, seed, k=8, spread=0.06):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-1, 1, (k, 3))
    centers /= np.maximum(np.linalg.norm(centers, axis=1, keepdims=True), 1e-9)
    centers *= rng.uniform(0.2, 1.0, (k, 1))
    which = rng.integers(0, k, n)
    return centers[which] + rng.normal(0, spread, (n, 3))


class TestCW04Baselines:
    def test_uniform_sphere_matches_cw04_3d0(self):
        """Uniform sphere Q ~ 0.79 (CW04 '3D0')."""
        Qs = [compute_q_parameter(_uniform_sphere(300, s)) for s in range(8)]
        Qm = np.mean(Qs)
        assert 0.75 < Qm < 0.85, f"uniform-sphere Q = {Qm:.3f} (CW04 3D0 ~ 0.79)"

    def test_concentration_raises_Q(self):
        """Centrally concentrated (Plummer) gives Q > uniform sphere (> 0.8)."""
        Qu = np.mean([compute_q_parameter(_uniform_sphere(300, s)) for s in range(6)])
        Qp = np.mean([compute_q_parameter(_plummer(300, s)) for s in range(6)])
        assert Qp > Qu and Qp > 0.80, f"Plummer Q={Qp:.3f} should exceed uniform {Qu:.3f} and 0.80"

    def test_substructure_lowers_Q(self):
        """Clumpy/substructured distribution gives Q < 0.80."""
        Qc = np.mean([compute_q_parameter(_clumpy(300, s)) for s in range(6)])
        assert Qc < 0.80, f"clumpy Q={Qc:.3f} should be < 0.80 (substructured)"

    def test_full_ordering(self):
        """Q orders the regimes: clumpy < uniform < concentrated."""
        Qc = np.mean([compute_q_parameter(_clumpy(300, s)) for s in range(5)])
        Qu = np.mean([compute_q_parameter(_uniform_sphere(300, s)) for s in range(5)])
        Qp = np.mean([compute_q_parameter(_plummer(300, s)) for s in range(5)])
        assert Qc < Qu < Qp, f"ordering failed: clumpy={Qc:.3f}, unif={Qu:.3f}, plummer={Qp:.3f}"


class TestQNIndependence:
    def test_q_roughly_n_independent(self):
        """CW04: Q is ~N-independent for N > 100 (range/mean < 0.25)."""
        Qs = []
        for N in (150, 300, 600, 1200):
            Qs.append(np.mean([compute_q_parameter(_uniform_sphere(N, s)) for s in range(4)]))
        Qs = np.array(Qs)
        assert (Qs.max() - Qs.min()) / Qs.mean() < 0.25, f"Q vs N not flat: {Qs}"


class TestQApprox:
    @pytest.mark.parametrize("gen", [_uniform_sphere, _clumpy])
    def test_approx_matches_exact_substructure_regime(self, gen):
        """In the substructure/uniform regime (Q <~ 0.8) -- where q_approx is used
        for substructure inference -- the differentiable q_approx tracks the exact
        CW04 Q tightly (uniform ~0.04, clumpy ~0.005). In the concentrated regime
        (Q > 0.8) it over-reads by ~0.1 (shown in the validation figure); ordering
        is preserved everywhere (next test)."""
        pos = gen(400, 0)
        q_ex = compute_q_parameter(pos)
        q_ap = float(q_approx(jnp.asarray(pos)))
        assert abs(q_ap - q_ex) < 0.06, f"q_approx={q_ap:.3f} vs exact={q_ex:.3f}"

    def test_approx_preserves_ordering(self):
        """q_approx preserves the concentrated > uniform > clumpy ordering across
        the full range (monotonic in substructure, even where it over-reads)."""
        qp = float(q_approx(jnp.asarray(_concentrated(400, 1))))
        qu = float(q_approx(jnp.asarray(_uniform_sphere(400, 1))))
        qc = float(q_approx(jnp.asarray(_clumpy(400, 1))))
        assert qc < qu < qp, f"q_approx ordering: clumpy={qc:.3f}, unif={qu:.3f}, conc={qp:.3f}"

    # AD-vs-FD for q_approx wrt a radial-concentration parameter is owned by the grad-audit
    # registry (tests/validation/grad_audit/registry.py :: q_approx[EFF], where the EFF slope
    # gamma is the canonical concentration channel); see
    # docs/website/50-validation/differentiability-audit.md. The former
    # test_differentiable_wrt_concentration was removed here (audit T6 consolidation; registry
    # is SoT). The physics tests (CW04 baselines, ordering, exact-vs-approx) stay.


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
