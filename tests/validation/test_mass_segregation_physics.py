"""Physical-correctness validation for the Λ_MSR mass-segregation diagnostic.

Validates ``progenax.diagnostics.compute_lambda_msr`` (Allison et al. 2009) against
*analytic / hand-constructed* ground truth — NO N-body evolution (that is deferred to the
gravax session; see docs/notes/2026-06-08-gravax-segregation-validation-followup.md).

Λ_MSR = ⟨L_random⟩ / L_massive  (>1 segregated, ≈1 none, <1 inverse), error = σ_random/L_massive,
verified against the held ApJ 700 L99 PDF (see docs/website/99-bibliography/per-paper/allison-2009.md).

Each test is *discriminating*: an inverted or mis-normalised estimator would fail it.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist

from progenax.diagnostics import compute_lambda_msr


# ── Tier A1: unsegregated → Λ ≈ 1 ──
def test_unsegregated_lambda_is_unity():
    """Masses uncorrelated with position ⇒ the massive subset is a random subset ⇒ Λ≈1.

    Averaged over many random mass assignments to remove the single-realization scatter.
    """
    rng = np.random.default_rng(0)
    N, N_massive = 120, 8
    positions = rng.normal(size=(N, 3))
    lambdas = []
    for s in range(60):
        masses = rng.random(N)  # random, position-independent
        lam, _ = compute_lambda_msr(positions, masses, N_massive=N_massive,
                                    N_random_samples=80, seed=s)
        lambdas.append(lam)
    mean_lambda = float(np.mean(lambdas))
    assert 0.9 < mean_lambda < 1.1, f"unsegregated mean Λ={mean_lambda:.3f}, expected ≈1"


# ── Tier A2: maximally segregated → Λ ≫ 1 ──
def test_maximal_segregation_lambda_large():
    """Massive stars packed in a tiny core, rest spread over unit radius ⇒ L_massive→0 ⇒ Λ≫1."""
    rng = np.random.default_rng(1)
    N, N_massive = 200, 10
    field = rng.normal(size=(N, 3))
    field /= np.linalg.norm(field, axis=1, keepdims=True)
    field *= rng.uniform(0, 1, (N, 1)) ** (1 / 3)  # uniform in unit ball
    masses = rng.random(N)
    top = np.argsort(-masses)[:N_massive]
    field[top] = rng.normal(scale=1e-3, size=(N_massive, 3))  # tight core
    lam, _ = compute_lambda_msr(field, masses, N_massive=N_massive, N_random_samples=200, seed=2)
    assert lam > 20.0, f"maximal-segregation Λ={lam:.2f}, expected ≫1"


# ── Tier A3: inverse segregation → Λ < 1 ──
def test_inverse_segregation_lambda_below_unity():
    """Massive stars on a wide shell, low-mass stars in a tight core ⇒ Λ<1."""
    rng = np.random.default_rng(3)
    N, N_massive = 200, 10
    positions = rng.normal(scale=0.05, size=(N, 3))  # everyone tight by default
    masses = rng.random(N)
    top = np.argsort(-masses)[:N_massive]
    shell = rng.normal(size=(N_massive, 3))
    shell /= np.linalg.norm(shell, axis=1, keepdims=True)
    positions[top] = shell  # massive stars on unit shell (spread out)
    lam, _ = compute_lambda_msr(positions, masses, N_massive=N_massive, N_random_samples=200, seed=4)
    assert lam < 0.7, f"inverse-segregation Λ={lam:.3f}, expected <1"


# ── Tier A4: exact hand-computed value (N_massive=2) ──
def test_exact_value_two_massive():
    """For N_massive=2, L_massive = the massive pair distance and ⟨L_random⟩ → the mean of ALL
    pairwise distances (uniform 2-subsets). So Λ_true = mean(all pair dists)/d(massive pair),
    computed independently via scipy.pdist — an exact analytic check of the estimator."""
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],   # the two most massive (d = 1.0)
        [0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0], [2.0, 2.0, 1.0],
    ])
    masses = np.array([10.0, 9.0, 1.0, 1.0, 1.0, 1.0])  # first two are most massive
    d_massive = float(np.linalg.norm(positions[0] - positions[1]))   # = 1.0
    lambda_true = float(np.mean(pdist(positions))) / d_massive        # independent reference
    lam, _ = compute_lambda_msr(positions, masses, N_massive=2, N_random_samples=20000, seed=5)
    assert lam == pytest.approx(lambda_true, rel=0.03), f"Λ={lam:.4f} vs exact {lambda_true:.4f}"


# ── Tier A5: estimator convergence (more random samples → tighter, → exact) ──
def test_estimator_converges_with_random_samples():
    """The Λ estimate's seed-to-seed scatter shrinks as N_random_samples grows, and converges
    to the exact value (Tier A4 config)."""
    positions = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 4.0], [2.0, 2.0, 1.0],
    ])
    masses = np.array([10.0, 9.0, 1.0, 1.0, 1.0, 1.0])
    lambda_true = float(np.mean(pdist(positions))) / 1.0

    def scatter(n_samples):
        vals = [compute_lambda_msr(positions, masses, N_massive=2,
                                   N_random_samples=n_samples, seed=s)[0] for s in range(40)]
        return float(np.std(vals)), float(np.mean(vals))

    std_lo, _ = scatter(20)
    std_hi, mean_hi = scatter(3000)
    assert std_hi < std_lo, f"scatter did not shrink: {std_hi:.3f} !< {std_lo:.3f}"
    assert mean_hi == pytest.approx(lambda_true, rel=0.02)


# ── Tier A6: binary-contamination caveat (documented behaviour) ──
def test_tight_massive_binary_inflates_lambda():
    """A tight massive binary gives a spuriously short L_massive → inflated Λ (docstring caveat).

    The effect bites hardest when the binary *dominates* the massive MST — i.e. small N_massive
    (here N_massive=2, where L_massive IS the pair separation → →0 → Λ→∞). At large N_massive the
    binary is only one of ~N_massive−1 edges, so a single pair inflates Λ only mildly; this test
    therefore demonstrates the caveat in the regime where it is dramatic (verified ~10³–10⁴×).
    Practical mitigation (docstring): use binary centre-of-mass positions.
    """
    rng = np.random.default_rng(6)
    N = 200
    positions = rng.normal(size=(N, 3))
    positions /= np.linalg.norm(positions, axis=1, keepdims=True)
    positions *= rng.uniform(0, 1, (N, 1)) ** (1 / 3)
    masses = rng.random(N)
    lam_clean, _ = compute_lambda_msr(positions, masses, N_massive=2,
                                      N_random_samples=400, seed=7)
    top2 = np.argsort(-masses)[:2]
    positions[top2[1]] = positions[top2[0]] + np.array([1e-4, 0.0, 0.0])  # tight massive binary
    lam_binary, _ = compute_lambda_msr(positions, masses, N_massive=2,
                                       N_random_samples=400, seed=7)
    assert lam_binary > 50.0 * lam_clean, (
        f"tight massive binary should strongly inflate Λ: clean={lam_clean:.2f} "
        f"binary={lam_binary:.1f}")


# ── Tier A7: input validation ──
def test_input_validation():
    pos = np.random.default_rng(8).normal(size=(20, 3))
    m = np.random.default_rng(9).random(20)
    with pytest.raises(ValueError):
        compute_lambda_msr(pos, m, N_massive=1)        # < 2
    with pytest.raises(ValueError):
        compute_lambda_msr(pos, m, N_massive=20)       # >= N


# ── Tier A8: reproducibility ──
def test_reproducible_with_seed():
    rng = np.random.default_rng(10)
    pos, m = rng.normal(size=(100, 3)), rng.random(100)
    a = compute_lambda_msr(pos, m, N_massive=10, N_random_samples=100, seed=123)
    b = compute_lambda_msr(pos, m, N_massive=10, N_random_samples=100, seed=123)
    assert a == b
