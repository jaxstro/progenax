"""Label-only variance diagnosis using the retained rung-4 IMF configuration.

For one star let u=m-E[m], v=m'-E[m'], s=p'/p, c=E[u*s], and
I=E[s*s]. For R independent N-star catalogues the mean-mass gradient is the
mean pathwise derivative plus the unbiased sample covariance of mean mass
and the whole-catalogue score. Its exact variance is

    (Var(m)*I+c*c)/(R-1)
    + (Var(m')+E[u*u*s*s]+2*E[u*v*s]-Var(m)*I-2*c*c)/(N*R).

The leading term survives N -> infinity. Squared total mass has the same
leading relative noise because its fluctuation is 2*N*E[m]*sum(u), while
its mean derivative is 2*N*N*E[m]*(E[m']+c). That statement is asymptotic;
it is not a new finite-N precision qualification. No physical catalogue or
equilibrium solve is executed here. Small exhaustive cases check algebra only.
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from progenax.imf.smooth import Maschberger
from progenax.profiles.limepy_multimass import _bin_imf


def mean_mass_gradient_variance(mass, probability, dm, dp, n, r):
    """Exact finite-N variance for the leave-one-out mean-mass gradient."""
    u = mass - probability @ mass
    v = dm - probability @ dm
    score = dp / probability
    mass_var = probability @ u**2
    fisher = probability @ score**2
    covariance = probability @ (u * score)
    limit = (mass_var * fisher + covariance**2) / (r - 1)
    correction = (
        probability @ (v**2 + u**2 * score**2 + 2 * u * v * score)
        - mass_var * fisher
        - 2 * covariance**2
    ) / (n * r)
    return limit + correction, limit


def check_exhaustive_algebra():
    """Check the formula against exact assignment sums, not synthetic clusters."""
    mass = np.array([1.0, 3.0])
    probability = np.array([0.4, 0.6])
    dm = np.array([0.2, -0.1])
    dp = np.array([0.1, -0.1])
    records = []
    for n, r in ((2, 2), (3, 3)):
        assignments = list(itertools.product(range(2), repeat=n))
        mean = second = 0.0
        for batch in itertools.product(assignments, repeat=r):
            labels = np.array(batch)
            weight = np.prod(probability[labels])
            value = mass[labels].mean(axis=1)
            score = (dp / probability)[labels].sum(axis=1)
            baseline = (value.sum() - value) / (r - 1)
            gradient = dm[labels].mean() + np.mean((value - baseline) * score)
            mean += weight * gradient
            second += weight * gradient**2
        variance = second - mean**2
        formula, _ = mean_mass_gradient_variance(mass, probability, dm, dp, n, r)
        np.testing.assert_allclose(mean, probability @ dm + dp @ mass, rtol=1e-10)
        np.testing.assert_allclose(variance, formula, rtol=1e-10, atol=0.0)
        records.append(
            dict(n=n, r=r, enumerated_variance=variance, formula=float(formula))
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Retain existing evidence: {args.output}")
    assert jax.config.jax_enable_x64
    path = Path(
        "docs/reviews/plummer-om-artifacts/general-observables-qualification.json"
    )
    frozen = json.loads(path.read_text())
    protocol = frozen["protocol"]
    n = protocol["target_stars"]
    r = protocol["realizations_per_batch"]
    b = len(protocol["seeds"])

    def components(alpha):
        mass, budget = _bin_imf(
            Maschberger(alpha=alpha, m_min=0.1, m_max=20.0), 4, (0.1, 20.0)
        )
        number = budget / mass
        return mass, number / number.sum()

    mass, probability = components(protocol["theta"][0])
    dm, dp = jax.jacfwd(components)(protocol["theta"][0])
    derivative = probability @ dm + dp @ mass
    variance, limit = mean_mass_gradient_variance(mass, probability, dm, dp, n, r)
    mu = probability @ mass
    mass_squared_derivative = (
        n * (2 * (probability @ (mass * dm)) + dp @ mass**2)
        + 2 * n * (n - 1) * mu * derivative
    )
    reference = frozen["reference_fd_1e-05"][0][0]
    np.testing.assert_allclose(mass, frozen["component_mass"], rtol=1e-10, atol=0.0)
    # Compare the mass marginal derivative to the retained full-constructor FD;
    # use its existing deterministic gate, not a new Monte Carlo criterion.
    np.testing.assert_allclose(
        mass_squared_derivative, reference, rtol=protocol["deterministic_relative_gate"]
    )
    sources = [
        Path(__file__),
        Path("src/progenax/profiles/limepy_multimass.py"),
        Path("src/progenax/cluster/sampling.py"),
    ]
    report = {
        "date": "2026-09-30",
        "mode": "SCIENCE Test, analytic variance diagnosis; not a new campaign",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "x64": True,
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "frozen_evidence": str(path),
        "frozen_evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "claim": "Increasing stars per catalogue does not remove the leading whole-catalogue score variance at fixed realization count.",
        "falsifier": "Exact small-assignment variance disagrees with the derived formula, or the leading variance term vanishes in the retained IMF marginal.",
        "exhaustive_algebra_checks": check_exhaustive_algebra(),
        "configuration": dict(
            n_stars=n,
            realizations_per_batch=r,
            batches=b,
            alpha=protocol["theta"][0],
            mass_range=[0.1, 20.0],
            n_components=4,
            bin_nodes=64,
        ),
        "mass": mass.tolist(),
        "target_probability": probability.tolist(),
        "max_probability_difference_from_fitted_frozen_model": float(
            jnp.max(jnp.abs(probability - jnp.array(frozen["component_probability"])))
        ),
        "mean_mass_alpha_derivative": float(derivative),
        "score_contribution": float(dp @ mass),
        "pathwise_contribution": float(probability @ dm),
        "exact_mean_mass_batch_variance": float(variance),
        "large_N_batch_variance": float(limit),
        "mean_mass_exact_relative_SE_of_batch_mean": float(
            jnp.sqrt(variance / b) / jnp.abs(derivative)
        ),
        "mass_squared_large_N_relative_SE_of_batch_mean": float(
            jnp.sqrt(limit / b) / jnp.abs(derivative)
        ),
        "frozen_mass_squared_empirical_relative_SE": frozen["standard_error"][0][0]
        / abs(reference),
        "mass_squared_bin_reference_derivative": float(mass_squared_derivative),
        "mass_squared_frozen_FD_reference_derivative": reference,
        "verdict": "Prediction supported by exact mean-mass algebra and a nonzero leading term; frozen mass-squared precision failure remains unchanged.",
        "limits": [
            "IMF target-number marginal; full fit probability differs at its retained numerical residual.",
            "Mass-squared variance statement is a leading large-N approximation, not exact finite-N qualification.",
            "Eight empirical batches do not establish bias or accurately determine the true variance.",
            "No estimator, public functionality, threshold, domain or physical sampling distribution changed.",
            "General observable precision and broader domains remain rung-4 release requirements; no brute-force or closed Engine B campaign repeated.",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "exact_mean_mass_batch_variance",
                    "large_N_batch_variance",
                    "mass_squared_large_N_relative_SE_of_batch_mean",
                    "frozen_mass_squared_empirical_relative_SE",
                    "verdict",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
