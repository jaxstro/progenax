"""Reference-LIMEPY worker: runs ONLY in the pinned ephemeral subprocess env.

The canonical published LIMEPY (Gieles & Zocchi 2015) at
``~/projects/jaxstro-dev/ref-repos/limepy`` does not run under the project env
(scipy 1.17 breaks it: float ``nsteps`` at limepy.py:488 and an incompatible
dopri5 ``solout`` API). ``scripts/validate_limepy_reference.py`` therefore
invokes THIS worker via::

    env -u VIRTUAL_ENV uv run --no-project --python 3.11 \
        --with "numpy==1.26.4" --with "scipy==1.11.4" \
        python scripts/_limepy_reference_worker.py '<config json>' <out.npz>

and caches the outputs as .npz under ``validation/data/limepy_reference/``.
numpy/scipy are allowed HERE (and only here) -- this file never imports into
the JAX-native progenax process.

Config JSON keys: W0, g, and optionally mj, Mj, delta, eta, ra (multimass /
anisotropy). Multimass models always use ``meanmassdef='central'`` so the
reference's mean mass is the GZ15 eq-26 central-density-weighted
m-bar = sum_j m_j alpha_j -- identical to progenax's ``bar_m`` (no Peuten
eq 8-9 W0 translation needed).
"""
import json
import os
import subprocess
import sys

# <progenax repo root>/../ref-repos/limepy (scripts/ is one level below the
# repo root); override with the LIMEPY_REF_REPO environment variable if set.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_REPO = os.environ.get(
    "LIMEPY_REF_REPO",
    os.path.join(os.path.dirname(_REPO_ROOT), "ref-repos", "limepy"))


def main():
    cfg = json.loads(sys.argv[1])
    out_path = sys.argv[2]

    sys.path.insert(0, REF_REPO)
    import numpy as np
    import scipy
    from limepy import limepy

    kwargs = {}
    multi = "mj" in cfg
    if multi:
        kwargs.update(mj=cfg["mj"], Mj=cfg["Mj"], delta=cfg["delta"],
                      eta=cfg.get("eta", 0.0), meanmassdef="central")
    if cfg.get("ra") is not None:
        kwargs["ra"] = cfg["ra"]

    m = limepy(cfg["W0"], cfg["g"], **kwargs)

    # Per-component arrays; single-mass models fall back to the global arrays.
    rhoj = np.atleast_2d(getattr(m, "rhoj", m.rho))
    v2j = np.atleast_2d(getattr(m, "v2j", m.v2))
    v2rj = np.atleast_2d(getattr(m, "v2rj", getattr(m, "v2r", m.v2 / 3.0)))
    v2tj = np.atleast_2d(getattr(m, "v2tj", getattr(m, "v2t", 2.0 * m.v2 / 3.0)))
    alpha = np.atleast_1d(getattr(m, "alpha", np.array([1.0])))
    mj = np.atleast_1d(getattr(m, "mj", np.array([1.0])))
    Mj = np.atleast_1d(getattr(m, "Mj", np.array([m.M])))
    raj = np.atleast_1d(getattr(m, "raj", np.array([m.ra])))

    try:
        sha = subprocess.run(["git", "-C", REF_REPO, "rev-parse", "HEAD"],
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "unknown"

    provenance = json.dumps({
        "ref_repo": REF_REPO,
        "ref_git_sha": sha,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "python": sys.version.split()[0],
        "config": cfg,
        "meanmassdef": "central" if multi else "n/a (single-mass)",
    })

    np.savez(
        out_path,
        r=m.r, rhoj=rhoj, v2j=v2j, v2rj=v2rj, v2tj=v2tj,
        alpha=alpha, mj=mj, Mj=Mj, raj=raj, mc=m.mc,
        rt=np.float64(m.rt), rh=np.float64(m.rh), rv=np.float64(m.rv),
        r0=np.float64(m.r0), mmean=np.float64(getattr(m, "mmean", 1.0)),
        converged=np.bool_(m.converged),
        provenance=np.str_(provenance),
    )
    print(f"wrote {out_path}: converged={m.converged}, rt/r0={m.rt / m.r0:.4f}, "
          f"rh/r0={m.rh / m.r0:.4f}, alpha={np.round(alpha, 4)}")


if __name__ == "__main__":
    main()
