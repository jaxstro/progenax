r"""Reusable forward-model pieces for the B12 binary-inflated dynamical-mass demo.

A scripts-local (NOT a packaged API) helper module for
``scripts/demo_binary_dynamical_mass.py``. It provides the three reusable parts
of the demo's forward model:

* :func:`project_los_velocity` -- isotropic line-of-sight projection of a 3-velocity;
* ``build_korb_kernel`` -- the sigma-independent flux-weighted binary blend kernel
  ``K_orb`` (Moe & Di Stefano P-q-e orbits + Tout+1996 ZAMS luminosity weighting),
  built in ``BINARY`` units (Msun, AU, yr) and returned in km/s;
* ``predict_vlos_counts`` -- the differentiable binned single+binary mixture model.

JAX-native (``jax.numpy``); the kernel/predict pieces are jit/grad-safe so the
demo can differentiate the mixture model in ``(sigma, f_b)``. float64 is the
demo's responsibility (``import progenax`` enables it before this module is used).
"""
import jax.numpy as jnp


def project_los_velocity(vel3, los_hat):
    r"""Line-of-sight component of a 3-velocity along a unit direction.

    ``v_los = vel3 . los_hat``. For ``los_hat`` drawn isotropically, the
    projection of a fixed velocity has zero mean and variance ``|vel3|^2 / 3``
    (the velocity's energy shared equally over three orthogonal axes).

    Parameters
    ----------
    vel3 : (3,) array
        Velocity vector [any consistent units; km/s in the B12 demo].
    los_hat : (3,) array
        Line-of-sight unit vector (caller normalizes).

    Returns
    -------
    v_los : scalar, the LOS velocity component (same units as ``vel3``).
    """
    return jnp.dot(vel3, los_hat)
