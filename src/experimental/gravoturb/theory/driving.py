r"""Turbulence-driving ↔ compressive-fraction relation (Federrath et al. 2010).

F10 measure the density-PDF driving parameter ``b`` (of σ_s² = ln(1+b²ℳ²)) against the
longitudinal power fraction of the VELOCITY FIELD,

    Ψ = E_long / E_tot                      (F10 Eq. 21)
    b ≈ √D · ⟨Ψ⟩                            (F10 Eq. 22; the radical is over D ONLY)

verified against the held PDF (docs/core-papers/Federrath-2010.pdf, p. 11–12, Fig. 8)
on 2026-07-16. In 3-D the inversion is therefore

    χ ≡ Ψ = b / √3,

the F10-consistent default for the Helmholtz-coupled construction's compressive power
weight. Paper anchors: solenoidal b = 1/3 → χ ≈ 0.19; compressive b = 1 → χ ≈ 0.577 —
both match F10 Fig. 14's measured E_long/E_tot (~0.2 and ~0.5–0.6). Two consequences:
**χ never reaches 1 for forced turbulence** (the χ→1 corner is an extrapolation beyond
F10's measurements), and F10 Eq. (23) — b̃(ζ) = 1/D + (D−1)/D·(F_long/F_tot)³ — is the
FORCING-side fit in the driving parameter ζ, deliberately NOT what this module inverts.

JAX-native, differentiable.
"""

import jax.numpy as jnp
from jaxtyping import Array, Float

# F10 Eq. 22 geometric factor for D = 3 (the diagonal of the unit cube).
_SQRT_D_3D = jnp.sqrt(3.0)


def chi_f10(b: Float[Array, ""]) -> Float[Array, ""]:
    r"""F10-consistent compressive power fraction χ = b/√3 (3-D; Eqs. 21–22).

    ``b`` is the FK10 driving parameter of the density PDF (1/3 solenoidal → 1
    compressive); the return is the longitudinal share of the velocity power the
    Helmholtz construction should impose. Range over b ∈ (0, 1]: χ ∈ (0, 0.577] —
    forced turbulence never reaches χ = 1. Differentiable (linear).
    """
    return b / _SQRT_D_3D
