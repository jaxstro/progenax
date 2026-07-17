"""F10 driving↔compressive-fraction relation (theory/driving.py; Phase 3).

chi_f10 inverts Federrath+2010 Eq. (22), b = √D·⟨Ψ⟩ with Ψ = E_long/E_tot (Eq. 21),
verified against the held PDF 2026-07-16 (p. 11–12, Fig. 8): the radical is over D
ONLY, so in 3-D χ ≡ Ψ = b/√3. Paper-anchored checks: solenoidal b=1/3 → χ≈0.19 and
compressive b=1 → χ≈0.577 match F10 Fig. 14's measured E_long/E_tot (~0.2 / ~0.5–0.6).
NB Eq. (23) is the FORCING-side cubic fit b̃(ζ) — deliberately not what we invert.
"""

import jax
import numpy as np
import pytest
from gravoturb.theory.driving import chi_f10

pytestmark = [pytest.mark.experimental, pytest.mark.unit]


def test_chi_f10_paper_anchors():
    assert float(chi_f10(1.0 / 3.0)) == pytest.approx(1.0 / (3.0 * np.sqrt(3.0)), rel=1e-12)
    assert float(chi_f10(1.0)) == pytest.approx(1.0 / np.sqrt(3.0), rel=1e-12)
    # forced turbulence never reaches chi=1 (F10 Fig. 8/14): the max over b∈(0,1] is ~0.577
    assert float(chi_f10(1.0)) < 0.6


def test_chi_f10_monotone_and_differentiable():
    bs = np.linspace(0.05, 1.0, 20)
    chis = [float(chi_f10(b)) for b in bs]
    assert all(c1 < c2 for c1, c2 in zip(chis, chis[1:]))
    g = float(jax.grad(chi_f10)(0.5))
    assert g == pytest.approx(1.0 / np.sqrt(3.0), rel=1e-12)  # linear relation
