"""gravoturb_fdf — gravoturbulent-1D + fractal-density-field (FDF) initial conditions.

Clean-room rewrite (2026-06) of the gravoturbulent + FDF IC subsystem, authored from
PDF-grounded theory (Burkhart & Mocz 2019; Parmentier & Pasquali 2020; Padoan &
Nordlund 2011; Federrath+2010; Heyer+2009; Kim & Ryu 2005).

**Status: EXPERIMENTAL — follow-up paper. NOT part of the initial progenax/jaxstro
release and NOT shipped in the progenax wheel.** This standalone package depends one-way
on ``progenax.cluster.turbulence``; nothing in released progenax imports it.

Layers
------
- ``theory``      : BM19 / PP20 / PN11 1D density-PDF theory (JAX-native, differentiable).
- ``field``       : 3D realization — GRF + rank copula, dense-tail mask, star sampling.
- ``diagnostics`` : CW04 Q substructure metric (numpy/scipy, non-differentiable).
- ``validation``  : AC1–AC10 acceptance scripts that print real numbers.

Every formula is re-derived from the held PDFs and re-validated against committed,
printing acceptance scripts before it is believed. See
``docs/plans/2026-06-05-fdf-clean-room-spec.md`` (§8 authoritative).
"""

__version__ = "0.0.0.dev0"
__all__: list[str] = []
