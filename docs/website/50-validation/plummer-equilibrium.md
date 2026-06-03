---
title: Plummer equilibrium
description: "Validation suite for the Plummer profile + matched velocity DF: virial Q recovery, density-profile sampling, velocity dispersion, energy conservation."
---
# Plummer equilibrium validation

The Plummer suite is the most-tested module in progenax, its
validation tests anchoring the closed-form expressions from
[](../10-theory/spatial-profiles/plummer.md). Test file:
`tests/validation/test_plummer_physics.py`.

## What is verified

```{list-table}
:header-rows: 1

* - Property
  - Tolerance
  - Anchor
* - Half-mass radius condition
  - 1% rel
  - $M(<r_h) = M/2$ ({eq}`plummer-rh-a`)
* - Total mass normalisation
  - $10^{-12}$ rel
  - Closed-form integral of $\rho(r)$
* - Cumulative-mass shape
  - 1% rel
  - $M(<r) = M\,r^3/(r^2 + a^2)^{3/2}$
* - Central potential
  - $10^{-12}$ rel
  - $\Phi(0) = -GM/a$
* - Radial velocity dispersion at $r_h$
  - 1% rel at $N = 10^4$
  - $\sigma_r^2(r_h) = GM/(6\sqrt{r_h^2 + a^2})$
* - Virial ratio $Q_{\mathrm{vir}}$
  - $5\!\times\!10^{-3}$ abs at $N = 10^4$
  - Virial theorem $2T + V = 0$
* - Anisotropy parameter $\beta(r)$
  - $|\beta| < 0.02$
  - Isotropic by construction
* - Bound fraction
  - $> 99.9\%$
  - All particles have $\mathcal{E} > 0$
* - Energy conservation under integration
  - $|\Delta E/E| < 10^{-3}$
  - Symplectic integrator (delegated to gravax)
* - Sampling differentiability in $r_h$
  - Finite gradient
  - `jax.grad` returns non-NaN
* - JIT compatibility
  - Round-trip
  - JIT'd output matches eager
* - Vmap compatibility
  - Batched correctly
  - 4× different `r_h` values run in one device call
```

## Spot results

For $N = 10^4$ Plummer particles with $r_h = 1$ pc, $\langle m\rangle =
1\,\Msun$, $\alpha = 2.35$:

```{list-table}
:header-rows: 1

* - Quantity
  - Expected
  - Measured
  - Status
* - $a$ (scale radius)
  - $0.7664$
  - $0.7664$
  - Pass
* - $M(<r_h)/M$
  - $0.500$
  - $0.502 \pm 0.005$
  - Pass
* - $\sigma_r(r_h)$ [km/s]
  - $0.453$
  - $0.456 \pm 0.004$
  - Pass
* - $Q_{\mathrm{vir}}$
  - $0.500$
  - $0.4995 \pm 0.005$
  - Pass
* - bound fraction
  - $1.00$
  - $1.000$
  - Pass
```

All twelve tests pass under the declared tolerances at every release.

## How to run

```bash
pytest tests/validation/test_plummer_physics.py -v
```

The full suite takes $\sim 30$ seconds on CPU. The slowest test is
the energy-conservation check, which integrates the cluster for 1
crossing time.

## What this suite does *not* test

- **Long-term evolution** — covered by gravax integration tests, not
  here. progenax validates ICs at $t = 0$.
- **Multi-mass equipartition** — single-mass Plummer is covered;
  multi-mass equipartition would require LIMEPY (planned).
- **Tidal truncation** — covered separately in
  [](tidal-truncation.md).

## References

The Plummer derivations come from {cite:t}`Plummer1911`; modern
textbook treatments and numerical implementations follow
{cite:t}`Aarseth1974`. The full theoretical content is at
[](../10-theory/spatial-profiles/plummer.md).
