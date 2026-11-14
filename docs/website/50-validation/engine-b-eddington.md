---
title: Multi-component Eddington equilibria (Engine B)
description: "Validation suite for MultiComponentCluster.from_density_profiles: Plummer/EFF/King density components in one shared self-consistent potential, per-component DFs by Eddington inversion (optionally Osipkov-Merritt), cross-validated against Engine A and analytic DF oracles."
---
# Multi-component Eddington equilibria (Engine B)

**Engine B** (`MultiComponentCluster.from_density_profiles`) builds Plummer, EFF,
and King **density** components into a shared potential by cumulative-trapezoid quadrature
(King component construction retains its own profile ODE), inverts each component's DF
by the generic `eddington_invert` in that shared $\Psi$ (optionally
Osipkov-Merritt per component, $r_{a,j}$), and samples component positions and velocities
with **no external virial rescale**. Hard truncation introduces the edge
approximation quantified below. It complements
[Engine A](multimass-equilibrium.md) (DF-defined lowered-isothermal/LIMEPY): A
starts from a DF family, B from prescribed densities — and the two engines must
agree where they overlap. Test file:
`tests/validation/test_engine_b_physics.py` (**6 tests**; plus the `TestEngineB`
unit suite in `tests/unit/cluster/test_multicomponent.py`); figures + PASS/FAIL
gate: `scripts/validate_multicomponent_eddington.py`.

## What is verified

Each row maps to assertions in `test_engine_b_physics.py`; the **Measured**
column is the refreshed standalone validation run (2026-09-29, **ALL PASS**,
11/11 gates, float64, 4096-node catalogue speeds). The interior-density row is
a separate retained diagnostic; its test was rerun and passes. The headline mix is a
Plummer halo ($r_h=2$ pc, 60% of the mass) + EFF $\gamma=5$ core ($a=0.8$ pc,
$r_t=9$ pc, 40%).

```{list-table}
:header-rows: 1

* - Check
  - Measured
  - Gate
* - King A-vs-B radial KS distance ($N=2\times10^4$, same seed)
  - $0.0009$
  - $< 0.02$
* - King A-vs-B max $|\sigma_B/\sigma_A - 1|$ (interior bins)
  - $0.0004$
  - $< 0.02$
* - Plummer $f(E)$ vs $E^{7/2}$ law (untruncated zero point)
  - $1.06\times10^{-4}$
  - $< 10^{-3}$
* - Plummer $f(E)$ vs exact truncated closed form
  - $8.86\times10^{-6}$
  - $< 10^{-4}$
* - Headline theory $Q_j$ (DF-weighted oracle)
  - $[0.50038,\,0.50012]$
  - $0.5 \pm 3\times10^{-3}$
* - Headline sampled global $Q$ ($N=30$k, unscaled)
  - $0.4969$
  - $0.5 \pm 0.02$
* - Headline predicted hybrid $Q_j$
  - $[0.4953,\,0.4985]$
  - (the prediction)
* - Headline sampled $Q_j$ (3 seeds $\times$ 16k)
  - $[0.4909 \pm 0.0076,\,0.5048 \pm 0.0016]$; max $|\Delta$ vs pred$| = 0.0063$
  - $< 0.012$
* - OM max $|\beta_{\rm sampled} - r^2/(r^2+r_a^2)|$ (4 seeds $\times$ 20k, 8 bins)
  - $0.0212$
  - $< 0.05$
* - DF-density fidelity (truncation-consistent, $r < r_{h,j}$)
  - $1.06\times10^{-3}$ (OM build) / $2.4\times10^{-4}$ (isotropic)
  - $< 5\times10^{-3}$ (test gate)
* - Gradients AD-vs-FD: halo $r_h$ / mass-fraction $t$ / $r_{a,j}[0]$
  - $8.51\times10^{-9}$ / $6.54\times10^{-7}$ / $2.01\times10^{-8}$
  - $< 10^{-3}$ each
```

The King A-vs-B anchor is *the* trust anchor: the same physical model
($W_0=5$, $r_c=1$ pc) built by two **independent** engines — A's
lowered-isothermal DF + coupled Poisson ODE vs B's prescribed King density +
Poisson quadrature + Eddington inversion — agrees in the sampled radial CDF
(KS $0.0009$) and dispersion profile (max dev $0.0004$). The exact-quadrature
$Q_j$ oracle alone is *necessary, not sufficient* ($2T_j + W_j = 0$ holds for
any positive $f$ in a consistent $(\Psi, d\Psi/dr)$ pair); the cross-engine and
closed-form-DF anchors carry the inversion-correctness burden.

## Figure

:::{figure} figures/engine_b_eddington.png
:label: val-engine-b
:width: 100%

**Engine B validation summary** (`scripts/validate_multicomponent_eddington.py`,
ALL PASS). **(a)** King A-vs-B $\sigma_{1d}(r)$ overlay: the two independent
engines coincide bin by bin (max dev $0.0004$). **(b)** Plummer ergodic DF:
the inverter's $f(E)$ on the BT2008 $f \propto E^{7/2}$ law (log–log), with a
residual inset showing $\le 1.06\times10^{-4}$ against the power law and
$\le 8.86\times10^{-6}$ against the exact truncated closed form. **(c)** DF
fidelity: $\rho_{{\rm DF},j}$ reconstructed from each component's $f_j$ table
integrates back to the prescribed $\rho_{{\rm presc},j} - \rho_j(r_t)$
(truncation-consistent form), halo and core. **(d)** Osipkov-Merritt: sampled
$\beta_{\rm halo}(r)$ tracks $r^2/(r^2+r_a^2)$ for $r_a = 3$ pc (max dev
$0.0212$). **(e)** Per-component virial summary — theory oracle, predicted
hybrid expectation, and sampled $Q_j$ (3 seeds $\pm$ sem): the sampled values
sit on the *prediction*, not on idealized 0.5 (see below).
:::

## Physical interpretation

1. **The $E^{7/2}$ oracle needs the untruncated zero point.** With the
   truncated zero point $\Psi = \Phi(r_t) - \Phi$ the energies shift by
   $c = M/\sqrt{r_t^2+a^2}$ in the reference convention $G=1$, and the power law picks up an $O(3.5c/E) \approx
   30\%$ deviation at $E = 0.1\Psi_0$ even at $r_t = 100a$. The law is tested
   with $\Psi = -\Phi$ (untruncated), and the truncated case is covered by the
   **exact closed form including the boundary term**
   for the reference density $\rho=k(\Psi+c)^5$, where $k$ fixes its normalization
   and $b=E+c$:

   ```{math}
   f(E) = \frac{20k\left(2b^3\sqrt{E}-2b^2E^{3/2}
   +\frac65 bE^{5/2}-\frac27 E^{7/2}\right)+5kc^4/\sqrt{E}}
   {\sqrt8\,\pi^2}.
   ```

   This checks amplitude and shape ($8.86\times10^{-6}$). The potential zero
   point is part of the oracle definition.
2. **The chosen ergodic/OM family need not realize a prescribed density mix.** The
   plan's drafted EFF core $a=0.4$ in the Plummer-halo shared potential has a
   *genuinely negative* Eddington DF ($\min f/\max|f| = -0.20$,
   resolution-independent; verified with the closed-form two-Plummer oracle,
   since $\gamma=5$ EFF $\equiv$ Plummer). The $f_j \ge 0$ realizability gate
   refuses such builds, naming the component, its $f_{\min}$, and a remedy;
   a close-out sweep located the gate flip between $a = 0.65$ (refused) and
   $a = 0.68$ (realizable). The headline therefore uses $a = 0.8$.
   Realizability is physics, not a numerical nuisance.
3. **The hybrid-sampling $Q_j$ plateau is predicted physics, not bias.**
   Engine B samples a *hybrid*: positions from the prescribed $\rho_j$, speeds
   from $f_j$. A hard-truncated component has $\rho(r_t) > 0$ — an edge offset
   no ergodic $f(E)$ can carry (the Eddington pair represents
   $\rho(\Psi) - \rho(0)$) — so the truncated halo's sampled $Q_j$ plateaus
   *below* 0.5. The exact-quadrature hybrid expectation predicts $0.4953$; an
   18-seed $\times$ 16k campaign measured $0.4947 \pm 0.0014$ ($0.4\sigma$ from
   the prediction — this is the headline figure the theory pages cite). The
   in-script PASS/FAIL table above runs a faster 3-seed $\times$ 16k check
   ($0.4909 \pm 0.0076$ for the halo, consistent with the 18-seed campaign
   within its larger SEM); both are consistent with the hybrid prediction.
   The refreshed figure uses sample standard deviation divided by the square
   root of the number of seeds for its SEM.
   The gate is against the **prediction**, never a tuned offset.

### King density derivatives

The King branch obtains its potential derivative from the integrated Poisson
identity using the closed-form density. This avoids differentiating a
piecewise-linear potential interpolant before Eddington inversion. The
[theory page](../10-theory/populations/eddington-engine.md#king-poisson-identity)
gives the identity and conventions.

## Differentiability

The full Engine B build (`build_engine_b_state`) is differentiable: AD matches
central FD through the shared-potential pass + Eddington inversion in the halo
scale $r_h$ ($8.51\times10^{-9}$), the mass fraction $t$
($6.54\times10^{-7}$), and the per-component anisotropy radius $r_{a,j}$
($2.01\times10^{-8}$) — all far inside the $10^{-3}$ gate. These checks hold the King component's internal profile solve fixed. The derived
cluster radius carries derivatives through the largest finite component extent,
or through the enclosed-mass constraint for all-infinite profiles. A switch
between equally large finite extents is not a smooth parameter point. Explicit
overrides involving a King component require concrete values for edge validation.

## Population-moment gradients

`estimate_population_moments` integrates mean mass and squared radius over the
component population and radial CDF. For mean squared speed, it integrates the
clipped, interpolated distribution function (DF) at energy knots and zero
crossings, averages Osipkov–Merritt directions analytically, and uses four-point
Gauss quadrature in each radial-CDF interval. The correlation between component
assignment and sampled total stellar mass is included. Engine B moments are
deterministic; random keys and integration draw counts do not change them.

:::{note} Checked configuration and limits
The Plummer halo/EFF core above passes the 0.3% numerical criterion in both
isotropic and anisotropic configurations (halo anisotropy radius 3 pc, isotropic
core). At default constructor resolution, the largest discrepancy is 0.000786%
across the three profile-parameter derivatives and the finite halo-anisotropy
radius derivative. The target population is 100,000 stars and calculations use
float64. The reference integrates conditional DF moments against the prescribed
density on a finer radial grid; it shares the potential and inversion code.

Single-component checks additionally cover the EFF outer slope at $\gamma=5$
and King concentration at $W_0=5$, through the complete profile construction.
Squared-radius and squared-speed derivatives pass the same 0.3% criterion at
two finite-difference steps; the largest discrepancy is 0.118%. Forward and
reverse AD agree within $10^{-10}$ relative tolerance. Length-scaling identities
also pass for Plummer, EFF, King and a three-component Plummer model.

These comparisons do not establish arbitrary-profile population gradients,
higher derivatives or general user-defined observables. The separate physics
anchors above assess the inversion itself.
:::

Catalogue sampling uses 4096 speed nodes in checkpointed batches of 32. Its
inverse-CDF discretization is separate from the continuous interpolated-DF
moment integral and requires convergence checks. Moment quadrature also uses
checkpointed batches of 32 to bound reverse-mode storage. Reproduce the retained
derivative diagnostic with `scripts/check_engine_b_deterministic_candidate.py
--public-defaults`; existing artifacts are protected from overwrite.

## How to run

```bash
# physics tests (6 validation tests; the unit suite covers realizability + extraction pins)
JAX_ENABLE_X64=true PYTHONPATH=src .venv/bin/pytest tests/validation/test_engine_b_physics.py -q
pytest tests/unit/cluster/test_multicomponent.py -q

# regenerate the 5-panel figure with the 11-row PASS/FAIL table
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_multicomponent_eddington.py
```

## What this suite does *not* test

- **No N-body stationarity test of the assembled ICs.** The retained virial,
  cross-engine and DF-moment checks constrain selected continuum/numerical
  properties. They do not establish dynamical stationarity of every finite
  hybrid realization.
- **The hard-truncation edge approximation is quantified, not eliminated** — a
  hard-truncated prescribed density is only approximately stationary at the
  edge ($\rho(r_t)>0$ cannot be carried by any ergodic $f(E)$); the sampled
  global-$Q$ gate is $0.02$ (not $0.002$) for this reason, and the
  per-component sampled $Q_j$ is gated on the hybrid prediction.
- **Broader population-gradient coverage remains open** — passing the retained
  halo/core cases does not certify the entire input domain.

## References

Eddington inversion and the $f \propto E^{7/2}$ Plummer DF: Binney & Tremaine
(2008); the King model is {cite:t}`King1966`, the EFF profile
{cite:t}`ElsonFallFreeman1987`. Engine A (the DF-defined family it cross-validates against)
is documented at [](multimass-equilibrium.md); the lowered-model-family
roadmap at [](../10-theory/spatial-profiles/lowered-model-family.md).
This validation backs the Phase-2 Engine-B close-out.
