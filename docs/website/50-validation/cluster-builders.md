---
title: Cluster-builder validation
description: "Validation of the build_cluster convenience IC-builder layer: virial equilibrium across all five profiles, density recovery, tidal truncation, solid-body / differential rotation, and Osipkov-Merritt anisotropy — all through the one-call build_cluster / alias API."
---
# Cluster-builder validation

The `build_cluster` convenience layer is a thin, differentiable wrapper over the
composable `build_spatial_ic` core: it resolves the mass spec, auto-pairs the
scale-matched equilibrium velocity DF (`matched_velocity_df`), and applies optional
tidal / rotation modifiers. Because the base case is proven **bit-identical** to the
manual composition in default STELLAR units (exact `==` in
`tests/unit/builders/test_cluster_builders.py`), the underlying profile and DF
validation retains its recorded scope. This page checks the **builder surface**:
virial ratios, modifier behavior, unit conversions and differentiable parameters.
Composition does not extend the profiles' physical domain or certify stationarity.

Figures: `scripts/validate_cluster_builders.py` (prints expected-vs-measured for every
check and exits non-zero on any failure). End-to-end energy-ratio and construction checks across five profiles:
`tests/integration/test_cluster_builders_integration.py`. Gradient correctness: the
differentiability audit ([](./differentiability-audit.md)) registers eight measured
AD-vs-FD cases for the builders plus a tidal straight-through teeth test.

:::{admonition} Conventions for this suite
:class: note
- **Q is the energy ratio** $Q=T/|V|$, where $T$ is kinetic energy and $V$ is
  gravitational potential energy. $Q=0.5$ satisfies $2T+V=0$. `build_cluster` virial-scales
  to $Q=0.5$ by **default**, so all five aliases land at $0.5000$. This is scalar
  virial balance, not certification of N-body stationarity. Pass `Q=None` for the
  unscaled output of the matched samplers. Finite draws fluctuate, and
  hard-edged EFF output has the documented density/DF boundary approximation.
- **`tidal_radius` is Plummer-only.** King/EFF/Michie/LIMEPY carry a native truncation radius
  $r_t$; passing `tidal_radius` to them would double-truncate, so it raises. Tidal stripping is
  applied to the one untruncated profile (Plummer); the truncated families set $r_t$ on the
  profile (and that $r_t$ is differentiable for inference).
- **Density** is compared against the **shell-volume-weighted** analytic density
  $\int\rho\,r^2\,dr / \int r^2\,dr$ — the matched estimator for a counts-per-shell-volume
  sampled density near a steep truncation.
:::

## What is verified

Each row is a PASS/FAIL check in `scripts/validate_cluster_builders.py`. The **Tolerance**
is the bound the script enforces (`sys.exit(1)` on any failure); the **Measured** value is
retained from the recorded figure run; no new current-source execution is
implied by this page.

```{list-table}
:header-rows: 1

* - Property
  - Tolerance (as enforced)
  - Measured
  - Anchor
* - Virial $Q=T/|V|$, all 5 aliases ($N=5000$)
  - $|Q-0.5|<0.03$
  - $0.5000$ (Plummer/King/EFF/Michie/LIMEPY)
  - default $Q=0.5$ virial scale
* - Density recovery, sampled vs shell-weighted analytic
  - max rel $<0.20$ (bins $\ge50$ counts)
  - Plummer $0.072$ / King $0.135$ / EFF $0.128$
  - inherited inverse-CDF sampling
* - Tidal cut: ghost mass beyond $r_t$
  - exact $0$
  - $0.000$ (survivor fraction $0.71$)
  - `apply_tidal_truncation` (hard cut)
* - Rotation $L_z(\omega)$ slope vs $\Sigma m R^2$
  - rel err $<10^{-9}$
  - $4.7\times10^{-16}$ (machine-exact)
  - solid-body overlay
* - Differential $v_\phi(R)$ overlay
  - max rel $<0.15$
  - $0.057$
  - peaked rotation curve
* - Anisotropy $\beta(r)$ vs $r^2/(r^2+r_a^2)$
  - max dev $<0.05$
  - $0.0187$
  - Osipkov-Merritt matched DF
```

## Density recovery across the families

:::{figure} figures/cluster_density_recovery.png
:label: fig-cluster-density
:width: 80%

Sampled radial number-density (points) vs the shell-volume-weighted analytic density
(lines) for **Plummer, King, and EFF** built through the named aliases, with a residual
panel below. The convenience builder reuses each profile's own inverse-CDF sampler, so the
shape residuals are compared in populated bins. The displayed outer-bin
residual alone does not distinguish finite sampling noise from numerical bias.
:::

## Tidal truncation (Plummer-only)

:::{figure} figures/cluster_tidal_cut.png
:label: fig-cluster-tidal
:width: 70%

Radial mass profile of a Plummer build **with** vs **without** `tidal_radius`. The cut at
$r_t$ is exact (the straight-through `apply_tidal_truncation` forward pass is a hard
Heaviside), and the truncated stars become **zero-mass ghosts** beyond $r_t$ (fixed shape
$N$, so the build stays `jit`/`grad`-safe). Survivors keep velocities drawn for the
untruncated potential. The cut changes mass and potential; `revirialize=True`
sets a global energy target, while a
native-$r_t$ King/LIMEPY model provides a matched lowered continuum DF. Revirializing sets
the energy ratio; it does not make the stripped distribution stationary.
:::

## Rotation: solid-body and differential

:::{figure} figures/cluster_rotation_Lz.png
:label: fig-cluster-rotation
:width: 80%

**Left:** solid-body angular momentum $L_z(\omega)$ is exactly linear with slope
$\Sigma m R^2$ (measured to machine precision, rel err $4.7\times10^{-16}$, since the
overlay leaves positions unchanged). **Right:** a differential `RotationSpec` produces a
peaked $v_\phi(R)$ curve recovered to $<6\%$ in the inner region. Rotation injects net
$L_z$; it is a streaming transform, not a self-consistent rotating DF.
:::

## Osipkov-Merritt anisotropy

:::{figure} figures/cluster_anisotropy_beta.png
:label: fig-cluster-anisotropy
:width: 70%

Measured anisotropy $\beta(r)=1-\sigma_t^2/(2\sigma_r^2)$ for a Plummer build with
`anisotropy_radius` set, against the analytic Osipkov-Merritt profile
$\beta(r)=r^2/(r^2+r_a^2)$ (max deviation $0.019$). `anisotropy_radius` is threaded into the
matched DF and is valid **only for Plummer/EFF**; King is isotropic and Michie/LIMEPY carry
their anisotropy intrinsically (set $r_a$ on the profile).
:::

## Differentiability

Selected builder parameters have registered checks in the release gradient gate
([](./differentiability-audit.md)) with a **measured** AD-vs-FD tolerance: `build_cluster`
in `r_h` (machine-exact), `anisotropy_radius`, and rotation $\omega$; the per-family alias
paths (King `r_c`, EFF $\gamma$, Michie/LIMEPY `W0`); and `build_cluster_from_params` in
`r_h` through the `ClusterParams` θ-PyTree. The `tidal_radius` channel is a **straight-through
surrogate** (live but not finite-difference-consistent by design) and is covered by a
dedicated live-gradient teeth test rather than a false FD-consistent case.

## Single-population unit contract

`tests/unit/builders/test_cluster_units.py` uses deterministic engineering
cases with 16 entries and float64. It compares identical physical Plummer
inputs in STELLAR, PLANETARY and CGS for supplied masses, `n=` equal stars and
solar-mass IMF draws, with `Q=None` and `Q=0.5`. Supplied native masses remain
unchanged; returned dynamical fields convert back to STELLAR, while
`stellar_radii` remains in solar radii. CGS tidal, rotation and combined
modifier paths also retain the stellar-label conversion, including zero-mass
ghosts. The low-level `mass_scale_to_msun` keyword is checked separately.

JIT, forward and reverse first derivatives are checked against the same
Tout relation for stellar labels and Plummer mass-scaling homology for squared
speeds: multiplying every mass by $\lambda$ at fixed sampled positions gives
$v^2\propto\lambda$. Here $\lambda$ is a dimensionless common mass multiplier
and $v$ is speed. Round trips reuse the suite's exact-formula relative budget
$10^{-10}$ with zero absolute slack. This is a unit-conversion check, not an
independent calibration of the stellar fit, stochastic precision or wider
equilibrium-domain qualification. The binary unit checks are described below;
multicomponent alternate-unit paths remain separate review work. See [](../20-architecture/units-policy.md)
for the conversion equations and the distinction between native masses and
solar stellar labels.

The follow-up CGS comparison passes for the retained King, Michie, LIMEPY and
EFF configurations. The EFF velocity table begins at $r_{\min}=10^{-5}a$,
where $a$ is the EFF scale radius. The fractional inner cutoff therefore
stays the same when lengths change units. The retained same-key STELLAR/CGS
check gives a maximum component-relative velocity discrepancy of
$2.62\times10^{-13}$ against the existing $10^{-10}$ budget. Four table/speed
homology cases at $a=0.5$ and $a=2$ (isotropic and Osipkov-Merritt) also pass.
These checks do not eliminate the existing hard-truncation mismatch or
certify wider EFF domains.

## Binary-population unit contract

`tests/unit/builders/test_binary_units.py` checks 25 deterministic construction
cases in float64. Its unit comparisons encode the same physical Plummer model
in STELLAR, PLANETARY and CGS: half-mass radius 1 pc, 16 systems, key 0 and a
primary power-law IMF with exponent $\alpha=-2.3$ over 0.5–4 solar masses,
so its density per unit mass rises as $p(m)\propto m^{2.3}$. Independent
companions use binary fraction 0.5, minimum mass ratio 0.2, periods over
100–10000 days and thermal eccentricities. These small fixtures isolate
unit routing; they are not population-calibration experiments.

| Cases | Contract checked |
|---|---|
| 16 | Legacy/cataloged and compact/masked results, Q=None/0.5, PLANETARY/CGS physical round trips |
| 4 | CGS eager Stars and TotalMass budgets retain their solar-mass definitions |
| 2 | Moe companion prescriptions still receive solar masses; no new empirical-domain qualification |
| 1 | CGS Kepler periods, catalogue identities and periapsis contact margins use consistent native arrays |
| 2 | STELLAR/CGS JIT and forward/reverse IMF-slope derivatives preserve total mass, stellar labels and orbital axes |

The existing relative budget is $10^{-10}$ with zero absolute slack.
Converted physical positions, velocities and masses agree; `stellar_radii`
stays in solar radii. Zero-mass ghost slots retain masks and identities, but
their arbitrary finite Cartesian sanitization values are not physical
particles and are excluded from coordinate round trips. A separate
preservation comparison gives bit-identical leaves for ten default-STELLAR
cases. The retained CGS masked catalogue has maximum component-relative
round-trip differences of $1.97\times10^{-16}$ in mass,
$2.76\times10^{-15}$ in position and $9.27\times10^{-15}$ in velocity.

The affected selection passes 155 checks, including existing binary assembly,
companion, catalogue, budget, Kepler, single-population unit, API and provenance
checks. The existing 1000-system [](../40-howto/add-binary-population.md) recipe
also executes with its declining $m^{-2.3}$ primary IMF and published output.
A separate analytic regression checks that recipe's density ratio and mean mass
without drawing a catalogue. These checks establish construction
and unit consistency in the recorded cases. They do not calibrate empirical
binary laws, qualify wider profile/unit combinations, establish expectation
precision or certify N-body stationarity. For the conversion and frame
contracts, see [](../20-architecture/units-policy.md#binary-population-boundaries)
and [](../20-architecture/primordial-system-handoff.md).
