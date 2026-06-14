---
title: Cluster-builder validation
description: "Validation of the build_cluster convenience IC-builder layer: virial equilibrium across all five profiles, density recovery, tidal truncation, solid-body / differential rotation, and Osipkov-Merritt anisotropy — all through the one-call build_cluster / alias API."
---
# Cluster-builder validation

The `build_cluster` convenience layer is a thin, differentiable wrapper over the
composable `build_spatial_ic` core: it resolves the mass spec, auto-pairs the
scale-matched equilibrium velocity DF (`matched_velocity_df`), and applies optional
tidal / rotation modifiers. Because the base case is proven **bit-identical** to the
manual composition (exact `==` in `tests/unit/builders/test_cluster_builders.py`), every
equilibrium guarantee is *inherited* from the already-validated profiles and DFs — this
page verifies only the **new surface**: that each alias builds a near-virial IC, that the
modifiers are physically correct, and that the differentiable knobs flow.

Figures: `scripts/validate_cluster_builders.py` (prints expected-vs-measured for every
check and exits non-zero on any failure). End-to-end equilibrium across all five profiles:
`tests/integration/test_cluster_builders_integration.py`. Gradient correctness: the
differentiability audit ([](./differentiability-audit.md)) registers eight measured
AD-vs-FD cases for the builders plus a tidal straight-through teeth test.

:::{admonition} Conventions for this suite
:class: note
- **Q is the virial ratio** $Q=T/|V|$; $Q=0.5$ is equilibrium. `build_cluster` virial-scales
  to $Q=0.5$ by **default**, so all five aliases land at $0.5000$. Pass `Q=None` for the
  faithful **unscaled** equilibrium of the true-DF samplers (King/EFF/Michie/LIMEPY sit near
  $0.5$ unscaled by construction).
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
regenerated on every run.

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
*shape* is recovered to a few percent per well-populated bin — the residual at the steep
King edge is honest Poisson noise in the outermost bin, not a sampler bias.
:::

## Tidal truncation (Plummer-only)

:::{figure} figures/cluster_tidal_cut.png
:label: fig-cluster-tidal
:width: 70%

Radial mass profile of a Plummer build **with** vs **without** `tidal_radius`. The cut at
$r_t$ is exact (the straight-through `apply_tidal_truncation` forward pass is a hard
Heaviside), and the truncated stars become **zero-mass ghosts** beyond $r_t$ (fixed shape
$N$, so the build stays `jit`/`grad`-safe). Survivors keep velocities drawn for the
untruncated potential — the set is super-virial (audit S4); use `revirialize=True`, or a
native-$r_t$ King/LIMEPY model, for a stationary truncated equilibrium.
:::

## Rotation: solid-body and differential

:::{figure} figures/cluster_rotation_Lz.png
:label: fig-cluster-rotation
:width: 80%

**Left:** solid-body angular momentum $L_z(\omega)$ is exactly linear with slope
$\Sigma m R^2$ (measured to machine precision, rel err $4.7\times10^{-16}$, since the
overlay leaves positions unchanged). **Right:** a differential `RotationSpec` produces a
peaked $v_\phi(R)$ curve recovered to $<6\%$ in the inner region. Rotation injects net
$L_z$ (audit S3, deliberately non-stationary).
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

Every differentiable knob of the builder is registered in the release gradient-gate
([](./differentiability-audit.md)) with a **measured** AD-vs-FD tolerance: `build_cluster`
in `r_h` (machine-exact), `anisotropy_radius`, and rotation $\omega$; the per-family alias
paths (King `r_c`, EFF $\gamma$, Michie/LIMEPY `W0`); and `build_cluster_from_params` in
`r_h` through the `ClusterParams` θ-PyTree. The `tidal_radius` channel is a **straight-through
surrogate** (live but not finite-difference-consistent by design) and is covered by a
dedicated live-gradient teeth test rather than a false FD-consistent case.
