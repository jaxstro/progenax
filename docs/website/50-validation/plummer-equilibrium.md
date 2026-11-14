---
title: Plummer equilibrium
description: "Validation suite for the Plummer profile + matched velocity DF: scale radius, density/CDF sampling, velocity dispersion, the Beta(3/2,9/2) speed law, virial Q, isotropy, and gradient correctness."
---
# Plummer equilibrium validation

The suite checks the analytic profile and raw matched velocity sampler against
the [spatial profile](../10-theory/spatial-profiles/plummer.md) and
[velocity DF](../10-theory/velocity-dfs/plummer-dfs.md). Physics assertions live in
`tests/validation/test_plummer_physics.py`; fixed-draw sampler gradient cases
live in `tests/validation/grad_audit/registry.py` and run through
`tests/validation/test_grad_audit.py`. Figures come from `scripts/validate_plummer.py`.

## Isotropic velocity-scale checks

Independent analytic scale checks exercise the default isotropic sampler with
16 particles, seed 42, explicit float64, and half-mass radii 0.5, 1, 2.5 and 7.
They distinguish changing the scale radius at fixed positions from scaling both
positions and radius together, and check the square-root dependence on total mass
and the gravitational constant. Eager and JIT results and fixed-draw JVPs agree
within relative tolerance $10^{-6}$ and absolute tolerance $10^{-8}$.

The [registered relations](../15-model-reference/velocity_dfs.md#card-plummer-speed-pdf)
also pass exact symbolic normalization, moment and scale-derivative checks.
All five new cases passed; the combined selection including existing physics,
provenance and gradient checks passed 36 cases. A controlled extra radius factor
failed all four numerical cases. The reproducible command and scope are in the
run record (`docs/reviews/plummer-velocity-artifacts/run.json`, repository-only record).
These checks concern fixed random draws in the isotropic branch, not OM sampling
or gradients of population expectations. Statistical distribution checks remain
separate from the fixed-draw scale comparison.

## OM conditional speeds and central derivatives

The OM speed sampler is checked against an independently integrated conditional
CDF, expressed through incomplete beta functions. The
[registered PDF and CDF](../15-model-reference/velocity_dfs.md#card-plummer-om-speed-cdf)
pass a symbolic derivative check. The quantile test uses 1,089 deterministic
probabilities, including both tails, at radius ratios $r/a=0,1,10,100,1000$
and anisotropy ratios $r_a/a=0.75,1,2,10$, where $a$ is the Plummer scale radius
and $r_a$ the anisotropy radius. Its absolute CDF-error limit is $10^{-4}$.

Full-sampler position Jacobians are checked in forward and reverse AD at the
origin, where they vanish for a fixed random key. Nearby derivatives are compared
with finite differences in both isotropic and OM branches, in eager and JIT
execution. These are float64 checks in
`tests/unit/kinematics/test_plummer_om_speed.py`; they do not establish gradients
of population expectations. Inverse-CDF interpolation remains piecewise smooth.

## Evidence and scope

The table records historical figure outputs, not a fresh run of every test.
Figure sample counts and statistics differ from test exposures; shared numeric
thresholds do not make them identical checks. Test fixtures are in
`tests/conftest.py`. The local tracker’s September 2026 pilot report (`docs/reviews/plummer-pilot-results.md`, repository-only record)
records an 18-case exposure, with the selected tests and command in its
run record (`docs/reviews/plummer-pilot-artifacts/run.json`, repository-only record). Its
continuum Q was 0.502096 and mean squared speed ratio was 0.247234. Those
values do not replace measurements from the older figure exposures below.
Later repair and registry checks are separate exposures; their counts are not
interchangeable. The pilot
observations (`docs/reviews/plummer-pilot-artifacts/observations.json`, repository-only record)
record its diagnostics. Revision and environment provenance for older figure
measurements is not established by this pilot.

```{list-table}
:header-rows: 1

* - Property
  - Tolerance (as tested)
  - Measured
  - Anchor
* - Scale radius $a = r_h\sqrt{2^{2/3}-1}$
  - $10^{-6}$ rel
  - $0.7664$ ($=$ exact)
  - $M(<r_h)=M/2$
* - Half-mass fraction (frac $r<r_h$)
  - $0.03$ abs
  - $0.500$
  - definition of $r_h$
* - $M(<a)/M = 1/2^{3/2}$
  - $0.03$ abs
  - $0.3513$ (exp $0.3536$)
  - Plummer CDF at $r=a$
* - Cumulative-mass shape $M(<r)=M\,r^3/(r^2+a^2)^{3/2}$
  - $0.03$ abs per radius
  - max $|\Delta\mathrm{CDF}| = 3.6\times10^{-3}$
  - closed-form CDF
* - Velocity dispersion $\sigma_{1d}(r)=\sqrt{GM/(6\sqrt{r^2+a^2})}$
  - $10\%$ rel per bin
  - max rel $= 1.2\%$
  - isotropic Jeans / DF 2nd moment
* - Velocity isotropy $\langle v_x^2\rangle\!\approx\!\langle v_y^2\rangle\!\approx\!\langle v_z^2\rangle$
  - $5\%$ rel spread
  - $0.6\%$
  - isotropic DF
* - Virial ratio $Q=T/|V|$
  - $0.05$ abs at $N=5000$
  - $0.5026$
  - $2T+V=0$, $V=-3\pi GM^2/32a$
* - Bound fraction ($v\le v_{\rm esc}$)
  - **100%**
  - **100.00%**
  - $v_{\rm esc}^2=2GM/\sqrt{r^2+a^2}$
* - Speed law $\langle q^2\rangle$, $q=v/v_{\rm esc}$
  - $0.02$ abs
  - $0.2496$ (exp $0.25$)
  - $q^2\sim\mathrm{Beta}(3/2,9/2)$
* - Speed-law variance $\mathrm{Var}(q^2)$
  - $15\%$ rel
  - $0.0267$ (exp $0.0268$)
  - $\mathrm{Beta}(3/2,9/2)$
* - Position isotropy $\cos\theta\sim U(-1,1)$, $\phi\sim U(0,2\pi)$
  - reduced $\chi^2_\nu\approx1$
  - $\chi^2_\nu = 0.81,\ 0.75$
  - spherical symmetry
* - Analytic density-shape and central-dispersion derivatives ($r_h$, $M$)
  - AD $=$ central-FD
  - rel err $6\times10^{-11}$, $1.6\times10^{-6}$
  - analytic observables; not sampler-gradient evidence
```

The live sampler gradient nodes are
`test_gradient_audit[PlummerProfile.sample_positions]` and
`test_gradient_audit[PlummerVelocityDF.sample_velocities]` in
`tests/validation/test_grad_audit.py`. They vary the matched half-mass radius
with fixed random draws. They do not establish population-expectation gradients.

## Figures

Generated by `scripts/validate_plummer.py` (prints expected-vs-measured
PASS/FAIL for every panel; PNG for the web + PDF vector for the paper).

:::{figure} figures/plummer_density.png
:label: val-plummer-density
:width: 95%
:align: center

**Density profile and cumulative mass.** (a) `PlummerProfile.density` (dashed)
on the analytic $(1+r^2/a^2)^{-5/2}$ (solid) and the $N=5\times10^4$ sample
(points). (b) The empirical CDF tracks the closed form
$M(<r)=M\,r^3/(r^2+a^2)^{3/2}$ to **max $|\Delta\mathrm{CDF}|=3.6\times10^{-3}$**;
the half-mass point $(r_h,\,0.5)$ is marked. Histogram amplitudes are fitted
to the analytic shape, so panel (a) does not test absolute normalization. The
reported discrepancy uses upper ECDF steps, not the exact two-sided KS statistic. The scale radius is
$a=r_h\sqrt{2^{2/3}-1}=0.7664\,r_h$ to machine precision.
:::

:::{figure} figures/plummer_velocity_equilibrium.png
:label: val-plummer-velocity
:width: 95%
:align: center

**Velocity-space equilibrium.** (a) Sampled $\sigma_{1d}(r)$ on the analytic
$\sqrt{GM/(6\sqrt{r^2+a^2})}$ (per-bin rel $\le 1.2\%$). (b) Every velocity
satisfies $v\le v_{\rm esc}(r)$ — **100% bound**. (c) The unscaled IC sits at
**$Q=0.5026$** (tested to $\pm0.05$), using the analytic continuum potential
energy and no external velocity rescale. This checks the integrated kinetic
moment; it does not measure finite-N pair energy or dynamical stationarity.
:::

:::{figure} figures/plummer_beta_distribution.png
:label: val-plummer-beta
:width: 70%
:align: center

**The Plummer speed law.** The sampled $q^2=(v/v_{\rm esc})^2$ histogram
($N=4\times10^4$) overlays the exact $\mathrm{Beta}(3/2,9/2)$ density — the
distribution {cite:t}`Aarseth1974` rejection-sampling reproduces. Measured
$\langle q^2\rangle = 0.2496$ (exact $0.25$) and $\mathrm{Var}(q^2)=0.0267$
(exact $0.0268$).
:::

:::{figure} figures/plummer_isotropy.png
:label: val-plummer-isotropy
:width: 95%
:align: center

**Position isotropy.** (a) $\cos\theta$ and (b) $\phi$ of the sampled positions
are flat against their uniform expectations $U(-1,1)$ and $U(0,2\pi)$, with
reduced $\chi^2_\nu = 0.81$ and $0.75$ (consistent with unity). Velocity
isotropy $\langle v_i^2\rangle$ agrees to $0.6\%$ (printed by the script).
:::

## Differentiability

The figure below differentiates the dimensionless density shape and analytic
central velocity dispersion. It is distinct from the fixed-draw sampler audit.
Gradient agreement alone does not establish parameter identifiability.

:::{figure} figures/plummer_gradient_validation.png
:label: val-plummer-gradient
:width: 80%
:align: center

**Gradient validation (autodiff vs central finite difference).** (a)
$\partial\log_{10}[\rho(1\,\mathrm{pc})/\rho_0]/\partial r_h$ and (b)
$\partial\sigma_0/\partial M_{\rm tot}$: autodiff (line) and finite difference
(circles) agree to the historical figure errors **$6\times10^{-11}$** and
$1.6\times10^{-6}$. The first panel differentiates the density shape; normalized
mass density also includes the derivative of its $a^{-3}$ normalization.
:::

## How to run

```bash
# physics tests
pytest tests/validation/test_plummer_physics.py -q

# regenerate the five figures with PASS/FAIL tables
python scripts/validate_plummer.py
```

## What this suite does *not* test

- **Long-term evolution and assembled IC stationarity** — not established here.
  The builder recenters positions and velocities even with `Q=None`; these raw
  sampler checks do not qualify that transformed distribution.
- **Multi-mass equipartition** — single-mass Plummer is covered; the multi-mass
  and anisotropic lowered families have separate existing owners and checks
  following {cite:t}`Gieles2015` — see the
  [family page](../10-theory/spatial-profiles/lowered-model-family.md).
  They are not qualified by the single-mass isotropic checks on this page.
- **Tidal truncation** — covered separately in [](tidal-truncation.md).

## References

The Plummer derivations come from {cite:t}`Plummer1911`; the inverse-CDF radius
draw and the $\mathrm{Beta}(3/2,9/2)$ speed law follow {cite:t}`Aarseth1974`.
The full theoretical content is at
[](../10-theory/spatial-profiles/plummer.md).
