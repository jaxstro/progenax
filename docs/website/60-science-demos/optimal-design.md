---
title: Optimal experimental design — where to spend telescope time on r_a (B14)
description: "A pure pre-data Bayesian optimal-experimental-design demo. Given a fixed star budget, the additive design Fisher allocates stars across (projected radius x {RV, PM_R, PM_T}) to minimize the marginal variance of the Osipkov-Merritt anisotropy radius r_a. The optimizer DISCOVERS that proper-motion stars belong in the outskirts (where beta(r) is largest): the PM fraction rises from 0.73 in the core to 0.95 outside, the c-optimal design reaches equal precision on r_a with 3.66x fewer stars than uniform (sigma(r_a)/r_a 12.1% -> 6.3%), and a 64-draw calibration confirms the pre-data Fisher predicts the realized scatter (mildly conservative)."
---

# Optimal experimental design — where to spend telescope time on $r_a$ (B14)

The other Batch-B/C demos answer *can you recover a parameter from data you
already have?* This one asks the question that comes **before** any data exist:
given a fixed star budget, **where on the sky and in which kinematic channel
should you spend it** to learn the most about one target parameter? That is
**optimal experimental design** (OED) — designing the *observation*, not just
analysing it.

The target here is the **Osipkov–Merritt anisotropy radius** $r_a$, which sets
where stellar orbits turn radial via $\beta(r)=r^2/(r^2+r_a^2)$ (the
[anisotropy demo, B6](anisotropy.md) measures it; here we design *how* to measure
it). The lever is the choice of **kinematic channel** — line-of-sight radial
velocity (RV) versus the two proper-motion (PM) components — across projected
radius. The headline result is **emergent and interpretable**: the optimizer is
told nothing about anisotropy physics, yet it *discovers* that proper-motion
stars belong in the **outskirts** and RV stars in the centre, because the
Binney & Mamon (1982) PM kernels carry the anisotropy $\beta(r)$, which grows
outward.

## Inputs and assumptions

The design optimises over a **fixed budget** of $N_{\rm total}$ stars allocated
across $K=12$ projected-radius bins $\times$ 3 kinematic channels (36 cells).
The science target is $r_a$; total mass $M$ and half-mass radius $r_h$ are
nuisances carried with fractional priors. Everything is computed *pre-data* — no
mock catalogue enters the optimisation loop (a 64-draw calibration **ensemble**
afterwards is a gate, not part of the design).

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b14-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $r_a$
  - Osipkov–Merritt anisotropy radius — **the science target**; $\beta_{\rm OM}(r)=r^2/(r^2+r_a^2)$. c-optimality minimises *its* marginal variance.
  - **target** ($6.0$ pc $=2\,r_h$; well inside OM validity $r_a\gg0.75\,a$)
* - $M$, $r_h$
  - Total mass and Plummer half-mass radius — **nuisances**, profiled out; each carries a 30% fractional ($d\ln\theta$) prior (e.g. $M$ from integrated light, $r_h$ from photometry).
  - nuisances ($M=10^5\,\Msun$, $r_h=3.0$ pc) with $\sigma_{\rm prior}/\theta=0.3$
* - 3 kinematic channels
  - RV ($\sigma_{\rm los}$), PM radial ($\sigma_{{\rm pm},R}$), PM tangential ($\sigma_{{\rm pm},T}$) — the **design lever**; each channel's information content is set by the B&M82 projection kernels (below).
  - fixed (the design allocates stars *among* them)
* - $d$, $\sigma_{\rm RV}$, $\sigma_{\rm PM}$
  - Distance converts the astrometric error to velocity; per-star errors enter the per-datum variance $\delta\sigma^2=(\sigma^2+\epsilon^2)/(2n)$. At $d=4$ kpc the two channels are at **deliberate error parity** — neither trivially dominates.
  - fixed ($d=4$ kpc; $\sigma_{\rm RV}=1.0$ km/s; $\sigma_{\rm PM}=0.05$ mas/yr $\to 4.74\,\sigma_{\rm PM}\,d\approx0.95$ km/s)
* - completeness $c(R)$
  - **Fixed realism**, identical across channels: a smooth logistic faint-end roll-off ($\approx1$ in the core, $<1$ outside, turnover $\sim2\,r_h$). Illustrative, *not* a real survey curve, and *not* a design knob in Stage 1.
  - fixed (folds into the per-star blocks)
* - $K$ bins, budget, optimiser
  - $K=12$ log-spaced bin centres $R_k\in[0.3,3]\,r_h$; budget $N_{\rm total}$ (default 4000, swept for the frontier); multi-start Adam over the softmax simplex.
  - numerical choices
* - units, $G$
  - `STELLAR` ($\Msun$, pc, Myr); errors converted to pc/Myr explicitly ($1$ km/s $=1/0.977792$ pc/Myr).
  - known / fixed
```

```{important}
:label: imp-b14-dimensionless
**The Fisher is built in the dimensionless $\ln\theta$ metric (ADR 0011).** The
parameters span five orders of magnitude ($M\approx10^5$ vs $r_h\approx3$), so the
raw-unit Fisher is wildly ill-conditioned ($\mathrm{cond}\approx1.7\times10^9$)
and the A/D criteria are not scale-invariant. Differentiating with respect to
$\ln\theta$ ($J\to J\cdot\mathrm{diag}(\theta_{\rm fid})$) makes $F$
**dimensionless** ($\mathrm{cond}\approx45$), every covariance entry a
**fractional variance**, and the c-headline a **fractional precision**
$\sigma(r_a)/r_a$. The nuisance prior is fractional too,
$\mathtt{PRIOR\_DIAG}=[0,\,1/0.3^2,\,1/0.3^2]$ (none on the target).
```

## The method: an additive, design-linear Fisher

The load-bearing idea (ADR 0004) is that the **design Fisher is additive and
linear in the design**. For a design that places $n_{b,c}$ stars in radial bin
$b$ and channel $c$,

```{math}
:label: b14-additive-fisher
F(\text{design}) \;=\; \sum_{b,\,c} n_{{\rm eff},\,b,c}\; M_{b,c},
\qquad
n_{{\rm eff},\,b,c} = n_{b,c}\cdot c(R_b),
```

where each **per-star block** is the design-*independent* $3\times3$ rank-1 matrix

```{math}
:label: b14-perstar-block
M_{b,c} \;=\; \frac{2\,J_{b,c}\,J_{b,c}^{\mathsf T}}{\sigma^2_{b,c} + \epsilon_c^2},
\qquad
J_{b,c} = \frac{\partial\,\sigma^{\rm pred}_{b,c}}{\partial\ln\theta},
```

from the Gaussian information of a dispersion measured from $n$ stars
($\delta\sigma^2=(\sigma^2+\epsilon^2)/(2n)$).

**Why this dodges re-differentiating the forward model.** The Jacobian $J_{b,c}$
is computed **once**, by a single reverse-mode `jacrev` through the packaged
`project_dispersion`, at the truth $\theta=(r_a,M,r_h)$. The design then enters
{eq}`b14-additive-fisher` *only* as the linear weights $n_{b,c}$ — so the entire
optimisation is $F=\sum n\,(c\,M)$, invert a $3\times3$, read one element. The
B&M82 projection (an ODE/quadrature stack) is never differentiated inside the
optimiser loop; the gradients $\partial(\text{criterion})/\partial(\text{design})$
are pure linear algebra over the precomputed blocks. We use `jacrev` (reverse
mode) because it is the supported, tested AD path for *all* profiles and stays
correct if a King/Michie mock is ever swapped in (those equilibrium-solver
profiles hit a `custom_vjp` ODE with no forward-mode rule). On the Plummer path
used here there is no ODE, so forward-mode would also work — `jacrev` is the
robust choice, not a forced one.

**The B&M82 projection — where the channels get their information.** The three
observed dispersions are the Binney & Mamon (1982, MNRAS 200, 361) line-of-sight
projection of the 3-D anisotropic Jeans model, with OM anisotropy
$\beta(r)=r^2/(r^2+r_a^2)$ {cite:p}`Merritt1985`:

```{math}
:label: b14-bm82
\begin{aligned}
\Sigma\,\sigma_{\rm los}^2(R)   &= 2\!\int_R^\infty \Big(1 - \beta\,\tfrac{R^2}{r^2}\Big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}},\\
\Sigma\,\sigma_{{\rm pm},R}^2(R)&= 2\!\int_R^\infty \Big(1 - \beta + \beta\,\tfrac{R^2}{r^2}\Big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}},\\
\Sigma\,\sigma_{{\rm pm},T}^2(R)&= 2\!\int_R^\infty \big(1 - \beta\big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}}.
\end{aligned}
```

In the isotropic limit ($\beta=0$) all three kernels collapse to 1 and
$\sigma_{\rm los}=\sigma_{{\rm pm},R}=\sigma_{{\rm pm},T}$ — **anisotropy lives
only in the ratios**, and the tangential PM kernel $(1-\beta)$ is the most direct
$\beta(r)$ probe. Since $\beta(r)$ grows outward, the PM channels carry *more*
information about $r_a$ at large $R$ — the physics the optimiser is about to
rediscover on its own.

**Three optimality criteria** ride the same $F=\sum n\,(c\,M)$:

```{math}
:label: b14-criteria
\underbrace{c\!:\;\min\,(F^{-1})_{r_a r_a}}_{\text{target the }r_a\text{ variance}}
\qquad
\underbrace{D\!:\;\max\,\log\det F}_{\text{the whole }\theta\text{ democratically}}
\qquad
\underbrace{A\!:\;\min\,\mathrm{tr}\,F^{-1}}_{\text{total fractional variance}}.
```

c-optimality targets $r_a$ *after profiling out* $M,r_h$ (the $(r_a,r_a)$ element
of the full inverse); D and A weight all three parameters together. Under the
$M\!\leftrightarrow\!r_a$ (through $GM$) and $r_h\!\leftrightarrow\!r_a$
degeneracies, c allocates to **different radii** than D/A — which is the
criterion-disagreement lesson made visible below.

## Figures

### The headline — PMs to the outskirts

:::{figure} figures/demo_oed_headline.png
:label: fig-oed-headline
:width: 90%

**The c-optimal design tracks the OM anisotropy $\beta(R)$.** Left axis: the
three predicted dispersion channels $\sigma_{\rm los}$, $\sigma_{{\rm pm},R}$,
$\sigma_{{\rm pm},T}$ (km/s) vs projected radius. Right axis: the c-optimal
**PM allocation fraction** $(n_{{\rm pm},R}+n_{{\rm pm},T})/\sum_c n$ per bin
(purple) plotted against $\beta_{\rm OM}(R)=R^2/(R^2+r_a^2)$ (dashed). The design
— told nothing about anisotropy — pushes the PM share from $0.73$ in the core to
$0.95$ in the outskirts, *following* $\beta(R)$: proper motions go where the
anisotropy signal is.
:::

### c vs D vs A — why the criteria disagree

:::{figure} figures/demo_oed_cda.png
:label: fig-oed-cda
:width: 100%

**The c-, D-, and A-optimal allocations differ.** Each panel stacks the effective
star count $n_{\rm eff}$ per channel over radius for one criterion. The c-design
(targeting $r_a$ alone) loads the **PM channels in the outskirts** hardest; the
D- and A-designs, which must constrain $M$ and $r_h$ too, also load the **core**,
where the density and overall normalisation are best pinned. Different objectives
genuinely want stars at different radii — the disagreement is shown, not asserted.
:::

### The precision frontier — fewer stars at equal precision

:::{figure} figures/demo_oed_frontier.png
:label: fig-oed-frontier
:width: 80%

**The c-optimal design reaches equal precision with $\approx3.9\times$ fewer
stars.** Realized fractional precision $\sigma(r_a)/r_a=\sqrt{c}$ vs star budget
$N_{\rm total}$, recomputed (not extrapolated) for the uniform and c-optimal
designs. The horizontal arrow is the equal-precision star factor at the reference
precision. The curves are **mildly non-$1/N$** because the fixed nuisance prior
dilutes as $N$ grows (the c-opt slope departs slightly from $-1/2$) — the
honest, real frontier rather than an idealised $c\propto1/N$ line.
:::

### Calibration — the pre-data Fisher is trustworthy

:::{figure} figures/demo_oed_calibration.png
:label: fig-oed-calibration
:width: 70%

**A 64-draw mock ensemble confirms the design Fisher predicts the realized
scatter.** The calibration is run at the **uniform** design (so the Fisher value
here, $0.121$, is the *uniform* precision — not the c-optimal $0.063$); this
suffices because the per-star Fisher blocks $M_{b,c}$ are **design-independent**,
so validating the Fisher machinery at one design transitively validates it at the
c-optimal design built from the same blocks. The realized fractional precision
$\sigma(r_a)/r_a$ (orange, with its Monte-Carlo error band from 64 draws) sits at
$0.109$, just below the Fisher-predicted $0.121$ (blue): the pre-data Fisher is
**mildly conservative** (the binned-dispersion estimator loses a little
information relative to the idealised per-star Fisher), and the two agree within
the MC error. This is the gate that makes the pre-data design trustworthy.
:::

### Optimiser convergence

:::{figure} figures/demo_oed_optpath.png
:label: fig-oed-optpath
:width: 70%

**All three alphabet-optimality objectives converge cleanly.** Each curve is the
best-start suboptimality gap $(c_t-c_\infty)/(c_0-c_\infty)$ vs Adam iteration,
normalised to the initial gap (the c, D, A objectives live on different scales and
signs, so the normalised gap is the like-for-like view). Every objective descends
monotonically to its converged design.
:::

## Quantitative results

All numbers are from the gated CLI's run-record (`--full`, 64-draw calibration,
$N_{\rm total}=4000$):

```{list-table} OED Stage-1 results
:header-rows: 1
:label: tbl-b14-results

* - Quantity
  - Result
* - Equal-precision star factor (c-design vs uniform, **fixed $N$**)
  - $\mathbf{3.66\times}$ fewer stars ($\approx3.9\times$ on the swept frontier)
* - $\sigma(r_a)/r_a$, uniform $\to$ c-optimal (at fixed $N$)
  - $\mathbf{12.1\% \to 6.3\%}$
* - PM allocation fraction, inner-half $\to$ outer-half (c-design)
  - $\mathbf{0.73 \to 0.95}$ (PMs favoured outward)
* - Calibration: realized $\sigma(r_a)/r_a$ vs Fisher
  - $0.109$ (realized) vs $0.121$ (Fisher) — $19\%$ variance-space dev, gate $35\%$
* - c-criterion $(F^{-1})_{r_a r_a}$: uniform / c-opt / D-opt / A-opt
  - $1.46 / 0.40 / 0.59 / 0.59 \times10^{-2}$ (c-design lowest on its own objective)
```

The equal-precision factor is **exact at fixed $N$** (the fixed nuisance prior
cancels in the $c_{\rm uniform}/c_{\rm designed}$ ratio); the "$\approx3.9\times$
fewer stars" gloss on the frontier is the same physics read off the swept budget
curve, where the prior no longer cancels and the slope departs mildly from
$1/N$.

## Current scope and planned extensions

```{warning}
Stage 1 is a **pre-data, single-shot OED on a clean self-consistent mock**. Its
boundaries, stated honestly:

- **The RV channel is under-utilised *at the chosen error parity*.** At
  $d=4$ kpc the per-star RV and PM errors are deliberately matched, but PM
  delivers **two** components ($R$ and $T$) at that same per-star error, so the
  optimiser prefers it almost everywhere — the RV ($v_{\rm los}$) channel carries
  little of the budget. This is a **real, documented consequence of the error
  parity**, not a bug: were the RV errors smaller (better spectroscopy, or PM
  errors that grow with distance), RV would carry more of the load. The
  "PMs to the outskirts" result is the *radial* trend within the PM-favoured
  regime, which is robust.
- **Single line-of-sight projection.** LOS $=\hat z$; no flattening, no rotation,
  no inclination — those are versatility-roadmap items.
- **Fixed completeness — depth is not yet a design knob.** The faint-end
  roll-off is applied identically to all channels as fixed realism. Making the
  magnitude limit / depth a *design variable* (and adding $M_{\rm dyn}$) is
  **Stage 2**.
- **OM-Plummer equilibrium mock only.** Truth and forward model share the same
  generative family (isotropic Plummer density under exact Osipkov–Merritt
  anisotropy); no model misspecification, no real catalogue, no cross-channel
  systematics.
- **The Fisher is a local, Gaussian (Cramér–Rao) approximation.** It is exact
  only in the high-information / linearised limit. **The calibration ensemble is
  precisely the check** that it predicts the realized scatter — and it does, to
  within the Monte-Carlo error (mildly conservative, {numref}`fig-oed-calibration`).
- **The $c\propto1/N$ star-count gloss is approximate.** With a fixed nuisance
  prior, $c\cdot N$ drifts ($\sim18\%$ over $N=10^3$–$8\times10^3$); the
  fixed-$N$ equal-precision factor is the exact headline, the frontier curve is
  the honest extrapolation.
- **Static single-shot — no sequential OED.** The design is computed once; there
  is no online/adaptive re-design as data arrive.

**Deferred:** **Stage 2** (magnitude-limit / $M_{\rm dyn}$ — depth as a design
knob) and **Stage 3** (multi-epoch astrometry / explicit cost budget). On the
versatility roadmap: rotation, flattening, and the tracer $\neq$ mass case.
```

## How to run

```bash
# quick (12-draw calibration, ~1 min)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed.py

# publication-grade (64-draw calibration, ~4 min) — regenerates the figures here
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed.py --full
```

The CLI is gated (exit 0 only if the headline factor, PM-outskirts trend, and
calibration all pass) and writes a JSON run-record alongside the five figures.

## References

The Osipkov–Merritt anisotropy law $\beta(r)=r^2/(r^2+r_a^2)$ is
{cite:t}`Merritt1985`; the line-of-sight projection of an anisotropic spherical
model into $\sigma_{\rm los}$, $\sigma_{{\rm pm},R}$, $\sigma_{{\rm pm},T}$ is
Binney & Mamon (1982, MNRAS 200, 361). The differentiable `project_dispersion`
forward model is documented on the
[velocity-DF kinematics](../10-theory/velocity-dfs/rotation-anisotropy.md)
pages; the anisotropy *recovery* counterpart is the
[anisotropy demo (B6)](anisotropy.md), and [B8](rotation.md) introduces the same
sky-projection helper.
```