---
title: OED for anisotropy — where to spend telescope time on r_a (B14)
subtitle: The optimiser discovers that proper motions belong in the outskirts — without being told the physics
description: "A pedagogical pure pre-data Bayesian optimal-experimental-design demo. Given a fixed star budget, the additive design Fisher allocates stars across (projected radius x {RV, PM_R, PM_T}) to minimize the marginal variance of the Osipkov-Merritt anisotropy radius r_a. The optimizer DISCOVERS that proper-motion stars belong in the outskirts (where beta(r) is largest): the PM fraction rises from 0.73 in the core to 0.95 outside, the c-optimal design reaches equal precision on r_a with 3.66x fewer stars than uniform (sigma(r_a)/r_a 12.1% -> 6.3%), and a 64-draw calibration confirms the pre-data Fisher predicts the realized scatter. Shared Fisher/projection theory is factored into the OED formalism page."
---

# OED for anisotropy — where to spend telescope time on $r_a$ (B14)

This is the first worked example of [optimal experimental design](index.md): given a
fixed budget of stars, *where on the sky* and *in which channel* should you measure to
pin one number — the cluster's velocity **anisotropy**? The answer the optimiser
returns, having been told none of the underlying physics, is the lesson:
**proper motions belong in the outskirts.**

:::{note} **Shared theory lives once, on the formalism page**
The Fisher information / Cramér–Rao foundation {ref}`oed-fisher-gaussian`, the
[additive design-linear backbone](background.md#sec-oed-additive)
{ref}`oed-additive-fisher`, the [c/D/A criteria](background.md#sec-oed-criteria), the
[dimensionless $\ln\theta$ metric](background.md#sec-oed-dimensionless), and the
[Binney & Mamon (1982) projection geometry](background.md#sec-oed-projection) are all
built on [the OED formalism page](background.md). This page is the *application* — read
the formalism first if any of those terms are unfamiliar.
:::

:::{note} **Reading paths**
:class: dropdown

- **Already comfortable with Fisher OED?** Skip to [the physical question](#sec-b14-physics)
  for the anisotropy science, or jump straight to [the headline figure](#fig-oed-headline)
  and [Quantitative results](#sec-b14-results).
- **Want the machinery first?** Read [the OED formalism page](background.md), then come
  back here.
:::

(sec-b14-physics)=
## The physical question: how do you measure orbital anisotropy?

Our target is the **velocity anisotropy** of a star cluster — the degree to which
stellar orbits are radially or tangentially biased, captured by the Binney parameter
$\beta(r)$ {ref}`oed-beta`. We adopt the **Osipkov–Merritt** law {cite:p}`Merritt1985`,

```{math}
:label: b14-om
\beta_{\rm OM}(r) \;=\; \frac{r^2}{r^2 + r_a^2},
```

isotropic in the core and increasingly radial outward, with a single knob: the
**anisotropy radius** $r_a$, the radius where $\beta=\tfrac12$. Measuring the cluster's
orbital structure *is* measuring $r_a$.

Why does this matter physically? Anisotropy is a fossil of how a cluster formed and how
it is being torn apart: violent relaxation, tidal stripping, and radial-orbit
instability all leave their signature in $\beta(r)$. In dwarf-galaxy dynamics the same
$\beta$ is the dominant nuisance in the **mass–anisotropy degeneracy** that limits
dark-matter cusp-versus-core measurements. So $r_a$ is both a number worth measuring
and a number that is *notoriously hard* to measure — which is what makes it a perfect
OED target.

### Why proper motions, and why the outskirts

The three sky channels — line-of-sight RV, radial PM, tangential PM — are the
[B&M82 projection](background.md#sec-oed-projection) {ref}`oed-bm82` of one internal
$\sigma_r(r)$ and $\beta(r)$. The key fact, derived there: **in the isotropic limit the
three channels are identical — the anisotropy lives only in their *ratios*.** The
tangential proper motion is the cleanest probe, $\sigma_{{\rm pm},T}^2\propto(1-\beta)$,
so as orbits turn radial it **collapses** while the other channels stay up. And because
$\beta(r)$ grows outward, *the channels disagree most in the outskirts*. A single
$\sigma_{\rm los}(R)$ profile cannot break the tie — a cold outer LOS profile can mean
"low $\sigma_r$" **or** "high $\beta$." Proper motions break it. Keep that picture in
mind: it is the physics the optimiser is about to rediscover, having been told none of
it.

## Inputs and assumptions

The design optimises over a **fixed budget** of $N_{\rm total}$ stars allocated across
$K=12$ projected-radius bins $\times$ 3 kinematic channels (36 cells). The science
target is $r_a$; total mass $M$ and half-mass radius $r_h$ are nuisances carried with
fractional priors. Everything is computed *pre-data* — no mock catalogue enters the
optimisation loop (a 64-draw calibration **ensemble** afterwards is a gate, not part of
the design).

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
  - RV ($\sigma_{\rm los}$), PM radial ($\sigma_{{\rm pm},R}$), PM tangential ($\sigma_{{\rm pm},T}$) — the **design lever**; each channel's information content is set by the B&M82 kernels {ref}`oed-bm82`.
  - fixed (the design allocates stars *among* them)
* - $d$, $\sigma_{\rm RV}$, $\sigma_{\rm PM}$
  - Distance converts the astrometric error to velocity; per-star errors enter the per-datum variance $\delta\sigma^2=(\sigma^2+\epsilon^2)/(2n)$. At $d=4$ kpc the two channels are at **deliberate error parity** — neither trivially dominates.
  - fixed ($d=4$ kpc; $\sigma_{\rm RV}=1.0$ km/s; $\sigma_{\rm PM}=0.05$ mas/yr $\to 4.74\,\sigma_{\rm PM}\,d\approx0.95$ km/s)
* - completeness $c(R)$
  - **Fixed realism**, identical across channels: a smooth logistic faint-end roll-off ($\approx1$ in the core, $<1$ outside, turnover $\sim2\,r_h$). Illustrative, *not* a real survey curve, and *not* a design knob here — promoting depth to a knob is [the dynamical-mass example](dynamical-mass.md).
  - fixed (folds into the per-star blocks)
* - $K$ bins, budget, optimiser
  - $K=12$ log-spaced bin centres $R_k\in[0.3,3]\,r_h$; budget $N_{\rm total}$ (default 4000, swept for the frontier); multi-start Adam over the softmax simplex.
  - numerical choices
* - units, $G$
  - `STELLAR` ($\Msun$, pc, Myr); errors converted to pc/Myr explicitly ($1$ km/s $=1/0.977792$ pc/Myr).
  - known / fixed
```

The Fisher is built in the [dimensionless $\ln\theta$ metric](background.md#sec-oed-dimensionless)
{ref}`imp-oed-dimensionless`, so $\sigma(r_a)/r_a$ is a *fractional* precision and the
nuisance prior $\mathtt{PRIOR\_DIAG}=[0,\,1/0.3^2,\,1/0.3^2]$ is fractional too (none on
the target).

## The method, specialised: c-optimality on the additive Fisher

This example pours the [additive backbone](background.md#sec-oed-additive)
{ref}`oed-additive-fisher` straight into a **c-optimal** allocation. The per-star blocks
$M_{b,c}$ come from one `jacrev` of `project_dispersion` at the truth; the design enters
only through the softmax weights {ref}`oed-design`, multiplied by the fixed completeness
$c(R_b)$. The headline criterion minimises the $r_a$ variance with $M,r_h$ profiled out:

```{math}
:label: b14-c-criterion
c\!:\quad \min_{\mathbf z}\;(F^{-1})_{r_a r_a},
\qquad F(\mathbf z)=\sum_{b,c} n_{b,c}\,c(R_b)\,M_{b,c}.
```

We also run **D** ($\max\log\det F$) and **A** ($\min\mathrm{tr}\,F^{-1}$) as the
contrast — see [why c and D *should* disagree](background.md#sec-oed-criteria). Under the
$M\!\leftrightarrow\!r_a$ (through $GM$) and $r_h\!\leftrightarrow\!r_a$ degeneracies they
put stars at **different radii**, which is the criterion-disagreement lesson made visible
below.

## Validating a pre-data calculation: the calibration ensemble

A Fisher forecast is a *promise*: "if you observe this way, your error bar will be this
large." A promise made before any data is only trustworthy if you check it, because the
Cramér–Rao bound {ref}`oed-cramer-rao` is a **local, Gaussian** approximation — exact in
the high-information limit, optimistic when the likelihood is curved or the estimator is
biased.

So we close the loop once, as a *gate* (not inside the optimiser). We draw a 64-member
ensemble of mock catalogues from the actual Osipkov–Merritt Plummer sampler, project each
to the sky, bin it, and fit $\hat r_a$ by maximum *a posteriori* (with the same
fractional prior the design Fisher used). The realised scatter should match the Fisher
prediction,

```{math}
:label: b14-calibration
\mathrm{Var}(\hat r_a)\big/ r_a^2 \;\approx\; (F^{-1})_{r_a r_a},
```

both sides being *fractional* variances in the $\ln\theta$ metric. The tolerance is not a
free knob: the Monte-Carlo error on a variance estimated from $n_{\rm draws}$ draws is
$\approx\sqrt{2/n_{\rm draws}}$, so the gate is a principled $2\sigma$ band,
$2\sqrt{2/64}\approx0.35$. The realised $\sigma(r_a)/r_a=0.109$ sits just below the
Fisher's $0.121$ — the pre-data Fisher is **mildly conservative** (the binned-dispersion
estimator loses a little information relative to the idealised per-star Fisher), and the
two agree well inside the band. The promise holds.

## Figures

### The headline — PMs to the outskirts

:::{figure} figures/demo_oed_headline.png
:label: fig-oed-headline
:width: 90%

**The c-optimal design tracks the OM anisotropy $\beta(R)$.** Left axis: the three
predicted dispersion channels $\sigma_{\rm los}$, $\sigma_{{\rm pm},R}$,
$\sigma_{{\rm pm},T}$ (km/s) vs projected radius. Right axis: the c-optimal **PM
allocation fraction** $(n_{{\rm pm},R}+n_{{\rm pm},T})/\sum_c n$ per bin (purple)
plotted against $\beta_{\rm OM}(R)=R^2/(R^2+r_a^2)$ (dashed). The design — told nothing
about anisotropy — pushes the PM share from $0.73$ in the core to $0.95$ in the
outskirts, *following* $\beta(R)$: proper motions go where the anisotropy signal is.
:::

### c vs D vs A — why the criteria disagree

:::{figure} figures/demo_oed_cda.png
:label: fig-oed-cda
:width: 100%

**The c-, D-, and A-optimal allocations differ.** Each panel stacks the effective star
count $n_{\rm eff}$ per channel over radius for one criterion. The c-design (targeting
$r_a$ alone) loads the **PM channels in the outskirts** hardest; the D- and A-designs,
which must constrain $M$ and $r_h$ too, also load the **core**, where the density and
overall normalisation are best pinned. Different objectives genuinely want stars at
different radii — the disagreement is shown, not asserted.
:::

### The precision frontier — fewer stars at equal precision

:::{figure} figures/demo_oed_frontier.png
:label: fig-oed-frontier
:width: 80%

**The c-optimal design reaches equal precision with $\approx3.9\times$ fewer stars.**
Realized fractional precision $\sigma(r_a)/r_a=\sqrt{c}$ vs star budget $N_{\rm total}$,
recomputed (not extrapolated) for the uniform and c-optimal designs. The horizontal
arrow is the equal-precision star factor at the reference precision. The curves are
**mildly non-$1/N$** because the fixed nuisance prior dilutes as $N$ grows (the c-opt
slope departs slightly from $-1/2$) — the honest, real frontier rather than an idealised
$c\propto1/N$ line.
:::

### Calibration — the pre-data Fisher is trustworthy

:::{figure} figures/demo_oed_calibration.png
:label: fig-oed-calibration
:width: 70%

**A 64-draw mock ensemble confirms the design Fisher predicts the realized scatter.** The
calibration is run at the **uniform** design (so the Fisher value here, $0.121$, is the
*uniform* precision — not the c-optimal $0.063$); this suffices because the per-star
Fisher blocks $M_{b,c}$ are **design-independent**, so validating the Fisher machinery at
one design transitively validates it at the c-optimal design built from the same blocks.
The realized fractional precision $\sigma(r_a)/r_a$ (orange, with its Monte-Carlo error
band from 64 draws) sits at $0.109$, just below the Fisher-predicted $0.121$ (blue): the
pre-data Fisher is **mildly conservative**, and the two agree within the MC error. This is
the gate that makes the pre-data design trustworthy.
:::

### Optimiser convergence

:::{figure} figures/demo_oed_optpath.png
:label: fig-oed-optpath
:width: 70%

**All three alphabet-optimality objectives converge cleanly.** Each curve is the
best-start suboptimality gap $(c_t-c_\infty)/(c_0-c_\infty)$ vs Adam iteration, normalised
to the initial gap (the c, D, A objectives live on different scales and signs, so the
normalised gap is the like-for-like view). Every objective descends monotonically to its
converged design.
:::

(sec-b14-results)=
## Quantitative results

All numbers are from the gated CLI's run-record (`--full`, 64-draw calibration,
$N_{\rm total}=4000$):

```{list-table} OED anisotropy results
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

The equal-precision factor is **exact at fixed $N$** (the fixed nuisance prior cancels in
the $c_{\rm uniform}/c_{\rm designed}$ ratio, because $c\propto1/N$ when the prior is held
fixed); the "$\approx3.9\times$ fewer stars" gloss on the frontier is the same physics
read off the swept budget curve, where the prior no longer cancels and the slope departs
mildly from $1/N$.

## What the optimum means: science implications

The $3.66\times$ is not a free lunch — it is the optimiser **discovering the physics of
where information lives**, and it generalises well beyond this mock.

**1. Information is localised — in space *and* in channel.** Anisotropy $\beta(r)$ grows
outward, and the tangential-PM sensitivity to $r_a$ grows with it. So the *value* of a
star for measuring $r_a$ depends sharply on where it sits and how you measure it. OED
makes that quantitative instead of intuitive: the headline figure shows the optimiser
routing PM stars to exactly the radii where $\beta(R)$ is turning over.

**2. Proper motions break the mass–anisotropy degeneracy where it bites.** A
line-of-sight dispersion profile alone is degenerate — recall
{ref}`oed-projection-geometry`: a cold outer $\sigma_{\rm los}$ can mean low $\sigma_r$
*or* high $\beta$. The two PM channels carry different $\beta$-weightings {ref}`oed-bm82`,
so they lift the degeneracy — and the OED tells you *where the lift is largest*: the
outskirts. **The result — RV in the core, PM in the outskirts — is the observing strategy
this degeneracy demands, derived from first principles.** That is directly actionable for
real Gaia-PM + spectroscopic campaigns on globular clusters and dwarf spheroidals.

**3. Design for your target, not for "everything."** The c-vs-D-vs-A divergence is the
deepest lesson. If your science is one number — anisotropy as a probe of a dark-matter
cusp/core, an IMBH's kinematic signature, a cluster's tidal state — a design that
minimises the joint ellipsoid (D) *wastes* stars tightening nuisances. c-optimality can
buy a multiplicative factor in the precision you actually care about.

**4. The honest caveat is itself a research direction.** The optimum is optimal *for the
assumed OM-Plummer model*. That model-dependence is the central limitation of all OED —
and it points straight at the most valuable extension: **robust** and
**model-discriminating** design (see [the section's capability map](index.md)), which
`progenax` is unusually well-placed to do because it ships *more than one* differentiable
forward model for the same observable.

## Current scope and planned extensions

```{warning}
This is a **pre-data, single-shot OED on a clean self-consistent mock**. Its boundaries,
stated honestly:

- **The RV channel is under-utilised *at the chosen error parity*.** At $d=4$ kpc the
  per-star RV and PM errors are deliberately matched, but PM delivers **two** components
  ($R$ and $T$) at that same per-star error, so the optimiser prefers it almost
  everywhere — the RV ($v_{\rm los}$) channel carries little of the budget. This is a
  **real, documented consequence of the error parity**, not a bug: were the RV errors
  smaller, RV would carry more of the load. The "PMs to the outskirts" result is the
  *radial* trend within the PM-favoured regime, which is robust.
- **Single line-of-sight projection.** LOS $=\hat z$; no flattening, no rotation, no
  inclination — those are versatility-roadmap items.
- **Fixed completeness — depth is not yet a design knob here.** The faint-end roll-off is
  applied identically to all channels as fixed realism. Making the magnitude limit /
  depth a *design variable* (and headlining $M_{\rm dyn}$) is
  [the dynamical-mass example](dynamical-mass.md).
- **OM-Plummer equilibrium mock only.** Truth and forward model share the same generative
  family; no model misspecification, no real catalogue, no cross-channel systematics.
- **The Fisher is a local, Gaussian (Cramér–Rao) approximation.** It is exact only in the
  high-information limit. **The calibration ensemble is precisely the check** that it
  predicts the realized scatter — and it does, to within the Monte-Carlo error
  ({numref}`fig-oed-calibration`).
- **Static single-shot — no sequential OED.** The design is computed once; there is no
  online/adaptive re-design as data arrive.
```

:::{note} What we just learned
The precision you will achieve is a *known function of the observing strategy* — the
Fisher information {ref}`oed-fisher-gaussian`, bounded below by Cramér–Rao
{ref}`oed-cramer-rao` — so you can **design the observation before taking it**. For
cluster anisotropy, the information lives in the *ratios* between the RV and the two PM
channels {ref}`oed-bm82`, and because $\beta(r)$ grows outward, a c-optimal design **puts
proper motions in the outskirts** and reaches the same precision on $r_a$ with
$3.66\times$ fewer stars. The whole optimisation is cheap because the Fisher is
[additive and design-linear](background.md#sec-oed-additive) {ref}`oed-additive-fisher` —
one `jacrev`, then linear algebra. The same recipe with a *different* design knob — survey
depth — weighs the cluster in [the dynamical-mass example](dynamical-mass.md).
:::

## How to run

```bash
# quick (12-draw calibration, ~1 min)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed.py

# publication-grade (64-draw calibration, ~4 min) — regenerates the figures here
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed.py --full
```

The CLI is gated (exit 0 only if the headline factor, PM-outskirts trend, and calibration
all pass) and writes a JSON run-record alongside the five figures.

## References

The shared Fisher / Cramér–Rao / projection theory and its references are on
[the OED formalism page](background.md). The Osipkov–Merritt anisotropy law is
{cite:t}`Merritt1985`; the line-of-sight projection into $\sigma_{\rm los}$,
$\sigma_{{\rm pm},R}$, $\sigma_{{\rm pm},T}$ is Binney & Mamon (1982, MNRAS 200, 361). The
differentiable `project_dispersion` forward model is documented on the
[velocity-DF kinematics](../../10-theory/velocity-dfs/rotation-anisotropy.md) pages; the
anisotropy *recovery* counterpart is the [anisotropy demo (B6)](../anisotropy.md), and
[B8](../rotation.md) introduces the same sky-projection helper.
