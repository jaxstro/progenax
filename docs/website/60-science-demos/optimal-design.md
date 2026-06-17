---
title: Optimal experimental design — where to spend telescope time on r_a (B14)
subtitle: Designing the observation before you take it — Fisher information, c-optimality, and why proper motions belong in the outskirts
description: "A pedagogical, pure pre-data Bayesian optimal-experimental-design demo. Given a fixed star budget, the additive design Fisher allocates stars across (projected radius x {RV, PM_R, PM_T}) to minimize the marginal variance of the Osipkov-Merritt anisotropy radius r_a. The optimizer DISCOVERS that proper-motion stars belong in the outskirts (where beta(r) is largest): the PM fraction rises from 0.73 in the core to 0.95 outside, the c-optimal design reaches equal precision on r_a with 3.66x fewer stars than uniform (sigma(r_a)/r_a 12.1% -> 6.3%), and a 64-draw calibration confirms the pre-data Fisher predicts the realized scatter. Includes a first-principles treatment of Fisher information / Cramer-Rao and a map of the OED progenax can do."
---

# Optimal experimental design — where to spend telescope time on $r_a$ (B14)

Every other demo on this site answers the same kind of question: *given data, can
you recover a parameter?* This one asks the question that comes **before any data
exist**. You have been awarded a fixed amount of telescope time — call it a budget
of $N$ stars you can afford to measure. Where on the sky should you point, and
*which* measurement (a radial velocity? a proper motion?) should you make on each
star, so that you learn the most about the one number you actually care about?

That is **optimal experimental design** (OED): you optimise the *observation
itself*, not just the analysis you do afterwards. It is rare in astronomy — almost
no simulator can compute $\partial(\text{information})/\partial(\text{observing
strategy})$ — and it is exactly what a *differentiable* forward model makes
possible. This page builds the idea from first principles, runs it on a star
cluster, and ends with a map of the OED problems the rest of `progenax` can pose.

:::{note} **Reading paths**
:class: dropdown

- **New to experimental design?** Read straight through — every equation is earned
  with a physical picture first.
- **Already comfortable with Fisher information and the Cramér–Rao bound?** Skip to
  [The physical question](#sec-b14-physics) for the anisotropy science, or to
  [The method](#sec-b14-method) for the additive-Fisher trick.
- **Just want the result?** Jump to [the headline figure](#fig-oed-headline) and
  [Quantitative results](#sec-b14-results).
:::

## Why design an observation?

Telescope time is the scarcest resource in observational astronomy. A night on an
8-metre is worth tens of thousands of dollars; a Gaia-quality proper motion or a
high-resolution spectrum of a faint cluster star is expensive in a way that the
analysis afterwards is not. And yet the decision of *what to observe* — how many
stars, how far out, in which channel — is usually made by intuition and heritage:
"measure the brightest stars near the centre, that's where the signal is."

Sometimes intuition is right. Often it is *exactly backwards*. The information
about a parameter is not where the signal is largest; it is where the signal is
most **sensitive to that parameter**. Those can be very different places, and the
gap between them is telescope time thrown away.

OED replaces the guess with a computation. The key realisation is that the
precision you will achieve is a *known function of the observing strategy* — known
**before you collect a single photon** — through the Fisher information. So you can
treat the observing strategy as a free variable and maximise the information you
expect to extract. The output is not just "a good plan"; it is a *quantified*
plan: "this allocation reaches the same precision with $3.7\times$ fewer stars."

## The foundation: Fisher information and the Cramér–Rao bound

Suppose your data $\mathbf{d}$ depend on parameters $\boldsymbol\theta$ through a
likelihood $\mathcal{L}(\mathbf{d}\mid\boldsymbol\theta)$, and write the
log-likelihood $\ell = \ln\mathcal{L}$. Near the truth, $\ell$ is a hill in
parameter space, and its **curvature** tells you how sharply the data pin the
parameters: a sharp peak means a tight measurement, a flat ridge means a
degeneracy. The **Fisher information matrix** is exactly that curvature, averaged
over data realisations:

```{math}
:label: b14-fisher-def
F_{ij} \;=\; -\,\Big\langle \frac{\partial^2 \ell}{\partial\theta_i\,\partial\theta_j} \Big\rangle .
```

Its meaning is delivered by the **Cramér–Rao bound**: *no* unbiased estimator can
do better than

```{math}
:label: b14-cramer-rao
\mathrm{Cov}(\hat{\boldsymbol\theta}) \;\succeq\; F^{-1}.
```

So $F^{-1}$ is the best achievable covariance, and $\sqrt{(F^{-1})_{ii}}$ is the
tightest $1\sigma$ error bar you can hope to put on $\theta_i$ — *computable from
the model alone*, before observing. That is the hinge of the whole method: the
error bar is a function of the design, so you can optimise it.

For data with Gaussian errors — our binned dispersions, with measurement
uncertainty $\delta\sigma$ — the Fisher matrix has a simple, **additive** form. If
$g(\boldsymbol\theta)$ is the model prediction for a datum,

```{math}
:label: b14-fisher-gaussian
F_{ij} \;=\; \sum_{\rm data}\; \frac{1}{\delta\sigma^2}\,
   \frac{\partial g}{\partial\theta_i}\,\frac{\partial g}{\partial\theta_j}.
```

Two features of {eq}`b14-fisher-gaussian` do all the work below. First, it is a
**sum over independent data** — Fisher information adds. Second, each term is the
*sensitivity* $\partial g/\partial\theta$ squared, divided by the noise. A datum is
informative not when $g$ is large, but when $g$ **moves** as $\theta$ moves.

:::{note} **Why a Gauss–Newton Fisher, not a Hessian**
:class: dropdown
We never form $\partial^2\ell/\partial\theta^2$ directly. For a Gaussian
likelihood the expected Fisher equals the **Gauss–Newton** form
$J^{\mathsf T}\Sigma^{-1}J$ with $J=\partial g/\partial\theta$ — one Jacobian, no
second derivatives. That is cheaper *and* it is the only form that survives our
diffrax-backed forward model on the equilibrium-solver profiles, whose
reverse-mode `custom_vjp` defines no forward-mode rule (a Hessian needs
forward-over-reverse). Here `progenax.kinematics.fisher_information_gn` is exactly
this $J$-based Fisher.
:::

(sec-b14-physics)=
## The physical question: how do you measure orbital anisotropy?

Our target is the **velocity anisotropy** of a star cluster — the degree to which
stellar orbits are radially or tangentially biased. In a spherical system each
star's velocity splits into a **radial** part (toward or away from the cluster
centre) and two **tangential** parts, with dispersions $\sigma_r$ and $\sigma_t$.
The Binney anisotropy parameter measures their asymmetry,

```{math}
:label: b14-beta
\beta(r) \;=\; 1 - \frac{\sigma_t^2(r)}{\sigma_r^2(r)},
```

which is $0$ for isotropic orbits and rises toward $1$ for radially-biased ones. We
adopt the **Osipkov–Merritt** law {cite:p}`Merritt1985`,

```{math}
:label: b14-om
\beta_{\rm OM}(r) \;=\; \frac{r^2}{r^2 + r_a^2},
```

isotropic in the core and increasingly radial outward, with a single knob: the
**anisotropy radius** $r_a$, the radius where $\beta=\tfrac12$. Measuring the
cluster's orbital structure *is* measuring $r_a$.

Why does this matter physically? Anisotropy is a fossil of how a cluster formed and
how it is being torn apart: violent relaxation, tidal stripping, and radial-orbit
instability all leave their signature in $\beta(r)$. In dwarf-galaxy dynamics the
same $\beta$ is the dominant nuisance in the **mass–anisotropy degeneracy** that
limits dark-matter cusp-versus-core measurements. So $r_a$ is both a number worth
measuring and a number that is *notoriously hard* to measure — which is what makes
it a perfect OED target.

### The three observables, and why $\sigma_r \neq$ the radial velocity

Here is the subtlety that makes anisotropy measurable at all, and it trips up
nearly everyone the first time. The word "radial" means **three different things**:

```{list-table}
:header-rows: 1
:label: tbl-b14-radials

* - Quantity
  - "Radial" with respect to…
  - Direction
* - $\sigma_r$
  - the **cluster centre** (3-D)
  - along $\hat r$, centre $\to$ star
* - $v_{\rm los}$ / RV
  - the **observer**
  - along the line of sight $\hat z$ (Doppler)
* - $\sigma_{{\rm pm},R}$
  - the **cluster centre, projected on-sky**
  - radial in the plane of the sky
```

The **radial velocity** you measure spectroscopically is *radial to the observer* —
the line-of-sight Doppler shift — **not** the 3-D motion toward the cluster centre.
A star at projected radius $R$ and line-of-sight depth $z$ sits at 3-D radius
$r=\sqrt{R^2+z^2}$, and its local radial direction $\hat r$ makes an angle $\psi$
with the sight line, $\cos\psi = z/r$. The observed line-of-sight velocity
dispersion at that point is therefore a *projection* that mixes the radial and
tangential dispersions:

```{math}
:label: b14-projection-geometry
\sigma_{\rm los}^2(\text{at }r) \;=\; \sigma_r^2\cos^2\psi + \sigma_t^2\sin^2\psi
   \;=\; \sigma_r^2\Big(1 - \beta\,\frac{R^2}{r^2}\Big),
```

using $\sigma_t^2=(1-\beta)\sigma_r^2$ and $z^2+R^2=r^2$. Only for a sight line
*through the centre* ($R\to0$, $\psi\to0$) does $\sigma_{\rm los}=\sigma_r$;
everywhere else the LOS velocity is a $\beta$-weighted blend. Integrating
{eq}`b14-projection-geometry` along the sight line, weighted by the density
$\rho(r)$, gives the **observed** profile. Doing the same for the two
plane-of-sky (proper-motion) components yields the full Binney & Mamon (1982)
projection — three distinct observables of the same underlying $\sigma_r(r)$ and
$\beta(r)$:

```{math}
:label: b14-bm82
\begin{aligned}
\Sigma\,\sigma_{\rm los}^2(R)   &= 2\!\int_R^\infty \Big(1 - \beta\,\tfrac{R^2}{r^2}\Big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}},\\
\Sigma\,\sigma_{{\rm pm},R}^2(R)&= 2\!\int_R^\infty \Big(1 - \beta + \beta\,\tfrac{R^2}{r^2}\Big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}},\\
\Sigma\,\sigma_{{\rm pm},T}^2(R)&= 2\!\int_R^\infty \big(1 - \beta\big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}}.
\end{aligned}
```

The $\sigma_r^2(r)$ inside these integrals is itself the solution of the
**anisotropic Jeans equation**; for the OM law it has the closed integrating-factor
form

```{math}
:label: b14-jeans
\rho\,\sigma_r^2(r) \;=\; \frac{1}{r^2+r_a^2}\int_r^\infty (s^2+r_a^2)\,\rho(s)\,
   \frac{G\,M(<s)}{s^2}\,ds,
```

evaluated by the packaged differentiable `project_dispersion`. The point of all
this machinery is one sentence:

> In the isotropic limit ($\beta=0$) the three kernels in {eq}`b14-bm82` collapse
> to $1$ and $\sigma_{\rm los}=\sigma_{{\rm pm},R}=\sigma_{{\rm pm},T}$ — **the
> anisotropy lives only in the *ratios* between channels.**

The tangential proper motion is the cleanest probe: $\sigma_{{\rm pm},T}^2\propto
(1-\beta)$, so as orbits turn radial ($\beta\to1$) the tangential PM **collapses**
while the LOS and radial-PM channels stay up. And because $\beta(r)$ grows outward,
*the channels disagree most in the outskirts*. A single $\sigma_{\rm los}(R)$
profile cannot break the tie — a cold outer LOS profile can mean "low $\sigma_r$"
**or** "high $\beta$." Proper motions break it. Keep that picture in mind: it is the
physics the optimiser is about to rediscover, having been told none of it.

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
  - RV ($\sigma_{\rm los}$), PM radial ($\sigma_{{\rm pm},R}$), PM tangential ($\sigma_{{\rm pm},T}$) — the **design lever**; each channel's information content is set by the B&M82 projection kernels {eq}`b14-bm82`.
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
and the A/D criteria are not scale-invariant — $\mathrm{tr}\,F^{-1}$ would be
dominated by whichever parameter happens to carry the largest *unit magnitude*,
not the largest *fractional* uncertainty. Differentiating with respect to
$\ln\theta$ ($J\to J\cdot\mathrm{diag}(\theta_{\rm fid})$) makes $F$
**dimensionless** ($\mathrm{cond}\approx45$), every covariance entry a
**fractional variance**, and the c-headline a **fractional precision**
$\sigma(r_a)/r_a$. The nuisance prior is fractional too,
$\mathtt{PRIOR\_DIAG}=[0,\,1/0.3^2,\,1/0.3^2]$ (none on the target). This is
standard Fisher-forecasting practice, here forced on us by the unit spread.
```

(sec-b14-method)=
## The method: an additive, design-linear Fisher

Now we make {eq}`b14-fisher-gaussian` concrete for this problem, and exploit a
structure that turns the whole optimisation into trivial linear algebra.

**The per-datum noise.** A velocity dispersion measured from $n$ stars, each with
intrinsic dispersion $\sigma$ and measurement error $\epsilon$, has a sampling
uncertainty

```{math}
:label: b14-dispersion-error
\delta\sigma^2 \;=\; \frac{\sigma^2 + \epsilon^2}{2\,n}.
```

The factor of $2n$ (not $n$) is the standard error on a *dispersion*, not a mean —
a dispersion is a second moment, so it converges more slowly. More stars in a bin
($n\uparrow$) sharpen that bin's datum; that is the design's only lever on the
noise.

**The design and the budget.** A design is an allocation of the budget across the
36 cells. We parametrise it on the softmax simplex so it stays a valid, positive
allocation under unconstrained optimisation,

```{math}
:label: b14-design
n_{b,c} \;=\; N_{\rm total}\,\big[\mathrm{softmax}(\mathbf z)\big]_{b,c},
\qquad
n_{{\rm eff},\,b,c} = n_{b,c}\cdot c(R_b),
```

where $c(R_b)$ is the fixed completeness (stars you *target* in the faint outskirts
that you do not actually detect). The design variables are the logits $\mathbf z$.

**The additive backbone (ADR 0004).** Substituting {eq}`b14-dispersion-error` into
{eq}`b14-fisher-gaussian` and grouping by cell, the design Fisher is a
budget-weighted **sum of design-independent per-star blocks**:

```{math}
:label: b14-additive-fisher
F(\text{design}) \;=\; \sum_{b,\,c} n_{{\rm eff},\,b,c}\; M_{b,c},
\qquad
M_{b,c} \;=\; \frac{2\,J_{b,c}\,J_{b,c}^{\mathsf T}}{\sigma^2_{b,c} + \epsilon_c^2},
\qquad
J_{b,c} = \frac{\partial\,\sigma^{\rm pred}_{b,c}}{\partial\ln\theta}.
```

This is the load-bearing trick. Each $M_{b,c}$ is a $3\times3$ rank-1 matrix — the
information *one star* in that cell-and-channel carries about
$\theta=(r_a,M,r_h)$ — and it **does not depend on the design**. So the Jacobian
$J_{b,c}$ is computed **once**, by a single reverse-mode `jacrev` through
`project_dispersion` at the truth. After that, the design enters
{eq}`b14-additive-fisher` *only* through the scalar weights $n_{b,c}$:

> Optimising the observing strategy is $F=\sum n\,(c\,M)$ → invert a $3\times3$ →
> read one element. The expensive B&M82 projection is **never** re-differentiated
> inside the optimiser loop, and the gradients
> $\partial(\text{criterion})/\partial(\text{design})$ are pure linear algebra over
> the precomputed blocks.

We use `jacrev` (reverse mode) because it is the supported, tested AD path for
*all* profiles and stays correct if a King/Michie mock is ever swapped in (those
equilibrium-solver profiles hit a `custom_vjp` ODE with no forward-mode rule). On
the analytic-density Plummer path used here there is no ODE, so forward-mode would
also work — `jacrev` is the robust choice, not a forced one.

**Three optimality criteria** ride the same $F=\sum n\,(c\,M)$. The choice of
criterion is the choice of *what "optimal" means*:

```{math}
:label: b14-criteria
\underbrace{c\!:\;\min\,(F^{-1})_{r_a r_a}}_{\text{the }r_a\text{ variance, nuisances profiled}}
\qquad
\underbrace{D\!:\;\max\,\log\det F}_{\text{the whole }\theta\text{ ellipsoid}}
\qquad
\underbrace{A\!:\;\min\,\mathrm{tr}\,F^{-1}}_{\text{total fractional variance}}.
```

**c-optimality** is our headline: it minimises the variance of $r_a$ *after
marginalising over* $M$ and $r_h$ — exactly the $(r_a,r_a)$ element of the full
inverse, which already accounts for how the target trades off against the
nuisances. **D-optimality** maximises the determinant of $F$, i.e. shrinks the
*volume* of the joint confidence ellipsoid over all three parameters.
**A-optimality** minimises the trace of $F^{-1}$, the average variance. They are
genuinely different objectives, and under the $M\!\leftrightarrow\!r_a$
(through $GM$) and $r_h\!\leftrightarrow\!r_a$ degeneracies they put stars at
**different radii** — the criterion-disagreement lesson, made visible below.

:::{note} **Why c and D *should* disagree — and why that's the point**
:class: dropdown
D-optimality maximises the volume of the whole confidence ellipsoid, so it happily
spends stars tightening $M$ and $r_h$. c-optimality only cares about the $r_a$ axis
of that ellipsoid *after the others are marginalised away*, so it refuses to spend
stars where they would mostly improve nuisances. A demo in which c and D agreed
would be hiding the most important idea in OED: **optimising for "everything" is
not optimising for the one number you came to measure.** If your science is a
single parameter — an anisotropy, a mass-function slope, a tidal radius — you want
c (or its subset generalisation, $D_s$), not D.
:::

## Validating a pre-data calculation: the calibration ensemble

A Fisher forecast is a *promise*: "if you observe this way, your error bar will be
this large." A promise made before any data is only trustworthy if you check it,
because the Cramér–Rao bound {eq}`b14-cramer-rao` is a **local, Gaussian**
approximation — exact in the high-information limit, optimistic when the
likelihood is curved or the estimator is biased.

So we close the loop once, as a *gate* (not inside the optimiser). We draw a
64-member ensemble of mock catalogues from the actual Osipkov–Merritt Plummer
sampler, project each to the sky, bin it, and fit $\hat r_a$ by maximum *a
posteriori* (with the same fractional prior the design Fisher used). The realised
scatter should match the Fisher prediction,

```{math}
:label: b14-calibration
\mathrm{Var}(\hat r_a)\big/ r_a^2 \;\approx\; (F^{-1})_{r_a r_a},
```

both sides being *fractional* variances in the $\ln\theta$ metric. The tolerance is
not a free knob: the Monte-Carlo error on a variance estimated from $n_{\rm draws}$
draws is $\approx\sqrt{2/n_{\rm draws}}$, so the gate is a principled
$2\sigma$ band, $2\sqrt{2/64}\approx0.35$. The realised $\sigma(r_a)/r_a=0.109$
sits just below the Fisher's $0.121$ — the pre-data Fisher is **mildly
conservative** (the binned-dispersion estimator loses a little information relative
to the idealised per-star Fisher), and the two agree well inside the band. The
promise holds.

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

(sec-b14-results)=
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
cancels in the $c_{\rm uniform}/c_{\rm designed}$ ratio, because $c\propto1/N$ when
the prior is held fixed); the "$\approx3.9\times$ fewer stars" gloss on the
frontier is the same physics read off the swept budget curve, where the prior no
longer cancels and the slope departs mildly from $1/N$.

## What the optimum means: science implications

The $3.66\times$ is not a free lunch — it is the optimiser **discovering the
physics of where information lives**, and it generalises well beyond this mock.

**1. Information is localised — in space *and* in channel.** Anisotropy $\beta(r)$
grows outward, and the tangential-PM sensitivity to $r_a$ grows with it. So the
*value* of a star for measuring $r_a$ depends sharply on where it sits and how you
measure it. OED makes that quantitative instead of intuitive: the headline figure
shows the optimiser routing PM stars to exactly the radii where $\beta(R)$ is
turning over.

**2. Proper motions break the mass–anisotropy degeneracy where it bites.** A
line-of-sight dispersion profile alone is degenerate — recall
{eq}`b14-projection-geometry`: a cold outer $\sigma_{\rm los}$ can mean low
$\sigma_r$ *or* high $\beta$. The two PM channels carry different $\beta$-weightings
{eq}`b14-bm82`, so they lift the degeneracy — and the OED tells you *where the
lift is largest*: the outskirts. **The result — RV in the core, PM in the outskirts
— is the observing strategy this degeneracy demands, derived from first
principles.** That is directly actionable for real Gaia-PM + spectroscopic
campaigns on globular clusters and dwarf spheroidals.

**3. Design for your target, not for "everything."** The c-vs-D-vs-A divergence is
the deepest lesson. If your science is one number — anisotropy as a probe of a
dark-matter cusp/core, an IMBH's kinematic signature, a cluster's tidal state — a
design that minimises the joint ellipsoid (D) *wastes* stars tightening nuisances.
c-optimality can buy a multiplicative factor in the precision you actually care
about.

**4. The honest caveat is itself a research direction.** The optimum is optimal
*for the assumed OM-Plummer model*. That model-dependence is the central limitation
of all OED — and it points straight at the most valuable extension: **robust** and
**model-discriminating** design (below), which `progenax` is unusually well-placed
to do because it ships *more than one* differentiable forward model for the same
observable.

(sec-b14-capabilities)=
## What OED can do with progenax

Stage 1 is one instance of a general capability: **every differentiable `progenax`
forward model, paired with a likelihood, is a Fisher matrix
{eq}`b14-fisher-gaussian` — and therefore an OED problem.** Because the Fisher is
additive {eq}`b14-additive-fisher`, you assemble the design Fisher for *any* of
these by computing per-star (or per-bin) blocks once and optimising weights. Below,
**[done]** is demonstrated here, **[B#]** means an existing demo already provides
the differentiable forward model, and **[enabled]** means the machinery supports it
but it is not yet built. We mark the line honestly between *demonstrated* and
*possible*.

### By design space — *what you optimise*

```{list-table}
:header-rows: 1
:label: tbl-b14-designspace

* - Design lever
  - What it allocates
  - Status
* - **Channel allocation** (RV $\leftrightarrow$ PM)
  - kinematic measurement type per star
  - **[done]**
* - **Radial / spatial allocation**
  - where on the sky to observe
  - **[done]**; generalises to number-count surveys (King/EFF $W_0$, $r_c$, **$r_t$** from binned counts via the Poisson channel) **[B11, B7]**
* - **Survey depth / magnitude limit**
  - how faint to go (ZAMS $L$ $\to$ completeness), trading area vs depth
  - **[enabled — this is Stage 2]**
* - **Epochs / cadence / cost**
  - astrometric epochs (PM precision $\propto$ epochs), spectroscopic cadence, under a cost model
  - **[enabled — this is Stage 3]**
* - **Multi-channel fusion**
  - the optimal *mix of instruments* — Fisher sums across photometry + kinematics + counts
  - **[enabled]**
```

### By science target — *what parameter you measure*

```{list-table}
:header-rows: 1
:label: tbl-b14-targets

* - Target
  - Forward model
  - Status
* - Anisotropy $r_a$ / mass $M$ / $r_h$ (kinematics)
  - `project_dispersion` (B&M82)
  - **[done]**
* - IMF slope $\alpha$ / environment $\alpha_3$
  - IMF likelihood + mass function
  - **[B5]** — design which mass range / how many stars pin the high-mass slope
* - Binary fraction $f_b$ + Moe $P$–$q$–$e$
  - binary-inflated dispersion / mass function
  - **[B12, B4]** — allocate RV epochs + photometry to constrain binarity
* - Concentration, tidal radius $r_t$ $\to$ Jacobi $R_{\rm gal}$
  - count profile + tidal truncation
  - **[B7]** — outer-bin allocation; 93% of $r_t$ info is in the outskirts
* - Rotation $\omega\sin i$
  - rotating projected-kinematics model
  - **[B8]** — break the rank-1 $(\omega,i)$ degeneracy with multiple channels
* - Multi-population (halo+core, mass segregation)
  - `MultiComponentCluster` (Engine A/B)
  - **[enabled]** — where to observe to *separate* species
```

### By optimality criterion — *what "best" means*

**c / $D_s$** (single target or a subset of targets) — **[done, headline]**.
**D** (all parameters) and **A** (average) — **[done, as the contrast]**. **E**
(worst-constrained eigendirection), **I/G** (prediction-oriented) — same
$F=\sum n\,M$ machinery, **[enabled]**. The standout is **T-optimality (model
discrimination)**: `progenax` carries *multiple* differentiable forward models for
the same observable — OM-Jeans vs the exact Michie second moment
(`df_moment_dispersion`), Engine-A vs Engine-B, OM vs a native $\beta(r)$. The
[anisotropy demo (B6)](anisotropy.md) already shows OM and Michie *diverge in the
outskirts*; a T-optimal design **maximises that divergence**, telling you where to
observe to *distinguish the models*, not merely fit one. **[enabled — high value]**.

### By robustness and adaptivity

- **Bayesian OED** — expected information averaged over a prior on the nuisances;
  the priors are already in the Fisher. **[enabled]**
- **Robust / maximin OED** — optimise the *worst case* over model or nuisance
  uncertainty; the honest answer to the model-dependence caveat. **[enabled]**
- **Sequential / adaptive OED** — re-optimise as data arrive (greedy or batch); the
  differentiable Fisher is the per-step ingredient. **[enabled, not built]**

The throughline: `progenax` is unusual in being a **fully differentiable
astrophysical IC-and-observable stack**, so
$\partial(\text{information})/\partial(\text{observing strategy})$ is computable for
essentially any cluster-science parameter — which turns "how should we observe?"
from a heritage decision into a gradient-ascent problem across all of the above.

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

:::{note} What we just learned
The precision you will achieve is a *known function of the observing strategy* —
the Fisher information {eq}`b14-fisher-gaussian`, bounded below by Cramér–Rao
{eq}`b14-cramer-rao` — so you can **design the observation before taking it**. For
cluster anisotropy, the information lives in the *ratios* between the RV and the
two PM channels {eq}`b14-bm82`, and because $\beta(r)$ grows outward, a c-optimal
design **puts proper motions in the outskirts** and reaches the same precision on
$r_a$ with $3.66\times$ fewer stars. The whole optimisation is cheap because the
Fisher is additive and design-linear {eq}`b14-additive-fisher` — one `jacrev`, then
linear algebra. And the same recipe — *forward model → Fisher → optimise the
design* — applies to every differentiable model in `progenax`
([the capability map](#sec-b14-capabilities)).
:::

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

The Fisher-information / Cramér–Rao foundation is standard estimation theory (see
e.g. Tegmark, Taylor & Heavens 1997 for the astrophysical forecasting form). The
Osipkov–Merritt anisotropy law $\beta(r)=r^2/(r^2+r_a^2)$ is {cite:t}`Merritt1985`;
the line-of-sight projection of an anisotropic spherical model into
$\sigma_{\rm los}$, $\sigma_{{\rm pm},R}$, $\sigma_{{\rm pm},T}$ is
Binney & Mamon (1982, MNRAS 200, 361). The differentiable `project_dispersion`
forward model is documented on the
[velocity-DF kinematics](../10-theory/velocity-dfs/rotation-anisotropy.md)
pages; the anisotropy *recovery* counterpart is the
[anisotropy demo (B6)](anisotropy.md), and [B8](rotation.md) introduces the same
sky-projection helper.
```
