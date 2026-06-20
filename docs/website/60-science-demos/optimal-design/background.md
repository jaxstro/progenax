---
title: The OED formalism — Fisher information, the additive backbone, and sky projection
subtitle: The shared machinery every worked example on this page reuses — derived once, from first principles
description: "The shared theory for progenax's optimal-experimental-design demos. Fisher information and the Cramer-Rao bound make the achievable error bar a known function of the observing strategy, computable before any photon is collected. For binned Gaussian dispersions the Fisher is additive and design-linear -- F = sum of budget-weighted, design-independent per-star blocks -- so a single reverse-mode Jacobian through the differentiable forward model is computed once and the optimisation reduces to linear algebra. Includes the dimensionless ln-theta metric (ADR 0011), the c/D/A optimality criteria, and the Binney & Mamon (1982) projection geometry that turns sigma_r(r) and beta(r) into the three sky observables (RV, PM_R, PM_T)."
---

# The OED formalism — the machinery every example reuses

Every worked example in this section — [the anisotropy design](anisotropy.md), [the
dynamical-mass depth knob](dynamical-mass.md) — runs on the *same* engine. This page
builds that engine once, from first principles, so the example pages can spend their
words on the science instead of re-deriving the algebra. If you have just landed here
from one of them, this is the page that explains *why* a pre-data error bar is even
computable.

:::{note} **Reading paths**
:class: dropdown

- **New to experimental design?** Read straight through — every equation is earned
  with a physical picture first.
- **Already comfortable with Fisher information and the Cramér–Rao bound?** Skip to
  [the additive backbone](#sec-oed-additive) — the trick that turns the optimisation
  into linear algebra — or to [the projection geometry](#sec-oed-projection) for the
  B&M82 kernels.
- **Just want to run an example?** Go back to [the anisotropy design](anisotropy.md)
  or [the dynamical-mass design](dynamical-mass.md); this page is reference theory.
:::

## Why design an observation at all?

Telescope time is the scarcest resource in observational astronomy. A night on an
8-metre is worth tens of thousands of dollars; a Gaia-quality proper motion or a
high-resolution spectrum of a faint cluster star is expensive in a way that the
analysis afterwards is not. And yet the decision of *what to observe* — how many
stars, how far out, how faint, in which channel — is usually made by intuition and
heritage: "measure the brightest stars near the centre, that's where the signal is."

Sometimes intuition is right. Often it is *exactly backwards*. The information about a
parameter is not where the signal is largest; it is where the signal is most
**sensitive to that parameter**. Those can be very different places, and the gap
between them is telescope time thrown away.

**Optimal experimental design** (OED) replaces the guess with a computation. The key
realisation is that the precision you will achieve is a *known function of the
observing strategy* — known **before you collect a single photon** — through the
Fisher information. So you can treat the observing strategy as a free variable and
maximise the information you expect to extract. The output is not just "a good plan";
it is a *quantified* plan: "this allocation reaches the same precision with
$3.66\times$ fewer stars," or "going one magnitude deeper buys you nothing." OED is
rare in astronomy because almost no simulator can compute
$\partial(\text{information})/\partial(\text{observing strategy})$ — and that is
exactly what a *differentiable* forward model makes possible.

## The foundation: Fisher information and the Cramér–Rao bound

Suppose your data $\mathbf{d}$ depend on parameters $\boldsymbol\theta$ through a
likelihood $\mathcal{L}(\mathbf{d}\mid\boldsymbol\theta)$, and write the
log-likelihood $\ell = \ln\mathcal{L}$. Near the truth, $\ell$ is a hill in parameter
space, and its **curvature** tells you how sharply the data pin the parameters: a
sharp peak means a tight measurement, a flat ridge means a degeneracy. The **Fisher
information matrix** is exactly that curvature, averaged over data realisations:

```{math}
:label: oed-fisher-def
F_{ij} \;=\; -\,\Big\langle \frac{\partial^2 \ell}{\partial\theta_i\,\partial\theta_j} \Big\rangle .
```

Its meaning is delivered by the **Cramér–Rao bound**: *no* unbiased estimator can do
better than

```{math}
:label: oed-cramer-rao
\mathrm{Cov}(\hat{\boldsymbol\theta}) \;\succeq\; F^{-1}.
```

So $F^{-1}$ is the best achievable covariance, and $\sqrt{(F^{-1})_{ii}}$ is the
tightest $1\sigma$ error bar you can hope to put on $\theta_i$ — *computable from the
model alone*, before observing. That is the hinge of the whole method: the error bar
is a function of the design, so you can optimise it.

For data with Gaussian errors — our binned dispersions, with measurement uncertainty
$\delta\sigma$ — the Fisher matrix has a simple, **additive** form. If
$g(\boldsymbol\theta)$ is the model prediction for a datum,

```{math}
:label: oed-fisher-gaussian
F_{ij} \;=\; \sum_{\rm data}\; \frac{1}{\delta\sigma^2}\,
   \frac{\partial g}{\partial\theta_i}\,\frac{\partial g}{\partial\theta_j}.
```

Two features of {eq}`oed-fisher-gaussian` do all the work below. First, it is a **sum
over independent data** — Fisher information adds. Second, each term is the
*sensitivity* $\partial g/\partial\theta$ squared, divided by the noise. A datum is
informative not when $g$ is large, but when $g$ **moves** as $\theta$ moves.

:::{note} **Why a Gauss–Newton Fisher, not a Hessian**
:class: dropdown
We never form $\partial^2\ell/\partial\theta^2$ directly. For a Gaussian likelihood
the expected Fisher equals the **Gauss–Newton** form $J^{\mathsf T}\Sigma^{-1}J$ with
$J=\partial g/\partial\theta$ — one Jacobian, no second derivatives. That is cheaper
*and* it is the only form that survives our diffrax-backed forward model on the
equilibrium-solver profiles, whose reverse-mode `custom_vjp` defines no forward-mode
rule (a Hessian would need forward-over-reverse). Here
`progenax.kinematics.fisher_information_gn` is exactly this $J$-based Fisher.
:::

(sec-oed-dimensionless)=
## The dimensionless $\ln\theta$ metric (ADR 0011)

Cluster parameters span enormous dynamic ranges: a total mass $M\approx10^5\,\Msun$
sits next to a half-mass radius $r_h\approx3$ pc in the same vector. Built in raw
units, the Fisher is wildly ill-conditioned ($\mathrm{cond}\approx1.7\times10^9$) and
the A/D criteria below are not scale-invariant — $\mathrm{tr}\,F^{-1}$ would be
dominated by whichever parameter happens to carry the largest *unit magnitude*, not
the largest *fractional* uncertainty.

```{important}
:label: imp-oed-dimensionless
**Every OED example builds its Fisher in the dimensionless $\ln\theta$ metric.**
Differentiating with respect to $\ln\theta$ (equivalently $J\to J\cdot
\mathrm{diag}(\theta_{\rm fid})$) makes $F$ **dimensionless**
($\mathrm{cond}\approx45$), every covariance entry a **fractional variance**, and the
headline a **fractional precision** $\sigma(\theta)/\theta$. Nuisance priors are
fractional too (e.g. $\mathtt{PRIOR\_DIAG}=[0,\,1/0.3^2,\,1/0.3^2]$ for a 30%
prior, none on the target). This is standard Fisher-forecasting practice, here forced
on us by the unit spread.
```

(sec-oed-additive)=
## The additive, design-linear backbone (ADR 0004)

This is the structural trick that makes the whole optimisation cheap. Start from the
per-datum noise: a velocity dispersion measured from $n$ stars, each with intrinsic
dispersion $\sigma$ and measurement error $\epsilon$, has a sampling uncertainty

```{math}
:label: oed-dispersion-error
\delta\sigma^2 \;=\; \frac{\sigma^2 + \epsilon^2}{2\,n}.
```

The factor of $2n$ (not $n$) is the standard error on a *dispersion*, not a mean — a
dispersion is a second moment, so it converges more slowly. More stars in a bin
($n\uparrow$) sharpen that bin's datum; that is the design's only lever on the noise.

**The design and the budget.** A design allocates a fixed budget $N_{\rm total}$
across the cells (projected-radius bin $b$ × kinematic channel $c$). We parametrise it
on the softmax simplex so it stays a valid, positive allocation under unconstrained
optimisation,

```{math}
:label: oed-design
n_{b,c} \;=\; N_{\rm total}\,\big[\mathrm{softmax}(\mathbf z)\big]_{b,c},
```

with the logits $\mathbf z$ as the design variables. Substituting
{eq}`oed-dispersion-error` into {eq}`oed-fisher-gaussian` and grouping by cell, the
design Fisher becomes a budget-weighted **sum of design-independent per-star blocks**:

```{math}
:label: oed-additive-fisher
F(\text{design}) \;=\; \sum_{b,\,c} n_{b,c}\; M_{b,c},
\qquad
M_{b,c} \;=\; \frac{2\,J_{b,c}\,J_{b,c}^{\mathsf T}}{\sigma^2_{b,c} + \epsilon_c^2},
\qquad
J_{b,c} = \frac{\partial\,\sigma^{\rm pred}_{b,c}}{\partial\ln\theta}.
```

Each $M_{b,c}$ is a $3\times3$ rank-1 matrix — the information *one star* in that
cell-and-channel carries about $\boldsymbol\theta$ — and it **does not depend on the
design**. So the Jacobian $J_{b,c}$ is computed **once**, by a single reverse-mode
`jacrev` through the differentiable forward model at the truth. After that, the design
enters {eq}`oed-additive-fisher` *only* through the scalar weights $n_{b,c}$:

> Optimising the observing strategy is $F=\sum n\,M$ → invert a $3\times3$ → read one
> element. The expensive projection is **never** re-differentiated inside the
> optimiser loop, and the gradients
> $\partial(\text{criterion})/\partial(\text{design})$ are pure linear algebra over
> the precomputed blocks.

We use `jacrev` (reverse mode) because it is the supported, tested AD path for *all*
profiles and stays correct if a King/Michie mock is ever swapped in (those
equilibrium-solver profiles hit a `custom_vjp` ODE with no forward-mode rule). On the
analytic-density Plummer path used in the examples there is no ODE, so forward-mode
would also work — `jacrev` is the robust choice, not a forced one.

:::{aside} **Where the depth knob plugs in**
The [dynamical-mass example](dynamical-mass.md) promotes the survey **depth** to a
design variable. It does so *without* touching this backbone: the predicted
dispersions $\sigma^{\rm pred}(r)$ are a property of the potential, independent of how
faint you observe, so $J$ is still computed once. Depth enters only through the
per-channel effective error $\epsilon_c \to \epsilon_{{\rm eff},c}(m_{\rm lim})$ and a
per-bin availability weight — both cheap, differentiable IMF/ZAMS integrals layered
*around* {eq}`oed-additive-fisher`. That the backbone survives is the whole reason
depth optimisation is affordable.
:::

(sec-oed-criteria)=
## What "optimal" means: the c / D / A criteria

The same $F=\sum n\,M$ supports several notions of "best." The choice of criterion is
the choice of *what you are optimising for*:

```{math}
:label: oed-criteria
\underbrace{c\!:\;\min\,(F^{-1})_{\theta_\star\theta_\star}}_{\text{the target's variance, nuisances profiled}}
\qquad
\underbrace{D\!:\;\max\,\log\det F}_{\text{the whole }\theta\text{ ellipsoid}}
\qquad
\underbrace{A\!:\;\min\,\mathrm{tr}\,F^{-1}}_{\text{total fractional variance}}.
```

**c-optimality** is the headline of both examples: it minimises the variance of one
target parameter $\theta_\star$ *after marginalising over* the nuisances — exactly the
$(\theta_\star,\theta_\star)$ element of the full inverse, which already accounts for
how the target trades off against the others. **D-optimality** maximises the
determinant of $F$, shrinking the *volume* of the joint confidence ellipsoid over all
parameters. **A-optimality** minimises the trace of $F^{-1}$, the average variance.
They are genuinely different objectives, and under parameter degeneracies they put
stars at **different radii**.

:::{note} **Why c and D *should* disagree — and why that's the point**
:class: dropdown
D-optimality maximises the volume of the whole confidence ellipsoid, so it happily
spends stars tightening every nuisance. c-optimality only cares about the
$\theta_\star$ axis of that ellipsoid *after the others are marginalised away*, so it
refuses to spend stars where they would mostly improve nuisances. A demo in which c
and D agreed would be hiding the most important idea in OED: **optimising for
"everything" is not optimising for the one number you came to measure.** If your
science is a single parameter — an anisotropy radius, a dynamical mass, a tidal radius
— you want c (or its subset generalisation $D_s$), not D.
:::

(sec-oed-projection)=
## The projection geometry: $\sigma_r(r)$ into the three sky observables (B&M82)

Both examples observe the cluster through the *same* three kinematic channels, and the
geometry that connects the cluster's internal velocity ellipsoid to what a telescope
records is worth getting exactly right — it trips up nearly everyone the first time.
In a spherical system each star's velocity splits into a **radial** part (toward or
away from the cluster centre) and two **tangential** parts, with dispersions
$\sigma_r$ and $\sigma_t$. The Binney anisotropy parameter measures their asymmetry,

```{math}
:label: oed-beta
\beta(r) \;=\; 1 - \frac{\sigma_t^2(r)}{\sigma_r^2(r)},
```

which is $0$ for isotropic orbits and rises toward $1$ for radially-biased ones.

The word "radial" means **three different things** here, and conflating them is the
classic mistake:

```{list-table}
:header-rows: 1
:label: tbl-oed-radials

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
the line-of-sight Doppler shift — **not** the 3-D motion toward the cluster centre. A
star at projected radius $R$ and line-of-sight depth $z$ sits at 3-D radius
$r=\sqrt{R^2+z^2}$, and its local radial direction $\hat r$ makes an angle $\psi$ with
the sight line, $\cos\psi = z/r$. The observed line-of-sight velocity dispersion at
that point is therefore a *projection* that mixes the radial and tangential
dispersions:

```{math}
:label: oed-projection-geometry
\sigma_{\rm los}^2(\text{at }r) \;=\; \sigma_r^2\cos^2\psi + \sigma_t^2\sin^2\psi
   \;=\; \sigma_r^2\Big(1 - \beta\,\frac{R^2}{r^2}\Big),
```

using $\sigma_t^2=(1-\beta)\sigma_r^2$ and $z^2+R^2=r^2$. Only for a sight line
*through the centre* ($R\to0$, $\psi\to0$) does $\sigma_{\rm los}=\sigma_r$;
everywhere else the LOS velocity is a $\beta$-weighted blend. Integrating
{eq}`oed-projection-geometry` along the sight line, weighted by the density
$\rho(r)$, gives the **observed** profile. Doing the same for the two plane-of-sky
(proper-motion) components yields the full Binney & Mamon (1982) projection — three
distinct observables of the same underlying $\sigma_r(r)$ and $\beta(r)$:

```{math}
:label: oed-bm82
\begin{aligned}
\Sigma\,\sigma_{\rm los}^2(R)   &= 2\!\int_R^\infty \Big(1 - \beta\,\tfrac{R^2}{r^2}\Big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}},\\
\Sigma\,\sigma_{{\rm pm},R}^2(R)&= 2\!\int_R^\infty \Big(1 - \beta + \beta\,\tfrac{R^2}{r^2}\Big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}},\\
\Sigma\,\sigma_{{\rm pm},T}^2(R)&= 2\!\int_R^\infty \big(1 - \beta\big)\,\rho\,\sigma_r^2\;\frac{r\,dr}{\sqrt{r^2-R^2}}.
\end{aligned}
```

The $\sigma_r^2(r)$ inside these integrals is itself the solution of the **anisotropic
Jeans equation**; for the Osipkov–Merritt law $\beta_{\rm OM}(r)=r^2/(r^2+r_a^2)$
{cite:p}`Merritt1985` it has the closed integrating-factor form

```{math}
:label: oed-jeans
\rho\,\sigma_r^2(r) \;=\; \frac{1}{r^2+r_a^2}\int_r^\infty (s^2+r_a^2)\,\rho(s)\,
   \frac{G\,M(<s)}{s^2}\,ds,
```

evaluated by the packaged differentiable `project_dispersion`. The point of all this
machinery is one sentence:

> In the isotropic limit ($\beta=0$) the three kernels in {eq}`oed-bm82` collapse to
> $1$ and $\sigma_{\rm los}=\sigma_{{\rm pm},R}=\sigma_{{\rm pm},T}$ — **the
> anisotropy lives only in the *ratios* between channels.**

The tangential proper motion is the cleanest probe: $\sigma_{{\rm pm},T}^2\propto
(1-\beta)$, so as orbits turn radial ($\beta\to1$) the tangential PM **collapses**
while the LOS and radial-PM channels stay up. And because $\beta(r)$ grows outward,
*the channels disagree most in the outskirts*. A single $\sigma_{\rm los}(R)$ profile
cannot break the tie — a cold outer LOS profile can mean "low $\sigma_r$" **or** "high
$\beta$." Proper motions break it. Notice too that the **total mass** $M$ enters every
one of these channels through $GM(<s)$ in {eq}`oed-jeans`: $\sigma_r^2\propto GM$, so a
dispersion measured *anywhere* weighs the cluster — the lever the
[dynamical-mass example](dynamical-mass.md) pulls.

:::{note} What this page gives you
The achievable error bar is a *known function of the observing strategy* — the Fisher
information {eq}`oed-fisher-gaussian`, bounded below by Cramér–Rao
{eq}`oed-cramer-rao` — so you can **design the observation before taking it**. For
binned Gaussian dispersions that Fisher is **additive and design-linear**
{eq}`oed-additive-fisher`: one `jacrev` for the per-star blocks, then linear algebra
over the budget weights. The three sky channels (RV, PM$_R$, PM$_T$) are the
{eq}`oed-bm82` projection of one internal $\sigma_r(r)$ and $\beta(r)$, and the
$\ln\theta$ metric {ref}`imp-oed-dimensionless` makes every headline a *fractional*
precision. The worked examples — [anisotropy](anisotropy.md) and
[dynamical mass](dynamical-mass.md) — pour different science into this same mould.
:::

## References

The Fisher-information / Cramér–Rao foundation is standard estimation theory (see e.g.
Tegmark, Taylor & Heavens 1997 for the astrophysical forecasting form). The
Osipkov–Merritt anisotropy law $\beta(r)=r^2/(r^2+r_a^2)$ is {cite:t}`Merritt1985`;
the line-of-sight projection of an anisotropic spherical model into $\sigma_{\rm los}$,
$\sigma_{{\rm pm},R}$, $\sigma_{{\rm pm},T}$ is Binney & Mamon (1982, MNRAS 200, 361).
The differentiable `project_dispersion` forward model is documented on the
[velocity-DF kinematics](../../10-theory/velocity-dfs/rotation-anisotropy.md) pages.
