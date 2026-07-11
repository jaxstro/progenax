---
title: OED for concentration — where to spend telescope time on W₀ (Stage 3)
subtitle: The optimiser sends the proper motions to the CORE to measure concentration — the mirror image of the anisotropy design
description: "A pre-data Bayesian optimal-experimental-design demo that targets a cluster's CONCENTRATION W0 with the same additive-Fisher machinery as the anisotropy (Stage 1) and dynamical-mass (Stage 2) designs. Given a fixed star budget allocated across (projected radius x {RV, PM_R, PM_T}), the c-optimal design that minimises the marginal variance of W0 puts ~71% of its stars in the CORE -- the mirror image of the anisotropy design, which wanted ~99% in the outskirts. Pre-registered hypothesis H1 (W0 differs from r_a) is CONFIRMED on the radial axis and REFUTED on the channel axis (W0 is strongly PM-dominated, a 2-for-1 PM efficiency at the assumed RV/PM error parity -- a wrong sub-prediction reported as a finding). The King c-optimal design reaches equal precision on W0 with ~3.4x fewer stars (sigma(lnW0) 0.104 -> 0.056, a 1.85x gain); Michie 0.095 -> 0.057 (1.67x). The W0<->r_a Fisher correlation is ~-0.02 (King) / -0.12 (Michie), so concentration is cleanly separable from anisotropy. Closes the open loop on the W0-differentiability work (PCHIP interpolation of the equilibrium-solver tables and a df_moment W0 lock) by exercising it through an actual Fisher/OED inference. Validated by AD-vs-FD on the W0 Jacobian and a real-star King calibration (realized/Fisher ratio 0.976). This is the OED tooling, which is planned for a separate package and is not part of v0.1.0: it ships as scripts plus this page."
---

# OED for concentration — where to spend telescope time on $W_0$ (Stage 3)

:::{warning} Deprecation — this OED demo is migrating to informax
The OED tooling has been ported to the dedicated inference-design package
**informax** (developed alongside progenax). This page and its
`scripts/` harness are retained temporarily while the port is being
verified, and will be removed from progenax afterwards. The demo's
dedicated test suite was retired in 2026-07 (the release gate no longer
exercises it), so treat the scripts here as frozen reference copies.
:::


You have a cluster and a star budget, and you want to know how **centrally concentrated**
it is. The [anisotropy design](anisotropy.md) asked *where on the sky* and *in which
channel* to measure to pin the orbital anisotropy $r_a$; the
[dynamical-mass design](dynamical-mass.md) asked *how deep* to survey to weigh the
cluster. This page asks the same kind of question for a third target — the cluster's
**concentration** $W_0$ — and the optimiser returns an answer that is the *mirror image*
of the anisotropy result. Anisotropy wanted the **outskirts**; concentration wants the
**core**. Same machinery, opposite strategy, and the contrast is the lesson.

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

- **Already comfortable with Fisher OED?** Skip to [the physical question](#sec-w0-physics)
  for the concentration science, or jump straight to [the headline figure](#fig-oedc-headline)
  and [Quantitative results](#sec-w0-results).
- **Want the machinery first?** Read [the OED formalism page](background.md), then come
  back here.
- **Why does this demo exist at all?** It closes the loop on the W₀-differentiability
  work — see [why this arc exists](#sec-w0-why).
:::

(sec-w0-why)=
## Why this arc exists: closing the W₀-differentiability loop

A short piece of history makes the rest of the page land. The forward models that turn a
cluster's structure into observable dispersions — the Osipkov–Merritt Jeans projection
`project_dispersion` and the exact Michie second moment `df_moment_dispersion` — were made
**differentiable in the concentration $W_0$** by two pieces of internal work: a $C^1$
PCHIP interpolation of the King/Michie equilibrium-solver tables and a $W_0$
lock on the `df_moment` path. Those changes made $\partial\sigma/\partial W_0$
*exist and be smooth*.

But they were validated **only at the gradient-audit level** — an automatic-versus-finite-difference
check that the derivative is numerically correct (98 entry points, 0 hazards). They were
*never* exercised through an actual Fisher / OED inference that **treats $W_0$ as a
parameter**. The stated goal of that whole line of work — "$W_0$-differentiable $\sigma$
for OED/Fisher" — had an open loop. This demo closes it: it builds the design Fisher with
$W_0$ in the parameter vector, optimises an observing strategy for it, and confirms the
forecast against real mock catalogues. It is the end-to-end use of the capability those
ADRs promised.

(sec-w0-physics)=
## The physical question: how do you measure concentration?

Our target is the **concentration** of a star cluster — how steeply its density climbs
from the tidal boundary to the centre. For the King (1966) lowered-isothermal family the
concentration is governed by a single dimensionless number, the **central potential depth**
$W_0$ (documented on the [King profile theory page](../../10-theory/spatial-profiles/king.md)):
a small $W_0$ is a diffuse, weakly-bound cluster; a large $W_0$ is a deep, centrally-peaked
one with an extended outer envelope. $W_0$ sets the King concentration
$c=\log_{10}(r_t/r_c)$ — the ratio of tidal to core radius — so measuring how concentrated
a cluster *is* means measuring $W_0$.

Why does this matter physically? Concentration is a dynamical clock and a structural
fingerprint: it tracks core collapse, two-body relaxation, and the cluster's tidal history,
and it is the first parameter any structural model of a globular cluster or a dwarf must
pin. It is usually read off the **surface-density / star-count profile** — that is the
[count-channel recovery demo (B11)](../king-concentration.md). Here we ask the
complementary question: what does the *kinematic* dataset say, and **where in it does the
concentration information live**?

### Why concentration is a core probe — the physical intuition

The key fact, which the optimiser is about to rediscover unaided: **concentration is an
isotropic, central property.** Changing $W_0$ rescales the depth of the potential well, and
its strongest observable fingerprint is the **velocity dispersion contrast between the deep
core and the cold outskirts**. A more concentrated cluster has a hotter, more sharply
peaked $\sigma(r)$ in the core, falling to a colder envelope; a diffuse one is flatter. The
*core* is where the dispersion is highest and the projected density is greatest, so it is
where a fixed number of stars pins that contrast most tightly. This is the opposite of
anisotropy, which is **zero in the core by construction** ($\beta_{\rm OM}(r)=r^2/(r^2+r_a^2)\to0$)
and only grows outward — so the anisotropy signal lives in the *outskirts*. Hold that
contrast in mind: concentration → core, anisotropy → outskirts. It is the physics behind
[the headline figure](#fig-oedc-headline).

## Inputs and assumptions

The design optimises over a **fixed budget** of $N_{\rm total}$ stars allocated across
$K=12$ projected-radius bins $\times$ 3 kinematic channels (36 cells). The science target
is $W_0$; the anisotropy radius $r_a$ is a co-determined kinematic parameter and the total
mass $M$ is a nuisance carried with a fractional prior. Everything is computed *pre-data*
— no mock catalogue enters the optimisation loop (a real-star calibration **ensemble**
afterwards is a gate, not part of the design).

```{list-table} Model inputs
:header-rows: 1
:label: tbl-w0-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $W_0$
  - King/Michie central potential depth — **the science target**; sets the concentration $c=\log_{10}(r_t/r_c)$. c-optimality minimises *its* marginal variance.
  - **target** ($W_0=6.0$; index 0 of $\boldsymbol\theta$)
* - $r_a$
  - Osipkov–Merritt anisotropy radius — a **co-determined kinematic parameter** (no external prior; constrained by the RV-vs-PM split alone). The *same* $r_a$ is the OM radius passed to `project_dispersion` (one source of truth).
  - free ($r_a=6.0\,r_c=2\,r_h$; index 1)
* - $M$
  - Total mass — the only **nuisance** with an external constraint (integrated light $\times$ $M/L$); carries a 30% fractional ($d\ln\theta$) prior.
  - nuisance ($M=10^5\,\Msun$; index 2) with $\sigma_{\rm prior}/M=0.3$
* - profile model
  - **King** (headline OM-King) and **Michie**, *both projected under Osipkov–Merritt* — they differ only in the density → $r_t(W_0)$ map (see [the modelling choice](#sec-w0-model)).
  - two models compared
* - 3 kinematic channels
  - RV ($\sigma_{\rm los}$), PM radial ($\sigma_{{\rm pm},R}$), PM tangential ($\sigma_{{\rm pm},T}$) — the **design lever**; each channel's information set by the B&M82 kernels {ref}`oed-bm82`.
  - fixed (the design allocates stars *among* them)
* - $d$, $\sigma_{\rm RV}$, $\sigma_{\rm PM}$
  - Distance converts the astrometric error to velocity; per-star errors enter the per-datum variance $\delta\sigma^2=(\sigma^2+\epsilon^2)/(2n)$. At $d=4$ kpc the two channels are at **deliberate error parity** — neither trivially dominates *per component*.
  - fixed ($d=4$ kpc; $\sigma_{\rm RV}=1.0$ km/s; $\sigma_{\rm PM}=0.05$ mas/yr $\to 4.74\,\sigma_{\rm PM}\,d\approx0.95$ km/s)
* - completeness $c(R)$
  - **Fixed realism**, identical across channels: a smooth logistic faint-end roll-off ($\approx1$ in the core, $<1$ outside, turnover $\sim6\,r_c$). Illustrative, *not* a real survey curve, and *not* a design knob here — promoting depth to a knob is [the dynamical-mass example](dynamical-mass.md).
  - fixed (folds into the per-star blocks)
* - $K$ bins, budget, optimiser
  - $K=12$ log-spaced bin centres $R_k\in[0.3,12]\,r_c$ (every bin inside both models' $r_t$, so all are bound); budget $N_{\rm total}=400$ (the King calibration operating point); multi-start Adam over the softmax simplex.
  - numerical choices
* - units, $G$
  - `STELLAR` ($\Msun$, pc, Myr) with $r_c\equiv1$ as the length unit; errors converted to pc/Myr explicitly.
  - known / fixed
```

The Fisher is built in the [dimensionless $\ln\theta$ metric](background.md#sec-oed-dimensionless)
{ref}`imp-oed-dimensionless`, so $\sigma(\ln W_0)$ is a *fractional* precision and the
nuisance prior $\mathtt{PRIOR\_DIAG}=[0,\,0,\,1/0.3^2]$ is fractional too — **none on the
target $W_0$, none on $r_a$, only on $M$.** That the data alone (three velocity channels
$\times$ 12 radial bins) keep the $(W_0, r_a)$ 2-block positive-definite — so no prior on
$W_0$ or $r_a$ is needed for a well-posed inverse — is a measured property of this design,
not an assumption (it is checked across random designs in the test suite).

(sec-w0-model)=
## The modelling choice: OM-King and OM-on-Michie-density

Two facts about the forward model set up the whole demo, and stating them plainly is the
honest way to read the result:

1. **`project_dispersion` is Osipkov–Merritt only.** It imposes
   $\beta_{\rm OM}(r)=r^2/(r^2+r_a^2)$ {ref}`oed-beta` on *whatever density it is handed* —
   it does not read a profile's own intrinsic anisotropy. So both models here are projected
   under the *same* OM anisotropy law; they differ **only** in the density profile, and
   hence in the concentration-dependent map $r_t(W_0)$. King is isotropic in density (OM is
   layered on top); Michie carries its own native anisotropy in its *density shape*, but it
   is still projected under OM. The headline is **OM-King**; **Michie** is added to exercise
   the native $r_t(W_0)$ path explicitly and to test that the qualitative answer is
   robust to the density model.

2. **The calibration's sampler equals the Fisher model.** The real-star gate (below) must
   draw stars from the *exact* distribution `project_dispersion` projects — "that density
   under OM" — or it would validate a mismatched model. So the OM particle sampler is
   assembled to be self-consistent with the Fisher forward model:
   for **King** via the Engine B `MultiComponentCluster.from_density_profiles` path (a
   single-component King density in its own shared-$\Psi$ potential with an Eddington/OM DF,
   documented on the [Engine B validation page](../../50-validation/engine-b-eddington.md));
   for **Michie** via the generic Eddington inversion
   ($\rho_Q=(1+r^2/r_a^2)\rho$, Merritt 1985 augmented density) on the Michie density, since
   Engine B does not ingest `MichieProfile`. In both cases **sampler $\equiv$ Fisher-model**,
   which is what makes the calibration a real test rather than a tautology. (Deliberately we
   do *not* use `MichieVelocityDF`'s native $\beta$ — that would mismatch the OM projection
   and bias the gate.)

## The method, specialised: c-optimality on the additive Fisher

This example pours the [additive backbone](background.md#sec-oed-additive)
{ref}`oed-additive-fisher` straight into a **c-optimal** allocation, exactly as the
anisotropy design does — only the target index changes. The per-star blocks $M_{b,c}$ come
from one `jacrev` of `project_dispersion` at the truth (reverse-mode by policy — the
King/Michie equilibrium solvers hit a `custom_vjp` ODE with no forward-mode rule); the
design enters only through the softmax weights {ref}`oed-design`, multiplied by the fixed
completeness $c(R_b)$. The headline criterion minimises the $W_0$ variance with $r_a$ and
$M$ profiled out:

```{math}
:label: w0-c-criterion
c\!:\quad \min_{\mathbf z}\;(F^{-1})_{W_0 W_0},
\qquad F(\mathbf z)=\sum_{b,c} n_{b,c}\,c(R_b)\,M_{b,c}.
```

We also run **D** ($\max\log\det F$) and **A** ($\min\mathrm{tr}\,F^{-1}$) as the contrast
— see [why c and D *should* disagree](background.md#sec-oed-criteria). Because the whole
Fisher is [additive and design-linear](background.md#sec-oed-additive), the expensive
projection is **never** re-differentiated inside the optimiser loop: one Jacobian, then
$F=\sum n\,M\to$ invert a $3\times3\to$ read one element.

## Validating a pre-data calculation: two gates

A Fisher forecast is a *promise* — "if you observe this way, your error bar will be this
large" — and a promise made before any data is only trustworthy if you check it, because
the Cramér–Rao bound {ref}`oed-cramer-rao` is a **local, Gaussian** approximation. This demo
checks it twice.

### Gate 1 — the gradient is correct (AD-vs-FD on $\partial\sigma/\partial\ln W_0$)

The load-bearing differentiability claim is the **$W_0$ column of the Jacobian**. We verify
it against finite differences at every radial bin. For **OM-King** the automatic and
finite-difference gradients agree to better than $10^{-3}$ across all bins. For **OM-Michie**
the inner bins ($R\lesssim r_a$) agree to the same $10^{-3}$ with a fixed step, but at
mid-to-outer radii the proximity of Michie's $r_t(W_0)$ truncation makes a *fixed-step*
finite difference an unreliable proxy — the function has high curvature there (the
truncation-curvature signature). So those bins are gated by **Richardson** finite differences instead: we confirm
that as the step $h\downarrow$, the finite difference **converges toward** the automatic
gradient. It does, to $\sim10^{-6}$, at every bin — the automatic gradient is correct
everywhere; the mid-radius $>10^{-3}$ values are pure $O(h^2\sigma''')$ truncation, not a
code defect.

### Gate 2 — the forecast is calibrated (real-star King calibration)

A Fisher forecast must predict the **realized scatter** of the estimator. We close the loop
with a real-star ensemble: for the **King** model we draw independent mock catalogues from
the OM sampler (which equals the Fisher forward model), project each to the sky, bin it by
projected radius, subsample the design counts, add per-star measurement error, and fit
$\hat W_0$ by maximum *a posteriori* in the $\ln\theta$ Gauss–Newton metric (a
Levenberg–Marquardt MAP fit, started at the truth). The realised fractional scatter should
match the Fisher prediction,

```{math}
:label: w0-calibration
\mathrm{Var}(\ln\hat W_0) \;\approx\; (F^{-1})_{W_0 W_0},
```

both sides being *fractional* variances in the $\ln\theta$ metric. The tolerance is not a
free knob: the Monte-Carlo error on a variance estimated from $n_{\rm draws}$ draws is
$\approx\sqrt{2/n_{\rm draws}}$, so the gate is a principled $2\sigma$ band. The realised /
Fisher variance ratio for King is **0.976** — comfortably inside the band, with no
significant bias. The promise holds.

:::{caution} **Two honest caveats — read these before quoting any number**
This demo is deliberately scoped, and two boundaries are load-bearing:

1. **It validates the Jeans (`project_dispersion`) $W_0$ path, *not* `df_moment_dispersion`.**
   The OED Fisher rides the Osipkov–Merritt Jeans projection; the *separate* exact-second-moment
   path `df_moment_dispersion` (also $W_0$-gradient-audited) is **not** exercised
   here. The closed loop is the Jeans path.
2. **It is not part of v0.1.0.** This is a scripts + demo-page
   validation — there is **no** `src/progenax/` API surface. The OED tooling is planned
   for a separate package; the whole arc ships as the optimal-design demo you are reading,
   not as released core.

The **Michie** calibration Monte-Carlo is also *not* run, and that is honest scope rather
than an omission: its reverse-mode-through-ODE MAP fit batches to $\sim$28 GB and OOM-crashes
the host, and it would add **no new anisotropy physics over King** (both are projected under
the same OM law). Michie's model is instead validated by the cheaper sampler-match, forward,
and gradient tests — the King calibration is the headline trust anchor.
:::

(sec-oedc-headline)=
## The headline result: concentration wants the core

The pre-registered hypothesis (**H1**) was: *$W_0$'s optimal allocation differs from
$r_a$'s — more core/intermediate weight, more channel-balanced, with only a modest outward
pull.* The result splits cleanly into a **confirmed** radial prediction and a **refuted**
channel sub-prediction. Reporting both honestly — the hit *and* the miss — is the point;
a wrong pre-registered sub-prediction is a genuine finding, not a failure.

### RADIAL: confirmed — the mirror image of the anisotropy design

The $W_0$ c-optimal design puts **$\sim71\%$ of its stars in the core** half of the radial
range (King; $\sim71\%$ Michie). The Stage-1 anisotropy design put **$\sim99\%$ in the
outskirts.** They are mirror images, and the physical reason is exactly the intuition above:
concentration is an isotropic, central property imprinted where the dispersion is highest
and the density greatest (the core), while anisotropy lives in $\beta(r)$'s outward growth
(the outskirts). The optimiser was told none of this — it discovered where the
concentration information lives.

:::{figure} figures/demo_oedc_headline_contrast.png
:label: fig-oedc-headline
:width: 95%

**Concentration wants the core; anisotropy wanted the outskirts — the same machinery,
opposite strategies.** Left: the $W_0$ c-optimal radial allocation (King), stacked by
channel (RV, PM$_R$, PM$_T$), as a fraction of the budget — **core-heavy** ($\sim71\%$ in
the inner half). Right: the Stage-1 $r_a$ c-optimal allocation — **outskirts-heavy**
($\sim99\%$ in the outer half). Each panel is normalised to its own budget so the *shape*
is compared (the $W_0$ grid is in $r_c$; the $r_a$ grid is in $r_h$). The mirror image is
the pre-registered radial prediction H1, confirmed.
:::

### CHANNEL: refuted — concentration is strongly PM-dominated

H1 also predicted that $W_0$ would be *more channel-balanced* than $r_a$ (RV pulling more
of its weight). It is not. The $W_0$ c-optimal design is **strongly proper-motion
dominated** — RV carries only $\sim1\%$ of the King budget. The reason is structural and
the same one the anisotropy page documents: at the assumed RV/PM error parity, a
proper-motion star delivers **two** information-bearing components (PM$_R$ *and* PM$_T$) for
one measurement, a **2-for-1 efficiency** that the optimiser exploits almost everywhere.
The error parity is *per component*, so PM wins on count. This sub-prediction was wrong, and
we say so: it is a null result reported with integrity, and it sharpens the real lesson —
the *radial* contrast (core vs outskirts) is the robust discriminator between the two
targets, not the channel mix.

:::{figure} figures/demo_oedc_alloc_king.png
:label: fig-oedc-alloc-king
:width: 80%

**The King $W_0$ c-optimal allocation, resolved in both axes.** Effective star count
$n_{\rm eff}$ per (channel $\times$ projected-radius) cell: rows are the three velocity
channels (RV, PM$_R$, PM$_T$), columns the $K=12$ log-spaced on-sky radii. The budget piles
into the **core bins** (left columns) and into the two **proper-motion rows** — the
two-axis view of "concentration wants PMs in the core." The RV row is nearly empty,
the 2-for-1 PM efficiency made visible.
:::

:::{figure} figures/demo_oedc_alloc_michie.png
:label: fig-oedc-alloc-michie
:width: 80%

**The Michie $W_0$ c-optimal allocation — the same qualitative answer.** As
{numref}`fig-oedc-alloc-king`, for the Michie density. The core-plus-PM structure is robust
to the density model; Michie wants modestly more RV and a slightly more outward balance
(see [King vs Michie](#sec-w0-king-michie)), but the headline strategy is unchanged.
:::

## The precision gain and the criterion contrast

How much does designing for concentration buy you? At fixed budget the King c-optimal design
reaches a fractional precision $\sigma(\ln W_0)=0.056$, versus $0.104$ for a uniform design
— a **$1.85\times$ gain in precision**, equivalent to reaching the same error bar with
$\approx3.4\times$ fewer stars (the precision gain squared, since variance scales as $1/N$
when the prior is held fixed). The Michie design improves $0.095\to0.057$, a $1.67\times$
gain. As in the anisotropy example, the c-, D-, and A-optimal designs genuinely differ —
the c-design, which only cares about $W_0$, beats the others on *its* objective and would
waste stars if it chased the whole ellipsoid.

:::{figure} figures/demo_oedc_gain_cda.png
:label: fig-oedc-gain
:width: 100%

**The precision gain and the c/D/A criterion contrast, King vs Michie.** Left: the
fractional precision $\sigma(\ln W_0)$ for the uniform vs the c-optimal design (the gain
factor is annotated on each c-optimal bar) — $1.85\times$ (King), $1.67\times$ (Michie).
Right: each of the c/D/A optima as a ratio to its own uniform-design value (so all three are
dimensionless "fraction of the uniform criterion"; $<1$ is better), King vs Michie. The
three alphabet-optimality designs are genuinely different objectives on the same
$F=\sum n\,M$.
:::

## The broken degeneracy: concentration is separable from anisotropy

The deepest science payoff is that this design **breaks the $W_0\leftrightarrow r_a$
degeneracy** — it measures concentration and anisotropy *independently*. The Fisher
correlation at the c-optimal design,

```{math}
:label: w0-rho
\rho(W_0, r_a) \;=\; \frac{(F^{-1})_{W_0 r_a}}{\sqrt{(F^{-1})_{W_0 W_0}\,(F^{-1})_{r_a r_a}}},
```

is the covariance off-diagonal normalised to $[-1,1]$: $|\rho|\to1$ would mean the two are
nearly degenerate (you cannot tell them apart), $|\rho|\to0$ that the design measures them
cleanly. We find $\rho(W_0, r_a)\approx-0.02$ for King and $-0.12$ for Michie — both far
from $\pm1$. Concentration is **cleanly separable** from anisotropy in this kinematic design
space, which is exactly what makes a $W_0$-targeted observing strategy worth computing:
the core-PM stars that pin $W_0$ are *not* the same stars that pin $r_a$, and the design
knows it.

:::{figure} figures/demo_oedc_degeneracy.png
:label: fig-oedc-degeneracy
:width: 65%

**Concentration is separable from anisotropy.** The Fisher correlation $\rho(W_0, r_a)$ at
the c-optimal design, King vs Michie. Both bars sit near zero ($-0.02$ King, $-0.12$
Michie), far from the shaded $|\rho|>0.9$ "near-degenerate" band: the design pins $W_0$
almost independently of $r_a$. (A small negative $\rho$ is the residual, broken degeneracy
— not a problem, but the quantitative statement of how well it is broken.)
:::

(sec-w0-king-michie)=
## King vs Michie: is the answer robust to the density model?

Yes, qualitatively. The headline — **core plus proper motions** — is the same for both
density models, and the precision gains ($1.85\times$ vs $1.67\times$) and core fractions
($\sim71\%$ vs $\sim71\%$) are close. The differences are second-order and model-dependent:
Michie wants modestly more of the RV channel ($\sim16\%$ vs $\sim1\%$) and a slightly more
outward balance, because its native-anisotropy density shape moves the $\sigma(r)$ contrast
a little. The robust, model-independent statement is the one to take away: *measure
concentration in the core, with proper motions.* The RV fraction and the fine radial balance
are where the model choice shows up.

(sec-w0-results)=
## Quantitative results

All numbers are from the gated CLI's run-record ($N_{\rm total}=400$, $K=12$ bins, the King
calibration operating point):

```{list-table} OED concentration results
:header-rows: 1
:label: tbl-w0-results

* - Quantity
  - King
  - Michie
* - $\sigma(\ln W_0)$, uniform $\to$ c-optimal (fixed $N$)
  - $\mathbf{0.104 \to 0.056}$
  - $0.095 \to 0.057$
* - Precision gain (c-design vs uniform)
  - $\mathbf{1.85\times}$ ($\approx3.4\times$ fewer stars)
  - $1.67\times$
* - Core fraction (inner-half share, c-design)
  - $\mathbf{0.71}$
  - $0.71$
* - Channel split RV / PM (c-design)
  - $0.01 / 0.99$ (PM-dominated)
  - $0.16 / 0.84$
* - Fisher correlation $\rho(W_0, r_a)$ at c-optimal
  - $-0.02$ (separable)
  - $-0.12$ (separable)
* - **Contrast:** Stage-1 $r_a$ c-optimal radial split
  - $\sim99\%$ outskirts
  - —
* - Real-star calibration (realized $\mathrm{Var}(\ln\hat W_0)$ / Fisher)
  - $\mathbf{0.976}$ (within band)
  - not run (OOM; see caveat)
```

The precision gain is **exact at fixed $N$** (the fixed nuisance prior cancels in the
$c_{\rm uniform}/c_{\rm designed}$ ratio, because $c\propto1/N$ when the prior is held
fixed); the "$\approx3.4\times$ fewer stars" gloss is the same physics read as a star factor
($\text{gain}^2$).

## What the optimum means: science implications

The result is the optimiser **discovering where concentration information lives**, and it
generalises well beyond this mock.

**1. The observing strategy is the *opposite* of an anisotropy campaign.** To measure
concentration, put the proper-motion budget in the **core**; to measure anisotropy, put it
in the **outskirts**. Same instrument, same channels, opposite radial allocation — because
concentration is a central, isotropic property and anisotropy is an outer, radial one. That
is directly actionable: a Gaia-PM + spectroscopic campaign optimised for $W_0$ should *not*
reuse an anisotropy campaign's footprint, and OED tells you so quantitatively.

**2. Concentration and anisotropy are independently measurable.** The near-zero
$\rho(W_0, r_a)$ {eq}`w0-rho` means the kinematic dataset can pin *both* without one
contaminating the other — the core-PM stars that constrain $W_0$ are not the outskirts-PM
stars that constrain $r_a$. A joint structural+orbital fit is well-posed in this design
space, and the design is what makes it so.

**3. Differentiable OED is a general tool.** This is the *third* distinct science target
($r_a$, $M$, now $W_0$) run through the *same* additive-Fisher machinery, each with a
different physical answer the optimiser found unaided. Because `progenax`'s forward models
are differentiable, $\partial(\text{information})/\partial(\text{observing strategy})$ is
computable for essentially any cluster-science parameter — which is the throughline of
[the whole OED section](index.md).

**4. The honest caveat is itself a research direction.** The optimum is optimal *for the
assumed OM-King/Michie model*, and it rides the Jeans path, not the exact-moment one. That
model-dependence is the central limitation of all OED, and it points at the same valuable
extensions the sibling pages flag: **robust**, **Bayesian**, and **model-discriminating**
(T-optimal) design — for which `progenax` is well-placed because it ships *more than one*
differentiable forward model for the same observable.

## Current scope and planned extensions

```{warning}
This is a **pre-data, single-shot OED on a clean self-consistent mock**, headlining
concentration. Its boundaries, stated honestly:

- **Jeans path only — not `df_moment_dispersion`.** The closed loop validates the
  Osipkov–Merritt `project_dispersion` $W_0$ path; the exact-second-moment path is
  grad-audited but **not** exercised here ([caveat 1](#sec-oedc-headline)).
- **Not part of v0.1.0.** Scripts + this page, **no** released-core API
  surface; the OED tooling is planned for a separate package (caveat 2).
- **The RV channel is under-utilised *at the chosen error parity*.** PM delivers two
  components per star at the matched per-component error, so the optimiser prefers it; the
  channel-balance sub-prediction was refuted on exactly this point. The **core** result (the
  radial trend) is robust; the RV/PM mix is parity-dependent.
- **OM-King / OM-on-Michie-density equilibrium mock only.** Truth and forward model share the
  same generative family (sampler $\equiv$ Fisher-model); no model misspecification, no real
  catalogue, no cross-channel systematics.
- **Michie calibration MC not run** — its reverse-mode-through-ODE fit is $\sim$28 GB / OOM,
  and adds no new anisotropy physics over King; Michie is validated by the sampler-match,
  forward, and gradient tests instead.
- **Single line-of-sight projection.** LOS $=\hat z$; no flattening, no rotation, no
  inclination.
- **The Fisher is a local, Gaussian (Cramér–Rao) approximation.** It is exact only in the
  high-information limit. **The King calibration is precisely the check** that it predicts
  the realized scatter — and it does, to a variance ratio of $0.976$
  ({numref}`fig-oedc-degeneracy` is the separability payoff; the calibration is the trust
  anchor).
- **Static single-shot — no sequential OED.** The design is computed once; there is no
  online/adaptive re-design as data arrive.
```

:::{note} What we just learned
The concentration $W_0$ is measured in the **core**, with **proper motions** — the mirror
image of the [anisotropy design](anisotropy.md), which wanted PMs in the *outskirts*.
Because $W_0$ is an isotropic, central property, a c-optimal design puts $\sim71\%$ of its
stars in the core and reaches the same fractional precision $\sigma(\ln W_0)$ with
$\approx3.4\times$ fewer stars ($0.104\to0.056$, a $1.85\times$ gain; Michie $1.67\times$).
The pre-registered hypothesis was **confirmed on the radial axis** (core, not outskirts) and
**refuted on the channel axis** (strongly PM-dominated, not balanced — a 2-for-1 PM
efficiency, reported as a finding). Concentration is **cleanly separable** from anisotropy
($\rho(W_0,r_a)\approx-0.02$). The forecast is trustworthy — a real-star King calibration
gives a realized/Fisher variance ratio of $0.976$, and the $W_0$ gradient passes AD-vs-FD at
every bin. This closes the open loop on the $W_0$-differentiability work (the $C^1$ PCHIP
interpolation and the df_moment $W_0$ lock) by exercising it through a real Fisher/OED
inference — the **Jeans path**.
The same machinery with a *channel* knob targets [anisotropy](anisotropy.md) and with a
*depth* knob targets [dynamical mass](dynamical-mass.md).
:::

## How to run

```bash
# the cheap design + figures (one jacrev per model + 3x3 linalg; ~1 min)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_concentration.py

# regenerate into the docs figure directory
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_concentration.py \
    --outdir docs/website/60-science-demos/optimal-design/figures
```

The CLI computes only the cheap parts — one `jacrev` per model at the truth, the c/D/A
$3\times3$-linear-algebra optimisation, the H1 radial/channel split contrasted with Stage-1,
and the five figures — and writes a JSON run-record alongside them. The expensive real-star
King calibration is the env-gated `@slow` test
`test_W0_fisher_calibration_matches_realized_scatter`; the King ratio of **0.976** quoted
here is cited from that gate, not re-run in the CLI (the Michie MAP-MC is intentionally never
run — see the caveat).

## References

The shared Fisher / Cramér–Rao / projection theory and its references are on
[the OED formalism page](background.md). The King (1966) lowered-isothermal model and its
$W_0$ parameter are documented on the
[King profile theory page](../../10-theory/spatial-profiles/king.md); the Michie–King
anisotropic model on the
[Michie–King theory page](../../10-theory/velocity-dfs/michie-king.md); the Engine B
Eddington/OM sampler on the
[Engine B validation page](../../50-validation/engine-b-eddington.md). The Osipkov–Merritt
anisotropy law is {cite:t}`Merritt1985`; the line-of-sight projection into $\sigma_{\rm los}$,
$\sigma_{{\rm pm},R}$, $\sigma_{{\rm pm},T}$ is {cite:t}`BinneyMamon1982`. The
*count-channel* concentration recovery — the complementary way to measure $W_0$, from star
counts rather than kinematics — is [the King-concentration demo (B11)](../king-concentration.md);
the companion OED designs are [anisotropy](anisotropy.md) (where) and
[dynamical mass](dynamical-mass.md) (how deep).
