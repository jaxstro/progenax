---
title: OED for robustness — when binaries lie to your mass estimate (Stage 4)
subtitle: The first OED demo whose headline is a BIAS, not a recovery — a binary-blind survey design over-weighs a cluster by 184% with 41× false confidence, and a binary-aware design removes it
description: "A pre-data optimal-experimental-design demo that turns the OED machinery from 'recovers the obvious' into a referee-resistant BIAS result. Unresolved binaries (Moe & Di Stefano massive-primary P-q-e blend) inflate the line-of-sight velocity dispersion by a flat pedestal sigma_bin=9.73 km/s (sigma_bin/sigma_cluster=1.08 at the center of an EFF-OM young-massive cluster, M=4e5 Msun). H1 (the headline): a naive binary-blind c-optimal-for-M RV design, fit with a binary-free model, recovers M_hat/M=2.84 (+184% bias) while CLAIMING sigma(M)/M=4.5% -- a 41x false-confidence disaster, gated by a cross-model Monte Carlo. The fix (measure-and-marginalize): a binary-aware fit drops the bias from +184% to +5%, recovers f_bin=0.50+/-0.08 from radial leverage alone, and reports an honest sigma(M)/M=6.9%. H2: the binary-aware DESIGN is 1.33x tighter than the binary-blind one at the operating point (growing to ~6x as binaries dominate). H3: the binary-aware allocation reshuffles to break the M<->f_bin degeneracy. Maximin robustness is an honest near-null (~0.2% hedge -- sigma(M) is monotone in f_bin). A sigma_bin/sigma_cluster sweep shows bias +26%->+850% across system mass while the fix stays ~0. The OED tooling is planned for a separate package and is not part of v0.1.0: scripts plus this page."
---

# OED for robustness — when binaries lie to your mass estimate (Stage 4)

:::{note} Status — candidate to remain as progenax's basic OED example
The OED tooling is migrating to the dedicated inference-design package
**informax**; most OED demos here will be removed once the port is
verified. This binary-robustness demo (binary RVs vs. the inferred
cluster mass) is the candidate to stay as progenax's single basic OED
example. Its 2026-07 test trim kept the load-bearing physics pins (the
V_bin/sigma_cluster ratio gate, the sampler-vs-projection integrity
check, the binary pedestal, and a CLI smoke); the full inference /
design-machinery gates now live in informax.
:::


The three OED demos before this one each end with the optimiser *rediscovering* a piece
of physics: [anisotropy](anisotropy.md) wants the **outskirts**,
[concentration](concentration.md) wants the **core**, [dynamical mass](dynamical-mass.md)
wants an **interior depth**. Each is a clean result — and each is one a referee could wave
away as "Fisher-OED recovers the obvious." This page is different. Its headline is not a
recovery; it is a **bias**. We hand a careful, binary-*blind* observer the optimal survey
design for weighing a cluster, let the real cluster harbour the binaries that every real
cluster has, and watch the analysis over-estimate the mass by **184% while reporting a 4.5%
error bar** — a measurement that is *confidently, expensively wrong*. Then we show the
design that protects against it.

:::{note} **Shared theory lives once, on the formalism page**
The Fisher information / Cramér–Rao foundation {ref}`oed-fisher-gaussian`, the
[additive design-linear backbone](background.md#sec-oed-additive) {ref}`oed-additive-fisher`,
the [c/D/A criteria](background.md#sec-oed-criteria), the
[dimensionless $\ln\theta$ metric](background.md#sec-oed-dimensionless), and the
[Binney & Mamon (1982) projection geometry](background.md#sec-oed-projection) are all built
on [the OED formalism page](background.md). This page is the *application* — read the
formalism first if any of those terms are unfamiliar.
:::

:::{note} **Reading paths**
:class: dropdown

- **Already comfortable with Fisher OED?** Skip to [the physical question](#sec-bin-physics)
  for the binary-systematic science, or jump straight to [the headline figure](#fig-oedb-false-confidence)
  and [Quantitative results](#sec-bin-results).
- **Want the machinery first?** Read [the OED formalism page](background.md), then come back here.
- **Why does this demo exist at all?** It converts OED from "recovers the obvious" into a
  referee-resistant bias result — see [why this arc exists](#sec-bin-why).
:::

(sec-bin-why)=
## Why this arc exists: a bias a referee cannot dismiss

The sibling OED demos are strong *engineering* — one differentiable Fisher backbone,
re-pointed at three science targets — but each headline is either already-known physics or
contingent on an arbitrary RV/PM error split that flips with the budget. None is a result a
referee couldn't read as "of course the optimiser put the proper motions where the signal
is."

This demo monetises a capability **nobody else has wired to OED**: `progenax` carries a
faithful, differentiable Moe & Di Stefano (2017) binary-population engine, so we can ask the
question that actually bites in real cluster mass measurements — *what does an unmodelled
population systematic do to an "optimal" design, and what design defends against it?* The
answer is a **parameter bias larger than its own forecast uncertainty**, which is the one
kind of result a Fisher-OED demo cannot be accused of staging. It is the canonical binary
$M/L$ systematic of dwarf-galaxy and young-cluster dynamics, turned into a *design* problem.

(sec-bin-physics)=
## The physical question: why binaries masquerade as mass

You weigh a star cluster with the **virial theorem**: a hotter cluster — larger
line-of-sight velocity dispersion $\sigma_{\rm los}$ — is a heavier one, $M \propto \sigma^2 R / G$.
That logic has a silent assumption: that every velocity you measure is a *single star
orbiting the cluster*. It is not. A large fraction of massive stars live in **binaries**,
and in a single-epoch radial-velocity survey you cannot resolve the binary — you measure the
star's *orbital* velocity around its companion folded into its cluster velocity. Those
orbital motions are fast (tens of km/s for short-period massive binaries), so they **inflate
the measured dispersion** above the true cluster value.

The inflation is not noise — noise averages out. It is a **systematic offset** in the second
moment. If the cluster's true dispersion is $\sigma_{\rm cluster}$ and a fraction $f_{\rm bin}$
of the stars carry a binary velocity drawn from a population with variance $V_{\rm bin}$, the
*observed* dispersion-squared is

```{math}
:label: bin-pedestal
\sigma_{\rm obs}^2(R) \;=\; \underbrace{\sigma_{\rm cluster}^2(R)}_{\text{falls with radius}}
\;+\; \underbrace{f_{\rm bin}\,V_{\rm bin}}_{\text{flat pedestal}} \;+\; \varepsilon_{\rm RV}^2,
```

a **flat pedestal** $f_{\rm bin}V_{\rm bin}$ added on top of the real cluster heat at *every*
radius. Feed $\sigma_{\rm obs}$ to a virial analysis that does not know about the pedestal,
and the analysis reads the extra dispersion as *extra mass*. For our young-massive-cluster
fiducial the pedestal is $\sqrt{f_{\rm bin}V_{\rm bin}}=6.9$ km/s sitting under a central
cluster dispersion of only $9.0$ km/s — and the cluster term *falls* to $0.4$ km/s in the
outskirts, while the pedestal does not budge. **That is the whole story:** in the cold
outskirts the binary pedestal *dominates* the signal.

### The intuition the optimiser is about to fall for — and then defeat

Hold one fact in mind: the cluster mass $M$ scales the dispersion *amplitude* and shows up at
*every* radius; the binary fraction $f_{\rm bin}$ adds a *flat offset*. The only thing that
tells them apart is **radial leverage** — the cluster dispersion has a *shape* $\sigma(R)$
that falls from core to edge, while the binary pedestal is flat. A design that ignores
binaries chases the loudest signal into the cold outskirts and parks its budget exactly where
the pedestal is largest relative to the cluster — and so confuses pedestal for mass most
badly. A design that *models* binaries instead spends its stars on the **core↔outskirts
contrast** that breaks the $M\leftrightarrow f_{\rm bin}$ degeneracy. Same instrument, same
budget, opposite outcome.

(sec-bin-inputs)=
## Inputs and assumptions

The design optimises a **fixed budget** of $N_{\rm total}$ radial-velocity measurements
allocated across $K=12$ log-spaced projected-radius bins — **RV only**, single-epoch. The
science target is the dynamical mass $M$; the EFF shape parameters $(\gamma, a)$ are
photometrically pinned nuisances, and the anisotropy radius $r_a$ and binary fraction
$f_{\rm bin}$ are kinematic nuisances the radial allocation must disentangle. Everything is
computed *pre-data* — the cross-model Monte Carlo afterwards is a calibration **gate**, not
part of the design loop.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-bin-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $M$
  - Total dynamical mass — **the science target**; scales the cluster dispersion amplitude. c-optimality minimises *its* marginal variance.
  - **target** ($M=4\times10^5\,\Msun$; index 0 of $\boldsymbol\theta$, no prior)
* - $r_a$
  - Osipkov–Merritt anisotropy radius — a kinematic **nuisance** with a weak (50% fractional) prior; the RV-only $M\!\leftrightarrow\! r_a$ degeneracy is regularised by it.
  - nuisance ($r_a=3$ pc; index 1; $\sigma_{\rm prior}/r_a=0.5$)
* - $\gamma$
  - EFF outer slope — the **concentration** shape knob. Photometrically pinned (measured from the surface-brightness profile).
  - nuisance ($\gamma=2.7$; index 2; tight 10% prior)
* - $a$
  - EFF **scale radius** — parameterised directly (EFF is natively $(a,\gamma,r_t)$ with no closed-form $r_h$ inversion; see [the modelling choice](#sec-bin-model)). Photometrically pinned.
  - nuisance ($a=1$ pc; index 3; tight 10% prior)
* - $f_{\rm bin}$
  - Binary fraction — the misspecification axis. In the binary-*aware* model it is a **free nuisance** with a *weak* (50%) prior, recovered from radial leverage; in the binary-*blind* model it is silently fixed at 0.
  - truth $0.5$ (index 4; weak prior or absent)
* - $V_{\rm bin}$
  - Variance of the single-epoch binary velocity blend — a **build-once population scalar** from the Moe & Di Stefano $P$–$q$–$e$ engine over a Maschberger massive-primary IMF with ZAMS flux weights.
  - fixed ($V_{\rm bin}=94.7$ (km/s)$^2$ $\to\sigma_{\rm bin}=9.73$ km/s)
* - density model
  - **EFF** (Elson–Fall–Freeman), projected under Osipkov–Merritt. Analytic density $\Rightarrow$ **no ODE / no OOM** (see [the modelling choice](#sec-bin-model)).
  - held fixed (the misspecification CONTROL)
* - kinematic channel
  - **RV only** ($\sigma_{\rm los}$) — no proper motions, no channel split (the binary $M/L$ systematic lives in the RV second moment; the dwarf/cluster $M/L$ regime).
  - fixed (the design allocates stars among *radii*)
* - $\varepsilon_{\rm RV}$
  - Per-star RV error, entering the per-datum variance $\delta\sigma^2=(\sigma^2+\varepsilon^2)/(2n)$.
  - fixed ($\varepsilon_{\rm RV}=1.0$ km/s)
* - $K$ bins, budget, optimiser
  - $K=12$ log-spaced bin centres $R_k\in[0.2\,a,\,0.95\,r_t]=[0.2,17.1]$ pc; $r_t=18$ pc; multi-start Adam over the softmax simplex.
  - numerical choices
* - units, $G$
  - `STELLAR` ($\Msun$, pc, Myr); errors converted to km/s explicitly.
  - known / fixed
```

The Fisher is built in the [dimensionless $\ln\theta$ metric](background.md#sec-oed-dimensionless)
{ref}`imp-oed-dimensionless`, so $\sigma(\ln M)$ is a *fractional* precision. The pinned
scales are the load-bearing ones: $\sigma_{\rm cluster,central}=8.98$ km/s and
$\sigma_{\rm bin}=9.73$ km/s, a contamination ratio $\sigma_{\rm bin}/\sigma_{\rm cluster}=1.08$
at the centre — *rising to $\sim24$ in the outskirts*, because the cluster term collapses
while the pedestal stays flat. That the binaries rival the cluster heat at this operating
point is **pinned before any inference**, exactly so that H1 below has a chance to bite.

(sec-bin-model)=
## The modelling choice: EFF-OM, RV-only, and a held-fixed cluster

Two facts about the forward model set up the whole demo, and stating them plainly is the
honest way to read the result.

1. **EFF density is analytic — no ODE, no OOM.** The Elson–Fall–Freeman profile has a
   closed-form density $\rho \propto (1+r^2/a^2)^{-\gamma/2}$, so `project_dispersion` does a
   plain quadrature with **no differential-equation solver in the tape**. This is what makes
   the cross-model Monte Carlo *runnable* — the King/Michie ODE solvers in the
   [concentration demo](concentration.md) forced its Michie calibration to OOM at $\sim$28 GB.
   The EFF slope $\gamma$ doubles as a concentration knob, so the bias vector spans
   $(M, r_a, \gamma, a, f_{\rm bin})$ with no ODE anywhere. We parameterise by the EFF
   *scale radius* $a$ directly rather than a derived half-mass radius, because EFF has no
   closed-form $r_h(a,\gamma,r_t)$ and inventing one would inject a spurious unpinned
   inversion.

2. **The cluster forward model is the held-fixed CONTROL.** This is a *misspecification*
   study, and a misspecification study is only clean if it changes **one thing**. So the
   EFF-OM cluster model — density shape, OM anisotropy law, RV projection — is identical
   between the design, the truth, and both fits. The *only* difference between the "disaster"
   and the "fix" is whether the analysis **models the binaries**. Every number on this page is
   the cost of that one omission, with the cluster physics deliberately frozen so it cannot be
   blamed.

## The method, specialised: c-optimality with a misspecified vs marginalized Fisher

This demo pours the [additive backbone](background.md#sec-oed-additive)
{ref}`oed-additive-fisher` into a **c-optimal-for-$M$** radial allocation, exactly as the
sibling designs do. One `jacrev` of the cluster term at the truth gives the per-bin Jacobian
$J=\partial\sigma_{\rm los}/\partial\ln\boldsymbol\theta$ (reverse-mode by policy); the design
enters only through the per-bin counts. The criterion minimises the $M$ variance with the
nuisances profiled out:

```{math}
:label: bin-c-criterion
c\!:\quad \min_{\mathbf z}\;(F^{-1})_{MM},
\qquad F(\mathbf z)=\sum_{b} n_{b}\,M_{b}+\mathrm{diag}(\text{priors}),
\qquad M_b=\frac{2\,J_b J_b^{\!\top}}{\sigma_{\rm obs}^2(R_b)+\varepsilon_{\rm RV}^2}.
```

The whole demo is the *contrast between two Fishers built on this one equation*:

- **Binary-blind** ($\boldsymbol\theta=(M,r_a,\gamma,a)$, no $f_{\rm bin}$): the observer who
  does not know binaries exist. Its $4\times4$ Fisher omits the $f_{\rm bin}$ column entirely.
- **Binary-aware / marginalized** ($\boldsymbol\theta=(M,r_a,\gamma,a,f_{\rm bin})$): adds the
  $\partial\sigma_{\rm los}/\partial\ln f_{\rm bin} = f_{\rm bin}V_{\rm bin}/(2\sigma_{\rm obs})$
  column (the [pluggable Fisher block](background.md#sec-oed-additive)) and profiles $f_{\rm bin}$
  out as a free nuisance.

Because the Fisher is additive and design-linear, the expensive projection is differentiated
**once**; the $f_{\rm bin}$ grid for the maximin criterion below only updates the
$\sigma_{\rm obs}(f_{\rm bin})$ denominator — no re-`jacrev`.

(sec-oedb-false-confidence)=
## The headline (H1): a confident, expensive, wrong mass

The pre-registered hypothesis (**H1**, [locked](#sec-bin-prereg) before the Monte Carlo ran)
was: *fitting the binary-blind model to binary-contaminated RV data biases $\hat M$ high, by
more than the design's own forecast $\sigma(M)$ — false confidence.* The rival H0 was that the
binary-blind fit would absorb the flat pedestal into its nuisances and stay unbiased. **H1 is
confirmed, decisively.**

The binary-blind c-optimal-for-$M$ design, fit with the binary-blind model on mock catalogues
that *do* contain Moe binaries, recovers

```{math}
:label: bin-h1
\frac{\hat M}{M} = 2.84 \quad(\text{bias } +184\%), \qquad
\text{forecast } \frac{\sigma(M)}{M}=4.5\%,
\qquad \frac{\text{bias}}{\text{forecast}} = 41\times.
```

The estimate is nearly **triple the truth**, and the design's own error bar — the precision it
*promised* — is **41 times smaller than the error it actually made**. This is the worst
failure mode in inference: not a wide error bar (which warns you), but a *tight* one around the
*wrong* answer.

The control is what makes it airtight. Run the *same* design and fit on mocks with
$f_{\rm bin}=0$ — where the fit model exactly matches the generative model — and the bias is
$-0.3\%$, well inside the forecast. The entire $+184\%$ is the binaries; nothing else moved.

:::{figure} figures/demo_oedb_false_confidence.png
:label: fig-oedb-false-confidence
:width: 80%

**False confidence: a binary-blind design weighs the cluster at $2.84\times$ its true mass
with a $4.5\%$ error bar.** The recovered $\hat M/M_{\rm true}$ for the binary-blind
c-optimal-for-$M$ design fit on binary-contaminated mocks (vermilion, $\approx2.84$), drawn
*with its own $\pm$forecast-$\sigma$ error bar* — visually a speck. The unbiased truth is the
dashed line at $1$; the $f_{\rm bin}=0$ baseline (blue square, mock $\equiv$ fit model)
recovers $\hat M/M\approx1$. The bias ($+184\%$) is annotated against the truth; the callout
states the headline — the claimed error bar is $41\times$ smaller than the bias. From the
env-gated cross-model Monte Carlo (48 draws).
:::

:::{figure} figures/demo_oedb_mechanism.png
:label: fig-oedb-mechanism
:width: 90%

**Why it happens: the design parks its budget in the contaminated outskirts.** The truth
cluster $\sigma_{\rm los}(R)$ (vermilion) falls steeply from $9.0$ km/s in the core to $0.40$
km/s at the edge — that fall *is* the mass signal. The binary-inflated observable
$\sqrt{\sigma_{\rm cluster}^2+f_{\rm bin}V_{\rm bin}}$ (green) sits on a near-flat pedestal at
the $\sqrt{f_{\rm bin}V_{\rm bin}}=6.9$ km/s floor (dotted). The binary-blind design's per-bin
allocation (blue bars, right axis) piles into the **outer** bins — exactly where the pedestal
dominates the signal — so the binary-blind fit reads the pedestal as extra mass. The design
optimised for $M$ under the wrong model finds the worst possible place to spend its stars.
:::

## The fix (measure-and-marginalize): model the binaries, recover them, lose the bias

The remedy is not to throw away the outskirts; it is to **model the pedestal**. Add the
$f_{\rm bin}$ parameter to the fit (and to the design Fisher), let the data determine it, and
the bias collapses:

- **Bias: $+184\% \to +5\%$.** The binary-aware fit on the *same* contaminated mocks recovers
  $\hat M$ within its honest forecast — the cross-model residual bias is consistent with zero
  ($+5\%$, inside $2\sigma$ of the marginalized forecast).
- **It recovers the binary fraction: $\hat f_{\rm bin}=0.50\pm0.08$.** Not from a prior — from
  **radial leverage alone**. The flat pedestal and the falling cluster shape are separable, and
  the design's core↔outskirts contrast measures both.
- **Honest precision: $\sigma(M)/M=6.9\%$**, up from the blind design's *claimed* $4.5\%$.
  Marginalizing over an unknown $f_{\rm bin}$ genuinely costs information — and saying so is the
  point. The blind $4.5\%$ was a fiction; the aware $6.9\%$ is the real cost of not knowing the
  binary fraction in advance.

That recovery of $f_{\rm bin}$ comes with a caveat sharp enough that it gets its own section
below — the mock is generated *at* the truth $f_{\rm bin}$ and the prior is *centred* there, so
the unbiased recovery alone proves only self-consistency. What proves $f_{\rm bin}$ is genuinely
**identifiable** is the [prior-insensitivity test](#sec-bin-identifiability), not this number.

### H2 — the binary-aware *design* is also tighter

The fix is not only a better *fit*; it is a better *design*. Comparing the two designs under
the same binary-aware (marginalized) fit, the binary-aware allocation reaches a target
$\sigma(M)$ with fewer stars:

```{math}
:label: bin-h2
\text{precision gain} \;=\;
\frac{\sigma(M)_{\text{binary-blind design}}}{\sigma(M)_{\text{binary-aware design}}}
\;=\; 1.33\times \quad\text{(at the operating point)},
```

confirming the pre-registered **H2** threshold ($\geq1.3\times$). This margin is *thin at the
fiducial ratio* $\sigma_{\rm bin}/\sigma_{\rm cluster}=1.08$ on purpose — and the
[sweep](#sec-bin-sweep) shows it grows to $\sim6\times$ as binaries come to dominate. **H3** is
confirmed too: the binary-aware allocation is *not* a monotone rescaling of the blind one — it
reshuffles, pulling budget toward the radii that constrain $f_{\rm bin}$ to break the
$M\leftrightarrow f_{\rm bin}$ degeneracy (the blind design spread over $\{1.5, 7.6, 17\}$ pc;
the aware design drops $7.6$ pc and concentrates $2.26$ pc while keeping $17$ pc).

(sec-bin-identifiability)=
## The honest test of identifiability: prior-insensitivity, not truth-centred recovery

This is the caveat to read **before quoting the $\hat f_{\rm bin}=0.50\pm0.08$ recovery** as
proof that binaries are measurable. The cross-model mock is generated at the *truth*
$f_{\rm bin}=0.5$, and the binary-aware fit's prior is *centred* on that same value. So an
unbiased recovery, by itself, demonstrates only **self-consistency** — the fit returns the
value it was pointed at. That is necessary but not sufficient.

What actually proves $f_{\rm bin}$ is **identifiable from the data** is a *prior-insensitivity*
test: **loosen the $f_{\rm bin}$ prior by $100\times$ and the recovered $\hat f_{\rm bin}$ moves
by only $0.4\%$.** The posterior is set by the radial leverage in the data, not by the prior —
which is the genuine evidence that the core↔outskirts contrast measures the binary fraction.
We state this plainly because it is the difference between a real measurement and a tautology,
and the truth-centred recovery number alone cannot tell them apart.

(sec-bin-maximin)=
## Robustness, honestly: maximin is a near-null

The measure-and-marginalize fix assumes you *trust* the Moe & Di Stefano $f_{\rm bin}$ enough
to model it. What if you do not — what if you want a design that is good across a whole *range*
of possible binary fractions you refuse to commit to? That is **maximin** (robust) design:
minimise the *worst-case* $\sigma(M)$ over $f_{\rm bin}\in[0, 0.7]$.

The honest answer here is a **near-null**, and reporting it as such is part of the integrity of
the arc. Because $\sigma(M)$ is **monotone** in $f_{\rm bin}$ (more binaries $\Rightarrow$ a
larger pedestal $\Rightarrow$ less mass information), the worst case always sits at the upper
endpoint $f_{\rm bin}=0.7$ — and the marginalize design, already built to handle a free
$f_{\rm bin}$, is *already nearly maximin-optimal*. The maximin design gives up only $+0.16\%$
of precision at the truth to gain $+0.24\%$ at the worst case: a $\sim0.2\%$ hedge. The lesson
is not "maximin wins" — it is that **measure-and-marginalize is the right tool here**, and the
expensive worst-case machinery buys almost nothing on top of it. A null reported faithfully is
a finding.

:::{figure} figures/demo_oedb_maximin.png
:label: fig-oedb-maximin
:width: 80%

**The robustness hedge is nearly free — and nearly pointless.** Forecast $\sigma(M)/M$ vs the
assumed binary fraction $f_{\rm bin}$ for the marginalize design (blue, optimal at the truth
$f_{\rm bin}=0.5$) and the maximin design (vermilion, optimal at the worst case
$f_{\rm bin}=0.7$). The two curves nearly coincide: $\sigma(M)$ is monotone in $f_{\rm bin}$,
so the marginalize design is already near-maximin-robust. The callout quantifies the $\sim0.2\%$
hedge; the inset shows the two per-bin allocations barely differ. An honest near-null.
:::

(sec-bin-sweep)=
## The sweep: the bias is catastrophic where binaries dominate

The fiducial operating point ($\sigma_{\rm bin}/\sigma_{\rm cluster}=1.08$) is one slice through
a continuum. Sweeping the system mass — which sets $\sigma_{\rm cluster}$ while $\sigma_{\rm bin}$
stays fixed — traces the bias across contamination regimes, and it is the regime-of-validity
map that turns a single number into a story.

The binary-blind-fit bias runs from $+26\%$ (massive, hot clusters where binaries are a minor
perturbation) to $+850\%$ (low-mass, cold clusters where the pedestal *swamps* the cluster
heat). Across the *entire* sweep the binary-aware fix holds the residual bias at $\approx0$. And
the H2 precision-gain grows from $\sim1.3\times$ at the fiducial point to $\sim6\times$ where
binaries dominate — so the thin operating-point margin is the *floor*, not the typical case.
**The colder the cluster, the more the binary-blind design lies, and the more the binary-aware
design is worth.**

:::{figure} figures/demo_oedb_sweep.png
:label: fig-oedb-sweep
:width: 95%

**Bias, fix, and OED payoff across the contamination ratio.** Left axis vs
$\sigma_{\rm bin}/\sigma_{\rm cluster}$ (top axis: system mass): the binary-blind-fit realized
$M$-bias (vermilion) rises from $+26\%$ to $+850\%$ as binaries come to dominate the cold
outskirts of lower-mass clusters; the binary-aware-fit residual (green) stays flat at $\approx0$
— the fix holds everywhere. Right axis: the H2 precision-gain (sky) grows from $\sim1.3\times$
to $\sim6\times$. The fiducial operating point (the single-point $+184\%$ headline) is marked.
:::

## Validating a pre-data calculation: two gates

A Fisher forecast is a *promise* — and this demo's promise is the unusual one that a design
will be *biased*. Both halves are checked.

### Gate 1 — the gradients are correct (AD-vs-FD)

The load-bearing new derivatives are the **$f_{\rm bin}$ Fisher block** and the **binary
$\sigma^2$-inflation term**. We verify them against finite differences to $\mathrm{rel}<10^{-3}$
(Richardson where a fixed-step FD is truncation-limited), per `gradient-validation`. The
analytic $f_{\rm bin}$ column $f_{\rm bin}V_{\rm bin}/(2\sigma_{\rm obs})$ matches AD to
$\sim10^{-16}$.

### Gate 2 — the bias is real (cross-model Monte Carlo)

The bias claim is gated by a **cross-model** Monte Carlo: *generate WITH Moe binaries, fit
WITHOUT*. The mock is **forward-model-consistent** — the cluster mock *is* the fit model, so the
$f_{\rm bin}=0$ baseline is unbiased by construction and any bias is attributable to binaries
alone. Each draw places the design's per-bin star counts directly, draws each star's cluster
velocity from $\mathcal{N}\!\left(0,\sigma_{\rm los}^2(R)\right)$ with $\sigma_{\rm los}$ from
`project_dispersion` at the truth, adds a flux-weighted Moe blend $\Delta$ (from a build-once
$K_{\rm orb}$ pool rescaled to $\mathrm{Var}=V_{\rm bin}$) to the per-star
$\mathrm{Bernoulli}(f_{\rm bin})$ fraction, adds per-star $\varepsilon_{\rm RV}$, forms the
per-bin $\hat\sigma$, and fits $\hat M$ by MAP in the $\ln\theta$ Gauss–Newton metric (honest
realized-$\hat\sigma$ weighting; unpopulated bins dropped, not floored). The H1 accept rule was fixed in advance: accept iff the mean
$\mathrm{bias}(\hat M)/M > 2\,\sigma_{\rm forecast}/M$ and the $2\,\text{SEM}$ band does not
straddle that threshold. It is met by a wide margin (bias/forecast $=41\times$; SEM $0.06$). The
binary-aware fit's accept rule — $|\mathrm{bias}| < 2\,\sigma_{M,\rm marg}$ — is met by the
$+5\%$ residual.

(sec-bin-prereg)=
## The pre-registration (locked before the Monte Carlo)

To keep the bias result honest, the hypotheses and accept/reject rules were **locked before**
the Monte Carlo ran, via `research-workflow:discriminating-experiment-design`. A reject on any
of them was defined in advance as a *reportable finding*, not a failure — and an H1 reject would
have *descoped the whole arc*.

```{list-table} Pre-registered hypotheses (locked 2026-06-19)
:header-rows: 1
:label: tbl-bin-prereg

* - Hypothesis
  - Accept rule (fixed in advance)
  - Outcome
* - **H1 — the bias.** Binary-blind design + binary-blind fit on contaminated data biases $\hat M$ high beyond its own forecast.
  - $\mathrm{bias}(\hat M)/M > 2\,\sigma_{\rm forecast}/M$, positive, $2\,$SEM not straddling.
  - **ACCEPT** ($+184\%$ vs $4.5\%$; $41\times$)
* - **H0 — the rival.** The binary-blind fit absorbs the pedestal into nuisances; $\hat M$ stays unbiased.
  - $|\mathrm{bias}|\le\sigma_{\rm forecast}/M$.
  - rejected (the $f_{\rm bin}=0$ control is the unbiased case; $f_{\rm bin}=0.5$ is not)
* - **H2 — OED payoff.** The binary-aware design beats the blind one under the binary-aware fit.
  - precision-gain $\geq1.3\times$.
  - **ACCEPT** ($1.33\times$ at the operating point; $\sim6\times$ in the sweep)
* - **H3 — non-obvious allocation.** The binary-aware allocation is not a monotone rescaling of the blind one.
  - per-bin weight rank-order changes.
  - **ACCEPT** (drops $7.6$ pc, concentrates $2.26$ pc)
```

(sec-bin-results)=
## Quantitative results

All numbers are from the gated CLI's run-record (EFF-OM YMC operating point,
$N_{\rm total}=5000$, $K=12$ bins, cross-model MC $48$ draws):

```{list-table} OED binary-robustness results
:header-rows: 1
:label: tbl-bin-results

* - Quantity
  - Binary-blind (naive)
  - Binary-aware (fix)
* - $\hat M/M$ (cross-model MC, $f_{\rm bin}=0.5$)
  - $\mathbf{2.84}$ (**$+184\%$ bias**)
  - $\approx1.05$ ($+5\%$, within forecast)
* - Claimed / honest $\sigma(M)/M$
  - $\mathbf{4.5\%}$ (claimed)
  - $\mathbf{6.9\%}$ (honest, marginalized)
* - bias / forecast ratio
  - $\mathbf{41\times}$ (false confidence)
  - $<2$ (calibrated)
* - $f_{\rm bin}=0$ control bias
  - $-0.3\%$ (unbiased — isolates the binaries)
  - —
* - Recovered $\hat f_{\rm bin}$
  - — (not modelled)
  - $\mathbf{0.50\pm0.08}$ (radial leverage)
* - Prior-insensitivity ($100\times$ looser prior)
  - —
  - $\hat f_{\rm bin}$ moves $0.4\%$ (identifiable)
* - H2 precision-gain (binary-aware design)
  - —
  - $\mathbf{1.33\times}$ (fiducial); $\sim6\times$ (binary-dominated)
* - Maximin hedge over the marginalize design
  - —
  - $\sim0.2\%$ (honest near-null)
* - Sweep: $M$-bias range across system mass
  - $+26\% \to +850\%$
  - $\approx0$ throughout
```

## What the optimum means: science implications

**1. "Optimal" is only optimal under a *correct* model.** The whole demo is a warning about
the silent assumption inside every OED (and every analysis): the design that minimises your
forecast variance under the wrong model can *maximise* your real error. A tighter forecast is
not a better measurement — it is only a better measurement *if the model is right*. The
binary-blind design's $4.5\%$ was the most dangerous kind of number.

**2. Population systematics are a *design* problem, not just an analysis problem.** You cannot
fix binaries purely after the fact if your survey spent its budget in the contaminated
outskirts. The binary-aware design spends differently *because* it knows the pedestal exists —
it earns the $f_{\rm bin}$ constraint from radial leverage at the telescope, not from a prior at
the keyboard.

**3. `progenax` can do this because it is differentiable end-to-end.** Wiring a faithful Moe &
Di Stefano binary population into a Fisher design requires
$\partial(\text{information})/\partial(\text{observing strategy})$ *through* the binary kernel.
That is the throughline of [the whole OED section](index.md): the same additive backbone that
targets [anisotropy](anisotropy.md), [mass](dynamical-mass.md), and
[concentration](concentration.md) here defends a measurement against a *model error* — it
extends OED from "what to measure" to "what to measure **robustly**."

## Current scope and planned extensions

```{warning}
This is a **pre-data, single-epoch, RV-only OED on a clean self-consistent mock**, headlining a
binary-misspecification *bias*. Its boundaries, stated honestly:

- **Single-epoch.** Binaries are an unresolvable statistical *inflation* handled via the second
  moment $\sigma^2$ + radial leverage, **not** via per-epoch RV-variability *detection*.
  Multi-epoch detection (epochs-vs-stars) is the explicit next arc; the API is built
  multi-epoch-ready.
- **Mass-follows-light, RV-only.** No dark-matter halo (tracer $\ne$ mass is deferred); the
  RV-only $M\leftrightarrow r_a$ mass–anisotropy degeneracy is handled by tight *photometric*
  priors on $(\gamma, a)$, not by proper motions.
- **The cluster forward model is the held-fixed CONTROL.** A misspecification study isolates one
  variable; here that variable is the binaries. The EFF-OM cluster physics is frozen across
  design, truth, and both fits, so the entire bias is attributable to the binaries.
- **Identifiability is proven by prior-insensitivity, not by the truth-centred recovery.** The
  mock is generated at the truth $f_{\rm bin}$ and the prior is centred there, so the unbiased
  $\hat f_{\rm bin}=0.50\pm0.08$ shows *self-consistency*; the $0.4\%$ move under a $100\times$
  looser prior is what shows $f_{\rm bin}$ is genuinely **identifiable from the data**
  ([the identifiability test](#sec-bin-identifiability)).
- **Second-moment identification only.** The $f_{\rm bin}\leftrightarrow\sigma_{\rm cluster}$
  degeneracy is broken by *radial leverage*, not by the non-Gaussian shape of the binary
  velocity tails (the full-histogram identification is a stronger, deferred extension).
- **The Fisher is a local, Gaussian (Cramér–Rao) approximation.** The cross-model Monte Carlo is
  precisely the check that it predicts the realized bias and scatter — and it does.
- **Not part of v0.1.0.** Scripts + this page, **no** `src/progenax/` API
  surface; the OED tooling is planned for a separate package.
```

:::{note} What we just learned
A binary-*blind* survey design, optimised to weigh a young-massive cluster and fit with a
binary-blind model, recovers $\hat M/M=2.84$ — a **$+184\%$ bias** — while reporting a $4.5\%$
error bar: a **$41\times$ false-confidence** disaster, because it parks its budget in the cold
outskirts where the flat binary pedestal dominates the falling cluster signal. The
**measure-and-marginalize** fix — modelling $f_{\rm bin}$ in both the design and the fit — drops
the bias to $+5\%$, recovers $f_{\rm bin}=0.50\pm0.08$ from radial leverage alone (identifiable:
$\hat f_{\rm bin}$ moves $0.4\%$ under a $100\times$ looser prior), and reports an honest
$\sigma(M)/M=6.9\%$. The binary-aware *design* is $1.33\times$ tighter at the operating point and
$\sim6\times$ tighter where binaries dominate; **maximin** robustness is an honest near-null
($\sim0.2\%$ hedge, because $\sigma(M)$ is monotone in $f_{\rm bin}$). Across system mass the
blind bias runs $+26\%\to+850\%$ while the fix stays $\approx0$. This is the first OED demo whose
headline is a **bias**, not a recovery — it extends [anisotropy](anisotropy.md),
[dynamical mass](dynamical-mass.md), and [concentration](concentration.md) from *what to
measure* to *how to measure it robustly*. Single-epoch; the OED tooling is planned for a
separate package and is not part of v0.1.0.
:::

## How to run

```bash
# the cheap design + forecast + mechanism figure (no MC; CI-safe, ~1 min)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_binary.py

# the full result: cross-model MC (the H1 bias + headline figure) + maximin + sweep
PROGENAX_RUN_OED_BINARY=1 env -u VIRTUAL_ENV uv run --no-sync \
    python scripts/demo_oed_binary.py --run-mc --maximin --sweep \
    --outdir docs/website/60-science-demos/optimal-design/figures
```

The cheap default path computes the binary-blind design, its forecast $\sigma(M)/M$, and the
mechanism figure, and exits 0. The env-gated cross-model Monte Carlo (the H1 bias, the
false-confidence figure) is `@slow` and **out of CI** — it is the gate
`test_H1_naive_design_is_biased_beyond_forecast` (and the binary-aware
`test_fix_binary_aware_fit_is_unbiased`) in `tests/unit/test_demo_oed_binary.py`, run with
`PROGENAX_RUN_OED_BINARY=1`. The figures and run-record land alongside this page.

## References

The shared Fisher / Cramér–Rao / projection theory and its references are on
[the OED formalism page](background.md). The line-of-sight projection into $\sigma_{\rm los}$ is
{cite:t}`BinneyMamon1982`; the Osipkov–Merritt anisotropy law is {cite:t}`Merritt1985`; the EFF
density profile is documented on the
[EFF profile theory page](../../10-theory/spatial-profiles/eff.md). The binary population — the
$P$–$q$–$e$ engine behind $V_{\rm bin}$ — follows {cite:t}`MoeDiStefano2017`, with ZAMS flux
weights from {cite:t}`Tout1996`; the single-epoch binary-inflated dispersion machinery is the
[binary dynamical-mass demo (B12)](../binary-dynamical-mass.md). The companion OED designs are
[anisotropy](anisotropy.md) (where the proper motions belong), [dynamical mass](dynamical-mass.md)
(how deep to survey), and [concentration](concentration.md) (where concentration information
lives) — this page extends them to model *robustness*.
