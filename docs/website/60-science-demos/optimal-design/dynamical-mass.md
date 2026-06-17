---
title: OED for dynamical mass — how deep to survey (Stage 2)
subtitle: The magnitude limit is a design knob — and there is an optimal depth to weigh a cluster
description: "An optimal-experimental-design demo that promotes survey DEPTH (the limiting magnitude m_lim) to a design variable, headlining the cluster's dynamical mass M. Going deeper unlocks more tracer stars (the IMF integrated above a falling mass floor) but those faint stars are photon-noisier, so the Fisher information on M peaks at an INTERIOR optimal depth: sigma(M)/M is minimised near m_lim ~ 13.3 at d = 4 kpc, ~0.10, beating a too-shallow (m_lim=10, 1.15x worse) and a too-deep (m_lim=16, 1.33x worse) survey. At the optimum the budget concentrates in the outskirts (93% outside the core) with radial PM dominant. A magnitude-selected Monte-Carlo calibration validates the depth Fisher to ~15%. Bolometric magnitudes; band-specific photometry is a planned fluxax follow-up."
---

# OED for dynamical mass — how deep to survey (Stage 2)

You have a cluster to weigh and a telescope to do it with. The
[anisotropy example](anisotropy.md) asked *where* to point and in *which* channel. This
one asks a question with a sharper answer: **how faint should you go?** Surveying deeper
costs real exposure time, and the usual instinct is "deeper is always better — more stars
is more information." It is not. There is a depth past which every extra magnitude *hurts*
your measurement of the cluster's mass. This page promotes the **limiting magnitude
$m_{\rm lim}$** from the fixed completeness of the anisotropy example to a genuine
**design knob**, and finds the depth that best weighs the cluster.

:::{note} **Shared theory lives once, on the formalism page**
The Fisher information / Cramér–Rao foundation {ref}`oed-fisher-gaussian`, the
[additive design-linear backbone](background.md#sec-oed-additive)
{ref}`oed-additive-fisher`, the [dimensionless $\ln\theta$ metric](background.md#sec-oed-dimensionless),
and the [B&M82 sky-projection geometry](background.md#sec-oed-projection) are built once
on [the OED formalism page](background.md). The crucial fact this page leans on:
**the total mass $M$ enters every dispersion channel** through $GM(<s)$ in the Jeans
integral {ref}`oed-jeans` ($\sigma_r^2\propto GM$), so a dispersion measured *anywhere*
weighs the cluster.
:::

## The depth knob: which stars exist to be observed

A limiting magnitude does not change the cluster — it changes which of its stars are
bright enough to land in your catalogue. A star of mass $m$ at distance $d$ has an
apparent magnitude set by its luminosity and the distance modulus,

```{math}
:label: dm-app-mag
m_{\rm app}(m) \;=\; M_{\rm bol}(m) + 5\log_{10}\!\big(d/10\,\mathrm{pc}\big),
\qquad
M_{\rm bol}(m) = 4.74 - 2.5\log_{10}\!\big(L(m)/\Lsun\big),
```

with $L(m)$ the **Tout et al. (1996) ZAMS** luminosity (in-package via
`progenax.stellar`). A star is detectable iff $m_{\rm app}\le m_{\rm lim}$ — equivalently,
iff its mass exceeds a floor $m_{\rm min}(m_{\rm lim},d)$ obtained by inverting the ZAMS
$L$–$M$ relation. **Deeper means a lower mass floor**: more of the steeply-rising IMF
becomes observable. For our fiducial cluster at $d=4$ kpc we sweep $m_{\rm lim}$ over
$[9,18]$ mag, and the mass floor falls monotonically across that range.

For the Sun at $10$ pc, $m_{\rm app}=M_{\rm bol,\odot}=4.74$; move it out to $4$ kpc and
the distance modulus alone adds $5\log_{10}(400)\approx13.0$ mag, putting a solar-mass
star at $m_{\rm app}\approx17.7$ — right at the faint edge of our deepest survey. That
single number is why $d=4$ kpc is the interesting distance: the depth sweep brackets the
turnoff from "only the bright giants" to "down to the solar-mass main sequence."

## The interior optimum — the headline

Hold the *measured-star budget* $N_{\rm total}$ fixed, exactly as in the anisotropy
example, and ask the optimiser to choose both the allocation **and** the depth. It does
not run to the deepest survey. It stops in the middle:

```{list-table} The optimal survey depth ($d=4$ kpc, $N_{\rm total}=400$)
:header-rows: 1
:label: tbl-dm-optimum

* - Survey depth
  - $\sigma(M)/M$
  - vs the optimum
* - too shallow ($m_{\rm lim}=10$)
  - $0.115$
  - $1.15\times$ worse
* - **optimal ($m_{\rm lim}\approx13.3$)**
  - $\mathbf{0.10}$
  - **best**
* - too deep ($m_{\rm lim}=16$)
  - $0.133$
  - $1.33\times$ worse
```

The joint optimiser settles at $m_{\rm lim}\approx13.2$ and an independent depth sweep
puts the minimum of $\sigma(M)/M$ at $m_{\rm lim}\approx13.1$ — consistent, with a broad
flat plateau through $\approx13.5$. Either way the result is the same: **$\sigma(M)/M$ has
an *interior* minimum**, not an endpoint one. A survey that stops too bright cannot place
tracers where the mass leverage is; a survey that goes too faint drowns in photon noise.
Both extremes are measurably worse than the depth in between.

:::{figure} figures/demo_oed2_depth_optimum.png
:label: fig-dm-optimum
:width: 85%

**$\sigma(M_{\rm dyn})/M_{\rm dyn}$ has an interior minimum in survey depth.** The
fractional precision on the dynamical mass vs the limiting magnitude $m_{\rm lim}$, at the
best allocation for each depth ($d=4$ kpc, $N_{\rm total}=400$). The curve falls from a
too-shallow survey, bottoms out at $\sigma(M)/M\approx0.10$ near $m_{\rm lim}\approx13.3$,
and *rises again* toward a too-deep one. The minimum is interior — depth is a real
trade-off, not a "more is better" knob.
:::

## The mechanism: supply versus noise

Why an interior optimum, with no cost model and no money changing hands? Because depth
pulls two levers in opposite directions, and the Fisher information on $M$ is their
**product**.

**Supply rises with depth.** The number of tracer stars depth makes available is the
Chabrier IMF integrated above the falling mass floor, summed over the radial bins. Across
$m_{\rm lim}=9\to18$ that available pool climbs steeply, from $\sim132$ to $\sim2110$
stars — and it rises *fastest in the star-starved outskirts*, exactly where the mass
leverage lives.

**Noise rises with depth too.** The stars depth unlocks are faint, and faint stars are
photon-noisy. We model the per-star measurement error with a photon-noise-like scaling
(below), so the IMF-weighted **effective error** $\epsilon_{\rm eff}(m_{\rm lim})$ climbs
as the survey admits ever-fainter tracers — in the RV channel, from $\sim0.23$ to
$\sim12$ km/s across the same sweep.

Each extra magnitude buys you more tracers but noisier ones. The Fisher information per
unit budget — supply divided by noise-variance — therefore peaks in between, and that peak
*is* the optimal depth.

:::{figure} figures/demo_oed2_depth_trade.png
:label: fig-dm-trade
:width: 90%

**The two competing trends that produce the optimum.** Left: the available tracer count
(IMF integrated above the mass floor) rises monotonically with depth. Centre: the
IMF-weighted per-star effective error rises with depth as fainter, noisier stars enter.
Right: their combination — the information per unit budget — peaks at an interior depth,
the same $m_{\rm lim}\approx13$ as {numref}`fig-dm-optimum`. Supply and noise pull in
opposite directions; the optimum is where they balance.
:::

## The allocation: weigh the cluster in the outskirts

At the optimal depth, *where* does the budget go? Almost entirely outside the core. Of the
optimal allocation, **only $\sim7\%$ of stars sit inside the core radius — $\sim93\%$ go
to the outskirts**, and within the kinematic channels the **radial proper motion
dominates** while the tangential PM is driven to nearly zero.

That allocation is not arbitrary — it is the mass leverage and the anisotropy talking at
once. The dynamical mass is best constrained by the dispersion in the outskirts, where the
enclosed-mass profile $M(<r)$ is flattening toward the total $M$. And the
Osipkov–Merritt anisotropy *suppresses the tangential dispersion* exactly there
($\sigma_{{\rm pm},T}^2\propto1-\beta\to0$ as $\beta\to1$), so a tangential-PM star in the
outskirts carries little signal — the optimiser correctly spends almost nothing on it.

:::{figure} figures/demo_oed2_allocation.png
:label: fig-dm-allocation
:width: 90%

**The optimal-depth allocation concentrates in the outskirts.** Effective star count per
(projected-radius bin × channel) at the joint optimum. The budget piles into the outer
bins ($\sim93\%$ outside the core), with the radial-PM channel carrying the load and the
tangential PM near-empty — the outskirts weigh the cluster, and OM anisotropy makes the
tangential channel barren there.
:::

## The frontier: precision versus budget at the optimal depth

Fix the depth at its optimum and vary the budget. The fractional precision on $M$ improves
as you add stars, then **flattens against the finite supply** — once your budget approaches
the number of stars the depth actually makes available, more budget cannot help, because
there are no more bright-enough stars to measure. At the demo's $N_{\rm total}=400$ the
optimal-depth design reaches $\sigma(M)/M\approx0.10$.

:::{figure} figures/demo_oed2_frontier.png
:label: fig-dm-frontier
:width: 80%

**The dynamical-mass frontier at the optimal depth.** $\sigma(M_{\rm dyn})/M_{\rm dyn}$ vs
the star budget $N_{\rm total}$, evaluated at the optimal $m_{\rm lim}$. Precision improves
with budget but flattens as the budget approaches the depth's supply ceiling — a real,
supply-limited frontier, not an idealised $1/\sqrt N$ line. The demo point sits at
$N_{\rm total}=400$, $\sigma(M)/M\approx0.10$.
:::

## Validating a pre-data calculation: the magnitude-selected calibration

The depth Fisher is a *promise* about a survey you have not run yet — it deserves the same
check as the anisotropy example, but harder: now the mock must respect the **magnitude
selection** itself. We draw a Monte-Carlo ensemble of mock catalogues in which masses are
sampled from the Chabrier IMF, *only* stars with $m_{\rm app}\le m_{\rm lim}$ are kept, and
each surviving star is given the magnitude-dependent error $\epsilon(m_{\rm app})$. We bin,
fit $\hat M$ by maximum *a posteriori*, and compare the realised $\mathrm{Var}(\hat M)/M^2$
to the Fisher prediction $(F^{-1})_{MM}$.

```{important}
:label: imp-dm-calibration
**The depth Fisher is validated, not merely consistent.** Across the optimal designs the
realised $\sigma(M)$ matches the prediction to within $\approx15\%$: the
realised/predicted variance ratio sits at $\sim0.84$–$1.05$ — consistent with $1.0$,
**with no significant systematic bias** in either direction. (Earlier drafts reported a
"conservative" offset; that was an artefact of a since-replaced fitter, not a property of
the Fisher. The honest result is *validated to $\sim15\%$*.) The realised
$\sigma(M)/M\approx0.103$ sits right on the Fisher's $0.100$.
```

:::{figure} figures/demo_oed2_calibration.png
:label: fig-dm-calibration
:width: 70%

**A magnitude-selected mock ensemble confirms the depth Fisher.** Realised $\sigma(M)$
(from MAP fits to mocks drawn under the *same* magnitude selection the design assumed)
against the Fisher prediction, across the optimal designs. The realised values track the
prediction to within $\approx15\%$ (variance ratio $\sim0.84$–$1.05$, consistent with
$1.0$, no significant bias). The depth-dependent selection physics in the Fisher is
trustworthy.
:::

(sec-dm-assumptions)=
## Assumptions & approximations

Every modelling choice below is a deliberate simplification with a reason. This is a
*pedagogical depth knob*, not a survey exposure-time calculator — the headline is the
**shape** of the information-versus-depth trade and *that* an interior optimum exists, both
of which are robust to the simplifications. Each choice gets its rationale and the evidence
that it is safe here.

### 1. Bolometric magnitudes (no band, no bolometric correction, no extinction)

We use bolometric magnitudes throughout: $M_{\rm bol}(m)=4.74-2.5\log_{10}(L/\Lsun)$ with
$L$ from the ZAMS relation, and no per-band magnitude, no bolometric correction, no
interstellar extinction.

**Why this is OK here.** The headline is the *shape* of the information-versus-depth curve
and the *existence* of an interior optimum — both set by two band-independent ingredients:
the IMF × ZAMS **supply** (how many stars sit above a given mass floor) and the
**photon-noise scaling** (how error grows with apparent faintness). A real photometric band
shifts the mass floor and the error normalisation, but to first order it does not change
whether supply-versus-noise produces an interior peak, nor roughly where. The bolometric
choice keeps the demo's physics legible without a colour model getting in the way.

:::{note}
**Band-specific photometry is a planned follow-up.** Real bolometric corrections,
wavelength-dependent extinction, and crowding-limited detection will come from the
[`fluxax`](https://github.com/drannarosen/progenax) package once it is hardened and
finalised — the next arc on this thread. **The numbers on this page are illustrative:**
the optimal $m_{\rm lim}\approx13.3$ and $\sigma(M)/M\approx0.10$ are correct *for the
bolometric model*, and should be read as demonstrating the method and the qualitative
optimum, not as a survey-design prescription for a specific instrument and filter.
:::

### 2. Photon-noise error model $\epsilon\propto10^{0.2(m_{\rm app}-m_{\rm ref})}$

Per-star measurement error scales as $\epsilon(m_{\rm app})=\epsilon_0\cdot
10^{0.2(m_{\rm app}-m_{\rm ref})}$, anchored so a reference-magnitude star ($m_{\rm ref}$)
has the fiducial error $\epsilon_0$ ($\epsilon_{0,\rm RV}=1$ km/s,
$\epsilon_{0,\rm PM}\to0.95$ km/s at $d=4$ kpc).

**Why this is OK here.** The exponent $0.2$ per magnitude is the photon-counting
$\mathrm{flux}^{-1/2}$ scaling: one magnitude fainter is a factor $10^{0.4}\approx2.5$ less
flux, hence $\sqrt{2.5}\approx1.6\times$ the error. That power law is the *correct
first-order* faint-end behaviour of any photon-limited measurement, which is what sets the
rising-noise arm of the trade. It is **not** a real survey exposure-time calculator — there
is no sky background, no detector read noise, no saturation at the bright end, no
wavelength dependence. We say so plainly: the scaling captures the *trend* that drives the
optimum, anchored to a bright reference star, and nothing finer.

### 3. Availability soft-cap $n_{\rm eff}=\mathrm{avail}\cdot\tanh(n_{\rm design}/\mathrm{avail})$

The design proposes $n_{\rm design}$ stars per cell, but a cell can only deliver as many
stars as actually exist there. We enforce this with a smooth cap, $n_{\rm eff}=
\mathrm{avail}\cdot\tanh(n_{\rm design}/\mathrm{avail})$.

**Why this is OK here.** A finite bright-star supply is real physics — you cannot observe
the 50th-brightest star in a bin that contains only 30 — and the optimiser needs to feel it
to find an *interior* depth (without a supply limit, "go shallow and pile every star into
the few bright outskirts cells" would be unbeatable). The $\tanh$ is the differentiable
realisation: for $n_{\rm design}\ll\mathrm{avail}$ it is $\approx n_{\rm design}$ (the
supply is irrelevant, you take what you ask for), and for $n_{\rm design}\gg\mathrm{avail}$
it saturates at $\mathrm{avail}$ (you have exhausted the cell). It is a soft constraint
chosen for smooth gradients, not a sharp $\min(n,\mathrm{avail})$, so the AD-vs-FD gate
(below) stays clean.

### 4. $\epsilon_{\rm eff}$ is per-channel global; availability is per-bin

The effective error $\epsilon_{\rm eff}(m_{\rm lim})$ is one number per channel (an
IMF-weighted RMS over the detectable masses), while the availability weight
$\mathrm{avail}_b(m_{\rm lim})$ is per radial bin.

**Why this is OK here.** All stars in the cluster share a *single distance* $d$. So the
mapping from mass to apparent magnitude — and hence the magnitude-dependent error — is the
same function of mass everywhere in the cluster; the IMF-weighted error distribution does
not depend on which radial bin a star sits in. That makes a per-channel global
$\epsilon_{\rm eff}$ *exact* for a single-distance cluster, not an approximation. What
*does* vary with radius is how many stars there are to detect — the projected density falls
outward — so availability is correctly carried per bin. The geometry (one distance) is what
cleanly factorises error-per-mass (global) from supply (radial).

### 5. Single-population, mass-follows-light

Every star traces the same potential, so the predicted dispersions $\sigma_{\rm pred}(r)$
are a property of the cluster's mass distribution — **independent of $m_{\rm lim}$**.

**Why this is OK here, and why it is the key technical point.** Because $\sigma_{\rm pred}$
does not depend on depth, the [additive backbone](background.md#sec-oed-additive)
{ref}`oed-additive-fisher` survives untouched: the per-star Jacobian
$J=\partial\sigma_{\rm pred}/\partial\ln\theta$ is **computed once**, and $m_{\rm lim}$
enters only through the cheap, differentiable scalars $\epsilon_{\rm eff}(m_{\rm lim})$ and
$\mathrm{avail}_b(m_{\rm lim})$. This is what makes optimising over depth affordable — the
expensive B&M82 projection is never re-differentiated as the depth changes. The cost is a
genuine scope limit: mass segregation and multi-mass kinematics (where lower-mass stars
have *hotter*, more extended orbits, so $\sigma_{\rm pred}$ *would* depend on which masses
you detect) are out of scope. For a single-mass tracer population, mass-follows-light holds
exactly.

### 6. The calibration's mild design-dependence

The realised/predicted variance ratio is not a single number — it varies across the optimal
designs, $\sim0.84$–$1.05$.

**Why this is honest, not a hidden bias.** The spread is consistent with a ratio of $1.0$
given the Monte-Carlo error; there is no significant systematic offset in either direction
({ref}`imp-dm-calibration`). The leading *candidate* for the residual scatter is a known
subtlety in how the per-channel $\epsilon_{\rm eff}$ is formed: it is an IMF-weighted
**RMS** error, whereas the information-optimal combination of heterogeneous per-star errors
is an **inverse-variance** weighting, and the two differ slightly when the error
distribution within a channel is broad. We flag this as the most likely source of the mild
design-to-design wobble *without* over-claiming it as a systematic — the calibration's
verdict is that the depth Fisher is trustworthy to $\approx15\%$, and that is what we
report.

### The evidence backing these choices

Three quantitative results, all from the gated CLI, are what license the simplifications:

- **The interior optimum exists and is robust** — $\sigma(M)/M$ minimised at
  $m_{\rm lim}\approx13.1$–$13.3$ ({numref}`fig-dm-optimum`), beating shallow ($1.15\times$)
  and deep ($1.33\times$). The shape — the actual claim — is exactly what the
  band-independent supply-versus-noise argument predicts.
- **The depth gradient is correct** — the joint optimiser differentiates through
  $m_{\rm lim}$, and an automatic-vs-finite-difference gate on
  $\partial(\text{criterion})/\partial[\,\mathbf z, m_{\rm lim}]$ passes at the
  $\sim10^{-9}$ level. The smooth $\tanh$ supply cap and the smooth mass-floor are what keep
  that gradient clean.
- **The Fisher is calibrated** — the magnitude-selected Monte-Carlo confirms the realised
  $\sigma(M)$ to $\approx15\%$ ({ref}`imp-dm-calibration`), under the *same* selection the
  design assumed.

## Current scope and planned extensions

```{warning}
This is a **pre-data, single-shot OED on a clean self-consistent mock**, headlining survey
depth. Its boundaries, stated honestly:

- **Bolometric magnitudes only** — no band, no bolometric correction, no extinction, no
  crowding. Real band-specific photometry is the planned [`fluxax`](https://github.com/drannarosen/progenax)
  follow-up; current numbers are illustrative ({ref}`sec-dm-assumptions`, item 1).
- **Illustrative photon-noise error model** — $\epsilon\propto10^{0.2(m_{\rm app}-m_{\rm ref})}$
  captures the faint-end trend, **not** a real survey exposure-time calculator (item 2).
- **Single-population, mass-follows-light** — $\sigma_{\rm pred}$ is $m_{\rm lim}$-independent,
  so $J$ is computed once. Mass segregation and multi-mass kinematics are out of scope (item 5).
- **A single global $m_{\rm lim}$** — depth is one number for the whole survey, not a
  per-region map. Explicit exposure **cost** and **multi-epoch astrometry** are the deferred
  Stage 3.
- **Chabrier IMF fixed** — the IMF is an input, not itself a design target here.
- **The Fisher is a local, Gaussian (Cramér–Rao) approximation** — exact only in the
  high-information limit. The magnitude-selected calibration is the check that it predicts
  the realised scatter, and it does, to $\approx15\%$ ({numref}`fig-dm-calibration`).
```

:::{note} What we just learned
The **limiting magnitude is a design knob**: it gates which stars exist to be observed
{ref}`dm-app-mag`, and a deeper survey unlocks more tracers (the IMF above a falling mass
floor) but photon-noisier ones. Because the dynamical mass $M$ enters every dispersion
through $GM$ {ref}`oed-jeans` while the predicted dispersions stay depth-independent, the
[additive backbone](background.md#sec-oed-additive) survives — depth enters only through a
per-channel effective error and a per-bin availability weight — and the Fisher information
on $M$ peaks at an **interior optimal depth**: $\sigma(M)/M\approx0.10$ near
$m_{\rm lim}\approx13.3$ at $d=4$ kpc, beating both a too-shallow ($1.15\times$) and a
too-deep ($1.33\times$) survey. At that depth the budget concentrates in the **outskirts**
($\sim93\%$ outside the core, radial PM dominant), and a magnitude-selected calibration
validates the prediction to $\approx15\%$. The same recipe with a *channel* knob instead of
a *depth* knob is [the anisotropy example](anisotropy.md).
:::

## How to run

```bash
# quick (small calibration ensemble, ~1 min)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_dynamical_mass.py

# publication-grade (64-draw calibration, ~minutes) — regenerates the figures here
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_oed_dynamical_mass.py --full
```

The CLI is gated (exit 0 only if the interior optimum exists, the AD-vs-FD depth gradient
passes, and the magnitude-selected calibration lands within band) and writes a JSON
run-record alongside the five figures.

## References

The shared Fisher / Cramér–Rao / projection theory and its references are on
[the OED formalism page](background.md). The ZAMS $L$–$M$ relation is Tout et al. (1996),
documented on the [ZAMS validation](../../50-validation/zams-relations.md) page; the
B&M82 sky projection is Binney & Mamon (1982, MNRAS 200, 361); the Osipkov–Merritt
anisotropy law is {cite:t}`Merritt1985`. The companion design — *where*, not *how deep* —
is [the anisotropy example](anisotropy.md), and the anisotropy *recovery* demo is
[B6](../anisotropy.md).
```
