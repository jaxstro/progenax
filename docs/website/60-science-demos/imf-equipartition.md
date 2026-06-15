---
title: IMF + equipartition recovery (B2)
description: "The self-consistent joint demo: one IMF high-mass slope alpha drives both the observed mass histogram and the equipartition mass groups of a LIMEPY multimass cluster, so (alpha, delta, W0) are recovered jointly by MLE + Fisher + NUTS — with the kinematics-only degeneracy, a wrong-IMF bias curve and an alpha_true robustness grid."
---

# IMF + equipartition recovery (B2)

This is the flagship demo: it couples a cluster's **mass function** to its
**internal kinematics** through a single physical parameter, and shows that
parameter is recoverable from both channels at once.

## The self-consistent physics

A multimass LIMEPY cluster {cite:p}`Gieles2015` is built from an
[initial mass function](../10-theory/imfs/index.md). progenax bins the IMF into
$J=4$ mass groups; each group $j$ has a representative mass $m_j$ and a population
weight set by the IMF. **Two-body relaxation** drives the cluster toward energy
equipartition — heavier stars sink and slow down — encoded in the multimass DF by
a per-group velocity scale,

```{math}
:label: b2-equipartition
\mu_j = \frac{m_j}{\bar m},
\qquad
s_j = s\,\mu_j^{-\delta},
\qquad
\delta = \tfrac12 \Rightarrow m_j s_j^2 = \text{const (full equipartition)},
```

with $\bar m$ the central-density-weighted mean mass and $\delta$ the
**equipartition degree** {cite:p}`Peuten2017`. Real clusters reach only
*partial* equipartition ($\delta < \tfrac12$); the heaviest stars approach
$\sigma\propto m^{-1/2}$ while the light stars saturate at an escape-speed
ceiling — the Bianchini relation, derived directly from this same DF on the
[multimass equipartition](../10-theory/spatial-profiles/multimass-equipartition.md)
theory page.

**The high-mass IMF slope.** The observed masses follow a Maschberger (2013) IMF
whose high-mass behaviour is the power law

```{math}
:label: b2-imf
\frac{dN}{dm}\Big|_{m\gg m_c}\ \propto\ m^{-\alpha},
\qquad \alpha_{\rm true} = 2.3\ (\approx\text{Salpeter}),
```

(full smooth form and characteristic mass: {cite:t}`Maschberger2013`). **Here is
the coupling that makes the demo self-consistent:** the *same* $\alpha$ that sets
the slope of the observed mass histogram also sets the masses $m_j$ and weights of
the equipartition groups in @b2-equipartition — so $\alpha$ is constrained by the
**masses** (the histogram) *and* by the **kinematics** (the group-by-group
$\sigma_j(r)$). The three recovered parameters are
$\theta = (\alpha,\ \delta,\ W_0)$, with $W_0$ the central dimensionless
concentration.

## The joint likelihood

Two channels, summed (`scripts/demo_delta_recovery.py`):

```{math}
:label: b2-joint
\ln\mathcal L(\alpha,\delta,W_0)
= \underbrace{-\tfrac12\sum_{j,k} w_{jk}
   \Big(\tfrac{\hat\sigma_{jk}-\sigma_{jk}^{\rm pred}}{\mathrm{SE}_{jk}}\Big)^2}_{\text{kinematics: per-group }\sigma_j(r)}
\ +\
\underbrace{\sum_{i=1}^{N}\ln f_{\rm Masch}(m_i;\alpha)}_{\text{masses: the IMF histogram}} .
```

The kinematic predictor is the differentiable Engine A multimass oracle rebuilt
from $\theta$ inside the traced loss (the `find_alpha_for_masses` eigenvalue solve
is differentiable in $\alpha,\delta,W_0$); the mass term is the analytic,
normalized Maschberger log-pdf — fully differentiable in $\alpha$.

```{note}
**Observed masses are a *global* IMF sample (clean-mock Option A).** The kinematic
groups carry only the $J=4$ representative labels $m_j$, so the per-star mass
likelihood is evaluated on a *separate* global draw $m_i \sim
\mathrm{Maschberger}(\alpha_{\rm true})$ over the full mass range, with **no**
per-star mass↔group correlation modeled. One $\alpha$ still drives both channels
at the population level; the demo does not claim to model the joint mass–velocity
covariance of individual stars.
```

## Inputs and assumptions

The fit recovers **three parameters** $(\alpha, \delta, W_0)$; the truncation order,
the total mass, and the mass range are assumed known. The headline result — that the
mass channel *pins* $\alpha$ and breaks the kinematic $(\alpha,\delta)$ degeneracy —
rests on the clean-mock decoupling described above.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b2-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $\alpha$
  - IMF high-mass slope; drives **both** the mass histogram and the equipartition group masses/weights — the self-consistency hinge.
  - **recovered** (2.3)
* - $\delta$
  - Equipartition degree ($s_j=s\,\mu_j^{-\delta}$); $\tfrac12$ = full equipartition.
  - **recovered** (0.4)
* - $W_0$
  - LIMEPY central concentration.
  - **recovered** (5.0)
* - $J$, $g$
  - Number of IMF mass groups (4); LIMEPY truncation order fixed to the King limit ($g=1$).
  - known / fixed
* - $M_{\rm fixed}$
  - **Measured** total cluster mass anchoring the velocity scale $s=\sqrt{GM/(9r_c\mu_{\rm tot})}$ — treated as exactly known (see below).
  - known / fixed (data scalar)
* - mass range, $G$
  - Maschberger bounds $[0.1,20]\,M_\odot$ (class bounds = draw bounds, so no truncation correction); $G$ in model units.
  - known / fixed
* - $N$, bins, occupancy, quadrature
  - $10^5$ stars; 16 equal-count radial bins; per-cell occupancy floor `n_min=30`; quadrature ($I_p$ moments 256 pts, $g(W)$ table 256).
  - numerical choices
* - boxes
  - expit bounds α $(1.5,3.2)$, $W_0$ $(3,8)$, and **$\delta\in(0,0.7)$ — physical, not just numerical**: it caps short of the Spitzer-unstable $\delta\gtrsim0.9$ that crashes the ODE.
  - known/fixed bounds
* - MLE / NUTS / bias-grid
  - 3 dispersed Adam starts (300 steps); NUTS 300+600 (off by default); wrong-$\alpha$ grid {1.9…2.7} and robustness truths {1.9,2.3,2.7}.
  - numerical choices
```

```{important}
:label: imp-b2-decoupling
**The headline degeneracy-breaking depends on the clean-mock "Option A" decoupling
of the mass channel from the kinematics.** The per-star mass likelihood is evaluated
on a *separate, global* draw $m_{\rm obs}\sim\mathrm{Maschberger}(\alpha)$ over the
full mass range, with no per-star mass$\leftrightarrow$group$\leftrightarrow$velocity
correlation. This is what makes the mass channel a clean analytic log-pdf in
$\alpha$ alone (its Hessian has only an $\alpha\alpha$ entry), and *why* the mass
channel **pins $\alpha$ rather than rotating the $(\alpha,\delta)$ ellipse** — the
entire result. If real mass–velocity covariance were modeled, the mass channel would
also carry $\delta$ information and this clean separation would not hold. The total
mass $M_{\rm fixed}$ is likewise treated as an exactly-known anchor; in a real
cluster it is itself uncertain and partly the quantity one wants.
```

## Result 1 — joint recovery (freshly run, ALL PASS)

Measured 2026-06-11 ($N=10^5$, three dispersed Adam starts; exit 0):

```{list-table}
:header-rows: 1

* - Parameter
  - Truth
  - $\hat\theta \pm \hat\sigma$
  - Pull
* - $\alpha$ (IMF high-mass slope)
  - $2.300$
  - $2.293 \pm 0.004$
  - $-1.68$
* - $\delta$ (equipartition degree)
  - $0.400$
  - $0.397 \pm 0.034$
  - $-0.08$
* - $W_0$ (central concentration)
  - $5.000$
  - $4.990 \pm 0.021$
  - $-0.48$
```

All within $3\sigma$ (the heaviest, sparsest group $j=3$ has occupancy $1153 \ge
300$, so the fit is not starved). The MLE is a robust optimum — the two best of
the three dispersed starts agree in loss to $3\times10^{-3}$.

:::{figure} figures/imf_equip_fit.png
:label: sci-b2-fit
:width: 100%

**Joint MLE fit** (`demo_delta_recovery.py`). **(a)** Per-group binned
$\hat\sigma_{1\mathrm D,j}(r)$ (points, finite-$N$ errors) with the best-fit
binned-expectation curves — the equipartition ordering ($\sigma$ decreasing with
group mass) is visible. **(b)** The observed mass histogram with the fitted
Maschberger pdf at $\hat\alpha$.
:::

## Result 2 — the degeneracy the mass channel breaks

Kinematics *alone* cannot cleanly separate $\alpha$ from $\delta$: a steeper
$\sigma(m)$ can come from a larger equipartition degree *or* from an $\alpha$ that
reweights the group masses. The Fisher information quantifies it (freshly
measured):

```{list-table}
:header-rows: 1

* - Quantity
  - Kinematics-only
  - Joint (+ mass channel)
* - $\rho(\alpha,\delta)$ correlation
  - $-0.265$
  - $-0.059$
* - $\sigma_\alpha$
  - $0.0192$
  - $0.0041$
* - $\sigma_\delta$
  - $0.0357$
  - $0.0344$
* - $(\Delta\chi^2{=}4)$ ellipse area
  - $8.30\times10^{-3}$
  - $1.78\times10^{-3}$
```

Adding the mass histogram **pins $\alpha$ $4.7\times$ tighter** (ellipse area
ratio $4.67$) and collapses the $(\alpha,\delta)$ correlation from $-0.265$ toward
zero. Crucially the mass channel *pins* $\alpha$ rather than *rotating* the
ellipse — $\delta$'s width barely moves ($0.0357\to0.0344$); the gain is almost
entirely in $\alpha$.

:::{figure} figures/imf_equip_fisher.png
:label: sci-b2-fisher
:width: 90%

**Fisher degeneracy panel.** $\Delta\chi^2=4$ ($86.5\%$ in 2-D) ellipses in
$(\alpha,\delta)$: the broad, tilted kinematics-only ellipse vs the compact joint
ellipse. The mass channel removes the degeneracy along $\alpha$.
:::

## Result 3 — full posterior (NUTS)

A vendored blackjax No-U-Turn sampler (300 warmup + 600 samples) draws the full
posterior. Recorded measured run (`--run-nuts`, $\sim 52$ min wall):

- **0 divergent transitions**;
- posterior mean within **$1\sigma$ of the MLE per parameter**
  ($\alpha\ -0.05$, $\delta\ +0.27$, $W_0\ -0.38\ \sigma$);
- the target is made flat in $\theta$ by adding the $\sum\ln(d\theta/dz)$
  reparametrization Jacobian, so the corner is a faithful posterior, not a
  box-edge artifact.

:::{figure} figures/imf_equip_corner.png
:label: sci-b2-corner
:width: 85%

**NUTS posterior corner** in $(\alpha,\delta,W_0)$ with the MLE and truth
overlaid; unimodal, no divergences, posterior mean on the MLE.
:::

## Result 4 — wrong-IMF bias + robustness grid

What if you **assume the wrong** $\alpha$ and refit only the kinematics? Freezing
$\alpha$ at a grid of wrong values and refitting $(\delta, W_0)$ from kinematics
alone (5 seeds each) gives the bias curve. The recovered $\hat\delta$ **peaks at
the true $\alpha$** ($\hat\delta = 0.42 \approx \delta_{\rm true}$ at
$\alpha_{\rm assumed}=2.3$) and is biased **low** on either side — assuming the
wrong IMF slope corrupts the equipartition measurement. The linear sensitivity
$d\hat\delta/d\alpha = -0.005 \pm 0.020$ (seed-ensemble SE) is near zero precisely
*because* the response is peaked, not monotonic — it is **reported, not gated**
(no published reference value to assert against).

The companion **robustness grid** regenerates fresh truth datasets at
$\alpha_{\rm true}\in\{1.9, 2.3, 2.7\}$ and refits the full joint $(\alpha,\delta,
W_0)$ — every case recovers within $3\sigma$:

```{list-table}
:header-rows: 1

* - $\alpha_{\rm true}$
  - $\hat\alpha$
  - $\hat\delta$
  - $\hat W_0$
  - max $|$pull$|$
* - $1.9$
  - $1.903 \pm 0.003$
  - $0.411 \pm 0.033$
  - $4.984 \pm 0.019$
  - $0.97$
* - $2.3$
  - $2.300 \pm 0.004$
  - $0.396 \pm 0.033$
  - $5.011 \pm 0.021$
  - $0.51$
* - $2.7$
  - $2.700 \pm 0.005$
  - $0.428 \pm 0.030$
  - $5.006 \pm 0.024$
  - $0.96$
```

:::{figure} figures/imf_equip_bias.png
:label: sci-b2-bias
:width: 100%

**Wrong-IMF bias (left)** — $\hat\delta(\alpha_{\rm assumed})$ with seed scatter,
peaking at truth; **robustness grid (right)** — joint recovery across three
$\alpha_{\rm true}$, all within $3\sigma$.
:::

## Caveats

```{warning}
- **Clean-mock, Option A masses** — observed masses are a global IMF sample
  uncorrelated with the kinematic group (see the note above); no measurement
  errors, projection, or mass-dependent completeness.
- **The wrong-IMF curve is a sensitivity *measurement*, not a calibrated test** —
  reported with seed-ensemble uncertainty, never gated.
- **NUTS is non-divergent but not SBC-calibrated** — the credible-interval
  coverage has not been checked by simulation-based calibration; that is future
  work.
- **$\delta$ is fit on $[0, 0.7]$** — $\delta\gtrsim 0.9$ approaches the Spitzer
  instability (heavy stars decouple) and hard-crashes the equilibrium ODE solve,
  so the box stops short of it by construction.
- **The wrong-IMF bias is peaked at truth, not linear.** $\hat\delta$ is *maximal*
  at the true $\alpha$ and biased low for a wrong $\alpha$ in **either** direction,
  so the quoted slope $d\hat\delta/d\alpha\approx0$ is not "no bias" — read the
  curve's shape, not its local slope.
- **Truncation order and binning are fixed inputs.** Only $(\alpha,\delta,W_0)$
  vary; $g=1$ (King truncation) is assumed known, and the 16-bin / `n_min=30`
  scheme (tuned to keep the sparse heavy group resolved) is a choice the result
  depends on. Uncertainties are inverse-Hessian Gaussian approximations.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py            # MLE + Fisher (minutes)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery.py --run-nuts # + NUTS corner (~52 min)
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_delta_recovery_bias.py       # wrong-IMF curve + grid
```

## References

The multimass LIMEPY DF is {cite:t}`Gieles2015` with the equipartition
$m$-convention of {cite:t}`Peuten2017`; the $\sigma(m)$ equipartition relation and
its derived equipartition mass are {cite:t}`Bianchini2016`; the IMF is
{cite:t}`Maschberger2013`. The equipartition physics is developed on the
[multimass equipartition theory page](../10-theory/spatial-profiles/multimass-equipartition.md),
and the underlying equilibrium is validated at
[multimass equilibrium](../50-validation/multimass-equilibrium.md).
