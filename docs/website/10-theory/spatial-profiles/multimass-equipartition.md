---
title: "Multimass equipartition saturation — the derived m_eq"
description: "Why the GZ15 multimass lowered-isothermal model saturates: the escape-speed ceiling at low mass, the deep-well m^(-1/2) branch at high mass, and the DERIVED (not fitted) equipartition mass m_eq = m-bar (g+5/2)(g+7/2)/W0 — validated fit-free against progenax's Engine A, plus the reference-LIMEPY parity result and the m-bar convention bookkeeping."
---
# Multimass equipartition saturation — the derived $m_{\rm eq}$

:::{admonition} Status — validated, zero new parameters
:class: tip
Everything on this page is a property of the **standard** Engine A multimass
model ([the lowered-model family](lowered-model-family.md)) — no new code, no
new parameter, **no fitting anywhere**. Evidence:
`scripts/validate_limepy_reference.py` (ours-vs-canonical-LIMEPY parity, 6
configs, ALL PASS) and `scripts/validate_equipartition_saturation.py`
(fit-free saturation gates, ALL PASS), both run 2026-06-11. Per-paper notes:
[Gieles & Zocchi 2015](../../99-bibliography/per-paper/gieles-zocchi-2015.md),
[Peuten et al. 2017](../../99-bibliography/per-paper/peuten-2017.md),
[Bianchini et al. 2016](../../99-bibliography/per-paper/bianchini-2016.md).
:::

:::{admonition} Who this page is for
:class: note
**Audience:** researchers studying mass segregation and energy equipartition in multimass clusters who want the *derived* (not fitted) saturation physics of the GZ15 model.
**Prerequisites:** the [lowered-model family](lowered-model-family.md) (the multimass Engine A model and the $w_j$ / $\delta$ machinery) and the [King profile](king.md) for the single-mass baseline.
**You'll get:** why $\sigma(m)$ saturates (an escape-speed ceiling at low mass, a deep-well $m^{-1/2}$ branch at high mass), the derived equipartition mass $m_{\rm eq}$, and why fitted literature $m_{\rm eq}$ values are window-dependent.
:::

## The model and the $\delta$ ansatz

In the {cite:t}`Gieles2015` (GZ15) multimass lowered-isothermal model, every
mass component $j$ shares **one** self-consistent potential but carries its
own velocity scale and anisotropy radius via power-law mass scalings
(GZ15 eqs 24–26; {cite:t}`Peuten2017` eqs 3–5):

```{math}
:label: multimass-delta-ansatz
s_j = s\,\mu_j^{-\delta},
\qquad \hat r_{a,j} = \hat r_a\,\mu_j^{\eta},
\qquad \mu_j \equiv \frac{m_j}{\bar m},
\qquad \bar m = \sum_j m_j\,\alpha_j ,
```

with $\bar m$ the **central-density-weighted** mean mass (GZ15 eq 26;
$\alpha_j = \rho_{0j}/\rho_0$, $\sum_j \alpha_j = 1$). The component density
rides the shared potential at the rescaled depth $W_j = \mu_j^{2\delta} W_0$
(GZ15 eq 29). For the traditional $\delta = \tfrac12$, $m_j s_j^2$ is constant
across bins by construction — but the model is **not** in energy
equipartition, and the way it falls short is the physics of this page.

## The honest sub-equipartition physics: why $\sigma(m)$ saturates

The *observable* central 1-D dispersion of component $j$ is not $s_j$. In
closed form (Bianchini et al. 2016 eq A1, with the paper's missing
$^{1/2}$ restored — [typo note](../../99-bibliography/per-paper/bianchini-2016.md)):

```{math}
:label: sigma-m-closed-form
\hat\sigma_{1d,j0} \;=\; \mu_j^{-\delta}
\left[\frac{E_\gamma\!\big(g+\tfrac52;\, \mu_j^{2\delta}\hat\phi_0\big)}
           {E_\gamma\!\big(g+\tfrac32;\, \mu_j^{2\delta}\hat\phi_0\big)}\right]^{1/2},
\qquad \hat\phi_0 = W_0 .
```

The two limits of {eq}`sigma-m-closed-form` are two different physical regimes:

- **Low mass — an escape-speed ceiling, not "equipartition failure".** A light
  component ($\mu_j \ll 1$) sees the shared well at the vanishing depth
  $W_j = \mu_j^{2\delta} W_0 \to 0$: it is **maximally tidally truncated**, a
  DF cut ever closer to the escape energy (Merritt 1981). Its dispersion is
  therefore pinned near the central escape speed — which is a property of the
  *shared* potential and is **mass-independent**. Hence $\sigma(m)$ flattens:
  the local slope $\eta(m) \equiv -\,d\ln\sigma/d\ln m \to 0$.
- **High mass — the deep-well thermalized branch.** A heavy component
  ($\mu_j \gg 1$) has $W_j \gg 1$: effectively an *untruncated* isothermal
  sphere, for which $\hat\sigma_{1d,j0} \to \mu_j^{-\delta}$ exactly — with
  $\delta = \tfrac12$, the true-equipartition $\sigma \propto m^{-1/2}$
  (GZ15 §3.2.1), $\eta \to \tfrac12$.

Both limits are verified **analytically** (autodiff of
{eq}`sigma-m-closed-form`, nothing fitted) in
`scripts/validate_equipartition_saturation.py`: for the 20-bin
$[0.1, 1]\,M_\odot$ model ($\delta=\tfrac12$, $g=1.5$),
$\eta(m_{\rm min}) = 0.026$–$0.039$ across $W_0 \in \{5, 7, 9\}$ (flat,
escape-limited end), $|\eta - \tfrac12| \le 1.1\times10^{-16}$ at $\mu = 20$
(deep well, float64 noise), and $\eta(m)$ is strictly monotone increasing.
The closed form itself is gated against the multimass solver's independent
quadrature oracle: max relative difference $1.5$–$1.7\times10^{-7}$
(gate $10^{-6}$).

## The crossover is DERIVED, not fitted

{cite:t}`Bianchini2016` introduced the exponential fitting function
$\sigma(m) = \sigma_0 \exp(-\tfrac12 m/m_{\rm eq})$ for $m \le m_{\rm eq}$,
$\propto m^{-1/2}$ above (their eq 3), with $m_{\rm eq}$ — the mass above
which the cluster is in full equipartition — as the fitted "degree of
equipartition". Their Appendix A shows the crossover **emerges from the GZ15
DF itself**: Taylor-expanding {eq}`sigma-m-closed-form` for low mass and
matching linear terms (A2 ↔ A3) identifies

```{math}
:label: meq-derived
\boxed{\;m_{\rm eq} \;=\; \bar m\,\frac{\big(g+\tfrac52\big)\big(g+\tfrac72\big)}{\hat\phi_0}\;}
```

[↗ model card](#card-meq-derived)

so $m_{\rm eq}$ is fixed by the mean mass, the truncation order $g$, and the
central concentration $\hat\phi_0 = W_0$ — **not** a free input. Note
$m_{\rm eq}/\bar m \propto 1/W_0$: **more concentrated clusters reach
equipartition down to lighter stars**, the model-side face of Bianchini's
headline observational result that $m_{\rm eq}$ tracks dynamical age
($n_{\rm eq} = T_{\rm age}/T_{\rm rc}$; relaxed clusters → smaller
$m_{\rm eq}$).

progenax validates {eq}`meq-derived` **fit-free**: since we hold the exact
curve {eq}`sigma-m-closed-form`, the headline gate is the exact $\mu \to 0$
identity — the *local saturation mass* $m/(2\eta(m))$, with $\eta$ by
autodiff, must approach the derived $m_{\rm eq}$. Measured at $\mu = 10^{-3}$
($g = 1.5$, 20 equal-$M_j$ log bins on $[0.1,1]\,M_\odot$):

```{list-table} Derived-m_eq identity (`scripts/validate_equipartition_saturation.py`, 2026-06-11, ALL PASS; gate 0.5%).
:header-rows: 1

* - $W_0$
  - $\bar m\ (M_\odot)$
  - derived $m_{\rm eq}$ {eq}`meq-derived`
  - $m/(2\eta)$ at $\mu=10^{-3}$
  - rel. dev.
* - 5
  - 0.5779
  - 2.3114
  - 2.3089
  - 0.11%
* - 7
  - 0.6595
  - 1.8842
  - 1.8813
  - 0.15%
* - 9
  - 0.7375
  - 1.6388
  - 1.6356
  - 0.19%
```

::::{figure} ../../50-validation/figures/equipartition_saturation.png
:label: fig-equipartition-saturation
:width: 100%

**Fit-free equipartition saturation** across $W_0 \in \{5,7,9\}$: the exact
Engine A $\sigma(m)$ {eq}`sigma-m-closed-form` (points: quadrature oracle;
line: closed form), the derived $m_{\rm eq}$ {eq}`meq-derived` marking the
crossover, the flat escape-limited low-mass end, the deep-well $m^{-1/2}$
branch, and the dashed tangent exponential
$\sigma_0\exp(-m/2m_{\rm eq})$ illustrating the finite-window fit bias
(see the caution below).
::::

## Sharper than exponential — a caution for literature comparisons

The exponential of Bianchini eq 3 matches the DF only to **first** order. The
exact quadratic Taylor coefficient of the DF's $\sigma(m)$ — eq A2's printed
$(6+3a-4a^2)/(8a^2b^2c)\cdot\hat\phi_0^2$ term, $a=g+\tfrac52$,
$b=g+\tfrac72$, $c=g+\tfrac92$ — is **negative** for all cluster-relevant
$g$, *opposite in sign* to the exponential's $+1/(8m_{\rm eq}^2)$: the DF
saturates **sharper than exponentially**. (A3's "first terms of the Taylor
expansion of the exponential" holds for the linear term only.)

:::{admonition} Caution — fitted m_eq values are window-dependent
:class: warning
Because the true curve bends *below* the tangent exponential, an eq-3
exponential fit over a finite mass window recovers an $m_{\rm eq}$ **biased
low**, by an amount that depends on the window. Measured in a one-off
windowed-fit analysis (2026-06-11; recorded in the validation script's
docstring — the gated validation itself is deliberately fit-free): over
$\mu = m/\bar m \in [0.1/\bar m,\, 1/\bar m]$ per model — the 20-bin
$[0.1,1]\,M_\odot$ window, $\approx[0.14, 1.7]$ across the $W_0$ grid since
$\bar m$ varies — such fits recover **~0.5×** the DF-derived
{eq}`meq-derived` (0.50–0.53 across $W_0 \in \{5,7,9\}$). So when
comparing a *fitted* literature $m_{\rm eq}$ with the derived one:
**window-match first** (fit eq 3 to the model over the data's mass window), or
— future work — skip eq 3 entirely and fit the exact differentiable
$\sigma(m;\, g, W_0, \bar m)$ {eq}`sigma-m-closed-form` directly to the
kinematic data.
:::

The convergence to the identity is itself the evidence that the bias is pure
finite-mass curvature — the local saturation mass over the derived
$m_{\rm eq}$ rises monotonically to 1 as $\mu \to 0$ (printed by the script:
e.g. $W_0=7$: 0.54 at $\mu=0.5$, 0.86 at $\mu=0.1$, 0.985 at $\mu=0.01$,
0.9985 at $\mu=10^{-3}$).

## The $\bar m$ conventions, and parity with canonical LIMEPY

Three $\bar m$-related facts the bookkeeping must keep straight
([Peuten et al. 2017](../../99-bibliography/per-paper/peuten-2017.md) §2):

1. **GZ15 eq 26 (= progenax):** $\bar m$ is the **central-density-weighted**
   mean, $\bar m = \sum_j m_j\,\alpha_j$ — what Engine A's `bar_m` computes
   and what every equation on this page uses.
2. **The reference LIMEPY code's default** is the **global** mean mass
   (faster, especially with black-hole bins); it supports the GZ15 convention
   via `meanmassdef='central'`.
3. **The two describe the same physical model**, related exactly by Peuten
   eqs 8–9 (in the corrected direction — heavier reference mass ⇒ colder
   reference scale ⇒ deeper $W_0^*$):
   $W_0^{*} = W_0\,(\bar m^{*}/\bar m)^{2\delta}$,
   $\hat r_a^{*} = \hat r_a\,(\bar m^{*}/\bar m)^{\eta+\delta}$.

The parity harness (`scripts/validate_limepy_reference.py`) runs the canonical
numpy/scipy LIMEPY (repo SHA `ef2a479`, pinned ephemeral env, outputs cached
with provenance) with `meanmassdef='central'`, so both codes share one
$(W_0, \bar m)$ convention and no eq 8–9 translation is needed. Six configs —
single-mass $g=1$ (King) and $g=1.5$; 2-component $\delta=0.5$; the 4-bin
Maschberger IMF ($\delta=0.4$); and the 2-component anisotropic pair
$\hat r_a = 5$, $\eta \in \{0, 0.5\}$ — compared on scale-invariant
quantities only:

```{list-table} Ours-vs-reference-LIMEPY parity (2026-06-11, ALL PASS; worst case across the 6 configs, per quantity).
:header-rows: 1

* - Quantity (scale-invariant)
  - worst measured
  - gate
* - density shape $\max|\Delta[\rho_j(r)/\rho_j(0)]|$
  - $1.3\times10^{-5}$
  - $2$–$5\times10^{-5}$
* - dispersion shape $\max|\Delta[\sigma_j(r)/\sigma_j(0)]|$
  - $8.4\times10^{-5}$
  - $2\times10^{-5}$–$3\times10^{-4}$
* - central density fractions $\max|\Delta\alpha_j|$
  - $4.9\times10^{-5}$
  - $10^{-8}$–$5\times10^{-4}$
* - realized mass fractions $\max|\Delta M_j/M|$
  - $7.1\times10^{-5}$
  - $10^{-8}$–$5\times10^{-4}$
* - concentration $|\Delta\log_{10}(r_t/r_0)|$
  - $1.5\times10^{-4}$
  - $5\times10^{-4}$
* - half-mass radius $|\Delta r_h|/r_h$
  - $6.3\times10^{-6}$
  - $10^{-4}$
```

All scale-invariant quantities agree to $\lesssim 1.5\times10^{-4}$ — Engine A
is a faithful (and differentiable) reimplementation of the published model.
Gates were measured first, then frozen with honest headroom (~3–10× the
measured deviations); the skip-if-absent regression test is
`tests/validation/test_limepy_reference_parity.py`.

::::{figure} ../../50-validation/figures/limepy_reference_parity.png
:label: fig-limepy-reference-parity
:width: 100%

**Reference parity** for the representative multimass config: per-component
$\rho_j(r)$ and $\sigma_j(r)$, ours (curves) vs canonical LIMEPY (points),
with residual strips.
::::

## What is deliberately NOT here: `meq` and `zeta`

The published LIMEPY *code* carries two extra knobs absent from the GZ15 and
Peuten et al. (2017) equations: `meq` (a low-mass softening
$\mu_j = (m_j + m_{\rm eq})/\bar m$, whose docstring citation of "GZ15 eq 24"
is incorrect — eq 24 is the pure power law) and `zeta` (a high-mass $s_j^2$
decoupling motivated by the Spitzer instability). Both are **deferred** in
progenax (decision 2026-06-11): they are *fitting freedom*, not the
saturation mechanism — as this page shows, the standard model **already
saturates**, with the crossover {eq}`meq-derived` set by $(\bar m, g, W_0)$.
If a future fit needs to decouple the equipartition degree from $g/W_0$
(`meq`) or model a dark-remnant subsystem (`zeta`), they will be added as
default-off kwargs and documented honestly as **code heuristics, not
published equations**.

(reproduce)=

## Reproduce

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_limepy_reference.py
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_equipartition_saturation.py
```

## Implementation, validation & references

- **In code:** `src/progenax/cluster/multicomponent.py` — the standard
  Engine A multimass model (`bar_m`, the closed-form
  $\hat\sigma_{1d,j0}$, and `component_virial_ratios`); no new code or
  parameter is introduced for this page. See the
  [`MultiComponentCluster` API](../../30-api/cluster.md).
- **Validated in:** [multimass equilibrium](../../50-validation/multimass-equilibrium.md);
  the fit-free saturation and reference-LIMEPY parity gates are reproduced
  by the scripts in the [Reproduce](#reproduce) block above
  (`scripts/validate_equipartition_saturation.py`,
  `scripts/validate_limepy_reference.py`).
- **Primary sources:** {cite:t}`Gieles2015` (the multimass DF and
  $\delta$ ansatz, eqs 24–29); {cite:t}`Peuten2017` (multimass methods +
  $\bar m$-convention translation, eqs 3–5, 8–9); {cite:t}`Bianchini2016`
  (the $\sigma(m)$ relation, its Appendix-A derivation, and the
  $m_{\rm eq}$–dynamical-age relation). Full notes in the bibliography:
  [Gieles & Zocchi 2015](../../99-bibliography/per-paper/gieles-zocchi-2015.md) ·
  [Peuten et al. 2017](../../99-bibliography/per-paper/peuten-2017.md) ·
  [Bianchini et al. 2016](../../99-bibliography/per-paper/bianchini-2016.md).
