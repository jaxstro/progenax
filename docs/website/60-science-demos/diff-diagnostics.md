---
title: Differentiable structure diagnostics (B10)
description: "Validate progenax's two differentiable cluster-structure surrogates against their exact combinatorial oracles: q_approx (a nearest-neighbor approximation for the Cartwright & Whitworth 2004 substructure Q) and lambda_msr_approx (a soft surrogate for the Allison et al. 2009 mass-segregation ratio). q_approx tracks exact Q in the substructure regime and its autodiff gradient matches finite differences; lambda_msr_approx has a separately defined nearest-neighbor sharp limit; its selected rank agreement does not make it the MST ratio."
---

# Differentiable structure diagnostics (B10)

The exact reference implementations use SciPy MSTs and hard subset
selections and do not expose a JAX derivative. MST length itself is
piecewise differentiable away from ties; the implementation and mathematical
questions are distinct. Progenax provides approximate JAX diagnostics. This
page records selected value, ranking and derivative checks from a retained
demo run; it does not qualify their use as general inference likelihoods.

| differentiable surrogate | exact oracle | statistic |
|---|---|---|
| `q_approx` (hard nearest-neighbor approximation) | `compute_q_parameter` (MST + scipy) | Cartwright & Whitworth (2004) $Q$ {cite:p}`Cartwright2004` |
| `lambda_msr_approx` (softmin + soft mass-cut) | `compute_lambda_msr` (MST) | Allison et al. (2009) $\Lambda_{\rm MSR}$ {cite:p}`Allison2009` |

## Inputs and assumptions

This demo **fits nothing** — it is a calibration check of differentiable surrogates
against exact oracles. Every quantity is a known/fixed generator setting, a swept
parameter, or a numerical tolerance.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b10-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $Q$ generators
  - Substructure sequence (clumpy $\to$ uniform sphere $\to$ $r^{-2}$ concentrated) spanning the CW04 range; **Plummer is deliberately excluded** ($Q>1$, outside the calibrated regime).
  - known / fixed (toy generators)
* - clump / segregation params
  - Clumpy generator ($k=8$ clumps, spread 0.06); bimodal segregation cluster ($n=400$, $n_{\rm massive}=20$, $0.5$ vs $10\,M_\odot$).
  - known / fixed
* - $\Lambda$ surrogate softness
  - `lambda_msr_approx` hyperparameters ($m_{\rm cut}=2.0$, $\tau=0.3$, $\beta=0.1$) that set the magnitude compression.
  - known / fixed
* - core-scale sweep
  - Segregation strength slid $1.0\to0.05$ (massive stars pulled into a shrinking core).
  - sweep input
* - $N$, seeds, FD step
  - 400 particles/cluster; 6 seeds averaged; finite-difference step $h=10^{-3}$ for the AD-vs-FD check.
  - numerical choices
* - gates
  - $Q$ full-sequence $<0.10$, substructure-regime $<0.06$, AD–FD rel-gap $<0.12$, $\Lambda$ Spearman $>0.8$, null $|\Lambda-1|<0.25$.
  - numerical choices
```

```{important}
:label: imp-b10-surrogates
**The soft surrogates are *not* drop-in replacements for the exact statistics —
they trade magnitude fidelity for differentiability.** `q_approx` is calibrated and
faithful ($|\Delta|<0.06$) only in the CW04 substructure regime $Q\lesssim0.8$, and
degrades toward high concentration in the displayed calibration. Both the
exact statistic and the implemented approximation remain defined for ordinary
nondegenerate inputs with $Q>1$; one is outside this calibration exposure.
`lambda_msr_approx` is **rank-faithful but magnitude-compressed** in the soft
regime (it rises to $\sim$3 where the exact MST $\Lambda$ reaches $\sim$20
in this example). Its sharp limit is a global nearest-neighbor distance ratio,
not the MST statistic. Ranking agreement does not establish gradient direction,
likelihood accuracy or inference calibration. Interpret it as a separately
defined diagnostic whose relationship to MST $\Lambda$ needs measurement.
```

## (a) Substructure $Q$ — calibration in the regime it is used

The CW04 parameter $Q = \bar m / \bar s$ (mean MST edge length over mean
separation) separates **substructured** ($Q<0.8$) from **centrally concentrated**
($Q>0.8$) clusters, anchored at $Q\approx0.79$ for a uniform sphere. Across a
sequence from a clumpy distribution through a uniform sphere to an $r^{-2}$
concentrated profile, `q_approx` tracks the exact $Q$:

```{list-table}
:header-rows: 1

* - distribution
  - exact $Q$
  - $q_{\rm approx}$
  - $|\Delta|$
* - clumpy (substructured)
  - $0.409$
  - $0.409$
  - $0.000$
* - uniform sphere
  - $0.774$
  - $0.824$
  - $0.051$
* - $r^{-2}$ concentrated
  - $0.924$
  - $1.010$
  - $0.085$
```

In the **substructure regime** $Q\lesssim0.8$ — the regime where `q_approx` is
*used* for substructure inference — it tracks the exact statistic to
$|\Delta|<0.06$, and it preserves the ordering throughout. The calibration degrades
mildly toward high central concentration ($Q\sim0.92$), and a steep Plummer
($Q>1$) is deliberately excluded as **outside the CW04-calibrated range**, where
the displayed calibration does not establish surrogate accuracy. The exact
MST statistic remains meaningful beyond $Q=1$.

## (b) Mass segregation — rank comparison with the MST statistic

Sweeping a bimodal-mass cluster's segregation strength (massive stars drawn into a
shrinking core), the soft `lambda_msr_approx` and exact `compute_lambda_msr` both
rise as the massive population concentrates. The rank correlation is **perfect
(Spearman $= 1.00$)**, and the unsegregated case returns $\Lambda\approx1$. In the
*soft* regime used for smooth gradients the surrogate is **magnitude-compressed**
(it rises to $\sim 3$ where the exact $\Lambda$ reaches $\sim 20$ at extreme
segregation). The Oracle-1 test compares its sharp limit with the hard
nearest-neighbor ratio, not with an MST. For example, positions $x=(0,1,3)$
with the first two objects selected have global NN distances $(1,1,2)$ and
a sharp NN ratio $4/3$, whereas averaging the three two-object subset MSTs
gives exact $\Lambda_{\rm MSR}=2$. No choice of vanishing softness makes
these definitions identical in general.

## (c) Derivative consistency of the approximate Q observable

With $q(p) = q_{\rm approx}(u^p\,\hat D)$ — raising each point's radius to the power
$p$, so larger $p$ is more centrally concentrated — the autodiff gradient
$\mathrm dq/\mathrm dp = 0.183$ matches the finite-difference slope $0.169$ at
$h=10^{-3}$. The reported **4.0%** uses the symmetric gap
$|g_{\rm AD}-g_{\rm FD}|/(|g_{\rm AD}|+|g_{\rm FD}|)$, not relative
error against the FD value. This one-step comparison concerns `q_approx`
itself, not the derivative of exact MST Q. It does not establish the cause
of the discrepancy or convergence as the FD step shrinks.

## Figure

:::{figure} figures/demo_diff_diagnostics.png
:label: sci-diff-diagnostics
:width: 100%

**Differentiable structure diagnostics** (`scripts/demo_diff_diagnostics.py`, ALL
PASS). **(a)** $q_{\rm approx}$ vs exact $Q$ across the substructure→concentration
sequence; the $\pm0.06$ band and the $y=x$ line, with the CW04 uniform-sphere anchor
$Q=0.79$ (dashed). **(b)** $\Lambda_{\rm MSR}$ vs segregation strength (log–log):
exact (blue) and the rank-faithful soft surrogate (vermilion), both rising as the
massive stars concentrate. **(c)** $q_{\rm approx}(u^p\hat D)$ vs the concentration
exponent $p$; the autodiff vs finite-difference gradients agree to 4% at $p=0.5$.
:::

## Caveats

```{warning}
- **Smooth + simple-clump axis only.** The substructured end of (a) is a Gaussian
  *blob* generator, not the experimental `gravoturb` **fractal-density field**.
  Full fractal-substructure inference lives in the repo-only `gravoturb` package
  (its own AC1–AC17 acceptance suite), not the released core.
- **`q_approx` is calibrated for $Q\lesssim0.8$.** It is the substructure-inference
  regime sampled here; accuracy outside the displayed families, sizes and
  concentrations is not established. There is no mathematical cutoff at $Q=1$.
- **`lambda_msr_approx` is rank-faithful, not magnitude-calibrated** in the soft
  regime. Its sharp limit is an NN ratio; use the exact MST estimator when
  the published absolute $\Lambda_{\rm MSR}$ is required. Ranking alone
  does not qualify a gradient-based objective.
- **Calibration is at one softness, one size, toy generators.** The compression is
  evaluated at fixed $(\tau,\beta,m_{\rm cut})=(0.3,0.1,2.0)$; the $Q$/$\Lambda$
  statistics are $N$-dependent and measured at a modest $N=400$ (6 seeds); the
  segregation test is a two-population ($0.5$ vs $10\,M_\odot$) toy, not a continuous
  IMF; and the $Q$ generators are sampling toys, not the named profiles' true DFs.
- **One FD step and one surrogate observable.** The displayed symmetric
  gradient gap is not an exact-MST derivative comparison or a step-convergence
  study. Tie/candidate boundaries and numerical regularization remain relevant.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_diff_diagnostics.py
```

## References

The substructure parameter $Q$ is {cite:t}`Cartwright2004`; the mass-segregation
ratio $\Lambda_{\rm MSR}$ is {cite:t}`Allison2009`. The surrogates and their
exact-limit convergence are documented on the
[JAX-native substructure $Q$](../20-architecture/jax-native-substructure-q.md) and
[mass-segregation](../10-theory/tidal-and-substructure/mass-segregation.md) pages.
