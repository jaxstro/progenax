---
title: Binary-inflated dynamical mass (B12)
description: "Unresolved binaries inflate a cluster's line-of-sight velocity dispersion, biasing the virial/dynamical mass high (~24% at f_b=0.5 in a UFD-like system). A dispersion-only analysis cannot remove the bias — the (sigma_true, f_b) problem is rank-1 degenerate. A differentiable joint fit to the non-Gaussian wings of the velocity distribution returns an unbiased mass, with a Fisher forecast vs sample size N and RV precision eps. The kinematic companion to B4."
---

# Binary-inflated dynamical mass (B12)

A star cluster's **dynamical mass** is read off its velocity dispersion,
$M \propto \sigma^2 r_h / G$. But an unresolved binary contributes its *orbital*
motion to the measured velocity, widening the dispersion and biasing the mass
**high**. In low-dispersion systems — ultra-faint dwarfs, low-mass globular
clusters — the binary motion is a large fraction of $\sigma$ itself, so the bias
is large and has driven real debates about mass-to-light ratios and dark-matter
content.

This demo (a) **quantifies** the bias, (b) shows a **dispersion-only** analysis
*cannot* remove it — the $(\sigma_{\rm true}, f_b)$ problem is rank-1 degenerate —
and (c) demonstrates a **differentiable joint recovery** from the *non-Gaussian
wings* of the velocity distribution that returns an **unbiased** dynamical mass,
with a Fisher/CRLB forecast vs sample size $N$ and RV precision $\epsilon$.

It is the kinematic companion to [B4](binary-mass-function.md): B4 recovers the
binary fraction $f_b$ *photometrically* (from the unresolved-binary mass
function); B12 measures the dynamical mass *kinematically* in the presence of the
same binaries. Both reuse the Moe & Di Stefano (2017) $P$–$q$–$e$ statistics
{cite:p}`MoeDiStefano2017` and the Tout et al. (1996) ZAMS mass–luminosity
relation {cite:p}`Tout1996`.

## The forward model

Each star has a cluster centre-of-mass line-of-sight velocity
$v_{\rm COM}\sim\mathcal N(0,\sigma_{\rm true}^2)$. A fraction $f_b$ are
**unresolved binaries**: you measure one blended velocity, not two. The blend
sits at $v_{\rm obs}=v_{\rm COM}+\Delta$, where $\Delta$ is the internal orbital
motion projected onto the line of sight and **flux-weighted** by the two
components' luminosities,

```{math}
:label: b12-blend
\Delta = \frac{L_1\,v_{1,\rm los} + L_2\,v_{2,\rm los}}{L_1 + L_2},
\qquad L_i = L_{\rm ZAMS}(m_i; Z),
```

with $L_{\rm ZAMS}$ the Tout (1996) ZAMS relation (in-package via
`progenax.stellar`). Two limits make $\Delta$ intuitive:

- **$q\to1$ (equal mass):** $L_1=L_2$ and $v_1=-v_2$, so the blend **cancels**,
  $\Delta\to0$. Twins are nearly invisible contaminants.
- **$q\to0$ (extreme ratio):** the faint secondary adds no light, so you see the
  primary's reflex, $\Delta\to v_{1,\rm los}$ (an SB1).

The orbits $(P, q, e)$ are drawn from the Moe coupling; periods become semimajor
axes via Kepler's third law, and the orbit is sampled at a **uniform mean
anomaly** (uniform in *time*, so eccentric orbits are correctly weighted toward
slow apocenter passages). The distribution of $\Delta$ over a large binary pool is
the contamination kernel $K_{\rm orb}$ — crucially **independent of
$\sigma_{\rm true}$**, so the observed velocity density factorizes,

```{math}
:label: b12-mixture
p(v) = (1-f_b)\,\mathcal N(0,\sigma_{\rm true}^2+\epsilon^2)
     + f_b\,\big[\mathcal N(0,\sigma_{\rm true}^2+\epsilon^2)\circledast K_{\rm orb}\big](v),
```

where $\epsilon$ is the per-star RV precision (added in quadrature). Because
$K_{\rm orb}$ does not depend on $\sigma_{\rm true}$, it is precomputed **once**.
The expected binned counts $\mu_k(\sigma_{\rm true},f_b)=N\!\int_{{\rm bin}\,k}p$
are differentiable in both parameters, and $(\sigma_{\rm true},f_b)$ are recovered
by a per-bin Poisson MLE (`scripts/_demo_inference.py`), with the Fisher from the
Poisson information.

```{note}
$K_{\rm orb}$ has standard deviation $\approx 3.3$ km/s but **kurtosis $\approx
100$** — extraordinarily heavy-tailed. Most binaries (wide, long-period)
contribute almost nothing; a rare population of ultra-short-period systems reaches
$>100$ km/s. This heavy tail is the whole story: it both *biases* the naive mass
and, through the non-Gaussian wings it imprints on $p(v)$, makes the bias
*removable*.
```

## Result — freshly run, ALL GATES PASS

UFD-like regime: $\sigma_{\rm true}=5$ km/s, $f_b=0.5$, $N=1500$ RV stars,
$\epsilon=1$ km/s, metal-poor $Z=10^{-3}$, $r_h=30$ pc (exit 0; wall $\approx
90$ s).

```{list-table}
:header-rows: 1

* - Gate
  - Expected
  - Measured
* - 1 — bias exists
  - $M_{\rm naive}/M_{\rm true}>1.10$
  - $1.28$ ($\sigma_{\rm obs}=5.66$ km/s)
* - 2 — dispersion-only degenerate
  - rank-1 (cond $>10^8$)
  - cond $=2.3\times10^{16}$
* - 3a — joint recovery unbiased
  - $M_{\rm ratio}\approx1$, $<3\sigma$
  - $\sigma=4.97\pm0.13$, $M_{\rm ratio}=0.99$
* - 3b — full-distribution full-rank
  - cond $<10^6$
  - cond $=878$
* - 4 — RV-precision floor
  - precision degrades, mass unbiased
  - $\sigma(\sigma_{\rm true})$: $0.12\to0.24$; $\max|M-1|=0.03$
* - 5 — null ($f_b=0$)
  - $\hat f_b<0.05$
  - $\hat f_b=0.02$, $\sigma=4.97$
* - 6 — AD-vs-FD gradient integrity
  - rel-err $<10^{-4}$
  - $1.6\times10^{-10}$
```

### The bias, and why dispersion-only cannot fix it

The dispersion grows by the variance budget
$\sigma_{\rm obs}^2 = \sigma_{\rm true}^2 + f_b\,\mathrm{Var}(K_{\rm orb}) +
\epsilon^2$, so the naive virial mass is biased high by exactly
$(\sigma_{\rm obs}/\sigma_{\rm true})^2$ — **$1.28\times$ at $f_b=0.5$**
([](#fig-b12-bias)). Crucially, this is a **systematic**: averaging more stars at
fixed $f_b$ measures the *wrong* mass more precisely, it does not remove the bias.

A dispersion-only analysis sees a single number $\sigma_{\rm obs}$, so any
$(\sigma_{\rm true}, f_b)$ on the curve
$\sigma_{\rm true}^2 + f_b\,\mathrm{Var}(K_{\rm orb}) = \text{const}$ fits equally
well — an infinite **ridge** ([](#fig-b12-constraint), orange). Its $2\times2$
Fisher is the outer product of one gradient: exactly **rank-1** (one zero
eigenvalue, cond $=2.3\times10^{16}$). One number cannot separate two unknowns.

### How the wings break the degeneracy

The fix is to fit the **whole shape** of $p(v)$, not just its width. The Gaussian
*core* constrains one combination of $(\sigma_{\rm true},f_b)$; the non-Gaussian
*wings* — the heavy $K_{\rm orb}$ tail no single-star Gaussian can mimic
([](#fig-b12-distribution)) — constrain a *different* combination. Two independent
constraints turn the open ridge into a closed **ellipse**
([](#fig-b12-constraint), blue): the full-distribution Fisher is **full-rank**
(cond $=878$, a $10^{13}$ improvement). The recovered mass is **unbiased**,
$\sigma=4.97\pm0.13$ km/s, $M_{\rm ratio}=0.99$ vs the naive $1.28$ (verified
unbiased over 24 realizations: $\langle\sigma\rangle=5.00$,
$\langle f_b\rangle=0.50$).

### The RV-precision floor

Because $\epsilon$ is *known* and folded into [](#b12-mixture), the recovered mass
stays **unbiased at every $\epsilon$** ([](#fig-b12-eps-floor)) — even at
$\epsilon=5$ km/s $\approx\sigma_{\rm obs}$. What degrades is **precision**:
$\sigma(\sigma_{\rm true})$ doubles as $\epsilon$ washes out the wing signature,
and the **detectable** binary fraction $f_b\,P(|\Delta|>\epsilon)$ collapses from
$0.21$ to $0.02$. The honest scope: *you can only correct for the binaries you can
detect, and that detectable fraction shrinks as the spectrograph gets noisier* —
but the correction you do make is unbiased. The Fisher forecast
([](#fig-b12-fisher-vs-n)) shows both uncertainties fall as $N^{-1/2}$; $f_b$ (and
hence the bias correction) is the harder parameter, since the wings it lives in are
sparsely populated.

## Figures

:::{figure} figures/demo_binary_dynamical_mass_distribution.png
:label: fig-b12-distribution
:width: 100%

**The non-Gaussian wings** (log-$y$). The mock $v_{\rm los}$ (grey) and the
single+binary mixture (blue) follow the binary wings out to $\pm35$ km/s, where the
single-only Gaussian (orange) predicts essentially nothing. The shaded regions
($|v|>2.5\,\sigma_{\rm obs}$) are where the binary excess lives — the signal the
joint fit uses to break the degeneracy.
:::

:::{figure} figures/demo_binary_dynamical_mass_bias.png
:label: fig-b12-bias
:width: 100%

**The bias.** $M_{\rm naive}/M_{\rm true}=(\sigma_{\rm obs}/\sigma_{\rm true})^2$
vs binary fraction $f_b$ (points, 20 realizations each) tracks the analytic
variance budget $1+(\epsilon^2+f_b\,\mathrm{Var}\,K_{\rm orb})/\sigma_{\rm true}^2$
(line). At $f_b=0.5$ the virial mass is biased $1.28\times$ high.
:::

:::{figure} figures/demo_binary_dynamical_mass_constraint.png
:label: fig-b12-constraint
:width: 100%

**Degeneracy and its breaking.** A dispersion-only analysis constrains only the
degenerate **ridge** (orange) — every point gives the same $\sigma_{\rm obs}$. The
full velocity distribution localizes both parameters to the finite $1$/$2\sigma$
**ellipses** (blue); truth (star) sits on the ridge and inside the ellipse. The
ellipse is elongated *along* the ridge — the residual correlation the wings only
partially lift.
:::

:::{figure} figures/demo_binary_dynamical_mass_eps_floor.png
:label: fig-b12-eps-floor
:width: 100%

**The RV-precision floor.** The recovered-mass precision
$\sigma(\sigma_{\rm true})$ (blue, left) and $\sigma(f_b)$ (sky, right) grow with
RV precision $\epsilon$, while the detectable fraction $f_b\,P(|\Delta|>\epsilon)$
(vermilion, right) collapses. The mass itself stays unbiased
($M_{\rm ratio}\approx1$) at all $\epsilon$ — only the precision degrades.
:::

:::{figure} figures/demo_binary_dynamical_mass_fisher_vs_N.png
:label: fig-b12-fisher-vs-n
:width: 100%

**Survey forecast.** Fisher $\sigma(\sigma_{\rm true})$ and $\sigma(f_b)$ vs sample
size $N$ (log–log), both scaling as $N^{-1/2}$ (dotted guide). Small-$N$ sits above
the guide — the few-star regime is more degenerate than the asymptotic CRLB.
:::

## Caveats

```{warning}
- **Moe statistics assumed known.** $K_{\rm orb}$ is a fixed template built from the
  Moe (2017) $P$–$q$–$e$ coupling at fixed IMF, $q_{\min}$, and metallicity; its own
  uncertainty (and age/metallicity dependence) is not marginalized. A misspecified
  kernel would re-introduce bias.
- **Single epoch.** Multi-epoch RVs see the orbital phase change between visits —
  additional information this single-epoch demo does not use (it marginalizes over
  phase). Real binary surveys would do better than this forecast.
- **One isotropic global dispersion.** No $\sigma(r)$ profile, anisotropy, or
  rotation; perfect membership (no foreground beyond $\epsilon$). A genuinely
  non-Gaussian intrinsic velocity distribution (radial orbits, tides) could
  partially mimic binary wings — a confound this idealized demo does not separate.
- **ZAMS photometry only.** The flux weighting uses the Tout (1996) main-sequence
  $L(m)$ from `progenax.stellar`; evolved (giant-dominated) systems change
  $K_{\rm orb}$'s shape. Full SED/bandpass photometry is the `fluxax` package's job.
- **Heavy tail is grid-limited.** $K_{\rm orb}$ is histogrammed on $\pm150$ km/s;
  the rarest ultra-short-period systems beyond that are dropped (negligible by
  count, but they are the high-leverage wing stars in the extreme-precision regime).
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_dynamical_mass.py
```

## References

The binary $P$–$q$–$e$ coupling is {cite:t}`MoeDiStefano2017`; the ZAMS
mass–luminosity relation is {cite:t}`Tout1996`. The Moe and companion models are
documented on the [binary statistics](../10-theory/imfs/multiplicity-statistics.md)
theory pages; the photometric sibling is [B4](binary-mass-function.md).
