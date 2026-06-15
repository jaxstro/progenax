---
title: Unresolved-binary mass function (B4)
description: "The headline binary demo: unresolved binaries blend into one photometric source of inflated mass, distorting the observed stellar mass function. Recover the binary fraction f_b from that distortion -- and show it requires the correct Moe & Di Stefano (2017) period-mass-ratio coupling, because the blended (close) binaries are a high-q subset an independent-(P,q) model mis-models, biasing f_b by ~5% (3.6 sigma at survey scale)."
---

# Unresolved-binary mass function (B4)

A binary that a survey cannot resolve appears as a **single, brighter** source, and
its inferred mass is biased high — so unresolved binaries distort the observed
stellar mass function. This demo recovers the binary fraction $f_b$ from that
distortion, and makes the central point: **getting $f_b$ right requires the correct
Moe & Di Stefano (2017) period–mass-ratio coupling** {cite:p}`MoeDiStefano2017`,
because the blended (close) binaries are a biased, high-$q$ subset that an
independent-$(P,q)$ model gets wrong.

## The forward model

Each binary has a primary $m_1$ (from a Maschberger IMF), a mass ratio $q$
($m_2 = q\,m_1$) and a period $P$; the semimajor axis $a$ follows Kepler's third
law. A survey **resolves** wide pairs and **blends** close ones, $a < a_{\rm crit}$
(here $50\,\mathrm{AU}$):

```{math}
:label: b4-blend
\text{single} \to m_1, \qquad
\text{resolved} \to \{m_1, m_2\}, \qquad
\text{blended} \to m_{\rm obs} = L^{-1}\!\big(L(m_1) + L(m_2)\big) > m_1,
```

where $L(m)$ is the **Tout et al. (1996) ZAMS mass–luminosity relation**
{cite:p}`Tout1996` (provided in-package by `progenax.stellar`). The observed
mass function is a linear mixture in $f_b$,

```{math}
:label: b4-mixture
\mu_k(f_b) = N_{\rm sys}\big[(1-f_b)\,S_k + f_b\,B_k\big],
```

with the single template $S_k$ (the IMF) and the binary template $B_k$ (per-binary
expected catalogue stars after resolution + blending) precomputed from a large
pool. $f_b$ is recovered by a per-bin Poisson MLE, with the Fisher variance from the
(linear, ODE-free) information.

## The controlled coupling test

The "independent" comparison uses the **same pool with $q$ shuffled against $P$** —
this preserves the $q$ and $P$ marginals *exactly* and changes only the $P$–$q$
correlation. So any bias is attributable to the coupling alone. The truth is
Moe-coupled; we fit $f_b$ with the (correct) Moe template and the (wrong)
independent template.

## Inputs and assumptions

The fit recovers **one parameter**, the binary fraction $f_b$; everything else is
an **assumed-known input**. The recovered-vs-known split is the key onboarding
lesson — the demo's punchline (that the $P$–$q$ coupling must be right) is a
statement about one of those *assumed* inputs.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b4-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $f_b$
  - Binary fraction — **the science target**, recovered from the Poisson mixture $\mu_k=N_{\rm sys}[(1-f_b)S_k+f_b B_k]$ (`FB_BOX=(0.05,0.95)`).
  - **recovered** (truth 0.5)
* - Moe $P$–$q$–$e$ coupling
  - The joint orbital model (`MoeJointOrbit`) that builds the *correct* binary template $B_k$; its $P$–$q$ correlation is the whole point.
  - known / fixed (assumed correct)
* - $q_{\min}$
  - Minimum mass ratio in `MoeDiStefano2017Full`; truncates the secondary masses feeding the blend.
  - known / fixed (0.1)
* - $a_{\rm crit}$
  - Resolution limit: pairs with semimajor axis $a<a_{\rm crit}$ **blend**, wider ones **resolve** (selects which binaries distort the MF).
  - known / fixed (`A_CRIT_AU=50` AU)
* - $Z$
  - Metallicity for the Tout ZAMS $L(m)$ used to compute the blend mass $m_{\rm obs}=L^{-1}(L_1+L_2)$.
  - known / fixed (`Z_MET=0.02`, solar)
* - IMF
  - Maschberger ($\alpha=2.3$, $0.08$–$100\,M_\odot$): draws primaries $m_1$, hence the single template $S_k$ and the blend masses.
  - known / fixed
* - $N_{\rm sys}$
  - Systems in the mock cluster; sets the Poisson counts and the **significance** of the wrong-model systematic ($\propto\sqrt{N_{\rm sys}}$).
  - known / fixed ($3\times10^5$)
* - $N_{\rm pool}$, `M_BINS`, `SEED`
  - Template pool size ($6\times10^5$), mass-function bins (30 log bins), and PRNG seeds (templates / data / $q$-shuffle).
  - numerical choices
```

```{important}
:label: imp-b4-coupling
**An unbiased $f_b$ requires the correct $P$–$q$ coupling — it cannot be inferred
from the marginals.** The blended subset is selected by $a<a_{\rm crit}$, i.e. by
*short period*, and under the Moe coupling short periods are preferentially
**high-$q$** (median $q=0.54$ for blends vs $0.48$ when $q$ is shuffled against
$P$). The blend bump is therefore set by the *correlation*, which the shuffle
destroys while preserving the $P$ and $q$ marginals exactly. Fitting with an
independent-$(P,q)$ template biases $f_b$ low by $\sim$5% — a **systematic** whose
fractional size is fixed but whose significance grows as $\sqrt{N_{\rm sys}}$. The
coupling is an *assumed-known model input*, not something the masses reveal.
```

## Result — freshly run, ALL PASS

Measured 2026-06-12 ($N_{\rm sys}=3\times10^5$, $N_{\rm pool}=6\times10^5$,
$a_{\rm crit}=50$ AU, solar $Z$; wall $\approx 10$ s; exit 0).

**The mechanism.** The blended (close-pair) subset is **more equal-mass under Moe**
than under the decoupled model — median $q = 0.540$ vs $0.479$ — because Moe couples
short periods to high $q$. That is the whole effect.

```{list-table}
:header-rows: 1

* - model fitted
  - recovered $f_b$
  - vs truth $0.500$
* - Moe template (correct)
  - $0.5016 \pm 0.0074$
  - $+0.22\sigma$  ✓
* - independent template (wrong)
  - $0.4749 \pm 0.0070$
  - $-3.6\sigma$  (≈5% low)
```

Ignoring the $P$–$q$ coupling biases $f_b$ **low by ~5%**. This is a *systematic*
(a wrong-model error), so its fractional size is fixed while its significance grows
as $\sqrt{N_{\rm sys}}$: only $-1.5\sigma$ at $N_{\rm sys}=5\times10^4$, but
$-3.6\sigma$ at the $3\times10^5$ survey scale here. At a real cluster survey it is
unambiguous. Gate summary:

```{list-table}
:header-rows: 1

* - Check
  - Gate
  - Status
* - mechanism: blended median $q$ (Moe > independent)
  - coupling present
  - **PASS**
* - self-consistency (Moe template @ truth)
  - $<4\sigma$ Poisson ($2.76$)
  - **PASS**
* - $f_b$ recovery (Moe template)
  - $<3\sigma$ ($+0.22$)
  - **PASS**
* - $f_b$ bias (independent template)
  - $>3\sigma$ ($-3.6$)
  - **PASS**
```

## Figure

:::{figure} figures/demo_binary_mass_function.png
:label: sci-binary-mass-function
:width: 100%

**Unresolved-binary mass function** (`scripts/demo_binary_mass_function.py`, ALL
PASS). **(a)** The observed mass function (black) decomposed into singles (blue) and
the binary contribution (vermilion) — the blended pairs push a bump to higher
inferred mass. **(b)** The mechanism: the blended subset's median $q$ is higher under
the Moe coupling (vermilion) than the $P$-shuffled independent model (sky) at the
same marginals. **(c)** Recovered $f_b$: the Moe template recovers the truth (dashed)
while the independent template is biased low.
:::

## Caveats

```{warning}
- **Photometry is the Tout (1996) ZAMS relation** ($L$–$M$ at fixed metallicity),
  from `progenax.stellar` (a startrax placeholder) — a clean main-sequence stand-in.
  Real surveys add band-dependent blending, extinction, and unequal evolutionary
  states; full SED/bandpass photometry is the `fluxax` package's job, not this demo.
- **No external dependency.** The ZAMS relations are in-package, so B4 runs in a
  clean environment / CI — nothing to install beyond progenax.
- **f_b only.** The demo recovers the binary fraction and *shows* (not fits) that the
  $q$-coupling drives the bias; jointly recovering the $q$-distribution shape is a
  natural extension. The recovery assumes a known IMF and resolution limit.
- **Clean census + flat $f_b$.** A constant binary fraction and complete catalogue;
  Moe's own mass-dependent $f_b(m_1)$ and selection are not modelled here.
- **$a_{\rm crit}$ is a single sharp, period-only cut.** Blending is a hard step at
  $a<a_{\rm crit}$ on the *physical* semimajor axis — no distance, projection, or
  flux-contrast dependence, and **eccentricity is drawn but unused** (resolution
  depends only on $a$, so eccentric pairs that would be resolvable near apocenter
  are not modelled).
- **Self-consistent by construction.** Templates and data share the same IMF,
  $q_{\min}$, $Z$ (=solar), $a_{\rm crit}$ and pool family; only the $P$–$q$
  coupling is varied between the right/wrong models. The mechanism is *shown*, not
  stress-tested against IMF / $a_{\rm crit}$ / metallicity misspecification.
- **No photometric noise.** The blend mass is the exact $L^{-1}(L_1+L_2)$ with no
  measurement scatter; binning is the only smoothing. The $f_b$ uncertainty is the
  inverse-Hessian (`fisher_cov`) Gaussian approximation, and the MLE is seeded at
  the truth (a clean-start convergence check, not a multimodality test).
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_mass_function.py
```

## References

The binary $P$–$q$–$e$ coupling is {cite:t}`MoeDiStefano2017`; the ZAMS
mass–luminosity relation is {cite:t}`Tout1996`. The Moe and companion models are
documented on the [binary statistics](../10-theory/imfs/multiplicity-statistics.md)
theory pages.
