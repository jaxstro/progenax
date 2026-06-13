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
{cite:p}`Tout1996` (imported from the `fluxax.photometry` sibling). The observed
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
  from the `fluxax` sibling — a clean main-sequence stand-in. Real surveys add
  band-dependent blending, extinction, and unequal evolutionary states; those live
  in `fluxax` proper, not this demo.
- **`fluxax` is a private, unlocked experimental dependency.** progenax is decoupled
  from it (the demos are local), so install it to run B4:
  `uv pip install -e ../fluxax --no-deps`.
- **f_b only.** The demo recovers the binary fraction and *shows* (not fits) that the
  $q$-coupling drives the bias; jointly recovering the $q$-distribution shape is a
  natural extension. The recovery assumes a known IMF and resolution limit.
- **Clean census + flat $f_b$.** A constant binary fraction and complete catalogue;
  Moe's own mass-dependent $f_b(m_1)$ and selection are not modelled here.
```

## How to run

```bash
uv pip install -e ../fluxax --no-deps   # one-time: the Tout ZAMS photometry
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_mass_function.py
```

## References

The binary $P$–$q$–$e$ coupling is {cite:t}`MoeDiStefano2017`; the ZAMS
mass–luminosity relation is {cite:t}`Tout1996`. The Moe and companion models are
documented on the [binary statistics](../10-theory/imfs/multiplicity-statistics.md)
theory pages.
