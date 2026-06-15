---
title: Tidal radius from the outskirts (B7)
description: "A stretch demo reusing the Poisson number-density channel: an EFF young cluster's truncation radius r_t is recovered from the count-limited OUTSKIRTS (where an honest Poisson likelihood matters), and converted to a Galactocentric distance via the Jacobi radius. The per-bin Fisher information shows 93% of the r_t constraint comes from the outer bins, and r_t is made differentiable by clipping the cumulative at r_t (a hard truncation has zero gradient through its boundary)."
---

# Tidal radius from the outskirts (B7)

A cluster's **tidal (Jacobi) radius** $r_t$ marks where the Galaxy strips stars,
and it is written in the **outskirts** — the sparse, count-limited edge of the
density profile. This stretch demo reuses the Poisson number-density channel
([B11](king-concentration.md)) to recover $r_t$ from those few outer stars, and
turns it into the cluster's **Galactocentric distance**.

## The physics

The Elson–Fall–Freeman (1987) {cite:p}`ElsonFallFreeman1987` profile
$\rho(r) = (1 + r^2/a^2)^{-\gamma/2}$ is sharply truncated at $r_t$. The scale
radius $a$ is pinned by the **inner** profile (where almost all the stars are), so
it is held fixed; $r_t$ is read from the **outer** counts. Bins are placed out past
$r_t$ so the outermost are empty: a too-large $r_t$ predicts stars in those empty
bins (penalized by the Poisson $-\mu$ term), a too-small $r_t$ cannot explain the
observed outer stars. The Poisson channel is honest there; a Gaussian-on-log-count
would be ill-defined. (A *steep* halo self-truncates — its density at $r_t$ is
already $\sim 0$, so $r_t$ sits in the noise; this uses a shallow, YMC-typical
$\gamma=2.5$ so stars reach $r_t$.)

```{note}
**Making $r_t$ differentiable.** A hard `jnp.where(r < r_t, ρ, 0)` truncation has
**zero gradient** through its boundary condition, so $\partial\mu/\partial r_t = 0$
and the Fisher information is singular. The demo instead integrates the
*untruncated* cumulative once, then evaluates it at bin edges **clipped at $r_t$**
and normalizes by $\mathrm{enc}(r_t)$ — so $r_t$ flows through the differentiable
`minimum` and `interp`, giving a real $\partial\mu/\partial r_t$ for a *sharp*
truncation. (The same lesson underlies progenax's custom-JVP `apply_tidal_truncation`.)
```

## Inputs and assumptions

The fit recovers **one parameter**, the truncation/tidal radius $r_t$; the scale
radius, the profile family, and both masses in the Jacobi conversion are assumed
known. The mass assumptions matter most — they, not the count statistics, would
dominate a real $R_{\rm gal}$ error budget.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b7-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $r_t$
  - EFF truncation = Jacobi/tidal radius — **the science target**, read from the count-limited outer bins and converted to $R_{\rm gal}$.
  - **recovered** (12.0 pc)
* - $a$
  - EFF scale radius; pinned by the *inner* profile and held fixed (jointly fitting $(a,r_t)$ from counts is degenerate).
  - known / fixed (1.0 pc)
* - $\gamma$
  - EFF outer slope; shallow enough that stars reach $r_t$ (a steep $\gamma$ self-truncates and hides $r_t$).
  - known / fixed (2.5)
* - $M_{\rm gal}$
  - Interior Galaxy (point-mass) for the Jacobi $\to R_{\rm gal}$ conversion.
  - known / fixed ($5\times10^{10}\,M_\odot$)
* - $M_{\rm cl}$
  - Cluster mass entering $R_{\rm gal}=r_t(3M_{\rm gal}/M_{\rm cl})^{1/3}$ — summed from the IMF draw, treated as exactly known.
  - known / fixed ($\sim10^4\,M_\odot$, derived)
* - IMF
  - Maschberger ($\alpha=2.3$, $0.08$–$100\,M_\odot$) — fixes $M_{\rm cl}$; the mass spectrum does *not* enter the count fit (positions only).
  - known / fixed
* - $N$, bins, $r_{\rm hi}$ factor, boxes, MLE
  - $2\times10^4$ stars; 22 geometric bins from $r_{\rm lo}=0.2$ pc to $1.3\,r_t$ (so outer bins are **empty** — the Poisson $-\mu$ term is what pins $r_t$); 3000-pt cumulative-EFF grid; $r_t\in(6,18)$ pc; 3 Adam starts.
  - numerical choices
```

```{important}
:label: imp-b7-assumptions
**The scale radius $a$ is held fixed, and the constraint lives in the empty outer
bins.** Jointly fitting $(a,r_t)$ from number counts alone is degenerate — the
normalized profile shape trades them off — so $a$ is asserted to be pinned by the
star-rich inner profile and $r_t$ is read purely from the outskirts (bins extending
to $1.3\,r_t$, mostly empty, supply $\sim$93% of the Fisher information via the
Poisson $-\mu$ penalty). The conversion to Galactocentric distance then assumes
$M_{\rm gal}$ and $M_{\rm cl}$ are **error-free**, so $\sigma(R_{\rm gal})/R_{\rm
gal}=\sigma(r_t)/r_t$ inherits only the counting uncertainty — in reality a
cluster-mass error (M/L, unseen low-mass stars, remnants) would dominate.
```

## Result — freshly run, ALL PASS

Measured 2026-06-12 ($N=2\times10^4$, $\gamma=2.5$, true $r_t=12$ pc; wall $\approx 5$ s).

```{list-table}
:header-rows: 1

* - quantity
  - value
* - $r_t$
  - $12.01 \pm 0.04$ pc  (truth $12.0$, pull $+0.24$)
* - $r_t$ Fisher info from the outer bins ($r>0.6\,r_t$)
  - **93%** — the count-limited outskirts pin $r_t$
* - forecast
  - $\sigma(r_t)\propto N^{-1/2}$
```

The per-bin Fisher information spikes at the truncation **edge** (panel b): $r_t$ is
constrained almost entirely by the handful of outermost bins, which is precisely
why the Poisson treatment of those low-count bins matters.

### From $r_t$ to the Galactocentric distance

The recovered $r_t$ *is* the Jacobi/tidal radius {cite:p}`King1962`,

```{math}
:label: b7-jacobi
r_t = R_{\rm gal}\left(\frac{M_{\rm cl}}{3 M_{\rm gal}}\right)^{1/3}
\;\Longrightarrow\;
R_{\rm gal} = r_t\left(\frac{3 M_{\rm gal}}{M_{\rm cl}}\right)^{1/3},
```

with $\sigma(R_{\rm gal})/R_{\rm gal} = \sigma(r_t)/r_t$. For the measured
$M_{\rm cl}\approx1.2\times10^4\,\Msun$ and a representative interior
$M_{\rm gal}=5\times10^{10}\,\Msun$, the recovered $r_t$ gives

```{math}
R_{\rm gal} = 2.79 \pm 0.01\ \mathrm{kpc},
```

(round-trips through `jacobi_radius`). So a cluster's *orbit* is encoded in its
faint outer star counts.

## Figure

:::{figure} figures/demo_tidal_radius.png
:label: sci-tidal-radius
:width: 100%

**Tidal radius from the outskirts** (`scripts/demo_tidal_radius.py`, ALL PASS).
**(a)** The EFF number-density profile (counts, $\sqrt N$ errors) with the MLE fit
and the recovered truncation $\hat r_t$. **(b)** The per-bin Fisher information for
$r_t$ on a log-$r$ axis: it **spikes at the truncation edge** — 93% of the
constraint comes from the count-limited outskirts. **(c)** Forecast
$\sigma(r_t)\propto N^{-1/2}$, annotated with the derived Galactocentric distance.
:::

## Caveats

```{warning}
- **$a$ is fixed, $r_t$ recovered.** The scale radius is pinned by the inner
  profile and held; jointly fitting $(a, r_t)$ from counts alone is degenerate
  (the normalized shape trades them off). The headline is the truncation.
- **Sharp truncation, complete census.** The EFF cut is hard and every star is
  counted; real outskirts have foreground/membership contamination and a soft,
  orbit-phase-dependent edge.
- **Point-mass Galaxy for the Jacobi step.** $M_{\rm gal}$ is a fixed representative
  interior mass; a realistic mass profile and an eccentric orbit shift $R_{\rm gal}$.
  The point is the *method* — outer counts → $r_t$ → orbit.
- **$M_{\rm cl}$ assumed exactly known.** Only $\sigma(r_t)$ propagates into
  $\sigma(R_{\rm gal})$; a real cluster-mass uncertainty would dominate it.
- **$\gamma$ and the EFF family are fixed at truth.** A wrong outer slope or a
  non-EFF profile would bias $r_t$ (only a shallow enough $\gamma$ lets stars reach
  $r_t$); profile-family misspecification is not explored here (contrast B6).
- **3-D radii, complete outskirts.** True 3-D radii are binned (no projection to
  surface density), and the result leans heavily on a few near-empty outer bins —
  robustness to a handful of interlopers there is not quantified.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_tidal_radius.py
```

## References

The EFF profile is {cite:t}`ElsonFallFreeman1987`; the tidal/Jacobi radius is
{cite:t}`King1962`. The Jacobi-radius derivation and progenax's
`apply_tidal_truncation` are on the
[tidal & substructure](../10-theory/tidal-and-substructure/tidal.md) page; the
Poisson count channel is shared with [B11](king-concentration.md).
