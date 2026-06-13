---
title: King concentration from star counts (B11)
description: "The methods showcase for the Poisson number-density channel: a single-population King model's concentration (W0, r_c) recovered from binned radial STAR COUNTS alone (no kinematics) by MLE + reverse-mode Poisson Fisher. Recovers both parameters and the King concentration c=log10(r_t/r_c), and exposes the strong W0-r_c degeneracy intrinsic to the count channel."
---

# King concentration from star counts (B11)

Every other demo in this series recovers structure from the *velocity* channel
$\sigma(r)$. This one recovers a King {cite:p}`King1966` cluster's concentration
from the observable a photometric survey actually delivers — the **radial star
counts** — and nothing else. It is the methods showcase for the Poisson
number-density channel (`binned_number_density` / `poisson_loglike` /
`poisson_fisher_information`).

## The physics being recovered

A King lowered-isothermal model is fixed by two structural parameters:

- $W_0$ — the dimensionless central potential (**concentration**). It sets the
  *shape* of the number-density profile in units of the core radius and the
  truncation ratio, summarized by the King concentration $c = \log_{10}(r_t/r_c)$.
- $r_c$ — the **core radius**, the physical *scale* at which the profile breaks.

So binned counts $N_k$ constrain both: the profile *shape* pins $W_0$, the physical
*break radius* pins $r_c$. The forward model is **Engine A** — a single King-limit
LIMEPY component {cite:p}`Gieles2015` ($g=1$): the coupled Poisson ODE gives
$\psi(r/r_c)$, and the King volume density is the closed form

```{math}
:label: b11-density
\hat n(r) = E_\gamma\!\left(g + \tfrac32,\, W(r)\right), \qquad W(r) = \psi(r/r_c),
```

(`limepy_density_hat`), differentiable in $(W_0, r_c)$ through `psi_grid`. Beyond
$r_t$ the potential $W\le 0$ and the density is **exactly zero** — the hard
truncation that pins the outer profile, and the regime where the Poisson model
(honest counting errors on low-occupancy bins) matters most.

## The likelihood

The expected counts are $\mu_k(W_0, r_c) = N_{\rm obs}\, p_k$, with $p_k$ the
fraction of the model's enclosed number in radial bin $k$ (the number-weighted
integral of $4\pi r^2\,\hat n(r)$ over the bin). The data are the **frozen** counts;
gradients flow through the model only. The per-bin Poisson log-likelihood is
$\sum_k\left(N_k\log\mu_k - \mu_k\right)$. The Fisher information is the
**reverse-mode** Poisson form $J^\top\mathrm{diag}(1/\mu)\,J$
(`poisson_fisher_information`): the Engine A diffrax solve carries a `custom_vjp`,
so `jax.hessian` would crash — exactly the constraint that makes the
[B2 demo](imf-equipartition.md) use Gauss–Newton.

## Result — freshly run, ALL PASS

Measured 2026-06-12 ($N=3\times10^4$ stars, $K=20$ log-spaced bins over
$[0.10, 16.8]$ pc, three dispersed Adam starts; wall $\approx 29$ s; exit 0).

```{list-table}
:header-rows: 1

* - Parameter
  - Truth
  - $\hat\theta \pm \hat\sigma$
  - Pull
* - $W_0$ (concentration)
  - $6.000$
  - $6.028 \pm 0.024$
  - $+1.15$
* - $r_c$ (core radius)
  - $1.000$ pc
  - $0.995 \pm 0.010$ pc
  - $-0.47$
```

Both within $1.2\sigma$. Supporting diagnostics, same run:

- **King concentration recovered.** $c = \log_{10}(r_t/r_c) = 1.263$ at the MLE
  vs $1.255$ at truth — the model's $c(W_0)$ relation is validated against
  King (1966) Table II on the [King validation page](../50-validation/physics-tests.md).
- **Robust optimum.** All three dispersed initializations converged to the
  *identical* loss $-2.02978\times10^{5}$.
- **Self-consistency.** The analytic prediction at truth matches the binned counts
  to $\max|N_k-\mu_k|/\sqrt{\mu_k} = 2.51$ over the 20 bins — no oracle/binning bias.

### The honest catch: counts alone are degenerate

The Gauss–Newton Fisher correlation is $\rho(W_0, r_c) = -0.91$. The number-density
profile alone leaves a **strong $W_0$–$r_c$ degeneracy**: a more concentrated model
($W_0\!\uparrow$) with a smaller core ($r_c\!\downarrow$) reproduces almost the same
count profile over the binned range. Both *marginals* are still tight (the demo
recovers each to $\lesssim 2\%$), but the parameters are far from independent — the
$2\sigma$ Fisher ellipse is a thin diagonal sliver (panel b). This is precisely the
degeneracy the *velocity* channel breaks: $\sigma(r)$ adds the gravitational-mass /
velocity-scale information that counts cannot supply, which is why the
multi-channel demos ([B2](imf-equipartition.md), [B3](halo-core.md)) pin their
parameters far more tightly. B11 isolates what the count channel *can* and *cannot*
do on its own.

## Figure

:::{figure} figures/demo_king_concentration.png
:label: sci-king-concentration
:width: 100%

**King concentration from star counts** (`scripts/demo_king_concentration.py`,
ALL PASS). **(a)** The radial number-density profile: binned counts per shell
volume (points, $\sqrt N$ errors) with the best-fit (vermilion) and truth (blue
dashed) King profiles, across the core-to-truncation dynamic range. **(b)** The
$2\sigma$ Fisher ellipse in $(W_0, r_c)$: a thin diagonal sliver
($\rho=-0.91$) — the strong count-channel degeneracy — with the truth ★ inside,
near the MLE.
:::

## Caveats

```{warning}
- **Clean mock, single family.** Truth and fit are the same Engine A King model;
  the data are 3-D radial counts with no projection, completeness, or
  contamination. This isolates the *count-channel inference*, not observational
  recovery.
- **Counts alone are degenerate** ($\rho(W_0,r_c)=-0.91$). The point of B11 is to
  *show* this, not hide it; breaking the degeneracy needs the velocity channel.
- **Inner $0.1$ pc excluded.** The flat King core inside the first bin edge
  ($0.1\,r_c$, $\sim$0.1% of the mass) is dropped from the fit; the normalization
  is over the binned range.
- **$W_0$ box capped at $7.5$.** Above $W_0\sim 8$ the Engine A diffrax solve hits
  its step limit; the fit is bounded to the safe range (truth $6$ is well interior).
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_king_concentration.py
```

## References

The King lowered-isothermal model is {cite:t}`King1966`; the LIMEPY family and the
$g=1$ King limit are {cite:t}`Gieles2015`. The Engine A construction and its
$c(W_0)$ validation against King (1966) Table II are documented on the
[populations](../10-theory/populations/index.md) and
[physics-tests](../50-validation/physics-tests.md) pages.
