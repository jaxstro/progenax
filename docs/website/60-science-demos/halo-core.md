---
title: Halo + core recovery (B3)
description: "Engine B inference: a Plummer halo (Osipkov-Merritt anisotropic) plus an EFF gamma=5 core, sampled in shared-potential equilibrium, with the mass split t, anisotropy radius r_a and halo scale r_h recovered jointly by MLE + Gauss-Newton Fisher from binned sigma(r) and the halo beta(r)."
---

# Halo + core recovery (B3)

A real cluster is not one population. This demo builds a **two-family** cluster —
a spatially extended, kinematically **anisotropic halo** plus a compact,
**isotropic core** — in one shared self-consistent potential, and asks whether
three structural parameters can be recovered jointly from its kinematics:

- $t$ — the **halo mass fraction** (the core carries $1-t$),
- $r_a$ — the halo's **Osipkov-Merritt anisotropy radius**,
- $r_h$ — the halo's **Plummer half-mass radius**.

The truth model (the science-headline mix of the
[Engine B validation suite](../50-validation/engine-b-eddington.md)):

```{math}
:label: b3-truth
\underbrace{\text{Plummer}(r_h=2\,\text{pc})}_{\text{halo, } 60\%,\ m_j=0.5\,\Msun,\ r_a=3\,\text{pc}}
\ +\
\underbrace{\text{EFF}(a=0.8\,\text{pc},\,\gamma=5,\,r_t=9\,\text{pc})}_{\text{core, } 40\%,\ m_j=1.0\,\Msun,\ \text{isotropic}}
```

sampled with $N = 3\times10^4$ stars. The EFF core shape $(a,\gamma,r_t)$, the
stellar-mass labels $m_j$, and the core's isotropy are held fixed; only
$(t, r_a, r_h)$ are recovered.

## The physics being recovered

**Anisotropy (Osipkov-Merritt).** The halo's DF depends on energy and angular
momentum only through $Q = E - L^2/(2 r_a^2)$, which makes the velocity
ellipsoid isotropic in the centre and increasingly radial outside $r_a$. The
**Binney anisotropy** has the closed form

```{math}
:label: b3-beta
\beta(r) \equiv 1 - \frac{\sigma_t^2}{2\sigma_r^2}
        = \frac{r^2}{r^2 + r_a^2},
```

so $\beta\to 0$ as $r\to 0$ (isotropic) and $\beta\to 1$ as $r\to\infty$ (radial).
The core, with $r_a\to\infty$, has $\beta\equiv 0$. The OM curve {cite:p}`Merritt1985`
is the second observable, and it is what pins $r_a$.

**Dispersion.** The 1-D dispersion of each component is built from the speed
moments of its Eddington DF $f_j$ in the shared potential, with the OM stretch
folded in:

```{math}
:label: b3-sigma
\sigma_{1\mathrm D, j}^2(r) = \frac{\langle v^2\rangle_j(r)}{3},
\qquad
\langle v^2\rangle_j(r) = \frac{m_{2,j}(r)}{m_{0,j}(r)}
\left(\frac13 + \frac23\,\frac{1}{1 + r^2/r_{a,j}^2}\right),
```

where $m_{p,j}(r) = \int_0^{\sqrt{2\Psi}} w^{p} f_j\!\big(\Psi - \tfrac12 w^2\big)\,dw$
are the stretched-frame speed moments. This is the **verbatim** oracle the Engine
B validation suite checks against sampled velocities. Physical units come from
the engine's own velocity scale $v = \sqrt{G M_{\rm sampled}/(4\pi\mu)}\,s$,
anchored to the *measured* total mass (never an input $M$).

**Why $\sigma(r)$ and $\beta(r)$ together recover all three.** The halo scale
$r_h$ sets where $\sigma(r)$ falls off; the mass split $t$ sets the relative
amplitude and shape of the two components' dispersion profiles; $r_a$ is read
almost entirely from the rising $\beta(r)$. The three are nearly orthogonal —
borne out by the near-zero Fisher correlations below.

## Result — joint MLE + Fisher (freshly run, ALL PASS)

Measured 2026-06-11 ($N=3\times10^4$, three dispersed Adam starts; exit 0):

```{list-table}
:header-rows: 1

* - Parameter
  - Truth
  - $\hat\theta \pm \hat\sigma$
  - Pull $(\hat\theta-\theta_{\rm true})/\hat\sigma$
* - $t$ (halo mass fraction)
  - $0.600$
  - $0.602 \pm 0.008$
  - $+0.22$
* - $r_a$ (halo OM radius)
  - $3.000$ pc
  - $3.051 \pm 0.053$ pc
  - $+0.96$
* - $r_h$ (Plummer half-mass)
  - $2.000$ pc
  - $2.015 \pm 0.029$ pc
  - $+0.51$
```

All three within $1\sigma$, comfortably inside the $3\sigma$ recovery gate.
Supporting diagnostics, all measured in the same run:

- **Robust optimum.** All three dispersed initializations converged to the
  *identical* loss $19.1479$ — the recovered minimum is not an initialization
  artifact.
- **Realizability.** Rebuilding Engine B at $\hat\theta$ gives DF positivity
  margins $f_{\min,j} = [0.085,\ 1.16\times10^{-4}]$, both $\ge -10^{-3}$ — the
  recovered model is a genuine equilibrium, not a negative-DF fiction.
- **Self-consistency.** The analytic prediction at truth matches the binned data
  to $\max|{\rm dev}/{\rm SE}| = 2.34$ over all 48 populated cells — no
  systematic oracle/binning bias.
- **Near-orthogonality.** Gauss-Newton Fisher correlations
  $\rho(r_a, r_h) = +0.03$, $\rho(t, r_a) = -0.04$ (condition number $13.4$):
  the three parameters are independently constrained.

## Figure

:::{figure} figures/halo_core.png
:label: sci-halo-core
:width: 100%

**Halo + core recovery** (`scripts/demo_halo_core.py`, ALL PASS). **(a, b)**
Per-component $\sigma_{1\mathrm D}(r)$: mock data (points, finite-$N$ error bars)
with the best-fit binned-expectation curve and the truth curve, for the Plummer
halo and EFF core. **(c)** Halo Binney anisotropy $\hat\beta(r)$ rising along the
OM curve $r^2/(r^2+r_a^2)$ ($\hat r_a = 3.05$ pc vs truth $3.0$), with the
isotropic-core reference at $\beta=0$. **(d)** $2\sigma$ Fisher ellipses in
$(r_a, r_h)$ (and inset $(t, r_a)$): the truth ★ sits inside, near the MLE; the
ellipses are nearly axis-aligned, reflecting the near-zero correlations.
:::

## Caveats

```{warning}
- **Clean mock, shared family.** Truth and fit are the same Engine B model; the
  data are the full 3-D $\sigma(r)$ and $\beta(r)$ with no projection, no
  measurement error, and no incompleteness. This isolates the *inference
  machinery*, not observational recovery.
- **The domain $r_t$ is fixed, not recovered.** The shared potential's outer
  radius is set to the (fixed) EFF extent $r_t = 9$ pc — both at truth and inside
  the traced fit — so a recovered $r_h$ never has to re-derive the domain. This
  is the model the truth draw uses; it is a construction choice, stated plainly.
- **Hard-truncation edge.** The EFF core is hard-truncated ($\rho(r_t)>0$), an
  edge offset no ergodic $f(E)$ carries exactly; the resulting small
  sub-equilibrium is quantified (not eliminated) on the
  [Engine B validation page](../50-validation/engine-b-eddington.md). It does not
  bias the parameter recovery, which is gated on the $3\sigma$ pulls above.
- **Cost is MLE-compile-dominated.** End-to-end $\sim 3.7$ min: the warm
  likelihood-and-gradient eval is $0.14$ s, but each of the three Adam runs is a
  jit-compiled 400-step `lax.scan` whose body rebuilds Engine B and backprops, so
  the three *scan compiles* dominate.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_halo_core.py
```

## References

The EFF profile is {cite:t}`ElsonFallFreeman1987`; the Plummer model
{cite:t}`Plummer1911`; Osipkov-Merritt anisotropy {cite:t}`Merritt1985`;
Eddington inversion follows Binney & Tremaine (2008). The Engine B construction
and its realizability/anisotropy validation are documented at
[Engine B (Eddington)](../50-validation/engine-b-eddington.md) and the
[multi-component theory pages](../10-theory/populations/index.md).
