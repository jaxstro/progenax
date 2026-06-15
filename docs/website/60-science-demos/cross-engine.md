---
title: Cross-engine agreement (B1)
description: "One King model built two ways — Engine A (DF-defined lowered-isothermal + coupled Poisson ODE) and Engine B (density-defined Eddington inversion) — shown to agree on rho(r), sigma_1d(r) and f(E) with freshly measured KS and dispersion gates."
---

# Cross-engine agreement (B1)

progenax builds multi-component equilibria two genuinely different ways, and the
first demo checks that they describe the *same physical cluster*.

- **Engine A** (`MultiComponentCluster.from_components`) starts from a
  **distribution function**: the lowered-isothermal / LIMEPY family
  {cite:p}`Gieles2015`. The density is whatever that DF produces in its own
  self-consistent potential, found by integrating the coupled Poisson ODE.
- **Engine B** (`MultiComponentCluster.from_density_profiles`) starts from a
  **prescribed density** (here a `KingProfile`), builds the shared potential by
  one cumulative-trapezoid Poisson pass, and recovers the DF by **Eddington
  inversion**.

If both engines are correct, the King model `(W₀=5, g=1, r_c=1 pc)` built either
way must sample the same radial profile, the same dispersion profile, and the
same energy distribution. Nothing forces this but the physics — the two code
paths share no quadrature, no ODE, and no DF evaluation.

## The two constructions

**Engine A — the lowered-isothermal DF.** For a single mass component with
truncation order $g$, the DF is a lowered (energy-shifted) Maxwellian
({cite:t}`Gieles2015`, eq. 2),

```{math}
:label: b1-lowered-df
f(E) \;\propto\; E_g\!\left(\frac{E}{s^2}\right),
\qquad
E_g(x) = e^{x}\,P(g, x)\ \ (x>0),\quad 0\ \text{otherwise},
```

with $E = \Psi(r) - \tfrac12 v^2$ the (positive, bound) relative energy, $s$ the
velocity scale, and $P(g,x)=\gamma(g,x)/\Gamma(g)$ the regularized lower
incomplete gamma. The index is **$g$ itself** (the density uses $g+\tfrac32$).
The single parameter $g$ slides continuously between classical models —
$g=0$ Woolley, **$g=1$ King**, $g=2$ Wilson — and at $g=1$ the lowered
exponential collapses to $E_1(x) = e^x - 1$, the textbook King DF. The density
$\hat\rho(\Psi)=\int f\,d^3v$ closes Poisson's equation, solved as a coupled ODE
in $(\Psi, d\Psi/dr)$ — full derivation on the
[lowered-model-family](../10-theory/spatial-profiles/lowered-model-family.md)
page.

**Engine B — Eddington inversion.** Given the *same* King density as a
prescribed shape, the shared potential $\Psi(r)$ follows from Poisson's equation
by quadrature, and the ergodic DF is recovered by the Eddington formula (Binney
& Tremaine 2008, eq. 4.46b),

```{math}
:label: b1-eddington
f(E) = \frac{1}{\sqrt8\,\pi^2}
\left[\int_0^{E}\frac{d^2\rho}{d\Psi^2}\,\frac{d\Psi}{\sqrt{E-\Psi}}
      + \frac{1}{\sqrt E}\left.\frac{d\rho}{d\Psi}\right|_{\Psi=0}\right].
```

The two routes invert each other in principle — A maps DF$\to$density, B maps
density$\to$DF — so agreement is a true round-trip cross-validation.

## Inputs and assumptions

This demo **fits nothing** — there are no recovered parameters. It builds one King
model two ways and compares them, so every quantity is either a known/fixed model
input or a numerical choice.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b1-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $W_0$
  - King central dimensionless potential (concentration); the shared model both engines build (`W0`).
  - known / fixed (5.0)
* - $g$
  - LIMEPY truncation order; $g=1$ is the classic King DF (`GG`).
  - known / fixed (1.0)
* - $r_c$
  - King core radius — the length scale (`R_C`).
  - known / fixed (1.0 pc)
* - $\alpha_j, w_j, m_j$
  - Single mass component (weight, velocity-scale weight, mass label all $=1$), so the two engines compare like-for-like.
  - known / fixed (one component)
* - $G$
  - Gravitational constant (STELLAR units); sets the physical velocity scale.
  - known / fixed
* - $N$
  - Stars drawn from **each** engine to measure the sampled radial CDF and binned $\sigma$, at the **same** PRNG key so residuals isolate model (not shot) differences.
  - numerical choice ($2\times10^4$, seed 0)
* - $\sigma$-bins, oracle grids, quadrature, gates
  - 6 bins over the 5%–90% radial quantiles; 300–400-pt analytic-oracle grids; 400-pt speed-moment quadrature; gate thresholds (KS $<0.02$, $|\sigma_B/\sigma_A-1|<0.02$, $Q_j=0.5\pm3\times10^{-3}$).
  - numerical choices
```

```{important}
:label: imp-b1-conventions
**The two engines are handed the *same physics under two incompatible
conventions*, so the comparison is fair only as a shape comparison.** Engine A's DF
lives on a dimensionless lowered-potential argument $\hat E\in[0,W_0]$; Engine B's
Eddington $f$ lives on physical relative energy $E\in[0,\Psi_0]$. They are
reconciled only through the linear map $\hat E=(W_0/\Psi_0)E$ and a peak
normalization, and the $\rho(r)$ residual is taken after re-normalizing each curve
to its central value. The agreement claim therefore rests on the *convention-free*
gates — the sampled-CDF KS distance and the binned dispersion ratio — while the
$f(E)$ overlay is a diagnostic that already carries a $\sim$5% edge residual *by
construction of the matching*, not a tested gate.
```

## What is measured (freshly run, ALL PASS)

Both engines are sampled at $N = 2\times10^4$ with the **same** PRNG key, so the
draws are strongly correlated and the residuals isolate genuine *model*
differences rather than shot noise. Measured 2026-06-11 (`key=0`):

```{list-table}
:header-rows: 1

* - Check
  - Measured
  - Gate
* - Tidal radius $r_t$ (A vs B)
  - $10.8054$ pc (both)
  - consistency
* - Theory virial $Q_j = T_j/|W_j|$ — Engine A
  - $0.50019$
  - $0.5 \pm 3\times10^{-3}$
* - Theory virial $Q_j$ — Engine B
  - $0.50000$
  - $0.5 \pm 3\times10^{-3}$
* - Radial Kolmogorov–Smirnov distance (sampled CDF)
  - $1.5\times10^{-4}$
  - $< 0.02$
* - Max $|\sigma_B/\sigma_A - 1|$ (interior bins)
  - $3.5\times10^{-4}$
  - $< 0.02$
* - $\rho(r)$ shape max fractional diff (diagnostic)
  - $7.95\times10^{-13}$
  - —
* - $f(E)$ peak-matched max diff (diagnostic)
  - $5.05\times10^{-2}$
  - —
```

The two independent engines agree on the sampled radial CDF to a KS distance of
$1.5\times10^{-4}$ and on the dispersion profile to $3.5\times10^{-4}$ — two
orders of magnitude inside the $0.02$ gate. The density *shape* agrees to
$8\times10^{-13}$ (essentially the floating-point floor, since both reduce the
same King profile after re-normalizing to the central value).

```{note}
The virial $Q_j = 0.5$ check is **necessary but not sufficient**: the Clausius
identity $2T_j + W_j = 0$ holds for *any* positive DF in a consistent
$(\Psi, d\Psi/dr)$ pair, so it tests self-consistency, not inversion
correctness. The KS and $\sigma$-dev anchors carry the cross-validation weight —
they would fail if either engine's DF were wrong even though $Q_j$ stayed $0.5$.
```

## Figure

:::{figure} figures/cross_engine.png
:label: sci-cross-engine
:width: 100%

**Cross-engine King agreement** (`scripts/demo_cross_engine.py`, ALL PASS).
**(a)** $\rho(r)/\rho_0$ for Engine A (DF-defined) and Engine B
(density-defined) with a residual strip — coincident to $8\times10^{-13}$ after
central re-normalization. **(b)** $\sigma_{1\mathrm D}(r)$ from the two
independent oracles plus both sampled profiles ($N=2\times10^4$, same key);
$\max|\sigma_B/\sigma_A-1| = 3.5\times10^{-4}$. **(c)** $f(E)$ overlay on the
dimensionless energy axis $\hat E = (W_0/\Psi_0)E$ — the lowered-isothermal DF
shape (A) and the Eddington $f$-row (B), peak-matched.
:::

## Caveats

```{warning}
- **The $f(E)$ overlay is matched by peak, and shows a $\sim 5\%$ edge
  residual.** The two engines parametrize energy and normalize the DF
  differently; after matching the peak on a common dimensionless axis, the
  largest discrepancy ($5.05\times10^{-2}$) sits at the truncation edge / central
  cusp, where the lowered-exponential cutoff and the Abel-inverted tail differ in
  shape. It is reported as a **diagnostic**, not gated — the sampled-CDF (KS) and
  dispersion gates carry the agreement claim, and a $5\%$ DF-shape edge mismatch
  is invisible in those sampled statistics.
- **This is one model, not a sweep.** A single concentration ($W_0=5$) at one
  resolution is checked. The wider parameter agreement is established separately
  by the [Engine B validation suite](../50-validation/engine-b-eddington.md),
  whose King A-vs-B anchor measures $0.0002$ / $0.0003$ across its own headline
  mix.
- **Single shared seed; interior bins only.** The headline KS / $\sigma$-deviation
  numbers are *one-seed* measurements (`key=0`, deliberately shared to cancel shot
  noise), not an ensemble; and the $\sigma$ gate is evaluated only over the 5%–90%
  radial quantiles — the cusp and truncation edge (exactly where the $f(E)$
  residual peaks) are excluded.
- **Single mass component.** The cross-check uses a one-component ($w_j=m_j=1$)
  model; the multimass machinery both engines support is exercised in the Engine B
  validation suite, not here.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_cross_engine.py
```

## References

The lowered-isothermal / LIMEPY family is {cite:t}`Gieles2015`; the King model
{cite:t}`King1966`; Eddington inversion and the ergodic DF follow Binney &
Tremaine (2008). Engine internals: the
[multi-component theory pages](../10-theory/populations/index.md) and the
[Engine B validation suite](../50-validation/engine-b-eddington.md).
