---
title: Rotation & the omega-inclination degeneracy (B8)
description: "The first observational-realism demo. A rotating cluster viewed at an inclination shows a mean line-of-sight velocity that is linear in the sky x-coordinate with slope omega*sin(i): <v_los> measures the PRODUCT omega*sin(i), not the rotation rate. So omega and the inclination are degenerate -- the (omega, i) Fisher is rank 1 -- and breaking it needs an independent inclination that line-of-sight velocities cannot supply. Introduces a line-of-sight projection helper (the bridge to a future observational-realism arc)."
---

# Rotation & the $\omega$–inclination degeneracy (B8)

The other demos work on *clean* 3-D mocks. This one takes the first step toward
**observational realism**: a rotating cluster is viewed at an inclination, and the
recovered rotation depends on a **line-of-sight projection**. The honest result is
the headline of the realism axis — the mean line-of-sight velocity measures the
*product* $\omega\sin i$, never the rotation rate $\omega$ alone.

## The physics

Solid-body rotation about the cluster $z$-axis gives $\mathbf v = \omega(\hat z
\times \mathbf r)$. The demo's projection helper `project_los` tilts the cluster by
the inclination $i$ (about the sky $x$-axis) and reads the line-of-sight (observer
$z$) velocity. The mean is **linear in the sky $x$-coordinate**:

```{math}
:label: b8-vlos
\langle v_{\rm los}\rangle(x_{\rm sky}) = \omega\,\sin i\;x_{\rm sky}.
```

A face-on cluster ($i=0$) shows **no** rotation signature; an edge-on one ($i=90^\circ$)
shows the full $\omega$. So the observable rotation amplitude is the **slope**
$k = \omega\sin i$. The recovered slope (a clean linear fit of the binned
$\langle v_{\rm los}\rangle$) is

```{list-table}
:header-rows: 1

* - quantity
  - truth
  - recovered
* - $k = \omega\sin i$
  - $1.732$  ($\omega=2.0$, $i=60^\circ$)
  - $1.69 \pm 0.03$  (pull $-1.34$)
```

## The degeneracy (the headline)

Because $\langle v_{\rm los}\rangle$ depends only on the product, $\omega$ and $i$
are **degenerate**: $\partial k/\partial(\omega, i) = (\sin i,\ \omega\cos i)$, so the
$(\omega, i)$ Fisher information $\mathcal F = (\nabla k)(\nabla k)^\top/\sigma_k^2$
is **rank 1**, with eigenvalues

```{math}
:label: b8-eig
\lambda(\mathcal F) = (8\times10^{-14},\ 1.9\times10^{3}),
```

— one machine-precision zero (condition number $\sim 10^{16}$). Every $(\omega, i)$
on the curve $\omega = \hat k/\sin i$ fits the line-of-sight velocities equally well
(panel c). **Recovering the rotation rate needs an independent inclination** — e.g.
from the projected flattening — which line-of-sight velocities alone cannot supply.
This is the same rank-deficient inverse-problem structure as the
[birth-environment demo](birth-environment.md), now from projection rather than a
many-to-one map.

## Figure

:::{figure} figures/demo_rotation.png
:label: sci-rotation
:width: 100%

**Rotation & the $\omega$–$i$ degeneracy** (`scripts/demo_rotation.py`, ALL PASS).
**(a)** The projected sky map coloured by $v_{\rm los}$: the rotation **dipole** —
one side approaching (blue), one receding (red). **(b)** The rotation curve
$\langle v_{\rm los}\rangle(x_{\rm sky})$ with the recovered slope $k=\omega\sin i$.
**(c)** The degeneracy: every $(\omega, i)$ on $\omega=\hat k/\sin i$ (purple) fits
equally well; the truth ★ lies on it.
:::

## Caveats

```{warning}
- **The bridge, not the destination.** This introduces the line-of-sight projection
  but stops at one realism effect (inclination). Measurement errors, selection /
  incompleteness, foreground contamination, and a realistic PSF are the rest of the
  observational-realism arc — applied to *all* the demos — and are future work.
- **Solid-body rotation, no anisotropy.** A differential rotation curve or a
  rotation–anisotropy coupling changes $\langle v_{\rm los}\rangle$; the
  degeneracy with $i$ persists but its breaking (via the flattening) is
  model-dependent.
- **Clean census, known projection geometry** otherwise (the tilt is a pure
  inclination about one axis; a real cluster also has a position-angle and possibly
  internal structure).
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_rotation.py
```

## References

The Plummer model is {cite:t}`Plummer1911`; the rotation overlays
(`apply_solid_body_rotation`, `apply_differential_rotation`) and their
non-equilibrium caveats are documented on the
[rotation & anisotropy](../10-theory/velocity-dfs/rotation-anisotropy.md) page.
