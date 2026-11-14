---
title: Baumgardt & Makino (2003)
description: Annotated reference for H. Baumgardt & J. Makino — Dynamical evolution of star clusters in tidal fields; source of the tidal-radius (Jacobi) relation r_t ∝ (G m_c / 2 V_G²)^(1/3) R_G^(2/3) used in progenax.tidal.
---

# Baumgardt & Makino (2003)

```{admonition} Dynamical evolution of star clusters in tidal fields
:class: note

**Authors.** Holger Baumgardt, Junichiro Makino (Department of Astronomy, University
of Tokyo).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **340**, 227–246
(2003). Accepted 2002 November 19.

**DOI.** [10.1046/j.1365-8711.2003.06286.x](https://doi.org/10.1046/j.1365-8711.2003.06286.x) ·
**ADS.** [2003MNRAS.340..227B](https://ui.adsabs.harvard.edu/abs/2003MNRAS.340..227B)
```

## Abstract (paraphrased)

Reports a large set of collisional *N*-body simulations (NBODY4 on GRAPE6, $N = 8192$ to
$131\,072$ stars) of multimass King clusters evolving in an external Galactic tidal field
under the combined action of stellar evolution, two-body relaxation and tidal stripping. The
clusters move on circular or eccentric orbits in a logarithmic Galactic potential with
circular velocity $V_G = 220\ \mathrm{km\,s^{-1}}$. The main result is that cluster lifetimes
scale as $T \sim T_{\rm rh}^x$ with $x \approx 0.75$ (weakly sub-linear in the relaxation
time), and that preferential loss of low-mass stars over the tidal boundary turns initially
rising mass functions into ones that decline towards low masses. For progenax, the
load-bearing content is the **initial setup**: the tidal radius of the King model is set
equal to the tidal radius of the external field (their Eq. 1).

## The load-bearing equation: tidal (Jacobi) radius in a logarithmic potential

For an orbit in a logarithmic external potential $\phi(R_G) = V_G^2 \ln R_G$ with circular
velocity $V_G$, the cluster radii are adjusted so the King tidal radius equals the tidal
radius of the external field, given by (**Eq. 1, p. 229**):

```{math}
:label: bm03-tidal-radius
r_t = \left(\frac{G\, m_c}{2\, V_G^2}\right)^{1/3} R_G^{2/3},
```

where $m_c$ is the cluster mass, $V_G$ the circular velocity of the Galaxy, and $R_G$ the
distance of the cluster from the Galactic Centre (for eccentric orbits, evaluated at
perigalacticon). This is the singular-isothermal-halo / flat-rotation-curve tidal radius.

```{note}
Writing the orbital angular velocity as $\Omega = V_G / R_G$, Eq. 1 is **algebraically
identical** to the angular-velocity form
$r_t = \big(G\,m_c / (2\,\Omega^2)\big)^{1/3}$, since
$\big(G m_c / 2 V_G^2\big)^{1/3} R_G^{2/3} = \big(G m_c / 2 (V_G/R_G)^2\big)^{1/3}
= \big(G m_c / 2 \Omega^2\big)^{1/3}$. The factor of 2 (rather than 3, as in the point-mass
case) is the signature of the flat rotation curve.
```

A corollary used qualitatively in the mass-loss discussion (text after **Eq. 11, p. 233**):
as a cluster loses mass its tidal radius shrinks as

```{math}
:label: bm03-rt-scaling
r_t \propto m_c^{1/3},
```

which follows directly from {eq}`bm03-tidal-radius` at fixed orbit ($R_G$, $V_G$).

## Other key relations (context, not used in `tidal.py`)

| Quantity | Form | Location |
|----------|------|----------|
| Adopted IMF | two-stage power law, $\xi(m)\,dm \sim m^{-1.3}\,dm$ ($m<0.5\,M_\odot$), $m^{-2.3}\,dm$ ($m\ge 0.5\,M_\odot$), $0.1$–$15\,M_\odot$ | Eq. 2, p. 229 |
| Relaxation time | $T_{\rm rh} \sim \dfrac{\sqrt{m_h}\,r_h^{3/2}}{\langle m\rangle\,\sqrt{G}\,\ln(\gamma N)}$ | Eq. 3, p. 230 |
| Dissolution-time scaling | $T_{\rm diss} = \beta \left[\dfrac{N}{\ln(\gamma N)}\right]^x \dfrac{R_G}{\rm kpc}\left(\dfrac{V_G}{220\,{\rm km\,s^{-1}}}\right)^{-1}$, $x\approx 0.75$ ($W_0=5$), $x\approx 0.82$ ($W_0=7$) | Eqs. 6–7, p. 232 |
| Eccentricity correction | $T_{\rm diss}(\epsilon) = T_{\rm diss}(0)\,(1-\epsilon)$ | Eq. 8, p. 232 |

These are recorded for completeness; progenax does not implement the lifetime/dissolution
fitting formulae.

## Code cross-reference

`src/progenax/tidal.py` now attributes the logarithmic-host radius to
Baumgardt & Makino Eq. 1 at the module and `jacobi_radius_isothermal` owners.
`fill_factor_to_r_h` is documented as the inversion of a definitional ratio,
not as a formula supplied by this paper.

The function whose formula **matches** {eq}`bm03-tidal-radius` is **`jacobi_radius_isothermal`**:

```python
# src/progenax/tidal.py, jacobi_radius_isothermal
Omega = V_circ / R_galactic
r_J = (G * M_cluster / (2.0 * Omega**2)) ** (1.0 / 3.0)
```

Substituting $\Omega = V_{\rm circ}/R_G$ gives $r_J = (G\,M / 2)^{1/3}\,(R_G/V_{\rm circ})^{2/3}
= (G\,M / 2 V_{\rm circ}^2)^{1/3} R_G^{2/3}$ — **exactly BM03 Eq. 1** with
$m_c \to M_{\rm cluster}$, $V_G \to V_{\rm circ}$, $R_G \to R_{\rm galactic}$.
The current function docstring includes this primary attribution. (`jacobi_radius`, the *point-mass* $r_J = R\,(M/3M_{\rm gal})^{1/3}$, is a
**different** relation — factor 3, not 2 — correctly attributed to King 1962 / Binney &
Tremaine Eq. 8.91, **not** to this paper.)

## Verification verdict

```{admonition} Verdict: formula and current attribution agree
:class: tip
The implemented isothermal relation is algebraically BM03 Eq. 1. The current
source cites it at that function and treats $r_h=f r_J$ as a definition.
The earlier citation-placement discrepancy is historical; it is not a pending
source repair. The PDF equation inspection recorded above is distinct from
this current-source reconciliation.
```

```{note}
All equation values above were read from the PDF's **embedded text layer** (the document is
text-based, not a scanned image), so no values are flagged UNVERIFIED-by-text-extract.
```

## Use in progenax

- `progenax.tidal.jacobi_radius_isothermal` — implements {eq}`bm03-tidal-radius` (BM03 Eq. 1).
- `progenax.tidal.fill_factor_to_r_h` — uses $r_h = f\,r_J$; the fill-factor concept is **not**
  from this paper (see verdict).
- [](../../10-theory/tidal-and-substructure/index.md) — tidal-radius and truncation theory.

## Notes

Eq. 1's factor of 2 (vs. 3 for a Keplerian point mass) is specific to the
logarithmic / flat-rotation-curve potential; do not confuse it with the point-mass Jacobi
radius in `jacobi_radius`.
