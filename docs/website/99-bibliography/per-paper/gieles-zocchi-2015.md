---
title: Gieles & Zocchi (2015)
description: Annotated reference for M. Gieles & A. Zocchi — A family of lowered isothermal models (the LIMEPY formalism progenax reimplements differentiably).
---

# Gieles & Zocchi (2015)

```{admonition} A family of lowered isothermal models (the LIMEPY formalism)
:class: note

**Authors.** Mark Gieles & Alice Zocchi (Department of Physics, University of Surrey).

**Reference.** *Monthly Notices of the Royal Astronomical Society* **454**, 576–592 (2015);
accepted 2015 August 7. Code: `limepy` (Lowered Isothermal Model Explorer in PYthon),
[github.com/mgieles/limepy](https://github.com/mgieles/limepy).

**ADS.** [2015MNRAS.454..576G](https://ui.adsabs.harvard.edu/abs/2015MNRAS.454..576G) ·
**DOI.** [10.1093/mnras/stv1848](https://doi.org/10.1093/mnras/stv1848)

**Erratum.** [2018MNRAS.474.3997G](https://ui.adsabs.harvard.edu/abs/2018MNRAS.474.3997G)
(DOI [10.1093/mnras/stx3144](https://doi.org/10.1093/mnras/stx3144)) — corrects printed
Eqs. 20, 21, 41 (see [below](#erratum)). The `limepy` code itself was correct; no published
LIMEPY results are affected.
```

This is the **keystone paper** for the progenax lowered-model roadmap
([](../../10-theory/spatial-profiles/lowered-model-family.md)). It unifies the classical
single-mass isotropic truncated models — Woolley, [King (1966)](king-1966.md), Wilson — into
**one** continuous family, and then extends that family in the two directions globular clusters
actually need: **radial anisotropy** (a [Michie (1963)](michie-1963.md) / Osipkov–Merritt term)
and **multiple mass components** (the physical origin of mass segregation). progenax
reimplements this formalism JAX-natively so every model parameter — including the truncation
sharpness $g$ and the equipartition degree $\delta$ — is differentiable for gradient-based
inference (the published `limepy` is numpy/scipy and not differentiable).

## Abstract (paraphrased)

Presents a family of self-consistent, spherical, lowered isothermal models with one or more
mass components, with *parametrized* prescriptions for the energy truncation and for the
radial-anisotropy content. The models extend the isotropic single-mass family of Gomez-Leyton
& Velazquez (2014, "GV14"), of which Woolley, King, and Wilson are members. Analytic
expressions for density and velocity-dispersion components in terms of potential and radius are
derived (so no double velocity integral is needed at each radial step), and a fast Poisson
solver (`limepy`) is provided for data fitting and for drawing discrete samples as N-body
initial conditions. The models are aimed at tidally limited, mass-segregated star clusters
across their life-cycle.

## The single-mass DF — one knob unifies Woolley/King/Wilson (verified, §2.1)

The distribution function (Eq. 1) is a lowered Maxwellian whose **truncation sharpness** is set
by a continuous parameter $g$, with an optional Michie/Osipkov–Merritt anisotropy factor:

```{math}
:label: limepy-df
f(E, J^2) = A\,\exp\!\left(-\frac{J^2}{2 r_a^2 s^2}\right)\,
            E_\gamma\!\left(g,\; \frac{\phi(r_t)-E}{s^2}\right),
\qquad E \le \phi(r_t),\ \ 0 \ \text{otherwise.}
```

Here $E=\tfrac12 v^2+\phi(r)$ is the specific energy, $J=|\mathbf r\times\mathbf v|$ the specific
angular momentum, $s$ a velocity scale, $r_a$ the anisotropy radius, and $r_t$ the truncation
radius. The relative energy $\hat E \equiv [\phi(r_t)-E]/s^2 \ge 0$ for bound stars. The whole
construction lives or dies on **one special function** (Eq. 2):

```{math}
:label: limepy-Egamma
E_\gamma(a, x) =
\begin{cases}
\exp(x), & a = 0,\\[4pt]
\exp(x)\,P(a, x), & a > 0,
\end{cases}
\qquad P(a,x) \equiv \frac{\gamma(a, x)}{\Gamma(a)},
```

where $P(a,x)$ is the **regularized lower incomplete gamma function** — exactly
`jax.scipy.special.gammainc(a, x)`. This is what makes a JAX reimplementation tractable *and*
differentiable: the truncation is not a hand-rolled series but a built-in special function with
analytic gradients in both arguments.

The integer-$g$ corners recover the textbook models (paper footnote 2, verified by direct
expansion of {eq}`limepy-Egamma`):

| $g$ | $E_\gamma(g,x)$ | Model | Truncation |
|-----|-----------------|-------|------------|
| $0$ | $e^{x}$ | **Woolley** (1954) | DF discontinuous at $E=\phi(r_t)$ |
| $1$ | $e^{x}-1$ | **[King](king-1966.md)** (1966) | DF continuous (the lowered Maxwellian) |
| $2$ | $e^{x}-1-x$ | **Wilson** (1975) | DF *and* derivative continuous (more extended) |

So $g$ is a **continuous dial** between these: a fitted $g$ lets the *data* choose
King-vs-Wilson as a posterior, rather than the modeller hard-coding it. The paper's $g$ is the
same symbol as progenax's truncation index; the central dimensionless potential
$\hat\phi_0$ (their notation) is identical to King's $W_0$ (their footnote 3).

## The computational win — density without a velocity integral (§2.1.3–2.1.4)

Self-consistency means solving Poisson's equation $\nabla^2\phi = 4\pi G\rho$ with
$\rho=\int f\,d^3v$. Naively that nests a velocity integral inside every radial step. The
family's key analytic property is that the **velocity-space integral collapses back into the
same $E_\gamma$ family at a shifted index**. In dimensionless variables
($\hat\phi=[\phi(r_t)-\phi]/s^2$, $\hat k=v^2/2s^2$, $\hat\rho=\rho/\rho_0$,
$\hat r = r/r_s$ with $r_s^2 = 9 s^2/(4\pi G\rho_0)$ the King radius), Poisson's equation reads
(Eq. 5)

```{math}
:label: limepy-poisson
\frac{1}{\hat r^2}\frac{d}{d\hat r}\!\left(\hat r^2 \frac{d\hat\phi}{d\hat r}\right) = -9\,\hat\rho,
\qquad \hat\phi(0)=\hat\phi_0,\ \ \left.\frac{d\hat\phi}{d\hat r}\right|_0 = 0,
```

with the **factor of $-9$** inherited from King's core-radius normalization
($8\pi G j^2 r_s^2\rho_0 = 9$). For the **isotropic** case the density and pressure integrals
(Eqs. 8–9) reduce to closed forms in $E_\gamma$ at indices $g+\tfrac12$ and $g+\tfrac32$:

```{math}
:label: limepy-rho-iso
\mathcal{I}^{\rho}(\hat\phi) = \frac{2}{\sqrt{\pi}}\int_0^{\hat\phi}\! \hat k^{1/2}\,
   E_\gamma(g,\hat\phi-\hat k)\,d\hat k,
\qquad
\mathcal{I}^{\rho\sigma^2}(\hat\phi) = \frac{4}{\sqrt{\pi}}\int_0^{\hat\phi}\! \hat k^{3/2}\,
   E_\gamma(g,\hat\phi-\hat k)\,d\hat k,
```

with $\hat\rho = \mathcal{I}^\rho/\mathcal{I}^\rho_0$ and
$\hat\sigma^2=\sigma^2/s^2 = \mathcal{I}^{\rho\sigma^2}/\mathcal{I}^\rho$ (central values carry
subscript $0$). These are evaluated as combinations of $E_\gamma$ via the convolution identity
of Appendix D — **no quadrature at runtime**.

```{admonition} The King corner, worked through (verified by hand)
:class: tip
For $g=1$, {eq}`limepy-rho-iso` integrates in closed form to
$\mathcal{I}^{\rho} = e^{\hat\phi}\,\mathrm{erf}(\sqrt{\hat\phi})
 - \tfrac{2}{\sqrt\pi}\sqrt{\hat\phi}\,(1+\tfrac23\hat\phi)$ — **exactly** the King (1966)
density that [`progenax.profiles.KingProfile`](king-1966.md) already integrates. This is the
single most useful cross-check for the progenax build: the general-$g$ density **must** reduce
to the existing King solver at $g=1$, isotropic. (The half-integer index bookkeeping in the
paper's compact "$=E_\gamma(g+\tfrac12,\hat\phi)$" shorthand is verified against this corner
*numerically* in the progenax implementation, not assumed.)
```

The **anisotropic** case (Eqs. 10–17) replaces the closed $E_\gamma$ with a radial integral that
the paper carries out via fractional calculus, producing the **confluent hypergeometric function**
$_1F_1(a,b,x)$ — also a `jax.scipy.special.hyp1f1` primitive. The anisotropy is a Michie/OM term
$\exp(-J^2/2r_a^2 s^2)$: isotropic in the core ($\hat r\ll \hat r_a$), radial in the envelope
($\beta\to 1$), and — characteristically — **suppressed again near the truncation radius**
(potential escapers with tangentially-biased velocities are above the escape energy and removed),
matching tidally-truncated systems (Oh & Lin 1992).

## Limits and special members (§2.1.5, §3) — built-in sanity anchors

- **Plummer is a member.** For $g=7/2$ in the polytropic (low-$\hat\phi_0$) regime the model is
  the $n=5$ polytrope, i.e. the [Plummer (1911)](plummer-1911.md) sphere (Eq. 20 limit). A
  direct corner check for the progenax core.
- **Finite vs infinite.** Isotropic models are finite in extent only for $g \le 3.5$
  ($g_{\max}=3.5$ as $\hat\phi_0\to\infty$); $g\ge 7/2$ gives infinite-mass models, excluded for
  clusters. progenax restricts to the cluster-relevant $g\lesssim 2$ band.
- **Isothermal limit.** Both $\hat\phi_0\to\infty$ and $g\to\infty$ approach the singular
  isothermal sphere (Eq. 23).
- **Stability.** Radial-orbit instability is flagged via $\kappa = 2K_r/K_t$ (Eq. 33): isotropic
  $\kappa=1$, and $\kappa\gtrsim 1.7\pm0.25$ (Polyachenko & Shukhman 1981) signals instability —
  a guard the anisotropic progenax models will report.

## Multi-mass models — the physics of mass segregation (§2.2, §3.2)

This is the section that motivates the whole progenax Phase 2. Each mass component $j$ shares
**one** self-consistent potential $\hat\phi(\hat r)$ but has its own velocity scale and
anisotropy radius set by **power-law mass scalings** (Eqs. 24–26):

```{math}
:label: limepy-multimass-scalings
s_j = s\,\mu_j^{-\delta},
\qquad \hat r_{a,j} = \hat r_a\,\mu_j^{\eta},
\qquad \mu_j \equiv \frac{m_j}{\bar m},
```

with $\bar m$ the **central density-weighted mean mass** (Eq. 26). The **equipartition parameter
$\delta$** is the key knob: heavier components ($\mu_j>1$) get a *smaller* velocity scale $s_j$,
hence a deeper effective well and **central concentration as a genuine equilibrium** — this is
mass segregation, not an imposed reshuffle. The dimensionless multi-mass Poisson equation
(Eqs. 27–29) sums the per-component densities:

```{math}
:label: limepy-multimass-poisson
\hat\nabla^2\hat\phi = -9\sum_j \alpha_j\,\hat\rho_j,
\qquad \sum_j\alpha_j = 1,
\qquad \hat\rho_j = \frac{\mathcal{I}^\rho\!\big(\mu_j^{2\delta}\hat\phi,\,\hat r\big)}
                        {\mathcal{I}^\rho\!\big(\mu_j^{2\delta}\hat\phi_0\big)},
```

where $\alpha_j$ is the **central** density fraction of component $j$. Because the $\alpha_j$ that
give a desired mass set $\{M_j\}$ are not known a priori, the solve is an **eigenvalue iteration**:
start from $\alpha_j = M_j/\sum_k M_k$, solve, then rescale. The paper finds Gunn & Griffin's
(1979) $\alpha_j \times (M_j/M_j')$ update unstable for wide mass functions and instead multiplies
by $\sqrt{M_j/M_j'}$ — the robust update progenax must replicate.

```{admonition} The honest physics of δ — why "full equipartition" is the wrong default
:class: important
Although $m_j s_j^2$ is constant across bins by construction, the models are **not** in energy
equipartition: the *observable* central dispersion ratio $\sigma_{1{\rm d},j}/s_j < 1$ for
realistic finite $\hat\phi_0$, because the lower-mass components feel a shallower central
potential (Merritt 1981; Miocchi 2006; their Fig. 9). True equipartition ($\sigma_{1d}\propto
m^{-1/2}$) is reached only as $\hat\phi_0\to\infty$ — and is **Spitzer-unstable** in real
clusters. The paper adopts $\delta=\tfrac12$ as the standard ("approximate at best" but
reproducing observed segregation; Trenti & van der Marel 2013; Sollima et al. 2015), with
$\eta=0$ (mass-independent anisotropy) typical. **progenax keeps $\delta$ a free, differentiable
parameter** rather than baking in $\delta=1/2$ or full equipartition — the data should constrain
it.
```

## The LIMEPY code (§4)

`limepy` solves {eq}`limepy-poisson` with `scipy`'s `dopri5` (adaptive RK4(5)), given
$(\hat\phi_0,\, g,\, r_a,\, \{m_j, M_j, \delta, \eta\})$, and can scale to physical units via a
mass $M$ and a radius scale, after which the velocity unit follows from
$G = 0.004302\ \mathrm{pc}\,(\mathrm{km\,s^{-1}})^2\,M_\odot^{-1}$ — **the STELLAR unit system
progenax already uses**. For large hypergeometric arguments ($x\gtrsim 700$) it switches to the
asymptotic $_1F_1$ forms (Appendix D24–D25) for numerical stability. progenax mirrors the solver
structure but on `diffrax.Tsit5` (differentiable, JIT-compatible) instead of `dopri5`, and on
`jax.scipy.special` instead of `scipy.special`.

(erratum)=
## Erratum (2018) — what was wrong on paper, and why the code was fine

The erratum corrects **printed typesetting mistakes that never entered the `limepy` code**:

- **Eqs. 20 & 21** — the small-$\hat\phi$ (polytropic, near-truncation) limits. As printed, their
  right-hand sides were the limits of the *unnormalized* integrals $\mathcal{I}^\rho$,
  $\mathcal{I}^{\rho\sigma^2}$, not of the normalized $\hat\rho$, $\hat\rho\hat\sigma^2$. Corrected:
  $\lim_{\hat\phi\to0}\hat\rho = \hat\phi^{g+3/2}/[\Gamma(g+\tfrac52)\,E_\gamma(g+\tfrac32,\hat\phi_0)]$
  and
  $\lim_{\hat\phi\to0}\hat\rho\hat\sigma^2 = 3\hat\phi^{g+5/2}/[\Gamma(g+\tfrac72)\,E_\gamma(g+\tfrac32,\hat\phi_0)]$
  — i.e. the missing central normalization $E_\gamma(g+\tfrac32,\hat\phi_0)$.
- **Eq. 41** — a notation typo in the projected $\sigma_S^2(r)$; corrected to
  $\sigma_S^2(r)=\sigma_t^2[1-\beta(r)(1-R^2/r^2)]$.

The paper states explicitly that these do not affect any figure and that the released `limepy`
implemented the correct expressions. **Implication for progenax:** ground the implementation on
the *corrected* limits and (where possible) cross-check against the `limepy` code's behaviour and
the King closed form — never the printed Eqs. 20/21.

## Use in progenax

This paper is the formalism behind the **multi-mass LIMEPY equilibrium hardening** of mass
segregation (the current build; complements the differentiable `lambda_seg` blend rather than
replacing it):

- [](../../10-theory/spatial-profiles/lowered-model-family.md) — the roadmap page this paper
  specifies; tracks the planned→implemented status of the family.
- **Phase 1** — single-mass general-$g$ core: `progenax.profiles.LIMEPYProfile` +
  `progenax.kinematics.LIMEPYVelocityDF`, solving {eq}`limepy-poisson` differentiably in
  $(W_0, g, r_a)$ via `diffrax`. Validated against the trusted internal corners: $g=1$ isotropic
  $\equiv$ [`KingProfile`](king-1966.md), $g=1$ anisotropic $\equiv$
  [`MichieProfile`](michie-1963.md), and King (1966) Table II concentrations $c(W_0)$.
- **Phase 2** — multi-mass coupling: `progenax.cluster.MultiComponentCluster`
  (`from_mass_segregation`; supersedes the earlier `MultiMassLIMEPY`), the coupled Poisson
  solve {eq}`limepy-multimass-poisson` with equipartition $\delta$, differentiable in
  $(W_0, g, r_a, \delta)$ and the mass-fraction parameters. The per-mass-group virial ratio
  $Q_j$ (progenax's `per_group_virial_ratio`) is the equilibrium proof: $Q_j\approx0.5$ for each
  component certifies a *true* partial equilibrium, the property the `lambda_seg` chord lacks.

## Notes

The lineage: Woolley (1954) lowered the isothermal sphere; [King (1966)](king-1966.md) made the
DF continuous; Wilson (1975) made its derivative continuous; [Michie (1963)](michie-1963.md) added
radial anisotropy; Gomez-Leyton & Velazquez (2014) parametrized the truncation continuously; and
Gieles & Zocchi (2015) unified all of it — multi-mass, anisotropic — into one solver. progenax's
contribution is orthogonal: making that unified family **differentiable**, so $(g,\delta,r_a,W_0)$
become *fitted* quantities flowing gradients through the Poisson solve, rather than fixed
modelling choices.
