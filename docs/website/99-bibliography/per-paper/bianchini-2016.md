---
title: Bianchini et al. (2016)
description: Annotated reference for Bianchini, van de Ven, Norris, Schinnerer & Varri — "A novel look at energy equipartition in globular clusters" (the σ(m)–m_eq equipartition relation and its derivation from the lowered-isothermal multimass DF).
---

# Bianchini et al. (2016)

```{admonition} A novel look at energy equipartition in globular clusters
:class: note

**Authors.** P. Bianchini, G. van de Ven, M. A. Norris, E. Schinnerer, A. L. Varri.

**Reference.** *Monthly Notices of the Royal Astronomical Society* **458**,
3644–3654 (2016). **DOI.** [10.1093/mnras/stw552](https://doi.org/10.1093/mnras/stw552) ·
**arXiv.** [1603.00878](https://arxiv.org/abs/1603.00878) ·
**ADS.** [2016MNRAS.458.3644B](https://ui.adsabs.harvard.edu/abs/2016MNRAS.458.3644B).

**Held PDF.** `docs/core-papers/Bianchini2016.pdf`. Equations below verified
page-by-page (2026-06-11).
```

Why progenax holds this: it supplies the **physical interpretation** of the
mass-dependence of velocity dispersion that our Engine A multimass models
([Gieles & Zocchi 2015](gieles-zocchi-2015.md), [Peuten et al. 2017](peuten-2017.md))
produce. Crucially, its **Appendix A derives the σ(m) relation directly from the
GZ15 lowered-isothermal multimass DF** — so it is the bridge that lets us state,
with provenance, that progenax's standard model already captures equipartition
saturation, and quantify it.

## The σ(m) fitting function (§3.1, eqs 3–4 — verified)

Globular clusters never reach *full* energy equipartition; two-body relaxation
drives them only to **partial** equipartition, with a mass-dependent local slope.
A single power law `σ ∝ m^{−η}` (Trenti & van der Marel 2013) fits only a
restricted mass range, so Bianchini introduces an **exponential** fitting
function valid across the whole range (eq 3):

```{math}
\sigma(m) =
\begin{cases}
\sigma_0 \, \exp\!\left(-\dfrac{1}{2}\dfrac{m}{m_{\rm eq}}\right) & m \le m_{\rm eq}\\[8pt]
\sigma_{\rm eq}\left(\dfrac{m}{m_{\rm eq}}\right)^{-1/2} & m > m_{\rm eq}
\end{cases}
\qquad \sigma_{\rm eq} = \sigma_0\, e^{-1/2}
```

equivalently **σ² ∝ exp(−m/m_eq)**. The local slope (eq 4) is
`η(m) = −d ln σ / d ln m = ½ (m/m_eq)` for `m ≤ m_eq` and `½` for `m > m_eq`.
The piecewise `m^{−1/2}` branch above `m_eq` is imposed so the slope cannot
exceed `½` (which would unphysically *exceed* equipartition) and to match the
asymptotic limits of the analytic multimass DF models (App. A).

**Meaning of `m_eq`.** It is the mass **above which the cluster has reached full
equipartition** (`σ ∝ m^{−1/2}`); below it, equipartition is only partial.
**Smaller `m_eq` ⇒ closer to global equipartition.** Typical fitted values are
`m_eq ≳ 1 M_⊙`, i.e. only stars/remnants above ~1 M_⊙ are in equipartition.

The fit is performed via a Gaussian likelihood over the observed `(m_i, v_i)`
(eq 5), and white dwarfs are excluded (recent mass-loss leaves their kinematics
inconsistent with their present mass).

## Appendix A — σ(m) derived FROM the GZ15 multimass DF (the bridge — verified)

This is the part progenax relies on. Bianchini shows the exponential is not
ad hoc: it is the **low-mass Taylor expansion of the central velocity dispersion
of a GZ15 multimass component.** The component central dispersion (eq A1):

```{math}
\hat\sigma_{1d,j0} = \frac{1}{\mu_j^{2}}\,
\frac{E_\gamma\!\left(g+\tfrac52;\; \mu_j^{2\delta}\hat\phi_0\right)}
     {E_\gamma\!\left(g+\tfrac32;\; \mu_j^{2\delta}\hat\phi_0\right)},
\qquad \mu_j = m_j/\bar m,
```

with `m̄` the central-density-weighted mean mass and `δ` the GZ15 equipartition
index (`δ=½` ⇒ `m_j s_j² = m̄ σ²` constant). Expanding for **low mass**
(`μ_j ≪ 1`, `δ=½`) to second order (eq A2) and collecting terms gives (eq A3):

```{math}
\sigma \sim \sigma_0\left[1 - \tfrac12\frac{m}{m_{\rm eq}}
                           + \tfrac18\left(\frac{m}{m_{\rm eq}}\right)^2\right]
        = \text{Taylor series of } \sigma_0\exp\!\left(-\tfrac12 m/m_{\rm eq}\right),
```

and the **high-mass** limit (`μ_j ≫ 1`) gives `σ̂_{1d,j0} ∼ μ_j^{−δ} ∝ m^{−1/2}`
(GZ15 §3.2.1) — the equipartition branch. Matching the linear terms of A2 ↔ A3
identifies the equipartition mass as a **derived** quantity:

```{math}
:label: bianchini-meq-derived
\boxed{\;m_{\rm eq} = \bar m\,\frac{(g+\tfrac52)(g+\tfrac72)}{\hat\phi_0}\;}
```

i.e. **`m_eq` is fixed by the mean mass `m̄`, the truncation order `g`, and the
central concentration `Ŵ₀ = φ̂₀`** — it is *not* a free input to the DF. The
standard multimass model (`μ_j = m_j/m̄`, `δ=½`, no extra parameter) therefore
*already* produces the Bianchini saturation. progenax uses
[](bianchini-meq-derived) to validate that its differentiable Engine A model
reproduces the equipartition relation analytically (see the multimass theory
page). The LIMEPY *code*'s `meq` (in `μ_j=(m_j+meq)/m̄`) is a **separate
phenomenological knob** that adds extra softening to decouple equipartition from
`g/Ŵ₀`; it is *not* the [](bianchini-meq-derived) physics.

## m_eq ↔ relaxation state (§4–6, the headline result — verified)

Fitting `m_eq` across Monte-Carlo cluster simulations (Downing et al. 2010) at
matched time-snapshots, Bianchini finds a **tight correlation between the degree
of equipartition and dynamical age**: with `n_eq ≡ T_age/T_rc` (cluster age in
core relaxation times), clusters older than **~20 core relaxation times reach a
maximum degree of equipartition** (more concentrated / older ⇒ smaller `m_eq`).
Consequences: (i) `m_eq` measured kinematically (HST proper motions) is a proxy
for a cluster's relaxation state; (ii) knowing `T_rc` photometrically *predicts*
the σ(m) trend (incl. the unobservable low-mass / remnant regime); (iii)
deviations from the tight `m_eq–n_rel` relation flag peculiar histories
(post-core-collapse, IMBH, accretion). Binaries and dark remnants follow the
same σ(m) as single stars (except recently-formed white dwarfs).

The **Spitzer (1969) instability** — heavy stars sinking and forming a
self-gravitating subsystem that decouples and never equipartitions — is the
physical origin of the LIMEPY code's `zeta` "decoupling" knob (deferred in
progenax).

## Use in progenax

- **Derived-m_eq validation** ([](bianchini-meq-derived)): a zero-new-parameter
  check that our Engine A σ(m) matches Bianchini eq 3 with the derived `m_eq`.
- **Honest provenance for the deferred `meq`/`zeta` knobs:** Bianchini motivates
  them physically but does *not* define the code's `(m_j+meq)` form — that is a
  code heuristic, documented as such (see [Peuten et al. 2017](peuten-2017.md)
  and the design note `docs/plans/2026-06-11-multimass-limepy-hardening-design.md`).
