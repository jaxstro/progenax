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

**Verified.** Equations below verified page-by-page against the published PDF (2026-06-11).
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
of a GZ15 multimass component.** The component central dispersion, in
corrected form (eq A1 — see the typo note below):

```{math}
:label: bianchini-A1-corrected
\hat\sigma_{1d,j0} = \mu_j^{-\delta}
\left[\frac{E_\gamma\!\left(g+\tfrac52;\; \mu_j^{2\delta}\hat\phi_0\right)}
     {E_\gamma\!\left(g+\tfrac32;\; \mu_j^{2\delta}\hat\phi_0\right)}\right]^{1/2},
\qquad \mu_j = m_j/\bar m,
```

with `m̄` the central-density-weighted mean mass and `δ` the GZ15 equipartition
index (`δ=½` ⇒ `m_j s_j² = m̄ σ²` constant).

```{admonition} Paper typo in eq A1 (confirmed via the paper's own A2)
:class: warning
As printed, the paper's A1 omits the `^{1/2}` on the `E_γ` ratio (it gives
`σ̂_{1d,j0}` with the prefactor `1/μ_j^δ` but the *unsquare-rooted* ratio) —
internally inconsistent with its own eq A2, whose expansion prefactor
`[(g+3/2)Γ(g+3/2) / ((g+5/2)Γ(g+5/2))]^{1/2} φ̂₀^{1/2}` matches only the
square-rooted form [](#bianchini-A1-corrected). (Dimensionally, the ratio is
a `σ̂²`, not a `σ̂`.) progenax validated [](#bianchini-A1-corrected) directly:
the multimass solver's quadrature-oracle central dispersion agrees with this
closed form to a max relative difference of `1.5–1.7e-7` across
`W₀ ∈ {5,7,9}` (`scripts/validate_equipartition_saturation.py`, gate `1e-6`).
An earlier revision of this note also mis-transcribed the prefactor as
`1/μ_j²`; both fixed 2026-06-11.
```

Expanding for **low mass** (`μ_j ≪ 1`, `δ=½`) to second order (eq A2) and
matching against the expansion of the exponential gives (eq A3):

```{math}
\sigma \sim \sigma_0\left[1 - \tfrac12\frac{m}{m_{\rm eq}} + \mathcal{O}(m^2)\right],
\qquad \text{matching } \sigma_0\exp\!\left(-\tfrac12 m/m_{\rm eq}\right)
\ \text{to FIRST order},
```

and the **high-mass** limit (`μ_j ≫ 1`) gives `σ̂_{1d,j0} ∼ μ_j^{−δ} ∝ m^{−1/2}`
(GZ15 §3.2.1) — the equipartition branch. Matching the linear terms of A2 ↔ A3
identifies the equipartition mass as a **derived** quantity:

```{math}
:label: bianchini-meq-derived
\boxed{\;m_{\rm eq} = \bar m\,\frac{(g+\tfrac52)(g+\tfrac72)}{\hat\phi_0}\;}
```

i.e. **`m_eq` is fixed by the mean mass `m̄`, the truncation order `g`, and the
central concentration `Ŵ₀ = φ̂₀`** — it is *not* a free input to the DF.

```{admonition} Second order: the DF saturates SHARPER than the exponential
:class: important
The eq-3 exponential matches the DF only to **first** order. The exact
quadratic Taylor coefficient of the DF's `σ(m)` — from eq A2's printed term,
`(6+3a−4a²)/(8a²b²c)·φ̂₀²` with `a=g+5/2`, `b=g+7/2`, `c=g+9/2` — is
**negative** for all cluster-relevant `g` (`6+3a−4a² < 0` for `a > 1.65`,
i.e. any `g > −0.85`), *opposite in sign* to the exponential's
`+1/(8 m_eq²)`. So A3's statement that A2 gives "the first terms of the Taylor
expansion of the exponential" holds for the **linear term only**. Practical
consequence (measured 2026-06-11 in a one-off windowed-fit analysis, recorded
in the docstring of `scripts/validate_equipartition_saturation.py` — the
gated validation itself is fit-free): eq-3 exponential fits over a
finite mass window recover a *window-dependent* `m_eq` biased low — over
`μ = m/m̄ ∈ [0.14, 1.7]` (20 bins on `[0.1, 1] M_⊙`) such fits recover
**~0.5×** the derived [](#bianchini-meq-derived) (measured `0.50–0.53`
across `W₀ ∈ {5,7,9}`). Window-match before comparing fitted literature
`m_eq` values with the derived one. See the
[multimass-equipartition theory page](../../10-theory/spatial-profiles/multimass-equipartition.md).
```

The
standard multimass model (`μ_j = m_j/m̄`, `δ=½`, no extra parameter) therefore
*already* produces the Bianchini saturation. progenax uses
[](#bianchini-meq-derived) to validate that its differentiable Engine A model
reproduces the equipartition relation analytically (see the multimass theory
page). The LIMEPY *code*'s `meq` (in `μ_j=(m_j+meq)/m̄`) is a **separate
phenomenological knob** that adds extra softening to decouple equipartition from
`g/Ŵ₀`; it is *not* the [](#bianchini-meq-derived) physics.

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

- **Derived-m_eq validation** ([](#bianchini-meq-derived)): a zero-new-parameter
  check that our Engine A σ(m) matches Bianchini eq 3 with the derived `m_eq`.
- **Honest provenance for the deferred `meq`/`zeta` knobs:** Bianchini motivates
  them physically but does *not* define the code's `(m_j+meq)` form — that is a
  code heuristic, documented as such (see [Peuten et al. 2017](peuten-2017.md)
  and the design note `docs/plans/2026-06-11-multimass-limepy-hardening-design.md`).
