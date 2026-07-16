---
title: Science capabilities
description: "What science progenax enables — the model inventory with validity regimes and measured validation numbers, twelve concrete research questions the package can address today, and the honest scope of what it does not do."
---

# Science capabilities

This page is for research astronomers deciding whether — and how — to
use progenax. It answers three questions: what science the package
enables, exactly which models are implemented and in what regime each
is valid, and what the package deliberately does *not* do. Every claim
on this page traces to a named validation test with a measured number;
the [validation dashboard](../50-validation/index.md) is the ledger.

:::{admonition} Who this page is for
:class: note
**Audience:** researchers evaluating whether/how to use progenax — **skim, don't read linearly.** This is a dense capability inventory, not a tutorial; jump to the model table or the twelve science questions for your use case.
**Prerequisites:** the getting-started basics ([what progenax is](index.md), [installation](installation.md)); the linked [theory](../10-theory/index.md) and [validation](../50-validation/index.md) pages carry the derivations and evidence behind each claim.
**You'll get:** the full model inventory with validity regimes and measured validation numbers, twelve concrete research questions progenax can address today, and an honest statement of what it does *not* do.
:::

## What progenax is for

progenax generates initial conditions for N-body simulations — but the
point of the package is that the IC generator is **differentiable
end-to-end**. Every structural parameter (half-mass radius, King
concentration $W_0$, truncation shape $g$, equipartition exponent
$\delta$, anisotropy radius $r_a$, IMF slope) supports
`jax.grad` through construction *and* sampling
([differentiability rules](../20-architecture/differentiability.md);
[AD-vs-FD evidence](../50-validation/multimass-equilibrium.md)).
That turns initial conditions from a preprocessing step into a
**likelihood component**: you can place a cluster model inside a
posterior, run HMC or gradient descent over its structural parameters
against star-count, dispersion, or photometric data, and let the
sampler — not a by-hand grid — explore the model family. The
[gradient demo](differentiable-ic.md) and the
[$r_h$-fit recipe](../40-howto/gradient-based-r_h-fit.md) show the
mechanics.

The second pillar is **true equilibria, with no virial-rescale
crutches**. Every released DF is sampled in detailed equilibrium —
the lowered-Maxwellian King DF, the Eddington-inverted EFF DF, the
coupled multi-component models — and the virial ratio $Q = T/|V| = 0.5$
*emerges* from the DF rather than being imposed by rescaling
velocities afterwards. For multi-component systems this is not a
nicety: a global rescale provably cannot fix a multi-population IC
(it moves every individually-correct component *away* from its own
equilibrium — the physics bug that retired progenax's legacy
two-component generator; see
[the per-component-equilibrium principle](../10-theory/populations/index.md)).
Controlled numerical experiments — relaxation, mass segregation,
dissolution — start from ICs whose equilibrium is proven by
exact-quadrature oracles, so any subsequent evolution is physics, not
initialization transient.

Third, progenax is the IC stage of an **end-to-end differentiable
forward-modeling chain** for survey-era (LSST/Gaia/HST) cluster
science: progenax ICs → [gravax](../40-howto/interface-with-gravax.md)
N-body integration and pixel-level rendering → likelihood against
observed images or catalogs, with gradients flowing the whole way back
to the IC parameters.

## The model inventory, with validity regimes

### Single-component profiles and velocity DFs

```{list-table}
:header-rows: 1

* - Model
  - Regime
  - Validated (measured)
* - **Plummer** {cite:p}`Plummer1911` — exact ergodic $f(E) \propto E^{7/2}$
  - Cored, untruncated systems; the canonical analytic test bed
  - Sampled $Q = 0.5026$ unscaled; defining condition $M(<r_h) = M/2$
    anchored ([evidence](../50-validation/plummer-equilibrium.md))
* - **King** {cite:p}`King1966` — lowered-Maxwellian DF, self-consistent
    tidal truncation
  - Relaxed, tidally truncated clusters (Galactic globulars)
  - $c(W_0)$ vs {cite:t}`King1966` Table II: max
    $|\Delta\log_{10}c| = 0.002$ over $W_0 = 2.5$–$15$; volume density vs
    independent moment oracle $1.2\times 10^{-10}$; $Q = 0.5$ unscaled
    ([evidence](../50-validation/king-profile.md))
* - **EFF** {cite:p}`ElsonFallFreeman1987` — power-law envelope
    $\rho \propto (1+r^2/a^2)^{-\gamma/2}$, **Eddington-inverted** DF
  - Young massive clusters with shallow untruncated envelopes
    (LMC/SMC clusters, $\gamma \gtrsim 4$ in the 3-D slope convention);
    mild truncation
  - $\gamma = 5$ reduces to Plummer exactly (max rel $= 0$); asymptotic
    slope to 1%; Eddington-DF $Q = 0.502$ unscaled
    ([evidence](../50-validation/eff-profile.md))
* - **Michie–King** {cite:p}`Michie1963` — self-consistent radially
    anisotropic lowered DF
  - Anisotropic relaxed clusters; radially biased halos
  - $\beta(r)$ vs the DF's own second-moment oracle: max dev $0.027$;
    isotropic limit recovers King to $2.7\times 10^{-3}$
    ([evidence](../50-validation/michie-anisotropy.md))
* - **Osipkov–Merritt anisotropy** {cite:p}`Merritt1985` (Plummer/EFF
    `anisotropy_radius`)
  - Imposed $\beta(r) = r^2/(r^2+r_a^2)$ when the anisotropy *profile*
    is the model
  - OM $\beta(r)$ realized exactly via the Merritt velocity stretch
    ([evidence](../50-validation/rotation-om-anisotropy.md))
* - **Rotation** (solid-body, differential) — additive streaming
    transforms
  - Rotating clusters; composable with any DF above
  - Streaming $v_\phi(R)$ and angular-momentum budget verified exactly
    (additive transform, no scatter noise)
    ([evidence](../50-validation/rotation-om-anisotropy.md))
```

A physically instructive contrast the suite makes explicit: the
Michie–King DF is *not* a function of the single OM integral $Q_{\rm OM}$,
so its self-consistent $\beta(r)$ sits **below** the OM ceiling and
returns toward isotropy at the tidal boundary — the
[suppressed-$\beta$ finding](../50-validation/michie-anisotropy.md).
If your science targets the anisotropy profile itself, that distinction
chooses your model for you.

### Multimass Engine A — the differentiable lowered-isothermal (LIMEPY) family

[Engine A](../10-theory/spatial-profiles/lowered-model-family.md) of
`MultiComponentCluster` implements the {cite:t}`Gieles2015` family
natively in JAX: one continuous truncation parameter $g$ spans
Woolley ($g=0$) → King ($g=1$) → Wilson ($g=2$), each of $n$ mass
components rides **one** shared self-consistent potential with its own
velocity-scale ratio $w_j = s_j/s$, mass segregation is the built-in
equipartition law $w_j = \mu_j^{-\delta}$, and each component can carry
its own Michie/OM anisotropy radius.

- **Regimes:** Galactic globular clusters and other relaxed, tidally
  truncated systems; mass-segregated multi-mass clusters; partial
  equipartition studies ($\delta \in [0, 0.5]$ standard); GC 1G/2G and
  binaries-vs-singles decompositions via $w_j$ directly.
- **Equilibrium proven:** exact-quadrature oracle reads per-component
  $Q_j = 0.5001$–$0.5002$ across $\delta \in [0, 0.6]$; sampled global
  $Q = 0.497$–$0.500$ unscaled; per-component $\sigma_{1d,j}(r)$ matches
  the analytic DF moment to $<1\%$ across resolved bins
  ([evidence](../50-validation/multimass-equilibrium.md)).
- **Differentiable structural inference:** the full parameter vector
  $(W_0, g, \{w_j\}, \delta, r_a)$ is differentiable through the coupled
  Poisson solve and the samplers — AD vs finite differences to
  $2.15\times 10^{-4}$, with $g$ itself a *fitted* quantity.
- **Performance:** DF tables make sampling practical at survey scale —
  **67×** (isotropic) and 21.7× (anisotropic) faster speed draws at
  $N = 10^5$, every approximation budget-asserted against the retained
  exact-quadrature oracle (density $6\times 10^{-6}$, speed moments
  $\le 0.28\%$/$1.5\%$).

### Multimass Engine B — density-defined Eddington equilibria

[Engine B](../10-theory/populations/eddington-engine.md) starts from
prescribed *densities*: Plummer/EFF/King density shapes with mass
fractions, one shared potential from a single quadrature pass, and each
component's DF recovered by Eddington inversion in that shared
potential — optionally with per-component OM anisotropy.

- **Regimes:** observed multi-population decompositions
  (surface-brightness halo+core fits, literature profiles taken
  verbatim), where the density — not a DF family — is the input.
- **Realizability as physics:** Eddington inversion yields the *unique*
  candidate $f_j(E)$; if it is negative anywhere, the prescribed
  decomposition **has no joint equilibrium**, and Engine B refuses,
  naming the component and the remedy. This is a falsifiable test, not
  an error message: the originally drafted halo+core example
  ($a_{\rm EFF} = 0.4$ inside a Plummer halo) is genuinely unrealizable
  ($f_{\min} = -0.20$, resolution-independent), with the realizability
  boundary measured between $a = 0.65$ and $0.68$
  ([worked example](../10-theory/populations/two-component.md)).
- **Equilibrium proven:** theory $Q_j = [0.50038, 0.50012]$; sampled
  global $Q = 0.4976$ at $N = 3\times 10^4$, unscaled; Plummer
  $f(E) \propto E^{7/2}$ analytic oracle to $1.06\times 10^{-4}$
  ([evidence](../50-validation/engine-b-eddington.md)).
- **Hard-truncation caveat, quantified:** a sharply truncated prescribed
  density carries an edge offset no ergodic $f(E)$ can represent, so
  the truncated component's sampled $Q_j$ plateaus slightly below 0.5 —
  and Engine B *predicts* the plateau (predicted $0.4953$, sampled
  $0.4947 \pm 0.0014$ over 18 seeds). The deviation is gated against
  the prediction, never rescaled away.

The two engines overlap at exactly one configuration — a single King
component — and that overlap is the **cross-engine trust anchor**: two
fully independent codepaths (coupled ODE + lowered-DF sampling vs.
quadrature potential + Eddington inversion) agree to a radial KS
distance of $2\times 10^{-4}$ and $|\sigma_B/\sigma_A - 1| \le 3\times 10^{-4}$.

### IMFs and binary populations

```{list-table}
:header-rows: 1

* - Capability
  - Regime / scope
  - Validated (measured)
* - Canonical IMFs — Salpeter, Kroupa, Chabrier, Maschberger, truncated
    power law {cite:p}`Salpeter1955,Kroupa2001,Chabrier2003,Maschberger2013`
  - Single-star birth-mass distributions; Maschberger is the smooth
    analytically invertible choice for inference
  - KS goodness-of-fit and recovered $\alpha$ vs truth
    ([evidence](../50-validation/imf-statistics.md))
* - Environment-dependent IMF — {cite:t}`Marks2012` cluster-scale
    variation + {cite:t}`Jerabkova2018` IGIMF
  - IMF as a function of birth density and metallicity
  - [evidence](../50-validation/environment-imf.md); segment-conversion
    documented at [](../10-theory/imfs/environment.md)
* - Binary statistics — {cite:t}`MoeDiStefano2017` joint $P$–$q$–$e$
    interrelation (`MoeCompanions`), {cite:t}`Sana2012` OB periods,
    thermal/uniform/Moe eccentricities
  - Field-calibrated multiplicity; the non-separable $P$–$q$–$e$
    coupling is sampled *jointly*, not as independent marginals
  - Moe+17 $q$ sampler vs implemented PDF: KS $D = 0.0021$; twin
    fraction $15.2\% \to 6.3\%$ from $1$ to $10\,M_\odot$
    ([evidence](../50-validation/binary-imf.md))
* - Binary-aware IMF recovery
  - Inference from unresolved system masses at survey $N$
  - The headline: a naive single-star fit at $N = 10^5$ recovers
    $\hat\alpha = 2.21$ against a true $2.30$ — **17.8σ confidently
    wrong** — while the binary-aware marginalised likelihood recovers
    $\hat\alpha = 2.298$ ($0.4\sigma$)
* - Binary→spatial connector + energy budgets
  - Resolved component positions/velocities for collisional N-body;
    COM-virialised per the McLuster convention {cite:p}`Kuepper2011`
  - Kepler III exact to machine precision; orbital energy
    $4.2\times 10^{-16}$; `binary_energy_budget` reports the internal
    binding-energy reservoir explicitly
    ([theory](../10-theory/binaries/index.md);
    [evidence](../50-validation/binary-imf.md))
```

### Substructure and segregation diagnostics

- **CW04 $Q$** {cite:p}`Cartwright2004` (`compute_q_parameter`, exact
  MST estimator) plus the differentiable kNN `q_approx` — separates
  centrally concentrated ($Q > 0.8$) from substructured ($Q < 0.8$)
  clusters; uniform-sphere and Table-1 anchors verified
  ([evidence](../50-validation/fractal-substructure.md)).
- **$\Lambda_{\rm MSR}$** {cite:p}`Allison2009` validated against
  analytic ground truth (realistic segregated regime
  $\Lambda = 4.8 \pm 0.7$, against the observed ONC-like range), plus
  three **differentiable segregation observables** (soft
  $\Lambda_{\rm MSR}$, radial concentration, $\Sigma$–$m$) that converge
  to their exact estimators in the hard limit (to $10^{-11}$–$5.6\times 10^{-3}$)
  and come **Fisher-information-ranked**: radial concentration carries
  $\mathcal{I} = 649$ vs $132$ (soft $\Lambda$) and $143$ ($\Sigma$–$m$)
  at the tested operating point — autodiff-exact identifiability
  guidance for which observable to infer with, including 2D-projection
  information loss ([evidence](../50-validation/mass-segregation.md)).
- **Primordial vs equilibrium segregation, as separate labeled routes:**
  `energy_sorted_segregation` ({cite:t}`Baumgardt2008`-style
  energy-ranked, explicitly *non-equilibrium/primordial*) vs Engine A's
  `from_mass_segregation` (segregation as a true equilibrium) — see
  [](../10-theory/tidal-and-substructure/mass-segregation.md).
- **Tidal truncation** — the Jacobi radius reproduces the restricted
  three-body L1 point across seven decades in mass ratio (residual
  $1.1\times 10^{-3}$ for a Galactic globular), differentiable in the
  truncation radius ([evidence](../50-validation/tidal-truncation.md)).

### Analytical anchors

Exact-solution IC builders for integrator validation: the two-body
Kepler ellipse ($E = -Gm_1m_2/2a$ to machine precision, closes to
$10^{-7}$), the Chenciner–Montgomery figure-eight (exactly $L = 0$,
closes to $4\times 10^{-8}$), the eight-planet solar system (Kepler III
vs observed sidereal periods to 0.7%), and the harmonic oscillator
([evidence](../50-validation/analytical-test-cases.md)). These exist so
that when you hand progenax ICs to an integrator, you can first prove
the *integrator* against ICs whose evolution is known exactly.

## Twelve concrete science questions

Each question maps to a documented, validated capability — no entry
here outruns the test suite.

```{list-table}
:header-rows: 1

* - Research question
  - Capability (page)
* - What are the joint posteriors of $(W_0, g, M, r_h)$ for a Galactic
    globular from HST/Gaia star counts — with the truncation shape $g$
    a *fitted* parameter (King-vs-Wilson as a posterior, not a
    modelling choice)?
  - Engine A differentiable structural inference
    ([lowered-model family](../10-theory/spatial-profiles/lowered-model-family.md))
* - Do two chemically distinct GC populations (1G/2G), each with its
    own concentration and kinematics, admit a *joint* dynamical
    equilibrium in one shared potential?
  - Engine A `from_components` with per-component $w_j$
    ([worked example](../10-theory/populations/two-component.md))
* - Is an observed halo+core surface-brightness decomposition
    dynamically realizable at all — and at what core scale does the
    answer flip? (Realizability as a falsifiable test.)
  - Engine B $f_j \ge 0$ gate, boundary measured to
    $a \in (0.65, 0.68)$ in the headline example
    ([Eddington engine](../10-theory/populations/eddington-engine.md))
* - What degree of equipartition $\delta$ does a cluster's
    per-mass-group velocity dispersion imply, fitted by gradient
    rather than by model grid?
  - Engine A $w_j = \mu_j^{-\delta}$, differentiable in $\delta$
    ([evidence](../50-validation/multimass-equilibrium.md))
* - Is an observed mass-segregation signal primordial or dynamical?
    Compare a primordial-segregated IC and an equilibrium-segregated IC
    at matched $\Lambda_{\rm MSR}$, then evolve both.
  - The two labeled segregation routes
    ([mass segregation](../10-theory/tidal-and-substructure/mass-segregation.md))
    + gravax handoff
* - Which segregation observable should a survey invest in — and how
    much of the signal survives sky projection?
  - Fisher-ranked differentiable observables; 2D/3D information ratios
    ([evidence](../50-validation/mass-segregation.md))
* - How badly do unresolved binaries bias an IMF slope measured from
    system masses, and does a binary-aware likelihood remove the bias
    at survey $N$?
  - The 17.8σ "confidently wrong" result + 0.4σ binary-aware recovery
    ([evidence](../50-validation/binary-imf.md);
    [theory](../10-theory/imfs/binary.md))
* - What were the initial conditions of LMC/SMC young massive clusters
    with shallow EFF envelopes — sampled in true Eddington equilibrium
    rather than with a Maxwellian approximation?
  - EFF profile + Eddington DF, $\gamma$-truncation regime
    ([theory](../10-theory/spatial-profiles/eff.md);
    [evidence](../50-validation/eff-profile.md))
* - What anisotropy radius do proper-motion dispersion profiles imply —
    and does the data prefer the self-consistent (suppressed-$\beta$)
    Michie model or an imposed OM profile?
  - Michie–King vs OM contrast
    ([evidence](../50-validation/michie-anisotropy.md);
    [rotation & anisotropy](../10-theory/velocity-dfs/rotation-anisotropy.md))
* - How does a cluster on a given Galactic orbit dissolve? Build a
    tidally truncated equilibrium IC, hand off to gravax, and forecast
    mass loss.
  - Jacobi radius + `apply_tidal_truncation`
    ([tidal physics](../10-theory/tidal-and-substructure/tidal.md);
    [gravax handoff](../40-howto/interface-with-gravax.md))
* - Does birth environment (cluster density, metallicity) imprint on
    the IMF in resolved-cluster data?
  - Environment-dependent IMF mapping
    ([theory](../10-theory/imfs/environment.md);
    [evidence](../50-validation/environment-imf.md))
* - Can cluster structural parameters be inferred from LSST
    crowded-field *pixels* directly, with gradients end-to-end through
    IC → N-body → rendered image?
  - progenax differentiable ICs
    ([differentiability](../20-architecture/differentiability.md)) +
    gravax integration/rendering — an ecosystem workflow; the rendering
    stage lives in gravax, not progenax
```

## What progenax does not do (honest scope)

:::{admonition} Read this before building a project on progenax
:class: warning
- **No dynamical evolution.** progenax generates $t = 0$. Integration,
  relaxation, encounters, and binary evolution are
  [gravax](../40-howto/interface-with-gravax.md)'s job. Primordial
  binary labels go stale under evolution — the *current* binary
  population of an evolved snapshot must be measured
  (`find_bound_pairs`), not read from the labels.
- **Hard truncation edges are approximate (Engine B), quantified.**
  A hard-truncated prescribed density is only approximately stationary
  at its edge; the $Q_j$ deviation is predicted and gated
  ($0.4947 \pm 0.0014$ vs predicted $0.4953$), not hidden — and not
  rescaled away.
- **Not every density decomposition is realizable.** Engine B will
  refuse decompositions whose Eddington DF is negative. This is a
  feature — the refusal is a physical result — but it means you cannot
  feed it arbitrary profile mixes and expect an IC back.
- **No exact binary-aware `logpdf` in the released API.** The
  binary-aware recovery result uses the marginalised-likelihood
  validation script; the released `BinaryIMF` exposes sampling helpers
  (see [](../10-theory/imfs/index.md)).
- **Turbulent/fractal IC generation is experimental and repo-only**
  (the `gravoturb` package, not in the released wheel). The
  released core retains only the substructure *diagnostics*
  ([CW04 Q](../50-validation/fractal-substructure.md)).
- **Known current limitations of the multi-component engines**
  (Engine-A-only accessors on Engine B models, sampler scale thresholds
  at $r_h \gtrsim 10^4$ pc, and related items) are documented in the
  [Engine B limitations admonition](../10-theory/populations/eddington-engine.md)
  with fixes tracked — check that page for the current state.
:::

## Why you can trust the numbers

progenax's persuasion strategy is measured numbers, not adjectives. The
released core carries a four-figure test suite spanning unit,
integration, and physics-validation tiers (see the
[test dashboard](../50-validation/test-dashboard.md) for the live
count), and the validation tier is built on three habits worth knowing
before you rely on the package:

1. **Independent oracles.** Equilibrium claims are proven by
   exact-quadrature oracles deliberately independent of the sampled
   draws (and of the DF-table approximations they check); sampled
   clusters are verified *unscaled*.
2. **A cross-engine anchor.** The one configuration both multimass
   engines describe — a single King component — agrees through two
   fully disjoint numerical pipelines at the $10^{-4}$ level.
3. **Anchors on defining conditions, not derived constants** — the
   pattern that has empirically caught transcription, inversion, and
   differentiability regressions
   ([methodology](../50-validation/methodology.md)).

Differentiability is what converts all of this into scientific
leverage: when the IC pipeline is a likelihood component, the same
validated physics that generates your simulation's $t = 0$ also
computes $\partial(\text{model})/\partial(\text{parameters})$ for your
posterior — exactly, by autodiff, through the same code path the
forward model uses.
