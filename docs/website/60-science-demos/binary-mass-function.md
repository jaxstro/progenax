---
title: Unresolved-binary mass function (B4)
description: "Historical synthetic binary-fraction recovery with a global mass-ratio shuffle, a semimajor-axis blending rule and a ZAMS luminosity model; scope and confounding correlations are stated explicitly."
---

# Unresolved-binary mass function (B4)

A binary that a survey cannot resolve appears as a **single, brighter** source, and
its inferred mass is biased high — so unresolved binaries distort the observed
stellar mass function. This demo recovers the binary fraction $f_b$ from that
distortion under a specified synthetic model. The retained comparison
shows sensitivity to globally removing mass-ratio correlations from the
{cite:t}`MoeDiStefano2017`-based template; it does not isolate period–mass-ratio
coupling from primary-mass dependence.

## The forward model

Each binary has a primary $m_1$ (from a Maschberger IMF), a mass ratio $q$
($m_2 = q\,m_1$) and a period $P$; the semimajor axis $a$ follows Kepler's third
law. A survey **resolves** wide pairs and **blends** close ones, $a < a_{\rm crit}$
(here $50\,\mathrm{AU}$):

```{math}
:label: b4-blend
\text{single} \to m_1, \qquad
\text{resolved} \to \{m_1, m_2\}, \qquad
\text{blended} \to m_{\rm obs} = L^{-1}\!\big(L(m_1) + L(m_2)\big) > m_1,
```

where $L(m)$ is the **Tout et al. (1996) ZAMS mass–luminosity relation**
{cite:p}`Tout1996` (provided in-package by `progenax.stellar`). The observed
mass function is a linear mixture in $f_b$,

```{math}
:label: b4-mixture
\mu_k(f_b) = N_{\rm sys}\big[(1-f_b)\,S_k + f_b\,B_k\big],
```

with the single template $S_k$ (the IMF) and the binary template $B_k$ (per-binary
expected catalogue stars after resolution + blending) precomputed from a large
pool. The script fits a per-bin independent-Poisson working likelihood.
Its uncertainty uses the observed objective curvature (`fisher_cov`).
Because data contain a fixed number of systems and a resolved binary can
contribute two correlated entries, that working likelihood does not
establish an independent-Poisson generative law or calibrated coverage.

## The global mass-ratio shuffle

The comparison globally permutes the pool's $q$ array while holding each
$(m_1,P)$ pair fixed (`scripts/demo_binary_mass_function.py::build_templates`).
It preserves the separate empirical $q$, $P$ and $m_1$ marginals, but removes
$q$ dependence on **both primary mass and period**. The contrast therefore
does not attribute a bias to period coupling alone. It compares the generating
joint template with this particular globally decoupled approximation.

## Inputs and assumptions

The fit recovers **one parameter**, the binary fraction $f_b$; everything else is
an **assumed-known input**. The recovered-vs-known split is the key onboarding
lesson — the demo's punchline (that the $P$–$q$ coupling must be right) is a
statement about one of those *assumed* inputs.

```{list-table} Model inputs
:header-rows: 1
:label: tbl-b4-inputs

* - Input
  - Meaning and role
  - Status (fiducial)
* - $f_b$
  - Binary fraction — **the science target**, recovered from the Poisson mixture $\mu_k=N_{\rm sys}[(1-f_b)S_k+f_b B_k]$ (`FB_BOX=(0.05,0.95)`).
  - **recovered** (truth 0.5)
* - Moe $P$–$q$–$e$ coupling
  - The joint orbital model (`MoeJointOrbit`) that builds the *correct* binary template $B_k$; the shuffle changes both $q$–$P$ and $q$–$m_1$ dependence.
  - known / fixed (assumed correct)
* - $q_{\min}$
  - Minimum mass ratio in `MoeDiStefano2017Full`; truncates the secondary masses feeding the blend.
  - known / fixed (0.1)
* - $a_{\rm crit}$
  - Resolution limit: pairs with semimajor axis $a<a_{\rm crit}$ **blend**, wider ones **resolve** (selects which binaries distort the MF).
  - known / fixed (`A_CRIT_AU=50` AU)
* - $Z$
  - Metallicity for the Tout ZAMS $L(m)$ used to compute the blend mass $m_{\rm obs}=L^{-1}(L_1+L_2)$.
  - known / fixed (`Z_MET=0.02`, solar)
* - IMF
  - Maschberger ($\alpha=2.3$, $0.08$–$100\,M_\odot$): draws primaries $m_1$, hence the single template $S_k$ and the blend masses.
  - known / fixed
* - $N_{\rm sys}$
  - Systems in the mock cluster; sets the Poisson counts and the **significance** of the wrong-model systematic ($\propto\sqrt{N_{\rm sys}}$).
  - known / fixed ($3\times10^5$)
* - $N_{\rm pool}$, `M_BINS`, `SEED`
  - Template pool size ($6\times10^5$), mass-function bins (30 log bins), and PRNG seeds (templates / data / $q$-shuffle).
  - numerical choices
```

```{important}
:label: imp-b4-coupling
**The joint template matters in the retained example.** The blending
selection uses $a<a_{\rm crit}$, where
$a^3=Gm_1(1+q)P^2/(4\pi^2)$, so it depends on primary mass and mass
ratio as well as period. Under the generating joint model, blends are preferentially
**high-$q$** (median $q=0.54$ for blends vs $0.48$ when $q$ is shuffled against
$P$). The global shuffle changes both $q$–$P$ and $q$–$m_1$ relations,
and hence also secondary masses and the blend selection. Its retained fit
is about 5% low. This is a model-comparison result for the chosen templates,
not a unique causal measurement of period coupling. Joint statistics are
assumed inputs; no joint-parameter recovery is performed.
```

## Retained June 12 result

The script records a 2026-06-12 run ($N_{\rm sys}=3\times10^5$, $N_{\rm pool}=6\times10^5$,
$a_{\rm crit}=50$ AU, solar $Z$; wall $\approx 10$ s; exit 0). These historical figures and gate values
were not regenerated for this review. Optimizer convergence, template-noise
propagation and repeated-catalogue coverage remain unqualified.

**The mechanism.** The blended (close-pair) subset is **more equal-mass under Moe**
than under the decoupled model — median $q = 0.540$ vs $0.479$ — under the generating joint model. The global shuffle changes more than
the conditional period relation, so this median is a mechanism diagnostic,
not a complete causal decomposition.

```{list-table}
:header-rows: 1

* - model fitted
  - recovered $f_b$
  - vs truth $0.500$
* - Moe template (correct)
  - $0.5016 \pm 0.0074$
  - $+0.22\sigma$  ✓
* - independent template (wrong)
  - $0.4749 \pm 0.0070$
  - $-3.6\sigma$  (≈5% low)
```

The globally shuffled template gives a retained $f_b$ estimate about 5% lower. This is a *systematic*
(a wrong-model error), so its fractional size is fixed while its significance grows
as $\sqrt{N_{\rm sys}}$: only $-1.5\sigma$ at $N_{\rm sys}=5\times10^4$, but
$-3.6\sigma$ at the $3\times10^5$ survey scale here. That scaling treats the templates and working-likelihood variance as fixed;
a real survey requires its selection, mass errors and count covariance.
Historical script gates:

```{list-table}
:header-rows: 1

* - Check
  - Gate
  - Status
* - mechanism: blended median $q$ (Moe > independent)
  - coupling present
  - **PASS**
* - self-consistency (Moe template @ truth)
  - $<4\sigma$ Poisson ($2.76$)
  - **PASS**
* - $f_b$ recovery (Moe template)
  - $<3\sigma$ ($+0.22$)
  - **PASS**
* - $f_b$ bias (independent template)
  - $>3\sigma$ ($-3.6$)
  - **PASS**
```

## Figure

:::{figure} figures/demo_binary_mass_function.png
:label: sci-binary-mass-function
:width: 100%

**Unresolved-binary mass function** (`scripts/demo_binary_mass_function.py`,
historical June 12 run reporting passing script gates). **(a)** The observed mass function (black) decomposed into singles (blue) and
the binary contribution (vermilion) — the blended pairs push a bump to higher
inferred mass. **(b)** The mechanism: the blended subset's median $q$ is higher under
the Moe coupling (vermilion) than the $P$-shuffled independent model (sky) at the
same marginals. **(c)** Recovered $f_b$: the Moe template recovers the truth (dashed)
while the independent template is biased low.
:::

## Caveats

```{warning}
- **Photometry is the Tout (1996) ZAMS relation** ($L$–$M$ at fixed metallicity),
  from `progenax.stellar` (a startrax placeholder) — a clean main-sequence stand-in.
  Real surveys add band-dependent blending, extinction, and unequal evolutionary
  states; full SED/bandpass photometry is the `fluxax` package's job, not this demo.
- **No external dependency.** The ZAMS relations are in-package, so B4 runs in a
  package environment without an external stellar-data service. This is
  not evidence that a particular clean installation or CI job was executed.
- **f_b only.** The demo fits the binary fraction while holding joint
  statistics fixed; jointly recovering the $q$-distribution shape is a
  natural extension. The recovery assumes a known IMF and resolution limit.
- **Clean census + flat $f_b$.** A constant binary fraction and complete catalogue;
  Moe's own mass-dependent $f_b(m_1)$ and selection are not modelled here.
- **$a_{\rm crit}$ is a sharp semimajor-axis cut.** Blending is a hard step at
  $a<a_{\rm crit}$ on the *physical* semimajor axis — no distance, projection, or
  flux-contrast dependence, and **eccentricity is drawn but unused** (resolution
  depends only on $a$, so eccentric pairs that would be resolvable near apocenter
  are not modelled).
- **Self-consistent by construction.** Templates and data share the same IMF,
  $q_{\min}$, $Z$ (=solar), $a_{\rm crit}$ and pool family; the relations of $q$ with both
  $P$ and $m_1$ are changed by the shuffle.
  No conditional-on-$m_1$ decorrelation control or IMF / $a_{\rm crit}$ /
  metallicity misspecification study is retained here.
- **Stellar-fit domain.** The Tout fit declares masses 0.1–100 solar masses.
  This primary support starts at 0.08, secondaries can be lower, and blended
  inversions can reach the inverse solver's bounds. Those cases are
  extrapolated or clipped, rather than calibrated stellar photometry.
- **No photometric noise.** The blend mass is the exact $L^{-1}(L_1+L_2)$ with no
  measurement scatter; binning is the only smoothing. The $f_b$ uncertainty is the
  inverse-Hessian (`fisher_cov`) Gaussian approximation, and the MLE is seeded at
  the truth (a clean-start convergence check, not a multimodality test).
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_mass_function.py
```

## References

The binary $P$–$q$–$e$ coupling is {cite:t}`MoeDiStefano2017`; the ZAMS
mass–luminosity relation is {cite:t}`Tout1996`. The Moe and companion models are
documented on the [binary statistics](../10-theory/imfs/multiplicity-statistics.md)
theory pages.
