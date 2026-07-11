---
title: "The differentiable lowered-model family (Engine A)"
description: "progenax's implemented, differentiable generalization of King/Wilson/Woolley into one lowered-isothermal family (continuous truncation parameter g, multi-component velocity scales, per-component anisotropy), following the Gieles & Zocchi (2015) formalism — implemented natively in JAX, not as an external dependency."
---
# The differentiable lowered-model family (Engine A)

:::{admonition} Status — implemented
:class: tip
This family is **implemented and validated**. It is "Engine A" of the unified
[`MultiComponentCluster`](../populations/index.md) model: every component is a
lowered-isothermal DF riding one shared self-consistent potential, solved by
the coupled-Poisson core `solve_multicomponent_limepy(alpha_j, rescale_j, W0,
g, ra_hat_j)`. The complementary density-defined route ("Engine B",
[Eddington inversion in a shared potential](../populations/eddington-engine.md))
covers the case where the *densities* are prescribed instead of the DF.
:::

:::{admonition} Who this page is for
:class: note
**Audience:** students & researchers who already know the single-mass King model and want the unified, differentiable lowered-isothermal family (Woolley/King/Wilson + multi-mass + anisotropy).
**Prerequisites:** the [King profile](king.md) (the $g = 1$ member this generalizes) and the [virial-$Q$ convention](../../20-architecture/q-virial-convention.md); the [multi-component populations](../populations/index.md) overview gives the Engine A/B framing.
**You'll get:** the continuous truncation parameter $g$, how $n$ mass components share one self-consistent potential via velocity-scale ratios $w_j$, and why progenax reimplemented LIMEPY natively to make $g$ and the segregation parameters *differentiable*.
:::

## The idea

The classical truncated-isothermal models — Woolley (no energy truncation),
{cite:t}`King1966` (a linear lowered Maxwellian), and Wilson (a quadratic
lowered form) — are members of **one** family distinguished by how sharply the
distribution function is truncated near the escape energy. {cite:t}`Gieles2015`
made this explicit with a single continuous **truncation parameter** $g$: their
LIMEPY DF reads

```{math}
:label: limepy-df-family
f(E) \;\propto\; e^{-E/\sigma^2}\,
\gamma\!\Big(g,\; \tfrac{E_{\rm cut}-E}{\sigma^2}\Big),
```

[↗ model card](#card-limepy-df-family)

with $g=0$ recovering the Woolley cutoff, $g=1$ the King model, and $g=2$ the
Wilson model, interpolating smoothly in between. The same framework extends
naturally to **multiple mass components** (each mass group its own
$\sigma_j$, sharing one potential) and to **radial anisotropy** (an
Osipkov–Merritt / Michie term).

## The coupled multi-component equilibrium

progenax generalizes the single-mass solve to $n$ components sharing one
dimensionless potential $\psi(\xi)$, $\xi = r/r_c$. The model-defining insight
is that a component's **single free Engine-A scale is its velocity-scale
ratio**

```{math}
:label: wj-def
w_j \;=\; \frac{s_j}{s}, \qquad \texttt{rescale}_j \;=\; w_j^{-2},
```

so component $j$ sees the shared potential at the rescaled depth
$W_j(\xi) = \texttt{rescale}_j\,\psi(\xi)$ and contributes a density
$\hat\rho_j \propto E_\gamma\!\big(g+\tfrac32,\, \texttt{rescale}_j\,\psi\big)$
(the "lowered exponential" of {cite:t}`Gieles2015`, Appendix B), giving the
coupled Poisson equation

```{math}
:label: multimass-poisson
\frac{1}{\xi^2}\frac{\mathrm{d}}{\mathrm{d}\xi}
\Big(\xi^2\frac{\mathrm{d}\psi}{\mathrm{d}\xi}\Big)
\;=\; -\,9\sum_j \alpha_j\,\hat\rho_j(\xi),
```

[↗ model card](#card-multimass-poisson)

with $\alpha_j$ the central density fractions ($\sum_j\alpha_j = 1$). A
*colder* component ($w_j < 1$) feels a deeper effective well and concentrates
— as a **genuine equilibrium**, not a post-hoc radial reshuffling. Mass
segregation is then the equipartition convenience

```{math}
:label: mass-seg-law
w_j \;=\; \mu_j^{-\delta}, \qquad \mu_j = m_j/\bar m, \quad
\bar m = \sum_j m_j\,\alpha_j,
```

[↗ model card](#card-mass-seg-law)

({cite:t}`Gieles2015`, Eqs. 24–26): $\delta = 1/2$ is the standard
partial-equipartition choice, $\delta = 0$ collapses every component to the
single-mass model exactly (the cleanest oracle). The representative stellar
masses $m_j$ are otherwise **decoupled labels** — the structure depends only
on $(\alpha_j, w_j, \texttt{ra\_hat}_j, W_0, g)$, so equal-mass populations of
different concentration (GC 1G/2G, halo+core, binaries-vs-singles) set $w_j$
directly.

Per-component radial anisotropy enters as an Osipkov–Merritt / Michie term
with its own anisotropy radius $\hat r_{a,j} = r_{a,j}/r_c$
({cite:t}`Michie1963`; {cite:t}`Merritt1985`); the anisotropic density is
evaluated by an exact, numerically stable 1-D quadrature (the bounded
$T(\beta)$ Poisson sum) because JAX's `hyp1f1` is NaN for arguments
$\gtrsim 100$.

:::{admonition} A misprint worth knowing about
:class: note
The density index of the lowered-isothermal family is
$E_\gamma(g+\tfrac32, \hat\phi)$ — printed **correctly** in
{cite:t}`Gieles2015` main-text Eq. 8 (p. 578). What the published 2018
**erratum** (MNRAS 474, 3997) corrects are the $\hat\phi \to 0$
*limiting expressions* Eqs. 20–21 (whose corrected right-hand sides carry
exactly these $E_\gamma(g+\tfrac32,\cdot)$ / $\Gamma(g+\tfrac52)$ indices)
and a typo in the $\sigma_S^2(r)$ Eq. 41 — the LIMEPY code itself was
always correct. progenax verified the index three independent ways
(main-text Eq. 8, the erratum, and the King $g=1$ corner) before
implementing `limepy_density_hat`. (An earlier revision of this note
mislocated the misprint as "main text $g+\tfrac12$ vs Appendix B" —
corrected 2026-07-11 against both PDFs.)
:::

## Entry points

The family is exposed through three `MultiComponentCluster` constructors
(see [](../populations/index.md) for the engine map and
[](../populations/two-component.md) for a worked example):

```python
from progenax import MultiComponentCluster

# General: per-component velocity-scale ratios (GC 1G/2G, halo+core, ...)
model = MultiComponentCluster.from_components(
    alpha_j=[0.5, 0.5], w_j=[0.7, 1.0], m_j=[1.0, 1.0], W0=7.0, g=1.0)

# Mass segregation as equipartition: w_j = mu_j^(-delta)
model = MultiComponentCluster.from_mass_segregation(
    alpha_j=[0.6, 0.4], m_j=[0.3, 3.0], W0=7.0, g=1.0, delta=0.5)

# IMF-driven: bin the IMF, eigenvalue-solve for the alpha_j that
# reproduce the per-bin mass budget
model = MultiComponentCluster.from_imf(imf, n_comp=8, W0=7.0, g=1.0, delta=0.5)

ic = model.sample_cluster(key, n_stars=50_000)   # ICResult with component_id
```

`sample_cluster` draws every star from its component's lowered DF at the
rescaled potential $W_j(r)$ and velocity scale $s_j = s\,w_j$ — **no external
virial rescale anywhere**: each component is individually virial
($Q_j = 0.5$) and the sampled cluster is globally virial for *any* mass
spectrum. The unequal-mass two-population regression measures theory
$Q_j = 0.5 \pm 2\times 10^{-3}$, sampled global $Q$ within $\pm 0.04$ and
per-component $Q_j$ within $\pm 0.07$ at test resolution. The single-mass
corners were validated against the released models directly: $g=1$ isotropic
$\equiv$ [King](king.md), $g=1$ anisotropic $\equiv$
[Michie](../velocity-dfs/michie-king.md), both with $Q = 0.5$ unscaled.

## The DF-table performance layer

The anisotropic density quadrature is the 86% hotspot of the coupled solve,
and per-star speed draws dominated sampling. Phase 1.5 replaced both with
three differentiable table primitives in `profiles/limepy_tables.py` — with
the **exact quadrature retained everywhere as a selectable oracle**
(`aniso_method="quadrature"`), and every approximation budget asserted in
tests:

```{list-table} DF tables — design and measured budgets (every row is a regression test against the retained quadrature oracle).
:header-rows: 1

* - Table
  - Grid / scheme
  - Measured vs oracle
* - `AnisoDensityTable` — $\hat\rho(W, p;\, g)$
  - $(\sqrt{W},\, \mathrm{asinh}\,p)$ grid, tensor-product 4-point **cubic
    Lagrange** ($O(h^4)$); 512×96 pointwise, 160×40 in-solve
  - density $6.05\times 10^{-6}$ (budget $10^{-5}$); solve
    $|\Delta\psi| \le 1.93\times 10^{-4}$ over 3 configs; mass CDF
    $4.48\times 10^{-5}$
* - `SpeedCDFTable` — isotropic inverse speed CDF
  - 256×256 on $(\sqrt{W},\, u/\sqrt{2W})$; row-relative CDF normalization;
    gated to $g \in [0, 3.5]$
  - speed moments $\le 0.28\%$ vs DF quadrature
* - `AnisoSpeedCDFTable` — anisotropic speed *marginal*
  - 192×48×192 on $(\sqrt{W},\, \mathrm{asinh}\,p,\, u/\sqrt{2W})$; the
    angular conditional $(\cos\theta \mid u, p)$ stays **exact**
  - speed moments $\le 1.5\%$; sampled $\beta(r) \equiv$ the DF's own
    quadrature $\beta$ within $0.06$ (unchanged)
```

Measured speedups (warm, two-component, $W_0 = 7$): anisotropic
**construction 5.6× faster** (957 → 170 ms); sampling at $N = 10^5$:
**isotropic 67×** (0.48 µs/star), **anisotropic 21.7×** (2.9 µs/star). The
anisotropic table build dominates at small $N$ — break-even is ~3k stars.
Crucially, the equilibrium oracle `component_virial_ratios` is deliberately
**quadrature-only** (it must not share the approximation it checks): a
table-built model proves $Q_j = 0.5001 \pm 1.5\times 10^{-4}$ against it.

## Why progenax reimplemented it (rather than depend on LIMEPY)

{cite:t}`Gieles2015` is the reference *formalism*, and the published `limepy`
code is the standard numpy/scipy implementation. progenax does **not** wrap or
depend on it, for one decisive reason: **`limepy` is not differentiable.** The
entire progenax thesis is JAX-native, end-to-end-differentiable initial
conditions, so that structural parameters can be **inferred** from data by
gradient descent or HMC (see [](../../20-architecture/differentiability.md)).

The implemented family delivers exactly this: a single continuous parameter
vector — $(W_0, g, \{w_j\}, \delta, r_a)$ — can be fit jointly to an observed
cluster, with $g$ itself a *fitted* quantity that selects the truncation
sharpness the data prefer (King-vs-Wilson as a posterior, not a modelling
choice). $\partial(\text{model})/\partial g$ flows through the coupled Poisson
solve **and** through the DF tables: AD agrees with finite differences to
$2.15\times 10^{-4}$, and the table-backed gradient agrees with the
quadrature-backed gradient to $2.1\times 10^{-4}$. The anisotropic and
mass-segregation parameters $(r_a, \eta, \delta)$ are likewise
differentiable through construction and sampling.

## Delivered scope

```{list-table}
:header-rows: 1

* - Capability
  - Status
  - Measured
* - King ($g=1$), single mass
  - ✅ released ([King](king.md))
  - $c(W_0)$ vs King (1966) Table II, $\Delta \le 0.02$
* - Continuous $g$ (Woolley $\to$ King $\to$ Wilson)
  - ✅ delivered (`solve_multicomponent_limepy`, differentiable in $g$)
  - $g=1$ corner $\equiv$ King (iso) / Michie (aniso), $Q = 0.5$ unscaled
* - Multi-mass / multi-component equilibrium (per-component $s_j = s\,w_j$)
  - ✅ delivered (`from_components` / `from_mass_segregation` / `from_imf`)
  - theory $Q_j = 0.5 \pm 2\times 10^{-3}$; sampled global $\pm 0.04$,
    per-component $\pm 0.07$
* - Radial anisotropy per component ($\hat r_{a,j}$, Osipkov–Merritt / Michie)
  - ✅ delivered
  - sampled $\beta_j(r) \equiv$ DF quadrature $\beta$ within $0.06$
* - DF-table acceleration (oracle-backed)
  - ✅ delivered (Phase 1.5)
  - 5.6× construction; 67× (iso) / 21.7× (aniso) sampling at $N = 10^5$
* - Differentiable structural inference $(W_0, g, w_j, \delta, r_a)$
  - ✅ delivered
  - AD-vs-FD $2.15\times 10^{-4}$; table-AD $\equiv$ quadrature-AD
    $2.1\times 10^{-4}$
```

## Relationship to the other roadmap item

This is independent of, but complementary to, the now-**resolved differentiable
tidal radius** $\partial r_t/\partial W_0$
([](../../20-architecture/differentiability.md#roadmap-differentiable-rt)): the
scalar $r_t$ already carries a finite, exact gradient through the unclamped
$\psi=0$ crossing. In the unified family $r_t$ is a function of $(g, W_0)$; the
same implicit-function-theorem treatment of the $\psi=0$ crossing carries over
to the $g$ generalization unchanged.

## Implementation, validation & references

- **In code:** `src/progenax/cluster/multicomponent.py`
  (`MultiComponentCluster` and the coupled-Poisson core
  `solve_multicomponent_limepy`), with the lowered-density kernel in
  `src/progenax/profiles/limepy_multimass.py` and the DF-table layer in
  `src/progenax/profiles/limepy_tables.py`. See the
  [`MultiComponentCluster` API](../../30-api/cluster.md).
- **Validated in:** [multimass equilibrium](../../50-validation/multimass-equilibrium.md)
  — per-component $Q_j$, the single-mass King/Michie corners, and the
  AD-vs-FD gradient checks for $(W_0, g, w_j, \delta, r_a)$.
- **Primary sources:** {cite:t}`Gieles2015` (the LIMEPY lowered-model
  formalism); single-model members are {cite:t}`King1966` (with Woolley
  1954 / Wilson 1975 for the $g = 0, 2$ endpoints), and anisotropy
  follows {cite:t}`Michie1963` and {cite:t}`Merritt1985`. Full notes in
  the bibliography:
  [Gieles & Zocchi 2015](../../99-bibliography/per-paper/gieles-zocchi-2015.md)
  and [King 1966](../../99-bibliography/per-paper/king-1966.md).
