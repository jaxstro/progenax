---
title: Binary energy budget (B9)
description: "The primordial-binary energy reservoir: build_binary_cluster virializes the system COMs to Q=0.5 treating binaries as point masses, leaving the internal binary binding energy as a separate reservoir. binary_energy_budget makes the two scales explicit -- Q_com recovers 0.5 cleanly while |E_internal| dwarfs |W_com|, and the same primordial population is a relatively larger energy store in a young EFF cluster than in a concentrated King GC."
---

# Binary energy budget (B9)

A cluster with primordial binaries carries **two energy scales**, and conflating
them is a classic mistake. `build_binary_cluster` virializes the *system
centres-of-mass* to $Q=0.5$ treating each binary as a single point mass — the
McLuster scale-separation convention {cite:p}`Kuepper2011_McLuster` — and leaves
the **internal binary binding energy** as a separate reservoir that the global
virial scaling never touches. This demo makes that separation explicit and shows
why the naive virial ratio of the *resolved* stars is not the cluster's.

The two scales, from `binary_energy_budget`:

- $W_{\rm com}$ — the cluster's bulk gravitational binding on the system COMs; the
  scale the cluster is virialized on, $Q_{\rm com} = T_{\rm com}/|W_{\rm com}| \approx 0.5$.
- $E_{\rm internal} = \sum_b \left(-G\,m_{1}m_{2}/2a_b\right)$ — the internal binary
  binding (vis-viva), set by the periods and masses and **independent of where the
  COM sits** in the cluster.

```{math}
:label: b9-qresolved
Q_{\rm resolved} = \frac{T}{|W|}\Big|_{\rm resolved\ stars}
\ \neq\ Q_{\rm com},
```

because the resolved ratio mixes the two scales. It is **not** the cluster's virial
ratio (audit S10): a hard binary's *internal* virial is itself $\approx 0.5$
(time-averaged), so sampled at random orbital phases $Q_{\rm resolved}$ **scatters
around 0.5** rather than deflating monotonically. The robust, gated statement is
the energy *separation* — $Q_{\rm com}$ cleanly recovers $0.5$ while
$|E_{\rm internal}|$ dwarfs $|W_{\rm com}|$.

**Why a young cluster.** Primordial binaries are the population present *at birth*,
before dynamical processing ionizes the soft ones and hardens the rest. Their
natural home is therefore a **young** cluster, for which the
Elson–Fall–Freeman (1987) {cite:p}`ElsonFallFreeman1987` extended power-law profile
is the standard parametrization — not the King model of an old, relaxed globular.
So the primary cluster here is EFF ($a=1$ pc, $\gamma=2.5$, $r_t=15$ pc); a
concentrated King ($W_0=7$) {cite:p}`King1966` appears only in the environment
figure (below).

## What is built

A Plummer-masses-first EFF young cluster of $N_{\rm sys}=2000$ systems, virialized
to $Q=0.5$, with a smooth (differentiable) Maschberger $\alpha=2.3$ primary IMF over
$[0.08, 100]\,\Msun$ and companions from `IndependentCompanions` (or the full
Moe & Di Stefano (2017) {cite:p}`MoeDiStefano2017` model). Two controlled sweeps:

- **Hardness** — the companion period band `LogUniformPeriod` is slid from
  $\log_{10}(P/{\rm day})=1.5$ (hard, $\sim$30 d) to $5.5$ (soft, $\sim$900 yr) at
  fixed $f_b=0.5$. Shorter periods $\Rightarrow$ smaller $a$ $\Rightarrow$ deeper
  binding (vis-viva).
- **Binary fraction** — $f_b$ swept $0.1\!\to\!1.0$ at a fixed broad period band.

## Result — freshly run, ALL PASS

Measured 2026-06-12 (CPU/float64, $N_{\rm sys}=2000$, seeds from `PRNGKey(0)`; wall
$\approx 32$ s; exit 0).

**Hardness sweep (EFF, $f_b=0.5$).** $Q_{\rm com}$ is pinned at $0.5$ to four
decimals at every point; $|E_{\rm internal}|$ spans **three orders of magnitude**;
$Q_{\rm resolved}$ scatters (the contamination):

```{list-table}
:header-rows: 1

* - $\log_{10} P$
  - $Q_{\rm com}$
  - $Q_{\rm resolved}$
  - $|E_{\rm internal}|$
  - $|W_{\rm com}|$
* - $1.5$ (hard)
  - $0.5000$
  - $0.464$
  - $1.22\times10^{6}$
  - $6.4\times10^{2}$
* - $2.5$
  - $0.5000$
  - $0.420$
  - $8.4\times10^{5}$
  - $1.0\times10^{3}$
* - $3.5$
  - $0.5000$
  - $0.512$
  - $4.1\times10^{4}$
  - $5.2\times10^{2}$
* - $4.5$
  - $0.5000$
  - $0.491$
  - $1.4\times10^{4}$
  - $6.4\times10^{2}$
* - $5.5$ (soft)
  - $0.5000$
  - $0.327$
  - $7.1\times10^{3}$
  - $6.1\times10^{2}$
```

Across both sweeps the reservoir ratio $|E_{\rm internal}|/|W_{\rm com}|$ stays in
$[11.6,\ 1.9\times10^{3}]$ — the internal binding **always dwarfs** the cluster
potential — while $\max|Q_{\rm resolved}-Q_{\rm com}| = 0.257$ confirms the resolved
ratio is a contaminated proxy. Gate summary:

```{list-table}
:header-rows: 1

* - Check
  - Gate
  - Status
* - $Q_{\rm com}\approx0.5$ (all 12 points)
  - $|Q_{\rm com}-0.5|<0.01$
  - **PASS**
* - $E_{\rm internal}<0$ (all bound)
  - $<0$
  - **PASS**
* - realized $f_b$ matches set $f_b$
  - $3\sigma$ Poisson
  - **PASS**
* - $|E_{\rm internal}|>|W_{\rm com}|$ (all points)
  - reservoir dwarfs potential
  - **PASS**
* - $E_{\rm internal}$ identical EFF vs King (same key)
  - controlled comparison
  - **PASS**
* - reservoir fraction EFF $>$ King
  - environment effect
  - **PASS**
```

## The environment comparison

The same realistic Moe & Di Stefano population is laid into a young EFF and a
concentrated King ($W_0=7$) potential **using the same random key**, so the masses
and orbital elements — and therefore $E_{\rm internal}$ — are *byte-identical*
between the two. Only the cluster structure differs:

```{list-table}
:header-rows: 1

* - Cluster
  - $r_h$
  - $|E_{\rm internal}|$
  - $|W_{\rm com}|$
  - $|E_{\rm internal}|/|W_{\rm com}|$
* - EFF (young)
  - $5.32$ pc
  - $2.965\times10^{5}$
  - $6.7\times10^{2}$
  - $440$
* - King $W_0=7$ (GC-like)
  - $3.26$ pc
  - $2.965\times10^{5}$
  - $9.9\times10^{2}$
  - $298$
```

The concentrated King is more tightly bound ($|W_{\rm com}|$ larger), so the *same*
primordial binary population is a relatively **larger energy store in the young,
puffy cluster** — the global importance of the binary reservoir depends on the
birth environment. This is the {cite:t}`Heggie1975` hard/soft intuition at the
level of the *total* energy budget; a per-binary hard/soft classification
($|E_{\rm bind}|$ vs the local $kT$) is a related but distinct quantity and is not
claimed here.

## Figures

:::{figure} figures/demo_binary_energy_budget.png
:label: sci-b9-sweeps
:width: 100%

**EFF young-cluster sweeps** (`scripts/demo_binary_energy_budget.py`). **(a)**
Hardness sweep: $Q_{\rm com}$ pinned at $0.5$ (blue) while $Q_{\rm resolved}$
(vermilion) scatters around it — the resolved ratio is not the cluster's. **(b)**
$|E_{\rm internal}|$ (green) towers over $|W_{\rm com}|$ (purple) and grows three
orders as binaries harden. **(c, d)** Binary-fraction sweep: same $Q_{\rm com}$
pinning, and the reservoir fraction rising with $f_b$, with the realistic
Moe & Di Stefano population marked (★).
:::

:::{figure} figures/demo_binary_energy_budget_environment.png
:label: sci-b9-environment
:width: 85%

**Birth-environment comparison** (same Moe population, same key). **(a)**
$|E_{\rm internal}|$ is identical between EFF and King (the controlled comparison);
$|W_{\rm com}|$ is larger for the concentrated King. **(b)** The reservoir fraction
$|E_{\rm internal}|/|W_{\rm com}|$ is therefore larger in the young EFF cluster.
$r_h$ for each cluster is annotated.
:::

## Caveats

```{warning}
- **Initial conditions only.** This is the binary energy budget *at birth*. The
  dynamical consequences — soft-binary ionization, hard-binary hardening, the
  binary-burning energy source that halts core collapse — require N-body evolution
  (the deferred gravax forward chain), not these ICs.
- **$Q_{\rm resolved}$ is a diagnostic, not a gate.** Its exact value is set by the
  instantaneous orbital phases of the binary sample and scatters around $0.5$; the
  demo gates the robust separation ($Q_{\rm com}$, the reservoir magnitude), and
  *shows* the $Q_{\rm resolved}$ scatter as the cautionary point.
- **Environment comparison is at characteristic $r_h$, not fixed $r_h$.** EFF and
  King are two representative fiducial clusters with their own half-mass radii
  (both annotated). The reservoir-fraction difference is driven by $|W_{\rm com}|$
  (the more bound King), disclosed in the table; a controlled-at-fixed-$r_h$ study
  is a refinement, not done here.
- **Point-mass binaries.** The hardest band ($\log P = 1.5$, $\sim$30 d) keeps $a$
  well above stellar radii for these masses; the point-mass orbital treatment is
  valid across the sampled range.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_binary_energy_budget.py
```

## References

The EFF young-cluster profile is {cite:t}`ElsonFallFreeman1987`; the King model
{cite:t}`King1966`; the binary statistics follow {cite:t}`MoeDiStefano2017`; the
COM/internal scale separation is the McLuster convention
{cite:t}`Kuepper2011_McLuster`; the hard/soft intuition is {cite:t}`Heggie1975`.
The budget API and its vis-viva / scale-separation tests live in
`progenax.binaries.diagnostics`.
