---
title: Two-component populations — worked examples
description: "Worked two-component clusters through both MultiComponentCluster engines: an Engine-A cold/hot lowered-isothermal pair and the Engine-B Plummer-halo + EFF-core decomposition, including an honestly unrealizable mix and the verified truncation-edge Q_j plateau."
---

# Two-component populations — worked examples

The simplest non-trivial multi-component IC has *two* populations in
one shared potential. This page builds one through **each** engine of
[`MultiComponentCluster`](index.md): a cold/hot pair where the DFs
define the model (Engine A), and a halo+core pair where prescribed
densities define it (Engine B). Both are *true joint equilibria* — no
external virial rescale anywhere — and both come with the measured
numbers that prove it.

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers wanting a worked two-population cluster through each engine, including the honest physics; no prior multi-component-equilibrium literature assumed.
**Prerequisites:** [the Eddington engine](eddington-engine.md) (Engine B theory) and the [multi-component overview](index.md) (the two-engine framing).
**You'll get:** an Engine-A cold/hot pair and the Engine-B Plummer-halo + EFF-core headline — with the unrealizable-mix gate and the verified truncation-edge $Q_j$ plateau spelled out.
:::

## Engine A: a cold population inside a hot one

Engine A components are lowered-isothermal DFs
([theory](../spatial-profiles/lowered-model-family.md)); the one free
per-component scale is the velocity-scale ratio $w_j = s_j/s$. A
*colder* component ($w_j < 1$) sees the shared potential at the deeper
rescaled depth $W_j = \psi/w_j^2$ and concentrates — concentration
here is an *equilibrium outcome*, not a sampling choice:

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import MultiComponentCluster

model = MultiComponentCluster.from_components(
    alpha_j=jnp.array([0.5, 0.5]),   # central density fractions
    w_j=jnp.array([0.7, 1.0]),       # cold (0.7) + hot (1.0)
    m_j=jnp.array([1.0, 1.0]),       # equal stellar masses: pure w_j physics
    W0=7.0, g=1.0, r_c=1.0,          # King-like (g = 1) truncation
)
ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=20_000, G=STELLAR.G)
# ic.component_id: 0 = cold, 1 = hot
```

Measured (from `scripts/validate_cluster_ic.py`, with $r_c = 1$ pc):
the cold ($w = 0.7$) population's median radius is **1.50 pc**, inside
the hot population's **8.36 pc** — spatial segregation from velocity
scale alone, with equal stellar masses. The exact-quadrature
equilibrium oracle reads $Q_j = [0.5,\, 0.5]$ for both components.

For *mass*-driven versions of the same physics, use
`from_mass_segregation` ($w_j = \mu_j^{-\delta}$) or `from_imf`; for a
GC 1G/2G setup, give the concentrated 2G the smaller $w_j$.

## Engine B: a Plummer halo plus an EFF core

Engine B starts from prescribed *densities*
([theory](eddington-engine.md)): the headline decomposition is an
extended Plummer halo carrying 60% of the mass plus a compact
truncated EFF core ({cite:t}`ElsonFallFreeman1987`) carrying 40%:

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import MultiComponentCluster, PlummerProfile, EFFProfile

model = MultiComponentCluster.from_density_profiles(
    profiles=[PlummerProfile(r_h=2.0),                  # halo
              EFFProfile(a=0.8, gamma=5.0, r_t=9.0)],   # core
    mass_fractions=jnp.array([0.6, 0.4]),               # M_j / M_total
    m_j=jnp.array([0.5, 1.0]),                          # stellar-mass labels
)
ic = model.sample_cluster(jax.random.PRNGKey(0), n_stars=30_000, G=STELLAR.G)
```

The shared potential is one direct quadrature pass over the summed
prescribed density; each component's DF is its Eddington inversion in
that shared potential. Measured at close-out: theoretical
$Q_j = [0.50038,\, 0.50012]$ (gate $0.5 \pm 3\times 10^{-3}$), sampled
global $Q = 0.4976$ at $N = 30{,}000$, **unscaled**.

### Honest physics 1: not every decomposition exists

Eddington inversion is a two-way street: a prescribed density in a
given potential corresponds to a *unique* candidate $f(E)$, and if that
$f$ is negative anywhere, **the component cannot exist as an
equilibrium in that potential** — full stop. The originally drafted
version of this very example used a *shallower* EFF core,
$a_{\rm EFF} = 0.4$, and it is **genuinely unrealizable**: its
Eddington DF has $\min f / \max|f| = -0.20$, resolution-independent
(verified against the closed-form two-Plummer oracle, since
$\gamma = 5$ EFF *is* Plummer). A close-out sweep located the
realizability boundary between $a = 0.65$ (refused,
$-6.4\times 10^{-3}$) and $a = 0.68$ (realizable,
$+1.9\times 10^{-2}$); the headline uses $a = 0.8$
($f_{\min,j} = +1.6\times 10^{-2}$ halo, $+1.2\times 10^{-4}$ core).

progenax treats this as physics, not failure: the constructor raises a
`ValueError` **naming the component** and the remedy (steepen it,
raise its mass fraction, or raise its $r_{a,j}$), and always stores
the `f_min_j` diagnostic. See the
[realizability gate](eddington-engine.md) for the mechanism.

### Honest physics 2: the truncation-edge $Q_j$ plateau

The hard-truncated halo's *sampled* per-component virial ratio
plateaus slightly **below** 0.5 — and this is verified physics, not a
bias to be rescaled away. A sharply truncated prescribed density has
$\rho(r_t) > 0$, a constant edge offset that *no* ergodic $f(E)$ can
carry (the Eddington pair represents $\rho(\Psi) - \rho(0)$). Engine B
samples a *hybrid* — positions from the prescribed $\rho_j$, speeds
from $f_j$ — so the exact-quadrature hybrid expectation predicts the
offset: **predicted $Q_{\rm halo} = 0.4953$, sampled
$0.4947 \pm 0.0014$** (18 seeds × 16k stars, a $0.4\sigma$ agreement).
The validation gate is against the *prediction*, never a tuned offset
— and emphatically **not** a rescale to 0.5, which would destroy the
core's equilibrium to cosmetically fix the halo's edge.

## Choosing between the engines for two-component work

```{list-table}
:header-rows: 1

* - You have…
  - Use
  - Because
* - A dynamical model in mind (relaxed, tidally truncated populations;
    mass segregation; fit $g, W_0, w_j, \delta$ to data)
  - Engine A (`from_components` et al.)
  - The lowered-isothermal family *is* the model; equilibrium and
    equipartition are built in.
* - Observed/prescribed density shapes (surface-brightness
    decomposition, halo+core, literature profiles)
  - Engine B (`from_density_profiles`)
  - Densities go in verbatim; Eddington tells you whether the
    decomposition is dynamically realizable.
```

The two engines agree where they overlap: a single King component
built both ways matches to a radial KS distance of $2\times 10^{-4}$
and $|\sigma_B/\sigma_A - 1| \le 3\times 10^{-4}$
(the A-vs-B trust anchor; see [](eddington-engine.md)).

## Domain of validity

1. **True joint equilibrium, by construction.** Unlike the retired
   layered superposition (each component sampled from its own
   isolated-cluster DF), both engines solve/integrate **one** shared
   potential and prove $Q_j = 0.5$ per component with no rescale.
2. **Truncation edges are approximate** (Engine B): hard-truncated
   prescribed profiles are only approximately stationary at the edge;
   the deviation is *predicted and gated*, not hidden (see above).
3. **Deliberate non-equilibrium ICs** (sub-/super-virial, mismatched
   DFs) remain a separate, explicit workflow — see
   [](../velocity-dfs/index.md) and
   [](../../20-architecture/q-virial-convention.md).

## Implementation, validation & references

- **In code:** both engines are `MultiComponentCluster` in
  `src/progenax/cluster/multicomponent.py`
  (`from_components` for Engine A, `from_density_profiles` for Engine B);
  the worked numbers come from `scripts/validate_cluster_ic.py` and
  `scripts/validate_multicomponent_eddington.py`. See the
  [cluster API](../../30-api/cluster.md).
- **Validated in:** [two-component](../../50-validation/two-component.md),
  with Engine-specific anchors in
  [multimass equilibrium](../../50-validation/multimass-equilibrium.md)
  and [Engine B (Eddington)](../../50-validation/engine-b-eddington.md).
- **Primary sources:** self-consistent multi-mass lowered models
  {cite:t}`Gieles2015`; Eddington inversion with Osipkov–Merritt
  anisotropy {cite:t}`Merritt1985`; EFF profile
  {cite:t}`ElsonFallFreeman1987`; layered multi-population ICs in N-body
  practice {cite:p}`Kuepper2011`. Full notes in the
  [bibliography](../../99-bibliography/index.md).
