---
title: IMF sampling
description: Hands-on tutorial — sample canonical IMFs, then compose the supported birth-environment mappings with a four-segment power-law sampler.
---

# IMF sampling

```{admonition} Run the Python blocks in order
:class: tip

A notebook version is planned. The Python blocks below form one executable
sequence: run Setup first, then the sampling and plotting blocks in order
with progenax and Matplotlib installed (see [](installation.md)).
```

progenax provides four canonical IMF parameterisations, each
appropriate to a different use case. This page samples from all
four, plots them, and explains when to use each.
The final section composes the supported birth-environment mappings with
the same power-law sampling interface.

## Setup

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
from progenax.imf import PowerLawIMF, Maschberger, ChabrierIMF

key = jax.random.PRNGKey(0)
key_salp, key_kroupa, key_chab, key_masch = jax.random.split(key, 4)
N = 100_000   # Draws per family; the rare high-mass tail remains noisy
```

## The four IMFs

### Salpeter (single power law)

```python
salpeter = PowerLawIMF(
    exponents=[2.35],
    breakpoints=[],
    m_min=0.1,
    m_max=100.0,
)
masses_salp = salpeter.sample(key_salp, N)
```

### Kroupa (multi-segment broken power law)

```python
kroupa = PowerLawIMF.kroupa()
masses_kroupa = kroupa.sample(key_kroupa, N)
```

### Chabrier (lognormal + power law)

This example keeps the Chabrier system-form parameters $m_c=0.22\,\Msun$
and $\sigma=0.57$ dex, where $m_c$ is the lognormal characteristic mass
and $\sigma$ is its width in $\log_{10}m$. Its draws represent system
masses, whereas the other examples use individual-star distributions.
`ChabrierIMF()` instead defaults to the single-star disk form
($m_c=0.08\,\Msun$, $\sigma=0.69$ dex). Match that distinction to the
intended catalogue; see [](../10-theory/imfs/classic.md).

```python
chabrier = ChabrierIMF(m_c=0.22, sigma=0.57, alpha=2.3)
masses_chab = chabrier.sample(key_chab, N)
```

### Maschberger (smooth, default)

```python
maschberger = Maschberger(alpha=2.3, beta=1.4, mu=0.2)
masses_masch = maschberger.sample(key_masch, N)
```

## Plotting

```python
fig, ax = plt.subplots(figsize=(8, 5))
for name, masses in [
    ("Salpeter", masses_salp),
    ("Kroupa", masses_kroupa),
    ("Chabrier (system form)", masses_chab),
    ("Maschberger", masses_masch),
]:
    log_masses = jnp.log10(masses)
    ax.hist(log_masses, bins=50, density=True, alpha=0.5, label=name)
ax.set_xlabel("log₁₀(M / M☉)")
ax.set_ylabel("Probability density per dex")
ax.set_yscale("log"); ax.legend()
plt.savefig("imf_comparison.png", dpi=120)
```

Expected: all four approach similar high-mass slopes, but normalized
heights differ with shape and mass bounds. Here Salpeter spans
$0.1$–$100\,\Msun$, Kroupa spans $0.01$–$100\,\Msun$, Maschberger
spans $0.01$–$300\,\Msun$, and Chabrier spans $0.08$–$100\,\Msun$.
Salpeter has no low-mass
turnover; Kroupa changes slope at $0.08$ and $0.5\,\Msun$; Chabrier's
lognormal body joins its power-law tail at $1\,\Msun$; Maschberger has
no interior join. The plot compares these configurations, not families
at a common support or a common single-star convention.

## Choosing the right IMF

```{list-table}
:header-rows: 1

* - Choose
  - When
* - **Maschberger**
  - **Default for new code.** Closed-form inverse-CDF, smooth,
    HMC-friendly. progenax production default
* - **Salpeter**
  - High-mass-only studies; comparison with classical work
* - **Kroupa**
  - Backwards compatibility with prior cluster modelling
* - **Chabrier**
  - Unresolved-population integrated colours
```

For binary-aware inference, the IMF combines with the {cite:t}`MoeDiStefano2017`
multiplicity statistics — see [](../10-theory/imfs/binary.md).

## Environment-dependent slopes

`env_to_imf_params` maps a cluster's birth conditions to four IMF slopes.
Here $M_{\rm ecl}$ is the embedded stellar mass in $\Msun$,
$[\mathrm{Fe/H}]$ is metallicity in dex relative to solar, and
$\varepsilon=M_{\rm ecl}/M_{\rm cl}$ is the star-formation efficiency;
$M_{\rm cl}$ is total birth-cloud mass. The high-mass slope $\alpha_3$
sets the number density $\xi(m)\propto m^{-\alpha_3}$ above $1\,\Msun$.
Positive `exponents` therefore give a declining density per unit mass.
See [](../10-theory/imfs/environment.md) for the adopted relations and
their empirical limitations.

The block below runs independently of Setup. It constructs each supported
mapping at the same example inputs, then draws from the Marks
Fundamental Plane. It keeps the default hard thresholds, fixed low-mass
slopes and input guards. All samplers retain the returned mass bounds and
breaks, including the $1\,\Msun$ break that separates $\alpha_2$ from
$\alpha_3$. The printed mean is the distribution's expected mass in
$\Msun$, rather than a mean estimated from the draw.

```python
import jax
jax.config.update("jax_enable_x64", True)

from progenax.imf import BirthEnvironment, PowerLawIMF, env_to_imf_params

env = BirthEnvironment.from_cluster_mass(M_ecl=1e6, FeH=-1.5)
models = (
    "kroupa",
    "jerabkova_generalized",
    "jerabkova_mecl",
    "jerabkova_rho",
    "marks_plane",
    "marks_mcl",
    "marks_mecl",
    "marks_rho",
    "marks_feh",
)
environment_imfs = {}
for model in models:
    params = env_to_imf_params(env, model=model)
    imf = PowerLawIMF(
        exponents=[params.alpha0, params.alpha1, params.alpha2, params.alpha3],
        breakpoints=[params.m_break0, params.m_break1, params.m_break2],
        m_min=params.m_min,
        m_max=params.m_max,
    )
    environment_imfs[model] = imf
    print(f"{model:23s} alpha3={float(params.alpha3):.6f} "
          f"mean={float(imf.mean_mass()):.6f} Msun")

key_env = jax.random.key(0)
masses_env = environment_imfs["marks_plane"].sample(key_env, 10000)
print(f"Marks-plane draw: {masses_env.shape[0]} masses in Msun")
```

At these inputs, `kroupa`, `marks_mcl` and `marks_mecl` give
$\alpha_3=2.3$; the other routes give smaller high-mass slopes.
`jerabkova_generalized` and `jerabkova_mecl` agree here because the
environment uses their shared reference efficiency $\varepsilon=0.33$.
The nine routes are alternative adopted prescriptions, not nine estimates
to average or an accuracy ranking.

```{list-table} Inputs used by the high-mass mapping
:header-rows: 1

* - `model`
  - Inputs that affect $\alpha_3$
* - `kroupa`
  - None; the canonical slopes are fixed
* - `jerabkova_generalized`
  - Stellar mass, metallicity and the supplied SFE
* - `jerabkova_mecl`
  - Stellar mass and metallicity; assumes SFE $0.33$
* - `jerabkova_rho`
  - Cloud density and metallicity
* - `marks_plane`
  - Cloud density and metallicity
* - `marks_mcl`
  - Cloud mass $M_{\rm ecl}/\varepsilon$
* - `marks_mecl`
  - Stellar mass
* - `marks_rho`
  - Cloud density
* - `marks_feh`
  - Metallicity
```

The three density routes compute cloud density from mass and SFE unless
`log_rho_cl` is supplied to `BirthEnvironment`. That override means
$\log_{10}[\rho_{\rm cl}/(10^6\,\Msun\,\mathrm{pc}^{-3})]$;
`log_rho_cl=1.0` therefore denotes $10^7\,\Msun\,\mathrm{pc}^{-3}$.
The mass-based and metallicity-only routes ignore that override.
Optional `include_lowmass_variation=True` makes $\alpha_1$ and $\alpha_2$
depend on metallicity for every route except `kroupa`; its extrapolation
limits are described in the theory chapter.

```{important} Mass input and catalogue size have different roles
$M_{\rm ecl}$ selects the mapping's birth environment. `sample(key, n)`
draws a fixed number of masses; it does not condition their sum to equal
$M_{\rm ecl}$ or impose a cluster-mass-dependent upper stellar mass.
This block illustrates construction and sampling on $[0.01,150]\,\Msun$;
it is not a recovered birth environment or a wider-domain validation.
```

Keep `model` and Boolean mapping options static when using JIT. Hard
thresholds and output clipping retain derivative boundaries; opting into
`smooth_alpha3=True` changes the mapping near a threshold. The
[API](../30-api/imf.md#api-imf-env_to_imf_params) owns these options, and
[environment validation](../50-validation/environment-imf.md) records
the named published anchors and bounded derivative checks.

## Next step

[](glossary.md) defines every term you've seen in the tutorial path.
After that, drop into [](../10-theory/index.md) for the full theory
section, [](../20-architecture/index.md) for design rationale, or
[](../30-api/index.md) for the full API reference.
