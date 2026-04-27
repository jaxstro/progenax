---
title: IMF sampling
description: Hands-on tutorial — sample masses from Salpeter, Kroupa, Chabrier, and Maschberger IMFs and compare them.
---

# IMF sampling

```{admonition} Executable notebook coming later
:class: tip

The hands-on **executable** version of this tutorial — `imf-sampling.ipynb`
with pre-executed JAX cells, ready to launch in Binder or Colab — is
a Phase E deliverable still to come. Until then, the prose walkthrough
below is sufficient to follow along: every code block runs as-is when
copied into a Python session with progenax installed (see
[](installation.md)).
```

progenax provides four canonical IMF parameterisations, each
appropriate to a different use case. This page samples from all
four, plots them, and explains when to use each.

## Setup

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from progenax.imf import PowerLawIMF, Maschberger, ChabrierIMF

key = jax.random.PRNGKey(0)
key_salp, key_kroupa, key_chab, key_masch = jax.random.split(key, 4)
N = 100_000   # Large enough for smooth histograms
m_grid = jnp.logspace(-1.5, 2.0, 100)   # 0.03 - 100 M_sun
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
    ("Chabrier", masses_chab),
    ("Maschberger", masses_masch),
]:
    log_masses = jnp.log10(masses)
    ax.hist(log_masses, bins=50, density=True, alpha=0.5, label=name)
ax.set_xlabel("log₁₀(M / M☉)"); ax.set_ylabel("dN/dlogM")
ax.set_yscale("log"); ax.legend()
plt.savefig("imf_comparison.png", dpi=120)
```

Expected: all four IMFs agree closely above $\sim 1\,\Msun$ (the
Salpeter regime) and differ in the low-mass turnover. Salpeter has
no turnover; Kroupa breaks at 0.5 $\Msun$; Chabrier and Maschberger
are smooth.

## Choosing the right IMF

```{list-table}
:header-rows: 1
:widths: 30 70

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

For binary-aware inference, the IMF combines with the {cite:t}`Moe2017`
multiplicity statistics — see [](../10-theory/imfs/binary.md).

For environment-dependent IMF (top-heavy in dense / metal-poor
populations), see [](../10-theory/imfs/environment.md).

## Next step

[](glossary.md) defines every term you've seen in the tutorial path.
After that, drop into [](../10-theory/index.md) for the full theory
section, [](../20-architecture/index.md) for design rationale, or
[](../30-api/index.md) for the full API reference.
