---
title: Two-component populations
description: "The implemented two-component cluster IC API: split pre-sampled masses across two profile/velocity-DF pairs."
---

# Two-component populations

The simplest non-trivial multi-component IC has *two* components,
typically representing a young dense core embedded in an older diffuse
envelope. progenax's implemented `TwoComponentConfig` gives each
component its own spatial profile and velocity distribution function.

Masses are sampled outside the two-component generator, then split into
populations either randomly or with a caller-supplied mask. The current
API does **not** provide per-component IMF sampling, inter-component
offsets, or a global virial-rescale option.

## The TwoComponentConfig

```python
from progenax.populations import TwoComponentConfig
from progenax.profiles import PlummerProfile
from progenax.kinematics import PlummerVelocityDF

cfg = TwoComponentConfig(
    f_A=0.25,                                # 25% in population A
    profile_A=PlummerProfile(r_h=2.0),       # Extended envelope
    profile_B=PlummerProfile(r_h=0.4),       # Compact core
    velocity_df_A=PlummerVelocityDF(r_h=2.0),
    velocity_df_B=PlummerVelocityDF(r_h=0.4),
)
```

`f_A` controls the random assignment probability for population A when
no mask is supplied. Population A is conventionally the extended
component (`pop_id == 0`) and population B the compact component
(`pop_id == 1`), but the code only requires profile and velocity-DF
objects satisfying the progenax protocols.

## Generation

```python
import jax
import jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax.imf import Maschberger
from progenax.populations import generate_two_component_cluster

key = jax.random.PRNGKey(42)
key_mass, key_ic = jax.random.split(key)
N = 2500

masses = Maschberger(alpha=2.3).sample(key_mass, N)
positions, velocities, pop_id = generate_two_component_cluster(
    masses, cfg, key=key_ic, G=STELLAR.G,
)
```

The function returns `(positions, velocities, pop_id)`. It does not
return masses because it never modifies the input mass array.

Internally, `generate_two_component_cluster`:

1. Assigns each input mass to population A or B. If `pop_mask` is
   supplied, `True` means population A and `False` means population B.
2. Samples positions from both profiles.
3. Samples velocities from both velocity DFs.
4. Selects the profile/DF result for each star according to the
   population mask.
5. Returns population labels with `0 = A` and `1 = B`.

## Composition decisions

Building a two-component IC requires three explicit choices:

```{list-table}
:header-rows: 1

* - Decision
  - Current default
  - When to override
* - **Per-component DF**
  - Match each profile, e.g. Plummer-on-Plummer
  - Studying deliberately mismatched kinematics
* - **Population assignment**
  - Random Bernoulli assignment with probability `f_A`
  - Use `pop_mask` for mass-sorted or deterministic assignment
* - **Post-processing**
  - Caller-controlled
  - Apply `to_com_frame` or `virial_scale` after generation if needed
```

For example, a mass-sorted core can be built by supplying a custom mask:

```python
massive_core = masses > jnp.quantile(masses, 0.75)
positions, velocities, pop_id = generate_two_component_cluster(
    masses,
    cfg,
    key=key_ic,
    G=STELLAR.G,
    pop_mask=~massive_core,  # True => A/envelope; False => B/core
)
```

Tidal truncation, fractal substructure, and global virial rescaling can
still be composed around this output, but they are not options on
`TwoComponentConfig` itself.

## When components do not equilibrate jointly

For two components with very different scale radii, the *joint*
gravitational potential is no longer well-described by either
component's individual DF. Component A's DF was designed for one
profile; the actual potential includes component B's mass too.

For marginal mass ratios ($M_b/M_a \sim 1$), self-consistent
multi-component DFs are needed. The {cite:t}`Gieles2015` LIMEPY family
provides this for King-style profiles; progenax does not currently
implement multi-component LIMEPY.

## Validation

The validation page at [](../../50-validation/two-component.md)
currently points at unit-test coverage for the implemented API. It does
not yet correspond to a dedicated
`tests/validation/test_two_component.py` file.

The real unit tests exercise population assignment, returned shapes,
the `pop_id` convention, and custom `pop_mask` behavior.

## Domain of validity

1. **Two components only.** The `TwoComponentConfig` API is specialised
   for the two-component case.
2. **No self-consistent joint DF.** Use this for controlled mixtures,
   merger studies, or initial conditions for violent-relaxation
   experiments, not as a proof of exact joint equilibrium.
3. **No built-in joint finalisation.** The current function does not
   shift to a centre-of-mass frame or rescale the joint virial ratio.

## References

Multi-component cluster modelling is standard in N-body work
{cite:p}`Aarseth1974,Kuepper2011`. progenax's two-component
implementation follows the layered single-component DF approach rather
than the self-consistent multi-mass DF approach; {cite:t}`Gieles2015`
LIMEPY is the right tool when self-consistency matters.
