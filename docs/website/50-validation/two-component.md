---
title: Two-component populations validation
description: Current validation status for TwoComponentConfig and generate_two_component_cluster.
---
# Two-component populations validation

```{important}
Status: **unit-backed, not a dedicated validation suite yet**.

There is no `tests/validation/test_two_component.py` file in this
checkout. The implemented two-component API is covered by
`tests/unit/test_populations.py`.
```

## What is currently verified

```{list-table}
:header-rows: 1
:widths: 35 25 40

* - Property
  - Status
  - Anchor
* - `TwoComponentConfig` accepts `f_A`, two profiles, and two velocity DFs
  - Unit-tested
  - `tests/unit/test_populations.py`
* - `generate_two_component_cluster` returns positions, velocities, and `pop_id`
  - Unit-tested
  - `tests/unit/test_populations.py`
* - Random population assignment follows the configured fraction statistically
  - Unit-tested
  - `tests/unit/test_populations.py`
* - Caller-supplied `pop_mask` controls assignment
  - Unit-tested
  - `tests/unit/test_populations.py`
* - Per-component IMF sampling, Q-target-global finalisation, COM finalisation, modifiers
  - Not implemented in this API
  - Planned/design-only unless added later
```

## Spot values

Earlier spot tables listing component virial ratios and joint
Q-target-global behavior were aspirational. The current function does
not implement that option, does not rescale the joint state, and
does not shift to the centre-of-mass frame.

## How to run current checks

```bash
pytest tests/unit/test_populations.py -v
```

## References

Theory at [](../10-theory/populations/two-component.md). The
{cite:t}`Gieles2015` LIMEPY family is the relevant external reference
when exact self-consistent multi-component DFs are required.
