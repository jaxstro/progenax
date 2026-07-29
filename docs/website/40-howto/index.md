---
title: How-to recipes
description: Task-oriented, runnable recipes for common progenax workflows — virial clusters, differentiable r_h fits, protocol composition, and binary populations.
---

(howto-index)=
# How-to recipes

Task-oriented snippets for common workflows. Each page answers **one
question** with a runnable recipe, an inputs table, and verified output —
every code block runs as-is in a Python session with progenax installed
(see [](../00-getting-started/installation.md)).

```{list-table} Recipes
:header-rows: 1

* - Recipe
  - Question it answers
* - [](set-up-virial-cluster.md)
  - How do I build a cluster in virial equilibrium ($Q = T/|V| = 0.5$)?
* - [](gradient-based-r_h-fit.md)
  - How do I recover the half-mass radius $\rh$ with `jax.grad` through the IC builder?
* - [](mix-plummer-positions-king-velocities.md)
  - How do I compose a Plummer spatial profile with a King velocity DF?
* - [](add-binary-population.md)
  - How do I add a primordial binary population with `build_binary_cluster`?
* - [](interface-with-gravax.md)
  - How do I retain primordial-system provenance and hand the physical IC to Gravax?
```
