# FDF spherical-cluster ICs — design (envelope + turbulent velocities + substructure diagnostic)

**Date:** 2026-06-07 · **Branch:** `gravoturb-fdf-sbc-validation` (experimental subsystem) · nothing pushed.
**Context:** strategic pivot (see brain inbox 2026-06-07 + STATUS.md) — the 2D-projected-β *inference*
is banked as a methods result; gravoturb_fdf is repositioned as a **forward generative tool**:
realistic, substructured, *spherical* young-cluster ICs for N-body / binary studies. This design adds
a spherical shape, coherent turbulent velocities, and a clean substructure parameterization to the
existing positions-only turbulent-cube pipeline.

## Current state

`build_fdf_field` → `mass_conserving_copula_field` → `cloud_to_stars`/`sample_positions` produces star
**positions** in a periodic cubic box following a BM19 turbulent density field (⟨e^s⟩=1). No spherical
envelope, **no velocities**. Validated by AC1–AC10 (BM19 marginal, mass conservation, f_dense_realized
cornerstone AC6, Q(f_sub), gradients). Core `progenax` provides `PlummerProfile`/`EFFProfile` (each with
`.density(r)`), `compute_q_parameter`/`q_approx`, and `virial_scale(pos, vel, m, Q_target, G)` (cites
Goodwin & Whitworth 2004).

## Decisions

1. **Spherical envelope — separable log-space.** `s_total(x) = s_turb(x) + ln ρ_env(r)`, where
   `s_turb` is the existing BM19 turbulent fluctuation (⟨e^s⟩=1, the **substructure**: β, ℳ, α) and
   `ρ_env(r)` is a radial profile (the **shape**: r_h, concentration). `ρ_total = ρ_env(r)·e^{s_turb}`
   — centrally concentrated **and** clumpy. The split keeps all existing BM19 validation (s_t,
   f_dense, AC6) on `s_turb` (dense clumps = *local* overdensities `s_turb > s_t`). Envelope = any
   progenax `SpatialProfile` via `.density(r)`; **default `PlummerProfile`** (EFF for young-cluster
   realism). Envelope centered in the box, taper well inside (r_h ≲ box/4) so periodic edges are empty.

2. **Velocities — turbulent + coherent, scaled to a chosen Q.** A 3-component Gaussian velocity field
   with a turbulent spectrum P_v(k) ∝ k^{−β_v} (reuse `gaussian_random_field` per component),
   trilinear-interpolated to star positions → spatially coherent stellar velocities (clumps move
   together; Goodwin & Whitworth 2004). Then `virial_scale(pos_total, v, m, Q_target, G)` sets the
   amplitude to a **chosen Q** (free IC parameter; **default 0.5**, sub-virial Q<0.5 supported — the
   dynamically interesting young-cluster regime). `virial_scale` computes |V| from the actual
   positions, so the **envelope is automatically accounted for** (deeper potential → larger v at fixed
   Q). **β_v default to be grounded against the Goodwin & Whitworth 2004 PDF before coding** (no
   assumption-from-memory).

3. **Substructure diagnostic — CW04 Q + (m̄, s̄) plane.** Reuse `compute_q_parameter`; report the
   **(m̄ clumpiness, s̄ concentration) components** separately so substructure is decoupled from the
   envelope. Calibrate m̄ ↔ β at fixed envelope (substructure tracks β) and confirm m̄ ≈ const as
   concentration varies at fixed β (decoupling). Q bridges to observed-cluster catalogs.

## Validation (AC-style, experimental)

(1) sampled ρ(r) matches the analytic envelope; (2) m̄(β) monotonic **and** concentration-decoupled;
(3) BM19/AC6 still hold on `s_turb`; (4) velocity field has the target spectrum, achieves target Q,
and shows near-neighbour velocity **coherence**; (5) `jax.grad` flows through the density construction.

## Plots (figure gallery)

3D + projected scatter (spherical & clumpy); ρ(r) sampled vs analytic; velocity-coherence map;
Q–(m̄,s̄) vs β; density PDF; power spectrum.

## Build order (TDD; released-core **814** invariant held; experimental-only)

1. `field/envelope.py`: `radius_grid`, `apply_spherical_envelope` (separable log-space) + tests.
2. `field/velocity.py`: `turbulent_velocity_field`, `sample_turbulent_velocities` (+ G&W β_v
   grounding) + tests; wire `virial_scale`.
3. diagnostic extension: (m̄, s̄) components, concentration-decoupled + tests.
4. validation script (AC-style) + figure gallery.

## Defaults

envelope `PlummerProfile`; Q_target 0.5 (sub-virial supported); β_v ← Goodwin & Whitworth 2004 (verify).

---

## NICE-TO-DO / REVISIT LATER — filamentary (non-Gaussian-phase) morphology

**Status: deferred (not on the critical path). Realism polish, not a requirement for the stated
N-body/binary purpose.** This section records *why* we wait and *how* to implement it when warranted,
so a future session can pick it up without re-deriving.

## What's missing, and the root cause

The current density field is a Gaussian random field (GRF) re-marginalized to the BM19 PDF by a
monotone copula. Write the GRF as

$$ \delta(\mathbf{x}) = \sum_{\mathbf{k}} |\delta_{\mathbf{k}}|\, e^{i\varphi_{\mathbf{k}}}\,
e^{i\mathbf{k}\cdot\mathbf{x}}, \qquad P(k) = \langle |\delta_{\mathbf{k}}|^2\rangle . $$

The power spectrum $P(k)$ fixes only the mode **amplitudes** $|\delta_{\mathbf{k}}|$. Coherent
structures — **filaments and sheets** — are **phase correlations** among the $\varphi_{\mathbf{k}}$.
A GRF has *random, independent* phases, and the copula $\rho = T(\delta)$ is **monotone**, so it
preserves the GRF's level-set morphology. Hence the field has the correct 1-pt PDF and 2-pt $P(k)$ but
**blob-like, not filamentary, morphology**. You cannot fix this by changing $P(k)$ — you must inject
phase structure / non-Gaussianity into the field itself.

## Why we wait (the scientific justification)

These ICs feed **N-body / binary dynamics**. Once stars are point masses, the evolution integrates the
**gravity of the mass distribution**, which depends on *where the mass is and how it clusters* — the
power spectrum, the clumpiness ($Q$, $\bar m$–$\bar s$), and the sub-clump masses/positions — **not**
on whether an overdensity is shaped like a filament or a blob. At **matched $P(k)$ + clumpiness +
virial ratio**, filamentary and blobby ICs produce nearly the same cluster-scale dynamics. The
*degree* of substructure (set by $\beta$ and $Q$) is first-order and **already controlled**;
morphology is largely cosmetic for the dynamics.

**Revisit only when one of these triggers fires:**
1. A **discriminating dynamical observable** is identified — i.e. segregation time, primordial-binary
   disruption rate, ejection rate, or $Q$-erasure timescale that differs *significantly* between
   filament and blob ICs at **matched** $P(k)$, $Q$, and virial $Q$. (Until shown, assume it doesn't.)
2. The science needs **direct morphology comparison** to observed cloud/cluster filament statistics.
3. The study concerns the **gas→star spatial mapping** (stars forming *along* filaments/hubs), where
   the initial stellar geometry — not just its clustering statistics — matters.

Until then, the higher-value path is finishing the forward pipeline (envelope + velocities + virial
wiring) and **demonstrating the forward science** (natal turbulence → dynamical outcomes).

## How to implement it (pedagogical overview + equations)

**Key enabling insight — the copula is morphology-agnostic.** Build a *filamentary base field*
$g_{\rm fil}(\mathbf{x})$, then apply the **same** copula to impose the exact BM19 marginal:

$$ s(\mathbf{x}) = F_{\rm BM19}^{-1}\!\big(\Phi(\hat g_{\rm fil}(\mathbf{x}))\big), \qquad
\hat g_{\rm fil} = (g_{\rm fil}-\langle g_{\rm fil}\rangle)/\sigma_{g_{\rm fil}} . $$

Because $T=F^{-1}\!\circ\Phi$ is monotone it **preserves the filamentary morphology** (level sets)
while fixing the 1-pt PDF to BM19. **Cost:** the displacement reshapes the spectrum, so $\beta$ is no
longer the clean input slope — it becomes a **measured** output, and AC1–AC6 must be re-validated.

### Option A — turbulent-shock displacement (preferred; most ISM-faithful)

Real ISM filaments are carved by **converging supersonic flows (compressive shocks)**. Lay tracers on
a uniform grid $\mathbf{q}$, generate a turbulent velocity field $\mathbf{v}(\mathbf{q})$ (we already
have `turbulent_velocity_field`, $P_v(k)\propto k^{-\beta_v}$), and displace (Zel'dovich-like,
single step):

$$ \mathbf{x}(\mathbf{q}) = \mathbf{q} + D\,\mathbf{v}(\mathbf{q}), $$

with $D$ a displacement amplitude (a "how far have the flows converged" / effective-time knob; tie to
$\mathcal{M}$). Mass piles up where the map is compressive — the Lagrangian-to-Eulerian Jacobian
shrinks:

$$ \rho(\mathbf{x}) = \frac{\rho_0}{\big|\det(\partial \mathbf{x}/\partial \mathbf{q})\big|}
= \frac{\rho_0}{\big|\det(\mathbf{I} + D\,\partial \mathbf{v}/\partial \mathbf{q})\big|}. $$

**Caustics** (sheets→filaments→knots, as 1, 2, then 3 eigenvalues of $D\,\partial\mathbf v/\partial\mathbf q$
reach $-1$) are exactly the filamentary skeleton. To avoid multi-stream blowup at shell-crossing, the
**adhesion model** adds Burgers viscosity,

$$ \partial_t \mathbf{v} + (\mathbf{v}\cdot\nabla)\mathbf{v} = \nu\,\nabla^2 \mathbf{v}, \qquad
\nu \to 0^+, $$

whose vanishing-viscosity limit yields a sharp, connected filament network. Deposit the displaced
tracers to a grid (CIC) → $g_{\rm fil}$ → copula → BM19 marginal.

*Keeps:* BM19 PDF, differentiability (smooth for finite $D$ before shell-crossing), physical filaments,
reuses the velocity field. *Loses:* clean $\beta$ control (measure it), AC re-validation needed.

### Option B — Zel'dovich/adhesion on a Gaussian potential (gravity cousin)

Identical machinery, gravity-sourced displacement:

$$ \mathbf{x}(\mathbf{q}) = \mathbf{q} - D\,\nabla\phi(\mathbf{q}), \qquad \nabla^2\phi = \delta_{\rm GRF}. $$

Physically appropriate once **self-gravity** dominates (the cosmic-web construction). For the *natal*,
pre-collapse turbulent cloud, **A is more faithful** (turbulent shocks precede strong self-gravity);
B is the simpler, well-established alternative. Mechanistically A and B are cousins (displace by a
field gradient), so the A-vs-B choice is second-order relative to "do we need filaments at all."

### Validation deltas when revisited

Re-run AC1–AC6 on the post-copula field (PDF should still pass; $f_{\rm dense}$ re-check); **measure**
$\beta$ from the realized $P(k)$ (no longer the input); add a morphology diagnostic (e.g. filamentarity
via the bispectrum or a Hessian/Minkowski-functional skeleton) and confirm filaments appear; confirm
gradients still flow through the displacement for finite $D$.
