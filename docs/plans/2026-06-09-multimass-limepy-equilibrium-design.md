# Multi-mass LIMEPY coupled equilibrium solver (Phase 2) — design

**Date:** 2026-06-09
**Status:** Approved (Anna, brainstorming session). Implementation pending (TDD).
**Branch:** `feat/multimass-limepy-equilibrium`
**Precedes:** Phase 2b (per-component anisotropy η), Phase 3 (blend-vs-equilibrium comparison).

## Goal

A first-principles, differentiable, multi-mass lowered-isothermal equilibrium
(Gieles & Zocchi 2015, §2.2) — the physically-realistic mass-segregation generator
that **complements** the `lambda_seg` blend. The headline deliverable: a cluster IC
in which mass segregation is a *true equilibrium*, proven by per-mass-group virial
ratios $Q_j \approx 0.5$ (unscaled) — the property the `lambda_seg` chord cannot have
(quantified in Phase 0).

Builds directly on the validated single-mass LIMEPY core (`profiles/limepy.py`,
`kinematics/limepy_df.py`). **Isotropic now**; η is Phase 2b (API kept η-ready).

## Physics (verified vs Gieles & Zocchi 2015, Eqs. 24–29)

One shared self-consistent potential $\hat\phi(\xi)$; each mass component $j$ has its
own velocity scale via the equipartition parameter $\delta$:

- $\bar m = \sum_j m_j\alpha_j$ — central density-weighted mean mass (Eq. 26),
  $\alpha_j$ = central density fraction, $\sum_j\alpha_j=1$.
- $\mu_j = m_j/\bar m$, $\quad s_j = s\,\mu_j^{-\delta}$ (Eq. 24).
- Component $j$ rides the shared potential on its own scale, so its dimensionless
  potential is $\mu_j^{2\delta}\hat\phi$ (since $s^2/s_j^2 = \mu_j^{2\delta}$), and
  (Eq. 29):
  $$\hat\rho_j(\xi) = \frac{E_\gamma(g+\tfrac32,\ \mu_j^{2\delta}\psi(\xi))}
                          {E_\gamma(g+\tfrac32,\ \mu_j^{2\delta}W_0)}
   = \frac{\texttt{limepy\_density\_hat}(\mu_j^{2\delta}\psi, g)}
          {\texttt{limepy\_density\_hat}(\mu_j^{2\delta}W_0, g)}.$$
- Coupled Poisson (Eq. 27), same $-9$ King-radius nondimensionalization:
  $$\frac{1}{\xi^2}\frac{d}{d\xi}\Big(\xi^2\frac{d\psi}{d\xi}\Big)
    = -9\sum_j \alpha_j\,\hat\rho_j(\xi).$$

Heavier component ($\mu_j>1$) → smaller $s_j$ → deeper effective well → central
concentration **as a genuine equilibrium** (mass segregation). $\delta=1/2$ is the
standard ("approximate at best" but reproduces observed segregation; full
equipartition $\delta=1$ is Spitzer-unstable). $\delta=0$ recovers the single-mass
model exactly.

## Architecture — three independently-tested layers (option 3)

### Layer A — physics core (`src/progenax/profiles/limepy_multimass.py`)
`solve_multimass_limepy(alpha_j, m_j, W0, g, delta, xi_max, n_points)`:
one coupled Poisson solve given $\alpha_j$. Computes $\bar m$, $\mu_j$, sums the
per-component sources (a `jax.vmap` over $j$ of `limepy_density_hat` at the rescaled
potentials), solves with `diffrax.Tsit5`. Returns $(\xi, \psi(\xi), \{\hat\rho_j\})$.
Pure, differentiable, **no iteration** — validated directly with hand-picked $\alpha_j$.
$\delta=0$ collapses every $\hat\rho_j$ to the single-mass $\hat\rho$, so the sum is
$(\sum\alpha_j)\hat\rho = \hat\rho$ — the cleanest single-mass oracle.

### Layer B — constraint solver (same module)
`find_alpha_for_masses(m_j, M_j, W0, g, delta, n_iter=30)`: the $\alpha_j$ that
reproduce target masses $M_j$. Realized mass fraction $f_j' =
\alpha_j\nu_j/\sum_k\alpha_k\nu_k$, $\nu_j=\int\hat\rho_j\xi^2 d\xi$; target
$f_j=M_j/\sum_k M_k$. Stabilized update (Gieles & Zocchi §4.1, **not** Gunn & Griffin's
linear form, which diverges for wide mass functions):
$$\alpha_j \leftarrow \alpha_j\sqrt{f_j/f_j'},\quad\text{renormalize }\sum_j\alpha_j=1.$$
A **fixed-`n_iter`** `jax.lax.scan` (never `while_loop`), each step = one Layer-A solve
+ the $\nu_j$ trapezoid + the update. Differentiable in $(M_j,\delta,g,W_0)$. Returns
$\alpha_j$ **and** the residual $\max_j|f_j'-f_j|$ (reported, never branched on).

### Layer C — user-facing model (`MultiMassLIMEPY`, Equinox)
Holds `W0, g, delta, m_j, alpha_j, N_j`, shared `xi_grid, psi_grid`, `r_c`, and
per-component position CDFs (built from $\hat\rho_j$ on the shared potential, reusing
the `LIMEPYProfile` CDF-trapezoid pattern). Constructors:
- `from_alpha(alpha_j, m_j, W0, g, delta, r_c)` — direct (Layer A); the controlled
  ground-truth knob for the diagnostics and the Phase-3 matched-$\Lambda_{\rm MSR}$ test.
- `from_imf(imf, n_comp, W0, g, delta, m_range, r_c)` — log-spaced mass bins; per bin
  $N_j=\int\xi\,dm$, $M_j=\int m\xi\,dm$ (reuse `IMFProtocol`), $m_j=M_j/N_j$; →
  Layer B → A.
- `sample_cluster(key, n_stars, M_total, G)` → `(positions, velocities, masses)`:
  allocate `n_stars` ∝ $N_j$; per-component inverse-CDF positions + isotropic
  directions (reuse `_sample_radii`); per-component speeds from
  $u^2 E_\gamma(g, \mu_j^{2\delta}\psi - u^2/2)$ at scale $s_j=s\mu_j^{-\delta}$,
  $s^2 = GM_{\rm tot}/(9 r_c\mu_{\rm tot})$ (reuse `_sample_unit_speed`); concatenate.

η-ready: constructors carry inert `r_a=None, eta=0.0` for Phase 2b.

## Validation (the evidence; per "Definition of Complete")

1. **$\delta=0\equiv$ single-mass** — `solve_multimass_limepy(δ=0)` $\psi$ identical to
   `solve_limepy_profile`; `MultiMassLIMEPY(δ=0).density == LIMEPYProfile.density`.
2. **Eigenvalue hits targets** — realized $M_j'$ match $M_j$ (residual $<10^{-3}$) for a
   Kroupa spectrum; $\bar m$ self-consistent.
3. **Per-group $Q_j\approx0.5$ — THE equilibrium proof** (Phase 0
   `per_group_virial_ratio` on the sampled cluster, split by component; unscaled).
4. **Segregation trend** — heavy more concentrated than light (validated $\Lambda_{\rm
   MSR}$ / radial concentration); segregation increases with $\delta$.
5. **Global $Q=0.5$** unscaled; all bound.
6. **Differentiability** — `jax.grad` through the core $(W_0,g,\delta,\alpha_j)$, the
   iteration $(M_j,\delta)$, and the sampler.
7. **Script + figure** `scripts/validate_multimass_equilibrium.py` →
   `seg_multimass_equilibrium.png` (per-component profiles; per-group $Q_j$ vs $\delta$;
   segregation vs $\delta$).

## Constraints

JAX-native core (no numpy/scipy in `src/`; scipy/limepy only in tests/scripts).
Fixed-iteration `scan` (never `while_loop`). Float64 (automatic). Function ≤100 LOC,
file ≤500 LOC (new module keeps `limepy.py` under the limit). `lambda_seg` kept
(complement, not replace). Released-core invariant stays green. Commit per layer; push
only when Anna says (HITL).

## Reuse (do not reinvent)

`limepy_density_hat`, the `diffrax.Tsit5` solve pattern, `_find_tidal_radius`, the
`LIMEPYProfile` CDF construction, `_sample_unit_speed`, `per_group_virial_ratio`
(Phase 0), `compute_lambda_msr` / radial concentration (`segregation_approx`),
`IMFProtocol` (`PowerLawIMF`/`ChabrierIMF`). STELLAR units; `G` threaded explicitly.
