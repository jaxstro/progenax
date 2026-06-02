# Code reviews

This page is the landing record for full-package technical reviews of progenax.

| Date | Reviewer | Type | Grade | Record |
|------|----------|------|-------|--------|
| 2026-06-01 | Claude Opus 4.8 (multi-agent audit) | Expert scientific audit + code review | **B+ (87/100)** | *this page* |
| 2025-12-07 | Claude Opus 4.5 | Comprehensive code/architecture/science review | A (95/100) | [`2025-12-07-progenax-review.md`](2025-12-07-progenax-review.md) |

```{admonition} Why this audit grades lower than the 2025-12-07 review
:class: note
The earlier review was a largely *static* read and graded the package **A (95/100)**.
This audit added **runtime verification, adversarial differentiation/JIT probing,
literature cross-checking, and execution of the test suite**. That deeper probing
surfaced two runtime-confirmed bugs, a non-collecting test suite, and a science
gap in the gravoturbulent field sampler that a static read did not catch. The
analytic core remains excellent; the grade reflects *correctness and
reproducibility gaps that block publication-grade results*, not a regression in
the codebase.
```

---

## Expert scientific audit & code review — 2026-06-01

**Document type:** Full-package expert audit (scientific correctness · numerics · JAX/differentiability · gravoturbulence & cluster FDF · testing · reproducibility · code craft)
**Date:** 2026-06-01
**Reviewer:** Claude Opus 4.8, orchestrating the `astro-code-review` skill suite across seven parallel review agents
**Package version:** 0.1.0 (audited at commit `22ad6ad`, branch `main`)
**Methodology:** Each dimension was reviewed by a dedicated agent that read the
source directly; every Critical/Major claim was then **independently re-derived or
re-run** by the lead reviewer against the installed package (`.venv`, JAX float64).
Formulas and coefficients were verified against primary literature (NASA ADS / arXiv
full text). Quantitative evidence is reproduced in [§3](#cr-2026-evidence).

```{contents}
:local:
:depth: 2
```

(cr-2026-summary)=
### 1. Executive summary — state of the package

progenax is a JAX-native library for **differentiable initial conditions** for
N-body simulations: spatial profiles (Plummer, King, EFF), matching velocity
distribution functions, initial mass functions (Kroupa, Chabrier, Maschberger,
IGIMF, environment-dependent, binary), Keplerian binary populations, gravoturbulent
density-PDF models (BM19, PN11, PP20), and a fractal/FDF cluster framework.

**The analytic foundations are excellent and literature-faithful.** Every one of the
nine physics formulas cross-checked against primary sources matched
([§4.7](#cr-2026-provenance)): the Plummer ergodic DF, the King `K`-function, the
Chabrier single-star parameters, the Moe & Di Stefano (2017) multiplicity trend, the
Federrath et al. (2010) PDF variance, the Burkhart & Mocz (2019) transition density
`s_t = (α−½)σ_s²` (verbatim), and the Parmentier & Pasquali (2020) magnification ζ(p).
The Plummer profile+DF, the Kepler solver, and the IMF inverse-CDF samplers are
correct to machine precision and verified differentiable.

**However, runtime and adversarial probing surfaced issues a static read missed.**
Two are genuine, confirmed bugs (a dropped `G` in velocity sampling; a NaN gradient
in the King model); the default `pytest tests/` invocation currently **fails to
collect**; and the headline novel capability — the BM19 *turbulent density field* —
does not reproduce its own target dense-gas fraction at usable resolution, while the
test that "validates" it uses white noise and so passes vacuously.

#### Corrected key metrics

| Metric | Documented (CLAUDE.md / README) | Actual (this audit) |
|--------|--------------------------------|---------------------|
| Source LOC | ~9,400 | **~18,600** (61 modules) |
| Test functions | 432 | **~740** across 49 files |
| Default `pytest tests/` | "432 passing" | **Aborts at collection** (broken import); **812 pass** once excluded |
| Public API exports | 57+ | 57+ (unchanged) |

The self-documentation is roughly **2× stale** — the package nearly doubled (cluster
FDF, gravoturbulence, differentiable-binary, environment-IMF) since those counts were
written. Because `CLAUDE.md` is read by AI assistants making design decisions, this
matters beyond cosmetics.

#### Findings at a glance

| Severity | Count | Headline items |
|----------|-------|----------------|
| 🔴 **Critical** | 2 | `G` dropped in `build_spatial_ic` velocity sampling; King `K`-function NaN gradient |
| 🟠 **Major** | 9 | King/EFF velocity DFs not in equilibrium; BM19 field under-samples the dense tail (test masks it); non-collecting suite; EFF/King left-Riemann CDF; King Poisson scaling; no lockfile/CI; loose test tolerances; oversize functions; stale docs |
| 🟡 **Minor** | 12 | Stale `moe2017()` docstring; Chabrier "system IMF" comment; field `dV` off-by-one; BM19 `s_max` tail clip; latent PN11 crash; OM anisotropy; non-diff `argmax`; tidal COM; etc. |
| 🟢 **Positive** | 9 | Plummer/Kepler/IMF exactness; correct reparameterization; literature-faithful formulas; clean PRNG; isolated numpy; no `while_loop` |

**Grade: B+ (87/100).** Excellent, literature-grounded analytic core, but several
correctness and reproducibility defects must be resolved before results are
publication-grade. None of the issues is architectural; all have concrete,
low-risk fixes.

---

(cr-2026-findings)=
### 2. Severity-ranked findings

(cr-2026-critical)=
#### 🔴 Critical

**C1 — `build_spatial_ic` drops `G` when sampling velocities.**
`src/progenax/builders.py:249` calls
`velocity_df.sample_velocities(positions, masses, key_vel)` with **no `G`**, even
though `G` is a required argument of `build_spatial_ic` (line 231) and is correctly
threaded into `virial_scale` (line 267). Every velocity DF therefore silently falls
back to `DEFAULT_UNITS.G` (STELLAR). When `Q` is set (the default 0.5) the global
virial rescale hides the error because it only fixes the *total* energy ratio. When
**`Q=None`** (a documented option to disable rescaling) the bug is fully exposed.

*Verified (this audit):* with `G=PLANETARY.G, Q=None`, the realized virial ratio is
**Q = 0.000055**, matching the predicted `0.5·G_STELLAR/G_PLANETARY = 5.70×10⁻⁵` to
four significant figures; the same call with `G=PLANETARY.G, Q=0.5` returns
Q = 0.500 (rescale masks it). `populations.py:169` already passes `G=G` correctly.
**Fix:** `velocity_df.sample_velocities(positions, masses, key_vel, G=G)` — one line.

**C2 — King `K`-function produces a NaN gradient at `W=0`, breaking differentiable
King ICs.** `src/progenax/profiles/king.py:66-76` computes `sqrt_W =
jnp.sqrt(jnp.maximum(W, 0.0))` and guards the *value* with a single `jnp.where(W <
1e-10, 0.0, K)`. The `1/(2√W)` derivative of `sqrt` at `W=0` still flows through the
dead branch (classic "`where`-NaN" trap). The argument `W₀−ψ` hits exactly 0 at the
cluster center (`ψ(0)=W₀`) and at the tidal radius (`ψ=0`), so the singularity is
unavoidable.

*Verified (this audit):* `jax.grad(king_K_function)(0.0)` → **`nan`**;
`jax.grad` of `Σψ` through `solve_king_profile` w.r.t. `W₀` → **`nan`**.
This silently violates the package's headline "fully differentiable" guarantee for
any King-based inference. **Fix:** the standard double-`where` / safe-primitive
pattern (compute `sqrt`/`exp` on a clamped `W_pos`, then `where` the result).

(cr-2026-major)=
#### 🟠 Major

**M1 — King velocity DF is not self-consistent with the King potential.**
`src/progenax/kinematics/king_df.py:124-156` samples an isotropic Gaussian with an
ad-hoc dispersion `σ₀² ≈ GM/(9 r_c)`, a *parabolic* `ψ(r) ≈ W₀(1−r²/(r_t²+r_c²))`,
and clips over-escape speeds down to `v_esc`. The parabola overestimates the true
ODE potential by ~2.8× at mid-radii; raw King velocities give a virial ratio of
**Q ≈ 6.7** (super-virial) before rescaling. `build_spatial_ic` then forces global
Q=0.5, but that fixes only total energy — the *radial* velocity structure remains
inconsistent with the King density, so King ICs are **not in detailed equilibrium**
and will visibly relax at *t*=0. (Honestly documented as "simplified," but the
magnitude warrants Major.) **Fix:** sample the true lowered Maxwellian
`f(E) ∝ e^{E/σ²}−1` using the ODE-interpolated `ψ(r)` already stored on the profile.

**M2 — EFF velocity DF has no self-consistent DF.** `src/progenax/kinematics/eff_df.py`
draws an isotropic Gaussian with a virial estimate `σ ≈ √(GM/6a)`. EFF ICs are
therefore not in equilibrium (documented limitation). Recommend a tabulated Eddington
inversion of the EFF potential, or a prominent non-equilibrium warning.

**M3 — BM19 turbulent field under-samples the dense tail; the validating test uses
white noise.** `src/progenax/cluster/fdf_density.py` (`init_bm19_density_field`) with
the physically-motivated power-spectrum slope β≈4 produces a dense-gas mass fraction
of **f_tail = 0.022 ± 0.020** against a theory value of **0.057** (~2.5× low, ~90%
realization scatter; the tail lives in 4–140 voxels of 64³). The 1-D ICDF and a
white-noise 64³ field both reproduce the target — so the analytic remap is correct —
but the production correlated field does not, at usable resolution. The unit test
`tests/unit/physics/test_bm19_pdf.py` passes only because it samples **white noise**
rather than the production field, giving false confidence in the framework's headline
claim. **Fix:** add a validation test that exercises `init_bm19_density_field` (β≈4)
and asserts the realization scatter; document that f_tail matches f_dense only in the
well-mixed limit; raise resolution or warn for tail-dominated work.

**M4 — Default test suite does not collect.**
`tests/integration/test_knobs_pipeline.py:13` imports
`progenax.profiles.mass_segregation`, but that module lives at
`progenax.cluster.mass_segregation`. A single collection error aborts the entire
`pytest tests/` run, so the documented "432 tests passing" is **not currently
reproducible** with the default invocation. **Fix:** correct the import path.

**M5 — EFF and King CDFs use a left-Riemann sum mislabeled "trapezoid."**
`src/progenax/profiles/eff.py:87` and `src/progenax/profiles/king.py:344` build the
cumulative mass as `jnp.cumsum(integrand) * dr` with a comment claiming a "trapezoid
approximation." *Verified:* for the EFF integrand the left-rule total-mass error is
**6.3×10⁻³** vs **1.0×10⁻⁶** for a true cumulative trapezoid on the *same grid*
(~6000× worse), biasing the sampled radial distribution. **Fix:** the one-line
cumulative-trapezoid form.

**M6 — King Poisson RHS omits the factor of 9 from the standard nondimensionalization.**
`src/progenax/profiles/king.py:130` integrates `d²ψ/dξ² + (2/ξ)dψ/dξ = −ρ̃`, whereas
the King (1966) / Binney & Tremaine (2008, eq. 4.131) scaling — in which `r_c` is the
King *core* radius `r_c = √(9σ²/4πGρ₀)` — carries `= −9 ρ̃`. As written, the model's
`r_c` does not correspond to the observational King core radius and the
W₀↔concentration mapping deviates from published King tables (the *shape* sampled is
still a valid King(W₀) family, since `from_W0_rc` sets `r_t = r_c·ξ_t` self-consistently).
**Action:** restore the factor of 9 (so `r_c` is King's core radius) **or** document
the nonstandard scaling, and validate `c(W₀)` against the King (1966) table.

**M7 — Environment unpinned / not reproducible.** progenax has lower-bound pins only
(`jax>=0.4.20`, …) and **no `uv.lock`**, unlike sibling packages in the workspace.
HMC/MCMC validation results are sensitive to JAX minor-version numerics. **Fix:**
`uv lock` and commit; document `uv sync --frozen` for exact reproduction.

**M8 — Provenance gaps on physically critical coefficients.** In
`src/progenax/imf/environment.py:62-88` the Jeřábková (`α₃ = −0.41x+1.94`) and Marks
(`−0.4072x̂+1.9383`, plus the 8-number `MARKS_TABLE3_COEFFICIENTS`) constants — which
set the high-mass IMF slope — are cited at block level but not pinned to a specific
equation/table/column. **Fix:** inline the figure/table/column citation per constant.
*(The `JERABKOVA_COEFFICIENTS` derivation block, by contrast, is exemplary — see
[Positives](#cr-2026-positive).)*

**M9 — FDF `D→χ` calibration is a stub in production with no user-facing warning.**
`src/progenax/cluster/fdf_calibration.py` ships `FDF_STUB_CALIBRATION`
(`version="v0_uncalibrated"`, identity placeholder map). Any fractal-substructure
result via the public `generate_cluster_ic(..., fractal=...)` path uses an
unvalidated heuristic, but no `UserWarning` is emitted at the call site. **Fix:**
warn at the call site and surface the calibration `version`.

```{admonition} Testing & tooling cluster (also Major)
:class: caution
Grouped because they share one root cause — no CI gate:
**no `.github/workflows/`**, no pytest markers (the 3-tier architecture is
undeclared; `@pytest.mark.slow` warns as unknown), and no coverage config. Test-side:
the Plummer virial-ratio tolerance is **20%** (`tests/conftest.py:86`; should be ~5%
at N=5000), several `rtol≈0.3` checks on quantities the code constructs exactly
(`test_fractal.py:354`, `test_cluster_ic.py:370`), the **SanaOBPeriod power-law slope
is never tested** against Sana+2012, and there are **no differentiability tests** for
the King/EFF velocity DFs, the binary period/eccentricity distributions, or the PN11
pipeline. Oversize units also sit here: `generate_cluster_ic` (226 LOC),
`BinaryIMF` (339-LOC class), `generate_fractal_ic_density` (163 LOC) all exceed the
100-LOC project limit.
```

(cr-2026-minor)=
#### 🟡 Minor (selected)

- **`moe2017()` docstring is stale.** `src/progenax/imf/differentiable_binary.py:150`
  advertises `(a=0.0416, b=1.3925)`, but the code calls `from_moe2017()` and *uses*
  `(a=−0.2799, b=1.417, c=0.4755)` — *verified at runtime*. The model is correct; the
  docstring lies. (An earlier agent over-rated this "Critical"; the code is sound, so
  it is a documentation defect.)
- **Chabrier "system IMF" mislabel.** `src/progenax/imf/chabrier.py` repeatedly calls
  its parameters the "system IMF," but `m_c=0.08, σ=0.69, A_ln=0.158` are the Chabrier
  (2003) **single-star** disk values (the system IMF is `m_c≈0.22, σ≈0.57, A≈0.086`).
  The numbers are right for sampling individual stars; only the label is wrong.
- **`DensityField3D` normalization off-by-one.** `fdf_density.py:525,687` use
  `dx = 2L/N` while the grid is `linspace(−L, L, N)` (spacing `2L/(N−1)`), so the
  documented `∫ρ dV = 1` is off by `(N/(N−1))³` = **1.048 at N=64** (1.024 at N=128).
  Harmless for sampling (PMFs renormalize) but breaks absolute normalization.
- **BM19 `s_max = s_t + 10/α`** (`bm19_pdf.py:208`) bounds the *volume* tail but
  truncates **18.9%** of the *mass* tail at α=1.2 (3.6% at α=1.5); use `10/(α−1)`.
- **Latent PN11 crash.** `tail_layer_from_env(..., model="pn11")` yields `mode="pn11"`,
  which is unrouted in the `generate_fractal_ic_density` dispatch and raises
  `ValueError("Invalid mode 'pn11'")`.
- Osipkov–Merritt transform imposes a deterministic `v_r/v_t` ratio (moment-matching),
  not a true anisotropic DF (`kinematics/anisotropy.py:77`); `argmax` in
  `_find_tidal_radius` (`king.py:228`) and `sample_m_total_packed` (`base.py:207`) are
  non-differentiable; `apply_tidal_truncation` does not recenter the COM
  (`tidal.py:108`); `from_W0_rc` silently returns the grid edge if ψ never crosses 0
  (high W₀); the `power_law` PPF loses precision in a thin band near α=1; the
  `fractal_gw_legacy` deprecated path emits no `DeprecationWarning`; the `fdf.py`
  module docstring is truncated mid-sentence.

(cr-2026-positive)=
#### 🟢 Positive (verified correct)

- **Plummer profile + DF are exact** (verified): scale radius, inverse-CDF
  `r=a√(u^{2/3}/(1−u^{2/3}))`, and the `q²(1−q²)^{7/2}` speed law sampled as
  `q²~Beta(3/2,9/2)` with σ²=v_esc²/12 — a rejection-free, differentiable equilibrium
  sampler.
- **Kepler solver is machine-precision** (`binaries/kepler.py`): `from_state ∘ to_state`
  recovers `(a,e)` to 1e-16 and position to 1e-14 even at e=0.99; the e→0 and i→0
  singularities are handled with explicit fallbacks.
- **IMF inverse-CDFs are accurate & differentiable**: Maschberger uses an exact
  analytic inverse; Chabrier's 30-iteration Newton reaches 4e-16; all PPF gradients are
  finite.
- **`DifferentiableBinaryModel` reparameterization is correct** and grad-verified
  (all parameters finite, JIT-safe); soft weights converge to the hard threshold as
  T→0; `f_b` = 0.23/0.43/0.91/0.99 at 0.1/1/20/100 M☉ matches the Moe+2017 trend.
- **Gravoturbulence analytic layer is literature-faithful** — BM19 `σ_s²`, `s_0`,
  `s_t`, the piecewise mass integrals (`f_dense` matches direct quadrature to <1e-13),
  PN11 critical density, and the PP20 ζ(p) closed form (ζ(0)=1, ζ(1.5)=√2) all check out.
- **JAX hygiene is clean**: no `jax.lax.while_loop` anywhere; numpy/scipy are confined
  to non-differentiable diagnostics/validation modules that the core never imports;
  PRNG keys are always split before independent draws; `stop_gradient` in the FDF
  displacement field is the correct reparameterization pattern.
- **Provenance done right**: the `JERABKOVA_COEFFICIENTS` derivation block and the
  PP20 docstring derivation are gold-standard documentation.

---

(cr-2026-evidence)=
### 3. Quantitative evidence

All checks were run against the installed package in `.venv` (JAX float64).

#### Test-suite status

`pytest tests/` (default) **aborts during collection**:

```text
ERROR tests/integration/test_knobs_pipeline.py
  ModuleNotFoundError: No module named 'progenax.profiles.mass_segregation'
!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!
```

With that one file excluded the suite is healthy — **812 passed, 17 warnings in
252 s** — so the broken import is a stale path, not a physics failure (and the real
passing count is ~2× the documented "432," consistent with the package having
roughly doubled). A `PytestUnknownMarkWarning: Unknown pytest.mark.slow` confirms
markers are undeclared, and several `DeprecationWarning: mode='pn11_legacy'` lines
appear (distinct from the unrouted `mode='pn11'` crash noted in §2).

#### Verification battery (independent re-runs of the headline claims)

| Check | Command essence | Result | Verdict |
|-------|-----------------|--------|---------|
| King `K` gradient | `jax.grad(king_K_function)(0.0)` | `nan` (finite 0.097 at W=3) | C2 confirmed |
| King solve gradient | `grad(Σψ) wrt W₀` | `nan` | C2 confirmed |
| `build_spatial_ic` `G`-drop | `G=PLANETARY, Q=None` | Q = **5.5×10⁻⁵** (predicted 5.70×10⁻⁵) | C1 confirmed |
| ↳ control | `G=PLANETARY, Q=0.5` | Q = 0.500 (rescale masks) | C1 scope confirmed |
| Moe2017 coefficients | `from_moe2017()` vs docstring | code (−0.28,1.42,0.48) ≠ docstring (0.042,1.39) | M-doc confirmed |
| EFF CDF rule | left vs trapezoid, same grid | err 6.3e-3 vs 1.0e-6 | M5 confirmed |
| Field `dV` | `(N/(N−1))³` | 1.048 (N=64), 1.024 (N=128) | Minor confirmed |
| BM19 field f_tail | `init_bm19_density_field`, β=4, 8 seeds | 0.022 ± 0.020 vs theory 0.057 | M3 confirmed |

---

(cr-2026-appendix)=
### 4. Detailed appendix

(cr-2026-physics)=
#### 4.1 Physics correctness

The Plummer pipeline is exact end-to-end (profile, inverse-CDF, ergodic DF, virial
machinery). `compute_potential_energy` correctly carries `G` and sums the strict
upper triangle (no self-pairs, no double counting); `Q=T/|V|` uses the 0.5-equilibrium
convention; `virial_scale` retargets kinetic energy correctly. Jacobi radii (point-mass
and isothermal), the uniform sphere, and the rotation transforms are all dimensionally
and structurally correct. The two genuine physics defects are **C1** (dropped `G`) and
the **King velocity DF** (M1) and **EFF velocity DF** (M2) equilibrium gaps. The King
Poisson scaling (M6) is a nondimensionalization-convention issue affecting the meaning
of `r_c`, not the sampled shape.

(cr-2026-numerics)=
#### 4.2 Numerical methods

No catastrophic-cancellation or convergence bugs in the solvers. The Kepler Newton
scan (50 fixed iterations) and the Chabrier/Maschberger/power-law PPFs are
float64-accurate; epsilon guards (`+1e-30`, `+1e-12`) sit in O(1) denominators and are
numerically inert. The actionable items are the **left-Riemann CDF** (M5, ~6000×
accuracy loss, one-line fix) and the **BM19 `s_max` mass-tail clip** for shallow
slopes (Minor). The `power_law` PPF has a thin precision gap near α=1 (widen the
log-branch threshold or use `expm1`), and `TaperedPowerLaw` PPF is limited to ~1e-4 by
its tabulated CDF — both negligible for standard IMFs.

(cr-2026-jax)=
#### 4.3 JAX & differentiability

The differentiable hot paths (IMF PPFs, Kepler solve, `DifferentiableBinaryModel`) are
all verified grad- and JIT-safe. The exception is the **King pipeline**: the
`K`-function NaN gradient (C2) and the `float(xi_t)` concretization in
`_find_tidal_radius` (`king.py:239`) make `from_W0_rc` neither JIT-able nor
differentiable w.r.t. W₀. *Note:* the `solve_king_profile` docstring blames the JIT
limitation on `n_points`, but `jax.jit(solve_king_profile)(7.0)` actually succeeds —
the real blockers are the `float()` cast and the NaN gradient. Fixing both makes the
King model match the package's differentiability guarantee. numpy/scipy isolation, the
absence of `while_loop`, PRNG hygiene, and `stop_gradient` usage are all clean.

(cr-2026-gravoturb)=
#### 4.4 Gravoturbulence & cluster FDF

The analytic BM19/PN11/PP20 layer is the strongest science in the package — every
coefficient was re-derived and confirmed (the `f_dense` piecewise integral matches
direct quadrature to <1e-13). The risk is entirely in the **sampled 3-D field** (M3):
with a realistic β≈4 spectrum the dense tail is carried by a handful of correlated
voxels, so the realized `f_tail` is biased low with ~90% scatter, and the unit test
masks this by using white noise. Mass (`Σm=M_total`), COM, and global virial ratio are
conserved exactly across all IC generators. The FDF velocity field currently assigns
smooth-Plummer velocities to a turbulent density field (no turbulent power spectrum /
position–velocity correlation) — a documented modeling limitation. The `D→χ`
calibration is an explicit stub (M9).

(cr-2026-testing)=
#### 4.5 Testing

Coverage is genuinely strong for the IMFs, the Kepler binary mechanics, the BM19/PP20
formulas, and `DifferentiableBinaryModel` (machine-precision conservation tests; broad
gradient checks). The gaps are: a non-collecting default suite (M4); the 20% virial
tolerance and a few 30% `rtol` checks on exactly-constructed quantities; an untested
SanaOBPeriod slope; and missing differentiability tests for the King/EFF DFs, binary
distributions, and PN11. No skipped/xfail tests and no mock-only assertions were found
— the suite tests real physics, it just needs tighter tolerances and a CI gate.

(cr-2026-craft)=
#### 4.6 Reproducibility & code craft

Add a `uv.lock` (M7) and CI. Pin the Jeřábková/Marks coefficients to specific
equations/tables (M8) and warn on the stub FDF calibration (M9). Update the stale
LOC/test counts in `CLAUDE.md` and `README.md`. Split the three oversize units
(`generate_cluster_ic`, `BinaryIMF`, `generate_fractal_ic_density`) — the existing
`# TODO: split binary.py into a binary/ subpackage` should be promoted to a tracked
task. Coefficient centralization (named dicts in `environment.py`) and the
`fdf_hyperparams` calibrated/uncalibrated separation are already good patterns.

(cr-2026-provenance)=
#### 4.7 Literature & provenance verification

All nine items cross-checked against primary sources **matched**:

| Item | Code | Source | Verdict |
|------|------|--------|---------|
| Plummer DF `f(E)∝E^{7/2}`, `q²~Beta(3/2,9/2)` | `plummer_df.py` | Plummer 1911; B&T 2008 §4.3; Dehnen 1993 | ✅ |
| King `K(W)=erf√W−(2/√π)√W e^{−W}`, lowered Maxwellian | `king.py`, `king_df.py` | King 1966 AJ 71 64; B&T §4.3 | ✅ |
| Chabrier `m_c=0.08, σ=0.69, A=0.158` | `chabrier.py` | Chabrier 2003 Table 1 — **single-star** (not system) | ✅ (label fix) |
| Moe+2017 `f_b(m)` logistic, endpoints | `differentiable_binary.py` | Moe & Di Stefano 2017 ApJS 230 15 | ✅ |
| Federrath `σ_s²=ln(1+b²M²)`, b=0.4 | `bm19_model.py` | Federrath et al. 2010 A&A 512 A81 | ✅ |
| BM19 `s_t=(α−½)σ_s²` | `bm19_pdf.py` | Burkhart & Mocz 2019 ApJ 879 129, Eq. 2 (verbatim) | ✅ |
| PN11 critical density / SFR-per-t_ff | `pn11_model.py` | Padoan & Nordlund 2011 ApJ 730 40 | ✅ |
| PP20 ζ(p)=(3−p)^{3/2}/[2.6(2−p)] | `pp20_magnification.py` | **Parmentier & Pasquali** 2020 ApJ 903 56 (arXiv:2009.10652); origin Parmentier 2019 | ✅ |
| Kepler `M=E−e sinE`, `T=2π√(a³/GM)`, R_z R_x R_z | `kepler.py` | Murray & Dermott 1999 | ✅ |

PP20 attribution (Parmentier & **Pasquali**, not Pfalzner) is correct and **consistent
across the source code and the website theory/API/bibliography pages**. The only
documentation defects are the Chabrier "system IMF" comment and the stale `moe2017()`
docstring (both Minor).

---

(cr-2026-recommendations)=
### 5. Prioritized recommendations

**Now (correctness — hours):**
1. **C1** — pass `G` in `builders.py:249`.
2. **C2** — double-`where` guard in `king_K_function`; drop `float(xi_t)` so the King
   pipeline is differentiable/JIT-able.
3. **M4** — fix the `mass_segregation` import in `test_knobs_pipeline.py`; re-run the
   full suite and refresh the documented pass count.
4. **M5** — cumulative-trapezoid in `eff.py`/`king.py`.

**Next (science fidelity — days):**
5. **M3** — add a BM19 *production-field* validation test (β≈4) asserting realized
   `f_tail` scatter; document the well-mixed caveat.
6. **M1/M2** — implement the true King lowered-Maxwellian (and an EFF Eddington
   inversion) or add prominent non-equilibrium warnings.
7. **M6** — restore the factor of 9 (or document the scaling) and validate `c(W₀)`
   against the King table.

**Soon (reproducibility & rigor — days):**
8. **M7/tooling** — `uv lock`; add `.github/workflows/tests.yml`, pytest markers, and
   coverage config.
9. **M8/M9** — pin the Jeřábková/Marks coefficient citations; warn on the stub FDF
   calibration.
10. Tighten the virial-ratio (→5%) and fractal-median (→5%) tolerances; add the
    SanaOBPeriod-slope and King/EFF/PN11/binary-distribution differentiability tests.
11. Refresh `CLAUDE.md`/`README.md` metrics; split the three oversize units; fix the
    `moe2017()` and Chabrier docstrings.

---

(cr-2026-methodology)=
### 6. Methodology & limitations

Seven parallel agents (physics, numerics, JAX/differentiability, gravoturbulence +
cluster FDF, testing, reproducibility + code craft, and literature verification) read
the source directly using the `astro-code-review` skill methodology. Every
Critical/Major claim was then independently re-derived or re-executed by the lead
reviewer against the installed package; results are in [§3](#cr-2026-evidence). Formula
provenance was checked against NASA ADS / arXiv full text.

**Limitations:** the audit did not run long N-body integrations of the generated ICs
(the equilibrium claims for M1/M2 rest on virial-ratio measurements and potential
comparisons, not on evolved relaxation tests); coverage percentages were not measured
(no coverage harness is configured — see M-tooling); and the cluster FDF framework is
research-grade code where some "limitations" are modeling choices rather than bugs and
are flagged as such. Findings reflect the package at commit `22ad6ad`.
