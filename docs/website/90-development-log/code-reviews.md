# Code reviews

```{admonition} API retirement notice (2026-06, Phase 1c)
:class: warning
The audit records below predate the 2026-06 unified multi-component
redesign and reference APIs that were since **retired**:
`generate_cluster_ic`, the `populations` module, and the fractal-IC
generator path (moved to the experimental `gravoturb_fdf` package).
Multi-population clusters are now built with `MultiComponentCluster` —
see [](../10-theory/populations/index.md) and [](../30-api/cluster.md).
The records are preserved unedited as historical evidence.
```

This page is the landing record for full-package technical reviews of progenax.

| Date | Reviewer | Type | Grade | Record |
|------|----------|------|-------|--------|
| 2026-06-03 | Claude Opus 4.8 (5-lane follow-up) | Post-hardening regression + new-code audit | **A− (90/100)** | *this page* |
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

```{admonition} 2026-06-05 update — gravoturbulent/FDF subsystem rebuilt clean-room
:class: warning
The **gravoturbulent + fractal-density-field findings** in these audits (CR-FU-1's
`mode="bm19"` categorical-sampler OOM; M3's white-noise dense-tail under-sampling; the
`fdf_density` normalization off-by-one; the `s_max` mass-truncation; the
`fractal_gw_legacy` deprecation gap) concern modules that were **deleted and rebuilt
clean-room on 2026-06-05** as the experimental, repo-only **`gravoturb_fdf`** package.
The `cluster.fdf*`, `gravoturb`, and `fdf_density` file:line references below are
therefore **historical**. In the rewrite: the cornerstone is now **AC6** (realized vs
BM19 `f_dense`, ensemble bias ≤0.004% at 128³ — versus the −37% / 2.5×-low white-noise
era this audit flagged), and the OOM-prone `jax.random.categorical` tail sampler no
longer exists (mass-conserving rank copula + `gravoturb_fdf.field.sampling`). See
`src/experimental/gravoturb_fdf/VALIDATION_SUMMARY.md`.
```

---

## Follow-up audit — 2026-06-03 (post-hardening)

```{admonition} Status: every finding below was RESOLVED (2026-06-03)
:class: important
All Critical / Major / Minor items in this audit have since been **fixed and merged**
(see [](../00-getting-started/whats-new.md) and the resolution table at the end of this
section). The present-tense wording below — "still OOMs", "crashes under `jax.grad`",
"returns a NaN gradient" — records the state **at the time of the audit**, not current
behaviour.
```

**Document type:** Regression + new-code audit of the 2026-06 hardening (the 41-commit
`hardening/audit-2026-06` branch, merged at `644c28f`).
**Date:** 2026-06-03
**Reviewer:** Claude Opus 4.8, five independent review lanes (equilibrium-DF physics ·
differentiability/JAX · gravoturbulence-FDF · testing/reproducibility · code-craft/API/provenance),
each charged to **re-run** every claim; all Critical/Major findings then independently re-executed
by the lead reviewer.
**Scope:** `git diff 22ad6ad..644c28f` (the original audit's commit → the hardening merge):
**+4,980 / −4,661 across 34 source files**, four >1000-LOC monoliths split into 20 subpackage modules.
**Package version:** 0.1.0 at `644c28f` (main).

### Bottom line

**Every claim the hardening made is independently verified true.** The King and EFF velocity DFs are
genuine equilibria (unscaled Q = 0.499–0.502), `c(W₀)` matches King (1966) Table II to ≤0.02, the
latent King density bug is gone, the BM19 rank copula reproduces `f_dense` at β=4 (0.0548 vs 0.0568,
realization scatter collapsed to ~0), C1/C2/M5 are correct, test coverage is **85 %** (measured — the
original audit could not), and the four subpackage splits are byte-identical with zero API loss. The
original audit's two Criticals and nine Majors are genuinely closed.

**But the follow-up surfaced two new Critical launch-blockers — both "untested twins" of a fix that
was applied to only one of two call sites:**

| Severity | Count | Headline items |
|----------|-------|----------------|
| 🔴 Critical | 2 | The **default/recommended** `mode="bm19"` tail sampler still OOMs (~26 GB @ 5000×64³) — the CI memory fix patched only the `pn11_legacy` twin; `build_spatial_ic` (the flagship public IC, and the CLAUDE.md "fully differentiable" example) **crashes under `jax.grad`** |
| 🟠 Major | 2 | `compute_potential_energy(softening=0.0)` (the default; the C1 doc example) returns a **NaN gradient**; the M3-guard test `test_phi_copula_reproduces_the_bug` is **seed-fragile** (flaky-CI risk) |
| 🟡 Minor | ~12 | M3's new `float()` guard makes `init_bm19_density_field` non-grad/JIT; a 4th phantom analytical doc ref (the real symbol is `harmonic_oscillator`); an `energy_sorted_segregation` export overclaim; `profiles/api.py` at 37 % coverage; stale `tests/README.md` & "848" count; 9 functions >100 LOC in the split files; etc. |
| 🟢 Verified | many | every hardening physics / numerics / reproducibility / API claim (see above) |

**Grade: A− (90/100)** as-is — up from B+ (87) on the back of the now-verified science, the nine
closed Majors, a measured 85 % coverage, and a green CI gate. It clears to a solid **A** the moment
the two 🔴 twins are patched: both are one-to-few-line fixes whose correct pattern *already lives in
the repo* (the `cumsum`+`searchsorted` inverse-CDF for the sampler; dropping the `float()` cast for
the grad path).

### The two Critical findings (independently re-verified by the lead)

**CR-FU-1 — the `mode="bm19"` (default, recommended) tail sampler still OOMs at production scale.**
The CI memory fix (`c5d2061`) replaced `jax.random.categorical` with a `cumsum`+`searchsorted`
inverse-CDF — but **only in the `pn11_legacy` branch** of `cluster/fdf_density/sampling.py`. The
default path `mode="bm19"` (`cluster/gravoturbulent.py:126`, `cluster/fdf_density/sampling.py:103`,
both labelled "RECOMMENDED") routes through `sample_positions_tail_bm19` → `sample_positions_from_pmfs`
→ **`sample_from_pmf` (`cluster/fdf_tail.py:382`)**, which still calls
`random.categorical(key, log_pmf, shape=(n_samples,))` — Gumbel-max, materializing an
`(n_samples × n_cells)` array, called twice (tail + smooth). No test exercises it at scale, so CI
stays green.

*Re-verified (this audit, macOS peak RSS, float64):*

```text
searchsorted (N=2000, n_cells=262144): peak RSS  0.52 GB  (delta +0.01 GB)
categorical  (N=2000, n_cells=262144): peak RSS 11.01 GB  (delta +10.49 GB)
# at N=5000 the categorical materializes ~10.5 GB x2 calls -> ~26 GB OOM
```

A user following the documented `mode="bm19"` workflow at 5000 stars × 64³ OOMs exactly as the fix
claimed to prevent. **Fix:** port the same `searchsorted` inverse-CDF into
`sample_from_pmf` / `sample_positions_from_pmfs`.

**CR-FU-2 — `build_spatial_ic` crashes under `jax.grad`, breaking the headline guarantee on the
flagship path.** `builders.py:258` does `softening = float(softening)`, where
`softening = softening_factor · characteristic_radius() / N^{1/3}` — a traced quantity when
differentiating w.r.t. a profile scale (`r_h`). The package's headline is "fully differentiable IC
generation," and the CLAUDE.md example differentiates a loss built from `build_spatial_ic`.

*Re-verified (this audit):*

```text
jax.grad(loss_built_from_build_spatial_ic)(r_h=1.0)
  -> ConcretizationTypeError at builders.py:258  (float(softening))
```

The existing grad tests miss it because they re-implement the IC via `sample_positions` /
`sample_velocities` directly and never call `build_spatial_ic`. **Fix:** `softening =
jnp.asarray(softening)` (it is only stored on `ICResult` and passed to `virial_scale`, both
array-safe — the same lesson was already applied at `king.py:240`), plus a
`jax.grad`-through-`build_spatial_ic` regression test.

### Major / Minor (selected)

- 🟠 **`compute_potential_energy(softening=0.0)` NaN gradient.** `builders.py:132-135` computes
  `r_soft = sqrt(r² + softening²)` then masks the diagonal with `where(eye, inf, ·)` *after* the
  `sqrt`. At the default `softening=0`, the diagonal `sqrt(0)` has an infinite derivative the later
  `where` cannot rescue (`0·inf = nan`). *Re-verified:* `softening=0.0 → grad nan`,
  `softening=0.01 → finite`. This is the exact call form in the CLAUDE.md C1 example. Fix: mask the
  `r²` diagonal to a positive constant *before* the `sqrt` (double-`where`).
- 🟠 **`test_phi_copula_reproduces_the_bug` is seed-fragile** (`tests/validation/test_bm19_field_tail.py`).
  Its `vals.std() > 0.2·f_dense` assertion fails on 3 of 6 seed-blocks (on some seeds the Φ-copula
  undersamples so hard the tail nearly vanishes, shrinking the scatter below the bound). The
  bug-reproduction itself (mean undersampling) is robust; only the scatter bound over-specifies — so
  a different default seed could flake CI in the very test guarding the headline M3 fix. Fix: assert
  the robust mean-undersampling signature.
- 🟡 **M3 introduced a new non-differentiability.** The resolution-guard `float()` at
  `field_init.py:284` makes `init_bm19_density_field` non-grad/JIT
  (`ConcretizationTypeError` w.r.t. `sigma_s_sq`, re-verified). Contained — it is a host-side setup
  routine (also un-JIT-able via the data-dependent `if expected_tail_cells < 5` warn) and not on the
  `generate_fractal_ic_density` hot path — but it defeats the design doc's "param gradients flow
  through the CDF table" claim for the standalone function. Guard behind a concreteness check.
- 🟡 **A 4th phantom doc ref survived the "3 phantom refs" fix:** a stale `harmonic_oscillator_*1d*`
  name (CLAUDE.md:217, README.md:86); the real symbol is `harmonic_oscillator`. Separately, CLAUDE.md:207/219
  lists `energy_sorted_segregation()` under "exported from `progenax.__init__`" though it is not in
  `__all__` (it exists as `progenax.cluster.energy_sorted_segregation`).
- 🟡 **`profiles/api.py` at 37 % coverage** — the one core-adjacent module below 60 % (the
  `make_profile` / `compute_profile_potential` dispatch wrapper; the physics modules it dispatches to
  are 97–100 %). Plus a stale `tests/README.md` (still lists the disproven `c≈0.8` and references the
  removed `king_K_function` tests) and a stale "collects 848" count (now 855).

### What the hardening got right (independently confirmed)

Equilibrium DFs re-derived from King (1966) / B&T (2008) eq. 4.46 and re-run: King lowered-Maxwellian
Q = 0.4995–0.5025 unscaled, EFF Eddington f(E) ≥ 0 with Q ≈ 0.50 (γ=5); `c(W₀)` =
0.681 / 1.034 / 1.529 / 2.119 vs Table II 0.67 / 1.03 / 1.53 / 2.12. C2 grad finite (FD rel-err 9.5e-9)
and JIT-safe. BM19 rank copula Spearman(g,s) = 1.000000 (monotone), f_tail 0.0548 vs analytic f_dense
0.0568 at β=4 with ~0 scatter; the resolution guard fires exactly at `expected_tail_cells < 5`; and the
inverse-CDF fix that *was* applied is distribution-faithful (χ²/dof = 1.02, zero leak into zero-prob
cells). Coverage 85 % (854 passed / 1 deselected, 0 collection errors, 0 unknown-mark warnings); the
new modules are 95–100 % covered; tolerances were genuinely tightened (virial 0.20→0.05, fractal/median
0.30→0.05) and the King/EFF tests assert the *unscaled* Q that truly tests M1/M2. uv.lock +
`[tool.uv.sources]` + `uv sync --locked` CI are wired; API parity is byte-identical with deterministic
RNG; M8 provenance (Jeřábková 2018 / Marks 2012) is pinned to specific equations/tables.

### Methodology & limitations

Five lanes read the source directly and re-ran claims against the installed package (`.venv`,
JAX 0.10.1, float64); the lead independently re-executed every Critical/Major (the F1 / F3 / OOM / F2
re-runs above). Lane sub-grades: physics 96, differentiability 78, gravoturbulence 78, testing 94,
craft 94. **Limitations:** as in the original audit, no long N-body integration of the generated ICs
was run (equilibrium rests on unscaled virial ratios + DF re-derivation, not evolved relaxation), and
the OOM was confirmed by RSS scaling at N=2000 plus the materialization arithmetic rather than by
allocating the full ~26 GB.

### Resolution status — 2026-06-03 follow-up

Worked on branch `hardening/followup-2026-06` (TDD: a failing test reproduced each finding before the
fix; per-batch human gates). All 2026-06-03 follow-up Critical/Major findings are resolved.

| Finding | Status | Commit(s) |
| --- | --- | --- |
| **CR-FU-1** bm19 default-path OOM (`random.categorical` Gumbel-max) | ✅ Resolved | `26a9ab9` (+ `0be73ff` degenerate-PMF guard) |
| **CR-FU-2** `build_spatial_ic` non-differentiable (`float(softening)`) | ✅ Resolved | `b227740` |
| 🟠 `compute_potential_energy` NaN grad at `softening=0` | ✅ Resolved | `5e7fc1e` |
| 🟠 flaky `test_phi_copula_reproduces_the_bug` | ✅ Resolved | `3566247` |
| 🟡 `init_bm19_density_field` resolution-guard breaks grad/JIT | ✅ Resolved | `0362dea` |
| 🟡 `energy_sorted_segregation` export overclaim | ✅ Resolved | `269d512` |
| 🟡 `profiles/api.py` coverage (37% → 100%) | ✅ Resolved | `2a8715c` |
| 🟡 `c(W₀)` ↔ King (1966) Table II guard | ✅ Resolved | `275f528` |
| 🟡 `harmonic_oscillator_*1d*` phantom + stale docs/counts | ✅ Resolved | `1e8ac50` |
| env: `--cov=<submodule>` narrow-scope abort root-caused + documented | ✅ Resolved | `efdac19` |

All 2026-06-03 follow-up Critical/Major resolved; package clears to A.

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
| MoeDiStefano2017 coefficients | `from_moe2017()` vs docstring | code (−0.28,1.42,0.48) ≠ docstring (0.042,1.39) | M-doc confirmed |
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

(cr-2026-resolution)=
### 6. Resolution status — 2026-06 hardening

Worked on branch `hardening/audit-2026-06` (TDD: a failing test reproduced each
finding before the fix; per-batch human gates). All Critical and Major **bug**
findings are resolved.

| Finding | Status | Commit(s) | Note |
| --- | --- | --- | --- |
| **C1** dropped `G` | ✅ Resolved | `645eed6` | `G` threaded into `sample_velocities`; Q correct with no rescale |
| **C2** King `K` NaN grad + non-JIT ctor | ✅ Resolved | `e9966f6`, `629770b` | double-`where` + array `xi_t`. `king_K_function` was later **removed** (it became orphaned when M6/the density bug were fixed); the W=0 clamp now lives on `king_lowered_maxwellian_density`, and the JIT-safe `from_W0_rc` + ODE-grad guards remain |
| **M1** King true DF | ✅ Resolved | `94e92ff` | lowered-Maxwellian `f(E)`; Q≈0.51 unscaled |
| **M2** EFF true DF | ✅ Resolved | `5fe5c90` | Eddington inversion; `f(E)≥0`; Q≈0.50 (γ=5) |
| **M4** non-collecting suite | ✅ Resolved | `08f98fa`, `5fe5c90` | quarantined, then the refactor-orphaned file deleted; suite collects **874** |
| **M5** CDF quadrature | ✅ Resolved | `ecc3198` | cumulative trapezoid (EFF + King) |
| **M6** King nondimensionalization | ✅ Resolved | `817b1a9` | factor-of-9 restored — **and a latent density–potential bug found during hardening** (the ODE sampled a non-King, 2–30× over-extended profile). `c(W₀)` now matches King (1966) Table II to ≤0.02 |
| **M3** BM19 dense-tail undersampling | ✅ Resolved | `d654e4f` | **rank/empirical-CDF copula** (the bug was copula non-Gaussianity at β=4, *not* finite resolution — a cascade was measured-and-rejected); realized `f_tail` 0.029±0.040 → 0.054±0.000 vs `f_dense` 0.057, + resolution guard |
| **M7** tooling | ✅ Resolved | `47568b4`, `4ef2dc2` | CI + markers + coverage; **`uv.lock` + `[tool.uv.sources]` + `uv sync --locked`** |
| **M8** IMF coefficient citations | ✅ Present | — | Jeřábková 2018 / Marks+2012 already cited in `imf/environment.py` + `cluster/README.md` |
| **M9** stub FDF calibration | ✅ Resolved | `5aa811e` | `UserWarning` + surfaced `version` (`v1_stub_uncalibrated`) at the `generate_cluster_ic` fractal path |
| Oversize units (rec. 11) | ✅ Resolved | `8a9208b`,`eb3e4f6`,`b660bf6`,`1d6878a`,`1301c7f` | `generate_cluster_ic` + `generate_fractal_ic_density` split to ≤100 code-lines; `binary.py`/`fdf_density.py`/`analytical/core.py`/`environment.py` → subpackages <500 LOC (API-preserving) |
| Test robustness (rec. 10) | ✅ Resolved | `7304ffa`,`ef6e69f`,`d4c12d1`,`28b2536`,`c1d8894` | virial→5%, fractal/median→5% (via N-bump), SanaOB −0.55 slope, thermal ⟨e⟩=2/3, binary + PN11 differentiability |
| Docs / docstrings (rec. 11) | ✅ Resolved | `b2e1f6d`,`ba55ebc`,`9f11753` | `moe2017()` + Chabrier single-star labels; CLAUDE/README/API metrics + DF-fidelity |
| Minor: `dV` off-by-one; pn11 routing | ✅ Resolved | `77e46d0`,`8abbf5b` | — |

**All audit Critical/Major findings are resolved.** One maintainability follow-up
remains ticketed: the eight files still 500–1000 LOC (the audit named only the
>1000-LOC files for splitting) → `docs/notes/2026-06-02-file-length-followup-ticket.md`.

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
