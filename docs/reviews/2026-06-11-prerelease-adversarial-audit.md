# progenax — Pre-Release Adversarial Audit (2026-06-11)

**Reviewer:** Claude Fable 5 orchestrating six specialist audit agents (profiles/kinematics science, IMF/binaries science, cluster/builders/diagnostics science, JAX/differentiability, test suite, architecture/packaging). Every claim below carries file:line evidence from this session; agents independently re-derived formulas, executed gradient finite-difference cross-checks, transcription-checked Moe & Di Stefano (2017) Table 13 against the PDF, measured King CDF errors numerically, built and inspected the wheel, and pulled live CI run logs.

---

## 1. Executive Summary

The scientific core of progenax is in **unusually good shape for research software** — and that praise is specific: the Plummer inverse-CDF, the King ODE (factor-9 nondimensionalization re-derived), the Merritt-1985 OM coefficient 63/4, the Eddington inversion incl. the boundary term, the Kepler solver (machine precision to e=0.9999 *with correct gradients*), the Murray & Dermott element rotation matrix, the Moe Table 13 transcription (all 80 cells), Sana+2012 and Maschberger 2013 parameters, the GZ15 multimass LIMEPY engine (gated by parity against the canonical reference code at a pinned SHA with embedded provenance), and the CW04 Q diagnostic were each independently verified this session. The differentiability engineering (double-where NaN guards, straight-through estimators, checkpointed blocked kernels, AD-vs-FD validation tests) is state of the art; every headline gradient claim tested empirically verified against finite differences to ≤1e-5 relative.

**But the repository is not releasable today**, for reasons that are mostly at the boundary rather than in the physics:

1. **CI is red on main and has run zero tests for ≥5 pushes** (missing `jaxstroviz` sibling checkout).
2. Even when green, **CI never runs the slow-marked headline validations** (LIMEPY parity, Engine A-vs-B anchor, multimass equilibrium) that the docs advertise as the project's trust anchors.
3. **Nobody outside this machine can install the package** (path deps on unpublished `jaxstro`/`jaxstroviz`; no LICENSE; no metadata).
4. The **README teaches a deleted API** (`TwoComponentConfig`) and omits the flagship `MultiComponentCluster`; docs advertise phantom classes (`IGIMF`, `EnvironmentIMF`) and a nonexistent `[all]` extra and `progenax-legacy` PyPI package.
5. **Two confirmed science bugs**: the Moe & Di Stefano F_twin normalization (twins mixed against q>0.1 instead of the paper's q>0.3 → +22% twin overweight at solar logP=1), and an inverted stellar mass–radius relation in `compute_stellar_radii`.
6. **One quantified numerical defect**: King/Michie position-CDF grids under-resolve the core at W0 ≳ 9 (measured +18% enclosed-mass error at 0.3 r_c for W0=9, ~3.6× at W0=12).
7. `BaseIMF.sample_fixed_n` **silently returns 30% less mass than requested** when the target is unreachable, and the whole sampling-mode family has zero tests.

All blockers are fixable in days, not months. Classification: **Release candidate** for the core (profiles/DFs/Kepler/builders), **Beta** for the binary-population layer until F_twin is fixed, **packaging/CI: pre-alpha**.

---

## 2. Overall Release Readiness

**Classification: Beta scientific software, close to Release Candidate.**

Why not RC: a release candidate must be installable, have green CI enforcing its advertised validation claims, and have no known science bugs in features the docs call "faithful." None of those holds today. Why not Alpha: the validation culture is far beyond alpha — external-reference gating (King 1966 Table II, canonical LIMEPY at pinned SHA, CW04 Table 1, IAU constants, Moe Table 13), measured-first frozen tolerances with written noise budgets, AD-vs-FD gradient tests, and honest-scope documentation throughout.

The gap to public release is ~2 weeks of mechanical work (CI, packaging, README, two science fixes) — **not** a rewrite.

---

## 3. Highest-Risk Findings

| # | Severity | Category | Finding | Files | Blocks release |
|---|---|---|---|---|---|
| R1 | **Critical** | CI | CI entirely red on main since ≥2026-06-10; `uv sync` dies on missing `../jaxstroviz`; zero tests enforced anywhere but the laptop | `pyproject.toml:43-46`, `.github/workflows/tests.yml:26-62` | **YES** |
| R2 | **Critical** | Packaging | Wheel not installable outside this machine: `jaxstro` unpinned + unpublished; `[tool.uv.sources]` paths don't ship; no LICENSE file, no author/classifier/URL metadata (README claims MIT with no license grant) | `pyproject.toml`, missing `LICENSE` | **YES** |
| R3 | **High** | Science | **Moe & Di Stefano F_twin mis-normalized**: twin excess mixed against the q∈[0.1,1] population; paper (p.5, Fig. 2) defines F_twin against q>0.3. Measured: realized paper-convention F_twin = 0.367 vs intended 0.30 at solar logP=1 (+22% twins), with companion deficit at q<0.3. Affects `MoeDiStefano2017Full`, `MoeJointOrbit`, `MoeCompanions` — everything labeled "faithful" | `src/progenax/imf/binary/moe_di_stefano.py:286-289` | **YES** (for the "faithful" label) |
| R4 | **High** | Science | **King/Michie/EFF CDF grid under-resolves the core at high W0**: linear 1000-pt grid over [0, r_t]; measured enclosed-mass error at 0.3 r_c: +0.9% (W0=7), **+17.7% (W0=9)**, **+264% (W0=12)**. Concentration tests pass (use the ODE, not the CDF); global Q passes (core mass tiny) — so the suite is blind to it | `src/progenax/profiles/king.py:359`, `michie.py:177`, `eff.py:83` | **YES** for advertised W0 ≳ 9 support; no for W0 ≤ 8 |
| R5 | **High** | Science | **`sample_fixed_n` silent mass shortfall**: docstring says "exactly n masses summing to m_total"; measured Maschberger n=1000, target 500 M☉ → returns **349 M☉** with no warning (one-sided quantile stretch hits its ceiling; Newton clips to bound silently). Zero tests for all four sampling modes | `src/progenax/imf/base.py:230-281` | **YES** (silent quantitative failure in documented API) |
| R6 | **High** | Science | **`compute_stellar_radii` mass–radius exponents inverted** vs standard MS homology (code: R∝M^0.8 above 1 M☉, M^0.57 below; literature: the opposite). 10 M☉ → 6.3 R☉ (should be ~4); 0.2 M☉ → 0.40 R☉ (observed ~0.22). No citation; tests assert the code's own formula back at itself. Propagates into every `ICResult.stellar_radii` (collision radii downstream) | `src/progenax/builders.py:155-193` | No (auxiliary), but fix before paper |
| R7 | **High** | Docs/API | README quickstart teaches deleted `TwoComponentConfig`/`generate_two_component_cluster`; `MultiComponentCluster` (the flagship) absent from README; docs advertise nonexistent `IGIMF`/`EnvironmentIMF` classes and 16 IMF symbols not exported from `progenax.__init__`; `installation.md` commands (`[all]` extra) fail | `README.md:154-175`, `progenax/CLAUDE.md`, `docs/website/00-getting-started/installation.md:21,40` | **YES** |
| R8 | **High** | CI/Testing | All headline external-reference validations (`test_limepy_reference_parity`, multimass equilibrium, Engine A-vs-B anchor, Engine B AD-vs-FD) are `@pytest.mark.slow` and CI runs `-m "not slow"` — the trust anchors are enforced by local convention only | `tests/validation/*:pytestmark`, `tests.yml:72` | **YES** |
| R9 | **High** | Packaging | `progenax.diagnostics` imports numpy + scipy eagerly; neither is a declared dependency — `from progenax.diagnostics import compute_q_parameter` raises ModuleNotFoundError in a clean install | `src/progenax/diagnostics/substructure.py:20-22`, `__init__.py:41-46` | **YES** for any README follower |
| R10 | **Medium** | JAX | `PowerLawIMF` NaN gradient at α=1.0 exactly (where-NaN; confirmed by execution). The correct `exp_safe` fix already exists in `imf/differentiable.py:47-54` — port it to 3 sites | `src/progenax/imf/power_law.py:92-96, 200-204, 249-252, 276-280` | Borderline; fix is 3 lines × 3 sites |

---

## 4. Scientific Formula Audit

### 4.1 Verified correct (independently re-derived this session — not taken on faith)

| Formula | Where | How verified |
|---|---|---|
| Plummer a(r_h) = r_h√(2^{2/3}−1); inverse CDF r = a√(u^{2/3}/(1−u^{2/3})) | `profiles/plummer.py:48,122-123` | Algebraic re-derivation vs Aarseth+74 |
| Plummer speed g(q)∝q²(1−q²)^{7/2} ⇒ Beta(3/2, 9/2) | `kinematics/plummer_df.py:250` | Change-of-variables re-derived |
| Plummer OM DF coefficient 63/4 and bound ρ_a≤16/9 ⇔ r_a≥0.75a | `plummer_df.py:33,130` | Re-derived from Merritt 1985 Eq. 45–46 Γ-ratio |
| King ρ̂(W) = e^W erf(√W) − (2/√π)√W(1+2W/3) | `profiles/king.py:43-77` | Matches BT08 Eq. 4.131 |
| King Poisson ODE factor 9; r_c=√(9σ²/4πGρ₀); BCs + L'Hôpital center | `king.py:85-135` | Re-derived from ∇²W=−4πGρ/σ² |
| King DF: true lowered Maxwellian at local ψ, truncated at σ√(2ψ), σ²=GM/(9 r_c μ) | `kinematics/king_df.py:46-61,142-151,201-222` | Re-derived; no clipping; unscaled Q=0.5 tested at W0=7 and 12 |
| King c(W0) | `tests/validation/test_king_physics.py:266-274` | vs King (1966) Table II (paper-PDF-verified per project memory), \|Δ\|≤0.02–0.03 |
| Eddington inversion incl. (dρ/dΨ)\|₀/√E boundary term, u=√(E−Ψ) desingularization; OM via augmented density with full Q-variable inversion (NOT an approximation) | `kinematics/eddington.py:31-98` | Re-derived vs BT08 4.46a; three oracles (analytic Plummer f∝E^{7/2}, truncated closed form, rescale equivariance) |
| OM direction split: v_r=s cosθ, v_t=s sinθ/√(1+r²/r_a²) ⇒ β(r)=r²/(r²+r_a²) by construction | `eddington.py:133-177` | Jacobian consistency re-derived; a-posteriori \|Δβ\|<0.03–0.05 at 4 radii |
| Michie f∝exp(−s²u_t²/2)[exp(W−u²/2)−1]; closed-form u_r integral | `profiles/michie.py:26-60` | Re-derived; isotropic limit → King <1% |
| Virial PE = −G Σ_{i<j} m_i m_j/r_ij (strict i<j, explicit G); Q = T/\|V\|, 0.5 = equilibrium | `dynamics/virial.py:33-82` | Inspection + e2e FD |
| Kepler solver: Danby starter, 50 fixed Newton iters via scan | `binaries/kepler.py:311-358` | Executed: residual ≤9e-16 to e=0.9999; gradients finite & correct at e=0.999 |
| Elements→Cartesian R_z(Ω)R_x(i)R_z(ω) | `kepler.py:360-408` | Element-by-element vs Murray & Dermott Eq. 2.122 |
| Barycentric split, COM/momentum | `to_binary_state` | Executed: exactly 0.0; E_orb=−GM/2a to 1e-16 |
| Kepler III both directions | `binaries/kepler_period.py` | Exact roundtrip, explicit G |
| Moe Table 13 grids (γ_largeq, γ_smallq, F_twin, companion freq) | `imf/binary/moe_di_stefano.py` | **All 80 cells match the paper p.52 exactly** (PDF read) |
| Sana+2012 π=−0.55, f_bin=0.69, logP∈[0.15,3.5]; α=−1 log-uniform limit | `binaries/period.py` | vs paper PDF |
| Maschberger L3 pdf/quantile, defaults | `imf/` | vs Maschberger (2013) Table 1 Eqs. 3–4 (PDF) |
| Chabrier lognormal + power-law join (continuity 4e-9; A_pl within paper rounding) | `imf/chabrier.py` | Re-derived erf integral |
| Kroupa 3- and 4-segment piecewise normalization; α=1 guarded in `differentiable.py` | `imf/power_law.py`, `differentiable.py` | Continuity ≤2e-8; inverse-CDF algebra checked |
| Engine A = GZ15 multimass LIMEPY (NOT GG79): one ψ ODE with Σ_j α_j ρ̂_j; w_j = μ_j^{−δ}, m̄ = Σ m_j α_j (GZ15 Eq. 26); GG79 divergence documented | `profiles/limepy_multimass.py:213-264,519-524`, `cluster/multicomponent.py:344-355` | Re-derived + parity vs canonical LIMEPY code (cached, provenance-stamped, gates 1e-8–5e-4) |
| Engine B: per-component Eddington inversion against shared total Ψ; realizability gate raising per component; hybrid-sampling Q offset turned into a *prediction* and gated | `cluster/eddington_engine.py:166-217` | Code-trace + King A-vs-B anchor + EFF(γ=5)≡Plummer identity |
| jacobi_radius: point-mass r_J=R(m/3M_g)^{1/3} and isothermal (Gm/2Ω²)^{1/3} | `tidal.py` | Validated vs full restricted-3-body L1 root-find + (3/2)^{1/3} ratio test |
| CW04 Q: exact MST, m̄=L_MST/√(N·A), A=πR² (convex-hull rejected with quantified +0.1 bias), 2D projection | `diagnostics/substructure.py:118-124` | vs CW04 Table 1 anchors (0.79/0.84/0.93) |
| Figure-eight ICs (Chenciner–Montgomery/Simó digits), closure <1e-6, L=0 | `analytical/` | Digit-checked + self-validating integration |
| Marks+2012/Jeřábková+2018 environment-IMF: erratum threshold −0.87 applied; deliberate Eq.-9 constant deviation (2.83→0.2161) documented | `imf/environment/coefficients.py:44-53`, `mapping.py:151-155` | Code + docstring provenance |

### 4.2 Defects (beyond §3)

| # | Sev | Finding | Where | Fix |
|---|---|---|---|---|
| S1 | Med | `KingProfile` accepts arbitrary inconsistent `r_t`; `KingVelocityDF.r_t` stored but **never used** — `KingProfile(W0=7, r_c=1, r_t=10)` silently yields a non-self-consistent, non-equilibrium IC. README:320 itself shows this inconsistent triple. Validation tests normalize the foot-gun | `king.py:332-404`, `king_df.py:101,123` | Derive r_t from c(W0) by default; warn on deviation; remove or use the DF's r_t |
| S2 | Med | Default EFF (γ=3) Eddington DF is **knowingly ~5–8% sub-virial** (truncated ρ(r_t)>0 not representable by f(E)). Honest module docstring, but the *default constructor config* is the broken case, only γ=5 is in the headline table, and no test pins Q(γ=3) — a regression to −15% would pass | `kinematics/eff_df.py:15-21` | Constructor-level caveat + pinned Q(γ=3) band test |
| S3 | Med | Rotation transforms are additive kinematic overlays (`velocities + v_rotation`) that inject energy/L_z and break equilibrium; not stated at function level; `api.py` hint to rescale to Q=0.5 is physically misleading (isotropic rescale ≠ stationarity) | `kinematics/rotation.py:14-117`, `api.py:199-235` | Explicit "breaks equilibrium" caveat; optionally sign-flip (Lynden-Bell) equilibrium-preserving variant |
| S4 | Med | `apply_tidal_truncation` leaves survivor velocities drawn for the untruncated potential → super-virial survivor set, some formally unbound stars near r_t; undocumented, untested | `tidal.py:119-165` | Docstring paragraph + recommend re-virialization; optional validation test |
| S5 | Med | `q_approx` calibration (1.375696) from ONE point (uniform sphere, N=500); ~0.1 over-read in concentrated regime documented only in test docstrings, not API docs; N-dependence untested | `diagnostics/q_approx.py:36-39` | Calibration sweep over (profile, N); regime statement in API docstring |
| S6 | Med | `energy_sorted_segregation` docstring oversells "clean equilibrium": segregated mass-weighted density ≠ parent profile, so self-consistent potential differs; the validation quietly applies a global virial rescale that the docstring never tells users to do | `cluster/mass_segregation.py:6-11,56-58` | Add self-consistency caveat + "finalize with rescale" |
| S7 | Med | `MoePeriod` uses companion *frequency* per decade as a binary period *pdf* (folds outer tertiaries in; O-star f_mult=2.1), with flat clamped extrapolation outside logP∈[1,7]; undocumented | `moe_di_stefano.py:315-360` | Document; consider edge anchors |
| S8 | Low | `MoeEccentricity` extends p(e)∝e^η to full [0, e_max]; paper calibrates only to 0.8 e_max and M1 ranges 0.8–3, >7 M☉ | `binaries/eccentricity.py:254-280` | One-line caveat; optional taper |
| S9 | Low | `MoeDiStefano2017` (period-averaged) twin = invented Gaussian N(1, 0.03), eyeballed reductions — honestly disclosed, but it is the **default** `q_distribution` of `BinaryIMF`, whose docs say "full Moe+17 model" (oversell) | `moe_di_stefano.py:17-146`, `binary/imf.py:75,127-130` | Caveat in BinaryIMF docstring; never cite this class as MD17 in the paper |
| S10 | Low | `BinaryEnergyBudget` docstring claims Q_resolved "inflated"; measured: **deflated** (0.31 vs 0.50) — physics right, adjective wrong | `binaries/diagnostics.py:222-223` | Fix word |
| S11 | Low | `MassDependentBinaryFraction` steps are a Moe/Raghavan blend (disclosed) — cite as "derived from," not "from," Table 13 | `binary_fraction.py:40-62` | Citation hygiene |
| S12 | Low | King ρ̂(W) catastrophic cancellation: W=1e-6 rel. err 4.8e-4; **W=1e-8 returns negative** (−5.4e-20). Downstream clamps make impact ~nil (tidal-skin only) | `king.py:71-77` | Optional Taylor-series branch below W<1e-4 |
| S13 | Low | r_t non-differentiable in W0 (argmax+clamp, d ξ_t/dW0 ≡ 0) — documented & deferred; fine for shape inference, silently biases W0-inference through tidal observables | `king.py:259-266` | Keep warning in public docs |
| S14 | Low | Headline Engine B numbers partly single-seed: global Q=0.4976 is PRNGKey(0), N=30k, one seed (gate ±0.02); β-dev 0.028 is 4 seeds. Per-component machinery is properly multi-seed (18-seed evidence quoted) | `test_engine_b_physics.py:328-333` | Quote multi-seed mean ± scatter in the paper; archive the 18-seed run |
| S15 | Low | Zero rotation axis: comment claims NaN, code silently no-ops | `rotation.py:49-50` | Align comment/behavior; eager ValueError |
| S16 | Low | Per-group W_j is origin-dependent for subgroups; docstring claims origin independence; callers all pre-center but users won't know | `dynamics/virial.py:201-215` | Fix docstring; center internally |
| S17 | Low | Maschberger `_primitive` breaks at β=1 or α=1 (non-default), undocumented | `imf/maschberger.py` | Document |
| S18 | Low | `find_bound_pairs` softening param doesn't soften E_internal (correct choice, one docstring line missing) | `binaries/diagnostics.py` | Document |

### 4.3 Ambiguous-conventions items
- Table 13 node masses (1.0/3.2/6.7/12/20 M☉) for the spectral-type bins are a progenax interpolation convention, not from the paper — document it.
- Moe Eqs. 17–18 literal text (η fits) verified only against Table 13 anchors, not the equation text — read pp. 41–43 of the PDF before the paper.
- "IGIMF" naming anywhere would be wrong: the implemented model is the Marks+2012/Jeřábková+2018 *environment-dependent stellar IMF*, not galaxy-wide IGIMF integration.

---

## 5. JAX / Differentiability / Performance Audit

### 5.1 Empirically verified gradient coverage (FD cross-checked this session)

| Entry point | Param | Status |
|---|---|---|
| `PlummerProfile.sample_positions` | r_h | ✅ FD ratio 1−1e-11 |
| `build_spatial_ic` (Plummer, end-to-end incl. virial scale, COM) | r_h | ✅ FD ratio 1−1e-10 |
| `KingProfile.from_W0_rc` → positions (through diffrax ODE + CDF) | W0 | ✅ FD ratio 1.000006 (W0≲9.5) |
| `KingVelocityDF.sample_velocities` | W0 | ✅ FD ratio 0.999994 |
| `EFFVelocityDF` (Eddington) / OM | γ / r_a | ✅ FD ratios 1±6e-7 |
| `PowerLawIMF.sample` | α interior | ✅ FD ratio 1−1e-10; **α=1.0 exactly → NaN** (R10) |
| `KeplerElements.to_state` | e ∈ {0.05, 0.9, 0.99, 0.999} | ✅ FD ratios 1.00000000 |
| `q_approx` (naive + fast Morton path) | positions | ✅ finite, nonzero |
| `apply_tidal_truncation` (STE custom_jvp) | r_t | ✅ surrogate ≈ coarse FD of the step |
| Michie DF, LIMEPY tables, MultiComponentCluster | W0/g/w_j/r_a_j | Claimed; repo has AD-vs-FD tests; **spot-check recommended pre-release** (~2 min each) |
| `gaussian_random_field` (experimental) | β | ❌ **NaN** (where-NaN at DC mode; off the production inference path) |

### 5.2 Findings

| # | Sev | Class | Finding | Where |
|---|---|---|---|---|
| J1 | High | correctness | PowerLawIMF α=1 NaN grad (=R10); port `exp_safe` from `differentiable.py:47-54` | `power_law.py` ×4 sites |
| J2 | High | docs-correctness | 16 documented symbols missing from `progenax.__init__`; `IGIMF`/`EnvironmentIMF` don't exist (=R7) | `__init__.py` |
| J3 | Med | correctness (exp.) | GRF NaN grad w.r.t. β at kmag=0 dead branch; standard double-where fix | `gravoturb_fdf/field/field.py:63` |
| J4 | Med | correctness | Traced-W0 `_auto_ode_domain` fallback silently pins ξ_t to the domain edge for W0≳10 under jit/grad — wrong answer, no warning, exactly where the user can't see it. Add an `is_pinned` diagnostic or raise | `king.py:159-166,282` |
| J5 | Low | edge | `virial_scale` returns NaN for T=0 input; sibling `rescale_velocities_to_virial` has a (different) guard — the near-duplicates disagree | `builders.py:256-259` vs `virial.py:251` |
| J6 | Low | perf | `q_approx(method="auto")`: `lax.cond` on a *static* Python N compiles both branches every shape (measured 0.4 s); use Python `if` | `q_approx.py:239,270-275` |
| J7 | Low | perf | `_sample_cluster_arrays` keeps G static under filter_jit → recompile per distinct float G; `king_df.py:178` already shows the `jnp.asarray(G)` fix | `cluster/sampling.py:23-35` |
| J8 | Low | API | `inverse_cdf_draw` zero-weight clamps to grid[-1]; contract enforced by caller convention only (all 5 current sites comply) — add `fallback=` to make it structural | `numerics.py:37-55` |
| J9 | Low | edge | Kepler solver unguarded for e≥1 (silent garbage); document | `kepler.py` |
| J10 | Info | acceptable | O(N²) energy kernels: blocked scan + checkpoint, 32.8 GB→0.12 GB at N=2e4 measured by authors. **Leave them alone** — model implementation |
| J11 | Info | acceptable | Eddington d²ρ/dΨ² via FD on the Ψ grid — empirically clean (γ/r_a grads match FD to 5e-7); no action |
| J12 | Info | strength | **No rejection sampling anywhere** — all samplers exact inverse-CDF (Beta/analytic PPF/tabulated). No acceptance-rate pathology class exists in this codebase |

**Hygiene verified clean:** no `lax.while_loop` (diffrax internals excepted), no numpy in traced paths, no PRNG key reuse, no `.item()`/`float()` on tracers, deliberate documented non-differentiable surfaces all correctly fenced (argsort diagnostics, Stars/TotalMass budgets, r_t).

---

## 6. Architecture and API Review

**Organization: strong.** Module layout maps cleanly onto physics; the 2026-06 redesign removed debt (string dispatch, buggy `populations.py`) instead of shimming it. Docstring sample (10 public symbols): genuinely excellent — units in brackets, equation-numbered references, gradient-behavior notes.

**Findings:**

| # | Sev | Finding | Where |
|---|---|---|---|
| A1 | Med | **Duplicate `VelocityDF` Protocol** — two structurally identical runtime-checkable definitions; will drift | `protocols.py:54-83` vs `kinematics/api.py:53-70` |
| A2 | Med | **Units-policy split**: the VelocityDF protocol bakes in `G=None→DEFAULT_UNITS` on *core* sampling entry points, while `build_spatial_ic` requires G — the package contradicts its own MANDATORY explicit-units policy. (Positive: zero `get_G()`/global-context violations found.) Decide once, pre-1.0 | `protocols.py:68` et al. |
| A3 | Med | KingVelocityDF over-parameterized (S1) — same r_t inconsistency at the API level | `king_df.py:107-115` |
| A4 | Med | Size-limit mandate violated: 8 functions >100 LOC (worst: `energy_sorted_segregation` 166, `two_body_kepler` 141, `env_to_imf_params` 137), 4 files >500 LOC (`limepy_multimass.py` 617, `king.py` 559, `multicomponent.py` 552, `builders.py` 502). Either split the two that genuinely read badly (`env_to_imf_params`, `energy_sorted_segregation`) or amend the rule | various |
| A5 | Low | Ghost cache dirs `src/progenax/gravoturb/` and `src/progenax/cluster/fdf_density/` (only `__pycache__`/mypy debris, git-ignored, **verified absent from the wheel**) — delete; they made this audit suspect stale code. Note `cluster/turbulence.py` is NOT stale (consumed by environment-IMF; correctly documented) | — |
| A6 | Low | Protocols advisory-only: `build_spatial_ic` never isinstance-checks; wrong objects die deep in sampling with AttributeError. One friendly boundary check would cost nothing | `builders.py` |
| A7 | Low | `UniformSphereProfile` exported from `progenax.profiles` but not top level (every other profile is); `CompanionModel.sample` missing return annotation; `ids=None` iff `id_offset==0` asymmetry; `TotalMass` top-up `while True` unbounded | `profiles/__init__.py:46`, `protocols.py:273-282`, `builders.py:317,364-375` |

**Extensibility:** new SpatialProfile/VelocityDF/IMF/CompanionModel = easy (protocols + reusable Eddington/numerics kernels extracted in the right places). New multimass engine = hard but appropriately so. Experimental quarantine verified at the wheel level — no core leakage of `gravoturb_fdf`.

---

## 7. Module-by-Module Completeness Table

| Module | Status | Key evidence / caveats |
|---|---|---|
| Plummer profile | **Release-ready** | Exact algebra; CDF validated; u-clamp negligible & documented |
| King profile | **Release-ready W0≲8; under-verified W0≳9** | R4 CDF-grid core errors (measured); fix n_grid scaling + add core-ρ test |
| EFF profile | **Release-ready** (positions) | Trapezoid CDF gated + order-2 convergence measured; untested r_t/a≫30 |
| Michie profile | **Release-ready with doc caveats** | Closed-form integral re-derived; shares R4 grid pattern |
| Plummer DF (+OM) | **Release-ready** | Beta(3/2,9/2) exact; 63/4 re-derived; Merritt bound enforced |
| King DF | **Release-ready** (after S1 doc fix) | True lowered Maxwellian; unscaled Q=0.5 at W0=7 & 12 |
| EFF DF | **Release-ready with doc caveats** | S2: default γ=3 sub-virial by construction; pin it |
| Michie DF | **Release-ready** | Factorization re-derived; β self-oracle + g=1 LIMEPY identity; one external β point would close it |
| OM anisotropy | **Release-ready** | Full Q-variable inversion; β by construction + a-posteriori |
| Rotation transforms | **Technically functional, scientifically under-verified** | S3: equilibrium-breaking overlay undocumented |
| Multimass Engine A (LIMEPY) | **Release-ready** | Parity vs canonical LIMEPY (pinned SHA, provenance-stamped); per-component Q oracles; implicit-VJP grads vs FD |
| Engine B (Eddington, shared Ψ) | **Release-ready with doc caveats** | A-vs-B anchor; EFF≡Plummer identity; hybrid-sampling edge offset *predicted and gated*; S14 single-seed headline |
| Mass segregation | **Release-ready with doc caveats** | S6 self-consistency caveat missing; algorithm sound, regression-locked |
| Tidal truncation | **Release-ready with doc caveats** | r_J validated vs L1 root-find; S4 equilibrium break undocumented |
| Builders / IC composition | **Release-ready after R6 fix** | Budget semantics unambiguous & tested; G threading audited |
| IMF: PowerLaw/Chabrier/Maschberger/Truncated/IMFParams | **Release-ready** (R10 = 3-line fix; `differentiable.py` is the gold standard) | Paper-verified parameters throughout |
| `BaseIMF` mass-target sampling modes | **Do not release yet** | R5: silent 30% shortfall, zero tests |
| Environment-dependent IMF | **Release-ready with doc caveats** | Erratum-corrected; Eq.-9 deviation documented; kill "IGIMF" naming (R7) |
| Binary orbital mechanics (Kepler, elements, periods) | **Release-ready** | Machine-precision + correct grads to e=0.999 |
| Period/ecc distributions (Öpik/DM91/Sana/Thermal) | **Release-ready** | Paper-verified; α=−1 limit correct |
| MoeDiStefano2017Full / MoeJointOrbit / MoeCompanions | **Do not call "faithful" until R3 fixed** | Table 13 exact; F_twin normalization wrong |
| MoeDiStefano2017 (period-averaged) / MoePeriod | **Plausible, needs stronger validation** | Honest approximations; S7/S9 caveats |
| Binary connector + diagnostics | **Release-ready with doc caveats** | COM exact; S10 wording; softening scope note |
| Binary population synthesis (`build_binary_cluster`) | **Release-ready** | Primary-IMF semantics preemptively documented; budgets tested |
| CW04 Q exact / q_approx | **Release-ready** / **needs stronger validation** | Table 1 anchors / S5 single-point calibration |
| Analytical test cases | **Release-ready** | Digit-checked; self-validating; honest ephemeris provenance |
| gravoturb_fdf (experimental) | **Experimental only** (correct labeling) | J3 GRF grad NaN off-path; AC tolerances are honest smoke reductions; full-strength AC run not gated anywhere |
| Docs & examples | **Do not release yet** | R7: README teaches deleted API; phantom classes; broken install commands |
| CI / packaging | **Do not release yet** | R1/R2: red CI; uninstallable wheel; no LICENSE |

---

## 8. Test Suite Audit

**What the suite genuinely proves** (best-in-class for research code): analytic Plummer σ²(r); King 1966 Table II; LIMEPY parity against the canonical GZ15 code at pinned SHA ef2a479 with provenance embedded in the .npz (numpy/scipy/python versions + config — verified by reading `provenance.npy`); measured-first frozen gates with written noise budgets (e.g. `conftest.py:86-89` documents the 40-seed calibration that tightened VIRIAL_RATIO 0.20→0.05); AD-vs-FD gradient tests at rel<1e-5; CW04/Allison/Küpper anchors. **The 5 suspiciously-tight tolerances were checked: none are seed-hacks** — each has a written derivation.

**What it does not prove / weaknesses:**

| # | Sev | Finding |
|---|---|---|
| T1 | **Critical** | CI red on main (=R1); ≥5 runs failed at dependency sync; nothing enforced anywhere |
| T2 | **Critical** | Headline validations slow-marked and never run in CI (=R8): LIMEPY parity, multimass equilibrium, A-vs-B anchor, Engine B AD-vs-FD — 47/1192 tests including the advertised trust anchors |
| T3 | High | No version matrix: Python 3.13 only vs `requires-python>=3.10`; JAX floor 0.4.20 never tested (lock has 0.6.2) |
| T4 | High | LIMEPY parity **skips silently** if the cache is absent (`test_limepy_reference_parity.py:62-64`) — a packaging mistake converts the strongest gate into a silent pass. Make it fail in strict/release mode |
| T5 | High | Stale counts: `tests/README.md` says 874 tests; CLAUDE.md says 1163; measured 1192 collected / 1145 not-slow |
| T6 | Med | Gradient tests bimodal: strong FD-anchored ones exist, but 7/9 in `test_jax_compatibility.py` assert sign/finiteness only — a wrong-by-2× gradient passes |
| T7 | Med | Integration tier ~2/3 smoke/tautology (COM-sums-to-zero true by construction). The dropped-G regression IS real (verified it catches the catastrophic case) but doesn't test G-*scaling* |
| T8 | Med | `tests/unit/test_numerics.py:15-61` bit-for-bit mirrors the implementation — refactoring guard, not correctness; needs an analytic-integral oracle |
| T9 | Med | Single-seed stochastic tests (EFF ordering, Λ_MSR, multimass) — **probed at fresh seeds this session: all pass with large margin**, so flakiness is a hygiene smell not a hazard; convert to 3-seed means |
| T10 | Med | Untested edges: EFF γ→2, e≥0.95 distributions, IMF α≈1, N∈{0,1}, W0→0; all four `BaseIMF` mass-mode samplers (=R5) |
| T11 | Low | ~11 `scripts/validate_*.py` never invoked by pytest — the project's own Definition-of-Complete validation tier enforced by convention only; `benchmark_batch_a.json` referenced by nothing |

**Loosest tolerances (for tightening):** `2.0<Λ<12.0` (mass-seg); `±0.20` on α₃ (environment); `0.40<mean<0.80` mean-mass; `0.05<m_peak<0.4`. Defensible given finite-N, but several could be 2–3× tighter with multi-seed means.

**Recommended pre-release test matrix:**
1. Fix CI (jaxstroviz checkout or de-lock) — prerequisite for everything.
2. PR gate: current 3-tier `-m "not slow"` sharding (keep).
3. **Nightly/release-tag full-physics lane**: `pytest tests -m slow -n auto` + full-strength `python -m gravoturb_fdf.validation.acceptance`; `PROGENAX_STRICT_REFS=1` makes reference-cache absence a failure.
4. Version matrix (nightly, fast suite): {3.10, 3.13} × {JAX floor, lock}.
5. Gradient tier: one AD-vs-FD per public differentiable module (add Michie DF, LIMEPY tables, resolve_binary_components); rename finite-only tests "grad_is_finite".
6. Statistical: 3-seed means for the 5 single-seed ordering tests; one explicit N-convergence test (Q error ∝ 1/√N).
7. `jax.jit(jax.grad(...))` composition smoke for the two builders.
8. CPU/GPU: no runner — document as untested + manual workflow_dispatch job.
9. **Docs-example smoke**: execute README/CLAUDE.md quickstart snippets in pytest — would have caught R7 automatically.
10. Wheel-build + clean-venv `import progenax` + `import progenax.diagnostics` job — would have caught R9.

---

## 9. Documentation and Packaging Review

| # | Sev | Finding | Blocks |
|---|---|---|---|
| D1 | **Critical** | No LICENSE file; wheel METADATA missing license/author/classifiers/URLs; README's "MIT" is an unenforceable claim (=R2) | YES |
| D2 | **Critical** | Not pip-installable: `Requires-Dist: jaxstro` unpinned/unpublished; `[tool.uv.sources]` doesn't ship; `uv.lock` drags an editable jaxstroviz→fluxax→(circular) progenax graph | YES |
| D3 | High | README quickstart = deleted API; flagship MultiComponentCluster absent; stale counts (815 vs 1192 tests, 14.7k vs 17.7k LOC); `apply_osipkov_merritt()` doesn't exist; "3 protocols" (there are 9); Michie/LIMEPY missing from tables (=R7) | YES |
| D4 | High | `installation.md`: nonexistent `[all]`/`[io]`/`[viz]`/`[ml]` extras (those are *gravax's*); conda instructions contradicting uv mandate; monorepo URL vs actual split repo; smoke test **reuses the same PRNG key** for positions and velocities — teaches the anti-pattern in example #1 | YES |
| D5 | High | `docs/website/20-architecture/ic-redesign-history.md:110` claims "`pip install progenax-legacy` still works" — **fabrication**, falsifiable in 10 s | YES |
| D6 | Med | Three documents, three different test counts (README 815 / CLAUDE.md 1163 / STATUS.md 1189) — generate or stop quoting |  |
| D7 | Med | No CHANGELOG; no upper bounds or tested-versions table for jax/equinox (floor 0.4.20 vs lock 0.6.2) |  |
| D8 | Low | STATUS.md = 5,000-word internal dev log with SHAs and .claude-work paths at repo root — move long-form history under docs/ for public consumption |  |
| D9 | Low | Stale `PLUMMER_FIXES.md (v0.3.0)` docstring reference (file doesn't exist) | |

**Positive (specific):** `docs/website/00-getting-started/science-capabilities.md` is exemplary honest-scope writing — every claim traced to a named validation test. Docstring quality across the 10-symbol sample is excellent.

**Sibling-dependency strategy (the decision behind D2):**
- **Option A (recommended): publish `jaxstro` to PyPI first.** Progenax core needs only `jaxstro.units` + `jaxstro.jaxconfig` — tiny, stable surface. Pin `jaxstro>=0.1,<0.2`. Remove jaxstroviz from progenax's lock graph entirely (validation-side only; it's what broke CI).
- Option B: vendor the ~2 modules into `progenax._vendor` — fastest standalone wheel, costs ecosystem coherence (shared UnitSystem objects with gravax).
- Option C: git-URL dep — fine for a GitHub soft launch; PyPI rejects URL deps.

---

## 10. Pre-Release Action Plan

### Blockers (must do)
1. **Fix CI** (R1): check out jaxstroviz(+fluxax) siblings in all 4 jobs, or better, remove jaxstroviz from progenax's dependency graph. Get main green.
2. **Add the nightly/release-tag slow lane** (R8): LIMEPY parity, A-vs-B, multimass equilibrium, full-strength gravoturb acceptance; strict reference-cache mode.
3. **Fix Moe F_twin normalization** (R3): unnormalized p ∝ p_pl + (ft/(1−ft))·I_B·U[0.95,1], renormalize over [q_min,1]; regression test: excess-twin/(q>0.3) = Table 13 F_twin at grid nodes.
4. **Fix high-W0 CDF grids** (R4): scale n_grid with ξ_t (or log-stretched grid) in king/michie/eff; add sampled core-density tests at W0 ∈ {9, 12, 15}.
5. **Fix or fence `sample_fixed_n`** (R5): reachability guard + tests for all 4 modes; soften "exactly".
6. **Fix `compute_stellar_radii`** (R6): adopt a cited fit (Demircan & Kahraman 1991 or Tout+1996 ZAMS); literature-anchored test.
7. **Port `exp_safe` to PowerLawIMF** (R10): 3 sites; grad test at α=1.0.
8. **README + installation rewrite** (R7/D3/D4/D5): current API only, MultiComponentCluster featured, real install commands, kill `progenax-legacy` claim, fix reused-key example, fix phantom IGIMF/EnvironmentIMF everywhere (incl. both CLAUDE.md files), reconcile `__init__` exports with documented API (decide: export the 14 IMF symbols or fix docs).
9. **Packaging** (R2/D1): LICENSE, metadata, sibling-dep decision (Option A recommended), `[diagnostics]` extra or lazy numpy/scipy imports (R9), CHANGELOG, version tag.
10. **Wheel-smoke CI job**: build wheel, clean venv, import progenax + progenax.diagnostics, run README snippet.

### Important, non-blocking
- S1 King r_t consistency guard; S2 pin Q(γ=3); S3/S4/S6 equilibrium-caveat docstrings; J4 traced-W0 pinning diagnostic; J5 virial_scale guard + dedupe with rescale_velocities_to_virial; A1 dedupe VelocityDF protocol; A2 units-policy decision; A5 delete ghost dirs; T6 upgrade finite-only grad tests; T9 3-seed conversions; S5 q_approx calibration sweep; M-dwarf multiplicity + DM91 constants PDF check; version matrix CI.

### Pre-paper verification (cheap insurance)
- Eyeball King 1966 Table II once more against `test_king_physics.py:267,397`.
- Read Moe Eqs. 17–18 text (pp. 41–43) for η fits + 3–7 M☉ interpolation prescription.
- Regenerate the LIMEPY reference cache once in the pinned env and diff.
- CW04 Table 1 + area definitions against the actual PDF.
- GZ15 §2.2 m̄ convention + η default direct read.
- Archive the 18-seed Engine B run behind the quoted 0.4953/0.5007 numbers.
- Spot FD-check Michie/LIMEPY/MultiComponentCluster gradients (3 × 2-min harness runs).
- One long N-body run per DF (King/Michie/EFF/Engine B) in gravax over ~10 t_cross asserting profile stability — the gold-standard equilibrium demonstration for the paper.

---

## 11. Methods Paper Plan

**Title options:**
1. "progenax: Differentiable Initial Conditions and Population Synthesis for Star Cluster Modeling and Inference"
2. "Differentiable Equilibrium Initial Conditions for Collisional Stellar Dynamics with progenax"
3. "progenax: A Composable, Differentiable Engine for Star Cluster Initial Conditions, Binary Populations, and Gradient-Based Inference"

**Abstract framing:** IC generation has been treated as a fire-and-forget preprocessing step; progenax reframes it as a differentiable model component, enabling gradients of any downstream observable with respect to structural (r_h, W0, c, r_t, r_a), population (IMF slopes, binary fractions, Moe parameters), and environmental parameters — while matching or exceeding the physical fidelity of standard generators (true lowered-Maxwellian/Eddington equilibria, no virial rescaling; LIMEPY-parity multimass models).

**Claims currently justified (cite the named tests):**
- True detailed equilibria sampled without external virial rescale (Q=0.5 unscaled: Plummer/King/Michie/EFF γ=5/Engine A/Engine B).
- Multimass LIMEPY-family models with reference-code parity at 1e-8–5e-4.
- Two independent equilibrium engines agreeing on King (A-vs-B σ/KS at 2–3e-4).
- OM anisotropy by construction with a-posteriori β(r) verification.
- End-to-end differentiability with AD-vs-FD verification through ODE solves and Eddington inversions.
- Faithful Kepler machinery to e→1 with correct gradients.
- Moe Table 13 transcription exactness (after R3 fix, the full P–q–e claim).

**Claims requiring more validation before being made:**
- "Faithful Moe & Di Stefano 2017" — blocked on R3 + Eqs. 17–18 verification.
- High-concentration (W0>9) sampled structure — blocked on R4.
- Long-term N-body stationarity (add the gravax 10-t_cross runs).
- q_approx as a general-purpose Q estimator (S5 regime limits).
- Anything about gravoturb_fdf beyond "experimental, repo-only" unless promoting it deliberately.

**Section outline:**
1. Introduction — ICs as inference bottleneck; differentiable-programming context.
2. Architecture — protocols, Equinox PyTrees, explicit units, composability (profile × DF × IMF × binaries).
3. Spatial profiles & velocity DFs — Plummer/King/Michie/EFF; Eddington inversion; OM; rotation overlays (with honest equilibrium caveats).
4. Multi-component clusters — Engine A (GZ15 multimass) and Engine B (density-defined shared-Ψ Eddington); mass segregation; tidal truncation.
5. Stellar populations — IMFs (paper-grounded constants incl. the Jeřábková Eq.-9 correction), binary populations (Moe P–q–e), the connector.
6. Differentiability — fixed-iteration solvers, double-where guards, STE truncation, non-differentiable surfaces (honest table).
7. Validation — external-reference parity (King Table II, LIMEPY, CW04), equilibrium gates, AD-vs-FD, N-body stationarity.
8. Inference demonstrations (§12 below).
9. Performance & scaling; 10. Discussion/roadmap (stellax/startrax/fluxax coupling, gravoturb_fdf as future work).

**Key figures:** (i) architecture/composability diagram; (ii) ρ(r), σ(r), β(r) sampled-vs-analytic 4-panel per DF family; (iii) c(W0) vs King 1966 + LIMEPY parity residuals; (iv) Engine A-vs-B overlay; (v) gradient-verification panel (AD vs FD across parameters/entry points); (vi) posterior corners from the inference demos; (vii) Moe P–q–e joint recovery vs Table 13; (viii) N-body stationarity over 10 t_cross.

**Key tables:** validated formula↔reference provenance table; gradient coverage map (from §5.1 — publish it); equilibrium Q table (the existing CLAUDE.md table, multi-seed); timing/compile-cost table.

---

## 12. Recommended Science and Inference Demonstrations

| Demo | Demonstrates | Validates | Figure | Difficulty | Priority |
|---|---|---|---|---|---|
| 1. Recover (r_h) of a Plummer sphere from particle positions via gradient descent / HMC | The core differentiable-inference loop; bias/variance vs N | Plummer profile+DF, build_spatial_ic grads | posterior + convergence trace | Easy (exists in spirit in tests) | **Release-critical** |
| 2. Recover (W0, r_c) of a King model — gradients through the diffrax ODE | The hard differentiability claim | King profile/DF, implicit grads | corner plot | Easy–moderate | **Release-critical** |
| 3. Model comparison: fit Plummer vs King vs EFF to one synthetic cluster, gradient-based MAP + evidence proxy | Composability as scientific capability | all three families | profile residual panel | Moderate | Paper-enhancing |
| 4. IMF slope α from a synthetic resolved population (incl. α near break masses) | Differentiable population inference | IMF subsystem, IMFParams | posterior vs truth across α grid | Easy | **Release-critical** |
| 5. Unresolved-binary bias on inferred IMF slope (with/without MoeCompanions) | A real survey systematic, only easy with this tool | binary connector + Moe model | Δα vs binary fraction | Moderate | Paper-enhancing (flagship science result) |
| 6. Binary-population inference: recover F_twin/γ/period params from resolved binaries | Moe machinery post-R3-fix | MoeDiStefano2017Full | corner | Moderate–hard | Paper-enhancing |
| 7. Concentration + anisotropy joint (W0, r_a) recovery from projected kinematics | OM machinery; degeneracy structure | Engine B / OM DFs | 2D posterior with β(r) overlay | Moderate | Paper-enhancing |
| 8. Two-component mass-segregated cluster: recover δ (equipartition parameter) | Engine A multimass | MultiComponentCluster | δ posterior | Moderate | Paper-enhancing |
| 9. IC-sensitivity through N-body evolution: d(final r_h)/d(initial Q) via gravax SymplecticIntegrator | Ecosystem differentiability end-to-end | gravax coupling | gradient flow vs FD through evolution | Hard (gravax-side) | Paper-enhancing |
| 10. Differentiable CMD forward model via fluxax (flux_from_mass) → IMF+binary posterior from photometry | The LSST-facing pitch | fluxax coupling | synthetic CMD + posterior | Hard | Paper-enhancing, defer if fluxax not ready |
| 11. Substructure: Q-parameter trajectories of collapsing subvirial clusters; q_approx as differentiable summary | diagnostics | CW04 Q + q_approx (post-S5) | Q(t) tracks | Easy | Paper-enhancing |
| 12. gravoturb_fdf β-headline 2D inference | The experimental arc | gravoturb_fdf | SBC + posterior | Already in progress | **Separate paper**, not this one |

Minimum release-critical set: demos 1, 2, 4 — each is a notebook-sized exercise on machinery already validated, and together they substantiate the paper's central claim ("differentiable inference workflows impossible with traditional IC generators").

---

## 13. Final Verdict

**Beta scientific software, ~2 weeks of mechanical work from Release Candidate.** The physics core would survive expert review today — the audit *failed to find a single error* in the headline equilibrium machinery (Plummer/King/Michie/EFF/Engine A/Engine B), and the validation culture (external references, pinned provenance, measured-first gates, AD-vs-FD) is better than most published astronomy packages. What would embarrass you publicly is entirely at the boundary: a red CI, an uninstallable wheel, a README teaching deleted APIs, two real but fixable science bugs in the binaries/IMF periphery (R3, R5), one quantified numerical defect at high W0 (R4), and a factual error in an auxiliary stellar mass–radius relation (R6).

### Top 10 actions, in order
1. Fix CI (jaxstroviz checkout / de-lock) and get main green — nothing else is enforced until then.
2. Add the nightly/release-tag slow-test lane so the advertised trust anchors actually run.
3. Fix the Moe F_twin normalization (R3) + Table-13 regression test.
4. Fix high-W0 CDF grids (R4) + core-density tests at W0 ∈ {9,12,15}.
5. Guard/fix/test the `BaseIMF` mass-target sampling modes (R5).
6. Swap the `compute_stellar_radii` exponents with a cited fit (R6).
7. Port `exp_safe` to PowerLawIMF (R10) + α=1 gradient test.
8. Rewrite README + installation docs; reconcile `__init__` exports vs documented API; kill IGIMF/EnvironmentIMF and `progenax-legacy` claims (R7/D5).
9. Packaging: LICENSE + metadata + jaxstro-first publication decision + `[diagnostics]` dependency story (R2/R9/D1).
10. Pre-paper verification sweep: Moe Eqs. 17–18 PDF read, CW04/GZ15 PDF checks, LIMEPY cache regen, Michie/multimass gradient spot-FD, and one 10-t_cross gravax stationarity run per DF.
