# Multimass LIMEPY Hardening — Implementation Plan (validation + docs)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> to implement this plan task-by-task (fresh subagent per task + independent
> code-review after each). FIRST verify the plan (see the handoff prompt).

**Goal:** Establish progenax's Engine A multimass equilibrium as a **faithful,
differentiable, validated reimplementation of canonical LIMEPY** — via (1) a
direct ours-vs-reference-LIMEPY cross-validation harness, (2) a zero-new-parameter
validation that our model reproduces Bianchini2016's equipartition-saturation σ(m)
with the *derived* m_eq, and (3) methods/theory documentation. **No released-core
behavior change** (no new model parameters; `meq`/`zeta` are DEFERRED).

**Architecture:** progenax's `MultiComponentCluster` (Engine A) solves the GZ15
multimass lowered-isothermal equilibrium (`s_j=s·μ_j^{−δ}`, `μ_j=m_j/m̄`,
`ρ̂_j` via `μ_j^{2δ}φ̂`, √-update MF iteration in `find_alpha_for_masses`). The
reference is the canonical numpy/scipy `limepy` at
`~/projects/jaxstro-dev/ref-repos/limepy/`. The harness builds the SAME physical
model in both and compares scale-invariant shapes, reconciling the central- vs
global-`m̄` convention via Peuten2017 eqs 8–9.

**Tech stack:** JAX (progenax side), numpy/scipy (reference, import-only via
`sys.path`), `scripts/_plotstyle.py` figures, pytest (skip-if-reference-absent).

**Design doc (read first):** `docs/plans/2026-06-11-multimass-limepy-hardening-design.md`
— it carries the VERIFIED equations (GZ15 eqs 24–26/29; Peuten eqs 3–5, 8–9;
Bianchini eqs 3–4, App. A2–A3 with the derived `m_eq=m̄(g+5/2)(g+7/2)/φ̂₀`).
Per-paper notes: `docs/website/99-bibliography/per-paper/{gieles-zocchi-2015,
peuten-2017,bianchini-2016}.md` (already written, equation-verified).

**Gates:** FAST gate (1121 not-slow) + FULL gate (1163) per `progenax/CLAUDE.md`
— must stay green (the new parity test is skip-if-absent, so it never breaks the
released gate). Demo/validation scripts exit-nonzero on gate failure.

**Git:** branch `feat/multimass-limepy-validation` off `main`; commit per task;
**NO push/merge without Anna's explicit go.**

**Hard rules:** JAX-native (progenax side). Equations are VERIFIED — do not
re-derive from memory; cite the design doc / PDFs. **Never weaken a gate to pass**
— if ours-vs-reference disagrees beyond the (measured) tolerance, that is a REAL
finding: STOP and report (do not loosen). No released-core behavior change. Every
figure via `_plotstyle`. HITL: Anna approves at each review checkpoint.

---

## Task 1: Reference-LIMEPY cross-validation harness (single-mass + multimass, meq=0)

**Files:**
- Create `scripts/validate_limepy_reference.py` (gated CLI, `validate_*` house style)
- Create `tests/validation/test_limepy_reference_parity.py` (skip-if-absent)

**Step 1: Study both APIs (no code yet).**
- Reference: read `ref-repos/limepy/limepy/limepy.py` — the `LIMEPY(phi0, g, **kw)`
  constructor and the OUTPUT attributes after construction (`.r`, `.rho`, `.v2`,
  `.rhat`, `.mc`, `.Mj`, `.rt`, `.rh`, `.rv`, per-component arrays for multimass:
  `.rhoj`, `.v2j` or similar — CONFIRM the exact attribute names by reading the
  `_poisson`/`_setup_phi`/density code; note which are per-component).
- Ours: read `src/progenax/cluster/multicomponent.py` (`from_components`,
  `from_imf`) and `src/progenax/profiles/limepy_multimass.py` — confirm how to get
  per-component `α_j`, `ρ_j(r)`, `σ_j(r)` / `v²_j(r)`, and the radial grid out of a
  built `MultiComponentCluster` (the σ_j oracle recipe is in
  `tests/validation/test_multimass_equilibrium_physics.py:70-86`).
- **Confirm the reference's `m̄` mode** (central-density-weighted vs global): read
  `_set_mass_function_variables`/`mmean` in the reference. If it differs from ours
  (central, GZ15 eq 26), the comparison MUST translate `W₀`/`r̂_a` via Peuten eqs
  8–9 (`W₀*=W₀(m̄/m̄*)^{2δ}`, `r̂_a*=r̂_a(m̄/m̄*)^{η+δ}`) — design doc §Equation
  verification. Decide: translate, or set the reference to central mode if it
  supports it. Document the choice in the script header.

**Step 2: Write the harness.** For each config, build the model in BOTH codes and
compare **scale-invariant** quantities (convention-light):
- converged per-component central-density ratios `α_j` (Σ=1);
- density shape `ρ_j(r)/ρ_j(0)` on a shared `r/r_scale` grid (interpolate);
- dispersion shape `σ_j(r)/σ_j(0)` (or `v²_j`);
- per-component mass fractions at convergence;
- global structure: concentration `c=log(r_t/r_c)`, `r_h`.

**Configs** (start simplest):
1. single-mass `nmbin=1`, g=1 (≡ King), iso — sanity (should match to solver tol);
2. single-mass g=1.5, iso;
3. 2-component (e.g. m=[0.3,1.0], M-fractions [0.7,0.3]), δ=0.5, g=1, iso;
4. the B2 4-component (binned Maschberger α=2.3, N_COMP=4, M_RANGE=(0.1,20)), g=1;
5. anisotropic: config 3 with finite r_a, η=0 and η=0.5.

**Step 3: MEASURE first, then set the gate.** Run once, print the max shape
deviation per config; set the PASS threshold at a defensible level (target ~1%,
both solve the same ODE — but set it from the measured numbers, honestly). Print
an expected-vs-measured table; `sys.exit(1)` if any config exceeds its gate.
**If a config disagrees badly (≫1%), STOP and report** — it is a real
parity finding (likely an m̄-convention or unit-mapping issue), not a gate to
loosen.

**Step 4: Figure** `validation/plots/limepy_reference_parity.png` (+pdf) via
`_plotstyle`: ρ_j(r) and σ_j(r) ours-vs-ref overlays + residual strips for a
representative multimass config.

**Step 5: Parity test** `tests/validation/test_limepy_reference_parity.py`:
`importorskip` the reference (`sys.path` insert to `ref-repos/limepy`; skip if not
importable) — assert the same shape agreement for 2–3 configs at the gate
tolerance. Mark `@pytest.mark.slow` if needed.

**Step 6: Run + commit.**
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_limepy_reference.py; echo "exit: $?"
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_limepy_reference_parity.py -q
git add scripts/validate_limepy_reference.py tests/validation/test_limepy_reference_parity.py
git commit -m "validation(multimass): ours-vs-reference-LIMEPY parity harness (faithful published LIMEPY)"
```

---

## Task 2: Derived-m_eq equipartition-saturation validation (zero new params)

**Files:** extend `scripts/validate_limepy_reference.py` (a new `--meq` section or
a sibling `scripts/validate_equipartition_saturation.py` — implementer's call;
keep house style).

**Step 1:** From a built progenax multimass model (a fine mass spectrum, e.g.
20 log-spaced bins over [0.1, 1.0] M_⊙ as in Bianchini Fig. 9; g=1.5, a few Ŵ₀),
extract the per-component **central** 1-D dispersion `σ_{1d,j0}` (the σ_j oracle
at r→0). This is the dimensionless central dispersion of each mass bin.

**Step 2:** Fit `σ(m)` to Bianchini eq 3 (`σ₀exp(−m/2m_eq)` low-mass,
`m^{−1/2}` high-mass) → recovered `m̂_eq`. Compute the **derived**
`m_eq = m̄(g+5/2)(g+7/2)/φ̂₀` (design doc; Bianchini App. A2↔A3). Note `m̄` is the
central-density-weighted mean (our `bar_m`), `φ̂₀=W₀`.

**Step 3: MEASURE then gate.** Assert recovered `m̂_eq` ≈ derived `m_eq` (set
tolerance from the measured agreement; the App.-A expansion is leading-order, so
expect a modest tolerance — report the actual % and explain). Also verify the
qualitative shape: σ(m) flat at low mass, → m^{−1/2} at the high-mass end. If the
agreement is poor, STOP and report (it may indicate a σ-oracle or m̄ definition
issue — a real finding).

**Step 4: Figure** `validation/plots/equipartition_saturation.png`: our σ(m)
points + the Bianchini eq-3 curve with derived m_eq + the fitted m_eq; annotate
both m_eq values. Print an expected-vs-measured table; exit-nonzero on gate fail.

**Step 5: Commit.**
```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/validate_equipartition_saturation.py; echo "exit: $?"
git add scripts/validate_equipartition_saturation.py validation/plots/  # if plots tracked; they are gitignored -> just the script
git commit -m "validation(multimass): derived m_eq reproduces Bianchini2016 sigma(m) saturation (zero new params)"
```

---

## Task 3: Theory/methods page + wire notes + docstring fixes + build-verify

**Files:**
- Create `docs/website/10-theory/.../multimass-equipartition.md` (or extend the
  existing `lowered-model-family.md`/multimass theory page — find the right home).
- Modify `docs/website/myst.yml` (nav: add the 2 new per-paper notes + the theory
  page).
- Modify `src/progenax/profiles/limepy_multimass.py` (docstring only): the
  `find_alpha_for_masses` "eigenvalue" wording → also name the GZ15 √-update MF
  iteration; ensure NO docstring repeats a "meq = GZ15 eq 24" citation.

**Step 1:** Theory page in paper-section prose: the GZ15 multimass DF, the δ
equipartition ansatz + the honest sub-equipartition physics, the Bianchini σ(m)
saturation with the **derived** m_eq (cite the validation figure + numbers from
Task 2), the m̄ conventions (Peuten eqs 8–9), and the explicit statement that
`meq`/`zeta` are deferred code-knobs (not GZ15). Every number from the scripts'
printed tables.

**Step 2:** Wire `bianchini-2016.md`, `peuten-2017.md`, and the theory page into
`myst.yml`. Build-verify:
```bash
# from docs/website (or wherever myst runs in this repo)
<the repo's myst build command>  # exit 0, 0 warnings; record page count
```
Fix any broken cross-refs (`{eq}` labels, inter-note links).

**Step 3:** The docstring-only source edit (no behavior change). Run the FAST gate
to confirm nothing moved.

**Step 4: Commit.**
```bash
git add docs/website/ src/progenax/profiles/limepy_multimass.py
git commit -m "docs(theory): multimass equipartition-saturation page + wire notes + docstring provenance fix"
```

---

## Task 4: Close-out

**Step 1: FULL released-core gate** (must be unchanged — no behavior change):
```bash
XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
  env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
```
Expected: the prior count + the new skip-if-present parity test. Record.

**Step 2:** Completion doc
`.claude-work/MULTIMASS_LIMEPY_VALIDATION_COMPLETE.md`: the reference-parity table
(per config, measured deviation vs gate), the derived-vs-fitted m_eq result, the
figures, the equation-provenance summary (what's GZ15 vs Peuten vs Bianchini vs
code-only), myst build result, files changed, and the DEFERRED meq/zeta record.

**Step 3:** Update `STATUS.md` (`next:` = meq/zeta knobs OR Batch B Tasks 6–9 OR
continuous-mass research — Anna's call). Capture a brain event.

**Step 4: STOP** — present the completion doc + evidence to Anna. Merge/push only
on her explicit go.

---

## Out of scope (do not drift)

`meq`/`zeta` implementation (DEFERRED — knobs, not the saturation); continuous-mass
DF f(E,m); Engine B; changing the √-update or N_COMP; the B2 demo science
(Tasks 6–9, demo-only, separate).
