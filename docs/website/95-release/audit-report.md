---
title: Release-readiness audit report
subtitle: Ten dimensions, verified with first-hand evidence
description: >-
  Dimension-by-dimension release-readiness audit of progenax at commit 24cb6b9
  (2026-06-17): tests, physics, gradients, API, architecture, docs, packaging,
  CI, examples, and release-scope decisions — each with the command, the number,
  and the verdict.
---

(release-audit-report)=
# Release-readiness audit report

This report verifies, dimension by dimension, whether progenax is ready for a
public release. The rule throughout: **no claim without evidence produced this
session.** Where a value could not be independently verified, it is marked
`UNVERIFIED` rather than asserted — a partial check that says so is worth more
than a confident guess.

**Audited:** commit `24cb6b9` on `main`, 2026-06-17. **Method:** a full local
test gate, the validation scripts, the gradient-audit run, a wheel build +
clean-venv install, a documentation build, and a citation sweep against the
source PDFs in `docs/core-papers/` — parallelised across read-only sub-audits and
synthesised here.

Severity legend: ✅ ready · ⚠️ should fix before a *polished public* launch ·
❌ blocks release.

(audit-summary-table)=
## Summary

| Dimension | Verdict | Headline evidence |
|---|---|---|
| [1. Correctness & tests](#audit-d1) | ✅ | 1553 passed / 3 skipped / 1 xfailed, exit 0; coverage 95.96% |
| [2. Scientific validation](#audit-d2) | ✅ (1 ⚠️) | All physics anchors reproduce; one stale validation *script* |
| [3. Gradient integrity](#audit-d3) | ✅ | 97 AD-vs-FD cases, 0 hazards; all 4 registries full |
| [4. Public API & naming](#audit-d4) | ⚠️ | 122 symbols; ~27 docstring gaps; one deferred units decision |
| [5. Architecture & JAX-native](#audit-d5) | ✅ | No hardcoded G; numpy/scipy lazy; `while_loop` correctly fenced |
| [6. Docs site](#audit-d6) | ✅ (2 ⚠️) | `myst build` 172 pages, 0 warnings; no fabricated content |
| [7. Packaging & metadata](#audit-d7) | ✅ | Wheel clean; `twine` passes; clean-venv import works |
| [8. Reproducibility & CI](#audit-d8) | ⚠️ | CI config excellent but **disabled**; demos exit 0 |
| [9. README & examples](#audit-d9) | ✅ | 9 README code blocks execute in CI |
| [10. Release-scope decisions](#audit-d10) | — | OED demos → informax; `gravoturb_fdf` experimental |

(audit-d1)=
## 1. Correctness & tests — ✅

The full released-core gate (`tests/unit tests/integration tests/validation -n
auto`, including the `@slow` trust anchors) ran to completion:

```text
1553 passed, 3 skipped, 1 xfailed in 1305.40s   (exit 0)
```

(The 21-minute wall time reflects four parallel audit agents contending for the
CPU; the inner-loop fast gate is ~4 minutes.) Line coverage is **95.96%** against
a stated floor of 90%, and that number is *not* stale: `git diff` shows
`src/progenax/` is unchanged since the commit the coverage was measured at, so the
src-based staleness gate correctly treats it as fresh.

Crucially, **no skip or xfail hides a failure** — each was read and reproduced
with `-rsxX`:

| Test | Status | Reason | Action |
|---|---|---|---|
| `test_strict_refs.py` | SKIP | "strict reference mode not requested" — gated on `PROGENAX_STRICT_REFS`; unset locally, **set =1 in the release-tag CI lane**, where it asserts the LIMEPY reference cache is present (so parity can't silently no-op). | None — by design (R8/T4). |
| `test_limepy_tables.py::test_table_solve_is_faster` | SKIP | Skips under xdist/coverage (`PYTEST_XDIST_WORKER` / `sys.gettrace()`); timing ratios are meaningless under instrumentation. **Runs + asserts ~5–10× serially.** | None — by design. |
| `test_dispersion_physics.py::test_plummer_om_jeans_vs_analytic` | ~~SKIP~~ → **now passing** | Originally **deferred rather than fabricated**. **Closed during this audit (2026-06-17):** the closed form $\sigma_r^2 = GM(a^2+3r^2+2r_a^2)/[12(r^2+r_a^2)\sqrt{a^2+r^2}]$ was *derived* from the OM anisotropic Jeans equation (BT §4.8.3) — not transcribed (Plummer is not a Carollo+1995 γ-model) — and verified (isotropic limit → Dejonghe to 2e-16; numerical OM Jeans to ~1e-6). | Done. |
| `test_dispersion.py::test_grad_jeans_michie_wrt_W0` | ~~XFAIL (strict)~~ → **resolved 2026-06-17** | Flagged at audit as a $\sim 5\times10^{-3}$ FD-inconsistent Michie $\partial\sigma_r/\partial W_0$ and *attributed to the ODE solver* (ADR 0009). **The #4 arc overturned that:** the AD gradient was always **correct** (a central FD *converges* to it as the step shrinks); the $\sim 5\times10^{-3}$ was a coarse-FD artifact of a **C⁰ piecewise-linear back-interpolation** in `_sigma_r2_from_tables` — as $r_t(W_0)$ moved the master $s$-grid nodes, the interp's slope jumped for a fixed query radius, kinking $\partial\sigma_r/\partial W_0$. A **C¹ PCHIP back-interp** (ADR 0016, supersedes 0009) removes it: rel $5.07\times10^{-3}\to3.5\times10^{-4}$. | Done — gate flipped xfail→pass on merit; released-core now carries **0 xfail**. |

The two design skips (env-gated guard, instrumentation-aware benchmark) are
*correct* behaviour, not debt. The OM-Plummer skip was the no-guessed-formula rule
in action — and was **closed during this audit** by deriving and verifying the
oracle rather than asserting it, so the released-core skip count drops from 3 to 2.
The Michie-$W_0$ xfail was **closed by the follow-on #4 arc (2026-06-17, ADR 0016)**:
it was never an incorrect gradient — the AD value is correct (a finite difference
converges to it as the step shrinks) — but a C⁰ back-interpolation kink that a coarse
fixed-step FD straddled; a C¹ PCHIP back-interp fixed it, so released-core now carries
**0 xfail**. (The audit's original "genuinely incorrect / ODE-solver" reading of this
row was a misdiagnosis, corrected by the arc's discriminating experiments.)

The experimental `gravoturb_fdf` suite (repo-only) ran **343 passed, 8 deselected,
exit 0** — green as documented.

(audit-d2)=
## 2. Scientific validation — ✅ (one stale script)

The headline physics anchors from the project's validation tier reproduce
quantitatively:

| Anchor | Measured this session | Expected |
|---|---|---|
| Plummer virial $Q=T/\lvert V\rvert$ | 0.5026 | 0.5 |
| King $c(W_0)$ vs King (1966) Table II | max $\lvert\Delta\rvert = 0.000$ | $\le 0.03$ |
| EFF Eddington-DF $Q$ ($\gamma=5$) | 0.502 | 0.5 |
| Engine-B King A-vs-B (KS / $\sigma$ dev) | 0.0002 / 0.0003 | $< 0.02$ |
| Engine-B global $Q$ (N=30k, unscaled) | 0.4976 | $0.5 \pm 0.02$ |
| Engine-B OM $\beta(r)$ deviation | 0.028 | $< 0.05$ |
| Kepler energy / period | $4.2\times10^{-16}$ / $0.0$ | exact |

The **one finding** here is tooling, not physics: `scripts/validate_king.py`
crashes (exit 1) in its velocity-equilibrium figure because it still calls
`KingVelocityDF(..., r_t=...)` — but the `r_t` field was *removed* in the S1/A3 fix
and the script was never updated. The c(W₀) table and density oracle pass before
the crash; the **package is fine**, the standalone script rotted. This is the
prior audit's T11 risk (validation scripts are not run in CI) materialising. The
defect is confined to that one script.

(audit-d3)=
## 3. Gradient / differentiability integrity — ✅

The gradient-audit run gate is clean:

```text
97 cases; 0 hazard(s).   (scripts/audit_gradients.py, exit 0)
```

Every public differentiable entry point — profiles, DFs, builders, cluster
constructors, the dispersion forward models, the ZAMS relations — reports an
AD-vs-FD ratio of $\approx 1.0$. All **four registries report `full: true`**:
api-coverage (120 symbol-tests, 0 untested), differentiability (41 audited, 97
rows, **0 hazards**), physics-validation (21 models, 0 untested), and
provenance-of-constants (34 constants, **0 unprovenanced**). No `jacfwd`/`hessian`
is taken through a reverse-only `custom_vjp` (the one place that would break is
explicitly documented as a warning, not exercised).

(audit-d4)=
## 4. Public API & naming — ⚠️

`progenax.__all__` exports **122 symbols**; roughly 95 carry release-complete
docstrings (summary + args + units + returns + differentiability note). The
binaries, dispersion, and rotation surfaces are exemplary — every $G$-consuming
binary function *requires* an explicit $G$, and the rotation overlays explicitly
document that they are kinematic, **non-equilibrium** transforms.

The gaps to close before a polished release:

- **`MichieProfile.from_W0_rc` and its methods are undocstringed** — a headline
  equilibrium model documented below the rest of the API.
- **`TruncatedIMF.ppf`** and the IMF mass-ratio `pdf`/`cdf`/`ppf` family lack
  structured Returns blocks.
- **Units-policy split (deferred):** the five `*VelocityDF.sample_velocities`
  methods and `MultiComponentCluster.sample_cluster` resolve `G=None →
  DEFAULT_UNITS.G`, while `build_spatial_ic` *requires* explicit $G$. This is the
  prior audit's A2, **explicitly deferred in the CHANGELOG** as a breaking sweep.
  It is documented behaviour, not a silent default — but the package contradicts
  its own MANDATORY explicit-units policy on those entry points, so it should be
  resolved before a 1.0.

None of these are physics defects; the remediation is bounded documentation work
plus one API-contract decision.

(audit-d5)=
## 5. Architecture & JAX-native — ✅

Clean against every JAX-native release criterion:

- **No hardcoded $G$** — every numeric-$G$ string in `src/progenax/` is a
  docstring or doctest; $G$ is always threaded as an argument or read from
  `units.G`.
- **No numpy/scipy on a core/differentiable path** — the only real numpy/scipy
  imports are *lazy* (inside function bodies) and confined to the optional,
  non-differentiable `[diagnostics]` extra, with actionable errors if absent.
- **`jax.lax.while_loop`** appears only inside `custom_vjp` forward passes whose
  autodiff is fully replaced by a hand-written implicit-function-theorem reverse
  rule — so it is never differentiated through.
- All stateful classes are immutable `eqx.Module` PyTrees; all `.at[]` updates are
  functional; Python loops iterate over static component lists, not particles.

Five files exceed the soft ~500-LOC target (`dispersion.py` 792, `king.py` 636,
`limepy_multimass.py` 623, `multicomponent.py` 563, `builders.py` 525); each is a
single coherent algorithm, which the project's size guidance explicitly permits.

(audit-d6)=
## 6. Documentation site — ✅ (two editorial ⚠️)

`myst build` produces **172 pages with zero content warnings**, zero broken
cross-references, all 64 bibliography keys and ~154 figures resolving, and a
perfect `(label)=` anchor convention (zero `{#id}` violations).

**The fabricated-content sweep found nothing fabricated in released-core.** Every
load-bearing literature value that could be read from a PDF text layer matched its
source exactly — King (1966) Table II, the LIMEPY $g+3/2$ index, the Marks (2014)
erratum threshold $x' \ge -0.87$, and the Salpeter/Chabrier/Maschberger/Moe
constants. The two *previously-caught* fabrications (a fake LIMEPY $g=1$ table and
a debunked King log-$c$ series) are confirmed **absent** from the live docs and
source — `limepy.py` even states explicitly that there is *no* misprint.

As an independent spot-check, the Demircan & Kahraman (1991) mass–radius
coefficients that anchor the `compute_stellar_radii` fix were verified
cell-for-cell against Table II of the PDF: the *empirical* row gives $\log R = a +
b\log M$ with $(a,b)=(0.026, 0.945)$ for $M<1.66\,\Msun$ and $(0.124, 0.555)$
above, i.e. $R = 1.06\,M^{0.945}$ and $1.33\,M^{0.555}$ — matching the code
exactly.

Two editorial issues remain for a polished launch:

- **The `40-howto/` section is six empty `TBD.` stubs** published in the table of
  contents.
- **Ten-plus links point to internal `docs/plans/` and `docs/notes/` files
  outside the site root** — they build locally (the files exist on disk) but will
  deploy as broken links exposing internal planning filenames.

:::{aside} Honesty caveat on the citation sweep
About five load-bearing PDFs (Demircan & Kahraman 1991, EFF 1987, Plummer 1911,
Michie 1963; Kroupa 2001 partially) are **scanned images with no text layer**, so
the automated sweep could not text-extract them. The Demircan & Kahraman case was
re-verified visually this session (above); the rest are canonical forms with no
red flags but are marked `UNVERIFIED-by-automated-sweep` rather than asserted.
:::

(audit-d7)=
## 7. Packaging & metadata — ✅

The wheel and sdist build cleanly and pass `twine check`. The wheel contains
**only `src/progenax/`** — the experimental `gravoturb_fdf`, the `scripts/`
demos, the tests, and the paper PDFs are all correctly excluded. `pyproject.toml`
carries a real author/email, version, license, classifiers, URLs, and a sensible
extras split; a `LICENSE` file (Apache-2.0) is present; the sdist ships no
copyrighted PDFs. A clean-venv install of the wheel plus the public `jaxstro`
sibling imports successfully:

```text
progenax 0.1.0 + diagnostics import OK
n public symbols: 122
```

Minor: the sdist is ~20 MB (it bundles `docs/`, including built figures); tightening
the sdist file set is optional polish.

(audit-d8)=
## 8. Reproducibility & CI — ⚠️

The CI configuration is genuinely strong: a sharded released-core matrix, a
**wheel-smoke clean-venv import** job (catches eager-import and packaging
regressions), a dedicated **gradient-gate** (run gate + JSON staleness comparator
+ coverage ratchet + `audit_gradients.py` exit-0), a release-tag slow-physics lane
with `PROGENAX_STRICT_REFS=1`, and a 3.10/3.13 version matrix — all unified by an
aggregator `tests` check.

**But both workflows are `disabled_manually` at GitHub**, and the nightly crons
were removed to conserve Actions minutes. So at present the advertised gate runs
*nowhere but a developer's machine*. Re-enabling the workflows is a release
prerequisite (see the [checklist](#release-checklist)). The gated demo CLIs were
spot-checked (`demo_cluster_builders` exits 0, all 5 demos pass) and the
run-record pattern is in place.

(audit-d9)=
## 9. README & examples — ✅

`tests/integration/test_readme_examples.py` executes **every** Python code block
in the README in a fresh namespace; all **9 blocks pass** (8.9 s). The README
itself reflects the current API (the prior audit's deleted-API examples are gone).
Minor gap: code blocks inside the `docs/` site pages are not auto-executed, so
they are not machine-protected the way the README blocks are.

(audit-d10)=
## 10. Release-scope decisions — surfaced

Two scoping calls were referred to the maintainer and **decided**:

- **OED demos** (`scripts/_demo_oed*.py`, `scripts/demo_oed*.py`, and the
  `60-science-demos/optimal-design/` pages) are the CAREER Aim-1 prototype slated
  to migrate to the planned **informax** package. **Decision: hold them out of the
  v0.1.0 release scope** (tracked in the [checklist](#release-checklist)).
- The experimental **`gravoturb_fdf`** subsystem stays repo-only, excluded from the
  wheel and the public API, and is correctly labelled experimental throughout the
  docs. It may be mentioned in release notes as future work.

(audit-prior-blockers)=
## Re-verification of the 2026-06-11 blockers

This release cycle claimed to resolve the prior adversarial audit's ten blockers.
Re-checked against the live tree:

| ID | Blocker | Status this session |
|---|---|---|
| R1 | CI red / missing `jaxstroviz` | ✅ lock decoupled; config sound (but workflows disabled — see D8) |
| R2 | Not pip-installable (`jaxstro` unpublished) | ❌ **deferred by design** — the one open blocker |
| R3 | Moe F_twin mis-normalised | ✅ rebuilt; pinned to Table-13 nodes (`test_moe_full.py`) |
| R4 | King/Michie/EFF core under-resolved at high $W_0$ | ✅ sqrt-stretched CDF grid in `king.py` |
| R5 | `sample_fixed_n` silent shortfall | ✅ raises `ValueError` on unreachable target |
| R6 | `compute_stellar_radii` inverted | ✅ D&K91 fit, verified vs PDF Table II |
| R7 | README/docs phantom API | ✅ rewritten; phantom classes gone; `progenax-legacy` removed |
| R8 | Slow trust anchors not in CI | ✅ release-tag slow lane + strict-refs (lane disabled with the rest) |
| R9 | `diagnostics` eager numpy/scipy | ✅ lazy imports + `[diagnostics]` extra |
| R10 | `PowerLawIMF` $\alpha=1$ NaN gradient | ✅ `exp_safe` double-where ported |

**Nine of ten verified fixed; R2 is the deferred packaging decision.**
