# Differentiability Gradient-Audit Harness — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan
> task-by-task. Execute in batches (one tier = one batch); STOP at every ⛔ CHECKPOINT and report
> to Anna — she is the non-negotiable human-in-the-loop and must approve before you continue.
> Never weaken a test to make it pass; if a fix changes expected physics, derive the new
> expectation analytically inside the test and show Anna.

**Goal:** Build a systematic AD-vs-FD gradient-audit harness over every public progenax sampling/
physics entry point that proves each parameter→IC and parameter→summary gradient is finite and
finite-difference-consistent (or is a documented, pinned limitation), fix the measurement-confirmed
hazards, and publish the result as a living validation website table whose numbers are the same
numbers the test gate asserts on.

**Architecture:** One pure engine (`audit_entry_point`) returns a structured `AuditResult` for a
`(entry_point, param, direction)` case; a thin pytest layer asserts per-`expect`-class, and a thin
script dumps the same results to JSON for the website. Confirmed-but-unfixed hazards are
`xfail(strict=True)` with a hazard-id (self-cleaning ratchet). Built in 4 tiers (headline samplers →
cluster/binary → binned-Fisher-path → consolidation of all scattered grad tests into the registry).

**Tech Stack:** JAX/Equinox (100% JAX-native core — `jnp`, `lax.scan`/`fori_loop`, never
`while_loop`, no numpy/scipy in `src/progenax` except the documented `diagnostics` carve-out),
pytest + pytest-xdist, diffrax, uv, mystmd (website).

**Design source of truth:** `docs/plans/2026-06-13-differentiability-gradient-audit-design.md`
(decisions D1–D6, the hazard map, the coverage matrix). Read it before starting.

---

## Ground rules for the executing session

- **Branch:** `feat/differentiability-audit` (already created off `main`; the design doc is
  committed there as `569d9dc`). Commit per task. Do **NOT** push or merge without Anna's go.
- **Test commands** (`progenax/CLAUDE.md`):
  ```bash
  # Single test during TDD:
  env -u VIRTUAL_ENV uv run --no-sync pytest <path>::<test> -v
  # FAST gate (inner loop, ~4 min, excludes slow):
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -m "not slow" -n auto
  # FULL gate (tier close-out, ~9 min):
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1" \
    env -u VIRTUAL_ENV uv run --no-sync pytest tests/unit tests/integration tests/validation -q -n auto
  ```
- **TDD discipline.** The *engine* (Tier 0) is real RED→GREEN: toy broken functions (NaN-grad,
  stop_gradient, wrong-by-2×) must be caught before the engine is trusted. The *per-case* tasks are
  characterization tests (they pin current behaviour); each tier includes at least one **mutation
  check** — temporarily break the entry point, confirm the audit flags it, revert — so a passing
  case is proven to have teeth (this is how we honour "if a test passes immediately, rethink it"
  for characterization tests). `known_zero` cases (king `r_t`) and suspected-hazard edges (H2) are
  real measurements that may legitimately reveal AD=0 or a hazard.
- **Units:** STELLAR (M☉, pc, Myr); G explicit (`STELLAR.G`); never hardcode G.
- **JAX policy:** reverse-mode `jax.grad`/`jacrev` only (ODE `custom_vjp`-safe); `jnp`; no in-place
  ops. float64 is automatic on `import progenax`.
- **After each tier:** run the FULL gate, report results + diffs at the ⛔ CHECKPOINT, wait for
  approval.

---

# Tier 0 — Harness scaffolding (TDD the engine itself)

The engine must be proven to *catch* broken gradients before any real entry point trusts it.

### Task 0.1: The `Case`/`AuditResult` types + `audit_entry_point` engine

**Files:**
- Create: `tests/validation/grad_audit/__init__.py` (empty)
- Create: `tests/validation/grad_audit/core.py`
- Test: `tests/validation/grad_audit/test_core_engine.py`

**Step 1: Write the failing tests** — `tests/validation/grad_audit/test_core_engine.py`:

```python
"""The grad-audit engine must CATCH broken gradients (NaN / silent-zero / wrong-by-2x)
and correctly classify the intentional-limitation expect-classes. These toy functions are
the engine's RED proof — they do not touch progenax physics."""
import jax
import jax.numpy as jnp
import progenax  # noqa: F401  (float64)

from tests.validation.grad_audit.core import Case, audit_entry_point


def _case(fn, expect="consistent", tol=1e-5, reduce=jnp.sum):
    return Case(id="toy", direction="params->IC", fn=fn, param="x",
                theta0=2.0, reduce=reduce, expect=expect, tol=tol)


def test_clean_linear_is_clean():
    r = audit_entry_point(_case(lambda x: jnp.array([3.0 * x, 5.0 * x])))
    assert r.finite and r.status == "clean"
    assert abs(r.ratio - 1.0) < 1e-6 and r.abs_ad > 1e-9


def test_silent_zero_is_hazard():
    # stop_gradient -> AD is 0 but FD is non-zero: the headline failure mode.
    r = audit_entry_point(_case(lambda x: jax.lax.stop_gradient(3.0 * x) * jnp.ones(2)))
    assert r.status == "hazard" and r.abs_ad < 1e-12


def test_nan_grad_is_hazard():
    # sqrt(x - x) has a 0/0 grad -> NaN; the engine must not call it clean.
    r = audit_entry_point(_case(lambda x: jnp.sqrt(x - x) * x * jnp.ones(1)))
    assert (not r.finite) and r.status == "hazard"


def test_wrong_by_two_is_hazard():
    # AD double-counts: a custom_jvp that lies about the derivative.
    @jax.custom_jvp
    def f(x):
        return x * jnp.ones(1)
    f.defjvp(lambda p, t: (f(p[0]), 2.0 * t[0] * jnp.ones(1)))  # claims 2x
    r = audit_entry_point(_case(f))
    assert r.status == "hazard" and abs(r.ratio - 2.0) < 1e-5


def test_known_zero_pins_zero_gradient():
    # An intentionally constant-in-x output: AD=0, FD=0 -> known-limitation, NOT hazard.
    r = audit_entry_point(_case(lambda x: jnp.ones(2), expect="known_zero"))
    assert r.status == "known-limitation"


def test_known_zero_flags_if_gradient_appears():
    # If a 'known_zero' case SUDDENLY has a gradient, that is a hazard (unannounced change).
    r = audit_entry_point(_case(lambda x: 3.0 * x * jnp.ones(2), expect="known_zero"))
    assert r.status == "hazard"


def test_known_blocked_requires_only_finite():
    r = audit_entry_point(_case(lambda x: jax.lax.stop_gradient(x) * jnp.ones(2),
                                expect="known_blocked"))
    assert r.status == "known-limitation"  # finite AD (0) is acceptable for a blocked site
```

**Step 2: Run RED**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/grad_audit/test_core_engine.py -v
```
Expected: collection/import error (`core` does not exist).

**Step 3: Implement `core.py`**

```python
"""Shared AD-vs-FD gradient-audit engine (single source of truth for the gate + the website).

audit_entry_point(case) returns an AuditResult whose `status` is COMPUTED from
(expect, finite, |ratio-1|<tol, |ad|>eps) -- never hand-set. The same results feed the pytest
gate (tests/validation/test_grad_audit.py) and scripts/audit_gradients.py -> results.json.
Reverse-mode jax.grad only (ODE custom_vjp-safe, mirrors _demo_inference.fisher_information_gn).
"""
from dataclasses import dataclass, field
from typing import Callable, Literal, Tuple

import jax
import jax.numpy as jnp

Direction = Literal["params->IC", "params->summary"]
Expect = Literal["consistent", "known_zero", "known_blocked"]


@dataclass(frozen=True)
class EdgeConfig:
    """A curated boundary probe for a Case (e.g. W0=12, alpha=1.0)."""
    label: str                       # appears in the case id, e.g. "W0=12"
    theta0: float                    # the edge parameter value
    hazard_id: str | None = None     # links to the hazard map; set if it probes a suspect
    tol: float | None = None         # per-edge tolerance override
    expect: Expect | None = None     # per-edge expect override (e.g. alpha=1.0 -> known_blocked)


@dataclass(frozen=True)
class Case:
    id: str
    direction: Direction
    fn: Callable[[jax.Array], jax.Array]   # theta (scalar) -> output array
    param: str
    theta0: float
    reduce: Callable[[jax.Array], jax.Array] = jnp.sum   # output -> scalar
    expect: Expect = "consistent"
    tol: float = 1e-3
    h_rel: float = 1e-4
    eps: float = 1e-9                        # |AD| silent-zero threshold
    edges: Tuple[EdgeConfig, ...] = ()


@dataclass(frozen=True)
class AuditResult:
    id: str
    direction: str
    param: str
    theta: float
    finite: bool
    ad: float
    fd: float
    ratio: float
    abs_ad: float
    expect: str
    tol: float
    status: str          # clean | known-limitation | hazard


def _scalar(case: Case, theta: jax.Array) -> jax.Array:
    return case.reduce(case.fn(theta))


def _classify(expect, finite, ad, fd, ratio, tol, eps) -> str:
    if expect == "known_zero":
        # Pinned: AD must be (and stay) zero. FD of a grid-snapped step is ~0 off node
        # crossings; we require AD~0 and treat a re-appeared gradient as a hazard.
        return "known-limitation" if abs(ad) < eps else "hazard"
    if expect == "known_blocked":
        return "known-limitation" if finite else "hazard"
    # consistent
    if finite and abs(ad) > eps and abs(ratio - 1.0) < tol:
        return "clean"
    return "hazard"


def audit_entry_point(case: Case, theta: float | None = None,
                      tol: float | None = None, expect: str | None = None) -> AuditResult:
    theta = case.theta0 if theta is None else theta
    tol = case.tol if tol is None else tol
    expect = case.expect if expect is None else expect

    t = jnp.asarray(theta, dtype=jnp.float64)
    ad = float(jax.grad(lambda x: _scalar(case, x))(t))

    h = case.h_rel * max(abs(float(theta)), 1.0)
    g = lambda x: float(_scalar(case, jnp.asarray(x, dtype=jnp.float64)))
    fd = (g(float(theta) + h) - g(float(theta) - h)) / (2.0 * h)

    finite = bool(jnp.isfinite(jnp.asarray(ad)))
    if fd != 0.0:
        ratio = ad / fd
    else:
        ratio = 1.0 if ad == 0.0 else float("inf")
    status = _classify(expect, finite, ad, fd, ratio, tol, case.eps)
    return AuditResult(case.id, case.direction, case.param, float(theta), finite,
                       ad, fd, ratio, abs(ad), expect, tol, status)
```

**Step 4: Run GREEN**

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/grad_audit/test_core_engine.py -v
```
Expected: all 7 PASS.

**Step 5: Commit**

```bash
git add tests/validation/grad_audit/__init__.py tests/validation/grad_audit/core.py \
        tests/validation/grad_audit/test_core_engine.py
git commit -m "test(grad-audit): AD-vs-FD engine that catches NaN/silent-zero/wrong-by-2x + expect taxonomy"
```

### Task 0.2: Per-channel reductions

**Files:**
- Create: `tests/validation/grad_audit/reductions.py`
- Test: `tests/validation/grad_audit/test_reductions.py`

**Step 1: Failing test** — assert each reduction returns a finite scalar and is sensitive to its
channel (mean_radius responds to a position scale, mean_speed to a velocity scale, mean_mass to a
mass scale). Full test in the file; key assertions:

```python
def test_mean_radius_scales_with_positions():
    pos = jnp.ones((10, 3))
    assert float(mean_radius(2.0 * pos)) == pytest.approx(2.0 * float(mean_radius(pos)))

def test_reductions_finite_on_zeros():
    assert jnp.isfinite(mean_radius(jnp.zeros((5, 3))))   # 1e-30 guard, no NaN
```

**Step 2–4:** RED (no module) → implement:

```python
"""Per-channel scalar reductions for the params->IC audit direction (design D4)."""
import jax.numpy as jnp

def mean_radius(positions):       # (N,3) -> ()
    return jnp.mean(jnp.sqrt(jnp.sum(positions**2, axis=-1) + 1e-30))

def mean_speed(velocities):       # (N,3) -> ()
    return jnp.mean(jnp.sqrt(jnp.sum(velocities**2, axis=-1) + 1e-30))

def mean_mass(masses):            # (N,) -> ()
    return jnp.mean(masses)

def identity_sum(x):              # params->summary: reduce a vector statistic
    return jnp.sum(x)
```
→ GREEN → commit `test(grad-audit): per-channel reductions (mean r / v / m)`.

### Task 0.3: Registry skeleton + the parametrized gate + xfail-from-hazard wiring

**Files:**
- Create: `tests/validation/grad_audit/registry.py` (`REGISTRY: list[Case] = []` + a `smoke` Case)
- Create: `tests/validation/test_grad_audit.py`
- Test: the gate file itself is the test.

**Step 1:** `registry.py` starts with ONE real smoke case (Plummer r_h, known-good) so the gate is
non-empty:

```python
"""The grad-audit case registry: every public entry point x direction x param.
Tiers are added incrementally (see the implementation plan)."""
import jax, jax.numpy as jnp
from jaxstro.units import STELLAR
from progenax import PlummerProfile
from tests.validation.grad_audit.core import Case
from tests.validation.grad_audit.reductions import mean_radius

_KEY = jax.random.PRNGKey(0)
_MASSES = jnp.ones(400)

def _plummer_positions(r_h):
    return PlummerProfile(r_h=r_h).sample_positions(_MASSES, _KEY)

REGISTRY: list[Case] = [
    Case(id="PlummerProfile.sample_positions", direction="params->IC",
         fn=_plummer_positions, param="r_h", theta0=1.0, reduce=mean_radius,
         expect="consistent", tol=1e-5),
]
```

**Step 2:** `tests/validation/test_grad_audit.py` — the thin gate that parametrizes over REGISTRY
and asserts per status, turning a confirmed-but-unfixed hazard into `xfail(strict)` via the case's
`hazard_id`:

```python
"""The release gradient-gate: every registered entry point is finite + FD-consistent
(or a documented, pinned limitation). Numbers are emitted to JSON by scripts/audit_gradients.py;
this gate asserts on the same engine. See docs/plans/2026-06-13-...-design.md."""
import pytest
import progenax  # noqa: F401  (float64)

from tests.validation.grad_audit.core import audit_entry_point
from tests.validation.grad_audit.registry import REGISTRY

_IDS = [c.id for c in REGISTRY]


@pytest.mark.parametrize("case", REGISTRY, ids=_IDS)
def test_gradient_audit(case):
    """Baseline (generic) params: assert the computed status is acceptable."""
    r = audit_entry_point(case)
    if getattr(case, "hazard_id", None):
        pytest.xfail(f"HAZARD {case.hazard_id}: confirmed, pending triage")
    assert r.status in ("clean", "known-limitation"), (
        f"{case.id} [{case.param}] -> {r.status}: AD={r.ad:.3e} FD={r.fd:.3e} "
        f"ratio={r.ratio:.6f} finite={r.finite}"
    )


def _edge_cases():
    out = []
    for c in REGISTRY:
        for e in c.edges:
            out.append((c, e))
    return out


@pytest.mark.parametrize("case,edge", _edge_cases(),
                         ids=[f"{c.id}::{e.label}" for c, e in _edge_cases()])
def test_gradient_audit_edges(case, edge):
    """Edge/boundary params: the hazard probes."""
    if edge.hazard_id:
        pytest.xfail(f"HAZARD {edge.hazard_id}: confirmed at {edge.label}, pending triage")
    r = audit_entry_point(case, theta=edge.theta0,
                          tol=edge.tol or case.tol, expect=edge.expect or case.expect)
    assert r.status in ("clean", "known-limitation"), (
        f"{case.id}::{edge.label} -> {r.status}: AD={r.ad:.3e} FD={r.fd:.3e} ratio={r.ratio:.6f}"
    )
```

**Step 3: Run** the gate (the smoke case must pass):

```bash
env -u VIRTUAL_ENV uv run --no-sync pytest tests/validation/test_grad_audit.py -v
```
Expected: 1 passed (Plummer clean).

**Step 4: Mutation check (teeth proof).** Temporarily wrap `_plummer_positions` return in
`jax.lax.stop_gradient(...)`, rerun → the case must FAIL with status `hazard`. Revert. Show both
outputs in the report.

**Step 5: Commit** `test(grad-audit): parametrized release gate + xfail-from-hazard wiring`.

### Task 0.4: The audit script → JSON + markdown table

**Files:**
- Create: `scripts/audit_gradients.py`
- Test: `tests/validation/grad_audit/test_audit_script.py`

**Step 1: Failing test** — importing the script's `run_audit()` returns a list of dicts with the
required keys and writes valid JSON:

```python
def test_run_audit_emits_required_keys(tmp_path):
    from scripts.audit_gradients import run_audit
    rows = run_audit(out_json=tmp_path / "r.json")
    assert rows and {"id","direction","param","ratio","status","ad","fd"} <= set(rows[0])
    import json; json.loads((tmp_path / "r.json").read_text())  # valid JSON
```

**Step 2–4:** RED → implement `scripts/audit_gradients.py`:

```python
"""Run the grad-audit REGISTRY and emit validation/data/grad_audit_results.json + a markdown
table. The website doc cites this JSON; the pytest gate asserts on the same engine (design D1)."""
import json
from dataclasses import asdict
from pathlib import Path

import progenax  # noqa: F401  (float64)
from tests.validation.grad_audit.core import audit_entry_point
from tests.validation.grad_audit.registry import REGISTRY

_DEFAULT_JSON = Path(__file__).resolve().parents[1] / "validation" / "data" / "grad_audit_results.json"


def run_audit(out_json: Path = _DEFAULT_JSON) -> list[dict]:
    rows = []
    for c in REGISTRY:
        rows.append(asdict(audit_entry_point(c)))
        for e in c.edges:
            rows.append(asdict(audit_entry_point(
                c, theta=e.theta0, tol=e.tol or c.tol, expect=e.expect or c.expect)))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rows, indent=2))
    return rows


def to_markdown(rows: list[dict]) -> str:
    head = "| id | dir | param | ratio | finite | status |\n|---|---|---|---|---|---|\n"
    body = "".join(
        f"| `{r['id']}` | {r['direction']} | {r['param']} | {r['ratio']:.6f} | "
        f"{r['finite']} | {r['status']} |\n" for r in rows)
    return head + body


if __name__ == "__main__":
    rows = run_audit()
    print(to_markdown(rows))
    n_haz = sum(r["status"] == "hazard" for r in rows)
    print(f"\n{len(rows)} cases; {n_haz} hazard(s).")
    raise SystemExit(1 if n_haz else 0)
```
→ GREEN → commit `feat(grad-audit): audit_gradients.py emits results.json + markdown table`.

### ⛔ CHECKPOINT 0
Report: the engine RED→GREEN (7 toy tests — show the failing import then the pass), the Plummer
mutation-check teeth proof (stop_gradient → hazard → reverted), the smoke gate pass, and the
script's JSON sample. **Wait for approval before Tier 1.**

---

# Tier 1 — headline samplers + packaged summaries + pinned limitations

Each task = append Case(s) to `registry.py`, run the gate (expect clean / known-limitation),
commit. One representative **mutation check** per task group. Suspected-hazard edges may xfail.

### Task 1.1: Plummer + Plummer-OM velocity DF (params→IC)
Add `PlummerVelocityDF.sample_velocities` (param `r_h`, reduce `mean_speed`) and an OM edge
(`r_a` near the Merritt bound `0.75a`). Run; both expected clean (`tol=1e-5` Plummer closed-form,
`1e-3` OM). Commit `test(grad-audit): Plummer velocity DF + OM r_a edge`.

### Task 1.2: King profile + DF (params→IC) + the known_zero r_t pin
Add `KingProfile.from_W0_rc → positions` (`W0`, `r_c`; `tol=1e-3`; edge `W0=12`) and
`KingVelocityDF.sample_velocities` (`W0`). **Add the pinned limitation** as its own case:

```python
def _king_r_t(W0):
    return jnp.atleast_1d(KingProfile.from_W0_rc(W0=W0, r_c=1.0).r_t)
Case(id="KingProfile.r_t", direction="params->IC", fn=_king_r_t, param="W0",
     theta0=7.0, reduce=jnp.sum, expect="known_zero")  # d r_t/dW0 = 0 (intentional; design)
```
RED reality check: assert (in the report) that the `known_zero` case measures `AD≈0` — if it ever
measures a non-zero AD, that is the documented limitation silently changing and the gate fails.
Choose `theta0=7.0` (a W0 off a grid-node crossing so FD≈0 too). Commit.

### Task 1.3: Michie + EFF profile/DF (params→IC)
Add `Michie` (profile+DF, `W0`, `r_a`) and `EFF` (profile+DF, `γ`, `r_t`, `r_a`; edge `γ=2.01`).
Run (expect clean, `tol=1e-3`). Commit.

### Task 1.4: build_spatial_ic end-to-end (params→IC)
Add `build_spatial_ic` (Plummer profile×DF, e2e incl. virial scale + COM) param `r_h`,
`reduce=mean_radius`, `tol=1e-3`. **Mutation check** here (e2e path is the headline promise).
Commit.

### Task 1.5: IMF samplers (params→IC) + the α=1 known_blocked split
Add `PowerLawIMF.sample`/`.ppf`, `ChabrierIMF.sample`, `Maschberger.ppf`, `Schechter.ppf`
(params α/m_c/σ/μ/β). Edges:
- `PowerLawIMF` α: `EdgeConfig("alpha=1.0", 1.0, expect="known_blocked")` (branch-limited point —
  preliminary probe AD=0 vs FD=−2.4e-4), plus `EdgeConfig("alpha=0.999", 0.999)` consistent.
- `ChabrierIMF` boundary sample: an edge that forces a sample to `m_min` (a frozen `u→1e-12`) —
  `hazard_id=None` initially; run it. **If clean** (preliminary probe says likely), leave consistent
  with the measured ratio; **if it confirms a zeroed boundary gradient**, set `hazard_id="H6"` →
  xfail. Show the measured number either way.
- `PowerLawIMF.cdf` at `m=m_min` and `m=m_min+1e-3` (H4): same — measure, classify.
Commit `test(grad-audit): IMF samplers + alpha=1 branch pin + H4/H6 boundary probes`.

### Task 1.6: Packaged summary diagnostics (params→summary)
Add `IMF.logpdf`/`mean_mass` (param `α`; the mass-function channel) and `q_approx`,
`lambda_msr_approx` (param `r_h` *through* sampled positions). For `lambda_msr_approx`, the
assertion is **consistent + non-zero** — this proves the `segregation_approx.py:145`
`stop_gradient` did NOT block the real gradient (the intentional-site verification; design
refinement: this is `consistent`, not `known_zero`). Commit.

### ⛔ CHECKPOINT 1
FULL gate + `python scripts/audit_gradients.py` output. Report the status table (clean / known-
limitation / any hazard), the H4/H6/α=1 measured numbers, and the mutation-check teeth proofs.
Flag any confirmed hazard for Anna's triage **before** any fix. **Wait for approval.**

---

# Tier 1.5 — the living validation website doc (Phase 3 of the mission)

### Task W.1: Create + wire the doc
**Files:**
- Create: `docs/website/50-validation/differentiability-audit.md`
- Modify: `docs/website/myst.yml` (add the page under the validation section)

The doc: a *gradient integrity = Fisher integrity* prose intro; the **living status table** (one
row per case × direction, columns: entry point | direction | gradient finite? | AD-vs-FD ratio
(params tested) | hazard found | fix (commit) | status ✅/⚠/🔧/⏳), every number copied from
`validation/data/grad_audit_results.json` (cite the JSON; never fabricate); an **"intentional
non-differentiable sites"** subsection (king `r_t` d/dW0=0 + deferral-doc link; segregation
`stop_gradient`; α=1 branch point) with rationale; and a **changelog of fixes**.

**Step: build to 0 warnings**

```bash
cd docs/website && make build
```
Expected: build succeeds, 0 warnings. Fix any xref/anchor warnings before committing.

Commit `docs(website): living differentiability-audit validation page (Tier 1 numbers)`.

> The doc is updated again at the end of each subsequent tier with that tier's measured numbers.

### ⛔ CHECKPOINT W
Show the `make build` 0-warning output and the rendered table. **Wait for approval.**

---

# Tier 2 — MultiComponentCluster, binaries, rotation (the H2 + r_t-kink probes)

### Task 2.1: MultiComponentCluster Engine A (params→IC + the H2 ψ=0 probe)
Add `MultiComponentCluster.from_imf(...).sample_cluster` (params `W0`, `g`, `δ`). Add an **H2 edge**:
choose `W0` such that the ψ=0 boundary shell is actually sampled, and a `reduce` (mean radius)
sensitive to the outer mask. Run. If the ψ=0 `where` (multicomponent.py:266/276) or the shared
`r_t` kink zeros/wrongs the gradient → `hazard_id="H2"` (xfail) and surface to Anna; else consistent
with the measured ratio. Commit.

### Task 2.2: MultiComponentCluster Engine B (params→IC)
Add `MultiComponentCluster.from_density_profiles(...).sample_cluster` (params `r_h_j`, `γ_j`,
`r_a_j`). Run (Engine B has AD-vs-FD anchors already; expect clean `tol=1e-3`). Commit.

### Task 2.3: Binaries (params→IC)
Add `KeplerElements.to_state` (param `e`; edge `e=0.999`, `tol=1e-5` — the audit verified machine-
precision grads to e=0.999), `resolve_binary_components`, and `MoeCompanions.sample` (param a Moe
coupling parameter; check differentiability — Moe sampling may have non-diff selection, classify
honestly as consistent or known_blocked). Commit.

### Task 2.4: Rotation overlays (params→IC)
Add `apply_solid_body_rotation` / `apply_differential_rotation` (param `ω`, reduce `mean_speed`).
Run (additive overlay — linear in ω, expect clean). Commit. Update the website doc + `make build`.

### ⛔ CHECKPOINT 2
FULL gate + script output + updated website table. Report H2 / r_t-kink findings (the highest-
probability real hazards) with measured numbers. Triage any confirmed hazard with Anna. **Wait.**

---

# Tier 3 — the binned-kinematic Fisher path (params→summary)

### Task 3.1: Vendored frozen-edge binner + σ(r)/β(r)/N(r) cases
**Files:**
- Create: `tests/validation/grad_audit/binners.py` (a minimal frozen-edge `binned_sigma1d` /
  `binned_number_density`, ported from `scripts/_demo_inference.py`, just enough for the audit —
  document the provenance).
- Modify: `registry.py` (add 3–4 `params→binned-summary` cases).

Add cases: `build_spatial_ic(Plummer) → binned_sigma1d` (param `r_h`), `→ binned σ_β/β(r)` (param
`r_a`, OM), `→ binned_number_density N(r)` (param `r_h`). `reduce = identity_sum` (the binned
vector → scalar). Run (expect clean `tol=1e-3`). **Mutation check** on the σ(r) case (this IS the
Fisher path). Commit. Update website doc + `make build`.

### ⛔ CHECKPOINT 3
FULL gate + script + website. Report the Fisher-path numbers. **Wait.**

---

# Tier 4 — consolidation: the registry becomes the release gradient-gate

### Task 4.1: Inventory every existing gradient test
**Files:** Create: `.claude-work/grad-test-inventory.md`

```bash
grep -rln "jax.grad\|jacrev\|_central_fd\|AD.*FD\|finite.diff" tests/ | sort
```
For each hit, record in the inventory table: file → what it asserts → {migrate AD-FD into registry
| keep (non-gradient property) | delete (redundant finite-only)}. Known entries: `test_imf_
gradients.py` (migrate FD cases; KEEP α=1 kink + boundary-finite), `test_jax_compatibility.py`
(DELETE the finite-only smoke tests — audit T6 — after registry covers them), `test_find_alpha_
ift.py` (KEEP the `REF_ALPHA` forward regression; migrate the FD-gradient assertion),
`test_engine_b_physics.py` (KEEP the β(r) physics anchor; migrate the AD-FD part),
`test_limepy_tables.py`, `test_michie_physics.py`, `test_king_physics.py` grad bits. **No deletion
without Anna signing off the inventory.** Commit the inventory.

### ⛔ CHECKPOINT 4a — Anna approves the inventory's migrate/keep/delete column before any deletion.

### Task 4.2: Migrate + dedupe (per the approved inventory)
For each "migrate" row: ensure the registry has an equivalent (or stronger) case, then delete the
duplicate assertion. For each "delete" row: remove the finite-only smoke test. For each "keep" row:
leave it, add a comment pointing to the registry as the gradient source of truth. Run the FULL gate
after **each** file change (no batch deletions). Commit per file: `refactor(tests): migrate
<file> AD-FD into the grad-audit registry; drop redundant finite-only (audit T6)`.

### Task 4.3: Final artifacts (the 5-requirement DoD)
- `validation/plots/grad_audit_ratio.png` (per-case AD/FD ratio scatter, log residual) +
  `grad_audit_summary.png` (per-direction bar): add a `--plots` flag to `scripts/audit_gradients.py`
  (matplotlib is a `[viz]` extra — guard the import; this is a script, not core `src/`).
- `.claude-work/TASK_differentiability-audit_COMPLETE.md`: files, API, measured results table,
  test counts, lessons, integration notes.
- Final website doc pass: every row current, changelog complete, `make build` 0 warnings.
- Update `STATUS.md` (`next:` / `blocker:` lines).
Commit `docs: differentiability-audit completion doc + plots + STATUS`.

### ⛔ CHECKPOINT 4b — FINAL
FULL gate green; `python scripts/audit_gradients.py` exit 0 (no unxfailed hazards); `make build` 0
warnings; the website table complete with real numbers; completion doc written. Report everything.
**Only when Anna approves and CI is green: ONE final PR** `feat/differentiability-audit → main`.

---

## Definition of Done (ecosystem 5-requirement standard)
1. **Test suite:** the AD-vs-FD harness passes (finite + FD-consistent on both directions, generic +
   edge params); engine has RED-proven teeth; ≥1 mutation check per tier.
2. **Validation script:** `scripts/audit_gradients.py` runs the full registry, emits JSON + table,
   exits non-zero on any unxfailed hazard.
3. **Plots:** `validation/plots/grad_audit_*.png`.
4. **Quantitative results:** the printed table + the website table, every number from `results.json`.
5. **Completion doc:** `.claude-work/TASK_differentiability-audit_COMPLETE.md`.
Plus: every in-scope hazard fixed (root-cause, RED→GREEN) **or** Anna-accepted as a pinned
documented limitation; FULL gate green; website builds 0 warnings; STATUS.md updated; ONE final PR.
