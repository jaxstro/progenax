"""Physics-validation registry ratchet (Phase 4 / Task 4.1).

The manifest partitions EVERY ``progenax.__all__`` symbol into exactly one of
MODEL_INVARIANTS / EXEMPT_NON_MODEL / EXEMPT_NON_EQUILIBRIUM_MODEL / UNTESTED_MODELS,
with no overlaps and no stale entries. The teeth:

  1. Categorization — a NEW public symbol uncategorized -> RED (catches additions).
  2. Operational IS_MODEL ratchet (C3) — every symbol that is OPERATIONALLY a model
     (implements SpatialProfile/VelocityDF/IMFProtocol by the runtime-checkable
     protocols, OR is a ``build_*_cluster`` entry point) must be in MODEL_INVARIANTS
     or EXEMPT_NON_EQUILIBRIUM_MODEL, NEVER in EXEMPT_NON_MODEL. A real profile/DF/IMF
     mis-filed as a non-model -> RED, so it cannot silently escape physics validation.
  3. No stale mappings — a manifest entry for a removed/renamed symbol -> RED.
  4. Cross-check vs grad-audit SYMBOL_CATEGORY (C4 anti-drift) — soft/documented where
     the two model/exempt splits disagree (different lenses); HARD-fail only the
     obviously-wrong case (a symbol EXEMPT_NON_MODEL here but a profile/DF/IMF by
     protocol — caught operationally by check 2, but cross-checked here too).
  5. Cited tests resolve — every node id in MODEL_INVARIANTS collects (no mapping to a
     non-existent test; the C2 anti-theater discipline: cite a test that actually runs).

Operational model definition (NOT a hand-list — hand-lists go inconsistent; see the
grad-audit ``build_plummer_cluster`` EXEMPT_HELPER vs ``build_king_cluster`` AUDITED
asymmetry): a symbol is a model iff
  - it is a class whose ``issubclass`` against the method-only ``SpatialProfile`` or
    ``VelocityDF`` runtime-checkable protocol is True, OR
  - it is a class structurally conforming to ``IMFProtocol`` (which has the data members
    ``m_min``/``m_max`` so ``issubclass`` raises — we test the method set the protocol
    requires, the class-level equivalent of ``isinstance(instance, IMFProtocol)``), OR
  - its name matches ``build_*cluster*`` (a cluster entry point).
The protocol classes themselves (SpatialProfile/VelocityDF/IMFProtocol/...) are NOT models.
"""
import re

import progenax
import pytest

from progenax.protocols import IMFProtocol, SpatialProfile, VelocityDF
from tests.validation.physics_registry.manifest import (
    EXEMPT_NON_EQUILIBRIUM_MODEL,
    EXEMPT_NON_MODEL,
    MODEL_INVARIANTS,
    UNTESTED_MODELS,
)

# The runtime-checkable typing Protocols themselves are NOT models (they are the
# yardstick, not a measured object). issubclass(SpatialProfile, SpatialProfile) is True,
# so they would otherwise self-classify as models.
_PROTOCOL_NAMES = frozenset({
    "SpatialProfile", "VelocityDF", "IMFProtocol", "PeriodDistribution",
    "EccentricityDistribution", "ConditionalEccentricityDistribution",
    "MassPeriodEccentricityDistribution", "BinaryFractionModel", "CompanionModel",
})

# The method set IMFProtocol REQUIRES (it cannot be used with issubclass because it also
# declares the data members m_min/m_max; this class-level method check is the structural
# equivalent of isinstance(instance, IMFProtocol) minus the data-attribute presence,
# which every IMF class has as instance state).
_IMF_METHODS = ("logpdf", "cdf", "ppf", "sample", "mean_mass")

_BUILD_CLUSTER_RE = re.compile(r"build_\w*cluster")


def _operational_model_kind(name):
    """Return the operational model kind of an __all__ symbol, or None if not a model.

    Kinds: 'SpatialProfile' | 'VelocityDF' | 'IMFProtocol' | 'build_cluster'.
    """
    if name in _PROTOCOL_NAMES:
        return None
    if _BUILD_CLUSTER_RE.search(name):
        return "build_cluster"
    obj = getattr(progenax, name)
    if not isinstance(obj, type):
        return None
    # Method-only protocols -> issubclass is the canonical runtime-checkable test.
    try:
        if issubclass(obj, SpatialProfile):
            return "SpatialProfile"
    except TypeError:
        pass
    try:
        if issubclass(obj, VelocityDF):
            return "VelocityDF"
    except TypeError:
        pass
    # IMFProtocol has data members -> structural method-set check (class-level isinstance).
    if all(hasattr(obj, m) for m in _IMF_METHODS):
        # Guard: do not double-count a profile/DF that happens to share a method name.
        try:
            if issubclass(obj, (SpatialProfile, VelocityDF)):
                return None  # already classified above
        except TypeError:
            pass
        return "IMFProtocol"
    return None


def _operational_models():
    """The set of __all__ symbols that are operationally models (kind != None)."""
    return {n for n in progenax.__all__ if _operational_model_kind(n) is not None}


# --------------------------------------------------------------------------------------
# 1. Categorization: every public symbol lands in EXACTLY ONE manifest dict.
# --------------------------------------------------------------------------------------


def test_every_public_symbol_is_categorized():
    """Each __all__ symbol is in EXACTLY ONE of the four manifest dicts (no gaps, no
    overlaps). A new public symbol with no home -> RED (catches additions)."""
    public = set(progenax.__all__)
    mi, en, ene, un = (
        set(MODEL_INVARIANTS), set(EXEMPT_NON_MODEL),
        set(EXEMPT_NON_EQUILIBRIUM_MODEL), set(UNTESTED_MODELS),
    )
    union = mi | en | ene | un

    missing = sorted(public - union)
    assert not missing, (
        "public symbols not categorized in MODEL_INVARIANTS / EXEMPT_NON_MODEL / "
        f"EXEMPT_NON_EQUILIBRIUM_MODEL / UNTESTED_MODELS: {missing}")

    # Pairwise disjointness (a symbol's status must be unambiguous).
    for a_name, a, b_name, b in [
        ("MODEL_INVARIANTS", mi, "EXEMPT_NON_MODEL", en),
        ("MODEL_INVARIANTS", mi, "EXEMPT_NON_EQUILIBRIUM_MODEL", ene),
        ("MODEL_INVARIANTS", mi, "UNTESTED_MODELS", un),
        ("EXEMPT_NON_MODEL", en, "EXEMPT_NON_EQUILIBRIUM_MODEL", ene),
        ("EXEMPT_NON_MODEL", en, "UNTESTED_MODELS", un),
        ("EXEMPT_NON_EQUILIBRIUM_MODEL", ene, "UNTESTED_MODELS", un),
    ]:
        overlap = sorted(a & b)
        assert not overlap, f"symbols in BOTH {a_name} and {b_name}: {overlap}"


# --------------------------------------------------------------------------------------
# 2. Operational IS_MODEL ratchet (C3): a real model cannot hide in EXEMPT_NON_MODEL.
# --------------------------------------------------------------------------------------


def test_operational_models_are_not_mis_filed_as_non_model():
    """Every OPERATIONAL model (profile/DF/IMF by protocol, or build_*_cluster) must be
    in MODEL_INVARIANTS or EXEMPT_NON_EQUILIBRIUM_MODEL (or, as an honest hole,
    UNTESTED_MODELS) — NEVER in EXEMPT_NON_MODEL. This is the C3 fix: do not let a real
    profile/DF silently escape physics validation by being hand-waved into 'not a model'.
    """
    ops = _operational_models()
    mis_filed = sorted(s for s in ops if s in EXEMPT_NON_MODEL)
    assert not mis_filed, (
        "OPERATIONAL models (profile/DF/IMF by protocol or build_*_cluster) mis-filed in "
        f"EXEMPT_NON_MODEL — they cannot escape physics validation: {mis_filed}. Move each "
        f"to MODEL_INVARIANTS (with its invariants) or EXEMPT_NON_EQUILIBRIUM_MODEL "
        f"(reference-parity / non-equilibrium, with a documented reason).")

    # And every operational model must be ACCOUNTED FOR by a model dict (the registry's
    # whole point). An operational model in none of the model dicts is an unguarded hole.
    accounted = set(MODEL_INVARIANTS) | set(EXEMPT_NON_EQUILIBRIUM_MODEL) | set(UNTESTED_MODELS)
    unguarded = sorted(s for s in ops if s not in accounted)
    assert not unguarded, (
        f"OPERATIONAL models not in any model dict (unguarded by the registry): {unguarded}. "
        f"Add each to MODEL_INVARIANTS, EXEMPT_NON_EQUILIBRIUM_MODEL, or UNTESTED_MODELS.")


# --------------------------------------------------------------------------------------
# 3. No stale mappings.
# --------------------------------------------------------------------------------------


def test_no_stale_mappings():
    """No manifest dict references a symbol no longer in __all__ (catches deletions)."""
    public = set(progenax.__all__)
    union = (
        set(MODEL_INVARIANTS) | set(EXEMPT_NON_MODEL)
        | set(EXEMPT_NON_EQUILIBRIUM_MODEL) | set(UNTESTED_MODELS)
    )
    stale = sorted(union - public)
    assert not stale, (
        f"manifest entries for symbols no longer in progenax.__all__ (remove them): {stale}")


# --------------------------------------------------------------------------------------
# 4. No untested-model holes (HARD as of Task 4.1: UNTESTED_MODELS is empty).
# --------------------------------------------------------------------------------------


def test_no_untested_model_holes():
    """HARD: every operational/equilibrium model has at least one enumerated physics
    invariant mapped to an asserting validation test, so UNTESTED_MODELS is empty. A NEW
    model whose invariant is not yet checked re-populates UNTESTED_MODELS and turns this
    RED until the validation test exists (a Task-4.2 hole for Anna)."""
    assert not UNTESTED_MODELS, (
        "models with NO enumerated physics invariant (real holes — write the validation "
        f"test and move to MODEL_INVARIANTS with Anna's sign-off): {sorted(UNTESTED_MODELS)}")


# --------------------------------------------------------------------------------------
# 5. Cross-check vs grad-audit SYMBOL_CATEGORY (C4 anti-drift).
# --------------------------------------------------------------------------------------


def test_crosscheck_grad_audit_symbol_category():
    """C4 anti-drift: the physics-registry model/exempt split vs grad-audit's
    AUDITED/EXEMPT split. The two registries answer DIFFERENT questions (physics
    invariants vs Fisher/gradient entry points), so divergence is EXPECTED and reported,
    not failed — EXCEPT the one obviously-wrong combination, which is HARD-fail: a symbol
    we mark EXEMPT_NON_MODEL here while it is OPERATIONALLY a profile/DF/IMF. If it is a
    real model by protocol, calling it a non-model is a mistake regardless of grad-audit.
    """
    from tests.validation.grad_audit.manifest import AUDITED, SYMBOL_CATEGORY

    ops = _operational_models()
    # Hard: an operationally-real model marked EXEMPT_NON_MODEL (also caught by check 2;
    # cross-checked here so the C4 lens flags it independently).
    hard_fail = sorted(s for s in EXEMPT_NON_MODEL if s in ops)
    assert not hard_fail, (
        f"symbols EXEMPT_NON_MODEL here but OPERATIONALLY a profile/DF/IMF: {hard_fail}. "
        f"A real model is not a non-model — move to MODEL_INVARIANTS or "
        f"EXEMPT_NON_EQUILIBRIUM_MODEL.")

    # Informational: where the physics-registry and grad-audit partitions disagree.
    # Documented for human review without making the suite brittle to each registry's lens.
    divergences = []
    for s in progenax.__all__:
        ga = SYMBOL_CATEGORY.get(s, "<absent>")
        if s in MODEL_INVARIANTS:
            pr = "MODEL_INVARIANTS"
        elif s in EXEMPT_NON_EQUILIBRIUM_MODEL:
            pr = "EXEMPT_NON_EQUILIBRIUM_MODEL"
        elif s in UNTESTED_MODELS:
            pr = "UNTESTED_MODELS"
        else:
            pr = "EXEMPT_NON_MODEL"
        ga_audited = ga == AUDITED
        pr_model = pr in ("MODEL_INVARIANTS", "EXEMPT_NON_EQUILIBRIUM_MODEL", "UNTESTED_MODELS")
        # Disagreement of interest: grad-audit AUDITED a symbol we call a non-model, or
        # we call a symbol a model that grad-audit marks EXEMPT.
        if (ga_audited and not pr_model) or (not ga_audited and pr_model and ga != "<absent>"):
            divergences.append(f"{s}: grad_audit={ga} physics_registry={pr}")
    if divergences:
        print(
            "\n[physics-registry vs grad-audit divergences — EXPECTED (different lenses: "
            "physics invariants vs Fisher/gradient entry points), documented for C4 "
            "anti-drift review]:\n  " + "\n  ".join(divergences)
        )


# --------------------------------------------------------------------------------------
# 6. Every cited validation test node id resolves (C2 anti-theater: cite a real test).
# --------------------------------------------------------------------------------------


def test_every_cited_test_node_id_resolves():
    """Collect-only every node id cited in MODEL_INVARIANTS; a mapping to a non-existent
    test -> RED. This enforces the C2 discipline (cite a test that actually runs and
    asserts the invariant) at the registry level, not just by convention.

    Runs ``pytest --collect-only`` over the unique cited node ids in a subprocess (the
    parsing/subprocess pattern mirrors the dashboard generator; this is a script-style
    check, not core JAX code). On any collection error the offending ids are surfaced.
    """
    import subprocess
    import sys
    from pathlib import Path

    node_ids = sorted({nid for inv in MODEL_INVARIANTS.values() for nid in inv.values()})
    assert node_ids, "MODEL_INVARIANTS cites no tests — the registry is empty."

    repo_root = Path(__file__).resolve().parents[3]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q",
         "--no-header", "-p", "no:cacheprovider", *node_ids],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    # pytest exits 0 when all cited node ids collect. A non-zero exit means at least one
    # id did not resolve (typo / renamed / deleted test).
    assert proc.returncode == 0, (
        "at least one MODEL_INVARIANTS node id did not collect (a mapping to a "
        "non-existent test — fix the citation):\n"
        f"--- stdout ---\n{proc.stdout[-4000:]}\n--- stderr ---\n{proc.stderr[-2000:]}")
