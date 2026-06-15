"""Provenance-of-constants registry (Phase 5 / Task 5.1).

A ratcheting source of truth ported from ``docs/provenance-ledger.md`` (the 2026-06
provenance & credibility audit; ZERO fabricated values). Every cited physical constant /
empirical coefficient / fit in released-core maps to its provenance (paper + eq/table, or
CODATA/IAU). An allowlist-scoped new-literal ratchet reds CI if a new unprovenanced
citable-shaped number appears in a constant-bearing module.
"""
