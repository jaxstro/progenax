"""Provenance-cards coverage manifest (the 5th enforcement registry — ADR-0034).

Frozen, hand-curated structures (NOT computed from the YAML at runtime — a derived
manifest could not catch a silently deleted card):

  CARDED_MODELS   : physics-registry model name -> card id that carries its provenance.
                    The RATCHET set: every entry here MUST resolve to a card, and the
                    set only grows (populate phases append; deleting an entry is a
                    reviewed decision, not a drive-by).
  REGISTRY_FULL   : flips True when EVERY MODEL_INVARIANTS model is carded (end of the
                    Slice-B populate). While False, the coverage test asserts the
                    ratchet subset only and reports the remaining gap as an xfail-style
                    reminder, mirroring the other registries' full flags.

Phase log (ADR-0034 sequencing):
  2026-07-10  pilot: plummer_profile, king_profile, eff_profile (ledger-verified trio).
  2026-07-11  B1 velocity DFs: plummer_df, king_df, michie_df, eff_df, om_anisotropy
              (overlay, extra-registry) + michie_profile. Coverage 8/21.
  2026-07-11  B2 IMFs: powerlaw_imf, chabrier, maschberger, tapered_powerlaw +
              truncated + environment_imf (last three partly extra-registry).
              Coverage 12/21. House guidance: Maschberger preferred (smooth).
  2026-07-11  B3 binaries: moe_pqe, period_distributions, eccentricity,
              kepler_elements, binary_cluster_assembly. Coverage 13/21.
"""

# physics-registry model name (tests/validation/physics_registry/manifest.py
# MODEL_INVARIANTS keys) -> provenance card id (docs/provenance/registry/*.yaml).
CARDED_MODELS: dict[str, str] = {
    "PlummerProfile": "plummer_profile",
    "KingProfile": "king_profile",
    "EFFProfile": "eff_profile",
    "MichieProfile": "michie_profile",
    "PlummerVelocityDF": "plummer_df",
    "KingVelocityDF": "king_df",
    "MichieVelocityDF": "michie_df",
    "EFFVelocityDF": "eff_df",
    "PowerLawIMF": "powerlaw_imf",
    "ChabrierIMF": "chabrier",
    "Maschberger": "maschberger",
    "TruncatedIMF": "truncated",
    "build_binary_cluster": "binary_cluster_assembly",
}

# True once every MODEL_INVARIANTS model has a card (Slice-B populate complete).
REGISTRY_FULL: bool = False
