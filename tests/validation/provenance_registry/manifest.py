"""Provenance-of-constants source of truth (Phase 5 / Task 5.1).

Mirrors the grad-audit / api-coverage / physics-registry frozen-literal pattern:
HAND-CURATED dict literals (NOT a regex float-literal scan of ``src/`` — that returns
~2,525 matches dominated by ``1.0``/``0.0``/``0.5``, array fills, shapes, and exponents
a regex cannot distinguish from a citable coefficient: a false-positive trap and the
exact thing the C6-hardened design forbids). This manifest is a faithful PORT of
``docs/provenance-ledger.md`` — the 2026-06 provenance & credibility audit that read the
held PDFs directly and found **ZERO fabricated values** in released-core. A derived scan
cannot catch a *deletion*; a hand-curated frozen literal can.

  PROVENANCE  : a stable id "module.py::CONSTANT_or_context" -> its citation (paper +
                eq/table/page, or CODATA/IAU, or "derivable identity"). One row per cited
                physical constant / empirical coefficient / fit the ledger verified.
  UNPROVENANCED : an allowlisted constant that genuinely lacks a citation TODAY -> a REAL
                hole for Task 5.2 / Anna adjudication. Honesty over coverage: an honest
                hole is correct; a fabricated PROVENANCE entry is a failure. The audit
                found ZERO fabricated values, so we expect mostly citation-porting and
                few/no real holes. Enforced by an xfail "no unprovenanced constants" test
                (mirrors api_coverage's holes pattern).
  ALLOWLIST_MODULES : the SCOPED set of constant-bearing ``src/`` modules the new-literal
                ratchet covers (NOT all of ``src/`` — the C6 scope). A new unprovenanced
                citable-shaped literal in one of these reds CI. A genuinely new unsourced
                number elsewhere is added to the allowlist with Anna's sign-off.
  ALLOWLIST_NON_COEFFICIENT : per allowlisted module, the numeric literals that are
                numerical-method / range-default / unit-conversion / guard values — NOT
                citable paper coefficients. Each is named with a reason so the carve is
                AUDITABLE, not a silent suppression (the C6 "keep the flagging conservative
                + documented" requirement). The scanner skips ONLY these (and the trivial
                0/1/2/±0.5 set + small ints).

Enforced by tests/validation/provenance_registry/test_provenance_coverage.py. The website
provenance section is generated from here (Task 5.2).
"""

# --- PROVENANCE: constant id -> citation (ported from docs/provenance-ledger.md) ---------
# Every value below was verified against a held PDF (or a derivable identity) by the
# 2026-06 audit — the ledger row is named in the trailing comment where it sharpens the
# trace. ZERO fabricated values were found; this is a citation-porting manifest.
PROVENANCE: dict[str, str] = {
    # ============================ Spatial profiles ===================================
    "profiles/plummer.py::scale-radius a = r_h*sqrt((1-0.5^(2/3))/0.5^(2/3))":
        "Plummer (1911) MNRAS 71, 460 — closed-form inverse of the Plummer mass CDF "
        "M(<r_h)/M=0.5; algebraically exact (= r_h*sqrt(2^(2/3)-1)). Ledger Batch-7 LOW "
        "spot-sample: canonical form, NOT the inverted-a bug.",
    "profiles/king.py::King(1966) Table II xi_t = r_t/r_c (c=4.699/131.4/2272 @ W0=3/9/15)":
        "King (1966) AJ 71, 64, Table II (p. 73) — verified Batch-3 against the held PDF; "
        "re-confirms log c {0.672/1.029/1.528/2.119}, NOT the debunked 0.84/1.18/1.48/1.76.",
    "profiles/limepy.py::LIMEPY density index g+3/2":
        "Gieles & Zocchi (2015) MNRAS 454, 576, Eq. 8 (p. 578) + erratum Eqs. 20/21 — "
        "index verified Batch-3 (main-text Eq. 8 = E_gamma(g+3/2, phi_hat); King g=1 corner).",
    "profiles/eff.py::EFF density rho(0)=1, rho(a)=2^(-gamma/2), rho~r^(-gamma)":
        "Elson, Fall & Freeman (1987) ApJ 323, 54 — closed-form EFF surface-density profile; "
        "derivable identity (asserted in tests/validation/test_eff_physics.py).",

    # ============================ IMFs (classic) =====================================
    "imf/power_law.py::Salpeter alpha = 2.35":
        "Salpeter (1955) ApJ 121, 161 (p. 165) — xi ∝ m^-1.35 per dlog m => alpha=2.35 "
        "per dm. Ledger Batch-5 ✅ verified against the held scanned PDF.",
    "imf/power_law.py::Kroupa exponents [0.3, 1.3, 2.3] + breaks [0.08, 0.5]":
        "Kroupa (2001) MNRAS 322, 231, Eq. 2 (p. 234, verbatim) — alpha0=0.3, alpha1=1.3, "
        "alpha2=alpha3=2.3; the 3-segment merge of alpha2=alpha3=2.3 is exact. Batch-5 ✅.",
    "imf/chabrier.py::m_c=0.08, sigma=0.69, A_ln=0.158, alpha=2.3":
        "Chabrier (2003) PASP 115, 763, Table 1 (p. 769) — single-star (disk) IMF: m_c=0.079 "
        "(≈0.08), sigma=0.69, A_ln=0.158, high-mass x=1.3 => alpha=2.3. Correctly distinct "
        "from the SYSTEM IMF (m_c≈0.2/sigma≈0.6). Batch-5 ✅.",
    "imf/smooth.py::Maschberger alpha=2.3, beta=1.4, mu=0.2, m_l=0.01":
        "Maschberger (2013) MNRAS 429, 1725, Table 1 (p. 1727, verbatim). Batch-5 ✅.",

    # ============================ Environment-dependent IMF (Marks/Jerabkova) ========
    "imf/environment/coefficients.py::JERABKOVA logMecl_coeff=0.6039, constant=0.2161":
        "Jerabkova et al. (2018) A&A 620, A39, Eq. 7/9 — 8pi half-mass-density convention: "
        "0.6039 = 0.99*0.61 (exact); 0.2161 is 8pi-derived (Batch-1 ✅; 8pi deviation from "
        "the Eq.9-printed 2.83 ratified 2026-06-09).",
    "imf/environment/coefficients.py::JERABKOVA x_threshold=-0.87, slope=-0.41, intercept=1.94, canonical=2.3":
        "Jerabkova (2018) Eq. 6 (cites the Marks 2014 erratum) — alpha3=2.3 (x<-0.87), "
        "-0.41x+1.94 (x>=-0.87). NEGATIVE threshold per Marks et al. 2014 erratum (MNRAS "
        "442, 3315, Eq. 1, corrects the printed +0.87 typo). Batch-1 ✅.",
    "imf/environment/coefficients.py::JERABKOVA FeH_coeff=-0.14, rho_logRho_coeff=0.99":
        "Jerabkova (2018) Eq. 7 (density-based, no constant term). Batch-1 ✅.",
    "imf/environment/coefficients.py::MARKS cos_theta=-0.139, sin_theta=0.990 (theta=98deg)":
        "Marks et al. (2012) MNRAS 422, 2246, Eq. 14/15 (p. 2252) — Fundamental-Plane "
        "rotation cos/sin(98deg). Batch-1 ✅.",
    "imf/environment/coefficients.py::MARKS slope=-0.4072, intercept=1.9383, x_hat_threshold=-0.87":
        "Marks (2012) Eq. 15 (full-precision 2012 slope/intercept; the 2014 erratum rounds "
        "to -0.41/1.94 while fixing the sign). Threshold -0.87 is the erratum-corrected value "
        "(was +0.87, a printed typo). Batch-1 ✅.",
    "imf/environment/coefficients.py::MARKS lowmass_slope=0.5":
        "Marks (2012) Eq. 12 (low-mass slope delta_alpha per [Fe/H]). Batch-1 ✅.",
    "imf/environment/coefficients.py::MARKS_TABLE3 mcl/mecl/rho/feh 1-D relations":
        "Marks (2012) Table 3 (p. 2251) — matches cell-for-cell: M_cl(-0.94/2.14/0.68), "
        "M_ecl(-0.77/1.59/0.27), rho_cl(-0.43/1.86/0.095), [Fe/H](0.66/2.63/<-0.5). Batch-1 ✅.",
    "imf/environment/coefficients.py::DEFAULT_SFE=0.33":
        "Jerabkova (2018) fiducial star-formation efficiency epsilon = 0.33. Batch-1.",
    "imf/environment/mapping.py::alpha3 clip [0.5, 2.3] + canonical 2.3":
        "Jerabkova (2018) Eq. 6 / Kroupa (2001) canonical high-mass slope 2.3; the clip "
        "bounds alpha3 to the physical [0.5, canonical] range. Batch-1 ✅ (consumer of "
        "coefficients.py).",

    # ============================ Binaries: Moe & Di Stefano (2017) ==================
    "imf/binary/moe_di_stefano.py::Table 13 grids (gamma_largeq, gamma_smallq, F_twin, f_logP)":
        "Moe & Di Stefano (2017) ApJS 230, 15, Table 13 (p. 52) — ALL 80 cells match exactly "
        "(incl. the '<0.03 -> 0' twin convention and the -1.1/-2.0 tail). Batch-2 ✅.",
    "imf/binary/moe_di_stefano.py::_MASS_NODES [1.0,3.2,6.7,12,20], _LOGP_NODES [1,3,5,7]":
        "Moe (2017) Table 13 representative mass bins (Solar/A-lateB/MidB/EarlyB/O) and "
        "logP rows. Batch-2 ✅ (the interpolation node grid of Table 13).",
    "binaries/eccentricity.py::Moe eta Eq.17 (0.6-0.7/(logP-0.5)) + Eq.18 (0.9-0.2/(logP-0.5))":
        "Moe & Di Stefano (2017) Eqs. 17/18 — late-type (0.8<M1<3 Msun) and early-type "
        "(M1>7 Msun) eccentricity power-law index eta(logP). Batch-2 ✅ (reproduces Table 13 eta).",

    # ============================ Binaries: Sana / Lucy / mass-ratio =================
    "binaries/period.py::Sana pi=-0.55, range logP in [0.15, 3.5]":
        "Sana et al. (2012) Science 337, 444 — OB period distribution p(logP) ∝ (logP)^pi, "
        "pi=-0.55±0.22 (main text + Fig. 2). Range logP [0.15, 3.5] is figure-read (cite "
        "'Fig. 2' -> SOM; ledger Batch-5 low-priority hygiene note). Value Batch-5 ✅.",
    "imf/binary/mass_ratio.py::Sana q-slope kappa ≈ -0.1":
        "Sana et al. (2012) Science 337, 444 — main text & Fig. 1 (Science Report has NO "
        "numbered equations; corrected from the stale 'Eq. 3' cite). Batch-5 🔧 fixed.",
    "imf/binary/mass_ratio.py::TwinPeaked f_twin=0.1, sigma_twin=0.03":
        "Lucy (2006) A&A 457, 629 — systematic statistical study of the q≈1 twin excess "
        "(F_twin≈0.1; the strong-twin hypothesis traces to Lucy & Ricco 1979). Batch-4 🔧 "
        "(relabeled 'systematic' — dropped the over-claimed 'First').",

    # ============================ Diagnostics (CW04 Q / M&C estimator) ===============
    "diagnostics/substructure.py::CW04 Table 1 radial Q (3D0/3D1/3D2 = 0.79/0.84/0.93)":
        "Cartwright & Whitworth (2004, CW04) MNRAS 348, 589, Table 1 (p. 590) — exact (3D1 "
        "corrected 0.03->0.02). The estimator's area convention A=πR² reproduces the RADIAL "
        "Q to <0.01; fractal Q is the estimator output (offset ~0.02-0.03). Batch-3 ✅.",
    "diagnostics/segregation_approx.py::Sigma=(k-1)/(pi r_k^2), k=6":
        "von Hoerner (1963) / Casertano & Hut (1985), as adopted by Maschberger & Clarke "
        "(2011) MNRAS 416, 541, Eq. 4 (p. 544) — kNN surface-density estimator + k=6 choice. "
        "Batch-5 🔧 (re-credited to the upstream von Hoerner / Casertano & Hut origin).",

    # ============================ Stellar radii (Demircan & Kahraman 1991) ===========
    "builders.py::compute_stellar_radii 1.06/0.945, 1.33/0.555, knee 1.66":
        "Demircan & Kahraman (1991) Ap&SS 181, 313, Table II (EMPIRICAL R=10^a M^b; "
        "a=0.026/0.124, b=0.945/0.555) + Section 4 knee 1.66±0.08 Msun. Batch-4 ✅ verified "
        "exact; 🔧 relabeled 'empirical' (not 'ZAMS').",

    # ============================ ZAMS relations (Tout et al. 1996) ==================
    "stellar.py::_TOUT_L_COEFFS (Tout+1996 Table 1, L(M,Z))":
        "Tout, Pols, Eggleton & Han (1996), MNRAS 281, 257, Table 1 (eq. 3) — 35/35 cells "
        "verified cell-by-cell vs held PDF; see "
        "docs/core-papers/tout1996_zams_coefficients_verified.md + provenance-ledger.md.",
    "stellar.py::_TOUT_R_COEFFS + _TOUT_R_NU (Tout+1996 Table 2, R(M,Z))":
        "Tout, Pols, Eggleton & Han (1996), MNRAS 281, 257, Table 2 (eq. 4) — 40 cells + "
        "Z-independent scalar ν verified cell-by-cell vs held PDF; see "
        "docs/core-papers/tout1996_zams_coefficients_verified.md + provenance-ledger.md. "
        "M2: Z is clipped to [1e-4, 0.03] per Tout p. 262 (extrapolation forbidden — the "
        "rational functions go negative); dL/dZ is therefore zero outside that band "
        "(Z-band-limited differentiability).",

    # ============================ Analytical ICs (IAU masses, figure-eight) ==========
    "analytical/base.py::planet/Sun mass ratios (IAU 2009 / Luzum et al. 2011 Table 1)":
        "IAU (2009) Current Best Estimates / Luzum et al. (2011) Cel. Mech. Dyn. Astron. "
        "109, 293, Table 1 — all 8 M_sun/M_planet reciprocals verified (Earth 332946.05; "
        "Jupiter 1.047348644e3; etc.). Batch-4 ✅.",
    "analytical/base.py::planet orbital elements (a,e,inc,Omega,omega)":
        "Standish & Williams (2012) / JPL Horizons J2000 — source NOT held; accepted as "
        "standard J2000 (Anna's Batch-4 call; noted source-not-held in the code comment).",
    "analytical/few_body.py::figure-eight period T0 = 6.32591398":
        "Chenciner & Montgomery (2000) Ann. Math. 152, 881 — dimensionless (G=1, m=1, "
        "scale=1) figure-eight choreography period; canonical Chenciner-Montgomery-Simo IC. "
        "Ledger: figure-eight cite already correct (NOT the debunked Aarseth 1974 mis-cite).",
}

# --- ALLOWLIST_MODULES: the SCOPED constant-bearing files the new-literal ratchet covers -
# These are the densest paper-coefficient modules — the "handful that hold citable
# coefficients" (C6 scope). Each citable-shaped literal in them must be a PROVENANCE value,
# carry an inline citation comment, or be in ALLOWLIST_NON_COEFFICIENT below. A NEW
# unprovenanced literal here reds CI. (Algorithm-heavy modules — limepy_tables.py grid
# sizes, builders.py sampling/unit-conversion, plummer.py CDF exponents — are deliberately
# OUT of scope: their literals are numerical-method choices, not citable coefficients, and
# would be a false-positive trap. Their cited constants still live in PROVENANCE above for
# the ledger-consistency ratchet.)
ALLOWLIST_MODULES: tuple[str, ...] = (
    "src/progenax/imf/power_law.py",
    "src/progenax/imf/chabrier.py",
    "src/progenax/imf/smooth.py",
    "src/progenax/imf/environment/coefficients.py",
    "src/progenax/imf/environment/mapping.py",
    "src/progenax/imf/binary/moe_di_stefano.py",
    "src/progenax/imf/binary/mass_ratio.py",
    "src/progenax/binaries/period.py",
    "src/progenax/stellar.py",
)

# --- ALLOWLIST_NON_COEFFICIENT: per-module literals that are NOT citable coefficients -----
# Numerical-method / range-default / guard / interpolation-node literals. Named with a
# reason so the carve is auditable (NOT a silent suppression). The scanner skips ONLY these
# (plus the trivial set in the test). A genuinely-new coefficient is NOT in here, so it reds.
ALLOWLIST_NON_COEFFICIENT: dict[str, dict[float, str]] = {
    "src/progenax/imf/power_law.py": {
        0.01: "Kroupa-family default m_min [Msun] (sampling range, not a fitted coefficient)",
        0.1: "Salpeter default m_min [Msun] (range default; ledger notes it is outside the "
             "fitted ~0.4-10 Msun range — a hygiene note, not a coefficient)",
        100.0: "default m_max [Msun] (sampling range default, not a fitted coefficient)",
    },
    "src/progenax/imf/chabrier.py": {
        100.0: "default m_max [Msun] (range default)",
        30.0: "internal sampling/grid size (numerical-method literal)",
        4000.0: "shared-grid CDF n_points (numerical-method literal)",
        10.0: "log base / decade arithmetic (jnp.log(10.0)); not a coefficient",
    },
    "src/progenax/imf/smooth.py": {
        0.01: "Maschberger fiducial lower limit m_l [Msun] — IS cited (Maschberger 2013) "
              "inline; listed here only because the bare default also appears in the "
              "Schechter/Larson helper defaults where it is a range default",
        0.3: "Larson-form turnover-mass default m_peak [Msun] (helper default, not a "
             "Maschberger-table coefficient)",
        100.0: "default m_max / m_star [Msun] (range default)",
        300.0: "internal grid size (numerical-method literal)",
        4000.0: "shared-grid CDF n_points (numerical-method literal)",
    },
    "src/progenax/imf/environment/coefficients.py": {
        # (none — every citable literal here is a Marks/Jerabkova coefficient in PROVENANCE)
    },
    "src/progenax/imf/environment/mapping.py": {
        # The 2.3 / [0.5, 2.3] clip is the Kroupa-canonical bound (in PROVENANCE). The one
        # residual literal is a numerical-smoothing default, not a paper coefficient:
        0.2: "smooth_width default — the sigmoid transition half-width for the soft alpha3 "
             "step (a numerical-smoothing knob, not a Marks/Jerabkova coefficient)",
    },
    "src/progenax/imf/binary/moe_di_stefano.py": {
        257.0: "MoeJointOrbit n_grid (inverse-CDF grid resolution; numerical-method literal)",
        512.0: "period/q n_grid (inverse-CDF grid resolution; numerical-method literal)",
    },
    "src/progenax/imf/binary/mass_ratio.py": {
        20.0: "Newton-iteration count in a fori_loop (numerical-method literal, not a coeff)",
    },
    "src/progenax/binaries/period.py": {
        4.8: "LogNormal/DM91 default mean log-period mu (a distribution default surfaced in "
             "the file; cited to Duquennoy & Mayor 1991 in the class docstring)",
        8.0: "MoePeriod logP upper bound [days] (sampling-range default)",
        10.0: "log base / decade arithmetic; not a coefficient",
    },
    "src/progenax/stellar.py": {
        # Tout L(M) exponents (eq. 1 formula structure, NOT fitted coefficients):
        5.5: "Tout L(M) numerator mass exponent M**5.5 (eq.1 structure, not a fitted coeff)",
        11.0: "Tout L/R mass exponent M**11 (eq.1/2 formula structure, not a fitted coeff)",
        3.0: "Tout L(M) denominator mass exponent M**3 (eq.1 structure, not a fitted coeff)",
        5.0: "Tout L(M) denominator mass exponent M**5 (eq.1 structure, not a fitted coeff)",
        7.0: "Tout L(M) denominator mass exponent M**7 (eq.1 structure, not a fitted coeff)",
        8.0: "Tout L(M) denominator mass exponent M**8 (eq.1 structure, not a fitted coeff)",
        9.5: "Tout L(M) denominator mass exponent M**9.5 (eq.1 structure, not a fitted coeff)",
        # Tout R(M) exponents (eq. 2 formula structure, NOT fitted coefficients):
        2.5: "Tout R(M) numerator mass exponent M**2.5 (eq.2 structure, not a fitted coeff)",
        6.5: "Tout R(M) numerator mass exponent M**6.5 (eq.2 structure, not a fitted coeff)",
        19.0: "Tout R(M) numerator mass exponent M**19 (eq.2 structure, not a fitted coeff)",
        19.5: "Tout R(M) numerator/denominator mass exponent M**19.5 (eq.2 structure, not a coeff)",
        2.0: "Tout R(M) denominator mass exponent M**2 (eq.2 structure, not a fitted coeff)",
        8.5: "Tout R(M) denominator mass exponent M**8.5 (eq.2 structure, not a fitted coeff)",
        18.5: "Tout R(M) denominator mass exponent M**18.5 (eq.2 structure, not a fitted coeff)",
        # Numerical guards (avoid 0/0 and NaN slopes; not paper coefficients):
        1e-10: "denominator guard jnp.maximum(den, 1e-10) (numerical-method literal)",
        1e-15: "inverse-Newton L_target floor jnp.clip(..., 1e-15, 1e8) (guard, not a coeff)",
        1e-30: "inverse-Newton slope/log-density floor (divide-by-zero guard, not a coeff)",
        # Inverse-Newton clip bounds / initial-guess + iteration count (numerical method):
        0.005: "inverse-Newton mass lower clip [Msun] (sampling-domain guard, not a coeff)",
        125.0: "inverse-Newton initial-guess upper clip [Msun] (numerical-method literal)",
        150.0: "inverse-Newton step upper clip [Msun] (numerical-method literal)",
        1e8: "inverse-Newton L_target ceiling jnp.clip(..., 1e-15, 1e8) (guard, not a coeff)",
        20.0: "_INVERSE_NEWTON_ITERS fixed lax.scan length (numerical-method literal)",
        # Solar-metallicity reference + Tout-mandated Z-clip range (cited in PROVENANCE/inline):
        0.02: "_Z_SUN Tout+1996 reference solar metallicity + default Z (cited inline; also "
              "the Z-band centre for the log10(Z/Zsun) basis)",
        1e-4: "Tout+1996 lower Z-clip bound (p.262 extrapolation guard, not a fitted coeff)",
        0.03: "Tout+1996 upper Z-clip bound (p.262 extrapolation guard, not a fitted coeff)",
    },
}

# --- UNPROVENANCED: allowlisted constant -> hole note (REAL holes; Task-5.2 / Anna) -------
# A genuinely-uncited number an allowlisted module asserts. The 2026-06 audit found ZERO
# fabricated values, so this is EMPTY: every allowlisted coefficient ports to a PROVENANCE
# citation, and every non-coefficient literal is named in ALLOWLIST_NON_COEFFICIENT. A NEW
# unsourced number re-populates this and turns the xfail RED (a hole for Anna to adjudicate).
UNPROVENANCED: dict[str, str] = {}
