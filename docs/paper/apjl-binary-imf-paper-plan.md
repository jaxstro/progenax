# ApJL Paper Plan: Binary-Aware IMF Inference for LSST

**Working title:** "Confidently Wrong: Why Ignoring Binaries Biases IMF Inference at LSST Sample Sizes"

**Target journal:** The Astrophysical Journal Letters
**Target length:** ~3,500 words + 3 figures (ApJL limit: 3,500 words or 4 journal pages)
**Authors:** Anna Rosen (SDSU)

---

## Core Thesis

Naive single-star IMF fitting produces a constant systematic bias of ~0.05--0.10 on the high-mass slope alpha. As sample size $N$ increases, posterior credible intervals shrink as $1/\sqrt(N)$ while the bias remains constant. At LSST sample sizes ($N > 10^4$), naive posteriors become *confidently wrong* -- the bias exceeds the credible interval width, meaning the true value falls outside the reported uncertainty. A binary-aware mixture likelihood that marginalizes over Moe & Di Stefano (2017) binary population models eliminates this bias across all environments and sample sizes.

**Why this matters now:** LSST will deliver photometry for thousands of clusters with 10^3--10^5 resolved stars each. If the community applies existing single-star tools (BASE-9, ASteCA) to these datasets, the resulting IMF constraints will be systematically wrong and will appear precise. This paper warns the field and provides the solution.

---

## What Already Exists

### Code

- `progenax/validation/imf/validate_binary_aware_recovery.py` (966 lines, complete)
- Full HMC/NUTS inference via NumPyro with Maschberger IMF + Moe+17 binaries
- 128-point Gauss-Legendre quadrature for mass marginalization
- Four astrophysical environments (Solar, YMC, Low-Z GC, Starburst)
- N-scaling analysis from N=500 to N=30,000

### Results (saved to JSON, reproducible)

- `results/binary_aware_results.json` -- full HMC samples for all environments
- `results/scaling_results.json` -- N-scaling analysis

### Existing Figure

- `plots/binary_aware_recovery.png` -- 4-panel figure (proposal Fig. 3)
  - (a) System MF vs single-star IMF distortion
  - (b) Parameter recovery: naive vs binary-aware
  - (c) Residual posteriors
  - (d) Precision scaling + "confidently wrong" regime

### Key Numerical Results (Solar, alpha_true = 2.30)

| N | Naive bias | Naive CI width | Bias > CI? | Aware bias | Aware CI width |
|---|-----------|---------------|------------|-----------|---------------|
| 500 | -0.045 | 0.278 | No | +0.036 | 0.292 |
| 1,000 | -0.035 | 0.197 | No | +0.055 | 0.207 |
| 3,000 | -0.057 | 0.115 | No | +0.021 | 0.122 |
| 10,000 | -0.082 | 0.060 | **Yes** | +0.004 | 0.066 |
| 30,000 | -0.098 | 0.035 | **Yes** | -0.013 | 0.040 |

Crossover to "confidently wrong" occurs at N ~ 5,000--10,000 -- squarely in the LSST regime.

---

## Figures (3 total)

### Figure 1: Binary Population Model (Methods)

- **(a)** Moe+17 mass-dependent binary fraction f_b(M): step function from 0.22 (M < 0.1 Msun) to 0.90 (M > 10 Msun)
- **(b)** Mass-ratio distribution p(q|M1): power-law + twin excess at q ~ 1, colored by primary mass to show gamma(M1) variation from +0.4 (M-dwarfs) to -0.5 (OB stars)
- **(c)** System MF distortion: single-star IMF (solid) vs observed system MF (dashed) across four environments

**Purpose:** Ground the paper in the Moe+17 observational constraints. Show these are fixed inputs, not free parameters. Showcases progenax implementation.

**Status:** Data exists in progenax. Needs plotting.

### Figure 2: Parameter Recovery Across Environments (Validation)

- **(a)** Recovery plot: true alpha vs recovered alpha for naive (diamonds) and binary-aware (circles), four environments with 95% CIs
- **(b)** Residual posteriors: distribution of (recovered - true) alpha, naive (dashed) vs binary-aware (solid fill), four colors

**Purpose:** Demonstrate the bias exists across all environments and the binary-aware method eliminates it.

**Status:** Exists (current panels b,c). Reformat for ApJL 2-panel layout.

### Figure 3: The Confidently Wrong Regime (THE KEY RESULT -- standalone)

- **Full-width standalone figure**
- X-axis: log N, from 200 to 100,000
- Y-axis: |bias| and CI width (log scale or linear, whichever reads better)
- **All four environments** overlaid (four colors)
- For each environment, show:
  - Solid line: 95% CI width (shrinks as 1/sqrt(N))
  - Dotted/dashed line: |naive bias| (constant)
  - Filled circles: binary-aware |bias| (fluctuates near zero)
- **Shaded region:** "confidently wrong" regime where |bias| > CI width
- **Vertical band:** LSST typical regime (N = 10^3 to 10^5)
- Reference line: theoretical 1/sqrt(N) scaling

**N sample points (11 values, log-spaced):**
N = [200, 500, 1000, 2000, 5000, 10000, 20000, 30000, 50000, 75000, 100000]

(Keep 30,000 since results already exist. New runs: 200, 2000, 5000, 20000, 50000, 75000, 100000. Reuse: 500, 1000, 3000→2000 optional, 10000, 30000.)

**Purpose:** This is the headline result. Shows the crossover is inevitable and universal across environments. At N=100,000 the naive bias is 8-10x the CI width -- catastrophic. The figure should be immediately comprehensible and memorable.

**Status:** Partially exists (5 N-values, Solar only). Needs: extended N range to 100,000, all four environments, reformatted as standalone.

---

## Paper Outline

### 1. Introduction (~500 words)

- IMF slope alpha is the key observable for testing IMF universality
- LSST will deliver unprecedented sample sizes (10^3--10^5 stars per cluster)
- Standard tools (BASE-9, ASteCA) treat all sources as single stars
- Unresolved binaries shift observed mass function systematically
- This bias is *constant* -- it does not shrink with sample size
- As posteriors sharpen with more data, they become confidently wrong
- We demonstrate this quantitatively and provide a binary-aware solution

Key citations: Kroupa (2001), Bastian+ (2010), Moe & Di Stefano (2017), Ivezic+ (2019), Stenning+ (2016), Perren+ (2015)

### 2. Method (~800 words)

#### 2.1 IMF Model

- Maschberger (2013) smooth parameterization
- Why smooth > piecewise: avoids breakpoint artifacts in likelihood
- Brief note on environment-dependent alpha via Jerabkova+ (2018)

#### 2.2 Binary Population Model (Figure 1)

- Moe & Di Stefano (2017) mass-dependent binary fraction:
  - f_b(M): 0.22 (M < 0.1 Msun) to 0.90 (M > 10 Msun)
- Mass-ratio distribution: power-law + twin excess at q ~ 1
  - gamma(M1): varies from +0.4 (M-dwarfs) to -0.5 (OB stars)
  - f_twin(M1): 0.03--0.10
- These are *observational constraints*, not free parameters

#### 2.3 Observation Operator

- Mass-addition: m_sys = m1 + q*m1 (worst-case bound)
- Why worst case: no photometric information used, maximum confusion with high-mass singles
- Real LSST photometry has smaller bias (0.75 mag shift, ~20% mass shift rather than 100%) -- structurally identical, quantitatively smaller
- This means our results are conservative

#### 2.4 Inference

- Binary-aware mixture likelihood:
  p(m_sys | alpha) = (1 - f_b) * xi(m_sys; alpha) + integral term
- Integral over primary mass m1 via 128-point Gauss-Legendre quadrature
- Naive model: treat m_sys as single-star draw from xi(m; alpha)
- Both models inferred via HMC/NUTS (NumPyro), 2 chains x 1000 samples
- Convergence diagnostics: ESS > 200, R-hat < 1.05

### 3. Results (~800 words)

#### 3.1 System MF Distortion (Figure 1c)

- Binaries shift mass function to higher apparent masses
- Distortion depends on environment (higher alpha = more low-mass stars = larger relative binary contribution at high mass)
- Effect is 5--15% at M > 1 Msun

#### 3.2 Bias on Alpha (Figure 2)

- Naive fit: systematic negative bias of 0.05--0.10 across all four environments
- Worst in solar-metallicity models (sparse sampling above 1 Msun breakpoint)
- Binary-aware fit: recovers true alpha to within 0.03 in all environments
- Residual posteriors confirm unbiased recovery (centered at zero)

#### 3.3 The Confidently Wrong Regime (Figure 3) -- THE KEY RESULT

- Vary N from 200 to 100,000 across all four environments
- Both methods: CI width shrinks as 1/sqrt(N) (expected)
- Naive: |bias| remains constant at ~0.05--0.10 (depends on environment)
- Binary-aware: |bias| fluctuates around zero at all N
- Crossover at N ~ 3,000--10,000 (environment-dependent): naive |bias| > naive CI width
- Beyond crossover: the true alpha falls outside the naive 95% CI
- At N = 100,000: naive bias is 8--10x the CI width -- catastrophic
- LSST clusters span N = 10^3--10^5 -- deep into this regime
- The crossover is universal: all four environments enter the confidently wrong regime at or below LSST sample sizes

### 4. Discussion (~800 words)

#### 4.1 Implications for Existing Results

- Published IMF slopes from single-star fitting (BASE-9, ASteCA) may carry ~0.1 systematic bias
- For small-N studies (N < 1000), this bias is absorbed into statistical uncertainty -- hidden
- As LSST enables large-N studies, this hidden bias becomes a dominant systematic
- Literature values of alpha should be revisited with binary-aware methods

#### 4.2 Why Mass-Addition is a Worst-Case Bound

- Real LSST photometry: unresolved binaries contribute combined *flux* (not mass)
- Equal-mass binary: 0.75 mag brighter, ~20% mass shift (vs 100% in mass-addition)
- Photometric bias is smaller but structurally identical
- Our mass-addition results are therefore conservative
- The full photometric case will be demonstrated with jaxstro (in prep)

#### 4.3 Path Forward

- Binary-aware inference requires:
  1. A binary population model (provided: Moe+17 in progenax)
  2. Marginalizing over binary parameters in the likelihood (demonstrated here)
  3. A differentiable forward model for efficiency (jaxstro pipeline, in development)
- The marginalization architecture transfers directly to photometric CMD fitting -- only the observation operator changes
- Full end-to-end pipeline (mass -> stellar evolution -> photometry -> likelihood) under development for LSST DP2

### 5. Summary (~300 words)

- Naive single-star IMF fitting produces constant bias of 0.05--0.10
- At LSST sample sizes, posteriors become confidently wrong
- Binary-aware mixture likelihood eliminates this bias
- The community must adopt binary-aware inference for LSST cluster science
- Code publicly available: progenax (Apache 2.0)

---

## What Needs to Be Done

### Already done

- [x] Binary population model (Moe+17 in progenax)
- [x] Maschberger IMF implementation
- [x] Four-environment validation at N=10,000
- [x] HMC inference (naive + binary-aware)
- [x] N-scaling analysis (N=500 to N=30,000, Solar only)
- [x] 4-panel figure (proposal version)
- [x] Numerical results for all environments

### New analysis needed

- [ ] **Extended N-scaling (Figure 3):** Rerun scaling analysis for N = [200, 2000, 5000, 20000, 50000, 75000, 100000] -- keep existing N = [500, 1000, 3000, 10000, 30000]. Run for **all four environments**, not just Solar. This is the same script with different N values -- straightforward rerun.
- [ ] **Figure 1 (Moe+17 model):** Plot the binary population model parameters (f_b(M), p(q|M1), system MF distortion). Data exists in progenax; needs plotting only.
- [ ] **Figure 2 (recovery):** Reformat existing panels (b,c) from 4-panel figure into standalone 2-panel.
- [ ] **Figure 3 (confidently wrong):** New standalone figure with extended N range and all four environments.

### Writing needed

- [ ] Draft text (~3,500 words)
- [ ] Abstract (~150 words)
- [ ] Figure formatting for ApJL style (AASTeX two-column)
- [ ] References

---

## Compute Estimate for Extended Scaling

Current runs: 2 chains x 500 warmup x 1000 samples per (N, environment, method) combination.

New runs needed: 7 new N values x 4 environments x 2 methods (naive + aware) = 56 HMC runs.
At ~1--5 min per run (depending on N), total wall time: ~1--5 hours on GPU.
Can parallelize across environments/methods.

---

## Strategic Considerations

### Timing

- Submit after LSST-DA proposal deadline (Feb 17) -- no rush to beat that
- Target submission: Spring 2026 (before DP2 data arrives July 2026)
- Establishes priority on binary-aware differentiable IMF inference
- Creates a citable reference for the jaxstro methods paper

### Framing

- Frame as a **warning + solution**, not just a methods paper
- The "confidently wrong" framing is the hook -- lead with it
- Emphasize that this affects *all* existing single-star IMF studies at large N
- Position progenax as the solution, with jaxstro as the full photometric pipeline (forthcoming)

### Scope discipline

- This is a *Letter* -- resist expanding to a full paper
- The core argument (constant bias + shrinking CI = confidently wrong) is self-contained
- Save the full photometric pipeline, end-to-end CMD fitting, and multi-cluster analysis for the jaxstro methods paper
- The Letter plants the flag; the methods paper builds the house

---

## Key References

- Bastian, Covey & Meyer (2010) -- IMF universality review
- Kroupa (2001) -- Canonical IMF
- Maschberger (2013) -- Smooth IMF parameterization
- Moe & Di Stefano (2017) -- Binary population statistics
- Jerabkova et al. (2018) -- Environment-dependent IMF
- Marks et al. (2012) -- IMF fundamental plane
- Stenning et al. (2016) -- BASE-9 methodology
- Perren et al. (2015) -- ASteCA methodology
- Ivezic et al. (2019) -- LSST reference
- Betancourt (2018) -- HMC conceptual introduction
- Usher et al. (2023) -- SMWLV Star Clusters Roadmap
