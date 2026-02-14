# Binary-Aware IMF Recovery: Theory, Method, and Limitations

> A pedagogical guide to the binary-contamination problem in stellar initial mass function inference, and how to solve it. This document is the single source of truth for the `progenax` binary-aware IMF recovery pipeline.

---

## 1. The Initial Mass Function

The **stellar initial mass function** (IMF) describes the distribution of stellar masses at birth. It governs chemical enrichment, supernova rates, and the integrated light of galaxies. Measuring its shape — and whether it varies with environment — is one of the central problems in stellar astrophysics.

### 1.1 The Maschberger (2013) functional form

The Maschberger (2013) IMF provides a smooth, analytically invertible form (the "L3 distribution"):

$$f(m) \propto \left(\frac{m}{\mu}\right)^{-\alpha} \left[1 + \left(\frac{m}{\mu}\right)^{1-\alpha}\right]^{-\beta}$$

Three parameters control the shape:

| Parameter | Default | Meaning |
|-----------|---------|---------|
| $\alpha$ | 2.3 | High-mass power-law slope (Salpeter 1955 value) |
| $\beta$ | 1.4 | Low-mass turnover strength |
| $\mu$ | 0.2 M$_\odot$ | Characteristic (peak) mass |

At high masses ($m \gg \mu$), the second factor approaches 1 and the IMF reduces to a pure power law $f(m) \propto m^{-\alpha}$. At low masses ($m \ll \mu$), the second factor suppresses the distribution, producing the observed turnover below ~0.3 M$_\odot$.

A key advantage of this form: its cumulative distribution function has a closed-form inverse, enabling exact analytical sampling without numerical root-finding (Maschberger 2013, Section 3). For a uniform variate $u \sim U[0,1]$, the mass is:

$$m(u) = \mu \left[\left(\frac{P_{\min} + u(P_{\max} - P_{\min})}{C}\right)^{1/(1-\beta)} - 1\right]^{1/(1-\alpha)}$$

where $C = \mu / [(1-\beta)(1-\alpha)]$ and $P_{\min}, P_{\max}$ are the primitive evaluated at the mass bounds.

### 1.2 Single-star IMF vs. system IMF

An important terminological distinction: the **single-star IMF** (also called the "individual-star IMF") describes the birth mass distribution of individual stars. The **system IMF** describes the distribution of observed system masses, including unresolved multiples. These are not the same function. Maschberger (2013, Table 1) explicitly provides separate canonical parameter values for single-star and system IMFs, reflecting the fact that unresolved binaries alter the observed mass distribution.

Throughout this document, $\xi(m \mid \alpha)$ denotes the **single-star (primary) IMF**. The system mass function is derived from it via the binary contamination model described below.

### 1.3 Environment-dependent IMF

The environment-dependent IMF hypothesis (Marks et al. 2012; Jerabkova et al. 2018) predicts that $\alpha$ varies with star-forming conditions: steeper ($\alpha \approx 2.3$) in quiescent environments like the Solar neighborhood, shallower ($\alpha \approx 1.6$) in extreme starbursts. Measuring $\alpha$ accurately is therefore central to testing IMF universality.

---

## 2. The Binary Contamination Problem

### 2.1 What observers actually measure

Stars do not form in isolation. A large fraction form in binary or higher-order multiple systems (Duchene & Kraus 2013). When a binary system is unresolved (as most are in crowded cluster fields), the observer measures the **system mass** rather than the individual stellar masses:

$$m_{\text{sys}} = \begin{cases} m_1 & \text{if single (probability } 1 - f_b(m_1)\text{)} \\ m_1 + m_2 & \text{if binary (probability } f_b(m_1)\text{)} \end{cases}$$

where $m_1$ is the primary mass, $m_2 = q \cdot m_1$ is the secondary mass, $q$ is the mass ratio, and $f_b(m_1)$ is the binary fraction.

### 2.2 Why this biases the IMF slope

Binaries transfer probability mass from intermediate to high stellar masses. A binary with $m_1 = 5\,M_\odot$ and $q = 0.5$ appears as a single $7.5\,M_\odot$ system. This excess at high masses makes the observed mass function appear **shallower** (smaller $\alpha$) than the true single-star IMF.

If one naively fits a single-star IMF to the system mass function, the recovered $\alpha$ will be **biased low**. The bias is systematic and always negative. Kroupa (2001) explicitly quantified this: unresolved binaries can bias inferred power-law indices by $\Delta\alpha \sim 0.05$--$0.1$, depending on mass range and binary fraction.

### 2.3 The naive approach

Most IMF studies fit the observed mass function directly:

$$\mathcal{L}_{\text{naive}}(\alpha \mid \{m_{\text{sys},i}\}) = \prod_{i=1}^{N} \xi(m_{\text{sys},i} \mid \alpha)$$

where $\xi(m \mid \alpha)$ is the single-star IMF. This assumes every observed mass is a single star. The resulting posterior converges to the **wrong value** as sample size grows.

---

## 3. The Moe & Di Stefano (2017) Binary Statistics

### 3.1 Overview

Moe & Di Stefano (2017, ApJS 230, 15) compiled the most comprehensive census of binary star properties to date. They combined observations from spectroscopy, eclipsing binaries, long-baseline interferometry, adaptive optics, and common proper motion surveys, then corrected each sample for its selection effects to derive the intrinsic joint distribution $f(M_1, q, P, e)$ of primary mass, mass ratio, orbital period, and eccentricity.

Two key findings:

1. **Binary properties are not universal.** The mass-ratio distribution, binary fraction, and period distribution all vary systematically with primary mass. Previous studies that assumed mass-independent binary statistics introduced systematic errors.

2. **The joint distribution is not separable:**

$$f(M_1, q, P, e) \neq f(M_1) \cdot f(q) \cdot f(P) \cdot f(e)$$

The mass-ratio distribution depends on both $M_1$ and $P$; the eccentricity distribution depends on $P$; the companion frequency depends on both $M_1$ and $P$. Treating these as independent factors introduces correlated errors.

### 3.2 Companion frequency vs. binary fraction

Moe & Di Stefano are careful about terminology, and we should be too:

- **Companion frequency** $f_{\log P; q>q_{\min}}(M_1, P)$: the mean number of companions per primary **per decade of orbital period** above some mass-ratio threshold. This is the fundamental quantity they measure.

- **Multiplicity frequency** $f_{\text{mult}; q>0.1}(M_1)$: the companion frequency integrated over all periods. This can exceed 1.0 — it represents the mean number of companions per primary, not a probability. For O-type stars, $f_{\text{mult}} \sim 2.1$.

- **Binary fraction** (or "multiplicity fraction"): the probability that a star has **at least one** companion. This is $\leq 1$ by definition, and is related to but distinct from companion frequency.

In simplified models (like ours) that allow at most one companion per primary, the "binary fraction" $f_b(m_1)$ is treated as a probability. This is a deliberate simplification; see Section 9.3 for its implications.

### 3.3 Mass-ratio distribution

For a primary of mass $M_1$, the mass-ratio distribution follows a hybrid model combining a power law with a twin excess:

$$p(q \mid M_1) = (1 - f_{\text{twin}}) \cdot \frac{q^{\gamma(M_1)}}{Z_{\text{pl}}} + f_{\text{twin}} \cdot \mathcal{N}(q \mid 1, \sigma_{\text{twin}})$$

where $Z_{\text{pl}} = \int_{q_{\min}}^{1} q^{\gamma} \, dq$ is the power-law normalization.

The two components:

1. **Power-law component** ($q^\gamma$): The bulk of the distribution. In the full MDS17 treatment, $\gamma$ is actually a **broken power law** with two slopes:

   - $\gamma_{\text{small-}q}$ for $0.1 < q < 0.3$
   - $\gamma_{\text{large-}q}$ for $0.3 < q < 1.0$

   joined continuously at $q = 0.3$. Both slopes depend on $M_1$ and $P$. Our simplified implementation uses a single effective $\gamma$ per mass bin (the period-averaged $\gamma_{\text{large-}q}$), which is appropriate for a total mass-function analysis but loses the small-$q$ structure.

   The period-averaged values from MDS17 Table 10:

   | Primary Mass | $\gamma$ | Behavior |
   |-------------|----------|----------|
   | $M_1 < 0.8\,M_\odot$ (M-dwarfs) | 0.4 | Preference for near-equal masses |
   | $0.8 < M_1 < 1.2\,M_\odot$ (Solar-type) | 0.3 | Mild preference for equal masses |
   | $1.2 < M_1 < 3.5\,M_\odot$ (A/F stars) | 0.0 | Flat (uniform in $q$) |
   | $M_1 > 3.5\,M_\odot$ (OB stars) | $-0.5$ | Preference for unequal masses |

   The trend reverses at high masses: massive stars preferentially have **low-$q$** companions, while low-mass stars prefer **equal-mass** companions.

2. **Twin excess** ($f_{\text{twin}}$): A narrow Gaussian peak at $q \approx 1$ with width $\sigma_{\text{twin}} \approx 0.03$. This represents an **excess** of near-equal-mass binaries ("twins") beyond what the power-law predicts. The key word is "excess" — $f_{\text{twin}}$ is the fraction of systems with $q > 0.95$ **above** the extrapolated power law, not the total fraction of near-equal-mass systems. The twin fraction varies with mass:

   | Primary Mass | $f_{\text{twin}}$ |
   |-------------|-------------------|
   | $M_1 < 0.8\,M_\odot$ | 0.05 |
   | $0.8 < M_1 < 1.2\,M_\odot$ | **0.10** (peak) |
   | $1.2 < M_1 < 3.5\,M_\odot$ | 0.08 |
   | $M_1 > 3.5\,M_\odot$ | 0.03 |

   Solar-type stars show the strongest twin excess.

### 3.4 Binary fraction

The probability that a star has a companion depends strongly on its mass. Moe & Di Stefano (2017, Table 13) report the **companion frequency** (number of companions per primary, integrated over all periods with $q > 0.1$):

| Primary Mass | $f_b$ | Type |
|-------------|-------|------|
| $M_1 < 0.1\,M_\odot$ | 0.22 | Very low mass / brown dwarfs |
| $0.1$-$0.5\,M_\odot$ | 0.26 | M-dwarfs |
| $0.5$-$1.0\,M_\odot$ | 0.44 | K/G-dwarfs |
| $1.0$-$2.0\,M_\odot$ | 0.50 | F/A-stars |
| $2.0$-$5.0\,M_\odot$ | 0.60 | B-stars |
| $5.0$-$10\,M_\odot$ | 0.80 | Early B |
| $M_1 > 10\,M_\odot$ | 0.90 | O-stars |

The trend is steep: O-type stars are almost always in multiples, while M-dwarfs are mostly single. For O-type stars, the companion frequency exceeds 1.0 ($\sim$2.1 companions per primary), meaning most are in triples or higher-order systems.

### 3.5 Three period regimes

Moe & Di Stefano identify three qualitatively different regimes of orbital period:

1. **Short periods** ($P \lesssim 20$ days): Tidally circularized orbits ($e \lesssim 0.4$), modest mass ratios ($\langle q \rangle \approx 0.5$), small twin excess.

2. **Intermediate periods** ($\log P \approx 3.5$ days, $a \approx 10$ AU): Peak companion frequency. Mass ratios weighted toward small values ($q \approx 0.2$-$0.3$). Thermal eccentricity distribution $f(e) = 2e$.

3. **Long periods** ($\log P \approx 5.5$-$7.5$ days, $a \approx 200$-$5000$ AU): Outer tertiary components in hierarchical triples. Mass-ratio distribution nearly consistent with random pairings drawn from the IMF.

### 3.6 How they determined these distributions

Moe & Di Stefano analyzed dozens of binary samples, each spanning a narrow interval of $M_1$ and $P$. For each sample, they:

1. Identified the relevant selection function (e.g., spectroscopic surveys are biased toward large $q$ and short $P$; visual surveys are biased toward wide separations).
2. Corrected for incompleteness using the known selection function of each survey technique.
3. Fit the intrinsic mass-ratio distribution as a power law plus twin excess.
4. Identified a ~30% contamination rate from white dwarf companions masquerading as main-sequence binaries in spectroscopic samples, and corrected for it.

They then fit smooth functions to the corrected parameters ($\gamma$, $f_{\text{twin}}$, $f_b$) as functions of $M_1$ and $P$, producing the joint probability density $f(M_1, q, P, e)$ that can be directly sampled in population synthesis codes.

---

## 4. The System Mass Function

### 4.1 Analytical derivation

Given the forward generative model (draw $m_1$ from IMF, decide binary or not, add $m_2$ if binary), the probability density of the observed system mass $M$ is:

$$p_{\text{sys}}(M) = \underbrace{\xi(M)\,(1 - f_b(M))}_{\text{single-star contribution}} + \underbrace{\int_{M/2}^{\;M/(1+q_{\min})} \frac{\xi(m_1)\, f_b(m_1)\, g\!\left(\frac{M}{m_1} - 1 \;\middle|\; m_1\right)}{m_1}\, dm_1}_{\text{binary contribution}}$$

**Derivation of the binary term.** For a binary with primary mass $m_1$ and mass ratio $q$:

$$m_{\text{sys}} = m_1(1 + q) = M$$

Solving for the mass ratio: $q = M/m_1 - 1$, with Jacobian $|dq/dM| = 1/m_1$.

The integration limits come from the constraint $q_{\min} \leq q \leq 1$:
- Upper limit on $m_1$: $q \geq q_{\min} \implies m_1 \leq M / (1 + q_{\min})$
- Lower limit on $m_1$: $q \leq 1 \implies m_1 \geq M / 2$

So the binary contribution integrates over all primary masses $m_1$ that could produce the observed system mass $M$, weighted by the IMF, binary fraction, and mass-ratio distribution.

### 4.2 Why no closed-form solution exists

This integral has no analytical solution because $f_b(m_1)$ and $g(q \mid m_1)$ from Moe & Di Stefano (2017) are piecewise functions with mass-dependent parameters. Numerical quadrature is required.

---

## 5. The Binary-Aware Likelihood

### 5.1 Formulation

The binary-aware likelihood correctly marginalizes over the latent binary status and mass ratio of each observed system. For a single observed system mass $M$:

$$p(M \mid \alpha) = \underbrace{(1 - \bar{f}_b) \cdot \xi(M \mid \alpha)}_{\text{single-star term}} + \underbrace{\bar{f}_b \cdot \int_{q_{\min}}^{1} \frac{\xi\!\left(\frac{M}{1+q} \;\middle|\; \alpha\right)}{1+q} \cdot g\!\left(q \;\middle|\; \frac{M}{1+q}\right) dq}_{\text{binary term}}$$

where $\bar{f}_b$ is the average binary fraction (or more precisely, $f_b(m_1)$ evaluated at each candidate primary mass $m_1 = M/(1+q)$ inside the integral).

**Interpretation:** For each system mass $M$, we sum over two hypotheses:
1. It is a single star of mass $M$, with probability $1 - f_b(M)$, contributing $\xi(M)$.
2. It is a binary, for every possible mass ratio $q \in [q_{\min}, 1]$: the primary had mass $m_1 = M/(1+q)$, the mass ratio was $q$, and the probability density involves the IMF at $m_1$, the mass-ratio distribution $g(q \mid m_1)$, and the Jacobian $1/(1+q)$ from the change of variables $m_{\text{sys}} = m_1(1+q)$.

### 5.2 Numerical evaluation

The integral over $q$ has no closed form (because $g(q \mid m_1)$ from Moe+17 changes functional form with $m_1$). We evaluate it using **128-point Gauss-Legendre quadrature**, which gives ~14 digits of accuracy for smooth integrands:

$$\int_{q_{\min}}^{1} h(q)\, dq \approx \frac{1 - q_{\min}}{2} \sum_{k=1}^{128} w_k \cdot h\!\left(\frac{(1 - q_{\min}) x_k + (1 + q_{\min})}{2}\right)$$

where $\{x_k, w_k\}$ are the Gauss-Legendre nodes and weights on $[-1, 1]$.

### 5.3 Computational cost

For $N$ observed systems and 128 quadrature points, each likelihood evaluation requires $N \times 128$ evaluations of the IMF and mass-ratio distribution. At $N = 30{,}000$ (LSST-scale), this is $\sim$3.8 million evaluations per NUTS step, with ~1500 NUTS steps per chain. Total: ~$6 \times 10^9$ floating-point operations per chain.

On a MacBook Pro CPU, this takes ~35 minutes for $N = 30{,}000$. On a GPU (A100), the $N \times 128$ evaluations are embarrassingly parallel via `jax.vmap`, yielding an estimated ~100-1000$\times$ speedup.

### 5.4 The full log-likelihood

For $N$ independent observations:

$$\ln \mathcal{L}(\alpha) = \sum_{i=1}^{N} \ln p(M_i \mid \alpha)$$

This is passed to NumPyro NUTS (No-U-Turn Sampler) as a `numpyro.factor` statement with a `Uniform(0.5, 4.0)` prior on $\alpha$.

---

## 6. Naive vs. Binary-Aware Inference

### 6.1 The naive model

Assumes all systems are single stars:

$$\ln \mathcal{L}_{\text{naive}}(\alpha) = \sum_{i=1}^{N} \ln \xi(M_i \mid \alpha)$$

This is a **misspecified model**: the data-generating process includes binaries, but the likelihood ignores them.

### 6.2 What happens as $N$ grows

Both models produce posterior distributions that shrink as $\sigma \propto 1/\sqrt{N}$ (the Bernstein-von Mises theorem). But:

- **Binary-aware:** The posterior shrinks around the **true** $\alpha$. The bias $|\hat\alpha - \alpha_{\text{true}}|$ decreases with $N$.
- **Naive:** The posterior shrinks around a **wrong** value. The bias stays constant (it is a property of the model, not the sample size). At large $N$, the 95% credible interval becomes **narrower than the bias**, so the posterior **excludes the true value**.

This is the "**confidently wrong**" regime. It is the central danger of ignoring binaries with large datasets.

| $N$ | Naive 95% CI width | Naive $|\text{bias}|$ | Status |
|-----|--------------------|-----------------------|--------|
| 500 | 0.28 | 0.045 | CI contains truth |
| 1,000 | 0.20 | 0.035 | CI contains truth |
| 3,000 | 0.12 | 0.057 | CI contains truth (barely) |
| 10,000 | 0.06 | **0.082** | **CI excludes truth** |
| 30,000 | 0.035 | **0.098** | **Confidently wrong** |

---

## 7. Data Generation: Forward Model

The validation uses a forward generative model to create synthetic observations:

1. **Draw primary masses** from the Maschberger IMF: $m_1 \sim \xi(m \mid \alpha)$
2. **Assign binary status** using the mass-dependent binary fraction: binary with probability $f_b(m_1)$ from Moe+17 Table 13
3. **Draw mass ratio** (if binary): $q \sim g(q \mid m_1)$ from Moe+17 Table 10
4. **Compute system mass**: $m_{\text{sys}} = m_1(1 + q)$ if binary, $m_1$ if single

This is the standard approach used in population synthesis codes (e.g., BPASS, Eldridge et al. 2017; COSMIC, Breivik et al. 2020; McLuster, Kupper et al. 2011). The "sample primary first, then conditionally generate companions" structure matches how Moe+17's empirical constraints are parameterized — they are conditional on primary mass.

Note: all system masses satisfy $m_{\text{sys}} \geq m_{\min}$ by construction. Singles have $m_{\text{sys}} = m_1 \geq m_{\min}$. Binaries have $m_{\text{sys}} = m_1(1+q) > m_1 \geq m_{\min}$.

---

## 8. The Observation Operator: What Does the Telescope Measure?

The forward model in Sections 4--7 uses the observation operator $m_{\text{sys}} = m_1 + m_2$. This is the **true dynamical mass** of the system. But different observational techniques measure different things, and the choice of observation operator determines the size and shape of the binary distortion.

### 8.1 Dynamical mass ($m_{\text{sys}} = m_1 + m_2$)

For eclipsing binaries, SB2 orbital solutions, and astrometric binaries, the measured quantity is the true system mass (or individual component masses). In this regime, mass addition is physically correct.

Our current implementation uses this operator. It gives the **maximum possible distortion** of the mass function — an upper bound on the binary contamination bias.

### 8.2 Photometric mass (flux-inferred)

Many published "mass functions" derive masses from **photometry** using a mass-luminosity relation. For an unresolved binary, the telescope measures combined flux, not combined mass:

$$F_{b,\text{sys}} = F_b(m_1) + F_b(m_2) \quad \text{for each band } b$$

The observer then inverts this total flux through a single-star mass-luminosity relation to get a "photometric mass":

$$m_{\text{phot}} = \mathcal{M}^{-1}\!\left(F_{\text{sys}}\right)$$

For a rough main-sequence scaling $L \propto m^s$ with $s \approx 3$--$4$:

$$m_{\text{phot}} \approx m_1 \cdot (1 + q^s)^{1/s}$$

The key insight: because $s$ is large, the secondary contributes little flux unless $q$ is near 1. Concrete examples:

| Mass ratio $q$ | Mass addition: $m_1 + m_2$ | Photometric mass ($s=3.5$) | Difference |
|------|------|------|------|
| 1.0 (equal) | $2.0\,m_1$ | $1.22\,m_1$ | $\Delta m = 0.75$ mag brighter |
| 0.5 | $1.5\,m_1$ | $1.04\,m_1$ | Companion barely changes flux |
| 0.3 | $1.3\,m_1$ | $1.01\,m_1$ | Negligible photometric effect |

**Implication:** The mass-addition observation operator ($m_1 + m_2$) gives the worst-case distortion. Photometric masses are much less affected by binaries, especially for moderate $q$. But the bias is still nonzero and systematic — it does not vanish, and it still causes the naive model to become confidently wrong at large $N$.

### 8.3 Multi-band photometry (LSST)

LSST does not measure "mass" at all. It measures **multi-band fluxes** (or magnitudes) with uncertainties, plus incompleteness and crowding effects. For an unresolved binary:

$$m_{b,\text{sys}} = -2.5\,\log_{10}\!\left[F_b(m_1) + F_b(m_2)\right] + \text{zeropoint} \quad \text{for each } b \in \{u, g, r, i, z, y\}$$

This is **not equivalent** to saying "the observed mass is $m_1 + m_2$." The mapping from (multi-band magnitudes) → (inferred mass) involves:

1. Comparison to theoretical isochrones at assumed age $\tau$, metallicity $Z$, and distance $d$
2. Extinction correction (dust column $A_V$ and reddening law $R_V$)
3. Color-magnitude diagram (CMD) fitting

Each of these steps introduces additional model dependence. The binary distortion in CMD space is qualitatively different from mass-space: an unresolved equal-mass binary appears 0.75 mag brighter at nearly the same color, placing it **above** the main sequence in the CMD. This mimics a more massive or more evolved star.

---

## 9. Limitations and Caveats

### 9.1 Mass addition as an upper bound

As discussed in Section 8, our model uses $m_{\text{sys}} = m_1 + m_2$. This is correct for dynamical mass measurements but overestimates the distortion for photometric surveys. The validation figure (panel d) demonstrates that model mismatch causes bias, but the **magnitude** of the bias in a real LSST analysis would depend on the photometric observation operator.

### 9.2 Known binary statistics

We assume perfect knowledge of $f_b(m_1)$ and $g(q \mid m_1)$. In our validation, the data-generating process and the likelihood model use identical Moe+17 statistics, so the binary-aware inference is unbiased by construction.

In real applications, uncertainty in the binary statistics propagates into uncertainty in $\alpha$. The Moe+17 binary fractions carry uncertainties of $\pm 0.04$ for solar-type stars, growing to $\pm 0.10$ for O-stars. Misspecifying these would reintroduce bias, though typically smaller than ignoring binaries entirely.

A natural extension: jointly infer $\alpha$ and binary parameters (e.g., the overall binary fraction normalization). This is planned for v0.4 of progenax.

### 9.3 Higher-order multiples

We model only singles and binaries. In reality, ~10% of solar-type systems are triples, rising to >50% for O-type stars (Moe & Di Stefano 2017, Section 10; Duchene & Kraus 2013). A triple system would contribute $m_{\text{sys}} = m_1 + m_2 + m_3$, further inflating the high-mass tail.

Ignoring triples means we **underestimate** the binary distortion slightly. For Solar-type stars ($\alpha = 2.3$, triple fraction ~10%), this is a small correction. For massive-star-dominated populations, it could matter.

For O-type stars, Moe+17 report a companion frequency of ~2.1 — meaning the typical O-star has 2+ companions. Our model, which allows at most one companion per primary, compresses this into a single binary with an effective mass ratio. This underestimates the system-mass inflation for the most massive systems.

### 9.4 Period-dependent effects

The Moe+17 mass-ratio distribution varies with orbital period (Section 3.5 above). Short-period binaries have different $q$ distributions than wide binaries. Our implementation integrates over all periods (using the period-averaged $\gamma$ and $f_{\text{twin}}$ from Table 10), which is appropriate for a total mass-function analysis but would need refinement for surveys sensitive to specific period ranges (e.g., spectroscopic surveys detect only short-period binaries).

### 9.5 Resolution-dependent binary fraction

Whether a binary is "unresolved" is not a fixed property of the binary — it depends on:

- **Angular separation** (set by orbital period, semi-major axis, and distance to the cluster)
- **Instrument PSF** (ground-based seeing vs. space-based diffraction limit)
- **Crowding and blending** (dense cluster cores vs. sparse fields)
- **Contrast ratio** (faint companions near bright primaries may be undetectable even if resolved)

The effective binary fraction relevant to unresolved photometry is therefore:

$$f_{\text{unresolved}}(m_1) = \int f(m_1, a) \cdot \mathbb{1}[\text{unresolved at } (a, d, \text{PSF})] \, da$$

If you treat all binaries as unresolved (as we do), you overestimate the distortion for nearby or sparse clusters where many wide binaries would be resolved. Conversely, in very distant or crowded fields, even moderately wide binaries are blended.

For a full treatment, the forward model should sample orbital period (or semi-major axis), compute the projected angular separation at the cluster distance, and compare to the instrument resolution. This couples the binary statistics to the cluster distance and instrument model.

### 9.6 Primary-IMF vs. stellar-IMF subtlety

Our procedure draws **primaries** from the IMF and then generates secondaries conditionally. This matches how Moe+17's constraints are parameterized. But the full set of individual stars $\{m_1, m_2\}$ in the resulting population does **not** follow the same IMF as the primaries.

Consider: if you draw $m_1$ from $\xi(m)$ and then generate $m_2 = q \cdot m_1$ with $q \sim g(q \mid m_1)$, the distribution of $m_2$ values is:

$$p(m_2) = \int \xi(m_1) \cdot f_b(m_1) \cdot g(m_2 / m_1 \mid m_1) \cdot \frac{1}{m_1} \, dm_1$$

This is not $\xi(m_2)$ in general. For instance, if $g(q)$ peaks at $q \approx 1$ (twin excess), the secondaries are biased toward masses near the primaries, overrepresenting intermediate masses relative to $\xi(m)$.

This distinction matters when interpreting what "$\xi(m)$" means. In our framework:
- $\xi(m \mid \alpha)$ is the **primary-star IMF** (the birth mass distribution of the more massive component in each system).
- The "all-stars IMF" (the distribution of every individual star, including companions) differs from $\xi(m)$ and depends on the binary statistics.

Be explicit in any writeup about which definition you use.

### 9.7 Metallicity dependence

Binary statistics may vary with metallicity. Moe & Di Stefano (2017) primarily calibrated on solar-metallicity populations. At low metallicity, the binary fraction may be higher (Moe et al. 2019), which would increase the distortion. Our "Low-Z" and "Starburst" environments use the same Moe+17 binary statistics as the Solar environment, which may underestimate the binary contamination in those regimes.

### 9.8 Mass segregation and observational selection

In real clusters, massive binaries sink to the center via dynamical friction (mass segregation). If a survey targets only the cluster core, it will observe a higher effective binary fraction than the cluster average. Conversely, flux-limited surveys preferentially detect more massive (and therefore more frequently binary) systems. Neither effect is modeled here.

---

## 10. Toward Photometric (LSST-Realistic) Inference

The current `progenax` pipeline demonstrates the **conceptual framework** — IMF $\to$ multiplicity $\to$ observation operator $\to$ likelihood — using the mass-addition operator. For real LSST cluster science, the same framework applies with a different observation operator.

### 10.1 LSST-realistic forward model

For each system:

1. **Sample primary mass** $m_1 \sim \xi(m \mid \alpha)$ from the Maschberger IMF.
2. **Assign multiplicity**: binary with probability $f_b(m_1)$, or better: sample a companion count from MDS17 (allowing triples for massive primaries).
3. **If companion exists**: sample mass ratio $q \sim g(q \mid m_1)$ and orbital period $\log P \sim f_P(M_1)$.
4. **Map to photometry**: given cluster parameters ($\tau$, $Z$, $d$, $A_V$), convert $(m_1, m_2)$ to per-band absolute magnitudes via theoretical isochrones (e.g., PARSEC, MIST).
5. **Apply distance and extinction**: convert to apparent magnitudes.
6. **Resolve or blend**: if the projected separation at distance $d$ exceeds the PSF FWHM, the system produces two detections; otherwise, add fluxes per band.
7. **Add photometric noise**: apply LSST-like magnitude uncertainties and completeness cuts.

### 10.2 Photometric likelihood

The mass-space likelihood (Section 5) generalizes to a photometric likelihood. For each observed object with multi-band magnitudes $\mathbf{m}_{\text{obs}}$ and error covariance $\Sigma$:

$$p(\mathbf{m}_{\text{obs}} \mid \alpha, \tau, Z, d, A_V) = (1 - f_b) \int \xi(m_1 \mid \alpha) \cdot p(\mathbf{m}_{\text{obs}} \mid m_1) \, dm_1 + f_b \iint \xi(m_1 \mid \alpha) \cdot g(q \mid m_1) \cdot p(\mathbf{m}_{\text{obs}} \mid m_1, q) \, dm_1 \, dq$$

where $p(\mathbf{m}_{\text{obs}} \mid m_1)$ is a multivariate Gaussian in magnitude space centered on the isochrone prediction for a single star of mass $m_1$, and $p(\mathbf{m}_{\text{obs}} \mid m_1, q)$ is centered on the flux-added binary prediction.

This is the direct photometric analog of the mass-space integral in Section 5. The additional complexity is the isochrone mapping $m \to \mathbf{M}(\tau, Z)$ and the extra nuisance parameters ($\tau$, $Z$, $d$, $A_V$).

### 10.3 Halfway-house upgrade

A minimal step between "mass-space toy" and "full CMD inference" that preserves the existing pipeline:

1. Generate fluxes for each binary using an isochrone (or approximate $L \propto m^s$).
2. Add fluxes of primary + secondary.
3. Invert the combined flux through a single-star mass-luminosity relation to get a "photometric-inferred system mass" $m_{\text{phot}}$.
4. Run the existing mass-based likelihood on the derived $m_{\text{phot}}$ values.

This bakes in the "photometry-weighted" behavior (where unequal-mass binaries barely shift the inferred mass) without requiring full CMD modeling. It would demonstrate that the binary bias in photometric mass space is smaller than in true-mass space — but still present and still growing with $N$.

### 10.4 What stays the same

The core logic is identical regardless of observation operator:

- **Forward model**: IMF $\to$ multiplicity model $\to$ observation operator $\to$ synthetic catalog
- **Likelihood**: marginalize over binary status and mass ratio
- **Inference**: NUTS samples the posterior on $\alpha$ (and potentially other parameters)
- **Validation**: compare naive vs. binary-aware posteriors as $N$ grows

The only thing that changes is the observation operator. Our mass-addition results are a proof-of-principle that the **structure** works. The photometric operator changes the **magnitude** of the bias but not the **existence** of the bias or the fact that the naive model becomes confidently wrong at large $N$.

---

## 11. Summary

| Component | What it does | Key reference |
|-----------|-------------|---------------|
| Maschberger IMF | Smooth, analytically invertible IMF | Maschberger (2013) |
| $f_b(m_1)$ | Mass-dependent binary fraction | Moe & Di Stefano (2017), Table 13 |
| $g(q \mid m_1)$ | Mass-dependent mass-ratio distribution | Moe & Di Stefano (2017), Table 10 |
| Forward model | Generate synthetic system masses | Standard population synthesis |
| Observation operator | Map $(m_1, m_2) \to$ observable | $m_1 + m_2$ (this work); photometric (future) |
| Binary-aware likelihood | Marginalize over binary status and $q$ | This work (Gauss-Legendre quadrature) |
| Naive likelihood | Ignore binaries entirely | Standard practice (shown to be biased) |

The central result: ignoring binaries produces a systematic negative bias in $\alpha$ that does not shrink with sample size. At LSST-scale samples ($N \sim 10^4$-$10^5$), the naive posterior becomes **confidently wrong** --- the credible interval excludes the true value. The binary-aware likelihood eliminates this bias at the cost of a ~128$\times$ increase in computation per likelihood evaluation, which is easily parallelized on GPUs.

The current implementation uses mass addition ($m_{\text{sys}} = m_1 + m_2$) as the observation operator, which gives an upper bound on the binary distortion. The next step for LSST-realistic inference is to replace this with a photometric observation operator that maps through isochrones and flux addition, reducing the distortion but not eliminating it.

---

## 12. Checklist for Writing Up

When presenting binary-aware IMF results (in a proposal, paper, or talk), explicitly state:

1. **What is the latent truth?** The stellar IMF of individual stars? The IMF of primaries? The IMF of systems? (These are three different things.)

2. **What is the observed quantity?** True dynamical mass? Photometric mass inferred under a single-star assumption? Multi-band magnitudes? Luminosity?

3. **What multiplicity model are you adopting?**
   - Do you cap at binaries, or allow multiple companions per primary?
   - Which mass-ratio distribution? Period-averaged or period-conditional?
   - Which binary fraction parameterization? What is $q_{\min}$?

4. **What is your observation operator?**
   - Mass addition ($m_1 + m_2$)?
   - Flux addition through an isochrone?
   - Full CMD forward model with photometric noise?

5. **What is your unresolved/resolved selection function?**
   - If your instrument resolves binaries wider than some separation threshold, those companions should not be blended into "system masses."
   - Does the selection function depend on primary mass (brighter primaries are easier to resolve)?

6. **Do you condition on period or marginalize?**
   - Because $g(q \mid m_1, P)$ depends on $P$, and "unresolved" depends on angular separation (which depends on $P$ and $d$).

---

## References

- **Maschberger, T.** (2013). On the function describing the stellar initial mass function. *MNRAS*, 429, 1725. [arXiv:1212.0939](https://arxiv.org/abs/1212.0939)
- **Moe, M. & Di Stefano, R.** (2017). Mind Your Ps and Qs: The Interrelation between Period (P) and Mass-ratio (Q) Distributions of Binary Stars. *ApJS*, 230, 15. [arXiv:1606.05347](https://arxiv.org/abs/1606.05347)
- **Duchene, G. & Kraus, A.** (2013). Stellar Multiplicity. *ARA&A*, 51, 269. [arXiv:1303.3028](https://arxiv.org/abs/1303.3028)
- **Salpeter, E. E.** (1955). The Luminosity Function and Stellar Evolution. *ApJ*, 121, 161.
- **Kroupa, P.** (2001). On the variation of the initial mass function. *MNRAS*, 322, 231. [ADS](https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K)
- **Marks, M. et al.** (2012). Evidence for top-heavy stellar initial mass functions with increasing density and decreasing metallicity. *MNRAS*, 422, 2246.
- **Jerabkova, T. et al.** (2018). The impact of binaries on the determination of the stellar initial mass function. *A&A*, 620, A39.
- **Eldridge, J. J. et al.** (2017). Binary Population and Spectral Synthesis Version 2.1: Construction, Observational Verification, and New Results. *PASA*, 34, e058.
- **Breivik, K. et al.** (2020). COSMIC Variance in Binary Population Synthesis. *ApJ*, 898, 71.
- **Chabrier, G.** (2003). Galactic Stellar and Substellar Initial Mass Function. *PASP*, 115, 763.
