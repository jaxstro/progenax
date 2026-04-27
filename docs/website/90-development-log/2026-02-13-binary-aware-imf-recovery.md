# Binary-Aware IMF Recovery via NUTS

**Goal:** Demonstrate that ignoring unresolved binaries biases the IMF slope, and that a Moe+17-aware mixture likelihood eliminates this bias.

**Context:** For the LSST-DA Early Science proposal. Companion to the IMF-only recovery figure (`validate_hmc_imf_recovery.py`). This figure shows progenax handles the IMF-binary degeneracy that LSST data will contain.

---

## The Problem

LSST will observe unresolved binaries as single sources. A binary with primary mass m1 and mass ratio q appears as a system with mass m_sys = m1(1+q). Fitting a single-star IMF to system masses produces a biased slope: binaries inflate the high-mass end, making the IMF appear shallower (smaller alpha) than it truly is.

## The Solution

A mixture likelihood that marginalizes over binary status:

```
p(m_sys | alpha) = [1 - f_b(m_sys)] * xi(m_sys; alpha)
                 + integral of f_b(m1) * xi(m1; alpha) * g(q | m1) / m1  dm1
```

where xi is the Maschberger IMF, f_b is the Moe+17 mass-dependent binary fraction, and g(q | m1) is the Moe+17 mass-ratio distribution. The integral runs over m1 in [m_sys/2, m_sys/(1+q_min)], computed with 128-point Gauss-Legendre quadrature.

---

## Script 1: `validation/imf/validate_binary_aware_recovery.py`

### Data Generation

For each of 4 environments (Solar, YMC, Low-Z, Starburst):

1. Draw N=10,000 primaries from Maschberger(alpha_true)
2. Apply Moe+17 mass-dependent binary fraction and mass-ratio distribution via `BinaryIMF`
3. Compute m_sys = m1(1+q) for binaries, m_sys = m1 for singles
4. Discard binary membership (observer sees only m_sys)

### Inference: Two NUTS Runs per Environment

**Naive model:** Fit Maschberger(alpha) directly to m_sys, pretending all are singles. Uses `Maschberger.logpdf(m_sys, alpha)`.

**Binary-aware model:** Fit alpha using the mixture likelihood above. Moe+17 binary physics are fixed (not inferred). The only free parameter is alpha.

### Expected Results

- Naive alpha biased negative (shallower) for all environments
- Bias largest for steep slopes (Solar, alpha=2.3) where high-mass stars are rare and binary contamination has outsized impact
- Binary-aware alpha recovers the true value within 95% CI

### Build Order

1. **Bernoulli version first:** Constant f_bin=0.5, flat mass-ratio distribution. Validates the quadrature likelihood machinery.
2. **Moe+17 upgrade:** Swap in `MassDependentBinaryFraction` and `MoeDiStefano2017`. Validates with realistic binary physics.
3. **Figure polish:** Match IMF-only figure standards.

---

## Figure Layout (1x3, same as IMF-only)

**Panel (a): System mass function**
- Solid lines: true single-star IMF m*xi(m) for each environment
- Dashed lines: observed system MF m*p(m_sys) with binary contamination
- Shows the distortion binaries introduce

**Panel (b): True vs recovered alpha**
- Filled circles + thick error bars: binary-aware (should land on 1:1 line)
- Open circles + thin error bars: naive (biased low)
- Same 4 environment colors as IMF figure

**Panel (c): Residual posteriors**
- Solid KDE + fill: binary-aware (centered on zero)
- Dashed KDE: naive (shifted negative)
- Same fixed x-range as IMF figure for comparability

---

## Key Implementation Details

### Mixture Log-Likelihood (~40 lines)

```python
def log_mixture_likelihood(m_sys, alpha, n_quad=128):
    imf = Maschberger(alpha=alpha)
    bf = MassDependentBinaryFraction()
    qd = MoeDiStefano2017()

    # Singles contribution
    log_p_single = imf.logpdf(m_sys)
    f_b_sys = bf.compute(m_sys)  # f_bin at observed mass
    p_single = (1 - f_b_sys) * jnp.exp(log_p_single)

    # Binary contribution: Gauss-Legendre quadrature over m1
    nodes, weights = gauss_legendre_nodes(n_quad)  # precomputed

    def binary_integrand(m_sys_i):
        m1_lo = jnp.maximum(m_sys_i / 2, m_min)
        m1_hi = m_sys_i / (1 + q_min)
        # Map nodes from [-1,1] to [m1_lo, m1_hi]
        m1 = 0.5 * (m1_hi - m1_lo) * nodes + 0.5 * (m1_hi + m1_lo)
        q = m_sys_i / m1 - 1
        f_b = bf.compute(m1)
        xi_m1 = jnp.exp(imf.logpdf(m1))
        g_q = qd.pdf_given_primary(q, m1)  # mass-dependent q dist
        integrand = f_b * xi_m1 * g_q / m1
        return 0.5 * (m1_hi - m1_lo) * jnp.dot(weights, integrand)

    p_binary = jax.vmap(binary_integrand)(m_sys)
    p_total = p_single + p_binary
    return jnp.sum(jnp.log(p_total + 1e-30))
```

### Dependencies from progenax

- `Maschberger` (IMF)
- `BinaryIMF` (data generation)
- `MassDependentBinaryFraction` (Moe+17 f_bin)
- `MoeDiStefano2017` (Moe+17 mass-ratio)

### Gauss-Legendre Quadrature

Precompute nodes and weights with `jnp.polynomial.legendre` or `scipy.special.roots_legendre` (at init time, not inside JIT). 128 points gives ~1e-12 accuracy for smooth integrands.

---

## Script 2: `validation/imf/validate_binary_param_recovery.py` (supplementary)

Recover binary population parameters from system masses.

**Parameters:** (alpha, f_bin_scale, gamma_q_offset) where:
- f_bin_scale multiplies the Moe+17 baseline (1.0 = standard, 1.5 = enhanced)
- gamma_q_offset shifts the mass-ratio power-law index (0.0 = standard)

**Purpose:** Show progenax can infer binary population properties, not just correct for them. Supplementary evidence, not the proposal figure.

Deferred until Script 1 passes.

---

## Success Criteria

1. Binary-aware recovery: all 4 alpha values within 95% CI
2. Naive bias visible: all 4 naive medians shifted negative
3. Bias scales with alpha: steeper slopes show larger naive bias
4. ESS > 200, R-hat < 1.05 for all 8 runs
5. Figure matches IMF-only layout for visual consistency

## Files

| File | Purpose |
|------|---------|
| `validation/imf/validate_binary_aware_recovery.py` | Script 1 (proposal figure) |
| `validation/imf/validate_binary_param_recovery.py` | Script 2 (supplementary) |
| `validation/imf/plots/binary_aware_recovery.{png,pdf}` | Proposal figure |
| `validation/imf/plots/binary_param_recovery.{png,pdf}` | Supplementary figure |
