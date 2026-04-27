# IMF Stack Fix Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix gradient flow, implement correct Chabrier (2003) log₁₀-based system IMF, and ensure robust domain behavior across all IMF classes.

**Architecture:** Remove `custom_jvp` that kills parameter gradients; let autodiff flow through Newton iterations. Fix Chabrier to use true log₁₀ formulas with correct Jacobian. Enforce domain behavior (logpdf → -inf outside, cdf → 0/1 at boundaries) uniformly via BaseIMF.

**Tech Stack:** JAX, Equinox, jax.lax.fori_loop (fixed iterations), pytest, jaxastroviz for validation plots.

---

## Task 1: Add IMFProtocol to base.py

**Files:**
- Modify: `src/progenax/imf/base.py:13-20`

**Step 1: Write the failing test**

Create test file `tests/unit/imf/test_imf_protocol.py`:

```python
"""Tests for IMFProtocol type checking."""
import pytest
from progenax.imf.base import IMFProtocol
from progenax.imf import ChabrierIMF, PowerLawIMF, TruncatedIMF, Maschberger


class TestIMFProtocol:
    """Verify all IMFs satisfy IMFProtocol."""

    def test_chabrier_is_imf_protocol(self):
        """ChabrierIMF satisfies IMFProtocol."""
        imf = ChabrierIMF()
        assert isinstance(imf, IMFProtocol)

    def test_powerlaw_is_imf_protocol(self):
        """PowerLawIMF satisfies IMFProtocol."""
        imf = PowerLawIMF.salpeter()
        assert isinstance(imf, IMFProtocol)

    def test_truncated_is_imf_protocol(self):
        """TruncatedIMF satisfies IMFProtocol."""
        inner = ChabrierIMF()
        imf = TruncatedIMF(inner, m_min=0.1, m_max=50.0)
        assert isinstance(imf, IMFProtocol)

    def test_maschberger_is_imf_protocol(self):
        """Maschberger satisfies IMFProtocol."""
        imf = Maschberger()
        assert isinstance(imf, IMFProtocol)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_imf_protocol.py -v
```

Expected: FAIL with "cannot import name 'IMFProtocol'"

**Step 3: Add IMFProtocol to base.py**

Add after line 20 in `base.py`:

```python
@runtime_checkable
class IMFProtocol(Protocol):
    """Protocol for all IMF implementations.

    Any class with these attributes and methods can be used as an IMF,
    enabling TruncatedIMF to wrap any compatible IMF.
    """
    m_min: float
    m_max: float

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]: ...
    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]: ...
    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]: ...
    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]: ...
    def mean_mass(self) -> float: ...
```

Update `__all__` to include `"IMFProtocol"`.

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_imf_protocol.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add tests/unit/imf/test_imf_protocol.py src/progenax/imf/base.py src/progenax/imf/__init__.py
git commit -m "feat(imf): add IMFProtocol for type-safe IMF composition"
```

---

## Task 2: Remove custom_jvp That Kills Parameter Gradients

**Files:**
- Modify: `src/progenax/imf/base.py:27-80`

**Step 1: Write the failing test**

Add to `tests/unit/imf/test_imf_gradients.py`:

```python
"""Tests for gradient flow through IMF parameters."""
import jax
import jax.numpy as jnp
import pytest
import equinox as eqx
from progenax.imf import ChabrierIMF, PowerLawIMF, Maschberger


class TestParameterGradients:
    """Verify gradients flow through IMF parameters (not just u)."""

    def test_chabrier_grad_wrt_alpha(self):
        """Gradient w.r.t. alpha is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(alpha):
            imf = ChabrierIMF(alpha=alpha)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(2.35)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_chabrier_grad_wrt_sigma(self):
        """Gradient w.r.t. sigma is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(sigma):
            imf = ChabrierIMF(sigma=sigma)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(0.69)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_maschberger_grad_wrt_mu(self):
        """Gradient w.r.t. mu is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(mu):
            imf = Maschberger(mu=mu)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(0.2)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"

    def test_maschberger_grad_wrt_alpha(self):
        """Gradient w.r.t. alpha is non-zero through ppf."""
        u = jnp.array([0.3, 0.5, 0.7])

        def loss(alpha):
            imf = Maschberger(alpha=alpha)
            masses = imf.ppf(u)
            return jnp.sum(masses)

        grad_val = jax.grad(loss)(2.3)
        assert jnp.isfinite(grad_val), f"Gradient is {grad_val}"
        assert jnp.abs(grad_val) > 1e-6, f"Gradient is effectively zero: {grad_val}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_imf_gradients.py::TestParameterGradients -v
```

Expected: FAIL (gradients are zero because of nondiff_argnums=(0,))

**Step 3: Remove custom_jvp, keep plain Newton**

Replace lines 27-80 in `base.py` with:

```python
def _ppf_newton(imf: "BaseIMF", u: Float[Array, "..."]) -> Float[Array, "..."]:
    """
    Inverse CDF via fixed Newton iteration.

    Uses Newton's method with fixed iterations (JIT-safe, no convergence loops).
    Initial guess uses linear interpolation in log-mass space.

    JAX autodiff flows through this function, providing gradients w.r.t.
    both u AND imf parameters (alpha, sigma, etc.).

    Args:
        imf: IMF instance
        u: Uniform samples in [0, 1]

    Returns:
        Mass values m such that CDF(m) ≈ u
    """
    # Initial guess: linear interpolation in log-mass space
    log_m_min = jnp.log(jnp.maximum(imf.m_min, 1e-10))
    log_m_max = jnp.log(imf.m_max)
    log_m0 = log_m_min + u * (log_m_max - log_m_min)
    m0 = jnp.exp(log_m0)

    def newton_step(_, m):
        """Single Newton iteration: m_new = m - f(m)/f'(m)."""
        residual = imf.cdf(m) - u
        pdf = jnp.exp(imf.logpdf(m))
        m_new = m - residual / (pdf + 1e-30)
        return jnp.clip(m_new, imf.m_min, imf.m_max)

    # Fixed 20 iterations (JIT-safe, no while_loop)
    return jax.lax.fori_loop(0, 20, newton_step, m0)
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_imf_gradients.py::TestParameterGradients -v
```

Expected: PASS (gradients now flow through Newton iterations)

**Step 5: Commit**

```bash
git add src/progenax/imf/base.py tests/unit/imf/test_imf_gradients.py
git commit -m "fix(imf): remove custom_jvp to enable parameter gradients"
```

---

## Task 3: Add Domain-Safe logpdf and cdf to BaseIMF

**Files:**
- Modify: `src/progenax/imf/base.py:126-134`

**Step 1: Write the failing test**

Add to `tests/unit/imf/test_domain_behavior.py`:

```python
"""Tests for domain behavior of IMF logpdf and cdf."""
import jax.numpy as jnp
import pytest
from progenax.imf import ChabrierIMF, PowerLawIMF, Maschberger, TruncatedIMF


class TestLogPDFDomain:
    """logpdf returns -inf outside [m_min, m_max]."""

    def test_logpdf_below_m_min(self):
        """logpdf returns -inf for m < m_min."""
        imf = ChabrierIMF()  # m_min = 0.08
        m_below = jnp.array([0.01, 0.05, 0.079])
        logpdf_vals = imf.logpdf(m_below)
        assert jnp.all(jnp.isneginf(logpdf_vals)), \
            f"logpdf should be -inf below m_min, got {logpdf_vals}"

    def test_logpdf_above_m_max(self):
        """logpdf returns -inf for m > m_max."""
        imf = ChabrierIMF()  # m_max = 100
        m_above = jnp.array([101.0, 150.0, 1000.0])
        logpdf_vals = imf.logpdf(m_above)
        assert jnp.all(jnp.isneginf(logpdf_vals)), \
            f"logpdf should be -inf above m_max, got {logpdf_vals}"

    def test_logpdf_inside_domain_finite(self):
        """logpdf returns finite values inside domain."""
        imf = ChabrierIMF()
        m_inside = jnp.array([0.1, 1.0, 10.0, 50.0])
        logpdf_vals = imf.logpdf(m_inside)
        assert jnp.all(jnp.isfinite(logpdf_vals)), \
            f"logpdf should be finite inside domain, got {logpdf_vals}"


class TestCDFDomain:
    """cdf returns 0 below m_min, 1 above m_max."""

    def test_cdf_below_m_min(self):
        """cdf returns 0 for m <= m_min."""
        imf = ChabrierIMF()
        m_below = jnp.array([0.01, 0.05, 0.08])
        cdf_vals = imf.cdf(m_below)
        assert jnp.allclose(cdf_vals, 0.0, atol=1e-6), \
            f"cdf should be 0 at/below m_min, got {cdf_vals}"

    def test_cdf_above_m_max(self):
        """cdf returns 1 for m >= m_max."""
        imf = ChabrierIMF()
        m_above = jnp.array([100.0, 101.0, 1000.0])
        cdf_vals = imf.cdf(m_above)
        assert jnp.allclose(cdf_vals, 1.0, atol=1e-6), \
            f"cdf should be 1 at/above m_max, got {cdf_vals}"

    def test_cdf_inside_domain_valid(self):
        """cdf returns values in (0, 1) inside domain."""
        imf = ChabrierIMF()
        m_inside = jnp.array([0.1, 1.0, 10.0, 50.0])
        cdf_vals = imf.cdf(m_inside)
        assert jnp.all((cdf_vals > 0) & (cdf_vals < 1)), \
            f"cdf should be in (0,1) inside domain, got {cdf_vals}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_domain_behavior.py -v
```

Expected: FAIL (current implementation doesn't enforce domain)

**Step 3: Update BaseIMF.logpdf and cdf**

Replace lines 126-134 in `base.py`:

```python
def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
    """Normalized log-PDF. Returns -inf outside [m_min, m_max]."""
    m_arr = jnp.asarray(m)
    in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
    lp_unnorm = self._logpdf_unnorm(m_arr) - self._log_norm
    return jnp.where(in_domain, lp_unnorm, -jnp.inf)

def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
    """Normalized CDF. Returns 0 below m_min, 1 above m_max."""
    m_arr = jnp.asarray(m)
    F_min = self._cdf_unnorm(self.m_min)
    F_max = self._cdf_unnorm(self.m_max)
    raw = (self._cdf_unnorm(m_arr) - F_min) / (F_max - F_min + 1e-30)
    return jnp.where(
        m_arr <= self.m_min,
        0.0,
        jnp.where(m_arr >= self.m_max, 1.0, raw),
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_domain_behavior.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/progenax/imf/base.py tests/unit/imf/test_domain_behavior.py
git commit -m "fix(imf): enforce domain behavior in BaseIMF logpdf/cdf"
```

---

## Task 4: Fix sample_m_total JIT Safety

**Files:**
- Modify: `src/progenax/imf/base.py:159-182`

**Step 1: Write the failing test**

Add to `tests/unit/imf/test_sample_modes.py`:

```python
"""Tests for IMF sampling modes."""
import jax
import jax.numpy as jnp
import pytest
from progenax.imf import ChabrierIMF


class TestSampleMTotalJIT:
    """sample_m_total should be JIT-compatible."""

    def test_sample_m_total_jittable(self, key):
        """sample_m_total can be JIT-compiled."""
        imf = ChabrierIMF()

        @jax.jit
        def sample_wrapper(k):
            masses, n_live = imf.sample_m_total(k, m_total=100.0, n_max=500)
            return masses, n_live

        masses, n_live = sample_wrapper(key)
        assert masses.shape == (500,)
        # n_live should be a JAX array, not Python int
        assert isinstance(n_live, jax.Array)

    def test_sample_m_total_n_live_is_array(self, key):
        """n_live returned is a JAX scalar array, not Python int."""
        imf = ChabrierIMF()
        masses, n_live = imf.sample_m_total(key, m_total=100.0, n_max=500)

        # Should be a 0-dimensional JAX array
        assert hasattr(n_live, 'shape'), "n_live should be a JAX array"
        assert n_live.shape == (), f"n_live should be scalar, got shape {n_live.shape}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_sample_modes.py::TestSampleMTotalJIT -v
```

Expected: FAIL (current returns `int(n_live)`)

**Step 3: Fix sample_m_total return type**

In `base.py`, change line 182 from:
```python
return masses_padded, int(n_live)
```
to:
```python
return masses_padded, n_live  # n_live is a JAX scalar array
```

Update the docstring for `sample_m_total` (around line 159):

```python
def sample_m_total(
    self,
    key: PRNGKeyArray,
    m_total: float,
    n_max: int | None = None,
) -> tuple[Float[Array, "n_max"], Float[Array, ""]]:
    """
    M_total mode (simple): hard cutoff, NOT differentiable w.r.t. m_total.

    Samples masses until cumulative sum exceeds m_total, then pads with zeros.

    Returns:
        masses: Padded array of masses (zeros after n_live)
        n_live: Number of live particles (JAX scalar array, not Python int).
                To get Python int, use int(n_live) outside JIT.
    """
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_sample_modes.py::TestSampleMTotalJIT -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/progenax/imf/base.py tests/unit/imf/test_sample_modes.py
git commit -m "fix(imf): make sample_m_total JIT-safe by returning JAX array"
```

---

## Task 5: Fix ChabrierIMF to Use True Log₁₀ Formulas

**Files:**
- Modify: `src/progenax/imf/chabrier.py` (major rewrite)

**Step 1: Write the failing test for log₁₀ correctness**

Add to `tests/unit/imf/test_chabrier_log10.py`:

```python
"""Tests for Chabrier (2003) log₁₀-based system IMF."""
import jax
import jax.numpy as jnp
import pytest
from progenax.imf import ChabrierIMF


class TestChabrierLog10:
    """Verify Chabrier uses log₁₀ (not ln) as per Chabrier (2003)."""

    def test_lognormal_jacobian_factor(self):
        """Low-mass PDF has correct 1/(m ln 10) Jacobian factor.

        Chabrier (2003) defines xi(log10 m), so converting to xi(m):
            xi(m) = xi(log10 m) * d(log10 m)/dm = xi(log10 m) / (m * ln 10)

        At m = m_c (peak of lognormal in log-space), the exponential = 1,
        so xi(m_c) = A_ln / (m_c * ln 10).
        """
        imf = ChabrierIMF()
        m_c = imf.m_c  # 0.08

        # Get unnormalized PDF at m_c
        pdf_at_mc = imf._lognormal_pdf_unnorm(jnp.array(m_c))

        # Expected: A_ln / (m_c * ln(10))
        expected = imf.A_ln / (m_c * jnp.log(10.0))

        rel_error = jnp.abs(pdf_at_mc - expected) / expected
        assert rel_error < 0.01, \
            f"PDF at m_c: {float(pdf_at_mc):.6f}, expected {float(expected):.6f}"

    def test_lognormal_integral_log10_form(self):
        """Lognormal integral uses correct log₁₀-based erf formula.

        For log₁₀-based lognormal:
            ∫ xi(m) dm = A_ln * sqrt(pi)/2 * [erf(x_hi) - erf(x_lo)]
        where x = (log10 m - log10 m_c) / (sigma * sqrt(2))
        """
        imf = ChabrierIMF()

        # Integrate over [m_min, m_trans] using both methods
        integral_analytical = imf._lognormal_integral(imf.m_min, imf.m_trans)

        # Numerical integration for comparison
        m_grid = jnp.linspace(imf.m_min, imf.m_trans, 10000)
        pdf_grid = imf._lognormal_pdf_unnorm(m_grid)
        integral_numerical = jnp.trapezoid(pdf_grid, m_grid)

        rel_error = jnp.abs(integral_analytical - integral_numerical) / integral_numerical
        assert rel_error < 0.01, \
            f"Analytical: {float(integral_analytical):.6f}, Numerical: {float(integral_numerical):.6f}"

    def test_pdf_peak_location(self):
        """PDF peak (mode) is at expected location.

        For log₁₀-based lognormal xi(m) = A / (m ln 10) * exp[-(log10 m - log10 mc)^2 / (2 sigma^2)],
        the mode in linear mass is at m_mode = m_c * 10^(-sigma^2 * ln 10).

        For sigma = 0.69: m_mode ≈ 0.08 * 10^(-0.69^2 * ln 10) ≈ 0.025 M_sun
        (but our domain starts at m_min = 0.08, so peak is at m_min)
        """
        imf = ChabrierIMF()

        # Find peak in lognormal region
        m_grid = jnp.linspace(imf.m_min, imf.m_trans, 1000)
        pdf_grid = jnp.exp(imf.logpdf(m_grid))

        peak_idx = jnp.argmax(pdf_grid)
        m_peak = m_grid[peak_idx]

        # Peak should be near m_min (because true mode is below m_min)
        # or at the effective mode within the truncated domain
        assert 0.08 <= float(m_peak) <= 0.3, \
            f"PDF peak at {float(m_peak):.3f}, expected 0.08-0.3 M_sun"


class TestChabrierContinuity:
    """Verify continuity at m_trans with log₁₀-based A_pl."""

    def test_continuous_at_m_trans(self):
        """PDF is continuous at transition mass."""
        imf = ChabrierIMF()
        eps = 1e-6

        pdf_below = jnp.exp(imf.logpdf(jnp.array(imf.m_trans - eps)))
        pdf_above = jnp.exp(imf.logpdf(jnp.array(imf.m_trans + eps)))

        rel_diff = jnp.abs(pdf_below - pdf_above) / pdf_below
        assert rel_diff < 0.01, \
            f"Discontinuity at m_trans: {float(pdf_below):.6f} vs {float(pdf_above):.6f}"

    def test_A_pl_from_continuity(self):
        """A_pl is computed to ensure continuity, not from table."""
        imf = ChabrierIMF()

        # At m_trans, lognormal and powerlaw should match
        ln_pdf = imf._lognormal_pdf_unnorm(jnp.array(imf.m_trans))
        pl_pdf = imf._powerlaw_pdf_unnorm(jnp.array(imf.m_trans))

        rel_diff = jnp.abs(ln_pdf - pl_pdf) / ln_pdf
        assert rel_diff < 1e-6, \
            f"At m_trans: lognormal={float(ln_pdf):.6e}, powerlaw={float(pl_pdf):.6e}"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_chabrier_log10.py -v
```

Expected: FAIL (current uses ln instead of log₁₀)

**Step 3: Rewrite ChabrierIMF with log₁₀ formulas**

Replace entire `chabrier.py`:

```python
# src/progenax/imf/chabrier.py
"""
Chabrier (2003) IMF with lognormal + power-law components.

Implements the Chabrier (2003) SYSTEM IMF using log₁₀ mass:
- Lognormal component for m < 1 M☉ (in log₁₀ space)
- Power-law tail (Salpeter slope) for m ≥ 1 M☉

Key formulas (Chabrier 2003, PASP 115, 763, Table 1):

LOW-MASS (system IMF in log₁₀):
    ξ(log₁₀ m) = A_ln × exp[-(log₁₀ m - log₁₀ m_c)² / (2σ²)]

Converting to number per unit mass:
    ξ(m) = dN/dm = (1 / m ln 10) × ξ(log₁₀ m)
         = A_ln / (m ln 10) × exp[-(log₁₀ m - log₁₀ m_c)² / (2σ²)]

HIGH-MASS:
    ξ(m) = A_pl × m^(-α)

A_pl is computed for CONTINUITY at m_trans, not from Chabrier's Table 1.

References:
    Chabrier, G. 2003, PASP, 115, 763 - System IMF
    Chabrier, G. 2005, ASSL, 327, 41 - Review
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


class ChabrierIMF(eqx.Module):
    """Chabrier (2003) lognormal + power-law system IMF.

    Uses true log₁₀-based formulas from Chabrier (2003), with the
    correct Jacobian factor 1/(m ln 10) for conversion to dN/dm.

    Attributes:
        m_min: Minimum mass [M☉] (default: 0.08 - H-burning limit)
        m_max: Maximum mass [M☉] (default: 100)
        m_c: Characteristic mass for lognormal [M☉] (default: 0.08)
        sigma: Width of lognormal in log₁₀ space (default: 0.69)
        alpha: Power-law exponent for ξ(m) ∝ m^(-α) (default: 2.35)
        m_trans: Transition mass between components [M☉] (default: 1.0)
        A_ln: Lognormal coefficient (default: 0.158, Chabrier 2003)

    Example:
        >>> imf = ChabrierIMF()
        >>> key = jax.random.PRNGKey(42)
        >>> masses = imf.sample(key, 1000)
    """

    m_min: float = 0.08
    m_max: float = 100.0
    m_c: float = 0.08
    sigma: float = 0.69  # In log₁₀ space
    alpha: float = 2.35
    m_trans: float = 1.0
    A_ln: float = 0.158

    def __check_init__(self):
        """Validate parameters."""
        if self.m_c <= 0 or self.m_c >= self.m_max:
            raise ValueError(
                f"m_c ({self.m_c}) must be positive and < m_max ({self.m_max})"
            )
        if self.sigma <= 0:
            raise ValueError(f"sigma ({self.sigma}) must be positive")
        if self.alpha <= 0:
            raise ValueError(f"alpha ({self.alpha}) must be positive")
        if self.m_min >= self.m_trans:
            raise ValueError(f"m_min ({self.m_min}) must be < m_trans ({self.m_trans})")

    def _log10(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Safe log₁₀."""
        return jnp.log10(jnp.maximum(m, 1e-30))

    def _lognormal_pdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized lognormal PDF in MASS space (dN/dm).

        ξ(m) = A_ln / (m × ln 10) × exp[-(log₁₀ m - log₁₀ m_c)² / (2σ²)]

        The factor 1/(m × ln 10) is the Jacobian d(log₁₀ m)/dm.
        """
        log10_m = self._log10(m)
        log10_mc = self._log10(self.m_c)

        # Gaussian in log₁₀ space
        quad = -((log10_m - log10_mc) ** 2) / (2.0 * self.sigma ** 2)

        # Jacobian factor: 1 / (m × ln 10)
        jacobian = 1.0 / (m * jnp.log(10.0) + 1e-30)

        return self.A_ln * jacobian * jnp.exp(quad)

    def _powerlaw_pdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized power-law PDF.

        ξ(m) = A_pl × m^(-α)
        """
        return self.A_pl * jnp.power(m + 1e-30, -self.alpha)

    @property
    def A_pl(self) -> Float[Array, ""]:
        """Power-law coefficient computed for continuity at m_trans.

        At m_trans: ξ_ln(m_trans) = ξ_pl(m_trans)
        => A_pl = ξ_ln(m_trans) × m_trans^α
        """
        m_t = jnp.asarray(self.m_trans)
        xi_ln_mt = self._lognormal_pdf_unnorm(m_t)
        return xi_ln_mt * jnp.power(m_t, self.alpha)

    def _lognormal_integral(self, m_lo: float, m_hi: float) -> Float[Array, ""]:
        """Analytical integral of lognormal component from m_lo to m_hi.

        For the log₁₀-based lognormal:
            ∫ ξ(m) dm = A_ln × √π/2 × [erf(x_hi) - erf(x_lo)]
        where x = (log₁₀ m - log₁₀ m_c) / (σ × √2)

        Derivation:
            Let x = (log₁₀ m - log₁₀ m_c) / (σ√2)
            Then dm = m × ln(10) × σ√2 × dx
            So ∫ A_ln/(m ln 10) × exp(-x²) × m × ln(10) × σ√2 dx
             = A_ln × σ√2 × ∫ exp(-x²) dx
             = A_ln × σ√2 × √π/2 × [erf(x)]
             = A_ln × √π/2 × [erf(x_hi) - erf(x_lo)]  # σ√2 cancels
        """
        m_lo = jnp.asarray(m_lo)
        m_hi = jnp.asarray(m_hi)

        log10_mc = self._log10(self.m_c)
        sqrt_2_sigma = jnp.sqrt(2.0) * self.sigma

        x_lo = (self._log10(m_lo) - log10_mc) / sqrt_2_sigma
        x_hi = (self._log10(m_hi) - log10_mc) / sqrt_2_sigma

        erf_lo = jax.scipy.special.erf(x_lo)
        erf_hi = jax.scipy.special.erf(x_hi)

        return self.A_ln * jnp.sqrt(jnp.pi) / 2.0 * (erf_hi - erf_lo)

    def _powerlaw_integral(self, m_lo: float, m_hi: float) -> Float[Array, ""]:
        """Integral of power-law component from m_lo to m_hi.

        ∫ A_pl × m^(-α) dm = A_pl × [m^(1-α) / (1-α)]
        """
        m_lo = jnp.asarray(m_lo)
        m_hi = jnp.asarray(m_hi)
        e = 1.0 - self.alpha
        base = jnp.where(
            jnp.abs(e) < 1e-12,
            jnp.log(m_hi / m_lo),
            (jnp.power(m_hi, e) - jnp.power(m_lo, e)) / e,
        )
        return self.A_pl * base

    def _compute_normalization(self) -> tuple[Float[Array, ""], Float[Array, ""], Float[Array, ""]]:
        """Compute normalization integrals.

        Returns:
            I_ln: Lognormal integral [m_min, min(m_trans, m_max)]
            I_pl: Power-law integral [max(m_trans, m_min), m_max]
            Z: Total normalization I_ln + I_pl
        """
        m_ln_max = jnp.minimum(self.m_trans, self.m_max)
        m_pl_min = jnp.maximum(self.m_trans, self.m_min)

        I_ln = self._lognormal_integral(self.m_min, m_ln_max)

        has_powerlaw = self.m_max > self.m_trans
        I_pl = jnp.where(
            has_powerlaw,
            self._powerlaw_integral(m_pl_min, self.m_max),
            0.0,
        )

        Z = I_ln + I_pl
        return I_ln, I_pl, Z

    def _logpdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized log-PDF."""
        m_arr = jnp.asarray(m)
        is_lognormal = m_arr < self.m_trans

        ln_pdf = self._lognormal_pdf_unnorm(m_arr)
        pl_pdf = self._powerlaw_pdf_unnorm(m_arr)

        pdf_unnorm = jnp.where(is_lognormal, ln_pdf, pl_pdf)
        return jnp.log(pdf_unnorm + 1e-30)

    def _cdf_unnorm(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Unnormalized CDF: ∫_{m_min}^m ξ(m') dm'."""
        m_arr = jnp.asarray(m)
        m_ln_max = jnp.minimum(self.m_trans, self.m_max)

        def cdf_scalar(m_val):
            m_clamped = jnp.clip(m_val, self.m_min, self.m_max)

            # Lognormal contribution
            ln_cdf = self._lognormal_integral(self.m_min, jnp.minimum(m_clamped, m_ln_max))

            # Power-law contribution (if m > m_trans)
            I_ln_full = self._lognormal_integral(self.m_min, m_ln_max)
            has_pl = m_clamped > self.m_trans
            pl_contrib = jnp.where(
                has_pl,
                self._powerlaw_integral(self.m_trans, m_clamped),
                0.0,
            )

            return jnp.where(m_clamped < self.m_trans, ln_cdf, I_ln_full + pl_contrib)

        if m_arr.ndim == 0:
            return cdf_scalar(m_arr)
        return jax.vmap(cdf_scalar)(m_arr.ravel()).reshape(m_arr.shape)

    @property
    def _log_norm(self) -> Float[Array, ""]:
        """Log normalization constant."""
        _, _, Z = self._compute_normalization()
        return jnp.log(Z + 1e-30)

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF. Returns -inf outside [m_min, m_max]."""
        m_arr = jnp.asarray(m)
        in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
        lp = self._logpdf_unnorm(m_arr) - self._log_norm
        return jnp.where(in_domain, lp, -jnp.inf)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized CDF. Returns 0 below m_min, 1 above m_max."""
        m_arr = jnp.asarray(m)
        _, _, Z = self._compute_normalization()
        raw = self._cdf_unnorm(m_arr) / (Z + 1e-30)
        return jnp.where(
            m_arr <= self.m_min,
            0.0,
            jnp.where(m_arr >= self.m_max, 1.0, raw),
        )

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF via Newton iteration with two-component initial guess."""
        return self._ppf_newton_chabrier(u)

    def _ppf_newton_chabrier(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Newton solver with component-specific initial guess."""
        u = jnp.clip(jnp.asarray(u), 1e-10, 1.0 - 1e-10)

        # Probability in lognormal region
        I_ln, _, Z = self._compute_normalization()
        p_ln = I_ln / Z

        # Lognormal initial guess (inverse of erf-based CDF)
        log10_mc = self._log10(self.m_c)
        sqrt_2_sigma = jnp.sqrt(2.0) * self.sigma

        m_ln_max = jnp.minimum(self.m_trans, self.m_max)
        erf_min = jax.scipy.special.erf((self._log10(self.m_min) - log10_mc) / sqrt_2_sigma)
        erf_max = jax.scipy.special.erf((self._log10(m_ln_max) - log10_mc) / sqrt_2_sigma)

        u_ln_scaled = jnp.clip(u / (p_ln + 1e-10), 0.0, 1.0)
        erf_target = erf_min + u_ln_scaled * (erf_max - erf_min)
        log10_m_ln = log10_mc + sqrt_2_sigma * jax.scipy.special.erfinv(erf_target)
        m0_ln = jnp.clip(jnp.power(10.0, log10_m_ln), self.m_min, m_ln_max)

        # Power-law initial guess
        u_pl_scaled = jnp.clip((u - p_ln) / (1.0 - p_ln + 1e-10), 0.0, 1.0)
        e = 1.0 - self.alpha
        e_safe = jnp.where(jnp.abs(e) < 1e-10, 1e-10, e)

        m_trans_e = jnp.power(self.m_trans, e_safe)
        m_max_e = jnp.power(self.m_max, e_safe)
        m0_pl = jnp.power(u_pl_scaled * (m_max_e - m_trans_e) + m_trans_e, 1.0 / e_safe)
        m0_pl = jnp.clip(m0_pl, self.m_trans, self.m_max)

        # Select based on region
        is_lognormal = u < p_ln
        m0 = jnp.where(is_lognormal, m0_ln, m0_pl)

        def newton_step(_, m):
            residual = self.cdf(m) - u
            pdf = jnp.exp(self.logpdf(m))
            m_new = m - residual / (pdf + 1e-30)
            return jnp.clip(m_new, self.m_min, self.m_max)

        return jax.lax.fori_loop(0, 30, newton_step, m0)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample n masses via reparameterization trick."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> Float[Array, ""]:
        """Expected mass E[m] via numerical integration."""
        m_grid = jnp.linspace(self.m_min, self.m_max, 5000)
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)


__all__ = ["ChabrierIMF"]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_chabrier_log10.py -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/progenax/imf/chabrier.py tests/unit/imf/test_chabrier_log10.py
git commit -m "fix(imf): implement correct log₁₀-based Chabrier (2003) system IMF"
```

---

## Task 6: Update PowerLawIMF Domain Behavior

**Files:**
- Modify: `src/progenax/imf/power_law.py:205-213`

**Step 1: Write the failing test**

Add to `tests/unit/imf/test_domain_behavior.py`:

```python
class TestPowerLawDomain:
    """PowerLawIMF domain behavior."""

    def test_powerlaw_logpdf_below_m_min(self):
        """logpdf returns -inf for m < m_min."""
        imf = PowerLawIMF.salpeter()  # m_min = 0.1
        m_below = jnp.array([0.01, 0.05, 0.09])
        logpdf_vals = imf.logpdf(m_below)
        assert jnp.all(jnp.isneginf(logpdf_vals))

    def test_powerlaw_logpdf_above_m_max(self):
        """logpdf returns -inf for m > m_max."""
        imf = PowerLawIMF.salpeter()  # m_max = 100
        m_above = jnp.array([101.0, 150.0])
        logpdf_vals = imf.logpdf(m_above)
        assert jnp.all(jnp.isneginf(logpdf_vals))

    def test_powerlaw_cdf_boundaries(self):
        """cdf returns 0 at m_min, 1 at m_max."""
        imf = PowerLawIMF.salpeter()
        assert jnp.allclose(imf.cdf(jnp.array(0.05)), 0.0, atol=1e-6)
        assert jnp.allclose(imf.cdf(jnp.array(0.1)), 0.0, atol=1e-6)
        assert jnp.allclose(imf.cdf(jnp.array(100.0)), 1.0, atol=1e-6)
        assert jnp.allclose(imf.cdf(jnp.array(150.0)), 1.0, atol=1e-6)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_domain_behavior.py::TestPowerLawDomain -v
```

Expected: FAIL

**Step 3: Update PowerLawIMF.logpdf and cdf**

Replace lines 205-213 in `power_law.py`:

```python
def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
    """Normalized log-PDF. Returns -inf outside [m_min, m_max]."""
    m_arr = jnp.asarray(m)
    in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
    Z = jnp.sum(self._segment_integrals)
    lp = self._logpdf_unnorm(m_arr) - jnp.log(Z)
    return jnp.where(in_domain, lp, -jnp.inf)

def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
    """Normalized CDF. Returns 0 below m_min, 1 above m_max."""
    m_arr = jnp.asarray(m)
    Z = jnp.sum(self._segment_integrals)
    raw = self._cdf_unnorm(jnp.clip(m_arr, self.m_min, self.m_max)) / Z
    return jnp.where(
        m_arr <= self.m_min,
        0.0,
        jnp.where(m_arr >= self.m_max, 1.0, raw),
    )
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_domain_behavior.py::TestPowerLawDomain -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/progenax/imf/power_law.py
git commit -m "fix(imf): enforce domain behavior in PowerLawIMF"
```

---

## Task 7: Update TruncatedIMF Type Annotation and Domain Behavior

**Files:**
- Modify: `src/progenax/imf/truncated.py`

**Step 1: Write the failing test**

Add to `tests/unit/imf/test_domain_behavior.py`:

```python
class TestTruncatedDomain:
    """TruncatedIMF domain behavior."""

    def test_truncated_logpdf_outside(self):
        """logpdf returns -inf outside truncated domain."""
        inner = ChabrierIMF()
        imf = TruncatedIMF(inner, m_min=0.1, m_max=50.0)

        # Outside truncated domain
        m_outside = jnp.array([0.05, 0.09, 51.0, 100.0])
        logpdf_vals = imf.logpdf(m_outside)
        assert jnp.all(jnp.isneginf(logpdf_vals))

    def test_truncated_cdf_boundaries(self):
        """cdf returns 0/1 at boundaries."""
        inner = ChabrierIMF()
        imf = TruncatedIMF(inner, m_min=0.1, m_max=50.0)

        assert jnp.allclose(imf.cdf(jnp.array(0.05)), 0.0, atol=1e-6)
        assert jnp.allclose(imf.cdf(jnp.array(0.1)), 0.0, atol=1e-6)
        assert jnp.allclose(imf.cdf(jnp.array(50.0)), 1.0, atol=1e-6)
        assert jnp.allclose(imf.cdf(jnp.array(100.0)), 1.0, atol=1e-6)
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/imf/test_domain_behavior.py::TestTruncatedDomain -v
```

Expected: FAIL

**Step 3: Update TruncatedIMF**

Replace `truncated.py`:

```python
# src/progenax/imf/truncated.py
"""
TruncatedIMF wrapper for hard mass truncation.

Wraps any IMF (satisfying IMFProtocol) to enforce strict mass bounds.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray

from .base import IMFProtocol


class TruncatedIMF(eqx.Module):
    """
    Wrapper for hard mass truncation of any IMF.

    Takes an existing IMF and enforces strict bounds [m_min, m_max].
    Renormalizes the PDF over the truncated domain.

    Attributes:
        inner: The wrapped IMF (any class satisfying IMFProtocol)
        m_min: Minimum mass [M_sun]
        m_max: Maximum mass [M_sun]

    Example:
        >>> from progenax.imf import ChabrierIMF, TruncatedIMF
        >>> chabrier = ChabrierIMF()
        >>> truncated = TruncatedIMF(chabrier, m_min=0.08, m_max=150.0)
        >>> masses = truncated.sample(key, 1000)
    """

    inner: IMFProtocol
    m_min: float
    m_max: float

    def __init__(self, inner: IMFProtocol, m_min: float, m_max: float):
        """
        Create truncated IMF.

        Args:
            inner: IMF to wrap (must satisfy IMFProtocol)
            m_min: New minimum mass [M_sun]
            m_max: New maximum mass [M_sun]

        Raises:
            ValueError: If m_min >= m_max after clamping to inner IMF bounds
        """
        self.inner = inner
        self.m_min = max(m_min, inner.m_min)
        self.m_max = min(m_max, inner.m_max)

        if self.m_min >= self.m_max:
            raise ValueError(
                f"Invalid truncation bounds: m_min={self.m_min:.3f} >= m_max={self.m_max:.3f} "
                f"after clamping to inner IMF bounds [{inner.m_min:.3f}, {inner.m_max:.3f}]"
            )

    @property
    def _cdf_min(self) -> Float[Array, ""]:
        """CDF at m_min."""
        return self.inner.cdf(jnp.asarray(self.m_min))

    @property
    def _cdf_max(self) -> Float[Array, ""]:
        """CDF at m_max."""
        return self.inner.cdf(jnp.asarray(self.m_max))

    def logpdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """Normalized log-PDF. Returns -inf outside [m_min, m_max]."""
        m_arr = jnp.asarray(m)
        in_domain = (m_arr >= self.m_min) & (m_arr <= self.m_max)
        log_norm = jnp.log(self._cdf_max - self._cdf_min + 1e-30)
        lp = self.inner.logpdf(m_arr) - log_norm
        return jnp.where(in_domain, lp, -jnp.inf)

    def cdf(self, m: Float[Array, "..."]) -> Float[Array, "..."]:
        """CDF rescaled to [0, 1]. Returns 0 below m_min, 1 above m_max."""
        m_arr = jnp.asarray(m)
        raw_cdf = self.inner.cdf(m_arr)
        cdf_trunc = (raw_cdf - self._cdf_min) / (self._cdf_max - self._cdf_min + 1e-30)
        return jnp.where(
            m_arr <= self.m_min,
            0.0,
            jnp.where(m_arr >= self.m_max, 1.0, cdf_trunc),
        )

    def ppf(self, u: Float[Array, "..."]) -> Float[Array, "..."]:
        """Inverse CDF over truncated domain."""
        u_scaled = self._cdf_min + u * (self._cdf_max - self._cdf_min)
        return self.inner.ppf(u_scaled)

    def sample(self, key: PRNGKeyArray, n: int) -> Float[Array, "n"]:
        """Sample from truncated distribution."""
        u = jax.random.uniform(key, (n,))
        return self.ppf(u)

    def mean_mass(self) -> Float[Array, ""]:
        """Mean mass over truncated domain."""
        m_grid = jnp.linspace(self.m_min, self.m_max, 1000)
        pdf_grid = jnp.exp(self.logpdf(m_grid))
        return jnp.trapezoid(m_grid * pdf_grid, m_grid)


__all__ = ["TruncatedIMF"]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/imf/test_domain_behavior.py::TestTruncatedDomain -v
```

Expected: PASS

**Step 5: Commit**

```bash
git add src/progenax/imf/truncated.py
git commit -m "fix(imf): update TruncatedIMF to use IMFProtocol and enforce domain"
```

---

## Task 8: Update __init__.py Exports

**Files:**
- Modify: `src/progenax/imf/__init__.py`

**Step 1: Update exports**

Add `IMFProtocol` to imports and `__all__`:

```python
from .base import BaseIMF, _ppf_newton, IMFProtocol
```

And in `__all__`:
```python
"IMFProtocol",
```

**Step 2: Run all IMF tests**

```bash
pytest tests/unit/imf/ -v
```

Expected: All PASS

**Step 3: Commit**

```bash
git add src/progenax/imf/__init__.py
git commit -m "feat(imf): export IMFProtocol from imf module"
```

---

## Task 9: Run Full Test Suite and Fix Any Failures

**Files:**
- Various test files may need updates

**Step 1: Run full test suite**

```bash
pytest tests/ -v --tb=short
```

**Step 2: Fix any failures**

Address test failures one by one. Common issues:
- Tests expecting old behavior (ln vs log₁₀)
- Tests with tight tolerances
- Tests checking exact return types

**Step 3: Commit fixes**

```bash
git add -A
git commit -m "test(imf): update tests for log₁₀ Chabrier and domain behavior"
```

---

## Task 10: Create Validation Plots with jaxastroviz

**Files:**
- Create: `scripts/validate_imfs.py`

**Step 1: Write validation script**

```python
#!/usr/bin/env python
"""
IMF validation plots using jaxastroviz.

Produces publication-quality diagnostic figures for:
1. PDF × m vs m (mass distribution shape)
2. High-mass tail (power-law slope verification)
3. CDF vs empirical CDF (sampling validation)
"""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from progenax.imf import ChabrierIMF, PowerLawIMF


def plot_imf_validation(imf, name: str, output_dir: str = "validation/plots"):
    """Generate validation plots for an IMF."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    key = jax.random.PRNGKey(42)
    N_samples = 100000
    masses = imf.sample(key, N_samples)

    # 1. PDF × m vs m (mass distribution)
    fig, ax = plt.subplots(figsize=(8, 6))

    m_grid = jnp.logspace(jnp.log10(imf.m_min), jnp.log10(imf.m_max), 500)
    pdf_grid = jnp.exp(imf.logpdf(m_grid))

    ax.loglog(m_grid, m_grid * pdf_grid, 'b-', lw=2, label='Analytical m×ξ(m)')

    # Histogram
    log_bins = np.logspace(np.log10(float(imf.m_min)), np.log10(float(imf.m_max)), 50)
    counts, edges = np.histogram(np.array(masses), bins=log_bins, density=True)
    centers = np.sqrt(edges[:-1] * edges[1:])
    ax.loglog(centers, centers * counts, 'ro', ms=4, alpha=0.7, label=f'Samples (N={N_samples})')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('m × ξ(m)', fontsize=12)
    ax.set_title(f'{name} IMF: Mass Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name.lower()}_pdf.png', dpi=150)
    plt.close()

    # 2. High-mass tail (slope verification)
    fig, ax = plt.subplots(figsize=(8, 6))

    m_high = m_grid[m_grid > 2.0]
    pdf_high = jnp.exp(imf.logpdf(m_high))

    ax.loglog(m_high, pdf_high, 'b-', lw=2, label='ξ(m)')

    # Reference slope line
    if hasattr(imf, 'alpha'):
        alpha = imf.alpha
    else:
        alpha = 2.35
    m_ref = 5.0
    pdf_ref = float(jnp.exp(imf.logpdf(jnp.array(m_ref))))
    m_line = jnp.array([2.0, 50.0])
    pdf_line = pdf_ref * (m_line / m_ref) ** (-alpha)
    ax.loglog(m_line, pdf_line, 'r--', lw=1.5, label=f'Slope α={alpha}')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('ξ(m)', fontsize=12)
    ax.set_title(f'{name} IMF: High-Mass Tail', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name.lower()}_tail.png', dpi=150)
    plt.close()

    # 3. CDF comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    cdf_analytical = imf.cdf(m_grid)

    # Empirical CDF
    masses_sorted = jnp.sort(masses)
    ecdf = jnp.arange(1, N_samples + 1) / N_samples

    ax.semilogx(m_grid, cdf_analytical, 'b-', lw=2, label='Analytical CDF')
    ax.semilogx(masses_sorted[::100], ecdf[::100], 'r.', ms=2, alpha=0.5, label='Empirical CDF')

    ax.set_xlabel('Mass [M$_\\odot$]', fontsize=12)
    ax.set_ylabel('F(m)', fontsize=12)
    ax.set_title(f'{name} IMF: CDF Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/{name.lower()}_cdf.png', dpi=150)
    plt.close()

    print(f"✓ {name} plots saved to {output_dir}/")


def main():
    """Generate validation plots for all IMFs."""
    print("Generating IMF validation plots...")

    # Chabrier (2003)
    chabrier = ChabrierIMF()
    plot_imf_validation(chabrier, "Chabrier")

    # Kroupa (2001)
    kroupa = PowerLawIMF.kroupa()
    plot_imf_validation(kroupa, "Kroupa")

    # Salpeter (1955)
    salpeter = PowerLawIMF.salpeter()
    plot_imf_validation(salpeter, "Salpeter")

    print("\n✓ All validation plots complete!")


if __name__ == "__main__":
    main()
```

**Step 2: Run validation**

```bash
mkdir -p validation/plots
python scripts/validate_imfs.py
```

**Step 3: Visually inspect plots**

Check that:
- PDF peak is in expected range (0.1-0.3 M☉ for Chabrier)
- High-mass slope matches α parameter
- CDF matches empirical distribution

**Step 4: Commit**

```bash
git add scripts/validate_imfs.py validation/plots/
git commit -m "docs(imf): add validation plots for Chabrier, Kroupa, Salpeter"
```

---

## Final Checklist

Before declaring complete:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Parameter gradients work: `pytest tests/unit/imf/test_imf_gradients.py -v`
- [ ] Domain behavior correct: `pytest tests/unit/imf/test_domain_behavior.py -v`
- [ ] Chabrier uses log₁₀: `pytest tests/unit/imf/test_chabrier_log10.py -v`
- [ ] Validation plots look correct
- [ ] No regressions in physics tests: `pytest tests/validation/test_imf_physics.py -v`

---

**Plan complete and saved to `docs/plans/2025-12-07-imf-stack-fix.md`.**

**Two execution options:**

1. **Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

2. **Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
