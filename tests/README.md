# progenax Test Suite

Comprehensive physics validation for stellar initial conditions generation.

**~1243 released-core tests** across 3 tiers validating astrophysics from published
literature (1911–2018). Counts drift as the suite grows — **see CI for the live number**.

## Table of Contents

- [Quick Start](#quick-start)
- [Test Architecture](#test-architecture)
- [Physics Validation](#physics-validation)
  - [1. Plummer Profile](#1-plummer-profile--kinematics)
  - [2. King Profile](#2-king-profile--kinematics)
  - [3. EFF Profile](#3-eff-profile--kinematics)
  - [4. Initial Mass Functions](#4-initial-mass-functions)
  - [5. Binary Orbital Mechanics](#5-binary-orbital-mechanics)
  - [6. Environment-Conditioned IMF](#6-environment-conditioned-imf)
  - [7. Galaxy-wide IMF (IGIMF) — scope](#7-environment-conditioned-imf--galaxy-wide-igimf-scope)
- [Unit Tests](#unit-tests)
- [Integration Tests](#integration-tests)
- [Test Fixtures & Constants](#test-fixtures--constants)
- [References](#references)

---

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run by tier
pytest tests/unit/ -v           # Unit tests (functional correctness)
pytest tests/integration/ -v    # JAX compatibility
pytest tests/validation/ -v     # Physics validation

# Run specific module
pytest tests/unit/imf/ -v       # IMF tests only
pytest tests/validation/test_plummer_physics.py -v
```

---

## Test Architecture

| Tier | Tests | Purpose | Runtime |
|------|-------|---------|---------|
| **Unit** | ~956 | Functional correctness (shapes, bounds, roundtrips) | — |
| **Integration** | ~43 | JAX transformations (JIT, grad, vmap) | — |
| **Validation** | ~244 | Physics accuracy against literature | — |
| **Total** | **~1243** | Comprehensive coverage (see CI for the live count) | — |

---

## Physics Validation

Each section documents the physics equations being tested, literature sources, and tolerance justifications.

### 1. Plummer Profile & Kinematics

**Source:** Plummer (1911) MNRAS 71:460, Binney & Tremaine (2008) "Galactic Dynamics"

#### Density Profile

$$\rho(r) = \frac{3M}{4\pi a^3} \left(1 + \frac{r^2}{a^2}\right)^{-5/2}$$

#### Scale Radius (from half-mass radius)

$$a = r_h \sqrt{2^{2/3} - 1} \approx 0.7664 \, r_h$$

**Derivation:** From $M(<r_h)/M = 0.5$ and the cumulative mass function.

#### Cumulative Mass Function

$$\frac{M(<r)}{M} = \frac{r^3}{(r^2 + a^2)^{3/2}}$$

At the scale radius: $M(<a)/M = 1/2^{3/2} \approx 0.354$

#### Velocity Dispersion Profile

$$\sigma^2(r) = \frac{GM}{6\sqrt{r^2 + a^2}}$$

Central dispersion: $\sigma(0) = \sqrt{GM/6a}$

#### Escape Velocity

$$v_{\rm esc}^2(r) = \frac{2GM}{\sqrt{r^2 + a^2}}$$

#### Analytical Potential Energy

$$V = -\frac{3\pi GM^2}{32a}$$

#### Velocity Distribution

Velocity magnitudes follow $q^2 \sim \text{Beta}(3/2, 9/2)$ where $q = v/v_{\rm esc}$

- Mean: $\langle q^2 \rangle = \frac{3/2}{3/2 + 9/2} = 0.25$
- Variance: $\text{Var}(q^2) = \frac{(3/2)(9/2)}{36 \times 7} \approx 0.027$

#### Tests & Tolerances

| Test | Physics Validated | Tolerance | Justification |
|------|-------------------|-----------|---------------|
| `test_scale_radius_formula_exact` | $a = r_h \sqrt{2^{2/3} - 1}$ | rtol < 1e-6 | Exact formula |
| `test_half_mass_radius_statistical` | 50% particles within $r_h$ | 3% | Statistical (N=5000) |
| `test_cumulative_mass_at_scale_radius` | $M(<a)/M = 0.354$ | 3% | Statistical |
| `test_cdf_formula_accuracy` | CDF at multiple radii | 3% | Statistical |
| `test_central_velocity_dispersion` | $\sigma(0) = \sqrt{GM/6a}$ | 10% | Statistical + sampling |
| `test_radial_dispersion_profile` | $\sigma(r)$ decreases outward | 10% | Statistical |
| `test_velocity_isotropy` | $\langle v_x^2 \rangle \approx \langle v_y^2 \rangle \approx \langle v_z^2 \rangle$ | 5% | Statistical |
| `test_virial_ratio` | $Q = T/|V| \approx 0.5$ | 20% | Statistical + PE approximation |
| `test_all_particles_bound` | 100% have $v < v_{\rm esc}$ | exact | Physical requirement |
| `test_q_squared_mean` | $\langle q^2 \rangle = 0.25$ | 2% | Beta distribution property |
| `test_q_squared_variance` | $\text{Var}(q^2) \approx 0.027$ | 15% | Higher-order statistic |

---

### 2. King Profile & Kinematics

**Source:** King (1966) AJ 71:64, Binney & Tremaine (2008) "Galactic Dynamics"

#### Lowered-Maxwellian Density

The 3-D King density is the lowered Maxwellian (King 1966; B&T 2008 eq. 4.131),
not the projected $K$-function — the latter over-extends the profile by 2-30x and was
removed during the 2026-06 hardening:

$$\rho(r) \propto \int_0^{v_{\rm esc}} v^2 \left[e^{\psi(r) - v^2/2\sigma^2} - 1\right] dv$$

The self-consistent $\sigma^2 = GM / (9\, r_c\, \mu(W_0))$ (factor-of-9 nondimensionalization)
puts the cluster in virial equilibrium ($Q = 0.5$) with no external rescale.

Reference concentrations $c = \log_{10}(r_t/r_c)$ (King 1966 Table II): $c(3) \approx 0.67$,
$c(7) \approx 1.53$, $c(9) \approx 2.12$.

#### Dimensionless Poisson Equation

$$\frac{d^2\psi}{d\xi^2} + \frac{2}{\xi}\frac{d\psi}{d\xi} = -\tilde{\rho}(\psi)$$

Boundary conditions: $\psi(0) = W_0$, $\left.\frac{d\psi}{d\xi}\right|_{\xi=0} = 0$

#### Density-Potential Relation

The density follows from the lowered-Maxwellian DF via the velocity integral above,
normalized to the central value:

$$\frac{\rho(r)}{\rho_0} = \frac{\int_0^{\sqrt{2\psi(r)}} v^2 \left[e^{\psi(r) - v^2/2} - 1\right] dv}{\int_0^{\sqrt{2 W_0}} v^2 \left[e^{W_0 - v^2/2} - 1\right] dv}$$

#### Tidal Truncation

$$\rho(r) = 0 \quad \text{for} \quad r > r_t$$

The tidal radius $r_t$ is where $\psi(r_t) = 0$.

#### Concentration Parameter

$c = \log_{10}(r_t/r_c)$ increases with $W_0$ (King 1966 Table II):
- $W_0 = 3$: Low concentration ($c \approx 0.67$)
- $W_0 = 7$: Medium concentration ($c \approx 1.53$)
- $W_0 = 9$: High concentration ($c \approx 2.12$)

#### Tests & Tolerances

| Test | Physics Validated | Tolerance | Justification |
|------|-------------------|-----------|---------------|
| `test_concentration_matches_king1966_table_ii` | $c = \log_{10}(r_t/r_c)$ vs Table II | 0.02 | King 1966 Table II |
| `test_density_shape_matches_direct_velocity_integral` | Lowered-Maxwellian $\rho(r)$ shape | 0.5% | Independent velocity-integral oracle |
| `test_grad_of_psi_through_solve_king_profile_is_finite` | $\partial\psi/\partial W_0$ finite (grad/JIT-safe) | finite | Differentiability |
| `test_boundary_conditions` | $\psi(0) \approx W_0$ | 0.1 | ODE solver |
| `test_potential_monotonic_decrease` | $d\psi/dr < 0$ | exact | Physical requirement |
| `test_potential_reaches_zero` | $\psi(r_t) = 0$ | 1e-3 | ODE termination |
| `test_tidal_truncation` | 100% at $r \leq r_t$ | exact | Hard boundary |
| `test_concentration_effect` | Higher $W_0 \to$ larger $r_t/r_c$ | qualitative | Physical trend |

---

### 3. EFF Profile & Kinematics

**Source:** Elson, Fall & Freeman (1987) ApJ 323:54

#### Density Profile

$$\rho(r) = \rho_0 \left(1 + \frac{r^2}{a^2}\right)^{-\gamma/2}$$

Parameters:
- $a$: Scale radius
- $\gamma$: Power-law index (typically 2-4)
- $r_t$: Tidal truncation radius

#### Asymptotic Behavior

For $r \gg a$:

$$\rho(r) \propto r^{-\gamma}$$

#### Special Cases

| $\gamma$ | Profile Type | Asymptotic Slope |
|----------|--------------|------------------|
| 2 | Shallow (extended halo) | $\rho \propto r^{-2}$ |
| 3 | Intermediate | $\rho \propto r^{-3}$ |
| 4 | Steep (concentrated core) | $\rho \propto r^{-4}$ |

#### Tests & Tolerances

| Test | Physics Validated | Tolerance | Justification |
|------|-------------------|-----------|---------------|
| `test_central_density_unity` | $\rho(0) = \rho_0$ | exact | Normalization |
| `test_density_at_scale_radius` | $\rho(a) = \rho_0 / 2^{\gamma/2}$ | exact | Formula |
| `test_power_law_slope_asymptotic` | $\rho \propto r^{-\gamma}$ at large $r$ | 1% | Asymptotic limit |
| `test_density_monotonic_decrease` | $d\rho/dr < 0$ | exact | Physical requirement |
| `test_gamma_affects_concentration` | Higher $\gamma \to$ more concentrated | qualitative | Physical trend |
| `test_all_particles_within_tidal_radius` | 100% at $r \leq r_t$ | exact | Hard boundary |

---

### 4. Initial Mass Functions

#### 4.1 Salpeter (1955)

**Source:** Salpeter (1955) ApJ 121:161

$$\xi(m) \propto m^{-\alpha}, \quad \alpha = 2.35$$

The original power-law IMF, still used for high-mass stars.

#### 4.2 Kroupa (2001)

**Source:** Kroupa (2001) MNRAS 322:231

Three-segment power law with continuity at breakpoints:

$$\xi(m) \propto \begin{cases}
m^{-0.3} & 0.01 \leq m < 0.08 \, M_\odot \\
m^{-1.3} & 0.08 \leq m < 0.5 \, M_\odot \\
m^{-2.3} & 0.5 \leq m \leq 100 \, M_\odot
\end{cases}$$

#### 4.3 Chabrier (2003)

**Source:** Chabrier (2003) PASP 115:763

Lognormal below 1 $M_\odot$, power-law above:

$$\xi_{\ln}(m) \propto \frac{1}{m} \exp\left[-\frac{(\ln m - \ln m_c)^2}{2\sigma^2}\right] \quad (m < 1 \, M_\odot)$$

$$\xi_{\rm pl}(m) \propto m^{-2.35} \quad (m \geq 1 \, M_\odot)$$

Parameters:
- Characteristic mass: $m_c = 0.08 \, M_\odot$
- Lognormal width: $\sigma = 0.69$
- High-mass slope: $\alpha = 2.35$ (Salpeter)

#### 4.4 Maschberger (2013)

**Source:** Maschberger (2013) MNRAS 429:1725

Smooth analytical IMF with closed-form PPF:
- Peak mass: $\mu = 0.2 \, M_\odot$
- High-mass slope: $\alpha \approx 2.3$

#### Tests & Tolerances

| Test | Physics Validated | Tolerance | Justification |
|------|-------------------|-----------|---------------|
| `test_salpeter_high_mass_slope` | $\alpha = 2.35$ | 0.01 | Exact parameter |
| `test_kroupa_breakpoint_masses` | Breaks at $[0.08, 0.5] \, M_\odot$ | exact | Exact parameters |
| `test_kroupa_segment_slopes` | $\alpha = [0.3, 1.3, 2.3]$ | 0.02 | Numerical derivative |
| `test_kroupa_pdf_continuous` | PDF continuous at breakpoints | 1% | Numerical continuity |
| `test_chabrier_characteristic_mass` | $m_c = 0.08 \, M_\odot$ | exact | Exact parameter |
| `test_chabrier_lognormal_width` | $\sigma = 0.69$ | exact | Exact parameter |
| `test_chabrier_high_mass_slope` | $\alpha = 2.35$ above 2 $M_\odot$ | 0.05 | Numerical derivative |
| `test_massive_star_fraction_salpeter` | 0.1-1% above 8 $M_\odot$ | range | Statistical |
| `test_few_very_massive` | <0.2% above 50 $M_\odot$ | range | Statistical |
| `test_grad_through_ppf` | Gradients flow through PPF | finite | Differentiability |

---

### 5. Binary Orbital Mechanics

**Source:** Murray & Dermott (1999) "Solar System Dynamics", Binney & Tremaine (2008)

#### Kepler's Third Law

$$T^2 = \frac{4\pi^2}{GM} a^3 \quad \Rightarrow \quad T \propto a^{3/2}$$

#### Kepler's Equation

$$M = E - e \sin E$$

where $M$ is mean anomaly and $E$ is eccentric anomaly.

#### Orbital Energy

$$E = -\frac{GM_1 M_2}{2a}$$

Independent of eccentricity.

#### Apsidal Distances

- Periapsis: $r_p = a(1 - e)$ at $M = 0$
- Apoapsis: $r_a = a(1 + e)$ at $M = \pi$

#### Circular Orbit Velocity

$$v = \sqrt{\frac{GM}{a}}$$

#### Conservation Laws

- Center of mass: $m_1 \mathbf{r}_1 + m_2 \mathbf{r}_2 = 0$
- Momentum: $m_1 \mathbf{v}_1 + m_2 \mathbf{v}_2 = 0$

#### Tests & Tolerances

| Test | Physics Validated | Tolerance | Justification |
|------|-------------------|-----------|---------------|
| `test_period_formula_circular` | $T = 2\pi\sqrt{a^3/GM}$ | 1e-10 | Exact formula |
| `test_period_scales_as_a_cubed` | $T \propto a^{3/2}$ | 1e-10 | Exact scaling |
| `test_period_semimajor_axis_roundtrip` | $a(T(a)) = a$ | 1e-10 | Inverse functions |
| `test_circular_orbit_mean_equals_eccentric` | $E = M$ for $e = 0$ | 0.01 | Kepler equation |
| `test_eccentric_orbit_periapsis` | $r = a(1-e)$ at $M = 0$ | 1e-6 | Exact formula |
| `test_eccentric_orbit_apoapsis` | $r = a(1+e)$ at $M = \pi$ | 1e-6 | Exact formula |
| `test_com_at_origin` | $m_1 r_1 + m_2 r_2 = 0$ | 1e-10 | Conservation law |
| `test_momentum_zero` | $m_1 v_1 + m_2 v_2 = 0$ | 1e-10 | Conservation law |
| `test_orbital_energy_formula` | $E = -GM_1 M_2 / 2a$ | 1% | Numerical precision |
| `test_circular_orbit_velocity` | $v = \sqrt{GM/a}$ | 1e-6 | Exact formula |
| `test_periapsis_velocity` | $v_{\rm peri} > v_{\rm apo}$ | qualitative | Kepler's 2nd law |

---

### 6. Environment-Conditioned IMF

**Source:** Marks et al. (2012) MNRAS 422:2246, Jeřábková et al. (2018) A&A 620:A39

#### Jeans Mass Scaling

$$M_J \propto T^{3/2} \rho^{-1/2}$$

Characteristic mass for gravitational fragmentation.

#### Bonnor-Ebert Mass

$$m_{\rm BE} \propto \frac{T^2}{P^{1/2}}$$

Includes pressure support.

#### Metallicity Effect on Cooling

Low metallicity $\to$ less cooling $\to$ higher gas temperature $\to$ larger $M_J$

#### Marks (2012) Prescription

High-mass slope $\alpha$ decreases with increasing density:

$$\alpha(\log n_H) = 2.3 - \Delta\alpha(\log n_H)$$

At $\log n_H \gtrsim 6$: top-heavy IMF ($\alpha < 2$)

#### Jeřábková (2018) Prescription

$\alpha$ depends on both metallicity $Z$ and density $n_H$:
- Low $Z$ + high density $\to$ very top-heavy ($\alpha \sim 1.5$)
- Solar $Z$ + normal density $\to$ Salpeter ($\alpha \sim 2.35$)

#### Tests & Tolerances

| Test | Physics Validated | Tolerance | Justification |
|------|-------------------|-----------|---------------|
| `test_temperature_scaling` | $M_J \propto T^{3/2}$ | 10% | Physical scaling |
| `test_density_scaling` | $M_J \propto \rho^{-1/2}$ | 10% | Physical scaling |
| `test_pressure_scaling` | $m_{\rm BE} \propto P^{-1/2}$ | exact | Exact formula |
| `test_metallicity_effect` | Low $Z \to$ larger $m_{\rm BE}$ | qualitative | Physical trend |
| `test_marks2012_density_dependence` | $\alpha$ decreases with $n_H$ | qualitative | Literature prescription |
| `test_jerabkova2018_metallicity_effect` | Low $Z \to$ lower $\alpha$ | qualitative | Literature prescription |
| `test_primordial_is_top_heavy` | $\alpha \leq 2.0$ for $Z \to 0$ | qualitative | Pop III IMF |

---

### 7. Environment-conditioned IMF — galaxy-wide (IGIMF) scope

progenax implements the **cluster-scale** environment→IMF mapping (see §6:
`BirthEnvironment` + `env_to_imf_params`), validated in
`tests/validation/test_environment_physics.py` and `tests/unit/imf/test_environment.py`.

The **galaxy-wide IGIMF integration** (ECMF aggregation + a Weidner–Kroupa
$M_{\rm ecl,max}(\mathrm{SFR})$ cap; Weidner & Kroupa 2004, Weidner et al. 2013) is
**background theory only — it is not implemented.** There is no `IGIMF`/`EnvironmentIMF`
class, no ECMF sampler, and no IGIMF/ECMF test suite. `tests/unit/test_documented_api.py`
asserts those phantom classes stay absent.

---

## Unit Tests

Verify functional correctness without full physics context.

| Module | Tests | Key Validations |
|--------|-------|-----------------|
| `profiles/` | 54 | Output shapes, finite values, isotropy, reproducibility |
| `kinematics/` | 33 | Velocity bounds, mean zero, dispersion scaling |
| `imf/` | 138 | CDF bounds [0,1], monotonicity, PPF inverse, sample bounds |
| `binaries/` | 14 | Element conversion, Kepler solver, roundtrips |
| `analytical/` | 14 | COM frame, momentum conservation, symmetry |
| `builders/` | 13 | IC generation, virial scaling, COM transform |

### IMF Parametrized Tests

`test_imf_core.py` runs 17 tests across 5 IMF types (85 total):

| IMF Type | Tests |
|----------|-------|
| Salpeter | CDF, PPF, sampling, normalization, JIT |
| Kroupa | CDF, PPF, sampling, normalization, JIT |
| Chabrier | CDF, PPF, sampling, normalization, JIT + specific tests |
| Maschberger | CDF, PPF, sampling, normalization, JIT |
| TruncatedChabrier | CDF, PPF, sampling, normalization, JIT |

**Note:** TaperedPowerLaw and Schechter excluded from parametrized tests due to O(10%) numerical precision from Newton iteration. Validated separately.

---

## Integration Tests

Verify JAX transformations work correctly.

### JIT Compilation (20 tests)

All sampling functions compile with `@jax.jit`:
- Profile `sample_positions()`
- Velocity DF `sample_velocities()`
- IMF `sample()`, `ppf()`, `cdf()`

### Gradient Flow (15 tests)

Gradients flow through:
- `jax.grad(loss)(r_h)` for profile parameters
- `jax.grad(loss)(m_min)` for IMF parameters
- End-to-end IC generation pipelines

### Vectorization (6 tests)

`jax.vmap` works for:
- PPF over quantile arrays
- logPDF over mass arrays

### No Spurious Recompilation (4 tests)

Functions don't recompile when called with same shapes.

---

## Test Fixtures & Constants

### Physics Constants

Defined in `conftest.py` from literature:

```python
class PlummerConstants:
    SCALE_RADIUS_FACTOR = sqrt(2^(2/3) - 1)  # ≈ 0.7664
    MASS_WITHIN_SCALE_RADIUS = 1/2^(3/2)     # ≈ 0.354
    MEAN_Q_SQUARED = 0.25                     # Beta(3/2, 9/2) mean

class KingConstants:
    W0_LOW = 3.0; W0_MED = 7.0; W0_HIGH = 12.0  # concentration ranges

class IMFConstants:
    SALPETER_ALPHA = 2.35
    KROUPA_ALPHAS = (0.3, 1.3, 2.3)
    KROUPA_BREAKS = (0.08, 0.5)  # M_sun
    CHABRIER_MC = 0.08           # M_sun
    CHABRIER_SIGMA = 0.69
```

### Tolerance Classes

```python
class PhysicsTolerances:
    EXACT = 1e-10              # Machine precision
    HIGH = 1e-6                # Numerical methods
    STANDARD = 0.05            # 5% (physics + statistics)
    VIRIAL_RATIO = 0.05        # 5% (regime-anchored: ~11-sigma at N=5000)
    HALF_MASS = 0.03           # 3% (half-mass radius)
    VELOCITY_DISPERSION = 0.10 # 10% (velocity statistics)
    BOUND_FRACTION = 1.0       # 100% (all particles bound)
```

### Sample Size Fixtures

```python
N_unit = 100        # Fast unit tests
N_integration = 1000  # Integration tests
N_validation = 5000   # Physics validation
N_stats = 10000       # Precise statistics
```

---

## References

### Density Profiles

- **Plummer (1911)** "On the problem of distribution in globular star clusters" MNRAS 71:460
- **King (1966)** "The structure of star clusters. III. Some simple dynamical models" AJ 71:64
- **Elson, Fall & Freeman (1987)** "The structure of young star clusters in the Large Magellanic Cloud" ApJ 323:54

### Velocity Distributions

- **Binney & Tremaine (2008)** "Galactic Dynamics" 2nd ed., Princeton University Press

### Initial Mass Functions

- **Salpeter (1955)** "The Luminosity Function and Stellar Evolution" ApJ 121:161
- **Kroupa (2001)** "On the variation of the initial mass function" MNRAS 322:231
- **Chabrier (2003)** "Galactic Stellar and Substellar Initial Mass Function" PASP 115:763
- **Maschberger (2013)** "On the function describing the stellar initial mass function" MNRAS 429:1725

### Environment-Conditioned IMF

- **Marks et al. (2012)** "Evidence for top-heavy stellar initial mass functions..." MNRAS 422:2246
- **Jeřábková et al. (2018)** "Impact of metallicity and star formation rate on the time-dependent IMF" A&A 620:A39

### Binary Mechanics

- **Murray & Dermott (1999)** "Solar System Dynamics" Cambridge University Press

---

*Last updated: 2026-06-12 | ~1243 released-core tests (see CI for the live count) | progenax v0.1.0*
