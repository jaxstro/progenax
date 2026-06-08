---
title: Mass segregation validation
description: Current validation status for the Λ_MSR diagnostic (analytic ground truth), the energy-ordered generator, and the experimental mass-weighted local-Σ metric — with figures and run commands.
---

# Mass segregation validation

```{important}
Status (2026-06-08): the **Λ_MSR diagnostic is validated against analytic ground truth**
(`tests/validation/test_mass_segregation_physics.py`, 8 tests). The energy-ordered
**generator** is unit-tested + end-to-end checked. The **mass-weighted local-Σ metric**
(experimental, repo-only) is validated against the held Maschberger & Clarke (2011) PDF.
Regenerate every figure on this page with the two commands in
[Run it yourself](#run-it-yourself).
```

## 1. Λ_MSR diagnostic — validated against analytic ground truth

`progenax.diagnostics.compute_lambda_msr` (Allison et al. 2009;
$\Lambda_{\mathrm{MSR}}=\langle\ell_{\mathrm{random}}\rangle/\ell_{\mathrm{massive}}$,
definition verified against the held ApJ 700 L99 PDF — see
[](../99-bibliography/per-paper/allison-2009.md)). Tests in
`tests/validation/test_mass_segregation_physics.py`:

```{list-table}
:header-rows: 1

* - Check
  - Construction
  - Expected
  - Measured
* - Unsegregated
  - random masses on a field
  - $\Lambda \approx 1$
  - $1.00 \pm 0.11$
* - Maximally segregated
  - massive stars in a tight core
  - $\Lambda \gg 1$
  - $\sim 410$
* - Inverse
  - massive stars on the rim
  - $\Lambda < 1$
  - $0.17$
* - Exact ($N_{\mathrm{massive}}=2$)
  - vs independent `scipy.pdist` enumeration
  - exact ratio
  - $3.047$ vs $3.046$ ✓
* - Estimator convergence
  - $\sigma(\Lambda)$ vs $N_{\mathrm{random}}$
  - $\propto 1/\sqrt{N}$
  - confirmed
* - Binary caveat
  - tight massive pair, $N_{\mathrm{massive}}=2$
  - spurious inflation
  - $\sim 10^3$–$10^4\times$
```

:::{figure} figures/lambda_msr_regimes.png
:label: val-lambda-regimes
:width: 100%

$\Lambda_{\mathrm{MSR}}$ on three hand-constructed regimes (unsegregated → 1, maximally
segregated → ≫1, inverse → <1). ▲ marks the $N_{\mathrm{massive}}$ set.
:::

:::{figure} figures/lambda_msr_monotonic_convergence.png
:label: val-lambda-convergence
:width: 100%

Left: $\Lambda$ rises monotonically with segregation. Right: convergence to the exact value,
$\sigma(\Lambda)\propto 1/\sqrt{N_{\mathrm{random}}}$.
:::

:::{figure} figures/lambda_msr_binary_caveat.png
:label: val-lambda-binary
:width: 70%
:align: center

The documented binary caveat, quantified: a tight massive pair drives
$\Lambda \propto 1/\text{separation}$ (use binary centre-of-mass positions to avoid).
:::

## 2. Energy-ordered generator — end-to-end check

`progenax.cluster.mass_segregation.energy_sorted_segregation` (+ `MassSegregationLayer`):
unit-tested in `tests/unit/cluster/` (shape/permutation validity, $S=0/1$ limits, massive-stars-
more-bound). End-to-end via `scripts/validate_cluster_ic.py`: applying it to an unsegregated
Kroupa Plummer pool drives the *independently validated* $\Lambda_{\mathrm{MSR}}$ from **1.40 → 14.3**
and yields Spearman $\rho(m,E)=-0.84$ (massive stars in the most-bound orbits).

:::{figure} figures/cluster_ic_energy_sorted_segregation.png
:label: val-energy-sorted
:width: 100%

`energy_sorted_segregation` produces real, $\Lambda_{\mathrm{MSR}}$-detectable segregation.
:::

```{caution}
The **partial**-segregation knob `lambda_seg` is a *linear phase-space blend* between the
unsegregated and fully-Baumgardt catalogs (differentiable in `lambda_seg`), **not** a
first-principles partial-equilibrium model — intermediate states are interpolated. Its
intermediate calibration is not yet analytically validated (see the hardening checklist in
`docs/plans/2026-06-08-fdf-methods-paper-and-hardening-design.md`).
```

## 3. Mass-weighted substructure metric (experimental, repo-only)

`gravoturb_fdf.diagnostics.mass_density` — Maschberger & Clarke (2011) local surface density
$\Sigma=(k-1)/(\pi r_k^2)$, $k=6$ (Eq. 4, verified vs the held PDF —
[](../99-bibliography/per-paper/maschberger-clarke-2011.md)) + the m–Σ plane. Robust to
substructure (a *local* density), unlike CW04 $\mathcal{Q}$ on small massive subsets. Validated in
`tests/experimental/unit/test_mass_density.py` (6 tests: exact Eq. 4 formula; uniform-density
recovery; dense > sparse; random masses → ρ≈0; **primordial correlation detected** ρ(m,Σ)>0.5 when
massive stars are placed in dense clumps via `gravoturb_fdf.masses.correlated_mass_assignment`;
reproducible).

## 4. Differentiability & inference status

```{list-table}
:header-rows: 1

* - Component
  - Differentiable?
  - Inference implication
* - Generator `lambda_seg`
  - **Yes** (verified ∂/∂λ_seg finite)
  - forward model diff in segregation strength
* - Λ_MSR diagnostic
  - No (scipy MST)
  - —
* - m–Σ metric, correlated placement
  - No (kNN / ranking)
  - —
```

**Clean inference is via SBI** (the forward model is fast + samplable; diagnostics are validated
summaries). **Gradient-based/HMC inference of segregation is not yet available** — it needs a
differentiable segregation observable (a Λ_MSR/Σ surrogate, à la the existing differentiable
`q_approx` for substructure geometry).

(run-it-yourself)=
## Run it yourself

```bash
# (1) Λ_MSR diagnostic — analytic validation tests (released core)
pytest tests/validation/test_mass_segregation_physics.py -v

# (2) regenerate the figures on this page (released-core diagnostic + generator)
python scripts/validate_mass_segregation.py     # -> lambda_msr_*.png
python scripts/validate_cluster_ic.py           # -> cluster_ic_*.png
#   figures land in validation/plots/; the curated copies embedded here live in
#   docs/website/50-validation/figures/

# (3) experimental mass-weighted Σ metric + density-correlated placement
PYTHONPATH=src:src/experimental pytest tests/experimental/unit/test_mass_density.py tests/experimental/unit/test_masses.py -v
```

## References

Theory at [](../10-theory/tidal-and-substructure/mass-segregation.md). Diagnostic
{cite:t}`Allison2009`; energy-ordered generator {cite:t}`Baumgardt2008`; mass-weighted local-Σ
{cite:t}`MaschbergerClarke2011`.
