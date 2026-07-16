# `gravoturb` — gravoturbulent initial conditions (turbulent density fields → star clusters)

> ## ⚠️ EXPERIMENTAL — follow-up paper
> This package is **not** part of the initial progenax/jaxstro release and is **not** shipped in
> the progenax wheel. It is a standalone, repo-only subsystem for a follow-up paper. It depends
> one-way on `progenax.cluster.turbulence`; **nothing in released progenax imports it.**
> Import it as `gravoturb` (after putting `src/experimental` on the path), never as
> `progenax.gravoturb` (that module was removed in the 2026-06 clean-room rewrite).

Clean-room rewrite (2026-06) of the gravoturbulent IC pipeline — 1-D density-PDF theory plus a
3-D turbulent-density-field realization — authored from PDF-grounded theory and validated by a
committed acceptance suite that prints real numbers.
See [`VALIDATION_SUMMARY.md`](VALIDATION_SUMMARY.md) for the current AC1–AC17 + AC-IC0–IC7 results.

## Why a rewrite

The previous subsystem was built experimentally and contained concrete, caught fabrications (a
PP20 ζ(p) typo with a spurious `p=1.3` pole; a fabricated BM19 `s_t`; a PN11 prefactor ~2.3× off;
and a "validated" suite that actually ran a *white-noise* field through a *√N-less* Q estimator,
yielding a nonphysical headline). This package re-derives every formula from the held PDFs and
re-validates it before believing it. The cornerstone that was −37% in the old path is now
≤0.004% (AC6).

## Layout

```
gravoturb/
  specs.py     CloudSpec GeometrySpec VelocitySpec CompositionSpec  # typed, validated parameter specs
  cluster.py   build_cluster_ic → ClusterIC             # end-to-end natal-parameters → N-body IC
  theory/      density_pdf.py density_cdf.py dense_gas_sfr.py collapse_threshold.py
               log_correlations.py projection.py counts_in_cells.py  # 1D PDF + predicted stats — JAX, differentiable
  realization/ gaussian_field.py copula.py placement.py pipeline.py  # 3D realization — GRF + rank copula → stars
               envelope.py turbulent_velocity.py mass_assignment.py  # cluster shape (log-envelope) + coherent turbulent velocities + density-correlated masses
  diagnostics/ q.py mass_density.py                     # CW04 Q substructure metric (numpy/scipy, non-diff) + mass-density diagnostics
  inference/   covariance.py likelihood.py fisher.py hmc.py       # differentiable predicted-statistics inference (blackjax NUTS)
               projected_logp.py priors.py sbc.py flow_npe.py diagnostics.py  # 2D projected-β forward model, priors, SBC, NPE baseline, HMC diagnostics
  validation/  acceptance.py calibration.py measure.py  # AC1–AC17 printing scripts + Q(f_sub) driver + oracles
               cluster_acceptance.py                    # AC-IC0–IC7 cluster-IC suite + figure gallery
```

| Layer | Key public symbols |
|-------|--------------------|
| `theory.density_pdf` | `sigma_s_squared`, `transition_density`, `dense_mass_fraction`, `dense_mass_fraction_lognormal`, `pdf_slope_to_radial` |
| `theory.dense_gas_sfr` | `magnification_factor`, `magnification_factor_with_core`, `zeta_from_field` |
| `theory.collapse_threshold` | `virial_parameter`, `critical_overdensity`, `critical_log_density` |
| `theory.density_cdf` | `log_density_pdf`, `log_density_icdf`, `log_density_icdf_analytic`, `mass_cdf`, `mean_density`, `build_cdf_table` |
| `realization.gaussian_field` | `gaussian_random_field`, `low_resolution_flag` |
| `realization.copula` | `rank_copula_field`, `mass_conserving_copula_field`, `rank_to_uniform` |
| `realization.placement` | `collapse_weights`, `f_tail_actual`, `sample_cell_indices`, `cells_to_positions`, `sample_positions` |
| `realization.pipeline` | `TurbulentField`, `build_turbulent_field`, `cloud_to_stars` |
| `diagnostics.q` | `compute_q_parameter` (CW04, `A = πR²`) |
| `theory.log_correlations` / `projection` / `cic` | `gaussianized_xi`, `gaussian_correlation_grid`, `cic_variance`, `count_distribution` |
| `inference` | `data_vector`, `gaussian_loglike`, `count_loglike`, `tail_exceedance_loglike`, `alpha_fisher_info`, `sigma_alpha`, `run_nuts` |

## Module reference

The package has two faces. **`theory/` + `inference/` are the analytic, differentiable forward
model and its likelihood** — they predict summary statistics as smooth functions of
$\theta=(\mathcal{M}, b, \alpha, \beta)$ and never realize a random field. **`realization/` +
`diagnostics/` + `validation/` are the stochastic realization simulator and its oracles** — they
draw actual fields and stars, and serve as the ground truth against which the analytic predictions
are checked. The design philosophy (predict the statistic, don't differentiate the simulator) is
explained pedagogically in
[`docs/.../gravoturbulence/differentiable-inference.md`](../../../docs/website/10-theory/gravoturbulence/differentiable-inference.md).

### `theory/` — analytic density-PDF physics + predicted statistics (JAX, differentiable)

- **`density_pdf.py`** — the one-point physics: lognormal width `sigma_s_squared` $=\ln(1+(b\mathcal{M})^2)$
  (FK10 Eq. 19), the lognormal→powerlaw transition `transition_density` $s_t=(\alpha-\tfrac12)\sigma_s^2$,
  and the self-gravitating mass fraction `dense_mass_fraction`.
- **`density_cdf.py`** — the BM19 *volume* PDF `log_density_pdf` (lognormal body + power-law tail), its CDF
  and analytic inverse-CDF (`log_density_icdf`, `log_density_icdf_analytic`) — the marginal the field is mapped to,
  and the engine of the rank copula — plus `mass_cdf` / `mean_density`.
- **`dense_gas_sfr.py`, `collapse_threshold.py`** — the dense-gas SFR side: the Parmentier & Pasquali ζ magnification factor
  and the Padoan & Nordlund critical density (the forward-SFR chapters, [](../../../docs/website/10-theory/gravoturbulence/index.md)).
- **`log_correlations.py`** — the log-density **2-point** $\xi_s(r)=\sum_n (c_n^2/n!)\,\rho_g(r)^n$ via
  the Hermite coefficients $c_n$ of the copula map (`gaussianized_xi`). The $\beta$-carrier; the
  Szapudi & Pan / Coles & Jones machinery.
- **`projection.py`** — the Gaussian correlation $\rho_g(r;\beta)$ from $k^{-\beta}$
  (`gaussian_correlation_grid`), Gaussian smoothing at a cell scale, and the **Limber** 3-D→2-D
  projection (the data are 2-D sky positions).
- **`counts_in_cells.py`** — counts-in-cells: the cell-averaged linear variance and CIC moment `cic_variance`
  $\sigma_N^2=\bar N+\bar N^2\bar\xi$, and the compound-Poisson count distribution `count_distribution`
  $P(N)$ (the locally-Poisson CIC of Szapudi & Pan).

### `realization/` — the stochastic 3-D realization (the ground-truth oracle; non-differentiable)

- **`gaussian_field.py`** — `gaussian_random_field` (FFT, $P(k)\propto k^{-\beta}$);
  `low_resolution_flag` guards the under-resolved-tail regime.
- **`copula.py`** — the copula map to the BM19 marginal: `rank_copula_field` (faithful volume
  marginal — used for the tail), `mass_conserving_copula_field` (exact `f_dense` — used for the
  cornerstone), and `rank_to_uniform`.
- **`placement.py`** — the soft dense-tail mask (`collapse_weights`, `f_tail_actual`) plus the star
  samplers: `sample_cic_counts` (clean inhomogeneous-Poisson counts), the categorical tail/smooth
  sampler, and the default multi-freefall (∝ w·ρ^{3/2}) placement (`sample_positions`).
- **`envelope.py`** — the cluster shape: any `SpatialProfile` applied additively in log-space to
  the turbulent field.
- **`turbulent_velocity.py`** — coherent turbulent velocity field (β_v) + virial `Q_target` rescale.
- **`mass_assignment.py`** — optional density-correlated (primordial-segregation) mass assignment.
- **`pipeline.py`** — the end-to-end `build_turbulent_field` and `cloud_to_stars`.

### `diagnostics/` — substructure metric (validation/demo only)

- **`q.py`** — the Cartwright & Whitworth `compute_q_parameter` ($Q=\bar m/\bar s$, area $A=\pi R^2$).
  Non-differentiable (MST); demoted to a diagnostic ("we reproduce fractal clusters"), never a fit
  observable.

### `inference/` — the differentiable inference layer

- **`covariance.py`** — power-spectrum band-powers (`power_spectrum_bandpowers`) and the
  Hartlap-corrected **mock** covariance/precision (the Gaussian band-power covariance underestimates
  the true non-Gaussian one, so the mock covariance is used).
- **`likelihood.py`** — the blocks: `data_vector` + `gaussian_loglike` (2-pt band-powers + CIC
  variance); `count_loglike` (compound-Poisson counts → $\mathcal{M}, \beta$);
  **`tail_exceedance_loglike`** (the peaks-over-threshold truncated-exponential tail block → $\alpha$;
  shift-immune in `s_thr`, so no validity barrier is needed); `density_pdf_loglike` (a full-PDF
  diagnostic, superseded for $\alpha$ by the POT block).
- **`fisher.py`** — `fisher_matrix` / `marginal_errors` (the field-level forecast) and the
  truncation-corrected POT forecast `alpha_fisher_info` / `sigma_alpha` (the honest σ(α)-vs-$N_{\rm tail}$ curve).
- **`hmc.py`** — `run_nuts` (a thin blackjax NUTS driver) with a bounded log/log-shift
  reparametrization ($\mathcal{M}>0$, $\alpha>1$, $\beta>0$) and the log-Jacobian.

### `validation/` — acceptance scripts + measurement oracles (numpy permitted)

- **`acceptance.py`** — the AC1–AC17 printing scripts that emit real expected-vs-measured tables
  (the evidence for every "validated" claim); `main()` runs the whole suite.
- **`calibration.py`** — `measure_q_ensemble`, `q_vs_fsub` (the AC7 $Q(f_{\rm sub})$ calibration).
- **`measure.py`** — the oracle measurements: `autocovariance_3d` / band-powers (Wiener–Khinchin),
  `smooth_copula_field`, and `measure_exceedances` (the gas-tail → exceedance histogram for the POT
  block).
- **`cluster_acceptance.py`** — the AC-IC0–IC7 cluster-IC acceptance suite + figure gallery.

## Use

The wheel packages only `src/progenax`, so `gravoturb` is dev/repo-only. Pytest sees it via
`pythonpath = ["src", "src/experimental"]` in `pyproject.toml`; for bare scripts, set the path:

```bash
cd progenax
PYTHONPATH=src:src/experimental python -m gravoturb.validation.acceptance   # print AC1–AC17
PYTHONPATH=src:src/experimental pytest tests/experimental -q                     # experimental tests (see CI for the live count)
```

```python
from gravoturb.theory.density_pdf import sigma_s_squared, dense_mass_fraction
from gravoturb.realization.pipeline import build_turbulent_field
import jax

sigma_s_squared(mach=5.0, b=0.4)              # ln(1 + b²ℳ²)  (FK10 Eq. 19)
dense_mass_fraction(mach=8.0, b=0.5, alpha=1.8) # dense-mass fraction (BM19 Eq. 19–20)

# Build a 3D realization and read back the realized dense fraction:
fld = build_turbulent_field(mach=8.0, b=0.5, alpha=1.8, beta=3.5,
                      shape=(128, 128, 128), key=jax.random.PRNGKey(0))
fld.f_dense, fld.f_dense_realized   # match to O(1/N) — the AC6 cornerstone

# End-to-end cluster IC via the typed specs (validated at construction):
from jaxstro.units import STELLAR
from progenax import PlummerProfile
from gravoturb import (CloudSpec, GeometrySpec, VelocitySpec, CompositionSpec,
                       build_cluster_ic)
import jax.numpy as jnp

ic = build_cluster_ic(
    jnp.ones(1000),
    cloud=CloudSpec(mach=8.0, b=0.5, alpha=1.8, beta=3.0),
    geometry=GeometrySpec(profile=PlummerProfile(r_h=0.5), box_size=4.0, shape=(32,)*3),
    velocity=VelocitySpec(beta_v=4.0, Q_target=0.5),
    composition=CompositionSpec(),  # multi_freefall default; f_sub is DERIVED
    # (legacy ablation mode: CompositionSpec(placement="two_population", f_sub=0.3))
    G=STELLAR.G, key=jax.random.PRNGKey(0),
)
ic.positions, ic.velocities, ic.Q_virial, ic.tail_star_fraction, ic.field.f_dense_realized
```

## Conventions

- **JAX-native cores.** `theory/` and `realization/` use `jax.numpy`, `lax`, `vmap`/`grad`/`jit`,
  Equinox/jaxtyping; sampling uses fixed-iteration `lax.scan`, never `while_loop`. float64 is
  enabled at import. `diagnostics/q.py` and `validation/` are the *only* places numpy/scipy appear.
- **Differentiable interface = the predicted-statistics inference layer.** Categorical star sampling
  and the CW04 Q metric are non-differentiable, so inference predicts summary statistics analytically
  as smooth functions of θ and differentiates *those* (`inference/`; AC11–AC17). Q is a
  validation/demo diagnostic only. (The earlier fitted `q_surrogate` prototype was retired.)
- **Units.** CGS for microphysics; the turbulence relations consumed from
  `progenax.cluster.turbulence` use M☉/pc/Myr/(km·s⁻¹).

## Grounding

Forward model: Burkhart & Mocz (2019); Parmentier & Pasquali (2020); Padoan & Nordlund (2011);
Federrath et al. (2010); Heyer (2009); Kim & Ryu (2005); Lomax et al. (2018); Cartwright &
Whitworth (2004). Inference layer (Gaussianization / log-density): Coles & Jones (1991); Szapudi &
Pan (2004); Szapudi et al. (2005); Neyrinck, Szapudi & Szalay (2009, 2011); Carron & Szapudi (2013,
2014); Carron, Wolk & Szapudi (2014); Hoffman & Gelman (2014, NUTS); contrasted with the SBI
approach of Bairagi & Wandelt (2026). Per-paper notes:
`docs/website/99-bibliography/per-paper/`. Authoritative spec:
an internal clean-room spec (§8).
