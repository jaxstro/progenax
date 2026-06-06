# `gravoturb_fdf` — gravoturbulent + fractal-density-field initial conditions

> ## ⚠️ EXPERIMENTAL — follow-up paper
> This package is **not** part of the initial progenax/jaxstro release and is **not** shipped in
> the progenax wheel. It is a standalone, repo-only subsystem for a follow-up paper. It depends
> one-way on `progenax.cluster.turbulence`; **nothing in released progenax imports it.**
> Import it as `gravoturb_fdf` (after putting `src/experimental` on the path), never as
> `progenax.gravoturb` (that module was removed in the 2026-06 clean-room rewrite).

Clean-room rewrite (2026-06) of the gravoturbulent-1D + FDF-3D IC pipeline, authored from
PDF-grounded theory and validated by a committed acceptance suite that prints real numbers.
See [`VALIDATION_SUMMARY.md`](VALIDATION_SUMMARY.md) for the current AC1–AC10 results.

## Why a rewrite

The previous subsystem was built experimentally and contained concrete, caught fabrications (a
PP20 ζ(p) typo with a spurious `p=1.3` pole; a fabricated BM19 `s_t`; a PN11 prefactor ~2.3× off;
and a "validated" suite that actually ran a *white-noise* field through a *√N-less* Q estimator,
yielding a nonphysical headline). This package re-derives every formula from the held PDFs and
re-validates it before believing it. The cornerstone that was −37% in the old path is now
≤0.004% (AC6).

## Layout

```
gravoturb_fdf/
  theory/      bm19.py pp20.py pn11.py pdf.py gaussianization.py projection.py cic.py  # 1D PDF + predicted stats — JAX, differentiable
  field/       field.py tail.py sampling.py pipeline.py  # 3D realization — GRF + rank copula → stars
  diagnostics/ q.py                                   # CW04 Q substructure metric (numpy/scipy, non-diff)
  inference/   covariance.py likelihood.py fisher.py hmc.py  # differentiable predicted-statistics inference (blackjax NUTS)
  validation/  acceptance.py calibration.py measure.py  # AC1–AC17 printing scripts + Q(f_sub) driver + oracles
```

| Layer | Key public symbols |
|-------|--------------------|
| `theory.bm19` | `sigma_s_squared`, `transition_density`, `f_dense_bm19_full`, `f_dense_lognormal_limit`, `pdf_slope_to_radial` |
| `theory.pp20` | `magnification_factor`, `magnification_factor_with_core`, `zeta_fdf_direct` |
| `theory.pn11` | `virial_parameter`, `critical_overdensity_pn11`, `s_crit_pn11` |
| `theory.pdf` | `bm19_volume_pdf`, `bm19_icdf`, `bm19_icdf_analytic`, `bm19_mass_cdf`, `bm19_mean_density`, `build_bm19_cdf_table` |
| `field.field` | `gaussian_random_field`, `rank_copula_field`, `mass_conserving_copula_field`, `low_resolution_flag` |
| `field.tail` | `tail_weights`, `f_tail_actual` |
| `field.sampling` | `sample_cell_indices`, `cells_to_positions`, `sample_positions` |
| `field.pipeline` | `FDFField`, `build_fdf_field`, `cloud_to_stars` |
| `diagnostics.q` | `compute_q_parameter` (CW04, `A = πR²`) |
| `theory.gaussianization` / `projection` / `cic` | `gaussianized_xi`, `gaussian_correlation_grid`, `cic_variance`, `count_distribution` |
| `inference` | `data_vector`, `gaussian_loglike`, `count_loglike`, `tail_exceedance_loglike`, `alpha_fisher_info`, `sigma_alpha`, `run_nuts` |

## Use

The wheel packages only `src/progenax`, so `gravoturb_fdf` is dev/repo-only. Pytest sees it via
`pythonpath = ["src", "src/experimental"]` in `pyproject.toml`; for bare scripts, set the path:

```bash
cd progenax
PYTHONPATH=src:src/experimental python -m gravoturb_fdf.validation.acceptance   # print AC1–AC9
PYTHONPATH=src:src/experimental pytest tests/experimental -q                     # 150 experimental tests
```

```python
from gravoturb_fdf.theory.bm19 import sigma_s_squared, f_dense_bm19_full
from gravoturb_fdf.field.pipeline import build_fdf_field
import jax

sigma_s_squared(mach=5.0, b=0.4)              # ln(1 + b²ℳ²)  (FK10 Eq. 19)
f_dense_bm19_full(mach=8.0, b=0.5, alpha=1.8) # dense-mass fraction (BM19 Eq. 19–20)

# Build a 3D realization and read back the realized dense fraction:
fld = build_fdf_field(mach=8.0, b=0.5, alpha=1.8, beta=3.5,
                      shape=(128, 128, 128), key=jax.random.PRNGKey(0))
fld.f_dense, fld.f_dense_realized   # match to O(1/N) — the AC6 cornerstone
```

## Conventions

- **JAX-native cores.** `theory/` and `field/` use `jax.numpy`, `lax`, `vmap`/`grad`/`jit`,
  Equinox/jaxtyping; sampling uses fixed-iteration `lax.scan`, never `while_loop`. float64 is
  enabled at import. `diagnostics/q.py` and `validation/` are the *only* places numpy/scipy appear.
- **Differentiable interface = the predicted-statistics inference layer.** Categorical star sampling
  and the CW04 Q metric are non-differentiable, so inference predicts summary statistics analytically
  as smooth functions of θ and differentiates *those* (`inference/`; AC11–AC17). Q is a
  validation/demo diagnostic only. (The earlier fitted `q_surrogate` prototype was retired.)
- **Units.** CGS for microphysics; the turbulence relations consumed from
  `progenax.cluster.turbulence` use M☉/pc/Myr/(km·s⁻¹).

## Grounding

Burkhart & Mocz (2019); Parmentier & Pasquali (2020); Padoan & Nordlund (2011); Federrath et al.
(2010); Heyer (2009); Kim & Ryu (2005); Lomax et al. (2018); Cartwright & Whitworth (2004).
Per-paper notes: `docs/website/99-bibliography/per-paper/`. Authoritative spec:
`docs/plans/2026-06-05-fdf-clean-room-spec.md` (§8).
