# Retained absolute LIMEPY reference fixtures

These 13 files contain one base solution and central finite-difference pairs
for the IMF slope $\alpha$, segregation parameter $\delta$ and central
potential $W_0$, at steps $10^{-4}$ and $10^{-5}$. They support
`tests/validation/test_limepy_reference_parity.py`; they are compact independent
reference fixtures, rather than generated stellar catalogues.

The source is the NumPy/SciPy LIMEPY implementation accompanying Gieles &
Zocchi (2015), upstream Git revision
`ef2a4796a006506a2d38f35b368c039a37ae15f1`, with source SHA256
`30a4573adf618cda6673bf19d07958fa4f6004bd05b45efa2c6d512518370ca1`.
The reference uses four Maschberger-derived components over
$0.1$–$20\,M_\odot$, $(\alpha,\delta,W_0)=(2.3,0.4,5)$, truncation index
$g=1$, the central mean-mass convention, and retained model units
$G=r_c=1$. Each file retains the component mass and kinetic integrals,
physical scales, fit residual, source version and solver controls. The finest
reference uses ODE relative/absolute tolerances $10^{-11}$ and relative
mass-fraction convergence $10^{-12}$.

The comparison targets a fixed population of 100,000 stars. The unchanged
tests require the reference fit residual to be at most $10^{-10}$,
nonzero first derivatives to agree within $0.3\%$, and theoretically zero
mass-shape derivatives to satisfy the scaled $10^{-8}$ identity criterion.
They compare the retained 2,000-node/$\xi_{\max}=300$ and
8,000-node/$\xi_{\max}=30$ Progenax grids. These are bounded first-order
checks; they do not qualify other domains, IMF binning or stochastic precision.

The fixtures were copied from the retained comparison. Only the local
repository-location string in `provenance.ref_repo` was replaced with the
upstream revision identity for publication. Every other metadata field and
every numerical array retains its original dtype, shape and bytes.

To load the fixtures without silently skipping missing files, set
`PROGENAX_STRICT_REFS=1` when running the parity test. The adapter is
`scripts/validate_limepy_reference.py`; its `--absolute-output` and
`--reference-python` options regenerate a comparison in a new output location
using a separately installed reference environment. No reference solve runs
when these cached files are loaded.
