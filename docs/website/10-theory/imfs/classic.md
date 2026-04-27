---
title: Classical IMFs (Salpeter, Kroupa, Chabrier, Maschberger)
description: The four canonical single-star IMFs in progenax — Salpeter power-law, Kroupa multi-segment broken power-law, Chabrier lognormal+power-law, Maschberger smooth analytic — their parameter conventions, sampling, and trade-offs.
---

# Classical IMFs

The single-star initial mass function $\xi(m) \equiv \mathrm{d}N/\mathrm{d}m$
captures the birth-mass distribution of *individual* stars formed in
a single star-formation event. It is the standard input for population
synthesis, chemical evolution, and the IMF-inference problems progenax
is built to solve. Four functional forms dominate the literature, each
with a regime where it is the natural choice:

```{list-table}
:header-rows: 1
:widths: 22 22 56

* - IMF
  - Class
  - Use when
* - **Salpeter** (1955)
  - `PowerLawIMF`
  - Single power law $\xi(m) \propto m^{-2.35}$. Use when only the high-mass slope matters and the stellar sample doesn't reach below $\sim 1\,\Msun$.
* - **Kroupa** (2001)
  - `PowerLawIMF` with multi-segment
  - 3- or 4-segment broken power-law that captures the low-mass turnover. Standard default for resolved-cluster work.
* - **Chabrier** (2003)
  - `Chabrier`
  - Lognormal below $1\,\Msun$ matched to a Salpeter-like power-law above. Standard for unresolved-population integrated colours.
* - **Maschberger** (2013)
  - `Maschberger`
  - Smooth, three-parameter, *analytically invertible* IMF. Default for progenax production work because of its closed-form sampling.
* - **Truncated power-law**
  - `TruncatedIMF`
  - Hard cutoffs at $m_{\min}$ and $m_{\max}$. Useful for testing edge cases or modelling stellar populations with explicit upper-mass limits.
```

This chapter derives each form, lists the parameter values that turn
it into "the" canonical IMF, and explains why progenax picks
{cite:t}`Maschberger2013` as its default in production.

## Salpeter (1955): the original power-law

The first published IMF, fit to Galactic-disk solar-neighbourhood
stars {cite:p}`Salpeter1955`:

```{math}
:label: imf-salpeter
\xi(m) \;=\; \xi_0\,m^{-\alpha},\qquad \alpha = 2.35
```

valid for $m \gtrsim 1\,\Msun$. Below $\sim 1\,\Msun$ the actual stellar
distribution turns over (the IMF flattens and eventually decreases
toward sub-stellar masses), which Salpeter does *not* capture. Modern
usage typically restricts {eq}`imf-salpeter` to $m \in [1, m_{\max}]$
and pairs it with a separate low-mass model — which is essentially
the Kroupa construction below.

The cumulative is closed-form:

```{math}
:label: imf-salpeter-cdf
N(<m) / N \;=\; \frac{m^{1-\alpha} - m_{\min}^{1-\alpha}}{m_{\max}^{1-\alpha} - m_{\min}^{1-\alpha}}
```

inverted analytically for inverse-CDF sampling. progenax's
`PowerLawIMF(alpha=2.35, m_min=1.0, m_max=150.0)` is the Salpeter
configuration.

## Kroupa (2001): multi-segment broken power-law

{cite:t}`Kroupa2001` extends Salpeter to a piecewise broken power-law
that captures the observed low-mass turnover:

```{math}
:label: imf-kroupa
\xi(m) \;\propto\;
\begin{cases}
  m^{-\alpha_0}, & 0.01 \le m < 0.08\,\Msun & (\alpha_0 = 0.3) \\
  m^{-\alpha_1}, & 0.08 \le m < 0.50\,\Msun & (\alpha_1 = 1.3) \\
  m^{-\alpha_2}, & 0.50 \le m < 1.00\,\Msun & (\alpha_2 = 2.3) \\
  m^{-\alpha_3}, & 1.00 \le m \le m_{\max} & (\alpha_3 = 2.3)
\end{cases}
```

with continuity coefficients enforced at each break. The four-segment
form includes the brown-dwarf regime ($m < 0.08\,\Msun$); a common
three-segment usage drops $\alpha_0$ and starts at $0.08\,\Msun$. The
two high-mass slopes $\alpha_2 = \alpha_3 = 2.3$ match the Salpeter
value (different normalisation conventions account for the small
difference between Salpeter's $2.35$ and Kroupa's $2.3$).

`PowerLawIMF` accepts arbitrary segment break-points and slopes:

```python
from progenax.imf import PowerLawIMF

kroupa = PowerLawIMF(
    breaks=jnp.array([0.01, 0.08, 0.5, 1.0, 150.0]),
    alphas=jnp.array([0.3, 1.3, 2.3, 2.3]),
)
```

The class internally computes the continuity coefficients $a_i$ (so
that $\xi$ is continuous at each break) and the per-segment integrals
needed for normalisation and sampling.

## Chabrier (2003): lognormal + power-law

{cite:t}`Chabrier2003` replaces the broken-power-law low-mass piece
with a lognormal:

```{math}
:label: imf-chabrier
\xi(m) \;\propto\;
\begin{cases}
  \dfrac{1}{m\,\ln 10}\,\exp\!\Bigl[-\dfrac{(\log_{10} m - \log_{10} m_c)^2}{2\sigma_{\log m}^2}\Bigr], & m \le 1\,\Msun \\
  m^{-\alpha_3}, & m > 1\,\Msun
\end{cases}
```

with $m_c \approx 0.22\,\Msun$, $\sigma_{\log m} \approx 0.57$, and
$\alpha_3 = 2.3$, joined continuously at $m = 1\,\Msun$. The lognormal
provides a smoother low-mass description than the broken power-law,
which matters for unresolved-population integrated luminosities and
colours. progenax's `Chabrier` class is the standard implementation;
it shares its high-mass tail with `PowerLawIMF`.

The Chabrier IMF does *not* admit a closed-form inverse CDF. Sampling
uses a Newton solver on $\mathrm{CDF}(m) = u$; the iteration count is
fixed at 30 steps which is enough for double-precision convergence
across the entire mass range.

## Maschberger (2013): smooth and analytically invertible

The {cite:t}`Maschberger2013` "L3" form is progenax's production default
because it captures the same physics as Chabrier+Kroupa with a
*single* smooth functional form *and* a closed-form inverse CDF:

```{math}
:label: imf-maschberger
\xi(m) \;\propto\; \biggl(\frac{m}{\mu}\biggr)^{\!-\alpha}\,
                  \biggl[\,1 + \biggl(\frac{m}{\mu}\biggr)^{\!1-\alpha}\,\biggr]^{-\beta}
```

with default parameters $\alpha = 2.3$ (Salpeter slope), $\beta = 1.4$,
$\mu = 0.2\,\Msun$ (peak mass). The two factors interact:

- For $m \gg \mu$: the second factor approaches 1, leaving the
  high-mass power-law tail $\xi \propto m^{-\alpha}$ (= Salpeter).
- For $m \ll \mu$: the second factor suppresses the distribution,
  producing the observed turnover below $\sim 0.3\,\Msun$.

The closed-form CDF and its inverse are derived in {cite:t}`Maschberger2013`
§3. For $u \sim \mathcal{U}(0, 1)$,

```{math}
:label: imf-maschberger-inverse
m(u) \;=\; \mu\,\biggl[\biggl(\frac{P_{\min} + u\,(P_{\max} - P_{\min})}{C}\biggr)^{\!1/(1-\beta)} - 1\biggr]^{1/(1-\alpha)}
```

with $C = \mu / [(1-\beta)(1-\alpha)]$ and $P_{\min}, P_{\max}$ the
primitive evaluated at the integration endpoints. progenax's
`Maschberger` class uses {eq}`imf-maschberger-inverse` directly, with
no Newton solver, no rejection sampling, fully `vmap`-compatible.

```{admonition} Why progenax defaults to Maschberger
:class: note
The closed-form inverse CDF makes Maschberger the *only* IMF in this
chapter that can be sampled with a fixed number of operations per
particle. Inverse-CDF on Salpeter and Kroupa is also closed-form, but
those IMFs miss the low-mass turnover. Chabrier's Newton-solve approach
works but adds 30 iterations per particle. For HMC inference of the
IMF parameters (Maschberger's $\alpha$, $\beta$, $\mu$), the
analytical invertibility translates into substantially cleaner
gradients and a $\sim 5\times$ speed-up over Chabrier in `vmap`'d
sampling.
```

## Truncated power-law

`TruncatedIMF` wraps any of the above with explicit hard cutoffs at
$m_{\min}$ and $m_{\max}$. This is useful for:

- Testing edge cases (e.g. forcing $m_{\max} = 5\,\Msun$ to study a
  specific stellar regime).
- Cluster-mass-dependent upper limits via the
  {cite:t}`Marks2012` Weidner-Kroupa relation
  $m_{\max}(M_{\mathrm{cl}})$.
- Survey selection effects where stars below some completeness
  threshold are not observed.

The truncation is implemented via inverse-CDF resampling within the
truncated range — *not* by rejection — so all sampled masses are
guaranteed inside $[m_{\min}, m_{\max}]$ in a single pass.

## Comparison and choice guide

```{list-table}
:header-rows: 1
:widths: 22 22 22 34

* - IMF
  - Sampling cost
  - Continuity
  - Best for
* - Salpeter
  - $\mathcal{O}(1)$ inverse CDF
  - Power-law only
  - High-mass-only studies
* - Kroupa
  - $\mathcal{O}(1)$ per segment
  - Continuous, breaks at fixed masses
  - Resolved cluster work; backwards compatibility
* - Chabrier
  - $\mathcal{O}(30)$ Newton solver
  - Smooth (lognormal join)
  - Unresolved-population colours
* - Maschberger
  - $\mathcal{O}(1)$ inverse CDF
  - $C^\infty$ smooth
  - **progenax production default**
* - Truncated
  - As above + clamp
  - As above
  - Edge cases, $m_{\max}(M_{\mathrm{cl}})$
```

For new users: use `Maschberger` unless you have a specific reason to
pick another. For comparison with prior work using a different IMF,
match the convention of the comparison; progenax's flexible
`PowerLawIMF` + `Chabrier` cover the standard alternatives.

## Connection to binary-aware and environment IMFs

The IMFs above describe *single-star* distributions. They feed into the
binary-aware framework ([](binary.md)) where the *primary* mass is
drawn from $\xi(m \mid \alpha)$ and the *secondary* is drawn
conditionally from the {cite:t}`Moe2017` mass-ratio distribution. They
also feed into the environment-dependent framework ([](environment.md))
where the slopes $\alpha_{1,2,3}$ become functions of cloud-core
density and metallicity per {cite:t}`Marks2012`.

Both extensions wrap the classical IMF rather than replacing it: the
single-star $\xi$ remains the underlying object, with binary
multiplicity and environment dependence layered on top.

## References

The original parameterisations are {cite:t}`Salpeter1955`,
{cite:t}`Kroupa2001`, {cite:t}`Chabrier2003`, and {cite:t}`Maschberger2013`.
The validation suite is at [](../../50-validation/imf-statistics.md).
