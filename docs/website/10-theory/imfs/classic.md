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

:::{admonition} Who this page is for
:class: note
**Audience:** new students & researchers choosing a single-star IMF; no prior IMF literature assumed.
**Prerequisites:** the [IMF overview](index.md) (the `IMFProtocol` contract and notation) — the entry point for this track.
**You'll get:** the four canonical IMFs (Salpeter, Kroupa, Chabrier, Maschberger) plus truncation, their parameter conventions and sampling costs, and why progenax defaults to the analytically-invertible Maschberger form.
:::

```{list-table}
:header-rows: 1

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
  - `ChabrierIMF`
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

```{figure} ../figures/imf_classic_slopes.webp
:label: fig-imf-classic-slopes
:width: 100%

The four classic IMFs and why smoothness matters. **(a)** $m\,\xi(m) =
\mathrm{d}N/\mathrm{d}\ln m$ for each family on its canonical support
(curves: analytic; faint steps: $2\times 10^5$ sampled masses each, seed 42 —
the sampler-equals-theory check). **(b)** the local slope
$S(m) = -\mathrm{d}\ln\xi/\mathrm{d}\ln m$ computed **by autodiff**:
Kroupa's breakpoints and Chabrier's $1\,\Msun$ join are $C^0$ kinks in
exactly the quantity gradient-based inference differentiates through, while
Maschberger is one smooth curve — the reason it is progenax's production
default. Regenerate: `python -m laboratory.icviz --only imf-classic-slopes`.
```


## Salpeter (1955): the original power-law

The first published IMF, fit to Galactic-disk solar-neighbourhood
stars {cite:p}`Salpeter1955`:

```{math}
:label: imf-salpeter
\xi(m) \;=\; \xi_0\,m^{-\alpha},\qquad \alpha = 2.35
```

[↗ model card](#card-imf-salpeter)

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

[↗ model card](#card-imf-salpeter-cdf)

inverted analytically for inverse-CDF sampling. progenax's
`PowerLawIMF.salpeter(m_min=1.0, m_max=150.0)` is the Salpeter
configuration (internally `exponents=[2.35], breakpoints=[]`).

:::{warning}
**The 1.35-vs-2.35 convention trap.** Salpeter's paper quotes the slope
$\Gamma = 1.35$ — in *logarithmic* mass, $\mathrm{d}N/\mathrm{d}\log m
\propto m^{-\Gamma}$. progenax's $\alpha = 2.35$ is the *linear*-mass
slope, $\xi(m) = \mathrm{d}N/\mathrm{d}m \propto m^{-\alpha}$, with
$\alpha = \Gamma + 1$. Mixing the two conventions is the single most common
IMF-fitting error; every progenax IMF uses the linear-mass $\alpha$.
:::

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

[↗ model card](#card-imf-kroupa)

with continuity coefficients enforced at each break. The four-segment
form includes the brown-dwarf regime ($m < 0.08\,\Msun$); a common
three-segment usage drops $\alpha_0$ and starts at $0.08\,\Msun$. The
two high-mass slopes $\alpha_2 = \alpha_3 = 2.3$ match the Salpeter
value (different normalisation conventions account for the small
difference between Salpeter's $2.35$ and Kroupa's $2.3$).

`PowerLawIMF` accepts arbitrary segment slopes and break-points
(`breakpoints` lists the *interior* breaks only, excluding `m_min` and
`m_max`):

```python
from progenax.imf import PowerLawIMF

# canonical Kroupa: the two equal high-mass slopes (2.3) merge to one
# segment, so the classmethod uses 3 segments / 2 breaks (exact).
kroupa = PowerLawIMF.kroupa(m_min=0.01, m_max=150.0)

# or build the explicit 4-segment form directly:
kroupa4 = PowerLawIMF(
    exponents=[0.3, 1.3, 2.3, 2.3],
    breakpoints=[0.08, 0.5, 1.0],
    m_min=0.01,
    m_max=150.0,
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

[↗ model card](#card-imf-chabrier)

with the {cite:t}`Chabrier2003` Table 1 **single-star (disk)** values
$m_c \approx 0.08\,\Msun$ ($0.079$), $\sigma_{\log m} = 0.69$, and
$\alpha_3 = 2.3$, joined continuously at $m = 1\,\Msun$. (The
often-quoted $m_c = 0.22$, $\sigma = 0.57$ are the *system* IMF;
progenax's `ChabrierIMF` defaults to the single-star disk form — its
defaults are `m_c=0.08, sigma=0.69, alpha=2.3, m_trans=1.0`.) The
lognormal provides a smoother low-mass description than the broken
power-law, which matters for unresolved-population integrated
luminosities and colours. `ChabrierIMF` is the standard implementation;
it shares its high-mass tail slope with `PowerLawIMF`.

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

[↗ model card](#card-imf-maschberger)

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

[↗ model card](#card-imf-maschberger-inverse)

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

Just as important as invertibility is **smoothness**: Maschberger is one
$C^\infty$ functional form, whereas Kroupa's segment breaks are $C^0$
kinks (and Chabrier keeps a residual kink at $1\,\Msun$). Kinks propagate
into the derivatives that gradient-based inference and Fisher forecasting
differentiate through — so prefer Maschberger unless a like-for-like
comparison against a Kroupa/Chabrier-defined literature analysis demands
the segmented forms.
```

## Check yourself

:::{dropdown} 1. The slope trap — reconcile 1.35 with 2.35
Salpeter (1955) reports a slope of $1.35$; `PowerLawIMF.salpeter()` uses
$2.35$. Before peeking: which convention is which, and how do they relate?

Verify numerically: sample $10^5$ masses, histogram them in $\log m$, and
fit the high-mass tail slope — you should recover $\Gamma \approx 1.35 =
\alpha - 1$ (the $\mathrm{d}\ln m = m\,\mathrm{d}m$ Jacobian shifts the
exponent by one).
:::

:::{dropdown} 2. Rank the mean masses — then explain the outlier
Predict the ordering of $\langle m \rangle$ for the four default
configurations, then run `imf.mean_mass()` for each. Measured values
(analytic, this checkout): Salpeter $0.351$, Maschberger $0.367$, Kroupa
$0.376$, **Chabrier $0.607\,\Msun$**.

The lesson: the biggest driver is the *support*, not the shape — Chabrier's
default $m_{\min} = 0.08$ (the hydrogen-burning limit) excludes the brown
dwarfs the other defaults include, nearly doubling $\langle m \rangle$;
Salpeter's missing turnover is almost exactly offset by its $m_{\min} = 0.1$
floor. For a fixed total mass, your IMF choice changes the *number of stars*
by the same factor.
:::

:::{dropdown} 3. Feel the kink with `jax.grad`
Compute `jax.grad(lambda mu: Maschberger(mu=mu).mean_mass())(0.2)` — a
clean $\mathrm{d}\langle m\rangle/\mathrm{d}\mu \approx 1.52$. Now try
the same through a Kroupa *breakpoint* (rebuild `PowerLawIMF` with a traced
break). Both are finite — but relate what you see to panel (b) of
{numref}`fig-imf-classic-slopes`: the Maschberger derivative is smooth in
*all* its parameters everywhere, while a sampled mass sitting exactly at a
Kroupa break feels a $C^0$ kink. Smoothness is why HMC chains on
($\alpha, \beta, \mu$) mix cleanly.
:::

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
`PowerLawIMF` + `ChabrierIMF` cover the standard alternatives.

## Connection to binary-aware and environment IMFs

The IMFs above describe *single-star* distributions. They feed into the
binary-aware framework ([](binary.md)) where the *primary* mass is
drawn from $\xi(m \mid \alpha)$ and the *secondary* is drawn
conditionally from the {cite:t}`MoeDiStefano2017` mass-ratio distribution. They
also feed into the environment-dependent framework ([](environment.md))
where the slopes $\alpha_{1,2,3}$ become functions of cloud-core
density and metallicity per {cite:t}`Marks2012`.

Both extensions wrap the classical IMF rather than replacing it: the
single-star $\xi$ remains the underlying object, with binary
multiplicity and environment dependence layered on top.

## Implementation, validation & references

- **In code:** `src/progenax/imf/power_law.py` (`PowerLawIMF`, the
  Salpeter/Kroupa classmethods), `chabrier.py` (`ChabrierIMF`),
  `smooth.py` (`Maschberger`), and `truncated.py` (`TruncatedIMF`) — see
  the [IMF API](../../30-api/imf.md).
- **Validated in:** [IMF statistics](../../50-validation/imf-statistics.md)
  — the regression suite that checks each sampled distribution against
  its analytic form.
- **Primary sources:** {cite:t}`Salpeter1955`, {cite:t}`Kroupa2001`,
  {cite:t}`Chabrier2003`, and {cite:t}`Maschberger2013` — full notes in
  the [bibliography](../../99-bibliography/index.md).
