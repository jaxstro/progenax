---
title: Anisotropy & the OM-vs-Michie formalism (B6)
description: "A paper-seed demo on velocity anisotropy. For an Osipkov-Merritt cluster the anisotropy radius r_a is cleanly recovered from beta(r), with a detectability forecast. But the anisotropy FORMALISM matters: fitting the OM form to a Michie (anisotropic-King) cluster -- a genuinely different DF -- inflates the reduced chi^2 by 12.9x and mis-estimates r_a (8.9 vs 6.0). The anisotropy model you assume shapes what you infer."
---

# Anisotropy & the OM–vs–Michie formalism (B6)

Velocity anisotropy is encoded in the **Binney anisotropy**
$\beta(r) = 1 - \sigma_t^2/(2\sigma_r^2)$ — zero for isotropic orbits, rising toward
1 for radially-biased ones — and parametrized by an **anisotropy radius** $r_a$.
This demo asks two questions: how well can $r_a$ be measured, and does it matter
*which anisotropy formalism* you assume?

## (a) Well-specified: recovering Osipkov–Merritt $r_a$

The Osipkov–Merritt (OM) ansatz {cite:p}`Merritt1985` makes the DF a function of
$Q = E - L^2/2r_a^2$, giving the closed-form profile

```{math}
:label: b6-om
\beta_{\rm OM}(r) = \frac{r^2}{r^2 + r_a^2}.
```

A Plummer+OM cluster ($r_h=1$, true $r_a=1.5$ pc) is sampled, $\beta(r)$ binned, and
$r_a$ recovered by a $\chi^2$ fit of {eq}`b6-om`:

```{list-table}
:header-rows: 1

* - quantity
  - truth
  - recovered
* - $r_a$
  - $1.5$ pc
  - $1.55 \pm 0.035$ pc  (pull $+1.49$)
```

The OM form fits an OM cluster well (reduced $\chi^2 = 0.43$; the conservative
$\beta$ error $\sigma_\beta = (1+|\beta|)/\sqrt n$ deflates it), and the Fisher
forecast gives $\sigma(r_a)\propto N^{-1/2}$. Anisotropy this strong
($r_a\sim r_h$) is detectable in a modest sample.

## (b) Misspecified: an OM fit to a Michie cluster

Anisotropy comes in **different formalisms**. The Michie (1963) anisotropic-King DF
{cite:p}`Michie1963`

```{math}
:label: b6-michie
f \;\propto\; \exp\!\left(-\frac{J^2}{2 r_a^2 \sigma^2}\right)
              \left[\exp\!\left(-\frac{E}{\sigma^2}\right) - 1\right]
```

is **not** a function of a single $Q$, so its $\beta(r)$ shape differs from
{eq}`b6-om`. Sampling a Michie cluster ($W_0=7$, true $r_a=6$ pc) and fitting it with
the OM form:

```{list-table}
:header-rows: 1

* - quantity
  - value
* - OM-fit $r_a$
  - $8.90 \pm 0.27$ pc  (vs true Michie $r_a = 6.0$)
* - OM-fit reduced $\chi^2$
  - $5.58$  — a **12.9× inflation** over the well-specified fit
```

So assuming the wrong anisotropy formalism does two things: it **mis-estimates**
the anisotropy radius, and it leaves a **systematic $\beta(r)$ residual** (the
$12.9\times$ $\chi^2$ inflation) that *detects* the mismatch. The anisotropy model
you assume shapes what you infer — anisotropy is not a single number read off
$\beta(r)$ independent of the DF family.

## Figure

:::{figure} figures/demo_anisotropy.png
:label: sci-anisotropy
:width: 100%

**Anisotropy & the OM–vs–Michie formalism** (`scripts/demo_anisotropy.py`, ALL
PASS). **(a)** A Plummer+OM cluster: $\beta(r)$ (points, conservative errors) with
the recovered OM fit — a good match (reduced $\chi^2=0.43$). **(b)** A Michie
cluster fit with the OM form: the OM curve cannot follow the Michie $\beta(r)$
shape (reduced $\chi^2=5.58$, $12.9\times$ worse), and the best-fit $r_a=8.9$
mis-locates the true $6.0$. **(c)** Forecast: $\sigma(r_a)\propto N^{-1/2}$ for the
well-specified OM case.
:::

## Caveats

```{warning}
- **No "King+OM" model.** King's anisotropic form *is* Michie, so there is no
  fixed-density OM-vs-Michie pair. The misspecification here is an OM fit to a
  Michie *sample* on the $\beta(r)$ channel — not a self-consistent fixed-density
  refit. It demonstrates the formalism-dependence of the inferred $r_a$, which is
  the point.
- **$\beta(r)$ channel only.** The demo fits anisotropy alone; it does not co-fit
  concentration or the density (the $r_a$–concentration degeneracy needs the
  $\sigma(r)$ channel, already exercised in [B3](halo-core.md)).
- **Conservative $\beta$ errors.** $\sigma_\beta=(1+|\beta|)/\sqrt n$ deflates the
  reduced $\chi^2$; the misspecification result is the *ratio* ($12.9\times$),
  robust to the error scale.
- **The forecast $N\sim150$ is the asymptotic CRLB** for this strong anisotropy
  ($r_a\sim r_h$); the small-$N$ bound is optimistic, and weaker anisotropy needs
  far more stars.
```

## How to run

```bash
env -u VIRTUAL_ENV uv run --no-sync python scripts/demo_anisotropy.py
```

## References

Osipkov–Merritt anisotropy is {cite:t}`Merritt1985`; the Michie anisotropic-King
DF is {cite:t}`Michie1963`. The OM and Michie velocity DFs are documented on the
[kinematics](../10-theory/populations/index.md) pages; B3's two-component OM
recovery is at [halo + core](halo-core.md).
