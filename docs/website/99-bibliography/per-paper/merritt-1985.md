---
title: Merritt (1985)
description: Annotated reference for D. Merritt — Spherical Stellar Systems with Spheroidal Velocity Distributions (the Osipkov–Merritt anisotropic DF method).
---

# Merritt (1985)

```{admonition} Spherical Stellar Systems with Spheroidal Velocity Distributions
:class: note

**Author.** David Merritt (Department of Astronomy, University of California, Berkeley).

**Reference.** *The Astronomical Journal* **90** (6), 1027–1037 (1985 June).
Received 1984 December 12; revised 1985 February 12.

**ADS.** [1985AJ.....90.1027M](https://ui.adsabs.harvard.edu/abs/1985AJ.....90.1027M)
(article locator 0004-6256/85/061027-11)
```

## Abstract (paraphrased)

A method for deriving families of **anisotropic** distribution functions (DFs) consistent with
*any* spherically symmetric density profile. Each family is labelled by a single free parameter
$r_a$, the **anisotropy radius**, and the radial-to-tangential velocity-dispersion ratio is
$\sigma_r^2/\sigma_t^2 = 1 \pm r^2/r_a^2$ (Eq. 15). The models are isotropic in the centre and
become radially ($+$) or tangentially ($-$) anisotropic outward. The radially anisotropic
("Type I") branch is what is now universally called the **Osipkov–Merritt** model (Osipkov 1979;
Merritt 1985). The construction reduces to a single Abel/Eddington inversion of an *augmented*
density, so analytic solutions follow whenever the isotropic Eddington integral is analytic.

## The method (verified against the paper, §II)

For an isotropic system the configuration density and phase-space density are related by

```{math}
:label: merritt-iso
\rho(r) = 4\pi \int_{U(r)}^{0} dE\,\sqrt{2[E-U(r)]}\,f(E),
\qquad
f(E) = \frac{\sqrt 2}{4\pi^2}\frac{d}{dE}\int_{E}^{0}\frac{d\rho}{dU}\frac{dU}{\sqrt{U-E}}
```

(Eqs. 1–2; the second is Eddington 1916). Merritt's key step: assume the DF depends on energy and
angular momentum *only* through the single variable

```{math}
:label: merritt-Q
Q_+ \equiv E + \frac{J^2}{2 r_a^2}
\qquad\text{(Eq. 4a), with}\qquad
f = f(Q_+).
```

Then the **augmented density**

```{math}
:label: merritt-augmented
\rho_1(r) \equiv \left(1 + \frac{r^2}{r_a^2}\right)\rho(r)
\qquad\text{(Eq. 9)}
```

plays exactly the role $\rho$ plays in the isotropic problem, so the anisotropic DF is recovered
by the **same Eddington inversion applied to $\rho_1$**:

```{math}
:label: merritt-fQ
f_{\mathrm I}(Q_+) = \frac{\sqrt 2}{4\pi^2}\frac{d}{dQ_+}
   \int_{Q_+}^{0}\frac{d\rho_1}{dU}\frac{dU}{\sqrt{U-Q_+}}
\qquad\text{(Eq. 11).}
```

This is the heart of architecture **(C)** in progenax: a single differentiable
augmented-density Eddington core, with the isotropic case recovered as $r_a \to \infty$
(then $\rho_1 \to \rho$ and {eq}`merritt-fQ` $\to$ {eq}`merritt-iso`).

## The anisotropy law (verified, Eqs. 15 & 17)

The velocity anisotropy of every Type I solution is **independent of the density profile**:

```{math}
:label: merritt-beta
\frac{\sigma_r^2}{\sigma_t^2} = 1 + \frac{r^2}{r_a^2}
\quad\Longleftrightarrow\quad
\beta(r) \equiv 1 - \frac{\sigma_t^2}{\sigma_r^2} = \frac{r^2}{r^2 + r_a^2}.
```

```{admonition} Convention reconciliation (so progenax's docstrings are consistent)
:class: important
Merritt writes $\sigma_t$ for a **single** tangential component (so $\beta = 1-\sigma_t^2/\sigma_r^2$,
Eq. 17). Binney & Tremaine define $\beta = 1 - \sigma_t^2/(2\sigma_r^2)$ with $\sigma_t^2 \equiv
\sigma_\theta^2 + \sigma_\phi^2$ the **two-component** tangential dispersion. The factor of 2 cancels,
so **both give the same** $\beta(r) = r^2/(r^2+r_a^2)$ — which is the profile already cited in
`progenax.kinematics`. Verified, not assumed.
```

$\beta(0)=0$ (isotropic centre), $\beta(r_a)=\tfrac12$, $\beta\to 1$ (radial) as $r\to\infty$;
radial motions already dominate by a factor $\sim2$ at $r=r_a$.

## The analytic anisotropic Plummer model (§III — progenax's validation anchor)

For the Plummer ($n=5$ polytrope) density $\rho \propto (1+r^2/r_0^2)^{-5/2}$ (Eq. 39) with
potential $U(r) = -6\sigma_0^2(1+r^2/r_0^2)^{-1/2}$ (Eq. 40), the inversion is analytic. The
**isotropic** DF is

```{math}
:label: merritt-plummer-iso
f(E) = \frac{\sqrt 2}{378\,\pi^3 G r_0^2 \sigma_0}\left(\frac{-E}{\sigma_0^2}\right)^{7/2},
\qquad -6 \le E/\sigma_0^2 \le 0
```

(Eq. 42; Eddington 1916) — i.e. $f(E)\propto(-E)^{7/2}$, confirming
`progenax.kinematics.PlummerVelocityDF`. The **Osipkov–Merritt (Type I) Plummer DF** is

```{math}
:label: merritt-plummer-aniso
f_{\mathrm I}(Q_+) = \frac{\sqrt 2}{378\,\pi^3 G r_0^2 \sigma_0}
   \left(\frac{-Q_+}{\sigma_0^2}\right)^{7/2}
   \left[\,1 - \frac{r_0^2}{r_a^2}
        + \frac{63}{4}\frac{r_0^2}{r_a^2}\left(\frac{-Q_+}{\sigma_0^2}\right)^{-2}\right]
\qquad\text{(Eq. 45).}
```

```{admonition} Non-negativity bound (Eq. 46) — a real constraint, not a tolerance
:class: warning
The phase-space density first goes negative (at $Q_+ = -6\sigma_0^2$) when $r_a = 3r_0/4$.
**A physical OM Plummer model therefore requires $r_a \ge \tfrac34 r_0 = 0.75\,r_0$.**
Smaller $r_a$ asks for more radial anisotropy than the finite-mass Plummer model can support
with $f\ge0$. progenax must enforce/flag this bound when constructing OM Plummer ICs.
```

The matched radial dispersion is $\sigma_r(r) = \sigma_0(1+r^2/r_0^2)^{-1/4}
[1 + \tfrac12 (r^2+r_0^2)/(r^2+r_a^2)]^{1/2}$, $\sigma_t = \sigma_r/\sqrt{1+r^2/r_a^2}$ (Eq. 47).
{eq}`merritt-plummer-aniso` is the closed form against which progenax's numerical
augmented-density inversion (and its $\beta(r)$ profile) are validated.

## Scope notes

- progenax uses only the **radially anisotropic Type I** branch ($\beta\ge0$). Merritt's
  **tangentially anisotropic** Type II/IIa/IIb solutions ($Q_-\equiv E - J^2/2r_a^2$, Eqs. 19–38)
  carry a velocity-dispersion discontinuity at $r=r_a$ and are out of scope.
- Merritt notes that Eddington's (1914) generalized isothermal sphere
  $f(E,J^2)\propto\exp[-(E+\beta J^2)]$ obeys the *same* anisotropy law — the lowered version of
  this is the **Michie (1963)** anisotropic King model, the route for an anisotropic
  `KingVelocityDF` (augmented-density inversion of the King density gives the same $\beta(r)$).
- **Linear superpositions** of different-$r_a$ solutions (Eq. 50) give a weighted-average
  $\beta(r)$ and a smooth way to avoid the Type II discontinuity — a possible future extension.

## Use in progenax

- `progenax.kinematics` — Osipkov–Merritt radial anisotropy ($\beta(r)=r^2/(r^2+r_a^2)$).
  The shared augmented-density Eddington core ({eq}`merritt-augmented`, {eq}`merritt-fQ`) is the
  basis of the true $f(Q_+)$ sampler that replaces the earlier heuristic velocity reshuffle.
- [](plummer-1911.md) · [](king-1966.md) · [](elson-fall-freeman-1987.md) — the spatial
  densities whose augmented forms are inverted.
- {eq}`merritt-plummer-aniso` + the $r_a\ge0.75\,r_0$ bound — the validation anchor for the
  anisotropic Plummer DF and its realized $\beta(r)$ profile.

## Notes

The Osipkov–Merritt model is the standard one-parameter route to radial anisotropy in spherical
ICs precisely because it reduces anisotropy to *one* Abel inversion of an augmented density,
keeping the construction (and, in progenax, its gradients) as cheap and differentiable as the
isotropic Eddington case. The analytic Plummer solution and its explicit $f\ge0$ bound make it an
unusually clean validation target.
