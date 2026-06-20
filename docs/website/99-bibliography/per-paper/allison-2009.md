---
title: Allison et al. (2009)
description: Annotated reference for the two companion Allison et al. (2009) mass-segregation papers — the Λ_MSR method (MNRAS) and its short-timescale application (ApJ).
---

# Allison et al. (2009)

```{admonition} Two companion papers — read them together
:class: note

**Authors.** R. J. Allison, S. P. Goodwin, R. J. Parker, R. de Grijs, S. F. Portegies Zwart, M. B. N. Kouwenhoven.

**Method paper.** "Using the minimum spanning tree to trace mass segregation," *MNRAS* **395, 1449** (2009) — the formal $\Lambda_{\mathrm{MSR}}$ definition + the ONC measurement. **DOI** [10.1111/j.1365-2966.2009.14508.x](https://doi.org/10.1111/j.1365-2966.2009.14508.x).

**Application paper.** "Dynamical Mass Segregation on a Very Short Timescale," *ApJ* **700, L99–L103** (2009) — the cool-fractal N-body demonstration. **DOI** [10.1088/0004-637X/700/2/L99](https://doi.org/10.1088/0004-637X/700/2/L99).

**Verified.** **ApJ 700 L99 checked against the published PDF** (pp. L99–L103, 2026-06-08): IC setup §3.1, the Λ method *as described in prose* §3.2, the Fig. 2 result, the §3.3 mechanism + Eqs. 1–3. ⚠️ **The MNRAS 395,1449 method paper has NOT been verified** — its formal $\Lambda_{\mathrm{MSR}}$ equation numbering and the exact ONC $\Lambda$ value are *not* verified here.
```

## The big idea

Two questions, one MST toolkit. **(MNRAS)** How do you *quantify* whether the massive stars in a
cluster are more spatially concentrated than average? **(ApJ)** *Why* are so many young clusters
observed mass-segregated when two-body relaxation is far too slow? The answer to the second is that
clusters born **cool (subvirial) and clumpy (fractal)** collapse into a short-lived dense core whose
violent relaxation segregates the most massive stars in $\sim 1$ crossing time — orders of magnitude
faster than the classical $t_{\mathrm{seg}}$.

## The $\Lambda_{\mathrm{MSR}}$ metric (as described in the held ApJ L99 PDF, §3.2)

Compare the minimum spanning tree (MST) of the $N$ most massive stars to the MSTs of many random
$N$-star subsets. The mass segregation ratio is **the ratio of the average random-subset MST length
to the massive-star MST length**:

$$
\Lambda_{\mathrm{MSR}} = \frac{\langle L_{\mathrm{random}}\rangle}{L_{\mathrm{massive}}}
\;\pm\; \frac{\sigma_{\mathrm{random}}}{L_{\mathrm{massive}}},
$$

with the uncertainty the **standard deviation** (L99 calls it the "instantaneous standard deviation")
of the random-subset MST lengths. Interpretation:

- $\Lambda_{\mathrm{MSR}} \approx 1$ — no mass segregation (massive subset is a typical random subset);
- $\Lambda_{\mathrm{MSR}} > 1$ — massive stars more concentrated (segregated); $\gg 1$ ⇒ strong;
- $\Lambda_{\mathrm{MSR}} < 1$ — inverse segregation (rare).

```{warning}
The **formal equation and its number live in the MNRAS method paper (395, 1449), which is not held
here.** Do not cite "ApJ 700 L99 Eq. 1" for $\Lambda_{\mathrm{MSR}}$: **Eq. 1 of L99 is the Spitzer
$t_{\mathrm{seg}}$ relation** (below), not $\Lambda_{\mathrm{MSR}}$, which L99 gives only in prose. The
`compute_lambda_msr` docstring currently mis-cites this — fix to "Allison et al. 2009, MNRAS 395,
1449" (and obtain that PDF before quoting an equation number or the ONC value).
```

## The short-timescale result (held ApJ L99 PDF)

**Initial conditions (§3.1, verified).** $N=1000$ single stars; Kroupa (2002) three-part power-law
MF, $m\in[0.08, 50]\,M_\odot$; **fractal** spatial distribution (dimension $D=1.6$; $D=3.0$ = uniform
sphere) in a sphere of radius **1 pc**; velocities coherent so nearby stars move together (Goodwin &
Whitworth 2004); **virial ratio $Q=0.3$** ($Q=T/|V|$, so $0.5$ = virial). $D=1.6$ + $Q=0.3$ are
chosen as the *most extreme* (fastest-segregating) case. Integrated with `kira`/`starlab`; stellar
evolution neglected over the 4 Myr runs.

**Result (Fig. 2, verified).** $\Lambda_{\mathrm{MSR}}(t)$ for the $N=10, 20, 50, 100$ most massive
stars. Initially $\Lambda=1$ (unsegregated); after $\sim 1$ Myr the **10 most massive reach
$\Lambda\sim 3$**. The 20 and 50 most massive segregate weakly; beyond the 50th, none. Because fractal
ICs are seed-dependent, an **ensemble** is essential: of **50** clusters, **29 segregate within 1 Myr,
44 within 4 Myr, 6 never** (segregation = an event lasting $>0.1$ Myr with significance $>1$).

**Mechanism (§3.3, verified).** Cool + clumpy ⇒ gravitational collapse + violent relaxation ⇒ a
short-lived **dense core** (~half the mass within ~0.1 pc, lasting 0.1–0.2 Myr ≈ 10–20 *core* crossing
times → dynamically old). Classical timescales: Spitzer (1969) $t_{\mathrm{seg}}(M)\approx
(\langle m\rangle/M)\,t_{\mathrm{relax}}$ **(L99 Eq. 1)**; $t_{\mathrm{relax}}\approx [N/(8\ln N)]\,
t_{\mathrm{cross}}$ **(Eq. 2)**; combined **(Eq. 3)**. Core params $N\sim300$–500, $R\sim0.1$–0.2 pc,
$\langle m\rangle=0.4\,M_\odot$, $\sigma\sim2$ km s$^{-1}$ give $t_{\mathrm{seg}}\sim0.1$ Myr ⇒
segregation only **above $M\sim2$–$4\,M_\odot$** (the 50th most massive $\approx2\,M_\odot$).

**Why smooth clusters don't (verified).** Collapse factor $R_0/R_f = (\alpha_0/\alpha_f)\,2(1-Q_0)$.
A smooth Plummer ($\alpha_0\approx\alpha_f\approx0.75$) at $Q_0=0.3$ collapses only $\times1.4$ — too
little. A fractal $D=1.6$ has $\alpha_0\approx1.5$ ⇒ collapses $\times2.5$ (1 pc → ~0.4 pc; core much
smaller), reaching the dense state that segregates. This is the crux: **substructure enables the deep
collapse; subvirial supplies the cold start.**

## Use in progenax

- [](../../10-theory/tidal-and-substructure/mass-segregation.md) — Λ_MSR diagnostic theory.
- `progenax.diagnostics.compute_lambda_msr` — released-core estimator; its formula
  ($\langle L_{\mathrm{rand}}\rangle/L_{\mathrm{massive}}$, std error) **matches the held L99
  description**. *Validation still owed* (see below) and the docstring citation needs the MNRAS fix.
- The cool-fractal pathway is the canonical alternative to {cite:t}`Baumgardt2008` *primordial*
  segregation; the experimental `gravoturb_fdf` cluster-IC forward tool targets exactly this regime
  (turbulent substructure + chosen sub-virial $Q$).

### Validation owed (planned)

- **Tier A (analytic, released-core):** $\Lambda_{\mathrm{MSR}}$ absolute correctness — unsegregated
  random masses $\to 1$; hand-constructed exact value; maximal $\to\gg1$; inverse $\to<1$; estimator
  convergence in $N_{\mathrm{random}}$; the binary-contamination caveat (tight massive binary → spurious
  spike).
- **Tier B (literature):** the **end-to-end** anchor is L99 Fig. 2 — a cool ($Q=0.3$) fractal
  ($D=1.6$) $N=1000$ cluster should evolve $\Lambda(N{=}10)$ from $1$ to a *few* within ~1 Myr,
  segregating only down to $\sim2$–$4\,M_\odot$. (The exact ONC $\Lambda$ and the formal equation need
  the **MNRAS 395,1449 PDF**, not yet held.)

## Notes

- **Two distinct $Q$'s.** Allison's virial ratio $Q=T/|V|$ (0.5 = virial) is **not** the CW04
  structure parameter $\mathcal{Q}=\bar m/\bar s$ — see [](cartwright-2004.md). Both appear in this
  problem (cool *and* clumpy); keep them separate.
- **Softening / collisionality.** L99's segregation is *violent-relaxation* (collapse-driven), not
  slow two-body relaxation — so a softened collisionless integrator captures the dominant effect over
  ~1 crossing time. State this when reproducing.
- **Ensembles are mandatory** for fractal ICs (seed-to-seed scatter is large) — L99 used 50.
