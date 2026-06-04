# Design: Michie–King anisotropic model (Batch 2c)

**Date:** 2026-06-03 · **Status:** design (gate before TDD) · **Papers:** Michie (1963)
MNRAS 125, 127; King (1966) AJ 71, 64. Grounded in
`docs/website/99-bibliography/per-paper/{michie-1963,king-1966}.md`.

## Decisions locked (brainstorm)

- **DF = standard Michie–King**: Michie's $e^{-J^2/2r_a^2\sigma^2}$ anisotropy + King's
  $[e^{-E/\sigma^2}-1]$ lowered cutoff (the Gunn & Griffin 1979 / LIMEPY model), **not**
  Michie's literal Fokker–Planck $Q$.
- **API = a new self-consistent pair** `MichieProfile` (spatial) + `MichieVelocityDF`
  (kinematics). Isotropic `KingProfile`/`KingVelocityDF` stay untouched (Michie's density
  differs from King's).
- **Parameter = $r_a$** (anisotropy radius, physical, consistent with Plummer/EFF; maps to
  Michie's $C\propto 1/r_a^2$). Sampler = 2-D marginal-then-conditional inverse-CDF.
- **Validation anchored on the isotropic limit** ($r_a\to\infty$ recovers our validated
  King), $\beta(r)$ increasing outward, virial $Q\approx0.5$. **No external table** (we do
  not have LIMEPY; avoid any fabricated comparison).

## The model

**DF** (dimensionless $u=v/\sigma$, $W(r)=(\Phi_t-\Phi)/\sigma^2$, $s=r/r_a$):

```{math}
f \propto \exp\!\Big(-\tfrac{s^2 u_t^2}{2}\Big)\,
   \big[\exp(W - \tfrac{u^2}{2}) - 1\big],\quad u^2=u_r^2+u_t^2 < 2W,\ u_t\ge 0.
```

$s\to 0$ ($r_a\to\infty$) gives the isotropic King DF.

**Density** $\tilde\rho(W,s)=\int f\,d^3u$ reduces to a **1-D integral over $u_t$**
(the $u_r$ integral is closed-form: error functions for the Maxwellian term minus the
flat bound term). Differentiable (vmap + trapezoid). $\tilde\rho(W,0)$ must equal the
King density.

**Self-consistent ODE** (King's factor-of-9 nondimensionalisation, $\xi=r/r_c$,
$\hat r_a = r_a/r_c$):

```{math}
\frac{d^2W}{d\xi^2} + \frac2\xi\frac{dW}{d\xi}
  = -9\,\frac{\tilde\rho(W,\ \xi/\hat r_a)}{\tilde\rho(W_0,0)},
\qquad W(0)=W_0,\ W'(0)=0,\ \text{integrate to } W(\xi_t)=0.
```

Radius-dependent RHS (via $\xi/\hat r_a$) ⇒ a *different*, more centrally-radial density
than isotropic King. Solved with the same `jax.lax.scan` fixed-step scheme as
`solve_king_profile` (differentiable); $\hat r_a\to\infty$ recovers it exactly.

**Self-consistent $\sigma$** (no external rescale, matching the Batch-1 King pattern):
$\sigma^2 = G M / (9\,r_c\,\mu)$, $\mu=\int_0^{\xi_t}\tilde\rho(W,\xi/\hat r_a)\,\xi^2 d\xi$.

**2-D velocity sampler** (no OM stretch trick — $f$ is not a function of one $Q$):
at each particle's $(r, W, s)$,
1. sample $u_t$ from its marginal $m(u_t)\propto e^{-s^2u_t^2/2}\!\int_{\rm bound}[\dots]du_r$
   (the density's $u_t$-integrand) by inverse-CDF;
2. sample $u_r\,|\,u_t$ from $[\exp(W-\tfrac{u_r^2+u_t^2}{2})-1]$ on $|u_r|<\sqrt{2W-u_t^2}$
   by inverse-CDF;
3. $v_r=\sigma u_r$ along $\hat r$; $v_t=\sigma u_t$ in a random azimuth $\perp\hat r$.

Both inverse-CDFs are per-particle fixed grids (vmap), differentiable, no while-loop.

## Files

- `src/progenax/profiles/michie.py` — `michie_density(W, s)`, `solve_michie_profile(W0,
  ra_hat, ...)`, `MichieProfile` (sample_positions via the mass-profile inverse-CDF,
  characteristic_radius = r_t). Reuses king.py ODE machinery where possible.
- `src/progenax/kinematics/michie_df.py` — `MichieVelocityDF` (the 2-D sampler + self-
  consistent $\sigma$).
- Exports in `progenax.__init__`, `profiles/__init__`, `kinematics/__init__`.
- Function/file-length budget respected (split the density integral + ODE + sampler).

## Validation (TDD, RED→GREEN; never weaken)

1. **Isotropic limit** — `solve_michie_profile(W0, ra_hat=inf-proxy)` matches
   `solve_king_profile(W0)` (W grid, ξ_t) to tolerance; `michie_density(W,0)` == King
   density.
2. **ODE sanity** — W monotonic, W(0)=W₀, W(ξ_t)=0; finite-r_a is more concentrated
   (ξ_t/r_c shifts the right way).
3. **Anisotropy** — realised $\beta(r)$ is ~0 at the centre and **increases outward**
   (the headline anisotropy check); large $r_a$ ⇒ $\beta\approx0$ everywhere.
4. **Virial** — $Q=T/|V|\approx0.5$ unscaled (self-consistent $\sigma$).
5. **Concentration** — $c(W_0)$ at large $r_a$ matches King 1966 Table II (already
   validated); finite $r_a$ shifts $c$ consistently.
6. **Differentiability** — FD grad-checks wrt $W_0$, $r_c$, $r_a$; JIT; byte-stable RNG.

## Risks / notes

- The 2-D conditional inverse-CDF per particle is the most novel/expensive piece; keep
  the grids modest (≈128–256) and validate the marginal+conditional reproduce
  `michie_density` (a self-consistency unit test).
- $\beta(r)$ for Michie–King is **not** the OM $r^2/(r^2+r_a^2)$ — it is the DF-implied
  profile; tests assert monotonic-increasing + isotropic-centre, not a closed form.
- Out of scope: Michie's literal $Q$ cutoff; multimass; rotation.
