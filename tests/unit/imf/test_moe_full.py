"""Faithful two-slope, period-dependent Moe & Di Stefano (2017) q-distribution (Batch 4i-a).

p_q(q | M1, P) = (1 - F_twin) * [two-slope power law, continuous at q=0.3]
                 + F_twin * Uniform[0.95, 1.0]
with gamma_smallq, gamma_largeq, F_twin bilinearly interpolated over Table 13
(verified against the PDF p.52). RED tests pin: Table-13 cell recovery, the pdf
shape (slopes + twin + normalization + q=0.3 continuity), the sampler, and grads.
"""

import jax
import jax.numpy as jnp


def _full():
    from progenax.imf.binary import MoeDiStefano2017Full

    return MoeDiStefano2017Full()


# Table-13 grid nodes: mass bins Solar/A/Mid-B/Early-B/O at representative masses,
# log P = 1,3,5,7. Verified values (P in days).
LOGP = jnp.array([1.0, 3.0, 5.0, 7.0])
MASS = jnp.array([1.0, 3.2, 6.7, 12.0, 20.0])


class TestTable13Interpolation:
    def test_gamma_largeq_cells(self):
        moe = _full()
        # (logP, M, expected gamma_largeq) from the verified grid
        cases = [
            (1.0, 1.0, -0.5),
            (3.0, 20.0, -1.7),
            (5.0, 3.2, -1.4),
            (7.0, 1.0, -1.1),
        ]
        P = 10.0 ** jnp.array([c[0] for c in cases])
        M = jnp.array([c[1] for c in cases])
        exp = jnp.array([c[2] for c in cases])
        assert jnp.allclose(moe.gamma_largeq(P, M), exp, atol=1e-6)

    def test_gamma_smallq_cells(self):
        moe = _full()
        cases = [(1.0, 1.0, 0.3), (5.0, 6.7, -1.2), (7.0, 3.2, -1.0), (1.0, 20.0, 0.1)]
        P = 10.0 ** jnp.array([c[0] for c in cases])
        M = jnp.array([c[1] for c in cases])
        exp = jnp.array([c[2] for c in cases])
        assert jnp.allclose(moe.gamma_smallq(P, M), exp, atol=1e-6)

    def test_ftwin_cells(self):
        moe = _full()
        # solar logP=1 -> 0.30 ; O logP=1 -> 0.08 ; midB logP=3 -> 0 (<0.03); solar logP=5 -> 0.10
        cases = [(1.0, 1.0, 0.30), (1.0, 20.0, 0.08), (3.0, 6.7, 0.0), (5.0, 1.0, 0.10)]
        P = 10.0 ** jnp.array([c[0] for c in cases])
        M = jnp.array([c[1] for c in cases])
        exp = jnp.array([c[2] for c in cases])
        assert jnp.allclose(moe.f_twin(P, M), exp, atol=1e-6)

    def test_clamped_outside_grid(self):
        moe = _full()
        # below logP=1 and above logP=7 clamp to edge rows; below 1 Msun clamps to solar
        g_lo = moe.gamma_largeq(
            jnp.array([10.0**0.2]), jnp.array([0.5])
        )  # logP<1, M<1 -> solar logP=1
        assert jnp.allclose(g_lo, -0.5, atol=1e-6)


class TestMoeFullPdf:
    def test_normalized(self):
        moe = _full()
        q = jnp.linspace(0.1, 1.0, 200000)
        # solar, logP=3 (Ftwin=0.20, gamma_largeq=-0.5, gamma_smallq=0.3)
        p = moe.pdf(q, jnp.array(1.0), jnp.array(1e3))
        assert jnp.abs(jnp.trapezoid(p, q) - 1.0) < 1e-2

    def test_largeq_slope_ratio(self):
        """In q in (0.3,0.95) the pdf is the power-law q^gamma_largeq."""
        moe = _full()
        m1, P = (
            jnp.array(20.0),
            jnp.array(1e7),
        )  # O-type logP=7 -> gamma_largeq=-2.0, Ftwin=0
        p = moe.pdf(jnp.array([0.5, 0.6]), m1, P)
        ratio = p[1] / p[0]
        assert jnp.abs(ratio - (0.6 / 0.5) ** (-2.0)) < 1e-4

    def test_continuous_at_break(self):
        moe = _full()
        m1, P = jnp.array(3.2), jnp.array(1e5)
        p = moe.pdf(jnp.array([0.3 - 1e-6, 0.3 + 1e-6]), m1, P)
        assert jnp.abs(p[1] - p[0]) / p[0] < 1e-3  # continuous at q=0.3

    def test_twin_excess_present(self):
        moe = _full()
        m1, P = jnp.array(1.0), jnp.array(10.0)  # solar logP=1 -> Ftwin=0.30
        # density just above 0.95 should exceed the power-law extrapolation by the twin term
        p_twin = moe.pdf(jnp.array([0.97]), m1, P)[0]
        p_below = moe.pdf(jnp.array([0.90]), m1, P)[0]
        assert p_twin > p_below  # twin pile-up


class TestMoeFullSampling:
    def test_range_and_twin_fraction(self):
        moe = _full()
        key = jax.random.PRNGKey(0)
        n = 200000
        m1 = jnp.full(n, 1.0)
        P = jnp.full(n, 10.0)  # solar logP=1, Ftwin=0.30
        q = moe.sample(key, m1, P)
        assert jnp.all((q >= 0.1) & (q <= 1.0))
        # P(q>0.95) is the twin block PLUS the power-law content in [0.95,1],
        # NOT F_twin itself (F_twin is the EXCESS over q>0.3 — audit R3). Derive
        # the expectation by quadrature of the fixed pdf rather than a magic
        # constant; the old 0.28 gate encoded the pre-fix +22% twin overweight.
        qs = jnp.linspace(0.1, 1.0, 200_001)
        p = jax.vmap(lambda qq: moe.pdf(qq, jnp.asarray(1.0), jnp.asarray(10.0)))(qs)
        expected = float(
            jnp.trapezoid(jnp.where(qs >= 0.95, p, 0.0), qs) / jnp.trapezoid(p, qs)
        )
        assert abs(float(jnp.mean(q > 0.95)) - expected) < 0.01

    def test_sampled_mean_matches_pdf(self):
        moe = _full()
        key = jax.random.PRNGKey(1)
        n = 300000
        m1 = jnp.full(n, 6.7)
        P = jnp.full(n, 1e3)  # mid-B logP=3
        q = moe.sample(key, m1, P)
        # analytic mean via the pdf
        qq = jnp.linspace(0.1, 1.0, 400000)
        p = moe.pdf(qq, jnp.array(6.7), jnp.array(1e3))
        mean_pdf = jnp.trapezoid(qq * p, qq)
        assert jnp.abs(jnp.mean(q) - mean_pdf) < 0.01

    def test_grad_finite(self):
        from progenax.imf.binary import MoeDiStefano2017Full

        key = jax.random.PRNGKey(2)
        m1 = jnp.full(2000, 5.0)
        P = jnp.full(2000, 1e2)

        def loss(qmin):
            return jnp.mean(MoeDiStefano2017Full(q_min=qmin).sample(key, m1, P))

        g = jax.grad(loss)(0.1)
        assert jnp.isfinite(g)

    def test_grad_fd_accurate(self):
        """The grid-based inverse-CDF is properly reparameterized: autodiff matches FD
        (a multi-uniform segment/twin sampler would lose the mixture-weight gradient)."""
        from progenax.imf.binary import MoeDiStefano2017Full

        key = jax.random.PRNGKey(7)
        m1 = jnp.full(8000, 6.7)
        P = jnp.full(8000, 1e3)

        def loss(qmin):
            return jnp.mean(MoeDiStefano2017Full(q_min=qmin).sample(key, m1, P))

        ad = jax.grad(loss)(0.1)
        h = 1e-5
        fd = (loss(0.1 + h) - loss(0.1 - h)) / (2 * h)
        assert jnp.abs(ad - fd) / (jnp.abs(ad) + 1e-12) < 1e-3, f"ad={ad} fd={fd}"


class TestFTwinPaperConvention:
    """MD17 p.5 / Fig.2: F_twin = excess-twin fraction of q > 0.3 companions.

    Audit finding R3: the pre-fix mixture realized F_twin = 0.367 instead of
    0.300 at (M1=1, logP=1). These tests pin the paper convention at four
    Table-13 nodes via quadrature of the pdf's mixture components.
    """

    NODES = [(1.0, 1.0), (1.0, 3.0), (3.2, 1.0), (12.0, 3.0)]  # (M1 [Msun], logP)

    def test_f_twin_excess_over_q_gt_03(self):
        md = _full()
        qs = jnp.linspace(md.q_min, 1.0, 200_001)
        for m1, logP in self.NODES:
            mass = jnp.asarray(m1)
            period = jnp.asarray(10.0**logP)
            ft_table = float(md.f_twin(period, mass))
            p_pl, p_twin = md._components(qs, mass, period)
            mask = qs >= 0.3
            twin_mass = float(jnp.trapezoid(jnp.where(mask, p_twin, 0.0), qs))
            total_gt03 = float(jnp.trapezoid(jnp.where(mask, p_pl + p_twin, 0.0), qs))
            realized = twin_mass / total_gt03
            assert abs(realized - ft_table) < 2e-3, (
                f"(M1={m1}, logP={logP}): realized paper-convention F_twin "
                f"{realized:.4f} != Table 13 value {ft_table:.4f}"
            )

    def test_pdf_still_normalized(self):
        md = _full()
        qs = jnp.linspace(md.q_min, 1.0, 200_001)
        for m1, logP in self.NODES:
            p = jax.vmap(lambda q: md.pdf(q, jnp.asarray(m1), jnp.asarray(10.0**logP)))(
                qs
            )
            Z = float(jnp.trapezoid(p, qs))
            assert abs(Z - 1.0) < 1e-3

    def test_sample_matches_pdf_twin_fraction(self):
        """Sampled P(q >= 0.95) must match the quadrature of the FIXED pdf."""
        md = _full()
        n = 200_000
        key = jax.random.PRNGKey(0)
        m1 = jnp.full((n,), 1.0)
        periods = jnp.full((n,), 10.0)
        q = md.sample(key, m1, periods)
        qs = jnp.linspace(md.q_min, 1.0, 200_001)
        p = jax.vmap(lambda qq: md.pdf(qq, jnp.asarray(1.0), jnp.asarray(10.0)))(qs)
        expected = float(
            jnp.trapezoid(jnp.where(qs >= 0.95, p, 0.0), qs) / jnp.trapezoid(p, qs)
        )
        observed = float(jnp.mean(q >= 0.95))
        assert abs(observed - expected) < 0.01  # shot noise ~0.001 at n=2e5


class TestMoePeriod:
    def test_range(self):
        from progenax.imf.binary import MoePeriod

        P = MoePeriod().sample(jax.random.PRNGKey(0), jnp.full(50000, 5.0))
        lp = jnp.log10(P)
        assert jnp.all((lp >= 0.2 - 1e-6) & (lp <= 8.0 + 1e-6))

    def test_mass_dependence(self):
        """Solar companion frequency peaks at long logP (~5); O-type is flatter/shorter
        => median logP of O-type binaries < median logP of solar-type."""
        from progenax.imf.binary import MoePeriod

        mp = MoePeriod()
        P_solar = mp.sample(jax.random.PRNGKey(1), jnp.full(100000, 1.0))
        P_O = mp.sample(jax.random.PRNGKey(1), jnp.full(100000, 20.0))
        assert jnp.median(jnp.log10(P_O)) < jnp.median(jnp.log10(P_solar))

    def test_grad_finite(self):
        from progenax.imf.binary import MoePeriod

        g = jax.grad(
            lambda m: jnp.mean(
                jnp.log10(MoePeriod().sample(jax.random.PRNGKey(2), jnp.full(3000, m)))
            )
        )(5.0)
        assert jnp.isfinite(g)


class TestMoeJointOrbit:
    def test_shapes_and_ranges(self):
        from progenax.imf.binary import MoeJointOrbit

        jo = MoeJointOrbit.default()
        n = 20000
        m1 = jnp.full(n, 5.0)
        P, q, e = jo.sample(jax.random.PRNGKey(0), m1)
        assert P.shape == (n,) and q.shape == (n,) and e.shape == (n,)
        assert jnp.all((q >= 0.1) & (q <= 1.0))
        assert jnp.all((e >= 0.0) & (e < 1.0))
        lp = jnp.log10(P)
        assert jnp.all((lp >= 0.2 - 1e-6) & (lp <= 8.0 + 1e-6))

    def test_p_q_interrelation(self):
        """The paper's central result: short-P binaries favour larger q (twins/equal),
        long-P favour smaller q. Bin a massive-star population by logP and check <q> falls."""
        from progenax.imf.binary import MoeJointOrbit

        jo = MoeJointOrbit.default()
        n = 400000
        m1 = jnp.full(n, 12.0)  # early-B: γlargeq steepens strongly with logP
        P, q, e = jo.sample(jax.random.PRNGKey(3), m1)
        lp = jnp.log10(P)
        q_short = jnp.mean(q[lp < 2.0])
        q_long = jnp.mean(q[lp > 5.0])
        assert q_short > q_long, (
            f"<q>_shortP={q_short:.3f} should exceed <q>_longP={q_long:.3f}"
        )

    def test_jit(self):
        from progenax.imf.binary import MoeJointOrbit

        jo = MoeJointOrbit.default()
        m1 = jnp.full(1000, 5.0)
        P, q, e = jax.jit(jo.sample)(jax.random.PRNGKey(4), m1)
        assert (
            jnp.all(jnp.isfinite(P))
            and jnp.all(jnp.isfinite(q))
            and jnp.all(jnp.isfinite(e))
        )
