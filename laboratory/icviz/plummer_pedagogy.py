"""Analytic Plummer teaching figures; no particle sampling or fitted curves."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from jaxstro.registry import load_source
from jaxstro.registry.symbolic import build_equation
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

from progenax.registry import registry_root

from .style import PALETTE, polish_axes, setup_style


def build_density_to_radius() -> plt.Figure:
    """Registered density and CDF in dimensionless radius x = r/a."""
    setup_style(font_scale=1.1)
    source = load_source(registry_root(), "progenaxderivations")
    density = build_equation(source, "EQ-PLUMMER-DENSITY")
    cdf = build_equation(source, "EQ-PLUMMER-CDF")
    r, a, mass = (density.symbols[n] for n in ("r", "a", "M"))
    rho_expr = density.expression.subs({a: 1, mass: 1})
    rho = sp.lambdify(r, rho_expr, "numpy")
    cumulative = sp.lambdify(
        cdf.symbols["r"], cdf.expression.subs(cdf.symbols["a"], 1), "numpy"
    )
    x = np.linspace(0, 5, 700)
    x_h = 1 / np.sqrt(2 ** (2 / 3) - 1)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
    curves = [rho(x) / rho(0), 4 * np.pi * x**2 * rho(x), cumulative(x)]
    titles = ["Volume density", "Radial probability", "Cumulative mass"]
    labels = [r"$\rho(r)/\rho(0)$", r"$p_x(x)=a\,p_r(r)$", r"$F(r)=M(<r)/M$"]
    for ax, y, title, label in zip(axes, curves, titles, labels):
        ax.plot(x, y, color=PALETTE[0], lw=2)
        ax.axvline(1, color=PALETTE[1], ls=":", label=r"$a$")
        ax.axvline(x_h, color=PALETTE[2], ls="--", label=r"$r_h$")
        ax.set(
            xlim=(0, 5),
            ylim=(0, 1.04 if title != titles[1] else 0.65),
            xlabel=r"$r/a$",
            ylabel=label,
            title=title,
        )
        polish_axes(ax)
    axes[0].legend(frameon=False, loc="upper right")
    axes[2].annotate(
        "",
        xy=(x_h, 0.5),
        xytext=(0, 0.5),
        arrowprops={"arrowstyle": "->", "color": PALETTE[1]},
    )
    axes[2].plot(x_h, 0.5, "o", color=PALETTE[2], ms=5)
    axes[2].text(2.0, 0.42, r"$u=1/2\ \longrightarrow\ r_h$", fontsize=10)
    return fig


def build_velocity_components() -> plt.Figure:
    """Local isotropic basis and normalized conditional speed probability."""
    setup_style(font_scale=1.1)
    fig, (basis, speed) = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)
    basis.set(
        xlim=(-1.4, 1.7),
        ylim=(-1.3, 1.6),
        aspect="equal",
        title="Local velocity components",
    )
    basis.axis("off")
    basis.add_patch(Circle((0, 0), 0.06, color=PALETTE[0]))
    # Projection of an orthogonal spherical basis; arrow lengths are schematic.
    for endpoint, label in [
        ((1.0, 0.6), r"$v_r$"),
        ((-0.9, 0.6), r"$v_\theta$"),
        ((0, -0.8), r"$v_\phi$"),
    ]:
        basis.annotate(
            "",
            xy=endpoint,
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": PALETTE[0], "lw": 2},
        )
        basis.text(
            endpoint[0] * 1.2,
            endpoint[1] * 1.2,
            label,
            ha="center",
            va="center",
            fontsize=13,
        )
    basis.text(
        0.1,
        1.23,
        r"$\sigma_r^2=\sigma_\theta^2=\sigma_\phi^2$",
        ha="center",
        fontsize=12,
    )
    basis.text(0.1, -1.2, r"$\langle v^2\rangle=3\sigma_r^2$", ha="center", fontsize=12)
    q = np.linspace(0, 1, 600)
    # z=q^2 ~ Beta(3/2,9/2); the Jacobian dz/dq=2q gives p(q).
    alpha, beta = sp.Rational(3, 2), sp.Rational(9, 2)
    normalization = float(sp.gamma(alpha) * sp.gamma(beta) / sp.gamma(alpha + beta))
    probability = 2 * q**2 * (1 - q**2) ** 3.5 / normalization
    speed.plot(q, probability, color=PALETTE[0], lw=2)
    speed.set(
        xlim=(0, 1),
        ylim=(0, 2.5),
        xlabel=r"$q=v/v_{\mathrm{esc}}$",
        ylabel=r"$p(q)$",
        title="Conditional speed distribution",
    )
    speed.text(
        0.97,
        0.94,
        r"$q^2\sim\mathrm{Beta}(3/2,9/2)$",
        transform=speed.transAxes,
        ha="right",
        va="top",
        fontsize=11,
    )
    polish_axes(speed)
    return fig


def build_projected_density() -> plt.Figure:
    """Line-of-sight integral of Plummer density, in units of Sigma(0).

    Integrating 3M/(4*pi*a**3) * (1 + (R**2+z**2)/a**2)**(-5/2)
    over z gives M/(pi*a**2) * (1 + R**2/a**2)**(-2).
    The field of view and logarithmic normalization are display choices.
    """
    setup_style(font_scale=1.6)
    coordinate = np.linspace(-5, 5, 601)
    x, y = np.meshgrid(coordinate, coordinate)
    surface_density = (1 + x**2 + y**2) ** -2
    half_mass = 1 / np.sqrt(2 ** (2 / 3) - 1)
    fig, (ax, radial) = plt.subplots(
        1,
        2,
        figsize=(10.5, 4.6),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.1, 1]},
    )
    image = ax.imshow(
        surface_density,
        extent=(-5, 5, -5, 5),
        origin="lower",
        cmap="magma",
        norm=LogNorm(vmin=1 / 51**2, vmax=1),
        interpolation="nearest",
        aspect="equal",
    )
    for radius, linestyle in [(1, "--"), (half_mass, ":")]:
        ax.add_patch(
            Circle((0, 0), radius, fill=False, edgecolor="white", lw=1.5, ls=linestyle)
        )
    ax.annotate(
        r"$a$",
        xy=(-1, 0),
        xytext=(-3, 1.6),
        color="white",
        arrowprops={"arrowstyle": "-", "color": "white"},
        fontsize=15,
    )
    ax.annotate(
        r"$r_h$",
        xy=(half_mass, 0),
        xytext=(2.7, -1.6),
        color="white",
        arrowprops={"arrowstyle": "-", "color": "white"},
        fontsize=15,
    )
    ax.set(xlabel=r"$x/a$", ylabel=r"$y/a$", title="Projected density")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02, shrink=0.8)
    colorbar.set_label(r"$\Sigma/\Sigma(0)$")
    ax.grid(False)
    radius = np.linspace(0, 5, 601)
    radial.semilogy(radius, (1 + radius**2) ** -2, color=PALETTE[0], lw=2)
    radial.axvline(
        1, color=PALETTE[1], ls="--", lw=1.5, label=r"$a$: Plummer scale radius"
    )
    radial.axvline(
        half_mass, color=PALETTE[2], ls=":", lw=1.8, label=r"$r_h$: 3D half-mass radius"
    )
    radial.set(
        xlim=(0, 5),
        ylim=(1e-3, 1.2),
        xlabel=r"$R/a$",
        ylabel=r"$\Sigma(R)/\Sigma(0)$",
        title="Radial surface-density profile",
    )
    radial.legend(frameon=False, loc="upper right", fontsize=12)
    polish_axes(radial)
    return fig
