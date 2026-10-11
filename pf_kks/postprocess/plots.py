"""Matplotlib figures: field maps, DSC curve and evolution curves."""

import numpy as np

from .histories import TEMPERATURES_OF_INTEREST


def plot_field(ax, field, dx, key, title=None):
    """Image of a 2-D field (``eta`` or ``X``) with the colour limits of the
    original ``plot_micro``: [0, 1] for eta, [0, 0.025] for X."""
    nx, ny = field.shape[0], field.shape[1]
    vmax = 1 if key == "eta" else 0.025
    extent_nm = [0, nx * dx * 1e9, ny * dx * 1e9, 0]
    im = ax.imshow(field.reshape(nx, ny).transpose(), extent=extent_nm, vmin=0, vmax=vmax)
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("y [nm]")
    if title:
        ax.set_title(title)
    return im


def plot_dsc(ax, dsc, experimental=None):
    """Simulated DSC with its contributions, and the measured curve."""
    s = dsc.scale
    ax.plot(dsc.T, dsc.background * s, linestyle="dashdot", label="Heat capacity term")
    ax.plot(dsc.T, (dsc.background + dsc.precipitation) * s, linestyle="dashdot", label="+ precipitation and diffusion")
    ax.plot(dsc.T, (dsc.background + dsc.surface) * s, linestyle="dashdot", label="+ surface energy")
    ax.plot(dsc.T, dsc.total, color="black", label="Phase field (total)")
    if experimental is not None:
        ax.plot(experimental[0], experimental[1], color="red", label="Experimental")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel(r"$C_p^* \sim[J.g^{-1}.K^{-1}]$")
    ax.legend()


def mark_temperatures(ax, temperatures=TEMPERATURES_OF_INTEREST):
    for i, T in enumerate(temperatures, start=1):
        ax.axvline(x=T, color="b", linestyle="dashed", linewidth=0.8)
        ax.text(T, 0.95, f" TOI{i}", transform=ax.get_xaxis_transform(), va="top", fontsize=8)


def plot_histories(axes, curves):
    """Matrix Si content, precipitate fraction, interface measure and (if
    present) vacancy site fraction versus temperature."""
    T = curves["T"]
    panels = [
        ("X_al", 100, r"$\langle X_{Si}^{\alpha}\rangle ~\left[mol. \% \right] $"),
        ("eta", 1, r"$\langle \eta \rangle$"),
        ("surf", 1, r"$Specific~surface~\left[m^2. g^{-1}\right]$"),
        ("vacancy", 1, "Vacancy site fraction [-]"),
    ]
    for ax, (key, factor, label) in zip(axes, [p for p in panels if p[0] in curves]):
        ax.plot(T, np.asarray(curves[key]) * factor, color="k")
        mark_temperatures(ax)
        ax.set_xlabel("T [K]")
        ax.set_ylabel(label)
