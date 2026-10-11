"""Simulated DSC (apparent heat capacity) from the phase-field histories.

The heat flow during heating is split into three contributions:

* heat-capacity background of the two-phase mixture,
* precipitation / matrix desaturation (enthalpy change with the precipitate
  fraction ``<eta>`` and the matrix Si content ``<X_Si^alpha>``),
* interface (surface-energy) release, from the interface measure ``surf``.

The histories are first sub-sampled and smoothed with cubic splines; their
derivatives give the rates. This module reproduces ``DSC_tool/compute_DSC.ipynb``
(checked by ``tests/test_dsc_equivalence.py``). Parameters default to the
values of that notebook.
"""

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import UnivariateSpline

from ..thermo import gibbs_energy, molar_volume

M_AL = 26.98154  # molar mass of Al [g mol^-1]
M_SI = 28.0855  # molar mass of Si [g mol^-1]


@dataclass
class DSCParameters:
    subsample: int = 90  # keep one history point out of ``subsample``
    s_eta: float = 5.6e-6  # spline smoothing factor of <eta>(t)
    s_X_al: float = 5.8e-6  # spline smoothing factor of <X_Si^alpha>(t)
    s_surf: float = 5.8e-13  # spline smoothing factor of surf(t)
    x_supersaturated: float = 0.025  # initial matrix Si fraction
    strict_supersaturation: bool = False  # ">" (Post_Process.ipynb) instead of ">=" (compute_DSC.ipynb)
    heating_rate: float = 20 / 60  # T_dot [K s^-1]
    interface_energy: float = 0.5  # gamma_H [J m^-2]
    simulated_area: float = (710e-9) ** 2 / 4  # V_sim [m^2]
    output_scale: float = 3  # multiplies the heat flow to give Cp* (= 1 / T_dot)


@dataclass
class DSCResult:
    T: np.ndarray  # temperature of the sub-sampled points [K]
    t: np.ndarray  # time [s]
    background: np.ndarray  # heat-capacity term
    precipitation: np.ndarray  # precipitation / desaturation term
    surface: np.ndarray  # interface-energy term
    scale: float

    @property
    def total(self):
        """Apparent heat capacity Cp* [J g^-1 K^-1]."""
        return (self.background + self.surface + self.precipitation) * self.scale


def subsample(series, step, start=0, trim=1):
    """``series[start:-1][0:-trim:step]`` as in the original notebooks (the last
    point is dropped)."""
    return series[start:-1][0:-trim:step]


def taper_supersaturation(X_al, x0=0.025, strict=False):
    """Replace matrix compositions at or above ``x0`` by a slowly decreasing
    ramp just above ``x0`` (helps the spline fit at the start of heating).
    Returns a copy."""
    X_al = np.array(X_al, dtype=float, copy=True)
    mask = X_al > x0 if strict else X_al >= x0
    n = len(X_al[mask])
    X_al[mask] = x0 + np.arange(1, 0.001, -1 / (n)) * 1e-4
    return X_al


def smooth(t, y, s):
    """Cubic smoothing spline of ``y(t)``."""
    return UnivariateSpline(x=t, y=y, w=None, bbox=[None, None], k=3, s=s, ext=0, check_finite=False)


def compute_dsc(history, params=None):
    """Simulated DSC signal.

    Parameters
    ----------
    history : mapping
        Map or history file contents with ``T_out``, ``time_out``,
        ``eta_mean_out``, ``X_al_mean_out`` and ``surf_out``.
    params : DSCParameters, optional

    Returns
    -------
    DSCResult
    """
    p = DSCParameters() if params is None else params
    T = subsample(history["T_out"], p.subsample)
    t = subsample(history["time_out"], p.subsample)
    eta = subsample(history["eta_mean_out"], p.subsample)
    X_al = subsample(history["X_al_mean_out"], p.subsample)
    surf = subsample(history["surf_out"], p.subsample)

    eta_s = smooth(t, eta, p.s_eta)
    X_al = taper_supersaturation(X_al, p.x_supersaturated, p.strict_supersaturation)
    X_al_s = smooth(t, X_al, p.s_X_al)
    surf_s = smooth(t[t > 0], surf[t > 0], p.s_surf)

    deta_dt = eta_s.derivative(n=1)(t)
    dX_al_dt = X_al_s.derivative(n=1)(t)
    dsurf_dt = surf_s.derivative(n=1)(t)

    n = len(T)
    cp_fcc, cp_dia, Hm_dia, Hm_fcc = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    Vm_dia, Vm_fcc, dHm_fcc_dSi = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n):
        x = X_al_s(t[i])
        cp_fcc[i] = gibbs_energy(T[i], x, "cp_fcc")
        Hm_fcc[i] = gibbs_energy(T[i], x, "Hm_fcc")
        cp_dia[i] = gibbs_energy(T[i], 0.9999, "cp_dia")
        Hm_dia[i] = gibbs_energy(T[i], 0.9999, "Hm_dia")
        Vm_dia[i] = molar_volume(298, T[i], 0.9999, "Vmdia")
        Vm_fcc[i] = molar_volume(298, T[i], x, "Vmfcc")
        dHm_fcc_dSi[i] = gibbs_energy(T[i], x, "d_Hm_fcc_dSi")

    f = eta_s(t)
    rho = f * M_SI / Vm_dia + (1 - f) * M_AL / Vm_fcc  # [g m^-3]
    background = p.heating_rate * (f * cp_dia * (1 / M_SI) + (1 - f) * cp_fcc * (1 / M_AL))
    surface = p.interface_energy * dsurf_dt / (p.simulated_area * rho)
    precipitation = (
        (1 - f) * dHm_fcc_dSi * (1 / M_AL) * dX_al_dt
        + deta_dt * (Hm_dia * (1 / M_SI))
        + (-deta_dt) * (Hm_fcc * (1 / M_AL))
    )
    return DSCResult(T, t, background, precipitation, surface, p.output_scale)


def load_experimental(path):
    """Measured DSC curve (``data/experimental/heat_capacity_dsc.csv``):
    temperature [K] and Cp* [J g^-1 K^-1] (the file stores J kg^-1 K^-1).

    The original notebook skipped the first row (``skiprows=1``) although the
    file has no header; all rows are read here.
    """
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0], data[:, 1] / 1000
