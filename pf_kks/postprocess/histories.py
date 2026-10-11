"""Smoothed evolution curves of a run (Post_Process.ipynb, "Si, eta, Surf").

Matrix Si content, precipitate fraction, interface measure and vacancy site
fraction versus temperature, each sub-sampled and smoothed by a cubic spline
with the factors of the original notebook.
"""

from dataclasses import dataclass

from .dsc import smooth, subsample, taper_supersaturation

# Temperatures of interest marked in the article figures [K].
TEMPERATURES_OF_INTEREST = (507, 541, 610, 700)


@dataclass
class HistoryParameters:
    subsample: int = 75
    s_X_al: float = 2.8e-8
    s_eta: float = 5.6e-8
    s_surf: float = 1.1e-12
    s_vacancy: float = 1.1e-12


def smoothed_histories(history, params=None):
    """Return a dict of smoothed curves sampled at the sub-sampled times.

    Keys: ``T``, ``t``, ``X_al`` (matrix Si fraction), ``eta`` (precipitate
    fraction), ``surf`` and, when the history has ``xv_out``, ``vacancy``.
    """
    p = HistoryParameters() if params is None else params
    T = subsample(history["T_out"], p.subsample)
    t = subsample(history["time_out"], p.subsample)
    eta = subsample(history["eta_mean_out"], p.subsample)
    X_al = taper_supersaturation(subsample(history["X_al_mean_out"], p.subsample), strict=True)
    surf = subsample(history["surf_out"], p.subsample)
    out = {
        "T": T,
        "t": t,
        "X_al": smooth(t, X_al, p.s_X_al)(t),
        "eta": smooth(t, eta, p.s_eta)(t),
        "surf": smooth(t[t > 0], surf[t > 0], p.s_surf)(t),
    }
    if "xv_out" in history:
        xv = subsample(history["xv_out"], p.subsample)
        # The original cell fitted the spline against T but evaluated it at t;
        # it is evaluated at T here.
        out["vacancy"] = smooth(T, xv, p.s_vacancy)(T)
    return out
