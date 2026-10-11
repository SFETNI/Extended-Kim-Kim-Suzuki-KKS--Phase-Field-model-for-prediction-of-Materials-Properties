"""Simulation maps (``.npz`` snapshots) and their file names.

A map stores the full fields (``X``, ``X_al``, ``X_th``, ``eta``, shape
``(nx, ny, 1)``) and the per-step histories up to the save step
(``time_out``, ``T_out``, ``X_al_mean_out``, ``X_th_mean_out``,
``eta_mean_out``, ``surf_out``) plus ``ttime``. Maps can seed a new run but
are not exact restart points (the vacancy state and the adaptive step are not
stored). File names follow ``map{ttime:.2f}s_{T_start}_{T_max:.2f}K.npz``.
"""

import glob
import os
import re

import numpy as np

MAP_PATTERN = re.compile(r"map(?P<ttime>[0-9.]+)s_(?P<T_start>[0-9.]+)_(?P<T_max>[0-9.]+)K\.npz$")


def map_filename(ttime, T_start, T_max):
    """Map file name without extension, as written by the original code."""
    return "map" + "{0:.2f}".format(ttime) + "s_" + str(T_start) + "_" + "{0:.2f}".format(T_max) + "K"


def save_map(path, ttime, hist, X, X_al, X_th, eta):
    """Write a map. ``hist`` holds the solver histories."""
    np.savez_compressed(
        path,
        ttime=ttime,
        time_out=hist["time"],
        T_out=hist["T"],
        X_al_mean_out=hist["X_al_mean"],
        X_th_mean_out=hist["X_th_mean"],
        eta_mean_out=hist["eta_mean"],
        surf_out=hist["surf"],
        X=X,
        X_al=X_al,
        X_th=X_th,
        eta=eta,
    )


def parse_map_filename(path):
    """Return ``(ttime, T_start, T_max)`` parsed from a map file name."""
    m = MAP_PATTERN.search(os.path.basename(path))
    if m is None:
        raise ValueError(f"not a map file name: {path}")
    return float(m["ttime"]), float(m["T_start"]), float(m["T_max"])


def list_maps(directory):
    """Map files of ``directory`` sorted by maximum temperature, then time."""
    files = [f for f in glob.glob(os.path.join(directory, "map*.npz")) if MAP_PATTERN.search(os.path.basename(f))]
    return sorted(files, key=lambda f: (parse_map_filename(f)[2], parse_map_filename(f)[0]))


def load_map(path, fields=True):
    """Load a map as a dict. With ``fields=False`` only the histories and
    ``ttime`` are read (fast for large grids)."""
    with np.load(path) as d:
        keys = d.files if fields else [k for k in d.files if k not in ("X", "X_al", "X_th", "eta")]
        return {k: d[k] for k in keys}
