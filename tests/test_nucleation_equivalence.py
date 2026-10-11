"""pf_kks.nucleation reproduces ``nucle`` of the original top-level pyFi_adapt.py."""

import numpy as np

from pf_kks.nucleation import nucleation_increment

from .legacy import LEGACY_THERMO, LEGACY_TOPLEVEL, TABLES, load_module


def test_nucleation_increment_matches_legacy():
    thermo = load_module(LEGACY_THERMO, "legacy_free_nrj_mob")
    legacy = load_module(LEGACY_TOPLEVEL, "legacy_pyfi_adapt_toplevel")
    # In the original workflow these came from the notebook namespace.
    legacy.elas_const = thermo.elas_const
    legacy.molar_volume = thermo.molar_volume
    d = np.load(TABLES)
    for T, X_al_mean, grid in [(450.0, 0.02, (64, 64, 1)), (520.0, 0.03, (32, 1, 1)), (600.0, 0.018, (16, 16, 16))]:
        coeffs = [np.interp(T, d["T"], d[k]) for k in ("A_fcc", "B_fcc", "C_fcc", "A_dia", "B_dia", "C_dia")]
        sigma = np.interp(T, d["T"], d["gamma"])
        args = (1e-3, 0.005, 12.0, 0.5, T, 0.5e-9, 0.5e-9, 0.5e-9, *grid, 1e-18, *coeffs, sigma, X_al_mean, 1e-5)
        assert nucleation_increment(*args) == legacy.nucle(*args)
