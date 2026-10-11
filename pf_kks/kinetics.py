"""Temperature-dependent kinetic coefficients: excess vacancies, Si
diffusivity, interface mobility and the Allen-Cahn kinetic coefficient."""

import numpy as np

from .constants import D0_AL, D0_SI_IN_AL, HB, HF1VA, Q_AL, Q_SI_IN_AL, SB_KB, SF1VA, Z_COORD, R


def equilibrium_vacancy_fraction(T):
    """Equilibrium vacancy site fraction at temperature ``T`` [K], including
    vacancy-solute binding."""
    return np.exp(-(HF1VA - T * SF1VA) / (R * T)) * (
        1 - Z_COORD * 0.01 + Z_COORD * 0.01 * np.exp((HB - T * SB_KB) / (R * T))
    )


def quenched_vacancy_fraction(T_quench):
    """Vacancy site fraction frozen in from ``T_quench`` (850 K in the study)."""
    return np.exp(-(HF1VA - T_quench * SF1VA) / (R * T_quench)) * (
        1 - Z_COORD * 0.01 + Z_COORD * 0.01 * np.exp((HB - T_quench * SB_KB) / (R * T_quench))
    )


def si_diffusivity(T, Xv, Xve, D0, QD, cap):
    """Si diffusivity enhanced by the vacancy supersaturation ``Xv / Xve``.

    The enhancement factor ``|1 + (Xv - Xve) / Xve|`` is capped at ``cap`` and
    ignored when below 1.
    """
    factor = np.abs(1 + (Xv - Xve) / Xve)
    if factor > cap:
        D = cap * D0 * np.exp(-QD / (R * T))
    else:
        D = factor * D0 * np.exp(-QD / (R * T))
    if factor < 1:
        D = D0 * np.exp(-QD / (R * T))
    return D


def effective_vacancy_diffusivity(T, Xv, Xve):
    """Diffusivity controlling vacancy annihilation at sinks."""
    D_Al = D0_AL * np.exp(-Q_AL / (R * T))
    D_Si = D0_SI_IN_AL * np.exp(-Q_SI_IN_AL / (R * T))
    return np.abs(1 + (Xv - Xve) / Xve) * (0.03 * D_Si + 0.97 * D_Al)


def interface_mobility(T, Meta0, QMeta):
    """Interface mobility ``Meta0 exp(-QMeta / RT)``."""
    return Meta0 * np.exp(-QMeta / (R * T))


def vacancy_relaxation(Xv, Xve, dtime, Deff, Lv2):
    """Relax ``Xv`` towards ``Xve`` over ``dtime`` with sink mean free path
    squared ``Lv2``."""
    dXv = (1 - np.exp(-dtime * (Deff / Lv2))) * (Xve - Xv)
    return Xv + dXv


def gradient_and_barrier(sigma, lambd, alpha):
    """Gradient-energy coefficient ``kappa^2`` and double-well height ``w``.

    ``lambd`` is the half interface thickness and ``alpha`` the interface-width
    constant (2.2 in the study).
    """
    kappa2 = 2 * lambd * sigma * 3 / alpha
    w = (sigma * 3 * alpha) / lambd
    return kappa2, w


def allen_cahn_coefficient(sigma, kappa2, w, Meta, D, A_alp, X_Si_fcc, Omega0):
    """Allen-Cahn kinetic coefficient L with the thin-interface correction.

    ``xi`` is the chemical contribution of the interface (Kim-Kim-Suzuki);
    the final factor 0.01 is the scaling used for the published runs.
    """
    kappa = np.sqrt(kappa2)
    xi = (19 / 30) * (1 / Omega0) * 2 * A_alp * (X_Si_fcc - 0.9999) ** 2
    return (sigma / kappa**2) * (1 / (1 / Meta + kappa * xi / (D * np.sqrt(2 * w)))) * 0.01


def normalize(
    D, D_0, Meta, Meta_0, C44, dx, sigma, alpha, A_thet, B_thet, C_thet, A_alp, B_alp, C_alp, Omega0, L, lambd
):
    """Dimensionless parameters (lengths in dx, energies in C44, time in
    dx^2 / D_0). Original name: ``normalize_parameters``.

    Returns
    -------
    dict
        ``L_s, kappa2_s, D_s, Meta_s, w_s, A_alp_s, B_alp_s, C_alp_s,
        A_thet_s, B_thet_s, C_thet_s``.
    """
    kappa2 = 2 * lambd * sigma * 3 / alpha
    w = (sigma * 3 * alpha) / lambd
    return {
        "L_s": (L * C44 * dx**2) / D_0,
        "kappa2_s": kappa2 / (C44 * dx**2),
        "D_s": D / D_0,
        "Meta_s": Meta / Meta_0,
        "w_s": w / (C44),
        "A_thet_s": A_thet / (C44 * Omega0),
        "B_thet_s": B_thet / (C44 * Omega0),
        "C_thet_s": C_thet / (C44 * Omega0),
        "A_alp_s": A_alp / (C44 * Omega0),
        "B_alp_s": B_alp / (C44 * Omega0),
        "C_alp_s": C_alp / (C44 * Omega0),
    }
