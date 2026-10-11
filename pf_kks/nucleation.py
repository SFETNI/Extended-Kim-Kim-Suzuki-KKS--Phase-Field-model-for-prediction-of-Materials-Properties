"""Classical nucleation of Si precipitates (optional; not used in the published
DSC runs).

Carried over from the top-level ``pyFi_adapt.py`` of the original repository,
where ``nucle`` relied on ``elas_const`` and ``molar_volume`` being imported in
the notebook namespace; they are now imported from :mod:`pf_kks.thermo`.
"""

import numpy as np

from .constants import K_B, N_A
from .thermo import elas_const, molar_volume


def nucleation_increment(
    B,
    X_al_eq,
    t,
    dt,
    T,
    dx,
    dy,
    dz,
    nx,
    ny,
    nz,
    D,
    A_alp,
    B_alp,
    C_alp,
    A_thet,
    B_thet,
    C_thet,
    sigma,
    X_al_mean,
    Xv,
):
    """Number of nuclei formed during ``dt`` and the critical radius.

    Classical nucleation theory with an elastic correction for an incoherent
    spherical precipitate and a heterogeneous-nucleation factor ``B``.
    Original name: ``nucle``.

    Returns
    -------
    n_nuclei : float
        Expected number of nuclei formed in the simulation volume during ``dt``.
    r_critical : float
        Critical radius [m]. Both are 0 once the matrix is below ``X_al_eq``.
    """
    if X_al_mean > 0.025:
        X_al_mean = 0.025

    C11_Si = elas_const(T, "C11_Si")
    C12_Si = elas_const(T, "C12_Si")
    C11_Al = elas_const(T, "C11")
    C12_Al = elas_const(T, "C12")
    K_si = (C11_Si + 2 * C12_Si) / 3
    mu_al = (C11_Al - C12_Al) / 2
    C6 = (3 * K_si) / (3 * K_si + 4 * mu_al)

    Vm_fcc = molar_volume(298, T, X_al_mean, "Vmfcc")
    V_fcc = Vm_fcc / N_A
    a_fcc = (4 * V_fcc) ** (1 / 3)  # FCC lattice parameter (4 atoms per cell) [m]
    Vm_dia = molar_volume(298, T, 0.9999, "Vmdia")
    V_dia = Vm_dia / N_A
    Vm = 0.1 * Vm_dia + 0.9 * Vm_fcc

    if (ny == 1) and (nz == 1):
        Vsim = ((nx - 1) * dx) ** (3)
    elif nz == 1:
        Vsim = ((nx - 1) * dx * (ny - 1) * dy) ** (3 / 2)
    else:
        Vsim = (nx - 1) * dx * (ny - 1) * dy * (nz - 1) * dz

    f_al_p = 2 * A_alp * (X_al_mean) + B_alp * (X_al_mean)
    f_th = A_thet * 0.99999**2 + B_thet * 0.99999 + C_thet
    f_al = A_alp * X_al_mean**2 + B_alp * X_al_mean + C_alp
    delt_gV = (1 / N_A) * (f_th - (f_al + f_al_p * (0.99999 - X_al_mean)))

    A_fac = (36 * np.pi * V_dia**2) ** (1 / 3.0)
    eps = (1 / 3) * ((Vm_dia - Vm_fcc) ** 2 / (Vm_dia)) / N_A
    delgs = 2 * mu_al * C6 * eps
    deltaG_homo = (4 / 27) * ((A_fac * sigma) ** 3 / ((delt_gV + delgs) ** 2))
    deltaG_hete = B * deltaG_homo

    n_s = (-2 * A_fac * sigma / (3 * (delt_gV + delgs))) ** 3
    r_s = (1 / (2 * np.sqrt(np.pi))) * (36 * np.pi) ** (1 / 6) * V_dia ** (1 / 3) * n_s ** (1 / 3)
    zeldovich = (3 * (delt_gV + delgs) ** 2) / (4 * np.sqrt(np.pi * K_B * T) * (sigma * A_fac) ** (3 / 2.0))
    beta = (4 * np.pi * r_s**2 * X_al_mean * D) / (a_fcc**4)
    tau = 1 / (2 * beta * zeldovich**2)
    J = zeldovich * Xv * beta * np.exp(-deltaG_hete / (K_B * T)) * np.exp(-tau / t) * (N_A / Vm) * Vsim

    n_nuclei = J * dt
    if X_al_mean < X_al_eq:
        n_nuclei = 0
        r_s = 0
    return n_nuclei, r_s


def insert_random_nucleus(candidates, eta, rng=None):
    """Set eta = 1 at one random grid point among ``candidates`` (flat indices).

    Original name: ``put_nucle`` (only eta was modified there as well).
    """
    rng = np.random if rng is None else rng
    shape = eta.shape
    eta1d = eta.reshape(-1)
    pick = rng.randint(0, len(candidates) - 1, size=1)
    eta1d[candidates[pick]] = 1
    return eta1d.reshape(shape)
