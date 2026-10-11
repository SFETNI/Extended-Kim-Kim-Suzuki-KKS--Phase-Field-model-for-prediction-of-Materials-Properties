"""Al-Si thermodynamic and physical property functions (CALPHAD).

These functions are carried over from the original ``DSC_tool/free_nrj_mob.py``.
The formulas, coefficients and evaluation order are unchanged (checked by
``tests/test_thermo_equivalence.py``); only indentation, docstrings and the
output selection were rewritten. Requesting an unknown ``output`` now raises
``ValueError`` instead of silently returning 0.

They are used by the nucleation model (:mod:`pf_kks.nucleation`) and by the DSC
post-processing (:mod:`pf_kks.postprocess.dsc`). The phase-field solver itself
reads pre-tabulated parameters from ``data/inputs/final_export.npz``.
"""

import numpy as np

__all__ = [
    "atomic_mobility",
    "gibbs_energy",
    "molar_volume",
    "elas_const",
    "ATOMIC_MOBILITY_OUTPUTS",
    "GIBBS_ENERGY_OUTPUTS",
    "MOLAR_VOLUME_OUTPUTS",
    "ELASTIC_CONSTANT_OUTPUTS",
]


def _select(outputs, output, function):
    try:
        return outputs[output]
    except KeyError:
        raise ValueError(f"{function}: unknown output {output!r}; expected one of {sorted(outputs)}") from None


ATOMIC_MOBILITY_OUTPUTS = (
    "M_Al",
    "M_Si",
    "LIQ_M_Al",
    "FCC_D_Al_s",
)

GIBBS_ENERGY_OUTPUTS = (
    "Gm_fcc",
    "Hm_fcc",
    "d_Hm_fcc_dSi",
    "d_Hm_liq_dSi",
    "dGm_fcc_dSi",
    "d2Gm_fcc_dSi2",
    "dGm_fcc_dAl",
    "d2Gm_fcc_dAl2",
    "cp_fcc",
    "cp_liq",
    "cp_dia",
    "Gm_dia",
    "Hm_dia",
    "dmu_liq_Si_dSi",
    "dmu_liq_Al_dSi",
    "dGm_dia_dSi",
    "d2Gm_fcc_dSiAl",
    "mu_fcc_Si",
    "mu_liq_Si",
    "mu_liq_Al",
    "dmu_fcc_Si_dSi",
    "dmu_fcc_Al_dSi",
    "dmu_dia_Si_dSi",
    "dmu_dia_Al_dSi",
    "mu_fcc_Al",
    "mu_dia_Si",
    "mu_dia_Al",
    "Hm_liq",
    "Gm_liq",
    "dGm_liq_dSi",
)

MOLAR_VOLUME_OUTPUTS = (
    "Vmfcc",
    "Vmdia",
    "Vmliq",
    "CLE_fccAl",
)

ELASTIC_CONSTANT_OUTPUTS = (
    "C11",
    "C12",
    "C44",
    "C11_Si",
    "C44_Si",
    "C12_Si",
    "B",
    "E",
)


def atomic_mobility(T, X_Si, output):
    """Atomic mobilities of Al and Si in the FCC (and liquid) Al-Si phases.

    Data: Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58-66.

    Parameters
    ----------
    T : float
        Temperature [K].
    X_Si : float
        Si molar fraction [-].
    output : str
        One of ``ATOMIC_MOBILITY_OUTPUTS``.
    """
    # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58–66.
    R = 8.314472  # = [J.K-1.mol-1]
    # mobility in the liquid
    liq_Phi_Al_Al = -24830.3 - 130.62 * T
    liq_Phi_Al_Si = -41794.2 - 121.96 * T
    liq_0Phi_Al_AlSi = 17050.5
    X_Al = 1 - X_Si
    # for Al
    liq_Phi_Al = X_Al * liq_Phi_Al_Al + X_Si * liq_Phi_Al_Si + X_Al * X_Si * liq_0Phi_Al_AlSi
    liq_M_Al = np.exp(liq_Phi_Al / (R * T)) * (1 / (R * T))
    # mobility in FCC
    # for Al
    Phi_Al_Al = -123111.6 - 97.34 * T
    Phi_Al_Si = -155634.921 - 81.0257 * T
    _0Phi_Al_AlSi = 1846710.24

    Phi_Al = X_Al * Phi_Al_Al + X_Si * Phi_Al_Si + X_Si * X_Al * _0Phi_Al_AlSi  # *(chi_Al_grid-chi_Si_grid)

    M_Al = np.exp(Phi_Al / (R * T)) * (1 / (R * T))
    D_Al_s = R * T * M_Al

    # for Si
    Phi_Si_Al = -117600.0 - 93.05 * T
    Phi_Si_Si = -155634.9 - 81.03 * T
    _0Phi_Si_AlSi = -152613.88

    Phi_Si = X_Si * Phi_Si_Si + X_Al * Phi_Si_Al + X_Si * X_Al * _0Phi_Si_AlSi  # *(chi_Al_grid-chi_Si_grid)
    M_Si = np.exp(Phi_Si / (R * T)) * (1 / (R * T))
    # output

    outputs = {
        "M_Al": M_Al,
        "M_Si": M_Si,
        "LIQ_M_Al": liq_M_Al,
        "FCC_D_Al_s": D_Al_s,
    }
    return _select(outputs, output, "atomic_mobility")


def gibbs_energy(T, X_Si, output):
    """CALPHAD molar Gibbs energy of the Al-Si FCC, diamond and liquid phases,
    with the derived enthalpies, heat capacities and chemical potentials.

    Data: COST 507 (Ansara, Dinsdale, Rand, 1998); SGTE unary data for the
    pure-element reference states.

    Parameters
    ----------
    T : float
        Temperature [K]. Scalar only (the SGTE piecewise ranges use ``if``).
    X_Si : float
        Si molar fraction [-], strictly inside (0, 1).
    output : str
        One of ``GIBBS_ENERGY_OUTPUTS``. Energies are in J/mol, heat
        capacities in J/(mol K).
    """
    # data from : Y. Tang, L. Zhang, Y. Du, Calphad 49 (2015) 58–66.   X_Al = 1-X_Si
    R = 8.314472  # = [J.K-1.mol-1]

    # so as to X_SI different from 0 or 1

    # if X_Si>=1:
    #   X_Si = 0.00000001
    # elif X_Si<=1:
    #   X_Si = 9.99999999e-1

    X_Al = 1 - X_Si

    _0G_dia_Si = 0
    d_0G_dia_Si_dT = 0
    d2_0G_dia_Si_dT2 = 0
    _0G_fcc_Al = 0
    d_0G_fcc_Al_dT = 0
    d2_0G_fcc_Al_dT2 = 0
    _0G_liq_Si = 0
    d_0G_liq_Si_dT = 0
    d2_0G_liq_Si_dT2 = 0
    _0G_liq_Al = 0
    d_0G_liq_Al_dT = 0
    d2_0G_liq_Al_dT2 = 0

    #
    # free energy for G0_FCC-A1
    # for AL
    if T > 298.15 and T <= 700:
        _0G_fcc_Al = (
            -7976.15
            + 137.093038 * T
            - 24.3671976 * T * np.log(T)
            - 1.884662e-3 * T**2
            - 0.877664e-6 * T**3
            + 74092 * T ** (-1)
        )
        d_0G_fcc_Al_dT = (
            137.093038
            - 24.3671976 * (1 + np.log(T))
            - (2) * 1.884662e-3 * T
            - (3) * 0.877664e-6 * T**2
            + (-1) * 74092 * T ** (-2)
        )
        d2_0G_fcc_Al_dT2 = (
            -24.3671976 * (1 / T) - (2) * 1.884662e-3 - (3 * 2) * 0.877664e-6 * T + (-1) * (-2) * 74092 * T ** (-3)
        )
    elif T > 700 and T <= 933.47:
        _0G_fcc_Al = (
            -11276.24
            + 223.048446 * T
            - 38.5844296 * T * np.log(T)
            + 18.531982e-3 * T**2
            - 5.764227e-6 * T**3
            + 74092 * T ** (-1)
        )
        d_0G_fcc_Al_dT = (
            223.048446
            - 38.5844296 * (1 + np.log(T))
            + (2) * 18.531982e-3 * T
            - (3) * 5.764227e-6 * T**2
            + (-1) * 74092 * T ** (-2)
        )
        d2_0G_fcc_Al_dT2 = (
            -38.5844296 * (1 / T) + (2) * 18.531982e-3 - (3) * (2) * 5.764227e-6 * T + (-1) * (-2) * 74092 * T ** (-3)
        )
    elif T > 933.47 and T < 2900.0:
        _0G_fcc_Al = -11278.378 + 188.6841536 * T - 31.748192 * T * np.log(T) - 1230.524e25 * T ** (-9)
        d_0G_fcc_Al_dT = 188.6841536 - 31.748192 * (1 + np.log(T)) - (-9) * 1230.524e25 * T ** (-10)
        d2_0G_fcc_Al_dT2 = -31.748192 * (1 / T) - (-9) * (-10) * 1230.524e25 * T ** (-11)
    #
    # free energy for G0_FCC-A1
    # for Si
    if T > 298.15 and T <= 1687:
        _0G_dia_Si = (
            -8162.609
            + 137.236859 * T
            - 22.8317533 * T * np.log(T)
            - 1.912904e-3 * T**2
            - 0.003552e-6 * T**3
            + 176667 * T ** (-1)
        )
        d_0G_dia_Si_dT = (
            137.236859
            - 22.8317533 * (1 + np.log(T))
            - (2) * 1.912904e-3 * T
            - (3) * 0.003552e-6 * T**2
            + (-1) * 176667 * T ** (-2)
        )
        d2_0G_dia_Si_dT2 = (
            -22.8317533 * (1 / T) - (2) * 1.912904e-3 - (3 * 2) * 0.003552e-6 * T + (-1) * (-2) * 176667 * T ** (-3)
        )
    elif T > 1687 and T < 3600:
        _0G_dia_Si = -9457.642 + 167.281367 * T - 27.196 * T * np.log(T) - 420.369e28 * T ** (-9)
        d_0G_dia_Si_dT = 167.281367 - 27.196 * (1 + np.log(T)) - (-9) * 420.369e28 * T ** (-10)
        d2_0G_dia_Si_dT2 = -27.196 * (1 / T) - (-9) * (-10) * 420.369e28 * T ** (-11)

    _0G_fcc_Si = 51000.00 - 21.8 * T + _0G_dia_Si
    d_0G_fcc_Si_dT = -21.8 + d_0G_dia_Si_dT
    d2_0G_fcc_Si_dT2 = d2_0G_dia_Si_dT2
    _0G_dia_Al = 30 * T + _0G_fcc_Al
    d_0G_dia_Al_dT = 30 + d_0G_fcc_Al_dT
    d2_0G_dia_Al_dT2 = d2_0G_fcc_Al_dT2

    # free energy for liquid
    # for Si

    if T > 298.15 and T <= 1687:
        _0G_liq_Si = 50696.4 - 30.0994 * T + 2.09307e-21 * T**7 + _0G_dia_Si
        d_0G_liq_Si_dT = -30.0994 + 7 * 2.09307e-21 * T**6 + d_0G_dia_Si_dT
        d2_0G_liq_Si_dT2 = 6 * 7 * 2.09307e-21 * T**5 + d2_0G_dia_Si_dT2
    elif T > 1687 and T < 6000:
        _0G_liq_Si = 49828.2 - 29.5591 * T + 4.20369e30 * T ** (-9) + _0G_dia_Si
        d_0G_liq_Si_dT = -29.5591 + (-9) * 4.20369e30 * T ** (-10) + d_0G_dia_Si_dT
        d2_0G_liq_Si_dT2 = (-10) * 4.20369e30 * T ** (-11) + d2_0G_dia_Si_dT2

    # for Al:

    if T > 298.15 and T <= 933.47:
        _0G_liq_Al = 11005.029 - 11.84187 * T + 7.934e-20 * T**7 + _0G_fcc_Al
        d_0G_liq_Al_dT = -11.84187 + (7) * 7.934e-20 * T**6 + d_0G_fcc_Al_dT
        d2_0G_liq_Al_dT2 = (6 * 7) * 7.934e-20 * T**5 + d2_0G_fcc_Al_dT2
    elif T > 933.47 and T < 2900:
        _0G_liq_Al = 10482.382 - 11.253974 * T + 1.231e28 * T ** (-9) + _0G_fcc_Al
        d_0G_liq_Al_dT = -11.253974 + (-9) * 1.231e28 * T ** (-10) + d_0G_fcc_Al_dT
        d2_0G_liq_Al_dT2 = (-10) * 1.231e28 * T ** (-11) + d2_0G_fcc_Al_dT2

    # fcc
    L0_fcc_AlSi = -3143.78 + 0.3929 * T
    dL0_fcc_AlSi_dT = 0.3929
    d2L0_fcc_AlSi_dT2 = 0

    # dia
    L0_dia_AlSi = 113246.16 - 47.5551 * T
    dL0_dia_AlSi_dT = -47.5551
    d2L0_dia_AlSi_dT2 = 0

    # liq
    L0_liq_AlSi = -11340.1 - 1.23394 * T
    dL0_liq_AlSi_dT = -1.23394
    d2L0_liq_AlSi_dT2 = 0

    L1_liq_AlSi = -3530.93 + 1.35993 * T
    dL1_liq_AlSi_dT = 1.35993
    d2L1_liq_AlSi_dT2 = 0

    L2_liq_AlSi = 2265.39
    dL2_liq_AlSi_dT = 0
    d2L2_liq_AlSi_dT2 = 0

    # free energy for the liquid phase

    Gm_liq = (
        X_Al * _0G_liq_Al
        + X_Si * _0G_liq_Si
        + R * T * (X_Al * np.log(X_Al) + X_Si * np.log(X_Si))
        + X_Al * X_Si * (L0_liq_AlSi + L1_liq_AlSi * (X_Al - X_Si) + L2_liq_AlSi * (X_Al - X_Si) ** 2)
    )

    dGm_liq_dT = (
        X_Al * d_0G_liq_Al_dT
        + X_Si * d_0G_liq_Si_dT
        + R * (X_Al * np.log(X_Al) + X_Si * np.log(X_Si))
        + X_Al * X_Si * (dL0_liq_AlSi_dT + dL1_liq_AlSi_dT * (X_Al - X_Si) + dL2_liq_AlSi_dT * (X_Al - X_Si) ** 2)
    )

    d2Gm_liq_dT2 = (
        X_Al * d2_0G_liq_Al_dT2
        + X_Si * d2_0G_liq_Si_dT2
        + X_Al * X_Si * (d2L0_liq_AlSi_dT2 + d2L1_liq_AlSi_dT2 * (X_Al - X_Si) + d2L2_liq_AlSi_dT2 * (X_Al - X_Si) ** 2)
    )

    Hm_liq = Gm_liq - T * dGm_liq_dT

    dGm_liq_dSi = (
        (-1) * _0G_liq_Al
        + _0G_liq_Si
        + R * T * (-1 - np.log(1 - X_Si) + 1 + np.log(X_Si))
        + (-2 * X_Si + 1) * L0_liq_AlSi
        + (6 * X_Si**2 - 6 * X_Si + 1) * L1_liq_AlSi
        + (-16 * X_Si**3 + 24 * X_Si**2 - 10 * X_Si + 1) * L2_liq_AlSi
    )  # = to check because not good  old (-6*X_Si**5+5*X_Si**4+8*X_Si**3-6*X_Si**2-2*X_Si+1)

    d2Gm_liq_dSi2 = (
        R * T * (-1 + 1 / (1 - X_Si) + 1 + 1 / (X_Si))
        + (-2) * L0_liq_AlSi
        + (12 * X_Si - 6) * L1_liq_AlSi
        + (-48 * X_Si**2 + 48 * X_Si - 10) * L2_liq_AlSi
    )

    d2Gm_liq_dTSi = (
        (-1) * d_0G_liq_Al_dT
        + d_0G_liq_Si_dT
        + R * (-1 - np.log(1 - X_Si) + 1 + np.log(X_Si))
        + (-2 * X_Si + 1) * dL0_liq_AlSi_dT
        + (6 * X_Si**2 - 6 * X_Si + 1) * dL1_liq_AlSi_dT
        + (-6 * X_Si**5 + 5 * X_Si**4 + 8 * X_Si**3 - 6 * X_Si**2 - 2 * X_Si + 1) * dL2_liq_AlSi_dT
    )

    d_Hm_liq_dSi = dGm_liq_dSi - T * d2Gm_liq_dTSi

    #
    # free energy for fcc phase
    #
    Gm_fcc = (
        X_Al * _0G_fcc_Al
        + X_Si * _0G_fcc_Si
        + R * T * (X_Al * np.log(X_Al) + X_Si * np.log(X_Si))
        + X_Al * X_Si * L0_fcc_AlSi
    )
    # derivative with respect to T
    dGm_fcc_dT = (
        X_Al * d_0G_fcc_Al_dT
        + X_Si * d_0G_fcc_Si_dT
        + R * (X_Al * np.log(X_Al) + X_Si * np.log(X_Si))
        + X_Al * X_Si * dL0_fcc_AlSi_dT
    )

    d2Gm_fcc_dT2 = X_Al * d2_0G_fcc_Al_dT2 + X_Si * d2_0G_fcc_Si_dT2 + X_Al * X_Si * d2L0_fcc_AlSi_dT2

    d2Gm_fcc_dTSi = (
        (-1) * d_0G_fcc_Al_dT
        + d_0G_fcc_Si_dT
        + R * (-1 - np.log(1 - X_Si) + 1 + np.log(X_Si))
        + (-2 * X_Si + 1) * dL0_fcc_AlSi_dT
    )

    dGm_fcc_dSi = (
        (-1) * _0G_fcc_Al
        + _0G_fcc_Si
        + R * T * (-1 - np.log(1 - X_Si) + 1 + np.log(X_Si))
        + (-2 * X_Si + 1) * L0_fcc_AlSi
    )

    d2Gm_fcc_dSi2 = R * T * (-1 + 1 / (1 - X_Si) + 1 + 1 / (X_Si)) + (-2) * L0_fcc_AlSi  # good
    dGm_fcc_dAl = -dGm_fcc_dSi
    d2Gm_fcc_dAl2 = d2Gm_fcc_dSi2
    d2Gm_fcc_dSiAl = -d2Gm_fcc_dSi2
    d2Gm_fcc_dAlSi = -d2Gm_fcc_dSi2
    # molar enthalpy of fcc

    Hm_fcc = Gm_fcc - T * dGm_fcc_dT
    d_Hm_fcc_dSi = dGm_fcc_dSi - T * d2Gm_fcc_dTSi

    #
    # free energy for diamond phase
    Gm_dia = (
        X_Al * _0G_dia_Al
        + X_Si * _0G_dia_Si
        + R * T * (X_Al * np.log(X_Al) + X_Si * np.log(X_Si))
        + X_Al * X_Si * L0_dia_AlSi
    )  # ok
    # derivative with respect to T
    dGm_dia_dT = (
        X_Al * d_0G_dia_Al_dT
        + X_Si * d_0G_dia_Si_dT
        + R * (X_Al * np.log(X_Al) + X_Si * np.log(X_Si))
        + X_Al * X_Si * dL0_dia_AlSi_dT
    )  # ok
    d2Gm_dia_dT2 = X_Al * d2_0G_dia_Al_dT2 + X_Si * d2_0G_dia_Si_dT2 + X_Al * X_Si * d2L0_dia_AlSi_dT2  # ok

    # its 1st and 2nd derivative with respect to Si
    dGm_dia_dSi = (
        (-1) * _0G_dia_Al
        + _0G_dia_Si
        + R * T * (-1 - np.log(1 - X_Si) + 1 + np.log(X_Si))
        + (-2 * X_Si + 1) * L0_dia_AlSi
    )  # ok

    # Heat capacity

    cp_liq = -T * d2Gm_liq_dT2
    cp_dia = -T * d2Gm_dia_dT2
    cp_fcc = -T * d2Gm_fcc_dT2

    # molar enthalpy of diam

    Hm_dia = Gm_dia - T * dGm_dia_dT

    d2Gm_dia_dSi2 = R * T * (-1 + 1 / (1 - X_Si) + 1 + 1 / (X_Si)) + (-2) * L0_dia_AlSi
    dGm_dia_dAl = -dGm_dia_dSi
    d2Gm_dia_dAlSi = -d2Gm_dia_dSi2

    dGm_liq_dAl = -dGm_liq_dSi
    d2Gm_liq_dAlSi = -d2Gm_liq_dSi2
    # dGm_dia_dAl = _0G_dia_Al+(-1)*_0G_dia_Si+R*T*(1+np.log(X_Al)-1-np.log(1-X_Al))+\
    #  +(-2*X_Al+1)*L0_dia_AlSi
    # d2Gm_dia_dAl2 =  R*T*(1/X_Al+1/(1-X_Al))+\
    #  +(-2*X_Al+1)*L0_dia_AlSi
    #
    #

    # chemical potential
    # for liquid
    mu_liq_Si = dGm_liq_dSi * (1 - X_Si) + Gm_liq
    mu_liq_Al = dGm_liq_dAl * (1 - X_Al) + Gm_liq

    dmu_liq_Si_dSi = d2Gm_liq_dSi2 * (1 - X_Si)
    dmu_liq_Al_dSi = d2Gm_liq_dAlSi * (1 - X_Al) + dGm_liq_dAl * 1 + dGm_liq_dSi
    # for dia
    mu_dia_Si = dGm_dia_dSi * (1 - X_Si) + Gm_dia  #  ok
    mu_dia_Al = dGm_dia_dAl * (1 - X_Al) + Gm_dia  # ok

    dmu_dia_Si_dSi = d2Gm_dia_dSi2 * (1 - X_Si) + dGm_dia_dSi * (-1) + dGm_dia_dSi
    dmu_dia_Al_dSi = d2Gm_dia_dAlSi * (1 - X_Al) + dGm_dia_dAl * 1 + dGm_dia_dSi

    # chemical potential
    mu_fcc_Si = dGm_fcc_dSi * (1 - X_Si) + Gm_fcc  # ok
    dmu_fcc_Si_dSi = d2Gm_fcc_dSi2 * (1 - X_Si) + dGm_fcc_dSi * (-1) + dGm_fcc_dSi  # ok
    # dmu_fcc_SidAl =  -dmu_fcc_SidSi

    mu_fcc_Al = dGm_fcc_dAl * (1 - X_Al) + Gm_fcc
    #
    # dmu_fcc_AldSi = d2Gm_fcc_dSiAl+dGm_fcc_dAl
    # dmu_fcc_AldAl = d2Gm_fcc_dAl2-(dGm_fcc_dAl+X_Al*d2Gm_fcc_dAl2)+dGm_fcc_dAl
    dmu_fcc_Al_dSi = d2Gm_fcc_dAlSi * (1 - X_Al) + dGm_fcc_dAl * 1 + dGm_fcc_dSi  # -dmu_fcc_Al_dAl# ok

    #
    # derivatives of chemical potential
    # dmu_fcc_Si_dSi =  d2Gm_fcc_dSi2-(dGm_fcc_dSi+X_Si*d2Gm_fcc_dSi2)+dGm_fcc_dSi # ok new activation
    # derivatives of chemical potential
    dmu_dia_Si_dSi = d2Gm_dia_dSi2 * (1 - X_Si)
    # dmu_dia_SidSi =  dGm_dia_dSi-(dGm_dia_dSi+X_Si*d2Gm_dia_dSi2)+dGm_dia_dSi

    outputs = {
        "Gm_fcc": Gm_fcc,
        "Hm_fcc": Hm_fcc,
        "d_Hm_fcc_dSi": d_Hm_fcc_dSi,
        "d_Hm_liq_dSi": d_Hm_liq_dSi,
        "dGm_fcc_dSi": dGm_fcc_dSi,
        "d2Gm_fcc_dSi2": d2Gm_fcc_dSi2,
        "dGm_fcc_dAl": dGm_fcc_dAl,
        "d2Gm_fcc_dAl2": d2Gm_fcc_dAl2,
        "cp_fcc": cp_fcc,
        "cp_liq": cp_liq,
        "cp_dia": cp_dia,
        "Gm_dia": Gm_dia,
        "Hm_dia": Hm_dia,
        "dmu_liq_Si_dSi": dmu_liq_Si_dSi,
        "dmu_liq_Al_dSi": dmu_liq_Al_dSi,
        "dGm_dia_dSi": dGm_dia_dSi,
        "d2Gm_fcc_dSiAl": d2Gm_fcc_dSiAl,
        "mu_fcc_Si": mu_fcc_Si,
        "mu_liq_Si": mu_liq_Si,
        "mu_liq_Al": mu_liq_Al,
        "dmu_fcc_Si_dSi": dmu_fcc_Si_dSi,
        "dmu_fcc_Al_dSi": dmu_fcc_Al_dSi,
        "dmu_dia_Si_dSi": dmu_dia_Si_dSi,
        "dmu_dia_Al_dSi": dmu_dia_Al_dSi,
        "mu_fcc_Al": mu_fcc_Al,
        "mu_dia_Si": mu_dia_Si,
        "mu_dia_Al": mu_dia_Al,
        "Hm_liq": Hm_liq,
        "Gm_liq": Gm_liq,
        "dGm_liq_dSi": dGm_liq_dSi,
    }
    return _select(outputs, output, "gibbs_energy")


def molar_volume(T0, T, X_Si, output):
    """Molar volume of the FCC, diamond and liquid phases from integrated
    thermal-expansion fits, referred to ``T0``.

    Parameters
    ----------
    T0 : float
        Reference temperature [K] (298 K throughout the study).
    T : float
        Temperature [K].
    X_Si : float
        Si molar fraction of the FCC / liquid phase [-].
    output : str
        One of ``MOLAR_VOLUME_OUTPUTS``; volumes in m^3/mol.
    """
    X_Al = 1 - X_Si

    V0_fccAl = 9.99987596e-6  # 9.77430e-6
    a_fccAl = 2.17303609e-05  # 6.91213e-5
    b_fccAl = 2.63550737e-09  # 0
    c_fccAl = 1.47000588e-11  # 4.86802e-11
    d_fccAl = -9.03030712e-02  # 0.413484
    # 0V_fccAl =                      # interaction parameters
    #
    V0_fccSi = 9.20000852e-6
    a_fccSi = 4.50736462e-06  # 1.12466e-5
    b_fccSi = 2.00632966e-09  # 8.92959e-9
    c_fccSi = -8.07642813e-13  # -3.09709e-12
    d_fccSi = -1.50848370e-01  # -0.251151
    #
    V0_diaSi = 1.20588085e-05  # 12.0588e-6
    a_diaSi = 3.43321444e-06  # 8.56511e-6
    b_diaSi = 1.54351184e-09  # 6.85005e-9
    c_diaSi = -6.13087337e-13  # -2.35401e-12
    d_diaSi = -1.14950173e-01  # -0.191371
    # al liquid
    V0_liqAl = 1.03371674e-05  # 12.0588e-6
    a_liqAl = 4.66770167e-05  # 8.56511e-6
    b_liqAl = -6.07205185e-09  # 6.85005e-9
    c_liqAl = 5.29104281e-13  # -2.35401e-12
    d_liqAl = 3.51948259e-02  # -0.191371
    # si liquid
    V0_liqSi = 9.05196044e-06  # 12.0588e-6
    a_liqSi = 5.77061979e-05  # 8.56511e-6
    b_liqSi = -9.03455430e-09  # 6.85005e-9
    c_liqSi = 8.94822667e-13  # -2.35401e-12
    d_liqSi = 7.31878142e-02  # -0.191371

    # fcc al
    alpha_fccAl = (1 / 3.0) * (a_fccAl + b_fccAl * T + c_fccAl * T**2 + d_fccAl * T ** (-2))
    CLE_fccAl = alpha_fccAl * (T - T0)
    s_alpha_dT_fccAl = (a_fccAl * T + (b_fccAl / 2) * T**2 + (c_fccAl / 3) * T**3 + d_fccAl * (-(T ** (-1)))) - (
        a_fccAl * T0 + (b_fccAl / 2) * T0**2 + (c_fccAl / 3) * T0**3 + d_fccAl * (-(T0 ** (-1)))
    )
    Vm_fccAl = V0_fccAl * np.exp(3 * s_alpha_dT_fccAl)

    # liq al

    s_alpha_dT_liqAl = (a_liqAl * T + (b_liqAl / 2) * T**2 + (c_liqAl / 3) * T**3 + d_liqAl * (-(T ** (-1)))) - (
        a_liqAl * T0 + (b_liqAl / 2) * T0**2 + (c_liqAl / 3) * T0**3 + d_liqAl * (-(T0 ** (-1)))
    )
    Vm_liqAl = V0_liqAl * np.exp(3 * s_alpha_dT_liqAl)

    # fcc si
    s_alpha_dT_fccSi = (a_fccSi * T + (b_fccSi / 2) * T**2 + (c_fccSi / 3) * T**3 + d_fccSi * (-(T ** (-1)))) - (
        a_fccSi * T0 + (b_fccSi / 2) * T0**2 + (c_fccSi / 3) * T0**3 + d_fccSi * (-(T0 ** (-1)))
    )
    Vm_fccSi = V0_fccSi * np.exp(3 * s_alpha_dT_fccSi)

    # liq si

    s_alpha_dT_liqSi = (a_liqSi * T + (b_liqSi / 2) * T**2 + (c_liqSi / 3) * T**3 + d_liqSi * (-(T ** (-1)))) - (
        a_liqSi * T0 + (b_liqSi / 2) * T0**2 + (c_liqSi / 3) * T0**3 + d_liqSi * (-(T0 ** (-1)))
    )
    Vm_liqSi = V0_liqSi * np.exp(3 * s_alpha_dT_liqSi)

    # dia si
    s_alpha_dT_diaSi = (a_diaSi * T + (b_diaSi / 2) * T**2 + (c_diaSi / 3) * T**3 + d_diaSi * (-(T ** (-1)))) - (
        a_diaSi * T0 + (b_diaSi / 2) * T0**2 + (c_diaSi / 3) * T0**3 + d_diaSi * (-(T0 ** (-1)))
    )
    Vm_diaSi = V0_diaSi * np.exp(3 * s_alpha_dT_diaSi)

    Vmfcc = X_Al * Vm_fccAl + X_Si * Vm_fccSi
    Vmdia = Vm_diaSi

    Vmliq = X_Al * Vm_liqAl + X_Si * Vm_liqSi

    # alpha    = (1/3.)*(a_fccAl+b_fccAl*T+c_fccAl*T**2+d_fccAl*T**(-2))
    # CLE      = alpha*(T-T0)
    # s_alpha_dT = (a_fccAl*T+(b_fccAl/2)*T**2+(c_fccAl/3)*T**3\
    #           +d_fccAl*(-T**(-1)))-\
    #           (a_fccAl*T0+(b_fccAl/2)*T0**2+(c_fccAl/3)*T0**3\
    #           +d_fccAl*(-T0**(-1)))
    # Vm_fccAl = V0_fccAl*np.exp(3*s_alpha_dT)

    outputs = {
        "Vmfcc": Vmfcc,
        "Vmdia": Vmdia,
        "Vmliq": Vmliq,
        "CLE_fccAl": CLE_fccAl,
    }
    return _select(outputs, output, "molar_volume")


def elas_const(T, output):
    """Temperature-dependent elastic constants of Al (from bulk and Young's
    moduli) and of diamond Si (http://www.ioffe.ru/SVA/NSM/Semicond/Si/mechanic.html).

    Parameters
    ----------
    T : float
        Temperature [K].
    output : str
        One of ``ELASTIC_CONSTANT_OUTPUTS``; values in Pa.
    """
    # B = bulk modulus
    # E = young modulus
    a_B_al = 8.030e1 * 1e9
    b_B_al = -1.319e-2 * 1e9
    c_B_al = -6.815e-6 * 1e9

    a_E_al = 7.028e1 * 1e9
    b_E_al = 2.114e-3 * 1e9
    c_E_al = -4.903e-5 * 1e9

    # data from http://www.ioffe.ru/SVA/NSM/Semicond/Si/mechanic.html
    a_C11_si = 16.38 * 1e10
    b_C11_si = -1.28e-3 * 1e10
    c_C11_si = 0.0

    a_C44_si = 8.17 * 1e10
    b_C44_si = -0.59e-3 * 1e10
    c_C44_si = 0.0

    a_C12_si = 5.92 * 1e10
    b_C12_si = -0.48e-3 * 1e10
    c_C12_si = 0.0

    B_al = a_B_al + b_B_al * T + c_B_al * T**2
    E_al = a_E_al + b_E_al * T + c_E_al * T**2

    C11_Si = a_C11_si + b_C11_si * T + c_C11_si * T**2
    C44_Si = a_C44_si + b_C44_si * T + c_C44_si * T**2
    C12_Si = a_C12_si + b_C12_si * T + c_C12_si * T**2
    # G = -3*E_al*B_al/(E_al-9*B_al) #shear modulus
    # C44 = G
    # C11 = (3*B_al+4*C44)/3
    # C12 = C11-2*C44
    C44 = -3 * E_al * B_al / (E_al - 9 * B_al)  # shear modulus
    C11 = B_al - 4 * E_al * B_al / (E_al - 9 * B_al)  # shear modulus
    C12 = B_al + 2 * E_al * B_al / (E_al - 9 * B_al)  # shear modulus

    outputs = {
        "C11": C11,
        "C12": C12,
        "C44": C44,
        "C11_Si": C11_Si,
        "C44_Si": C44_Si,
        "C12_Si": C12_Si,
        "B": B_al,
        "E": E_al,
    }
    return _select(outputs, output, "elas_const")
