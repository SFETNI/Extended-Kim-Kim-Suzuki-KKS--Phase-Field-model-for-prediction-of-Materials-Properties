"""Temperature-dependent material tables (``final_export.npz``).

The tables give, at a few temperatures, the parabolic free-energy coefficients
of both phases, the elastic constants, the precipitate misfit, molar volume,
interface energy and equilibrium matrix composition. The solver interpolates
them linearly in temperature.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class MaterialTables:
    T: np.ndarray  # temperatures of the table [K]
    A_alp: np.ndarray  # matrix (FCC) parabola: A c^2 + B c + C [J mol^-1]
    B_alp: np.ndarray
    C_alp: np.ndarray
    A_thet: np.ndarray  # precipitate (diamond) parabola [J mol^-1]
    B_thet: np.ndarray
    C_thet: np.ndarray
    C11: np.ndarray  # matrix elastic constants [Pa]
    C12: np.ndarray
    C44: np.ndarray
    epsX: np.ndarray  # dilatational eigenstrain (misfit x scaling factor) [-]
    Vm: np.ndarray  # molar volume [m^3 mol^-1]
    gamma: np.ndarray  # interface energy (x scaling factor) [J m^-2]
    X_Si_fcc: np.ndarray  # equilibrium Si fraction of the matrix [-]
    D0: float  # Si diffusion pre-factor tabulated with the data [m^2 s^-1]
    QD: float  # Si diffusion activation energy tabulated with the data [J mol^-1]

    @classmethod
    def from_npz(cls, path, c_gamma=1.0, c_eps0=1.0):
        """Load the tables, scaling the interface energy by ``c_gamma`` and the
        misfit strain by ``c_eps0`` (calibration factors of the study)."""
        d = np.load(path)
        return cls(
            T=d["T"],
            A_alp=d["A_fcc"],
            B_alp=d["B_fcc"],
            C_alp=d["C_fcc"],
            A_thet=d["A_dia"],
            B_thet=d["B_dia"],
            C_thet=d["C_dia"],
            C11=d["C11"],
            C12=d["C12"],
            C44=d["C44"],
            epsX=d["delta"] * c_eps0,
            Vm=d["Vm"],
            gamma=d["gamma"] * c_gamma,
            X_Si_fcc=d["X_Si_fcc"],
            D0=d["D0"],
            QD=d["QD"],
        )

    def at(self, T, name):
        """Linear interpolation of table ``name`` at temperature ``T``."""
        return np.interp(T, self.T, getattr(self, name))
