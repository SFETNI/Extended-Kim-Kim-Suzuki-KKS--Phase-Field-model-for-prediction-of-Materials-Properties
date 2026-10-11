"""Energy-balance residual driving the adaptive time step.

The residual compares the rate of change of the total energy with the
dissipation of the Cahn-Hilliard and Allen-Cahn fluxes and the temperature
contribution:

    RES = dE/dt - sum(dX/dt * mu) + sum((1/L) (deta/dt)^2) - sum(dG/dt)

All quantities are dimensionless. Original name: ``compute_residual_KKS``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Residual:
    total: float
    energy_derivative: float
    diffusion_term: float
    allen_cahn_term: float
    temperature_term: float


def kks_residual(time_s_history, energy_history, good_step, istep, eta_pair, X_pair, mu_pair, G_pair, L_s, dtime_s):
    """Energy-balance residual of the last accepted step pair.

    Parameters
    ----------
    time_s_history : list of float
        Dimensionless time at each completed step.
    energy_history : ndarray
        Dimensionless total energy at each step.
    good_step : int
        Step from which the adaptive scheme is active.
    istep : int
        Current step.
    eta_pair, X_pair, mu_pair, G_pair : list of ndarray
        Last two values of eta, X, the diffusion potential df_alpha/dc and the
        free-energy density.
    L_s : float
        Dimensionless Allen-Cahn coefficient.
    dtime_s : float
        Current dimensionless time step.
    """
    nx, ny = eta_pair[-1].shape[0], eta_pair[-1].shape[1]
    time_s = np.asarray(time_s_history[good_step:])
    energy = energy_history[good_step:istep]

    eta_plus = eta_pair[-1].reshape(nx, ny)
    eta_minus = eta_pair[-2].reshape(nx, ny)
    X_plus = X_pair[-1].reshape(nx, ny)
    X_minus = X_pair[-2].reshape(nx, ny)
    mu_mean = np.asarray((mu_pair[-1].reshape(nx, ny) + mu_pair[-2].reshape(nx, ny)) / 2)
    G = np.asarray(G_pair).reshape(len(G_pair), nx, ny)

    dXdt = (X_plus - X_minus) / dtime_s
    diffusion = dXdt * mu_mean
    energy_derivative = (np.diff(energy) / np.diff(time_s)).mean()
    detadt = (eta_plus - eta_minus) / dtime_s
    allen_cahn = (1 / L_s) * (detadt) ** 2
    temperature = (G[-1] - G[-2]) / dtime_s

    total = energy_derivative + (-1 * diffusion + allen_cahn - temperature).sum()
    return Residual(total, energy_derivative, diffusion.sum(), allen_cahn.sum(), temperature.sum())
