"""Spectral grid: wave vectors of the periodic simulation box."""

import numpy as np


def _frequencies(n, d):
    """Angular wave numbers of an ``n``-point axis with spacing ``d``.

    Same ordering as the original ``prepar_fft``. For an even ``n`` the Nyquist
    entry ``n/2`` is left at zero, as in the original code.
    """
    lin = np.zeros(n)
    if n % 2 == 1:
        half = int((n - 1) / 2.0 + 1)
        lin[:half] = np.arange(0, half, 1) * 2 * np.pi / (n * d)
        lin[half:] = np.arange(int(-(n + 1) / 2.0 + 1), 0, 1) * 2 * np.pi / (n * d)
    else:
        lin[0 : int(n / 2.0)] = np.arange(0, int(n / 2.0), 1) * 2 * np.pi / (n * d)
        lin[int(n / 2.0 + 1) :] = np.arange(int(-n / 2.0 + 1), 0, 1) * 2 * np.pi / (n * d)
    return lin


def wave_vectors(shape, spacing):
    """Wave vectors of the grid.

    Parameters
    ----------
    shape : tuple of int
        ``(Nx, Ny, Nz)``.
    spacing : tuple of float
        ``(dx, dy, dz)``; the solver uses the dimensionless spacing ``(1, 1, 1)``.

    Returns
    -------
    k : ndarray, shape (3, Nx, Ny, Nz)
        Wave-vector components.
    k2 : ndarray, shape (Nx, Ny, Nz)
        Squared norm ``|k|^2``.
    k4 : ndarray, shape (Nx, Ny, Nz)
        ``|k|^4``.
    """
    nx, ny, nz = shape
    dx, dy, dz = spacing
    k = np.zeros((3, nx, ny, nz))
    k[0] = _frequencies(nx, dx)[:, None, None]
    k[1] = _frequencies(ny, dy)[None, :, None]
    k[2] = _frequencies(nz, dz)[None, None, :]
    k2 = k[0] ** 2 + k[1] ** 2 + k[2] ** 2
    k4 = k2**2
    return k, k2, k4
