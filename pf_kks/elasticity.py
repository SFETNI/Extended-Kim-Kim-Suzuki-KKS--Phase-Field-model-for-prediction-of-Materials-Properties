"""Micro-elasticity: Khachaturyan homogeneous-modulus solution in Fourier space.

The eigenstrain of the precipitate is the dilatational ``epsX * eta^2 * I``.
"""

import itertools

import numpy as np
from numba import njit

# Symmetric component pairs (the lower triangle is filled by symmetry).
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _voigt_index(i, j):
    """Voigt index of the tensor index pair (i, j)."""
    if i == j:
        return i
    return 6 - i - j


def stiffness_tensor(C11, C12, C44):
    """Fourth-order stiffness tensor ``C_ijkl`` of a cubic crystal.

    Original name: ``elas_modul``.
    """
    C = np.array(
        [
            [C11, C12, C12, 0, 0, 0],
            [C12, C11, C12, 0, 0, 0],
            [C12, C12, C11, 0, 0, 0],
            [0, 0, 0, C44, 0, 0],
            [0, 0, 0, 0, C44, 0],
            [0, 0, 0, 0, 0, C44],
        ]
    )
    C_ijkl = np.zeros((3, 3, 3, 3), dtype=float)
    for i, j, k, l in itertools.product(range(3), range(3), range(3), range(3)):
        C_ijkl[i, j, k, l] = C[_voigt_index(i, j), _voigt_index(k, l)]
    return C_ijkl


@njit(cache=True)
def _det_and_cofactor_3x3(mat):
    """Determinant and transposed cofactor matrix (adjugate) of a 3x3 matrix."""
    A = +(mat[1][1] * mat[2][2] - mat[2][1] * mat[1][2])
    B = -(mat[1][0] * mat[2][2] - mat[2][0] * mat[1][2])
    C = +(mat[1][0] * mat[2][1] - mat[2][0] * mat[1][1])
    D = -(mat[0][1] * mat[2][2] - mat[2][1] * mat[0][2])
    E = +(mat[0][0] * mat[2][2] - mat[2][0] * mat[0][2])
    F = -(mat[0][0] * mat[2][1] - mat[2][0] * mat[0][1])
    G = +(mat[0][1] * mat[1][2] - mat[1][1] * mat[0][2])
    H = -(mat[0][0] * mat[1][2] - mat[1][0] * mat[0][2])
    I = +(mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1])
    comat = np.array([[A, D, G], [B, E, H], [C, F, I]])
    det = mat[0][0] * A + mat[0][1] * B + mat[0][2] * C
    return det, comat


@njit(cache=True)
def green_operator(nx, ny, nz, C_ijkl, k):
    """Green operator ``Omega[x, y, z, k, h, i, j]`` in Fourier space.

    Computed once per simulation (the stiffness is frozen at the initial
    temperature). Original name: ``compute_green``.
    """
    omega = np.zeros((nx, ny, nz, 3, 3, 3, 3))
    K_ik = np.zeros((nx, ny, nz, 3, 3))
    N_ki = np.zeros((nx, ny, nz, 3, 3))
    for i_x in range(nx):
        for i_y in range(ny):
            for i_z in range(nz):
                for i in range(3):
                    for j in range(3):
                        for kk in range(3):
                            for l in range(3):
                                K_ik[i_x, i_y, i_z, i, kk] += (
                                    C_ijkl[i, j, kk, l] * k[j, i_x, i_y, i_z] * k[l, i_x, i_y, i_z]
                                )
                det, co_mat = _det_and_cofactor_3x3(K_ik[i_x, i_y, i_z, :, :])
                if det == 0:
                    det = 1
                N_ki[i_x, i_y, i_z, :, :] = co_mat / det
                for kk in range(3):
                    for h in range(3):
                        for i in range(3):
                            for j in range(3):
                                omega[i_x, i_y, i_z, kk, h, i, j] += 0.25 * (
                                    N_ki[i_x, i_y, i_z, h, i] * k[j, i_x, i_y, i_z] * k[kk, i_x, i_y, i_z]
                                    + N_ki[i_x, i_y, i_z, kk, i] * k[j, i_x, i_y, i_z] * k[h, i_x, i_y, i_z]
                                    + N_ki[i_x, i_y, i_z, h, j] * k[i, i_x, i_y, i_z] * k[kk, i_x, i_y, i_z]
                                    + N_ki[i_x, i_y, i_z, kk, j] * k[i, i_x, i_y, i_z] * k[h, i_x, i_y, i_z]
                                )
    return omega


def _symmetric_transform(transform, field, out):
    """Apply ``transform`` to the six independent components of a symmetric
    tensor field and fill the lower triangle by symmetry."""
    for i, j in _PAIRS:
        out[i, j] = transform(field[i, j])
        if i != j:
            out[j, i] = out[i, j]
    return out


def solve_elasticity(eta, omega, C_ijkl, epsX, fft):
    """Elastic fields, elastic energy density and its derivative w.r.t. eta.

    A single Khachaturyan iteration starting from zero total strain, as in the
    original ``solve_elas``.

    Parameters
    ----------
    eta : ndarray, shape (Nx, Ny, Nz)
        Order parameter.
    omega : ndarray
        Green operator from :func:`green_operator`.
    C_ijkl : ndarray, shape (3, 3, 3, 3)
        Stiffness tensor [Pa].
    epsX : float
        Dilatational eigenstrain of the precipitate [-].
    fft : pf_kks.fft.FFT
        FFT backend.

    Returns
    -------
    sig_el : ndarray, shape (3, 3, Nx, Ny, Nz)
        Elastic stress.
    eps_el : ndarray, shape (3, 3, Nx, Ny, Nz)
        Elastic strain.
    el_eta : ndarray, shape (Nx, Ny, Nz)
        Derivative of the elastic energy density w.r.t. eta.
    el : ndarray, shape (Nx, Ny, Nz)
        Elastic energy density.
    """
    nx, ny, nz = eta.shape
    sigk = np.zeros((3, 3, nx, ny, nz)) * 1j
    epsk = np.zeros((3, 3, nx, ny, nz)) * 1j
    eps = np.zeros((3, 3, nx, ny, nz))

    epsX_kl = np.einsum("xyz,kl->klxyz", eta**2, np.eye(3)) * epsX
    epsX_kl_eta = np.einsum("xyz,kl->klxyz", 2 * eta, np.eye(3)) * epsX

    eps_el = eps - epsX_kl
    sig_el = np.einsum("ijkl,klxyz->ijxyz", C_ijkl, eps_el)

    _symmetric_transform(fft.fft_planned, sig_el, sigk)
    _symmetric_transform(fft.fft_planned, eps, epsk)
    epsk = epsk - np.einsum("xyzklij,ijxyz->klxyz", omega, sigk)
    _symmetric_transform(lambda a: np.real(fft.ifft_planned(a)), epsk, eps)

    eps_el = eps - epsX_kl
    eps_ijkl = (
        -np.einsum("ijxyz,klxyz->ijklxyz", eps, epsX_kl_eta)
        - np.einsum("ijxyz,klxyz->ijklxyz", epsX_kl_eta, eps)
        + np.einsum("ijxyz,klxyz->ijklxyz", epsX_kl_eta, epsX_kl)
        + np.einsum("ijxyz,klxyz->ijklxyz", epsX_kl, epsX_kl_eta)
    )
    sig_el = np.einsum("ijkl,klxyz->ijxyz", C_ijkl, eps_el)
    el = 0.5 * np.einsum("ijxyz,ijxyz->xyz", sig_el, eps_el)
    el_eta = 0.5 * np.einsum("ijkl,ijklxyz->xyz", C_ijkl, eps_ijkl)
    return sig_el, eps_el, el_eta, el
