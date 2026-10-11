"""Initial microstructures: circular/spherical Si precipitates in the Al matrix."""

import numpy as np
from numba import njit


@njit(cache=True)
def place_precipitates(nx, ny, nz, dx, dy, dz, x_matrix, x_precipitate, r, ox, oy, oz):
    """Insert spherical precipitates (circles in 2-D) in a uniform matrix.

    Original name: ``micro_numba``.

    Parameters
    ----------
    nx, ny, nz : int
        Grid size.
    dx, dy, dz : float
        Grid spacing [m].
    x_matrix, x_precipitate : float
        Si molar fraction of the matrix and of the precipitates.
    r, ox, oy, oz : ndarray
        Radius and centre coordinates of each precipitate [m].

    Returns
    -------
    eta : ndarray
        Order parameter (1 inside precipitates, 0 elsewhere).
    X : ndarray
        Si molar fraction.
    X_matrix : ndarray
        Si molar fraction of the matrix phase (0 inside precipitates).
    X_precipitate : ndarray
        Si molar fraction of the precipitate phase (0 in the matrix).
    """
    eta = np.zeros((nx, ny, nz))
    X = np.ones((nx, ny, nz)) * x_matrix
    X_mat = np.ones((nx, ny, nz)) * x_matrix
    X_pre = np.zeros((nx, ny, nz))
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    for i_x in range(nx):
        for i_y in range(ny):
            for i_z in range(nz):
                for i in range(len(r)):
                    norm = np.sqrt((ox[i] - x[i_x]) ** 2 + (oy[i] - y[i_y]) ** 2 + (oz[i] - z[i_z]) ** 2)
                    if norm <= r[i]:
                        X_pre[i_x][i_y][i_z] = x_precipitate
                        X_mat[i_x][i_y][i_z] = 0
                        X[i_x][i_y][i_z] = x_precipitate
                        eta[i_x][i_y][i_z] = 1
    return eta, X, X_mat, X_pre


@njit(cache=True)
def add_nuclei(nx, ny, nz, dx, dy, dz, eta, X, X_mat, X_pre, r, ox, oy, oz):
    """Insert small nuclei (X_Si = 0.99999) into existing fields, in place.

    Original name: ``nucle_numba``.
    """
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    z = np.arange(nz) * dz
    for i_x in range(nx):
        for i_y in range(ny):
            for i_z in range(nz):
                for i in range(len(r)):
                    norm = np.sqrt((ox[i] - x[i_x]) ** 2 + (oy[i] - y[i_y]) ** 2 + (oz[i] - z[i_z]) ** 2)
                    if norm <= r[i]:
                        eta[i_x][i_y][i_z] = 1
                        X_pre[i_x][i_y][i_z] = 0.99999
                        X_mat[i_x][i_y][i_z] = 0
                        X[i_x][i_y][i_z] = 0.99999
    return eta, X, X_mat, X_pre


def load_precipitates(path):
    """Read precipitate centres and radii.

    Accepts the ``micro_final.npz`` format (keys ``ox_cyl``, ``oy_cyl``,
    ``oz_cyl``, ``r_cyl``, in metres) or a 4-column text file ``x y z r``
    (``precipitates_xyz_r.txt``).

    Returns
    -------
    ox, oy, oz, r : ndarray
    """
    path = str(path)
    if path.endswith(".npz"):
        micro = np.load(path)
        xyz_r = np.asarray([micro["ox_cyl"], micro["oy_cyl"], micro["oz_cyl"], micro["r_cyl"]]).transpose()
    else:
        xyz_r = np.atleast_2d(np.loadtxt(path))
    return xyz_r[:, 0], xyz_r[:, 1], xyz_r[:, 2], xyz_r[:, 3]


def interface_cells(nx, ny):
    """Grid-point indices and the quadrilateral cells of a 2-D grid.

    Used to measure the precipitate/matrix interface area. Original name:
    ``compu_cell``.

    Returns
    -------
    grid : ndarray, shape (nx, ny)
        Linear index of each grid point.
    cells : ndarray, shape ((nx-1)*(ny-1), 4)
        Corner indices of each cell.
    """
    grid = np.resize(np.arange(0, nx * ny, 1, dtype=int), (nx, ny))
    i, j = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    i, j = i.ravel(), j.ravel()
    cells = np.stack([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]], axis=1)
    return grid, cells


def interface_area(eta, grid, cells, dx, lambd):
    """Interface measure used for the surface-energy DSC contribution.

    Counts the cells whose four corners all lie in the diffuse interface
    (0.1 <= eta <= 0.9) and scales by ``dx^2 / (2 lambd)`` (cell area divided by
    the interface thickness), as in the original code. Returns 0 for a 1-D grid.
    """
    if eta.shape[1] == 1:
        return 0
    selected = grid[(eta[:, :, 0] >= 0.1) & (eta[:, :, 0] <= 0.9)]
    inside = np.isin(cells, selected)
    return np.sum(np.all(inside, axis=1)) * (dx**2 / (2 * lambd))
