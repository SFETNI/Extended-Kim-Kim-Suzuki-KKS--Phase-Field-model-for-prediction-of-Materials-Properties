"""KKS free-energy functional with parabolic phase free energies.

Phase free energies (dimensionless, divided by ``C44 * Vm``):

    f_alpha(c) = A_alp c^2 + B_alp c + C_alp      (Al matrix, FCC)
    f_theta(c) = A_thet c^2 + B_thet c + C_thet   (Si precipitate, diamond)

Interpolation ``h(eta) = 3 eta^2 - 2 eta^3`` and double well
``g(eta) = eta^2 (1 - eta)^2``. References: Kim, Kim & Suzuki, Phys. Rev. E 60
(1999) 7186; Hu et al., Acta Mater. (2007).
"""

import numpy as np


def interpolation(eta):
    """``h(eta) = 3 eta^2 - 2 eta^3``."""
    return 3.0 * eta**2 - 2 * eta**3


def free_energy_terms(c_alpha, c_theta, eta, w_s, A_alp_s, B_alp_s, C_alp_s, A_thet_s, B_thet_s, C_thet_s):
    """Dimensionless free energy and its derivatives.

    Original name: ``free_energy_para_new``.

    Returns
    -------
    G_s : ndarray
        Free-energy density.
    G_eta_s : ndarray
        dG/deta at constant phase compositions (Allen-Cahn driving force).
    G_cc_s : ndarray
        Effective d2G/dc2 (Hu 2007, KKS 1999).
    G_ceta_G_cc : ndarray
        ``h'(eta) (c_alpha - c_theta)``, the coupling term of the Cahn-Hilliard
        flux.
    G_ceta_s : ndarray
        d2G/(dc deta) (Hu 2007, eq. 27).
    h_eta : ndarray
        h'(eta).
    fa_ca_s : ndarray
        df_alpha/dc (diffusion potential).
    ft_ct_s : ndarray
        df_theta/dc.
    """
    fa_s = A_alp_s * c_alpha**2 + B_alp_s * c_alpha + C_alp_s
    fa_ca_s = 2 * A_alp_s * c_alpha + B_alp_s
    fa_caca_s = 2 * A_alp_s

    ft_s = A_thet_s * c_theta**2 + B_thet_s * c_theta + C_thet_s
    ft_ct_s = 2 * A_thet_s * c_theta + B_thet_s
    ft_ctct_s = 2 * A_thet_s

    h = 3.0 * eta**2 - 2 * eta**3
    h_eta = 6.0 * eta - 6 * eta**2
    g = eta**2 - 2 * eta**3 + eta**4
    g_eta = 2.0 * eta - 6.0 * eta**2 + 4.0 * eta**3

    G_s = (1 - h) * fa_s + h * ft_s + w_s * g
    G_eta_s = -h_eta * (fa_s - ft_s - (c_alpha - c_theta) * fa_ca_s) + w_s * g_eta
    G_cc_s = (ft_ctct_s * fa_caca_s) / ((1 - h) * ft_ctct_s + h * fa_caca_s)
    G_ceta_G_cc = h_eta * (c_alpha - c_theta)

    denom = (1 - h) * ft_ctct_s + h * fa_caca_s
    c_alpha_c = ft_ctct_s / denom
    c_theta_c = fa_caca_s / denom
    fa_c_s = fa_ca_s * c_alpha_c
    ft_c_s = ft_ct_s * c_theta_c
    fa_ca_c_s = 2 * A_alp_s * c_alpha_c
    G_ceta_s = -h_eta * (fa_c_s - ft_c_s - (c_alpha_c - c_theta_c) * fa_ca_s - (c_alpha - c_theta) * fa_ca_c_s)
    return G_s, G_eta_s, G_cc_s, G_ceta_G_cc, G_ceta_s, h_eta, fa_ca_s, ft_ct_s


def phase_compositions(X, eta, A_alp, B_alp, A_thet, B_thet):
    """Phase compositions from the KKS equal-diffusion-potential condition.

    With parabolic free energies the condition ``df_alpha/dc = df_theta/dc`` and
    the mixture rule ``X = (1-h) c_alpha + h c_theta`` give closed forms.
    Non-finite values are replaced by +/-1e-6 as in the original code.

    Returns
    -------
    X_al, X_th : ndarray
        Si molar fraction in the matrix and in the precipitate phase.
    """
    h = 3.0 * eta**2 - 2 * eta**3
    X_al = (X - h * ((B_alp - B_thet) / (2 * A_thet))) / ((1 - h) + h * (A_alp / A_thet))
    X_al[X_al == np.inf] = 1e-6
    X_al[X_al == -np.inf] = -1e-6
    X_al[np.isnan(X_al)] = 1e-6
    X_th = (A_alp / A_thet) * X_al + (B_alp - B_thet) / (2 * A_thet)
    X_th[X_th == np.inf] = 1e-6
    X_th[X_th == -np.inf] = -1e-6
    X_th[np.isnan(X_th)] = 1e-6
    return X_al, X_th


def total_energy(eta, G, kappa2, el):
    """Dimensionless total energy used by the residual-based time stepping.

    Kept identical to the original ``calculate_energ`` (2-D fields only). Note
    that ``np.gradient`` returns one array per axis, so the chemical and elastic
    densities are broadcast over both gradient components and therefore enter
    the sum twice; this is what the archived runs used.
    """
    nx, ny = eta.shape[0], eta.shape[1]
    eta = eta.reshape(nx, ny)
    G = G.reshape(nx, ny)
    el = el.reshape(nx, ny)
    return (G + 0.5 * kappa2 * (np.asarray(np.gradient(eta))) ** 2 + el).sum()
