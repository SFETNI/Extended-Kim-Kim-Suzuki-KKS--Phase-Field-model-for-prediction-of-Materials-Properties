"""Basic properties of the model components (independent of the legacy code)."""

import numpy as np
import pytest

from pf_kks.config import load_config
from pf_kks.elasticity import stiffness_tensor
from pf_kks.free_energy import free_energy_terms, phase_compositions
from pf_kks.grid import wave_vectors
from pf_kks.io import map_filename, parse_map_filename
from pf_kks.materials import MaterialTables
from pf_kks.microstructure import interface_area, interface_cells, place_precipitates

from .legacy import ROOT, TABLES


@pytest.mark.parametrize("n", [7, 8, 735])
def test_wave_vectors_match_fft_frequencies(n):
    k, k2, _ = wave_vectors((n, 3, 1), (1.0, 1.0, 1.0))
    expected = 2 * np.pi * np.fft.fftfreq(n)
    if n % 2 == 0:
        expected[n // 2] = 0.0  # Nyquist mode is left at zero
    np.testing.assert_allclose(k[0, :, 0, 0], expected, atol=1e-12)
    np.testing.assert_allclose(k2, k[0] ** 2 + k[1] ** 2 + k[2] ** 2)


def test_stiffness_tensor_symmetries():
    C = stiffness_tensor(108e9, 62e9, 28e9)
    np.testing.assert_array_equal(C, C.transpose(1, 0, 2, 3))
    np.testing.assert_array_equal(C, C.transpose(0, 1, 3, 2))
    np.testing.assert_array_equal(C, C.transpose(2, 3, 0, 1))
    assert C[0, 0, 0, 0] == 108e9 and C[0, 0, 1, 1] == 62e9 and C[0, 1, 0, 1] == 28e9


def test_phase_compositions_satisfy_kks_conditions():
    t = MaterialTables.from_npz(TABLES)
    T = 520.0
    A_a, B_a, A_t, B_t = (t.at(T, n) for n in ("A_alp", "B_alp", "A_thet", "B_thet"))
    eta = np.linspace(0, 1, 11).reshape(11, 1, 1)
    X = np.full_like(eta, 0.3)
    X_al, X_th = phase_compositions(X, eta, A_a, B_a, A_t, B_t)
    h = 3 * eta**2 - 2 * eta**3
    np.testing.assert_allclose((1 - h) * X_al + h * X_th, X, rtol=1e-12)  # mixture rule
    np.testing.assert_allclose(2 * A_a * X_al + B_a, 2 * A_t * X_th + B_t, rtol=1e-9)  # equal diffusion potential


def test_allen_cahn_driving_force_vanishes_in_bulk_phases():
    c = np.array([0.01, 0.02])
    for eta in (0.0, 1.0):
        _, G_eta, *_ = free_energy_terms(c, c + 0.9, np.full(2, eta), 1.0, 2.0, -1.0, 0.1, 3.0, -2.0, 0.5)
        np.testing.assert_array_equal(G_eta, 0.0)


def test_place_precipitates_and_interface_area():
    n, dx = 40, 0.5e-9
    eta, X, X_mat, X_pre = place_precipitates(
        n, n, 1, dx, dx, dx, 0.025, 0.99999, np.array([5e-9]), np.array([10e-9]), np.array([10e-9]), np.zeros(1)
    )
    assert eta.sum() == np.sum(
        (np.arange(n)[:, None] * dx - 10e-9) ** 2 + (np.arange(n)[None, :] * dx - 10e-9) ** 2 <= (5e-9) ** 2
    )
    np.testing.assert_array_equal(X[eta == 1], 0.99999)
    np.testing.assert_array_equal(X[eta == 0], 0.025)
    grid, cells = interface_cells(n, n)
    assert cells.shape == ((n - 1) ** 2, 4)
    assert interface_area(eta, grid, cells, dx, 2.5 * dx) == 0  # sharp initial interface
    smooth = np.clip(eta * 0.8 + 0.1, 0, 1)
    assert interface_area(smooth, grid, cells, dx, 2.5 * dx) == pytest.approx(
        (n - 1) ** 2 * dx**2 / (5 * dx), rel=1e-14
    )


def test_map_filename_roundtrip():
    name = map_filename(2221.716369079099, 400, 722.5789845064727)
    assert name == "map2221.72s_400_722.58K"
    assert parse_map_filename(name + ".npz") == (2221.72, 400.0, 722.58)


def test_reproducible_run_config_matches_archived_script():
    cfg = load_config(ROOT / "configs" / "reproducible_run.toml")
    assert (cfg.nx, cfg.ny, cfg.nz, cfg.dx) == (735, 735, 1, 0.5e-9)
    assert (cfg.dtime_s, cfg.dtime_max, cfg.dtime_min_divisor) == (75, 400, 1000)
    assert (cfg.c_Meta, cfg.c_eps0, cfg.c_gamma, cfg.c_D) == (0.6, 0.25, 1, 1)
    assert cfg.schedule_time == [0, 3 * (850 - 400)] and cfg.schedule_temperature == [400, 850]
    assert (cfg.n_steps, cfg.n_map_saves, cfg.n_surface_samples) == (30000, 40, 500)


def test_unknown_config_key_is_rejected(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[grid]\nnx = 10\nnot_a_key = 1\n")
    with pytest.raises(ValueError, match="not_a_key"):
        load_config(p)
