import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pf_kks.config import SimulationConfig  # noqa: E402
from pf_kks.microstructure import place_precipitates  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full-size (735 x 735) checks against the archived run")


def small_config(**overrides):
    """64 x 64 grid with the parameters of the reproducible run."""
    params = dict(
        nx=64,
        ny=64,
        n_steps=40,
        n_map_saves=3,
        n_surface_samples=10,
        material_tables=str(ROOT / "data" / "inputs" / "final_export.npz"),
    )
    params.update(overrides)
    return SimulationConfig(**params)


def three_precipitates(cfg):
    dx = cfg.dx
    r = np.array([3e-9, 2e-9, 4e-9])
    ox = np.array([8e-9, 20e-9, cfg.nx * dx * 0.6])
    oy = np.array([10e-9, 22e-9, cfg.ny * dx * 0.3])
    oz = np.zeros(3)
    return place_precipitates(cfg.nx, cfg.ny, cfg.nz, dx, dx, dx, cfg.x_matrix, cfg.x_precipitate, r, ox, oy, oz)


@pytest.fixture(scope="session")
def relaxed_state(tmp_path_factory):
    """Small microstructure relaxed for 300 steps without the adaptive scheme,
    so that the energy residual is in the admissible range."""
    from pf_kks import solver

    cfg = small_config(n_steps=300, x_equilibrium=0.0)
    state = three_precipitates(cfg)
    res = solver.run(cfg, *state, out_dir=str(tmp_path_factory.mktemp("relax")), verbose=False)
    return res.eta, res.X, res.X_al, res.X_th


@pytest.fixture
def copy_state():
    def _copy(state):
        return tuple(np.array(a, copy=True) for a in state)

    return _copy


os.environ.setdefault("MPLBACKEND", "Agg")
