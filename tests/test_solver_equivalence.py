"""The refactored solver reproduces the original ``pyFi_adapt.main`` (version of
``ReproducibleRun.zip``) on small grids, through every branch of the time loop:
fixed steps before the adaptive scheme, accepted, enlarged and rejected
adaptive steps, and the abort after too many rejections."""

import os

import numpy as np
import pytest

from pf_kks import solver

from .conftest import small_config, three_precipitates
from .legacy import legacy_dict, run_legacy

FIELDS = ("X", "X_al", "X_th", "eta")
HISTORIES = (
    "eta_mean",
    "deta_mean",
    "X_mean",
    "X_al_mean",
    "dX_al_mean",
    "X_th_mean",
    "dX_th_mean",
    "dT",
    "T",
    "dtime",
    "time",
    "surf",
    "dsurf",
    "L",
    "energy_s",
    "dtime_s",
    "residual_total",
    "residual_energy_derivative",
    "residual_diffusion",
    "residual_allen_cahn",
    "residual_temperature",
)


def _run_both(tmp_path, cfg, state, copy_state, **legacy_kwargs):
    legacy_dir = tmp_path / "legacy"
    new_dir = tmp_path / "new"
    legacy_dir.mkdir()
    new_dir.mkdir()
    old = run_legacy(cfg, *copy_state(state), str(legacy_dir), **legacy_kwargs)
    try:
        new = solver.run(cfg, *copy_state(state), out_dir=str(new_dir), verbose=False)
    except solver.AdaptiveStepError as exc:
        new = exc
    return old, new, legacy_dir, new_dir


def _assert_same_outputs(old, new, legacy_dir, new_dir):
    old = legacy_dict(old)
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(new, name), old[name], err_msg=name)
    for name in HISTORIES:
        np.testing.assert_array_equal(np.asarray(new.history[name], float), np.asarray(old[name], float), err_msg=name)
    assert new.good_step == old["good_step"]
    assert new.ttime == old["ttime"]
    _assert_same_files(legacy_dir, new_dir)


def _assert_same_files(legacy_dir, new_dir):
    old_maps = sorted(os.listdir(legacy_dir / "saved_maps"))
    assert old_maps == sorted(os.listdir(new_dir / "saved_maps"))
    for name in old_maps:
        with np.load(legacy_dir / "saved_maps" / name) as a, np.load(new_dir / "saved_maps" / name) as b:
            assert a.files == b.files
            for key in a.files:
                np.testing.assert_array_equal(a[key], b[key], err_msg=f"{name}:{key}")
                assert a[key].dtype == b[key].dtype, f"{name}:{key}"
    for log in ("log.txt", "dtime.txt"):
        assert (legacy_dir / log).read_text() == (new_dir / log).read_text(), log


def test_fixed_steps_before_adaptive_scheme(tmp_path, copy_state):
    cfg = small_config(n_steps=40)
    old, new, ld, nd = _run_both(tmp_path, cfg, three_precipitates(cfg), copy_state)
    # The matrix never reaches x_equilibrium here: the legacy code then fails on
    # its return statement (good_step undefined) after having written its maps.
    assert isinstance(old, UnboundLocalError)
    assert new.good_step is None
    _assert_same_files(ld, nd)


def test_accepted_adaptive_steps(tmp_path, relaxed_state, copy_state):
    cfg = small_config(n_steps=30, x_equilibrium=0.2)
    old, new, ld, nd = _run_both(tmp_path, cfg, relaxed_state, copy_state)
    _assert_same_outputs(old, new, ld, nd)
    assert new.good_step == 2
    assert "We get  (resmin <RE <resmax)" in (nd / "log.txt").read_text()


def test_enlarged_adaptive_steps(tmp_path, relaxed_state, copy_state):
    # A residual threshold inside the residual range of this run makes the
    # scheme switch from accepting to enlarging the step, up to dtime_max.
    cfg = small_config(n_steps=40, x_equilibrium=0.2, dtime_s=60, dtime_max=150, residual_min=7.6e-4)
    old, new, ld, nd = _run_both(tmp_path, cfg, relaxed_state, copy_state, residual_min=7.6e-4)
    _assert_same_outputs(old, new, ld, nd)
    log = (nd / "log.txt").read_text()
    assert "RE <resmin" in log and "We get  (resmin <RE <resmax)" in log
    # The recorded step is the one proposed for the next step (enlarged after the
    # solve, clamped to dtime_max only at the next step): see TECHNICAL_REPORT.md.
    assert max(new.history["dtime_s"]) == 150 * cfg.step_factor


def test_rejected_steps_and_abort(tmp_path, copy_state):
    # From the unrelaxed microstructure the residual stays above residual_max:
    # every attempt is rejected until the iteration limit stops the run.
    cfg = small_config(n_steps=40, x_equilibrium=0.2)
    old, new, ld, nd = _run_both(tmp_path, cfg, three_precipitates(cfg), copy_state)
    assert isinstance(old, SystemExit)
    assert isinstance(new, solver.AdaptiveStepError)
    _assert_same_files(ld, nd)
    assert (nd / "dtime.txt").read_text().count("RE>resmax") == cfg.max_iterations - 1


@pytest.mark.parametrize("backend", ["numpy"])
def test_numpy_fft_backend_matches_to_round_off(tmp_path, relaxed_state, copy_state, backend):
    cfg = small_config(n_steps=20, x_equilibrium=0.2)
    ref = solver.run(cfg, *copy_state(relaxed_state), out_dir=str(tmp_path / "a"), verbose=False)
    cfg.fft_backend = backend
    alt = solver.run(cfg, *copy_state(relaxed_state), out_dir=str(tmp_path / "b"), verbose=False)
    for name in FIELDS:
        np.testing.assert_allclose(getattr(alt, name), getattr(ref, name), rtol=0, atol=1e-10)
