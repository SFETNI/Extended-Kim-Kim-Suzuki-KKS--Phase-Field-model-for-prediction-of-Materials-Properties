"""Full-size checks against the run archived in ReproducibleRun.zip (2022).

* ``map0.07s_400_400.00K.npz`` is the map the original run saved after its
  first step; ``data/histories/reproducible_run.npz`` holds the per-step histories stored in
  the final map of that run (30 000 steps).
* ``dtime.txt`` is the archived run log.

The first step costs ~20 s and 20 steps ~3 min on 4 cores; both are marked
``slow`` (run with ``pytest -m slow``).
"""

import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data" / "reference" / "reproducible_run"

pytestmark = pytest.mark.slow


def _run(tmp_path, n_steps):
    from pf_kks.cli import main

    out = tmp_path / f"run{n_steps}"
    assert (
        main([str(ROOT / "configs" / "reproducible_run.toml"), "--out", str(out), "--n-steps", str(n_steps), "--quiet"])
        == 0
    )
    return out


def test_first_step_matches_archived_map(tmp_path):
    out = _run(tmp_path, 1)
    new_files = os.listdir(out / "saved_maps")
    assert new_files == ["map0.07s_400_400.00K.npz"]
    with np.load(out / "saved_maps" / new_files[0]) as new, np.load(REF / "map0.07s_400_400.00K.npz") as ref:
        assert new.files == ref.files
        for key in ref.files:
            assert new[key].dtype == ref[key].dtype, key
            # 2022 cluster vs today's libraries: identical up to FFT round-off.
            np.testing.assert_allclose(new[key], ref[key], rtol=0, atol=1e-14, err_msg=key)
    first_log_line = (REF / "dtime.txt").read_text().splitlines()[1].rstrip()
    assert (out / "dtime.txt").read_text().splitlines()[1].rstrip() == first_log_line


def test_twenty_steps_match_archived_histories(tmp_path):
    out = _run(tmp_path, 20)
    new = np.load(out / "history.npz")
    ref = np.load(ROOT / "data" / "histories" / "reproducible_run.npz")
    for ref_key, new_key in [
        ("time_out", "time"),
        ("T_out", "T"),
        ("surf_out", "surf"),
        ("X_al_mean_out", "X_al_mean"),
        ("X_th_mean_out", "X_th_mean"),
        ("eta_mean_out", "eta_mean"),
    ]:
        np.testing.assert_allclose(new[new_key], ref[ref_key][:20], rtol=1e-12, atol=0, err_msg=ref_key)
