"""pf_kks.postprocess.dsc reproduces DSC_tool/compute_DSC.ipynb.

The notebook's code cells are executed as they are (plots go to a non-interactive
backend) on the archived history of the run it was written for, and the DSC
terms it computes are compared with the package's.
"""

import json
import shutil
import sys
import types

import numpy as np

from pf_kks.postprocess.dsc import compute_dsc

from .legacy import ROOT, working_directory

NOTEBOOK = ROOT / "legacy_version" / "DSC_tool" / "compute_DSC.ipynb"
HISTORY = ROOT / "data" / "histories" / "small_grid_735_dsc.npz"


def _run_notebook(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    cells = [c for c in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
    # Execute up to the DSC terms; the last cells need a file absent from the
    # original repository (Yang.txt).
    source = "\n".join("".join(c["source"]) for c in cells)
    source = source.split("exp_data=np.loadtxt")[0]
    shutil.copy(HISTORY, tmp_path / "map1727.75s_400_739.09K.npz")
    sys.modules.setdefault("pyvista", types.ModuleType("pyvista"))  # imported, not used
    sys.path.insert(0, str(ROOT / "legacy_version" / "DSC_tool"))
    namespace = {}
    try:
        with working_directory(tmp_path):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    finally:
        sys.path.remove(str(ROOT / "legacy_version" / "DSC_tool"))
    return namespace


def test_dsc_terms_match_notebook(tmp_path):
    nb = _run_notebook(tmp_path)
    new = compute_dsc(np.load(HISTORY))
    np.testing.assert_array_equal(new.T, nb["T"])
    np.testing.assert_array_equal(new.background, nb["dsc_bkg"])
    np.testing.assert_array_equal(new.surface, nb["dsc_surf"])
    np.testing.assert_array_equal(new.precipitation, nb["dsc_preci"])
    np.testing.assert_array_equal(new.total, (nb["dsc_bkg"] + nb["dsc_surf"] + nb["dsc_preci"]) * 3)
