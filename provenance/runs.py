"""The simulation runs archived in the original repository.

Each run is identified by its final map; earlier maps of the same run hold the
same histories truncated at an earlier step. Used by
``scripts/extract_histories.py`` and ``scripts/make_zenodo_bundle.py``.
"""

RUNS = {
    "big_grid_1420": {
        "description": "1420 x 1420 grid (710 nm), 400 K -> 719 K, 30 000 steps: maps shown in the article",
        "maps": "saved_maps_Big_grid/map*.npz",
        "final": "saved_maps_Big_grid/map7413.37s_400_719.00K.npz",
    },
    "small_grid_735_dsc": {
        "description": "735 x 735 grid, 400 K -> 739 K, 70 000 steps: run used by DSC_tool/compute_DSC.ipynb",
        "maps": "saved_maps_small_grids/map*.npz",
        "final": "saved_maps_small_grids/map1727.75s_400_739.09K.npz",
    },
    "small_grid_735_36842": {
        "description": "735 x 735 grid, 400 K -> 700 K, 36 842 steps (DSC_tool/saved_npz_DSC)",
        "maps": "DSC_tool/saved_npz_DSC/map[59]*.npz",
        "final": "DSC_tool/saved_npz_DSC/map909.19s_400_700.29K.npz",
    },
    "small_grid_735_671K": {
        "description": "735 x 735 grid, 400 K -> 671 K, 30 000 steps (DSC_tool)",
        "maps": "DSC_tool/map740.23s_400_670.96K.npz",
        "final": "DSC_tool/map740.23s_400_670.96K.npz",
    },
    "small_grid_735_708K": {
        "description": "735 x 735 grid, 400 K -> 708 K, 30 000 steps (save_ipynb)",
        "maps": "save_ipynb/map*.npz",
        "final": "save_ipynb/map7415.84s_400_708.08K.npz",
    },
    "reproducible_run": {
        "description": "735 x 735 grid, 400 K -> 723 K, 30 000 steps, ReproducibleRun.zip",
        "maps": "ReproducibleRun/saved_maps/map*.npz",
        "final": "ReproducibleRun/saved_maps/map2221.72s_400_722.58K.npz",
    },
    "nucleation_438K": {
        "description": "735 x 735 grid, isothermal 438 K with nucleation, 76 531 steps",
        "maps": "saved_map_nucle/map*.npz",
        "final": "saved_map_nucle/map9111.05s_438_438.00K.npz",
    },
}

# Top-level map of the big-grid run saved by a later code version: same
# histories as saved_maps_Big_grid/map4754.92s_400_700.29K.npz plus the vacancy
# site fraction ``xv_out`` (used by Post_Process.ipynb).
VACANCY_MAP = "map4754.92s_400_700.29K.npz"
