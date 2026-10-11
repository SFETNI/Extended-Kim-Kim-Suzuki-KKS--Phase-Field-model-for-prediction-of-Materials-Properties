# Extended KKS phase-field model — simulation data (AlSi10Mg, LPBF)

Simulation outputs of the extended Kim-Kim-Suzuki phase-field model for Si
precipitation in LPBF AlSi10Mg under non-isothermal heating (DSC conditions),
accompanying:

> S. Fetni, J. Delahaye, H. Sepúlveda, L. Duchêne, A.M. Habraken, A. Mertens,
> *Extension of a phase-field KKS model to predict the microstructure evolution
> in LPBF AlSi10Mg alloy submitted to non isothermal processes*,
> Computational Materials Science 244 (2024) 113197.
> https://doi.org/10.1016/j.commatsci.2024.113197

Code: https://github.com/SFETNI/Extended-Kim-Kim-Suzuki-KKS--Phase-Field-model-for-prediction-of-Materials-Properties

These files were previously stored in the code repository; they are unchanged
(byte-identical, see `MANIFEST.tsv`). Duplicate copies were stored once.

## Contents

| Path | Content |
|---|---|
| `maps/big_grid_1420/` | 1420 x 1420 grid (710 nm), 400 → 719 K, 30 000 steps; maps shown in the article |
| `maps/big_grid_1420/extra/` | a map of the same run saved with the vacancy site fraction `xv_out` |
| `maps/small_grid_735_dsc/` | 735 x 735 grid, 400 → 739 K, 70 000 steps; run used for the DSC curve |
| `maps/small_grid_735_36842/` | 735 x 735 grid, 400 → 700 K, 36 842 steps |
| `maps/small_grid_735_671K/` | 735 x 735 grid, 400 → 671 K, 30 000 steps |
| `maps/small_grid_735_708K/` | 735 x 735 grid, 400 → 708 K, 30 000 steps |
| `maps/reproducible_run/` | 735 x 735 grid, 400 → 723 K, 30 000 steps; extracted from `original/ReproducibleRun.zip` |
| `maps/nucleation_438K/` | 735 x 735 grid, isothermal 438 K with nucleation, 76 531 steps |
| `inputs/` | material tables (`final_export.npz`), initial precipitates (`micro_final.npz`, text files) |
| `figures/` | rendered microstructures (PyVista) and the supplementary animation |
| `original/ReproducibleRun.zip` | the original self-contained reproducible run (scripts, logs, maps) |

## Map format

NumPy `.npz`, one file per saved step, named
`map{ttime:.2f}s_{T_start}_{T_max:.2f}K.npz`:

* `X`, `X_al`, `X_th`, `eta` — fields of shape `(nx, ny, 1)`: Si molar fraction,
  Si fraction in the matrix and in the precipitate phase, order parameter;
* `time_out`, `T_out`, `X_al_mean_out`, `X_th_mean_out`, `eta_mean_out`,
  `surf_out` — per-step histories up to the save step;
* `ttime` — accumulated time used in the file name.

Grid spacing 0.5 nm. Load with `numpy.load`, or with `pf_kks.io.load_map` and
plot with `scripts/plot_maps.py` from the code repository.

## License

CC-BY-4.0.
