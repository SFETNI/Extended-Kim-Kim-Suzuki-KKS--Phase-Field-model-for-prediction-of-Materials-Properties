# Data

Simulation inputs and light-weight results live in this repository. The
full-field simulation maps (2.3 GB) are in a Zenodo record. Nothing was
re-simulated: every file is either a byte-identical copy of an original file or
extracted from one, with its source and checksum recorded.

## In this repository

| Path | Content | Source |
|---|---|---|
| `data/inputs/final_export.npz` | temperature-dependent material tables (free-energy parabolas, elastic constants, misfit, molar volume, interface energy, equilibrium composition; 400–850 K) | original `final_export.npz` |
| `data/inputs/micro_final.npz` | initial microstructure: centres and radii of the 73 precipitates (`ox_cyl`, `oy_cyl`, `oz_cyl`, `r_cyl`, metres) and candidate nucleation sites (`idx_nuc`) | original `micro_final.npz` |
| `data/inputs/precipitates_xyz_r.txt` | the same 73 precipitates as text (`x y z r`) | original `xyz_r.txt` |
| `data/inputs/precipitates_xyz_r_big_cell.txt`, `cyl_packing.npz` | alternative precipitate sets of the original notebooks | original files |
| `data/histories/<run>.npz` | per-step histories of each archived run (below): all keys of its last map except the fields, plus `grid_shape`, `source_file`, `source_sha256` | extracted by `scripts/extract_histories.py` |
| `data/histories/big_grid_1420_vacancies.npz` | histories of the 1420 × 1420 run with the vacancy site fraction `xv_out` (step 19 231) | top-level `map4754.92s_400_700.29K.npz` |
| `data/experimental/heat_capacity_dsc.csv` | measured DSC: temperature [K], apparent heat capacity [J kg⁻¹ K⁻¹] | original `DSC_tool/heat capacity_data.csv` |
| `data/reference/reproducible_run/` | first-step map and run log (`dtime.txt`) of the reproducible run, used by the tests | `ReproducibleRun.zip` |

## Archived runs

| Run (`data/histories/`, Zenodo `maps/`) | Grid | Heating | Steps | Notes |
|---|---|---|---|---|
| `big_grid_1420` | 1420 × 1420 (710 nm) | 400 → 719 K | 30 000 | maps shown in the article; 36 maps |
| `small_grid_735_dsc` | 735 × 735 | 400 → 739 K | 70 000 | used by `compute_DSC.ipynb` for the DSC curve |
| `small_grid_735_36842` | 735 × 735 | 400 → 700 K | 36 842 | |
| `small_grid_735_671K` | 735 × 735 | 400 → 671 K | 30 000 | |
| `small_grid_735_708K` | 735 × 735 | 400 → 708 K | 30 000 | |
| `reproducible_run` | 735 × 735 | 400 → 723 K | 30 000 | complete script and logs archived: `configs/reproducible_run.toml` |
| `nucleation_438K` | 735 × 735 | isothermal 438 K | 76 531 | with nucleation |

Maps of the same run share their histories (each map holds the histories up to
its save step); runs were identified this way. Several maps were stored two or
three times in the original repository; they are stored once in the record.

## Zenodo record

DOI: [10.5281/zenodo.23300608](https://doi.org/10.5281/zenodo.23300608)

The record is assembled from a clone of the original repository by

```bash
python scripts/make_zenodo_bundle.py /path/to/original-repo --out zenodo_bundle
```

and contains:

```text
README.md                    description of the record (provenance/ZENODO_README.md)
MANIFEST.tsv                 path, size, sha256 and original path(s) of every file
maps/<run>/map*.npz          full-field maps of the seven runs (2.2 GB)
maps/big_grid_1420/extra/    the map with xv_out
inputs/                      material tables and initial microstructures
figures/                     PyVista renderings of X_Si and eta, supplementary animation
original/ReproducibleRun.zip the original self-contained reproducible run
```

To verify a download: `tail -n +2 MANIFEST.tsv | awk -F'\t' '{print $3"  "$1}' | sha256sum -c`.

## Map format

One `.npz` per saved step, named `map{ttime:.2f}s_{T_start}_{T_max:.2f}K.npz`
(`ttime` is a bookkeeping time, not the physical time; see the
[Technical Report](TECHNICAL_REPORT.md#implementation-notes), note 7):

| Key | Shape | Content |
|---|---|---|
| `X`, `X_al`, `X_th`, `eta` | `(nx, ny, 1)` | Si fraction; in the matrix; in the precipitate phase; order parameter |
| `time_out`, `T_out` | `(n,)` | physical time [s] and temperature [K] at each step |
| `X_al_mean_out` | `(n,)` | mean Si fraction of the matrix (`η < 0.05`) |
| `X_th_mean_out` | `(n,)` | mean Si fraction of the precipitates (`η > 0.95`) |
| `eta_mean_out` | `(n,)` | mean `η` (precipitate fraction) |
| `surf_out` | `(n,)` | interface measure [m] |
| `ttime` | `()` | see above |

```python
from pf_kks.io import list_maps, load_map
maps = list_maps("maps/big_grid_1420")        # sorted by temperature
m = load_map(maps[-1])                        # dict of arrays
h = load_map(maps[-1], fields=False)          # histories only (fast)
```

## Provenance

[`provenance/ORIGINAL_INVENTORY.tsv`](provenance/ORIGINAL_INVENTORY.tsv) lists
all 162 files of the original repository (commit `09d2ac1`) with size, sha256 and
destination: `repo:` (this repository), `record:` (Zenodo) or `dropped:` (5
files: byte-code caches and an empty log).
[`provenance/runs.py`](provenance/runs.py) defines the runs.
