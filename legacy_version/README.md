# Legacy Version — Original Published Code

This directory preserves the **original code** that accompanied the article, for
lineage and reference:

> Fetni, Delahaye, Sepúlveda, Duchêne, Habraken, Mertens, *Extension of a
> phase-field KKS model to predict the microstructure evolution in LPBF AlSi10Mg
> alloy submitted to non isothermal processes*, Computational Materials Science
> **244** (2024) 113197. [DOI](https://doi.org/10.1016/j.commatsci.2024.113197)

The original top-level README is kept as [`README_original.md`](README_original.md).

## Code only

Files are byte-identical to the original repository (commit `09d2ac1`), except
that notebook outputs were cleared. Simulation maps, figures and byte-code caches
were removed; maps and figures are in the Zenodo record (see [`../DATA.md`](../DATA.md)).
[`../provenance/ORIGINAL_INVENTORY.tsv`](../provenance/ORIGINAL_INVENTORY.tsv)
records where every original file went.

The code is the historical record and is not maintained. The current
implementation is the `pf_kks` package (`../pf_kks/`). The tests in `../tests/`
import the files below and check that `pf_kks` reproduces them exactly.

## Contents

| Path | Content | Current equivalent |
|---|---|---|
| `ReproducibleRun/pyFi_adapt.py` | solver of the archived reproducible run (**reference version**) | `pf_kks/solver.py` and modules |
| `ReproducibleRun/main_RVE_cell.py` | run script of the reproducible run | `configs/reproducible_run.toml`, `pf-kks-run` |
| `ReproducibleRun/pyFi_2D.slurm.sh`, `slurm-job.4219077.out` | SLURM script and job output (NIC5, Nov 2022) | — |
| `pyFi_adapt.py` | top-level solver: the `save_ipynb/` copy plus the nucleation functions | `pf_kks/nucleation.py` |
| `pyFi_RVE_cell.py` | script export of the simulation notebook (contains a syntax error at line 408) | `pf-kks-run` |
| `pyFi_origin_RVE_cell.ipynb` | main simulation notebook (setup, run, PyVista plots) | `pf-kks-run`, `scripts/plot_maps.py` |
| `Post_Process.ipynb` | maps and evolution curves | `scripts/plot_maps.py`, `scripts/make_figures.py` |
| `save_ipynb/` | another copy of the solver (smallest step `dtime_s/20` instead of `/1000`), run script and notebook | `dtime_min_divisor` |
| `DSC_tool/free_nrj_mob.py` | CALPHAD thermodynamics of Al-Si | `pf_kks/thermo.py` |
| `DSC_tool/compute_DSC.ipynb` | simulated DSC curve | `pf_kks/postprocess/dsc.py` |
| `DSC_tool/3d02a_dsc_mean.py`, `3f01a_2dmap.py` | thesis figure scripts (need a `jupyter_style` module, absolute paths and input files that were not in the repository) | — |

Apart from the nucleation functions and the smallest time step, the three solver
copies differ only in line endings, comments and blank lines.
See the [Technical Report](../TECHNICAL_REPORT.md#implementation-notes) for the
behaviours of the original code that `pf_kks` preserves.
