"""Figures from the archived histories (no simulation needed).

    python scripts/make_figures.py --out figures

* ``dsc.png``: simulated DSC with its contributions and the measurement
  (history of the 735 x 735, 70 000-step run, as in DSC_tool/compute_DSC.ipynb);
* ``evolution.png``: matrix Si content, precipitate fraction, interface measure
  and vacancy site fraction of the 1420 x 1420 run (Post_Process.ipynb);
* ``microstructure.png`` (with ``--maps``): Si field of the 1420 x 1420 run at
  the four temperatures of interest, from the maps of the Zenodo record
  (``maps/big_grid_1420``).
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pf_kks.io import list_maps, load_map, parse_map_filename  # noqa: E402
from pf_kks.postprocess.dsc import compute_dsc, load_experimental  # noqa: E402
from pf_kks.postprocess.histories import smoothed_histories  # noqa: E402
from pf_kks.postprocess.plots import plot_dsc, plot_field, plot_histories  # noqa: E402

HISTORIES = ROOT / "data" / "histories"


def plot_microstructures(maps_dir, path, temperatures=(507, 541, 610, 700), dx=0.5e-9):
    """Si field at the saved maps closest to ``temperatures``."""
    maps = list_maps(maps_dir)
    T_max = np.array([parse_map_filename(m)[2] for m in maps])
    fig, axes = plt.subplots(1, len(temperatures), figsize=(3.6 * len(temperatures), 3.6), layout="constrained")
    for ax, T in zip(axes, temperatures):
        i = int(np.argmin(np.abs(T_max - T)))
        im = plot_field(ax, load_map(maps[i])["X"], dx, "X", title=f"{T_max[i]:.0f} K")
        if ax is not axes[0]:
            ax.set_ylabel("")
            ax.set_yticklabels([])
    fig.colorbar(im, ax=axes, label=r"X$_{Si}$", shrink=0.85)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="figures", help="output directory")
    parser.add_argument("--maps", help="directory with the maps of the 1420 x 1420 run (Zenodo: maps/big_grid_1420)")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    dsc = compute_dsc(np.load(HISTORIES / "small_grid_735_dsc.npz"))
    exp = load_experimental(ROOT / "data" / "experimental" / "heat_capacity_dsc.csv")
    fig, ax = plt.subplots(figsize=(6, 4.5))
    plot_dsc(ax, dsc, exp)
    ax.set_xlim(400, 750)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "dsc.png"), dpi=200)

    curves = smoothed_histories(np.load(HISTORIES / "big_grid_1420_vacancies.npz"))
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    plot_histories(axes.ravel(), curves)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "evolution.png"), dpi=200)
    if args.maps:
        plot_microstructures(args.maps, os.path.join(args.out, "microstructure.png"))
    print(f"figures written to {args.out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
