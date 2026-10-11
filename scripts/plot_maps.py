"""Images of the X_Si and eta fields of every map in a directory.

    python scripts/plot_maps.py path/to/maps --out figures/maps [--dx 0.5e-9]

Maps of the archived runs are in the Zenodo record (see DATA.md); new runs
write them to ``<out>/saved_maps``. Maps are ordered by maximum temperature.
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf_kks.io import list_maps, load_map, parse_map_filename  # noqa: E402
from pf_kks.postprocess.plots import plot_field  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("maps", help="directory containing map*.npz files")
    parser.add_argument("--out", default="figures/maps")
    parser.add_argument("--dx", type=float, default=0.5e-9, help="grid spacing [m]")
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    for path in list_maps(args.maps):
        _, _, T_max = parse_map_filename(path)
        m = load_map(path)
        for key, label in (("X", r"X$_{Si}$"), ("eta", r"$\eta$")):
            fig, ax = plt.subplots(figsize=(5, 4.2))
            im = plot_field(ax, m[key], args.dx, key, title=f"{label} at {T_max:.0f} K")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            name = f"{key}_{os.path.basename(path)[:-4]}.png"
            fig.savefig(os.path.join(args.out, name), dpi=150)
            plt.close(fig)
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
