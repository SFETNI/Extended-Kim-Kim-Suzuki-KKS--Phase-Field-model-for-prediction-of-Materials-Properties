"""Extract the per-step histories of every archived run into small files.

One-off provenance script: reads a clone of the original repository (with
``ReproducibleRun.zip`` extracted next to it) and writes
``data/histories/<run>.npz`` (all map keys except the full fields) and
``data/histories/big_grid_1420_vacancies.npz``.

    python scripts/extract_histories.py /path/to/original-repo /path/to/extracted/zip-parent
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from provenance.runs import RUNS, VACANCY_MAP  # noqa: E402

FIELDS = ("X", "X_al", "X_th", "eta")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(src, dst):
    with np.load(src) as d:
        data = {k: d[k] for k in d.files if k not in FIELDS}
        data["grid_shape"] = np.array(d["eta"].shape)
    data["source_file"] = np.array(src.name)
    data["source_sha256"] = np.array(sha256(src))
    np.savez_compressed(dst, **data)
    print(f"{dst.relative_to(ROOT)}  <-  {src}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original", type=Path, help="clone of the original repository")
    parser.add_argument("zip_parent", type=Path, help="directory containing the extracted ReproducibleRun/")
    args = parser.parse_args(argv)
    out = ROOT / "data" / "histories"
    out.mkdir(parents=True, exist_ok=True)
    for name, run in RUNS.items():
        base = args.zip_parent if run["final"].startswith("ReproducibleRun/") else args.original
        extract(base / run["final"], out / f"{name}.npz")
    extract(args.original / VACANCY_MAP, out / "big_grid_1420_vacancies.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
