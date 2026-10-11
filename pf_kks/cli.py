"""Command-line entry point: ``pf-kks-run`` / ``python scripts/run_simulation.py``."""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import solver
from .config import load_config
from .microstructure import load_precipitates, place_precipitates

DESCRIPTION = """Run a phase-field simulation from a TOML configuration.

Relative data paths in the configuration are resolved against the directory of
the configuration file.

Outputs (in --out): saved_maps/map*.npz snapshots, history.npz (per-step
histories and residuals), log.txt, dtime.txt and config.json (the resolved
configuration).
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=DESCRIPTION, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="TOML configuration file")
    parser.add_argument("--out", default="outputs/run", help="output directory (default: outputs/run)")
    parser.add_argument("--n-steps", type=int, help="override the number of steps")
    parser.add_argument("--fft-threads", type=int, help="override the number of FFT threads")
    parser.add_argument("--print-config", action="store_true", help="print the resolved configuration and exit")
    parser.add_argument("--quiet", action="store_true", help="do not print progress")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, n_steps=args.n_steps, fft_threads=args.fft_threads)
    base_dir = str(Path(args.config).resolve().parent)
    if args.print_config:
        print(json.dumps(cfg.to_dict(), indent=2))
        return 0

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    ox, oy, oz, r = load_precipitates(cfg.resolve_path(cfg.precipitates, base_dir))
    state = place_precipitates(
        cfg.nx, cfg.ny, cfg.nz, cfg.dx, cfg.dy, cfg.dz, cfg.x_matrix, cfg.x_precipitate, r, ox, oy, oz
    )
    result = solver.run(cfg, *state, out_dir=args.out, base_dir=base_dir, verbose=not args.quiet)
    np.savez_compressed(
        os.path.join(args.out, "history.npz"),
        good_step=-1 if result.good_step is None else result.good_step,
        ttime=result.ttime,
        **{k: np.asarray(v) for k, v in result.history.items()},
    )
    return 0
