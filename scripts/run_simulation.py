"""Run a phase-field simulation from a TOML configuration.

    python scripts/run_simulation.py configs/reproducible_run.toml --print-config
    python scripts/run_simulation.py configs/demo_small.toml --out outputs/demo

Equivalent to the ``pf-kks-run`` command installed with the package.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pf_kks.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
