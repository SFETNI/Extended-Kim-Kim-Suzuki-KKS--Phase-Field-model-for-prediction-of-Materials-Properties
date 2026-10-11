"""Helpers to run the original (legacy) solver side by side with pf_kks."""

import contextlib
import importlib.util
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEGACY_SOLVER = ROOT / "legacy_version" / "ReproducibleRun" / "pyFi_adapt.py"
LEGACY_THERMO = ROOT / "legacy_version" / "DSC_tool" / "free_nrj_mob.py"
LEGACY_TOPLEVEL = ROOT / "legacy_version" / "pyFi_adapt.py"
TABLES = ROOT / "data" / "inputs" / "final_export.npz"


def load_module(path, name, replacements=()):
    """Import a legacy file as a module, optionally after text replacements
    (used to change hard-coded residual thresholds in tests)."""
    source = Path(path).read_text(encoding="utf-8")
    for old, new in replacements:
        assert old in source, old
        source = source.replace(old, new)
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path)
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module


@contextlib.contextmanager
def working_directory(path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def run_legacy(cfg, eta, X, X_al, X_th, workdir, residual_max=None, residual_min=None):
    """Run the legacy ``main`` with the parameters of ``cfg``.

    Returns the legacy output tuple, or the exception raised (the legacy code
    calls ``sys.exit`` when the adaptive scheme fails).
    """
    replacements = []
    if residual_max is not None:
        replacements.append(("resmax=1e-1 # trials", f"resmax={residual_max!r} # trials"))
    if residual_min is not None:
        replacements.append(("resmin= 1e-4", f"resmin= {residual_min!r}"))
    legacy = load_module(LEGACY_SOLVER, "legacy_pyfi_adapt", replacements)
    import matplotlib

    matplotlib.use("Agg")
    d = np.load(cfg.material_tables)
    nstep = int(cfg.n_steps)
    ask_step = np.linspace(1, nstep, cfg.n_map_saves, dtype=int)
    ask_step2 = np.linspace(1, int(cfg.surface_sample_last_step or nstep), cfg.n_surface_samples, dtype=int)
    dx = cfg.dx
    os.makedirs(os.path.join(workdir, "saved_maps"), exist_ok=True)
    with working_directory(workdir):
        try:
            return legacy.main(
                cfg.T_start,
                cfg.x_equilibrium,
                eta,
                X,
                X_al,
                X_th,
                cfg.vacancy_sink_distance**2,
                cfg.diffusion_enhancement_cap,
                cfg.nx,
                cfg.ny,
                cfg.nz,
                dx,
                1 * dx,
                dx,
                d["T"],
                d["delta"] * cfg.c_eps0,
                d["D_inter_max"],
                d["A_dia"],
                d["B_dia"],
                d["C_dia"],
                d["A_fcc"],
                d["B_fcc"],
                d["C_fcc"],
                d["C11"],
                d["C12"],
                d["C44"],
                d["L_max"],
                d["gamma"] * cfg.c_gamma,
                cfg.alpha,
                None,
                None,
                None,
                None,
                np.array(cfg.schedule_time),
                np.array(cfg.schedule_temperature),
                cfg.x_matrix,
                cfg.x_precipitate,
                d["Vm"],
                0,
                cfg.dtime_s,
                np.max(ask_step),
                ask_step,
                ask_step2,
                d["D0"] * cfg.c_D,
                d["QD"],
                cfg.Meta0 * cfg.c_Meta,
                cfg.QMeta,
                d["X_Si_fcc"],
                cfg.half_width_cells * dx,
                cfg.dtime_max,
                "precip_si",
                cfg.good_step_offset,
            )
        except (SystemExit, Exception) as exc:  # legacy aborts with sys.exit
            return exc


LEGACY_OUTPUTS = (
    "X",
    "X_al",
    "X_th",
    "eta",
    "sig_el",
    "eps_el",
    "eta_mean",
    "deta_mean",
    "X_mean",
    "X_al_mean",
    "dX_al_mean",
    "X_th_mean",
    "dX_th_mean",
    "dT",
    "T",
    "dtime",
    "time",
    "surf",
    "dsurf",
    "el_eta_s",
    "el",
    "G_eta_s",
    "G_ceta_G_cc",
    "L",
    "energy_s",
    "dtime_s",
    "good_step",
    "ttime",
    "residual_total",
    "residual_energy_derivative",
    "residual_diffusion",
    "residual_allen_cahn",
    "residual_temperature",
    "time_res",
)


def legacy_dict(out):
    return dict(zip(LEGACY_OUTPUTS, out))
