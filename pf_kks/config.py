"""Simulation configuration (TOML)."""

import os
from dataclasses import asdict, dataclass, field, fields

try:
    import tomllib
except ImportError:  # Python < 3.11
    import tomli as tomllib


@dataclass
class SimulationConfig:
    # Grid (2-D: nz = 1). dy = dz = dx.
    nx: int = 735
    ny: int = 735
    nz: int = 1
    dx: float = 0.5e-9  # [m]

    # Heat treatment: piecewise-linear temperature [K] vs time [s].
    schedule_time: list = field(default_factory=lambda: [0, 1350])
    schedule_temperature: list = field(default_factory=lambda: [400, 850])

    # Initial microstructure.
    precipitates: str = "../data/inputs/micro_final.npz"  # relative to the config file
    x_matrix: float = 0.025  # initial Si fraction of the matrix
    x_precipitate: float = 0.99999  # Si fraction of the precipitates
    x_equilibrium: float = 0.025  # matrix Si fraction that activates the adaptive scheme

    # Material data and calibration factors.
    material_tables: str = "../data/inputs/final_export.npz"  # relative to the config file
    c_gamma: float = 1.0  # interface-energy factor
    c_eps0: float = 0.25  # misfit-strain factor
    c_D: float = 1.0  # diffusivity factor
    c_Meta: float = 0.6  # interface-mobility factor
    D0: float | None = None  # [m^2 s^-1]; None = value stored in the tables
    QD: float | None = None  # [J mol^-1]; None = value stored in the tables
    Meta0: float = 5.60e-8  # interface-mobility pre-factor
    QMeta: float = 1.10e5  # interface-mobility activation energy [J mol^-1]

    # Interface.
    alpha: float = 2.2  # interface-width constant
    half_width_cells: float = 2.5  # half interface thickness lambda / dx

    # Excess vacancies.
    T_quench: float = 850  # temperature at which vacancies are frozen in [K]
    vacancy_sink_distance: float = 250e-9  # vacancy mean free path [m]
    diffusion_enhancement_cap: float = 1e9  # cap on Xv / Xve

    # Time stepping (dimensionless time, unit dx^2 / D(t=0)).
    dtime_s: float = 75  # initial step
    dtime_max: float = 400  # largest step
    dtime_min_divisor: float = 1000  # smallest step = dtime_s / divisor
    residual_max: float = 1e-1  # reject the step above this residual
    residual_min: float = 1e-4  # enlarge the step below this residual
    step_factor: float = 1.25  # step reduction / enlargement factor
    max_iterations: int = 50  # attempts per step before aborting
    good_step_offset: int = 0  # steps between matrix equilibrium and adaptive scheme

    # Run length and outputs.
    n_steps: int = 30000
    n_map_saves: int = 40  # full-field maps saved at evenly spaced steps
    n_surface_samples: int = 500  # steps at which the interface area is measured
    surface_sample_last_step: int | None = None  # None = n_steps

    # FFT.
    fft_backend: str = "auto"  # "pyfftw", "numpy" or "auto"
    fft_threads: int = 12

    @property
    def dy(self):
        return 1 * self.dx

    @property
    def dz(self):
        return self.dx

    @property
    def lambd(self):
        return self.half_width_cells * self.dx

    @property
    def T_start(self):
        return self.schedule_temperature[0]

    def resolve_path(self, path, base):
        return path if os.path.isabs(path) else os.path.join(base, path)

    def to_dict(self):
        return asdict(self)


def load_config(path, **overrides):
    """Read a TOML configuration; keyword arguments override file values."""
    with open(path, "rb") as f:
        data = tomllib.load(f)
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(value)
        else:
            flat[key] = value
    flat.update({k: v for k, v in overrides.items() if v is not None})
    known = {f.name for f in fields(SimulationConfig)}
    unknown = set(flat) - known
    if unknown:
        raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
    return SimulationConfig(**flat)
