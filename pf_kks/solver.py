"""Time integration of the extended KKS model.

Each step solves, semi-implicitly in Fourier space,

* the Allen-Cahn equation for the order parameter ``eta`` (with the elastic
  driving force), and
* the Cahn-Hilliard equation for the Si molar fraction ``X``,

then recovers the phase compositions from the KKS condition. Temperature
follows the prescribed heat treatment; diffusivity, interface mobility and all
tabulated properties are updated every step. An excess-vacancy model enhances
diffusion. Once the matrix reaches equilibrium, an energy-balance residual
controls the time step.

The loop reproduces ``main`` of the original ``pyFi_adapt.py`` (the version
archived in ``ReproducibleRun.zip``) operation for operation, so that published
results are reproduced to round-off. Its documented peculiarities are listed in
TECHNICAL_REPORT.md ("Implementation notes") and are intentionally preserved.
"""

import datetime
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import kinetics
from .elasticity import green_operator, solve_elasticity, stiffness_tensor
from .fft import FFT
from .free_energy import free_energy_terms, phase_compositions, total_energy
from .grid import wave_vectors
from .io import map_filename, save_map
from .materials import MaterialTables
from .microstructure import interface_area, interface_cells
from .residual import kks_residual
from .runlog import RunLog


@dataclass
class Schedule:
    """Steps at which outputs are produced."""

    n_steps: int
    map_steps: set
    surface_steps: set

    @classmethod
    def from_config(cls, cfg):
        map_steps = np.linspace(1, int(cfg.n_steps), cfg.n_map_saves, dtype=int)
        last = cfg.surface_sample_last_step or cfg.n_steps
        surface_steps = np.linspace(1, int(last), cfg.n_surface_samples, dtype=int)
        return cls(int(np.max(map_steps)), set(map_steps.tolist()), set(surface_steps.tolist()))


@dataclass
class SimulationResult:
    """Final fields and per-step histories (lists indexed by step)."""

    X: np.ndarray
    X_al: np.ndarray
    X_th: np.ndarray
    eta: np.ndarray
    sig_el: np.ndarray
    eps_el: np.ndarray
    good_step: int | None
    ttime: float
    map_files: list = field(default_factory=list)
    history: dict = field(default_factory=dict)


class AdaptiveStepError(RuntimeError):
    """Raised when the adaptive time stepping cannot find an admissible step."""


def run(cfg, eta, X, X_al, X_th, out_dir=".", base_dir=".", write_logs=True, verbose=True):
    """Run the simulation.

    Parameters
    ----------
    cfg : pf_kks.config.SimulationConfig
        Configuration.
    eta, X, X_al, X_th : ndarray, shape (nx, ny, nz)
        Initial order parameter, Si fraction and phase compositions (see
        :func:`pf_kks.microstructure.place_precipitates`).
    out_dir : str
        Directory for ``saved_maps/``, ``log.txt`` and ``dtime.txt``.
    base_dir : str
        Directory against which relative data paths of ``cfg`` are resolved.
    write_logs : bool
        Write ``log.txt`` / ``dtime.txt``.
    verbose : bool
        Print progress (residual every 1000 steps) to stdout.

    Returns
    -------
    SimulationResult
    """
    t_wall = time.time()
    log = RunLog(out_dir, enabled=write_logs)
    maps_dir = os.path.join(out_dir, "saved_maps")
    os.makedirs(maps_dir, exist_ok=True)

    tables = MaterialTables.from_npz(cfg.resolve_path(cfg.material_tables, base_dir), cfg.c_gamma, cfg.c_eps0)
    D0 = (tables.D0 if cfg.D0 is None else cfg.D0) * cfg.c_D
    QD = tables.QD if cfg.QD is None else cfg.QD
    Meta0 = cfg.Meta0 * cfg.c_Meta
    QMeta = cfg.QMeta
    Lv2 = cfg.vacancy_sink_distance**2
    lambd = cfg.lambd
    alpha = cfg.alpha
    dx = cfg.dx
    nx, ny, nz = cfg.nx, cfg.ny, cfg.nz
    time_table = np.array(cfg.schedule_time)
    temp_table = np.array(cfg.schedule_temperature)
    schedule = Schedule.from_config(cfg)

    fft = FFT((nx, ny, nz), threads=cfg.fft_threads, backend=cfg.fft_backend)

    t_s = 0  # dimensionless time
    t = 0  # physical time [s]
    t_b = 0
    ttime = 0  # accumulated time used in map file names (see implementation notes)
    flag = 0  # 1 once the adaptive scheme is armed
    good_step = None

    hist = {
        name: []
        for name in (
            "deta_mean",
            "eta_mean",
            "T",
            "dT",
            "dtime",
            "time",
            "time_s",
            "surf",
            "X_al_mean",
            "X_th_mean",
            "X_mean",
            "G_mean",
            "dX_al_mean",
            "dX_th_mean",
            "dsurf",
            "L",
        )
    }

    # Spectral operators (dimensionless grid spacing) and Green operator,
    # computed once with the stiffness at the initial temperature.
    k, k2, k4 = wave_vectors((nx, ny, nz), (dx / dx, cfg.dy / cfg.dy, cfg.dz / cfg.dz))
    T = np.interp(t, time_table, temp_table)
    C_ijkl = stiffness_tensor(tables.at(T, "C11"), tables.at(T, "C12"), tables.at(T, "C44"))
    omega = green_operator(nx, ny, nz, C_ijkl, k)

    dtime_s_ini = cfg.dtime_s
    dtime_s = cfg.dtime_s
    dtime_s_min = dtime_s_ini / cfg.dtime_min_divisor
    surf = 0
    surf_b = 0
    T_b = None
    grid_pt, cells = interface_cells(nx, ny)
    Xv = kinetics.quenched_vacancy_fraction(cfg.T_quench)

    nstep = schedule.n_steps
    energy_hist = np.zeros(nstep)
    dtime_s_hist = np.zeros(nstep)
    res_hist = {
        name: np.zeros(nstep) for name in ("total", "energy_derivative", "diffusion", "allen_cahn", "temperature")
    }
    # Last two states, needed by the residual.
    pair = {name: [] for name in ("mu", "X", "X_al", "X_th", "eta", "D", "G_cc", "G_ceta", "G")}
    num_elem = 2
    map_files = []

    for istep in range(nstep):
        iteration = 1
        while True:
            T = np.interp(t, time_table, temp_table)
            if istep + 1 in schedule.surface_steps:
                surf = interface_area(eta, grid_pt, cells, dx, lambd)

            Xve = kinetics.equilibrium_vacancy_fraction(T)
            D = kinetics.si_diffusivity(T, Xv, Xve, D0, QD, cfg.diffusion_enhancement_cap)
            Meta = kinetics.interface_mobility(T, Meta0, QMeta)
            Deff = kinetics.effective_vacancy_diffusivity(T, Xv, Xve)

            if istep == 0:
                D_0 = D
                Meta_0 = Meta
                dtime_s = dtime_s_ini

            if dtime_s > cfg.dtime_max:
                dtime_s = cfg.dtime_max
            if dtime_s < dtime_s_min:
                dtime_s = dtime_s_min

            # Physical time increment of the vacancy model and of ``ttime``.
            dtime = (dx**2 / D_0) * dtime_s_ini
            ttime = ttime + dtime
            Xv = kinetics.vacancy_relaxation(Xv, Xve, dtime, Deff, Lv2)

            A_alp = tables.at(T, "A_alp")
            B_alp = tables.at(T, "B_alp")
            C_alp = tables.at(T, "C_alp")
            C44 = tables.at(T, "C44")
            A_thet = tables.at(T, "A_thet")
            B_thet = tables.at(T, "B_thet")
            C_thet = tables.at(T, "C_thet")
            sigma = tables.at(T, "gamma")
            Omega0 = tables.at(T, "Vm")
            epsX = tables.at(T, "epsX")
            X_Si_fcc = tables.at(T, "X_Si_fcc")

            kappa2, w = kinetics.gradient_and_barrier(sigma, lambd, alpha)
            L = kinetics.allen_cahn_coefficient(sigma, kappa2, w, Meta, D, A_alp, X_Si_fcc, Omega0)

            if istep % 100 == 0:
                log.state(istep, T, D, L, Meta, Omega0, dtime_s)

            p = kinetics.normalize(
                D,
                D_0,
                Meta,
                Meta_0,
                C44,
                dx,
                sigma,
                alpha,
                A_thet,
                B_thet,
                C_thet,
                A_alp,
                B_alp,
                C_alp,
                Omega0,
                L,
                lambd,
            )
            coeffs = (p["w_s"], p["A_alp_s"], p["B_alp_s"], p["C_alp_s"], p["A_thet_s"], p["B_thet_s"], p["C_thet_s"])

            G_s, G_eta_s, G_cc_s, G_ceta_G_cc, G_ceta_s, h_eta, fa_ca_s, ft_ct_s = free_energy_terms(
                X_al, X_th, eta, *coeffs
            )
            sig_el, eps_el, el_eta, el = solve_elasticity(eta, omega, C_ijkl, epsX, fft)
            el_eta_s = el_eta / C44
            el_s = el / (C44)

            # Allen-Cahn (eta), semi-implicit.
            etak = fft.fft(eta)
            etak_old = etak
            drive_k = fft.fft(G_eta_s + el_eta_s)
            etak = (-dtime_s * p["L_s"] * drive_k + etak) / (1 + dtime_s * p["L_s"] * p["kappa2_s"] * k2)
            eta_b = np.real(fft.ifft(etak))

            # Cahn-Hilliard (X), semi-implicit.
            Xk = fft.fft(X)
            flux_k = (
                1j * k[0] * fft.fft(G_ceta_G_cc * np.real(fft.ifft(1j * k[0] * etak_old)))
                + 1j * k[1] * fft.fft(G_ceta_G_cc * np.real(fft.ifft(1j * k[1] * etak_old)))
                + 1j * k[2] * fft.fft(G_ceta_G_cc * np.real(fft.ifft(1j * k[2] * etak_old)))
            )
            Xk = (dtime_s * p["D_s"] * flux_k + Xk) / (1 + dtime_s * k2 * p["D_s"])
            X_b = np.real(fft.ifft(Xk))

            # Phase compositions at the new eta; as in the original code they are
            # evaluated from the Si field at the start of the step.
            X_al_b, X_th_b = phase_compositions(X, eta_b, A_alp, B_alp, A_thet, B_thet)

            energy_s = total_energy(eta_b, G_s, p["kappa2_s"], el_s)
            energy_hist[istep] = energy_s
            dtime_s_hist[istep] = dtime_s

            G_b_s, _, G_cc_b_s, G_ceta_G_cc, G_ceta_b_s, h_eta, fa_ca_s, ft_ct_s = free_energy_terms(
                X_al_b, X_th_b, eta_b, *coeffs
            )

            if flag == 1:
                pair["mu"].append(fa_ca_s)
                pair["X"].append(X_b)
                pair["X_al"].append(X_al_b)
                pair["X_th"].append(X_th_b)
                pair["eta"].append(eta_b)
                pair["D"].append(p["D_s"])
                pair["G_cc"].append(G_cc_b_s)
                pair["G_ceta"].append(G_ceta_b_s)
                pair["G"].append(G_b_s)
            if flag == 1 and istep >= good_step + 1:
                for name in pair:
                    pair[name] = pair[name][-num_elem:]

            if iteration == cfg.max_iterations:
                raise AdaptiveStepError(
                    f"step {istep}: no admissible time step after {iteration} attempts (dtime_s={dtime_s:.2e})"
                )

            if flag == 0 or istep < good_step + num_elem:
                break

            res = kks_residual(
                hist["time_s"],
                energy_hist,
                good_step,
                istep,
                pair["eta"],
                pair["X"],
                pair["mu"],
                pair["G"],
                p["L_s"],
                dtime_s,
            )
            if not np.isfinite(res.total):
                raise AdaptiveStepError(f"step {istep}: non-finite energy residual")
            res_hist["total"][istep] = res.total
            res_hist["energy_derivative"][istep] = res.energy_derivative
            res_hist["diffusion"][istep] = res.diffusion_term
            res_hist["allen_cahn"][istep] = res.allen_cahn_term
            res_hist["temperature"][istep] = res.temperature_term

            if verbose and istep % 1000 == 0:
                print(
                    "step, |RES-KKS|  , energy_term , gradient_term, eta_term, temp_term : ",
                    istep,
                    "{0:.2e}".format(np.abs(res.total)),
                    "{0:.2e}".format(res.energy_derivative),
                    "{0:.2e}".format(res.diffusion_term),
                    "{0:.2e}".format(res.allen_cahn_term),
                    "{0:.2e}".format(res.temperature_term),
                )
            if verbose and istep == good_step + num_elem and iteration == 1:
                print(
                    "resmax= ",
                    "{0:.2e}".format(cfg.residual_max),
                    " resmin= ",
                    "{0:.2e}".format(cfg.residual_min),
                    "\n",
                )
                print("good step: ", good_step, "\n")

            if np.abs(res.total) > cfg.residual_max:
                dtime_s /= cfg.step_factor
                log.rejected(istep, iteration, dtime_s, res, energy_s)
                iteration += 1
                continue
            elif np.abs(res.total) > cfg.residual_min:
                log.accepted(istep, res, energy_s)
                break
            else:
                log.enlarged(istep, res, energy_s)
                dtime_s *= cfg.step_factor
                break

        if flag == 1:
            # Bookkeeping of the original code: the free-energy entries are
            # overwritten with the values evaluated before the update.
            pair["X"][-1] = X_b
            pair["X_al"][-1] = X_al_b
            pair["X_th"][-1] = X_th_b
            pair["eta"][-1] = eta_b
            pair["D"][-1] = p["D_s"]
            pair["G_cc"][-1] = G_cc_s
            pair["G_ceta"][-1] = G_ceta_s
            pair["G"][-1] = G_s
            energy_hist[istep] = energy_s
            dtime_s_hist[istep] = dtime_s

        t_s += dtime_s
        dtime = (dx**2 / D_0) * dtime_s
        t_b += (dx**2 / D_0) * dtime_s

        if istep + 1 in schedule.surface_steps:
            T_b = np.interp(t_b, time_table, temp_table)
            surf_b = interface_area(eta_b, grid_pt, cells, dx, lambd)

        X_al_mean = np.mean(X[eta < 0.05])
        if X_al_mean <= cfg.x_equilibrium and flag == 0 and istep > 1:
            good_step = istep + cfg.good_step_offset
            flag = 1

        hist["X_al_mean"].append(X_al_mean)
        hist["X_th_mean"].append(np.mean(X[eta > 0.95]))
        hist["X_mean"].append(np.mean(X[(eta > 0.05) & (eta > 0.95)]))
        hist["dX_al_mean"].append(np.mean(X_b[eta_b < 0.05]) - np.mean(X[eta < 0.05]))
        hist["dX_th_mean"].append(np.mean(X_b[eta_b > 0.95]) - np.mean(X[eta > 0.95]))
        hist["deta_mean"].append(np.mean(eta_b) - np.mean(eta))
        hist["dT"].append(T_b - T)
        hist["eta_mean"].append(np.mean(eta))
        hist["G_mean"].append(np.mean(G_s.reshape(nx, ny)))
        hist["surf"].append(surf_b)
        hist["dsurf"].append(surf_b - surf)
        hist["T"].append(T)
        hist["dtime"].append(dtime)
        hist["time"].append(t)
        hist["time_s"].append(t_s)
        hist["L"].append(L)

        eta, X, X_al, X_th = eta_b, X_b, X_al_b, X_th_b
        t = t_b

        if istep + 1 in schedule.map_steps:
            name = map_filename(ttime, cfg.T_start, np.asarray(hist["T"]).max())
            path = os.path.join(maps_dir, name)
            save_map(path, ttime, hist, X, X_al, X_th, eta)
            map_files.append(path + ".npz")

    if verbose:
        print("Temperature reached : %s K ---" % (np.asarray(hist["T"]).max()))
        print("simulation time : " + str(datetime.timedelta(seconds=time.time() - t_wall)))

    hist["energy_s"] = energy_hist
    hist["dtime_s"] = dtime_s_hist
    for name, values in res_hist.items():
        hist["residual_" + name] = values
    return SimulationResult(X, X_al, X_th, eta, sig_el, eps_el, good_step, ttime, map_files, hist)
