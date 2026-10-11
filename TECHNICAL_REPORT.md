# Extended KKS Phase-Field Model — Technical Report

This report documents the governing equations, the numerical scheme, the
parameters of the archived runs and the implementation details of `pf_kks`. It is
the technical companion to the [README](README.md); the physical background and
the results are in the article (Fetni et al., Comput. Mater. Sci. 244 (2024)
113197) and in J. Delahaye's PhD thesis (ULiège, 2022).

Notation follows the code: `η` order parameter, `X` Si molar fraction, `X_al`
(`c_α`) and `X_th` (`c_θ`) Si fractions of the Al matrix (FCC, α) and of the Si
precipitates (diamond, θ).

## Free energy

The total free energy of the 2-D periodic cell is

```text
F = ∫ [ (1 − h(η)) f_α(c_α) + h(η) f_θ(c_θ) + w g(η) + κ²/2 |∇η|² + e_el ] dV

h(η) = 3η² − 2η³            g(η) = η²(1 − η)²
f_α(c) = A_α c² + B_α c + C_α        f_θ(c) = A_θ c² + B_θ c + C_θ
```

The parabolic coefficients `A, B, C` of both phases, the elastic constants
`C11, C12, C44` of the matrix, the misfit `δ`, the molar volume `V_m`, the
interface energy `σ` and the equilibrium matrix composition `X_Si^eq` are
tabulated at 400, 490, …, 850 K in `data/inputs/final_export.npz` (fitted to the
COST 507 CALPHAD description, see `pf_kks/thermo.py`). They are interpolated
linearly in temperature at every step. Calibration factors scale the interface
energy (`c_gamma`) and the misfit (`c_eps0`).

Interface parameters, with half width `λ = 2.5 dx` and `α = 2.2`:

```text
κ² = 6 λ σ / α        w = 3 α σ / λ
```

**KKS condition.** At each point the phase compositions have equal diffusion
potentials, `f_α'(c_α) = f_θ'(c_θ)`, and satisfy `X = (1 − h) c_α + h c_θ`. With
parabolic free energies this has a closed form (`free_energy.phase_compositions`):

```text
c_α = (X − h (B_α − B_θ)/(2 A_θ)) / ((1 − h) + h A_α/A_θ)
c_θ = (A_α/A_θ) c_α + (B_α − B_θ)/(2 A_θ)
```

## Evolution equations

```text
∂η/∂t = −L [ ∂f/∂η + ∂e_el/∂η − κ² ∇²η ]                          (Allen-Cahn)
∂X/∂t = ∇·( D [ ∇X + h'(η) (c_α − c_θ) ∇η ] )                      (Cahn-Hilliard, KKS form)

∂f/∂η = −h'(η) [ f_α − f_θ − (c_α − c_θ) f_α' ] + w g'(η)
```

Both are solved in Fourier space with a semi-implicit first-order scheme (the
gradient and diffusion operators implicit, the rest explicit). With `k` the wave
vector and `^` the FFT:

```text
η̂ⁿ⁺¹ = ( η̂ⁿ − Δt L (∂f/∂η + ∂e_el/∂η)^ ) / (1 + Δt L κ² k²)
X̂ⁿ⁺¹ = ( X̂ⁿ + Δt D i k · [ h'(c_α − c_θ) F⁻¹(i k η̂ⁿ) ]^ ) / (1 + Δt D k²)
```

For an even number of points the Nyquist wave number is set to zero.

## Micro-elasticity

The precipitate carries a dilatational eigenstrain `ε*_ij = ε_X η² δ_ij` with
`ε_X = c_eps0 δ(T)`. The elastic field follows Khachaturyan's homogeneous-modulus
solution: one iteration in Fourier space from zero total strain, with the Green
operator `Ω(k)` built from the cubic stiffness tensor (`elasticity.py`). The
elastic energy density `e_el = ½ σ_ij ε^el_ij` and its derivative with respect to
`η` enter the Allen-Cahn equation (normalised by `C44`).

## Kinetics

**Excess vacancies.** Fast LPBF solidification freezes in the vacancy site
fraction of `T_quench = 850 K`. During heating it relaxes towards equilibrium:

```text
X_v^eq(T) = exp(−(H_f − T S_f)/RT) · (1 − 12·0.01 + 12·0.01 exp((H_b − T S_b)/RT))
X_v ← X_v + (1 − exp(−Δt D_eff / L_v²)) (X_v^eq − X_v)
D_eff = (X_v / X_v^eq) (0.03 D_Si + 0.97 D_Al)
```

with `L_v = 250 nm` the vacancy mean free path to sinks. Constants are in
`pf_kks/constants.py`.

**Si diffusivity**, enhanced by the vacancy supersaturation (factor ignored when
below 1, capped at `1e9`):

```text
D = (X_v / X_v^eq) D_0 exp(−Q_D / RT)
```

**Interface kinetics.** Interface mobility `M_η = M_0 exp(−Q_M / RT)`. The
Allen-Cahn coefficient includes the thin-interface correction of Kim, Kim and
Suzuki:

```text
L = 0.01 · (σ/κ²) · [ 1/M_η + κ ξ / (D √(2w)) ]⁻¹
ξ = (19/30) (1/V_m) · 2 A_α (X_Si^eq − 0.9999)²
```

## Non-dimensionalisation

Lengths in `dx`, energy densities in `C44` (molar quantities in `C44 V_m`), time
in `dx² / D(t = 0)`. The time step `dtime_s` in the configuration is in these
units.

## Heat treatment and adaptive time stepping

Temperature is interpolated linearly in a time table (400 → 850 K in 1350 s,
i.e. 20 K/min). The dimensionless step starts at `dtime_s = 75`, bounded by
`[dtime_s / 1000, 400]`.

The adaptive scheme is armed once the mean Si fraction of the matrix
(`η < 0.05`) drops to `x_equilibrium` (step `good_step`). From then on, every
step is checked with an energy-balance residual (`residual.py`):

```text
R = dE/dt − Σ (dX/dt) f_α'(c_α) + Σ (1/L) (dη/dt)² − Σ dG/dt
```

* `|R| > 0.1`: the step is rejected and repeated with `Δt / 1.25`, at most 50
  times, after which the run stops;
* `1e-4 < |R| ≤ 0.1`: the step is accepted;
* `|R| ≤ 1e-4`: the step is accepted and the next one enlarged by 1.25.

The archived 30 000-step reproducible run armed the scheme at step 4866 and
recorded 25 127 accepted, 19 rejected and 5 enlarged steps.

## Outputs

* **Maps** (`saved_maps/map*.npz`), saved at `n_map_saves` evenly spaced steps:
  the fields `X`, `X_al`, `X_th`, `eta` and the histories up to that step. They
  can seed a new run, but they are not exact restart points: they do not store
  the vacancy site fraction, the current adaptive step, the dimensionless time
  or `good_step`, so a run started from a map does not continue the original
  trajectory. Neither the original code nor `pf_kks` has a restart option.
* **Histories**: per step, time and temperature, mean matrix composition
  `X_al_mean_out` (`η < 0.05`), mean precipitate composition (`η > 0.95`), mean
  `η` (precipitate fraction) and the interface measure `surf_out`.
* **Interface measure**: number of grid cells whose four corners lie in the
  diffuse interface (`0.1 ≤ η ≤ 0.9`), times `dx² / (2λ)`. It is updated at
  `n_surface_samples` evenly spaced steps and held in between.

## DSC post-processing

`pf_kks.postprocess.dsc` (from `DSC_tool/compute_DSC.ipynb`) sub-samples the
histories (1 point in 90), fits cubic smoothing splines to `⟨η⟩(t)`,
`⟨c_α⟩(t)` and `surf(t)`, and builds the heat flow per unit mass from three
contributions:

```text
background    = Ṫ [ η c_p,dia / M_Si + (1 − η) c_p,fcc / M_Al ]
precipitation = (1 − η) (∂H_fcc/∂X_Si) ċ_α / M_Al + η̇ (H_dia / M_Si − H_fcc / M_Al)
surface       = γ_H (d surf/dt) / (A_sim ρ)
C_p* = (background + precipitation + surface) / Ṫ
```

Enthalpies and heat capacities come from the CALPHAD functions
(`pf_kks/thermo.py`), molar volumes from `molar_volume`, and the density from the
local phase fractions. Parameters (`Ṫ = 20 K/min`, `γ_H = 0.5 J m⁻²`,
`A_sim = (710 nm)²/4`, the spline factors) are those of the original notebook;
see `DSCParameters`.

## Parameters of the archived runs

Only the reproducible run (`ReproducibleRun.zip`) archived its complete script;
its parameters are in `configs/reproducible_run.toml` and are checked by the
tests:

| Parameter | Value |
|---|---|
| grid | 735 × 735, `dx = 0.5 nm` |
| heating | 400 → 850 K, 20 K/min |
| initial microstructure | 73 precipitates of `micro_final.npz`, `X = 0.99999` in, `0.025` out |
| `c_gamma`, `c_eps0`, `c_D`, `c_Meta` | 1, 0.25, 1, 0.6 |
| `D_0`, `Q_D` | from the tables: 2.717e-5 m²/s, 118.18 kJ/mol |
| `M_0`, `Q_M` | 5.60e-8, 110 kJ/mol |
| `dtime_s`, max, min | 75, 400, 75/1000 |
| steps | 30 000 (reached 722.6 K) |

The other archived runs (in particular the 1420 × 1420 run used for the maps of
the article and the 70 000-step 735 × 735 run used for the DSC curve) are
documented by their histories and maps only. Their full parameter sets were not
archived in the original repository.

**Relation to the article.** The reproducible run is not the parameter set of
Table 2 of the article (`Δt* = 250`, `iter_max = 15`, `f_γ = 2.15`,
`f_ε0 = 0.01`, `f_Mη = f_D = 1`, `f_L = 0.01`). It differs in particular in the
initial step (75), the iteration limit (50), the interface-energy factor (1)
and the misfit factor (0.25), and its order parameter stays below 0.95 from
step 4118 on (`X_th_mean_out` is then undefined, NaN). Likewise, the DSC curve
computed here from the 70 000-step run reproduces the curve saved in
`compute_DSC.ipynb` but not Fig. 8 of the article, whose amplitudes differ
(the article quotes `γ_H = 1.5 J m⁻²`, the notebook uses 0.5; changing `γ_H`
alone does not recover Fig. 8). The run behind Fig. 8 is not identified among
the archived runs.

## Implementation notes

`pf_kks` reproduces the original solver operation for operation, so that the
archived results remain reproducible. The following behaviours of the original
code are **kept on purpose** and documented here. Changing any of them would
change the results:

1. **Phase compositions use the Si field of the start of the step.** After the
   update, `c_α`, `c_θ` are computed from the new `η` but the previous `X`.
2. **Elastic stiffness frozen.** The stiffness tensor and Green operator are
   built at the initial temperature. Only the normalisation by `C44(T)` follows
   temperature (the original code comments on this).
3. **Single elasticity iteration.** The elastic solution is one Khachaturyan
   iteration from zero total strain.
4. **Energy in the residual.** `total_energy` broadcasts the chemical and
   elastic densities over the two components of `np.gradient(η)`, so they enter
   the energy twice. The residual thresholds were tuned with this definition.
5. **Residual after a rejected step.** The residual uses the last two states
   computed, so after a rejection the "previous" state is the rejected attempt.
   After the step loop, the free-energy entries of this pair are replaced by the
   values evaluated before the update.
6. **Time after an enlarged step.** When `|R| < 1e-4`, the step is enlarged
   *after* the solve, and the clock (`t`, `t_s`) advances by the enlarged step.
   The recorded `dtime_s` history is likewise the enlarged value, which can
   exceed `dtime_max` by the factor 1.25 (it is clamped at the next step). This
   happened 5 times in the 30 000-step reproducible run.
7. **`ttime` is not the physical time.** `ttime`, used in map file names
   (`map{ttime}s_…`) and in the vacancy relaxation, accumulates
   `(dx²/D(0)) · dtime_s_initial` at every attempt, rejected ones included,
   independently of the adaptive step. The physical time is `time_out`. For
   example the final map of the reproducible run is named `map2221.72s_…` but
   corresponds to `t = 967.7 s` (722.6 K).
8. **Interface measure is step-held** between its `n_surface_samples` updates.
9. **2-D only.** The energy and residual reshape fields to `(nx, ny)`.
10. **`X_mean` history** uses the mask `(η > 0.05) & (η > 0.95)`, i.e. `η > 0.95`.

**Differences from the original code** (behaviour outside the published
results only):

* When the adaptive scheme fails, `AdaptiveStepError` is raised. The original
  code plotted the time steps and called `sys.exit`.
* A non-finite residual raises `AdaptiveStepError`. In the original code no
  branch matched, and the step loop never terminated.
* When the matrix never reaches `x_equilibrium`, `good_step` is `None`. The
  original `main` failed on its return statement (`good_step` undefined), after
  having written its maps.
* Two FFTs and one elasticity solve (before the time loop) whose results were
  never used are no longer computed.
* `pf_kks.thermo` raises `ValueError` for an unknown output name; the original
  returned 0. The `d_Hm_dia_dSi` output, which referenced an undefined variable,
  was removed.

**Versions of the original solver.** The original repository held three copies
of `pyFi_adapt.py`. The copy in `ReproducibleRun.zip` is the reference. The
`save_ipynb/` copy differs from it only by the smallest time step
(`dtime_s / 20` instead of `/ 1000`; configurable as `dtime_min_divisor`). The
top-level copy is the `save_ipynb/` one plus the nucleation functions (now
`pf_kks/nucleation.py`).

**Post-processing fixes** (figures only):

* `compute_DSC.ipynb` labelled the heat-capacity + precipitation curve "Surface
  energy term" and the heat-capacity + surface curve "Diffusion and
  precipitation term"; the labels are now correct.
* `Post_Process.ipynb` fitted the vacancy spline against temperature but
  evaluated it at time values; it is now evaluated at temperature.
* The measured DSC file has no header; the original notebook skipped its first
  data row.
* The second experimental curve (`Yang.txt`) and the thesis figure scripts'
  `jupyter_style` module and input files were not part of the original
  repository; those scripts are kept in `legacy_version/DSC_tool/` for reference
  only.
* The top-level `pyFi_RVE_cell.py` had a syntax error (line 408); it is kept
  unchanged in `legacy_version/`.

## Verification

See the [README](README.md#verification). In short: bit-identical to the
original code on every branch of the time loop (small grids), and identical to
round-off (≤ 9e-16 on the fields, ≤ 2e-14 relative on the histories) with the
archived 2022 run on its full-size grid.
