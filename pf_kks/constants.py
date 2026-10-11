"""Physical constants and fixed model constants used by the solver.

Values are the ones hard-coded in the original ``pyFi_adapt.py``.
"""

R = 8.314472  # gas constant [J K^-1 mol^-1]
N_A = 6.02214086e23  # Avogadro constant [mol^-1]
K_B = 1.380649e-23  # Boltzmann constant [J K^-1]

# Excess-vacancy model (formation/binding energies and entropies).
HF1VA = 63680  # vacancy formation enthalpy [J mol^-1]
SF1VA = 2.49  # vacancy formation entropy [-, multiplied by T]
HB = 10613  # vacancy-solute binding enthalpy [J mol^-1]
SB_KB = 42453  # vacancy-solute binding entropy term (as in the original code)
Z_COORD = 12  # FCC coordination number

# Diffusivities entering the effective vacancy-annihilation diffusivity.
D0_AL = 1.7e-4  # Al self-diffusion pre-factor [m^2 s^-1]
Q_AL = 142e3  # Al self-diffusion activation energy [J mol^-1]
D0_SI_IN_AL = 3.66e-6  # Si-in-Al diffusion pre-factor [m^2 s^-1]
Q_SI_IN_AL = 110958  # Si-in-Al diffusion activation energy [J mol^-1]
