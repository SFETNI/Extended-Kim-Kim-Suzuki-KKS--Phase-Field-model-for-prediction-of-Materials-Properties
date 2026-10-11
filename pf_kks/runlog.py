"""Run logs written next to the outputs, in the format of the original code.

``dtime.txt`` records the material state every 100 steps and every rejected
step; ``log.txt`` records the residual of every adaptive step. Keeping the
original format lets new runs be compared line by line with archived logs.
"""

import os


def _e(x):
    return "{0:.2e}".format(x)


class RunLog:
    def __init__(self, directory, enabled=True):
        self.enabled = enabled
        self.log_path = os.path.join(directory, "log.txt")
        self.dtime_path = os.path.join(directory, "dtime.txt")
        if enabled:
            os.makedirs(directory, exist_ok=True)
            with open(self.log_path, "w") as f:
                f.write("start simulation\n")
            with open(self.dtime_path, "w") as f:
                f.write("print time steps\n")

    def _write(self, path, text):
        if self.enabled:
            with open(path, "a") as f:
                f.write(text)

    def state(self, istep, T, D, L, Meta, Omega0, dtime_s):
        self._write(
            self.dtime_path,
            "step: "
            + str(istep)
            + ", T= "
            + "{0:.3f} ".format(T)
            + " D= "
            + _e(D)
            + " L= "
            + _e(L)
            + " Meta= "
            + _e(Meta)
            + " Vm= "
            + _e(Omega0)
            + " dtime_s= "
            + "{0:.2f}".format(dtime_s)
            + " \n",
        )

    def rejected(self, istep, iteration, dtime_s, res, energy_s):
        self._write(
            self.log_path,
            "step "
            + str(istep)
            + ": RE>resmax\n"
            + "Residual = "
            + _e(abs(res.total))
            + ","
            + " Energy_deriv= "
            + _e(res.energy_derivative)
            + ","
            + " Potentiel_grad= "
            + _e(res.diffusion_term)
            + ","
            + " eta_term= "
            + _e(res.allen_cahn_term)
            + " temp_term= "
            + _e(res.temperature_term)
            + ","
            + "         Energy= "
            + _e(energy_s)
            + "\n",
        )
        self._write(
            self.dtime_path,
            "RE>resmax\n"
            + "repeat istep "
            + str(istep)
            + " : iteration "
            + str(iteration)
            + " dtime_s:  "
            + _e(dtime_s)
            + "\n"
            + "\n",
        )

    def accepted(self, istep, res, energy_s):
        self._write(
            self.log_path,
            "step: "
            + str(istep)
            + ", We get  (resmin <RE <resmax) ;  go to step "
            + str(istep + 1)
            + "\n"
            + "Residual = "
            + _e(abs(res.total))
            + ","
            + " Energy_deriv= "
            + _e(res.energy_derivative)
            + ","
            + " Potentiel_grad= "
            + _e(res.diffusion_term)
            + ","
            + " eta_term= "
            + _e(res.allen_cahn_term)
            + ","
            + " temp_term= "
            + _e(res.temperature_term)
            + ","
            + "         Energy= "
            + _e(energy_s)
            + "\n",
        )

    def enlarged(self, istep, res, energy_s):
        self._write(
            self.log_path,
            " We get  :RE <resmin;  go to step "
            + str(istep)
            + "\n"
            + "Residual = "
            + _e(abs(res.total))
            + ","
            + " Energy_deriv= "
            + _e(res.energy_derivative)
            + ","
            + " Potentiel_grad= "
            + _e(res.diffusion_term)
            + ","
            + " eta_term= "
            + _e(res.allen_cahn_term)
            + ","
            + " temp_term= "
            + _e(res.temperature_term)
            + "         Energy= "
            + _e(energy_s)
            + "\n",
        )
