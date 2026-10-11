"""pf_kks.thermo returns exactly the values of the original free_nrj_mob.py."""

import numpy as np
import pytest

from pf_kks import thermo

from .legacy import LEGACY_THERMO, load_module

legacy = load_module(LEGACY_THERMO, "legacy_free_nrj_mob")

TEMPERATURES = [300.0, 400.0, 523.7, 650.0, 700.0, 701.0, 850.0, 933.0, 1000.0]
COMPOSITIONS = [1e-5, 0.0123, 0.025, 0.5, 0.9999]


def _same(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)


@pytest.mark.parametrize("output", thermo.GIBBS_ENERGY_OUTPUTS)
def test_gibbs_energy(output):
    for T in TEMPERATURES:
        for x in COMPOSITIONS:
            assert _same(thermo.gibbs_energy(T, x, output), legacy.gibbs_energy(T=T, X_Si=x, output=output)), (T, x)


@pytest.mark.parametrize("output", thermo.MOLAR_VOLUME_OUTPUTS)
def test_molar_volume(output):
    for T in TEMPERATURES:
        for x in COMPOSITIONS:
            assert _same(
                thermo.molar_volume(298, T, x, output), legacy.molar_volume(T0=298, T=T, X_Si=x, output=output)
            )


@pytest.mark.parametrize("output", thermo.ELASTIC_CONSTANT_OUTPUTS)
def test_elastic_constants(output):
    for T in TEMPERATURES:
        assert _same(thermo.elas_const(T, output), legacy.elas_const(T=T, output=output))


@pytest.mark.parametrize("output", thermo.ATOMIC_MOBILITY_OUTPUTS)
def test_atomic_mobility(output):
    for T in TEMPERATURES:
        for x in COMPOSITIONS:
            assert _same(thermo.atomic_mobility(T, x, output), legacy.atomic_mobility(T=T, X_Si=x, output=output))


def test_unknown_output_raises():
    with pytest.raises(ValueError):
        thermo.gibbs_energy(500.0, 0.01, "not_an_output")
