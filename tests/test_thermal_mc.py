"""Thermal Monte-Carlo with parallel tempering (magcalc/thermal_mc.py).

Pinned to EXACT classical results, never a self-generated number:

  * N non-interacting spins in a field: <m·B̂>/S = −L(βgμ_B|B|S) (Langevin), per T;
  * the classical Heisenberg dimer <E>(T) = −J S² L(βJS²) from the exact 1-D
    partition-function integral;
  * the dimer heat capacity C(T) = (JS²)²/T² L'(βJS²) — the fluctuation estimator
    Var(E)/(N kT²) must reproduce it;
  * parallel tempering reproduces independent single-temperature Metropolis.
"""
import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.thermal_mc import (build_supercell, parallel_tempering, MU_B, GAMMA)


def _L(x):
    return 1.0 / np.tanh(x) - 1.0 / x


def _Lp(x):
    return 1.0 / x**2 - 1.0 / np.sinh(x)**2


def test_noninteracting_spins_follow_langevin():
    """N free spins in a field: magnetization is the Langevin function, exactly."""
    S, Bz = 1.0, 12.0
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": []},
        "parameters": {"Hz": Bz}, "parameter_order": ["Hz"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    m = GenericSpinModel(cfg)
    H, b, N, S, _pos = build_supercell(m, [Bz], supercell=(4, 4, 1))
    assert abs(np.linalg.norm(b[:3]) - GAMMA * MU_B * Bz) < 1e-9
    bmag = np.linalg.norm(b[:3])
    temps = np.array([2.0, 5.0, 12.0])
    res = parallel_tempering(H, b, N, S, temps, n_sweeps=8000, n_equil=2500, seed=1)
    for i, T in enumerate(res.temperatures):
        x = bmag * S / T
        assert abs(res.mag_vector[i, 2] - (-_L(x))) < 0.02, f"kT={T}"


def _dimer_model(S=1.0, J=1.0):
    cfg = {"crystal_structure": {"lattice_vectors": [[10., 0, 0], [0, 10, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                         {"label": "B", "pos": [0.05, 0, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 0.5,
                                             "value": "J"}]},
        "parameters": {"J": J}, "parameter_order": ["J"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "antiferromagnetic",
                               "direction": [0, 0, 1], "propagation_vector": [0, 0, 0]}}
    return GenericSpinModel(cfg)


def test_classical_dimer_energy_and_heat_capacity():
    """Isolated classical Heisenberg dimers vs the exact <E>(T) and C(T)."""
    S, J = 1.0, 1.0
    m = _dimer_model(S, J)
    H, b, N, S, _pos = build_supercell(m, [J], supercell=(3, 3, 1))
    temps = np.array([0.5, 0.9, 1.5, 3.0])
    res = parallel_tempering(H, b, N, S, temps, n_sweeps=12000, n_equil=4000, seed=3)
    for i, T in enumerate(res.temperatures):
        a = J * S**2 / T
        E_exact = -J * S**2 * _L(a) / 2.0                 # per spin (2 spins/dimer)
        C_exact = (J * S**2)**2 / T**2 * _Lp(a) / 2.0
        assert abs(res.energy[i] - E_exact) < 0.01, f"E kT={T}"
        assert abs(res.heat_capacity[i] - C_exact) < 0.03, f"C kT={T}"


def test_parallel_tempering_matches_independent_metropolis():
    """PT (with swaps) and independent single-T Metropolis (no swaps) must agree on
    <E>(T) within statistics — swaps change sampling efficiency, not the distribution."""
    m = _dimer_model()
    H, b, N, S, _pos = build_supercell(m, [1.0], supercell=(3, 3, 1))
    temps = np.array([0.4, 0.8, 1.6, 3.2])
    pt = parallel_tempering(H, b, N, S, temps, n_sweeps=9000, n_equil=3000,
                            swap_every=1, seed=5)
    ind = parallel_tempering(H, b, N, S, temps, n_sweeps=9000, n_equil=3000,
                             swap_every=0, seed=6)
    assert np.max(np.abs(pt.energy - ind.energy)) < 0.01
