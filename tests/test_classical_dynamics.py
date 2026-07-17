"""Finite-T classical spin dynamics / SampledCorrelations (magcalc/classical_dynamics.py).

Pinned to exact/independent results, never a self-generated number:

  * a single spin in a field precesses at the Larmor frequency ω = gμ_B B — the
    S(0,ω) peak, within the FFT resolution (pins integrator + time/ω convention);
  * the undamped RK4 Landau–Lifshitz integrator conserves energy (drift → 0);
  * for a Heisenberg ferromagnet the low-T dynamical S(q,ω) peaks fall on the EXACT
    LSWT magnon dispersion the pyMagCalc engine computes — the dynamics inherit the
    validated spin-wave engine as their oracle.
"""
import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel
from magcalc.thermal_mc import build_supercell, MU_B, GAMMA
from magcalc.classical_dynamics import (evolve, energy, dynamical_structure_factor,
                                        sampled_correlations)


def _peak(e, y):
    i = int(np.argmax(y[1:])) + 1
    if 0 < i < len(y) - 1:
        d = 0.5 * (y[i - 1] - y[i + 1]) / (y[i - 1] - 2 * y[i] + y[i + 1] + 1e-30)
        return e[i] + d * (e[1] - e[0])
    return e[i]


def test_larmor_precession_frequency():
    S, Bz = 1.0, 20.0
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": []},
        "parameters": {"Hz": Bz}, "parameter_order": ["Hz"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [1, 0, 0]}}
    m = GenericSpinModel(cfg)
    H, b, N, S, pos = build_supercell(m, [Bz], supercell=(1, 1, 1))
    dt, nsteps = 0.01, 4096
    traj = evolve(H, b, S, np.array([[S, 0, 0.0]]), dt, nsteps)
    e, sqw = dynamical_structure_factor(traj, pos, np.array([[0, 0, 0]]), dt, "trace")
    peak = e[int(np.argmax(sqw[1:, 0])) + 1]
    d_omega = 2 * np.pi / (nsteps * dt)
    assert abs(peak - GAMMA * MU_B * Bz) < 1.5 * d_omega


def test_undamped_integrator_conserves_energy():
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 10, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    m = GenericSpinModel(cfg)
    H, b, N, S, pos = build_supercell(m, [], supercell=(8, 1, 1))
    rng = np.random.default_rng(0)
    mm = rng.standard_normal((N, 3))
    mm *= S / np.linalg.norm(mm, axis=1, keepdims=True)
    traj = evolve(H, b, S, mm, 0.01, 3000)
    assert abs(energy(H, b, traj[-1]) - energy(H, b, traj[0])) < 1e-6


def test_ferromagnet_dynamics_peaks_track_lswt_dispersion():
    """Low-T classical S(q,ω) peaks must fall on the exact LSWT magnon dispersion."""
    S, J = 1.0, -1.0
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 10, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": J},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    gm = GenericSpinModel(cfg)
    sm = SUNModel.from_generic_model(gm, [])
    B = 2 * np.pi * np.linalg.inv(np.array([[1., 0, 0], [0, 10, 0], [0, 0, 10]])).T
    L = 16
    qs_rlu = [4.0 / L, 8.0 / L]
    qs = np.array([[r, 0, 0] for r in qs_rlu]) @ B
    res = sampled_correlations(gm, [], qs, kT=0.03, supercell=(L, 1, 1),
                               dt=0.02, n_steps=4096, n_traj=4, therm_sweeps=1200,
                               cross_section="perp", seed=1)
    for i, r in enumerate(qs_rlu):
        w_dyn = _peak(res.energies, res.sqw[:, i])
        wl = np.sort(np.real(sm.dispersion(qs[i])))
        wl = wl[wl > 1e-6][0]
        assert abs(w_dyn - wl) / wl < 0.05, f"q={r}: {w_dyn} vs LSWT {wl}"
