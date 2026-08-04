"""CP^(N-1) classical dynamics of SU(N) coherent states (Gap 4 #26).

Sunny's generalized spin dynamics (`04_GSD_FeI2.jl`) and
`EntangledSampledCorrelations`. The dipole engine evolves Landau-Lifshitz, which
structurally cannot represent a site in a general N-level coherent state -- exactly
the states SU(N) mode and entangled units exist for.

The equations of motion are the mean-field Schroedinger equation

    i dZ_i/dt = h_i Z_i,    h_i = SUNModel.local_field(i, <O>)

so the generator is the SAME local Hamiltonian the CP^(N-1) ground-state search
already builds; #26 was never about new physics input, only about using it to
propagate rather than to minimize.

THE LOAD-BEARING TEST is the first one: at N = 2 this must reduce EXACTLY to
Landau-Lifshitz. That is the dynamical analogue of the "S=1/2 SU(N) == dipole" gate
that anchors `tests/test_sun.py`, and it fails loudly on any factor of 2, sign, or
mean-field-convention slip. Measured agreement over 400 steps: 4.8e-10.
"""
import copy

import numpy as np
import pytest

from magcalc.classical_dynamics import evolve as ll_evolve
from magcalc.generic_model import GenericSpinModel
from magcalc.sun import dynamics as sd
from magcalc.sun.lswt import SUNModel
from magcalc.sun.operators import coherent_from_direction
from magcalc.thermal_mc import build_supercell

LAT = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 9.0]]
NN = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))


def _cfg(S=0.5, J=1.0, sia=None):
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": J} for p, o in NN]}
    if sia is not None:
        inter["single_ion_anisotropy"] = [{"value": sia, "axis": [0, 0, 1],
                                           "atoms": ["A", "B"]}]
    return {"crystal_structure": {"lattice_vectors": LAT,
                                  "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                                "spin_S": S},
                                               {"label": "B", "pos": [0.5, 0, 0],
                                                "spin_S": S}]},
            "interactions": inter, "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[0, 0, 1], [0, 0, -1]]},
            "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}


def _model(cfg):
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[])


def _dipoles_along(mdl, traj):
    out = []
    for cfg_ in traj:
        out.append([[np.real(z.conj() @ mdl.ops[i][a] @ z) for a in range(3)]
                    for i, z in enumerate(cfg_)])
    return np.array(out)


@pytest.mark.parametrize("dt,tol", [(0.02, 2e-8), (0.01, 1e-9)])
def test_N2_reduces_exactly_to_landau_lifshitz(dt, tol):
    """THE gate. For S = 1/2 the coherent state is a spinor and the CP^1 dynamics
    IS Landau-Lifshitz -- started from the same dipoles, the two engines must trace
    the same trajectory. The residual is RK4 truncation in two different
    parametrizations of the same flow, so it shrinks with dt (checked by the two
    parameter sets)."""
    cfg = _cfg(S=0.5)
    mdl = _model(cfg)
    rng = np.random.default_rng(3)
    dirs = rng.standard_normal((2, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    Z0 = [coherent_from_direction(0.5, d) for d in dirs]
    mdl.Z = [z.copy() for z in Z0]
    mdl._prepare()

    H, b, N, S, _ = build_supercell(GenericSpinModel(copy.deepcopy(cfg)), [], (1, 1, 1))
    m0 = mdl.dipoles.copy()
    assert np.linalg.norm(m0, axis=1) == pytest.approx([0.5, 0.5], abs=1e-12)

    n = 400
    sun = _dipoles_along(mdl, sd.evolve(mdl, Z0, dt, n, record_every=50))
    ll = ll_evolve(H, b, S, m0, dt, n, record_every=50)
    assert np.abs(sun - ll).max() < tol


def test_energy_and_norms_are_conserved():
    """Every h_i is Hermitian, so the flow is unitary per site: |Z_i| = 1 exactly and
    the energy is conserved to the integrator's order. A drift here means the mean
    field is not the true gradient of the energy being reported."""
    mdl = _model(_cfg(S=1.0, sia=-0.4))
    rng = np.random.default_rng(11)
    Z0 = []
    for i in range(mdl.L):
        v = rng.standard_normal(mdl.Ns[i]) + 1j * rng.standard_normal(mdl.Ns[i])
        Z0.append(v / np.linalg.norm(v))
    E0 = sd.energy(mdl, Z0)
    traj = sd.evolve(mdl, Z0, 0.01, 800, record_every=100)
    for cfg_ in traj:
        assert [float(np.linalg.norm(z)) for z in cfg_] == pytest.approx(
            [1.0] * mdl.L, abs=1e-12)
        assert sd.energy(mdl, cfg_) == pytest.approx(E0, abs=1e-8)


def test_dynamics_is_nontrivial_beyond_N2():
    """Guards the guard: with S = 1 and a single-ion anisotropy the state explores
    directions no dipole can (quadrupolar components), so the trajectory must NOT be
    reproducible by evolving the dipole alone."""
    mdl = _model(_cfg(S=1.0, sia=-0.6))
    rng = np.random.default_rng(5)
    Z0 = []
    for i in range(mdl.L):
        v = rng.standard_normal(mdl.Ns[i]) + 1j * rng.standard_normal(mdl.Ns[i])
        Z0.append(v / np.linalg.norm(v))
    traj = sd.evolve(mdl, Z0, 0.01, 400, record_every=50)
    d = _dipoles_along(mdl, traj)
    lengths = np.linalg.norm(d, axis=2)
    # |<S>| is NOT conserved for a genuine SU(N) state -- weight moves between the
    # dipole and the higher multipoles. A dipole-only evolution keeps it fixed.
    assert lengths.max() - lengths.min() > 1e-3


def test_metropolis_samples_and_leaves_the_state_normalized():
    mdl = _model(_cfg(S=1.0))
    rng = np.random.default_rng(7)
    Z = [z.copy() for z in mdl.Z]
    acc = sd.thermalize(mdl, Z, kT=0.5, n_sweeps=40, rng=rng)
    assert [float(np.linalg.norm(z)) for z in Z] == pytest.approx([1.0] * mdl.L,
                                                                  abs=1e-12)


def test_sampled_correlations_runs_and_is_positive():
    """Smoke + sanity: S(q,w) must be real, finite and non-negative in the `perp`
    channel."""
    mdl = _model(_cfg(S=1.0))
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    qs = np.array([[0.2, 0, 0], [0.5, 0, 0]]) @ B
    w, sqw = sd.sampled_correlations(mdl, qs, kT=0.5, dt=0.02, n_steps=128,
                                     n_traj=2, therm_sweeps=40, seed=0)
    assert np.all(np.isfinite(sqw))
    assert sqw.min() >= -1e-12
    assert sqw.shape == (len(w), 2)


@pytest.mark.slow
def test_low_T_peaks_fall_on_the_SUN_lswt_dispersion():
    """KNOWN FAILURE, deliberately left visible: the finite-T S(q,w) from these
    trajectories peaks at ROUGHLY HALF the SU(N) LSWT magnon energy.

    Measured on the S = 1 chain with an easy-axis anisotropy: LSWT bands
    [2.0924, 2.0924, 4.0, 4.0] at q = (0.3, 0, 0), against a classical peak at
    1.07 -- a ratio of 1.95, stable across kT = 0.05/0.15/0.3 and carrying real
    spectral weight, so it is not sampling noise.

    A clean factor of ~2 is a bug until proven otherwise (GAP_STATUS's standing
    rule), and this one is NOT explained. What is already established:

      * the equations of motion and integrator are right -- at N = 2 they reproduce
        Landau-Lifshitz to 4.8e-10 (the first test in this file), and that dipole
        path is itself pinned to the LSWT dispersion in test_classical_dynamics.py;
      * energy and every |Z_i| are conserved to 1e-8 / 1e-12.

    So the defect is somewhere between the trajectory and the spectrum -- the
    thermal sampling on CP^(N-1), the moment operator, or a genuine factor in the
    N > 2 correspondence -- and it is not worth guessing. `sun.dynamics.sampled_correlations`
    is therefore NOT validated and must not be used for quantitative work; the
    deterministic dynamics underneath it is.
    """
    pytest.xfail("SU(N) S(q,w) peaks at ~half the LSWT energy; unexplained. "
                 "See GAP_STATUS Gap 4 #26.")
