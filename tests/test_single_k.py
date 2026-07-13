"""Tests for the unified single-k (propagation-vector) magnetic structures.

Covers: analytic helix dispersion with q +/- k satellites, the legacy
'spiral' shim, the S0 (lab-frame) back-rotation convention, the three-channel
incommensurate S(q,w) validated against Sunny's SpinWaveTheorySpiral, the
k_case 2 (k = 1/2) cross-terms, the spiral k-optimizer, and the
rotational-symmetry gate.

The Sunny reference numbers were generated with Sunny.jl v0.8.1
(SpinWaveTheorySpiral, ssf_perp with apply_g=false) on the same models; see
the in-repo source at Sunny.jl-main/src/Spiral/.
"""

import copy
import logging
import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import magcalc as mc
from magcalc.core import MagCalc, reciprocal_b_matrix
from magcalc.generic_model import (
    GenericSpinModel,
    normalize_magnetic_structure,
    rotation_about_axis,
    spiral_propagation_case,
)
from magcalc.spiral_opt import optimize_spiral

# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

K_SW03 = float(np.arccos(1.0 / 8.0) / (2 * np.pi))  # 0.2300534561...


def chain_config(J1, J2=None, k=(0.0, 0.0, 0.0), axis=(0, 0, 1), ms_extra=None,
                 interactions_extra=None):
    """1-site chain along a (a=3): J1 at +/-1 cells, optional J2 at +/-2."""
    inter = [
        {'type': 'heisenberg', 'pair': ['A', 'A'], 'rij_offset': [1, 0, 0], 'value': 'J1'},
        {'type': 'heisenberg', 'pair': ['A', 'A'], 'rij_offset': [-1, 0, 0], 'value': 'J1'},
    ]
    params = {'J1': J1, 'S': 1.0}
    order = ['J1']
    if J2 is not None:
        inter += [
            {'type': 'heisenberg', 'pair': ['A', 'A'], 'rij_offset': [2, 0, 0], 'value': 'J2'},
            {'type': 'heisenberg', 'pair': ['A', 'A'], 'rij_offset': [-2, 0, 0], 'value': 'J2'},
        ]
        params['J2'] = J2
        order.append('J2')
    if interactions_extra:
        inter += interactions_extra
    ms = {'enabled': True, 'type': 'single_k', 'k': list(k), 'axis': list(axis)}
    if ms_extra:
        ms.update(ms_extra)
    return {
        'crystal_structure': {
            'lattice_vectors': [[3.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
            'atoms_uc': [{'label': 'A', 'pos': [0, 0, 0], 'spin_S': 1.0}],
        },
        'interactions': inter,
        'parameters': params,
        'parameter_order': order,
        'magnetic_structure': ms,
    }


def build_calc(cfg, params, S=1.0, cache_base='test_single_k'):
    sm = GenericSpinModel(copy.deepcopy(cfg))
    thetas, phis = sm.generate_magnetic_structure()
    if thetas is not None:
        sm.set_magnetic_structure(thetas, phis)
    calc = MagCalc(spin_model_module=sm, spin_magnitude=S, hamiltonian_params=params,
                   cache_file_base=cache_base, cache_mode='none')
    return sm, calc


def qcart_chain(sm, qxs):
    B = reciprocal_b_matrix(np.array(sm.unit_cell(), float))
    return np.array([[q, 0, 0] for q in qxs]) @ B


# ---------------------------------------------------------------------------
# 1. Analytic helix: central + satellite branches (SW03 physics)
# ---------------------------------------------------------------------------

def analytic_helix_w(q, k, J1, J2, S=1.0):
    A = lambda x: J1 * np.cos(2 * np.pi * x) + J2 * np.cos(4 * np.pi * x)
    val = (A(q) - A(k)) * ((A(q + k) + A(q - k)) / 2.0 - A(k))
    return 2 * S * np.sqrt(np.maximum(val, 0.0))


def test_satellite_dispersion_analytic_helix():
    J1, J2, k = -1.0, 2.0, K_SW03
    cfg = chain_config(J1, J2, k=(k, 0, 0))
    sm, calc = build_calc(cfg, [J1, J2])
    assert calc.k_case == 3

    qs = np.linspace(0.02, 0.98, 21)
    res = calc.calculate_dispersion(qcart_chain(sm, qs), serial=True, satellites=True)
    assert res.energies.shape == (len(qs), 3)
    assert res.branch_labels == ['q-k', 'q', 'q+k']
    np.testing.assert_allclose(res.energies[:, 1], analytic_helix_w(qs, k, J1, J2),
                               atol=1e-8)
    np.testing.assert_allclose(res.energies[:, 0], analytic_helix_w(qs - k, k, J1, J2),
                               atol=1e-8)
    np.testing.assert_allclose(res.energies[:, 2], analytic_helix_w(qs + k, k, J1, J2),
                               atol=1e-8)


def test_satellites_off_by_default():
    J1, J2, k = -1.0, 2.0, K_SW03
    cfg = chain_config(J1, J2, k=(k, 0, 0))
    sm, calc = build_calc(cfg, [J1, J2])
    res = calc.calculate_dispersion(qcart_chain(sm, [0.1, 0.4]), serial=True)
    assert res.energies.shape == (2, 1)  # central branch only


# ---------------------------------------------------------------------------
# 2. Legacy 'spiral' shim is a pure rename
# ---------------------------------------------------------------------------

def test_legacy_spiral_shim_identical():
    J1, J2, k = -1.0, 2.0, K_SW03
    cfg_new = chain_config(J1, J2, k=(k, 0, 0))
    cfg_old = copy.deepcopy(cfg_new)
    cfg_old['magnetic_structure']['type'] = 'spiral'

    sm_new, calc_new = build_calc(cfg_new, [J1, J2], cache_base='shim_new')
    sm_old, calc_old = build_calc(cfg_old, [J1, J2], cache_base='shim_old')
    assert sm_old.mag_struct_cfg['type'] == 'single_k'

    qs = np.linspace(0.05, 0.95, 11)
    e_new = calc_new.calculate_dispersion(qcart_chain(sm_new, qs), serial=True).energies
    e_old = calc_old.calculate_dispersion(qcart_chain(sm_old, qs), serial=True).energies
    np.testing.assert_allclose(e_new, e_old, atol=1e-10)


def test_normalize_shim_propagation_vector():
    cfg = normalize_magnetic_structure(
        {'type': 'propagation_vector', 'k': [0.5, 0, 0], 'subtype': 'planar'})
    assert cfg['type'] == 'single_k'
    assert cfg['real_space'] is True
    assert cfg['u'] == [1, 0, 0] and cfg['v'] == [0, 1, 0]
    assert cfg['k_case'] == 2


def test_spiral_propagation_case():
    assert spiral_propagation_case([0, 0, 0]) == 1
    assert spiral_propagation_case([1, 2, -1]) == 1
    assert spiral_propagation_case([0.5, 0, 0]) == 2
    assert spiral_propagation_case([0.5, 0.5, 1.0]) == 2
    assert spiral_propagation_case([K_SW03, 0, 0]) == 3


# ---------------------------------------------------------------------------
# 3. S0 (lab-frame) input reproduces local_directions (rotating-frame) input
# ---------------------------------------------------------------------------

def two_site_chain_config(k, ms):
    """2-site cell along a (both sites on the chain, a=6, sites at 0 and 1/2)."""
    return {
        'crystal_structure': {
            'lattice_vectors': [[6.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
            'atoms_uc': [
                {'label': 'A', 'pos': [0.0, 0, 0], 'spin_S': 1.0},
                {'label': 'B', 'pos': [0.5, 0, 0], 'spin_S': 1.0},
            ],
        },
        'interactions': [
            {'type': 'heisenberg', 'pair': ['A', 'B'], 'rij_offset': [0, 0, 0], 'value': 'J1'},
            {'type': 'heisenberg', 'pair': ['B', 'A'], 'rij_offset': [0, 0, 0], 'value': 'J1'},
            {'type': 'heisenberg', 'pair': ['B', 'A'], 'rij_offset': [1, 0, 0], 'value': 'J1'},
            {'type': 'heisenberg', 'pair': ['A', 'B'], 'rij_offset': [-1, 0, 0], 'value': 'J1'},
        ],
        'parameters': {'J1': -1.0, 'S': 1.0},
        'parameter_order': ['J1'],
        'magnetic_structure': dict({'enabled': True, 'type': 'single_k',
                                    'k': [k, 0, 0], 'axis': [0, 0, 1]}, **ms),
    }


def test_s0_back_rotation_matches_local_directions():
    k = 0.123
    # Rotating-frame input: both sites along x in the rotating frame.
    cfg_local = two_site_chain_config(k, {'local_directions': [[1, 0, 0], [1, 0, 0]]})
    # Equivalent lab-frame input: S0_i = R(+2*pi*k*d_i) x_hat.
    n = np.array([0.0, 0.0, 1.0])
    d = [0.0, 0.5]
    S0 = [list(rotation_about_axis(2 * np.pi * k * di, n) @ np.array([1.0, 0.0, 0.0]))
          for di in d]
    cfg_s0 = two_site_chain_config(k, {'S0': S0})

    sm_a, calc_a = build_calc(cfg_local, [-1.0], cache_base='s0_a')
    sm_b, calc_b = build_calc(cfg_s0, [-1.0], cache_base='s0_b')

    th_a, ph_a = sm_a.generate_magnetic_structure()
    th_b, ph_b = sm_b.generate_magnetic_structure()
    np.testing.assert_allclose(th_a, th_b, atol=1e-12)
    np.testing.assert_allclose(np.unwrap(ph_a), np.unwrap(ph_b), atol=1e-12)

    qs = np.linspace(0.05, 0.45, 5)
    e_a = calc_a.calculate_dispersion(qcart_chain(sm_a, qs), serial=True).energies
    e_b = calc_b.calculate_dispersion(qcart_chain(sm_b, qs), serial=True).energies
    np.testing.assert_allclose(e_a, e_b, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Three-channel S(q,w) against Sunny (SpinWaveTheorySpiral, v0.8.1)
# ---------------------------------------------------------------------------

# SW03 J1-J2 chain (J1=-1, J2=2, S=1, k=0.2300534561..., axis z), channels
# ordered [q-k | q | q+k]. Sunny reference from intensities_bands (energies
# paired with intensities; Sunny sorts descending, we match by energy).
SUNNY_SW03 = {
    0.05: ([2.165200, 2.026510, 2.466077], [0.041698, 0.185613, 0.039653]),
    0.20: ([1.277279, 1.394911, 3.993543], [0.575311, 2.574605, 0.265638]),
    0.35: ([3.285769, 4.714906, 4.266245], [0.111102, 0.580010, 0.234935]),
    0.50: ([1.988738, 2.250000, 1.988738], [0.031427, 0.111111, 0.031427]),
}


def test_sqw_three_channels_vs_sunny():
    J1, J2, k = -1.0, 2.0, K_SW03
    cfg = chain_config(J1, J2, k=(k, 0, 0))
    sm, calc = build_calc(cfg, [J1, J2], cache_base='sqw_sunny')

    qs = sorted(SUNNY_SW03.keys())
    res = calc.calculate_sqw(qcart_chain(sm, qs))
    assert res.energies.shape == (len(qs), 3)

    for i, q in enumerate(qs):
        e_ref, i_ref = SUNNY_SW03[q]
        np.testing.assert_allclose(res.energies[i], e_ref, atol=2e-5)
        np.testing.assert_allclose(res.intensities[i], i_ref, atol=2e-5)


def test_sqw_k_to_zero_matches_sunny():
    """FM chain at tiny k: the k_case 3 decomposition is applied (Sunny does
    the same; the spiral formulas are intentionally discontinuous between
    k -> 0 and k = 0). Sunny v0.8.1 reference at q = 0.15, 0.35:
    E branches degenerate at the FM magnon energy, I = [0.125, 0.5, 0.125]."""
    J1 = -1.0
    cfg_sk = chain_config(J1, k=(1e-7, 0, 0),
                          ms_extra={'local_directions': [[1, 0, 0]]})
    sm_a, calc_a = build_calc(cfg_sk, [J1], cache_base='cont_a')

    qs = [0.15, 0.35]
    fm_energy = lambda q: 2 * 1.0 * (-J1) * (1 - np.cos(2 * np.pi * q))
    res_a = calc_a.calculate_sqw(qcart_chain(sm_a, qs))
    for i, q in enumerate(qs):
        np.testing.assert_allclose(res_a.energies[i], [fm_energy(q)] * 3, atol=1e-4)
        np.testing.assert_allclose(res_a.intensities[i], [0.125, 0.5, 0.125],
                                   atol=1e-6)


# AFM chain (J=1, S=1) treated as spiral with k=[1/2,0,0] (k_case 2). Sunny
# branch intensities are compared as sorted multisets: all three branches are
# degenerate, so the per-branch assignment is convention-dependent.
SUNNY_CASE2 = {
    0.15: ([1.6180340011] * 3, [0.1273813629, 0.1273813629, 0.2547627259]),
    0.30: ([1.9021130431] * 3, [0.3440954795, 0.3440954795, 0.6881909591]),
    0.45: ([0.6180340211] * 3, [1.5784378001, 1.5784378001, 3.1568756001]),
    0.60: ([1.1755705216] * 3, [0.7694208753, 0.7694208753, 1.5388417506]),
    0.75: ([2.0000000100] * 3, [0.25, 0.25, 0.5]),
}


def test_sqw_k_case2_vs_sunny():
    cfg = chain_config(1.0, k=(0.5, 0, 0))
    sm, calc = build_calc(cfg, [1.0], cache_base='case2')
    assert calc.k_case == 2

    qs = sorted(SUNNY_CASE2.keys())
    res = calc.calculate_sqw(qcart_chain(sm, qs))
    for i, q in enumerate(qs):
        e_ref, i_ref = SUNNY_CASE2[q]
        np.testing.assert_allclose(res.energies[i], e_ref, atol=1e-6)
        np.testing.assert_allclose(np.sort(res.intensities[i]), np.sort(i_ref),
                                   atol=1e-6)


# ---------------------------------------------------------------------------
# 5. Spiral k-optimizer
# ---------------------------------------------------------------------------

def test_optimizer_recovers_j1j2_k():
    J1, J2 = -1.0, 2.0
    cfg = chain_config(J1, J2, k=(0.1, 0, 0))
    sm = GenericSpinModel(copy.deepcopy(cfg))
    res = optimize_spiral(sm, [J1, J2], {'num_starts': 3}, S_val=1.0)
    err = min(abs(res.k_rlu[0] - K_SW03), abs(1.0 - res.k_rlu[0] - K_SW03))
    assert err < 1e-5
    assert abs(res.energy_per_site - (J1 * np.cos(2 * np.pi * K_SW03)
                                      + J2 * np.cos(4 * np.pi * K_SW03))) < 1e-8
    # committed to the model
    assert sm.mag_struct_cfg['type'] == 'single_k'
    assert sm.optimized_matrices is not None


def test_optimizer_fm_returns_k_zero():
    cfg = chain_config(-1.0, k=(0.3, 0, 0))
    sm = GenericSpinModel(copy.deepcopy(cfg))
    res = optimize_spiral(sm, [-1.0], {'num_starts': 2}, S_val=1.0)
    err = min(res.k_rlu[0], 1.0 - res.k_rlu[0])
    assert err < 1e-5
    assert abs(res.energy_per_site - (-1.0)) < 1e-8


def triangular_config():
    """Triangular-lattice AFM (NN J=1): ground state is the 120-degree spiral
    with k = [1/3, 1/3, 0]."""
    return {
        'crystal_structure': {
            'lattice_vectors': [[3.0, 0, 0],
                                [-1.5, 3.0 * np.sqrt(3) / 2, 0],
                                [0, 0, 8.0]],
            'atoms_uc': [{'label': 'A', 'pos': [0, 0, 0], 'spin_S': 1.0}],
        },
        'interactions': [
            {'type': 'heisenberg', 'pair': ['A', 'A'], 'rij_offset': off, 'value': 'J1'}
            for off in ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                        [1, 1, 0], [-1, -1, 0])
        ],
        'parameters': {'J1': 1.0, 'S': 1.0},
        'parameter_order': ['J1'],
        'magnetic_structure': {'enabled': True, 'type': 'single_k',
                               'k': [0.2, 0.2, 0], 'axis': [0, 0, 1]},
    }


def test_optimizer_triangular_lattice():
    cfg = triangular_config()
    sm = GenericSpinModel(copy.deepcopy(cfg))
    res = optimize_spiral(sm, [1.0], {'num_starts': 3}, S_val=1.0)
    # 1/3 or 2/3 (equivalent chirality domains)
    for comp in res.k_rlu[:2]:
        assert min(abs(comp - 1.0 / 3.0), abs(comp - 2.0 / 3.0)) < 1e-4
    assert abs(res.energy_per_site - (-1.5)) < 1e-6  # 3*J*S^2*cos(120)


# ---------------------------------------------------------------------------
# 6. Rotational-symmetry gate
# ---------------------------------------------------------------------------

def test_symmetry_gate_dm_not_parallel(caplog):
    dm = [
        {'type': 'dm', 'pair': ['A', 'A'], 'rij_offset': [1, 0, 0],
         'value': [0.0, 0.3, 0.0]},
        {'type': 'dm', 'pair': ['A', 'A'], 'rij_offset': [-1, 0, 0],
         'value': [0.0, -0.3, 0.0]},
    ]
    cfg = chain_config(-1.0, k=(0.25, 0, 0), interactions_extra=dm)
    sm = GenericSpinModel(copy.deepcopy(cfg))
    with caplog.at_level(logging.WARNING, logger='magcalc.generic_model'):
        sm.spin_interactions([-1.0])
    assert any('NOT invariant' in r.message for r in caplog.records)


def test_symmetry_gate_error_mode():
    dm = [
        {'type': 'dm', 'pair': ['A', 'A'], 'rij_offset': [1, 0, 0],
         'value': [0.0, 0.3, 0.0]},
        {'type': 'dm', 'pair': ['A', 'A'], 'rij_offset': [-1, 0, 0],
         'value': [0.0, -0.3, 0.0]},
    ]
    cfg = chain_config(-1.0, k=(0.25, 0, 0), interactions_extra=dm,
                       ms_extra={'enforce_rotational_symmetry': 'error'})
    sm = GenericSpinModel(copy.deepcopy(cfg))
    with pytest.raises(ValueError, match='NOT invariant'):
        sm.spin_interactions([-1.0])


def test_symmetry_gate_dm_parallel_ok(caplog):
    dm = [
        {'type': 'dm', 'pair': ['A', 'A'], 'rij_offset': [1, 0, 0],
         'value': [0.0, 0.0, 0.3]},
        {'type': 'dm', 'pair': ['A', 'A'], 'rij_offset': [-1, 0, 0],
         'value': [0.0, 0.0, -0.3]},
    ]
    cfg = chain_config(-1.0, k=(0.25, 0, 0), interactions_extra=dm)
    sm = GenericSpinModel(copy.deepcopy(cfg))
    with caplog.at_level(logging.WARNING, logger='magcalc.generic_model'):
        sm.spin_interactions([-1.0])
    assert not any('NOT invariant' in r.message for r in caplog.records)
