"""Tests for magnetic_supercell auto-expansion (SpinW nExt / Sunny
resize_supercell parity).

Covers: dims resolution ('auto' from k), atom/interaction/SIA remapping,
single_k -> real-space pattern conversion (Sunny repeat_periodically_as_spiral
convention), chemical-RLU q handling, per-chemical-cell intensity
normalization, agreement with the hand-written SW03 13-site supercell, and
S(q,w) validated against Sunny (v0.8.1, :dipole_uncorrected — pyMagCalc, like
SpinW, does not apply Sunny's :dipole-mode quadratic-anisotropy
renormalization).

Also regression-tests the degenerate-subspace -q eigenvector matching in
linalg._match_and_reorder_minus_q: collinear two-sublattice AFM cells with
anisotropy have doubly-degenerate bands whose S(q,w) weight used to vanish.
"""

import copy
import os
import sys

import numpy as np
import pytest

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from magcalc.core import MagCalc
from magcalc.generic_model import GenericSpinModel, resolve_supercell_dims
from magcalc.runner import compute_b_matrix

K_SW03 = 3.0 / 13.0


def chain_cfg(J1, J2=None, supercell=None, ms=None, sia=None):
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
    if sia is not None:
        inter.append(sia)
    cs = {
        'lattice_vectors': [[3.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        'atoms_uc': [{'label': 'A', 'pos': [0, 0, 0], 'spin_S': 1.0}],
    }
    if supercell is not None:
        cs['magnetic_supercell'] = supercell
    return {
        'crystal_structure': cs,
        'interactions': inter,
        'parameters': params,
        'parameter_order': order,
        'magnetic_structure': ms or {},
    }


def build(cfg, params, base='.'):
    sm = GenericSpinModel(copy.deepcopy(cfg), base_path=base)
    thetas, phis = sm.generate_magnetic_structure()
    if thetas is not None:
        sm.set_magnetic_structure(thetas, phis)
    calc = MagCalc(spin_model_module=sm, spin_magnitude=1.0, hamiltonian_params=params,
                   cache_file_base='test_supercell', cache_mode='none')
    return sm, calc


def qcart(sm, qxs):
    """Chemical-RLU (h00) -> Cartesian via the runner's B-matrix."""
    B = compute_b_matrix(sm)
    return np.array([[q, 0, 0] for q in qxs]) @ B


# ---------------------------------------------------------------------------
# dims resolution
# ---------------------------------------------------------------------------

def test_resolve_supercell_dims():
    assert resolve_supercell_dims([2, 1, 3]) == [2, 1, 3]
    assert resolve_supercell_dims({'matrix': [4, 1, 1]}) == [4, 1, 1]
    assert resolve_supercell_dims('auto', k_rlu=[0.5, 0, 0]) == [2, 1, 1]
    assert resolve_supercell_dims('auto', k_rlu=[3 / 13, 0.25, 0]) == [13, 4, 1]
    with pytest.raises(ValueError, match='incommensurate'):
        resolve_supercell_dims('auto', k_rlu=[1 / np.sqrt(5), 0, 0])
    with pytest.raises(ValueError):
        resolve_supercell_dims([2, 0, 1])
    with pytest.raises(ValueError):
        resolve_supercell_dims('auto')


# ---------------------------------------------------------------------------
# AFM chain: auto supercell, analytic bands, chemical-RLU q
# ---------------------------------------------------------------------------

AFM_MS = {'enabled': True, 'type': 'single_k', 'k': [0.5, 0, 0],
          'axis': [0, 1, 0], 'local_directions': [[0, 0, 1]]}


def test_afm_auto_supercell_analytic():
    cfg = chain_cfg(1.0, supercell='auto', ms=AFM_MS)
    sm, calc = build(cfg, [1.0])
    assert sm.supercell_dims == [2, 1, 1]
    labels = [a['label'] for a in sm.config['crystal_structure']['atoms_uc']]
    assert labels == ['A', 'A@1_0_0']
    # up-down pattern generated from the single_k structure
    dirs = sm.config['magnetic_structure']['directions']
    np.testing.assert_allclose(np.abs(np.dot(dirs[0], dirs[1])), 1.0, atol=1e-10)
    np.testing.assert_allclose(np.dot(dirs[0], [0, 0, 1]), 1.0, atol=1e-10)

    qs = np.array([0.1, 0.25, 0.4])
    res = calc.calculate_dispersion(qcart(sm, qs), serial=True)
    analytic = 2.0 * np.abs(np.sin(2 * np.pi * qs))
    for i in range(len(qs)):
        np.testing.assert_allclose(res.energies[i], [analytic[i]] * 2, atol=1e-8)


def test_bond_wrapping():
    """Bonds crossing the supercell boundary must wrap with adjusted offsets."""
    cfg = chain_cfg(1.0, supercell=[2, 1, 1], ms=AFM_MS)
    sm = GenericSpinModel(copy.deepcopy(cfg))
    bonds = {(tuple(e['pair']), tuple(e['rij_offset']))
             for e in sm.config['interactions'] if e['type'] == 'heisenberg'}
    assert bonds == {
        (('A', 'A@1_0_0'), (0, 0, 0)),
        (('A@1_0_0', 'A'), (1, 0, 0)),
        (('A', 'A@1_0_0'), (-1, 0, 0)),
        (('A@1_0_0', 'A'), (0, 0, 0)),
    }


# ---------------------------------------------------------------------------
# SW03: auto supercell reproduces the hand-written 13-site reference
# ---------------------------------------------------------------------------

def test_sw03_supercell_matches_handwritten():
    import yaml
    cfg = chain_cfg(-1.0, 2.0, supercell='auto',
                    ms={'enabled': True, 'type': 'single_k',
                        'k': [K_SW03, 0, 0], 'axis': [0, 0, 1]})
    sm_a, calc_a = build(cfg, [-1.0, 2.0])
    assert sm_a.supercell_dims == [13, 1, 1]

    ref_path = os.path.join(project_root, 'examples', 'spinw_tutorials',
                            'SW03_frustrated_chain', 'config.yaml')
    with open(ref_path) as f:
        ref_cfg = yaml.safe_load(f)
    ref_cfg['tasks'] = {}
    sm_r, calc_r = build(ref_cfg, [-1.0, 2.0],
                         base=os.path.dirname(ref_path))

    qs_chem = np.array([0.15, 0.35])
    q_cart = np.array([[2 * np.pi * q / 3.0, 0, 0] for q in qs_chem])
    e_a = np.sort(calc_a.calculate_dispersion(q_cart, serial=True).energies, axis=1)
    e_r = np.sort(calc_r.calculate_dispersion(q_cart, serial=True).energies, axis=1)
    np.testing.assert_allclose(e_a, e_r, atol=1e-9)


# ---------------------------------------------------------------------------
# S(q,w) vs Sunny (hardcoded references, Sunny v0.8.1)
# ---------------------------------------------------------------------------

# AFM chain J=1, S=1, dims (2,1,1), spins +/-z, ssf_perp(apply_g=false),
# q in chemical RLU. Sunny normalizes per chemical cell.
SUNNY_AFM_SC = {
    0.10: ([1.17557052, 1.17557052], 0.16245986),
    0.25: ([2.00000000, 2.00000000], 0.50000000),
    0.40: ([1.17557052, 1.17557052], 1.53884176),
}


def test_supercell_sqw_vs_sunny():
    cfg = chain_cfg(1.0, supercell=[2, 1, 1], ms=AFM_MS)
    sm, calc = build(cfg, [1.0])
    qs = sorted(SUNNY_AFM_SC.keys())
    res = calc.calculate_sqw(qcart(sm, qs))
    for i, q in enumerate(qs):
        e_ref, i_tot = SUNNY_AFM_SC[q]
        np.testing.assert_allclose(np.sort(res.energies[i]), e_ref, atol=1e-6)
        np.testing.assert_allclose(res.intensities[i].sum(), i_tot, atol=1e-6)


# Easy-axis AFM chain J=1, D=-0.2 (bare, SpinW convention), S=1, dims (2,1,1).
# Sunny reference computed in :dipole_uncorrected mode (Sunny's :dipole mode
# renormalizes quadratic SIA by (2s-1)/2s and would give a smaller gap).
# Doubly degenerate bands: only the intensity SUM per q is gauge-independent.
SUNNY_SIA_SC = {
    0.10: (1.77255918, 0.04674932 + 0.17382611),
    0.25: (2.40000000, 0.24377697 + 0.25622303),
}


def test_supercell_sia_sqw_vs_sunny():
    sia = {'type': 'sia', 'value': -0.2, 'axis': [0, 0, 1], 'atoms': ['A']}
    cfg = chain_cfg(1.0, supercell=[2, 1, 1], ms=AFM_MS, sia=sia)
    sm, calc = build(cfg, [1.0])
    # SIA target list must have been expanded to the replicas
    sia_entry = [e for e in sm.config['interactions'] if e['type'] == 'sia'][0]
    assert set(sia_entry['atoms']) == {'A', 'A@1_0_0'}

    qs = sorted(SUNNY_SIA_SC.keys())
    res = calc.calculate_sqw(qcart(sm, qs))
    for i, q in enumerate(qs):
        e_ref, i_tot = SUNNY_SIA_SC[q]
        np.testing.assert_allclose(res.energies[i], [e_ref] * 2, atol=1e-6)
        np.testing.assert_allclose(res.intensities[i].sum(), i_tot, atol=1e-6)


def test_degenerate_matching_regression():
    """Hand-written 2-site easy-axis AFM (no supercell feature): the
    degenerate bands must carry S(q,w) weight (linalg subspace matching)."""
    cfg = {
        'crystal_structure': {
            'lattice_vectors': [[6.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
            'atoms_uc': [{'label': 'A', 'pos': [0.0, 0, 0], 'spin_S': 1.0},
                         {'label': 'B', 'pos': [0.5, 0, 0], 'spin_S': 1.0}]},
        'interactions': [
            {'type': 'heisenberg', 'pair': ['A', 'B'], 'rij_offset': [0, 0, 0], 'value': 'J1'},
            {'type': 'heisenberg', 'pair': ['B', 'A'], 'rij_offset': [0, 0, 0], 'value': 'J1'},
            {'type': 'heisenberg', 'pair': ['B', 'A'], 'rij_offset': [1, 0, 0], 'value': 'J1'},
            {'type': 'heisenberg', 'pair': ['A', 'B'], 'rij_offset': [-1, 0, 0], 'value': 'J1'},
            {'type': 'sia', 'value': -0.2, 'axis': [0, 0, 1]}],
        'parameters': {'J1': 1.0, 'S': 1.0}, 'parameter_order': ['J1'],
        'magnetic_structure': {'enabled': True, 'type': 'pattern',
                               'pattern_type': 'generic',
                               'directions': [[0, 0, 1], [0, 0, -1]]},
    }
    sm, calc = build(cfg, [1.0])
    q_cart = np.array([[2 * np.pi * 0.1 / 3.0, 0, 0]])
    res = calc.calculate_sqw(q_cart)
    # 2-site cell without the supercell feature: intensities are per model
    # cell (2 chemical cells) -> twice the Sunny per-chemical-cell value.
    np.testing.assert_allclose(res.energies[0], [1.77255918] * 2, atol=1e-6)
    np.testing.assert_allclose(res.intensities[0].sum(),
                               2 * (0.04674932 + 0.17382611), atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-atom chemical cell + pattern/explicit remapping
# ---------------------------------------------------------------------------

def test_multiatom_pattern_supercell():
    """2-atom chemical cell replicated [2,1,1]: folded bands must contain the
    unreplicated spectrum at the same Cartesian q."""
    def two_site(supercell=None):
        cs = {
            'lattice_vectors': [[6.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
            'atoms_uc': [{'label': 'A', 'pos': [0.0, 0, 0], 'spin_S': 1.0},
                         {'label': 'B', 'pos': [0.5, 0, 0], 'spin_S': 1.0}]}
        if supercell:
            cs['magnetic_supercell'] = supercell
        return {
            'crystal_structure': cs,
            'interactions': [
                {'type': 'heisenberg', 'pair': ['A', 'B'], 'rij_offset': [0, 0, 0], 'value': 'J1'},
                {'type': 'heisenberg', 'pair': ['B', 'A'], 'rij_offset': [0, 0, 0], 'value': 'J1'},
                {'type': 'heisenberg', 'pair': ['B', 'A'], 'rij_offset': [1, 0, 0], 'value': 'J1'},
                {'type': 'heisenberg', 'pair': ['A', 'B'], 'rij_offset': [-1, 0, 0], 'value': 'J1'}],
            'parameters': {'J1': -1.0, 'S': 1.0}, 'parameter_order': ['J1'],
            'magnetic_structure': {'enabled': True, 'type': 'pattern',
                                   'pattern_type': 'generic',
                                   'directions': [[0, 0, 1], [0, 0, 1]]},
        }

    sm1, calc1 = build(two_site(), [-1.0])
    sm2, calc2 = build(two_site([2, 1, 1]), [-1.0])
    assert len(sm2.config['crystal_structure']['atoms_uc']) == 4
    assert len(sm2.config['magnetic_structure']['directions']) == 4

    q_cart = np.array([[2 * np.pi * 0.13 / 6.0, 0, 0]])
    e1 = np.sort(calc1.calculate_dispersion(q_cart, serial=True).energies[0])
    e2 = np.sort(calc2.calculate_dispersion(q_cart, serial=True).energies[0])
    # every unfolded band appears among the folded ones
    for e in e1:
        assert np.min(np.abs(e2 - e)) < 1e-8
