"""Entangled units (dimers) -- Tier 2 #11.

A "unit" is a small cluster treated as one effective SU(N) site: the intra-unit
coupling is diagonalized exactly (the reference is the unit's ground state, e.g. a
dimer SINGLET with zero dipole), and the excitations are transitions within the unit
spectrum (the triplon). Dipole -- and single-site SU(N) -- LSWT cannot represent this.

Pinned to exact / analytic references:
  * isolated S=1/2 dimer -> flat triplon at omega = J (the singlet-triplet gap), and the
    singlet has <S_tot> = 0 (nothing for dipole LSWT to expand);
  * coupled-dimer chain -> omega(q) = sqrt(J^2 - J J' cos(2 pi q)), the harmonic
    bond-operator (Sachdev-Bhatt) triplon dispersion, to machine precision -- reproduced
    both directly and through the full config -> runner path;
  * the isolated-dimer neutron intensity vanishes at q=0 (the dimer selection rule -- the
    total-spin operator is silent) and follows the (1 - cos(q.d)) structure factor.
"""
import os

import numpy as np
import pytest
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel
from magcalc.sun.operators import spin_matrices
from magcalc.sun.entangled import build_entangled_model

HERE = os.path.dirname(__file__)
CFG = os.path.join(HERE, "..", "examples", "entangled", "dimer_chain", "config.yaml")


def _dimer_pieces(J=1.0):
    S = spin_matrices(0.5)
    I2 = np.eye(2)
    S1 = [np.kron(S[a], I2) for a in range(3)]
    S2 = [np.kron(I2, S[a]) for a in range(3)]
    A = J * sum(S1[a] @ S2[a] for a in range(3))
    Z = np.linalg.eigh(A)[1][:, 0]                 # singlet
    return S1, S2, A, Z


def test_isolated_dimer_is_a_flat_triplon_at_J():
    J = 1.7
    S1, S2, A, Z = _dimer_pieces(J)
    Stot = [S1[a] + S2[a] for a in range(3)]
    m = SUNModel(spins=[1.5], coherent_states=[Z], bonds=[], onsite=[(0, A)],
                 operators=[S1 + S2])
    m.pos = np.zeros((1, 3))
    for q in ([0, 0, 0], [0.4, 0, 0], [1.3, 0.7, 0]):
        w = np.sort(np.real(m.dispersion(np.array(q, float))))
        assert np.allclose(w, J, atol=1e-9), f"{w} != flat {J}"
    # the singlet carries no dipole -> dipole LSWT has nothing to expand about.
    assert np.allclose([Z.conj() @ Stot[a] @ Z for a in range(3)], 0, atol=1e-12)


def test_coupled_dimer_triplon_matches_bond_operator_dispersion():
    """Direct construction: omega(k) = sqrt(J^2 - J J' cos k)."""
    J, Jp = 1.0, 0.3
    S1, S2, A, Z = _dimer_pieces(J)
    C = np.zeros((6, 6))
    for a in range(3):
        C[3 + a, a] = Jp                            # S2(i).S1(j) at +a
    bonds = [(0, 0, np.array([1., 0, 0]), C), (0, 0, np.array([-1., 0, 0]), C.T)]
    m = SUNModel(spins=[1.5], coherent_states=[Z], bonds=bonds, onsite=[(0, A)],
                 operators=[S1 + S2])
    for qr in np.linspace(0, 0.5, 11):
        w = np.max(np.real(m.dispersion(np.array([2 * np.pi * qr, 0, 0]))))
        assert abs(w - np.sqrt(J**2 - J * Jp * np.cos(2 * np.pi * qr))) < 1e-9


def test_config_runner_path_reproduces_the_analytic_triplon():
    """Through GenericSpinModel + build_entangled_model, from the shipped example config."""
    cfg = yaml.safe_load(open(CFG))
    J, Jp = cfg["parameters"]["J"], cfg["parameters"]["Jp"]
    m = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    sm = build_entangled_model(m, pv, units=[[0, 1]])
    assert sm.L == 1 and sm.N == 4 and len(sm.bonds) == 2
    a = cfg["crystal_structure"]["lattice_vectors"][0][0]        # chain lattice const
    for qr in (0.0, 0.15, 0.3, 0.5):
        w = np.max(np.real(sm.dispersion(np.array([2 * np.pi * qr / a, 0, 0]))))
        assert abs(w - np.sqrt(J**2 - J * Jp * np.cos(2 * np.pi * qr))) < 1e-8


def test_dimer_structure_factor_selection_rule():
    """Isolated dimer: intensity vanishes at q=0 and follows (1 - cos(q.d))."""
    J = 1.0
    S1, S2, A, Z = _dimer_pieces(J)
    d = 1.2
    dA, dB = np.array([-d / 2, 0, 0]), np.array([d / 2, 0, 0])
    m = SUNModel(spins=[1.5], coherent_states=[Z], bonds=[], onsite=[(0, A)],
                 operators=[S1 + S2],
                 moment_terms=[[(dA, (0, 1, 2)), (dB, (3, 4, 5))]])
    m.pos = np.zeros((1, 3))

    def I(qx):
        _, inten = m.structure_factor(np.array([qx, 0, 0]), cross_section="trace")
        return float(np.sum(inten))

    assert I(0.0) < 1e-12                                        # dimer selection rule
    # proportional to (1 - cos(q.d)): the ratio is constant across q
    qs = [0.4, 0.9, 1.5, 2.1]
    ratios = [I(q) / (1 - np.cos(q * d)) for q in qs]
    assert np.allclose(ratios, ratios[0], rtol=1e-6), ratios


def test_units_must_partition_all_sites():
    cfg = yaml.safe_load(open(CFG))
    m = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    with pytest.raises(ValueError, match="partition"):
        build_entangled_model(m, pv, units=[[0]])                # site 1 left out
