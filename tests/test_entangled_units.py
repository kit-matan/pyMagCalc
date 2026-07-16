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


def test_Cu5SbO6_reproduces_the_paper_dimer_expansion():
    """The shipped Cu5SbO6 example (Piyakulworawat et al., PRR 8, 013247 (2026)) must
    reproduce the paper's J1-J2-J4 dimer expansion: the triplon dispersion is the
    full bond-operator resummation of Eq. (A11), and the structure factor obeys the
    dimer selection rule (zero at q1 = 0)."""
    cfg = yaml.safe_load(open(os.path.join(
        HERE, "..", "examples", "entangled", "Cu5SbO6", "config.yaml")))
    J1, J2, J4 = cfg["parameters"]["J1"], cfg["parameters"]["J2"], cfg["parameters"]["J4"]
    m = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    sm = build_entangled_model(m, pv, units=[[0, 1]])
    L = np.array(cfg["crystal_structure"]["lattice_vectors"], float)
    B = 2 * np.pi * np.linalg.inv(L).T

    # intra-dimer on-site: singlet at -3J1/4, triplet at +J1/4 -> gap J1.
    ev = np.linalg.eigvalsh(sm.onsite[0][1])
    assert abs((ev[-1] - ev[0]) - J1) < 1e-9

    # dispersion == sqrt(J1^2 - J1 J2 cos 2pi q1 - J1 J4 cos 2pi(q3-q1)) (first-order
    # expansion Eq. A11 is its leading term); band spans ~11-21 meV as in the paper.
    tops = []
    for (h, k, l) in [(0, 0, 0), (0.25, 0, 0), (0.5, 0, 0), (0.5, 0, 0.5), (0.3, 0, 0.7)]:
        w = np.max(np.real(sm.dispersion(np.array([h, k, l]) @ B)))
        full = np.sqrt(J1**2 - J1 * J2 * np.cos(2 * np.pi * h)
                       - J1 * J4 * np.cos(2 * np.pi * (l - h)))
        assert abs(w - full) < 1e-9, f"{w} vs {full} at {(h, k, l)}"
        tops.append(w)
    grid = [np.max(np.real(sm.dispersion(np.array([h, 0, l]) @ B)))
            for h in np.linspace(0, 1, 21) for l in np.linspace(0, 1, 21)]
    assert 10.5 < min(grid) < 11.5 and 20.0 < max(grid) < 21.0    # paper: ~11 to ~21 meV

    # dimer selection rule S ~ 1 - cos(4 pi q1 / 3): silent at q1 = 0.
    _, I0 = sm.structure_factor(np.array([0.0, 0, 0]) @ B, cross_section="trace")
    assert np.sum(I0) < 1e-9


def test_Rb2Cu3SnF12_dimer_DM_and_field_mechanism():
    """The Rb2Cu3SnF12 pinwheel-dimer example (Matan et al., Nat. Phys. 6, 865 (2010)):
    out-of-plane DM splits the triplet into Stot^z = 0 (raised) and the degenerate
    Stot^z = +/-1 branch, and a c-axis field Zeeman-splits the +/-1 doublet while the
    0 branch stays fixed -- all exact analytic single-dimer results (the paper's Fig. 4
    mechanism)."""
    cfg0 = yaml.safe_load(open(os.path.join(
        HERE, "..", "examples", "entangled", "Rb2Cu3SnF12", "config.yaml")))
    J1, Dz, muB = 18.6, 3.348, 5.788e-2

    def gaps(B):
        cfg = yaml.safe_load(yaml.safe_dump(cfg0))
        cfg["parameters"]["H_mag"] = float(B)
        m = GenericSpinModel(cfg)
        pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
        sm = build_entangled_model(m, pv, units=[[0, 1]])
        return np.sort(np.real(sm.dispersion(np.zeros(3))))

    # zero field: Stot^z=+/-1 (degenerate) below Stot^z=0.
    d_pm = J1 / 2 + 0.5 * np.sqrt(J1**2 + Dz**2)
    d_0 = np.sqrt(J1**2 + Dz**2)
    g0 = gaps(0.0)
    assert np.allclose(g0, sorted([d_pm, d_pm, d_0]), atol=1e-6)
    assert d_pm < d_0                                    # DM raises the Sz=0 branch

    # field: the +/-1 doublet splits by +/- 2 mu_B B (g=2); the Sz=0 branch is fixed.
    gB = gaps(20.0)
    assert np.allclose(gB, sorted([d_pm - 2 * muB * 20, d_0, d_pm + 2 * muB * 20]), atol=1e-6)
    assert abs(gB[1] - g0[-1]) < 1e-6                    # middle (Sz=0) unchanged by field


def test_Rb2Cu3SnF12_triplet_polarizations():
    """Ehlers et al., PRB 89, 024414 (2014): the Stot^z = +/-1 doublet is polarized
    IN-PLANE and the Stot^z = 0 singlet OUT-OF-PLANE. The entangled structure factor
    must reproduce this: the +/-1 modes carry only S_xx/S_yy, the 0 mode only S_zz."""
    Sm = spin_matrices(0.5)
    I2 = np.eye(2)
    S1 = [np.kron(Sm[a], I2) for a in range(3)]
    S2 = [np.kron(I2, Sm[a]) for a in range(3)]
    J1, Dz = 18.6, 0.18 * 18.6
    A = J1 * sum(S1[a] @ S2[a] for a in range(3)) + Dz * (S1[0] @ S2[1] - S1[1] @ S2[0])
    Z = np.linalg.eigh(A)[1][:, 0]
    d = 1.0
    m = SUNModel(spins=[1.5], coherent_states=[Z], bonds=[], onsite=[(0, A)],
                 operators=[S1 + S2],
                 moment_terms=[[(np.array([-d / 2, 0, 0]), (0, 1, 2)),
                                (np.array([d / 2, 0, 0]), (3, 4, 5))]])
    m.pos = np.zeros((1, 3))
    q = np.array([np.pi / d, 0, 0])

    def chan(cs):
        w, I = m.structure_factor(q, cross_section=cs)    # sort by this channel's own E
        return np.real(I)[np.argsort(np.real(w))]         # ascending E: [Sz=+/-1, +/-1, 0]

    Ixx, Izz = chan("xx"), chan("zz")
    assert Ixx[0] > 1e-3 and Ixx[1] > 1e-3 and Ixx[2] < 1e-9     # +/-1 in-plane, 0 not
    assert Izz[2] > 1e-3 and Izz[0] < 1e-9 and Izz[1] < 1e-9     # 0 out-of-plane, +/-1 not


def test_units_must_partition_all_sites():
    cfg = yaml.safe_load(open(CFG))
    m = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    with pytest.raises(ValueError, match="partition"):
        build_entangled_model(m, pv, units=[[0]])                # site 1 left out
