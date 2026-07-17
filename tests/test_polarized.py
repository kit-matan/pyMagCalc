"""Polarized / chiral neutron cross-sections, validated against Sunny 0.8.1.

With the neutron polarization along q (the usual longitudinal SF/NSF setup) all
magnetic scattering is spin-flip, and the two beam polarizations differ by the chiral
term:

    M_ch(q, w) = i * qhat . [ sum_abc eps_abc S^ab(q, w) ]
    sigma_SF^(+/-) = S_perp -/+ M_ch

M_ch is the antisymmetric (imaginary) part of the correlation tensor, so it vanishes
identically for any collinear structure and is nonzero only for a chiral one.

NORMALIZATION. pyMagCalc's absolute S(q,w) is 3/4 of Sunny's -- a pre-existing
convention difference that affects the ordinary `perp` channel identically (verified),
and which a fit's free `scale` absorbs. So these tests compare the
normalization-INDEPENDENT ratio chiral/perp, which is what actually pins the sign
convention and the physics.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

A = 3.0
LAT = [[3 * A, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]      # 3-site supercell along a


def _model(directions):
    """J1-J2 chain, J2 = J1/2 -> commensurate k = 1/3 helix, as an explicit 3-site cell
    (so this is plain LSWT in both codes -- no rotating frame, no convention gap)."""
    atoms = [{"label": f"S{i}", "pos": [i / 3.0, 0.0, 0.0], "spin_S": 1.0,
              "ion": "Fe2+"} for i in range(3)]
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": A, "value": 1.0},
            {"type": "heisenberg", "distance": 2 * A, "value": 0.5},
        ]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": directions},
    }
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="pol", hamiltonian_params=[])


def _q(hs):
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return [np.array([3 * h, 0, 0]) @ B for h in hs]      # chemical rlu -> supercell


# proper screw: rotation axis || k || x, spins rotate in the yz plane
SCREW = [[0.0, np.cos(2 * np.pi * i / 3), np.sin(2 * np.pi * i / 3)] for i in range(3)]
# cycloid: rotation axis || z, PERPENDICULAR to k
CYCLOID = [[np.cos(2 * np.pi * i / 3), np.sin(2 * np.pi * i / 3), 0.0] for i in range(3)]

HS = [0.10, 0.20, 0.45]

# Sunny 0.8.1 (chiral via ssf_custom, perp via ssf_perp), bands sorted by energy
SUNNY_E = {0.10: [0.84326, 1.17364, 1.35592],
           0.20: [0.96139, 1.44517, 1.55379],
           0.45: [0.91255, 1.32395, 1.47572]}
SUNNY_CHIRAL = {0.10: [0.40561, -0.62277, 0.0],
                0.20: [0.47561, 0.0, -1.7594],
                0.45: [0.0, 0.75704, -2.09444]}
SUNNY_PERP = {0.10: [0.40561, 0.62277, 0.0],
              0.20: [0.47561, 0.0, 1.7594],
              0.45: [0.0, 0.75704, 2.09444]}


def test_chiral_matches_sunny_sign_and_magnitude():
    calc = _model(SCREW)
    qs = _q(HS)
    ch = calc.calculate_sqw(qs, cross_section="chiral")
    pp = calc.calculate_sqw(qs, cross_section="perp")
    E = np.real(ch.energies)

    for i, h in enumerate(HS):
        order = np.argsort(E[i])
        assert np.allclose(E[i][order], SUNNY_E[h], atol=1e-3)

        c_py = np.real(ch.intensities)[i][order]
        p_py = np.real(pp.intensities)[i][order]
        c_su = np.array(SUNNY_CHIRAL[h])
        p_su = np.array(SUNNY_PERP[h])

        # normalization-independent: chiral / perp, band by band
        m = p_py > 1e-9
        assert np.allclose(c_py[m] / p_py[m], c_su[m] / p_su[m], atol=1e-4), (
            f"chiral/perp mismatch at q={h}: {c_py[m]/p_py[m]} vs {c_su[m]/p_su[m]}")


def test_proper_screw_magnons_are_fully_circularly_polarized():
    """For a proper screw each magnon is fully circularly polarized, so |chiral| = perp
    band by band (Sunny shows the same). This is what makes the chiral channel the
    handedness probe it is."""
    calc = _model(SCREW)
    qs = _q(HS)
    c = np.real(calc.calculate_sqw(qs, cross_section="chiral").intensities)
    p = np.real(calc.calculate_sqw(qs, cross_section="perp").intensities)
    m = p > 1e-9
    assert np.allclose(np.abs(c[m]) / p[m], 1.0, atol=1e-6)


def test_chiral_vanishes_for_a_cycloid_when_q_is_perpendicular_to_the_axis():
    """The chiral term goes as qhat . n. A cycloid has its rotation axis perpendicular
    to k, so scattering along k sees no chirality -- Sunny gives exactly 0 here too.
    Guards against a formula that 'finds' chirality in the wrong geometry."""
    calc = _model(CYCLOID)
    c = np.real(calc.calculate_sqw(_q(HS), cross_section="chiral").intensities)
    assert np.max(np.abs(c)) < 1e-9


def test_chiral_vanishes_for_a_collinear_structure():
    collinear = [[0, 0, 1], [0, 0, -1], [0, 0, 1]]
    calc = _model(collinear)
    c = np.real(calc.calculate_sqw(_q(HS), cross_section="chiral").intensities)
    assert np.max(np.abs(c)) < 1e-9


@pytest.mark.slow
def test_spin_flip_channels_split_by_twice_the_chiral_term():
    """sigma_SF^(+/-) = S_perp -/+ M_ch, so SF- minus SF+ must be exactly 2 M_ch."""
    calc = _model(SCREW)
    qs = _q(HS)
    perp = np.real(calc.calculate_sqw(qs, cross_section="perp").intensities)
    chi = np.real(calc.calculate_sqw(qs, cross_section="chiral").intensities)
    sfp = np.real(calc.calculate_sqw(qs, cross_section="sf+").intensities)
    sfm = np.real(calc.calculate_sqw(qs, cross_section="sf-").intensities)
    # SF channels are clamped at zero (they are cross-sections), so compare where
    # both are positive
    assert np.allclose(sfp + sfm, 2 * perp, atol=1e-6)
    m = (sfp > 1e-9) & (sfm > 1e-9)
    assert np.allclose((sfm - sfp)[m], 2 * chi[m], atol=1e-6)


def test_unknown_cross_section_still_raises():
    calc = _model(SCREW)
    with pytest.raises(ValueError, match="Unknown cross_section"):
        calc.calculate_sqw(_q([0.1]), cross_section="nonsense")
