"""Polarized / chiral neutron cross-sections, validated against Sunny 0.8.1.

With the neutron polarization along q (the usual longitudinal SF/NSF setup) all
magnetic scattering is spin-flip, and the two beam polarizations differ by the chiral
term:

    M_ch(q, w) = i * qhat . [ sum_abc eps_abc S^ab(q, w) ]
    sigma_SF^(+/-) = S_perp -/+ M_ch

M_ch is the antisymmetric (imaginary) part of the correlation tensor, so it vanishes
identically for any collinear structure and is nonzero only for a chiral one.

NORMALIZATION. There is none to correct for: pyMagCalc's absolute S(q,w) EQUALS
Sunny's, here and on a collinear ferromagnet and Neel antiferromagnet
(`test_absolute_normalization.py`). This file used to carry hardcoded Sunny numbers
that were uniformly 4/3 too large, which was written up -- in this docstring, in
CLAUDE.md and in GAP_STATUS.md -- as a "pre-existing 3/4 convention difference" that
users were told to apply before comparing absolute intensities with Sunny. It was not
a convention; the reference numbers were simply wrong, and the tests could not see it
because they only ever compared the ratio chiral/perp, in which any overall factor
cancels. The values below were regenerated from Sunny 0.8.1 and are now asserted
ABSOLUTELY, so a normalization regression fails here.
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
    # No `ion`: the Sunny reference is form-factor-free, so this compares the
    # spin correlation function itself rather than f(Q)^2 times it.
    atoms = [{"label": f"S{i}", "pos": [i / 3.0, 0.0, 0.0], "spin_S": 1.0}
             for i in range(3)]
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

# Sunny 0.8.1, regenerated (bands sorted by energy). ssf_perp / ssf_custom, both
# with apply_g=false and no form factor:
#
#   chiral = ssf_custom(sys; apply_g=false) do q, ssf
#       qh = q / max(norm(q), 1e-12)
#       Float64(real(im * (qh[1]*(ssf[2,3]-ssf[3,2]) + qh[2]*(ssf[3,1]-ssf[1,3])
#                          + qh[3]*(ssf[1,2]-ssf[2,1]))))
#   end
SUNNY_E = {0.10: [0.8432638, 1.1736404, 1.3559224],
           0.20: [0.9613918, 1.4451732, 1.5537861],
           0.45: [0.9125506, 1.3239505, 1.4757244]}
SUNNY_CHIRAL = {0.10: [0.3042106, -0.4670783, 0.0],
                0.20: [0.3567096, 0.0, -1.3195507],
                0.45: [0.0, 0.5677769, -1.5708310]}
SUNNY_PERP = {0.10: [0.3042106, 0.4670783, 0.0],
              0.20: [0.3567096, 0.0, 1.3195507],
              0.45: [0.0, 0.5677769, 1.5708310]}


def test_chiral_and_perp_match_sunny_absolutely():
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

        # ABSOLUTE, band by band -- magnitude as well as sign.
        assert np.allclose(p_py, p_su, atol=1e-6), (
            f"perp mismatch at q={h}: {p_py} vs {p_su}")
        assert np.allclose(c_py, c_su, atol=1e-6), (
            f"chiral mismatch at q={h}: {c_py} vs {c_su}")

        # and the ratio, which is what pins the sign convention itself
        m = p_py > 1e-9
        assert np.allclose(c_py[m] / p_py[m], c_su[m] / p_su[m], atol=1e-6)


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
