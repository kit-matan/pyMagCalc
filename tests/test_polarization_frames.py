"""Blume-Maleev frames and arbitrary polarization axes (Gap 4 #25).

pyMagCalc's polarized cross-sections were P || q only: `perp`, `trace`, `chiral`,
`sf+`/`sf-` and lab-frame components. Two dict forms now generalize that:

    cross_section: {polarization: [x, y, z], channel: sf+|sf-|sf|nsf}
    cross_section: {bm: {u: [...], v: [...]}, component: '23'}

The BM frame follows Sunny `ssf_custom_bm`: e1 = qhat, e3 = the scattering-plane
normal, e2 = e3 x qhat, and q outside the plane is an error rather than something to
approximate. u/v/normal are CARTESIAN lab vectors here, as `domains.axis` already is.

Pinned to Sunny for the frame itself, and to exact identities for the channels --
in particular that P || q must reproduce the existing `sf+`/`sf-` BIT FOR BIT, so the
new general path cannot quietly disagree with the one already validated against Sunny
in test_polarized.py.
"""
import numpy as np
import pytest

from magcalc.numerical import blume_maleev_axes, contract_cross_section
# Shared with test_polarized.py ON PURPOSE: the Sunny reference numbers below
# were generated for exactly that model, so if it changes this must fail too.
from tests.test_polarized import CYCLOID, SCREW, _model, _q

HS = [0.10, 0.20, 0.45]

# Sunny 0.8.1: ssf_custom_bm(sys; u=[1,0,0], v=[0,1,0], apply_g=false) taking
# real(ssf[a,b]); bands sorted by energy. The cycloid is included because for the
# screw S^22 and S^33 coincide by symmetry, so that model alone could not tell the
# BM axes apart.
SUNNY_BM = {
    ("screw", "11"): [[0.0, 0.0, 0.5934792], [0.0, 1.6560211, 0.0],
                      [3.3640022, 0.0, 0.0]],
    ("screw", "22"): [[0.1521053, 0.2335392, 0.0], [0.1783548, 0.0, 0.6597753],
                      [0.0, 0.2838884, 0.7854155]],
    ("screw", "33"): [[0.1521053, 0.2335392, 0.0], [0.1783548, 0.0, 0.6597753],
                      [0.0, 0.2838884, 0.7854155]],
    ("screw", "23"): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ("cycloid", "11"): [[0.1521053, 0.2335392, 0.0], [0.1783548, 0.0, 0.6597753],
                        [0.0, 0.2838884, 0.7854155]],
    ("cycloid", "22"): [[0.1521053, 0.2335392, 0.0], [0.1783548, 0.0, 0.6597753],
                        [0.0, 0.2838884, 0.7854155]],
    ("cycloid", "33"): [[0.0, 0.0, 0.5934792], [0.0, 1.6560211, 0.0],
                        [3.3640022, 0.0, 0.0]],
    ("cycloid", "23"): [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
}
STRUCTURES = {"screw": SCREW, "cycloid": CYCLOID}
# scattering plane spanned by x and y -> normal along z; q runs along x, in plane.
BM_PLANE = {"u": [1, 0, 0], "v": [0, 1, 0]}


def _sorted(calc, qs, cs):
    r = calc.calculate_sqw(qs, cross_section=cs)
    E, I = np.real(r.energies), np.real(r.intensities)
    o = np.argsort(E, axis=1)
    return np.take_along_axis(E, o, 1), np.take_along_axis(I, o, 1)


@pytest.mark.parametrize("key", sorted(SUNNY_BM))
def test_bm_components_match_sunny(key):
    name, comp = key
    calc = _model(STRUCTURES[name])
    _, I = _sorted(calc, _q(HS), {"bm": BM_PLANE, "component": comp})
    assert I == pytest.approx(np.array(SUNNY_BM[key]), abs=2e-6)


def test_bm_axes_follow_sunnys_convention():
    """e1 = qhat, e3 = plane normal, e2 = e3 x qhat -- right-handed, orthonormal."""
    R = blume_maleev_axes([1.0, 1.0, 0.0], [0, 0, 1])
    e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
    assert e1 == pytest.approx(np.array([1, 1, 0]) / np.sqrt(2), abs=1e-12)
    assert e3 == pytest.approx([0, 0, 1], abs=1e-12)
    assert e2 == pytest.approx(np.cross(e3, e1), abs=1e-12)
    assert R.T @ R == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_bm_trace_is_the_ordinary_trace():
    """A rotation cannot change the trace: S^11 + S^22 + S^33 in the BM frame must
    equal `trace`. Catches a non-orthonormal or mis-transposed frame."""
    calc = _model(SCREW)
    qs = _q(HS)
    tot = sum(_sorted(calc, qs, {"bm": BM_PLANE, "component": c})[1]
              for c in ("11", "22", "33"))
    assert tot == pytest.approx(_sorted(calc, qs, "trace")[1], abs=1e-9)


def test_bm_refuses_q_out_of_the_scattering_plane():
    """A mis-specified geometry is an error, not something to approximate (Sunny
    refuses it too)."""
    with pytest.raises(ValueError, match="not in the scattering plane"):
        blume_maleev_axes([0.0, 0.0, 1.0], [0, 0, 1])
    calc = _model(SCREW)
    with pytest.raises(ValueError, match="not in the scattering plane"):
        calc.calculate_sqw(_q(HS), cross_section={"bm": {"normal": [1, 0, 0]},
                                                  "component": "11"})


# --------------------------------------------------------------------------
# Arbitrary polarization axis
# --------------------------------------------------------------------------
def test_polarization_along_q_reproduces_the_existing_sf_channels():
    """THE reduction: with P || q the general path must return exactly what the
    already-Sunny-pinned 'sf+'/'sf-' strings return. Bit for bit, not approximately
    -- the two share no code beyond the tensor itself."""
    calc = _model(SCREW)
    qs = _q(HS)
    q_dir = [1.0, 0.0, 0.0]                      # the q path runs along x
    for sign in ("sf+", "sf-"):
        E0, I0 = _sorted(calc, qs, sign)
        E1, I1 = _sorted(calc, qs, {"polarization": q_dir, "channel": sign})
        assert E1 == pytest.approx(E0, abs=0)
        assert I1 == pytest.approx(I0, abs=1e-12)


def test_nsf_vanishes_when_the_polarization_is_along_q():
    """M_perp is by construction perpendicular to q, so P || q sees no non-spin-flip
    magnetic scattering at all. This is the statement that makes P || q the
    'all scattering is spin-flip' geometry."""
    calc = _model(SCREW)
    _, nsf = _sorted(calc, _q(HS), {"polarization": [1, 0, 0], "channel": "nsf"})
    assert np.abs(nsf).max() < 1e-12


@pytest.mark.parametrize("P", [[0, 0, 1], [0, 1, 0], [1, 1, 0], [0.3, -0.5, 0.81]])
def test_sf_plus_nsf_is_the_unpolarized_total(P):
    """Exact sum rule for ANY polarization axis: the spin-flip and non-spin-flip
    channels partition the unpolarized magnetic cross-section. Fails on any
    projector slip."""
    calc = _model(SCREW)
    qs = _q(HS)
    _, sf = _sorted(calc, qs, {"polarization": P, "channel": "sf"})
    _, nsf = _sorted(calc, qs, {"polarization": P, "channel": "nsf"})
    _, perp = _sorted(calc, qs, "perp")
    assert sf + nsf == pytest.approx(perp, abs=1e-9)


def test_chiral_term_cancels_between_the_two_spin_flip_channels():
    """(SF+ + SF-)/2 = SF: the chiral term is what distinguishes them and nothing
    else may."""
    calc = _model(SCREW)
    qs = _q(HS)
    P = [0.3, -0.5, 0.81]
    _, sfp = _sorted(calc, qs, {"polarization": P, "channel": "sf+"})
    _, sfm = _sorted(calc, qs, {"polarization": P, "channel": "sf-"})
    _, sf = _sorted(calc, qs, {"polarization": P, "channel": "sf"})
    assert 0.5 * (sfp + sfm) == pytest.approx(sf, abs=1e-9)


# Sunny 0.8.1 on the SCREW helix, P = normalize([0.3, -0.5, 0.81]), bands ascending.
# The helix is used because its three bands are non-degenerate: on a collinear magnet
# the two magnons are degenerate, and the split of any quantity between a degenerate
# pair is basis-dependent, so per-band values there are not observables at all (both
# codes give a legitimate but different split). Learned the hard way -- see
# test_collinear_chirality_cancels_only_in_the_band_sum.
GEN_P = [0.3, -0.5, 0.81]
SUNNY_GENERAL_P = {
    "nsf": [[0.1383622, 0.2124383, 0.0], [0.1622400, 0.0, 0.6001631],
            [0.0, 0.2582384, 0.7144513]],
    "sf": [[0.1658484, 0.2546400, 0.0], [0.1944696, 0.0, 0.7193876],
           [0.0, 0.3095384, 0.8563796]],
    "chiral": [[0.0914417, -0.1403975, 0.0], [0.1072222, 0.0, -0.3966394],
               [0.0, 0.1706662, -0.4721709]],
}


@pytest.mark.parametrize("channel", ["nsf", "sf"])
def test_general_polarization_channels_match_sunny(channel):
    """NSF(P) and SF(P) about an arbitrary axis, band by band."""
    calc = _model(SCREW)
    _, I = _sorted(calc, _q(HS), {"polarization": GEN_P, "channel": channel})
    assert I == pytest.approx(np.array(SUNNY_GENERAL_P[channel]), abs=2e-6)


def test_general_polarization_chiral_term_matches_sunny():
    """The chiral term itself, extracted as (SF- - SF+)/2. Sign AND magnitude, which
    is what distinguishes a correct polarized calculation from a plausible one."""
    calc = _model(SCREW)
    qs = _q(HS)
    _, sfp = _sorted(calc, qs, {"polarization": GEN_P, "channel": "sf+"})
    _, sfm = _sorted(calc, qs, {"polarization": GEN_P, "channel": "sf-"})
    chiral = 0.5 * (sfm - sfp)
    assert chiral == pytest.approx(np.array(SUNNY_GENERAL_P["chiral"]), abs=2e-6)


NEEL_HS = [[0.13, 0, 0], [0.31, 0, 0], [0.45, 0, 0]]
NEEL_LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]


def _neel_chain():
    """A GENUINELY collinear ground state.

    The obvious thing -- an up-down-up state on the J1-J2 helix cell above -- is not
    an energy minimum at all (Sunny refuses it outright: "wavevector unstable"), and
    `_model` builds MagCalc directly so no guard catches it. Expanding about a
    non-minimum is the #1 source of silent wrongness here, and it is just as wrong
    inside a test as in a calculation.
    """
    import copy

    import magcalc as mc
    from magcalc.generic_model import GenericSpinModel
    nn = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
          (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
    cfg = {"crystal_structure": {
               "lattice_vectors": NEEL_LAT,
               "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0], "spin_S": 1.0},
                            {"label": "B", "pos": [0.5, 0, 0], "spin_S": 1.0}]},
           "interactions": {"heisenberg": [{"pair": p, "rij_offset": o, "value": 1.0}
                                           for p, o in nn]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="polfr_neel", hamiltonian_params=[])
    B = 2 * np.pi * np.linalg.inv(np.array(NEEL_LAT, float)).T
    return calc, [np.array(q) @ B for q in NEEL_HS]


def test_collinear_chirality_cancels_only_in_the_band_sum():
    """A collinear magnet is achiral as an OBSERVABLE, but not band by band: its two
    magnons are degenerate and oppositely handed, so how the chirality is split
    between them depends on the arbitrary eigenvector basis inside that degenerate
    subspace -- Sunny puts +/-0.168 on the two bands, pyMagCalc puts ~0 on each, and
    both are correct. What is physical, and what is asserted, is that the SUM
    vanishes for any polarization axis.

    (With P || q it does vanish per band too, because P is then perpendicular to the
    chiral vector; that narrower statement is pinned in test_polarized.py.)
    """
    calc, qs = _neel_chain()
    for P in ([1, 0, 0], [0, 1, 0], [0.3, -0.5, 0.81]):
        _, sfp = _sorted(calc, qs, {"polarization": P, "channel": "sf+"})
        _, sfm = _sorted(calc, qs, {"polarization": P, "channel": "sf-"})
        assert sfp.sum(axis=1) == pytest.approx(sfm.sum(axis=1), abs=1e-9)


# --------------------------------------------------------------------------
def test_bad_specs_raise_with_an_actionable_message():
    calc = _model(SCREW)
    with pytest.raises(ValueError, match="needs .polarization"):
        contract_cross_section(np.zeros((3, 3, 2), complex), np.array([1.0, 0, 0]),
                               {"channel": "sf"})
    with pytest.raises(ValueError, match="channel must be"):
        contract_cross_section(np.zeros((3, 3, 2), complex), np.array([1.0, 0, 0]),
                               {"polarization": [0, 0, 1], "channel": "nonsense"})
    with pytest.raises(ValueError, match="either `normal` or both"):
        contract_cross_section(np.zeros((3, 3, 2), complex), np.array([1.0, 0, 0]),
                               {"bm": {"u": [1, 0, 0]}, "component": "11"})
    with pytest.raises(ValueError, match="component must be"):
        contract_cross_section(np.zeros((3, 3, 2), complex), np.array([1.0, 0, 0]),
                               {"bm": {"normal": [0, 0, 1]}, "component": "q"})
    with pytest.raises(ValueError, match="parallel"):
        contract_cross_section(np.zeros((3, 3, 2), complex), np.array([1.0, 0, 0]),
                               {"bm": {"u": [1, 0, 0], "v": [2, 0, 0]},
                                "component": "11"})
    # and the runner-facing validator rejects it before any pool worker sees it
    with pytest.raises(ValueError):
        calc.calculate_sqw(_q([0.1]), cross_section={"polarization": [0, 0, 0],
                                                     "channel": "sf"})


def test_domains_refuse_a_lab_anchored_polarization():
    """A polarization axis is fixed in the LAB, so it is not the same measurement on
    a rotated twin -- exactly why the xx/zz components already refuse."""
    calc = _model(SCREW)
    with pytest.raises(ValueError, match="rotation-covariant"):
        calc.calculate_sqw(_q(HS), domains={"axis": [0, 0, 1], "n_fold": 3},
                           cross_section={"polarization": [0, 0, 1],
                                          "channel": "sf"})
