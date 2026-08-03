"""Twin/domain averaging in SU(N) and entangled mode (Gap 4 #23).

Both engines used to raise NotImplementedError. The dipole engine has always done
this by evaluating each domain at R^T q and averaging, which is not engine-specific:
a crystallite rotated by R scatters at q exactly as the unrotated one does at R^T q.
All three engines now share `numerical.sqw_domain_average`.

VALIDATION. Asserting "the average equals the average over rotated q" would just
restate the implementation. So the identity here is the PHYSICAL one underneath it,
established without touching the domain code at all:

    S_{rotated crystal}(q)  ==  S_{crystal}(R^T q)

built by rotating the lattice vectors and the spin directions explicitly. Only then
is the domain average required to equal the weighted sum of the *separately built*
twin models. If the rotation convention were transposed or the weights misapplied,
the second check fails even though the first still passes.
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.linalg import rotation_matrix
from magcalc.sun.adapter import SUNCalculator
from magcalc.sun.entangled import EntangledCalculator

LAT = [[6.0, 0, 0], [0, 7.0, 0], [0, 0, 8.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
# generic q directions: not along the rotation axis, not along a lattice vector
QS = np.array([[0.31, 0.17, 0.0], [0.44, -0.23, 0.11], [0.13, 0.41, -0.07]])


def _rotate(v, R):
    return list(np.asarray(R, float) @ np.asarray(v, float))


def _cfg(mode, R=None, S=1.0):
    """`R` rotates the whole crystal rigidly: lattice vectors AND spin directions.

    SU(N): a Neel chain, spins in a generic direction so the rotation bites.
    Entangled: strong intra-dimer J with a weak inter-dimer J', i.e. a singlet
    ground state with a dispersing triplon -- the regime the engine is exact in.
    Its reference is the unit ground state, so the only thing to rotate is the
    lattice (there are no classical directions to speak of).
    """
    lat = [list(np.asarray(R, float) @ np.asarray(v, float)) for v in LAT] \
        if R is not None else [list(v) for v in LAT]
    S = 0.5 if mode == "entangled" else S
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S, "ion": "Cu2+"},
             {"label": "B", "pos": [0.3, 0, 0], "spin_S": S, "ion": "Cu2+"}]
    if mode == "entangled":
        # intra-dimer A-B in cell, inter-dimer B-A across the cell boundary
        bonds = [{"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": 10.0},
                 {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": 10.0},
                 {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": 2.0},
                 {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": 2.0}]
        mag = {"type": "pattern", "pattern_type": "ferromagnetic",
               "direction": [0, 0, 1]}
    else:
        bonds = [{"pair": p, "rij_offset": o, "value": 1.0} for p, o in NN]
        dirs = [[0.0, 0.6, 0.8], [0.0, -0.6, -0.8]]
        if R is not None:
            dirs = [_rotate(d, R) for d in dirs]
        mag = {"type": "pattern", "pattern_type": "generic", "directions": dirs}
    cfg = {"crystal_structure": {"lattice_vectors": lat, "atoms_uc": atoms},
           "interactions": {"heisenberg": bonds},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": mag,
           "calculation": {"mode": mode, "on_imaginary": "off"}, "tasks": {}}
    if mode == "entangled":
        cfg["units"] = [["A", "B"]]
    return cfg


def _calc(mode, R=None, S=1.0):
    cfg = _cfg(mode, R, S)
    m = GenericSpinModel(copy.deepcopy(cfg))
    if mode == "entangled":
        return EntangledCalculator(m, cfg, [])
    return SUNCalculator(m, cfg, hamiltonian_params=[])


def _sorted_sqw(calc, qs, **kw):
    r = calc.calculate_sqw(np.asarray(qs, float), **kw)
    E, I = np.real(r.energies), np.real(r.intensities)
    o = np.argsort(E, axis=1)
    return np.take_along_axis(E, o, 1), np.take_along_axis(I, o, 1)


def _grouped(E, I, tol=1e-6):
    """Collapse each DEGENERATE multiplet into (energy, summed intensity).

    Both models here are symmetric enough to have exactly degenerate modes (the
    Neel chain's two sublattices; the dimer's three-fold triplon), and inside a
    degenerate subspace the eigenvector basis -- hence how the weight is split
    between the individual bands -- is arbitrary. Summing over the multiplet is the
    basis-independent observable, and still band-resolved, unlike a per-q total.
    """
    out = []
    for e_row, i_row in zip(np.atleast_2d(E), np.atleast_2d(I)):
        groups = []
        for e, i in zip(e_row, i_row):
            if groups and abs(e - groups[-1][0]) <= tol:
                groups[-1][1] += i
            else:
                groups.append([e, i])
        out.append(groups)
    lens = {len(g) for g in out}
    return np.array(out) if len(lens) == 1 else out


MODES = ["SUN", "entangled"]


@pytest.mark.parametrize("mode", MODES)
def test_structure_factor_is_rotationally_covariant(mode):
    """The physics the domain average rests on, checked WITHOUT the domain code:
    rotating the crystal by R and measuring at q is the same experiment as leaving
    it and measuring at R^T q."""
    R = rotation_matrix([0, 0, 1], 40.0)
    plain = _calc(mode)
    rotated = _calc(mode, R)
    E_rot, I_rot = _sorted_sqw(rotated, QS)
    E_ref, I_ref = _sorted_sqw(plain, [R.T @ q for q in QS])
    assert E_rot == pytest.approx(E_ref, abs=1e-8)
    assert _grouped(E_rot, I_rot) == pytest.approx(_grouped(E_ref, I_ref), abs=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_domain_average_equals_the_explicit_twin_average(mode):
    """Two 50/50 twins: the averaged result must be the concatenation of the two
    SEPARATELY BUILT crystals' spectra, each weighted 1/2."""
    angle = 40.0
    R = rotation_matrix([0, 0, 1], angle)
    plain = _calc(mode)
    domains = [{"axis": [0, 0, 1], "angle": 0.0, "weight": 1.0},
               {"axis": [0, 0, 1], "angle": angle, "weight": 1.0}]
    r = plain.calculate_sqw(QS, domains=domains)
    E_avg, I_avg = np.real(r.energies), np.real(r.intensities)

    E0, I0 = _sorted_sqw(plain, QS)
    E1, I1 = _sorted_sqw(_calc(mode, R), QS)
    n = E0.shape[1]
    assert E_avg.shape[1] == 2 * n, "domains must concatenate along the mode axis"

    # Columns come back DOMAIN-MAJOR, so split on that boundary and collapse each
    # domain's degenerate multiplets separately -- grouping the concatenated
    # spectrum would merge modes that merely happen to coincide across domains
    # (the flat Delta-m modes sit at the same energy in both).
    blocks = []
    for d in range(2):
        Eb, Ib = E_avg[:, d * n:(d + 1) * n], I_avg[:, d * n:(d + 1) * n]
        o = np.argsort(Eb, axis=1)
        blocks.append(_grouped(np.take_along_axis(Eb, o, 1),
                               np.take_along_axis(Ib, o, 1)))
    got = np.sort(np.concatenate(blocks, axis=1), axis=1)
    want = np.sort(np.concatenate(
        [_grouped(E0, I0 * 0.5), _grouped(E1, I1 * 0.5)], axis=1), axis=1)
    assert got == pytest.approx(want, abs=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_identity_domain_is_a_no_op(mode):
    """A single 0-degree domain must return exactly the undomained result -- not a
    numerically-close one."""
    calc = _calc(mode)
    base = _sorted_sqw(calc, QS)
    same = _sorted_sqw(calc, QS, domains=[{"axis": [0, 0, 1], "angle": 0.0}])
    assert same[0] == pytest.approx(base[0], abs=0)
    assert same[1] == pytest.approx(base[1], abs=0)


@pytest.mark.parametrize("mode", MODES)
def test_weights_are_normalized_and_respected(mode):
    """Unequal weights: a 3:1 twin must weight the two crystals 0.75 / 0.25, and the
    total intensity must be the weighted mean of the two totals (weights normalize,
    they do not scale the result)."""
    angle = 40.0
    R = rotation_matrix([0, 0, 1], angle)
    plain = _calc(mode)
    E_avg, I_avg = _sorted_sqw(plain, QS, domains=[
        {"axis": [0, 0, 1], "angle": 0.0, "weight": 3.0},
        {"axis": [0, 0, 1], "angle": angle, "weight": 1.0}])
    tot0 = _sorted_sqw(plain, QS)[1].sum(axis=1)
    tot1 = _sorted_sqw(_calc(mode, R), QS)[1].sum(axis=1)
    assert I_avg.sum(axis=1) == pytest.approx(0.75 * tot0 + 0.25 * tot1, abs=1e-8)


@pytest.mark.parametrize("mode", MODES)
def test_lab_frame_components_still_refuse(mode):
    """A lab-frame component of a rotated crystal is NOT the same component of the
    unrotated one, so rotating q alone would be silently wrong. That guard predates
    this work and must survive being shared."""
    calc = _calc(mode)
    with pytest.raises(ValueError, match="rotation-covariant"):
        calc.calculate_sqw(QS, domains={"axis": [0, 0, 1], "n_fold": 3},
                           cross_section="zz")
