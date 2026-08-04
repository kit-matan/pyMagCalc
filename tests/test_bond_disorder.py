"""Bond disorder in LSWT (Gap 4 #16b) -- Sunny's `set_exchange_at!` + KPM.

Gap 4 #16a gave the CLASSICAL samplers vacancies and open boundaries. This is the
other half: randomizing the exchange on every bond of a large supercell, which is
what Sunny's disorder tutorial (`09_Disorder_KPM.jl`) needs, and which pairs with the
existing KPM solver (`sun/kpm.py`) because that needs no eigensolve.

THE CORRECTNESS CATCH is the bond bookkeeping. pyMagCalc lists every bond in BOTH
directions -- (i, j, dr, J) and (j, i, -dr, J^T) -- which is how `H = (1/2)
sum_ordered` is encoded. Perturbing the two independently makes H(q) non-Hermitian
and yields complex "energies" that the ground-state guard would report as an
INSTABILITY, i.e. as physics rather than as the bookkeeping error it is. So the
spectrum being real is asserted directly, not assumed.
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel, apply_bond_disorder

LAT = [[3.0, 0, 0], [-1.5, 3 * np.sqrt(3) / 2, 0], [0, 0, 10.0]]
OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0])


def _cfg():
    return {"crystal_structure": {"lattice_vectors": LAT,
                                  "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                                "spin_S": 0.5}]},
            "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": o,
                                             "value": 1.0} for o in OFFS]},
            "magnetic_structure": {"type": "pattern",
                                   "pattern_type": "ferromagnetic",
                                   "direction": [0, 0, 1]},
            "parameters": {}, "parameter_order": [],
            "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}


def _model(n=4):
    """A supercell. Disorder on the CHEMICAL cell is not disorder -- it is a
    different clean model repeated periodically."""
    return SUNModel.from_generic_model(
        GenericSpinModel(copy.deepcopy(_cfg())), params=[],
        supercell=[[n, 0, 0], [0, n, 0], [0, 0, 1]])


def _q():
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return np.array([0.25, 0.25, 0]) @ B


def _bands(m):
    return np.sort(np.real(m.dispersion(_q())))


def test_zero_disorder_is_bit_identical():
    """The sigma -> 0 limit, exactly -- not merely close."""
    clean = _bands(_model())
    got = _bands(apply_bond_disorder(_model(), 0.0, seed=1))
    assert np.abs(got - clean).max() == 0.0


def test_hermiticity_is_preserved():
    """THE bookkeeping check. Both directions of each bond must get the SAME draw;
    independent draws break Hermiticity and the imaginary parts masquerade as an
    instability."""
    for seed in range(4):
        m = apply_bond_disorder(_model(), 0.25, seed=seed)
        ev = np.linalg.eigvals(m.hamiltonian(_q()))
        assert np.abs(np.imag(ev)).max() < 1e-10, f"seed {seed}"


def test_disorder_broadens_the_spectrum():
    """Physics: randomizing the couplings must spread the bands, and more disorder
    must spread them more."""
    clean = _bands(_model()).std()
    widths = {}
    for sig in (0.05, 0.2, 0.4):
        widths[sig] = np.mean([_bands(apply_bond_disorder(_model(), sig, seed=s)).std()
                               for s in range(4)])
    assert widths[0.05] > clean
    assert widths[0.2] > widths[0.05]
    assert widths[0.4] > widths[0.2]


def test_each_bond_pair_gets_one_draw():
    """A bond and its reverse must carry the same perturbation, so J_ji stays J_ij^T
    -- checked on the matrices themselves rather than through the spectrum."""
    m = apply_bond_disorder(_model(), 0.3, seed=2)
    table = {}
    for (i, j, dr, J) in m.bonds:
        table[(i, j, tuple(np.round(dr, 8)))] = J
    checked = 0
    for (i, j, dr), J in table.items():
        rev = table.get((j, i, tuple(np.round(-np.asarray(dr), 8))))
        if rev is None:
            continue
        assert rev == pytest.approx(np.asarray(J).T, abs=1e-12)
        checked += 1
    assert checked > 0


def test_results_self_average_across_seeds():
    """A single realization is not an answer; the seed-to-seed spread must be small
    compared with the effect being reported."""
    vals = [_bands(apply_bond_disorder(_model(6), 0.2, seed=s)).std()
            for s in range(5)]
    vals = np.array(vals)
    assert vals.std() / vals.mean() < 0.05


def test_bad_kind_raises():
    with pytest.raises(ValueError, match="relative.*absolute"):
        apply_bond_disorder(_model(), 0.1, kind="gaussian")
