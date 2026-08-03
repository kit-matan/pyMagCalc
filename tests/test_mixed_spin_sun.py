"""Mixed-spin SU(N): sites with different S in one cell (Gap 4 #24a).

`SUNModel` assumed a uniform N = 2S+1 and laid out its Nambu blocks as `i * M`. Any
model with two different spins raised. Now each site carries its own M_i = N_i - 1
and the blocks are addressed through an offsets table.

VALIDATION ORDER MATTERS HERE, and follows GAP4_PLAN: the decoupled-sublattice
identity below is written so it passes on the OLD uniform-N code (equal spins) before
the refactor, then must keep passing for unequal spins after it. A refactor of
validated core code that is only checked afterwards proves nothing about what it
broke.

The identity: two sublattices with NO coupling between them are two independent
problems, so the spectrum and the intensities of the combined model must be exactly
the union of the two computed separately. Every block-offset, normalization or
Bogoliubov-metric slip breaks it, and it needs no oracle at all.
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel

LAT = [[6.0, 0, 0], [0, 6.0, 0], [0, 0, 12.0]]
HS = (0.13, 0.31, 0.5)


def _cfg(spins, couple_ab):
    """Two AFM chains stacked along z, optionally coupled to each other.

    A and B run along x at z = 0 and z = 1/2. `couple_ab` switches on the interchain
    bond; with it off the two chains are literally independent problems.
    """
    atoms = [{"label": "A0", "pos": [0.0, 0.0, 0.0], "spin_S": spins[0]},
             {"label": "A1", "pos": [0.5, 0.0, 0.0], "spin_S": spins[0]},
             {"label": "B0", "pos": [0.0, 0.0, 0.5], "spin_S": spins[1]},
             {"label": "B1", "pos": [0.5, 0.0, 0.5], "spin_S": spins[1]}]
    bonds = []
    for lab, J in (("A", 1.0), ("B", 0.7)):
        for p, o in ((["%s0" % lab, "%s1" % lab], [0, 0, 0]),
                     (["%s1" % lab, "%s0" % lab], [0, 0, 0]),
                     (["%s1" % lab, "%s0" % lab], [1, 0, 0]),
                     (["%s0" % lab, "%s1" % lab], [-1, 0, 0])):
            bonds.append({"pair": p, "rij_offset": o, "value": J})
    if couple_ab:
        for p, o in ((["A0", "B0"], [0, 0, 0]), (["B0", "A0"], [0, 0, 0])):
            bonds.append({"pair": p, "rij_offset": o, "value": 0.3})
    return {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
            "interactions": {"heisenberg": bonds},
            "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[0, 0, 1], [0, 0, -1],
                                                  [0, 0, 1], [0, 0, -1]]},
            "calculation": {"mode": "SUN"}, "tasks": {}}


def _single(S, J, tag):
    """One AFM chain on its own, in the same cell geometry."""
    atoms = [{"label": "A0", "pos": [0.0, 0.0, 0.0], "spin_S": S},
             {"label": "A1", "pos": [0.5, 0.0, 0.0], "spin_S": S}]
    bonds = [{"pair": p, "rij_offset": o, "value": J}
             for p, o in ((["A0", "A1"], [0, 0, 0]), (["A1", "A0"], [0, 0, 0]),
                          (["A1", "A0"], [1, 0, 0]), (["A0", "A1"], [-1, 0, 0]))]
    cfg = {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
           "interactions": {"heisenberg": bonds},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"mode": "SUN"}, "tasks": {}}
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[])


def _build(cfg):
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[])


def _qs():
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return [np.array([h, 0, 0]) @ B for h in HS]


def _bands(mdl):
    return np.sort(np.array([np.sort(np.real(mdl.dispersion(q))) for q in _qs()]),
                   axis=1)


def _sqw(mdl, tol=1e-6):
    """(energy, intensity) per DEGENERATE MULTIPLET at each q.

    Third time this pattern has been needed (see also test_domains_sun.py and
    test_polarization_frames.py): inside a degenerate subspace the eigenvector basis
    is arbitrary, so how the weight is split between the individual bands is not an
    observable -- here each AFM chain's two magnons are degenerate and the two codes
    paths split them differently (0.11439/0.09270 vs 0.10355/0.10355, same sum). The
    multiplet sum is the physical quantity.
    """
    out = []
    for q in _qs():
        w, i = mdl.structure_factor(q)
        o = np.argsort(w)
        w, i = np.asarray(w)[o], np.real(i)[o]
        groups = []
        for e, v in zip(w, i):
            if groups and abs(e - groups[-1][0]) <= tol:
                groups[-1][1] += v
            else:
                groups.append([e, v])
        out.append(groups)
    return out


def _merge_sqw(a, b, tol=1e-6):
    """Combine two per-q multiplet lists, merging multiplets that coincide."""
    out = []
    for ga, gb in zip(a, b):
        merged = sorted(ga + gb, key=lambda g: g[0])
        acc = []
        for e, v in merged:
            if acc and abs(e - acc[-1][0]) <= tol:
                acc[-1][1] += v
            else:
                acc.append([e, v])
        out.append(acc)
    return out


@pytest.mark.parametrize("spins", [(1.0, 1.0), (0.5, 1.0), (1.0, 1.5), (0.5, 1.5)])
def test_decoupled_sublattices_are_the_union_of_two_independent_problems(spins):
    """The load-bearing identity. Equal spins exercised the old code path too; the
    unequal ones are what #24a adds."""
    both = _build(_cfg(spins, couple_ab=False))
    a = _single(spins[0], 1.0, "a")
    b = _single(spins[1], 0.7, "b")

    assert both.energy_per_site() == pytest.approx(
        0.5 * (a.energy_per_site() + b.energy_per_site()), abs=1e-10)

    got = _bands(both)
    want = np.sort(np.concatenate([_bands(a), _bands(b)], axis=1), axis=1)
    assert got == pytest.approx(want, abs=1e-9)


@pytest.mark.parametrize("spins", [(1.0, 1.0), (0.5, 1.5)])
def test_decoupled_sublattice_intensities_also_add(spins):
    """Energies alone would not catch a per-site normalization slip -- the mixed-spin
    prefactor bug in the dipole engine was exactly that, and it was a 60% error."""
    both = _build(_cfg(spins, couple_ab=False))
    a = _single(spins[0], 1.0, "a")
    b = _single(spins[1], 0.7, "b")
    got = _sqw(both)
    want = _merge_sqw(_sqw(a), _sqw(b))
    assert [len(g) for g in got] == [len(g) for g in want]
    assert np.array(got) == pytest.approx(np.array(want), abs=1e-8)


def test_mixed_spin_model_builds_and_has_per_site_boson_counts():
    """S = 1/2 gives 1 boson, S = 1 gives 2, S = 3/2 gives 3 -- the whole point."""
    mdl = _build(_cfg((0.5, 1.5), couple_ab=False))
    assert sorted(mdl.Ns) == [2, 2, 4, 4]
    assert sorted(mdl.Ms) == [1, 1, 3, 3]
    assert mdl.D == sum(mdl.Ms)
    assert len(_bands(mdl)[0]) == mdl.D


def test_coupled_mixed_spin_chain_is_not_just_the_decoupled_answer():
    """Guards the guard: if the interchain bond were silently dropped, the identity
    test above would pass for the wrong reason."""
    dec = _bands(_build(_cfg((0.5, 1.5), couple_ab=False)))
    cou = _bands(_build(_cfg((0.5, 1.5), couple_ab=True)))
    assert np.abs(cou - dec).max() > 1e-3


def test_uniform_spin_models_are_bit_identical_to_before():
    """The refactor must be inert where it does not apply: a uniform-S model must
    give exactly what the uniform-N code gave (these values come from the
    Sunny-pinned Neel chain in test_sun_missing_terms.py)."""
    mdl = _single(1.0, 1.0, "u")
    assert mdl.energy_per_site() == pytest.approx(-1.0, abs=1e-12)
    b = _bands(mdl)
    assert b[0] == pytest.approx([0.7942958, 0.7942958, 4.0, 4.0], abs=1e-6)
