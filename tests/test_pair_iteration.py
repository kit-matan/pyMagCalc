"""`iter_pair_matrices` is the double loop over `_pair_matrix` it replaced (2026-10-28).

Every real-space bond builder -- SCGA's `_exchange_cache`, `thermal_mc.build_supercell`,
the entangled-unit and dimer-series builders -- used to call `_pair_matrix` for every
(site, image) pair: 442k pairs and 6.9 s at 128 sites, for 768 bonds, almost all of it
SymPy's `is_zero_matrix` on the one shared zero entry. The iterator finds the candidate
pairs cheaply and then lets `_pair_matrix` decide each one, so its contract is exact
equality with the loop: same pairs, same order, same matrices bit for bit. That is what
these tests pin, against the loop itself, which is the definition.
"""
import os

import numpy as np
import pytest
import sympy as sp
import yaml

from magcalc.sun.entangled import _pair_matrix, iter_pair_matrices


def _loop(Jex, DM, Kex, n, no):
    out = []
    for i in range(n):
        for j in range(no):
            M = _pair_matrix(Jex, DM, Kex, i, j)
            if np.any(M):
                out.append((i, j, M))
    return out


def _same(a, b):
    return (len(a) == len(b)
            and all(x[0] == y[0] and x[1] == y[1] and np.array_equal(x[2], y[2])
                    for x, y in zip(a, b)))


def _tables(n=3, no=9):
    """Every entry kind the model builder produces: exact and float exchange, DM
    vectors, full and diagonal Kex, None, the SHARED zero default, and a pair
    whose terms cancel exactly (J = 2, Kex = -2 I), which must be dropped."""
    Jex = sp.zeros(n, no)
    Jex[0, 1] = sp.Rational(3, 2)
    Jex[0, 4] = 2
    Jex[1, 2] = sp.sqrt(3) / 3
    Jex[2, 7] = 0.25
    zero_vec, zero_mat = sp.Matrix([0, 0, 0]), sp.zeros(3, 3)
    DM = [[zero_vec] * no for _ in range(n)]
    Kex = [[zero_mat] * no for _ in range(n)]
    DM[1][5] = sp.Matrix([0, sp.Rational(1, 5), -0.3])          # DM only, no J
    DM[0][1] = sp.Matrix([0.1, 0, 0])
    Kex[0][4] = -2 * sp.eye(3)                                    # cancels J exactly
    Kex[2][3] = sp.Matrix([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0.1]])  # Kex only
    Kex[2][8] = [0.0, 0.0, 0.7]                                   # diagonal, a list
    DM[2][0] = None
    Kex[1][0] = None
    return Jex, DM, Kex, n, no


def test_iterator_equals_the_double_loop_entry_for_entry():
    Jex, DM, Kex, n, no = _tables()
    want = _loop(Jex, DM, Kex, n, no)
    got = list(iter_pair_matrices(Jex, DM, Kex, n, no))
    assert _same(got, want)
    pairs = [(i, j) for i, j, _ in got]
    assert (0, 4) not in pairs                   # the cancelling pair is dropped
    assert {(1, 5), (2, 3), (2, 8)} <= set(pairs)  # DM-only / Kex-only pairs kept
    assert pairs == sorted(pairs)                # row-major, as the loop


def test_numpy_exchange_table_is_handled_like_the_sympy_one():
    Jex, DM, Kex, n, no = _tables()
    Jnp = np.array(Jex.evalf(), dtype=float)
    assert _same(list(iter_pair_matrices(Jnp, DM, Kex, n, no)),
                 _loop(Jnp, DM, Kex, n, no))


def test_an_empty_model_gives_no_bonds():
    Jex = sp.zeros(2, 6)
    z = sp.Matrix([0, 0, 0])
    DM = [[z] * 6 for _ in range(2)]
    Kex = [[sp.zeros(3, 3)] * 6 for _ in range(2)]
    assert list(iter_pair_matrices(Jex, DM, Kex, 2, 6)) == []


def test_scga_bond_list_equals_the_loop_on_a_dm_model():
    """End to end on KFe3J (J1 + J2 + the allowed DM on the kagome bond): the bond
    list `_exchange_cache` builds is the one the loop builds."""
    from magcalc import scga
    from magcalc.generic_model import GenericSpinModel

    here = os.path.dirname(__file__)
    path = os.path.join(here, "..", "examples", "materials", "KFe3J", "config_kfe3j.yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f)
    params = [cfg["parameters"][k] for k in ("Dy", "Dz", "J1", "J2")] + [0.0, [0, 0, 1]]
    model = GenericSpinModel(cfg)
    Jex, DM, Kex = model.spin_interactions(params)
    n, no = len(model.atom_pos()), len(model.atom_pos_ouc())
    want = _loop(Jex, DM, Kex, n, no)
    assert any(np.any(M - M.T) for _, _, M in want)       # the DM is really there
    cache = scga._exchange_cache(model, params)
    got = [(i, jc, M) for i, jc, _R, M in cache["bonds"]]
    assert len(got) == len(want)
    for (i, jc, M), (wi, wj, wM) in zip(got, want):
        assert i == wi and jc == wj % n and np.array_equal(M, wM)
