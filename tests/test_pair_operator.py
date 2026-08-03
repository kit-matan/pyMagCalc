"""Arbitrary two-site couplings in SU(N) (Gap 4 #21).

Sunny's `set_pair_coupling!`. The engine already accepted generalized operator-pair
couplings (that is how biquadratic is expanded); what was missing was the front end
that turns an arbitrary Hermitian two-site operator into that form:

    D = sum_k A_k (x) B_k          (`operators.decompose_pair_operator`)

Config:

    interactions:
      pair_operator:
      - {pair: [A, B], rij_offset: [0, 0, 0], poly: [0, J, B2, B3]}   # sum_n c_n (Si.Sj)^n
      - {pair: [A, B], rij_offset: [0, 0, 0], matrix: [[...]]}        # explicit

Two independent oracles. The internal one is exact and costs nothing: a biquadratic
entered through the GENERAL path must reproduce the dedicated `interactions.biquadratic`
path to machine precision -- if the decomposition, the operator bookkeeping or the
1/2 convention were wrong, that fails. The external one is Sunny `set_pair_coupling!`
on a cubic polynomial coupling, which has no dedicated path on either side.
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel
from magcalc.sun.operators import (decompose_pair_operator, dot_product_operator,
                                   hermitian_basis, spin_matrices)

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = (0.13, 0.31, 0.5)


def _model(S, interactions):
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S},
             {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]
    cfg = {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
           "interactions": copy.deepcopy(interactions),
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"mode": "SUN"}, "tasks": {}}
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)), params=[])


def _bands(mdl):
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    qs = [np.array([h, 0, 0]) @ B for h in HS]
    return np.sort(np.array([np.sort(np.real(mdl.dispersion(q))) for q in qs]), axis=1)


# --------------------------------------------------------------------------
# The decomposition itself
# --------------------------------------------------------------------------
@pytest.mark.parametrize("S", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("power", [1, 2, 3])
def test_decomposition_is_exact_and_hermitian(S, power):
    """D = sum_k A_k (x) B_k exactly, with every factor Hermitian.

    Hermiticity is the part that is easy to get subtly wrong: a plain SVD of the
    reshaped operator gives non-Hermitian factors whenever singular values are
    degenerate, which for S_i.S_j at s = 1/2 is ALL of them. Expanding in a Hermitian
    basis first makes the coefficient matrix real and sidesteps it.
    """
    n = int(round(2 * S + 1))
    d = dot_product_operator(S, S)
    D = np.linalg.matrix_power(d, power)
    terms = decompose_pair_operator(D, n, n)
    recon = sum(np.kron(a, b) for a, b in terms)
    assert recon == pytest.approx(D, abs=1e-12)
    for a, b in terms:
        assert a == pytest.approx(a.conj().T, abs=1e-13)
        assert b == pytest.approx(b.conj().T, abs=1e-13)


def test_schmidt_rank_is_minimal_for_a_dot_product():
    """S_i.S_j has Schmidt rank exactly 3 (one term per spin component) at any S --
    a full Hermitian-basis expansion would give N^2 terms instead, and the engine
    cost is quadratic in the operator count."""
    for S in (0.5, 1.0, 1.5, 2.0):
        n = int(round(2 * S + 1))
        assert len(decompose_pair_operator(dot_product_operator(S, S), n, n)) == 3


def test_hermitian_basis_is_orthonormal():
    for n in (2, 3, 4):
        G = hermitian_basis(n)
        assert len(G) == n * n
        gram = np.array([[np.trace(a @ b) for b in G] for a in G])
        assert gram == pytest.approx(np.eye(n * n), abs=1e-12)


def test_decomposition_rejects_a_non_hermitian_operator():
    S = 1.0
    n = int(round(2 * S + 1))
    D = dot_product_operator(S, S).astype(complex)
    D[0, 1] += 0.5j                                  # break Hermiticity
    with pytest.raises(ValueError, match="Hermitian"):
        decompose_pair_operator(D, n, n)


# --------------------------------------------------------------------------
# Internal identity: the general path must reproduce the dedicated one
# --------------------------------------------------------------------------
@pytest.mark.parametrize("S", [1.0, 1.5])
def test_biquadratic_via_the_general_path_matches_the_dedicated_one(S):
    """THE load-bearing check. `poly: [0, J, B]` is J(Si.Sj) + B(Si.Sj)^2, which is
    exactly what `heisenberg` + `biquadratic` build by a completely different route
    (9 hardcoded S^a S^b products vs a Schmidt decomposition). Energies and every
    band must agree to machine precision."""
    J, B = 1.0, -0.4
    dedicated = _model(S, {
        "heisenberg": [{"pair": p, "rij_offset": o, "value": J} for p, o in NN],
        "biquadratic": [{"pair": p, "rij_offset": o, "value": B} for p, o in NN]})
    general = _model(S, {
        "pair_operator": [{"pair": p, "rij_offset": o, "poly": [0.0, J, B]}
                          for p, o in NN]})
    assert general.energy_per_site() == pytest.approx(dedicated.energy_per_site(),
                                                      abs=1e-12)
    assert _bands(general) == pytest.approx(_bands(dedicated), abs=1e-10)


def test_operator_registry_is_deduplicated():
    """Every bond of an orbit yields the same factors, and the H(q) assembly is
    quadratic in the operator count -- so they must be registered once, not per
    bond (4 bonds x 9 terms x 2 factors would be 75 operators instead of 21)."""
    mdl = _model(1.0, {"pair_operator": [
        {"pair": p, "rij_offset": o, "poly": [0.0, 1.0, -0.4]} for p, o in NN]})
    assert mdl.n_ops <= 24


def test_explicit_matrix_and_poly_agree():
    """The two config spellings of the same operator must give the same model."""
    S = 1.0
    d = dot_product_operator(S, S)
    mat = (1.0 * d - 0.4 * (d @ d)).real.tolist()
    a = _model(S, {"pair_operator": [{"pair": p, "rij_offset": o,
                                      "poly": [0.0, 1.0, -0.4]} for p, o in NN]})
    b = _model(S, {"pair_operator": [{"pair": p, "rij_offset": o, "matrix": mat}
                                     for p, o in NN]})
    assert _bands(b) == pytest.approx(_bands(a), abs=1e-10)


# --------------------------------------------------------------------------
# External oracle: Sunny set_pair_coupling!
# --------------------------------------------------------------------------
# Sunny 0.8.1, :SUN mode, Neel chain:
#   fn = (Si, Sj) -> (d = Si'*Sj; c1*d + c2*d^2 + c3*d^3)
#   set_pair_coupling!(sys, fn, Bond(1,2,[0,0,0])); ... Bond(2,1,[1,0,0])
SUNNY_PAIR_OP = {
    (1.0, 1.0, -0.4, 0.15): {
        "E": -2.2500000000,
        "bands": [[1.4694472, 1.4694472, 5.8608143, 5.8608143],
                  [3.0601981, 3.0601981, 5.9481730, 5.9481730],
                  [3.7000000, 3.7000000, 6.0000000, 6.0000000]]},
    (1.0, 1.0, 0.3, 0.05): {
        "E": -0.5500000000,
        "bands": [[0.6751514, 0.6751514, 3.7822264, 3.7822264],
                  [1.4060370, 1.4060370, 3.7933429, 3.7933429],
                  [1.7000000, 1.7000000, 3.8000000, 3.8000000]]},
    (1.5, 1.0, -0.2, 0.03): {
        "E": -4.3748437500,
        "bands": [[2.2138513, 2.2138513, 9.3248869, 9.3248869, 11.9055466, 11.9055466],
                  [4.6104573, 4.6104573, 9.4056202, 9.4056202, 11.9071579, 11.9071579],
                  [5.5743750, 5.5743750, 9.4537500, 9.4537500, 11.9081250, 11.9081250]]},
}


@pytest.mark.parametrize("key", sorted(SUNNY_PAIR_OP))
def test_cubic_polynomial_coupling_matches_sunny(key):
    """A coupling with no dedicated path on either side: c1(Si.Sj) + c2(Si.Sj)^2 +
    c3(Si.Sj)^3. All three coefficient sets keep the Neel state a genuine energy
    minimum -- Sunny refuses to compute the ones that do not, which is how the
    unstable candidates were weeded out."""
    S, c1, c2, c3 = key
    ref = SUNNY_PAIR_OP[key]
    mdl = _model(S, {"pair_operator": [{"pair": p, "rij_offset": o,
                                        "poly": [0.0, c1, c2, c3]} for p, o in NN]})
    assert mdl.energy_per_site() == pytest.approx(ref["E"], abs=1e-9)
    assert _bands(mdl) == pytest.approx(np.array(ref["bands"]), abs=2e-6)


# --------------------------------------------------------------------------
def test_asymmetric_operator_is_refused():
    """Both bond directions are listed by the user and the engine supplies the 1/2,
    which is only consistent for an operator symmetric under site exchange. An
    asymmetric one would silently realize a different Hamiltonian on the reverse
    bond -- the same class of bug as the ref_pair orientation trap."""
    S = 1.0
    n = int(round(2 * S + 1))
    Sx, Sy, Sz = spin_matrices(S)
    D = np.kron(Sz, np.eye(n)) @ np.kron(np.eye(n), Sx)   # Sz_i Sx_j: not symmetric
    D = 0.5 * (D + D.conj().T)
    with pytest.raises(ValueError, match="symmetric under exchanging"):
        _model(S, {"pair_operator": [{"pair": p, "rij_offset": o,
                                      "matrix": D.tolist()} for p, o in NN]})


def test_entry_matching_no_bonds_is_refused():
    """A term that matches nothing must not vanish silently.

    NB a large `rij_offset` does NOT test this: the over-cell is sized from the
    offsets in the config, so [5, 0, 0] happily finds a (30 A) bond. A distance that
    matches nothing does.
    """
    with pytest.raises(ValueError, match="matched no bonds"):
        _model(1.0, {"pair_operator": [{"pair": ["A", "B"], "distance": 123.0,
                                        "poly": [0.0, 1.0]}]})


def test_missing_operator_spec_is_refused():
    with pytest.raises(ValueError, match="needs `matrix` or `poly`"):
        _model(1.0, {"pair_operator": [{"pair": p, "rij_offset": o}
                                       for p, o in NN]})
