"""The `kitaev` interaction type (coverage-audit item 2).

The 2026-08-04 audit found `interactions.kitaev` in zero examples -- SW16, the
Kitaev tutorial, expresses its bond-dependent couplings as `interaction_matrix` --
and, in tests, only `test_new_interactions.py::test_kitaev_interaction`, which
asserts

    assert any(str(s) in ['kx','ky','kz'] for s in hm.free_symbols) or len(...) >= 0
    assert not hm.is_zero_matrix

whose first line is a tautology (`len(...) >= 0`), leaving "the Hamiltonian is not
identically zero". That passes for a wrong axis, a wrong sign, or a wrong magnitude:
a check a wrong answer passes is not a check. Effectively, no coverage.

Three silent-drop bugs were sitting behind that gap, all fixed 2026-08-04:
an unresolvable `value` logged a WARNING and skipped the bond; an unrecognised
`axis` fell through `.get(axis, 2)` to z; and `type: kitaev` under `symmetry_rules`
-- the route CLAUDE.md documents as REQUIRING a `ref_pair` -- had no propagation
branch at all and expanded to ZERO bonds without a word.

NO ORACLE IS NEEDED. `generic_model` builds the Kitaev term as a 3x3 matrix with a
single diagonal entry K at the bond's axis, so it has an EXACT equivalent in
`interaction_matrix`. Every test below is that identity, which fails loudly on a
wrong axis index, a dropped factor, or a term that never reaches the Hamiltonian.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = [[0.13, 0, 0], [0.31, 0, 0], [0.5, 0, 0]]
S_VAL = 1.0


def _bands(interactions, tag):
    cfg = {"crystal_structure": {
               "lattice_vectors": LAT,
               "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S_VAL},
                            {"label": "B", "pos": [0.5, 0, 0], "spin_S": S_VAL}]},
           "interactions": copy.deepcopy(interactions),
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]},
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S_VAL, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    e = np.real(calc.calculate_dispersion([np.array(q) @ B for q in HS]).energies)
    return np.sort(e, axis=1)


def _heis(J=1.0):
    return [{"type": "heisenberg", "pair": p, "rij_offset": o, "value": J}
            for p, o in NN]


def _matrix_for(axis, K):
    """The 3x3 a Kitaev term is documented to build: K on the axis diagonal only."""
    m = [[0.0] * 3 for _ in range(3)]
    m[{"x": 0, "y": 1, "z": 2}[axis]][{"x": 0, "y": 1, "z": 2}[axis]] = K
    return m


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.parametrize("K", [0.4, -0.7])
def test_kitaev_equals_its_explicit_matrix(axis, K):
    """THE identity. A Kitaev bond along `axis` with strength K is by construction
    an interaction_matrix carrying K at that diagonal entry and nothing else. Any
    axis-index slip (the obvious bug, since 'x'/'y'/'z' must map to 0/1/2) shows up
    as a different spectrum."""
    kit = _heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                      "value": K, "axis": axis} for p, o in NN]
    mat = _heis() + [{"type": "interaction_matrix", "pair": p, "rij_offset": o,
                      "value": _matrix_for(axis, K)} for p, o in NN]
    assert _bands(kit, f"kit_{axis}_{K}") == pytest.approx(
        _bands(mat, f"mat_{axis}_{K}"), abs=1e-10)


def test_bond_direction_is_an_accepted_alias_for_axis():
    """`generic_model` reads `axis` OR `bond_direction`; the config builder emits the
    latter (`_add_kitaev_entry`), so the two spellings must agree."""
    a = _heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                    "value": 0.5, "axis": "y"} for p, o in NN]
    b = _heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                    "value": 0.5, "bond_direction": "y"} for p, o in NN]
    assert _bands(a, "kit_axis") == pytest.approx(_bands(b, "kit_bdir"), abs=1e-12)


def test_kitaev_terms_on_different_axes_add():
    """Two Kitaev bonds with different axes must sum to the diagonal matrix carrying
    both -- i.e. the term accumulates rather than overwriting. `Kex[i][j] += k_mat`
    is easy to write as `=` and the bug would be invisible with a single term."""
    both = _heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                       "value": 0.4, "axis": "x"} for p, o in NN] \
                   + [{"type": "kitaev", "pair": p, "rij_offset": o,
                       "value": -0.7, "axis": "z"} for p, o in NN]
    m = [[0.4, 0, 0], [0, 0, 0], [0, 0, -0.7]]
    mat = _heis() + [{"type": "interaction_matrix", "pair": p, "rij_offset": o,
                      "value": m} for p, o in NN]
    assert _bands(both, "kit_sum") == pytest.approx(_bands(mat, "mat_sum"), abs=1e-10)


def test_kitaev_actually_changes_the_spectrum():
    """Guards the identity above: if BOTH paths silently dropped the term, the
    comparison would pass on two copies of the Heisenberg answer."""
    plain = _bands(_heis(), "kit_plain")
    kit = _bands(_heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                             "value": 0.6, "axis": "z"} for p, o in NN], "kit_on")
    assert np.abs(kit - plain).max() > 0.1


def test_default_axis_is_z():
    """Undocumented but real: `axis` defaults to 'z' when absent. Pinned so the
    default cannot drift silently -- a config relying on it would otherwise change
    meaning."""
    implicit = _heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                           "value": 0.5} for p, o in NN]
    explicit = _heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                           "value": 0.5, "axis": "z"} for p, o in NN]
    assert _bands(implicit, "kit_imp") == pytest.approx(
        _bands(explicit, "kit_exp"), abs=1e-12)


def test_unresolvable_value_raises():
    """A Kitaev entry whose `value` does not resolve USED TO log a WARNING and
    `continue`, dropping the bond. Every other term raises in that situation
    (cf. the `stevens B_k^q resolved to None` check) because a Hamiltonian quietly
    missing a term still produces a perfectly plausible spectrum. Hardened
    2026-08-04."""
    with pytest.raises(ValueError, match="kitaev value resolved to None"):
        _bands(_heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                           "value": None, "axis": "z"} for p, o in NN], "kit_none")


def test_unrecognised_axis_raises():
    """The nastier of the two silent paths: `.get(axis, 2)` mapped ANY unrecognised
    axis to z. `axis: c` (a crystallographic label where a spin component is wanted)
    or a plain typo therefore built a z-Kitaev term and ran without complaint."""
    with pytest.raises(ValueError, match="kitaev axis"):
        _bands(_heis() + [{"type": "kitaev", "pair": p, "rij_offset": o,
                           "value": 0.5, "axis": "c"} for p, o in NN], "kit_badax")


# ---------------------------------------------------------------------------
# The `symmetry_rules` route. CLAUDE.md section 2 lists `kitaev` among the types for
# which `ref_pair` is REQUIRED -- but until 2026-08 `add_symmetry_interaction` had no
# `kitaev` branch at all. The rule ran the reference-bond search, looped over every
# symmetry op, and added ZERO bonds, with no warning: a documented config route that
# produced a Hamiltonian missing the interaction entirely. The dispatch now also
# raises on any unhandled type, so a future one cannot vanish the same way.
# ---------------------------------------------------------------------------

def _sc_builder():
    """Simple cubic, Pm-3m -- 48 ops, and the 6 nearest-neighbour bonds fall into the
    x/y/z families a Kitaev term distinguishes."""
    from magcalc.config_builder import MagCalcConfigBuilder
    b = MagCalcConfigBuilder()
    b.set_lattice(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90, space_group=221)
    b.add_wyckoff_atom(label="A", pos=[0, 0, 0], spin=0.5)
    b.detect_symmetry_from_structure()
    return b


def _table(entries):
    return sorted((e["pair"][0], e["pair"][1], tuple(e["rij_offset"]),
                   tuple(tuple(round(float(x), 9) for x in row) for row in e["value"]))
                  for e in entries)


def test_symmetry_rule_kitaev_expands_to_bonds():
    """The regression proper. This rule used to expand to NOTHING."""
    b = _sc_builder()
    b.add_symmetry_interaction(type="kitaev", ref_pair=["A", "A"], value=0.7,
                               offset=[0, 0, 1], axis="z")
    assert len(b.config["interactions"]["interaction_matrix"]) == 6


def test_symmetry_rule_kitaev_equals_equivalent_matrix_rule():
    """Same exact-equivalent oracle as the explicit-bond tests, one level up: the
    propagated bond table must be identical to the one an `interaction_matrix` rule
    carrying diag(0,0,K) produces, entry for entry."""
    bk, bm = _sc_builder(), _sc_builder()
    bk.add_symmetry_interaction(type="kitaev", ref_pair=["A", "A"], value=0.7,
                                offset=[0, 0, 1], axis="z")
    bm.add_symmetry_interaction(type="interaction_matrix", ref_pair=["A", "A"],
                                value=_matrix_for("z", 0.7), offset=[0, 0, 1])
    assert _table(bk.config["interactions"]["interaction_matrix"]) == \
           _table(bm.config["interactions"]["interaction_matrix"])


def test_symmetry_propagation_permutes_the_kitaev_axis():
    """What makes the rule worth having, and a check no single bond can pass: one
    z-axis reference bond propagates to the bond-dependent coupling that defines a
    Kitaev model -- K^xx on the x bonds, K^yy on the y, K^zz on the z. A propagation
    that ignored the axis (or forgot to rotate the tensor) would put K^zz on all six."""
    b = _sc_builder()
    b.add_symmetry_interaction(type="kitaev", ref_pair=["A", "A"], value=0.7,
                               offset=[0, 0, 1], axis="z")
    got = {}
    for e in b.config["interactions"]["interaction_matrix"]:
        m = np.array(e["value"], float)
        assert np.allclose(m, np.diag(np.diag(m)), atol=1e-9), "not diagonal"
        axis_of = int(np.argmax(np.abs(np.diag(m))))
        bond_of = int(np.argmax(np.abs(e["rij_offset"])))
        got[tuple(e["rij_offset"])] = (bond_of, axis_of)
        assert np.diag(m)[axis_of] == pytest.approx(0.7)
        assert np.count_nonzero(np.abs(np.diag(m)) > 1e-9) == 1
    assert len(got) == 6
    for bond_of, axis_of in got.values():
        assert bond_of == axis_of, "bond direction and Kitaev axis must agree"


def test_symmetry_rule_kitaev_rejects_a_tensor_value():
    b = _sc_builder()
    with pytest.raises(ValueError, match="expects a SCALAR"):
        b.add_symmetry_interaction(type="kitaev", ref_pair=["A", "A"],
                                   value=_matrix_for("z", 0.7), offset=[0, 0, 1])


def test_symmetry_rule_kitaev_rejects_a_bad_axis():
    b = _sc_builder()
    with pytest.raises(ValueError, match="kitaev symmetry rule"):
        b.add_symmetry_interaction(type="kitaev", ref_pair=["A", "A"], value=0.7,
                                   offset=[0, 0, 1], axis="c")


def test_unhandled_symmetry_rule_type_raises():
    """The structural guard. `kitaev` fell through an if/elif chain with no `else`;
    anything else unhandled would have done the same. It now raises."""
    b = _sc_builder()
    with pytest.raises(ValueError, match="no propagation branch"):
        b.add_symmetry_interaction(type="biquadratic", ref_pair=["A", "A"],
                                   value=0.7, offset=[0, 0, 1])


def _sc_bands(rule, tag):
    """Full config route: simple cubic via space group + a single symmetry rule."""
    cfg = {"crystal_structure": {
               "lattice_parameters": {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90.0,
                                      "beta": 90.0, "gamma": 90.0, "space_group": 221},
               "wyckoff_atoms": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0}]},
           "interactions": {"symmetry_rules": [
               {"type": "heisenberg", "distance": 4.0, "value": 1.0},
               copy.deepcopy(rule)]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])
    lat = np.array(m.config["crystal_structure"]["lattice_vectors"], float)
    B = 2 * np.pi * np.linalg.inv(lat).T
    qs = [np.array(q) @ B for q in ([0.13, 0.07, 0], [0.3, 0.2, 0.1], [0.5, 0, 0])]
    return np.sort(np.real(calc.calculate_dispersion(qs).energies), axis=1)


def test_symmetry_rule_kitaev_end_to_end_matches_matrix_rule():
    """Config level, so it covers the `axis` pass-through from `symmetry_rules` into
    the builder -- which the builder-level tests above bypass by calling directly.
    Before the fix this raised or produced the bare Heisenberg spectrum."""
    kit = _sc_bands({"type": "kitaev", "ref_pair": ["A", "A"], "offset": [0, 0, 1],
                     "value": 0.7, "axis": "z"}, "sc_kit")
    mat = _sc_bands({"type": "interaction_matrix", "ref_pair": ["A", "A"],
                     "offset": [0, 0, 1], "value": _matrix_for("z", 0.7)}, "sc_mat")
    heis = _sc_bands({"type": "heisenberg", "distance": 4.0, "value": 0.0}, "sc_heis")
    assert kit == pytest.approx(mat, abs=1e-9)
    assert np.abs(kit - heis).max() > 0.1     # the term really is present
