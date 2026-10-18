"""A string `value` on a symmetry rule must reach sympify component by component, intact.

`add_symmetry_interaction` accepts a vector value as ONE string ("0,Dy,Dz",
"[0, Dy, Dz]"). Until 2026-10-18 it parsed that string by deleting every `[`, `]`,
`(` and `)` and then splitting on every comma, so `sqrt(3)*D1x/3` became
`sqrt3*D1x/3` -- a new free symbol `sqrt3` that no parameter binds. That is exactly
the form `magcalc symmetry` prints for the allowed DM on any hexagonal bond, and the
form the symmetry-conflict error tells users to paste. Measured on KFe3J (R-3m) with
the moments in the kagome plane: the bond table carried `sqrt3` and the dispersion
came out all-NaN, with no error. The same deletion also ungrouped `(J1+J2)/2, 0, 0`
and split `atan2(Dy, Dx)` in two.

Every route that PROPAGATES a value shares this parser -- `symmetry_rules` from the
CLI and both GUI endpoints, and the bare-`distance` entries under
`interactions.dm_interaction` / `anisotropic_exchange` / `interaction_matrix`,
which the expanders hand to `add_symmetry_interaction`. Explicit `pair:` entries do
not: `generic_model` evaluates those with `parse_expr`, which was always right.

Oracles, none of them a golden number:
* Moriya's rule from the bond geometry alone. KFe3J's NN bond has a little group of
  order 2 whose non-trivial element is a mirror that reverses the bond, so D lies in
  that mirror: its in-plane part is perpendicular to the bond, D_y/D_x = -r_x/r_y =
  sqrt(3)/3. That pins the premise that (D1x, sqrt(3)*D1x/3, D1z) IS the allowed
  form -- and the allowed matrix the CLI prints must say the same.
* The YAML LIST form never touches the string parser, so the string form must expand
  to the identical bond table; so must the same value with a decimal coefficient.
"""
import copy
import logging
import os

import numpy as np
import pytest
import sympy as sp
import yaml

from magcalc.config_builder import MagCalcConfigBuilder, _split_rule_value
from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(__file__)
KFE3J = os.path.join(HERE, "..", "examples", "materials", "KFe3J", "config_kfe3j.yaml")

D_NN = 3.66497                                   # KFe3J kagome NN bond (A)
EXACT = "D1x,sqrt(3)*D1x/3,D1z"                  # the form `magcalc symmetry` prints
DECIMAL = "D1x,0.5773502691896258*D1x,D1z"      # the form magpipe now emits
SAMPLES = [(0.2, 0.1), (-0.37, 0.53)]            # (D1x, D1z)


def _kfe3j_cfg(dm_value):
    cfg = yaml.safe_load(open(KFE3J))
    cfg["parameter_order"] = ["J1", "D1x", "D1z"]
    cfg["parameters"] = {"J1": 3.23, "D1x": 0.2, "D1z": 0.1}
    cfg["interactions"]["symmetry_rules"] = [
        {"type": "heisenberg", "distance": D_NN, "value": "J1",
         "ref_pair": ["Fe0", "Fe2"], "offset": [0, 0, 0]},
        {"type": "dm", "distance": D_NN, "value": dm_value,
         "ref_pair": ["Fe0", "Fe2"], "offset": [0, 0, 0]}]
    cfg["tasks"] = {}
    return cfg


def _dm_table(dm_value):
    """The expanded DM bond table, in emission order: [(i, j, offset, [3 strings])]."""
    m = GenericSpinModel(_kfe3j_cfg(dm_value))
    return m, [(e["pair"][0], e["pair"][1], tuple(e["rij_offset"]), e["value"])
               for e in m.interactions_config if e.get("type") == "dm_manual"]


def _evaluate(table, d1x, d1z):
    subs = {sp.Symbol("D1x"): d1x, sp.Symbol("D1z"): d1z}
    out = []
    for *_key, vec in table:
        exprs = [sp.sympify(v) for v in vec]
        stray = set().union(*(e.free_symbols for e in exprs)) - set(subs)
        assert not stray, f"unbound symbols {stray} in the bond table"
        out.append([float(e.subs(subs)) for e in exprs])
    return np.array(out)


def test_premise_the_printed_form_is_the_allowed_dm():
    """D1y = sqrt(3)/3 D1x is Moriya's mirror rule on this bond, and what the CLI prints."""
    b = MagCalcConfigBuilder.from_config(yaml.safe_load(open(KFE3J)))
    pos = {a["label"]: np.array(a["pos"], float) for a in b.atoms_uc}
    r = (pos["Fe2"] - pos["Fe0"]) @ np.array(b.lattice_vectors, float)
    assert abs(np.linalg.norm(r) - D_NN) < 1e-4
    assert abs(r[2]) < 1e-12                               # an in-plane kagome bond
    # in-plane D perpendicular to the bond (positions are given to 5 decimals)
    assert abs(-r[0] / r[1] - np.sqrt(3) / 3) < 1e-6

    orbit = next(o for o in b.analyze_bond_symmetry(max_distance=4.0)
                 if abs(o["distance"] - D_NN) < 1e-3)
    c = b.get_bond_constraints(orbit)
    rep = orbit["representative"]
    assert (rep["atom_i"], rep["atom_j"], [int(v) for v in rep["offset"]]) == ("Fe0", "Fe2", [0, 0, 0])
    J = sp.Matrix([[sp.sympify(x) for x in row] for row in c["symbolic_matrix"]])
    rng = np.random.default_rng(0)
    for _ in range(3):
        Jn = np.array(J.subs({s: rng.uniform(-1, 1) for s in J.free_symbols}), dtype=float)
        D = 0.5 * np.array([Jn[1, 2] - Jn[2, 1], Jn[2, 0] - Jn[0, 2], Jn[0, 1] - Jn[1, 0]])
        assert abs(D[1] - np.sqrt(3) / 3 * D[0]) < 1e-9 * max(1.0, abs(D[0]))


def test_printed_allowed_dm_expands_with_the_reference_bond_exact():
    """The exact form expands (no conflict, no stray symbol) and the reference bond
    carries the numeric value it denotes."""
    _, table = _dm_table(EXACT)
    assert len(table) == 36                    # 18 bonds x 2 directions
    ref = [k for k, row in enumerate(table) if row[:3] == ("Fe0", "Fe2", (0, 0, 0))]
    assert len(ref) == 1
    for d1x, d1z in SAMPLES:
        D = _evaluate(table, d1x, d1z)
        np.testing.assert_allclose(D[ref[0]], [d1x, np.sqrt(3) / 3 * d1x, d1z],
                                   rtol=0, atol=1e-12)
        # the propagated vectors keep D's length: an orthogonal transform of it
        np.testing.assert_allclose(np.linalg.norm(D, axis=1),
                                   np.hypot(d1x * 2 / np.sqrt(3), d1z), rtol=1e-12)


@pytest.mark.parametrize("variant", [
    "[D1x, sqrt(3)*D1x/3, D1z]",
    "(D1x, sqrt(3)*D1x/3, D1z)",
    " D1x , sqrt(3) * D1x / 3 , D1z ",
    DECIMAL,
    ["D1x", "sqrt(3)*D1x/3", "D1z"],           # YAML list: never string-parsed
])
def test_every_spelling_gives_the_identical_bond_table(variant):
    """Same bonds, same order, same numbers -- whichever way the value is written."""
    _, ref = _dm_table(EXACT)
    _, got = _dm_table(variant)
    assert [row[:3] for row in got] == [row[:3] for row in ref]
    for d1x, d1z in SAMPLES:
        np.testing.assert_allclose(_evaluate(got, d1x, d1z), _evaluate(ref, d1x, d1z),
                                   rtol=0, atol=1e-12)


def test_string_and_list_forms_agree_for_tensor_rules():
    """`anisotropic_exchange` and `interaction_matrix` share the parser. The full
    allowed matrix `magcalc symmetry` prints for this bond, pasted as ONE string,
    must expand to the same table as the YAML list of its rows; it used to lose its
    parentheses AND flatten to a 9-vector."""
    b0 = MagCalcConfigBuilder.from_config(yaml.safe_load(open(KFE3J)))
    orbit = next(o for o in b0.analyze_bond_symmetry(max_distance=4.0)
                 if abs(o["distance"] - D_NN) < 1e-3)
    J_list = b0.get_bond_constraints(orbit)["symbolic_matrix"]
    assert any("sqrt(3)" in x and "(" in x.replace("sqrt(3)", "") for row in J_list
               for x in row)                   # a tied entry with its own grouping
    J_str = "[" + ", ".join("[" + ", ".join(row) + "]" for row in J_list) + "]"
    rng = np.random.default_rng(1)
    free = sorted(set().union(*(sp.sympify(x).free_symbols for row in J_list for x in row)),
                  key=str)
    assert len(free) == 6
    subs = {s: rng.uniform(-1, 1) for s in free}
    subs.update({sp.Symbol("A"): 0.7, sp.Symbol("C"): -0.3})

    def table(kind, value):
        b = MagCalcConfigBuilder.from_config(yaml.safe_load(open(KFE3J)))
        b.add_symmetry_interaction(kind, ("Fe0", "Fe2"), value, distance=D_NN,
                                   offset=[0, 0, 0])
        rows = b.config["interactions"][kind]
        return ([(e["pair"], e["rij_offset"]) for e in rows],
                np.array([np.array(sp.Matrix(e["value"]).subs(subs), dtype=float)
                          for e in rows]))

    # The allowed DIAGONAL on this bond is (A, A, C): p2 = p3 = p4 = p5 = 0 above.
    for kind, s, lst in [("anisotropic_exchange", "sqrt(2)*A, sqrt(2)*A, (A+C)/2",
                          ["sqrt(2)*A", "sqrt(2)*A", "(A+C)/2"]),
                         ("interaction_matrix", J_str, J_list)]:
        keys_s, vals_s = table(kind, s)
        keys_l, vals_l = table(kind, lst)
        assert keys_s == keys_l and len(keys_s) == 36, kind
        np.testing.assert_allclose(vals_s, vals_l, rtol=0, atol=1e-12, err_msg=kind)


@pytest.mark.parametrize("text, parsed", [
    ("0,Dy,Dz", ["0", "Dy", "Dz"]),
    ("[0, -Dy, Dz]", ["0", "-Dy", "Dz"]),
    ("(0, -Dy, Dz)", ["0", "-Dy", "Dz"]),
    ("D1x,sqrt(3)*D1x/3,D1z", ["D1x", "sqrt(3)*D1x/3", "D1z"]),
    ("(J1+J2)/2, 0, 0", ["(J1+J2)/2", "0", "0"]),       # inner grouping kept
    ("atan2(Dy, Dx), 0, 1", ["atan2(Dy, Dx)", "0", "1"]),  # comma inside a call
    ("0,Dy,Dz,", ["0", "Dy", "Dz"]),                    # trailing comma
    ("[[a, b], [c, sqrt(2)*d]]", [["a", "b"], ["c", "sqrt(2)*d"]]),
    ("J1", "J1"),                                       # scalars pass through as-is
    ("(J1+J2)/2", "(J1+J2)/2"),
    ("(a+b)*(c+d)", "(a+b)*(c+d)"),                     # not ONE pair around the whole
])
def test_split_rule_value(text, parsed):
    assert _split_rule_value(text) == parsed


@pytest.mark.parametrize("text", [
    "[0, Dy, Dz",          # unclosed
    "0, Dy), Dz",          # stray close
    "[0, Dy, Dz)",         # mismatched pair
    "0,,Dz",               # empty component would shift Dz onto the y axis
])
def test_split_rule_value_refuses_malformed_input(text):
    with pytest.raises(ValueError):
        _split_rule_value(text)


def test_two_component_string_is_still_padded_to_3d():
    """Pre-existing convenience, kept: "0, -Dy" means (0, -Dy, 0)."""
    _, a = _dm_table("D1x, sqrt(3)*D1x/3")
    _, b = _dm_table("D1x, sqrt(3)*D1x/3, 0")
    for d1x, d1z in SAMPLES:
        np.testing.assert_allclose(_evaluate(a, d1x, d1z), _evaluate(b, d1x, d1z),
                                   rtol=0, atol=1e-12)
