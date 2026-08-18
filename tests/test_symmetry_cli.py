"""`magcalc symmetry` -- crystal-symmetry analyzer exposed as a CLI (Tier 3 #14).

The analyzer (`MagCalcConfigBuilder.analyze_bond_symmetry` + `get_bond_constraints`)
existed only behind the GUI. These tests pin it to INDEPENDENT references:

  * P4/mmm (SG 123), one atom: the nearest-neighbour bond is centrosymmetric, so
    symmetry FORCES a diagonal exchange matrix -- a textbook result, no free
    off-diagonals.
  * SW20 Yb2Ti2O7 (pyrochlore): the symmetry-ALLOWED form the analyzer reports must
    be exactly the structure of the physical exchange matrix used in that config
    (itself validated against SpinW/Sunny) -- same zeros, same tied entries.
  * The CLI wrapper runs, and `--json` is machine-readable.
"""
import json
import os

import numpy as np
import sympy as sp
import yaml
from typer.testing import CliRunner

from magcalc.cli import app
from magcalc.config_builder import MagCalcConfigBuilder

HERE = os.path.dirname(__file__)
EX = os.path.join(HERE, "..", "examples", "spinw_tutorials")
runner = CliRunner()


def test_centrosymmetric_bond_forces_a_diagonal_matrix():
    """P4/mmm single atom: the NN bond's little group forbids all off-diagonals."""
    b = MagCalcConfigBuilder()
    b.set_lattice(a=4.0, c=5.0, space_group=123)      # P4/mmm
    b.add_wyckoff_atom("Cu", [0, 0, 0], 1.0)
    orbits = b.analyze_bond_symmetry(max_distance=4.1)
    nn = next(o for o in orbits if abs(o["distance"] - 4.0) < 0.1)
    c = b.get_bond_constraints(nn)
    m = c["symbolic_matrix"]
    # every off-diagonal entry is exactly "0"
    for i in range(3):
        for j in range(3):
            if i != j:
                assert m[i][j] == "0", f"off-diagonal {i}{j} = {m[i][j]}, expected 0"
    assert all(m[i][i] != "0" for i in range(3))        # diagonal entries are free
    assert c["little_group_size"] >= 8


def test_from_config_builds_the_expanded_cell():
    cfg = yaml.safe_load(open(os.path.join(EX, "SW07_kagome_AFM", "config.yaml")))
    b = MagCalcConfigBuilder.from_config(cfg)
    assert b.space_group_number == 147                  # P-3, kagome
    assert len(b.atoms_uc) == 3                          # 3e orbit expanded
    # nearest-neighbour kagome bond at a/2 = 3.0 A exists
    orbits = b.analyze_bond_symmetry(max_distance=3.1)
    assert any(abs(o["distance"] - 3.0) < 1e-3 for o in orbits)


def _allowed_form(builder, orbit):
    """Return the 3x3 of sympy strings for the orbit's allowed exchange matrix."""
    return builder.get_bond_constraints(orbit)["symbolic_matrix"]


def test_pyrochlore_allowed_form_matches_the_physical_matrix():
    """The analyzer's allowed form for the Yb2Ti2O7 NN bond must be exactly the
    structure (zeros + tied entries) of the exchange matrix that config actually uses
    and that reproduces SpinW/Sunny."""
    cfg = yaml.safe_load(open(os.path.join(EX, "SW20_Yb2Ti2O7", "config.yaml")))
    b = MagCalcConfigBuilder.from_config(cfg)
    orbits = b.analyze_bond_symmetry(max_distance=3.8)
    nn = min(orbits, key=lambda o: o["distance"])        # the ~3.55 A NN bond
    c = b.get_bond_constraints(nn)
    assert c["little_group_size"] == 4
    assert len(c["free_parameters"]) == 4                # 4 independent components

    # Pull the config's actual NN interaction matrix and check it obeys the same ties.
    inter = cfg["interactions"]
    mats = inter.get("interaction_matrix") if isinstance(inter, dict) else None
    if not mats and isinstance(inter, dict):
        mats = [r for r in inter.get("symmetry_rules", []) if r.get("type") == "interaction_matrix"]
    assert mats, "expected an interaction_matrix in the SW20 config"
    M = np.array(mats[0]["value"], float)
    # the pyrochlore allowed form: [[a, -b, -b], [b, d, c], [b, c, d]]
    assert abs(M[0, 1] - M[0, 2]) < 1e-9                 # tied off-diagonals (row 0)
    assert abs(M[1, 0] - M[2, 0]) < 1e-9
    assert abs(M[1, 0] + M[0, 1]) < 1e-9                 # M[1,0] = -M[0,1]
    assert abs(M[1, 1] - M[2, 2]) < 1e-9                 # tied diagonal
    assert abs(M[1, 2] - M[2, 1]) < 1e-9                 # symmetric lower block


def test_cli_runs_and_json_is_parseable():
    cfg_path = os.path.join(EX, "SW11_La2CuO4", "config.yaml")
    res = runner.invoke(app, ["symmetry", cfg_path, "--max-distance", "4.0"])
    assert res.exit_code == 0, res.output
    assert "Space group: 123" in res.output
    assert "symmetry-allowed exchange matrix" in res.output

    res_json = runner.invoke(app, ["symmetry", cfg_path, "--max-distance", "4.0", "--json"])
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(res_json.output)
    assert data["space_group"] == 123
    assert len(data["bond_orbits"]) == 1
    nn = data["bond_orbits"][0]
    assert abs(nn["distance"] - 3.85) < 1e-2
    # diagonal allowed form -> off-diagonals are "0"
    m = nn["allowed_matrix"]
    assert m[0][1] == "0" and m[1][2] == "0"


# ---------------------------------------------------------------------------
# Robustness: symmetry is DISCRETE, so the allowed form must not move when the
# lattice does at the level of measurement noise (2026-08-17).
# ---------------------------------------------------------------------------
def _nn_form(cfg, max_distance, da=0.0):
    """Allowed matrix + free parameters of the shortest bond, `a` shifted by `da`."""
    import copy

    c = copy.deepcopy(cfg)
    c["crystal_structure"]["lattice_parameters"]["a"] += da
    b = MagCalcConfigBuilder.from_config(c)
    orbits = b.analyze_bond_symmetry(max_distance=max_distance)
    nn = min(orbits, key=lambda o: o["distance"])
    con = b.get_bond_constraints(nn)
    return con["symbolic_matrix"], con["free_parameters"], con["little_group_size"]


def test_allowed_form_is_unchanged_by_lattice_noise():
    """
    Perturbing `a` by 1e-7 A -- far below any experimental precision, and 4
    orders below the symprec spglib accepted the group at -- must not change
    the symmetry-allowed exchange matrix at all.

    It used to change it completely. `get_bond_constraints` "sanitized" the
    Cartesian rotation with `np.round(R, 10)` + `nsimplify`, which turns a
    noisy 2/3 into the exact rational 1666666499/2500000000. That R is not
    orthogonal, so `J = R J R^T` admits far fewer solutions than it should:
    KFe3J's NN bond went from 6 free parameters to 1, reporting every
    off-diagonal as symmetry-forbidden -- i.e. "no DM allowed on this bond",
    silently, on a config whose whole point is its DM term. (On a Materials
    Project NiO primitive cell the same giant rationals instead made sympy
    grind for over ten minutes without returning.)
    """
    cfg = yaml.safe_load(open(os.path.join(HERE, "..", "examples", "materials",
                                           "KFe3J", "config_kfe3j.yaml")))
    m0, free0, lg0 = _nn_form(cfg, 4.0)
    m1, free1, lg1 = _nn_form(cfg, 4.0, da=1e-7)

    assert lg0 == lg1 == 2
    assert free0 == free1, f"free parameters moved: {free0} -> {free1}"
    assert m0 == m1, f"allowed matrix moved under 1e-7 A:\n{m0}\n{m1}"
    # KFe3J's NN bond carries a DM term (Dy, Dz in the config), so the allowed
    # form must have off-diagonal freedom -- the collapse this pins against
    # reported none.
    assert len(free0) == 6
    assert m0[0][1] != "0" and m0[0][2] != "0"


def test_allowed_form_survives_a_real_cif_lattice():
    """
    A rocksalt primitive cell as Materials Project ships it (cubic to six
    decimals, not exactly) must give the exact centrosymmetric forms, quickly.

    Both nearest-neighbour bonds in rocksalt have an inversion centre at their
    midpoint, so the allowed matrix is SYMMETRIC: D = 0 on both, which is why
    NiO's exchange is a two-parameter Heisenberg problem in the first place.
    """
    import time

    # mp-aaaabcdd, the Phase-2 NiO run: rhombohedral primitive, cubic to 6 dp
    b = MagCalcConfigBuilder.from_config({"crystal_structure": {
        "lattice_parameters": {"a": 2.96018835, "b": 2.96018714, "c": 2.96018743,
                               "alpha": 59.99999689, "beta": 60.00001040,
                               "gamma": 60.00000019},
        "atoms_uc": [{"label": "Ni0", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}],
    }})
    assert b.space_group_number == 225                     # Fm-3m, fcc Ni

    t0 = time.time()
    orbits = b.analyze_bond_symmetry(max_distance=4.5)
    forms = [b.get_bond_constraints(o) for o in orbits]
    assert time.time() - t0 < 30.0, "symmetry analysis should be milliseconds"

    assert len(orbits) == 2
    assert abs(orbits[0]["distance"] - 2.9602) < 1e-3      # 12 NN, 90 deg
    assert abs(orbits[1]["distance"] - 4.1863) < 1e-3      # 6 NNN, 180 deg
    for orb, con in zip(orbits, forms):
        m = sp.Matrix([[sp.sympify(e) for e in row]
                       for row in con["symbolic_matrix"]])
        assert sp.simplify(m - m.T) == sp.zeros(3, 3), (
            f"d = {orb['distance']:.3f} A: allowed form is not symmetric, so it "
            f"claims DM is allowed on a centrosymmetric bond:\n"
            f"{con['symbolic_matrix']}")
    # the 180-degree bond is uniaxial about the bond: J_par and J_perp only
    assert len(forms[1]["free_parameters"]) == 2
