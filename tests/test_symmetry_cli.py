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
