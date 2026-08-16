"""The Studio's bond visualiser must expand `symmetry_rules` exactly as the CLI does.

`/get-visualizer-data` re-derives the bond table from the config through
`MagCalcConfigBuilder` (the run path itself is config-as-source -- see
test_gui_run_parity.py -- so this endpoint is the only place the apps still
expand rules themselves). Its expansion therefore has to agree with
`GenericSpinModel`'s, which is what `magcalc run` actually diagonalizes; the
CLI's bond table is the oracle here.

Three independent ways it did not, all found on the NaCVO manuscript configs
(`lattice_vectors` + `atoms_uc` + `symmetry_rules`), where the visualiser drew
ZERO bonds while `magcalc run` on the same file expanded 72:

  1. `_hydrate_builder` skipped symmetry detection whenever
     `atom_mode == "explicit"`. How the ATOMS were given says nothing about
     whether a bond RULE may be propagated -- every rule died on
     "No symmetry operations loaded".
  2. the explicit branch never built `builder._atom_label_to_idx`, so a rule's
     `ref_pair` could not be resolved at all ("Atom 'Cu1' not found in unit
     cell (keys: [])").
  3. `/get-visualizer-data` reset `builder.config["interactions"]` to a dict
     with no `interaction_matrix` key, so an `interaction_matrix` rule (usually
     the strongest exchange in the model) died with a bare
     `KeyError('interaction_matrix')` -- including on the wyckoff/space-group
     path, where everything else propagated fine.

Each failure printed a warning and returned HTTP 200, i.e. a plausible-looking
but incomplete structure picture.
"""
import os
import sys
import copy
import asyncio

import pytest
import yaml

pytest.importorskip("fastapi")

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..")
EX = os.path.join(ROOT, "examples")

# CLI bond-type names -> the visualiser's display names.
_TYPE_MAP = {
    "heisenberg": "heisenberg",
    "dm": "dm",
    "dm_manual": "dm",
    "anisotropic_exchange": "anisotropic",
    "interaction_matrix": "matrix",
    "kitaev": "matrix",
}


@pytest.fixture(scope="module")
def server():
    sys.path.insert(0, os.path.join(ROOT, "gui"))
    import server as srv
    return srv


def _cli_bonds(cfg):
    """{(type, (label_i, label_j), offset)} as `magcalc run` expands the config."""
    from magcalc.generic_model import GenericSpinModel
    model = GenericSpinModel(copy.deepcopy(cfg))
    bonds = set()
    for b in model.config["interactions"]:
        if not isinstance(b, dict) or not b.get("pair"):
            continue
        t = _TYPE_MAP.get(b.get("type"))
        if t is None:          # single-ion / field terms are not bonds
            continue
        off = tuple(int(v) for v in (b.get("rij_offset") or [0, 0, 0]))
        bonds.add((t, tuple(b["pair"]), off))
    return bonds


def _gui_bonds(server, cfg, atom_mode=None):
    """The same set, as /get-visualizer-data reports it."""
    data = copy.deepcopy(cfg)
    if atom_mode is not None:
        data["crystal_structure"]["atom_mode"] = atom_mode
    res = asyncio.run(server.get_visualizer_data({"data": data}))
    labels = [a["label"] for a in res["atoms"]]
    return {(b["type"], (labels[b["atom_i"]], labels[b["atom_j"]]),
             tuple(int(v) for v in b["offset"]))
            for b in res["bonds"]}


@pytest.mark.parametrize("atom_mode", [None, "explicit", "symmetry"])
def test_explicit_cell_rules_match_cli(server, atom_mode):
    """SW20: `lattice_vectors` + `atoms_uc` + an `interaction_matrix` rule
    propagated over the spglib-detected Fd-3m -- 96 ordered NN bonds. The
    visualiser reported 0 of them for every `atom_mode` the apps write."""
    cfg = yaml.safe_load(open(os.path.join(EX, "spinw_tutorials/SW20_Yb2Ti2O7/config.yaml")))
    cli = _cli_bonds(cfg)
    assert len(cli) == 96, f"CLI expansion changed: {len(cli)} bonds"
    assert _gui_bonds(server, cfg, atom_mode) == cli


@pytest.mark.parametrize("atom_mode", [None, "explicit", "symmetry"])
def test_distance_rules_match_cli(server, atom_mode):
    """SW26: two bare-`distance` heisenberg rules on an explicit one-site cell.
    This endpoint routed them to `add_symmetry_interaction`, which refuses a
    rule with no `ref_pair`, and then ran the distance expanders only for
    `atom_mode == "symmetry"` -- so both J1 and J2 were missing."""
    cfg = yaml.safe_load(open(os.path.join(EX, "spinw_tutorials/SW26_spiral_chain/config.yaml")))
    cli = _cli_bonds(cfg)
    assert len(cli) == 4, f"CLI expansion changed: {len(cli)} bonds"   # J1, J2, both directions
    assert _gui_bonds(server, cfg, atom_mode) == cli


def test_wyckoff_matrix_rule_reaches_visualizer(server):
    """SW09: wyckoff + `space_group: 147`. Symmetry propagation worked here, but
    the `interaction_matrix` bonds were dropped by the missing dict key, so the
    kagome triangles came back with no bonds at all."""
    cfg = yaml.safe_load(open(os.path.join(EX, "spinw_tutorials/SW09_kagome_AFM_DM/config.yaml")))
    cli = _cli_bonds(cfg)
    gui = _gui_bonds(server, cfg)
    assert gui, "no bonds reached the visualiser"
    assert all(t == "matrix" for t, _, _ in gui)
    assert gui == cli
