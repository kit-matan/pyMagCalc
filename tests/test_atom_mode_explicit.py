"""`atom_mode: explicit` must not empty the unit cell.

The Studio (web + native) writes `atom_mode` into every config it emits; CLI
configs omit it. `_expand_config_inplace` read explicit atoms from
`wyckoff_atoms` ONLY, so a config whose atoms sit in `atoms_uc` -- which is what
the app writes for an explicit cell, and what 41 of the shipped examples use --
had its atom list replaced by `[]` and died in `_load_structure` with

    ValueError: shapes (0,) and (3,3) not aligned: 0 (dim 0) != 3 (dim 0)

Adding that one key to a working config reproduces it, which is exactly what
"runs from the CLI, fails in the app" looked like to a user.

The reference here is an exact identity, not a golden number: the emitted model
must be the SAME model, so the spectrum must be bit-identical with and without
`atom_mode`, whichever key holds the atoms.
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel


def _square_afm():
    """S=1 square-lattice Neel AFM on a 2x2 cell (SW10's model, trimmed)."""
    return {
        "crystal_structure": {
            "lattice_vectors": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0},
                {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": 1.0},
                {"label": "C", "pos": [0.0, 0.5, 0.0], "spin_S": 1.0},
                {"label": "D", "pos": [0.5, 0.5, 0.0], "spin_S": 1.0},
            ],
        },
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["A", "C"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["C", "A"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["B", "D"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["D", "B"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["C", "D"], "rij_offset": [0, 0, 0], "value": "J"},
            {"pair": ["D", "C"], "rij_offset": [0, 0, 0], "value": "J"},
        ]},
        "parameters": {"J": 1.0},
        "parameter_order": ["J"],
        "magnetic_structure": {
            "type": "pattern", "pattern_type": "generic",
            "directions": [[0, 0, 1], [0, 0, -1], [0, 0, -1], [0, 0, 1]],
        },
        "tasks": {},
    }


def _positions(cfg):
    return np.asarray(GenericSpinModel(copy.deepcopy(cfg))._r_pos, dtype=float)


def test_atom_mode_explicit_keeps_atoms_uc():
    """The app's shape (atoms_uc + atom_mode) must build the CLI's cell."""
    cli = _square_afm()
    app = copy.deepcopy(cli)
    app["crystal_structure"]["atom_mode"] = "explicit"      # the only difference

    ref = _positions(cli)
    assert ref.shape == (4, 3), "fixture broken: the CLI config lost its atoms"
    np.testing.assert_array_equal(_positions(app), ref)


def test_atom_mode_explicit_accepts_either_atom_key():
    """`wyckoff_atoms` is the designer's spelling of the same explicit list."""
    cli = _square_afm()
    designer = copy.deepcopy(cli)
    cs = designer["crystal_structure"]
    cs["wyckoff_atoms"] = cs.pop("atoms_uc")
    cs["atom_mode"] = "explicit"

    np.testing.assert_array_equal(_positions(designer), _positions(cli))


@pytest.mark.parametrize("atom_key", ["atoms_uc", "wyckoff_atoms"])
def test_explicit_bonds_resolve_by_label(atom_key):
    """Label -> index resolution must work for both spellings: without it the
    Hamiltonian silently loses the bonds it could not resolve."""
    cfg = _square_afm()
    cs = cfg["crystal_structure"]
    if atom_key == "wyckoff_atoms":
        cs["wyckoff_atoms"] = cs.pop("atoms_uc")
    cs["atom_mode"] = "explicit"

    def exchange(c):
        Jex, _DM, _Kex = GenericSpinModel(copy.deepcopy(c)).spin_interactions([1.0])
        return np.array(Jex.tolist(), dtype=float)

    J_app = exchange(cfg)
    # 8 directed NN bonds carrying J = 1 (the exchange matrix is over the
    # out-of-cell atom list, so check the entries, not its size). A label that
    # fails to resolve drops its bond from the Hamiltonian without a word.
    assert np.count_nonzero(J_app) == 8, f"got {np.count_nonzero(J_app)} bonds"
    # ... and it is the same Hamiltonian the CLI form builds.
    np.testing.assert_array_equal(J_app, exchange(_square_afm()))


def _chain_with_distance_rule():
    """SW26's shape: an explicit cell whose bonds come from a DISTANCE rule."""
    return {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}],
        },
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": "J1"},
        ]},
        "parameters": {"J1": 1.0},
        "parameter_order": ["J1"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "tasks": {},
    }


def test_distance_rules_expand_in_explicit_mode():
    """`atom_mode` describes the ATOMS, not whether a bond rule may propagate.

    Gating the distance-based expanders on `atom_mode == 'symmetry'` made an
    explicit cell's distance rules expand to ZERO bonds -- an empty Hamiltonian,
    reported downstream as imaginary magnons rather than as the missing exchange
    it was. The CLI omits `atom_mode`; the apps write it.
    """
    def bonds(cfg):
        Jex, _DM, _Kex = GenericSpinModel(copy.deepcopy(cfg)).spin_interactions([1.0])
        return np.count_nonzero(np.array(Jex.tolist(), dtype=float))

    cli = _chain_with_distance_rule()
    n_cli = bonds(cli)
    assert n_cli > 0, "fixture broken: the distance rule matched nothing"

    for mode in ("symmetry", "explicit"):
        app = copy.deepcopy(cli)
        app["crystal_structure"]["atom_mode"] = mode
        assert bonds(app) == n_cli, f"atom_mode: {mode} lost bonds"
