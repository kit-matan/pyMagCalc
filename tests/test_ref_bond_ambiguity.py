"""Reference-bond resolution for `ref_pair` symmetry rules must never guess.

A `ref_pair` rule without an explicit `offset` has to pick which periodic image
of the second atom is the reference bond. When two images are symmetry-degenerate
the old resolver took whichever the enumeration reached first -- and for the
CCSF P2_1/n cell below the two candidates differ in length by ~9e-16 A, one ULP,
so the winner was decided by floating-point rounding.

That is invisible for a scalar J (every bond of the orbit carries the same
value) but not for a direction-carrying value: the two Cu0-Cu1 images are
related by the 2_1 screw, which acts on axial vectors as C2x = diag(1,-1,-1),
so attaching D to the other one realizes -C2x.D -- a sign flip on Dx alone.
Same bond count, plausible spectrum, different Hamiltonian.

These tests pin the guard and the expansion it protects.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from magcalc.config_builder import MagCalcConfigBuilder
from magcalc.generic_model import GenericSpinModel

# CCSF Cs2Cu3SnF12 monoclinic (P2_1/n) six-Cu cell. Cu0-Cu1 is the J12 pair
# whose two images at 3.5467973802 A are the degenerate case.
LATTICE = [
    [0.0, 4.148717, -6.711579],
    [7.08465, 0.0, 0.0],
    [0.0, 8.158324, 6.711583],
]
POSITIONS = [
    ("Cu0", [0.25364, 0.77059, 0.24238]),
    ("Cu1", [0.24636, 0.27059, 0.25762]),
    ("Cu2", [0.5, 0.0, 0.5]),
    ("Cu3", [0.74636, 0.22941, 0.75762]),
    ("Cu4", [0.75364, 0.72941, 0.74238]),
    ("Cu5", [0.0, 0.5, 0.0]),
]
DEGENERATE_PAIR = ["Cu0", "Cu1"]


def _config(rules):
    return {
        "name": "ccsf_ref_bond_ambiguity",
        "crystal_structure": {
            "lattice_vectors": [row[:] for row in LATTICE],
            "atoms_uc": [
                {"label": lbl, "pos": pos[:], "spin_S": 0.5, "ion": "Cu2+"}
                for lbl, pos in POSITIONS
            ],
        },
        "parameters": {"J12": 8.0, "D12x": -0.4, "D12y": 2.1, "D12z": 0.9},
        "parameter_order": ["J12", "D12x", "D12y", "D12z"],
        "interactions": {"symmetry_rules": rules},
        "magnetic_structure": {
            "type": "pattern",
            "pattern_type": "generic",
            "directions": [[0.54, 0.842, 0.009], [0.54, -0.842, -0.009],
                           [-0.899, 0.432, -0.075], [0.54, 0.842, 0.009],
                           [0.54, -0.842, -0.009], [-0.899, -0.432, 0.075]],
        },
        "calculation": {"mode": "dipole"},
    }


def _dm_rule(offset=None):
    rule = {"type": "dm", "ref_pair": DEGENERATE_PAIR[:],
            "value": ["D12x", "D12y", "D12z"]}
    if offset is not None:
        rule["offset"] = list(offset)
    return rule


def _heisenberg_rule(offset=None):
    rule = {"type": "heisenberg", "ref_pair": DEGENERATE_PAIR[:], "value": "J12"}
    if offset is not None:
        rule["offset"] = list(offset)
    return rule


PARAM_VALUES = {"J12": 8.0, "D12x": -0.4, "D12y": 2.1, "D12z": 0.9}


def _dm_bonds(model):
    """{(pair, rij_offset): D} for every expanded DM bond.

    The expansion stores D symbolically (e.g. '1.0*D12x'), so substitute the
    parameter values to get comparable numbers.
    """
    import sympy as sp

    subs = {sp.Symbol(k): v for k, v in PARAM_VALUES.items()}
    out = {}
    for entry in model.config.get("interactions", []):
        if entry.get("type") not in ("dm", "dm_manual", "dm_interaction"):
            continue
        val = entry.get("value", entry.get("vector"))
        D = tuple(round(float(sp.sympify(c).subs(subs)), 9) for c in val)
        key = (tuple(entry.get("pair", ())),
               tuple(entry.get("rij_offset", ())))
        out[key] = D
    return out


def test_reference_bond_images_are_degenerate_to_one_ulp():
    """The premise: two Cu0-Cu1 images are the same length to within one ULP,
    so any `<` comparison between them is decided by rounding, not physics."""
    lat = np.array(LATTICE, dtype=float)
    pos = dict(POSITIONS)
    pi = np.array(pos["Cu0"])
    pj = np.array(pos["Cu1"])
    d0 = np.linalg.norm(((pj + np.array([0, 0, 0])) - pi) @ lat)
    d1 = np.linalg.norm(((pj + np.array([0, 1, 0])) - pi) @ lat)

    assert abs(d0 - d1) < 1e-12, "expected two equidistant images"
    assert d0 != d1, "expected a ULP-level difference that a raw < would act on"


def test_degenerate_reference_bond_raises_for_dm():
    """Direction-carrying rule + degenerate images = refuse to guess."""
    with pytest.raises(ValueError) as excinfo:
        GenericSpinModel(_config([_dm_rule()]), base_path=os.path.dirname(__file__))

    msg = str(excinfo.value)
    assert "Ambiguous reference bond" in msg
    # the message must name both candidates and the way out
    assert "(0, 0, 0)" in msg and "(0, 1, 0)" in msg
    assert "offset" in msg


def test_degenerate_reference_bond_allowed_for_scalar_heisenberg():
    """A scalar J is identical on every bond of the orbit, so the same
    degeneracy must NOT be an error -- guarding it would be a false alarm."""
    model = GenericSpinModel(_config([_heisenberg_rule()]),
                             base_path=os.path.dirname(__file__))
    bonds = [e for e in model.config.get("interactions", [])
             if e.get("type") == "heisenberg"]
    assert bonds, "scalar rule should still expand"


@pytest.mark.parametrize("offset", [[0, 0, 0], [0, 1, 0]])
def test_explicit_offset_is_honoured(offset):
    """Pinning resolves the ambiguity and the rule expands."""
    model = GenericSpinModel(_config([_dm_rule(offset)]),
                             base_path=os.path.dirname(__file__))
    assert _dm_bonds(model), "pinned DM rule should expand to bonds"


def test_the_two_pinnings_are_physically_different():
    """The whole point of the guard: the two candidate reference bonds give
    different Hamiltonians. For the 2_1 screw the difference is D -> -C2x.D,
    i.e. Dx flips sign while Dy, Dz do not."""
    base = os.path.dirname(__file__)
    a = _dm_bonds(GenericSpinModel(_config([_dm_rule([0, 0, 0])]), base_path=base))
    b = _dm_bonds(GenericSpinModel(_config([_dm_rule([0, 1, 0])]), base_path=base))

    assert len(a) == len(b) and len(a) > 0
    assert a != b, (
        "the two reference-bond choices must not produce identical DM tables; "
        "if they do, the ambiguity guard is protecting nothing"
    )

    shared = set(a) & set(b)
    assert shared, "expected the two expansions to cover common bonds"
    differing = [k for k in shared if a[k] != b[k]]
    assert differing, "expected at least one bond to differ"
    for k in differing:
        Da, Db = np.array(a[k]), np.array(b[k])
        assert np.isclose(Da[0], -Db[0]), f"Dx should flip on bond {k}"
        assert np.allclose(Da[1:], Db[1:]), f"Dy,Dz should not change on {k}"


def test_scalar_expansion_covers_the_whole_orbit():
    """Why the scalar exemption is safe: a heisenberg `ref_pair` rule expands
    to the full orbit -- both the [0,0,0] and the [0,+-1,0] images -- so the
    bond table does not depend on which of the two was the reference."""
    model = GenericSpinModel(_config([_heisenberg_rule()]),
                             base_path=os.path.dirname(__file__))
    offsets = {tuple(e["rij_offset"]) for e in model.config.get("interactions", [])
               if e.get("type") == "heisenberg"}
    assert (0, 0, 0) in offsets
    assert (0, 1, 0) in offsets and (0, -1, 0) in offsets


@pytest.mark.parametrize("jitter", [0.0, 1e-15, -1e-15])
def test_canonical_choice_is_stable_against_ulp_noise(jitter):
    """The suggested reference bond must be the home-cell image every time.

    A resolver that ranks the degenerate images by raw distance picks whichever
    wins by ~1e-16 A, so a perturbation far below any physical tolerance flips
    it. The canonical order (rounded distance, then closeness to the home cell)
    must not move.
    """
    cfg = _config([_dm_rule()])
    cfg["crystal_structure"]["lattice_vectors"][1][0] += jitter

    with pytest.raises(ValueError) as excinfo:
        GenericSpinModel(cfg, base_path=os.path.dirname(__file__))

    # the guard recommends a concrete offset; it must be the home-cell one
    assert "offset: [0, 0, 0]" in str(excinfo.value)


def test_distance_window_spanning_two_lengths_raises():
    """The `distance:` branch must not first-match across distinct bond
    lengths: those are different orbits, and binding a value to the wrong one
    is wrong for every rule type, scalar included."""
    builder = MagCalcConfigBuilder()
    builder.set_lattice(a=10.0, b=10.0, c=10.0, space_group=1)
    builder.add_wyckoff_atom("A", [0.0, 0.0, 0.0], 1.0)
    builder.add_wyckoff_atom("B", [0.4985, 0.0, 0.0], 1.0)

    # images at 4.985 A and 5.015 A -- both inside the 0.05 A match window
    with pytest.raises(ValueError) as excinfo:
        builder.add_symmetry_interaction(type="heisenberg", ref_pair=("A", "B"),
                                         value=1.0, distance=5.0)

    msg = str(excinfo.value)
    assert "Ambiguous reference bond" in msg
    assert "distinct bond lengths" in msg
