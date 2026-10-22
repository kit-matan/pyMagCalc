"""A `symmetry_rules` value must have the symmetry of its own bond (Moriya's rules).

`add_symmetry_interaction` propagates a reference bond's value over the space group,
D' = det(R) R D for a DM vector, J' = R J R^T for a tensor. Several operations reach
each bond of the orbit, and they agree on its value only if the value is invariant
under the bond's little group -- the operations mapping the bond onto itself (value
unchanged) or onto its reverse (D -> -D, J -> J^T). Until 2026-10-18 the FIRST
operation to reach a bond won, silently, so a component the site symmetry forbids
came out with whichever sign spglib's enumeration order happened to give it.

Found on alpha-Cu2V2O7 (Fdd2): the midpoint of the J2 bond Cu0-Cu13 lies on the
2-fold axis along c, so D_z is forbidden there (Moriya: a 2-fold axis perpendicular
to the bond through its midpoint). The engine put D_z = +-1 on all 8 J2 bonds with
signs that disagreed with a hand propagation on 8 of 16 directed entries and were
not invariant under the F-centring translations (32 of 48 translated entries
differed) -- a bond table of lower symmetry than the lattice, whose minimized
ground state then broke translation symmetry. Same bond count, plausible spectrum.

Oracle: the 16 operations of Fdd2 written out by hand (four coset representatives
times the four F-centring translations), NOT spglib's list. The hand propagation
detects the conflict itself, and for every allowed value the engine's table must
equal it entry for entry, be antisymmetric, and be invariant under the centring
translations. The J1 table of the shipped aCVO config is pinned to it too, because
that one must not move.
"""
import os

import numpy as np
import pytest
import sympy as sp
import yaml

from magcalc.config_builder import MagCalcConfigBuilder, SymmetryConflictError

HERE = os.path.dirname(__file__)
ACVO = os.path.join(HERE, "..", "examples", "materials", "aCVO", "config_acvo.yaml")

A, B, C = 20.645, 8.383, 6.442
CU = [0.16572, 0.3646, 0.7545]
D_J1, D_J2, D_J3 = 3.13253, 3.97512, 5.2572

# Fdd2 (No. 43), conventional F cell, origin on the 2-fold axis (ITA setting):
# coset representatives (x,y,z), (-x,-y,z), (-x+1/4,y+1/4,z+1/4), (x+1/4,-y+1/4,z+1/4)
# times the four F-centring translations. Orthorhombic, so the fractional rotation
# IS the Cartesian one and det(R) R is the axial-vector transform.
_COSETS = [(np.diag([1, 1, 1]), [0, 0, 0]),
           (np.diag([-1, -1, 1]), [0, 0, 0]),
           (np.diag([-1, 1, 1]), [0.25, 0.25, 0.25]),
           (np.diag([1, -1, 1]), [0.25, 0.25, 0.25])]
_CENTRING = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
FDD2_OPS = [(R, np.array(t, float) + np.array(tc, float))
            for R, t in _COSETS for tc in _CENTRING]


def _builder():
    b = MagCalcConfigBuilder()
    b.set_lattice(a=A, b=B, c=C, alpha=90, beta=90, gamma=90, space_group=43)
    b.add_wyckoff_atom(label="Cu", pos=CU, spin=0.5, ion="Cu2+")
    assert len(b.atoms_uc) == 16                      # the 16b orbit
    return b


def _frac(b):
    return {a["label"]: np.array(a["pos"], float) for a in b.atoms_uc}


def _site_at(p, frac, tol=1e-4):
    for lbl, r in frac.items():
        d = p - r
        if np.linalg.norm(d - np.round(d)) < tol:
            return lbl, np.round(d).astype(int)
    raise AssertionError(f"no site at {p}")


def _hand_orbit(frac, i, j, off):
    """Every directed bond the 16 hand-written ops generate from i->j(off), with the
    coefficient matrix M of D' = M D, plus the conflicts (a bond reached twice with
    different M, after the reverse bond is given -M)."""
    bonds, conflicts = {}, []
    ri, rj = frac[i], frac[j] + np.array(off, float)
    for R, t in FDD2_OPS:
        k, ok = _site_at(R @ ri + t, frac)
        l, ol = _site_at(R @ rj + t, frac)
        key = (k, l, tuple((ol - ok).tolist()))
        rkey = (l, k, tuple((ok - ol).tolist()))
        M = np.linalg.det(R) * R
        for kk, MM in ((key, M), (rkey, -M)):
            if kk in bonds and not np.allclose(bonds[kk], MM):
                conflicts.append((kk, bonds[kk] - MM))
            bonds.setdefault(kk, MM)
    return bonds, conflicts


def _engine_table(entries, D):
    """{(i, j, offset): D_ij as floats} from the builder's dm_interaction list, with a
    symbolic value evaluated at D = (Dx, Dy, Dz)."""
    subs = {"Dx": D[0], "Dy": D[1], "Dz": D[2]}
    out = {}
    for e in entries:
        key = (e["pair"][0], e["pair"][1], tuple(e["rij_offset"]))
        assert key not in out, f"duplicate entry {key}"
        out[key] = np.array([float(sp.sympify(v).subs(subs)) if isinstance(v, str)
                             else float(v) for v in e["value"]])
    return out


def _check_against_hand(frac, table, j, D):
    """The engine table equals the hand propagation of D from Cu0->j entry for entry,
    is antisymmetric, and is invariant under the three F-centring translations."""
    hand, conflicts = _hand_orbit(frac, "Cu0", j, (0, 0, 0))
    for key, dM in conflicts:          # the ops may disagree on a general D, not on this one
        assert np.allclose(dM @ np.asarray(D, float), 0, atol=1e-12), key
    assert set(table) == set(hand)
    for key, M in hand.items():
        assert np.allclose(table[key], M @ np.asarray(D, float), atol=1e-9), key
    for (k, l, off), v in table.items():
        assert np.allclose(table[(l, k, tuple(-np.array(off)))], -v, atol=1e-12)
        for tc in _CENTRING[1:]:
            k2, ok2 = _site_at(frac[k] + tc, frac)
            l2, ol2 = _site_at(frac[l] + np.array(off) + tc, frac)
            key2 = (k2, l2, tuple((ol2 - ok2).tolist()))
            assert key2 in table and np.allclose(table[key2], v, atol=1e-12), (
                f"bond table not invariant under centring translation {tc}: "
                f"{(k, l, off)} -> {key2}")


# ---------------------------------------------------------------------------
# The oracle itself.
# ---------------------------------------------------------------------------

def test_hand_propagation_finds_the_j2_dz_conflict_and_nothing_else():
    """The 16 hand-written Fdd2 operations disagree on the J2 orbit, and the
    disagreement lives entirely in the z row (D_z); J1 and J3 carry a general D."""
    frac = _frac(_builder())
    _, conflicts = _hand_orbit(frac, "Cu0", "Cu13", (0, 0, 0))
    assert conflicts
    for _key, dM in conflicts:
        assert np.allclose(dM[:2], 0) and not np.allclose(dM[2], 0)
    for j in ("Cu2", "Cu9"):
        assert _hand_orbit(frac, "Cu0", j, (0, 0, 0))[1] == []


# ---------------------------------------------------------------------------
# The forbidden case: J2 D_z (2-fold axis along c through the bond midpoint).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["Dx,Dy,Dz", [1.0, 0.5, 0.25]],
                         ids=["symbolic", "numeric"])
def test_fdd2_j2_dz_is_forbidden_and_raises(value):
    """The regression proper. This used to build 16 entries with an order-dependent
    sign on D_z; it must now refuse, name the component, and add nothing."""
    b = _builder()
    with pytest.raises(SymmetryConflictError) as excinfo:
        b.add_symmetry_interaction("dm", ("Cu0", "Cu13"), value, offset=[0, 0, 0])
    msg = str(excinfo.value)
    assert "Cu0->Cu13" in msg
    assert "forbids its component z" in msg
    assert "onto its own reverse" in msg
    assert "Moriya" in msg
    assert "centring" in msg
    assert b.config["interactions"]["dm_interaction"] == []


def test_fdd2_j2_message_suggests_the_allowed_value():
    b = _builder()
    with pytest.raises(SymmetryConflictError, match=r"value: Dx,Dy,0"):
        b.add_symmetry_interaction("dm", ("Cu0", "Cu13"), "Dx,Dy,Dz", offset=[0, 0, 0])
    with pytest.raises(SymmetryConflictError, match=r"value: \[1, 0\.5, 0\]"):
        b.add_symmetry_interaction("dm", ("Cu0", "Cu13"), [1.0, 0.5, 0.25],
                                   offset=[0, 0, 0])


def test_fdd2_j2_in_plane_dm_propagates_consistently():
    """What the J2 bond DOES allow: D in the ab plane. 16 directed entries, equal to
    the hand propagation, antisymmetric, centring-invariant."""
    b = _builder()
    b.add_symmetry_interaction("dm", ("Cu0", "Cu13"), "Dx,Dy,0", offset=[0, 0, 0])
    entries = b.config["interactions"]["dm_interaction"]
    assert len(entries) == 16
    D = (1.0, 0.5, 0.0)
    _check_against_hand(_frac(b), _engine_table(entries, D), "Cu13", D)


def test_forbidden_component_raises_through_the_config_route():
    """`GenericSpinModel` wraps builder errors; the conflict must surface, not be
    swallowed into a Hamiltonian missing the rule."""
    from magcalc.generic_model import GenericSpinModel
    cfg = yaml.safe_load(open(ACVO))
    cfg["interactions"]["symmetry_rules"].append(
        {"type": "dm", "distance": D_J2, "value": "D2x,D2y,D2z",
         "ref_pair": ["Cu0", "Cu13"], "offset": [0, 0, 0]})
    cfg["parameters"].update({"D2x": 1.0, "D2y": 0.5, "D2z": 0.25})
    cfg["parameter_order"] = ["D2x", "D2y", "D2z"] + cfg["parameter_order"]
    with pytest.raises(ValueError, match="forbids its component z"):
        GenericSpinModel(cfg, base_path=os.path.dirname(ACVO))


# ---------------------------------------------------------------------------
# The allowed cases: J1 (no site symmetry) and J3 -- must stay exactly as they were.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["Dx,Dy,Dz", [2.79, 1.65, 0.7]],
                         ids=["symbolic", "numeric"])
def test_fdd2_j1_all_three_components_propagate_unchanged(value):
    """J1 Cu0-Cu2 has trivial site symmetry: all three components propagate, to 32
    directed entries in the (+-Dx, +-Dy, +-Dz) pattern, equal to the hand table."""
    b = _builder()
    b.add_symmetry_interaction("dm", ("Cu0", "Cu2"), value, offset=[0, 0, 0])
    entries = b.config["interactions"]["dm_interaction"]
    assert len(entries) == 32
    D = (2.79, 1.65, 0.7)
    table = _engine_table(entries, D)
    _check_against_hand(_frac(b), table, "Cu2", D)
    for v in table.values():                           # pure sign pattern
        assert np.allclose(np.abs(v), D, atol=1e-9)
    if isinstance(value, str):
        # literally +-Dx, +-Dy, +-Dz per component (float noise aside)
        for e in entries:
            for comp, name in zip(e["value"], ("Dx", "Dy", "Dz")):
                coeffs = sp.sympify(comp).as_coefficients_dict()
                assert abs(abs(float(coeffs[sp.Symbol(name)])) - 1.0) < 1e-12
                for s, c in coeffs.items():
                    if s != sp.Symbol(name):
                        assert abs(float(c)) < 1e-12


def test_fdd2_j3_two_images_carry_opposite_dz_without_a_false_conflict():
    """Cu0->Cu9 [0,0,0] and [0,0,1] are DIFFERENT directed bonds (the 2_1 screw at
    (1/4,1/2,z) maps one onto the reverse of the other), so they carry opposite D_z.
    Distinct keys are not a conflict: the full D propagates to 32 entries."""
    b = _builder()
    b.add_symmetry_interaction("dm", ("Cu0", "Cu9"), "Dx,Dy,Dz", offset=[0, 0, 0])
    entries = b.config["interactions"]["dm_interaction"]
    assert len(entries) == 32
    D = (1.0, 0.5, 0.25)
    table = _engine_table(entries, D)
    assert np.allclose(table[("Cu0", "Cu9", (0, 0, 0))], [1.0, 0.5, 0.25])
    assert np.allclose(table[("Cu0", "Cu9", (0, 0, 1))], [1.0, 0.5, -0.25])
    _check_against_hand(_frac(b), table, "Cu9", D)


def test_shipped_acvo_config_builds_with_its_j1_dm_table_unchanged():
    """The example that found the bug. Its J1 rule `Dx,0,0` is allowed and must come
    out exactly as before: 32 directed DM entries equal to the hand propagation of
    (Dx, 0, 0), plus the 32 anisotropic-exchange and 80 Heisenberg entries."""
    from magcalc.generic_model import GenericSpinModel
    cfg = yaml.safe_load(open(ACVO))
    model = GenericSpinModel(cfg, base_path=os.path.dirname(ACVO))
    inter = model.config["interactions"]
    dm = [e for e in inter if e.get("type") in ("dm", "dm_manual")]
    aniso = [e for e in inter if e.get("type") == "anisotropic_exchange"]
    heis = [e for e in inter if e.get("type") == "heisenberg"]
    assert (len(dm), len(aniso), len(heis)) == (32, 32, 80)

    frac = {a["label"]: np.array(a["pos"], float)
            for a in model.config["crystal_structure"]["atoms_uc"]}
    D = (2.79, 0.0, 0.0)
    _check_against_hand(frac, _engine_table(dm, D), "Cu2", D)
    for e in aniso:
        assert [str(v) for v in e["value"]] == ["1.0*G1", "-1.0*G1", "-1.0*G1"]


# ---------------------------------------------------------------------------
# The same little-group condition for a full tensor, and for other lattices.
# ---------------------------------------------------------------------------

def test_interaction_matrix_on_j2_obeys_the_same_little_group():
    """Not a DM special case. The 2-fold axis through the J2 midpoint reverses the
    bond, so R J R^T must equal J^T with R = diag(-1,-1,1): the xy block has to be
    SYMMETRIC (Gamma_xy allowed, D_z forbidden) and the xz, yz blocks ANTISYMMETRIC
    (D_x, D_y allowed, Gamma_xz, Gamma_yz forbidden)."""
    allowed = [["J", "G", "-Dy"], ["G", "J", "Dx"], ["Dy", "-Dx", "J"]]
    b = _builder()
    b.add_symmetry_interaction("interaction_matrix", ("Cu0", "Cu13"), allowed,
                               offset=[0, 0, 0])
    assert len(b.config["interactions"]["interaction_matrix"]) == 16

    for bad, names in (([["J", "Dz", 0], ["-Dz", "J", 0], [0, 0, "J"]], "entries xy, yx"),
                       ([["J", 0, "G"], [0, "J", 0], ["G", 0, "J"]], "entries xz, zx")):
        b = _builder()
        with pytest.raises(SymmetryConflictError, match=names):
            b.add_symmetry_interaction("interaction_matrix", ("Cu0", "Cu13"), bad,
                                       offset=[0, 0, 0])
        assert b.config["interactions"]["interaction_matrix"] == []


def test_anisotropic_exchange_on_j2_is_allowed():
    b = _builder()
    b.add_symmetry_interaction("anisotropic_exchange", ("Cu0", "Cu13"), "G1,-G1,-G1",
                               offset=[0, 0, 0])
    assert len(b.config["interactions"]["anisotropic_exchange"]) == 16


def test_centrosymmetric_bond_forbids_every_dm_component():
    """Moriya's first rule: an inversion centre at the midpoint kills D entirely.
    P4/mmm, one atom, the a-axis bond."""
    b = MagCalcConfigBuilder()
    b.set_lattice(a=4.0, c=5.0, space_group=123)
    b.add_wyckoff_atom("Cu", [0, 0, 0], 1.0)
    with pytest.raises(SymmetryConflictError, match="forbids its components x, y, z"):
        b.add_symmetry_interaction("dm", ("Cu", "Cu"), "Dx,Dy,Dz", offset=[1, 0, 0])


def _p23(c=4.0):
    """P23 (No. 195), one atom at 1a: the bond along c has 2_z along it (preserving)
    and 2_x across it (reversing), so D must be PARALLEL to the bond (Moriya: an
    n-fold axis along the bond); the 3-fold axes carry it to the a and b bonds."""
    b = MagCalcConfigBuilder()
    b.set_lattice(a=4.0, b=4.0, c=c, space_group=195)
    b.add_wyckoff_atom("A", [0, 0, 0], 1.0)
    return b


def test_cubic_p23_dm_along_the_bond_is_allowed_and_transverse_is_not():
    b = _p23()
    b.add_symmetry_interaction("dm", ("A", "A"), "0,0,Dz", offset=[0, 0, 1])
    entries = b.config["interactions"]["dm_interaction"]
    assert len(entries) == 6                           # +-a, +-b, +-c bonds
    for e in entries:                                  # D parallel to its own bond
        v = np.array([float(sp.sympify(x).subs({"Dz": 1.0})) for x in e["value"]])
        off = np.array(e["rij_offset"], float)
        assert np.allclose(np.cross(v, off), 0, atol=1e-12) and abs(abs(v @ off) - 1) < 1e-12
    with pytest.raises(SymmetryConflictError, match="forbids its component x"):
        _p23().add_symmetry_interaction("dm", ("A", "A"), "Dx,0,Dz", offset=[0, 0, 1])


def test_lattice_noise_is_not_a_conflict_but_a_forbidden_component_still_is():
    """A CIF-quality 'cubic' cell (c off by 1e-6) makes every axis-mixing operation
    slightly non-orthogonal, so ALL values propagate slightly differently through
    different operations. That must not read as a symmetry conflict (the tolerance
    follows the lattice's own noise), while a genuinely forbidden component, which
    disagrees by twice its size, still must."""
    b = _p23(c=4.0 * (1 + 1e-6))
    b.add_symmetry_interaction("dm", ("A", "A"), [0.0, 0.0, 1.0], offset=[0, 0, 1])
    assert len(b.config["interactions"]["dm_interaction"]) == 6
    with pytest.raises(SymmetryConflictError, match="forbids its component x"):
        _p23(c=4.0 * (1 + 1e-6)).add_symmetry_interaction(
            "dm", ("A", "A"), [0.01, 0.0, 1.0], offset=[0, 0, 1])


def test_exact_lattice_catches_a_tiny_forbidden_component():
    """On an exact lattice the check is tight: a transverse component 1e-7 of the
    allowed one is still refused -- it would still break the lattice symmetry."""
    with pytest.raises(SymmetryConflictError, match="forbids its component x"):
        _p23().add_symmetry_interaction("dm", ("A", "A"), [1e-7, 0.0, 1.0],
                                        offset=[0, 0, 1])


def test_symmetry_conflict_is_a_value_error():
    """So every `except ValueError` around rule expansion (and `pytest.raises`
    guards in older tests) still catches it."""
    assert issubclass(SymmetryConflictError, ValueError)
