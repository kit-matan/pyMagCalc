"""Cell utilities: primitive / standardized cells, sub-crystals, BZ paths (Gap 4 #27).

Sunny analogues: `primitive_cell`, `standardize`, `subcrystal`,
`print_irreducible_bz_paths`. The first three are spglib wrappers, so the oracle is
partly spglib itself -- which would be circular if that were all. It is not: the
checks below are *arithmetic identities about cells* that hold independently of which
library computed them (a body-centred cell has exactly half the conventional volume
and one lattice point; standardization is idempotent; a sub-crystal is a subset), plus
a cross-check against the analytic answer for a lattice whose primitive cell is known
by hand.
"""
import numpy as np
import pytest

from magcalc.cell_utils import (cell_from_structure, describe_cell,
                                irreducible_bz_path, primitive_cell, standardize_cell,
                                subcrystal)

A = 4.0


def _bcc_conventional():
    """Body-centred cubic written in its CONVENTIONAL cell: 2 atoms, volume a^3.
    Its primitive cell has 1 atom and volume a^3/2 -- known analytically."""
    lattice = np.diag([A, A, A]).astype(float)
    positions = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    numbers = np.array([1, 1])
    return (lattice, positions, numbers)


def _rocksalt():
    """NaCl-type: FCC lattice, two species. Conventional cell has 8 atoms and volume
    a^3; the primitive has 2 atoms and a^3/4."""
    lattice = np.diag([A, A, A]).astype(float)
    cat = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
    an = [[.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]
    positions = np.array(cat + an, dtype=float)
    numbers = np.array([1] * 4 + [2] * 4)
    return (lattice, positions, numbers)


def test_bcc_primitive_cell_is_the_analytic_one():
    """Half the volume and one atom -- a fact about body-centring, not about spglib."""
    conv = _bcc_conventional()
    prim = primitive_cell(conv)
    assert len(prim[2]) == 1
    assert abs(np.linalg.det(prim[0])) == pytest.approx(A ** 3 / 2, rel=1e-9)
    # the primitive vectors are the half-diagonals: every one has length a*sqrt(3)/2
    lens = np.linalg.norm(prim[0], axis=1)
    assert lens == pytest.approx(np.full(3, A * np.sqrt(3) / 2), rel=1e-9)


def test_rocksalt_primitive_cell_is_the_analytic_one():
    prim = primitive_cell(_rocksalt())
    assert sorted(prim[2].tolist()) == [1, 2]
    assert abs(np.linalg.det(prim[0])) == pytest.approx(A ** 3 / 4, rel=1e-9)


def test_standardize_recovers_the_conventional_cell():
    """Standardizing the PRIMITIVE bcc cell must give the conventional one back:
    volume a^3, two atoms, cubic angles. This is the round trip, so an error in
    either direction shows up."""
    prim = primitive_cell(_bcc_conventional())
    conv = standardize_cell(prim)
    d = describe_cell(conv)
    assert d["n_sites"] == 2
    assert d["volume"] == pytest.approx(A ** 3, rel=1e-9)
    assert (d["alpha"], d["beta"], d["gamma"]) == pytest.approx((90, 90, 90), abs=1e-9)


def test_standardization_is_idempotent():
    """Standardizing twice must change nothing -- true of any correct canonical
    form, and it needs no reference implementation at all."""
    once = standardize_cell(_rocksalt())
    twice = standardize_cell(once)
    assert twice[0] == pytest.approx(once[0], abs=1e-10)
    assert len(twice[2]) == len(once[2])
    assert abs(np.linalg.det(twice[0])) == pytest.approx(
        abs(np.linalg.det(once[0])), rel=1e-12)


def test_standardize_to_primitive_agrees_with_find_primitive():
    """Two spglib routes to the same object must not disagree."""
    conv = _rocksalt()
    a = primitive_cell(conv)
    b = standardize_cell(conv, to_primitive=True)
    assert abs(np.linalg.det(a[0])) == pytest.approx(abs(np.linalg.det(b[0])), rel=1e-9)
    assert len(a[2]) == len(b[2])


# --------------------------------------------------------------------------
STRUCT = {
    "lattice_vectors": [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 8.0]],
    "atoms_uc": [
        {"label": "Cu0", "pos": [0.0, 0.0, 0.0], "spin_S": 0.5, "element": "Cu"},
        {"label": "Cu1", "pos": [0.5, 0.5, 0.0], "spin_S": 0.5, "element": "Cu"},
        {"label": "O0", "pos": [0.5, 0.0, 0.0], "element": "O"},
        {"label": "O1", "pos": [0.0, 0.5, 0.25], "element": "O"},
    ],
}


def test_subcrystal_keeps_only_the_named_species():
    out = subcrystal(STRUCT, ["Cu"])
    assert [a["label"] for a in out["atoms_uc"]] == ["Cu0", "Cu1"]
    assert out["lattice_vectors"] == STRUCT["lattice_vectors"]
    # original untouched
    assert len(STRUCT["atoms_uc"]) == 4


def test_subcrystal_matches_on_chemistry_not_site_index():
    """`Cu` must keep Cu0 AND Cu1: the trailing digits are site indices, not part of
    the element. (This is the same rule the symmetry detector uses -- treating them
    as distinct species collapses the detected group to P1.)"""
    assert len(subcrystal(STRUCT, ["Cu"])["atoms_uc"]) == 2
    assert len(subcrystal(STRUCT, ["Cu", "O"])["atoms_uc"]) == 4


def test_subcrystal_refuses_an_absent_species():
    """A typo must not quietly return an empty magnetic sublattice."""
    with pytest.raises(ValueError, match="not in the structure"):
        subcrystal(STRUCT, ["Fe"])


def test_cell_from_structure_round_trips_species():
    cell, species = cell_from_structure(STRUCT)
    assert species == ["Cu", "Cu", "O", "O"]
    assert cell[1].shape == (4, 3)
    assert sorted(set(cell[2].tolist())) == [1, 2]
    assert describe_cell(cell, sorted(set(species)))["composition"] == {"Cu": 2, "O": 2}


def test_cell_from_structure_needs_an_expanded_structure():
    with pytest.raises(ValueError, match="lattice_vectors"):
        cell_from_structure({"lattice_parameters": {"a": 5.0}})


# --------------------------------------------------------------------------
try:                                              # pragma: no cover - env-dependent
    import seekpath  # noqa: F401
    _HAS_SEEKPATH = True
except ImportError:
    _HAS_SEEKPATH = False


@pytest.mark.skipif(not _HAS_SEEKPATH, reason="optional dependency `seekpath` absent")
def test_bz_path_for_bcc():
    """seekpath's own space-group determination must agree with spglib's, and the
    path must start from Gamma with Gamma at the origin."""
    bz = irreducible_bz_path(_bcc_conventional())
    assert bz["spacegroup_number"] == 229           # Im-3m
    assert "GAMMA" in bz["point_coords"]
    assert bz["point_coords"]["GAMMA"] == pytest.approx([0, 0, 0], abs=1e-12)
    assert len(bz["path"]) > 0


def test_bz_path_reports_a_missing_optional_dependency_clearly():
    """Absent seekpath must produce an actionable ImportError, not a traceback from
    deep inside. (When it IS installed there is nothing to check.)"""
    if _HAS_SEEKPATH:
        pytest.skip("seekpath is installed")
    with pytest.raises(ImportError, match="pip install seekpath"):
        irreducible_bz_path(_bcc_conventional())
