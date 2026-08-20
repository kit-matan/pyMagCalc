"""Magnetic CIF (mCIF) import.

An mCIF encodes a magnetic structure through a magnetic space group (symmetry ops with a
time-reversal parity). Expanding those ops over the asymmetric-unit moments gives the full
magnetic cell.

Validated against Sunny 0.8.1 on the real-world TbSb file (tests/data_TbSb.mcif, from
Sunny's own test set): same 6 Tb sites, same moment directions, including the G-type
alternation driven by the R-centering ANTI-translations.
"""
import os

import numpy as np
import pytest

from magcalc.mcif import mcif_to_config_fragment, parse_magnetic_symop, read_mcif

HERE = os.path.dirname(__file__)
TBSB = os.path.join(HERE, "data_TbSb.mcif")
INPLANE = os.path.join(HERE, "..", "examples", "materials", "mcif", "afm_inplane.mcif")


# ------------------------------------------------------------------ symop parsing
def test_parse_magnetic_symop():
    R, T, p = parse_magnetic_symop("x-y,-x,z+1/2,+1")
    assert np.allclose(R, [[1, -1, 0], [-1, 0, 0], [0, 0, 1]])
    assert np.allclose(T, [0, 0, 0.5])
    assert p == 1
    _, _, p2 = parse_magnetic_symop("-x,-y,z+1/2,-1")
    assert p2 == -1


def test_parse_rejects_non_magnetic_symop():
    with pytest.raises(ValueError, match="4 comma"):
        parse_magnetic_symop("x,y,z")           # missing the parity


# ------------------------------------------------------------------ TbSb vs Sunny
# Sunny 0.8.1: parse_mcif_data + transform_dipole, deduped by position.
SUNNY_TBSB = {
    (0.0, 0.0, 0.0): [0, 0, -1],
    (0.0, 0.0, 0.5): [0, 0, 1],
    (0.3333, 0.6667, 0.1667): [0, 0, 1],
    (0.3333, 0.6667, 0.6667): [0, 0, -1],
    (0.6667, 0.3333, 0.3333): [0, 0, -1],
    (0.6667, 0.3333, 0.8333): [0, 0, 1],
}


def test_tbsb_matches_sunny():
    """Real R-centred magnetic space group with anti-translations. Every Tb site and its
    moment direction must match Sunny."""
    d = read_mcif(TBSB)
    assert len(d["sites"]) == 6
    got = {tuple(np.round(s["pos"], 4)): np.round(s["direction"], 4) for s in d["sites"]}
    for pos, want in SUNNY_TBSB.items():
        key = min(got, key=lambda k: np.linalg.norm(np.array(k) - np.array(pos)))
        assert np.allclose(np.array(key), pos, atol=1e-3), f"missing site {pos}"
        assert np.allclose(got[key], want, atol=1e-3), f"{pos}: {got[key]} vs {want}"


def test_tbsb_is_a_g_type_afm_along_c():
    """Physics sanity: all moments along +/-c, and half up / half down."""
    d = read_mcif(TBSB)
    dirs = np.array([s["direction"] for s in d["sites"]])
    assert np.allclose(np.abs(dirs[:, 2]), 1.0)          # all along c
    assert np.isclose(dirs[:, 2].sum(), 0.0)             # compensated (AFM)


# ------------------------------------------------------------------ in-plane + Cartesian
def test_inplane_moment_and_cartesian_conversion():
    """Hand-checkable: m = 3 mu_B along a (a = 4 A), body-centring anti-translation
    flips it. Directions +a and -a; |moment| = 3 mu_B.

    This assertion used to read 12.0 = 3 * a, i.e. the components were multiplied
    by the FULL lattice vectors. That number was never checked against anything
    outside this package -- it was the reader's own rule written down twice -- and
    it is wrong: FullProf's own mCIF export (Ho2BaNiO5, in the FullProf Examples)
    states `spherical_modulus 8.99423` for components (-0.1441, 0, -8.9931) in a
    7.51 x 5.74 x 22.56 A cell, which is their plain Euclidean norm and could not
    be anything else for a Ho3+ ion. So the components are mu_B on UNIT crystal
    axes, and `moment_basis='lattice_vectors'` is kept only to read files written
    under the other reading. See tests/test_diffraction.py.
    """
    d = read_mcif(INPLANE)
    assert len(d["sites"]) == 2
    by_pos = {tuple(np.round(s["pos"], 3)): s for s in d["sites"]}
    s0 = by_pos[(0.0, 0.0, 0.0)]
    s1 = by_pos[(0.5, 0.5, 0.5)]
    assert np.allclose(s0["direction"], [1, 0, 0])
    assert np.allclose(s1["direction"], [-1, 0, 0])       # time-reversed image
    assert np.isclose(np.linalg.norm(s0["moment"]), 3.0)
    # the old reading remains available, and reproduces the old number exactly
    d_old = read_mcif(INPLANE, moment_basis="lattice_vectors")
    old0 = [s for s in d_old["sites"] if np.allclose(s["pos"], [0, 0, 0], atol=1e-3)][0]
    assert np.isclose(np.linalg.norm(old0["moment"]), 12.0)   # 3 * a
    assert np.allclose(old0["direction"], s0["direction"])    # cubic: same direction


def test_inconsistent_mcif_is_rejected(tmp_path):
    """A moment that is not invariant under its own site symmetry is unphysical.

    The reader must refuse rather than pick one of the two images silently. The
    message names the STABILISER, because that is the field to go and look at:
    eight MAGNDATA deposits fail exactly this way, and "internally inconsistent"
    did not say where to look or whose problem it was.
    """
    p = tmp_path / "bad.mcif"
    p.write_text(
        "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_space_group_symop_magn_operation.id\n"
        "_space_group_symop_magn_operation.xyz\n1 x,y,z,+1\n2 y,-x,z,+1\n"   # 4-fold about z
        "loop_\n_space_group_symop_magn_centering.id\n"
        "_space_group_symop_magn_centering.xyz\n1 x,y,z,+1\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\n_atom_site_occupancy\nFe1 0.0 0.0 0.0 1.0\n"
        "loop_\n_atom_site_moment.label\n_atom_site_moment.crystalaxis_x\n"
        "_atom_site_moment.crystalaxis_y\n_atom_site_moment.crystalaxis_z\n"
        "Fe1 3.0 0.0 0.0\n")   # in-plane moment on a site fixed by a 4-fold -> inconsistent
    with pytest.raises(ValueError, match="not invariant under its own magnetic stabiliser"):
        read_mcif(str(p))


# ------------------------------------------------------------------ config + runner
def test_config_fragment_shape():
    frag = mcif_to_config_fragment(TBSB, spin_S=6.0, ion="Tb3+")
    assert len(frag["crystal_structure"]["atoms_uc"]) == 6
    assert frag["magnetic_structure"]["type"] == "pattern"
    assert all(a["spin_S"] == 6.0 and a["ion"] == "Tb3+"
               for a in frag["crystal_structure"]["atoms_uc"])
    assert len(frag["magnetic_structure"]["directions"]) == 6


def test_runner_from_mcif_end_to_end():
    from magcalc.runner import run_calculation
    cfg = os.path.join(HERE, "..", "examples", "materials", "mcif",
                       "config_afm_inplane.yaml")
    run_calculation(cfg)          # must not raise


def test_a_cif_uncertainty_with_no_closing_bracket_is_still_a_number():
    """Deposited files do not always close the bracket, and one of them is real.

    MAGNDATA 1.400 (TbAg) carries its Tb moment as `8.95(5`, never closed.
    Requiring the `)` made that published entry unreadable; the uncertainty is
    discarded either way, so stripping from the bracket to the end of the token
    loses nothing and reads one more real file.
    """
    from magcalc.mcif import _cif_float

    assert _cif_float("8.95(5") == 8.95
    assert _cif_float("8.95(5)") == 8.95
    assert _cif_float(" 7.6637(4) ") == 7.6637
    assert _cif_float('"3.25(12)"') == 3.25
    assert _cif_float("-0.5(1)") == -0.5
    assert _cif_float("5.0") == 5.0


def test_two_species_on_one_site_is_a_solid_solution_not_a_broken_file(tmp_path):
    """NdMn(0.8)Fe(0.2)O3 is the real case: MAGNDATA 0.659, and 21 more like it.

    Mn1 and Fe1 sit at the same coordinates with occupancies 0.8 and 0.2. Calling
    that "internally inconsistent" blamed the deposit for being a solid solution;
    it is a structure this reader does not model, which is a different statement
    and a different exception, so a caller can skip one and investigate the other.
    """
    from magcalc.mcif import SharedSiteError

    p = tmp_path / "alloy.mcif"
    p.write_text(
        "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 4.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_space_group_symop_magn_operation.id\n"
        "_space_group_symop_magn_operation.xyz\n1 x,y,z,+1\n"
        "loop_\n_space_group_symop_magn_centering.id\n"
        "_space_group_symop_magn_centering.xyz\n1 x,y,z,+1\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\n_atom_site_occupancy\n"
        "Mn1 0.0 0.0 0.5 0.8\nFe1 0.0 0.0 0.5 0.2\n"
        "loop_\n_atom_site_moment.label\n_atom_site_moment.crystalaxis_x\n"
        "_atom_site_moment.crystalaxis_y\n_atom_site_moment.crystalaxis_z\n"
        "Mn1 0.0 0.0 3.0\nFe1 0.0 0.0 1.0\n")

    with pytest.raises(SharedSiteError, match=r"Mn1 \(occ 0.8\) and Fe1 \(occ 0.2\) share"):
        read_mcif(str(p))
    assert issubclass(SharedSiteError, ValueError)


def test_two_images_of_one_site_are_matched_by_distance_not_by_a_grid_bucket(tmp_path):
    """A coordinate on a bucket boundary split one site into two, and lost four.

    The old key was `round(r/tol) % (1/tol)`. MAGNDATA 1.14 has Ho at
    z = 0.10125, so z/tol = 1012.5 exactly: round-half-to-even sends it to 1012
    while the same image computed a floating hair higher goes to 1013. The
    duplicate then survived expansion, and a write/read round trip -- where the
    second read happened to merge it -- silently lost four of twenty-eight sites.
    """
    p = tmp_path / "boundary.mcif"
    p.write_text(
        "_cell_length_a 4.0\n_cell_length_b 4.0\n_cell_length_c 8.0\n"
        "_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\n"
        "loop_\n_space_group_symop_magn_operation.id\n"
        "_space_group_symop_magn_operation.xyz\n"
        "1 x,y,z,+1\n2 -x,-y,z,+1\n"
        "loop_\n_space_group_symop_magn_centering.id\n"
        "_space_group_symop_magn_centering.xyz\n1 x,y,z,+1\n"
        "loop_\n_atom_site_label\n_atom_site_fract_x\n_atom_site_fract_y\n"
        "_atom_site_fract_z\n_atom_site_occupancy\nHo1 0.0 0.0 0.10125 1.0\n"
        "loop_\n_atom_site_moment.label\n_atom_site_moment.crystalaxis_x\n"
        "_atom_site_moment.crystalaxis_y\n_atom_site_moment.crystalaxis_z\n"
        "Ho1 0.0 0.0 9.0\n")

    # (0,0,z) is fixed by -x,-y,z, so there is exactly ONE site, not two
    assert len(read_mcif(str(p))["sites"]) == 1
