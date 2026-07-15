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
    """Hand-checkable: m = 3 along a (a = 4 A), body-centring anti-translation flips it.
    Cartesian |moment| = 3 * 4 = 12; directions +a and -a."""
    d = read_mcif(INPLANE)
    assert len(d["sites"]) == 2
    by_pos = {tuple(np.round(s["pos"], 3)): s for s in d["sites"]}
    s0 = by_pos[(0.0, 0.0, 0.0)]
    s1 = by_pos[(0.5, 0.5, 0.5)]
    assert np.allclose(s0["direction"], [1, 0, 0])
    assert np.allclose(s1["direction"], [-1, 0, 0])       # time-reversed image
    assert np.isclose(np.linalg.norm(s0["moment"]), 12.0)  # 3 * a


def test_inconsistent_mcif_is_rejected(tmp_path):
    """A magnetic symop set that maps a fixed site to two different moments is
    unphysical; the reader must refuse rather than pick one silently."""
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
    with pytest.raises(ValueError, match="internally inconsistent"):
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
