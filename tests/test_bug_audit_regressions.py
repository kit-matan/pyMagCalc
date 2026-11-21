"""Regressions for the 2026-11-21 bug audit, one test per defect.

Each test is the audit's reproduction turned into an assertion: it fails on the code
before the fix and passes after. The degenerate-block defect (`linalg.KKdMatrix` and
its Fortran port) has its own file, `test_accidental_degeneracy.py`.
"""
import copy
from unittest import mock

import numpy as np
import pytest
import sympy as sp

import magcalc as mc
from magcalc import classical_dynamics, fitting, scga, spiral_opt
from magcalc import thermal_mc as tmc
from magcalc.config_builder import MagCalcConfigBuilder
from magcalc.forward import GaussianEnergyResolution
from magcalc.generic_model import GenericSpinModel
from magcalc.mcif import read_mcif
from magcalc.qe_resolution import GeometrySpread, QEResolution
from magcalc.runner import generate_q_path_from_config, resolve_S_and_params

CUBIC = {"a": 4.0, "b": 4.0, "c": 4.0, "alpha": 90, "beta": 90, "gamma": 90}
FM_Z = {"type": "pattern", "pattern_type": "ferromagnetic", "direction": [0, 0, 1]}


def _neel_chain(S=1.0, J=1.0, a=3.0):
    m = GenericSpinModel({
        "crystal_structure": {"lattice_vectors": [[2 * a, 0, 0], [0, 10, 0], [0, 0, 10]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                                           {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": a, "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]}})
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base="bug_audit", hamiltonian_params=[])


def _spin_symbols(model):
    return [[sp.Symbol(f"S{i}{a}") for a in "xyz"]
            for i in range(len(model.atom_pos_ouc()))]


def _sia_term(model, params):
    S = _spin_symbols(model)
    return sp.expand(model._compute_sia_terms(S, params, model._resolve_param_map(params))), S


def _one_site_sia(sia_entry):
    return GenericSpinModel({
        "crystal_structure": {"lattice_parameters": dict(CUBIC),
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0}]},
        "parameters": {"S": 1.0, "J": 1.0, "D": 0.5},
        "interactions": [{"type": "heisenberg", "pair": ["A", "A"],
                          "rij_offset": [1, 0, 0], "value": "J"}, sia_entry],
        "magnetic_structure": dict(FM_Z)})


# --------------------------------------------------------------------------- generic_model

@pytest.mark.parametrize("value, factor", [("D", 1.0), ("-D", -1.0), ("2*D", 2.0)])
def test_sia_expression_value_resolves_through_parameters(value, factor):
    """`type: sia` with value '-D' or '2*D' used to fall back to the FIRST parameter
    (J) on the symbolic LSWT path."""
    m = _one_site_sia({"type": "sia", "value": value, "axis": [0, 0, 1], "atoms": ["A"]})
    J, D = sp.symbols("J D")
    sym, S = _sia_term(m, [J, D])
    assert sp.simplify(sym - factor * D * S[0][2] ** 2) == 0
    num, S = _sia_term(m, [1.0, 0.5])
    assert sp.simplify(num - factor * 0.5 * S[0][2] ** 2) == 0


def test_sia_accepts_atom_label_and_K_keys():
    """The builder's old spelling {atom_label, K} was read as 'all atoms, value missing'."""
    m = _one_site_sia({"type": "sia", "K": "D", "axis": [0, 0, 1], "atom_label": "A"})
    J, D = sp.symbols("J D")
    sym, S = _sia_term(m, [J, D])
    assert sp.simplify(sym - D * S[0][2] ** 2) == 0


def test_builder_sia_lands_on_the_named_site_only():
    b = MagCalcConfigBuilder()
    b.set_lattice(a=4.0, b=4.0, c=4.0, alpha=90, beta=90, gamma=90, space_group=1)
    b.add_wyckoff_atom("A", [0, 0, 0], 1.0)
    b.add_wyckoff_atom("B", [0.5, 0.5, 0.5], 1.0)
    b.add_symmetry_interaction("heisenberg", ("A", "B"), "J", offset=[0, 0, 0])
    b.add_single_ion_anisotropy("A", "D", axis=[0, 0, 1])
    b.config["parameters"] = {"S": 1.0, "J": 1.0, "D": -0.5}
    b.config["magnetic_structure"] = {"type": "pattern", "pattern_type": "generic",
                                      "directions": [[0, 0, 1], [0, 0, -1]]}
    m = GenericSpinModel(copy.deepcopy(b.config))
    J, D = sp.symbols("J D")
    sym, S = _sia_term(m, [J, D])
    assert sp.simplify(sym - D * S[0][2] ** 2) == 0


def test_sia_unknown_site_label_raises():
    """`atoms: ['Fe']` on a cell whose site is 'Fe1' used to drop the term silently."""
    m = GenericSpinModel({
        "crystal_structure": {"lattice_parameters": dict(CUBIC, space_group=221),
                              "atoms_uc": [{"label": "Fe1", "pos": [0, 0, 0], "spin_S": 1.0}]},
        "parameters": {"S": 1.0, "J": 1.0, "D": 0.5},
        "interactions": [{"type": "heisenberg", "pair": ["Fe1", "Fe1"],
                          "rij_offset": [1, 0, 0], "value": "J"},
                         {"type": "sia", "value": "D", "axis": [0, 0, 1], "atoms": ["Fe"]}],
        "magnetic_structure": dict(FM_Z)})
    J, D = sp.symbols("J D")
    with pytest.raises(ValueError, match="match no site"):
        _sia_term(m, [J, D])


def test_lattice_b_and_c_default_to_a():
    """A cubic cell written with only `a` was expanded with b = c = 1."""
    m = GenericSpinModel({
        "crystal_structure": {
            "lattice_parameters": {"a": 4.0, "alpha": 90, "beta": 90, "gamma": 90,
                                   "space_group": 221},
            "wyckoff_atoms": [{"label": "M", "pos": [0, 0, 0], "spin_S": 1.0}]},
        "parameters": {"J": 1.0, "S": 1.0},
        "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 4.0,
                                             "value": "J"}]},
        "magnetic_structure": dict(FM_Z)})
    assert len([e for e in m.config["interactions"] if e.get("type") == "heisenberg"]) == 6
    cell = np.array(m.unit_cell(), float)
    np.testing.assert_allclose(cell, 4.0 * np.eye(3), atol=1e-12)


# -------------------------------------------------------------------------- config_builder

def test_anisotropic_exchange_rule_keeps_off_diagonal_terms():
    """On a triangular lattice the 6-fold images of diag(Jx, Jy, Jz), Jx != Jy, have xy
    terms. The rule used to emit only the diagonal; it is now promoted to
    interaction_matrix and equals the same tensor given as a matrix."""
    def build():
        b = MagCalcConfigBuilder()
        b.set_lattice(a=3.0, b=3.0, c=10.0, alpha=90, beta=90, gamma=120, space_group=191)
        b.add_wyckoff_atom(label="M", pos=[0, 0, 0], spin=1.0, ion="Fe3+")
        return b

    V = [1.0, 0.2, 0.5]
    b1 = build()
    b1.add_symmetry_interaction("anisotropic_exchange", ("M", "M"), V, offset=[1, 0, 0])
    b2 = build()
    b2.add_symmetry_interaction("interaction_matrix", ("M", "M"), np.diag(V).tolist(),
                                offset=[1, 0, 0])

    def tensors(b):
        return {tuple(e["rij_offset"]): np.array(e["value"], float)
                for e in b.config["interactions"].get("interaction_matrix", [])}

    got, want = tensors(b1), tensors(b2)
    assert not b1.config["interactions"].get("anisotropic_exchange")
    assert len(want) == 6 and set(got) == set(want)
    for off in want:
        np.testing.assert_allclose(got[off], want[off], atol=1e-12)
    assert max(abs(t[0, 1]) for t in got.values()) > 0.3


@pytest.mark.parametrize("z", [0.5, 2.5])
def test_symmetry_rule_finds_site_stored_outside_first_cell(z):
    """bcc nearest neighbours: a B site stored at z = 2.5 used to lose its bonds."""
    m = GenericSpinModel({
        "crystal_structure": {
            "lattice_parameters": {"a": 3.0, "b": 3.0, "c": 3.0, "alpha": 90, "beta": 90,
                                   "gamma": 90, "space_group": 229},
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0},
                         {"label": "B", "pos": [0.5, 0.5, z], "spin_S": 1.0}]},
        "parameters": {"J": 1.0, "S": 1.0},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "ref_pair": ["A", "B"],
             "offset": [0, 0, -int(np.floor(z))], "value": "J"}]},
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]}})
    assert len([e for e in m.config["interactions"] if e.get("type") == "heisenberg"]) == 16


def test_analyze_bond_symmetry_reaches_max_distance():
    """The search was a fixed +-2 cells: on a 2.5 A cubic cell it stopped short of the
    7.5 A shell, which holds two orbits ([3,0,0] x6 and [2,2,1] x24)."""
    b = MagCalcConfigBuilder()
    b.set_lattice(a=2.5, b=2.5, c=2.5, alpha=90, beta=90, gamma=90, space_group=221)
    b.add_wyckoff_atom("M", [0, 0, 0], 1.0)
    orbits = b.analyze_bond_symmetry(max_distance=8.0)
    at75 = sorted(o["multiplicity"] for o in orbits if abs(o["distance"] - 7.5) < 1e-3)
    assert at75 == [6, 24]


def test_space_group_origin_choice_is_selectable_and_warned():
    """Fd-3m (227) silently meant origin choice 1. A site at (0,0,0) is the 8-site
    diamond there and the 16-site pyrochlore 16c in origin choice 2 (Hall 526)."""
    def n_sites(hall):
        b = MagCalcConfigBuilder()
        with mock.patch("magcalc.config_builder.logger.warning") as warn:
            b.set_lattice(a=10.0, b=10.0, c=10.0, alpha=90, beta=90, gamma=90,
                          space_group=227, hall_number=hall)
        b.add_wyckoff_atom("Yb", [0.0, 0.0, 0.0], 0.5)
        warned = any("origin choice" in str(c.args[0]) for c in warn.call_args_list)
        return len(b.atoms_uc), warned

    assert n_sites(None) == (8, True)
    assert n_sites(526) == (16, False)
    with pytest.raises(ValueError):
        MagCalcConfigBuilder().set_lattice(a=10.0, b=10.0, c=10.0, alpha=90, beta=90,
                                           gamma=90, space_group=227, hall_number=1)


# --------------------------------------------------------------------------------- fitting

def test_nexus_loader_feeds_fit_problem():
    """The loader returned no `q` and zero errors on empty bins, so FitProblem either
    raised or divided by zero."""
    nx = {"x": np.array([0.5, 1.0]), "energy": np.array([1.0, 2.0, 3.0]),
          "signal": np.array([[4.0, 0.0, 1.0], [9.0, 0.0, 0.0]]), "params": {}}
    with mock.patch("magcalc.binning.load_nxs", return_value=nx):
        d = fitting.load_fit_data("powder", "x.nxs")
    assert "q" in d
    assert np.all(np.asarray(d["error"]) > 0)
    fitting.FitProblem("powder", None, d, [], {}, np.eye(3))


def test_fit_problem_takes_powder_samples():
    d = {"q": np.array([1.0]), "energy": np.array([1.0]),
         "intensity": np.array([1.0]), "error": np.array([1.0])}
    fp = fitting.FitProblem("powder", mock.Mock(), d, [], {}, np.eye(3), powder_samples=7)
    assert fp._powder_samples == 7


def test_fit_expr_on_non_varied_parameter_is_kept():
    p = fitting.build_lmfit_params(
        {"type": "dispersion", "vary": ["J1"], "expr": {"J2": "0.5*J1"}},
        {"J1": 1.0, "J2": 0.5, "D": 0.1}, ["J1", "J2", "D"])
    assert p["J2"].expr == "0.5*J1" and p["J1"].vary


@pytest.mark.parametrize("bounds, lo, hi", [([0, None], 0.0, np.inf),
                                            ([None, 5], -np.inf, 5.0),
                                            (None, -np.inf, np.inf)])
def test_fit_bounds_accept_null(bounds, lo, hi):
    p = fitting.build_lmfit_params(
        {"type": "dispersion", "vary": ["J1"], "bounds": {"J1": bounds}},
        {"J1": 1.0, "J2": 0.5, "D": 0.1}, ["J1", "J2", "D"])
    assert (p["J1"].min, p["J1"].max) == (lo, hi)


# ------------------------------------------------------------------------------------ core

def test_calculate_sqw_accepts_a_single_1d_q():
    c = _neel_chain()
    q = np.array([0.3, 0.1, 0.0])
    one = c.calculate_sqw(q)
    ref = c.calculate_sqw(q[None])
    assert one.energies.shape == (1, 2)
    np.testing.assert_allclose(one.energies, ref.energies, rtol=1e-12)
    np.testing.assert_allclose(one.intensities, ref.intensities, rtol=1e-12)


def test_fortran_backend_takes_dict_cross_section():
    pytest.importorskip("fmagcalc")
    c = _neel_chain()
    q = np.array([[0.3, 0.1, 0.0], [0.7, 0.0, 0.2]])
    pol = {"polarization": [0, 0, 1], "channel": "sf"}
    f = c.calculate_sqw(q, backend="fortran", cross_section=pol)
    n = c.calculate_sqw(q, cross_section=pol)
    np.testing.assert_allclose(f.intensities.sum(axis=1), n.intensities.sum(axis=1),
                               rtol=1e-8)


# ------------------------------------------------------------- spiral, SCGA, MC, contraction

def test_spiral_optimizer_uses_per_site_spin_lengths():
    """Mixed-spin (1, 5/2) AFM chain: the classical energy per cell is -2 J S_A S_B = -5.
    One common spin length gave -2."""
    m = GenericSpinModel({
        "crystal_structure": {"lattice_vectors": [[2., 0, 0], [0, 10, 0], [0, 0, 10]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0},
                                           {"label": "B", "pos": [0.5, 0, 0], "spin_S": 2.5}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": 1.0},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": 1.0},
            {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": 1.0},
            {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": dict(FM_Z)})
    S_val, p = resolve_S_and_params(m.config)
    r = spiral_opt.optimize_spiral(m, p, {"optimize_k": True, "num_starts": 3}, S_val=S_val)
    assert r.energy == pytest.approx(-5.0, abs=1e-6)


@pytest.mark.parametrize("lv", [
    [[1., 0, 0], [0, 10, 0], [0, 0, 10]],
    [[0, 1., 0], [10, 0, 0], [0, 0, 10]],
    [[2 ** -.5, 2 ** -.5, 0], [-7, 7, 0], [0, 0, 10]],
])
def test_scga_mesh_does_not_depend_on_lattice_orientation(lv):
    """The mesh axes were read off Cartesian bond components, so a chain along y or
    (1,1,0) got the wrong mesh and the wrong lambda."""
    m = GenericSpinModel({
        "crystal_structure": {"lattice_vectors": lv,
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 0.5}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": 1.0},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": dict(FM_Z)})
    kT = 0.3
    assert scga._wraps(scga._exchange_cache(m, [])).tolist() == [True, False, False]
    lam = scga.solve_lambda(m, [], kT, nq=200)[0]
    assert lam == pytest.approx(np.sqrt(4 + (3 * kT / 0.25) ** 2), rel=1e-5)


def test_perp_contraction_at_q0_agrees_between_modules():
    """'perp' at q = 0 is the powder limit (2/3) Tr in both SCGA and classical dynamics."""
    C = np.eye(3)
    for q in (np.zeros(3), np.array([1e-6, 0, 0])):
        assert scga._contract(C, q, "perp") == pytest.approx(2.0)
        assert classical_dynamics._contract(C[None], q, "perp")[0].real == pytest.approx(2.0)


def test_parallel_tempering_refuses_run_with_no_measured_sweeps():
    N = 4
    H = np.zeros((3 * N, 3 * N))
    for i in range(N):
        j = (i + 1) % N
        for a in range(3):
            H[3 * i + a, 3 * j + a] += 1.0
            H[3 * j + a, 3 * i + a] += 1.0
    with pytest.raises(ValueError, match="n_equil"):
        tmc.parallel_tempering(H, np.zeros(3 * N), N, 1.0, [0.5, 1.0], n_sweeps=1000, seed=0)


# ----------------------------------------------------------------- runner, resolution, mCIF

def test_q_path_with_undefined_label_raises():
    """A typo ('m' for 'M') used to truncate the path without a message."""
    cfg = {"q_path": {"Gamma": [0, 0, 0], "X": [0.5, 0, 0], "M": [0.5, 0.5, 0],
                      "points_per_segment": 10, "path": ["Gamma", "X", "m"]}}
    with pytest.raises(ValueError, match="undefined point"):
        generate_q_path_from_config(cfg)


def test_qe_resolution_fraction_is_finite_at_zero_width():
    """rho = 0 or sigma0 = 0 gave 0/0 = NaN. The limit is a step: 1 inside the bin,
    1/2 on an edge."""
    lo, hi = np.array([1.0]), np.array([2.0])
    tc = np.diag([1e-10, 1e-14])
    geom = dict(ei=10.0, l1=30.0, l2=4.0, time_cov=tc)
    g = GaussianEnergyResolution(fwhm0=0.2, poly=(1.0,), variable="E")
    qa = QEResolution(g, None, GeometrySpread(), **geom)
    assert qa.fraction(lo, hi, np.array([1.0]), 0.0, 0.0)[0] == pytest.approx(0.5)
    assert qa.fraction(lo, hi, np.array([1.5]), 0.0, 0.0)[0] == pytest.approx(1.0)
    g0 = GaussianEnergyResolution(fwhm0=0.0, poly=(1.0,), variable="E")
    q0 = QEResolution(g0, None, GeometrySpread(), **geom)
    assert q0.fraction(lo, hi, np.array([1.5]), 0.0, 1.0)[0] == pytest.approx(1.0)
    qi = QEResolution(g0, None, GeometrySpread(), intrinsic_sigma=0.1, **geom)
    f = qi.fraction(lo, hi, np.array([1.5]), 0.0, 1.0)[0]
    assert np.isfinite(f) and f == pytest.approx(1.0, abs=1e-5)


MCIF_ORTHOHEX = """data_t
_cell_length_a 1.0
_cell_length_b 1.7320508075688772
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_magn_operation.id
_space_group_symop_magn_operation.xyz
1 x,y,z,+1
2 -1/2x-3/2y,1/2x-1/2y,z,+1
3 -1/2x+3/2y,-1/2x-1/2y,z,+1
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
M1 0.3 0.0 0.0
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
M1 1.0 0.0 0.0
"""


@pytest.mark.parametrize("basis", ["unit_axes", "lattice_vectors"])
def test_mcif_moments_transform_under_axis_mixing_symop(tmp_path, basis):
    """A 3-fold written in an orthohexagonal cell (a = 1, b = sqrt 3) mixes axes of
    unequal length. With `unit_axes` the images came out with the wrong length and
    angle; they must be unit moments at 0, 120, 120 degrees."""
    path = tmp_path / "t.mcif"
    path.write_text(MCIF_ORTHOHEX)
    sites = read_mcif(str(path), moment_basis=basis)["sites"]
    m0 = np.asarray(sites[0]["moment"], float)
    angles = []
    for s in sites:
        m = np.asarray(s["moment"], float)
        assert np.linalg.norm(m) == pytest.approx(1.0, abs=1e-9)
        angles.append(np.degrees(np.arccos(np.clip(m @ m0, -1, 1))))
    np.testing.assert_allclose(sorted(angles), [0.0, 120.0, 120.0], atol=1e-6)
