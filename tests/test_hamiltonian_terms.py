"""Hamiltonian terms added to close the 'Gap 2' feature set.

Every check here is an *identity* against an independent reference -- an exact
analytic reduction, an equivalent formulation of the same physics, or Sunny --
rather than a golden number, so a regression cannot be papered over by updating
an expected value.

Covered:
  * biquadratic exchange           -> exact collinear reduction J_eff = J + 2*B*sigma*S^2
  * anisotropic / per-site g-tensor -> reduces to the legacy Zeeman at g = 2
  * full 3x3 single-ion anisotropy  -> reduces to the uniaxial `sia`
  * Stevens operators O_k^q         -> O_2^0 equals its 3x3 equivalent; O_4^0 survives
                                       the LSWT truncation (it used to be deleted)
  * dipole-dipole                   -> matches Sunny's truncated dipolar sum
  * multi-k                         -> one-component k=1/2 reproduces the Neel chain
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel, resolve_supercell_dims

S_VAL = 1.5
QS = [[0.13, 0.0, 0.0], [0.31, 0.0, 0.0], [0.5, 0.0, 0.0]]
NN_BONDS = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
            (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]


def bands(cfg, qs=QS, S=S_VAL, tag="t"):
    """Dispersion at `qs` (RLU).

    NB: the magnetic structure must be applied to the model BEFORE MagCalc is
    built -- gen_HM runs in MagCalc.__init__ and reads the rotations via mpr(),
    which is the identity until set_magnetic_structure() has run. Building it
    the other way round silently does LSWT about the unrotated state.
    """
    m = GenericSpinModel(copy.deepcopy(cfg))
    order = cfg.get("parameter_order") or []
    params = cfg.get("parameters") or {}
    pv = []
    for k in order:
        v = params[k]
        pv.extend(v) if isinstance(v, (list, tuple)) else pv.append(v)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=pv)
    A = np.array(m.config["crystal_structure"]["lattice_vectors"], float)
    B = 2 * np.pi * np.linalg.inv(A).T
    e = np.real(calc.calculate_dispersion([np.array(q) @ B for q in qs]).energies)
    return np.sort(e, axis=1)


def neel_chain(extra=None, J=1.0, atoms_extra=None):
    """2-site Neel chain, a = 6 A, NN at 3 A. `extra` merges into interactions."""
    atoms = [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"},
             {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"}]
    if atoms_extra:
        for at, ex in zip(atoms, atoms_extra):
            at.update(ex)
    inter = {"symmetry_rules": [{"type": "heisenberg", "distance": 3.0, "value": J}]}
    if extra:
        inter.update(copy.deepcopy(extra))
    return {
        "crystal_structure": {
            "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": atoms,
        },
        "interactions": inter,
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]},
    }


FIELD = {"parameters": {"H_mag": 3.0, "H_dir": [0.0, 0.0, 1.0]},
         "parameter_order": ["H_mag", "H_dir"]}


def with_field(cfg):
    cfg = copy.deepcopy(cfg)
    cfg.update(copy.deepcopy(FIELD))
    return cfg


# --------------------------------------------------------------------------
# Biquadratic
# --------------------------------------------------------------------------
@pytest.mark.slow
def test_biquadratic_reduces_to_exact_collinear_bilinear():
    """On a collinear structure (S_i.S_j has no boson-linear part), the quadratic
    part of B (S_i.S_j)^2 is exactly 2*B*D0 times that of (S_i.S_j), with
    D0 = sigma*S^2. So it must coincide with a bilinear J_eff = J + 2*B*sigma*S^2.
    """
    B, J = -0.037, 1.0
    sigma = -1.0  # every NN bond of the Neel chain is antiparallel
    biq = {"biquadratic": [{"pair": p, "rij_offset": o, "value": B}
                           for p, o in NN_BONDS]}
    native = bands(neel_chain(extra=biq, J=J), tag="biq_n")
    mapped = bands(neel_chain(J=J + 2.0 * B * sigma * S_VAL**2), tag="biq_m")
    assert np.allclose(native, mapped, atol=1e-9)

    # ... and the term must actually do something (guards against silent deletion)
    plain = bands(neel_chain(J=J), tag="biq_p")
    assert np.max(np.abs(native - plain)) > 1e-3


def test_biquadratic_empty_match_raises():
    # The on-site/bond terms are evaluated when the symbolic Hamiltonian is built
    # (MagCalc.__init__ -> gen_HM), which runs in the main process, so the error
    # surfaces to the caller rather than being swallowed into a NaN spectrum.
    biq = {"biquadratic": [{"pair": ["A", "B"], "distance": 99.0, "value": -0.01}]}
    with pytest.raises(ValueError, match="matched no bonds"):
        bands(neel_chain(extra=biq), tag="biq_bad")


# --------------------------------------------------------------------------
# g-tensor
# --------------------------------------------------------------------------
def test_gtensor_g2_reduces_to_legacy_zeeman():
    """The legacy global Zeeman is calibrated (SW29) so H_mag = B[T] gives the
    electron g = 2 result; an explicit isotropic g = 2 must therefore be a no-op."""
    legacy = bands(with_field(neel_chain()), tag="g_leg")
    g2 = bands(with_field(neel_chain(atoms_extra=[{"g": 2.0}, {"g": 2.0}])), tag="g_2")
    assert np.allclose(legacy, g2, atol=1e-10)


@pytest.mark.slow
def test_gtensor_axial_selects_g_par_along_field():
    """With the local axis along z and B || z, only g_par couples, so an axial
    tensor (g_par, g_perp) must equal an isotropic g = g_par."""
    axial = [{"g": {"g_par": 1.8, "g_perp": 4.32, "axis": [0, 0, 1]}}] * 2
    e_axial = bands(with_field(neel_chain(atoms_extra=axial)), tag="g_ax")
    e_iso = bands(with_field(neel_chain(atoms_extra=[{"g": 1.8}, {"g": 1.8}])),
                  tag="g_18")
    assert np.allclose(e_axial, e_iso, atol=1e-10)
    # and it is genuinely different from g = 2
    e_g2 = bands(with_field(neel_chain(atoms_extra=[{"g": 2.0}] * 2)), tag="g_2b")
    assert np.max(np.abs(e_axial - e_g2)) > 1e-3


def test_gtensor_diagonal_and_matrix_forms_agree():
    diag = bands(with_field(neel_chain(atoms_extra=[{"g": [4.32, 4.32, 1.8]}] * 2)),
                 tag="g_d")
    full = bands(with_field(neel_chain(atoms_extra=[
        {"g": [[4.32, 0, 0], [0, 4.32, 0], [0, 0, 1.8]]}] * 2)), tag="g_f")
    assert np.allclose(diag, full, atol=1e-12)


def _classical_energy(atoms_extra=None, tilt=0.3):
    """Classical energy of a fixed CANTED configuration (plain floats -> no boson
    ops -> no LSWT truncation). This is the energy the minimizer optimizes."""
    import sympy as sp
    cfg = with_field(neel_chain(atoms_extra=atoms_extra))
    m = GenericSpinModel(cfg)
    n, n_ouc = len(m.atom_pos()), len(m.atom_pos_ouc())
    Sxyz = []
    for j in range(n_ouc):
        # Canted, NOT Neel: a compensated state has zero net moment and hence zero
        # Zeeman energy for any g, which would make this check vacuous.
        th = tilt if (j % n) == 0 else np.pi - 3.0 * tilt
        Sxyz.append([S_VAL * np.sin(th), 0.0, S_VAL * np.cos(th)])
    return float(sp.N(m.Hamiltonian(Sxyz, [3.0, 0.0, 0.0, 1.0])))


def test_gtensor_enters_the_classical_energy_too():
    """REGRESSION: the minimizer must optimize the SAME Hamiltonian that LSWT
    diagonalizes. The g-tensor used to be applied only on the symbolic-parameter
    path, so with numeric params (the classical/minimization path) the field fell
    back to an isotropic term pointing along z -- giving a wrong ground state and
    imaginary magnon energies (seen on SW20 in field)."""
    e_none = _classical_energy(None)
    e_g2 = _classical_energy([{"g": 2.0}] * 2)
    e_g18 = _classical_energy([{"g": 1.8}] * 2)
    assert abs(e_g2 - e_none) < 1e-10          # g = 2 reduces to the legacy term
    assert abs(e_g18 - e_none) > 1e-6          # ... and g is genuinely felt


def test_scalar_field_uses_configured_direction_not_z():
    """REGRESSION: with a scalar H_mag the classical path used to assume B || z,
    ignoring H_dir. Fields along x and z must give different classical energies."""
    import sympy as sp

    def energy_for(hdir):
        cfg = neel_chain()
        cfg["parameters"] = {"H_mag": 3.0, "H_dir": hdir}
        cfg["parameter_order"] = ["H_mag", "H_dir"]
        m = GenericSpinModel(cfg)
        n, n_ouc = len(m.atom_pos()), len(m.atom_pos_ouc())
        Sxyz = []
        for j in range(n_ouc):
            th = 0.3 if (j % n) == 0 else np.pi - 0.9
            Sxyz.append([S_VAL * np.sin(th), 0.0, S_VAL * np.cos(th)])
        return float(sp.N(m.Hamiltonian(Sxyz, [3.0] + list(hdir))))

    assert abs(energy_for([0.0, 0.0, 1.0]) - energy_for([1.0, 0.0, 0.0])) > 1e-6


def test_gtensor_bad_spec_raises():
    with pytest.raises(Exception):
        bands(with_field(neel_chain(atoms_extra=[{"g": {"g_par": 1.8}}] * 2)),
              tag="g_bad")   # axial spec missing g_perp


# --------------------------------------------------------------------------
# Single-ion anisotropy: 3x3 tensor and Stevens operators
# --------------------------------------------------------------------------
def test_sia_matrix_matches_uniaxial():
    D = -0.25
    uni = bands(neel_chain(extra={"single_ion_anisotropy": [
        {"type": "sia", "value": D, "axis": [0, 0, 1]}]}), tag="sia_u")
    mat = bands(neel_chain(extra={"sia_matrix": [
        {"matrix": [[0, 0, 0], [0, 0, 0], [0, 0, D]]}]}), tag="sia_m")
    assert np.allclose(uni, mat, atol=1e-10)


def test_stevens_O20_matches_equivalent_tensor():
    """O_2^0 = 2Sz^2 - Sx^2 - Sy^2 (Sunny's classical convention), i.e. exactly the
    3x3 anisotropy diag(-1, -1, 2)."""
    B20 = 0.11
    stev = bands(neel_chain(extra={"stevens": [{"B": {"2,0": B20}}]}), tag="st")
    equiv = bands(neel_chain(extra={"sia_matrix": [
        {"matrix": [[-B20, 0, 0], [0, -B20, 0], [0, 0, 2 * B20]]}]}), tag="st_e")
    assert np.allclose(stev, equiv, atol=1e-10)


def test_stevens_quartic_survives_lswt_truncation():
    """REGRESSION: the quadratic-boson part of a quartic operator carries S^3. The
    old `coeff(S,1)*S + coeff(S,2)*S^2` filter deleted it outright, so O_4^0 was
    silently a no-op. It must now shift the spectrum."""
    base = bands(neel_chain(), tag="o4_0")
    quartic = bands(neel_chain(extra={"stevens": [{"B": {"4,0": 0.02}}]}), tag="o4")
    assert np.max(np.abs(quartic - base)) > 1e-6


def test_stevens_rejects_unsupported_order():
    # k must be even (time reversal) and <= 6 -- O_3^0 does not exist.
    with pytest.raises(ValueError, match="not available"):
        bands(neel_chain(extra={"stevens": [{"B": {"3,0": 0.1}}]}), tag="st_bad")


# --------------------------------------------------------------------------
# Dipole-dipole -- validated against Sunny 0.8.1
# --------------------------------------------------------------------------
def test_dipole_dipole_matches_sunny():
    """FM chain (J = -1) + truncated dipolar sum, cutoff 20 A, s = 1, g = 2.

    Reference from Sunny 0.8.1 (:dipole_uncorrected):
        sys = System(cryst, [1 => Moment(s=1, g=2)], :dipole_uncorrected)
        set_exchange!(sys, -1.0, Bond(1, 1, [1, 0, 0]))
        modify_exchange_with_truncated_dipole_dipole!(sys, 20.0)
    """
    sunny = {0.1: 0.3801768, 0.25: 1.99924954, 0.4: 3.61760366, 0.5: 3.99959145}
    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0,
                          "ion": "Fe2+", "g": 2.0}],
        },
        "interactions": {
            "symmetry_rules": [{"type": "heisenberg", "distance": 6.0, "value": -1.0}],
            "dipole_dipole": {"cutoff": 20.0},
        },
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    qs = sorted(sunny)
    e = bands(cfg, qs=[[q, 0, 0] for q in qs], S=1.0, tag="dip")
    got = np.max(e, axis=1)
    want = np.array([sunny[q] for q in qs])
    assert np.allclose(got, want, atol=1e-5), f"got {got}, Sunny {want}"


def test_dipole_dipole_cutoff_too_small_raises():
    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}],
        },
        "interactions": {"dipole_dipole": {"cutoff": 1.0}},
        "parameters": {}, "parameter_order": [],
    }
    with pytest.raises(ValueError, match="matched no bonds"):
        GenericSpinModel(cfg)


# --------------------------------------------------------------------------
# Multi-k
# --------------------------------------------------------------------------
@pytest.mark.parametrize("ks,want", [
    ([[0.5, 0, 0]], [2, 1, 1]),
    ([[0.5, 0, 0], [0, 0.5, 0]], [2, 2, 1]),
    ([[1 / 3, 0, 0], [0.5, 0, 0]], [6, 1, 1]),   # per-axis LCM of 3 and 2
])
def test_multik_supercell_is_per_axis_lcm(ks, want):
    assert resolve_supercell_dims("auto", k_rlu=ks) == want


def test_multik_single_component_reproduces_neel_chain():
    """A one-component multi_k with k = (1/2,0,0) and m || z on the 1-site chemical
    chain IS the Neel chain; it must give the hand-built 2-site cell's spectrum."""
    mk = {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": S_VAL,
                          "ion": "Fe2+"}],
            "magnetic_supercell": "auto",
        },
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "multi_k", "components": [
            {"k": [0.5, 0.0, 0.0], "m": [0.0, 0.0, 1.0]}]},
    }
    e_mk = bands(mk, tag="mk")
    e_neel = bands(neel_chain(), tag="mk_ref")
    assert np.allclose(np.sort(e_mk.ravel()), np.sort(e_neel.ravel()), atol=1e-8)


def test_multik_two_k_gives_noncollinear_unit_moments():
    mk = {
        "crystal_structure": {
            "lattice_vectors": [[3.0, 0, 0], [0, 3.0, 0], [0, 0, 9.0]],
            "atoms_uc": [{"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": S_VAL,
                          "ion": "Fe2+"}],
            "magnetic_supercell": "auto",
        },
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "multi_k", "components": [
            {"k": [0.5, 0.0, 0.0], "m": [0.0, 0.0, 1.0]},
            {"k": [0.0, 0.5, 0.0], "m": [1.0, 0.0, 0.0]}]},
    }
    m = GenericSpinModel(copy.deepcopy(mk))
    assert m.supercell_dims == [2, 2, 1]
    th, ph = m.generate_magnetic_structure()
    dirs = np.array([[np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]
                     for t, p in zip(th, ph)])
    assert np.allclose(np.linalg.norm(dirs, axis=1), 1.0)      # normalize=True
    assert len({tuple(np.round(d, 6)) for d in dirs}) == 4     # non-collinear
