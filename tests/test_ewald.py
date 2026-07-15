"""Ewald summation of the long-range dipole-dipole interaction.

The dipolar sum is only CONDITIONALLY convergent, so a truncated real-space sum depends
on the cutoff and on the sample shape. Ewald is exact.

Validated two independent ways:
  1. against Sunny 0.8.1's `enable_dipole_dipole!` (true Ewald);
  2. against pyMagCalc's OWN truncated sum, which must converge toward the Ewald result
     as the cutoff grows -- a check that needs no Julia and would catch a wrong
     prefactor or a wrong tensor form even if the Sunny reference were misread.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.ewald import MU0_MUB2_MEV_A3, dipole_ewald_at_q
from magcalc.generic_model import GenericSpinModel

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]

# Sunny 0.8.1: FM chain (J = -1), s = 1, g = 2, enable_dipole_dipole!()
SUNNY = {0.1: 0.38080116, 0.25: 1.9992156, 0.4: 3.61761512, 0.5: 3.99966539}


def _calc(method, cutoff=None):
    dd = {"method": method}
    if cutoff:
        dd["cutoff"] = cutoff
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "ion": "Fe2+",
             "g": 2.0}]},
        "interactions": {
            "symmetry_rules": [{"type": "heisenberg", "distance": 6.0, "value": -1.0}],
            "dipole_dipole": dd,
        },
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base=f"ew_{method}", hamiltonian_params=[])


def _disp(calc, hs):
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    e = calc.calculate_dispersion([np.array([h, 0, 0]) @ B for h in hs]).energies
    return np.max(np.real(e), axis=1)


def test_ewald_matches_sunny():
    hs = sorted(SUNNY)
    got = _disp(_calc("ewald"), hs)
    want = np.array([SUNNY[h] for h in hs])
    assert np.allclose(got, want, atol=1e-5), f"got {got}, Sunny {want}"


def test_truncated_sum_converges_to_ewald():
    """Independent of Sunny: the truncated sum must approach Ewald as the cutoff grows.
    (They differ by the surface/demagnetisation term, which is what makes a truncated
    dipolar sum shape-dependent in the first place.)"""
    hs = sorted(SUNNY)
    ew = _disp(_calc("ewald"), hs)
    errs = [np.max(np.abs(_disp(_calc("truncated", cutoff=c), hs) - ew))
            for c in (12.0, 30.0, 45.0)]
    assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
    assert errs[-1] < 1e-4


SUNNY_E_PER_SITE = -0.9992729042524618      # Sunny energy_per_site, same model


def test_ewald_classical_energy_matches_sunny():
    """The minimiser must optimise the SAME Hamiltonian LSWT diagonalises -- if the
    dipolar term were missing from the classical energy, the ground state would not be a
    minimum of the real model (that is the g-tensor bug, in a new costume).

    This checks it directly against Sunny's energy_per_site for the z-polarised FM."""
    calc = _calc("ewald")
    e_now, _ = calc.relax_from_current()
    assert abs(e_now - SUNNY_E_PER_SITE) < 1e-9, e_now


def test_dipolar_shape_anisotropy_reorients_the_spins_along_the_chain():
    """A genuinely physical consequence, and a strong sign the tensor form is right:
    dipolar coupling on a CHAIN favours spins pointing ALONG the chain. So the
    z-polarised ferromagnet (which Sunny also uses, and about which LSWT is perfectly
    valid -- it is a local minimum, no imaginary modes) is NOT the global minimum, and
    relaxing lowers the energy. An isotropic-by-mistake dipolar term could not do this."""
    calc = _calc("ewald")
    e_now, e_relaxed = calc.relax_from_current()
    assert e_relaxed < e_now - 1e-4               # relaxing finds the lower state
    assert calc.max_imaginary_energy() < 1e-6     # ... yet z-FM is still a local min

    # the relaxed state points along the chain (x), not along z
    res = calc.minimize_energy(method="anneal", num_starts=4, n_sweeps=500, seed=0)
    th, ph = res.x[0::2], res.x[1::2]
    d = np.array([np.sin(th[0]) * np.cos(ph[0]), np.sin(th[0]) * np.sin(ph[0]),
                  np.cos(th[0])])
    assert abs(d[0]) > 0.99, f"expected spins along the chain, got {d}"


def test_A_is_hermitian_in_the_right_sense():
    """A_ij(q) must satisfy A_ji(-q) = A_ij(q)^dagger -- a basic consistency check on the
    lattice sum and the phase convention."""
    lat = np.array(LAT, float)
    pos = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    q = np.array([0.13, 0.0, 0.0])
    Ap = dipole_ewald_at_q(lat, pos, q)
    Am = dipole_ewald_at_q(lat, pos, -q)
    for i in range(2):
        for j in range(2):
            assert np.allclose(Am[j, i], Ap[i, j].conj().T, atol=1e-8)


def test_prefactor_matches_sunny_units():
    # Sunny: Units(:meV, :angstrom).vacuum_permeability = mu0 * muB^2
    assert abs(MU0_MUB2_MEV_A3 - 0.6745817653) < 1e-9


def test_ewald_rejects_unknown_method():
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}]},
        "interactions": {"dipole_dipole": {"method": "magic"}},
        "parameters": {}, "parameter_order": [],
    }
    with pytest.raises(ValueError, match="truncated"):
        GenericSpinModel(cfg)


def test_ewald_with_single_k_is_rejected_not_silently_wrong():
    """The three q +/- k channels each need their own A(q); rather than quietly use the
    wrong one, refuse."""
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "g": 2.0}]},
        "interactions": {
            "symmetry_rules": [{"type": "heisenberg", "distance": 6.0, "value": 1.0}],
            "dipole_dipole": {"method": "ewald"},
        },
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "single_k", "k": [0.23, 0.0, 0.0],
                               "axis": [0.0, 0.0, 1.0]},
    }
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="ew_sk", hamiltonian_params=[])
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    with pytest.raises(NotImplementedError, match="single-k"):
        calc.calculate_sqw([np.array([0.2, 0, 0]) @ B], satellites=True)


def test_ewald_reaches_the_sqw_path_too():
    """Regression: the dipolar term is injected into the S(Q,w) workers as well as the
    dispersion ones. (An earlier edit silently no-op'd on the S(Q,w) worker, so its
    energies would have quietly been the NON-dipolar ones.) S(Q,w) energies must equal
    the Ewald dispersion energies, and must differ from the no-dipole model."""
    hs = [0.1, 0.3]
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    qs = [np.array([h, 0, 0]) @ B for h in hs]

    ew = _calc("ewald")
    e_disp = np.max(np.real(ew.calculate_dispersion(qs).energies), axis=1)
    e_sqw = np.max(np.real(ew.calculate_sqw(qs).energies), axis=1)
    assert np.allclose(e_disp, e_sqw, atol=1e-8)

    # and the dipolar term genuinely changes S(Q,w) energies
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "ion": "Fe2+",
             "g": 2.0}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 6.0, "value": -1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    plain = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                       cache_file_base="ew_none", hamiltonian_params=[])
    e_plain = np.max(np.real(plain.calculate_sqw(qs).energies), axis=1)
    assert np.max(np.abs(e_sqw - e_plain)) > 1e-4
