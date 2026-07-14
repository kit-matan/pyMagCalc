"""SU(N) (generalized) spin-wave theory.

The validation gates from SUN_PLAN.md, in order of how loudly they fail:

  GATE 1  S=1/2 (N=2): SU(N) is IDENTICAL to dipole LSWT. One boson per site, so it is
          literally the same theory -- any convention error (phase, Bogoliubov metric,
          factor of 2, on-site mean field) shows up here as a hard mismatch.
  GATE 2  With no single-ion terms, the DIPOLE band must be reproduced exactly for any
          S, with the extra SU(N) bands being the flat multipolar (Dm >= 2) modes.
  GATE 3  With single-ion anisotropy, match Sunny :SUN mode by mode -- including the
          quadrupolar band that dipole LSWT cannot represent at all.
"""
import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.sun import SUNModel, coherent_from_direction, spin_matrices, stevens_matrices

A = 4.0
LAT = [[A, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
QS = [np.array([h, 0, 0]) @ B for h in (0.1, 0.27, 0.45)]


# ----------------------------------------------------------------- operators
def test_spin_matrices_match_sunny():
    assert np.allclose(np.real(spin_matrices(0.5)[2]), [[0.5, 0], [0, -0.5]])
    assert np.allclose(np.real(spin_matrices(1.0)[0]),
                       [[0, 0.707107, 0], [0.707107, 0, 0.707107], [0, 0.707107, 0]],
                       atol=1e-5)
    # commutator algebra: [Sx, Sy] = i Sz
    for S in (0.5, 1.0, 1.5, 2.0):
        Sx, Sy, Sz = spin_matrices(S)
        assert np.allclose(Sx @ Sy - Sy @ Sx, 1j * Sz, atol=1e-10)


def test_stevens_matrices_match_sunny():
    """These are the N x N MATRICES, a different object from the classical polynomials
    in magcalc/stevens.py (which are the s -> inf limit used by dipole LSWT)."""
    assert np.allclose(np.real(stevens_matrices(1.0, 2, 0)),
                       [[1, 0, 0], [0, -2, 0], [0, 0, 1]], atol=1e-8)
    assert np.allclose(np.real(stevens_matrices(2.0, 4, 0)),
                       np.diag([12, -48, 72, -48, 12]), atol=1e-8)
    assert np.allclose(np.real(stevens_matrices(2.0, 4, 3)),
                       [[0, 0, 0, 3, 0], [0, 0, 0, 0, -3], [0, 0, 0, 0, 0],
                        [3, 0, 0, 0, 0], [0, -3, 0, 0, 0]], atol=1e-8)


def test_coherent_state_reproduces_the_classical_dipole():
    for S in (0.5, 1.0, 2.0):
        for d in ([0, 0, 1], [1, 0, 0], [1, 1, 1]):
            Z = coherent_from_direction(S, d)
            Sxyz = spin_matrices(S)
            exp = np.array([np.real(Z.conj() @ Sxyz[a] @ Z) for a in range(3)])
            want = S * np.asarray(d, float) / np.linalg.norm(d)
            assert np.allclose(exp, want, atol=1e-9)


# ----------------------------------------------------------------- helpers
def _dipole(S, J, directions, sia=None):
    n = len(directions)
    atoms = [{"label": f"S{i}", "pos": [i / n, 0.0, 0.0], "spin_S": S, "ion": "Fe2+"}
             for i in range(n)]
    inter = {"symmetry_rules": [{"type": "heisenberg", "distance": A / n, "value": J}]}
    if sia is not None:
        inter["single_ion_anisotropy"] = [
            {"type": "sia", "value": sia, "axis": [0, 0, 1]}]
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
        "interactions": inter,
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": directions},
    }
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base="sun_dip", hamiltonian_params=[])
    return np.sort(np.real(calc.calculate_dispersion(QS).energies), axis=1)


def _sun(S, J, directions, sia=None):
    n = len(directions)
    d = A / n
    bonds = []
    for i in range(n):
        j = (i + 1) % n
        bonds.append((i, j, np.array([d, 0, 0]), J * np.eye(3)))
        bonds.append((j, i, np.array([-d, 0, 0]), J * np.eye(3)))
    onsite = None
    if sia is not None:
        Sz = spin_matrices(S)[2]
        onsite = [(i, sia * (Sz @ Sz)) for i in range(n)]
    mdl = SUNModel.from_directions([S] * n, directions, bonds, onsite)
    return mdl, np.array([mdl.dispersion(q) for q in QS])


# ----------------------------------------------------------------- GATE 1
@pytest.mark.parametrize("J,dirs", [
    (-1.0, [[0, 0, 1]]),                       # FM chain
    (+1.0, [[0, 0, 1], [0, 0, -1]]),           # Neel chain
])
def test_gate1_spin_half_is_identical_to_dipole_lswt(J, dirs):
    """THE load-bearing test. At S=1/2 there is one boson per site, so SU(N) and dipole
    LSWT are the same theory and must agree to machine precision."""
    dip = _dipole(0.5, J, dirs)
    _, sun = _sun(0.5, J, dirs)
    assert np.allclose(np.sort(dip, axis=1), np.sort(sun, axis=1), atol=1e-10)


# ----------------------------------------------------------------- GATE 2
@pytest.mark.parametrize("S,J,dirs", [
    (1.0, -1.0, [[0, 0, 1]]),
    (1.0, +1.0, [[0, 0, 1], [0, 0, -1]]),
    (2.0, -1.0, [[0, 0, 1]]),
])
def test_gate2_dipole_band_is_reproduced_and_extras_are_multipolar(S, J, dirs):
    """Without single-ion terms the DIPOLE bands must appear exactly among the SU(N)
    bands. The extras are flat Dm >= 2 modes: the spin operators cannot connect states
    differing by more than one unit of m, so those modes do not disperse."""
    dip = _dipole(S, J, dirs)
    _, sun = _sun(S, J, dirs)
    for iq in range(len(QS)):
        for e in dip[iq]:
            assert np.min(np.abs(sun[iq] - e)) < 1e-8, f"dipole band {e} missing"
    # the extra bands are flat in q
    extra = sun.shape[1] - dip.shape[1]
    assert extra > 0
    flat = np.sort(sun, axis=1)[:, -extra:]
    assert np.allclose(flat, flat[0], atol=1e-8), "multipolar modes should be flat"


# ----------------------------------------------------------------- GATE 3
def test_gate3_single_ion_physics_matches_sunny_SUN():
    """S=1 FM chain with EASY-AXIS anisotropy. The Dm=2 (quadrupolar) level hybridises
    with the magnon -- the mechanism behind FeI2's single-ion bound state, and something
    dipole LSWT cannot represent at all.

    Reference: Sunny 0.8.1, :SUN mode.
    """
    sunny = {0.10: [0.981966, 4.0], 0.27: [2.850666, 4.0], 0.45: [4.0, 4.502113]}
    mdl, sun = _sun(1.0, -1.0, [[0, 0, 1]], sia=-0.6)

    assert abs(mdl.classical_energy() - (-1.6)) < 1e-9      # Sunny energy_per_site

    for iq, h in enumerate((0.10, 0.27, 0.45)):
        assert np.allclose(np.sort(sun[iq]), np.sort(sunny[h]), atol=1e-5), \
            f"q={h}: {sun[iq]} vs {sunny[h]}"


def test_gate3_dipole_mode_misses_the_single_ion_band():
    """The point of the whole exercise: dipole LSWT has only ONE band here and simply
    does not contain the quadrupolar excitation."""
    dip = _dipole(1.0, -1.0, [[0, 0, 1]], sia=-0.6)
    _, sun = _sun(1.0, -1.0, [[0, 0, 1]], sia=-0.6)
    assert dip.shape[1] == 1
    assert sun.shape[1] == 2
    # and the extra SU(N) band is nowhere near the dipole one
    assert np.min(np.abs(sun[0] - dip[0][0])) > 0.5


def test_sun_rejects_mixed_spin_for_now():
    with pytest.raises(NotImplementedError, match="same N"):
        SUNModel.from_directions([0.5, 1.0], [[0, 0, 1], [0, 0, 1]], [])


# ----------------------------------------------------------------- config bridge
def test_bridge_from_generic_model_matches_sunny():
    """SU(N) built from a pyMagCalc CONFIG -- reusing the whole existing front end
    (structure, symmetry propagation of the exchange matrices, supercells) and only
    swapping the LSWT engine. Same model as GATE 3, so it must land on the same Sunny
    numbers; if the bridge drops or mis-signs a bond or an on-site term, it diverges."""
    sunny = {0.10: [0.981966, 4.0], 0.27: [2.850666, 4.0], 0.45: [4.0, 4.502113]}
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": [
            {"label": "Fe", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "ion": "Fe2+"}]},
        "interactions": {
            "symmetry_rules": [{"type": "heisenberg", "distance": A, "value": -1.0}],
            "single_ion_anisotropy": [
                {"type": "sia", "value": -0.6, "axis": [0, 0, 1]}],
        },
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    mdl = SUNModel.from_generic_model(GenericSpinModel(cfg))
    assert abs(mdl.classical_energy() - (-1.6)) < 1e-9
    for h, want in sunny.items():
        got = mdl.dispersion(np.array([h, 0, 0]) @ B)
        assert np.allclose(np.sort(got), np.sort(want), atol=1e-5)


def test_bridge_carries_anisotropic_exchange_matrices():
    """FeI2-style: full 3x3 exchange must survive the bridge (it is what makes the
    single-ion bound state hybridise)."""
    Jm = [[0.3, 0.0, 0.0], [0.0, 0.1, -0.2], [0.0, -0.2, 0.5]]
    cfg = {
        "crystal_structure": {"lattice_vectors": LAT, "atoms_uc": [
            {"label": "Fe", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0, "ion": "Fe2+"}]},
        "interactions": {"interaction_matrix": [
            {"pair": ["Fe", "Fe"], "rij_offset": [1, 0, 0], "value": Jm},
            {"pair": ["Fe", "Fe"], "rij_offset": [-1, 0, 0], "value": Jm},
        ]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
    }
    mdl = SUNModel.from_generic_model(GenericSpinModel(cfg))
    assert len(mdl.bonds) == 2
    got = mdl.bonds[0][3]
    assert np.allclose(np.sort(np.abs(got).ravel()), np.sort(np.abs(np.array(Jm)).ravel()))


def test_energy_per_site_is_cell_independent():
    """REGRESSION: `classical_energy` is the TOTAL cell energy, not per site. That
    distinction is invisible on a one-site cell -- which every validation gate happened
    to use -- so it silently looked correct until a 4-site model came along. Use
    `energy_per_site` when comparing with Sunny."""
    Sz = spin_matrices(1.0)[2]
    aniso = -0.6 * (Sz @ Sz)

    b1 = [(0, 0, np.array([A, 0, 0]), -np.eye(3)),
          (0, 0, np.array([-A, 0, 0]), -np.eye(3))]
    m1 = SUNModel.from_directions([1.0], [[0, 0, 1]], b1, [(0, aniso)])

    b2 = []
    for i in (0, 1):
        j = 1 - i
        b2 += [(i, j, np.array([A / 2, 0, 0]), -np.eye(3)),
               (i, j, np.array([-A / 2, 0, 0]), -np.eye(3))]
    m2 = SUNModel.from_directions([1.0] * 2, [[0, 0, 1]] * 2, b2,
                                  [(i, aniso) for i in (0, 1)])

    assert abs(m1.energy_per_site() - (-1.6)) < 1e-9          # Sunny
    assert abs(m2.energy_per_site() - (-1.6)) < 1e-9          # same physics, 2-site cell
    assert abs(m2.classical_energy() - 2 * m1.classical_energy()) < 1e-9


def test_cpn_ground_state_search_finds_the_minimum():
    """The CP^(N-1) search (self-consistent local-field diagonalisation -- the SU(N)
    analogue of optmagsteep). It must NOT be seeded from the dipole ground state: with
    anisotropy present a coherent state has <Sz^2> != (S n_z)^2, so the SU(N) and dipole
    classical energies genuinely differ for a canted structure."""
    Sz = spin_matrices(1.0)[2]
    bonds = [(0, 0, np.array([A, 0, 0]), -np.eye(3)),
             (0, 0, np.array([-A, 0, 0]), -np.eye(3))]
    mdl = SUNModel.from_directions([1.0], [[1, 0, 0]], bonds,   # deliberately wrong start
                                   [(0, -0.6 * (Sz @ Sz))])
    E = mdl.minimize_energy(n_restarts=6, seed=0)
    assert abs(E / mdl.L - (-1.6)) < 1e-8
    assert abs(abs(mdl.dipoles[0][2]) - 1.0) < 1e-6      # easy axis -> spins along z
