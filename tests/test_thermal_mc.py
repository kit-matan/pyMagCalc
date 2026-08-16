"""Thermal Monte-Carlo with parallel tempering (magcalc/thermal_mc.py).

Pinned to EXACT classical results, never a self-generated number:

  * N non-interacting spins in a field: <m·B̂>/S = −L(βgμ_B|B|S) (Langevin), per T;
  * the classical Heisenberg dimer <E>(T) = −J S² L(βJS²) from the exact 1-D
    partition-function integral;
  * the dimer heat capacity C(T) = (JS²)²/T² L'(βJS²) — the fluctuation estimator
    Var(E)/(N kT²) must reproduce it;
  * parallel tempering reproduces independent single-temperature Metropolis.
"""
import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.thermal_mc import (build_supercell, parallel_tempering, MU_B, GAMMA)


def _L(x):
    return 1.0 / np.tanh(x) - 1.0 / x


def _Lp(x):
    return 1.0 / x**2 - 1.0 / np.sinh(x)**2


def test_noninteracting_spins_follow_langevin():
    """N free spins in a field: magnetization is the Langevin function, exactly."""
    S, Bz = 1.0, 12.0
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": []},
        "parameters": {"Hz": Bz}, "parameter_order": ["Hz"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    m = GenericSpinModel(cfg)
    H, b, N, S, _pos = build_supercell(m, [Bz], supercell=(4, 4, 1))
    assert abs(np.linalg.norm(b[:3]) - GAMMA * MU_B * Bz) < 1e-9
    bmag = np.linalg.norm(b[:3])
    temps = np.array([2.0, 5.0, 12.0])
    res = parallel_tempering(H, b, N, S, temps, n_sweeps=8000, n_equil=2500, seed=1)
    for i, T in enumerate(res.temperatures):
        x = bmag * S / T
        assert abs(res.mag_vector[i, 2] - (-_L(x))) < 0.02, f"kT={T}"


def _dimer_model(S=1.0, J=1.0):
    cfg = {"crystal_structure": {"lattice_vectors": [[10., 0, 0], [0, 10, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                         {"label": "B", "pos": [0.05, 0, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 0.5,
                                             "value": "J"}]},
        "parameters": {"J": J}, "parameter_order": ["J"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "antiferromagnetic",
                               "direction": [0, 0, 1], "propagation_vector": [0, 0, 0]}}
    return GenericSpinModel(cfg)


@pytest.mark.slow
def test_classical_dimer_energy_and_heat_capacity():
    """Isolated classical Heisenberg dimers vs the exact <E>(T) and C(T)."""
    S, J = 1.0, 1.0
    m = _dimer_model(S, J)
    H, b, N, S, _pos = build_supercell(m, [J], supercell=(3, 3, 1))
    temps = np.array([0.5, 0.9, 1.5, 3.0])
    res = parallel_tempering(H, b, N, S, temps, n_sweeps=12000, n_equil=4000, seed=3)
    for i, T in enumerate(res.temperatures):
        a = J * S**2 / T
        E_exact = -J * S**2 * _L(a) / 2.0                 # per spin (2 spins/dimer)
        C_exact = (J * S**2)**2 / T**2 * _Lp(a) / 2.0
        assert abs(res.energy[i] - E_exact) < 0.01, f"E kT={T}"
        assert abs(res.heat_capacity[i] - C_exact) < 0.03, f"C kT={T}"


@pytest.mark.slow
def test_parallel_tempering_matches_independent_metropolis():
    """PT (with swaps) and independent single-T Metropolis (no swaps) must agree on
    <E>(T) within statistics — swaps change sampling efficiency, not the distribution."""
    m = _dimer_model()
    H, b, N, S, _pos = build_supercell(m, [1.0], supercell=(3, 3, 1))
    temps = np.array([0.4, 0.8, 1.6, 3.2])
    pt = parallel_tempering(H, b, N, S, temps, n_sweeps=9000, n_equil=3000,
                            swap_every=1, seed=5)
    ind = parallel_tempering(H, b, N, S, temps, n_sweeps=9000, n_equil=3000,
                             swap_every=0, seed=6)
    assert np.max(np.abs(pt.energy - ind.energy)) < 0.01


# --- single-ion anisotropy in the sampler's Hamiltonian (OPEN_WORK item 11) -------
#
# `build_supercell` assembled H from the BOND table alone until 2026-08-15, so
# `single_ion_anisotropy` / `sia_matrix` / `stevens` never reached thermal_mc,
# wang_landau, static_correlations or the classical sampled_correlations: an
# anisotropic magnet was silently sampled as exchange-only. Every test above is
# bond-only, which is why the property had never been false in the suite.
#
# The oracle is the exact single-spin partition function, not a recorded number:
# one classical spin of length S in a field b ẑ with a uniaxial anisotropy D has
#     E(u) = D S² u² + b S u,   u = cos θ,
# so Z(β) = ∫₋₁¹ e^{-βE} du and every average is a 1-D quadrature -- the same
# shape of oracle the Langevin and dimer tests above use.

def _sia_model(D=2.0, S=1.0, Bz=0.0, axis=(0, 0, 1), rcs=False):
    """One isolated spin per cell, NO bonds, with a uniaxial anisotropy D(S·n)²."""
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"heisenberg": [],
                         "single_ion_anisotropy": [
                             {"type": "sia", "value": "D", "axis": list(axis),
                              "atoms": ["A"]}]},
        "parameters": {"D": D, "Hz": Bz}, "parameter_order": ["D", "Hz"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [1, 0, 0]}}
    if rcs:
        cfg["calculation"] = {"anisotropy_renormalization": "rcs"}
    return GenericSpinModel(cfg), [D, Bz]


def _exact_single_spin(D, S, bz, kT):
    """(<E>, <m_z>) for one classical spin: E(u) = D S² u² + bz S u, u = cos θ."""
    from scipy.integrate import quad
    def E(u):
        return D * S**2 * u**2 + bz * S * u
    w = lambda u: np.exp(-(E(u) - E(-1.0)) / kT)      # shifted: no overflow
    Z = quad(w, -1, 1, limit=200)[0]
    Eav = quad(lambda u: E(u) * w(u), -1, 1, limit=200)[0] / Z
    mz = S * quad(lambda u: u * w(u), -1, 1, limit=200)[0] / Z
    return Eav, mz


def test_build_supercell_carries_single_ion_anisotropy():
    """The structural pin: D(S·ẑ)² must appear in H as 2D on the zz entry.

    E = ½ mᵀH m, so the on-site block of a uniaxial D along ẑ is 2D·ẑẑᵀ. Before
    the fix this whole matrix was zero for a bond-free model -- H had exactly no
    nonzero entries -- which is the defect stated as a number.
    """
    D, S = 2.5, 1.0
    m, pr = _sia_model(D=D, S=S)
    H, b, N, _S, _pos = build_supercell(m, pr, supercell=(2, 2, 1))
    assert N == 4
    expected = np.zeros((3, 3))
    expected[2, 2] = 2.0 * D
    for a in range(N):
        blk = H[3 * a:3 * a + 3, 3 * a:3 * a + 3]
        assert np.allclose(blk, expected, atol=1e-12), blk
    assert np.allclose(b, 0.0)
    # and the energy of an explicit configuration is D Σ (m·ẑ)², exactly
    rng = np.random.default_rng(0)
    v = rng.standard_normal((N, 3))
    v *= S / np.linalg.norm(v, axis=1, keepdims=True)
    mv = v.ravel()
    assert abs(0.5 * mv @ (H @ mv) - D * np.sum(v[:, 2]**2)) < 1e-12


def test_anisotropy_axis_is_not_assumed_to_be_z():
    """An off-axis n gives the full 2D·nnᵀ block, not a z-only one."""
    D, n = 1.7, np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    m, pr = _sia_model(D=D, axis=tuple(n))
    H, _b, _N, _S, _pos = build_supercell(m, pr, supercell=(1, 1, 1))
    assert np.allclose(H[:3, :3], 2.0 * D * np.outer(n, n), atol=1e-12)


def test_sia_matrix_and_rank2_stevens_reach_the_sampler_too():
    """`sia_matrix` and a k = 2 Stevens term are quadratic, so both are exact.

    O₂⁰ = 2Sz² − Sx² − Sy² (the classical Sunny convention), so B·O₂⁰ must give the
    block 2B·diag(−1, −1, 2). Pinned against the polynomial, not a recorded number.
    """
    base = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
             "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0}]},
            "magnetic_structure": {"type": "pattern",
                                   "pattern_type": "ferromagnetic",
                                   "direction": [0, 0, 1]}}
    A = [[0.3, 0.1, 0.0], [0.1, -0.2, 0.0], [0.0, 0.0, 0.7]]
    cfg = dict(base, interactions={"heisenberg": [],
                                   "sia_matrix": [{"matrix": A, "atoms": ["A"]}]},
               parameters={}, parameter_order=[])
    H, _b, _N, _S, _pos = build_supercell(GenericSpinModel(cfg), [], supercell=(1, 1, 1))
    assert np.allclose(H[:3, :3], 2.0 * np.array(A), atol=1e-12)

    B20 = 0.45
    cfg = dict(base, interactions={"heisenberg": [],
                                   "stevens": [{"B": {"2,0": B20}, "atoms": ["A"]}]},
               parameters={}, parameter_order=[])
    H, _b, _N, _S, _pos = build_supercell(GenericSpinModel(cfg), [], supercell=(1, 1, 1))
    assert np.allclose(H[:3, :3], 2.0 * B20 * np.diag([-1.0, -1.0, 2.0]), atol=1e-12)


def test_rank4_stevens_refuses_rather_than_dropping_the_term():
    """A quartic on-site term cannot live in E = ½mᵀHm + bᵀm, so it must RAISE.

    Silently sampling the model without it is exactly the defect this item is
    about; `mode: SUN` is the route that carries the full operator.
    """
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 1]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 2.0}]},
        "interactions": {"heisenberg": [],
                         "stevens": [{"B": {"4,0": 0.01}, "atoms": ["A"]}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}
    with pytest.raises(NotImplementedError, match="NOT quadratic"):
        build_supercell(GenericSpinModel(cfg), [], supercell=(1, 1, 1))


def test_rcs_renormalization_reaches_the_sampler():
    """`anisotropy_renormalization: rcs` must scale the sampler's H too.

    λ₂(s) = 1 − 1/(2s) = ½ at s = 1, exactly — so the sampler and LSWT cannot
    disagree about which Hamiltonian is being simulated.
    """
    D = 3.0
    m_off, pr = _sia_model(D=D, S=1.0)
    m_on, _ = _sia_model(D=D, S=1.0, rcs=True)
    H_off, *_ = build_supercell(m_off, pr, supercell=(1, 1, 1))
    H_on, *_ = build_supercell(m_on, pr, supercell=(1, 1, 1))
    assert np.allclose(H_off[:3, :3], 2.0 * D * np.diag([0, 0, 1.0]), atol=1e-12)
    assert np.allclose(H_on[:3, :3], 0.5 * H_off[:3, :3], atol=1e-12)


@pytest.mark.slow
def test_single_spin_anisotropy_thermodynamics_vs_exact_partition_function():
    """<E>(T) and <m_z>(T) of a spin with an EASY-AXIS D and a field, exactly.

    Bond-free, so the sampler's answer is the single-spin quadrature above, and the
    pre-fix code (which dropped D) returns the free-spin Langevin result instead --
    at kT = 0.5 that is <E>/spin = -0.28 against the exact -1.72, i.e. this is not a
    tolerance question.
    """
    D, S, Bz = -1.5, 1.0, 6.0
    m, pr = _sia_model(D=D, S=S, Bz=Bz)
    H, b, N, S, _pos = build_supercell(m, pr, supercell=(4, 4, 1))
    bz = float(b[2])
    assert abs(bz - GAMMA * MU_B * Bz) < 1e-9
    temps = np.array([0.5, 1.0, 3.0])
    res = parallel_tempering(H, b, N, S, temps, n_sweeps=9000, n_equil=3000, seed=2)
    for i, T in enumerate(res.temperatures):
        E_exact, mz_exact = _exact_single_spin(D, S, bz, T)
        assert abs(res.energy[i] - E_exact) < 0.02, f"E kT={T}"
        assert abs(res.mag_vector[i, 2] * S - mz_exact) < 0.02, f"mz kT={T}"
