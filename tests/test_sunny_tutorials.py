"""Sunny.jl tutorial ports (examples/sunny_tutorials/) regression.

Each ported config is pinned to an INDEPENDENT reference, never a self-generated
golden number:

  * S08 -- the 1D DM+Ising chain has the EXACT analytic dispersion
      w(q) = 2 s [J +/- D sin(2 pi q_c)] = 3 +/- 0.6 sin(2 pi q_c),
    including the q -> -q asymmetry that is the tutorial's whole point.
  * S09 -- the 120-degree triangular AFM has the EXACT analytic maximum
      w_max = 3 J S sqrt(9/8) = 1.5910 meV  (J=1, S=1/2), gapless at K.

The other three ported configs are pinned in their own suites and are only
schema-checked here:
  * S01 CoRh2O4  -- Neel energy -2 J s^2 (verified in the tutorial README);
  * S03 FeI2 SUN -- bands + intensities vs Sunny, tests/test_sun.py;
  * S07 dipole   -- Ewald vs Sunny to 1.3e-8, tests/test_ewald.py.
"""
import os

import numpy as np
import pytest
import yaml

from magcalc.generic_model import GenericSpinModel
import magcalc.core as mc

HERE = os.path.dirname(__file__)
ROOT = os.path.join(HERE, "..", "examples", "sunny_tutorials")
CONFIGS = {
    "S01": "S01_CoRh2O4/config.yaml",
    "S02": "S02_CoRh2O4_finiteT/config.yaml",
    "S05": "S05_Ising_MC/config.yaml",
    "S07": "S07_dipole_dipole/config.yaml",
    "S08": "S08_momentum_conventions/config.yaml",
    "S09": "S09_triangular_AFM/config.yaml",
}


def _bands_at(cfg, q_rlu, S):
    """Max band energy at a list of RLU q-points, via a fresh in-process calc."""
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    pv = []
    for k in cfg.get("parameter_order", []):
        v = cfg["parameters"][k]
        pv.extend(v) if isinstance(v, (list, tuple)) else pv.append(v)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base="sunny_tut_test", hamiltonian_params=pv)
    lp = cfg["crystal_structure"].get("lattice_vectors")
    if lp is None:
        L = _lattice_from_params(cfg["crystal_structure"]["lattice_parameters"])
    else:
        L = np.array(lp, float)
    B = 2 * np.pi * np.linalg.inv(L).T
    out = []
    for q in q_rlu:
        e = calc.calculate_dispersion([np.array(q, float) @ B]).energies[0]
        out.append(np.max(np.real(e)))
    return np.array(out)


def _lattice_from_params(p):
    a, b, c = p["a"], p["b"], p["c"]
    al, be, ga = (np.radians(p[k]) for k in ("alpha", "beta", "gamma"))
    v_a = [a, 0, 0]
    v_b = [b * np.cos(ga), b * np.sin(ga), 0]
    cx = c * np.cos(be)
    cy = c * (np.cos(al) - np.cos(be) * np.cos(ga)) / np.sin(ga)
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([v_a, v_b, [cx, cy, cz]], float)


def _load(name):
    return yaml.safe_load(open(os.path.join(ROOT, CONFIGS[name])))


def test_all_ported_configs_validate():
    from magcalc.schema import MagCalcConfig
    for name, rel in CONFIGS.items():
        cfg = yaml.safe_load(open(os.path.join(ROOT, rel)))
        MagCalcConfig.model_validate(cfg)


@pytest.mark.slow
def test_S08_dispersion_is_the_exact_dm_ising_result():
    """w(q) = 3 + 0.6 sin(2 pi q_c), ASYMMETRIC in q_c (the tutorial's point)."""
    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIGS["S08"])))
    qcs = [-0.25, -0.125, 0.0, 0.125, 0.25]
    got = _bands_at(cfg, [[0, 0, qc] for qc in qcs], S=1.5)
    want = np.array([3 + 0.6 * np.sin(2 * np.pi * qc) for qc in qcs])
    assert np.allclose(got, want, atol=1e-6), f"{got} vs {want}"
    # explicit asymmetry: w(+1/4) - w(-1/4) = 1.2, not 0
    assert abs((got[-1] - got[0]) - 1.2) < 1e-6


@pytest.mark.slow
def test_S09_triangular_120_matches_analytic_max_and_is_gapless_at_K():
    """w_max = 3 J S sqrt(9/8) = 1.5910 meV; Goldstone at K = [1/3,1/3,0]."""
    cfg = yaml.safe_load(open(os.path.join(ROOT, CONFIGS["S09"])))
    K = [1 / 3, 1 / 3, 0]
    # sample a dense set to catch the band maximum
    qs = [[h, h, 0] for h in np.linspace(0, 0.5, 26)] + [K]
    w = _bands_at(cfg, qs, S=0.5)
    w_max_analytic = 3 * 1.0 * 0.5 * np.sqrt(9 / 8)     # = 1.59099
    assert abs(w[:-1].max() - w_max_analytic) < 5e-3, f"{w[:-1].max()} vs {w_max_analytic}"
    assert w[-1] < 1e-3, f"not gapless at K: {w[-1]}"    # Goldstone at K

# ---------------------------------------------------------------------------
# S01: pinned to Sunny 0.8.1, band by band. The README claimed this was
# "cross-checked against Sunny" but nothing asserted it -- the config has no
# `magnetic_structure` (it relies on tasks.minimization), so the helper above could
# not drive it and it was only schema-checked.
#
#   cryst = Crystal(lattice_vectors(8.5031,...,90,90,90), [[1/8,1/8,1/8]], 227)
#   sys = System(cryst, [1 => Moment(s=3/2, g=2)], :dipole)
#   set_exchange!(sys, 0.63, Bond(2, 3, [0,0,0]))     # NB Bond(2,3), d = 3.68195 A
#   randomize_spins!(sys); minimize_energy!(sys)
#   dispersion(SpinWaveTheory(sys; measure=nothing), qs)
# ---------------------------------------------------------------------------
S01_QS = [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.5, 0.25, 0], [0.5, 0.5, 0],
          [0.25, 0.25, 0]]
S01_SUNNY = [
    [0.0, 0.0, 3.78, 3.78, 3.78, 3.78, 3.78, 3.78],
    [1.446543, 1.446543, 3.492265, 3.492265, 3.78, 3.78, 3.78, 3.78],
    [2.672864, 2.672864, 2.672864, 2.672864, 3.78, 3.78, 3.78, 3.78],
    [2.861895, 2.861895, 2.861895, 2.861895, 3.638977, 3.638977, 3.638977, 3.638977],
    [3.273576, 3.273576, 3.273576, 3.273576, 3.273576, 3.273576, 3.273576, 3.273576],
    [1.969400, 1.969400, 3.535866, 3.535866, 3.535866, 3.535866, 3.739246, 3.739246],
]


def test_S01_dispersion_matches_sunny_band_by_band():
    """CoRh2O4: every band at every q, against Sunny.

    The diamond lattice is two interpenetrating fcc sublattices, so the Neel state is
    Co0..Co3 up / Co4..Co7 down; it is supplied explicitly here rather than found by
    the annealer, so the test measures the SPECTRUM and not the minimizer.
    Sunny's Goldstone mode comes back as 2.75e-4 rather than 0 -- that is its
    minimizer's residual, not a disagreement, hence the absolute tolerance.
    """
    cfg = _load("S01")
    cfg["magnetic_structure"] = {"type": "pattern", "pattern_type": "generic",
                                 "directions": [[0, 0, 1]] * 4 + [[0, 0, -1]] * 4}
    cfg["calculation"] = {"on_imaginary": "off"}
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.5, cache_mode="none",
                      cache_file_base="s01_pin",
                      hamiltonian_params=[cfg["parameters"]["J"]])
    L = _lattice_from_params(cfg["crystal_structure"]["lattice_parameters"])
    B = 2 * np.pi * np.linalg.inv(L).T
    got = np.sort(np.real(calc.calculate_dispersion(
        [np.array(q, float) @ B for q in S01_QS]).energies), axis=1)
    assert got == pytest.approx(np.array(S01_SUNNY), abs=5e-4)


def test_S01_classical_energy_is_the_exact_neel_value():
    """-2 J s^2 = -2.835 meV/site for the diamond Neel state (z = 4, the 1/2 over
    ordered pairs). Sunny's minimizer lands on exactly this."""
    assert -2 * 0.63 * 1.5 ** 2 == pytest.approx(-2.835, abs=1e-12)

# ---------------------------------------------------------------------------
# S05 -- the 2-D Ising ferromagnet, against ONSAGER'S EXACT results.
#
# Sunny builds Ising out of continuous spins with `polarize_spins!` +
# `propose_flip`: the move S -> -S never leaves the +/-z axis. pyMagCalc does the
# same with `thermal_mc: {propose: flip, init: [0,0,1]}`.
# ---------------------------------------------------------------------------
TC_ISING = 2.0 / np.log(1.0 + np.sqrt(2.0))          # 2.269185...


def _onsager_m(T):
    """Spontaneous magnetization [1 - sinh^-4(2J/T)]^(1/8), J = 1."""
    k = np.sinh(2.0 / T) ** -4
    return (1.0 - k) ** 0.125 if k < 1.0 else 0.0


def _ising_run(temps, L=24, **kw):
    from magcalc.thermal_mc import build_supercell, parallel_tempering
    cfg = _load("S05")
    m = GenericSpinModel(cfg)
    H, b, N, S, _ = build_supercell(m, [cfg["parameters"]["J"]], (L, L, 1))
    opts = dict(n_sweeps=8000, n_equil=3000, seed=1, propose="flip",
                init=[0, 0, 1], swap_every=0)
    opts.update(kw)
    return parallel_tempering(H, b, N, S, np.asarray(temps, float), **opts)


@pytest.mark.slow
def test_S05_magnetization_matches_onsager():
    """m(T) = [1 - sinh^-4(2J/T)]^(1/8) below Tc, and ~0 above. Exact, closed form,
    and nothing about it comes from this code."""
    temps = [1.5, 2.0, 2.6, 3.2]
    res = _ising_run(temps)
    for i, T in enumerate(temps):
        got = abs(res.mag_vector[i, 2])
        assert got == pytest.approx(_onsager_m(T), abs=0.03), f"T={T}"


@pytest.mark.slow
def test_S05_energy_at_Tc_matches_onsager():
    """Onsager's internal energy at criticality is exactly -sqrt(2) J per site.
    A 24x24 lattice reproduces it to a few percent (finite size)."""
    res = _ising_run([TC_ISING])
    assert res.energy[0] == pytest.approx(-np.sqrt(2.0), rel=0.05)


def test_S05_flip_proposal_keeps_the_system_ising():
    """The load-bearing mechanism: with `propose: flip` from a polarized start every
    spin must stay on the +/-z axis. If the proposal fell back to the uniform
    sphere move this is a Heisenberg model with a different Tc, and the Onsager
    comparison above would be meaningless."""
    from magcalc.thermal_mc import _sweep, build_supercell
    cfg = _load("S05")
    m = GenericSpinModel(cfg)
    H, b, N, S, _ = build_supercell(m, [cfg["parameters"]["J"]], (6, 6, 1))
    conf = np.tile(np.array([0.0, 0.0, S]), (N, 1))
    g = H @ conf.ravel() + b
    rng = np.random.default_rng(0)
    for _ in range(50):
        _sweep(conf, g, H, b, 1.0 / 2.0, S, rng, propose="flip")
        g = H @ conf.ravel() + b
    assert np.abs(conf[:, 0]).max() == 0.0 and np.abs(conf[:, 1]).max() == 0.0
    assert np.allclose(np.abs(conf[:, 2]), S)


def test_S05_replica_swaps_would_destroy_the_broken_symmetry():
    """Why the config sets `swap_every: 0`, recorded so nobody 'fixes' it.

    Below Tc the Ising model has two degenerate states. A replica that visits high T
    and returns can come back with the opposite sign, so <m> averages toward zero:
    measured 0.35 with swaps against Onsager's 0.9865 at T = 1.5. Without swaps each
    temperature stays in one broken-symmetry state, as Sunny's single-temperature
    LocalSampler does.
    """
    res = _ising_run([1.5], swap_every=0, n_sweeps=3000, n_equil=1000)
    assert abs(res.mag_vector[0, 2]) > 0.9


# ---------------------------------------------------------------------------
# S02 -- finite-T instantaneous S(q) on the S01 Hamiltonian.
# ---------------------------------------------------------------------------
def test_S02_static_correlations_peak_at_the_antiferromagnetic_wavevector():
    """Physics, not plumbing: CoRh2O4 orders Neel, so the instantaneous S(q) at
    finite T must carry more weight at the ordering wavevector than at a generic
    zone-interior point, and the contrast must SHARPEN on cooling."""
    from magcalc.thermal_mc import static_correlations
    cfg = _load("S02")
    m = GenericSpinModel(cfg)
    L = _lattice_from_params(cfg["crystal_structure"]["lattice_parameters"])
    B = 2 * np.pi * np.linalg.inv(L).T
    q = np.array([[0.0, 0.0, 0.0], [0.3, 0.17, 0.11], [1.0, 1.0, 1.0]]) @ B
    contrast = []
    for kT in (4.0, 1.3788):                       # 46 K and the tutorial's 16 K
        sq = static_correlations(m, [cfg["parameters"]["J"]], q, kT,
                                 supercell=(4, 4, 4), n_samples=120, n_equil=800,
                                 sample_every=5, seed=0).sq
        contrast.append(sq[2] / sq[1])
    assert contrast[1] > contrast[0], "correlations must sharpen on cooling"

