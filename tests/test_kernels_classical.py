"""K1 of RUST_KERNELS_PLAN.md: the classical Metropolis sweep and Landau-Lifshitz
integrator in rsMagCalc, behind `calculation: {kernels: rust}`.

The oracle tests (test_thermal_mc, test_classical_absolute_normalization,
test_integrators, test_classical_window, test_classical_to_quantum,
test_static_correlations, test_wang_landau, test_disorder, the S05 Ising case in
test_sunny_tutorials) run under both kernel sets through the `kernels` fixture; the
1e-12 parity of the deterministic kernels lives in
`rsMagCalc/tests/test_classical_parity.py`. What is pinned HERE is the wiring:

  * every kernel entry point is reached from the Python API and from a config
    (`thermal_mc`, `sampled_correlations`, `static_correlations`, `wang_landau`
    with `kernels: rust`), quickly, outside `slow`, so the fast suite touches it;
  * a kernel that raises at run time is logged and the NumPy path answers --
    the fMagCalc contract, which `kernels.resolve` alone cannot provide;
  * the per-site spin contract (a scalar S is broadcast) and the chunked
    trajectory loop give the same answer as a single call.
"""
import logging

import numpy as np
import pytest

from magcalc import classical_dynamics as cd
from magcalc import kernels as K
from magcalc import thermal_mc as tmc
from magcalc.generic_model import GenericSpinModel

LAT = [[4.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
B_MAT = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T


def _fm_chain():
    cfg = {"crystal_structure": {"lattice_vectors": LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
               {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0}]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": [],
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    return GenericSpinModel(cfg)


def _rust():
    mod = K.rust_module()
    if mod is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    return mod


# --------------------------------------------------------------------------
# Every entry point, quickly, under the fixture
# --------------------------------------------------------------------------
def test_make_rng_selects_the_stream_and_the_sweep_kernel(kernels):
    rng = K.make_rng(3, kernels)
    assert K.kernel_of_rng(rng) == kernels
    model = _fm_chain()
    H, b, N, S, _ = tmc.build_supercell(model, [], (6, 1, 1))
    m = rng.standard_normal((N, 3))
    m *= S / np.linalg.norm(m, axis=1, keepdims=True)
    g = H @ m.ravel() + b
    acc = tmc._sweep(m, g, H, b, 1.0, S, rng)
    assert 0.0 <= acc <= 1.0
    assert np.abs(g - (H @ m.ravel() + b)).max() < 1e-10
    assert np.linalg.norm(m, axis=1) == pytest.approx(S, abs=1e-12)


def test_parallel_tempering_runs_and_is_deterministic(kernels):
    model = _fm_chain()
    H, b, N, S, _ = tmc.build_supercell(model, [], (8, 1, 1))
    temps = np.array([0.5, 1.0])
    a = tmc.parallel_tempering(H, b, N, S, temps, n_sweeps=200, n_equil=50, seed=4,
                               kernels=kernels)
    c = tmc.parallel_tempering(H, b, N, S, temps, n_sweeps=200, n_equil=50, seed=4,
                               kernels=kernels, n_workers=2)
    assert np.array_equal(a.energy, c.energy)
    assert a.n_spins == N and 0.0 < a.accept_rate < 1.0
    assert a.energy[0] < a.energy[1]           # colder replica sits lower


def test_evolve_and_sampled_correlations_run(kernels):
    model = _fm_chain()
    H, b, N, S, pos = tmc.build_supercell(model, [], (6, 1, 1))
    m0 = K.make_rng(1, "numpy").standard_normal((N, 3))
    m0 *= S / np.linalg.norm(m0, axis=1, keepdims=True)
    for integ in ("rk4", "midpoint"):
        traj = cd.evolve(H, b, S, m0, 0.01, 30, record_every=3, integrator=integ,
                         kernels=kernels)
        assert traj.shape == (10, N, 3)
        assert np.linalg.norm(traj, axis=-1) == pytest.approx(S, abs=1e-9)
    qs = np.array([[0.25, 0, 0]]) @ B_MAT
    res = cd.sampled_correlations(model, [], qs, 0.5, supercell=(6, 1, 1), dt=0.02,
                                  n_steps=64, n_traj=3, therm_sweeps=20, seed=2,
                                  kernels=kernels, n_workers=2)
    assert res.sqw.shape == (32, 1) and np.all(np.isfinite(res.sqw))
    assert res.sqw.sum() > 0


def test_static_correlations_wang_landau_and_langevin_run(kernels):
    model = _fm_chain()
    qs = np.array([[0.25, 0, 0]]) @ B_MAT
    r = tmc.static_correlations(model, [], qs, kT=1.0, supercell=(6, 1, 1),
                                n_samples=10, n_equil=10, sample_every=1, seed=1,
                                kernels=kernels)
    assert r.sq.shape == (1,) and r.sq[0] > 0
    H, b, N, S, _ = tmc.build_supercell(model, [], (3, 1, 1))
    lo, hi = tmc.wang_landau_window(H, b, N, S, seed=0)
    w = tmc.wang_landau(H, b, N, S, lo, hi, n_bins=20, f_final=0.25,
                        sweeps_per_check=20, seed=1, kernels=kernels)
    assert (w.histogram > 0).sum() > 5 and w.f_final <= 0.25
    m = cd.langevin_thermalize(H, b, S, np.tile([0.0, 0.0, S], (N, 1)), 0.5, 0.02,
                               50, damping=0.5, seed=1, kernels=kernels)
    assert np.linalg.norm(m, axis=1) == pytest.approx(S, abs=1e-12)
    rng = K.make_rng(2, kernels)
    m2 = cd.langevin_step(H, b, m, 0.02, S, 0.5, 0.5, rng)
    assert m2.shape == (N, 3) and not np.array_equal(m2, m)


# --------------------------------------------------------------------------
# The chunked trajectory loop and the per-site spin contract
# --------------------------------------------------------------------------
def test_chunked_trajectories_equal_one_call_and_a_scalar_s_is_broadcast():
    rs = _rust()
    model = _fm_chain()
    H, b, N, S, _ = tmc.build_supercell(model, [], (5, 1, 1))
    one, _ = rs.sampled_trajectories(H, b, S, 0.5, 0, 5, 10, 0.01, 8, seed=9)
    parts = [rs.sampled_trajectories(H, b, np.full(N, S), 0.5, k, 2, 10, 0.01, 8,
                                     seed=9, n_workers=1)[0]
             for k in (0, 2, 4)]
    assert np.array_equal(one, np.concatenate(parts)[:5])
    with pytest.raises(ValueError, match="one entry per site"):
        rs.sampled_trajectories(H, b, np.ones(N + 1), 0.5, 0, 1, 1, 0.01, 2)


# --------------------------------------------------------------------------
# The fallback contract: a failing kernel is logged, NumPy answers
# --------------------------------------------------------------------------
def test_a_failing_kernel_falls_back_to_numpy_with_a_logged_exception(monkeypatch,
                                                                     caplog):
    rs = _rust()
    model = _fm_chain()
    H, b, N, S, _ = tmc.build_supercell(model, [], (6, 1, 1))

    def boom(*a, **k):
        raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(rs, "parallel_tempering", boom)
    monkeypatch.setattr(rs, "evolve", boom)
    monkeypatch.setattr(rs, "sampled_trajectories", boom)
    monkeypatch.setattr(rs, "wang_landau", boom)
    monkeypatch.setattr(rs, "sweep", boom)
    monkeypatch.setattr(rs, "langevin", boom)
    with caplog.at_level(logging.ERROR):
        pt = tmc.parallel_tempering(H, b, N, S, np.array([0.5]), n_sweeps=50,
                                    n_equil=10, seed=1, kernels="rust")
        ref = tmc.parallel_tempering(H, b, N, S, np.array([0.5]), n_sweeps=50,
                                     n_equil=10, seed=1, kernels="numpy")
        assert np.array_equal(pt.energy, ref.energy)     # the NumPy stream answered
        m0 = np.tile([0.0, 0.0, S], (N, 1))
        assert cd.evolve(H, b, S, m0, 0.01, 4, kernels="rust").shape == (4, N, 3)
        qs = np.array([[0.25, 0, 0]]) @ B_MAT
        cd.sampled_correlations(model, [], qs, 0.5, supercell=(6, 1, 1), n_steps=16,
                                n_traj=1, therm_sweeps=2, kernels="rust")
        lo, hi = tmc.wang_landau_window(H, b, N, S, seed=0)
        tmc.wang_landau(H, b, N, S, lo, hi, n_bins=10, f_final=0.5,
                        sweeps_per_check=5, kernels="rust")
        rng = K.make_rng(1, "rust")
        m = m0.copy()
        g = H @ m.ravel() + b
        tmc._sweep(m, g, H, b, 1.0, S, rng)
        cd.langevin_step(H, b, m, 0.02, S, 0.5, 0.5, K.make_rng(1, "rust"))
    text = caplog.text
    for what in ("parallel_tempering", "evolve", "sampled_trajectories",
                 "wang_landau", "sweep", "langevin"):
        assert f"rsmagcalc {what} failed" in text, what
    assert "synthetic kernel failure" in text


# --------------------------------------------------------------------------
# From a config: the four classical tasks under kernels: rust
# --------------------------------------------------------------------------
def _config(task, block, extra=None):
    cfg = {"crystal_structure": {"lattice_vectors": LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
               {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0}]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": [],
           "tasks": {task: True},
           task: block,
           "q_path": {"G": [0, 0, 0], "X": [0.5, 0, 0], "path": ["G", "X"],
                      "points_per_segment": 2},
           "plotting": {"enabled": False, "save_plot": False, "show_plot": False,
                        "plot_structure": False},
           "output": {"save_data": False},
           "calculation": {"cache_mode": "none", "kernels": "rust"}}
    cfg.update(extra or {})
    return cfg


@pytest.mark.parametrize("task,block,success", [
    ("thermal_mc", {"temperatures": [0.5, 1.0], "supercell": [6, 1, 1],
                    "n_sweeps": 60, "n_equil": 20, "n_workers": 2}, "Thermal MC ("),
    ("sampled_correlations", {"temperature": 0.5, "supercell": [6, 1, 1],
                              "n_steps": 32, "n_traj": 2, "therm_sweeps": 5,
                              "n_workers": 2}, "SampledCorrelations ("),
    ("static_correlations", {"temperature": 0.5, "supercell": [6, 1, 1],
                             "n_samples": 5, "n_equil": 5, "sample_every": 1},
     "Static correlations ("),
    ("wang_landau", {"supercell": [3, 1, 1], "n_bins": 20, "f_final": 0.25,
                     "temperatures": [0.5, 1.0]}, "Wang-Landau: "),
])
def test_config_paths_run_under_kernels_rust(tmp_path, caplog, task, block, success):
    """`calculation: {kernels: rust}` reaches each classical task from YAML. The
    runner swallows a task's exception, so "no error" proves nothing: the success
    line must be there, the failure line must not, and the kernel line must say
    rust (or, without rsmagcalc, the named fallback)."""
    import os

    import yaml

    from magcalc.runner import run_calculation

    p = tmp_path / f"{task}.yaml"
    p.write_text(yaml.safe_dump(_config(task, block)))
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        with caplog.at_level(logging.INFO):
            run_calculation(str(p))
    finally:
        os.chdir(cwd)
    text = caplog.text
    assert f"{task} failed" not in text, text
    assert success in text, text
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("kernels:")]
    expected = "kernels: rust" if K.rust_module() is not None else "kernels: numpy"
    assert msgs and msgs[0].startswith(expected), msgs
    assert "falling back" not in text.lower() or K.rust_module() is None, text
