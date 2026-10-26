"""K2 of RUST_KERNELS_PLAN.md: the CP^(N-1) Metropolis sampler, RK4 integrator and
damped quench in rsMagCalc, behind `calculation: {kernels: rust}`.

The oracle tests (test_sun_dynamics, test_sun_sampler_equilibration, test_sun_quench,
the SU(N) half of test_classical_absolute_normalization and test_classical_window,
the S06 quench in test_sunny_tutorials) run under both kernel sets through the
`kernels` fixture; the 1e-12 parity of the deterministic kernels and the local-dE
identity on a mixed-spin cell live in `rsMagCalc/tests/test_cpn_parity.py`. What is
pinned HERE is the wiring:

  * every kernel entry point is reached from the Python API and from a config
    (`sun_sampled_correlations` with `kernels: rust` and `n_workers`), quickly,
    outside `slow`, so the fast suite touches it;
  * a kernel that raises at run time is logged and the NumPy path answers -- the
    fMagCalc contract;
  * MIXED SPIN goes through the kernel (the NumPy fast path refuses it and takes
    the per-site loop), and the chunked trajectory loop equals one call.
"""
import copy
import logging

import numpy as np
import pytest

from magcalc import kernels as K
from magcalc.generic_model import GenericSpinModel
from magcalc.sun import dynamics as sd
from magcalc.sun.lswt import SUNModel

LAT = [[4.0, 0, 0], [0, 7.0, 0], [0, 0, 7.0]]
NN = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))
B_MAT = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T


def _cfg(spins=(1.0, 1.0)):
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": 0.8} for p, o in NN],
             "single_ion_anisotropy": [{"value": -0.45, "axis": [0.3, 0, 1],
                                        "atoms": ["A", "B"]}]}
    return {"crystal_structure": {"lattice_vectors": LAT,
                                  "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                                "spin_S": spins[0]},
                                               {"label": "B", "pos": [0.5, 0, 0],
                                                "spin_S": spins[1]}]},
            "interactions": inter, "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[0, 0, 1], [0, 0, -1]]},
            "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}


def _model(spins=(1.0, 1.0), ncells=2):
    return SUNModel.from_generic_model(
        GenericSpinModel(copy.deepcopy(_cfg(spins))), params=[],
        supercell=[[ncells, 0, 0], [0, 1, 0], [0, 0, 1]])


def _rust():
    mod = K.rust_module()
    if mod is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    return mod


def _random_state(mdl, seed):
    rng = np.random.default_rng(seed)
    return [(lambda v: v / np.linalg.norm(v))(rng.standard_normal(n)
                                              + 1j * rng.standard_normal(n))
            for n in mdl.Ns]


# --------------------------------------------------------------------------
# Every entry point, quickly, under the fixture -- on a MIXED-spin cell
# --------------------------------------------------------------------------
@pytest.mark.parametrize("spins", [(1.0, 1.0), (0.5, 1.0)])
def test_thermalize_evolve_quench_and_sampled_correlations_run(kernels, spins):
    mdl = _model(spins)
    assert sd._fast(mdl)["ok"] == (spins[0] == spins[1])
    Z = [z.copy() for z in mdl.Z]
    rng = K.make_rng(3, kernels)
    assert K.kernel_of_rng(rng) == kernels
    _, info = sd.thermalize(mdl, Z, 0.3, 30, rng, sigma=0.2, return_info=True,
                            on_unequilibrated="off")
    assert len(info.energies) == 15 and 0.0 < info.acceptance < 1.0
    assert [float(np.linalg.norm(z)) for z in Z] == pytest.approx([1.0] * mdl.L,
                                                                  abs=1e-12)
    assert info.energies[-1] == pytest.approx(sd.energy(mdl, Z), abs=1e-10)

    traj = sd.evolve(mdl, Z, 0.02, 20, record_every=4, kernels=kernels)
    assert len(traj) == 5 and len(traj[0]) == mdl.L
    assert sd.energy(mdl, traj[-1]) == pytest.approx(sd.energy(mdl, Z), abs=1e-8)

    seen = []
    Zf, snaps = sd.quench(mdl, _random_state(mdl, 1), 0.02, 30, damping=0.1,
                          record_at=(0, 10, 30, 99), kernels=kernels,
                          callback=lambda step, state: seen.append(step))
    assert seen == [0, 10, 30] and sorted(snaps) == [0, 10, 30]
    assert sd.energy(mdl, Zf) < sd.energy(mdl, snaps[10]) < sd.energy(mdl, snaps[0])
    assert max(np.abs(a - b).max() for a, b in zip(Zf, snaps[30])) == 0.0
    Z1 = sd.damped_step(mdl, snaps[0], 0.02, 0.1, kernels=kernels)
    assert sd.energy(mdl, Z1) < sd.energy(mdl, snaps[0])

    qs = np.array([[0.25, 0, 0]]) @ B_MAT
    w, sqw = sd.sampled_correlations(mdl, qs, 0.3, dt=0.02, n_steps=64, n_traj=3,
                                     therm_sweeps=20, seed=2, sigma=0.2,
                                     kernels=kernels, n_workers=2)
    assert sqw.shape == (32, 1) and np.all(np.isfinite(sqw)) and sqw.sum() > 0


def test_sampled_correlations_is_deterministic_and_chunk_independent():
    rs = _rust()
    mdl = _model((0.5, 1.0), ncells=3)
    qs = np.array([[1 / 3, 0, 0], [0.5, 0, 0]]) @ B_MAT
    kw = dict(dt=0.02, n_steps=32, n_traj=5, therm_sweeps=20, seed=4, sigma=0.2,
              kernels="rust")
    a = sd.sampled_correlations(mdl, qs, 0.3, n_workers=1, **kw)[1]   # 5 chunks of 1
    b = sd.sampled_correlations(mdl, qs, 0.3, n_workers=3, **kw)[1]   # chunks of 3, 2
    c = sd.sampled_correlations(mdl, qs, 0.3, **kw)[1]                # one chunk
    assert np.array_equal(a, b) and np.array_equal(a, c)
    # and it is the kernel's trajectories, FFT-transformed by the shared code
    cm = sd._rust_model(mdl, rs)
    traj = cm.sampled_trajectories(mdl.Z, 0.3, 0, 5, 20, 0.2, True, 0.5, 0.02, 32,
                                   1, 4)[0]
    acc = None
    for t in range(5):
        _, s = sd.dynamical_structure_factor(mdl, [cm.unpack(r) for r in traj[t]], qs,
                                             0.02)
        acc = s if acc is None else acc + s
    from magcalc.classical_dynamics import classical_to_quantum_factor
    want = acc / 5 * classical_to_quantum_factor(
        2 * np.pi * np.fft.fftfreq(32, d=0.02)[:16], 0.3)[:, None]
    assert np.abs(a - want).max() < 1e-12 * max(want.max(), 1.0)


def test_the_sampler_knobs_reach_the_kernel(monkeypatch):
    """`adapt_sigma` / `target_acceptance` / `sigma` / `random_start` are forwarded
    to the kernel call (the NumPy counterpart is pinned in
    test_sun_sampler_equilibration)."""
    rs = _rust()
    mdl = _model()
    seen = {}
    real = rs.CpnModel.sampled_trajectories

    def spy(self, Z0, kT, start, k, n, sigma, adapt, target, *rest, **kw):
        seen.update(Z0=Z0, kT=kT, n=n, sigma=sigma, adapt=adapt, target=target)
        return real(self, Z0, kT, start, k, n, sigma, adapt, target, *rest, **kw)

    monkeypatch.setattr(rs.CpnModel, "sampled_trajectories", spy)
    qs = np.array([[0.25, 0, 0]]) @ B_MAT
    sd.sampled_correlations(mdl, qs, 0.5, dt=0.05, n_steps=16, n_traj=1,
                            therm_sweeps=20, seed=0, sigma=0.07, adapt_sigma=False,
                            target_acceptance=0.31, random_start=True,
                            on_unequilibrated="off", kernels="rust")
    assert seen["adapt"] is False and seen["target"] == 0.31
    assert seen["sigma"] == 0.07 and seen["Z0"] is None and seen["n"] == 20


def test_the_model_cache_notices_a_changed_hamiltonian():
    """Same invalidation key as `_fast`: removing every bond must rebuild the packed
    model, or the kernel would integrate a stale Hamiltonian."""
    rs = _rust()
    mdl = _model()
    Z = _random_state(mdl, 2)
    before = sd.evolve(mdl, Z, 0.02, 3, kernels="rust")[-1]
    mdl.bonds = []
    after = sd.evolve(mdl, Z, 0.02, 3, kernels="rust")[-1]
    numpy_after = sd.evolve(mdl, Z, 0.02, 3, kernels="numpy")[-1]
    assert max(np.abs(a - b).max() for a, b in zip(after, before)) > 1e-6
    assert max(np.abs(a - b).max() for a, b in zip(after, numpy_after)) < 1e-12


# --------------------------------------------------------------------------
# The fallback contract: a failing kernel is logged, NumPy answers
# --------------------------------------------------------------------------
def test_a_failing_kernel_falls_back_to_numpy_with_a_logged_exception(monkeypatch,
                                                                     caplog):
    rs = _rust()
    mdl = _model()

    class Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("synthetic kernel failure")

    monkeypatch.setattr(rs, "CpnModel", Boom)
    mdl._dyn_rust = None
    Z = [z.copy() for z in mdl.Z]
    qs = np.array([[0.25, 0, 0]]) @ B_MAT
    with caplog.at_level(logging.ERROR):
        rng = K.make_rng(1, "rust")
        assert K.is_rust_rng(rng)
        _, info = sd.thermalize(mdl, Z, 0.3, 10, rng, sigma=0.2, return_info=True,
                                on_unequilibrated="off")
        assert len(info.energies) == 5
        assert len(sd.evolve(mdl, Z, 0.02, 4, kernels="rust")) == 4
        sd.damped_step(mdl, Z, 0.02, 0.1, kernels="rust")
        sd.quench(mdl, Z, 0.02, 4, damping=0.1, kernels="rust")
        w, sqw = sd.sampled_correlations(mdl, qs, 0.3, dt=0.02, n_steps=16, n_traj=1,
                                         therm_sweeps=4, kernels="rust")
        assert sqw.shape == (8, 1)
    text = caplog.text
    for what in ("thermalize", "evolve", "damped_step", "quench", "sampled_trajectories"):
        assert f"rsmagcalc cpn {what} failed" in text, what
    assert "synthetic kernel failure" in text


# --------------------------------------------------------------------------
# From a config: sun_sampled_correlations under kernels: rust
# --------------------------------------------------------------------------
def test_config_path_runs_under_kernels_rust(tmp_path, caplog):
    """`calculation: {kernels: rust}` reaches the SU(N) sampler from YAML, with the
    `n_workers` key. The runner swallows a task's exception, so "no error" proves
    nothing: the success line must be there, the failure line must not."""
    import os

    import yaml

    from magcalc.runner import run_calculation

    cfg = _cfg()
    cfg.update({
        "tasks": {"minimization": True, "sun_sampled_correlations": True},
        "minimization": {"enabled": True, "num_starts": 2, "seed": 0},
        "sun_sampled_correlations": {"temperature": 0.3, "dt": 0.02, "n_steps": 32,
                                     "n_traj": 2, "therm_sweeps": 10, "sigma": 0.2,
                                     "n_workers": 2, "on_unequilibrated": "off"},
        "q_path": {"G": [0, 0, 0], "X": [0.5, 0, 0], "path": ["G", "X"],
                   "points_per_segment": 2},
        "plotting": {"enabled": False, "save_plot": False, "show_plot": False,
                     "plot_structure": False},
        "output": {"save_data": False},
        "calculation": {"mode": "SUN", "cache_mode": "none", "kernels": "rust",
                        "on_imaginary": "off"}})
    p = tmp_path / "sun_sc.yaml"
    p.write_text(yaml.safe_dump(cfg))
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        with caplog.at_level(logging.INFO):
            run_calculation(str(p))
    finally:
        os.chdir(cwd)
    text = caplog.text
    assert "sun_sampled_correlations failed" not in text, text
    assert "SU(N) SampledCorrelations (" in text, text
    msgs = [r.getMessage() for r in caplog.records if r.getMessage().startswith("kernels:")]
    expected = "kernels: rust" if K.rust_module() is not None else "kernels: numpy"
    assert msgs and msgs[0].startswith(expected), msgs
    assert "falling back" not in text.lower() or K.rust_module() is None, text


# --------------------------------------------------------------------------
# The vectorized trajectory moments (what dominates once the loops are in Rust)
# --------------------------------------------------------------------------
def test_moments_along_equals_moment_of_on_an_entangled_unit():
    """`moments_along` is `moment_of` at every configuration, as one einsum. Checked
    on an ENTANGLED unit, whose `moment_terms` carry non-zero constituent offsets
    and operator indices beyond (0, 1, 2) -- the plain-dipole case cannot tell a
    dropped phase from a right one, the dimer's (1 - cos q.d) form factor can."""
    import os

    import yaml

    from magcalc.sun.entangled import build_entangled_model

    here = os.path.dirname(__file__)
    cfg = yaml.safe_load(open(os.path.join(here, "..", "examples", "entangled",
                                           "dimer_chain", "config.yaml")))
    gm = GenericSpinModel(cfg)
    mdl = build_entangled_model(gm, params=[4.0, 1.0], units=[[0, 1]])
    assert any(np.linalg.norm(d) > 0 for ml in mdl.moment_terms for (d, _) in ml)
    rng = np.random.default_rng(0)
    traj = []
    for _ in range(3):
        traj.append([(lambda v: v / np.linalg.norm(v))(rng.standard_normal(n)
                                                       + 1j * rng.standard_normal(n))
                     for n in mdl.Ns])
    B = 2 * np.pi * np.linalg.inv(np.array(cfg["crystal_structure"]["lattice_vectors"],
                                           float)).T
    qs = np.array([[0.1, 0, 0], [0.5, 0, 0], [0.37, 0.2, 0.1]]) @ B
    want = np.array([[sd.moment_of(mdl, c, q) for q in qs] for c in traj])
    got = sd.moments_along(mdl, traj, qs)
    assert got.shape == want.shape == (3, 3, 3)
    assert np.abs(got - want).max() < 1e-12 * np.abs(want).max()
