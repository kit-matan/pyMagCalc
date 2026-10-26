"""K3 of RUST_KERNELS_PLAN.md: the SU(N) LSWT hot path in rsMagCalc, behind
`calculation: {kernels: rust}`.

The oracle tests (test_sun's Sunny gates on FeI2 and the S=1/2 chain,
test_mixed_spin_sun, test_powder_sun, test_sun_zeeman, test_sun_missing_terms,
test_pair_operator, test_entangled_units, test_kpm_stability, test_bond_disorder and
the S09 SU(N) case in test_sunny_tutorials) run under both kernel sets through the
`kernels` fixture; the elementwise parity (H(q) to 1e-13, energies and per-band
tensors to 1e-10, the guard verdict by verdict) lives in
`rsMagCalc/tests/test_sun_lswt_parity.py`. What is pinned HERE is the wiring:

  * every SUNModel entry point reaches the kernel from the Python API and from a
    config (mode SUN and mode entangled, with `n_workers`), quickly, outside `slow`;
  * the kernel answers what the NumPy path answers, on a MIXED-spin supercell;
  * a kernel that raises is logged and the NumPy path answers -- the fMagCalc
    contract -- while a NON-minimum still raises `LinAlgError` under both kernel
    sets and is NOT mistaken for a kernel failure;
  * the packed model is rebuilt whenever the reference state or the bond list
    changes (a relaxation, a disorder draw, a swapped bond list), and a model
    holding one still copies and pickles.
"""
import copy
import logging
import os
import pickle

import numpy as np
import pytest
import yaml

from magcalc import kernels as K
from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel, apply_bond_disorder

LAT = [[4.0, 0, 0], [0, 7.0, 0], [0, 0, 7.0]]
NN = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))
B_MAT = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
QS = np.array([[0.13, 0.21, 0.05], [0.31, -0.07, 0.11], [0.47, 0.03, 0.0]]) @ B_MAT


def _cfg(spins=(0.5, 1.0)):
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


def _model(kernels="numpy", spins=(0.5, 1.0), ncells=2):
    mdl = SUNModel.from_generic_model(
        GenericSpinModel(copy.deepcopy(_cfg(spins))), params=[],
        supercell=[[ncells, 0, 0], [0, 1, 0], [0, 0, 1]], kernels=kernels)
    mdl.minimize_energy(n_restarts=4, seed=0)
    return mdl


def _rust():
    mod = K.rust_module()
    if mod is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    return mod


def _twin(mdl):
    """The same reference state under the other kernel set."""
    other = SUNModel(mdl.S, mdl.Z, mdl.bonds, mdl.onsite, operators=mdl.ops,
                     moment_terms=mdl.moment_terms,
                     kernels="numpy" if mdl.kernels == "rust" else "rust")
    other.pos, other.n_cells = mdl.pos, mdl.n_cells
    return other


# --------------------------------------------------------------------------
# Every entry point, on a MIXED-spin supercell, under the fixture
# --------------------------------------------------------------------------
def test_every_entry_point_agrees_with_the_numpy_path(kernels):
    mdl = _model(kernels)
    assert mdl.N is None and mdl.D == 6          # mixed spin: M = (1, 2) x 2 cells
    ref = _twin(mdl) if kernels == "rust" else mdl
    assert (mdl._rust() is not None) == (kernels == "rust")

    H = mdl.hamiltonians(QS)
    assert H.shape == (len(QS), 2 * mdl.D, 2 * mdl.D)
    for k, q in enumerate(QS):
        assert np.abs(H[k] - ref._hamiltonian_numpy(q)).max() < 1e-13
        assert np.abs(mdl.hamiltonian(q) - H[k]).max() < 1e-13
        assert np.abs(mdl.dispersion(q) - ref.dispersion(q)).max() < 1e-10
        assert abs(mdl.max_imaginary(q) - ref.max_imaginary(q)) < 1e-10
        w, T = mdl._bogoliubov(q)
        w0, _ = ref._bogoliubov(q)
        assert np.abs(w - w0).max() < 1e-10
        g = np.diag([1.0] * mdl.D + [-1.0] * mdl.D)
        assert np.abs(T.conj().T @ g @ T - g).max() < 1e-10
    assert np.abs(mdl.dispersion_batch(QS) - ref.dispersion_batch(QS)).max() < 1e-10

    for cs in ("perp", "trace", "zz"):
        w, I = mdl.structure_factor_batch(QS, ion="Fe2", cross_section=cs)
        w0, I0 = ref.structure_factor_batch(QS, ion="Fe2", cross_section=cs)
        assert np.abs(w - w0).max() < 1e-10
        # degenerate pairs may split differently; their sums may not
        for a, b, e in zip(I, I0, w):
            for lvl in np.unique(np.round(e, 8)):
                sel = np.abs(e - lvl) < 1e-7
                assert abs(a[sel].sum() - b[sel].sum()) < 1e-10, cs
        w1, I1 = mdl.structure_factor(QS[1], ion="Fe2", cross_section=cs)
        assert np.array_equal(w1, w[1]) and np.array_equal(I1, I[1])

    rep = mdl.assert_stable(QS)
    assert rep["stable"] and rep["n_checked"] == len(QS)
    assert [mdl.is_stable_at(q) for q in QS] == [ref.is_stable_at(q) for q in QS]


def test_the_empty_batch_is_empty(kernels):
    mdl = _model(kernels)
    empty = np.zeros((0, 3))
    assert mdl.hamiltonians(empty).shape == (0, 2 * mdl.D, 2 * mdl.D)
    assert mdl.dispersion_batch(empty).shape == (0, mdl.D)
    w, I = mdl.structure_factor_batch(empty)
    assert w.shape == (0, mdl.D) and I.shape == (0, mdl.D)
    assert mdl.assert_stable(empty)["stable"]


# --------------------------------------------------------------------------
# A non-minimum is NOT a kernel failure
# --------------------------------------------------------------------------
def _ferro_on_afm(kernels):
    cfg = _cfg((1.0, 1.0))
    cfg["interactions"].pop("single_ion_anisotropy")
    cfg["magnetic_structure"] = {"type": "pattern", "pattern_type": "ferromagnetic",
                                 "direction": [0, 0, 1]}
    return SUNModel.from_generic_model(GenericSpinModel(cfg), params=[],
                                       kernels=kernels)


def test_a_non_minimum_raises_LinAlgError_under_both_and_logs_no_fallback(kernels,
                                                                           caplog):
    mdl = _ferro_on_afm(kernels)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
            mdl.structure_factor(QS[0])
        with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
            mdl._bogoliubov(QS[0])
        assert not mdl.assert_stable(QS, on_failure="off")["stable"]
    assert "falling back" not in caplog.text


# --------------------------------------------------------------------------
# The fMagCalc contract: a failing kernel is logged, NumPy answers
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["hamiltonians", "eigenvalues", "bogoliubov",
                                    "correlations", "stable"])
def test_a_failing_kernel_falls_back_to_numpy_and_says_so(method, monkeypatch, caplog):
    rs = _rust()
    mdl = _model("rust")
    ref = _twin(mdl)
    assert mdl._rust() is not None

    def boom(self, *a, **k):
        raise RuntimeError("injected kernel failure")

    monkeypatch.setattr(rs.SunModel, method, boom)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        q = QS[1]
        if method == "hamiltonians":
            got, want = mdl.hamiltonian(q), ref._hamiltonian_numpy(q)
        elif method == "eigenvalues":
            got, want = mdl.dispersion(q), ref.dispersion(q)
        elif method == "bogoliubov":
            got, want = mdl._bogoliubov(q)[0], ref._bogoliubov(q)[0]
        elif method == "correlations":
            got, want = mdl.structure_factor(q)[0], ref.structure_factor(q)[0]
        else:
            got, want = mdl.assert_stable(QS)["stable"], True
            assert mdl.is_stable_at(q)
        assert np.allclose(got, want, atol=1e-10)
    assert "falling back to the NumPy implementation" in caplog.text
    assert "injected kernel failure" in caplog.text


def test_a_model_the_kernel_cannot_pack_runs_in_numpy(monkeypatch, caplog):
    rs = _rust()
    mdl = _model("rust")

    class Refuses:
        def __init__(self, *a, **k):
            raise ValueError("cannot pack")

    monkeypatch.setattr(rs, "SunModel", Refuses)
    mdl._prepare()                      # drop the packed copy
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        w = mdl.dispersion(QS[0])
    assert "SunModel packing" in caplog.text
    assert np.allclose(w, _twin(mdl).dispersion(QS[0]), atol=1e-12)


# --------------------------------------------------------------------------
# The packed copy never goes stale
# --------------------------------------------------------------------------
def test_the_packed_model_follows_relaxation_disorder_and_a_swapped_bond_list():
    _rust()
    mdl = _model("rust")
    first = mdl._rust()
    assert mdl._rust() is first                          # cached between calls

    mdl.minimize_energy(n_restarts=2, seed=5)            # _prepare -> rebuilt
    assert mdl._rust() is not first
    assert np.allclose(mdl.dispersion(QS[0]), _twin(mdl).dispersion(QS[0]), atol=1e-10)

    apply_bond_disorder(mdl, 0.3, seed=1)                # new bonds + _prepare
    assert np.allclose(mdl.dispersion(QS[0]), _twin(mdl).dispersion(QS[0]), atol=1e-10)

    before = mdl._rust()
    mdl.bonds = list(mdl.bonds)[:-2]                     # replaced WITHOUT _prepare
    assert mdl._rust() is not before
    assert np.abs(mdl.hamiltonian(QS[0]) - mdl._hamiltonian_numpy(QS[0])).max() < 1e-13


def test_a_model_holding_a_packed_kernel_still_copies_and_pickles():
    _rust()
    mdl = _model("rust")
    mdl.dispersion(QS[0])
    assert mdl._rust() is not None
    for clone in (copy.deepcopy(mdl), pickle.loads(pickle.dumps(mdl))):
        assert clone.kernels == "rust"
        assert np.allclose(clone.dispersion(QS[0]), mdl.dispersion(QS[0]), atol=1e-12)


# --------------------------------------------------------------------------
# From a config: mode SUN (S(Q,w) + powder) and mode entangled, with n_workers
# --------------------------------------------------------------------------
def _run_cfg(tmp_path, cfg, name):
    from magcalc.runner import run_calculation
    d = tmp_path / name
    d.mkdir()
    p = d / "c.yaml"
    p.write_text(yaml.safe_dump(cfg))
    cwd = os.getcwd()
    os.chdir(str(d))
    try:
        run_calculation(str(p))
    finally:
        os.chdir(cwd)
    return {k: np.load(d / f"{k}.npz") for k in ("disp", "sqw")}


def _sun_config(kernels):
    cfg = _cfg((0.5, 1.0))
    cfg["calculation"] = {"mode": "SUN", "cache_mode": "none", "kernels": kernels,
                          "n_workers": 2}
    cfg["tasks"] = {"minimization": True, "dispersion": True, "sqw_map": True}
    cfg["minimization"] = {"num_starts": 4, "seed": 0}
    cfg.pop("magnetic_structure")
    cfg["q_path"] = {"G": [0.0, 0.1, 0.0], "X": [0.5, 0.1, 0.0], "path": ["G", "X"],
                     "points_per_segment": 6}
    cfg["plotting"] = {"enabled": False, "save_plot": False, "show_plot": False}
    cfg["output"] = {"save_data": True, "disp_data_filename": "disp.npz",
                     "sqw_data_filename": "sqw.npz"}
    return cfg


def test_mode_sun_config_runs_the_kernel_and_reproduces_numpy(tmp_path, caplog):
    _rust()
    with caplog.at_level(logging.INFO):
        rust = _run_cfg(tmp_path, _sun_config("rust"), "rust")
    assert "kernels: rust (rsmagcalc" in caplog.text
    numpy_ = _run_cfg(tmp_path, _sun_config("numpy"), "numpy")
    assert np.abs(rust["disp"]["energies"] - numpy_["disp"]["energies"]).max() < 1e-10
    assert np.abs(rust["sqw"]["energies"] - numpy_["sqw"]["energies"]).max() < 1e-10
    # Per DEGENERATE GROUP: at X two bands meet exactly (1.6 and 1.6), and how the
    # weight splits inside a degenerate pair is a basis choice (5e-9 moved between
    # them here, the sum to 1e-16), not an observable.
    for e, a, b in zip(rust["sqw"]["energies"], rust["sqw"]["intensities"],
                       numpy_["sqw"]["intensities"]):
        for lvl in np.unique(np.round(e, 7)):
            sel = np.abs(e - lvl) < 1e-6
            assert abs(a[sel].sum() - b[sel].sum()) < 1e-10


def test_the_calculators_carry_kernels_and_n_workers_to_the_model():
    _rust()
    from magcalc.sun.adapter import SUNCalculator
    from magcalc.sun.entangled import EntangledCalculator

    cfg = _cfg((0.5, 1.0))
    cfg["calculation"] = {"mode": "SUN", "on_imaginary": "off", "n_workers": 3}
    sun = SUNCalculator(GenericSpinModel(copy.deepcopy(cfg)), cfg, [], kernels="rust")
    assert sun.model.kernels == "rust" and sun.model.n_workers == 3
    sun.update_hamiltonian_params([])
    assert sun.model.kernels == "rust"
    rep = sun.stability_report(n_q=4, q_cart=QS)
    assert rep["max_imag"] < 1e-8 and rep["band_scale"] > 0

    ecfg = yaml.safe_load(open(os.path.join(
        os.path.dirname(__file__), "..", "examples", "entangled", "dimer_chain",
        "config.yaml")))
    ecfg.setdefault("calculation", {})["n_workers"] = 2
    ent = EntangledCalculator(GenericSpinModel(copy.deepcopy(ecfg)), ecfg,
                              [ecfg["parameters"][k] for k in ecfg["parameter_order"]],
                              kernels="rust")
    assert ent.model.kernels == "rust" and ent.model.n_workers == 2
    a = ecfg["crystal_structure"]["lattice_vectors"][0][0]
    J, Jp = ecfg["parameters"]["J"], ecfg["parameters"]["Jp"]
    qs = [[2 * np.pi * h / a, 0, 0] for h in (0.0, 0.2, 0.5)]
    w = ent.calculate_dispersion(qs).energies.max(axis=1)
    assert np.allclose(w, [np.sqrt(J ** 2 - J * Jp * np.cos(2 * np.pi * h))
                           for h in (0.0, 0.2, 0.5)], atol=1e-9)
    sq = ent.calculate_sqw(qs)
    assert sq.intensities.shape == sq.energies.shape and sq.intensities[0].sum() < 1e-9
