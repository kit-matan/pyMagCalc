"""K5 of RUST_KERNELS_PLAN.md: the SCGA loops in rsMagCalc, behind
`calculation: {kernels: rust}`.

The oracle tests (test_scga: Sunny SCGA on the square and kagome antiferromagnets,
the free-spin sum rule, the high-T Curie limit, the classical-chain closed form, the
sublattice refusal; test_hamiltonian_terms: a parameterised g-tensor through S(q))
run under both kernel sets through the `kernels` fixture; the tensor / residual /
lambda parity at 1e-12 (with an anisotropic g, form factors, DM, a 16-site cell)
lives in `rsMagCalc/tests/test_scga_parity.py`. What is pinned HERE is the wiring,
quickly, outside `slow`:

  * every entry point -- `solve_lambda` (mesh eigenvalues + sublattice residuals),
    `sublattice_sum_rule_residuals`, `scga_intensities`, `compute_scga` -- reaches
    the kernel and answers what the NumPy loop answers, lambda bit for bit;
  * an empty q batch is an empty answer under both;
  * a kernel that raises, at any of the three calls or while packing, is logged and
    the NumPy loop answers -- the fMagCalc contract;
  * the inequivalent-sublattice refusal is unchanged under "rust";
  * a config run with `tasks: {scga: true}` and `kernels: rust` names the kernel in
    its log and saves what the NumPy run saves.
"""
import logging
import os
import sys

import numpy as np
import pytest
import yaml

from magcalc import kernels as K
from magcalc import scga

sys.path.insert(0, os.path.dirname(__file__))
from test_scga import _kagome, _mixed_spin_chain  # noqa: E402

KT = 1.5


def _rust():
    mod = K.rust_module()
    if mod is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    return mod


def _qs(lv):
    B = 2 * np.pi * np.linalg.inv(np.asarray(lv, float)).T
    hk = [[0, 0, 0], [1 / 3, 1 / 3, 0], [0.5, 0, 0], [0.21, 0.67, 0.3], [1.2, -0.4, 0]]
    return np.asarray(hk, float) @ B


@pytest.fixture
def spy(monkeypatch):
    """Count calls into the three K5 entry points of `rsmagcalc.ScgaModel`."""
    rs = _rust()
    calls = {"eigenvalues": 0, "sum_rule_residuals": 0, "correlations": 0}
    for name in calls:
        orig = getattr(rs.ScgaModel, name)

        def wrap(self, *a, _orig=orig, _name=name, **k):
            calls[_name] += 1
            return _orig(self, *a, **k)
        monkeypatch.setattr(rs.ScgaModel, name, wrap)
    return calls


# --------------------------------------------------------------------------
# Every entry point reaches the kernel and agrees with NumPy
# --------------------------------------------------------------------------
def test_every_entry_point_reaches_the_kernel_and_agrees(spy, caplog):
    m, lv = _kagome()
    qs = _qs(lv)
    with caplog.at_level(logging.WARNING):
        lam_r, lmin_r, res_r, cache_r, mesh = scga.solve_lambda(m, [1.0], KT, nq=10,
                                                                kernels="rust")
        assert spy["eigenvalues"] == 1 and spy["sum_rule_residuals"] == 1
        r_r = scga.sublattice_sum_rule_residuals(m, [1.0], KT, lam_r, cache_r, mesh,
                                                 kernels="rust")
        assert spy["sum_rule_residuals"] == 2
        I_r = scga.scga_intensities(m, [1.0], qs, KT, lam=lam_r, mesh=mesh,
                                    cache=cache_r, kernels="rust")
        assert spy["correlations"] == 1
        res = scga.compute_scga(m, [1.0], qs, KT, nq=10, kernels="rust", n_workers=2)
        assert spy == {"eigenvalues": 2, "sum_rule_residuals": 3, "correlations": 2}
    assert "falling back" not in caplog.text

    lam_n, lmin_n, res_n, cache_n, _ = scga.solve_lambda(m, [1.0], KT, nq=10)
    assert lam_r == lam_n and lmin_r == lmin_n           # bit-identical eigenvalues
    assert abs(res_r - res_n) < 1e-15
    r_n = scga.sublattice_sum_rule_residuals(m, [1.0], KT, lam_n, cache_n, mesh)
    assert np.max(np.abs(r_r - r_n)) < 1e-12
    I_n = scga.scga_intensities(m, [1.0], qs, KT, lam=lam_n, mesh=mesh, cache=cache_n)
    assert np.max(np.abs(I_r - I_n)) < 1e-12 * np.max(np.abs(I_n))
    assert res.lam == lam_n
    assert np.allclose(res.intensities, I_n, rtol=1e-12, atol=0)


def test_lam_none_solves_lambda_through_the_kernel_too(spy):
    m, lv = _kagome()
    qs = _qs(lv)
    I_r = scga.scga_intensities(m, [1.0], qs, KT, nq=8, kernels="rust")
    assert spy["eigenvalues"] == 1 and spy["correlations"] == 1
    I_n = scga.scga_intensities(m, [1.0], qs, KT, nq=8)
    assert np.allclose(I_r, I_n, rtol=1e-12, atol=0)


@pytest.mark.parametrize("which", ["numpy", "rust"])
def test_an_empty_q_batch_is_an_empty_answer(which):
    if which == "rust":
        _rust()
    m, _lv = _kagome()
    lam, _l, _r, cache, mesh = scga.solve_lambda(m, [1.0], KT, nq=6, kernels=which)
    I = scga.scga_intensities(m, [1.0], np.zeros((0, 3)), KT, lam=lam, mesh=mesh,
                              cache=cache, kernels=which)
    assert I.shape == (0,)
    res = scga.compute_scga(m, [1.0], np.zeros((0, 3)), KT, nq=6, kernels=which)
    assert res.intensities.shape == (0,)


# --------------------------------------------------------------------------
# A failing kernel falls back to NumPy, loudly
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method,what", [("eigenvalues", "SCGA mesh eigenvalues"),
                                         ("sum_rule_residuals", "SCGA sum-rule residuals"),
                                         ("correlations", "SCGA correlations")])
def test_a_failing_kernel_falls_back_to_numpy_and_says_so(method, what, monkeypatch,
                                                          caplog):
    rs = _rust()

    def boom(self, *a, **k):
        raise RuntimeError("injected kernel failure")
    monkeypatch.setattr(rs.ScgaModel, method, boom)
    m, lv = _kagome()
    qs = _qs(lv)
    want = scga.compute_scga(m, [1.0], qs, KT, nq=8)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        got = scga.compute_scga(m, [1.0], qs, KT, nq=8, kernels="rust")
    assert got.lam == want.lam
    assert np.allclose(got.intensities, want.intensities, rtol=1e-12, atol=0)
    assert f"rsmagcalc {what} failed" in caplog.text
    assert "falling back to the NumPy implementation" in caplog.text
    assert "injected kernel failure" in caplog.text


def test_a_cache_the_kernel_cannot_pack_runs_in_numpy(monkeypatch, caplog):
    rs = _rust()

    def boom(cls, cache):
        raise ValueError("injected packing failure")
    monkeypatch.setattr(rs.ScgaModel, "from_cache", classmethod(boom))
    m, lv = _kagome()
    want = scga.compute_scga(m, [1.0], _qs(lv), KT, nq=8)
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        got = scga.compute_scga(m, [1.0], _qs(lv), KT, nq=8, kernels="rust")
    assert got.lam == want.lam
    assert np.allclose(got.intensities, want.intensities, rtol=1e-12, atol=0)
    assert "injected packing failure" in caplog.text


def test_the_packed_model_is_reused_and_follows_the_bond_list():
    _rust()
    m, _lv = _kagome()
    _lam, _l, _r, cache, mesh = scga.solve_lambda(m, [1.0], KT, nq=6, kernels="rust")
    first = cache["_rust"][1]
    scga.sublattice_sum_rule_residuals(m, [1.0], KT, 3.0, cache, mesh, kernels="rust")
    assert cache["_rust"][1] is first
    cache["bonds"] = list(cache["bonds"])           # a new bond list: repack
    scga.sublattice_sum_rule_residuals(m, [1.0], KT, 3.0, cache, mesh, kernels="rust")
    assert cache["_rust"][1] is not first


# --------------------------------------------------------------------------
# The refusal is the same refusal
# --------------------------------------------------------------------------
def test_the_inequivalent_sublattice_refusal_is_unchanged_under_rust(spy):
    m = _mixed_spin_chain()
    with pytest.raises(ValueError, match="PER SUBLATTICE"):
        scga.compute_scga(m, [], np.array([[0.1, 0, 0], [0.5, 0, 0]]), kT=2.0, nq=12,
                          kernels="rust")
    assert spy["eigenvalues"] == 1 and spy["sum_rule_residuals"] == 1
    assert spy["correlations"] == 0                  # refused before any S(q)
    lam, *_x, cache, mesh = scga.solve_lambda(m, [], 2.0, nq=12, kernels="rust",
                                              sublattice_tol=float("inf"))
    r = scga.sublattice_sum_rule_residuals(m, [], 2.0, lam, cache, mesh, kernels="rust")
    assert r == pytest.approx([2.625, -0.42], abs=1e-6)


# --------------------------------------------------------------------------
# From a config
# --------------------------------------------------------------------------
def _scga_config(kernels):
    t = np.radians(120.0)
    return {
        "crystal_structure": {
            "lattice_vectors": [[1., 0, 0], [float(np.cos(t)), float(np.sin(t)), 0],
                                [0, 0, 10.]],
            "atoms_uc": [{"label": "A", "pos": [0.5, 0, 0], "spin_S": 1.0},
                         {"label": "B", "pos": [0, 0.5, 0], "spin_S": 1.0},
                         {"label": "C", "pos": [0.5, 0.5, 0], "spin_S": 1.0}]},
        "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 0.5,
                                             "value": "J1"}]},
        "parameters": {"J1": 1.0}, "parameter_order": ["J1"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "calculation": {"cache_mode": "none", "kernels": kernels},
        "tasks": {"scga": True},
        "scga": {"temperature": KT, "mesh_density": 10, "cross_section": "perp",
                 "n_workers": 2},
        "q_path": {"G": [0.0, 0.0, 0.0], "K": [1 / 3, 1 / 3, 0.0], "M": [0.5, 0.0, 0.0],
                   "path": ["G", "K", "M"], "points_per_segment": 8},
        "plotting": {"enabled": False, "save_plot": False, "show_plot": False},
        "output": {"save_data": True, "scga_filename": "scga.npz"},
    }


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
    return np.load(d / "scga.npz")


def test_a_scga_config_runs_the_kernel_and_reproduces_numpy(tmp_path, caplog, spy):
    with caplog.at_level(logging.INFO):
        rust = _run_cfg(tmp_path, _scga_config("rust"), "rust")
    assert "kernels: rust (rsmagcalc" in caplog.text
    assert spy["eigenvalues"] == 1 and spy["correlations"] == 1
    assert "falling back" not in caplog.text
    numpy_ = _run_cfg(tmp_path, _scga_config("numpy"), "numpy")
    assert spy["correlations"] == 1                  # the numpy run did not call it
    assert float(rust["lam"]) == float(numpy_["lam"])
    assert np.allclose(rust["intensities"], numpy_["intensities"], rtol=1e-12, atol=0)
    assert np.array_equal(rust["q_vectors"], numpy_["q_vectors"])
