"""K4 of RUST_KERNELS_PLAN.md: the KPM Chebyshev recursion in rsMagCalc, behind
`calculation: {kernels: rust}`.

The oracle tests (test_kpm: convergence to the exact `structure_factor`, the
integrated intensity, the chiral sign, the bound; test_kpm_stability: the guard at
every q, through the runner too; test_s09_disorder_kpm and test_bond_disorder: S09
with disorder) run under both kernel sets through the `kernels` fixture; the moments
parity (1e-12 at the same gamma, dense and CSR, on collinear, non-collinear,
mixed-spin and entangled models) lives in `rsMagCalc/tests/test_kpm_parity.py`.
What is pinned HERE is the wiring, quickly, outside `slow`:

  * every entry point -- `kpm_sqw` without and with a shared `hmat`,
    `kpm_sqw_batch`, `SUNModel.unstable_indices`, the runner's `kpm_sqw` task --
    reaches the kernel and answers what the NumPy loop answers, on a MIXED-spin
    supercell and on the shipped S09 supercell config;
  * an empty batch is an empty answer under both;
  * a kernel that raises (or returns a moment count that is not NumPy's) is logged
    and the NumPy loop answers -- the fMagCalc contract;
  * the guard still refuses a non-minimum under "rust", naming the q, BEFORE any
    moment is computed, and `on_imaginary: warn` still warns and computes.
"""
import copy
import logging
import os
import sys

import numpy as np
import pytest
import yaml

from magcalc import kernels as K
from magcalc.generic_model import GenericSpinModel
from magcalc.sun import kpm
from magcalc.sun.lswt import SUNModel

sys.path.insert(0, os.path.dirname(__file__))
from test_kernels_sun import QS, _model, _rust, _twin  # noqa: E402
from test_kpm_stability import _chain_cfg, _run  # noqa: E402

EGRID = np.linspace(0.0, 6.0, 61)
ROOT = os.path.join(os.path.dirname(__file__), "..", "examples", "sunny_tutorials")


def _rel(a, b):
    return float(np.abs(np.asarray(a) - np.asarray(b)).max() / np.abs(np.asarray(b)).max())


@pytest.fixture
def spy(monkeypatch):
    """Count calls into the K4 batch and the standalone recursion."""
    rs = _rust()
    calls = {"batch": 0, "single": 0}
    batch, single = rs.SunModel.kpm_moments, rs.kpm_moments

    def b(self, *a, **k):
        calls["batch"] += 1
        return batch(self, *a, **k)

    def s(*a, **k):
        calls["single"] += 1
        return single(*a, **k)

    monkeypatch.setattr(rs.SunModel, "kpm_moments", b)
    monkeypatch.setattr(rs, "kpm_moments", s)
    return calls


# --------------------------------------------------------------------------
# Every entry point reaches the kernel and agrees with NumPy
# --------------------------------------------------------------------------
def test_every_entry_point_reaches_the_kernel_and_agrees(spy, caplog):
    mdl = _model("rust")                     # mixed spin (1/2, 1), 2-cell supercell
    ref = _twin(mdl)
    q = QS[1]
    want = kpm.kpm_sqw(ref, q, EGRID, 0.3, tol=0.01)
    with caplog.at_level(logging.WARNING):
        got = kpm.kpm_sqw(mdl, q, EGRID, 0.3, tol=0.01)
        assert spy["batch"] == 1
        got_h = kpm.kpm_sqw(mdl, q, EGRID, 0.3, tol=0.01, hmat=mdl.hamiltonian(q))
        assert spy["single"] == 1
        batch = kpm.kpm_sqw_batch(mdl, QS, EGRID, 0.3, tol=0.01, n_workers=2)
        assert spy["batch"] == 2
    assert "falling back" not in caplog.text
    for r in (got, got_h, batch[1]):
        assert r.n_moments == want.n_moments
        assert abs(r.gamma - want.gamma) < 1e-12 * want.gamma
        assert _rel(r.intensities, want.intensities) < 1e-10
    for qk, r in zip(QS, batch):
        w = kpm.kpm_sqw(ref, qk, EGRID, 0.3, tol=0.01)
        assert _rel(r.intensities, w.intensities) < 1e-10
    # an explicit kernels= overrides the model's own choice, both ways
    assert spy["batch"] == 2
    kpm.kpm_sqw(ref, q, EGRID, 0.3, tol=0.01, kernels="rust")
    assert spy["batch"] == 3
    kpm.kpm_sqw(mdl, q, EGRID, 0.3, tol=0.01, kernels="numpy")
    assert spy["batch"] == 3


def test_unstable_indices_is_the_guard_verdict_per_q(kernels):
    """Every q, the same verdict as `is_stable_at`, and the closed form: on the
    frustrated chain at J1 = -1, J2 = 1/2 the curvature is proportional to
    (1 - cos k) - J2/|J1| (1 - cos 2k) = -(1 - cos k) cos k, NEGATIVE exactly for
    0 < h < 1/4 and zero at h = 0 and 1/4."""
    from test_kpm_stability import CHAIN_B, _chain_model
    m = _chain_model(J2=0.5, kernels=kernels)
    qs = np.array([[h, 0, 0] for h in np.linspace(0, 0.5, 11)], float) @ CHAIN_B
    bad = m.unstable_indices(qs)
    assert bad == [k for k, q in enumerate(qs) if not m.is_stable_at(q)]
    hs = np.linspace(0, 0.5, 11)
    assert bad == [k for k, h in enumerate(hs) if 0 < h < 0.25 - 1e-9] == [1, 2, 3, 4]
    assert _chain_model(J2=0.1, kernels=kernels).unstable_indices(qs) == []


def test_empty_batch(kernels):
    mdl = _model(kernels)
    assert kpm.kpm_sqw_batch(mdl, np.zeros((0, 3)), EGRID, 0.3) == []
    assert mdl.unstable_indices(np.zeros((0, 3))) == []


# --------------------------------------------------------------------------
# The fMagCalc contract: a failing kernel is logged, NumPy answers
# --------------------------------------------------------------------------
@pytest.mark.parametrize("where", ["batch", "single", "count"])
def test_a_failing_kernel_falls_back_to_numpy_and_says_so(where, monkeypatch, caplog):
    rs = _rust()
    mdl = _model("rust")
    ref = _twin(mdl)
    q = QS[2]
    real = rs.SunModel.kpm_moments

    def boom(*a, **k):
        raise RuntimeError("injected kernel failure")

    def wrong_count(self, *a, **k):
        g, nm, mom = real(self, *a, **k)
        return g, nm + 1, mom

    if where == "batch":
        monkeypatch.setattr(rs.SunModel, "kpm_moments", boom)
    elif where == "single":
        monkeypatch.setattr(rs, "kpm_moments", boom)
    else:
        monkeypatch.setattr(rs.SunModel, "kpm_moments", wrong_count)
    want = kpm.kpm_sqw(ref, q, EGRID, 0.3, tol=0.01)
    caplog.clear()
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        hm = mdl.hamiltonian(q) if where == "single" else None
        got = kpm.kpm_sqw(mdl, q, EGRID, 0.3, tol=0.01, hmat=hm)
        if where != "single":
            got_b = kpm.kpm_sqw_batch(mdl, QS, EGRID, 0.3, tol=0.01)[2]
            assert _rel(got_b.intensities, want.intensities) < 1e-10
    assert "falling back to the NumPy implementation" in caplog.text
    assert ("injected kernel failure" if where != "count" else "moment count") \
        in caplog.text
    assert got.n_moments == want.n_moments
    assert _rel(got.intensities, want.intensities) < 1e-10


# --------------------------------------------------------------------------
# The guard, through the runner, under "rust"
# --------------------------------------------------------------------------
def test_the_guard_refuses_a_non_minimum_before_any_moment(tmp_path, spy):
    msg = _run(tmp_path, _chain_cfg(kernels="rust"))
    assert msg is not None, "kpm_sqw ran about a non-minimum"
    assert "NOT a classical minimum" in msg and "KPM" in msg
    assert "q-point 2 of" in msg, msg              # Gamma (q-point 1) is stable
    assert spy["batch"] == 0, "the K4 batch ran before the guard refused"


def test_on_imaginary_warn_warns_and_computes_with_the_kernel(tmp_path, spy, caplog):
    cfg = _chain_cfg(on_imaginary="warn", kernels="rust")
    with caplog.at_level(logging.WARNING):
        assert _run(tmp_path, cfg) is None
    assert "NOT a classical minimum" in caplog.text and "of 9 q-points" in caplog.text
    assert spy["batch"] == 1


def _kpm_map(tmp_path, cfg, name):
    cfg = copy.deepcopy(cfg)
    cfg["output"] = {"save_data": True, "kpm_filename": f"{name}.npz"}
    assert _run(tmp_path, cfg, name=f"{name}.yaml") is None
    return np.load(os.path.join(str(tmp_path), f"{name}.npz"))["intensities"]


def test_the_runner_task_gives_the_numpy_map_under_rust(tmp_path, spy):
    cfg = _chain_cfg(J2=0.1)
    cfg["calculation"]["n_workers"] = 2
    a = _kpm_map(tmp_path, dict(cfg, calculation=dict(cfg["calculation"],
                                                      kernels="numpy")), "np")
    assert spy["batch"] == 0
    b = _kpm_map(tmp_path, dict(cfg, calculation=dict(cfg["calculation"],
                                                      kernels="rust")), "rs")
    assert spy["batch"] == 1
    assert a.shape == b.shape and _rel(b, a) < 1e-10


def test_the_shipped_S09_supercell_KPM_task_under_both_kernels(tmp_path, spy):
    """S09's `config_supercell.yaml` as shipped (minimization, then the KPM task over
    Gamma-K-M-Gamma, Gamma and K both Goldstone points of the non-collinear 120-degree
    state): the rust map equals the NumPy one."""
    cfg = yaml.safe_load(open(os.path.join(ROOT, "S09_triangular_AFM",
                                           "config_supercell.yaml")))
    cfg["tasks"] = {"minimization": True, "kpm_sqw": True}
    cfg["plotting"] = {"save_plot": False, "show_plot": False}
    cfg["q_path"]["points_per_segment"] = 6
    maps = {}
    for kern in ("numpy", "rust"):
        c = copy.deepcopy(cfg)
        c["calculation"]["kernels"] = kern
        maps[kern] = _kpm_map(tmp_path, c, kern)
    assert spy["batch"] == 1
    assert np.all(np.isfinite(maps["rust"])) and maps["numpy"].max() > 0
    assert _rel(maps["rust"], maps["numpy"]) < 1e-9
