"""K7: the dimer series' per-cluster perturbation theory in rsMagCalc, behind
`calculation: {kernels: rust}`.

The oracle tests (test_dimer_series: the PT engine against ED to O(lambda^(n+1)), the
chain's analytic first order, the linked-cluster pipeline against momentum-resolved
ED at J'/J = 0.4 and 0.8, Cu5SbO6 against PRR 8, 013247 Eq. (A11)) run under both
kernel sets through the `kernels` fixture. The kernel-level parity on random and
malformed inputs lives in `rsMagCalc/tests/test_dimer_series_parity.py`. What is
pinned HERE, quickly, outside `slow`:

  * `_cluster_pt` and the kernel agree to 1e-12 (relative to the largest entry) on
    the chain's clusters and on Rb2Cu3SnF12's, with the total-Sz sectors and
    without them;
  * the whole hopping series of Rb2Cu3SnF12 (6 dimers, out-of-plane DM, order 4)
    agrees to 1e-12, every hopping and the ground-state energy, and it is the
    kernel that computed it (spy);
  * a kernel that raises is logged ONCE and the NumPy PT answers the same series --
    the fMagCalc contract;
  * a level sent in chunks (`chunk_bytes`, 2026-10-31) gives the same series bit for
    bit as one call, each call's pool statistics are recorded in `kernel_stats`
    with the budget the model asked for, and an rsmagcalc built before the budget
    existed is still called the old way;
  * the subtraction in forked processes (`subtract_workers`, 2026-10-31) is the
    serial one bit for bit, a killed child falls back to it rather than hanging,
    every weight vanishes below its cluster's size (the linked-cluster theorem),
    and a level's failed additivity checks are one WARNING, counted exactly
    against clusters corrupted on purpose;
  * `mode: entangled` with `series_order` passes `calculation.kernels` and
    `n_workers` through to the series model, applies `series_resum`, and the
    dispersion it returns does not depend on the kernel set.
"""
import copy
import importlib.util
import logging
import os

import numpy as np
import pytest
import yaml

from magcalc import kernels as K
from magcalc.sun import dimer_series as ds

HERE = os.path.dirname(__file__)
EX = os.path.join(HERE, "..", "examples", "entangled")


def _rust():
    mod = K.rust_module()
    if mod is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    if not hasattr(mod, "cluster_pt_batch"):
        pytest.fail("rsmagcalc is installed but predates K7 (no cluster_pt_batch); "
                    "rebuild it")
    return mod


def _rb2cu3snf12():
    spec = importlib.util.spec_from_file_location(
        "rb_series", os.path.join(EX, "Rb2Cu3SnF12", "series_dispersion.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build_model()[0]


def _chain():
    I3 = np.eye(3)
    bonds = [(0, 1, (0, 0, 0), I3), (1, 0, (1, 0, 0), 0.6 * I3)]
    return ds.DimerSeriesModel.from_spin_arrays(
        np.diag([1.0, 10.0, 10.0]), [0.5, 0.5], [[0, 0, 0], [0.35, 0, 0]], bonds,
        units=[[0, 1]])


def _tasks(m, max_size, per_size):
    """(levels_list, links_local, sz_list) of the first topologies of each size."""
    out = []
    levels = m._enumerate(max_size)
    for size in range(1, max_size + 1):
        for form in sorted(levels[size])[:per_size]:
            _k, node_order = m._struct(form)
            lv, sz = m._node_levels(node_order)
            out.append((lv, m._local_links(form, node_order), sz))
    return out


def _close(a, b, tol=1e-12):
    scale = max(float(np.abs(a).max()), 1.0)
    return float(np.abs(np.asarray(a) - np.asarray(b)).max()) <= tol * scale


@pytest.fixture
def spy(monkeypatch):
    rs = _rust()
    calls = {"clusters": 0, "batches": 0}
    orig = rs.cluster_pt_batch

    def wrap(clusters, order, n_workers=None, **kw):
        clusters = list(clusters)
        calls["batches"] += 1
        calls["clusters"] += len(clusters)
        return orig(clusters, order, n_workers, **kw)
    monkeypatch.setattr(rs, "cluster_pt_batch", wrap)
    return calls


# --------------------------------------------------------------------------
# Cluster-level parity
# --------------------------------------------------------------------------
@pytest.mark.parametrize("model", ["chain", "Rb2Cu3SnF12"])
@pytest.mark.parametrize("sectors", [True, False])
def test_cluster_pt_matches_numpy(model, sectors):
    rs = _rust()
    m = _chain() if model == "chain" else _rb2cu3snf12()
    assert m.sz_levels is not None              # both conserve total Sz
    order = 5 if model == "chain" else 4
    tasks = _tasks(m, 4 if model == "chain" else 3, 6)
    if not sectors:
        tasks = [(lv, links, None) for (lv, links, _sz) in tasks]
    got = rs.cluster_pt_batch(tasks, order)
    assert len(got) == len(tasks)
    for (lv, links, sz), (g, d) in zip(tasks, got):
        g0, d0 = ds._cluster_pt(lv, links, order, sz)
        assert g.shape == g0.shape and d.shape == d0.shape
        assert _close(g, g0) and _close(d, d0)


# --------------------------------------------------------------------------
# The series end to end
# --------------------------------------------------------------------------
def test_rb2cu3snf12_hopping_series_matches_numpy(spy, caplog):
    order = 4
    mn = _rb2cu3snf12()
    mr = _rb2cu3snf12()
    mr.kernels = "rust"
    with caplog.at_level(logging.WARNING):
        tn = mn.hopping_series(order)
        tr = mr.hopping_series(order)
    assert spy["clusters"] > 0 and not mr._rust_failed
    assert not [r for r in caplog.records if "falling back" in r.getMessage()]
    assert set(tn) == set(tr)
    scale = max(np.abs(w).max() for w in tn.values())
    err = max(np.abs(tn[k] - tr[k]).max() for k in tn)
    assert err <= 1e-12 * scale, err
    assert _close(mn.ground_state_energy(order), mr.ground_state_energy(order))
    # one PT per topology, each computed by the kernel exactly once
    assert spy["clusters"] == len({k for (k, o) in mr._w_memo if o == order})


def test_kernel_failure_falls_back_to_numpy_once(monkeypatch, caplog):
    rs = _rust()
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("simulated kernel failure")
    monkeypatch.setattr(rs, "cluster_pt_batch", boom)
    order = 4
    ref = _chain().hopping_series(order)
    m = _chain()
    m.kernels = "rust"
    with caplog.at_level(logging.ERROR, logger="magcalc.kernels"):
        got = m.hopping_series(order)
    assert m._rust_failed and calls["n"] == 1
    msgs = [r.getMessage() for r in caplog.records if "falling back" in r.getMessage()]
    assert len(msgs) == 1 and "dimer-series cluster PT" in msgs[0]
    assert set(got) == set(ref)
    for k in ref:
        assert np.array_equal(got[k], ref[k])


# --------------------------------------------------------------------------
# Config wiring: mode entangled + series_order
# --------------------------------------------------------------------------
def test_entangled_series_passes_kernels_through(spy):
    from magcalc.generic_model import GenericSpinModel
    from magcalc.sun.entangled import EntangledCalculator

    cfg = yaml.safe_load(open(os.path.join(EX, "Cu5SbO6", "config.yaml")))
    cfg["calculation"].update(series_order=3, series_resum="sum", n_workers=2)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    L = np.array(cfg["crystal_structure"]["lattice_vectors"], float)
    qs = np.array([[0, 0, 0], [0.25, 0, 0], [0.3, 0, 0.7]]) @ (2 * np.pi * np.linalg.inv(L).T)
    out = {}
    for kern in ("numpy", "rust"):
        c = copy.deepcopy(cfg)
        calc = EntangledCalculator(GenericSpinModel(copy.deepcopy(c)), c, pv, kernels=kern)
        out[kern] = calc.calculate_dispersion(qs).energies
        assert calc._series.kernels == kern and calc._series.n_workers == 2
        # `series_resum: sum` reached the dispersion: the plain truncated sums
        plain = [np.sort(calc._series.dispersion(q, 3, "sum")[0]) for q in qs]
        assert np.allclose(np.sort(out[kern], axis=1), plain, rtol=0, atol=1e-12)
    assert spy["clusters"] > 0
    assert np.allclose(out["rust"], out["numpy"], rtol=0, atol=1e-10)


# --------------------------------------------------------------------------
# Chunks, the scratch budget and the recorded pool statistics (2026-10-31)
# --------------------------------------------------------------------------
def test_chunked_level_is_identical_and_records_its_pool(spy):
    order = 4
    ref = _rb2cu3snf12()
    tn = ref.hopping_series(order)                       # NumPy, the oracle
    one = _rb2cu3snf12()
    one.kernels = "rust"
    t1 = one.hopping_series(order)
    calls_one = spy["batches"]
    many = _rb2cu3snf12()
    many.kernels, many.chunk_bytes, many.n_workers = "rust", 1, 3
    many.scratch_budget_gb = 0.5
    tm = many.hopping_series(order)
    # chunk_bytes = 1: every topology in a call of its own
    n_topo = len({k for (k, o) in many._w_memo if o == order})
    assert spy["batches"] - calls_one == n_topo > calls_one
    assert set(tm) == set(t1) == set(tn)
    for k in t1:
        assert np.array_equal(tm[k], t1[k])
    scale = max(np.abs(w).max() for w in tn.values())
    assert max(np.abs(tn[k] - tm[k]).max() for k in tn) <= 1e-12 * scale
    st = many.kernel_stats
    assert len(st) == n_topo and all(s["n_clusters"] == 1 for s in st)
    assert all(s["budget_source"] == "argument" and s["budget_bytes"] == 2**29
               and s["threads"] == 1 and s["order"] == order for s in st)
    assert {s["size"] for s in st} == set(range(1, order + 1))


def test_an_rsmagcalc_without_the_budget_is_called_the_old_way(monkeypatch):
    rs = _rust()
    orig = rs.cluster_pt_batch
    monkeypatch.delattr(rs, "scratch_budget")
    monkeypatch.setattr(rs, "cluster_pt_batch",
                        lambda clusters, order, n_workers=None:
                        orig(clusters, order, n_workers))
    order = 4
    ref = _chain().hopping_series(order)
    m = _chain()
    m.kernels = "rust"
    got = m.hopping_series(order)
    assert not m._rust_failed and m.kernel_stats == []
    for k in ref:
        assert _close(got[k], ref[k])


# --------------------------------------------------------------------------
# The subtraction in forked processes, and one additivity line per level
# (2026-10-31)
# --------------------------------------------------------------------------
def test_forked_subtraction_is_the_serial_one_bit_for_bit(spy, monkeypatch):
    order = 4
    tn = _rb2cu3snf12().hopping_series(order)            # NumPy, serial: the oracle
    ser = _rb2cu3snf12()
    ser.kernels, ser.subtract_workers = "rust", 1
    ts = ser.hopping_series(order)
    monkeypatch.setattr(ds, "SUBTRACT_MIN_PER_PROC", 1)
    par = _rb2cu3snf12()
    par.kernels, par.subtract_workers = "rust", 3
    tp = par.hopping_series(order)
    # every level of more than one new topology went to the pool
    assert ser.subtract_stats == []
    assert [s["procs"] for s in par.subtract_stats] == [3] * len(par.subtract_stats)
    assert {s["size"] for s in par.subtract_stats} >= set(range(2, order + 1))
    assert set(tp) == set(ts) == set(tn)
    for k in ts:
        assert np.array_equal(tp[k], ts[k])
    assert np.array_equal(par.ground_state_energy(order), ser.ground_state_energy(order))
    assert set(par._w_memo) == set(ser._w_memo)
    scale = max(np.abs(w).max() for w in tn.values())
    assert max(np.abs(tn[k] - tp[k]).max() for k in tn) <= 1e-12 * scale
    # the linked-cluster theorem: every weight vanishes below its cluster's size
    assert par.additivity == [] and ser.additivity == []


@pytest.mark.parametrize("procs", [1, 3])
def test_additivity_failures_are_one_line_per_level(spy, monkeypatch, caplog, procs):
    """Corrupt order 0 -- below every cluster's size -- of each 3-link cluster's raw
    PT: exactly the level-3 topologies fail, and they are reported in one WARNING."""
    rs = _rust()
    orig = rs.cluster_pt_batch
    hit = {"n": 0}

    def corrupt(clusters, order, n_workers=None, **kw):
        clusters = list(clusters)
        res = orig(clusters, order, n_workers, **kw)
        out = res[0] if kw.get("return_stats") else res
        for (_lv, links, _sz), (_g, d) in zip(clusters, out):
            if len(links) == 3:
                d[0, 0, 0] += 1e-3
                hit["n"] += 1
        return res
    monkeypatch.setattr(rs, "cluster_pt_batch", corrupt)
    monkeypatch.setattr(ds, "SUBTRACT_MIN_PER_PROC", 1)
    order = 4
    m = _rb2cu3snf12()
    m.kernels, m.subtract_workers = "rust", procs
    with caplog.at_level(logging.DEBUG, logger="magcalc.sun.dimer_series"):
        m.hopping_series(order)
    assert hit["n"] > 1
    warn = [r.getMessage() for r in caplog.records
            if r.levelno == logging.WARNING and "additivity" in r.getMessage()]
    assert len(warn) == 1 and warn[0].startswith("  L=3:")
    assert f"in {hit['n']} of {hit['n']} new topologies" in warn[0]
    assert m.additivity == [dict(order=order, size=3, violated=hit["n"],
                                 topologies=hit["n"], worst=m.additivity[0]["worst"],
                                 worst_cluster=m.additivity[0]["worst_cluster"])]
    assert len(m.additivity[0]["worst_cluster"]) == 3
    # the per-cluster detail is still there, at DEBUG, from the parent either way
    dbg = [r for r in caplog.records
           if r.levelno == logging.DEBUG and "additivity violated for" in r.getMessage()]
    assert len(dbg) == hit["n"]
    assert bool(m.subtract_stats) == (procs > 1)


def test_a_killed_subtraction_child_falls_back_to_serial(spy, monkeypatch, caplog):
    """A child that dies (as under the OOM killer) must not hang the run: the
    executor raises, the level is subtracted serially, and the series is the
    serial one bit for bit."""
    order = 3
    ser = _rb2cu3snf12()
    ser.kernels, ser.subtract_workers = "rust", 1
    ts = ser.hopping_series(order)

    def die(i):
        os._exit(1)
    monkeypatch.setattr(ds, "_subtract_worker", die)
    monkeypatch.setattr(ds, "SUBTRACT_MIN_PER_PROC", 1)
    m = _rb2cu3snf12()
    m.kernels, m.subtract_workers = "rust", 2
    with caplog.at_level(logging.WARNING, logger="magcalc.sun.dimer_series"):
        tm = m.hopping_series(order)
    assert m.subtract_stats == []
    assert any("forked subtraction failed" in r.getMessage() for r in caplog.records)
    assert set(tm) == set(ts)
    for k in ts:
        assert np.array_equal(tm[k], ts[k])
