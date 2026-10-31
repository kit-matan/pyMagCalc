"""Save and resume for the dimer series (OPEN_WORK item 10, 2026-10-31).

`DimerSeriesModel.checkpoint_dir` writes the memo of subtracted cluster weights to
disk chunk by chunk, so an order that outlives one job's wall time continues in the
next. The oracle is an identity: the memo is exact, so a run that is killed and
resumed must give the uninterrupted run's hoppings BIT FOR BIT, under both kernel
sets. Pinned here on Rb2Cu3SnF12 with its out-of-plane DM (the model the order-9
LANTA run is for):

  * fast, order 4: a crash in the middle of a chunk of the last level, and a clean
    stop by a real SIGUSR1 (exit status 75, the handlers restored), both resumed
    from a fresh model to the uninterrupted hoppings and ground-state energy;
  * fast: a checkpoint of another model (one DM sign flipped), another kernel set,
    a foreign or corrupt chunk file, and a pickle that names a global are all
    refused with the reason; a temporary file left by a killed write is ignored;
    one directory holds several orders side by side;
  * fast: off by default (no files, no signal handlers touched), on through
    $MAGCALC_SERIES_CHECKPOINT and through `calculation.series_checkpoint`;
  * slow, order 6: a separate process SIGKILLed after a few chunks of L = 6 --
    nothing flushed on the way out -- and the run resumed here.
"""
import copy
import importlib.util
import os
import pickle
import signal
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import yaml

from magcalc.sun import dimer_series as ds
from magcalc.sun import series_checkpoint as sc

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EX = os.path.join(ROOT, "examples", "entangled")
RB = os.path.join(EX, "Rb2Cu3SnF12", "series_dispersion.py")


def _rb(kernels="numpy"):
    spec = importlib.util.spec_from_file_location("rb_series_ckpt", RB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.build_model(kernels=kernels)[0]
    assert m.sz_levels is not None
    assert any(np.abs(V.imag).max() > 0 for (_u, _v, _o, V) in m.link_types)  # DM on
    return m


def _model(kernels, n_workers=1, chunk_bytes=1 << 20):
    m = _rb(kernels)
    m.n_workers = n_workers
    m.chunk_bytes = chunk_bytes          # many chunks per level
    return m


_REF = {}


def _reference(kernels, order):
    """The uninterrupted hoppings and ground-state energy, checkpoint off."""
    if (kernels, order) not in _REF:
        m = _model(kernels)
        assert m.checkpoint_dir is None
        _REF[(kernels, order)] = (m.hopping_series(order), m.ground_state_energy(order))
    return _REF[(kernels, order)]


def _assert_identical(m, kernels, order):
    t_ref, gs_ref = _reference(kernels, order)
    t = m.hopping_series(order)
    assert set(t) == set(t_ref)
    for k in t_ref:
        assert np.array_equal(t[k], t_ref[k]), k
    assert np.array_equal(m.ground_state_energy(order), gs_ref)


def _topologies(m, order):
    return len({k for (k, o) in m._w_memo if o == order})


def _no_env(monkeypatch):
    monkeypatch.delenv(sc.ENV, raising=False)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    _no_env(monkeypatch)


# --------------------------------------------------------------------------
# Kill and resume, bit for bit
# --------------------------------------------------------------------------
def test_crash_mid_chunk_resumes_bit_identically(kernels, tmp_path, monkeypatch):
    order = 4
    m = _model(kernels)
    m.checkpoint_dir = str(tmp_path)
    stores = {"top": 0}
    orig = ds.DimerSeriesModel._store

    def crashing_store(self, memo_key, form, *a):
        orig(self, memo_key, form, *a)
        if len(form) == order:
            stores["top"] += 1
            if stores["top"] == 150:          # chunks hold ~60 L=4 topologies
                raise RuntimeError("simulated crash")
    monkeypatch.setattr(ds.DimerSeriesModel, "_store", crashing_store)
    with pytest.raises(RuntimeError, match="simulated crash"):
        m.hopping_series(order)
    monkeypatch.setattr(ds.DimerSeriesModel, "_store", orig)
    files = sorted(os.listdir(tmp_path / f"order-{order}"))
    assert "manifest.json" in files and len([f for f in files if f.endswith(".pkl")]) > 2
    assert m._ckpt is None                    # released on the way out

    m2 = _model(kernels)
    m2.checkpoint_dir = str(tmp_path)
    _assert_identical(m2, kernels, order)
    assert sum(n for (o, s), n in m2._resumed.items() if o == order and s == order) == 150
    assert sum(m2._resumed.values()) < _topologies(m2, order)


def test_stop_signal_saves_and_exits_75(kernels, tmp_path, monkeypatch):
    import threading
    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signal handlers need the main thread")
    order = 4
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    m = _model(kernels)
    m.checkpoint_dir = str(tmp_path)
    calls = {"top": 0}
    orig = ds.DimerSeriesModel._subtract_level

    def signalled(self, items, order_):
        orig(self, items, order_)
        if items and len(items[0][0]) == order:
            calls["top"] += 1
            if calls["top"] == 3:             # mid-level, between two chunks
                os.kill(os.getpid(), signal.SIGUSR1)
    monkeypatch.setattr(ds.DimerSeriesModel, "_subtract_level", signalled)
    with pytest.raises(sc.SeriesInterrupted) as ei:
        m.hopping_series(order)
    e = ei.value
    assert isinstance(e, SystemExit) and e.code == sc.EXIT_INTERRUPTED == 75
    assert e.signum == signal.SIGUSR1 and e.order == order and "SIGUSR1" in str(e)
    assert calls["top"] == 3                  # the chunk in flight finished, no more
    assert {s: signal.getsignal(s) for s in before} == before
    monkeypatch.setattr(ds.DimerSeriesModel, "_subtract_level", orig)

    m2 = _model(kernels)
    m2.checkpoint_dir = str(tmp_path)
    _assert_identical(m2, kernels, order)
    assert m2._resumed[(order, order)] > 0


# --------------------------------------------------------------------------
# Refusals: never reuse a checkpoint that is not this calculation's
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """A complete order-3 checkpoint of the numpy model."""
    d = tmp_path_factory.mktemp("ckpt")
    m = _model("numpy")
    m.checkpoint_dir = str(d)
    m.hopping_series(3)
    return d


def test_other_model_is_refused(saved):
    m = _model("numpy")
    u, v, off, V = m.link_types[0]
    m.link_types[0] = (u, v, off, V.conj())   # flips that link's DM
    m.checkpoint_dir = str(saved)
    with pytest.raises(sc.CheckpointMismatch, match="links") as ei:
        m.hopping_series(3)
    assert "Refusing" in str(ei.value)
    assert not m._w_memo                      # nothing was loaded


def test_other_kernel_set_is_refused(saved):
    m = _model("rust")
    m.checkpoint_dir = str(saved)
    with pytest.raises(sc.CheckpointMismatch, match="kernels"):
        m.hopping_series(3)


def test_other_sz_setting_is_refused(saved):
    m = _model("numpy")
    m.sz_levels = None                        # the PT without Sz sectors
    m.checkpoint_dir = str(saved)
    with pytest.raises(sc.CheckpointMismatch, match="sz_sectors"):
        m.hopping_series(3)


def test_each_order_has_its_own_directory(saved):
    m = _model("numpy")
    m.checkpoint_dir = str(saved)
    m.hopping_series(2)
    assert {"order-2", "order-3"} <= set(os.listdir(saved))
    m3 = _model("numpy")
    m3.checkpoint_dir = str(saved)
    _assert_identical(m3, "numpy", 3)
    assert sum(m3._resumed.values()) == _topologies(m3, 3)   # nothing recomputed


def test_foreign_or_corrupt_files_are_refused(saved, tmp_path):
    import shutil
    d = tmp_path / "c"
    shutil.copytree(saved, d)
    od = d / "order-3"
    first = sorted(p for p in os.listdir(od) if p.endswith(".pkl"))[0]

    # a chunk file from another calculation
    with open(od / first, "rb") as fh:
        fmt, _fp, entries = sc._NoGlobals(fh).load()
    with open(od / "w-999998.pkl", "wb") as fh:
        pickle.dump((fmt, "0" * 32, entries), fh, protocol=5)
    m = _model("numpy")
    m.checkpoint_dir = str(d)
    with pytest.raises(sc.CheckpointMismatch, match="different calculation"):
        m.hopping_series(3)
    os.remove(od / "w-999998.pkl")

    # a pickle that would build an object (here: call os.getcwd) is not unpickled
    class Evil:
        def __reduce__(self):
            return (os.getcwd, ())
    with open(od / "w-999999.pkl", "wb") as fh:
        pickle.dump((fmt, "x", Evil()), fh, protocol=5)
    with pytest.raises(sc.CheckpointMismatch, match="cannot be read.*holds no objects"):
        m.hopping_series(3)
    os.remove(od / "w-999999.pkl")

    # a write killed before its rename leaves a temporary file: ignored, removed
    (od / f".w-000123.pkl.{os.getpid()}.tmp").write_bytes(b"\x80half a pickle")
    _assert_identical(m, "numpy", 3)
    assert not [p for p in os.listdir(od) if p.endswith(".tmp")]


def test_a_directory_in_use_is_refused(saved, tmp_path):
    import shutil
    d = tmp_path / "busy"
    shutil.copytree(saved, d)
    m = _model("numpy")
    held = sc.SeriesCheckpoint(str(d), m, 3, m._enumerate(3))
    try:
        m2 = _model("numpy")
        m2.checkpoint_dir = str(d)
        with pytest.raises(RuntimeError, match="in use"):
            m2.hopping_series(3)
    finally:
        held.close()


# --------------------------------------------------------------------------
# The switch: off by default, the environment, the config key
# --------------------------------------------------------------------------
def test_off_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGUSR1)}
    m = _model("numpy")
    assert m.checkpoint_dir is None
    t = m.hopping_series(3)
    assert t and not os.listdir(tmp_path)
    assert {s: signal.getsignal(s) for s in before} == before


def test_environment_variable_turns_it_on(tmp_path, monkeypatch):
    monkeypatch.setenv(sc.ENV, str(tmp_path / "env"))
    m = _model("numpy")
    assert m.checkpoint_dir == str(tmp_path / "env")
    m.hopping_series(2)
    assert os.path.exists(tmp_path / "env" / "order-2" / "manifest.json")


def test_config_key_series_checkpoint(tmp_path):
    from magcalc.generic_model import GenericSpinModel
    from magcalc.sun.entangled import EntangledCalculator

    cfg = yaml.safe_load(open(os.path.join(EX, "Cu5SbO6", "config.yaml")))
    cfg["calculation"].update(series_order=3, series_resum="sum",
                              series_checkpoint=str(tmp_path / "cfg"))
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    calc = EntangledCalculator(GenericSpinModel(copy.deepcopy(cfg)), cfg, pv)
    e = calc.calculate_dispersion(np.zeros((1, 3))).energies
    assert np.isfinite(e).all()
    assert calc._series.checkpoint_dir == str(tmp_path / "cfg")
    assert os.path.exists(tmp_path / "cfg" / "order-3" / "manifest.json")


# --------------------------------------------------------------------------
# Slow: a separate process SIGKILLed mid-level at order 6, resumed here
# --------------------------------------------------------------------------
_KILLER = textwrap.dedent("""
    import importlib.util, os, signal, sys
    from magcalc.sun import dimer_series as ds


    def main():
        ckdir, kernels = sys.argv[1], sys.argv[2]
        order, after = int(sys.argv[3]), int(sys.argv[4])
        spec = importlib.util.spec_from_file_location("rb", {rb!r})
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        m = mod.build_model(kernels=kernels)[0]
        m.n_workers, m.chunk_bytes, m.checkpoint_dir = {n_workers}, {chunk}, ckdir
        top = {{"n": 0}}
        store, flush = ds.DimerSeriesModel._store, ds.DimerSeriesModel._checkpoint_flush

        def counting_store(self, memo_key, form, *a):
            store(self, memo_key, form, *a)
            top["n"] += len(form) == order

        def killing_flush(self):
            flush(self)
            if top["n"] >= after:      # some chunks into the last level: no goodbye
                os.kill(os.getpid(), signal.SIGKILL)
        ds.DimerSeriesModel._store = counting_store
        ds.DimerSeriesModel._checkpoint_flush = killing_flush
        m.hopping_series(order)


    if __name__ == "__main__":         # the NumPy pool spawns workers that import this
        main()
""")


@pytest.mark.slow
def test_sigkill_mid_level_order6_resumes_bit_identically(kernels, tmp_path):
    order, n_workers, chunk = 6, 4, 32 << 20
    ref_m = _rb(kernels)
    ref_m.n_workers, ref_m.chunk_bytes = n_workers, chunk
    t_ref = ref_m.hopping_series(order)
    gs_ref = ref_m.ground_state_energy(order)
    n_top = _topologies(ref_m, order)
    script = tmp_path / "killer.py"
    script.write_text(_KILLER.format(rb=RB, n_workers=n_workers, chunk=chunk))
    env = dict(os.environ, MAGCALC_SHADOW_GUARD="off", MAGCALC_NO_GUI="1",
               PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    env.pop(sc.ENV, None)
    p = subprocess.run([sys.executable, str(script), str(tmp_path / "ck"), kernels,
                        str(order), str(n_top // 3)], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=3600)
    assert p.returncode == -signal.SIGKILL, p.stderr[-3000:]

    m = _rb(kernels)
    m.n_workers, m.chunk_bytes, m.checkpoint_dir = n_workers, chunk, str(tmp_path / "ck")
    t = m.hopping_series(order)
    resumed_top = m._resumed[(order, order)]
    assert 0 < resumed_top < n_top            # killed in the middle of L = 6
    assert set(t) == set(t_ref)
    for k in t_ref:
        assert np.array_equal(t[k], t_ref[k]), k
    assert np.array_equal(m.ground_state_energy(order), gs_ref)
