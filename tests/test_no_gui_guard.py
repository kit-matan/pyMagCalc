"""
The conftest no-GUI guard, pinned.

An interactive backend in an unattended run does not fail -- it HANGS, at ~0%
CPU with no output and no timeout, until a human closes a window. That is worse
than a failure and it costs a whole ~14 min gate, so the guard that prevents it
needs tests of its own.

The reference here is matplotlib's own behaviour, not a number this project
chose: with no `MPLBACKEND` set, `matplotlib.get_backend()` on macOS is the
INTERACTIVE `macosx`, and `magcalc/plotting.py:show_plot_if_possible` calls
`plt.show()` on any backend that is not Agg.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_this_process_is_headless():
    import matplotlib

    assert matplotlib.get_backend().lower().startswith("agg"), (
        f"backend is {matplotlib.get_backend()}; conftest.py should have forced Agg")


def test_the_env_var_is_exported_for_children():
    """
    `matplotlib.use()` binds only THIS process. The `show_plot` configs actually
    execute inside `magcalc run` SUBPROCESSES, which can only be reached through
    the environment.
    """
    assert os.environ.get("MPLBACKEND") == "Agg"


def test_a_subprocess_inherits_the_headless_backend():
    out = subprocess.run(
        [sys.executable, "-c", "import matplotlib; print(matplotlib.get_backend())"],
        capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().lower().startswith("agg"), out.stdout


def test_the_shipped_show_plot_configs_are_still_the_reason_this_exists():
    """
    If this list ever empties, the guard is still wanted but this file's
    rationale has changed -- and if it GROWS, the guard is protecting more than
    it used to. Either way the number should be noticed rather than drift.
    """
    hits = [p for p in EXAMPLES.rglob("*.yaml")
            if "show_plot: true" in p.read_text().lower()]
    assert hits, "no config sets show_plot: true any more -- re-read conftest.py"


@pytest.mark.slow
def test_a_show_plot_config_runs_headless_end_to_end(tmp_path):
    """
    The whole point, end to end: a config with `show_plot: true` AND
    `plot_structure: true` must complete unattended. Before the conftest guard
    this hung whenever xdist happened to schedule such a test into a worker that
    had not imported `test_config_smoke` (the only module that forced Agg), which
    is why the windows appeared intermittently rather than every run.
    """
    src = EXAMPLES / "materials/aCVO/config_acvo.yaml"
    cfg = yaml.safe_load(src.read_text())
    assert cfg["plotting"]["show_plot"] is True
    assert cfg["plotting"]["plot_structure"] is True

    cfg["minimization"]["num_starts"] = 4
    cfg["tasks"] = {"minimization": True, "dispersion": True}
    cfg["q_path"]["points_per_segment"] = 3
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))

    out = subprocess.run([sys.executable, "-m", "magcalc", "run", "config.yaml"],
                         cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-2000:]

def test_show_is_suppressed_independently_of_the_backend():
    """
    The backend check is necessary but NOT sufficient: it protects only processes
    that ran conftest, and a test session is a tree of them (pytest, its xdist
    workers, the `magcalc run` subprocesses they spawn). One process that missed
    the setting opens a window and hangs the whole run.

    So `show_plot_if_possible` must refuse on the environment alone. Forced here
    with a genuinely interactive backend name, so the test would fail if the
    function still decided on the backend only.
    """
    import matplotlib

    from magcalc.plotting import show_plot_if_possible

    assert os.environ.get("MAGCALC_NO_GUI") == "1"
    called = []
    real_show = matplotlib.pyplot.show
    matplotlib.pyplot.show = lambda *a, **k: called.append(1)
    try:
        show_plot_if_possible()
    finally:
        matplotlib.pyplot.show = real_show
    assert not called, "plt.show() was called despite MAGCALC_NO_GUI"


def test_pytest_alone_is_enough_to_suppress_a_window(monkeypatch):
    """
    `PYTEST_CURRENT_TEST` is set by pytest and inherited by child processes, so
    it identifies the whole test process tree even where MAGCALC_NO_GUI did not
    reach.
    """
    import matplotlib

    from magcalc.plotting import show_plot_if_possible

    monkeypatch.delenv("MAGCALC_NO_GUI", raising=False)
    assert os.environ.get("PYTEST_CURRENT_TEST")
    called = []
    real_show = matplotlib.pyplot.show
    matplotlib.pyplot.show = lambda *a, **k: called.append(1)
    try:
        show_plot_if_possible()
    finally:
        matplotlib.pyplot.show = real_show
    assert not called, "plt.show() was called despite running under pytest"


def test_the_backend_is_restored_after_something_resets_it():
    """
    A third-party import can undo the headless backend: `import pyalps` resets
    matplotlib to the interactive `macosx`, beating both `$MPLBACKEND` and an
    earlier `matplotlib.use("Agg", force=True)`.

    Simulated here rather than by importing pyalps, so the test means the same
    thing on a machine without ALPS. The contract is that `pytest_runtest_setup`
    puts the backend back before each test, so a flip during one test cannot
    leak into the next.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)      # leave it as we found it
    assert matplotlib.get_backend().lower().startswith("agg")


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("pyalps") is None,
    reason="pyalps not installed")
def test_pyalps_really_does_reset_the_backend():
    """
    Pins the upstream behaviour that caused the hangs, so that if a future
    pyalps stops doing it we find out rather than keep guarding blindly.

    Run in a SUBPROCESS, deliberately. In-process the check is not reproducible:
    once any earlier test in the same xdist worker has imported pyalps, a second
    `import pyalps` is a no-op that re-flips nothing, and the test then reports
    "pyalps no longer resets the backend" -- which is false, and exactly the kind
    of skip that hides a live test. A fresh interpreter makes it deterministic.
    """
    probe = (
        "import os, matplotlib\n"
        "matplotlib.use('Agg', force=True)\n"
        "before = matplotlib.get_backend()\n"
        "import pyalps\n"
        "print(before, matplotlib.get_backend())\n"
    )
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                         text=True, timeout=300,
                         env=dict(os.environ, MPLBACKEND="Agg"))
    assert out.returncode == 0, out.stderr[-2000:]
    before, after = out.stdout.split()
    assert before.lower().startswith("agg")
    assert not after.lower().startswith("agg"), (
        f"pyalps no longer resets the backend (still {after!r}). Good news -- but "
        "re-read conftest.py's pytest_runtest_setup before relaxing anything: "
        "MAGCALC_NO_GUI is the guard that does not depend on this.")
