"""
Test-session guard: never open a GUI window.

Setting this here rather than relying on the caller exporting `MPLBACKEND=Agg`
is the whole point. The default backend on macOS is `macosx`, which is
INTERACTIVE, and `magcalc/plotting.py:show_plot_if_possible` calls `plt.show()`
on any non-Agg backend. Unattended, that blocks until a human closes the window:
the run wedges at ~0% CPU with no output and no timeout. A hang is far worse
than a failure -- it costs a whole gate (~14 min) and looks identical to a slow
test.

Seven shipped configs under `examples/` set `plotting: {show_plot: true}`
(FeI2, ZnCVO, aCVO x2, KFe3J, aRuCl3 x2), and they are reached from at least
seven test modules -- test_config_smoke, test_form_factor, test_zncvo,
test_gui_roundtrip, test_config_robustness, test_kfe3j_config, and magpipe's
test_excite.

`test_config_smoke.py` has always called `matplotlib.use("Agg", force=True)` at
import, but that only binds the ONE pytest-xdist worker that imports that
module. Both pytest.ini files run `-n auto --dist worksteal`, so which worker
draws which test is not fixed between runs -- which is exactly why the windows
appeared intermittently rather than every time.

Two mechanisms, because one is not enough:

* `os.environ["MPLBACKEND"]` is inherited by SUBPROCESSES. Tests spawn
  `magcalc run` (magpipe's `run_magcalc`, `test_fit_example`,
  `test_config_smoke`), and those child processes are where the `show_plot`
  configs actually execute. `matplotlib.use()` in this process cannot reach
  them. The two call sites that build a custom `env=` both start from
  `dict(os.environ)`, so they inherit it too.
* `matplotlib.use(..., force=True)` covers THIS process, including the case
  where something has already imported pyplot before conftest ran.

Escape hatch, mirroring `MAGCALC_SHADOW_GUARD=off`: set `MAGCALC_TEST_GUI=1` to
skip both and get your interactive backend back. Do not set it for a gate run.
"""
import os

if not os.environ.get("MAGCALC_TEST_GUI"):
    os.environ["MPLBACKEND"] = "Agg"
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
    except ImportError:      # matplotlib is not a hard dependency of every project
        pass

    # Backend-independent second line of defence. `show_plot_if_possible` is the
    # ONE place that can call `plt.show()`, and it honours this in any process
    # that inherits the environment -- including `magcalc run` subprocesses,
    # which is where the `show_plot` configs actually execute. The backend
    # setting above is necessary but not sufficient: it only protects processes
    # that ran this conftest, and a window opened by one that did not will hang
    # the entire session with no indication of which test is responsible.
    os.environ["MAGCALC_NO_GUI"] = "1"


def pytest_runtest_setup(item):
    """
    Re-assert the headless backend before every test.

    Necessary because a third-party import can UNDO it: `import pyalps` (ALPS's
    Python bindings, imported by `magpipe/tests/test_alps.py` and
    `test_thermo.py`) resets matplotlib's backend from Agg to the interactive
    `macosx`, overriding both `$MPLBACKEND` and an explicit
    `matplotlib.use("Agg", force=True)` that ran earlier. Setting the backend
    once at conftest import is therefore not durable.

    That is what made the windows INTERMITTENT rather than constant: under
    `-n auto --dist worksteal` only the workers that happened to import those
    modules were flipped, and only the config runs that landed in those workers
    afterwards opened a window -- so the same gate could be clean one run and
    hang the next.

    Restoring is the right response rather than failing: pyalps is a legitimate
    dependency doing something legitimate to its own process, and a test suite
    that refused to run because of it would help nobody. The backend-independent
    `MAGCALC_NO_GUI` guard in `magcalc.plotting.show_plot_if_possible` is the
    actual seatbelt; this keeps the backend itself sane so figures are still
    rendered offscreen rather than through a GUI toolkit.
    """
    if os.environ.get("MAGCALC_TEST_GUI"):
        return
    try:
        import matplotlib
    except ImportError:
        return
    if not matplotlib.get_backend().lower().startswith("agg"):
        matplotlib.use("Agg", force=True)
