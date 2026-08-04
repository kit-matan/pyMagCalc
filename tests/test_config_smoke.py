"""Every shipped config must actually RUN (config-surface smoke test).

Prompted by the 2026-08-04 coverage audit: 11 of 69 documented config keys never
appeared in a test, and the recurring shape was not untested physics but untested
CONFIG PATHS to tested physics. `tasks.powder_average` is used by 9 shipped configs
while the tests call `powder_sample_modes` directly; `from_mcif` has a thorough
reader suite and nothing checking the key reaches it.

That gap is not theoretical. `sun_sampled_correlations` shipped referencing a variable
named `calc` where the runner calls it `calculator` -- invisible to every unit test,
caught by one manual run.

THE LOAD-BEARING DETAIL: the runner CATCHES task exceptions and logs them
(`logger.exception("... failed.")`, 33 such sites). So a config can complete
"successfully" with a task silently skipped -- which is exactly what the calc/calculator
bug did. Asserting "no exception escaped" would therefore have PASSED on it. This test
asserts on the LOG RECORDS instead.

Marked slow: it runs real calculations and belongs in the merge gate (`pytest -m ""`),
not the 4-minute iteration loop.
"""
import glob
import logging
import os

import pytest

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# Configs excluded from the smoke run, each with a reason. Keep this list SHORT and
# justified: every entry is a config nothing verifies end to end.
SKIP = {
    # minutes each: large supercells, long MC/dynamics runs, or high series orders.
    "S02_CoRh2O4_finiteT": "finite-T sampling, ~40 s+",
    "S04_FeI2_finiteT": "CP^(N-1) ground state + trajectories, ~2.5 min",
    "S05_Ising_MC": "8000 MC sweeps x 5 temperatures",
    "Rb2Cu3SnF12": "high-order dimer series (batch run)",
}


def _configs():
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "*", "*",
                                              "config*.yaml"))):
        name = os.path.basename(os.path.dirname(path))
        if any(k in path for k in SKIP):
            continue
        # `future_exmaples` is a staging area, not shipped material
        if "future_exmaples" in path:
            continue
        out.append(pytest.param(path, id=f"{name}/{os.path.basename(path)}"))
    return out


class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.ERROR)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.mark.slow
@pytest.mark.parametrize("config_path", _configs())
def test_shipped_config_runs_without_logging_an_error(config_path, tmp_path,
                                                      monkeypatch):
    """Run the config and require that NOTHING logged at ERROR level.

    Not "no exception": the runner swallows task failures into the log, so an
    exception check is satisfied by a task that never ran.
    """
    from magcalc import runner

    monkeypatch.chdir(tmp_path)          # keep plots/caches out of the repo
    handler = _Collector()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        runner.run_calculation(config_path)
    finally:
        root.removeHandler(handler)

    if handler.records:
        msgs = "\n".join(f"  [{r.name}] {r.getMessage()}" for r in handler.records[:5])
        pytest.fail(f"{os.path.relpath(config_path, ROOT)} logged "
                    f"{len(handler.records)} error(s):\n{msgs}")


def test_the_skip_list_stays_short_and_justified():
    """A skip list is a list of things nothing verifies. Keep it visible and small so
    it cannot quietly absorb every config that starts failing."""
    assert len(SKIP) <= 6, f"skip list has grown to {len(SKIP)}: {sorted(SKIP)}"
    assert all(reason for reason in SKIP.values()), "every skip needs a reason"


def test_smoke_covers_most_shipped_configs():
    """Guards the guard: if the discovery glob broke, the parametrization would
    silently shrink to nothing and every run would pass."""
    assert len(_configs()) >= 40, f"only {len(_configs())} configs discovered"
