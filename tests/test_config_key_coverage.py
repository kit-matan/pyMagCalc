"""Every config key the CODE reads must be exercised somewhere (OPEN_WORK item 5).

The 2026-08-04 coverage audit swept the DOCUMENTED keys against `tests/`. Its blind
spot was structural rather than accidental: `calculation.imaginary_rel_tolerance` was
in neither the docs nor the tests, so the process meant to find gaps could not see
it. A key nobody wrote down and nobody tested is precisely the one that rots.

`tests/config_keys.py` therefore enumerates from the SOURCE -- it parses the package
and records every `<block>.get("key")` -- and this test asserts the result is
covered, with an explicit, reasoned allow-list for the rest. Running it from the code
immediately paid for itself: it found `calculation.h2_rel_tolerance`, the KPM
ground-state guard's threshold added by OPEN_WORK item 10, prominently documented in
CLAUDE.md, read by `runner.py`, and named by NO test and NO shipped config -- the
same shape as `imaginary_rel_tolerance`, one item later. It is covered now
(`test_kpm_stability.py`), which is why it is not in the list below.

WHAT AN ENTRY IN `ALLOWED` MEANS. Not "this key is fine": "nothing verifies this key
reaches the code, and here is why we accept that today". Keep it short. Adding a key
here is a decision; the cheap alternative is usually one line in an existing test.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import config_keys  # noqa: E402

# block -> {key: reason}
ALLOWED = {
    "<top-level>": {
        # The AST heuristic attributes these to the whole config because they are
        # read off a local rebound from a nested block; they are magnetic_structure
        # / domains sub-keys, covered there.
        "cone_angle_deg": "magnetic_structure sub-key, mis-attributed by the scan",
        "explicit_list": "domains sub-key, mis-attributed by the scan",
    },
    "calculation": {
        "series_resum": "dimer-series resummation; its only config is in "
                        "test_config_smoke.SKIP (batch run). Exercised via "
                        "sun/dimer_series tests through the Python API, not the key.",
    },
    "fitting": {
        "fit_kws": "pass-through to lmfit.minimize; nothing here to verify beyond "
                   "'it is forwarded'.",
    },
    "minimization": {
        "initial_configuration": "legacy seed for the multistart path.",
    },
    "output": {
        "optimized_structure_filename": "output filename only.",
    },
    "plotting": {
        "fit_title": "cosmetic.",
        "plot_dir": "output directory; exercised implicitly by every run.",
        "sampled_correlations_plot_filename": "output filename only.",
        "scga_plot_filename": "output filename only.",
        "structure_plot_filename": "output filename only.",
    },
    "powder_average": {
        "q_magnitudes": "alias for the |Q| list; the shipped configs use `q_range`.",
    },
    "tasks": {
        "run_plotting": "legacy umbrella flag, superseded by the per-plot keys.",
        "run_powder_average": "legacy alias for `powder_average`.",
    },
    "wang_landau": {
        "flatness": "WL histogram flatness criterion; the algorithm is pinned on the "
                    "classical dimer, the knob only changes how long it runs.",
    },
}


def test_every_config_key_the_code_reads_is_exercised_or_explicitly_allowed():
    """The audit itself. New keys arrive covered, or arrive with a reason."""
    missing = config_keys.uncovered()
    unexpected = {}
    for block, keys in missing.items():
        extra = [k for k in keys if k not in ALLOWED.get(block, {})]
        if extra:
            unexpected[block] = extra
    assert not unexpected, (
        "config keys read by the code but named in no config and no test:\n"
        + "\n".join(f"  {b}: {', '.join(k)}" for b, k in sorted(unexpected.items()))
        + "\n\nAdd a test or a config that sets them, or an entry in ALLOWED "
          "with the reason.")


def test_the_allow_list_has_not_gone_stale():
    """An ALLOWED entry for a key that IS now covered, or that the code no longer
    reads, is a lie about the coverage. Fail on it rather than let it accumulate."""
    missing = config_keys.uncovered()
    stale = [f"{block}.{key}" for block, keys in ALLOWED.items()
             for key in keys if key not in missing.get(block, [])]
    assert not stale, (f"ALLOWED lists keys that are covered now (or gone): "
                       f"{stale}. Delete them.")


def test_the_scan_actually_finds_the_config_surface():
    """Guards the guard: if the AST walk broke, it would find nothing and the audit
    above would pass vacuously. The blocks named here are the ones whose keys the
    tutorial documents, so a scan that misses them is broken, not merely thin."""
    found = config_keys.code_keys()
    for block in ("calculation", "tasks", "plotting", "minimization", "scga",
                  "thermal_mc", "sampled_correlations", "kpm", "fitting"):
        assert found.get(block), f"no keys discovered for `{block}:`"
    assert sum(len(v) for v in found.values()) > 150


def test_the_key_that_motivated_this_is_covered():
    """`calculation.h2_rel_tolerance` -- guard 3's threshold -- was the audit's first
    catch. Pinned by name so it cannot silently fall out of coverage again."""
    assert "h2_rel_tolerance" in config_keys.code_keys()["calculation"]
    assert "h2_rel_tolerance" in config_keys.exercised_keys()
