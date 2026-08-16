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

import yaml

import matplotlib
import pytest

# Force a headless backend BEFORE anything imports pyplot. Seven shipped configs
# set `plotting: {show_plot: true}`, and `plt.show()` on an interactive backend
# blocks until a human closes the window -- which in an unattended run is
# forever. It happens not to bite under pytest today, but that is an accident of
# the environment, not a guarantee: running one of these configs directly on
# macOS wedges at ~0% CPU with no output and no timeout. A hang is worse than a
# failure, so pin it here instead of relying on the accident.
matplotlib.use("Agg", force=True)

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
    """Every runnable config under `examples/`, found BY CONTENT.

    Discovery used to be a two-level glob (`examples/*/*/config*.yaml`) plus a
    hand-maintained EXTRA list, and that shape is itself a defect: a config is
    invisible unless someone remembers to name it `config*.yaml` AND to put it
    exactly two directories deep. `examples/fitting/fit_dispersion.yaml` failed both
    tests, which is how it went on shipping -- as TUTORIAL.md's own `magcalc fit`
    example -- with every bond listed in one direction only (halving each J) and no
    `magnetic_structure` at all (expanding about a stationary maximum), while its
    "recovers the true values" check passed because the shipped data had been
    generated from that same broken model. Adding it to a list fixed that ONE file
    and left the shape.

    So the criterion is now what a config IS, not what it is called: any *.yaml at
    any depth that parses to a mapping carrying `crystal_structure` or `from_mcif`.
    The three `*_fit_params.yaml` outputs (`best_fit_parameters:` only) are excluded
    by that same test, with no name-based special case. Cross-checked when it landed:
    it discovers EXACTLY the same 58 configs the glob-plus-list did, so it is a
    change of shape and not of coverage.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "**", "*.yaml"),
                                 recursive=True)):
        if any(k in path for k in SKIP):
            # `future_exmaples` used to be skipped here too, as "a staging area, not
            # shipped material". That is exactly how its FeI2 config came to assert a
            # spiral 2.5 meV/site above the ground state -- nothing ran it. Staging is
            # a reason to expect churn, not a reason to skip.
            continue
        try:
            with open(path) as handle:
                doc = yaml.safe_load(handle)
        except Exception:
            # A config that does not parse is a failure, not something to skip; give
            # it to the runner and let the test report it.
            doc = {"crystal_structure": None}
        if not isinstance(doc, dict):
            continue
        if "crystal_structure" not in doc and "from_mcif" not in doc:
            continue
        rel = os.path.relpath(path, os.path.join(ROOT, "examples"))
        out.append(pytest.param(path, id=rel))
    return out


# WARNINGS THAT A SHIPPED CONFIG IS ALLOWED TO EMIT.
#
# The smoke test failed on ERROR records only, and OPEN_WORK item 5 has wanted the
# warnings escalated for a while. The blocker was never the mechanism, it was the
# PREREQUISITE: escalating is only viable once benign warnings stop firing on
# correctly-written configs, and two that did are gone (2026-08-12) -- `num_starts <
# early_stopping` for the Monte-Carlo methods, where `early_stopping` is meaningless,
# and `plt.show()` on a non-interactive backend.
#
# Each entry is a SUBSTRING matched against the message, with a reason. An entry means
# "this warning is expected on a correct config", not "this warning is unimportant" --
# most of these are the engine correctly telling the user something about the physics.
# Anything not listed FAILS the config, which is the point: a new warning on a shipped
# config is either a defect or a documentation gap, and both deserve a look.
#
# To see what a run actually emits (e.g. after adding a config), set
# MAGCALC_SMOKE_HARVEST=<path> and the test appends every warning to that file
# instead of asserting.
ALLOWED_WARNINGS = {
    "single-ion (multipolar) excitations":
        "The dipole-mode advisory (CLAUDE.md 5c). It is CORRECT on the 9 configs that "
        "raise it -- S >= 1 with an anisotropy, run in dipole mode on purpose (the "
        "SU(N) versions live next door, e.g. FeI2's config_fei2_sun.yaml). Telling "
        "the user which bands are absent is the warning's job.",
    "is NOT a classical energy minimum":
        "The energy-audit guard, downgraded by an explicit `on_imaginary: warn` in "
        "the 3 configs that raise it (SW23, ZnCVO, aRuCl3) -- each with a comment "
        "saying why. The guard firing is the config working as written; a config "
        "that raised this WITHOUT `on_imaginary: warn` would have failed as an "
        "ERROR long before reaching here.",
    "Magnon energies are IMAGINARY":
        "Same three configs plus SW03 (its commensurate approximation to an "
        "incommensurate spiral) and SW18. All carry `on_imaginary: warn` "
        "deliberately; CLAUDE.md requires a comment saying why, and they have one.",
}

HARVEST = os.environ.get("MAGCALC_SMOKE_HARVEST")


class _Collector(logging.Handler):
    """Collects ERROR records always, and WARNING records for the escalation."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.records = []

    def emit(self, record):
        self.records.append(record)

    @property
    def errors(self):
        return [r for r in self.records if r.levelno >= logging.ERROR]

    @property
    def warnings(self):
        return [r for r in self.records if r.levelno == logging.WARNING]


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

    rel = os.path.relpath(config_path, ROOT)
    if handler.errors:
        msgs = "\n".join(f"  [{r.name}] {r.getMessage()}" for r in handler.errors[:5])
        pytest.fail(f"{rel} logged {len(handler.errors)} error(s):\n{msgs}")

    unexpected = [r for r in handler.warnings
                  if not any(frag in r.getMessage() for frag in ALLOWED_WARNINGS)]
    if HARVEST:
        # Harvest mode: record instead of asserting, so the allow-list can be built
        # from a real run rather than guessed. Deliberately not the default -- a test
        # that only ever records is not a test.
        with open(HARVEST, "a") as fh:
            for r in handler.warnings:
                fh.write(f"{rel}\t{r.name}\t{r.getMessage()}\n")
        return
    if unexpected:
        msgs = "\n".join(f"  [{r.name}] {r.getMessage()}" for r in unexpected[:5])
        pytest.fail(
            f"{rel} logged {len(unexpected)} unexpected warning(s):\n{msgs}\n\n"
            f"If the warning is correct and expected for this config, add a substring "
            f"of it to ALLOWED_WARNINGS with a reason. If it is not, that is the bug.")


def test_the_skip_list_stays_short_and_justified():
    """A skip list is a list of things nothing verifies. Keep it visible and small so
    it cannot quietly absorb every config that starts failing."""
    assert len(SKIP) <= 6, f"skip list has grown to {len(SKIP)}: {sorted(SKIP)}"
    assert all(reason for reason in SKIP.values()), "every skip needs a reason"


def test_smoke_covers_most_shipped_configs():
    """Guards the guard: if the discovery glob broke, the parametrization would
    silently shrink to nothing and every run would pass."""
    assert len(_configs()) >= 40, f"only {len(_configs())} configs discovered"


# --------------------------------------------------------------------------
# The deprecated `type: spiral` spelling, and why migrating it was safe
# --------------------------------------------------------------------------
def test_spiral_and_single_k_normalize_identically():
    """`type: spiral` is a pure alias for `single_k`, so migrating the seven shipped
    configs that used it is a no-op BY CONSTRUCTION -- which is the only reason it
    could be done without re-verifying seven spectra band by band.

    `normalize_magnetic_structure` rewrites `cfg['type']` and touches nothing else on
    that branch; this asserts the resulting dicts are equal, which is the exact
    identity that claim rests on.
    """
    from magcalc.generic_model import normalize_magnetic_structure

    base = {"k": [-1 / 3, -1 / 3, 0.0], "axis": [0, 0, 1],
            "local_directions": [[0.5, 0.8660254, 0], [0.5, 0.8660254, 0],
                                 [-0.5, -0.8660254, 0]], "enabled": True}
    a = normalize_magnetic_structure(dict(base, type="spiral"), quiet=True)
    b = normalize_magnetic_structure(dict(base, type="single_k"), quiet=True)
    assert a == b


def test_no_shipped_config_still_uses_a_deprecated_structure_type():
    """The escalation in `test_config_smoke.py` cannot police this one, and that is
    the point of pinning it separately: the deprecation warning is emitted **once per
    PROCESS** (`_LEGACY_MS_WARNED`), so under pytest it attaches itself to whichever
    config happens to run first and is invisible on the other six. A warning-based
    check on it would be order-dependent -- and the suite randomizes order.

    Seven configs carried `type: spiral` when this was written (SW08, SW15, SW18,
    SW22, SW23, SW26, SW37) and exactly one of them ever warned.
    """
    import glob
    import yaml

    offenders = []
    for path in sorted(glob.glob(os.path.join(ROOT, "examples", "**", "*.yaml"),
                                 recursive=True)):
        with open(path) as fh:
            try:
                doc = yaml.safe_load(fh)
            except Exception:
                continue
        ms = (doc or {}).get("magnetic_structure") if isinstance(doc, dict) else None
        if isinstance(ms, dict) and ms.get("type") in ("spiral", "propagation_vector"):
            offenders.append(os.path.relpath(path, ROOT))
    assert not offenders, (f"deprecated magnetic_structure type in: {offenders}. "
                           f"`spiral` -> `single_k` is a rename; "
                           f"`propagation_vector` also needs `real_space: true`.")
