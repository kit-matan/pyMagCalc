"""The ground-state guards' TOLERANCE knobs (coverage-audit item 3).

The 2026-08-04 audit found `calculation.imaginary_tolerance` and
`calculation.energy_tolerance` in zero tests. The guards themselves are well covered
-- they are the engine's defence against the documented #1 source of silent wrongness
(LSWT expanded about a non-minimum) -- but nothing checked that these keys are READ,
or that they move the threshold in the right direction. A tolerance silently ignored
would leave a guard that cannot be relaxed; one read from the wrong key, or compared
with the wrong sign, would leave a guard that never fires.

The model is a Neel chain whose two spins are tilted TOWARDS each other by an angle
theta, which gives an exact analytic handle:

    Delta_E(theta) = 2 J S^2 (1 - cos theta)     (two bonds per cell)

so the audit's arithmetic can be pinned against a closed form rather than a golden
number, and the tolerance can be bracketed around a drop whose size is known exactly.

The same structure is unstable, so it also carries imaginary magnons -- one model
exercises BOTH guards, and loosening the energy audit hands off to the imaginary one.
That is the design ("neither guard alone is sufficient", CLAUDE.md) observed directly.

FOUND WHILE WRITING THIS: `calculation.imaginary_rel_tolerance` is a THIRD tolerance,
documented nowhere -- not CLAUDE.md, not TUTORIAL.md, not schema.py -- and the
imaginary guard fires only when the absolute AND the relative threshold are both
exceeded. A key-level audit could not have found it: it enumerates DOCUMENTED keys, so
an undocumented one is invisible to exactly the process meant to catch gaps.
"""
import logging
import math
import os
import tempfile

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc import runner
from magcalc.generic_model import GenericSpinModel

J, S_VAL = 1.0, 1.0
LAT = [[3.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]]


def delta_e(theta_deg):
    """Exact classical energy excess of the tilted Neel chain, per cell."""
    return 2 * J * S_VAL ** 2 * (1 - math.cos(math.radians(theta_deg)))


def _config(theta_deg, **calculation):
    t = math.radians(theta_deg)
    return {
        "crystal_structure": {
            "lattice_vectors": LAT,
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S_VAL},
                         {"label": "B", "pos": [0.5, 0, 0], "spin_S": S_VAL}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": J},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": J},
            {"pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": J},
            {"pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {
            "type": "pattern", "pattern_type": "generic",
            "directions": [[0, 0, 1], [math.sin(t), 0, -math.cos(t)]]},
        "calculation": dict(calculation),
        "tasks": {"dispersion": True},
        "q_path": {"points": [[0, 0, 0], [0.5, 0, 0]], "n_points": 5},
        "plotting": {"enabled": False},
    }


def _run(tmp_path, theta_deg, **calculation):
    """Run a config through the RUNNER, so the config key is what is exercised.
    Returns None on success, else the exception message."""
    path = os.path.join(str(tmp_path), "c.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(_config(theta_deg, **calculation), f)
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        runner.run_calculation(path)
        return None
    except Exception as e:                       # noqa: BLE001 - message is the assert
        return str(e)
    finally:
        os.chdir(cwd)


def _which_guard(msg):
    if msg is None:
        return "none"
    if "NOT a classical energy minimum" in msg:
        return "energy"
    if "IMAGINARY" in msg:
        return "imaginary"
    if "q!=0 SPIRAL" in msg:
        return "lt"
    return f"other: {msg.splitlines()[0][:80]}"


# ---------------------------------------------------------------------------
# The energy audit, and `energy_tolerance`
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [2.0, 5.0, 10.0])
def test_energy_audit_measures_the_exact_tilt_energy(theta):
    """The audit's arithmetic against a closed form. `relax_from_current` must report
    a drop of exactly 2 J S^2 (1 - cos theta) -- an identity, not a golden number, so
    it fails on a factor of 2, a per-site/per-cell mix-up, or a wrong relaxed state."""
    cfg = _config(theta)
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S_VAL, cache_mode="none",
                      cache_file_base=f"gt{theta}", hamiltonian_params=[])
    e_now, e_relaxed = calc.relax_from_current()
    assert e_now - e_relaxed == pytest.approx(delta_e(theta), rel=1e-4)
    # and the relaxed state is the true Neel minimum, -2 J S^2 per cell
    assert e_relaxed == pytest.approx(-2 * J * S_VAL ** 2, abs=1e-6)


def test_energy_tolerance_brackets_the_audit(tmp_path):
    """THE test for the knob: the SAME structure must trip the energy audit when the
    tolerance sits below the (exactly known) drop and clear it when the tolerance sits
    above. A tolerance that was parsed but never compared, or compared with the wrong
    sign, fails one side or the other."""
    d = delta_e(5.0)
    assert _which_guard(_run(tmp_path, 5.0, energy_tolerance=d / 10)) == "energy"
    # Above the drop the energy audit is satisfied -- and the IMAGINARY guard then
    # catches the same bad structure. The guard that fires must CHANGE; asserting
    # merely "no longer the energy error" would also pass if nothing ran at all.
    assert _which_guard(_run(tmp_path, 5.0, energy_tolerance=d * 10)) == "imaginary"


def test_energy_tolerance_default_is_tight_enough_to_see_a_small_tilt(tmp_path):
    """The documented default is 1e-6 meV. A 2 degree tilt is a 1.2e-3 meV drop --
    small, plainly physical, and it must not slip through."""
    assert delta_e(2.0) > 1e-6
    assert _which_guard(_run(tmp_path, 2.0)) == "energy"


def test_a_true_minimum_passes_every_guard(tmp_path):
    """Guards the guards. If the checks fired unconditionally, every test above would
    pass while the engine refused all legitimate work."""
    assert _which_guard(_run(tmp_path, 0.0)) == "none"


# ---------------------------------------------------------------------------
# The imaginary-energy guard, `imaginary_tolerance`, and the undocumented
# `imaginary_rel_tolerance`
# ---------------------------------------------------------------------------

LOOSE_E = {"energy_tolerance": 1.0}      # step past the energy audit to reach guard 1


def test_imaginary_tolerance_brackets_the_guard(tmp_path):
    """Measured on this structure: max |Im(omega)| = 0.15 meV, 7.5% of the bandwidth.
    An absolute tolerance above that silences it; zero cannot rescue it."""
    assert _which_guard(_run(tmp_path, 5.0, imaginary_tolerance=1.0,
                             **LOOSE_E)) == "none"
    assert _which_guard(_run(tmp_path, 5.0, imaginary_tolerance=0.0,
                             **LOOSE_E)) == "imaginary"


def test_imaginary_guard_needs_BOTH_the_absolute_and_relative_threshold(tmp_path):
    """`imaginary_rel_tolerance` (undocumented) is ANDed with `imaginary_tolerance`,
    so EITHER knob alone silences the guard -- and, more importantly, lowering
    `imaginary_tolerance` alone cannot make it fire. That is deliberate: an absolute
    meV cutoff cannot separate a real instability from numerical noise across models
    whose energy scales differ by orders of magnitude (the code cites SW07's kagome,
    1e-3 meV of noise on a 2.4 meV band). Pinned so the AND is not silently loosened
    to an OR, which would make every near-Goldstone model start failing."""
    assert _which_guard(_run(tmp_path, 5.0, imaginary_rel_tolerance=0.5,
                             **LOOSE_E)) == "none"
    # ... with the relative threshold back below the measured 7.5%, it fires again
    assert _which_guard(_run(tmp_path, 5.0, imaginary_rel_tolerance=1e-4,
                             **LOOSE_E)) == "imaginary"


# ---------------------------------------------------------------------------
# `on_imaginary`, which gates BOTH guards
# ---------------------------------------------------------------------------

def test_on_imaginary_warn_downgrades_to_a_warning(tmp_path, caplog):
    """`warn` must not raise -- and must still SAY something. A downgrade that also
    lost the message would leave the user with a silently meaningless spectrum, the
    exact outcome the guards exist to prevent."""
    with caplog.at_level(logging.WARNING):
        assert _which_guard(_run(tmp_path, 5.0, on_imaginary="warn")) == "none"
    assert "NOT a classical energy minimum" in caplog.text


def test_on_imaginary_off_disables_both_guards(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert _which_guard(_run(tmp_path, 5.0, on_imaginary="off")) == "none"
    assert "NOT a classical energy minimum" not in caplog.text
    assert "IMAGINARY" not in caplog.text


def test_on_imaginary_rejects_an_unknown_value(tmp_path):
    msg = _run(tmp_path, 0.0, on_imaginary="yes")
    assert msg is not None and "on_imaginary" in msg


# ---------------------------------------------------------------------------
# The SU(N) energy audit reads the SAME key, from its own code path
# ---------------------------------------------------------------------------

def _sun_config(**calculation):
    """Easy-PLANE anisotropy: an S=1 coherent state along x is not the SU(N) ground
    state, so the audit has something to find. (Easy-AXIS would not work -- there the
    collinear state IS an exact SU(N) eigenstate and nothing relaxes.)"""
    cfg = _config(0.0, mode="SUN", **calculation)
    cfg["interactions"]["single_ion_anisotropy"] = [
        {"value": 4.0, "axis": [0, 0, 1], "atoms": ["A", "B"]}]
    cfg["magnetic_structure"]["directions"] = [[1, 0, 0], [-1, 0, 0]]
    return cfg


def _run_sun(tmp_path, **calculation):
    path = os.path.join(str(tmp_path), "sun.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(_sun_config(**calculation), f)
    cwd = os.getcwd()
    os.chdir(str(tmp_path))
    try:
        runner.run_calculation(path)
        return None
    except Exception as e:                       # noqa: BLE001
        return str(e)
    finally:
        os.chdir(cwd)


@pytest.mark.slow
def test_sun_energy_tolerance_is_read_from_the_same_key(tmp_path):
    """`sun/adapter.py` has its OWN `energy_tolerance` read, in meV/SITE rather than
    per cell. Covered separately because a shared key name is not shared code -- and
    this is the audit that catches the documented SU(N) trap (a dipole-derived state
    pasted under `mode: SUN`), which the imaginary check provably cannot see."""
    msg = _run_sun(tmp_path)
    assert msg is not None and "NOT the SU(N) ground state" in msg
    assert _run_sun(tmp_path, energy_tolerance=10.0) is None


@pytest.mark.slow
def test_sun_audit_respects_on_imaginary_off(tmp_path):
    assert _run_sun(tmp_path, on_imaginary="off") is None
