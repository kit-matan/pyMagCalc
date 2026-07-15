"""Guards against the 'wrong ground state' failure class.

LSWT expanded about anything that is not a classical energy MINIMUM has imaginary
magnon energies. Historically the engine only logged per-q warnings, discarded the
imaginary part, and completed the run -- so a local minimum (too few minimization
starts) or a hand-written non-minimum structure silently produced a
plausible-looking but meaningless spectrum. That is exactly what happened to SW20
in field.

These tests pin the three guards:
  1. MagCalc.max_imaginary_energy() surfaces the instability as a NUMBER.
  2. The runner fails the run by default (calculation.on_imaginary: error).
  3. The multistart reports how many starts reached the best energy.
"""
import copy

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.runner import run_calculation

S_VAL = 1.0

# AFM nearest-neighbour chain (J > 0), 2-site cell.
BASE = {
    "crystal_structure": {
        "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
        "atoms_uc": [
            {"label": "A", "pos": [0.0, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"},
            {"label": "B", "pos": [0.5, 0.0, 0.0], "spin_S": S_VAL, "ion": "Fe2+"}],
    },
    "interactions": {"symmetry_rules": [
        {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
    "parameters": {}, "parameter_order": [],
    "tasks": {"minimization": False, "dispersion": True},
    "q_path": {"G": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0],
               "path": ["G", "X"], "points_per_segment": 5},
    "plotting": {"save_plot": False, "show_plot": False, "plot_structure": False},
    "output": {"save_data": False},
    "calculation": {"cache_mode": "none"},
}

# The true ground state of an AFM chain: Neel.
NEEL = {"type": "pattern", "pattern_type": "generic",
        "directions": [[0, 0, 1], [0, 0, -1]]}
# NOT the ground state: ferromagnetic alignment with antiferromagnetic exchange.
# LSWT about this state is unstable -> imaginary magnon energies.
WRONG = {"type": "pattern", "pattern_type": "generic",
         "directions": [[0, 0, 1], [0, 0, 1]]}


def _calc(structure):
    cfg = copy.deepcopy(BASE)
    cfg["magnetic_structure"] = copy.deepcopy(structure)
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=S_VAL, cache_mode="none",
                      cache_file_base="gsguard", hamiltonian_params=[])


def test_max_imaginary_energy_is_zero_for_the_true_ground_state():
    assert _calc(NEEL).max_imaginary_energy() < 1e-8


def test_ferromagnetic_state_for_an_afm_is_NOT_caught_by_the_imaginary_check():
    """Documents the blind spot that motivates the second guard.

    A collinear FM reference has no anomalous terms, so the Bogoliubov problem stays
    diagonal; process_calc_disp then sorts the +/- omega pairs and returns the upper
    half, i.e. |omega|. The spectrum comes back REAL and POSITIVE even though the
    state is a maximum. Imaginary energies alone are therefore NOT a sufficient
    ground-state check."""
    assert _calc(WRONG).max_imaginary_energy() < 1e-8      # blind to it!


def test_energy_audit_detects_the_ferromagnetic_state_for_an_afm():
    """The second guard: a downhill step from the FM state finds the (much lower)
    Neel energy, so the structure is provably not a minimum."""
    e_now, e_relaxed = _calc(WRONG).relax_from_current()
    assert e_relaxed < e_now - 1e-3


def test_energy_audit_passes_for_the_true_ground_state():
    e_now, e_relaxed = _calc(NEEL).relax_from_current()
    assert e_relaxed >= e_now - 1e-6


def test_max_imaginary_energy_detects_a_non_stationary_state():
    """A canted, non-stationary structure DOES produce imaginary magnons (this is
    the SW20-in-field failure class)."""
    canted = {"type": "pattern", "pattern_type": "generic",
              "directions": [[0, 0, 1], [0.7, 0, 0.7]]}
    assert _calc(canted).max_imaginary_energy() > 1e-3


def _write(tmp_path, structure, calculation=None):
    cfg = copy.deepcopy(BASE)
    cfg["magnetic_structure"] = copy.deepcopy(structure)
    if calculation:
        cfg["calculation"].update(calculation)
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


def test_runner_raises_on_a_wrong_ground_state_by_default(tmp_path):
    """The whole point: a wrong ground state must FAIL the run, not quietly write a
    plausible plot. Here the FM-for-an-AFM state is caught by the ENERGY guard (the
    imaginary check is blind to it), and the message must say so."""
    with pytest.raises(ValueError, match="NOT a classical energy minimum") as exc:
        run_calculation(_write(tmp_path, WRONG))
    assert "minimization" in str(exc.value)


def test_runner_raises_on_a_non_stationary_structure(tmp_path):
    """The SW20 class: imaginary magnon energies. Message must name the ground state
    and the minimization knobs."""
    canted = {"type": "pattern", "pattern_type": "generic",
              "directions": [[0, 0, 1], [0.7, 0, 0.7]]}
    with pytest.raises(ValueError) as exc:
        run_calculation(_write(tmp_path, canted))
    msg = str(exc.value)
    assert "ground state" in msg or "minimum" in msg
    assert "num_starts" in msg or "minimization" in msg


def test_runner_accepts_the_true_ground_state(tmp_path):
    run_calculation(_write(tmp_path, NEEL))          # must not raise


def test_on_imaginary_warn_downgrades_to_a_warning(tmp_path):
    run_calculation(_write(tmp_path, WRONG, {"on_imaginary": "warn"}))   # no raise


def test_on_imaginary_off_disables_the_gate(tmp_path):
    run_calculation(_write(tmp_path, WRONG, {"on_imaginary": "off"}))    # no raise


def test_on_imaginary_rejects_a_bad_value(tmp_path):
    with pytest.raises(ValueError, match="on_imaginary"):
        run_calculation(_write(tmp_path, NEEL, {"on_imaginary": "yes please"}))


def test_gui_shaped_calculation_block_controls_the_guard(tmp_path):
    """The Studio front-ends expose the guard as a 'Ground-State Check' picker and post
    it inside the `calculation` block. `warn` must genuinely downgrade the guard and
    `error` must genuinely fail — a UI control that does not change behaviour is worse
    than no control at all, because it looks like an escape hatch and isn't one.
    """
    # exactly what the GUI sends (cache_mode/backend alongside the new key)
    gui_warn = {"cache_mode": "none", "backend": "numpy", "on_imaginary": "warn"}
    gui_error = {"cache_mode": "none", "backend": "numpy", "on_imaginary": "error"}

    run_calculation(_write(tmp_path, WRONG, gui_warn))        # metastable: allowed

    with pytest.raises(ValueError):
        run_calculation(_write(tmp_path, WRONG, gui_error))   # default: still fails


def test_multistart_reports_how_many_starts_hit_the_best_energy():
    """`hits` is the evidence that the minimum is global; hits == 1 is the smell
    that preceded SW20's wrong ground state."""
    calc = _calc(NEEL)
    res = calc.minimize_energy(num_starts=8, early_stopping=8, method="L-BFGS-B",
                               seed=0)
    assert hasattr(res, "hits") and hasattr(res, "num_starts")
    assert 1 <= res.hits <= res.num_starts
