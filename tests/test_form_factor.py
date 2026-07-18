"""Magnetic form factor application (regression for a silent ordering bug).

GenericSpinModel.__init__ used to RESET `_ion_list = []` AFTER `_load_structure`
had populated it -- so `ion_list()` was empty for every config and the magnetic
form factor was silently dropped from ALL intensity calculations (dipole, SU(N),
entangled). Invisible to every Sunny/SpinW cross-check (those were computed
form-factor-free on both sides); caught physically: the Cu5SbO6 powder map carried
far too much intensity at high |Q| compared to PRR 8, 013247 Fig. 5.
"""
import os

import numpy as np
import pytest
import yaml

from magcalc.form_factors import get_form_factor
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import powder_sample_modes
from magcalc.sun.entangled import EntangledCalculator

HERE = os.path.dirname(__file__)


def _dimer_cfg(with_ion):
    atoms = [{"label": "A", "pos": [0., 0, 0], "spin_S": 0.5},
             {"label": "B", "pos": [0.2, 0, 0], "spin_S": 0.5}]
    if with_ion:
        for a in atoms:
            a["ion"] = "Cu2+"
    return {"crystal_structure": {"lattice_vectors": [[30., 0, 0], [0, 30, 0],
                                                      [0, 0, 30]],
                                  "atoms_uc": atoms},
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": 16.5},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": 16.5}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "calculation": {"mode": "entangled"}, "units": [["A", "B"]]}


def test_ion_list_survives_construction():
    """The ordering-bug regression: ion_list must reflect the atoms' `ion` keys."""
    m = GenericSpinModel(_dimer_cfg(True))
    assert m.ion_list() == ["Cu2+", "Cu2+"]


def test_wyckoff_expansion_propagates_ion():
    """Regression: the wyckoff_atoms -> atoms_uc expansion dropped the `ion` key
    (add_wyckoff_atom was called without it, and the expanded atoms carried an
    explicit `ion: None` that defeated the .get() fallback chain), so every
    symmetry-mode config computed intensities with form factor 1.0 and logged
    "Ion 'None' not found" (reported on the Cu2V2O7 example)."""
    cfg = {"crystal_structure": {
               "lattice_parameters": {"a": 6.0, "b": 6.0, "c": 8.0,
                                      "alpha": 90.0, "beta": 90.0, "gamma": 120.0,
                                      "space_group": 147},
               "wyckoff_atoms": [{"label": "K", "pos": [0.5, 0.0, 0.0],
                                  "spin_S": 0.5, "ion": "Cu2+"}],
               "atom_mode": "symmetry"},
           "interactions": {"symmetry_rules": [
               {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern",
                                  "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]}}
    m = GenericSpinModel(cfg)
    assert m.ion_list() == ["Cu2+"] * 3          # 3e orbit, ion on every site


def test_intensities_carry_the_squared_form_factor():
    """I_with_ion / I_without must equal f_Cu2+(|Q|)^2 exactly, at every |Q|."""
    with_ion = EntangledCalculator(GenericSpinModel(_dimer_cfg(True)),
                                   _dimer_cfg(True), [])
    without = EntangledCalculator(GenericSpinModel(_dimer_cfg(False)),
                                  _dimer_cfg(False), [])
    for qm in (0.8, 1.5, 3.0, 5.0):
        q = np.array([[0.6, 0.8, 0.0]]) * qm      # generic direction, |q| = qm
        Iw = with_ion.calculate_sqw(q).intensities[0].sum()
        Io = without.calculate_sqw(q).intensities[0].sum()
        assert Iw / Io == pytest.approx(get_form_factor("Cu2+", qm) ** 2, rel=1e-9)


def test_isolated_dimer_powder_modulation_with_form_factor():
    """EXACT identity: the isolated dimer's energy-integrated powder intensity is
    (1 - sin(Qd)/(Qd)) * f_Cu2+(Q)^2 -- interference factor times form factor."""
    cfg = _dimer_cfg(True)
    calc = EntangledCalculator(GenericSpinModel(cfg), cfg, [])
    d = 6.0
    # |Q| <= 3: at Qd ~ 30 the interference integrand oscillates too fast for a
    # few hundred Fibonacci points (pure quadrature error; the high-|Q| form
    # factor itself is pinned EXACTLY by the per-shell ratio test above).
    qm = np.linspace(0.5, 3.0, 10)
    E, I = powder_sample_modes(calc, qm, num_samples=400)
    tot = I.sum(axis=1)
    ana = (1.0 - np.sin(qm * d) / (qm * d)) * \
        np.array([get_form_factor("Cu2+", q) ** 2 for q in qm])
    ratio = tot / ana
    assert (ratio.max() - ratio.min()) / ratio.mean() < 5e-3


def test_cu5sbo6_powder_high_q_is_suppressed():
    """The user-reported symptom, pinned: with the Cu2+ form factor applied, the
    Cu5SbO6 powder intensity at high |Q| is strongly suppressed (paper Fig. 5(b)).
    The COUPLED dimers deviate from the bare interference-factor product by ~15%
    (Bogoliubov redistribution over the sphere), so only the robust physical
    statement is asserted: the 5 A^-1 shell carries a small fraction of the peak,
    and the |Q| modulation tracks the analytic product to ~20%."""
    doc = yaml.safe_load(open(os.path.join(
        HERE, "..", "examples", "entangled", "Cu5SbO6", "config.yaml")))
    m = GenericSpinModel(doc)
    calc = EntangledCalculator(m, doc,
                               [doc["parameters"][k] for k in doc["parameter_order"]])
    d = 6.0
    qm = np.linspace(0.5, 5.0, 10)
    E, I = powder_sample_modes(calc, qm, num_samples=120)
    tot = I.sum(axis=1)
    ana = (1.0 - np.sin(qm * d) / (qm * d)) * \
        np.array([get_form_factor("Cu2+", q) ** 2 for q in qm])
    ratio = tot / ana
    assert (ratio.max() - ratio.min()) / ratio.mean() < 0.2
    assert tot[-1] < 0.25 * tot[np.argmax(tot)]
