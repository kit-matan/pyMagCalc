"""Magnetic form factors: the VALUES (vs Sunny) and their application.

Two independent failure modes are pinned here.

1. THE VALUES. The hand-written coefficient table this module used to carry was
   wrong. Every entry was normalized so that f(0) = 1 -- so a Q -> 0 check passed
   -- but the Q-dependence was not the tabulated one, reaching +22% in intensity
   at |Q| = 2.5 A^-1, +53% at 3.8 A^-1 and +113% at 5 A^-1 (Mn2+). It was
   invisible because the only test compared I_ion / I_bare against
   get_form_factor(...)**2, which is self-consistent BY CONSTRUCTION: the same
   wrong f(Q) appears on both sides. `test_j0_matches_sunny` fixes that by pinning
   f(Q) itself to an independent oracle.

2. THE APPLICATION. GenericSpinModel.__init__ used to RESET `_ion_list = []` AFTER
   `_load_structure` had populated it -- so `ion_list()` was empty for every config
   and the form factor was silently dropped from ALL intensities (dipole, SU(N),
   entangled). Invisible to every Sunny/SpinW cross-check (those were computed
   form-factor-free on both sides); caught physically: the Cu5SbO6 powder map
   carried far too much intensity at high |Q| compared to PRR 8, 013247 Fig. 5.
"""
import os

import numpy as np
import pytest
import yaml

from magcalc.form_factors import get_form_factor, get_j0
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import powder_sample_modes
from magcalc.sun.entangled import EntangledCalculator

HERE = os.path.dirname(__file__)

# --------------------------------------------------------------------------
# The values themselves, against Sunny 0.8.1 (the standard P. J. Brown /
# International Tables <j0>, <j2> expansions). Generated with
#
#   julia -e 'using Sunny; ff = FormFactor("Cu2");
#             println(Sunny.compute_form_factor(ff, Q^2))'
#
# NOT transcribed from a paper and NOT self-generated.
# --------------------------------------------------------------------------
FF_QS = [0.0, 0.5, 1.25, 2.5, 3.75, 5.0, 6.5]          # 1/Angstrom

SUNNY_J0 = {
    "Cu2": [1.000000000, 0.987884886, 0.927772939, 0.751876969, 0.548723675,
            0.370997868, 0.214916238],
    "Fe2": [1.000000000, 0.983985990, 0.905460497, 0.685760046, 0.453978553,
            0.272847976, 0.131803286],
    "Fe3": [0.999700000, 0.986608044, 0.921210419, 0.727044981, 0.502863048,
            0.313131311, 0.156815139],
    "Mn2": [0.999200000, 0.981925716, 0.897173038, 0.661061208, 0.418456000,
            0.238384267, 0.106173274],
    "Ni2": [0.999800000, 0.986644321, 0.921470131, 0.732379302, 0.519233551,
            0.339523289, 0.187646871],
    "Co2": [0.998600000, 0.984617070, 0.915185390, 0.713399470, 0.489369234,
            0.307542188, 0.161064878],
    "Cr3": [1.000100000, 0.983582092, 0.902099821, 0.670693645, 0.424571263,
            0.235856960, 0.096822280],
    "Yb3": [0.999700000, 0.992202805, 0.954159656, 0.834278081, 0.677506794,
            0.518736317, 0.353203979],
    "Nd3": [1.000000000, 0.987120182, 0.923142788, 0.735945882, 0.521543048,
            0.336400368, 0.176495731],
    # 5d, configuration-dependent: keeps Sunny's disambiguating suffix.
    "Ir0a": [0.999000000, 0.953745593, 0.752717629, 0.344586150, 0.091299704,
             -0.011583499, -0.028261066],
}

# g != 2 activates the <j2> term of the dipole approximation.
SUNNY_G_LANDE = {
    ("Yb3", 1.2): [0.999700000, 0.994217679, 0.966217837, 0.876000900,
                   0.753286934, 0.622655573, 0.477220334],
    ("Nd3", 0.7272): [1.000000000, 0.996097515, 0.975658148, 0.904454282,
                      0.796524290, 0.670684184, 0.520245591],
    ("Fe2", 1.5): [1.000000000, 0.986105636, 0.917687499, 0.723144156,
                   0.510836997, 0.337049468, 0.193033011],
}


@pytest.mark.parametrize("ion", sorted(SUNNY_J0))
def test_j0_matches_sunny(ion):
    """f(Q) itself, not a ratio in which it cancels."""
    got = np.array([get_form_factor(ion, q) for q in FF_QS])
    assert got == pytest.approx(np.array(SUNNY_J0[ion]), abs=1e-9)


@pytest.mark.parametrize("key", sorted(SUNNY_G_LANDE))
def test_dipole_approximation_with_lande_g_matches_sunny(key):
    """f(Q) = <j0> + ((2-g)/g) <j2> -- the <j2> branch, which used to be missing
    entirely (and whose docstring formula had the wrong sign)."""
    ion, g = key
    got = np.array([get_form_factor(ion, q, g=g) for q in FF_QS])
    assert got == pytest.approx(np.array(SUNNY_G_LANDE[key]), abs=1e-9)


def test_ion_spellings_are_equivalent():
    """'Fe2+' (pyMagCalc), 'Fe2' (Sunny) and 'Fe' (neutral) all resolve."""
    for q in FF_QS:
        assert get_form_factor("Fe2+", q) == get_form_factor("Fe2", q)
        assert get_form_factor("Fe", q) == get_j0("Fe0", q)


def test_unknown_ion_falls_back_to_unity(caplog):
    """An unrecognised label must warn and return f = 1, never a wrong number."""
    with caplog.at_level("WARNING"):
        assert get_form_factor("Unobtainium3+", 2.5) == 1.0
    assert "form-factor table" in caplog.text


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
