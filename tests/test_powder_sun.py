"""Powder averaging for the SU(N) and entangled calculators (previously a
NotImplementedError -- user-reported when running Cu5SbO6 powder from the app).

Two layers of validation:

  * GATE-1 CROSS-ENGINE ORACLE: for S=1/2, SU(N) is identical to dipole LSWT, so the
    new SU(N) powder average must reproduce the validated DIPOLE powder average
    end-to-end (same Fibonacci sampling, same per-shell averaging) on an AFM chain.
  * EXACT PLUMBING IDENTITY: for the entangled dimer chain (Cu5SbO6-like), the
    powder result must equal a manual spherical average over the calculator's own
    calculate_sqw at the same sphere points -- to machine precision. The sqw itself
    is validated elsewhere (tests/test_entangled_units.py), so this pins the only
    NEW code: the averaging.
"""
import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import fibonacci_sphere_points
from magcalc.sun.adapter import SUNCalculator
from magcalc.sun.entangled import EntangledCalculator

Q_MAGS = [0.8, 1.6, 2.4]
N_SAMP = 16


def _chain_cfg(S=0.5, J=1.0):
    return {
        "crystal_structure": {
            "lattice_vectors": [[4.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
            "atoms_uc": [
                {"label": "A", "pos": [0.0, 0, 0], "spin_S": S, "ion": "Cu2+"},
                {"label": "B", "pos": [0.5, 0, 0], "spin_S": S, "ion": "Cu2+"}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": 2.0, "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]},
    }


def test_sun_intensities_match_sunny_on_the_afm_chain():
    """Gate 1b — INTENSITIES (the original Gate 1 only compared dispersions, which hid
    a real bug): at generic q in a 2-atom cell, the S=1/2 AFM chain's one-magnon
    intensities must match Sunny 0.8.1 (ssf_perp, apply_g=false) absolutely. Before
    the fix the engine double-counted intracell phases (its H(q) is built in the
    full-position gauge) and picked the SUPPRESSED Bogoliubov combination: ~60x too
    weak at the zone boundary."""
    from magcalc.sun.lswt import SUNModel
    cfg = _chain_cfg()
    for a in cfg["crystal_structure"]["atoms_uc"]:
        a.pop("ion")                       # Sunny reference computed without form factor
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    sm = SUNModel.from_generic_model(m, [])
    sunny = {(0.9, 0.3, 0.2): 0.656891, (1.7, 0.0, 0.4): 4.050179}   # Sunny 0.8.1
    for q, i_sunny in sunny.items():
        w, I = sm.structure_factor(np.array(q), cross_section="perp")
        assert abs(I.sum() - i_sunny) < 1e-5 * i_sunny, \
            f"q={q}: {I.sum()} vs Sunny {i_sunny}"


def test_gate1_sun_powder_equals_dipole_powder():
    """S=1/2 AFM chain: the new SU(N) powder path must match the validated dipole
    powder path (identical sphere sampling + averaging conventions)."""
    cfg = _chain_cfg()
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    dip_calc = mc.MagCalc(spin_model_module=m, spin_magnitude=0.5, cache_mode="none",
                          cache_file_base="powder_dip", hamiltonian_params=[])
    dip = dip_calc.calculate_powder_average(Q_MAGS, num_samples=N_SAMP)

    m2 = GenericSpinModel(cfg)
    th, ph = m2.generate_magnetic_structure()
    m2.set_magnetic_structure(th, ph)
    sun_calc = SUNCalculator(m2, {"calculation": {}}, [])
    sun = sun_calc.calculate_powder_average(Q_MAGS, num_samples=N_SAMP)

    d_sorted = np.sort(dip.energies, axis=1)
    s_sorted = np.sort(sun.energies, axis=1)
    assert np.allclose(d_sorted, s_sorted, atol=1e-6), "powder mode energies differ"
    # total scattered weight per shell must agree (mode ordering may differ)
    assert np.allclose(dip.intensities.sum(axis=1), sun.intensities.sum(axis=1),
                       rtol=1e-6), "powder intensities differ"


def _dimer_calculator():
    cfg = {
        "crystal_structure": {
            "lattice_vectors": [[4.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
            "atoms_uc": [
                {"label": "CuA", "pos": [0.0, 0, 0], "spin_S": 0.5, "ion": "Cu2+"},
                {"label": "CuB", "pos": [0.3, 0, 0], "spin_S": 0.5, "ion": "Cu2+"}]},
        "interactions": {"heisenberg": [
            {"pair": ["CuA", "CuB"], "rij_offset": [0, 0, 0], "value": 5.0},
            {"pair": ["CuB", "CuA"], "rij_offset": [0, 0, 0], "value": 5.0},
            {"pair": ["CuB", "CuA"], "rij_offset": [1, 0, 0], "value": 1.0},
            {"pair": ["CuA", "CuB"], "rij_offset": [-1, 0, 0], "value": 1.0}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "calculation": {"mode": "entangled"},
        "units": [["CuA", "CuB"]],
    }
    m = GenericSpinModel(cfg)
    return EntangledCalculator(m, cfg, [])


def test_entangled_powder_is_the_exact_spherical_average():
    """Cu5SbO6-like dimer chain: powder == manual average of calculate_sqw over the
    same Fibonacci sphere points, to machine precision."""
    calc = _dimer_calculator()
    res = calc.calculate_powder_average(Q_MAGS, num_samples=N_SAMP)
    assert res.energies.shape[0] == len(Q_MAGS)

    for i, qm in enumerate(Q_MAGS):
        pts = fibonacci_sphere_points(qm, N_SAMP)
        ref = calc.calculate_sqw(pts)
        assert np.allclose(res.energies[i], np.nanmean(ref.energies, axis=0),
                           atol=1e-12)
        assert np.allclose(res.intensities[i], np.nanmean(ref.intensities, axis=0),
                           atol=1e-12)
    # a dimer's triplon powder intensity must be finite and positive somewhere
    assert res.intensities.max() > 0
