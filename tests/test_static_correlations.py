"""Static / energy-integrated correlations (Gap 4 #19).

Two different quantities that Sunny gives similar names, and which must not be
validated by one test:

  * `tasks: {static_sqw: true}` -- the ENERGY-INTEGRATED LSWT S(q), i.e. the band sum
    of S(q,w) (Sunny `intensities_static`). Zero temperature, ordered state required.
  * `tasks: {static_correlations: true}` -- the classical INSTANTANEOUS S(q) from
    Metropolis samples (Sunny `SampledCorrelationsStatic`). No dynamics, no ordered
    state, valid above T_N where LSWT has nothing to say.

The classical one is normalized PER SITE, which makes free isotropic spins an exact,
temperature-independent, model-free identity: 2S^2/3 in `perp` and S^2 in `trace` at
every q. That is the oracle for the normalization, and it fixes it without appeal to
any other code.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import static_intensities
from magcalc.thermal_mc import static_correlations

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = [[0.13, 0, 0], [0.31, 0, 0], [0.45, 0, 0]]

# Sunny 0.8.1: intensities_static(SpinWaveTheory(sys; measure=ssf_perp(sys;
# apply_g=false)), qs) on the same 2-site chain, s = 1.
SUNNY_STATIC = {
    "neel_z": [0.20709005, 0.52947275, 0.85408069],
    "fm": [1.00000000, 1.00000000, 1.00000000],
}
CASES = {"neel_z": (1.0, [0, 0, 1], [0, 0, -1]),
         "fm": (-1.0, [0, 0, 1], [0, 0, 1])}


def _lswt_static(case, tag):
    J, d1, d2 = CASES[case]
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": 1.0},
             {"label": "B", "pos": [0.5, 0, 0], "spin_S": 1.0}]
    cfg = {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
           "interactions": {"heisenberg": [{"pair": p, "rij_offset": o, "value": J}
                                           for p, o in NN]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [d1, d2]},
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return calc, static_intensities(calc, [np.array(q) @ B for q in HS])


@pytest.mark.parametrize("case", sorted(SUNNY_STATIC))
def test_lswt_static_matches_sunny(case):
    _, got = _lswt_static(case, f"stat_{case}")
    assert got == pytest.approx(np.array(SUNNY_STATIC[case]), abs=1e-7)


def test_lswt_static_is_exactly_the_band_sum():
    """Definitional, but it is the definition that must not drift: the static
    intensity is S(q,w) integrated over w, i.e. summed over bands."""
    calc, got = _lswt_static("neel_z", "stat_bandsum")
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    res = calc.calculate_sqw([np.array(q) @ B for q in HS], cross_section="perp")
    assert got == pytest.approx(np.real(res.intensities).sum(axis=1), abs=0)


# --------------------------------------------------------------------------
# Classical instantaneous S(q)
# --------------------------------------------------------------------------
FREE_LAT = [[5.0, 0, 0], [0, 5.0, 0], [0, 0, 9.0]]


def _free_model(S):
    """Genuinely free spins: a bond list with value 0, so the supercell builder has
    something to walk but the Hamiltonian is empty."""
    cfg = {"crystal_structure": {"lattice_vectors": FREE_LAT,
                                 "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                               "spin_S": S}]},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": 0.0},
               {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": 0.0}]},
           "magnetic_structure": {"type": "pattern",
                                  "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": [],
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    B = 2 * np.pi * np.linalg.inv(np.array(FREE_LAT, float)).T
    return GenericSpinModel(copy.deepcopy(cfg)), B


@pytest.mark.parametrize("S,cs,exact", [(0.5, "perp", 2 * 0.25 / 3),
                                        (1.0, "perp", 2.0 / 3),
                                        (2.0, "perp", 8.0 / 3),
                                        (1.0, "trace", 1.0),
                                        (2.0, "trace", 4.0)])
def test_free_spins_give_the_exact_sum_rule(S, cs, exact):
    """N uncorrelated spins of length S: <S^a(q) S^b(q)*> = N (S^2/3) delta_ab at
    EVERY q, so the per-site S(q) is (S^2/3) Tr P = 2S^2/3 for `perp` and S^2 for
    `trace`, independent of q and of temperature. This fixes the normalization
    outright -- no oracle, no fitted constant."""
    m, B = _free_model(S)
    q = np.array([[0.13, 0.2, 0], [0.31, 0, 0], [0.5, 0.5, 0]]) @ B
    r = static_correlations(m, [], q, kT=1.0, supercell=(6, 6, 1), n_samples=400,
                            n_equil=300, sample_every=2, cross_section=cs, seed=3)
    assert r.sq == pytest.approx(exact, rel=0.12)
    # and it must be FLAT in q, not merely right on average
    assert (r.sq.max() - r.sq.min()) / r.sq.mean() < 0.15


def test_free_spins_are_temperature_independent():
    """The free-spin sum rule has no temperature in it. A T-dependence here would
    mean the sampler, not the estimator, is wrong."""
    m, B = _free_model(1.0)
    q = np.array([[0.13, 0.2, 0], [0.31, 0, 0]]) @ B
    vals = [static_correlations(m, [], q, kT=kT, supercell=(6, 6, 1), n_samples=300,
                                n_equil=300, sample_every=2, seed=5).sq.mean()
            for kT in (0.25, 4.0)]
    assert vals[0] == pytest.approx(vals[1], rel=0.12)
    assert vals[0] == pytest.approx(2.0 / 3, rel=0.12)


@pytest.mark.slow
def test_antiferromagnet_peaks_at_the_ordering_wavevector():
    """Physics, not plumbing: a Neel chain's instantaneous S(q) must peak at the
    zone boundary h = 1/2 (its ordering vector) and be suppressed at h = 0, and the
    contrast must grow as the temperature falls."""
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": 1.0}]
    cfg = {"crystal_structure": {"lattice_vectors": [[4.0, 0, 0], [0, 9.0, 0],
                                                     [0, 0, 9.0]],
                                 "atoms_uc": atoms},
           "interactions": {"heisenberg": [
               {"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": 1.0},
               {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": 1.0}]},
           "magnetic_structure": {"type": "pattern",
                                  "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": [],
           "calculation": {"on_imaginary": "off"}, "tasks": {}}
    m = GenericSpinModel(copy.deepcopy(cfg))
    B = 2 * np.pi * np.linalg.inv(np.array(cfg["crystal_structure"]
                                           ["lattice_vectors"], float)).T
    q = np.array([[0.0, 0, 0], [0.25, 0, 0], [0.5, 0, 0]]) @ B
    contrast = []
    for kT in (2.0, 0.25):
        sq = static_correlations(m, [], q, kT=kT, supercell=(24, 1, 1),
                                 n_samples=600, n_equil=3000, sample_every=5,
                                 cross_section="trace", seed=11).sq
        assert sq[2] > sq[1] > sq[0], f"kT={kT}: not peaked at the zone boundary: {sq}"
        contrast.append(sq[2] / sq[0])
    assert contrast[1] > contrast[0], "AFM correlations must sharpen on cooling"
