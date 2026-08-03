"""Absolute S(q,w) normalization against Sunny 0.8.1.

There used to be a documented rule that "pyMagCalc's S(Q,w) is 3/4 of Sunny's --
a pre-existing convention difference", repeated in CLAUDE.md, in GAP_STATUS.md and
in tests/test_polarized.py, together with the instruction not to compare absolute
intensities with Sunny. It was false. The factor came from hardcoded reference
numbers in test_polarized.py that were uniformly 4/3 too large; the tests could not
detect it because they only compared the RATIO chiral/perp, in which any overall
factor cancels, and the caveat then explained the discrepancy away. Measured
against Sunny today, the two codes agree exactly.

So this file exists to make the absolute scale a pinned quantity rather than a
disclaimer: a ferromagnet at three spin values, and a Neel antiferromagnet in two
orientations, band by band.

    latvecs = lattice_vectors(6, 9, 9, 90, 90, 90)
    cryst   = Crystal(latvecs, [[0,0,0],[0.5,0,0]], 1)
    sys     = System(cryst, [1 => Moment(s=s, g=2), 2 => Moment(s=s, g=2)], :dipole)
    set_exchange!(sys, J, Bond(1,2,[0,0,0])); set_exchange!(sys, J, Bond(2,1,[1,0,0]))
    set_dipole!(...)
    intensities_bands(SpinWaveTheory(sys; measure=ssf_perp(sys; apply_g=false)), qs)

One real convention difference remains, and it is NOT an overall factor: Sunny's
`ssf_perp` applies the g-tensor by DEFAULT (`apply_g=true`), measuring magnetic
moments rather than spins -- 4x at g = 2, pinned below. pyMagCalc's S(Q,w) is
spin-only, i.e. Sunny's `apply_g=false`.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = [[0.13, 0, 0], [0.31, 0, 0], [0.45, 0, 0]]

# (case, s) -> per-q list of per-band perp intensities (bands ascending in energy)
SUNNY = {
    ("fm", 0.5): [[0.5000000, 0.0], [0.5000000, 0.0], [0.5000000, 0.0]],
    ("fm", 1.0): [[1.0000000, 0.0], [1.0000000, 0.0], [1.0000000, 0.0]],
    ("fm", 2.0): [[2.0000000, 0.0], [2.0000000, 0.0], [2.0000000, 0.0]],
    ("neel_z", 1.0): [[0.1035450, 0.1035450], [0.2647364, 0.2647364],
                      [0.3939707, 0.4601100]],
    ("neel_x", 1.0): [[0.2070901, 0.2070901], [0.5294727, 0.5294727],
                      [0.8540807, 0.8540807]],
}
# Sunny's DEFAULT ssf_perp (apply_g=true) on the s = 1 ferromagnet: g^2/4 = 4x.
SUNNY_FM_APPLY_G = [[4.0, 0.0], [4.0, 0.0], [4.0, 0.0]]

CASES = {"fm": (-1.0, [0, 0, 1], [0, 0, 1]),
         "neel_z": (1.0, [0, 0, 1], [0, 0, -1]),
         "neel_x": (1.0, [1, 0, 0], [-1, 0, 0])}


def _sqw(case, S, tag):
    J, d1, d2 = CASES[case]
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S},
             {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]
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
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    r = calc.calculate_sqw([np.array(q) @ B for q in HS], cross_section="perp")
    E, I = np.real(r.energies), np.real(r.intensities)
    return np.array([I[i][np.argsort(E[i])] for i in range(len(HS))])


@pytest.mark.parametrize("key", sorted(SUNNY))
def test_total_intensity_per_q_matches_sunny(key):
    """The per-q sum over bands: basis-independent, so it is well defined even
    where two magnons are degenerate (Neel along z at h = 0.45, where the two
    codes split one degenerate pair differently but must agree on the total)."""
    case, S = key
    got = _sqw(case, S, f"abs_{case}_{S}").sum(axis=1)
    assert got == pytest.approx(np.array(SUNNY[key]).sum(axis=1), abs=1e-6)


@pytest.mark.parametrize("key", [("fm", 0.5), ("fm", 1.0), ("fm", 2.0),
                                 ("neel_x", 1.0)])
def test_band_resolved_intensity_matches_sunny(key):
    """Band by band, for the cases with no degenerate-eigenvector ambiguity."""
    case, S = key
    got = _sqw(case, S, f"absb_{case}_{S}")
    assert got == pytest.approx(np.array(SUNNY[key]), abs=1e-6)


def test_there_is_no_three_quarters_factor():
    """The claim this file retires, stated so it cannot come back: the ratio to
    Sunny is 1, not 3/4, and it is 1 for a ferromagnet, a Neel antiferromagnet AND
    the non-collinear helix of test_polarized.py."""
    for key in sorted(SUNNY):
        case, S = key
        ratio = (_sqw(case, S, f"abs34_{case}_{S}").sum()
                 / np.array(SUNNY[key]).sum())
        assert ratio == pytest.approx(1.0, rel=1e-6), f"{key}: ratio {ratio}"


def test_sunny_default_measure_applies_g():
    """Documented, not corrected: Sunny's `ssf_perp` default measures moments
    (g S), so its numbers are g^2/4 = 4x ours at g = 2. Comparing against Sunny
    means passing apply_g=false -- which every reference in this repo does."""
    got = _sqw("fm", 1.0, "abs_g").sum(axis=1)
    assert got == pytest.approx(np.array(SUNNY_FM_APPLY_G).sum(axis=1) / 4.0,
                                abs=1e-6)
