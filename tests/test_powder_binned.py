"""Sample-resolved powder averaging (numerical.powder_sample_modes).

Regression for a user-reported physics mismatch: the powder task averaged the mode
ENERGIES over the sphere before broadening, collapsing Cu5SbO6's 10 meV-wide triplon
band into a ~1 meV blob at J1 — contradicting the published powder spectrum
(Piyakulworawat et al., PRR 8, 013247 (2026), Fig. 5). The fix keeps every sphere
direction's own mode energies (SpinW `powspec` convention); plots and fits broaden
those.

Oracles:
  * the ISOLATED DIMER's energy-integrated powder intensity must follow the exact
    textbook interference factor 1 − sin(Qd)/(Qd) — this pins sphere sampling,
    the entangled structure factor, and the representation, analytically;
  * Cu5SbO6 (the shipped example): the |Q|-integrated spectrum must peak at the
    harmonic van Hove energies ≈14.92/17.94 meV — the paper's measured M1 = 15.0(1)
    and M2 = 18.0(2) — with the band spanning ~11–21 meV and negligible weight
    outside;
  * plumbing identity: one sample direction ≡ calculate_sqw at that direction.
"""
import os

import numpy as np
import pytest
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import fibonacci_sphere_points, powder_sample_modes
from magcalc.plotting import broaden_spectrum
from magcalc.sun.entangled import EntangledCalculator

HERE = os.path.dirname(__file__)


def _isolated_dimer(d_frac=0.2, a=30.0, J=16.5):
    cfg = {"crystal_structure": {"lattice_vectors": [[a, 0, 0], [0, a, 0], [0, 0, a]],
            "atoms_uc": [{"label": "A", "pos": [0., 0, 0], "spin_S": 0.5},
                         {"label": "B", "pos": [d_frac, 0, 0], "spin_S": 0.5}]},
        "interactions": {"heisenberg": [
            {"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": J},
            {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]},
        "calculation": {"mode": "entangled"}, "units": [["A", "B"]]}
    m = GenericSpinModel(cfg)
    return EntangledCalculator(m, cfg, []), d_frac * a


def test_dimer_powder_interference_factor():
    """Energy-integrated powder intensity of an isolated dimer = 1 − sin(Qd)/(Qd)."""
    calc, d = _isolated_dimer()
    qm = np.linspace(0.3, 3.0, 12)
    E, I = powder_sample_modes(calc, qm, num_samples=300)
    tot = I.sum(axis=1)
    ana = 1.0 - np.sin(qm * d) / (qm * d)
    ratio = tot / ana
    assert (ratio.max() - ratio.min()) / ratio.mean() < 5e-3, \
        "powder |Q| modulation deviates from the exact dimer interference factor"


def test_cu5sbo6_powder_matches_published_spectrum():
    """The shipped Cu5SbO6 example's powder spectrum must reproduce the published
    band (~11–21 meV) and DOS peaks (measured 15.0(1)/18.0(2); harmonic van Hove
    14.92/17.94). Before the fix everything collapsed to ~15.8–16.7 meV."""
    doc = yaml.safe_load(open(os.path.join(
        HERE, "..", "examples", "entangled", "Cu5SbO6", "config.yaml")))
    m = GenericSpinModel(doc)
    calc = EntangledCalculator(m, doc,
                               [doc["parameters"][k] for k in doc["parameter_order"]])
    qm = np.linspace(1.0, 3.0, 12)
    E, I = powder_sample_modes(calc, qm, num_samples=100)
    grid = np.arange(8.0, 24.0, 0.05)
    spec = np.zeros_like(grid)
    for u in range(len(qm)):
        spec += broaden_spectrum(E[u], I[u], grid, width=0.7, kind="gaussian")

    from scipy.signal import find_peaks
    pk, _ = find_peaks(spec, height=0.3 * spec.max())
    peaks = grid[pk]
    assert len(peaks) == 2, f"expected 2 DOS peaks, got {peaks}"
    assert abs(peaks[0] - 14.92) < 0.3, f"M1 peak at {peaks[0]} (paper: 15.0(1))"
    assert abs(peaks[1] - 17.94) < 0.3, f"M2 peak at {peaks[1]} (paper: 18.0(2))"
    # band edges: significant weight from ~11 to ~21 meV, nothing outside
    nz = grid[spec > 0.02 * spec.max()]
    assert 10.4 < nz[0] < 11.6 and 20.2 < nz[-1] < 21.6
    outside = spec[(grid < 10.6) | (grid > 21.0)].sum()
    assert outside / spec.sum() < 0.02


def test_one_sample_equals_sqw_at_that_direction():
    """Plumbing identity: num_samples=1 must reproduce calculate_sqw exactly."""
    calc, _d = _isolated_dimer()
    qm = np.array([1.3])
    E, I = powder_sample_modes(calc, qm, num_samples=1)
    q = fibonacci_sphere_points(1.3, 1)
    ref = calc.calculate_sqw(q)
    assert np.allclose(np.sort(E[0]), np.sort(ref.energies[0]), atol=1e-12)
    assert np.allclose(np.sort(I[0]), np.sort(ref.intensities[0]), atol=1e-12)
