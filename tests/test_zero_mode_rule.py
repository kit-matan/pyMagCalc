"""The exact-zero-mode rule at an antiferromagnetic Bragg point (2026-10-31).

At a Neel magnet's Bragg point the Goldstone mode is a Jordan block of g H: the
true intensity diverges like 1/omega and the convention is to report zero. KKdMatrix
did so only through an ABSOLUTE cut, |N_ii| < 1e-12 on the pseudo-norm. Rounding of
order eps in H lifts the mode to omega ~ sqrt(eps) x scale with N_ii ~ omega /
scale, so which side of the cut it fell on -- zero, or a ~1e9 spike -- was decided
by rounding: the feasibility study of a numeric H(q) builder saw SW28 go from 0 on
SymPy's H to 2.7e9 on the prototype's at +-(1,1,1).

The rule (`linalg.ZERO_MODE_ENERGY_REL`, `ZERO_MODE_NORM`, fMagCalc's `calc_alpha`
line for line): a mode is an exact zero mode when omega <= 1e-5 x max|H_ij| AND
|N_ii| < 1e-3. Pinned on the Neel chain, where every mode at the Bragg point is
zero (which is why the scale is |H|, not the largest eigenvalue):

  * a noise gap H2 -> H2 + eps max|H| I, eps = 1e-14 .. 1e-11 (omega up to 4.5e-6
    of the scale): zero intensity under both backends, and -- with the rule off --
    the 1e6-1e7 spike, so the pin is known to bite;
  * next to the Bragg point, where the mode is physical (omega / scale = 1e-3),
    the intensity is untouched, and it is the chain's closed form
    2 S (1 - cos qd) / |sin qd| (8000 there);
  * a ferromagnet's Goldstone mode (N_ii = 1) keeps its finite intensity down to
    omega -> 0: `test_gapless_backends.py`.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc import linalg, numeric_hq, runner
from magcalc.generic_model import GenericSpinModel

J, S, A = 1.0, 1.0, 6.0


def _neel(monkeypatch):
    cfg = {
        "crystal_structure": {"lattice_vectors": [[A, 0, 0], [0, 9, 0], [0, 0, 9]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                                           {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]},
        "interactions": [
            {"type": "heisenberg", "pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": J},
            {"type": "heisenberg", "pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": J},
            {"type": "heisenberg", "pair": ["A", "B"], "rij_offset": [-1, 0, 0], "value": J},
            {"type": "heisenberg", "pair": ["B", "A"], "rij_offset": [1, 0, 0], "value": J}],
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]},
        "calculation": {"on_imaginary": "off"},
    }
    sm = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = sm.generate_magnetic_structure()
    sm.set_magnetic_structure(th, ph)
    monkeypatch.setenv(numeric_hq.ENV, "symbolic")
    s, p = runner.resolve_S_and_params(sm.config)
    return mc.MagCalc(spin_model_module=sm, spin_magnitude=s, hamiltonian_params=p,
                      cache_file_base="zero_mode_rule", cache_mode="none")


def _stacks(calc, q, eps):
    hp = calc._build_h_stack(q, S)
    hm = calc._build_h_stack(q, S, negate=True)
    g = np.diag([1.0, 1.0, -1.0, -1.0])
    return (hp + eps * np.abs(hp).max(axis=(1, 2))[:, None, None] * g,
            hm + eps * np.abs(hm).max(axis=(1, 2))[:, None, None] * g)


def _numpy_intensity(calc, q, hp, hm):
    """Band-summed trace S^aa(q) from KKdMatrix (positive-energy modes)."""
    out = []
    for k in range(len(q)):
        K, Kd, _ = linalg.KKdMatrix(S, hp[k], hm[k], calc.Ud_numeric, q[k], 2)
        out.append(sum(np.real(np.sum(K[a::3, m]) * np.sum(Kd[a::3, m + 2]))
                       for m in range(2) for a in range(3)))
    return np.array(out)


BRAGG = np.array([[2 * np.pi / A, 0, 0], [-2 * np.pi / A, 0, 0], [6 * np.pi / A, 0, 0]])


@pytest.mark.parametrize("eps", [1e-14, 1e-12, 1e-11])
def test_a_noise_lifted_goldstone_mode_carries_no_intensity(eps, monkeypatch):
    calc = _neel(monkeypatch)
    hp, hm = _stacks(calc, BRAGG, eps)
    assert np.all(np.abs(_numpy_intensity(calc, BRAGG, hp, hm)) == 0)
    monkeypatch.setattr(linalg, "ZERO_MODE_NORM", 0.0)       # the rule off
    assert np.min(np.abs(_numpy_intensity(calc, BRAGG, hp, hm))) > 1e5


def test_the_fortran_backend_applies_the_same_rule(monkeypatch):
    fm = pytest.importorskip("fmagcalc")
    if getattr(fm, "backend", None) != "ctypes":
        pytest.skip("fMagCalc's compiled library is not available")
    calc = _neel(monkeypatch)
    for eps in (1e-14, 1e-11):
        hp, hm = _stacks(calc, BRAGG, eps)
        res = fm.run_sqw(hp, hm, np.asarray(calc.Ud_numeric, complex), np.ones((len(BRAGG), 2)),
                         S, BRAGG)
        assert np.all(np.isfinite(res["intensities"]))
        assert np.all(res["intensities"] == 0)


def test_next_to_the_bragg_point_the_intensity_is_untouched(monkeypatch):
    """omega / scale = 1e-3 there: a physical mode, which the rule must not touch.
    The Heisenberg AFM chain's transverse weight is S sqrt((1 - g)/(1 + g)) =
    S (1 - cos qd) / |sin qd| per spin, g = cos qd, so 2 S (1 - cos qd) / |sin qd|
    per two-site cell -- 8000 here, against 2 S = 2 well away from the peak."""
    calc = _neel(monkeypatch)
    d = A / 2
    w_rel = 1e-3
    dq = np.arcsin(w_rel / 2) / d             # omega = 2 J S |sin(q d)| = w_rel x 2JS
    q = BRAGG + [dq, 0, 0]
    hp, hm = _stacks(calc, q, 0.0)
    on = _numpy_intensity(calc, q, hp, hm)
    monkeypatch.setattr(linalg, "ZERO_MODE_NORM", 0.0)
    off = _numpy_intensity(calc, q, hp, hm)
    assert np.allclose(on, off, rtol=1e-12, atol=0)
    qd = q[:, 0] * d
    closed = 2 * S * (1 - np.cos(qd)) / np.abs(np.sin(qd))
    assert np.allclose(on, closed, rtol=1e-8), (on, closed)
