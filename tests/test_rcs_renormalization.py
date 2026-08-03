"""Classical-to-quantum (RCS) renormalization of on-site and biquadratic terms.

Sunny runs TWO dipole modes. Its default `:dipole` rescales every rank-k on-site
Stevens coefficient by lambda_k(s) = 1 - 1/(2s) + ... (D. Dahlbom et al.,
arXiv:2304.03874) so that dipole LSWT reproduces the exact `:SUN` answer;
`:dipole_uncorrected` does not. pyMagCalc's dipole engine has always been the
UNCORRECTED one -- which matches SpinW, and is what every config in this repo
means, but is NOT what a Sunny user gets by default. Nothing recorded the
difference except one line in a docstring, and it is not small:

    s = 1,   D = -0.5 (S.z)^2  ->  lambda_2 = 1/2,      every band off by 0.5 meV
    s = 2,   B40 O_4^0         ->  lambda_4 = 0.09375,  the gap off by 8.6x
                                                        (13.13 meV vs 1.53 meV)

`calculation: {anisotropy_renormalization: rcs}` now selects Sunny's `:dipole`;
the default 'none' keeps the historical (SpinW) behaviour. Both branches are
pinned here so neither can drift.

Reference values from Sunny 0.8.1, same 2-site FM chain in both codes:

    cryst = Crystal(lattice_vectors(6, 9, 9, 90, 90, 90), [[0,0,0],[0.5,0,0]], 1)
    sys = System(cryst, [1 => Moment(s=s, g=2), 2 => Moment(s=s, g=2)], mode)
    set_exchange!(sys, -1.0, Bond(1,2,[0,0,0]); biquad)
    set_exchange!(sys, -1.0, Bond(2,1,[1,0,0]); biquad)
    set_onsite_coupling!(sys, op, 1); set_onsite_coupling!(sys, op, 2)
    polarize_spins!(sys, (0,0,1))
    dispersion(SpinWaveTheory(sys; measure=nothing), qs)
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.stevens import rcs_lambda

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = [[0.13, 0, 0], [0.31, 0, 0], [0.5, 0, 0]]

# (term, 2s, mode) -> per-q bands (ascending), q = HS
SUNNY = {
    ("sia", 2, "uncorrected"): [[1.1644908, 4.8355093], [1.8758333, 4.1241668],
                                [3.0000000, 3.0000000]],
    ("sia", 2, "rcs"): [[0.6644908, 4.3355093], [1.3758333, 3.6241668],
                        [2.5000000, 2.5000000]],
    ("sia", 3, "uncorrected"): [[1.7467361, 7.2532639], [2.8137499, 6.1862501],
                                [4.5000000, 4.5000000]],
    ("sia", 3, "rcs"): [[1.2467361, 6.7532639], [2.3137499, 5.6862501],
                        [4.0000000, 4.0000000]],
    ("sia", 5, "uncorrected"): [[2.9112269, 12.0887731], [4.6895831, 10.3104169],
                                [7.5000000, 7.5000000]],
    ("sia", 5, "rcs"): [[2.4112269, 11.5887731], [4.1895831, 9.8104169],
                        [7.0000000, 7.0000000]],
    ("stevens40", 4, "uncorrected"): [[13.1289815, 20.4710185],
                                      [14.5516665, 19.0483335],
                                      [16.8000000, 16.8000000]],
    ("stevens40", 4, "rcs"): [[1.5289815, 8.8710185], [2.9516665, 7.4483335],
                              [5.2000000, 5.2000000]],
    ("stevens40", 5, "uncorrected"): [[25.4112269, 34.5887731],
                                      [27.1895831, 32.8104169],
                                      [30.0000000, 30.0000000]],
    ("stevens40", 5, "rcs"): [[5.2112269, 14.3887731], [6.9895831, 12.6104169],
                              [9.8000000, 9.8000000]],
    ("biquad", 2, "uncorrected"): [[0.2631852, 6.1368148], [1.4013332, 4.9986668],
                                   [3.2000000, 3.2000000]],
    ("biquad", 2, "rcs"): [[0.1644908, 3.8355093], [0.8758333, 3.1241668],
                           [2.0000000, 2.0000000]],
    ("biquad", 3, "uncorrected"): [[0.5798299, 13.5201701], [3.0873122, 11.0126878],
                                   [7.0500000, 7.0500000]],
    ("biquad", 3, "rcs"): [[0.3577674, 8.3422326], [1.9049373, 6.7950627],
                           [4.3500000, 4.3500000]],
}


def _cfg(term, S, renorm):
    atoms = [{"label": "A", "pos": [0.0, 0, 0], "spin_S": S},
             {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]
    inter = {"heisenberg": [{"pair": p, "rij_offset": o, "value": -1.0}
                            for p, o in NN]}
    if term == "sia":
        inter["single_ion_anisotropy"] = [
            {"value": -0.5, "axis": [0, 0, 1], "atoms": ["A", "B"]}]
    elif term == "stevens40":
        inter["stevens"] = [{"B": {"4,0": -0.02}, "atoms": ["A", "B"]}]
    elif term == "biquad":
        inter["biquadratic"] = [{"pair": p, "rij_offset": o, "value": -0.3}
                                for p, o in NN]
    return {"crystal_structure": {"lattice_vectors": LAT, "atoms_uc": atoms},
            "interactions": inter, "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern",
                                   "pattern_type": "ferromagnetic",
                                   "direction": [0, 0, 1]},
            "calculation": {"on_imaginary": "off",
                            "anisotropy_renormalization": renorm},
            "tasks": {}}


def _bands(cfg, S, tag):
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=[])
    A = np.array(LAT, float)
    B = 2 * np.pi * np.linalg.inv(A).T
    e = np.real(calc.calculate_dispersion([np.array(q) @ B for q in HS]).energies)
    return np.sort(e, axis=1)


@pytest.mark.parametrize("key", sorted(SUNNY))
def test_matches_sunny(key):
    """Both branches, against the mode of Sunny each one claims to be."""
    term, two_s, mode = key
    renorm = "none" if mode == "uncorrected" else "rcs"
    got = _bands(_cfg(term, two_s / 2.0, renorm), two_s / 2.0,
                 f"rcs_{term}_{two_s}_{mode}")
    assert got == pytest.approx(np.array(SUNNY[key]), abs=2e-6)


def test_default_is_uncorrected():
    """Omitting the key must not change any existing config's meaning."""
    default = _bands(_cfg("sia", 1.0, "none"), 1.0, "rcs_default_a")
    cfg = _cfg("sia", 1.0, "none")
    cfg["calculation"].pop("anisotropy_renormalization")
    assert _bands(cfg, 1.0, "rcs_default_b") == pytest.approx(default, abs=1e-12)


def test_quadratic_anisotropy_vanishes_at_spin_half():
    """EXACT identity, no oracle: (S.n)^2 is a constant for s = 1/2, so a
    quadratic single-ion anisotropy can have no effect at all -- and indeed
    lambda_2(1/2) = 0. The un-renormalized classical polynomial says otherwise,
    which is the clearest statement of what the correction is for."""
    assert rcs_lambda(2, 0.5) == 0.0
    with_sia = _bands(_cfg("sia", 0.5, "rcs"), 0.5, "rcs_half_a")
    cfg = _cfg("sia", 0.5, "rcs")
    cfg["interactions"].pop("single_ion_anisotropy")
    assert with_sia == pytest.approx(_bands(cfg, 0.5, "rcs_half_b"), abs=1e-12)


def test_no_effect_without_onsite_or_biquadratic_terms():
    """A pure Heisenberg model must be untouched by the flag -- the factor
    applies to rank-k on-site terms, never to bilinear exchange."""
    plain = _cfg("none", 1.0, "none")
    rcs = _cfg("none", 1.0, "rcs")
    assert _bands(plain, 1.0, "rcs_none_a") == pytest.approx(
        _bands(rcs, 1.0, "rcs_none_b"), abs=1e-12)


def test_lambda_values_match_sunny_rcs_factors():
    """lambda_k transcription, against Sunny's `rcs_factors` (OnsiteCoupling.jl).

        julia> Sunny.rcs_factors(s)[k]
    """
    expected = {              # (k, s) -> lambda, verbatim from Sunny
        (2, 0.5): 0.0, (2, 1.0): 0.5, (2, 1.5): 0.66666666666666674,
        (2, 2.5): 0.80000000000000004,
        (4, 2.0): 0.09375, (4, 2.5): 0.19200000000000006,
        (6, 3.0): 0.01543209876543207,
    }
    for (k, s), lam in expected.items():
        assert rcs_lambda(k, s) == pytest.approx(lam, rel=1e-12)


def test_bad_value_raises():
    with pytest.raises(ValueError, match="anisotropy_renormalization"):
        GenericSpinModel(_cfg("sia", 1.0, "yes-please"))
