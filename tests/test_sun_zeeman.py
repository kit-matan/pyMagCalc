"""The magnetic field was silently dropped in `mode: SUN`.

`SUNModel.from_generic_model` built its on-site terms from sia / sia_matrix /
stevens and nothing else, so the Zeeman term never entered the Hamiltonian: every
SU(N)-in-field calculation quietly solved the ZERO-FIELD problem. The entangled
engine had always applied it (`sun/entangled.py`), which is what makes the omission
a plain oversight rather than a design choice.

WHY NOTHING CAUGHT IT. The SU(N) suite is thorough -- Gate 1/2/3, FeI2 bands and
intensities against Sunny to 1e-4 -- and not one of those models applies a field.
FeI2 included. A field-free test suite cannot see a dropped field term, however
many tests it has. Found only when porting Sunny tutorial 06, whose skyrmions exist
because a field competes with an easy-plane anisotropy: without it the texture
decayed to the trivial |m=0> state.

THE FIX IS PINNED BY THE LOAD-BEARING SU(N) IDENTITY. At S = 1/2 (N = 2) SU(N) LSWT
is *identical* to dipole LSWT -- the gate that anchors `test_sun.py`. Extending it to
a finite field checks the term's PRESENCE and its SIGN at once, against an engine
whose Zeeman convention is itself pinned (`test_zeeman_calibration.py`). Measured
agreement: exactly 0.0.
"""
import copy

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel

LAT = [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]]
NN = [(["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]),
      (["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0])]
HS = (0.13, 0.31, 0.5)


def _cfg(mode, h_mag, S=0.5, h_dir=(0, 0, 1)):
    return {"crystal_structure": {"lattice_vectors": LAT,
                                  "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                                "spin_S": S},
                                               {"label": "B", "pos": [0.5, 0, 0],
                                                "spin_S": S}]},
            "interactions": {"heisenberg": [{"pair": p, "rij_offset": o, "value": -1.0}
                                            for p, o in NN]},
            "parameters": {"H_mag": h_mag, "H_dir": list(h_dir)},
            "parameter_order": ["H_mag", "H_dir"],
            "magnetic_structure": {"type": "pattern",
                                   "pattern_type": "ferromagnetic",
                                   "direction": [0, 0, 1]},
            "calculation": {"mode": mode, "on_imaginary": "off"}, "tasks": {}}


def _qs():
    B = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T
    return [np.array([h, 0, 0]) @ B for h in HS]


def _params(h_mag, h_dir=(0, 0, 1)):
    return [h_mag, float(h_dir[0]), float(h_dir[1]), float(h_dir[2])]


def _dipole_bands(h_mag, S, h_dir=(0, 0, 1)):
    cfg = _cfg("dipole", h_mag, S, h_dir)
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base=f"zee_d{h_mag}_{S}_{h_dir}",
                      hamiltonian_params=_params(h_mag, h_dir))
    return np.sort(np.real(calc.calculate_dispersion(_qs()).energies), axis=1)


def _sun_bands(h_mag, S, h_dir=(0, 0, 1)):
    cfg = _cfg("SUN", h_mag, S, h_dir)
    mdl = SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(cfg)),
                                      params=_params(h_mag, h_dir))
    return np.sort(np.array([np.sort(np.real(mdl.dispersion(q))) for q in _qs()]),
                   axis=1)


@pytest.mark.parametrize("h_mag", [0.0, 2.0, 5.0])
def test_spin_half_sun_equals_dipole_in_a_field(h_mag):
    """THE gate, extended to finite field. Presence and sign in one assertion."""
    assert _sun_bands(h_mag, 0.5) == pytest.approx(_dipole_bands(h_mag, 0.5), abs=1e-12)


@pytest.mark.parametrize("h_dir", [(0, 0, 1), (1, 0, 0), (0, 1, 0),
                                  (0.3, -0.5, 0.81)])
def test_field_direction_is_respected(h_dir):
    """A SECOND BUG, surfaced by fixing the first, and now fixed too.

    `_resolve_param_map` FLATTENS vector-valued parameters, so `H_dir: [1, 0, 0]`
    came back as the scalar 1.0. `_resolve_field` tested `isinstance(h_dir, list)` on
    that scalar, the test failed, and it fell through to a hardcoded [0, 0, H]:
    EVERY field was silently forced along +z whatever `H_dir` said. It affected both
    engines and every consumer of `_resolve_field` -- SU(N), entangled, thermal_mc
    and annealing.

    `H_dir` is deliberately NOT normalized: the dipole engine (the pinned reference)
    uses it as given, and normalizing here made the two disagree by |H_dir| -- 0.2%
    for [0.3, -0.5, 0.81]. Small, plausible, and hard to notice later.
    """
    assert _sun_bands(4.0, 0.5, h_dir) == pytest.approx(
        _dipole_bands(4.0, 0.5, h_dir), abs=1e-12)


def test_transverse_field_is_not_just_the_z_field():
    """Guards the guard: with the direction ignored, every one of the cases above
    returned the SAME numbers as a field along z, so agreement alone proved nothing.
    """
    z = _sun_bands(4.0, 0.5, (0, 0, 1))
    x = _sun_bands(4.0, 0.5, (1, 0, 0))
    assert np.abs(x - z).max() > 1e-6


def test_the_field_actually_changes_the_sun_spectrum():
    """The regression itself, stated directly: before the fix the SU(N) spectrum was
    bit-identical with and without a field, because the term never entered."""
    assert np.abs(_sun_bands(5.0, 0.5) - _sun_bands(0.0, 0.5)).max() > 0.1


def test_zeeman_convention_matches_the_dipole_engine_across_field_strengths():
    """Rather than assert a shape for the field dependence, assert that SU(N)
    reproduces the DIPOLE engine's, whose Zeeman convention is pinned separately by
    `test_zeeman_calibration.py`.

    I tried twice to assert something stronger and was wrong both times. The field
    dependence here is neither linear nor monotone -- band[0] runs
    0.0822 -> 0.0335 -> 0.1493 -> 0.3808 as H goes 0 -> 1 -> 2 -> 4 -- because the
    convention is E = +gamma mu_B H.S, so moments prefer to lie ANTI-parallel to the
    field, and this model supplies them parallel. The state is therefore not the
    field's ground state and the gap closes before it reopens. That is a property of
    the test model, not of the Zeeman term, and it is exactly the kind of "obvious"
    expectation that should be checked against the engine rather than asserted.
    """
    for h in (0.0, 1.0, 2.0, 4.0, 8.0):
        assert _sun_bands(h, 0.5) == pytest.approx(_dipole_bands(h, 0.5), abs=1e-12)


@pytest.mark.slow
def test_higher_spin_sun_in_field_matches_dipole_without_anisotropy():
    """With no single-ion term, SU(N) reproduces the dipole bands for any S (the
    extra flat Delta-m modes aside), so the field must agree there too."""
    for S in (1.0, 1.5):
        dip = _dipole_bands(3.0, S)
        sun = _sun_bands(3.0, S)
        for iq in range(len(HS)):
            for e in dip[iq]:
                assert np.min(np.abs(sun[iq] - e)) < 1e-9, f"S={S} band {e}"
