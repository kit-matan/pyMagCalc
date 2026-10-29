"""The classical energy as a quadratic form, numeric and symbolic (2026-10-29).

`MagCalc._extract_classical_quadratic` gives E(m) = 0.5 m^T H m + b^T m + c for
the minimizer, the annealer and the energy-audit guard. It used to build the
whole classical Hamiltonian in SymPy on every run, twice: 17 s of a 24 s warm
FeI2 run. `calculation.classical_energy: numeric` (the default) now sums the
bond-bilinear part from the interaction arrays and builds only the on-site,
Zeeman and biquadratic terms in SymPy; `symbolic` keeps the old build as an
option. The result is memoized on the spin model the runner's two MagCalc
instances share.

References, kept apart:

  * the symbolic build itself, on every shipped example config (slow) and a
    fast subset that covers DM, a full interaction matrix, single-ion
    anisotropy, a field, mixed exchange and a rotating-frame spiral;
  * closed forms the code never sees: SW29's two-site chain,
    E = 2J mA.mB + K (mA_z^2 + mB_z^2) + 2 mu_B B (mA_z + mB_z) at random
    (not unit) m; the kagome q = 0 state, -(J1 + J2) S^2 per site on SW07;
    and SW09's antisymmetric exchange, which splits the two vector chiralities
    to -J S^2 -+ sqrt(3) |D| S^2 per site whatever the DM sign convention.
"""
import copy
import glob
import os

import numpy as np
import pytest
import yaml

import magcalc as mc
from magcalc import runner
from magcalc.constants import MU_B
from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(__file__)
EX = os.path.join(HERE, "..", "examples")


def _path(*parts):
    return os.path.join(EX, *parts)


FAST = [
    _path("spinw_tutorials", "SW09_kagome_AFM_DM", "config.yaml"),
    _path("spinw_tutorials", "SW29_AFM_chain_field", "config.yaml"),
    _path("spinw_tutorials", "SW12_triangular_easyplane", "config.yaml"),
    _path("spinw_tutorials", "SW20_Yb2Ti2O7", "config_field.yaml"),
    _path("spinw_tutorials", "SW03_frustrated_chain", "config_spiral.yaml"),
    _path("materials", "KFe3J", "config_kfe3j.yaml"),
    _path("sunny_tutorials", "S01_CoRh2O4", "config.yaml"),
]


def _every_config():
    out = []
    for p in sorted(glob.glob(os.path.join(EX, "**", "*.yaml"), recursive=True)):
        if any(s in p for s in ("/plots/", "/cache/", "/data/", "/reference/",
                                "future_exmaples", "SpinW_examples", "_params.yaml")):
            continue
        cfg = yaml.safe_load(open(p))
        if isinstance(cfg, dict) and cfg.get("crystal_structure") \
                and not cfg.get("python_model_file"):
            out.append(p)
    return out


def _calc(path, calculation=None):
    cfg = yaml.safe_load(open(path))
    if calculation is not None:
        cfg["calculation"] = {**(cfg.get("calculation") or {}), **calculation}
    m = GenericSpinModel(copy.deepcopy(cfg), base_path=os.path.dirname(path))
    S, p = runner.resolve_S_and_params(m.config)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, hamiltonian_params=p,
                      cache_file_base="classical_energy_test", cache_mode="none",
                      initialize=False)
    return calc, len(m.atom_pos()), len(m.atom_pos_ouc()), float(calc.spin_magnitude)


def _assert_same(path):
    calc, n, no, S = _calc(path)
    sym = calc._classical_quadratic_symbolic(calc.hamiltonian_params, n, no, S)
    num = calc._classical_quadratic_numeric(calc.hamiltonian_params, n, no, S)
    assert sym is not None and num is not None and not isinstance(num, str)
    scale = max(1.0, float(np.abs(sym[0]).max()))
    assert np.abs(num[0] - sym[0]).max() < 1e-12 * scale
    assert np.abs(num[1] - sym[1]).max() < 1e-12 * scale
    assert abs(num[2] - sym[2]) < 1e-12 * scale


def _energy(quad, m):
    H, b, c = quad
    m = np.ravel(m)
    return 0.5 * m @ H @ m + b @ m + c


@pytest.mark.parametrize("path", FAST, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_numeric_equals_symbolic(path):
    _assert_same(path)


@pytest.mark.slow
@pytest.mark.parametrize("path", _every_config(),
                         ids=lambda p: os.path.relpath(p, EX).replace(os.sep, "/"))
def test_numeric_equals_symbolic_on_every_example(path):
    _assert_same(path)


def test_sw29_closed_form():
    """Two sites, two A-B bonds per cell, easy axis K (S_z)^2, field B || z."""
    calc, n, no, S = _calc(FAST[1])
    quad = calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S)
    J, K, B = 1.0, -0.1, 7.0
    rng = np.random.default_rng(3)
    for _ in range(20):
        mA, mB = rng.normal(size=3), rng.normal(size=3)
        exact = (2 * J * mA @ mB + K * (mA[2] ** 2 + mB[2] ** 2)
                 + 2 * MU_B * B * (mA[2] + mB[2]))
        assert _energy(quad, [mA, mB]) == pytest.approx(exact, abs=1e-12)


def _q0_state(chirality):
    ang = chirality * np.deg2rad([0.0, 120.0, 240.0])
    return np.stack([np.cos(ang), np.sin(ang), np.zeros(3)], axis=1)


def test_kagome_q0_energy_is_minus_J1_plus_J2():
    calc, n, no, S = _calc(_path("spinw_tutorials", "SW07_kagome_AFM", "config.yaml"))
    quad = calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S)
    for chi in (+1, -1):
        assert _energy(quad, S * _q0_state(chi)) / n == pytest.approx(
            -(1.0 + 0.11) * S ** 2, abs=1e-12)


def test_kagome_dm_splits_the_two_chiralities():
    calc, n, no, S = _calc(FAST[0])
    quad = calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S)
    e = sorted(_energy(quad, S * _q0_state(chi)) / n for chi in (+1, -1))
    d = np.sqrt(3) * 0.08 * S ** 2
    assert e == pytest.approx([-S ** 2 - d, -S ** 2 + d], abs=1e-12)


def test_the_symbolic_option_uses_the_symbolic_build(monkeypatch):
    ref, *_ = _calc(FAST[0])
    calc, n, no, S = _calc(FAST[0], {"classical_energy": "symbolic"})
    num = ref._classical_quadratic_numeric(ref.hamiltonian_params, n, no, S)
    monkeypatch.setattr(type(calc), "_classical_quadratic_numeric",
                        lambda *a, **k: pytest.fail("numeric path used"))
    quad = calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S)
    assert np.abs(quad[0] - num[0]).max() < 1e-12


def test_an_unknown_mode_is_refused():
    calc, n, no, S = _calc(FAST[0], {"classical_energy": "sympy"})
    with pytest.raises(ValueError, match="classical_energy"):
        calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S)


def test_a_numeric_failure_falls_back_to_the_symbolic_build(monkeypatch):
    calc, n, no, S = _calc(FAST[1])
    calls = []
    sym = type(calc)._classical_quadratic_symbolic
    monkeypatch.setattr(type(calc), "_classical_quadratic_numeric", lambda *a, **k: None)
    monkeypatch.setattr(type(calc), "_classical_quadratic_symbolic",
                        lambda self, *a: calls.append(1) or sym(self, *a))
    assert calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S) is not None
    assert calls == [1]


def test_the_form_is_built_once_per_model_and_parameters(monkeypatch):
    """The runner's minimization and guard MagCalc share one spin model."""
    calc, n, no, S = _calc(FAST[1])
    other = mc.MagCalc(spin_model_module=calc.sm, spin_magnitude=S,
                       hamiltonian_params=calc.hamiltonian_params,
                       cache_file_base="classical_energy_test", cache_mode="none",
                       initialize=False)
    calls = []
    num = type(calc)._classical_quadratic_numeric
    monkeypatch.setattr(type(calc), "_classical_quadratic_numeric",
                        lambda self, *a: calls.append(1) or num(self, *a))
    q1 = calc._extract_classical_quadratic(calc.hamiltonian_params, n, no, S)
    q1[0][0, 0] += 99.0                       # a caller's edit must not reach the memo
    q2 = other._extract_classical_quadratic(other.hamiltonian_params, n, no, S)
    assert calls == [1]
    assert q2[0][0, 0] != q1[0][0, 0]
    p = list(calc.hamiltonian_params)
    p[0] = 2.0                                # J changes -> a new form
    q3 = calc._extract_classical_quadratic(p, n, no, S)
    assert calls == [1, 1]
    assert not np.allclose(q3[0], q2[0])
