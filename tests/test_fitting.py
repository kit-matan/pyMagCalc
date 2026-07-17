"""
Tests for the data-fitting engine (magcalc/fitting.py).

Strategy: build a tiny 1D ferromagnetic chain, generate synthetic data with the
forward model at a known parameter value, then fit from a perturbed start and
assert the true value is recovered. This exercises the full residual / lmfit /
single-calculator-reuse path without depending on any external data file.
"""

import os
from types import SimpleNamespace

import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.runner import compute_b_matrix
from magcalc import fitting
from magcalc.plotting import broaden_spectrum

pytestmark = pytest.mark.filterwarnings("ignore")


def _fm_chain_config(J1=1.0):
    """A minimal 1D FM chain along a (single magnetic site)."""
    return {
        "crystal_structure": {
            "lattice_parameters": {"a": 4.0, "b": 4.0, "c": 4.0,
                                   "alpha": 90, "beta": 90, "gamma": 90},
            "atoms_uc": [{"label": "Fe1", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}],
        },
        "interactions": [
            {"type": "heisenberg", "pair": ["Fe1", "Fe1"],
             "value": "J1", "rij_offset": [1, 0, 0]},
        ],
        "parameters": {"J1": J1, "S": 1.0},
    }


def _build_calc(config, base_path, cache_base):
    spin_model = GenericSpinModel(config, base_path=base_path)
    params = [config["parameters"]["J1"]]
    calc = mc.MagCalc(
        spin_model_module=spin_model,
        spin_magnitude=1.0,
        hamiltonian_params=params,
        cache_file_base=cache_base,
        cache_mode="none",
    )
    B = compute_b_matrix(spin_model)
    return calc, B


def test_broaden_spectrum_basic():
    grid = np.linspace(-5, 5, 501)
    out = broaden_spectrum(np.array([0.0]), np.array([1.0]), grid, width=0.4, kind="lorentzian")
    # Peak at the center, integrates to ~weight, symmetric.
    assert grid[np.argmax(out)] == pytest.approx(0.0, abs=0.05)
    assert _trapezoid(out, grid) == pytest.approx(1.0, rel=0.05)
    # Empty centers -> zeros.
    assert np.all(broaden_spectrum(np.array([]), np.array([]), grid) == 0)


def test_canonical_name_order():
    cfg = {"parameters": {"J1": 1.0, "J2": 2.0, "S": 0.5}}
    assert fitting.canonical_name_order(cfg) == ["J1", "J2"]
    cfg2 = {"parameters": {"J1": 1.0, "J2": 2.0, "S": 0.5},
            "parameter_order": ["J2", "J1", "S"]}
    assert fitting.canonical_name_order(cfg2) == ["J2", "J1"]


def test_dispersion_roundtrip(tmp_path):
    true_J1 = 1.3
    cfg_true = _fm_chain_config(J1=true_J1)
    calc_true, B = _build_calc(cfg_true, str(tmp_path), "fit_test_true")

    # Synthetic data: dispersion at a handful of q-points along (h,0,0).
    hkl = np.array([[h, 0.0, 0.0] for h in np.linspace(0.05, 0.45, 8)])
    q_cart = np.dot(hkl, B)
    res = calc_true.calculate_dispersion(q_cart, serial=True)
    E = np.real(np.asarray(res.energies))[:, 0]  # single mode

    data_file = tmp_path / "disp.txt"
    rows = np.column_stack([hkl, E, np.full_like(E, 0.02)])
    np.savetxt(data_file, rows, delimiter=",", header="h,k,l,E,sigma")

    # Fit from a perturbed start with a fresh calculator.
    cfg_fit = _fm_chain_config(J1=0.7)  # wrong start
    calc_fit, B2 = _build_calc(cfg_fit, str(tmp_path), "fit_test_fit")
    final_config = dict(cfg_fit)
    final_config["fitting"] = {
        "type": "dispersion",
        "data_file": str(data_file),
        "vary": ["J1"],
        "match": "nearest",
        "method": "leastsq",
    }

    out = fitting.run_fit(final_config, calc_fit, ["J1"], B2, config_dir=str(tmp_path))
    assert out["best_values"]["J1"] == pytest.approx(true_J1, rel=1e-3)
    # Calculator was reused (single instance) and ended at the optimum.
    assert calc_fit.hamiltonian_params[0] == pytest.approx(true_J1, rel=1e-2)


class _FakeCalc:
    """
    Minimal stand-in for MagCalc that returns fixed, nonzero mode energies and
    intensities. Lets us test the intensity-fitting machinery (Q-grouping,
    shared broadening, scale/background, output caching) deterministically and
    fast, independent of the LSWT physics (which gives ~zero S(Q,w) intensity
    for the toy FM chain, making `scale` unidentifiable there).
    """

    def __init__(self, energies, intensities):
        self._e = np.asarray(energies, dtype=float)
        self._i = np.asarray(intensities, dtype=float)
        self.hamiltonian_params = [1.0]
        self.n_sqw_calls = 0

    def update_hamiltonian_params(self, p):
        self.hamiltonian_params = list(p)

    def calculate_sqw(self, q_cart, backend="numpy", **kwargs):
        # **kwargs absorbs the measurement model (temperature / domains /
        # cross_section) that FitProblem now forwards to the real MagCalc.
        # Direction-independent fake: each q maps to the shell whose |Q| is
        # nearest 0.1/0.2/0.3, so the sample-resolved powder path (which asks for
        # sphere points around each |Q|) sees the same fixed modes per shell.
        self.n_sqw_calls += 1
        qs = np.atleast_2d(np.asarray(q_cart, dtype=float))
        mags = np.linalg.norm(qs, axis=1)
        idx = np.clip(np.round(mags / 0.1).astype(int) - 1, 0, len(self._e) - 1)
        return SimpleNamespace(q_vectors=qs,
                               energies=self._e[idx], intensities=self._i[idx])


def _intensity_fit(fit_type, tmp_path):
    """Shared body for the sqw/powder intensity-fitting tests."""
    # 3 distinct Q-points, 2 modes each.
    energies = np.array([[2.0, 5.0], [3.0, 6.0], [4.0, 7.0]])
    intensities = np.array([[1.0, 0.4], [0.8, 0.5], [0.6, 0.3]])
    calc = _FakeCalc(energies, intensities)

    true_scale, true_bg, width = 3.0, 0.1, 0.4
    e_axis = np.linspace(0.0, 9.0, 24)
    rows = []
    for u in range(3):
        spec = broaden_spectrum(energies[u], intensities[u], e_axis, width=width)
        I = true_scale * spec + true_bg
        for j, e in enumerate(e_axis):
            if fit_type == "sqw":
                rows.append([0.1 * (u + 1), 0.0, 0.0, e, I[j], 0.01])
            else:  # powder: single |Q| column
                rows.append([0.1 * (u + 1), e, I[j], 0.01])
    header = "h,k,l,energy,intensity,error" if fit_type == "sqw" else "|Q|,energy,intensity,error"
    data_file = tmp_path / f"{fit_type}.txt"
    np.savetxt(data_file, np.array(rows), delimiter=",", header=header)

    final_config = {
        "parameters": {"J1": 1.0, "S": 1.0},
        "fitting": {
            "type": fit_type,
            "data_file": str(data_file),
            "vary": [],  # Hamiltonian fixed; fit only intensity nuisance params
            "method": "leastsq",
            "scale": {"value": 1.0, "vary": True},
            "background": {"value": 0.0, "vary": True},
            "energy_broadening": {"value": width, "vary": False},
        },
    }
    B = np.eye(3)
    out = fitting.run_fit(final_config, calc, ["J1"], B, config_dir=str(tmp_path))
    assert out["result"].params["scale"].value == pytest.approx(true_scale, rel=1e-3)
    assert out["result"].params["background"].value == pytest.approx(true_bg, abs=1e-3)
    # Hamiltonian never changed -> only one forward-model call (caching works).
    assert calc.n_sqw_calls == 1


def test_sqw_intensity_fit(tmp_path):
    _intensity_fit("sqw", tmp_path)


def test_powder_intensity_fit(tmp_path):
    _intensity_fit("powder", tmp_path)


# --------------------------------------------------------------------------- #
# Fast dispersion evaluator (compile once over q+S+params)
# --------------------------------------------------------------------------- #
def test_fast_evaluator_matches_dispersion(tmp_path):
    """DispersionEvaluator must reproduce calculate_dispersion exactly, both at
    the compile-time parameters and at new parameters passed per call."""
    cfg = _fm_chain_config(J1=1.1)
    calc, B = _build_calc(cfg, str(tmp_path), "fast_eval_test")
    hkl = np.array([[h, 0.0, 0.0] for h in np.linspace(0.05, 0.45, 6)])
    q_cart = np.dot(hkl, B)

    ev = calc.compile_dispersion_evaluator()

    E_slow = np.real(np.asarray(calc.calculate_dispersion(q_cart, serial=True).energies))
    E_fast = ev.energies(q_cart)
    assert np.allclose(E_fast, E_slow, atol=1e-9)

    # New parameters, without mutating the calculator.
    E_fast2 = ev.energies(q_cart, [1.7])
    assert calc.hamiltonian_params[0] == pytest.approx(1.1)
    calc.update_hamiltonian_params([1.7])
    E_slow2 = np.real(np.asarray(calc.calculate_dispersion(q_cart, serial=True).energies))
    assert np.allclose(E_fast2, E_slow2, atol=1e-9)

    # Argument-count guard.
    with pytest.raises(ValueError):
        ev.energies(q_cart, [1.7, 0.3])


def test_dispersion_fit_fast_matches_slow(tmp_path):
    """run_fit must give the same optimum with the fast path on and off, and
    must leave the shared calculator synchronized with the optimum."""
    true_J1 = 1.3
    cfg_true = _fm_chain_config(J1=true_J1)
    calc_true, B = _build_calc(cfg_true, str(tmp_path), "fastslow_true")

    hkl = np.array([[h, 0.0, 0.0] for h in np.linspace(0.05, 0.45, 8)])
    q_cart = np.dot(hkl, B)
    res = calc_true.calculate_dispersion(q_cart, serial=True)
    E = np.real(np.asarray(res.energies))[:, 0]

    data_file = tmp_path / "disp_fastslow.txt"
    rows = np.column_stack([hkl, E, np.full_like(E, 0.02)])
    np.savetxt(data_file, rows, delimiter=",", header="h,k,l,E,sigma")

    results = {}
    for fast in (True, False):
        cfg_fit = _fm_chain_config(J1=0.7)
        calc_fit, B2 = _build_calc(cfg_fit, str(tmp_path), f"fastslow_{fast}")
        final_config = dict(cfg_fit)
        final_config["fitting"] = {
            "type": "dispersion",
            "data_file": str(data_file),
            "vary": ["J1"],
            "match": "nearest",
            "method": "leastsq",
            "fast": fast,
        }
        out = fitting.run_fit(final_config, calc_fit, ["J1"], B2,
                              config_dir=str(tmp_path))
        results[fast] = out["best_values"]["J1"]
        # calculator ends synchronized with the optimum on both paths
        assert calc_fit.hamiltonian_params[0] == pytest.approx(true_J1, rel=1e-3)
        # fast path actually compiled an evaluator (and vice versa)
        assert (out["problem"]._fast_eval is not None) == fast

    assert results[True] == pytest.approx(true_J1, rel=1e-3)
    assert results[True] == pytest.approx(results[False], rel=1e-6)


# --------------------------------------------------------------------------- #
# Regression: update_hamiltonian_params must flatten vector parameters so the
# stored list matches params_sym_flat (else _calculate_numerical_ud mismatches
# for any model with a field-direction vector; surfaced by post-fit sync).
# --------------------------------------------------------------------------- #
def _fm_chain_field_config(J1=1.0):
    """1D FM chain with a Zeeman field, so H_dir is a *vector* parameter."""
    return {
        "parameter_order": ["J1", "H_mag", "H_dir"],
        "parameters": {"J1": J1, "H_mag": 0.0, "H_dir": [0, 0, 1], "S": 1.0},
        "crystal_structure": {
            "lattice_parameters": {"a": 4.0, "b": 4.0, "c": 4.0,
                                   "alpha": 90, "beta": 90, "gamma": 90},
            "atoms_uc": [{"label": "Fe1", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0}],
        },
        "interactions": [
            {"type": "heisenberg", "pair": ["Fe1", "Fe1"],
             "value": "J1", "rij_offset": [1, 0, 0]},
        ],
    }


def test_update_hamiltonian_params_flattens_vector(tmp_path):
    cfg_a = _fm_chain_field_config(J1=1.5)
    sm_a = GenericSpinModel(cfg_a, base_path=str(tmp_path))
    calc_a = mc.MagCalc(spin_model_module=sm_a, spin_magnitude=1.0,
                        hamiltonian_params=[1.5, 0.0, [0, 0, 1]],
                        cache_file_base="upd_a", cache_mode="none")

    cfg_b = _fm_chain_field_config(J1=1.0)
    sm_b = GenericSpinModel(cfg_b, base_path=str(tmp_path))
    calc_b = mc.MagCalc(spin_model_module=sm_b, spin_magnitude=1.0,
                        hamiltonian_params=[1.0, 0.0, [0, 0, 1]],
                        cache_file_base="upd_b", cache_mode="none")

    # The vector param must not raise, and must be flattened to match init.
    calc_b.update_hamiltonian_params([1.5, 0.0, [0, 0, 1]])
    assert len(calc_b.hamiltonian_params) == len(calc_b.params_sym_flat)
    assert calc_b.hamiltonian_params == calc_a.hamiltonian_params
    assert all(not isinstance(v, (list, tuple)) for v in calc_b.hamiltonian_params)

    # And the forward model agrees with a freshly-constructed calculator.
    B = compute_b_matrix(sm_a)
    q = np.dot(np.array([[0.2, 0.0, 0.0], [0.35, 0.0, 0.0]]), B)
    Ea = np.real(np.asarray(calc_a.calculate_dispersion(q, serial=True).energies))
    Eb = np.real(np.asarray(calc_b.calculate_dispersion(q, serial=True).energies))
    assert np.allclose(Ea, Eb, atol=1e-9)
