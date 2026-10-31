"""Gapless models: both S(Q,w) backends against exact answers (2026-10-28).

OPEN_WORK listed this as "gapless powder averages differ between backends: fMagCalc
zeroes the intensity of modes below 1e-3 meV and NumPy does not; decide which is right,
against an independent reference, before changing either". Two independent references
decided it, and they found TWO Fortran defects, not one:

1. The 1e-3 meV cut. The ferromagnetic chain's one-magnon intensity is
   (S/2)(1 + Qz_hat^2) EXACTLY at every q, however small its energy. NumPy reproduces
   that down to 4e-11 meV; the cut zeroed every mode from 4e-11 to 9.9e-4 meV. A
   genuine zero mode needs no energy cut on the exact-H path: both backends' shared
   pseudo-norm rule already gives it zero, which is the convention (an exactly-zero
   mode has no normalizable eigenvector, and is in any case a set of measure zero).
   The cut now applies only to fMagCalc's bond-model path, where reconstruction noise
   needs it.

2. Degenerate modes. A collinear antiferromagnet's two magnons are degenerate at every
   q. fMagCalc matched each +q eigenvector to a SINGLE best-overlapping -q column, which
   inside a degenerate pair need not exist, so the mode went unmatched and lost all its
   intensity: on the Neel chain, 172 of 1495 powder points came out exactly 0 at
   energies far from zero. pyMagCalc fixed this in July (5449196, projection onto the
   degenerate subspace); the Fortran port predated it. Sunny is the reference.

Only a degenerate group's TOTAL is an observable -- how a pair splits its weight
depends on the eigenbasis each LAPACK returns -- so the comparisons are of totals.
"""
import json
import logging
import os

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import fibonacci_sphere_points

logging.disable(logging.CRITICAL)

A = 3.0
HERE = os.path.dirname(__file__)


def _calc(cfg, S):
    m = GenericSpinModel(cfg)
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base="gapless", hamiltonian_params=[])


def _fm_chain(S=1.0, J=-1.0):
    return _calc({
        "crystal_structure": {"lattice_vectors": [[A, 0, 0], [0, 10, 0], [0, 0, 10]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": A, "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                               "direction": [0, 0, 1]}}, S)


def _neel_chain(S=1.0, J=1.0):
    return _calc({
        "crystal_structure": {"lattice_vectors": [[2 * A, 0, 0], [0, 10, 0], [0, 0, 10]],
                              "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                                           {"label": "B", "pos": [0.5, 0, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [
            {"type": "heisenberg", "distance": A, "value": J}]},
        "parameters": {}, "parameter_order": [],
        "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                               "directions": [[0, 0, 1], [0, 0, -1]]}}, S)


# h = 0 is the exact zero; every other h has a small but NONZERO energy, down to 4e-11
# meV, and the closed form holds there exactly.
FM_H = [1e-6, 1e-4, 3e-3, 5e-3, 1e-2, 0.05]


def test_fm_chain_intensity_is_the_closed_form_down_to_tiny_energies(backend):
    S, J = 1.0, -1.0
    qs = [np.array([2 * np.pi * h / A, 0.4, 0.3]) for h in FM_H]
    r = _fm_chain(S, J).calculate_sqw(qs, backend=backend)
    for q, h, E, I in zip(qs, FM_H, r.energies, r.intensities):
        w = 2 * abs(J) * S * (1 - np.cos(2 * np.pi * h))
        assert np.real(E[0]) == pytest.approx(w, rel=1e-6, abs=1e-14)
        qz = q[2] / np.linalg.norm(q)
        assert np.real(I[0]) == pytest.approx(S / 2 * (1 + qz ** 2), rel=1e-9), (
            f"h = {h} (omega = {w:.1e} meV): a well-conditioned mode lost its intensity")


def test_an_exactly_zero_mode_carries_no_intensity_under_either_backend(backend):
    """The convention both backends share: H(q) = 0 at the FM chain's q = 0, the mode
    has no normalizable eigenvector, and it gets zero (not a NaN, not 3S)."""
    r = _fm_chain().calculate_sqw([np.array([0.0, 0.4, 0.3])], backend=backend)
    assert np.real(r.energies[0][0]) == 0.0
    assert np.real(r.intensities[0][0]) == 0.0


def _sunny_fixture():
    with open(os.path.join(HERE, "data_afm_chain_sunny.json")) as f:
        return json.load(f)


def test_the_fixture_points_are_pymagcalcs_powder_points():
    d = _sunny_fixture()
    pts = np.vstack([fibonacci_sphere_points(q, d["num_samples"]) for q in d["shells"]])
    assert np.abs(pts - np.array(d["q_cart"])).max() < 1e-12


def test_neel_chain_degenerate_totals_match_sunny(backend):
    """Every point of two powder shells, the degenerate pair's total against Sunny's.
    Above 0.2 meV only: Sunny regularizes H by 1e-8, which moves energies near the
    Goldstone points (8e-6 relative at 0.05 meV, 5e-7 at 0.2)."""
    d = _sunny_fixture()
    qs = np.array(d["q_cart"])
    E_s = np.sort(np.array(d["energies"]), axis=1)
    I_s = np.array(d["total_intensity"])
    r = _neel_chain().calculate_sqw(list(qs), backend=backend)
    E = np.sort(np.real(r.energies), axis=1)
    I = np.real(r.intensities).sum(axis=1)
    ok = E.min(axis=1) > 0.2
    assert ok.sum() > 300
    np.testing.assert_allclose(E[ok], E_s[ok], rtol=2e-6)
    rel = np.abs(I[ok] - I_s[ok]) / I_s[ok]
    assert rel.max() < 1e-5, (
        f"{int(np.sum(rel > 1e-5))} of {ok.sum()} points miss Sunny's total "
        f"({int(np.sum(I[ok] == 0))} of them exactly zero)")


def test_gapless_powder_averages_agree_between_backends(fortran_backend):
    """The item as OPEN_WORK stated it: the powder average of a gapless model no
    longer depends on `backend:`. Per shell, every mode's weight summed (the per-mode
    split of a degenerate pair is basis-dependent and not an observable)."""
    qm = np.linspace(0.2, 3.0, 8)
    for calc in (_fm_chain(), _neel_chain()):
        pn = calc.calculate_powder_average(qm, num_samples=200, backend="numpy")
        pf = calc.calculate_powder_average(qm, num_samples=200, backend=fortran_backend)
        np.testing.assert_allclose(np.real(pf.energies), np.real(pn.energies), atol=1e-12)
        tn = np.real(pn.intensities).sum(axis=1)
        tf = np.real(pf.intensities).sum(axis=1)
        np.testing.assert_allclose(tf, tn, rtol=1e-8)


@pytest.fixture(autouse=True)
def _under_both_builders(builder):
    """Every test here runs under both H(q) builders (K8; tests/conftest.py)."""
    return builder
