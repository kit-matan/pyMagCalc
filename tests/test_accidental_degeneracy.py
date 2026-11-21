"""Accidentally degenerate bands: every kernel gives the same group total (2026-11-21).

A Bogoliubov transformation needs T^dagger g T = g, so the eigenvectors inside a
degenerate block must be orthogonal in the bosonic metric g = diag(1, -1). The per-q
path (`linalg.KKdMatrix`, and its Fortran port `magcalc_kkd.f90`) orthonormalized a
degenerate block with a EUCLIDEAN Gram-Schmidt. That is g-orthogonal only when the
block's modes share one u/v ratio -- a symmetry-related pair, which is every case the
suite had. Where two bands with DIFFERENT Bogoliubov angles cross, and the eigensolver
hands back a mixed basis, the group's summed intensity came out wrong: up to 12 % here,
1.2 % on two weakly coupled chains through `calculate_sqw(backend="fortran")`.

Colpa is paraunitary by construction and is the reference. Only a degenerate group's
TOTAL is an observable, so totals are what is compared.

The Rust kernels have no Gram-Schmidt path (rsMagCalc's LSWT kernel is Colpa); the last
test pins it to NumPy at an accidental crossing all the same.
"""
import copy
import logging

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from magcalc import kernels as K
from magcalc.generic_model import GenericSpinModel
from magcalc.linalg import KKdMatrix
from magcalc.numerical import colpa_amplitudes_from_h, contract_cross_section_batch

logging.disable(logging.CRITICAL)

UD = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 6)))[0].astype(complex)
Q = np.array([0.3, 0.1, 0.2])
# (site-mixing angle, phase): 0 is the unmixed pair every kernel always got right
MIXINGS = [(0.0, 0.0), (0.4, 0.0), (0.7, 0.0), (0.7, 0.9), (1.1, 0.3)]


def _two_modes(theta, phi, split=0.0):
    """H(+q), H(-q) of two modes at the same energy 2 with different u/v ratios
    ((A, B) = (2.5, 1.5) and (sqrt(13), 3)), mixed by a site rotation."""
    A = np.diag([2.5, np.sqrt(13.0) + split])
    B = np.diag([1.5, 3.0])
    c, s = np.cos(theta), np.sin(theta)
    U = np.array([[c, -s * np.exp(1j * phi)], [s * np.exp(-1j * phi), c]])
    A2 = U.conj().T @ A @ U
    B2 = U.conj().T @ B @ U.conj()
    H2 = np.block([[A2, B2], [B2.conj().T, A2.conj()]])
    g = np.diag([1, 1, -1, -1.0])
    P = np.block([[np.zeros((2, 2)), np.eye(2)], [np.eye(2), np.zeros((2, 2))]])
    return g @ H2, g @ (P @ H2.conj() @ P)


def _colpa_total(Hp, cs):
    amps = colpa_amplitudes_from_h(Hp[None], Q[None], UD, np.full(2, 0.5))
    assert amps.ok[0]
    return float(np.sum(amps.contract(Q[None], cs)[0]))


def _perq_total(Hp, Hm, cs):
    Kmat, Kd, ev = KKdMatrix(0.5, Hp, Hm, UD, Q, 2)
    assert abs(ev[0] - ev[1]) < 1e-8
    corr = np.einsum("iam,jbm->abm", Kmat[:, :2].reshape(2, 3, 2),
                     Kd[:, 2:].reshape(2, 3, 2))
    return float(np.real(contract_cross_section_batch(corr[None], Q[None], cs)[0][0]).sum())


@pytest.mark.parametrize("theta,phi", MIXINGS)
@pytest.mark.parametrize("cs", ["perp", "trace"])
def test_numpy_per_q_total_equals_colpa(theta, phi, cs):
    Hp, Hm = _two_modes(theta, phi)
    assert _perq_total(Hp, Hm, cs) == pytest.approx(_colpa_total(Hp, cs), rel=1e-10)


@pytest.mark.parametrize("theta,phi", MIXINGS)
def test_fortran_total_equals_numpy_and_colpa(theta, phi):
    fmagcalc = pytest.importorskip("fmagcalc", reason="fMagCalc not installed")
    if getattr(fmagcalc, "backend", None) != "ctypes":
        pytest.skip("fMagCalc compiled library unavailable")
    Hp, Hm = _two_modes(theta, phi)
    out = fmagcalc.run_sqw(Hp[None], Hm[None], UD, np.ones((1, 2)), 0.5, Q[None])
    fort = float(out["intensities"][0].sum())
    assert np.allclose(out["energies"][0], 2.0, atol=1e-9)
    # An fMagCalc built before 2026-11-21 fails here by 2-12 % for theta != 0.
    assert fort == pytest.approx(_colpa_total(Hp, "perp"), rel=1e-10)
    assert fort == pytest.approx(_perq_total(Hp, Hm, "perp"), rel=1e-10)


def test_a_split_pair_is_untouched():
    """Above the degeneracy threshold nothing is orthonormalized; the paths agree as
    they always did."""
    Hp, Hm = _two_modes(0.7, 0.0, split=1e-5)
    Kmat, Kd, _ = KKdMatrix(0.5, Hp, Hm, UD, Q, 2)
    corr = np.einsum("iam,jbm->abm", Kmat[:, :2].reshape(2, 3, 2),
                     Kd[:, 2:].reshape(2, 3, 2))
    perq = float(np.real(contract_cross_section_batch(corr[None], Q[None], "perp")[0][0]).sum())
    assert perq == pytest.approx(_colpa_total(Hp, "perp"), rel=1e-9)


# --------------------------------------------------------------------------
# SU(N): NumPy against the Rust kernel where two inequivalent chains' bands cross
# --------------------------------------------------------------------------
LAT = [[4.0, 0, 0], [0, 7.0, 0], [0, 0, 7.0]]
B_MAT = 2 * np.pi * np.linalg.inv(np.array(LAT, float)).T


def _chains_cfg(j_ab):
    heis = [{"pair": ["A", "A"], "rij_offset": [1, 0, 0], "value": -1.0},
            {"pair": ["A", "A"], "rij_offset": [-1, 0, 0], "value": -1.0},
            {"pair": ["B", "B"], "rij_offset": [1, 0, 0], "value": -0.55},
            {"pair": ["B", "B"], "rij_offset": [-1, 0, 0], "value": -0.55}]
    if j_ab:
        heis += [{"pair": ["A", "B"], "rij_offset": [0, 0, 0], "value": j_ab},
                 {"pair": ["B", "A"], "rij_offset": [0, 0, 0], "value": j_ab}]
    return {"crystal_structure": {"lattice_vectors": LAT,
                                  "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": 1.0},
                                               {"label": "B", "pos": [0.3, 0.5, 0.5],
                                                "spin_S": 1.0}]},
            "interactions": {"heisenberg": heis,
                             "single_ion_anisotropy": [
                                 {"value": 0.2, "axis": [0, 0, 1], "atoms": ["A"]},
                                 {"value": 1.4, "axis": [0, 0, 1], "atoms": ["B"]}]},
            "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[1, 0, 0], [1, 0, 0]]},
            "calculation": {"mode": "SUN", "on_imaginary": "off"}, "tasks": {}}


@pytest.mark.parametrize("j_ab", [0.0, -1e-9])
def test_sun_rust_total_equals_numpy_at_an_accidental_crossing(j_ab):
    if K.rust_module() is None:
        pytest.skip("rsmagcalc not installed (optional Rust kernels)")
    from magcalc.sun.lswt import SUNModel

    ref = SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(_chains_cfg(j_ab))),
                                      params=[], kernels="numpy")
    ref.minimize_energy(n_restarts=4, seed=0)
    rust = SUNModel(ref.S, ref.Z, ref.bonds, ref.onsite, operators=ref.ops,
                    moment_terms=ref.moment_terms, kernels="rust")
    rust.pos, rust.n_cells = ref.pos, ref.n_cells
    assert rust._rust() is not None

    def q_of(h):
        return np.array([[h, 0.13, 0.07]]) @ B_MAT

    def gap(h):
        return float(np.diff(np.sort(ref.dispersion(q_of(h)[0])))[0])

    h = minimize_scalar(gap, bounds=(0.185, 0.198), method="bounded",
                        options={"xatol": 1e-15}).x
    assert gap(h) < 1e-7                      # the two lowest bands meet
    for cs in ("perp", "trace"):
        w0, i0 = ref.structure_factor_batch(q_of(h), cross_section=cs)
        w1, i1 = rust.structure_factor_batch(q_of(h), cross_section=cs)
        assert np.abs(w0 - w1).max() < 1e-10
        assert i1.sum() == pytest.approx(i0.sum(), rel=1e-10)
