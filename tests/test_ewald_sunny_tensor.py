"""The Ewald dipolar tensor A_ij(q) itself, pinned to Sunny 0.8.1 element by element.

`tests/test_ewald.py` pins what A(q) PRODUCES (bands to 1e-5, the classical energy) on
one ferromagnetic chain. That leaves the tensor's pair structure, its phases, the
surface term and the q folding to be checked only through their effect on a spectrum.
Here the oracle is Sunny's own `precompute_dipole_ewald_at_wavevector`, evaluated for
four cells chosen to exercise every branch of the sum: a non-orthogonal 3-site cell, a
triclinic 2-site cell with a non-trivial demagnetization tensor, the 1-site chain, and
an irregular 8-site cell. The q points cover Gamma (the k = 0 surface term), generic
points, a zone-boundary point and points outside the first zone (the folding).
Regenerate the data with `tests/ewald_sunny_reference.jl`.

Sunny returns A without the mu0*muB^2 prefactor and phased over lattice translations
only; pyMagCalc returns meV/muB^2 in the full-bond gauge, so the reference is
mu0*muB^2 * A_sunny * exp(i q.(r_j - r_i)). The bar is 1e-8 of the largest element,
which is stricter than an absolute 1e-8 since |A| ~ 1e-2 meV/muB^2. Measured: ~1e-15.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from magcalc import ewald
from magcalc.constants import MU0_MUB2_MEV_A3

DATA = json.loads((Path(__file__).parent / "data_ewald_sunny.json").read_text())
MODELS = DATA["models"]


def _sunny_reference(m, qi):
    lat = np.array(m["latvecs"], float)
    pos = np.array(m["positions"], float)
    cart = pos @ lat
    q_cart = np.array(m["q_rlu"][qi], float) @ (2 * np.pi * np.linalg.inv(lat).T)
    gauge = np.exp(1j * ((cart[None, :, :] - cart[:, None, :]) @ q_cart))
    S = np.array(m["A_re"][qi]) + 1j * np.array(m["A_im"][qi])
    return MU0_MUB2_MEV_A3 * S * gauge[:, :, None, None]


def test_the_reference_is_sunny_0_8_1():
    assert DATA["sunny_version"] == "0.8.1"
    assert {len(m["positions"]) for m in MODELS} == {1, 2, 3, 8}


@pytest.mark.parametrize("m", MODELS, ids=[m["name"] for m in MODELS])
def test_A_matches_sunny_element_by_element(m):
    eng = ewald.DipoleEwald(m["latvecs"], m["positions"], demag=np.array(m["demag"]))
    for qi, q in enumerate(m["q_rlu"]):
        want = _sunny_reference(m, qi)
        got = eng.at_q(q)
        err = np.abs(got - want).max() / np.abs(want).max()
        assert err < 1e-8, f"{m['name']} q={q}: relative error {err:.2e}"


@pytest.mark.parametrize("m", MODELS[:2], ids=[m["name"] for m in MODELS[:2]])
def test_the_one_off_function_equals_the_reusable_engine(m):
    eng = ewald.DipoleEwald(m["latvecs"], m["positions"], demag=np.array(m["demag"]))
    for q in m["q_rlu"]:
        one_off = ewald.dipole_ewald_at_q(np.array(m["latvecs"]), np.array(m["positions"]),
                                          np.array(q), demag=np.array(m["demag"]))
        assert np.array_equal(one_off, eng.at_q(q))


def test_the_engine_is_reusable_in_any_order():
    """Precomputation must hold no per-q state: evaluating q1, q2, q1 gives q1 twice."""
    m = MODELS[-1]
    eng = ewald.DipoleEwald(m["latvecs"], m["positions"], demag=np.array(m["demag"]))
    q1, q2 = m["q_rlu"][1], m["q_rlu"][3]
    first = eng.at_q(q1)
    eng.at_q(q2)
    assert np.array_equal(first, eng.at_q(q1))


def test_exchange_from_A_is_g_transpose_A_g_for_every_pair():
    """The vectorized J_ij = g_i^T A_ij g_j against the definition, pair by pair, for
    scalar g and for anisotropic, non-symmetric g tensors."""
    rng = np.random.default_rng(0)
    n = 4
    A = rng.standard_normal((n, n, 3, 3)) + 1j * rng.standard_normal((n, n, 3, 3))
    for g in (np.array([2.0, 1.9, 2.1, 2.2]), rng.standard_normal((n, 3, 3))):
        J = ewald.exchange_from_A(A, g)
        gt = g if g.ndim == 3 else np.einsum("i,ab->iab", g, np.eye(3))
        for i in range(n):
            for j in range(n):
                assert np.allclose(J[i, j], gt[i].T @ A[i, j] @ gt[j], atol=1e-14)
