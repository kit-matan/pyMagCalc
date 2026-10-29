"""The Luttinger-Tisza J(k) used by the q!=0 guard and the spiral initial guess.

`spiral_opt._fourier_exchange` compresses the OUC bonds into one 3N x 3N block
J(R) per cell translation and evaluates J(k) = sum_R J(R) exp(2 pi i k.R) for a
whole stack of k at once (2026-10-29). Before that, J(k) was a Python loop over
N^2 site pairs at every point of a 24^3 grid, which made the guard the largest
cost of a plain dipole run: 30 s on aCVO, 47 s on YIG.

Two kinds of reference, kept apart:

  * the per-pair loop it replaced, copied below verbatim, as the parity
    reference: J(k) to rounding, and the same grid point picked by the same
    first-wins-by-1e-12 rule;
  * closed-form band minima the loop never touched: the triangular
    antiferromagnet, eigmin J(k) = -3 J_aa at K = (1/3, 1/3) for each component,
    and the DM chain, whose in-plane block 2J cos q +- 2D sin q has its minimum
    -2 sqrt(J^2 + D^2) at q = pi - atan(D/J). The second exercises the
    antisymmetric (complex) part of J(k), which the isotropic tests cannot.
"""
import copy
import os
from itertools import product

import numpy as np
import pytest
import yaml

from magcalc import runner, spiral_opt
from magcalc.generic_model import GenericSpinModel

HERE = os.path.dirname(__file__)
EXAMPLES = os.path.join(HERE, "..", "examples")
CONFIGS = [
    os.path.join(EXAMPLES, "sunny_tutorials", "S01_CoRh2O4", "config.yaml"),
    os.path.join(EXAMPLES, "spinw_tutorials", "SW09_kagome_AFM_DM", "config.yaml"),
    os.path.join(EXAMPLES, "spinw_tutorials", "SW03_frustrated_chain", "config_spiral.yaml"),
]


def _reference_J_of_k(J_full, offsets, ouc_map, k):
    """The per-pair loop `luttinger_tisza_guess` used until 2026-10-28 (bbadb64)."""
    N = J_full.shape[0]
    ph = np.exp(2j * np.pi * (offsets @ k))
    Jk = np.zeros((3 * N, 3 * N), dtype=complex)
    for i in range(N):
        blocks = J_full[i] * ph[:, None, None]
        for j in range(N):
            sel = (ouc_map == j)
            if np.any(sel):
                Jk[3 * i:3 * i + 3, 3 * j:3 * j + 3] += blocks[sel].sum(axis=0)
    return 0.5 * (Jk + Jk.conj().T)


def _reference_grid_pick(J_full, offsets, ouc_map, k_grid):
    """The old grid scan: eigh per point, first point beating the best by 1e-12."""
    bonded = np.any(np.abs(J_full) > 1e-12, axis=(0, 2, 3))
    bond_offsets = offsets[bonded] if np.any(bonded) else offsets
    active = [d for d in range(3) if np.any(np.abs(bond_offsets[:, d]) > 0.5)]
    grid = np.arange(k_grid) / float(k_grid)
    axes_vals = [grid if d in active else np.array([0.0]) for d in range(3)]
    best_k, best_e = np.zeros(3), np.inf
    for kx, ky, kz in product(*axes_vals):
        k = np.array([kx, ky, kz])
        e = np.linalg.eigh(_reference_J_of_k(J_full, offsets, ouc_map, k))[0][0]
        if e < best_e - 1e-12:
            best_e, best_k = e, k
    return best_k, best_e


def _arrays(path):
    cfg = yaml.safe_load(open(path))
    m = GenericSpinModel(copy.deepcopy(cfg), base_path=os.path.dirname(path))
    _S, p = runner.resolve_S_and_params(m.config)
    return spiral_opt._build_bond_arrays(m, p)


def _triangular(Jaa):
    """One site, six neighbours at +-(1,0,0), +-(0,1,0), +-(1,1,0), exchange diag(Jaa)."""
    offs = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0]],
                    dtype=float)
    J_full = np.repeat(np.diag(Jaa)[None, None], len(offs), axis=1)
    return J_full, offs, np.zeros(len(offs), dtype=int)


def _dm_chain(J, D):
    """One site, bonds to +-x with J I + skew'(+-D z); D_ji = -D_ij."""
    def skew(d):
        return np.array([[0.0, d[2], -d[1]], [-d[2], 0.0, d[0]], [d[1], -d[0], 0.0]])
    Dz = np.array([0.0, 0.0, D])
    J_full = np.stack([J * np.eye(3) + skew(Dz), J * np.eye(3) + skew(-Dz)])[None]
    return J_full, np.array([[1.0, 0, 0], [-1.0, 0, 0]]), np.zeros(2, dtype=int)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_fourier_exchange_equals_the_per_pair_loop(path):
    J_full, offsets, ouc_map = _arrays(path)
    J_of_k = spiral_opt._fourier_exchange(J_full, offsets, ouc_map)
    ks = np.random.default_rng(7).random((40, 3))
    stack = J_of_k(ks)
    for k, Jk in zip(ks, stack):
        ref = _reference_J_of_k(J_full, offsets, ouc_map, k)
        assert np.abs(Jk - ref).max() < 1e-13
        assert np.abs(J_of_k(k) - ref).max() < 1e-13        # the single-k form too


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: os.path.basename(os.path.dirname(p)))
def test_grid_scan_picks_the_same_point_as_the_loop(path):
    J_full, offsets, ouc_map = _arrays(path)
    k_ref, e_ref = _reference_grid_pick(J_full, offsets, ouc_map, 12)
    k_new, _axis = spiral_opt.luttinger_tisza_guess(J_full, offsets, ouc_map, k_grid=12,
                                                    refine=False)
    assert np.array_equal(k_new, k_ref % 1.0)
    e_new = np.linalg.eigvalsh(spiral_opt._fourier_exchange(J_full, offsets, ouc_map)(k_new))[0]
    assert abs(e_new - e_ref) < 1e-12


@pytest.mark.parametrize("Jaa", [(1.0, 1.0, 1.0), (1.0, 1.0, 0.4), (0.3, 0.3, 2.0)])
def test_triangular_afm_minimum_is_minus_three_J_at_K(Jaa):
    J_full, offsets, ouc_map = _triangular(Jaa)
    k, _axis = spiral_opt.luttinger_tisza_guess(J_full, offsets, ouc_map, k_grid=24)
    lam = np.linalg.eigvalsh(spiral_opt._fourier_exchange(J_full, offsets, ouc_map)(k))[0]
    assert lam == pytest.approx(-3.0 * max(Jaa), abs=1e-9)
    # K or its time-reversed partner -K = (2/3, 2/3). kz has no bonds: the refine
    # step leaves ~1e-5 of noise there, which ordering_wavevector snaps at 1e-3.
    assert (np.allclose(k[:2], [1 / 3, 1 / 3], atol=1e-5)
            or np.allclose(k[:2], [2 / 3, 2 / 3], atol=1e-5)), k
    assert min(k[2], 1 - k[2]) < 1e-3


@pytest.mark.parametrize("J,D", [(1.0, 0.3), (1.0, -0.3), (-1.0, 0.5)])
def test_dm_chain_minimum_is_minus_two_sqrt_J2_plus_D2(J, D):
    J_full, offsets, ouc_map = _dm_chain(J, D)
    k, _axis = spiral_opt.luttinger_tisza_guess(J_full, offsets, ouc_map, k_grid=24)
    J_of_k = spiral_opt._fourier_exchange(J_full, offsets, ouc_map)
    lam = np.linalg.eigvalsh(J_of_k(k))[0]
    assert lam == pytest.approx(-2.0 * np.hypot(J, D), abs=1e-9)
    # 2J cos q + 2D sin q (or -) is minimal where the in-plane band touches -2 sqrt(J^2+D^2)
    q = 2 * np.pi * k[0]
    assert -2 * abs(J * np.cos(q)) - 2 * abs(D * np.sin(q)) == pytest.approx(lam, abs=1e-6)
    q_star = (np.pi - np.arctan(abs(D / J))) if J > 0 else np.arctan(abs(D / J))
    k_star = q_star / (2 * np.pi)
    assert min(abs(k[0] - k_star), abs(k[0] - (1 - k_star))) < 1e-5, (k[0], k_star)


def test_a_model_with_no_bonds_has_a_zero_J_of_k():
    J_full = np.zeros((2, 3, 3, 3))
    offsets = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    J_of_k = spiral_opt._fourier_exchange(J_full, offsets, np.array([0, 1, 0]))
    assert J_of_k(np.array([0.2, 0.1, 0.0])).shape == (6, 6)
    assert not J_of_k(np.random.default_rng(0).random((5, 3))).any()
