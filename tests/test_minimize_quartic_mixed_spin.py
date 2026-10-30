"""`minimize_energy` on a NON-quadratic energy with more than one spin length per cell.

A `biquadratic`, or a `stevens` term of rank >= 4, has no quadratic form (H, b, c), so
the multistart search falls back to an angle-based SymPy energy (the `("sym", ...)`
payload). Until 2026-10-30 its spin vectors were `S_sym * (sin th cos ph, ...)` with
`S_sym` bound to the REFERENCE spin, i.e. |m_i| = S_ref on every site: the mixed-spin
bug `_classical_spin_lengths` fixed for quadratic models on 2026-08-18, still live on
this path (OPEN_WORK item 6). Uniform-spin models were unaffected.

The oracles are exact identities, or the energy written out by hand in this file:

* a (1, 1/2) Neel chain with J = 1 and B = 0.05 on the intracell bond: per cell
  E = 2 J s_A s_B cos(a) + B (s_A s_B)^2 cos^2(a), minimal at a = 180 deg, -0.9875 meV.
  The old fallback reported -1.95 for the same state (one length, s = 1, for both);
* the same chain with B = 4 > J / (s_A s_B): the minimum is the CANTED state
  cos(a) = -J / (B s_A s_B) = -1/2, at E = -J^2/B. The angle depends on the lengths
  (the old fallback returned cos(a) = -1/4), so this pins directions, not just a number;
* the (1, 1, 1/2) trimer of `test_mixed_spin_classical.py` with a biquadratic on its
  three bonds, against E = sum_<ij> J m_i.m_j + B (m_i.m_j)^2 written out here and
  minimised by scipy on the true spheres.
"""
import copy
import logging

import numpy as np
import pytest
from scipy.optimize import minimize

import magcalc as mc
from magcalc.generic_model import GenericSpinModel

J = 1.0
SA, SB = 1.0, 0.5


def _calc(cfg, tag, params=()):
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    return mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base=tag, hamiltonian_params=list(params))


def _chain_cfg(B, sB=SB, mode="numeric"):
    bonds = lambda v, pairs: [{"pair": p, "rij_offset": o, "value": v} for p, o in pairs]
    intra = ((["A", "B"], [0, 0, 0]), (["B", "A"], [0, 0, 0]))
    inter = ((["B", "A"], [1, 0, 0]), (["A", "B"], [-1, 0, 0]))
    return {"crystal_structure": {
                "lattice_vectors": [[6.0, 0, 0], [0, 9.0, 0], [0, 0, 9.0]],
                "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0], "spin_S": SA},
                             {"label": "B", "pos": [0.5, 0, 0], "spin_S": sB}]},
            "interactions": {"heisenberg": bonds(J, intra + inter),
                             "biquadratic": bonds(B, intra)},
            "parameters": {}, "parameter_order": [],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[0, 0, 1], [0, 0, -1]]},
            "calculation": {"on_imaginary": "off", "classical_energy": mode},
            "tasks": {}}


def _units(x):
    th, ph = np.asarray(x)[0::2], np.asarray(x)[1::2]
    return np.c_[np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)]


def _minimize(cfg, tag, params=(), **kw):
    kw = {"method": "L-BFGS-B", "num_starts": 8, "seed": 0, **kw}
    return _calc(cfg, tag, params).minimize_energy(**kw)


@pytest.mark.parametrize("mode", ["numeric", "symbolic"])
def test_neel_chain_reports_the_exact_energy(mode):
    B = 0.05
    res = _minimize(_chain_cfg(B, mode=mode), f"q6_neel_{mode}")
    u = _units(res.x)
    assert u[0] @ u[1] == pytest.approx(-1.0, abs=1e-6)
    assert res.fun == pytest.approx(-2 * J * SA * SB + B * (SA * SB) ** 2, abs=1e-9)
    assert res.fun > -1.95 + 0.5        # the reference-length answer


def test_neel_chain_through_the_worker_pool():
    """The `("sym", ...)` payload is lambdified inside each worker: the per-site
    lengths must survive the trip to `n_workers > 1`."""
    res = _minimize(_chain_cfg(0.05), "q6_pool", n_workers=2)
    assert res.fun == pytest.approx(-0.9875, abs=1e-9)


@pytest.mark.parametrize("sB", [SB, 1.0])
def test_canting_angle_follows_the_per_site_lengths(sB):
    """B s_A s_B > J: the bond energy 2 J s c + B s^2 c^2 (s = s_A s_B, c = cos a) is
    minimal inside, at c = -J/(B s), E = -J^2/B. With s_B = 1/2 that is 120 degrees;
    the uniform-spin control (s_B = 1) gives c = -1/4."""
    B = 4.0
    res = _minimize(_chain_cfg(B, sB=sB), f"q6_cant_{sB}")
    u = _units(res.x)
    assert u[0] @ u[1] == pytest.approx(-J / (B * SA * sB), abs=1e-5)
    assert res.fun == pytest.approx(-J ** 2 / B, abs=1e-9)


# --------------------------------------------------------------------------
# The mixed-spin trimer, with a biquadratic on its three bonds
# --------------------------------------------------------------------------
TRIMER_S = np.array([1.0, 1.0, 0.5])
B_TRI = 0.05


def _trimer_cfg():
    return {"crystal_structure": {
                "lattice_vectors": [[20.0, 0, 0], [0, 20.0, 0], [0, 0, 20.0]],
                "atom_mode": "explicit",
                "atoms_uc": [
                    {"label": "A1", "pos": [0.0, 0.0, 0.0], "spin_S": 1.0},
                    {"label": "A2", "pos": [0.15, 0.0, 0.0], "spin_S": 1.0},
                    {"label": "B1", "pos": [0.075, 0.129903811, 0.0], "spin_S": 0.5}]},
            "interactions": {
                "symmetry_rules": [{"type": "heisenberg", "distance": 3.0, "value": "J"}],
                "biquadratic": [{"distance": 3.0, "value": B_TRI}]},
            "parameters": {"J": J}, "parameter_order": ["J"],
            "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                   "directions": [[0, 0, 1]] * 3},
            "calculation": {"on_imaginary": "off"}, "tasks": {}}


def _trimer_energy(x, S=TRIMER_S):
    m = _units(x) * S[:, None]
    d = [m[i] @ m[j] for i, j in ((0, 1), (1, 2), (0, 2))]
    return sum(J * v + B_TRI * v * v for v in d)


def _trimer_oracle():
    rng = np.random.default_rng(1)
    best = np.inf
    for _ in range(40):
        x0 = rng.random(6) * np.repeat([[np.pi, 2 * np.pi]], 3, axis=0).ravel()
        best = min(best, minimize(_trimer_energy, x0, method="BFGS", tol=1e-12).fun)
    return best


def test_trimer_minimum_matches_the_hand_written_energy():
    res = _minimize(_trimer_cfg(), "q6_trimer", params=[J], num_starts=24)
    assert res.fun == pytest.approx(_trimer_energy(res.x), abs=1e-9)   # its own state
    assert res.fun == pytest.approx(_trimer_oracle(), abs=1e-7)
    # one length for every site scores the same angles differently
    assert abs(_trimer_energy(res.x, np.ones(3)) - res.fun) > 0.1


def test_uniform_spin_fallback_is_unchanged():
    """With every s_i equal to the reference, the per-site lengths are the old S_sym:
    the B_4^0-free uniform chain lands where it always did."""
    B = 4.0
    res = _minimize(_chain_cfg(B, sB=SA), "q6_uniform")
    assert res.fun == pytest.approx(-J ** 2 / B, abs=1e-9)


def test_quartic_models_take_the_fallback(caplog):
    """The fallback these tests pin is the path actually taken: no quadratic form."""
    calc = _calc(_chain_cfg(0.05), "q6_path")
    with caplog.at_level(logging.INFO, logger="magcalc.core"):
        calc.minimize_energy(method="L-BFGS-B", num_starts=1, seed=0)
    assert "Energy not quadratic" in caplog.text
