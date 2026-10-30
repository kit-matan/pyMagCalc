"""MagCalc.sqw_batch: batched Colpa S(Q,w), against Sunny, closed forms and the per-q path.

`sqw_batch` diagonalizes a stack of H(q) by Colpa's method (one Cholesky and one
Hermitian `eigh` per q, batched) instead of the per-q pool's `eig` at +q and -q plus
basis matching (`process_calc_Sqw` -> `KKdMatrix`). It is the engine under the
forward model (`magcalc.forward`) and, since 2026-10-30, under `calculate_sqw` itself
(every cross-section and single-k: `test_sqw_colpa_routing.py`), so it must be the
SAME quantity as the per-q path:

- energies and per-mode intensities equal to `calculate_sqw` to 1e-9 (inside an
  exactly degenerate group only the group total is an observable, so totals);
- the absolute scale equal to Sunny's (`ssf_perp(apply_g=false)`), directly;
- with `apply_g=True`, equal to Sunny's `ssf_perp(apply_g=true)` on a model whose
  per-site g tensors are off-diagonal and non-symmetric (a transposed or swapped g
  fails), with form factors on both ions;
- at an exact Goldstone point the Cholesky of a numerically singular H2 can still
  "succeed"; those points must take the per-q path and get its conventional zero.
"""
import copy
import json
import logging
import os
import sys

import numpy as np
import pytest

import magcalc as mc
from magcalc.generic_model import GenericSpinModel
from magcalc.numerical import colpa_batch

sys.path.insert(0, os.path.dirname(__file__))
from test_gapless_backends import _fm_chain, _neel_chain  # noqa: E402
from test_mixed_spin_sqw import _ferri_calc  # noqa: E402

logging.disable(logging.CRITICAL)
HERE = os.path.dirname(os.path.abspath(__file__))


def _groups(e, tol=1e-6):
    """Index groups of (near-)degenerate modes of one q, energies ascending."""
    out, cur = [], [0]
    for k in range(1, len(e)):
        if abs(e[k] - e[k - 1]) < tol:
            cur.append(k)
        else:
            out.append(cur)
            cur = [k]
    out.append(cur)
    return out


def _per_q(calc, q, exact):
    """The per-q worker (`process_calc_Sqw` -> `KKdMatrix`), run in-process.

    exact=True switches off KKdMatrix's zeroing of K/Kd elements below
    ZERO_MATRIX_ELEMENT_THRESHOLD = 1e-6. That cut is a ~1e-6 RELATIVE error on a
    16-site cell (measured on aCVO: 1.7073801560 with it, 1.7073812084 without, and
    Colpa gives the latter to 10 digits), so a 1e-9 comparison needs it off. It must
    stay on at a Goldstone point, where it is what gives a zero mode zero weight.
    """
    import sympy as sp
    from magcalc import linalg, numerical
    numerical._init_worker(calc.HMat_sym,
                           [s for s in calc.full_symbol_list if isinstance(s, sp.Symbol)])
    saved = linalg.ZERO_MATRIX_ELEMENT_THRESHOLD
    if exact:
        linalg.ZERO_MATRIX_ELEMENT_THRESHOLD = 1e-300
    try:
        mags = calc.sm.spin_magnitudes()
        out = [numerical.process_calc_Sqw((
            calc.Ud_numeric, qv, calc.nspins, calc.spin_magnitude,
            list(calc.hamiltonian_params), calc.sm.ion_list(), "perp", mags, None))
            for qv in q]
    finally:
        linalg.ZERO_MATRIX_ELEMENT_THRESHOLD = saved
    E = np.real(np.array([o[1] for o in out]))
    I = np.real(np.array([o[2] for o in out])) / calc.supercell_ncells
    return E, I


def _assert_same_spectrum(E, I, Er, Ir, rtol):
    order = np.argsort(Er, axis=1)
    Er = np.take_along_axis(Er, order, 1)
    Ir = np.take_along_axis(Ir, order, 1)
    assert np.abs(E - Er).max() <= rtol * np.abs(Er).max()
    scale_i = Ir.sum(axis=1).max()
    for k in range(len(E)):
        for g in _groups(Er[k]):
            assert I[k, g].sum() == pytest.approx(Ir[k, g].sum(), abs=rtol * scale_i), (
                f"q index {k}, modes {g}")


def _assert_equal_to_per_q(calc, q, rtol=1e-9, exact=True):
    """sqw_batch against the per-q worker (exact) to rtol, against calculate_sqw's
    per-q process pool (`sqw_method: perq`, thresholds on) to 1e-5, and equal to
    calculate_sqw's default, which is this path."""
    E, I, ok = calc.sqw_batch(q, return_ok=True)
    Er, Ir = _per_q(calc, q, exact)
    _assert_same_spectrum(E, I, Er, Ir, rtol)
    calc.sqw_method = "perq"
    try:
        ref = calc.calculate_sqw(list(q))
    finally:
        calc.sqw_method = None
    _assert_same_spectrum(E, I, np.real(np.asarray(ref.energies)),
                          np.real(np.asarray(ref.intensities)), max(rtol, 1e-5))
    default = calc.calculate_sqw(list(q))
    assert np.array_equal(default.energies, E, equal_nan=True)
    assert np.array_equal(default.intensities, I, equal_nan=True)
    return ok


# ---------------------------------------------------------------- closed forms
def test_fm_chain_is_the_closed_form():
    """omega = 2|J|S(1 - cos qa), I = (S/2)(1 + qz^2/q^2) exactly (per chemical cell)."""
    S, J, A = 1.0, -1.0, 3.0
    rng = np.random.default_rng(4)
    q = rng.normal(size=(200, 3)) * 1.3
    E, I = _fm_chain(S, J).sqw_batch(q)
    w = 2 * abs(J) * S * (1 - np.cos(q[:, 0] * A))
    assert E[:, 0] == pytest.approx(w, rel=1e-10, abs=1e-12)
    qz2 = q[:, 2] ** 2 / np.sum(q ** 2, axis=1)
    assert I[:, 0] == pytest.approx(S / 2 * (1 + qz2), rel=1e-9)


def test_fm_chain_zero_mode_takes_the_per_q_path():
    """H(q) = 0 at the FM chain's q = 0: no Cholesky, the per-q convention (E = 0,
    I = 0) -- not a NaN, and not a rounding-sized energy with a huge intensity."""
    E, I, ok = _fm_chain().sqw_batch(np.array([[0.0, 0.4, 0.3], [0.3, 0.1, 0.0]]),
                                     return_ok=True)
    assert list(ok) == [False, True]
    assert E[0, 0] == 0.0 and I[0, 0] == 0.0


def test_ferrimagnetic_chain_is_the_closed_form():
    """Mixed spin: the per-site sqrt(S_i/2) and form factors, both branches."""
    from test_mixed_spin_sqw import _closed_form, _qs
    from test_mixed_spin_sqw import _RLU
    calc = _ferri_calc()
    q, h = _qs(), _RLU[:, 0]
    E, I = calc.sqw_batch(q)
    for k in range(len(q)):
        E_cf, I_cf = _closed_form(q[k], h[k])
        assert E[k] == pytest.approx(E_cf, rel=1e-10)
        assert I[k] == pytest.approx(I_cf, rel=1e-9, abs=1e-14)


# ---------------------------------------------------------------- per-q path
def test_neel_chain_equals_the_per_q_path():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(30, 3)) * 1.5
    ok = _assert_equal_to_per_q(_neel_chain(), q)
    assert ok.all()


def test_neel_chain_goldstone_points_fall_back():
    """q = 0 and the zone centres of the doubled cell (q_x = pi/3 and 2 pi/3 for a = 6 A
    with sites at 0 and 1/2): H2 is singular, so these must be the per-q result."""
    q = np.array([[0.0, 0, 0], [np.pi / 3, 0, 0], [2 * np.pi / 3, 0.1, 0],
                  [0.5, 0.2, 0.1]])
    ok = _assert_equal_to_per_q(_neel_chain(), q, exact=False)
    assert list(ok) == [False, False, False, True]


def test_mixed_spin_equals_the_per_q_path():
    rng = np.random.default_rng(11)
    q = rng.normal(size=(25, 3))
    _assert_equal_to_per_q(_ferri_calc(), q)


def test_neel_chain_matches_sunny_directly():
    """Absolute scale: Sunny's degenerate-pair totals on two powder shells."""
    with open(os.path.join(HERE, "data_afm_chain_sunny.json")) as f:
        d = json.load(f)
    q = np.array(d["q_cart"])
    E, I = _neel_chain().sqw_batch(q)
    E_s = np.sort(np.array(d["energies"]), axis=1)
    I_s = np.array(d["total_intensity"])
    keep = E_s[:, 0] > 0.2
    assert E[keep] == pytest.approx(E_s[keep], rel=1e-6)
    assert I[keep].sum(axis=1) == pytest.approx(I_s[keep], rel=1e-6)


def test_supercell_normalization_matches_calculate_sqw():
    """A magnetic_supercell: per chemical cell, as calculate_sqw divides."""
    cfg = {"crystal_structure": {"lattice_vectors": [[3.0, 0, 0], [0, 10, 0], [0, 0, 10]],
                                 "magnetic_supercell": [2, 1, 1],
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": 1.0, "ion": "Mn2+"}]},
           "interactions": {"symmetry_rules": [
               {"type": "heisenberg", "distance": 3.0, "value": 1.0}]},
           "parameters": {}, "parameter_order": [],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [[0, 0, 1], [0, 0, -1]]}}
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=1.0, cache_mode="none",
                      cache_file_base="sc", hamiltonian_params=[])
    assert calc.supercell_ncells == 2
    rng = np.random.default_rng(2)
    _assert_equal_to_per_q(calc, rng.normal(size=(12, 3)))


# ---------------------------------------------------------------- g tensor
G_REF = json.load(open(os.path.join(HERE, "data_g_tensor_sqw_sunny.json")))


def _g_chain(with_g=True):
    d = G_REF
    n = np.array(d["axis"])
    atoms = []
    for k, (lab, x) in enumerate((("A", 0.0), ("B", 0.5))):
        a = {"label": lab, "pos": [x, 0, 0], "spin_S": d["spins"][k], "ion": d["ions"][k]}
        if with_g:
            a["g"] = d["g"][k]
        atoms.append(a)
    cfg = {"crystal_structure": {"lattice_vectors": d["lattice"], "atoms_uc": atoms},
           "interactions": {"symmetry_rules": [
               {"type": "heisenberg", "distance": 3.0, "value": "J"}]},
           "parameters": {"J": d["J"]}, "parameter_order": ["J"],
           "magnetic_structure": {"type": "pattern", "pattern_type": "generic",
                                  "directions": [n.tolist(), (-n).tolist()]}}
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=d["spins"][0],
                      cache_mode="none", cache_file_base="gchain",
                      hamiltonian_params=[d["J"]])
    B = 2 * np.pi * np.linalg.inv(np.array(d["lattice"], float)).T
    return calc, np.array(d["q_rlu"]) @ B


def test_apply_g_matches_sunny_with_off_diagonal_g():
    calc, q = _g_chain()
    E, I = calc.sqw_batch(q, apply_g=True)
    E_s = np.array(G_REF["energies"])
    order = np.argsort(E_s, axis=1)
    E_s = np.take_along_axis(E_s, order, 1)
    I_s = np.take_along_axis(np.array(G_REF["intensity_apply_g"]), order, 1)
    assert E == pytest.approx(E_s, abs=1e-6)
    assert I == pytest.approx(I_s, rel=1e-6)


def test_apply_g_false_matches_sunny_bare():
    calc, q = _g_chain()
    E, I = calc.sqw_batch(q)
    E_s = np.array(G_REF["energies"])
    order = np.argsort(E_s, axis=1)
    I_s = np.take_along_axis(np.array(G_REF["intensity_bare"]), order, 1)
    assert I == pytest.approx(I_s, rel=1e-6)


def test_a_transposed_g_is_detected():
    """The reference is sensitive to g vs g^T, so the pin above means something."""
    calc, q = _g_chain()
    gT = np.array([np.array(g).T for g in G_REF["g"]])
    _, I = calc.sqw_batch(q, apply_g=True, g_tensors=gT)
    E_s = np.array(G_REF["energies"])
    I_s = np.take_along_axis(np.array(G_REF["intensity_apply_g"]),
                             np.argsort(E_s, axis=1), 1)
    assert np.abs(I / I_s - 1).max() > 1e-2


def test_apply_g_with_no_g_configured_is_four_times_bare():
    """Every site at the electron g = 2: exactly 4x (Sunny's apply_g default)."""
    calc, q = _g_chain(with_g=False)
    _, I0 = calc.sqw_batch(q)
    _, I1 = calc.sqw_batch(q, apply_g=True)
    assert I1 == pytest.approx(4 * I0, rel=1e-13)


def test_apply_g_fallback_matches_colpa():
    """The per-q fallback applies g the same way: force it by declaring every q bad."""
    from magcalc import numerical
    calc, q = _g_chain()
    E0, I0 = calc.sqw_batch(q, apply_g=True)
    real = numerical.colpa_batch

    def broken(h, n):
        e, t, ok = real(h, n)
        return e, t, np.zeros_like(ok)

    numerical.colpa_batch = broken
    try:
        E1, I1, ok = calc.sqw_batch(q, apply_g=True, return_ok=True)
    finally:
        numerical.colpa_batch = real
    assert not ok.any()
    assert E1 == pytest.approx(E0, rel=1e-9)
    assert I1 == pytest.approx(I0, rel=1e-8)


def test_parameter_dependent_g_follows_update_hamiltonian_params():
    """A g entry that names a parameter is resolved at the CURRENT value."""
    d = G_REF
    calc, q = _g_chain()
    calc.sm.config["parameters"]["gxz"] = 0.3
    calc.sm.config["parameter_order"].append("gxz")
    calc.sm.config["crystal_structure"]["atoms_uc"][0]["g"] = [
        [2.0, 0, "gxz"], [0, 2.0, 0], ["gxz", 0, 2.0]]
    calc.hamiltonian_params = [d["J"], 0.3]
    g = calc._numeric_g_tensors()
    assert g[0] == pytest.approx(np.array([[2, 0, .3], [0, 2, 0], [.3, 0, 2]]))
    calc.hamiltonian_params = [d["J"], -0.1]
    assert calc._numeric_g_tensors()[0][0, 2] == pytest.approx(-0.1)


# ---------------------------------------------------------------- guards
def test_unsupported_requests_raise():
    calc = _fm_chain()
    with pytest.raises(ValueError):
        calc.sqw_batch(np.ones((2, 3)), cross_section="qq")
    with pytest.raises(ValueError):
        calc.sqw_batch(np.ones((2, 3)), g_tensors=np.eye(3)[None])


def test_colpa_paraunitary():
    """T^dagger g T = I on the positive block, and T^dagger H2 T = diag(omega)."""
    calc = _ferri_calc()
    q = np.random.default_rng(3).normal(size=(5, 3))
    h = calc._build_h_stack(q, calc.spin_magnitude)
    E, T, ok = colpa_batch(h, 2)
    g = np.diag([1, 1, -1, -1])
    for k in range(len(q)):
        assert np.conj(T[k]).T @ g @ T[k] == pytest.approx(np.eye(2), abs=1e-12)
        h2 = g @ h[k]
        assert np.conj(T[k]).T @ h2 @ T[k] == pytest.approx(np.diag(E[k]), abs=1e-12)


# ---------------------------------------------------------------- 16 sites
@pytest.mark.slow
def test_acvo_16_sites_equals_the_per_q_path():
    """The shipped alpha-Cu2V2O7 model (Fdd2, 16 Cu, DM + symmetric anisotropy) at
    zero field, in its annealed ground state: every mode against calculate_sqw."""
    import yaml
    path = os.path.join(HERE, "..", "examples", "materials", "aCVO", "config_acvo.yaml")
    cfg = yaml.safe_load(open(path))
    cfg["parameters"]["H_mag"] = 0.0
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    sm = GenericSpinModel(copy.deepcopy(cfg))
    pre = mc.MagCalc(spin_model_module=sm, spin_magnitude=0.5, hamiltonian_params=pv,
                     cache_mode="none", cache_file_base="acvo_b", initialize=False)
    res = pre.minimize_energy(method="anneal", num_starts=4, n_workers=1, seed=3)
    sm.set_magnetic_structure(res.x[::2], res.x[1::2])
    calc = mc.MagCalc(spin_model_module=sm, spin_magnitude=0.5, hamiltonian_params=pv,
                      cache_mode="none", cache_file_base="acvo_b")
    e0, e1 = calc.relax_from_current()
    assert e1 >= e0 - 1e-9          # a minimum, so H2 > 0 away from Goldstone points
    B = 2 * np.pi / np.array([20.645, 8.383, 6.442])
    rng = np.random.default_rng(5)
    q = rng.uniform([-1, 0, -1], [9, 3, 1], size=(40, 3)) * B
    ok = _assert_equal_to_per_q(calc, q)
    assert ok.all()
