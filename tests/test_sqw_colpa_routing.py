"""`calculate_sqw` on Colpa's method (OPEN_WORK item 7): every cross-section, single-k.

Since 2026-10-30 the dipole engine's `calculate_sqw` runs on `MagCalc.sqw_amplitudes`
(one Cholesky and one `eigh` per q, batched, threaded) instead of the per-q process
pool (`process_calc_Sqw` -> `KKdMatrix`), which stays selectable for one release as
`calculation: {sqw_method: perq}`. The amplitudes are contracted by
`numerical.contract_cross_section_batch`, the one implementation of every
cross-section.

The independent oracles are the existing Sunny and closed-form pins, which now run
on this path by default (`test_polarized.py`, `test_polarization_frames.py`,
`test_single_k.py`, `test_absolute_normalization.py`, `test_ewald_spiral.py`, ...).
This file adds:

- an EXACT identity for the single-k path: a commensurate helix in the rotating frame
  (three channels, Toth-Lake projectors) equals the same helix written out as an
  explicit magnetic supercell (plain LSWT, Sunny-pinned in `test_polarized.py`), per
  chemical cell, for EVERY cross-section -- k_case 3 (k = 1/3) and k_case 2 (k = 1/2);
- equality with the per-q pool, every cross-section, commensurate and single-k
  (1e-5 of the band scale: the pool zeroes K/Kd elements below 1e-6);
- the per-q fallback (Goldstone points, unstable q) assembled correctly in every
  channel, and a supplied non-minimum (the `on_imaginary: warn` class) giving the
  pool's answer.

Inside an exactly degenerate group only the group total is an observable, so every
comparison is of energy-group totals.
"""
import functools
import logging
import os
import sys

import numpy as np
import pytest

import magcalc as mc
from magcalc import numerical
from magcalc.core import reciprocal_b_matrix
from magcalc.generic_model import GenericSpinModel

sys.path.insert(0, os.path.dirname(__file__))
from test_single_k import K_SW03, build_calc, chain_config, qcart_chain  # noqa: E402

logging.disable(logging.CRITICAL)

P_GEN = [0.3, -0.5, 0.81]
CROSS_SECTIONS = [
    "perp", "trace", "chiral", "sf+", "sf-", "xx", "yy", "zz", "xy", "yz", "zx",
    {"polarization": P_GEN, "channel": "sf+"},
    {"polarization": P_GEN, "channel": "sf-"},
    {"polarization": P_GEN, "channel": "nsf"},
    {"bm": {"normal": [0, 0, 1]}, "component": "12"},
    {"bm": {"normal": [0, 0, 1]}, "component": "33"},
]
CS_IDS = [c if isinstance(c, str) else
          ("bm" + c["component"] if "bm" in c else "P-" + c["channel"])
          for c in CROSS_SECTIONS]


def _assert_same(res_a, res_b, rtol, scale):
    """Per q, the intensity summed over each energy cluster (the union of both
    spectra's energies, so a 3-channel k_case 2 result compares with a 2-mode
    supercell) agrees to rtol * scale; with equal mode counts the sorted energies
    agree too. `scale` is the model's trace band sum, so a cross-section that
    vanishes (chiral of a collinear state, an off-diagonal component) is held to
    the same absolute standard rather than to its own rounding noise."""
    Ea, Ia = np.real(res_a.energies), np.real(res_a.intensities)
    Eb, Ib = np.real(res_b.energies), np.real(res_b.intensities)
    e_tol = 1e-6 * max(1.0, np.abs(Eb).max())
    if Ea.shape == Eb.shape:
        assert np.sort(Ea, 1) == pytest.approx(np.sort(Eb, 1), abs=e_tol)
    for k in range(len(Ea)):
        es = np.sort(np.concatenate([Ea[k], Eb[k]]))
        edges = [es[0]] + [e for p, e in zip(es[:-1], es[1:]) if e - p > e_tol]
        for lo, hi in zip(edges, edges[1:] + [np.inf]):
            ina = (Ea[k] >= lo - e_tol / 2) & (Ea[k] < hi - e_tol / 2)
            inb = (Eb[k] >= lo - e_tol / 2) & (Eb[k] < hi - e_tol / 2)
            da = Ia[k, ina].sum() - Ib[k, inb].sum()
            assert abs(da) <= rtol * scale, (k, lo, Ia[k, ina], Ib[k, inb], scale)


def _scale(calc, q):
    return float(np.real(calc.sqw_batch(np.asarray(q), cross_section="trace")[1])
                 .sum(axis=1).max())


def _sqw(calc, q, cs, method):
    calc.sqw_method = method
    try:
        return calc.calculate_sqw(list(q), cross_section=cs)
    finally:
        calc.sqw_method = None


# ------------------------------------------------- single-k == explicit supercell
@functools.lru_cache(maxsize=None)
def _helix(k, n_cells, supercell):
    """J1-J2 chain screw (axis || chain) at commensurate k, either in the rotating
    frame or as the explicit magnetic_supercell built from the SAME single_k block
    (`repeat_periodically_as_spiral`), so the two are one structure by construction."""
    J1, J2 = (1.0, 0.5) if n_cells == 3 else (1.0, None)
    cfg = chain_config(J1, J2, k=(k, 0, 0), axis=(1, 0, 0),
                       ms_extra={"local_directions": [[0, 1, 0]]})
    if supercell:
        cfg["crystal_structure"]["magnetic_supercell"] = [n_cells, 1, 1]
    params = [J1] if J2 is None else [J1, J2]
    sm, calc = build_calc(cfg, params, cache_base=f"helix{n_cells}{int(supercell)}")
    return sm, calc


# k_case 2 with a polarization axis off q: the three channel modes of one degenerate
# group can carry a NEGATIVE sf+/sf- each (the channel tensors (I - nn^T) S' R1 are
# not Hermitian), and the per-mode clamp zeroes it before the group is summed: +6 %
# on the group total at one q here. Unclamped, the identity holds to 1e-15. The per-q
# pool clamps the same way, so this predates Colpa (OPEN_WORK item 9).
_CLAMP_XFAIL = pytest.mark.xfail(strict=True, reason="per-mode clamp, OPEN_WORK item 9")
_SUPERCELL_CASES = [
    pytest.param(n, kc, cs, id=f"{cid}-k{kc}",
                 marks=[_CLAMP_XFAIL] if (kc == 2 and cid in ("P-sf+", "P-sf-")) else [])
    for n, kc in ((3, 3), (2, 2)) for cs, cid in zip(CROSS_SECTIONS, CS_IDS)]


@pytest.mark.parametrize("n_cells,k_case,cs", _SUPERCELL_CASES)
def test_single_k_equals_the_explicit_supercell(n_cells, k_case, cs):
    sm, spiral = _helix(1.0 / n_cells, n_cells, supercell=False)
    _, cell = _helix(1.0 / n_cells, n_cells, supercell=True)
    assert spiral.k_case == k_case and cell.k_cart is None
    assert cell.supercell_ncells == n_cells
    # q in the scattering plane z = 0 (the BM frame needs it), off the chain axis.
    B = reciprocal_b_matrix(np.array(sm.unit_cell(), float))
    rng = np.random.default_rng(n_cells)
    rlu = np.column_stack([rng.uniform(-1, 1, 7), rng.uniform(-0.6, 0.6, 7),
                           np.zeros(7)])
    q = rlu @ B
    a = spiral.calculate_sqw(list(q), cross_section=cs)
    b = cell.calculate_sqw(list(q), cross_section=cs)
    assert a.energies.shape == (7, 3) and b.energies.shape == (7, n_cells)
    _assert_same(a, b, 1e-9, _scale(cell, q))


@pytest.mark.parametrize("channel", ["sf+", "sf-"])
def test_k_case_2_polarized_identity_holds_before_the_clamp(channel):
    """The unclamped contraction of the amplitudes satisfies the identity exactly, so
    the xfail above is the clamp and nothing else."""
    sm, spiral = _helix(0.5, 2, supercell=False)
    _, cell = _helix(0.5, 2, supercell=True)
    B = reciprocal_b_matrix(np.array(sm.unit_cell(), float))
    rng = np.random.default_rng(2)
    q = np.column_stack([rng.uniform(-1, 1, 7), rng.uniform(-0.6, 0.6, 7),
                         np.zeros(7)]) @ B
    cs = {"polarization": P_GEN, "channel": channel}
    va, _ = numerical.contract_cross_section_batch(
        spiral.sqw_amplitudes(q).correlation(), q, cs)
    vb, _ = numerical.contract_cross_section_batch(
        cell.sqw_amplitudes(q).correlation(), q, cs)
    assert np.real(va).min() < -1e-3                     # the clamp has work to do
    assert np.real(va).sum(1) == pytest.approx(np.real(vb).sum(1), abs=1e-12)


def test_the_supercell_identity_is_not_vacuous():
    """The k = 1/3 screw's chiral term is NOT small, so the identity above pins it."""
    sm, spiral = _helix(1.0 / 3, 3, supercell=False)
    q = qcart_chain(sm, [0.1, 0.2, 0.45])
    ch = spiral.calculate_sqw(list(q), cross_section="chiral").intensities
    pp = spiral.calculate_sqw(list(q), cross_section="perp").intensities
    assert np.abs(ch).max() > 0.5 * np.abs(pp).max()


# ---------------------------------------------------------- Colpa == per-q pool
# Slow: every call spawns the pool (about 15 s under xdist load). The fast suite
# reaches the pool through the two tests further down.
@pytest.mark.slow
@pytest.mark.parametrize("cs", CROSS_SECTIONS, ids=CS_IDS)
def test_commensurate_screw_equals_the_per_q_pool(cs):
    _, cell = _helix(1.0 / 3, 3, supercell=True)
    B = reciprocal_b_matrix(np.array(cell.sm.unit_cell(), float))
    rng = np.random.default_rng(1)
    q = np.column_stack([rng.uniform(-3, 3, 9), rng.uniform(-1, 1, 9),
                         np.zeros(9)]) @ B
    _assert_same(_sqw(cell, q, cs, "colpa"), _sqw(cell, q, cs, "perq"), 1e-5,
                 _scale(cell, q))


@pytest.mark.slow
@pytest.mark.parametrize("cs", ["perp", "chiral", "xy",
                                {"polarization": P_GEN, "channel": "sf-"}],
                         ids=["perp", "chiral", "xy", "P-sf-"])
@pytest.mark.parametrize("which", ["sw03", "case2"])
def test_single_k_equals_the_per_q_pool(which, cs):
    """Incommensurate SW03 (k_case 3, Sunny-pinned in test_single_k) and the k = 1/2
    chain (k_case 2): all three channels, against the pool."""
    if which == "sw03":
        sm, calc = build_calc(chain_config(-1.0, 2.0, k=(K_SW03, 0, 0)), [-1.0, 2.0],
                              cache_base="sw03_colpa")
    else:
        sm, calc = build_calc(chain_config(1.0, k=(0.5, 0, 0)), [1.0],
                              cache_base="case2_colpa")
    q = qcart_chain(sm, [0.05, 0.2, 0.35, 0.5, 0.61]) + np.array([0, 0.3, 0.2])
    _assert_same(_sqw(calc, q, cs, "colpa"), _sqw(calc, q, cs, "perq"), 1e-5,
                 _scale(calc, q))


# ------------------------------------------------------------ the per-q fallback
def _force_fallback(monkeypatch):
    real = numerical.colpa_batch

    def broken(h, n):
        e, t, ok = real(h, n)
        return e, t, np.zeros_like(ok)

    monkeypatch.setattr(numerical, "colpa_batch", broken)


@pytest.mark.parametrize("which", ["sw03", "case2"])
@pytest.mark.parametrize("cs", ["perp", "chiral", "xy"])
def test_single_k_fallback_assembles_the_same_amplitudes(monkeypatch, which, cs):
    """Every channel forced down KKdMatrix (left from K at +q, right from Kd at -q,
    each through its projector) must give the Colpa amplitudes' cross-section."""
    if which == "sw03":
        sm, calc = build_calc(chain_config(-1.0, 2.0, k=(K_SW03, 0, 0)), [-1.0, 2.0],
                              cache_base="sw03_fb")
    else:
        sm, calc = build_calc(chain_config(1.0, k=(0.5, 0, 0)), [1.0],
                              cache_base="case2_fb")
    q = qcart_chain(sm, [0.13, 0.27, 0.41]) + np.array([0, 0.4, -0.1])
    E0, I0, ok0 = calc.sqw_batch(q, cross_section=cs, return_ok=True)
    assert ok0.all()
    _force_fallback(monkeypatch)
    E1, I1, ok1 = calc.sqw_batch(q, cross_section=cs, return_ok=True)
    assert not ok1.any()
    _assert_same(numerical.SqwResult(q, E1, I1), numerical.SqwResult(q, E0, I0), 1e-5,
                 _scale(calc, q))


def test_a_supplied_non_minimum_gives_the_pools_answer():
    """The ferromagnet of a J1-J2 chain past the Lifshitz point (J2 > |J1|/4) is a
    stationary state that is unstable near the zone boundary: omega(q) < 0 there and
    H2 is not positive definite. That is what an `on_imaginary: warn` run computes
    about. Those q must take the per-q path (ok False), exactly where the closed form
    says, and the whole map must be the pool's."""
    J1, J2 = -1.0, 0.5
    sm, calc = build_calc(chain_config(J1, J2, ms_extra={"local_directions": [[0, 0, 1]]}),
                          [J1, J2], cache_base="fm_unstable")
    hs = np.linspace(0.02, 0.98, 25)
    q = qcart_chain(sm, hs) + np.array([0, 0.3, 0.1])
    w = 2 * 1.0 * (J1 * (np.cos(2 * np.pi * hs) - 1) + J2 * (np.cos(4 * np.pi * hs) - 1))
    _, _, ok = calc.sqw_batch(q, return_ok=True)
    unstable = w < -1e-9
    assert unstable.any() and (~unstable).any()
    assert list(ok) == list(~unstable)
    _assert_same(_sqw(calc, q, "perp", "colpa"), _sqw(calc, q, "perp", "perq"), 1e-5,
                 1.0)


# -------------------------------------------------------------------- selection
def _tiny():
    sm, calc = build_calc(chain_config(1.0, k=(0.5, 0, 0)), [1.0], cache_base="sel")
    return sm, calc


def test_colpa_is_the_default_and_the_pool_is_not_used(monkeypatch):
    sm, calc = _tiny()
    assert calc._sqw_method() == "colpa"
    from magcalc import core

    def no_pool(*a, **k):
        raise AssertionError("the per-q pool ran under sqw_method: colpa")

    monkeypatch.setattr(core, "Pool", no_pool)
    res = calc.calculate_sqw(list(qcart_chain(sm, [0.2, 0.3])))
    assert np.isfinite(res.intensities).all()


def test_calculation_sqw_method_perq_routes_to_the_pool(monkeypatch):
    sm, calc = _tiny()
    calc.sm.config.setdefault("calculation", {})["sqw_method"] = "perq"
    assert calc._sqw_method() == "perq"

    def no_colpa(*a, **k):
        raise AssertionError("sqw_batch ran under sqw_method: perq")

    monkeypatch.setattr(type(calc), "sqw_batch", no_colpa)
    res = calc.calculate_sqw(list(qcart_chain(sm, [0.2, 0.3])))
    assert np.isfinite(res.intensities).all()
    calc.sqw_method = "colpa"                   # the attribute overrides the config
    assert calc._sqw_method() == "colpa"


def test_an_unknown_sqw_method_raises():
    sm, calc = _tiny()
    calc.sm.config.setdefault("calculation", {})["sqw_method"] = "fast"
    with pytest.raises(ValueError, match="sqw_method"):
        calc.calculate_sqw(list(qcart_chain(sm, [0.2])))


def test_apply_g_refuses_a_single_k_structure():
    sm, calc = _tiny()
    with pytest.raises(NotImplementedError):
        calc.sqw_batch(qcart_chain(sm, [0.2]), apply_g=True)


def test_bm_out_of_plane_raises_before_any_work():
    sm, calc = _tiny()
    with pytest.raises(ValueError, match="scattering plane"):
        calc.sqw_batch(np.array([[0.3, 0.2, 0.5]]),
                       cross_section={"bm": {"normal": [0, 0, 1]}, "component": "11"})
