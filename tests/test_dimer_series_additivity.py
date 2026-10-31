"""The linked-cluster theorem when the vacuum and a one-particle sector mix
(OPEN_WORK item 11, fixed 2026-10-31).

After the subtraction, an L-link cluster's weight must vanish below order L. The
engine checks this for every topology (`ADDITIVITY_TOL`); it is an exact identity.
With out-of-plane DM only total Sz is conserved, so the dressed vacuum and the T0
(Sz = 0) one-triplon states mix, and the des Cloizeaux one-particle H_eff is then
not cluster additive (Hoermann and Schmidt, SciPost Phys. 15, 097 (2023), Sec. 2.4):
on Rb2Cu3SnF12, 9 088 of the 35 566 seven-link topologies kept an order-6 weight
of up to 4.2e-3 at order 7. `_pca_one_particle` (their Sec. 3.1) removes each
one-particle state's vacuum component before orthonormalizing. Pinned here, under
both kernel sets:

  * one seven-link, six-node Rb2Cu3SnF12 cluster at order 6 (below its size, so
    its whole weight must vanish): additive with the DM off, additive with it on,
    and -- with the des Cloizeaux H_eff put back -- the old 3.506e-3, confined to
    the T0 block, so the pin is known to bite;
  * on a cluster, the new and old H_eff have the same eigenvalue series (one is a
    unitary rotation of the other) and identical Sz = +-1 blocks;
  * an independent oracle, momentum-resolved ED of Hoermann and Schmidt's own toy
    model (a transverse-field Ising ring with a parity-breaking sz sx term, where
    the old H_eff hops a particle between disconnected clusters at order 4): the
    order-6 series' error must fall as g^7 with the couplings g; with the old
    H_eff it falls as g^4.
"""
import importlib.util
import os

import numpy as np
import pytest
import scipy.sparse as sp

from magcalc.constants import GAMMA_ELECTRON as GAMMA, MU_B
from magcalc.sun import dimer_series as ds

HERE = os.path.dirname(__file__)
RB = os.path.join(HERE, "..", "examples", "entangled", "Rb2Cu3SnF12",
                  "series_dispersion.py")

# a 7-link cluster on 6 dimers (two loops) of the Rb2Cu3SnF12 model as built by
# series_dispersion.build_model: (link type, cell) per link
CLUSTER = ((0, (0, 0, 0)), (2, (-1, -1, 0)), (5, (0, -1, 0)), (6, (0, 0, 0)),
           (7, (0, 0, 0)), (8, (-1, -1, 0)), (9, (-1, -1, 0)))
ORDER = 6            # below the cluster's size, so its whole weight must vanish


def _des_cloizeaux(Om1, S, M, Om0, vac, order):
    """The one-particle H_eff before the fix."""
    return ds._sandwich(ds._inv_sqrt_series(S, order), M, order)


def _model(kernels, dz=None):
    spec = importlib.util.spec_from_file_location("rb_additivity", RB)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if dz is not None:
        mod.DZ = dz
    m = mod.build_model()[0]
    m.kernels = kernels
    return m


def _residual(m):
    """The subtracted weight of CLUSTER at every order (none is kept: L > ORDER)."""
    assert len(CLUSTER) == 7 and m._connected(frozenset(CLUSTER))
    got = {}
    orig = m._subtracted

    def keep(form, node_order, order, raw=None):
        if raw is None and form == CLUSTER:
            lv, sz = m._node_levels(node_order)
            links = m._local_links(form, node_order)
            raw = (ds._cluster_pt(lv, links, order, sz) if m.kernels == "numpy"
                   else m._rust_pt([(lv, links, sz)], order)[0])
        out = orig(form, node_order, order, raw)
        if form == CLUSTER:
            got["delta"], got["node_order"] = raw[1], node_order
        return out
    m._subtracted = keep
    m._weight_by_struct(CLUSTER, ORDER)
    return got["delta"], got["node_order"]


def _sz_rows(m, node_order):
    return np.array([m.sz_levels[d][a] for (d, _c) in node_order
                     for a in range(1, m.nloc)])


@pytest.mark.parametrize("dz", [0.0, None], ids=["dm_off", "dm_on"])
def test_the_cluster_is_additive(kernels, dz):
    m = _model(kernels, dz=dz)
    delta, _no = _residual(m)
    assert np.abs(delta).max() < 1e-12
    assert m._violations == {}


def test_the_old_heff_fails_in_the_t0_block_only(monkeypatch):
    monkeypatch.setattr(ds, "_pca_one_particle", _des_cloizeaux)
    m = _model("numpy")
    delta, node_order = _residual(m)
    sz = _sz_rows(m, node_order)
    t0, tpm = np.flatnonzero(sz == 0), np.flatnonzero(sz != 0)
    assert len(t0) == 6 and len(tpm) == 12
    assert np.abs(delta[:ORDER]).max() < 1e-12
    assert np.abs(delta[ORDER][np.ix_(tpm, tpm)]).max() < 1e-12
    assert np.abs(delta[ORDER][np.ix_(t0, tpm)]).max() < 1e-12
    assert np.abs(delta[ORDER][np.ix_(t0, t0)]).max() == pytest.approx(3.506e-3, rel=1e-3)
    (count, worst, form), = m._violations.values()
    assert count == 1 and form == CLUSTER and worst > ds.ADDITIVITY_TOL


def test_new_and_old_heff_share_eigenvalues_and_the_sz_pm1_blocks(monkeypatch):
    m = _model("numpy")
    skey, node_order = m._struct(CLUSTER)
    lv, sz = m._node_levels(node_order)
    links = m._local_links(CLUSTER, node_order)
    order = 7
    _g, new = ds._cluster_pt(lv, links, order, sz)
    monkeypatch.setattr(ds, "_pca_one_particle", _des_cloizeaux)
    _g, old = ds._cluster_pt(lv, links, order, sz)
    rows = _sz_rows(m, node_order)
    t0, tpm = np.flatnonzero(rows == 0), np.flatnonzero(rows != 0)
    assert np.array_equal(new[:, tpm][:, :, tpm], old[:, tpm][:, :, tpm])
    assert np.abs(new[:, t0][:, :, t0] - old[:, t0][:, :, t0]).max() > 1e-3
    # one is a unitary rotation of the other: the T0 block's eigenvalue series agree
    loc0 = np.diag([lv[d][a] - lv[d][0] for d in range(len(lv))
                    for a in range(1, len(lv[0]))])[np.ix_(t0, t0)]
    en = ds.eigenvalue_series([new[k][np.ix_(t0, t0)] + (loc0 if k == 0 else 0)
                               for k in range(order + 1)], order)
    eo = ds.eigenvalue_series([old[k][np.ix_(t0, t0)] + (loc0 if k == 0 else 0)
                               for k in range(order + 1)], order)
    en = np.array(sorted(map(tuple, np.real(en))))
    eo = np.array(sorted(map(tuple, np.real(eo))))
    assert np.abs(en - eo).max() < 1e-10


# --------------------------------------------------------------------------
# Independent oracle: Hoermann and Schmidt's toy model against momentum ED
# --------------------------------------------------------------------------
L_RING = 10


def _ring_ed(g):
    """E_1(k) - E_0 on a periodic ring of L_RING spins for
    H = sum sz + sum [g sx sx + g (sz sx + sx sz)], k = 2 pi n / L, n = 0..L/2."""
    L = L_RING
    N = 2 ** L
    sx = sp.csr_matrix([[0, 1], [1, 0]], dtype=float)
    sz = sp.csr_matrix([[1, 0], [0, -1]], dtype=float)

    def op(o, i):
        return sp.kron(sp.kron(sp.identity(2 ** i), o), sp.identity(2 ** (L - i - 1)),
                       format="csr")
    H = sum(op(sz, i) for i in range(L))
    for i in range(L):
        j = (i + 1) % L
        H = H + g * op(sx, i) @ op(sx, j) + g * (op(sz, i) @ op(sx, j) + op(sx, i) @ op(sz, j))
    H = H.toarray()
    idx = np.arange(N)
    shift = ((idx << 1) & (N - 1)) | (idx >> (L - 1))

    def lowest(kk, n):
        seen, cols = np.zeros(N, bool), []
        for s0 in range(N):
            if seen[s0]:
                continue
            orb, s = [s0], s0
            for _ in range(1, L):
                s = shift[s]
                orb.append(s)
            seen[orb] = True
            v = np.zeros(N, complex)
            for t, s in enumerate(orb):
                v[s] += np.exp(-1j * kk * t)
            if np.linalg.norm(v) > 1e-9:
                cols.append(v / np.linalg.norm(v))
        B = np.array(cols).T
        return np.linalg.eigvalsh(B.conj().T @ H @ B)[:n]
    e0 = lowest(0.0, 1)[0]
    ks = [2 * np.pi * n / L for n in range(L // 2 + 1)]
    return ks, np.array([(lowest(k, 2)[1] if n == 0 else lowest(k, 1)[0]) - e0
                         for n, k in enumerate(ks)])


def _ring_series(g, kernels, ks, order):
    M = 4 * np.array([[g, 0, g], [0, 0, 0], [g, 0, 0]])     # sigma = 2 S
    m = ds.DimerSeriesModel.from_spin_arrays(
        np.diag([1.0, 10.0, 10.0]), [0.5], [[0, 0, 0]], [(0, 0, (1, 0, 0), M)],
        units=[[0]], field=[0, 0, 2.0 / (GAMMA * MU_B)], kernels=kernels)
    assert m.sz_levels is None and np.allclose(m.levels[0], [-1, 1])
    return np.array([sum(float(np.real(h[0, 0]))
                         for h in m.h_series_at(np.array([k, 0, 0]), order)) for k in ks])


def _error_ratio(kernels, order=6):
    errs = []
    for g in (0.04, 0.02):
        ks, exact = _ring_ed(g)
        errs.append(np.abs(_ring_series(g, kernels, ks, order) - exact).max())
    return errs[0] / errs[1], errs


def test_toy_ring_series_converges_as_g_to_the_order_plus_one(kernels):
    ratio, errs = _error_ratio(kernels)
    assert errs[0] < 1e-6
    assert 0.7 * 2 ** 7 < ratio < 1.3 * 2 ** 7, (ratio, errs)


def test_toy_ring_series_with_the_old_heff_does_not(monkeypatch):
    monkeypatch.setattr(ds, "_pca_one_particle", _des_cloizeaux)
    ratio, errs = _error_ratio("numpy")
    assert ratio < 2 ** 5, (ratio, errs)          # ~2^4: a g^4 error never cancels
