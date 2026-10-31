"""The linked-cluster theorem on Rb2Cu3SnF12: OPEN_WORK item 11 (2026-10-31).

After the subtraction, an L-link cluster's weight must vanish below order L. The
engine checks this for every topology (`ADDITIVITY_TOL`); it is an exact identity,
so a failure is a defect, not rounding. On the dimer chain it holds to 1e-15 through
order 8, with and without DM and with two kinds of dimer. On Rb2Cu3SnF12 it fails
from order 6 on: at order 7, 9 088 of the 35 566 seven-link topologies keep a weight
at order 6 of up to 4.2e-3 of their scale (and 82 342 eight-link ones at order 8,
on LANTA). Pinned here, on one seven-link, six-node cluster at order 6 (0.2-0.7 s):

  * with the DM switched off (d_z = 0, SU(2) symmetric) it is additive;
  * with the DM on, the residual sits ONLY in the block of the T0 (Sz = 0)
    one-triplon states, which share the Sz = 0 sector with the vacuum; the Sz = +-1
    blocks are additive;
  * additivity itself, with the DM on: a strict xfail until item 11 is fixed.

Both kernel sets run every case: the NumPy and Rust PT agree on the residual, so it
is the method (a two-block canonical transformation, one-triplon block against the
rest, vacuum included), not a kernel.
"""
import importlib.util
import os

import numpy as np
import pytest

from magcalc.sun import dimer_series as ds

HERE = os.path.dirname(__file__)
RB = os.path.join(HERE, "..", "examples", "entangled", "Rb2Cu3SnF12",
                  "series_dispersion.py")

# a 7-link cluster on 6 dimers (two loops) of the Rb2Cu3SnF12 model as built by
# series_dispersion.build_model: (link type, cell) per link
CLUSTER = ((0, (0, 0, 0)), (2, (-1, -1, 0)), (5, (0, -1, 0)), (6, (0, 0, 0)),
           (7, (0, 0, 0)), (8, (-1, -1, 0)), (9, (-1, -1, 0)))
ORDER = 6            # below the cluster's size, so its whole weight must vanish


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
        lv, sz = m._node_levels(node_order)
        if raw is None and form == CLUSTER:
            raw = ds._cluster_pt(lv, m._local_links(form, node_order), order, sz) \
                if m.kernels == "numpy" else \
                m._rust_pt([(lv, m._local_links(form, node_order), sz)], order)[0]
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


def test_without_dm_the_cluster_is_additive(kernels):
    m = _model(kernels, dz=0.0)
    delta, _no = _residual(m)
    assert np.abs(delta).max() < 1e-12
    assert m._violations == {}


def test_with_dm_the_residual_is_confined_to_the_t0_block(kernels):
    m = _model(kernels)
    delta, node_order = _residual(m)
    sz = _sz_rows(m, node_order)
    t0, tpm = np.flatnonzero(sz == 0), np.flatnonzero(sz != 0)
    assert len(t0) == 6 and len(tpm) == 12
    # below order 6 everything cancels; at order 6 only the T0-T0 block survives
    assert np.abs(delta[:ORDER]).max() < 1e-12
    assert np.abs(delta[ORDER][np.ix_(tpm, tpm)]).max() < 1e-12
    assert np.abs(delta[ORDER][np.ix_(t0, tpm)]).max() < 1e-12
    assert np.abs(delta[ORDER][np.ix_(t0, t0)]).max() == pytest.approx(3.506e-3, rel=1e-3)
    # and the engine's check sees it, tallied for the level report
    (count, worst, form), = m._violations.values()
    assert count == 1 and form == CLUSTER and worst > ds.ADDITIVITY_TOL


@pytest.mark.xfail(strict=True, reason="the T0 sector is not cluster additive under "
                                       "DM, OPEN_WORK item 11")
def test_with_dm_the_cluster_is_additive(kernels):
    m = _model(kernels)
    delta, _no = _residual(m)
    assert np.abs(delta).max() < 1e-12
