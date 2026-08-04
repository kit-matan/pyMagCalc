"""Histogram binning and NeXus import (Gap 4 #20).

Sunny's `BinningParameters` / `load_nxs`. The point is to put a calculation on the
same (|Q|, E) grid an experiment was reduced onto, so model and data can be
subtracted rather than compared by eye.

WHAT PINS WHAT. The binning needs no oracle: weight conservation, a delta landing in
the bin that contains it, and coarsening equalling the sum of the fine bins are exact
statements about what a histogram IS. `load_nxs` is pinned by a round trip through a
file this module writes -- which proves reader and writer agree, and nothing more.
Whether an arbitrary instrument's NeXus dialect is understood is a separate question
and is documented as such rather than asserted.
"""
import numpy as np
import pytest

from magcalc.binning import (BinningParameters, bin_events, bin_mode_list, load_nxs,
                             nxs_report, rebin, save_nxs)


def _params(nx=6, ne=8):
    return BinningParameters.uniform((0.0, 3.0), nx, (0.0, 8.0), ne)


def test_edges_centers_and_shape_are_consistent():
    p = _params(4, 5)
    assert p.shape == (4, 5)
    assert len(p.x_centers) == 4 and len(p.e_centers) == 5
    assert p.x_centers[0] == pytest.approx(0.375)
    assert np.all(np.diff(p.x_edges) > 0)


def test_weight_is_conserved():
    """Every event inside the grid lands in exactly one bin, so the total is
    preserved. A histogram that loses or duplicates weight would show up later as an
    unexplained scale factor -- the failure mode this repo keeps meeting."""
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(0.0, 3.0, n)
    e = rng.uniform(0.0, 8.0, n)
    w = rng.uniform(0.1, 2.0, n)
    hist = bin_events(x, e, w, _params())
    assert hist.sum() == pytest.approx(w.sum(), rel=1e-12)


def test_events_outside_the_grid_are_dropped_and_reported(caplog):
    """Dropping is correct; dropping SILENTLY is not."""
    p = _params()
    with caplog.at_level("INFO"):
        hist = bin_events([1.0, 99.0], [1.0, 1.0], [1.0, 5.0], p)
    assert hist.sum() == pytest.approx(1.0)
    assert "outside the grid" in caplog.text


def test_a_delta_lands_in_the_bin_that_contains_it():
    """The defining property, checked against arithmetic rather than a reference."""
    p = _params(6, 8)                       # dx = 0.5, dE = 1.0
    hist = bin_events([1.2], [3.4], [7.0], p)
    ix, ie = int(1.2 // 0.5), int(3.4 // 1.0)
    assert hist[ix, ie] == pytest.approx(7.0)
    assert hist.sum() == pytest.approx(7.0)


def test_density_divides_by_bin_area():
    p = _params(6, 8)
    counts = bin_events([1.2], [3.4], [7.0], p)
    dens = bin_events([1.2], [3.4], [7.0], p, density=True)
    area = (3.0 / 6) * (8.0 / 8)
    assert dens.max() == pytest.approx(counts.max() / area)


def test_rebin_equals_binning_on_the_coarse_grid():
    """Exact identity: coarsening a histogram and histogramming onto the coarse grid
    are the same operation."""
    rng = np.random.default_rng(3)
    n = 800
    x, e = rng.uniform(0, 3, n), rng.uniform(0, 8, n)
    w = rng.uniform(0.5, 1.5, n)
    fine = bin_events(x, e, w, _params(12, 16))
    coarse = bin_events(x, e, w, _params(6, 8))
    assert rebin(fine, 2, 2) == pytest.approx(coarse, rel=1e-12)


def test_rebin_rejects_an_indivisible_factor():
    with pytest.raises(ValueError, match="not divisible"):
        rebin(np.zeros((5, 4)), 2, 2)


def test_bin_mode_list_matches_flattening_by_hand():
    """LSWT gives (n_q, n_modes) deltas; binning them must equal binning the same
    events flattened."""
    p = _params()
    xs = np.array([0.4, 1.1, 2.2])
    E = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]])
    I = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    got = bin_mode_list(xs, E, I, p)
    want = bin_events(np.repeat(xs[:, None], 2, axis=1), E, I, p)
    assert got == pytest.approx(want, rel=1e-12)
    assert got.sum() == pytest.approx(I.sum())


def test_mode_list_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        bin_mode_list([0.5], np.zeros((1, 2)), np.zeros((1, 3)), _params())


# --------------------------------------------------------------------------
def test_nxs_round_trip(tmp_path):
    """Reader and writer agree -- which is what this proves, and all it proves."""
    p = _params(5, 7)
    rng = np.random.default_rng(1)
    hist = rng.uniform(size=p.shape)
    f = tmp_path / "map.nxs"
    save_nxs(f, hist, p, title="unit-test")
    out = load_nxs(f)
    assert out["signal"] == pytest.approx(hist, rel=1e-12)
    assert out["x"] == pytest.approx(p.x_centers, rel=1e-12)
    assert out["energy"] == pytest.approx(p.e_centers, rel=1e-12)
    assert out["params"].x_edges == pytest.approx(p.x_edges, rel=1e-12)
    assert out["edges_reconstructed"] is False


def test_edges_are_reconstructed_when_absent_and_flagged(tmp_path):
    """Without stored edges the reader assumes a uniform grid. That assumption is
    reported, because getting it wrong shifts every bin by half a width -- invisible
    on a colour map."""
    import h5py
    p = _params(4, 4)
    f = tmp_path / "no_edges.nxs"
    with h5py.File(f, "w") as h:
        d = h.create_group("entry/data")
        d.attrs["axes"] = [np.bytes_("x"), np.bytes_("energy")]
        d.create_dataset("signal", data=np.ones(p.shape))
        d.create_dataset("x", data=p.x_centers)
        d.create_dataset("energy", data=p.e_centers)
    out = load_nxs(f)
    assert out["edges_reconstructed"] is True
    assert out["params"].x_edges == pytest.approx(p.x_edges, rel=1e-9)


def test_non_uniform_axis_refuses_rather_than_guessing(tmp_path):
    import h5py
    f = tmp_path / "nonuniform.nxs"
    with h5py.File(f, "w") as h:
        d = h.create_group("entry/data")
        d.attrs["axes"] = [np.bytes_("x"), np.bytes_("energy")]
        d.create_dataset("signal", data=np.ones((3, 3)))
        d.create_dataset("x", data=np.array([0.0, 1.0, 4.0]))   # not uniform
        d.create_dataset("energy", data=np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError, match="not uniformly spaced"):
        load_nxs(f)


def test_a_file_without_a_histogram_is_refused_with_a_hint(tmp_path):
    import h5py
    f = tmp_path / "events.nxs"
    with h5py.File(f, "w") as h:
        h.create_dataset("entry/events/tof", data=np.arange(10.0))
    with pytest.raises(ValueError, match="reduced HISTOGRAMS"):
        load_nxs(f)
    assert "entry/events/tof" in nxs_report(f)
