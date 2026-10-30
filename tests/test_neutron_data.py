"""magcalc.neutron_data: readers, frames, the t0 energy map and the Q quadrature tables.

Every check is an exact identity on synthetic files written in the two real layouts
(Mantid MDNorm 4-D exports; per-run count-cell exports). No real data enters the repo.
"""
import numpy as np
import pytest

from magcalc.neutron_data import (
    BinnedData, QFrame, T0EnergyMap, read_count_cells, read_mdnorm_npz,
    NEUTRON_V_PER_SQRT_MEV)


def write_mdnorm(path, zone="Z", shape=(4, 5, 2, 6), dims=None, seed=0, weight=1.1,
                 empty_cell=True):
    """A synthetic MDNorm export: Poisson events times a per-energy event weight."""
    rng = np.random.default_rng(seed)
    w = weight * (1 + 0.1 * np.arange(shape[3]))
    counts = rng.poisson(3.0, shape).astype(float)
    norm = rng.uniform(0.5, 2.0, shape)
    if empty_cell:
        counts[0, 0, 0, :] = 0
        norm[0, 0, 0, :] = 0
    data = counts * w
    err2 = counts * w ** 2
    dims = dims or ["[H,0,0]|7.8|8.2|4", "[0,K,0]|-0.5|0.5|5", "[0,0,L]|-0.2|0.2|2",
                    "DeltaE|-1|5|6"]
    np.savez(path, **{f"{zone}_data": data.astype("f4"), f"{zone}_err2": err2.astype("f4"),
                      f"{zone}_norm": norm.astype("f4"), f"{zone}_dims": np.array(dims)},
             UB=np.eye(3), runs=np.array([1, 2]), proton_charge_uAh=np.array([10., 11.]))
    return counts, norm, w


def write_cells(path, run, H, K, L, E, counts, exposure, dE=0.05):
    np.savez(path, H=np.asarray(H, "f4"), K=np.asarray(K, "f4"), L=np.asarray(L, "f4"),
             E=np.asarray(E, "f4"), counts=np.asarray(counts, "u4"),
             exposure=np.asarray(exposure, "f8"), dE=np.array(dE), run=np.array(run, "u4"))


# ------------------------------------------------------------------ MDNorm reader
def test_mdnorm_totals_and_weighted_to_counts(tmp_path):
    p = tmp_path / "m.npz"
    counts, norm, w = write_mdnorm(p)
    d = read_mdnorm_npz(p, "Z", samples_per_cell=8)
    tot = d.meta["file_totals"]
    assert tot["data"] == pytest.approx((counts * w).sum(), rel=1e-6)
    assert tot["norm"] == pytest.approx(norm.sum(), rel=1e-6)
    # w(E) = median(err2/data) recovers the event weight; n = data/w the event counts.
    assert np.array(d.meta["event_weight"]) == pytest.approx(w, rel=1e-6)
    assert d.counts.sum() == pytest.approx(counts.sum(), rel=1e-6)
    assert d.exposure.sum() == pytest.approx((norm / w).sum(), rel=1e-6)
    # Zero-norm bins are dropped: one empty (H, K, L) cell, all energies.
    assert d.n_bins == counts.size - counts.shape[3]
    assert d.n_cells == np.prod(counts.shape[:3]) - 1
    assert d.meta["runs"] == [1, 2]


def test_mdnorm_supplied_weight_and_raw_arrays(tmp_path):
    p = tmp_path / "m.npz"
    counts, norm, w = write_mdnorm(p)
    d = read_mdnorm_npz(p, "Z", weight=2.0 * w)
    assert d.counts.sum() == pytest.approx(counts.sum() / 2, rel=1e-6)
    assert d.meta["weight_source"] == "supplied array"
    assert d.raw["signal"].sum() == pytest.approx((counts * w).sum(), rel=1e-6)


def test_mdnorm_refuses_data_without_normalization(tmp_path):
    p = tmp_path / "bad.npz"
    write_mdnorm(p)
    z = dict(np.load(p))
    z["Z_data"][0, 0, 0, 0] = 1.0
    np.savez(p, **z)
    with pytest.raises(ValueError, match="zero normalization"):
        read_mdnorm_npz(p, "Z")


def test_mdnorm_samples_lie_in_their_cells(tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p)
    d = read_mdnorm_npz(p, "Z", samples_per_cell=16)
    for b in range(0, d.n_bins, 37):
        c = d.bin_cells[d.bin_ptr[b]]
        s = d.samples_hkl[d.cell_ptr[c]:d.cell_ptr[c + 1]]
        for ax, name, width in ((0, "H", 0.1), (1, "K", 0.2), (2, "L", 0.2)):
            assert np.all(np.abs(s[:, ax] - d.coords[name][b]) <= width / 2 + 1e-12)
        # A scrambled Sobol set of 2^k points is balanced: the mean is the centre.
        assert s.mean(0) == pytest.approx(
            [d.coords["H"][b], d.coords["K"][b], d.coords["L"][b]], abs=0.01)


def test_mdnorm_region_and_aggregate(tmp_path):
    p = tmp_path / "m.npz"
    counts, norm, w = write_mdnorm(p)
    d = read_mdnorm_npz(p, "Z", region={"E": (0, 5)})
    assert np.all(d.coords["E"] > 0)
    a = d.aggregate(d.coords["iK"] * 100 + d.coords["iE"])
    assert a.counts.sum() == pytest.approx(d.counts.sum(), rel=1e-12)
    assert a.exposure.sum() == pytest.approx(d.exposure.sum(), rel=1e-12)
    # Each merged bin's cell weights are its members' exposure fractions.
    for g in range(a.n_bins):
        mem = np.flatnonzero((d.coords["iK"] == a.coords["iK"][g]) &
                             (d.coords["iE"] == a.coords["iE"][g]))
        cells = a.bin_cells[a.bin_ptr[g]:a.bin_ptr[g + 1]]
        wts = a.bin_cell_w[a.bin_ptr[g]:a.bin_ptr[g + 1]]
        want = {int(d.bin_cells[m]): d.exposure[m] / d.exposure[mem].sum() for m in mem}
        assert dict(zip(cells.tolist(), wts)) == pytest.approx(want, rel=1e-12)


def test_aggregate_refuses_mismatched_energy_edges(tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p)
    d = read_mdnorm_npz(p, "Z")
    with pytest.raises(ValueError, match="energy edges"):
        d.aggregate(np.zeros(d.n_bins, int))


def test_save_load_round_trip(tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p)
    d = read_mdnorm_npz(p, "Z", energy_map=T0EnergyMap.from_elastic_offset(12, 0.2),
                        frame=QFrame.from_2d([[1.01, 0.02], [0.0, 0.99]]))
    s = d.select_region(E=(0, 4))
    out = tmp_path / "native.npz"
    s.save(out)
    t = BinnedData.load(out)
    for name in ("counts", "exposure", "e_lo", "e_hi", "samples_hkl", "cell_ptr",
                 "bin_ptr", "bin_cells", "bin_cell_w", "sample_w", "smear_normals"):
        assert np.array_equal(getattr(s, name), getattr(t, name)), name
    assert t.frame.to_model == pytest.approx(s.frame.to_model)
    assert t.energy_map(1.0) == s.energy_map(1.0)
    assert t.meta == s.meta
    assert set(t.coords) == set(s.coords)


def test_load_refuses_a_future_format(tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p)
    out = tmp_path / "n.npz"
    read_mdnorm_npz(p, "Z").save(out)
    z = dict(np.load(out))
    z["format_version"] = np.array(99)
    np.savez(out, **z)
    with pytest.raises(ValueError, match="format version"):
        BinnedData.load(out)


# ------------------------------------------------------------------ frames
def test_frame_map_gives_the_true_cartesian_q():
    """Data reduced in a rotated, rescaled frame plus its map == the true frame."""
    rng = np.random.default_rng(1)
    B = 2 * np.pi * np.diag([1 / 20.6, 1 / 8.4, 1 / 6.4])
    hkl_true = rng.uniform(-3, 3, (20, 3))
    U = np.array([[0.993, 0.009, 0.0], [0.005, 1.006, 0.0], [0.0, 0.0, 1.0]])
    hkl_data = hkl_true @ np.linalg.inv(U).T          # what a wrong UB reports
    f = QFrame(U)
    assert f.cartesian(hkl_data, B) == pytest.approx(hkl_true @ B, rel=1e-13)


def test_frame_from_two_bragg_peaks():
    """The CNCS construction: 800 observed at (7.9476, 0.0420), 020 at (0.0729, 2.0126);
    M = diag(8, 2) . inv([[7.9476, 0.0729], [0.0420, 2.0126]])."""
    obs = [[7.9476, 0.0420, 0], [0.0729, 2.0126, 0]]
    f = QFrame.from_peaks(obs, [[8, 0, 0], [0, 2, 0]])
    U = np.array([[7.9476, 0.0729], [0.0420, 2.0126]])
    M = np.diag([8.0, 2.0]) @ np.linalg.inv(U)
    assert f.to_model[:2, :2] == pytest.approx(M, rel=1e-14)
    assert f.model_hkl(obs) == pytest.approx(np.array([[8.0, 0, 0], [0, 2, 0]]), abs=1e-12)
    assert f.to_model[2] == pytest.approx([0, 0, 1])


# ------------------------------------------------------------------ energy map
def test_t0_map_is_the_flight_time_identity_and_inverts():
    ei, el = 12.0, 0.2346
    m = T0EnergyMap.from_elastic_offset(ei, el)
    assert m(el) == pytest.approx(0.0, abs=1e-12)
    e = np.linspace(-1, 9, 41)
    assert m.inverse(m(e)) == pytest.approx(e, abs=1e-12)
    # Explicitly: t_meas from E_meas, then the true transfer from t_meas - delta.
    v, L2 = NEUTRON_V_PER_SQRT_MEV, 3.5
    t = L2 / (v * np.sqrt(ei - e))
    vf = L2 / (t - m.delta)
    assert m(e) == pytest.approx(ei - (vf / v) ** 2, abs=1e-12)
    assert np.all(np.diff(m(e)) > 0)


def test_energy_map_applies_to_bin_edges(tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p)
    m = T0EnergyMap.from_elastic_offset(12.0, 0.1)
    d = read_mdnorm_npz(p, "Z", energy_map=m)
    raw_lo = d.coords["E_raw"] - 0.5
    assert d.e_lo == pytest.approx(m(raw_lo), abs=1e-12)


# ------------------------------------------------------------------ count cells
def _cell_files(tmp_path):
    a = tmp_path / "run_1.npz"
    b = tmp_path / "run_2.npz"
    c = tmp_path / "run_3.npz"
    write_cells(a, 1, H=[0.0, 0.1, -0.1], K=[1.01, 1.02, 1.07], L=[0, 0.05, -0.05],
                E=[1.0, 1.0, 1.05], counts=[3, 0, 2], exposure=[1.0, 3.0, 2.0])
    write_cells(b, 2, H=[], K=[], L=[], E=[], counts=[], exposure=[])        # empty run
    write_cells(c, 3, H=[0.05, 0.02], K=[1.03, 1.08], L=[0.01, 0.0], E=[1.0, 1.05],
                counts=[1, 4], exposure=[2.0, 1.5])
    return [a, b, c]


def test_count_cells_totals_and_empty_runs(tmp_path):
    cl = read_count_cells(_cell_files(tmp_path))
    rt = cl.meta["read_totals"]
    assert (rt["files"], rt["cells"], rt["counts"], rt["empty_runs"]) == (3, 5, 10, 1)
    assert len(cl) == 5


def test_count_cell_aggregation_is_count_and_exposure_preserving(tmp_path):
    cl = read_count_cells(_cell_files(tmp_path))
    d = cl.aggregate({"K": [1.0, 1.05, 1.1]}, samples_per_bin=8, seed=3)
    # (K bin, native E): (0, 1.0) has cells 0, 1, 3; (1, 1.05) has 2 and 4.
    assert d.n_bins == 2
    assert d.counts.tolist() == [4.0, 6.0]
    assert d.exposure == pytest.approx([6.0, 3.5])
    assert d.e_lo == pytest.approx([0.975, 1.025])
    assert d.e_hi == pytest.approx([1.025, 1.075])


def test_exposure_weighted_draws(tmp_path):
    """Systematic sampling of the exposure CDF: with exposures 1:3 and 8 samples the
    cells get exactly 2 and 6; all exposure in one cell puts every sample there."""
    p = tmp_path / "r.npz"
    write_cells(p, 9, H=[0.0, 0.1], K=[1.01, 1.02], L=[0, 0], E=[1.0, 1.0],
                counts=[0, 0], exposure=[1.0, 3.0])
    d = read_count_cells([p]).aggregate({"K": [1.0, 1.1]}, samples_per_bin=8, seed=0)
    h = d.samples_hkl[:, 0]
    assert (np.isclose(h, 0.0).sum(), np.isclose(h, 0.1).sum()) == (2, 6)
    write_cells(p, 9, H=[0.0, 0.1], K=[1.01, 1.02], L=[0, 0], E=[1.0, 1.0],
                counts=[0, 0], exposure=[0.0, 3.0])
    d = read_count_cells([p]).aggregate({"K": [1.0, 1.1]}, samples_per_bin=8, seed=0)
    assert np.allclose(d.samples_hkl, [0.1, 1.02, 0.0], atol=1e-6)


def test_count_cell_draws_are_reproducible(tmp_path):
    files = _cell_files(tmp_path)
    a = read_count_cells(files).aggregate({"K": [1.0, 1.05, 1.1]}, samples_per_bin=16, seed=5)
    b = read_count_cells(files).aggregate({"K": [1.0, 1.05, 1.1]}, samples_per_bin=16, seed=5)
    assert np.array_equal(a.samples_hkl, b.samples_hkl)


def test_count_cell_read_window(tmp_path):
    cl = read_count_cells(_cell_files(tmp_path), windows={"E": (0.99, 1.01)})
    assert len(cl) == 3
    assert cl.meta["read_totals"]["cells"] == 5     # totals are of every file


def test_count_cell_samples_carry_their_run(tmp_path):
    """Each drawn sample keeps the run of the cell it came from (the Q-E resolution
    needs it: the run fixes the rotation and so the pixel), through select and
    save/load."""
    paths = []
    for run, k0 in ((7, 0.1), (9, 0.3)):
        p = tmp_path / f"r{run}.npz"
        write_cells(p, run, [0.0, 0.0], [k0, k0 + 0.01], [0.0, 0.0], [1.0, 1.0], [3, 4],
                    [1.0, 2.0])
        paths.append(p)
    d = read_count_cells(paths).aggregate({"K": [0.0, 0.2, 0.4]}, samples_per_bin=6)
    k = d.samples_hkl[:, 1]
    assert np.all(d.sample_run[k < 0.2] == 7) and np.all(d.sample_run[k >= 0.2] == 9)
    sub = d.select([1])
    assert np.all(sub.sample_run == 9)
    d.save(tmp_path / "d.npz")
    assert np.array_equal(BinnedData.load(tmp_path / "d.npz").sample_run, d.sample_run)
