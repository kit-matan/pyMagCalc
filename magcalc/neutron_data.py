"""Reduced inelastic neutron data on its own bins: one container for every instrument.

The forward model (`magcalc.forward`) predicts the expected count in each bin of a
`BinnedData`. This module holds everything that is a property of the DATA rather than
of the model:

- the bins: counts and exposure per bin, and each bin's true energy edges;
- the Q quadrature of each bin: a two-level table bin -> Q cells (with exposure
  weights) -> sample points (with weights), so that a bin's model value is the
  exposure-weighted average of the model over the Q it actually integrates;
- the Q frame (`QFrame`): a 3x3 map from the data's r.l.u. to the model's;
- the energy map (`T0EnergyMap`): a direct-geometry emission-time correction applied
  to the energy-bin EDGES;
- metadata (Ei, temperature, field, instrument, sources and their hashes).

Two kinds of reduced data are read:

- `read_mdnorm_npz`: Mantid MDNorm 4-D histograms (H, K, L, E) saved as
  `<zone>_data`, `<zone>_err2`, `<zone>_norm` and `<zone>_dims`. The data are
  WEIGHTED counts; with the per-energy event weight w(E) = median(err2/data) they
  become scaled-Poisson effective counts n = data/w and exposure X = norm/w, so the
  model in units of data/norm predicts mu = X * s. The Q cells are the histogram's
  own (H, K, L) boxes; their samples are scrambled Sobol points uniform in the box and
  shared by every energy bin of that cell, since Q does not move with E there.
- `read_count_cells` + `CellList.aggregate`: true detector/energy cells (counts and
  exposure per cell, each with its own (H, K, L, E)). For a fixed pixel Q moves with
  E, so each analysis bin gets its own samples, drawn from its member cells'
  (H, K, L) in proportion to their exposure (systematic sampling of the exposure CDF:
  every draw is marginally exposure-distributed, so the estimate is exact in
  expectation).

`BinnedData.aggregate` merges histogram bins with identical energy edges (e.g. summing
H and L within a window, per K column), carrying exposure-weighted mixture weights over
the member cells -- the exact per-bin, per-energy exposure, never an energy-averaged
coverage.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
NEUTRON_V_PER_SQRT_MEV = 437.393   # neutron speed in m/s per sqrt(meV)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- frames
@dataclass
class QFrame:
    """Map from the data's r.l.u. to the model's: hkl_model = to_model @ hkl_data.

    A reduction done with a nominal or slightly wrong UB leaves the data in a frame that
    is not the model's; fixing the map from measured Bragg peaks (e.g. 800 and 020
    observed at (7.9476, 0.0420) and (0.0729, 2.0126)) is `from_peaks`. Cartesian Q is
    the map followed by the model's B matrix (rows b1, b2, b3, as
    `runner.compute_b_matrix` returns them).
    """
    to_model: np.ndarray = field(default_factory=lambda: np.eye(3))
    label: str = ""

    def __post_init__(self):
        self.to_model = np.asarray(self.to_model, dtype=float).reshape(3, 3)

    @classmethod
    def identity(cls, label="identity"):
        return cls(np.eye(3), label)

    @classmethod
    def from_2d(cls, m2, axes=(0, 1), label=""):
        """Embed a 2x2 in-plane map (acting on data axes `axes`) into 3x3."""
        m = np.eye(3)
        i, j = axes
        m2 = np.asarray(m2, dtype=float)
        m[np.ix_([i, j], [i, j])] = m2
        return cls(m, label)

    @classmethod
    def from_peaks(cls, observed, true, label=""):
        """The linear map taking observed peak positions (data frame) to their true
        indices; three non-coplanar peaks fix it, two in-plane peaks fix the in-plane
        block (the third axis is then the identity)."""
        obs = np.atleast_2d(np.asarray(observed, float))
        tru = np.atleast_2d(np.asarray(true, float))
        if obs.shape[0] == 3:
            return cls(tru.T @ np.linalg.inv(obs.T), label)
        if obs.shape[0] == 2:
            plane = [a for a in range(3) if np.any(obs[:, a] != 0) or np.any(tru[:, a] != 0)]
            if len(plane) != 2:
                raise ValueError("two peaks must lie in a coordinate plane")
            o = obs[:, plane].T
            t = tru[:, plane].T
            return cls.from_2d(t @ np.linalg.inv(o), axes=tuple(plane), label=label)
        raise ValueError("from_peaks needs two (in-plane) or three peaks")

    def model_hkl(self, hkl):
        return np.asarray(hkl, float) @ self.to_model.T

    def cartesian(self, hkl, b_matrix):
        return self.model_hkl(hkl) @ np.asarray(b_matrix, float)

    def to_dict(self):
        return {"to_model": self.to_model.tolist(), "label": self.label}

    @classmethod
    def from_dict(cls, d):
        return cls(np.asarray(d["to_model"], float), d.get("label", ""))


# --------------------------------------------------------------------------- energy
@dataclass
class T0EnergyMap:
    """Direct-geometry energy transfer recomputed with an emission-time error delta.

    The reduction computed E_meas = Ei - (m/2)(L2/t)^2 from a final flight time t that
    is delta too long; the true transfer is

        t = L2 / (V sqrt(Ei - E_meas)),   E_true = Ei - (L2 / ((t - delta) V))^2

    with V = 437.393 m/s per sqrt(meV), L2 in m and delta in s. `from_elastic_offset`
    fixes delta by requiring the measured elastic line E_el to map to 0. The map is
    applied to bin EDGES (a histogram bin [a, b] in E_meas is [f(a), f(b)] in E_true,
    since f is monotonic).
    """
    ei: float
    delta: float
    l2: float = 3.5
    kind: str = "t0"

    @classmethod
    def from_elastic_offset(cls, ei, e_elastic, l2=3.5):
        v = NEUTRON_V_PER_SQRT_MEV
        delta = l2 / (v * np.sqrt(ei - e_elastic)) - l2 / (v * np.sqrt(ei))
        return cls(float(ei), float(delta), float(l2))

    def __call__(self, e_meas):
        v = NEUTRON_V_PER_SQRT_MEV
        t = self.l2 / (v * np.sqrt(self.ei - np.asarray(e_meas, float))) - self.delta
        return self.ei - (self.l2 / t / v) ** 2

    def inverse(self, e_true):
        v = NEUTRON_V_PER_SQRT_MEV
        t = self.l2 / (v * np.sqrt(self.ei - np.asarray(e_true, float))) + self.delta
        return self.ei - (self.l2 / t / v) ** 2

    def to_dict(self):
        return {"kind": self.kind, "ei": self.ei, "delta": self.delta, "l2": self.l2}


def energy_map_from_dict(d):
    if not d:
        return None
    if d.get("kind") == "t0":
        return T0EnergyMap(d["ei"], d["delta"], d.get("l2", 3.5))
    raise ValueError(f"unknown energy map {d}")


# --------------------------------------------------------------------------- container
def _check_rows_sum_to_one(ptr, w, what):
    sums = np.add.reduceat(w, ptr[:-1]) if len(w) else np.zeros(0)
    empty = ptr[1:] == ptr[:-1]
    if np.any(empty):
        raise ValueError(f"{what}: a row has no entries")
    if len(sums) and np.max(np.abs(sums - 1.0)) > 1e-9:
        raise ValueError(f"{what}: weights do not sum to 1 per row "
                         f"(max |sum - 1| = {np.max(np.abs(sums - 1.0)):.2e})")


class BinnedData:
    """Bins of reduced INS data with their exact Q quadrature.

    Per bin b (length Nb): `counts` n_b (effective Poisson counts), `exposure` X_b (the
    model in the data's intensity units times X_b is the expected count), `e_lo`/`e_hi`
    true energy edges (meV), and `coords` (named per-bin arrays for display/selection).

    Quadrature: bin b covers cells bin_cells[bin_ptr[b]:bin_ptr[b+1]] with weights
    bin_cell_w (summing to 1); cell c owns samples cell_ptr[c]:cell_ptr[c+1] of
    `samples_hkl` (data-frame r.l.u.) with weights `sample_w` (summing to 1 per cell),
    and optional `smear_normals` (standard normal deviates per sample, for a Gaussian Q
    smearing applied in the data frame by the forward model) and `sample_run` (the run
    each sample was measured in, for count cells: it fixes the goniometer angle and so
    the pixel, which the Q-E resolution needs).

    The bin's model value is s_b = sum_c w_bc sum_s w_s f(Q_s, [e_lo, e_hi]).
    """

    def __init__(self, counts, exposure, e_lo, e_hi, *, samples_hkl, cell_ptr,
                 bin_ptr, bin_cells, bin_cell_w, sample_w=None, smear_normals=None,
                 frame=None, coords=None, meta=None, raw=None, energy_map=None,
                 sample_run=None):
        self.counts = np.asarray(counts, float)
        self.exposure = np.asarray(exposure, float)
        self.e_lo = np.asarray(e_lo, float)
        self.e_hi = np.asarray(e_hi, float)
        self.samples_hkl = np.asarray(samples_hkl, float).reshape(-1, 3)
        self.cell_ptr = np.asarray(cell_ptr, np.int64)
        self.bin_ptr = np.asarray(bin_ptr, np.int64)
        self.bin_cells = np.asarray(bin_cells, np.int64)
        self.bin_cell_w = np.asarray(bin_cell_w, float)
        ns = len(self.samples_hkl)
        nc = len(self.cell_ptr) - 1
        self.sample_w = (np.asarray(sample_w, float) if sample_w is not None else
                         np.repeat(1.0 / np.diff(self.cell_ptr), np.diff(self.cell_ptr)))
        self.smear_normals = (None if smear_normals is None
                              else np.asarray(smear_normals, float).reshape(ns, 3))
        self.sample_run = None if sample_run is None else np.asarray(sample_run, np.int64)
        if self.sample_run is not None and len(self.sample_run) != ns:
            raise ValueError("sample_run needs one entry per sample")
        self.frame = frame if frame is not None else QFrame.identity()
        self.coords = {k: np.asarray(v) for k, v in (coords or {}).items()}
        self.meta = dict(meta or {})
        self.raw = {k: np.asarray(v) for k, v in (raw or {}).items()}
        self.energy_map = energy_map
        nb = len(self.counts)
        for name in ("exposure", "e_lo", "e_hi"):
            if len(getattr(self, name)) != nb:
                raise ValueError(f"{name} has length {len(getattr(self, name))}, not {nb}")
        if len(self.bin_ptr) != nb + 1 or self.cell_ptr[-1] != ns:
            raise ValueError("inconsistent quadrature table")
        if len(self.bin_cells) and (self.bin_cells.min() < 0 or self.bin_cells.max() >= nc):
            raise ValueError("bin_cells refers to a cell that does not exist")
        if np.any(self.e_hi <= self.e_lo):
            raise ValueError("every bin needs e_hi > e_lo")
        _check_rows_sum_to_one(self.bin_ptr, self.bin_cell_w, "bin -> cell weights")
        _check_rows_sum_to_one(self.cell_ptr, self.sample_w, "cell -> sample weights")

    # ------------------------------------------------------------------ basics
    @property
    def n_bins(self):
        return len(self.counts)

    @property
    def n_cells(self):
        return len(self.cell_ptr) - 1

    @property
    def n_samples(self):
        return len(self.samples_hkl)

    def __repr__(self):
        return (f"BinnedData({self.n_bins} bins, {self.n_cells} Q cells, "
                f"{self.n_samples} samples, counts {self.counts.sum():.6g}, "
                f"instrument {self.meta.get('instrument', '?')})")

    def sample_cartesian(self, b_matrix, smear_sigma=None):
        """Cartesian Q (1/A) of every sample in the MODEL frame; `smear_sigma` (3,)
        r.l.u. in the data frame shifts each sample by sigma * its smear normal."""
        hkl = self.samples_hkl
        if smear_sigma is not None and np.any(np.asarray(smear_sigma) != 0):
            if self.smear_normals is None:
                raise ValueError("this dataset carries no smear normals")
            hkl = hkl + self.smear_normals * np.asarray(smear_sigma, float)[None, :]
        return self.frame.cartesian(hkl, b_matrix)

    # ------------------------------------------------------------------ selection
    def select(self, mask) -> "BinnedData":
        """A dataset holding only the bins where `mask` is True; cells and samples no
        bin uses are dropped."""
        mask = np.asarray(mask)
        if mask.dtype != bool:
            idx = np.asarray(mask, np.int64)
            mask = np.zeros(self.n_bins, bool)
            mask[idx] = True
        keep = np.flatnonzero(mask)
        lens = np.diff(self.bin_ptr)[keep]
        starts = self.bin_ptr[:-1][keep]
        pos = (np.repeat(starts - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
               + np.arange(lens.sum()))
        cells = self.bin_cells[pos]
        used = np.unique(cells)
        remap = -np.ones(self.n_cells, np.int64)
        remap[used] = np.arange(len(used))
        clen = np.diff(self.cell_ptr)[used]
        cstart = self.cell_ptr[:-1][used]
        spos = (np.repeat(cstart - np.concatenate([[0], np.cumsum(clen)[:-1]]), clen)
                + np.arange(clen.sum()))
        return BinnedData(
            self.counts[keep], self.exposure[keep], self.e_lo[keep], self.e_hi[keep],
            samples_hkl=self.samples_hkl[spos],
            cell_ptr=np.concatenate([[0], np.cumsum(clen)]),
            bin_ptr=np.concatenate([[0], np.cumsum(lens)]),
            bin_cells=remap[cells], bin_cell_w=self.bin_cell_w[pos],
            sample_w=self.sample_w[spos],
            smear_normals=None if self.smear_normals is None else self.smear_normals[spos],
            frame=self.frame, coords={k: v[keep] for k, v in self.coords.items()},
            meta=self.meta, raw={k: v[keep] for k, v in self.raw.items()},
            energy_map=self.energy_map,
            sample_run=None if self.sample_run is None else self.sample_run[spos])

    def select_region(self, **ranges) -> "BinnedData":
        """Keep bins whose coords lie in every given closed range, e.g.
        `select_region(h=(7.85, 8.15), E=(0.4, 7.0))`."""
        mask = np.ones(self.n_bins, bool)
        for name, (lo, hi) in ranges.items():
            if name not in self.coords:
                raise KeyError(f"no coordinate {name!r}; have {sorted(self.coords)}")
            v = self.coords[name]
            mask &= (v >= lo - 1e-9) & (v <= hi + 1e-9)
        return self.select(mask)

    def aggregate(self, labels) -> "BinnedData":
        """Merge bins that share an integer label (ignore label < 0).

        Merged bins must have identical energy edges (they are the same energy bin of
        different Q cells). Counts and exposure add; the merged quadrature is the
        exposure-weighted mixture of the members', so mu_merged = sum of the members'
        mu exactly. Raw arrays add; coords become exposure-weighted means (plain means
        where the total exposure is zero).
        """
        labels = np.asarray(labels, np.int64)
        use = labels >= 0
        uniq, inv = np.unique(labels[use], return_inverse=True)
        members = np.flatnonzero(use)
        order = np.argsort(inv, kind="stable")
        members = members[order]
        group = inv[order]
        ng = len(uniq)
        starts = np.searchsorted(group, np.arange(ng))
        ends = np.searchsorted(group, np.arange(ng), side="right")
        lo = self.e_lo[members]
        hi = self.e_hi[members]
        first = members[starts]
        g_lo = self.e_lo[first]
        g_hi = self.e_hi[first]
        if (np.max(np.abs(lo - np.repeat(g_lo, ends - starts)), initial=0) > 1e-9 or
                np.max(np.abs(hi - np.repeat(g_hi, ends - starts)), initial=0) > 1e-9):
            raise ValueError("aggregate: bins with one label must share energy edges")
        X = self.exposure[members]
        Xg = np.add.reduceat(X, starts) if ng else np.zeros(0)
        cnt = np.add.reduceat(self.counts[members], starts) if ng else np.zeros(0)
        nmem = ends - starts
        frac = np.where(np.repeat(Xg, nmem) > 0, X / np.where(np.repeat(Xg, nmem) > 0,
                                                               np.repeat(Xg, nmem), 1.0),
                        1.0 / np.repeat(nmem, nmem))
        # Expand member rows of the bin->cell table, scaled by each member's fraction.
        lens = np.diff(self.bin_ptr)[members]
        bstart = self.bin_ptr[:-1][members]
        pos = (np.repeat(bstart - np.concatenate([[0], np.cumsum(lens)[:-1]]), lens)
               + np.arange(lens.sum()))
        cells = self.bin_cells[pos]
        w = self.bin_cell_w[pos] * np.repeat(frac, lens)
        gid = np.repeat(np.arange(ng).repeat(nmem), lens)
        # Combine duplicate (group, cell) pairs.
        key = gid * (self.n_cells + 1) + cells
        ukey, kinv = np.unique(key, return_inverse=True)
        wsum = np.bincount(kinv, weights=w, minlength=len(ukey))
        g_of = ukey // (self.n_cells + 1)
        c_of = ukey % (self.n_cells + 1)
        bin_ptr = np.concatenate([[0], np.cumsum(np.bincount(g_of, minlength=ng))])
        coords = {}
        wt = np.where(np.repeat(Xg, nmem) > 0, frac, 1.0 / np.repeat(nmem, nmem))
        for k, v in self.coords.items():
            if not np.issubdtype(v.dtype, np.number) or not ng:
                continue
            vm = v[members]
            const = (np.maximum.reduceat(vm, starts) == np.minimum.reduceat(vm, starts))
            mean = np.add.reduceat(vm * wt, starts)
            # A coordinate shared by every member (an index, the E of an energy bin)
            # is kept exactly; one that varies becomes the exposure-weighted mean.
            coords[k] = (vm[starts] if np.all(const) else
                         np.where(const, vm[starts], mean))
        raw = {k: np.add.reduceat(v[members], starts) for k, v in self.raw.items()
               if k != "weight"} if ng else {}
        if "weight" in self.raw:
            # The event weight is a per-energy property: identical within a group.
            raw["weight"] = self.raw["weight"][first]
        coords["n_members"] = nmem
        return BinnedData(
            cnt, Xg, g_lo, g_hi, samples_hkl=self.samples_hkl, cell_ptr=self.cell_ptr,
            bin_ptr=bin_ptr, bin_cells=c_of, bin_cell_w=wsum, sample_w=self.sample_w,
            smear_normals=self.smear_normals, frame=self.frame, coords=coords,
            meta={**self.meta, "aggregated": True}, raw=raw, energy_map=self.energy_map,
            sample_run=self.sample_run)

    # ------------------------------------------------------------------ io
    def save(self, path):
        """Versioned native .npz (FORMAT_VERSION); `load` reads it back exactly."""
        arrays = dict(
            format_version=np.array(FORMAT_VERSION), counts=self.counts,
            exposure=self.exposure, e_lo=self.e_lo, e_hi=self.e_hi,
            samples_hkl=self.samples_hkl, cell_ptr=self.cell_ptr, bin_ptr=self.bin_ptr,
            bin_cells=self.bin_cells, bin_cell_w=self.bin_cell_w, sample_w=self.sample_w,
            header=np.array(json.dumps({
                "meta": self.meta, "frame": self.frame.to_dict(),
                "energy_map": None if self.energy_map is None else self.energy_map.to_dict(),
                "coords": sorted(self.coords), "raw": sorted(self.raw)})))
        if self.smear_normals is not None:
            arrays["smear_normals"] = self.smear_normals
        if self.sample_run is not None:
            arrays["sample_run"] = self.sample_run
        for k, v in self.coords.items():
            arrays[f"coord__{k}"] = v
        for k, v in self.raw.items():
            arrays[f"raw__{k}"] = v
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "BinnedData":
        with np.load(path, allow_pickle=False) as z:
            ver = int(z["format_version"])
            if ver != FORMAT_VERSION:
                raise ValueError(f"{path}: format version {ver}, this code reads "
                                 f"{FORMAT_VERSION}")
            head = json.loads(str(z["header"]))
            return cls(
                z["counts"], z["exposure"], z["e_lo"], z["e_hi"],
                samples_hkl=z["samples_hkl"], cell_ptr=z["cell_ptr"], bin_ptr=z["bin_ptr"],
                bin_cells=z["bin_cells"], bin_cell_w=z["bin_cell_w"], sample_w=z["sample_w"],
                smear_normals=z["smear_normals"] if "smear_normals" in z.files else None,
                sample_run=z["sample_run"] if "sample_run" in z.files else None,
                frame=QFrame.from_dict(head["frame"]),
                coords={k: z[f"coord__{k}"] for k in head["coords"]},
                raw={k: z[f"raw__{k}"] for k in head["raw"]},
                meta=head["meta"], energy_map=energy_map_from_dict(head["energy_map"]))


# --------------------------------------------------------------------------- sampling
def _unit_samples(n, seed, dims=6):
    """n scrambled Sobol points in [0,1)^dims (power-of-two n keeps the balance)."""
    from scipy.stats import qmc
    sob = qmc.Sobol(dims, scramble=True, seed=seed)
    if n & (n - 1) == 0:
        return sob.random_base2(int(np.log2(n)))
    return sob.random(n)


# --------------------------------------------------------------------------- MDNorm
def _parse_dims(dims):
    """'name|min|max|nbins' strings -> [(name, edges)] in float64."""
    out = []
    for s in dims:
        name, lo, hi, n = str(s).split("|")
        out.append((name, np.linspace(float(lo), float(hi), int(n) + 1)))
    return out


def read_mdnorm_npz(path, zone, *, frame=None, energy_map=None, weight="median",
                    samples_per_cell=16, seed=0, region=None, meta=None):
    """Read one zone of a Mantid MDNorm 4-D export (H, K, L, E).

    The file holds `<zone>_data`, `<zone>_err2`, `<zone>_norm` (shape (nH, nK, nL, nE))
    and `<zone>_dims` ('name|min|max|nbins' per axis), plus optionally `UB`, `runs` and
    `proton_charge_uAh`.

    weight: the per-energy event weight w(E) that turns weighted data into effective
        counts. "median" (default): median(err2/data) over this zone's bins with data
        > 0 at each energy (the global median where an energy has none); a float; or an
        array (nE,) supplied from elsewhere (e.g. a higher-statistics run). The choice
        is recorded in meta['weight_source'].
    region: optional closed ranges on bin CENTRES in the data frame, e.g.
        {'H': (7.8, 8.2), 'E': (0.4, 7)} (E in measured energy), applied before any
        sampling. Model-frame selection is `BinnedData.select_region(h=..., k=...)`.

    Bins with zero normalization are dropped (they have no exposure); data in such a
    bin would be an inconsistency and raises.
    """
    with np.load(path, allow_pickle=False) as z:
        D = z[f"{zone}_data"].astype(float)
        V = z[f"{zone}_err2"].astype(float)
        N = z[f"{zone}_norm"].astype(float)
        dims = _parse_dims(z[f"{zone}_dims"])
        extra = {k: z[k] for k in ("UB", "runs", "proton_charge_uAh") if k in z.files}
    if D.ndim != 4 or D.shape != V.shape or D.shape != N.shape:
        raise ValueError(f"{path}:{zone}: expected three equal 4-D arrays")
    if [len(e) - 1 for _, e in dims] != list(D.shape):
        raise ValueError(f"{path}:{zone}: dims do not match the array shape")
    if np.any((D != 0) & ~(N > 0)):
        raise ValueError(f"{path}:{zone}: data in bins with zero normalization")
    totals = {"data": float(D.sum()), "err2": float(V.sum()), "norm": float(N.sum())}
    edges = [e for _, e in dims]
    centres = [0.5 * (e[1:] + e[:-1]) for e in edges]
    nE = D.shape[3]

    # Per-energy event weight.
    if isinstance(weight, str):
        if weight != "median":
            raise ValueError(f"weight must be 'median', a number or an array, not {weight!r}")
        pos = D > 0
        glob = float(np.median(V[pos] / D[pos])) if pos.any() else 1.0
        w = np.full(nE, glob)
        for ie in range(nE):
            p = pos[..., ie]
            if p.any():
                w[ie] = float(np.median(V[..., ie][p] / D[..., ie][p]))
        wsrc = "median(err2/data) per energy over this zone"
    elif np.ndim(weight) == 0:
        w = np.full(nE, float(weight))
        wsrc = f"constant {float(weight)}"
    else:
        w = np.asarray(weight, float)
        if w.shape != (nE,):
            raise ValueError(f"weight array must have shape ({nE},)")
        wsrc = "supplied array"
    if np.any(w <= 0):
        raise ValueError("event weights must be positive")

    keep_axis = []
    for ax, (name, c) in enumerate(zip("HKLE", centres)):
        r = (region or {}).get(name)
        keep_axis.append(np.ones(len(c), bool) if r is None else
                         (c >= r[0] - 1e-9) & (c <= r[1] + 1e-9))
    sel = np.ix_(*keep_axis)
    D, V, N = D[sel], V[sel], N[sel]
    idx_axes = [np.flatnonzero(k) for k in keep_axis]
    wE = w[idx_axes[3]]

    # Q cells: (H, K, L) boxes with any exposure.
    q_exposed = (N > 0).any(axis=3)
    ih, ik, il = np.nonzero(q_exposed)
    ncell = len(ih)
    cell_of = -np.ones(q_exposed.shape, np.int64)
    cell_of[ih, ik, il] = np.arange(ncell)
    u = _unit_samples(samples_per_cell, seed)
    from scipy.stats import norm as _norm
    lo = np.stack([edges[a][idx_axes[a][i]] for a, i in zip(range(3), (ih, ik, il))], 1)
    hi = np.stack([edges[a][idx_axes[a][i] + 1] for a, i in zip(range(3), (ih, ik, il))], 1)
    samples = (lo[:, None, :] + u[None, :, :3] * (hi - lo)[:, None, :]).reshape(-1, 3)
    normals = np.tile(_norm.ppf(u[:, 3:6]), (ncell, 1))
    cell_ptr = np.arange(ncell + 1) * samples_per_cell

    # Bins: every (cell, E) with norm > 0.
    bh, bk, bl, be = np.nonzero(N > 0)
    nb = len(bh)
    e_edges = edges[3]
    e_lo_raw = e_edges[idx_axes[3][be]]
    e_hi_raw = e_edges[idx_axes[3][be] + 1]
    e_lo = energy_map(e_lo_raw) if energy_map is not None else e_lo_raw
    e_hi = energy_map(e_hi_raw) if energy_map is not None else e_hi_raw
    d = D[bh, bk, bl, be]
    v = V[bh, bk, bl, be]
    nn = N[bh, bk, bl, be]
    wb = wE[be]
    frame = frame or QFrame.identity()
    hkl_c = np.stack([centres[0][idx_axes[0][bh]], centres[1][idx_axes[1][bk]],
                      centres[2][idx_axes[2][bl]]], 1)
    model_c = frame.model_hkl(hkl_c)
    coords = {"H": hkl_c[:, 0], "K": hkl_c[:, 1], "L": hkl_c[:, 2],
              "h": model_c[:, 0], "k": model_c[:, 1], "l": model_c[:, 2],
              "E": 0.5 * (e_lo + e_hi), "E_raw": 0.5 * (e_lo_raw + e_hi_raw),
              "iH": idx_axes[0][bh], "iK": idx_axes[1][bk], "iL": idx_axes[2][bl],
              "iE": idx_axes[3][be]}
    md = {"instrument": "MDNorm histogram", "source": os.path.abspath(str(path)),
          "source_sha256": _sha256(path), "zone": zone,
          "axes": [n for n, _ in dims], "weight_source": wsrc,
          "event_weight": w.tolist(), "file_totals": totals,
          "samples_per_cell": int(samples_per_cell), "seed": int(seed)}
    for k, val in extra.items():
        md[k] = np.asarray(val).tolist()
    md.update(meta or {})
    return BinnedData(
        d / wb, nn / wb, e_lo, e_hi, samples_hkl=samples, cell_ptr=cell_ptr,
        bin_ptr=np.arange(nb + 1), bin_cells=cell_of[bh, bk, bl], bin_cell_w=np.ones(nb),
        smear_normals=normals, frame=frame, coords=coords, meta=md,
        raw={"signal": d, "err2": v, "norm": nn, "weight": wb}, energy_map=energy_map)


# --------------------------------------------------------------------------- cell lists
CELL_COLUMNS = ("H", "K", "L", "E", "counts", "exposure")


class CellList:
    """Native detector/energy cells: (H, K, L, E) per cell (data-frame r.l.u.; E the
    bin centre, the bin [E - dE/2, E + dE/2]), integer counts and exposure (the expected
    count per unit of the model's intensity)."""

    def __init__(self, H, K, L, E, counts, exposure, dE, run=None, meta=None,
                 frame=None, energy_map=None):
        self.H = np.asarray(H, float)
        self.K = np.asarray(K, float)
        self.L = np.asarray(L, float)
        self.E = np.asarray(E, float)
        self.counts = np.asarray(counts, float)
        self.exposure = np.asarray(exposure, float)
        self.dE = float(dE)
        self.run = None if run is None else np.asarray(run)
        self.meta = dict(meta or {})
        self.frame = frame or QFrame.identity()
        self.energy_map = energy_map

    def __len__(self):
        return len(self.counts)

    def aggregate(self, q_edges, *, e_edges=None, windows=None, samples_per_bin=32,
                  seed=0, drop_unexposed=True) -> BinnedData:
        """Sum cells into analysis bins and draw each bin's Q samples.

        q_edges: {'K': edges, ...} bin edges along data-frame axes (the others are
            integrated over, within `windows`).
        e_edges: analysis energy edges (measured energy); default: one bin per native
            energy, [E - dE/2, E + dE/2]. A cell belongs to the bin containing its E.
        windows: {'H': (lo, hi), 'L': (lo, hi), 'E': (lo, hi)} closed selections on the
            cell coordinates, applied first.

        Each bin's samples are `samples_per_bin` of its member cells' own (H, K, L),
        chosen by systematic sampling of the exposure CDF along a Morton order of the
        cells (one seeded random offset per bin): exact in expectation, and with all
        the exposure in one cell every sample is that cell.
        """
        sel = np.ones(len(self), bool)
        for name, (lo, hi) in (windows or {}).items():
            v = getattr(self, name)
            sel &= (v >= lo - 1e-6) & (v <= hi + 1e-6)
        H, K, L, E = self.H[sel], self.K[sel], self.L[sel], self.E[sel]
        cnt, ex = self.counts[sel], self.exposure[sel]
        axes = list(q_edges)
        idx = []
        ok = np.ones(len(H), bool)
        dims = []
        for name in axes:
            ed = np.asarray(q_edges[name], float)
            v = {"H": H, "K": K, "L": L}[name]
            i = np.searchsorted(ed, v, side="right") - 1
            ok &= (i >= 0) & (i < len(ed) - 1)
            idx.append(i)
            dims.append(len(ed) - 1)
        if e_edges is None:
            evals = np.unique(np.round(E, 6))
            ie = np.searchsorted(evals, np.round(E, 6))
            e_lo_raw = evals - self.dE / 2
            e_hi_raw = evals + self.dE / 2
        else:
            ee = np.asarray(e_edges, float)
            ie = np.searchsorted(ee, E, side="right") - 1
            ok &= (ie >= 0) & (ie < len(ee) - 1)
            e_lo_raw, e_hi_raw = ee[:-1], ee[1:]
        nE = len(e_lo_raw)
        flat = np.zeros(len(H), np.int64)
        for i, n in zip(idx, dims):
            flat = flat * n + np.where(ok, i, 0)
        flat = flat * nE + np.where(ok, ie, 0)
        flat[~ok] = -1
        use = flat >= 0
        if not use.any():
            raise ValueError("no cells fall in the requested bins")
        nbins_full = int(np.prod(dims)) * nE
        counts = np.bincount(flat[use], weights=cnt[use], minlength=nbins_full)
        expo = np.bincount(flat[use], weights=ex[use], minlength=nbins_full)
        present = np.zeros(nbins_full, bool)
        present[flat[use]] = True
        keep = present & (expo > 0) if drop_unexposed else present
        if drop_unexposed and np.any(present & (expo <= 0) & (counts > 0)):
            raise ValueError("counts in a bin with zero exposure")
        bins = np.flatnonzero(keep)
        nb = len(bins)
        # Morton order of the member cells inside each bin, then systematic sampling.
        cells_use = np.flatnonzero(use & np.isin(flat, bins))
        b_of = flat[cells_use]
        crd = np.stack([H[cells_use], K[cells_use], L[cells_use]], 1)
        span = np.ptp(crd, axis=0)
        span[span == 0] = 1.0
        c = np.clip(((crd - crd.min(0)) / span * 1023).astype(np.uint64), 0, 1023)
        morton = np.zeros(len(cells_use), np.uint64)
        for bit in range(10):
            for a in range(3):
                morton |= ((c[:, a] >> np.uint64(bit)) & np.uint64(1)) << np.uint64(3 * bit + a)
        order = np.lexsort((morton, b_of))
        b_sorted = b_of[order]
        rng = np.random.default_rng(seed)
        starts = np.searchsorted(b_sorted, bins)
        ends = np.searchsorted(b_sorted, bins, side="right")
        n = int(samples_per_bin)
        samples = np.empty((nb, n, 3))
        sample_run = np.full((nb, n), -1, np.int64)
        runs = self.run[sel] if self.run is not None else None
        offs = rng.random(nb)
        for j in range(nb):
            ids = cells_use[order[starts[j]:ends[j]]]
            w = ex[ids]
            if w.sum() <= 0:
                w = np.ones(len(ids))
            cum = np.cumsum(w)
            pick = np.searchsorted(cum, (np.arange(n) + offs[j]) / n * cum[-1], side="right")
            pick = np.minimum(pick, len(ids) - 1)
            samples[j] = np.stack([H[ids[pick]], K[ids[pick]], L[ids[pick]]], 1)
            if runs is not None:
                sample_run[j] = runs[ids[pick]]
        ie_b = bins % nE
        qidx = bins // nE
        q_coord = {}
        for name, nd in zip(reversed(axes), reversed(dims)):
            q_coord[name] = qidx % nd
            qidx = qidx // nd
        e_lo_r, e_hi_r = e_lo_raw[ie_b], e_hi_raw[ie_b]
        em = self.energy_map
        e_lo = em(e_lo_r) if em is not None else e_lo_r
        e_hi = em(e_hi_r) if em is not None else e_hi_r
        coords = {"E": 0.5 * (e_lo + e_hi), "E_raw": 0.5 * (e_lo_r + e_hi_r)}
        for name in axes:
            ed = np.asarray(q_edges[name], float)
            coords[name] = 0.5 * (ed[q_coord[name]] + ed[q_coord[name] + 1])
            coords["i" + name] = q_coord[name]
        coords["iE"] = ie_b
        mean_hkl = samples.mean(axis=1)
        mc = self.frame.model_hkl(mean_hkl)
        for a, name in enumerate("hkl"):
            coords[name] = mc[:, a]
        md = {**self.meta, "aggregation": {
            "q_edges": {k: np.asarray(v, float).tolist() for k, v in q_edges.items()},
            "e_edges": None if e_edges is None else np.asarray(e_edges, float).tolist(),
            "windows": {k: list(v) for k, v in (windows or {}).items()},
            "samples_per_bin": n, "seed": int(seed),
            "cells_used": int(use.sum()), "counts_used": float(cnt[use].sum())}}
        return BinnedData(
            counts[bins], expo[bins], e_lo, e_hi, samples_hkl=samples.reshape(-1, 3),
            cell_ptr=np.arange(nb + 1) * n, bin_ptr=np.arange(nb + 1),
            bin_cells=np.arange(nb), bin_cell_w=np.ones(nb), frame=self.frame,
            coords=coords, meta=md, energy_map=em,
            sample_run=None if runs is None else sample_run.reshape(-1))


def read_count_cells(paths: Iterable, *, frame=None, energy_map=None, windows=None,
                     meta=None) -> CellList:
    """Read per-run count-cell exports (`K, L, H, E` float32, `counts` uint32,
    `exposure` float64, scalar `dE` and `run`). Runs with no cells are kept in the
    tally (they are real runs with nothing in the exported region).

    windows: optional closed ranges on H, K, L or E applied while reading, so a large
    set need not be held in memory; meta['read_totals'] records the cells and counts of
    EVERY file before the window, for comparison with the export's own manifest.
    """
    parts = {k: [] for k in CELL_COLUMNS}
    runs, per_run, de = [], [], None
    tot_cells = tot_counts = 0
    for p in paths:
        with np.load(p, allow_pickle=False) as z:
            missing = [k for k in CELL_COLUMNS + ("dE",) if k not in z.files]
            if missing:
                raise ValueError(f"{p}: missing {missing}")
            d = float(z["dE"])
            if de is None:
                de = d
            elif abs(d - de) > 1e-12:
                raise ValueError(f"{p}: dE {d} differs from {de}")
            arr = {k: z[k] for k in CELL_COLUMNS}
            run = int(z["run"]) if "run" in z.files else -1
        nc = len(arr["counts"])
        c = int(arr["counts"].sum(dtype=np.int64))
        tot_cells += nc
        tot_counts += c
        per_run.append({"run": run, "cells": nc, "counts": c,
                        "sha256": _sha256(p)})
        sel = np.ones(nc, bool)
        for name, (lo, hi) in (windows or {}).items():
            v = arr[name]
            sel &= (v >= lo - 1e-6) & (v <= hi + 1e-6)
        for k in CELL_COLUMNS:
            parts[k].append(np.asarray(arr[k][sel], float))
        runs.append(np.full(int(sel.sum()), run))
    if de is None:
        raise ValueError("no files given")
    cat = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
    md = {"instrument": "count cells", "dE": de, "read_totals": {
        "files": len(per_run), "cells": tot_cells, "counts": tot_counts,
        "empty_runs": sum(1 for r in per_run if r["cells"] == 0)},
        "runs": per_run, "windows": {k: list(v) for k, v in (windows or {}).items()}}
    md.update(meta or {})
    return CellList(cat["H"], cat["K"], cat["L"], cat["E"], cat["counts"],
                    cat["exposure"], de, run=np.concatenate(runs) if runs else None,
                    meta=md, frame=frame, energy_map=energy_map)
