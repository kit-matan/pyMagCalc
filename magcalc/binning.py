"""Histogram binning and NeXus import (Gap 4 #20) -- Sunny's `BinningParameters`
and `load_nxs`.

Two jobs, deliberately separable:

  * `BinningParameters` + `bin_events` / `bin_curve` put a calculation onto the SAME
    (|Q|, E) or (q-index, E) grid an experiment was reduced onto, so model and data
    can be subtracted rather than eyeballed;
  * `load_nxs` reads a reduced NeXus/NXSPE-style histogram back out.

`h5py` is required only for `load_nxs`; the binning half has no extra dependency and
a missing h5py raises an actionable message rather than an ImportError traceback.

WHAT IS AND IS NOT VALIDATED. The binning is pinned by exact identities that need no
reference at all -- weight is conserved, a delta lands in the bin containing it, and
coarsening equals summing the fine bins it contains. `load_nxs` is pinned by a
ROUND TRIP through a file this module writes, which proves the reader matches the
writer's layout; it is NOT a guarantee that every instrument's NeXus dialect is
understood, and there are many. Point it at a real file from your reduction pipeline
before trusting it, and see `nxs_report` for what it found.
"""
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BinningParameters:
    """Uniform bin edges along an abscissa and along energy.

    `x_edges` is whatever the abscissa is -- |Q| for a powder map, a path index for a
    q-path -- and `e_edges` the energy axis. Edges, not centres: a histogram is
    defined by its boundaries, and storing centres forces a guess about widths at the
    ends.
    """
    x_edges: np.ndarray
    e_edges: np.ndarray

    @classmethod
    def uniform(cls, x_range: Tuple[float, float], n_x: int,
                e_range: Tuple[float, float], n_e: int) -> "BinningParameters":
        return cls(np.linspace(x_range[0], x_range[1], int(n_x) + 1),
                   np.linspace(e_range[0], e_range[1], int(n_e) + 1))

    @property
    def x_centers(self) -> np.ndarray:
        return 0.5 * (self.x_edges[:-1] + self.x_edges[1:])

    @property
    def e_centers(self) -> np.ndarray:
        return 0.5 * (self.e_edges[:-1] + self.e_edges[1:])

    @property
    def shape(self) -> Tuple[int, int]:
        return (len(self.x_edges) - 1, len(self.e_edges) - 1)


def bin_events(x, energy, weight, params: BinningParameters,
               density: bool = False) -> np.ndarray:
    """Histogram scattered (x, E, weight) triples onto `params`.

    This is the natural way to bin a MODE LIST: LSWT gives discrete (omega, I) per q,
    which is a set of delta functions, and putting them in bins is exactly what an
    experiment's detector does. Weight is CONSERVED (every event lands in exactly one
    bin, or is dropped if outside the grid -- and the count of dropped events is
    logged, because silently discarding half the intensity is the kind of thing that
    looks like a scale factor later).

    `density=True` divides by the bin area, giving an intensity per unit (x, E)
    rather than a total -- the right choice when comparing grids of different
    resolution.
    """
    x = np.asarray(x, float).ravel()
    e = np.asarray(energy, float).ravel()
    w = np.asarray(weight, float).ravel()
    if not (x.shape == e.shape == w.shape):
        raise ValueError(
            f"x, energy and weight must have the same length; got {x.shape}, "
            f"{e.shape}, {w.shape}.")
    inside = ((x >= params.x_edges[0]) & (x <= params.x_edges[-1])
              & (e >= params.e_edges[0]) & (e <= params.e_edges[-1]))
    n_out = int((~inside).sum())
    if n_out:
        logger.info(f"bin_events: {n_out} of {len(x)} events fell outside the grid "
                    f"and were dropped ({w[~inside].sum():.4g} of "
                    f"{w.sum():.4g} total weight).")
    hist, _, _ = np.histogram2d(x[inside], e[inside],
                                bins=[params.x_edges, params.e_edges],
                                weights=w[inside])
    if density:
        area = np.outer(np.diff(params.x_edges), np.diff(params.e_edges))
        hist = hist / area
    return hist


def bin_mode_list(x_values, energies, intensities,
                  params: BinningParameters, density: bool = False) -> np.ndarray:
    """Bin an LSWT result: `energies`/`intensities` are (n_q, n_modes), `x_values`
    the abscissa of each q (|Q| or path index)."""
    e = np.asarray(energies, float)
    i = np.asarray(intensities, float)
    if e.shape != i.shape:
        raise ValueError(f"energies {e.shape} and intensities {i.shape} must match.")
    x = np.repeat(np.asarray(x_values, float).ravel()[:, None], e.shape[1], axis=1)
    return bin_events(x, e, i, params, density=density)


def rebin(hist: np.ndarray, factor_x: int = 1, factor_e: int = 1) -> np.ndarray:
    """Coarsen a histogram by integer factors, summing. Exact: the result equals
    binning the same events on the coarser grid."""
    fx, fe = int(factor_x), int(factor_e)
    nx, ne = hist.shape
    if nx % fx or ne % fe:
        raise ValueError(
            f"histogram shape {hist.shape} is not divisible by ({fx}, {fe}).")
    return hist.reshape(nx // fx, fx, ne // fe, fe).sum(axis=(1, 3))


# --------------------------------------------------------------------------
def _require_h5py():
    try:
        import h5py
    except ImportError as exc:              # pragma: no cover - env-dependent
        raise ImportError(
            "reading NeXus files needs the optional `h5py` package "
            "(pip install h5py). The binning half of magcalc.binning does not.") from exc
    return h5py


def save_nxs(path, hist: np.ndarray, params: BinningParameters,
             title: str = "magcalc") -> None:
    """Write a minimal NXdata histogram -- used by the round-trip test, and handy for
    handing a calculated map to a plotting tool that speaks NeXus."""
    h5py = _require_h5py()
    with h5py.File(path, "w") as f:
        entry = f.create_group("entry")
        entry.attrs["NX_class"] = np.bytes_("NXentry")
        entry["title"] = np.bytes_(title)
        data = entry.create_group("data")
        data.attrs["NX_class"] = np.bytes_("NXdata")
        data.attrs["signal"] = np.bytes_("signal")
        data.attrs["axes"] = [np.bytes_("x"), np.bytes_("energy")]
        data.create_dataset("signal", data=np.asarray(hist, float))
        data.create_dataset("x", data=np.asarray(params.x_centers, float))
        data.create_dataset("energy", data=np.asarray(params.e_centers, float))
        data.create_dataset("x_edges", data=np.asarray(params.x_edges, float))
        data.create_dataset("energy_edges", data=np.asarray(params.e_edges, float))


def load_nxs(path) -> Dict[str, np.ndarray]:
    """Read a reduced NeXus histogram into {signal, x, energy, params, title}.

    Looks for an NXdata group and takes its `signal` plus two axes. Edge arrays are
    used when present and otherwise reconstructed from the centres by assuming a
    uniform grid -- reported in the result as `edges_reconstructed` so the caller
    knows which happened, because a wrong assumption there shifts every bin by half a
    width and that is very hard to see on a colour map.
    """
    h5py = _require_h5py()
    with h5py.File(path, "r") as f:
        grp = _find_nxdata(f)
        if grp is None:
            raise ValueError(
                f"{path}: no NXdata group with a `signal` dataset was found. "
                f"magcalc.binning reads reduced HISTOGRAMS, not raw event files; "
                f"`nxs_report` lists what is in the file.")
        signal = np.asarray(grp["signal"][()], float)
        axes = [a.decode() if isinstance(a, bytes) else str(a)
                for a in np.atleast_1d(grp.attrs.get("axes", [b"x", b"energy"]))]
        x = np.asarray(grp[axes[0]][()], float).ravel()
        e = np.asarray(grp[axes[1]][()], float).ravel()
        recon = False
        if f"{axes[0]}_edges" in grp and f"{axes[1]}_edges" in grp:
            xe = np.asarray(grp[f"{axes[0]}_edges"][()], float).ravel()
            ee = np.asarray(grp[f"{axes[1]}_edges"][()], float).ravel()
        else:
            recon = True
            xe, ee = _edges_from_centers(x), _edges_from_centers(e)
        title = _read_title(grp)
    return {"signal": signal, "x": x, "energy": e,
            "params": BinningParameters(xe, ee),
            "edges_reconstructed": recon, "title": title}


def _read_title(grp):
    node = grp.parent.get("title") if grp.parent is not None else None
    if node is None:
        return ""
    raw = node[()]
    if isinstance(raw, bytes):
        return raw.decode()
    return str(raw)


def _find_nxdata(f):
    found = []

    def visit(name, obj):
        # a Group whose members include `signal`; Datasets have no membership test
        if hasattr(obj, "keys") and "signal" in obj.keys():
            found.append(obj)
    f.visititems(visit)
    return found[0] if found else None


def _edges_from_centers(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c, float).ravel()
    if c.size == 1:
        return np.array([c[0] - 0.5, c[0] + 0.5])
    d = np.diff(c)
    if not np.allclose(d, d[0], rtol=1e-6):
        raise ValueError(
            "axis centres are not uniformly spaced, so bin edges cannot be "
            "reconstructed; the file should carry explicit edges.")
    return np.concatenate([[c[0] - d[0] / 2], c + d[0] / 2])


def nxs_report(path) -> str:
    """Human-readable listing of a NeXus file's datasets -- for when `load_nxs`
    cannot find what it needs and you have to see what the dialect actually is."""
    h5py = _require_h5py()
    lines = []
    with h5py.File(path, "r") as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                lines.append(f"  {name}  shape={obj.shape} dtype={obj.dtype}")
        f.visititems(visit)
    return f"{path}\n" + ("\n".join(lines) if lines else "  (no datasets)")
