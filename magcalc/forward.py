"""Forward model: the expected count in every bin of a `BinnedData` for a dipole model.

For bin b with exposure X_b and true energy edges [E_lo, E_hi],

    mu_b = X_b * sum_crystals w_c * sum_cells w_bc * sum_samples w_s *
           sum_modes I_m(R_c Q_s) * { (n(eps)+1) [Phi((E_hi - eps)/sigma) - Phi((E_lo - eps)/sigma)]
                                    + n(eps) [Phi((E_hi + eps)/sigma') - Phi((E_lo + eps)/sigma')] }

with eps = eps_m(R_c Q_s), sigma = sigma(eps) the resolution width at the MODE energy (and
sigma' = sigma(-eps) for the energy-gain line), n the Bose occupation at the data's
temperature (no gain line and no factor at T = 0), I_m the one-magnon intensity from
`MagCalc.sqw_batch` (form factor, sqrt(S_i/2), optional g), and the Q quadrature the
dataset's own (see `magcalc.neutron_data`). The energy integral over the bin is exact
(an erf difference), not a sample at the bin centre, so a mode at a bin edge splits in
half and a sharp mode inside one bin stays there.

The erf difference is the Gaussian case of the resolution protocol
`fraction(lo, hi, centre)`, the probability that a mode at `centre` is recorded in
[lo, hi]. Any resolution with that method can replace it, keeping the exact bin
integral: `magcalc.resolution.ChopperEnergyResolution` gives an instrument's own line
shape (moderator, choppers, flight paths) that way.

Nothing here is fitted: no amplitude, no background. Those belong to the fitting layer.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
from scipy.special import ndtr, xlogy

from .linalg import rotation_matrix
from .numerical import thermal_bose_prefactor

logger = logging.getLogger(__name__)

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class GaussianEnergyResolution:
    """Gaussian energy resolution with FWHM(E) = fwhm0 * sum_k poly[k] x^k.

    x = E / ei (variable='E/Ei', e.g. AMATERAS's 1 - 1.04 x + 0.38 x^2) or x = E in meV
    (variable='E', e.g. a polynomial fitted to an instrument's resolution chart; then
    fwhm0 is usually 1). `poly` is in ASCENDING powers. `extra_sigma` (meV) adds in
    quadrature (an intrinsic or sample width). A width <= 0 anywhere it is used raises.
    """
    fwhm0: float = 1.0
    poly: Sequence[float] = (1.0,)
    variable: str = "E/Ei"
    ei: Optional[float] = None
    extra_sigma: float = 0.0

    def __post_init__(self):
        if self.variable not in ("E/Ei", "E"):
            raise ValueError("variable must be 'E/Ei' or 'E'")
        if self.variable == "E/Ei" and not self.ei:
            raise ValueError("variable='E/Ei' needs ei")

    def fwhm(self, energy):
        e = np.asarray(energy, float)
        x = e / self.ei if self.variable == "E/Ei" else e
        p = np.polynomial.polynomial.polyval(x, np.asarray(self.poly, float))
        return self.fwhm0 * p

    def sigma(self, energy):
        f = self.fwhm(energy)
        if np.any(f < 0):
            raise ValueError("the resolution FWHM is negative at some mode energy "
                             f"(min {np.min(f):.3g} meV); check the law's range")
        return np.sqrt((f / FWHM_PER_SIGMA) ** 2 + self.extra_sigma ** 2)

    def fraction(self, lo, hi, centre):
        """Probability that a mode at `centre` is recorded in [lo, hi]: the resolution
        protocol `ForwardModel` uses (see `magcalc.resolution`)."""
        return _bin_fraction(lo, hi, centre, self.sigma(centre))


@dataclass
class Crystal:
    """One crystal (or twin) of the sample: Q in the lab frame is evaluated on the
    model at R @ Q, with weight `weight` (weights are used as given). A mosaic spread
    rotates further about `mosaic_axis` (Cartesian) by angles ~ N(0, mosaic_sigma_deg),
    integrated at `n_mosaic` Gauss-Hermite nodes."""
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    weight: float = 1.0
    mosaic_sigma_deg: float = 0.0
    mosaic_axis: Optional[Sequence[float]] = None
    n_mosaic: int = 5

    @classmethod
    def from_axis_angle(cls, axis, angle_deg, weight=1.0, mosaic_sigma_deg=0.0,
                        mosaic_axis=None, n_mosaic=5):
        """Rotation by angle_deg about a Cartesian axis (Rodrigues, active); the mosaic
        axis defaults to the same axis."""
        return cls(rotation_matrix(axis, angle_deg), weight, mosaic_sigma_deg,
                   axis if mosaic_axis is None else mosaic_axis, n_mosaic)

    def nodes(self):
        """[(R, w)] whose weights sum to self.weight."""
        R0 = np.asarray(self.rotation, float)
        if not self.mosaic_sigma_deg:
            return [(R0, float(self.weight))]
        if self.mosaic_axis is None:
            raise ValueError("a mosaic spread needs mosaic_axis")
        x, w = np.polynomial.hermite_e.hermegauss(int(self.n_mosaic))
        w = w / np.sqrt(2.0 * np.pi)
        return [(rotation_matrix(self.mosaic_axis, self.mosaic_sigma_deg * xi) @ R0,
                 float(self.weight * wi)) for xi, wi in zip(x, w)]


def crystals_from_domains(domains) -> List[Crystal]:
    """Crystals equivalent to a `calculation.domains` spec (`numerical.parse_domains`).

    A domain is the crystal rotated by R, which `calculate_sqw` evaluates at R^T q; a
    Crystal's rotation acts on Q, so the domain is Crystal(R^T, weight).
    """
    from .numerical import parse_domains
    parsed = parse_domains(domains)
    if parsed is None:
        return [Crystal()]
    return [Crystal(R.T, w) for R, w in parsed]


def _bin_fraction(lo, hi, centre, sigma):
    """Fraction of a Gaussian (centre, sigma) inside [lo, hi], stable in both tails;
    sigma == 0 is the delta-function limit (a centre on an edge counts one half)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        a = (lo - centre) / sigma
        b = (hi - centre) / sigma
        upper = a > 0
        frac = np.where(upper, ndtr(-a) - ndtr(-b), ndtr(b) - ndtr(a))
    if np.any(sigma == 0):
        step = 0.5 * (np.sign(hi - centre) - np.sign(lo - centre))
        frac = np.where(sigma == 0, step, frac)
    return frac


def poisson_deviance(n, mu):
    """2 sum [mu - n + n log(n / mu)] (0 log 0 = 0); infinite if mu = 0 where n > 0."""
    n = np.asarray(n, float)
    mu = np.asarray(mu, float)
    if np.any((mu <= 0) & (n > 0)):
        return np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(2.0 * np.sum(mu - n + xlogy(n, n) - xlogy(n, np.where(mu > 0, mu, 1.0))))


class ForwardModel:
    """Expected counts mu_b for `data` under the dipole model `calc` (a MagCalc).

    resolution: anything with fraction(lo, hi, centre) -- a GaussianEnergyResolution, or
        a `resolution.ChopperEnergyResolution` -- or, failing that, sigma(E) (Gaussian).
    crystals: list of Crystal (default: one, unrotated, weight 1).
    temperature: K for the Bose factor; default data.meta['temperature'], else T = 0.
    q_smear: Gaussian Q smearing sigma (3,) in DATA-frame r.l.u., applied through each
        sample's stored standard-normal deviates (deterministic).
    apply_g / g_tensors: passed to `sqw_batch` (scatter from g.S).
    energy_shift: meV subtracted from the data's energies (an energy-zero offset).
    b_matrix: the model's reciprocal basis (rows); default `runner.compute_b_matrix`.

    The model is evaluated only at the samples of bins with exposure. Results are cached
    on the Hamiltonian parameter vector; call `clear_cache()` after changing settings.
    """

    def __init__(self, calc, data, resolution, *, crystals=None, temperature=None,
                 q_smear=None, apply_g=False, g_tensors=None, energy_shift=0.0,
                 include_gain=True, b_matrix=None, entry_chunk=100_000):
        self.calc = calc
        self.data = data
        self.resolution = resolution
        self.crystals: List[Crystal] = list(crystals) if crystals else [Crystal()]
        if temperature is None:
            temperature = data.meta.get("temperature")
        self.temperature = None if temperature is None else float(temperature)
        self.q_smear = None if q_smear is None else np.asarray(q_smear, float)
        self.apply_g = bool(apply_g)
        self.g_tensors = g_tensors
        self.energy_shift = float(energy_shift)
        self.include_gain = bool(include_gain)
        if b_matrix is None:
            from .runner import compute_b_matrix
            b_matrix = compute_b_matrix(calc.sm)
        self.b_matrix = np.asarray(b_matrix, float)
        self.entry_chunk = int(entry_chunk)
        self._cache = {}
        self.last_timing = {}
        # Expand the two-level table once: entries (bin, sample, weight).
        d = data
        lens = np.diff(d.bin_ptr)
        cells = d.bin_cells
        cw = d.bin_cell_w
        clen = np.diff(d.cell_ptr)[cells]
        bins = np.repeat(np.repeat(np.arange(d.n_bins), lens), clen)
        cstart = np.repeat(d.cell_ptr[:-1][cells], clen)
        within = np.arange(clen.sum()) - np.repeat(np.cumsum(clen) - clen, clen)
        samp = cstart + within
        self._e_bin = bins
        self._e_samp = samp
        self._e_w = np.repeat(cw, clen) * d.sample_w[samp]
        self._active = d.exposure > 0

    # ------------------------------------------------------------------ cache
    def clear_cache(self):
        self._cache.clear()

    def _key(self):
        flat = []
        for p in self.calc.hamiltonian_params:
            flat.extend(np.ravel(p).tolist())
        return tuple(np.round(flat, 15))

    # ------------------------------------------------------------------ model
    def sample_q(self):
        """Lab-frame Cartesian Q of every sample (model reciprocal basis)."""
        return self.data.sample_cartesian(self.b_matrix, self.q_smear)

    def _spectra(self):
        """[(weight, energies (Ns, M), intensities (Ns, M))] per crystal node."""
        import time
        q = self.sample_q()
        out = []
        t = time.perf_counter()
        for cr in self.crystals:
            for R, w in cr.nodes():
                E, I = self.calc.sqw_batch(q @ R.T, apply_g=self.apply_g,
                                           g_tensors=self.g_tensors)
                if not np.all(np.isfinite(E)) or not np.all(np.isfinite(I)):
                    raise FloatingPointError("sqw_batch returned non-finite values at "
                                             "some samples")
                out.append((w, E, I))
        self.last_timing["sqw_seconds"] = time.perf_counter() - t
        self.last_timing["sqw_points"] = len(q) * len(out)
        return out

    def _fraction_function(self):
        """fraction(lo, hi, centre) of the resolution: its own method (the protocol), or,
        for an object that has only sigma(E), the Gaussian bin integral with that width."""
        res = self.resolution
        if hasattr(res, "fraction"):
            return res.fraction
        return lambda lo, hi, c: _bin_fraction(lo, hi, c, res.sigma(c))

    def _line_weights(self, E):
        """(stokes, gain) Bose weights per mode; gain is None at T = 0."""
        T = self.temperature
        if not T or T <= 0:
            return np.ones_like(E), None
        stokes = thermal_bose_prefactor(E, T)          # n + 1 for E > 0
        gain = thermal_bose_prefactor(-E, T) if self.include_gain else None   # n
        return stokes, gain

    def signal_per_exposure(self, params=None):
        """The model in the data's intensity units per bin (mu_b / X_b)."""
        import time
        if params is not None:
            self.calc.update_hamiltonian_params(params)
        key = self._key()
        if key in self._cache:
            return self._cache[key]
        lo = self.data.e_lo - self.energy_shift
        hi = self.data.e_hi - self.energy_shift
        nb = self.data.n_bins
        sig = np.zeros(nb)
        spectra = self._spectra()
        t = time.perf_counter()
        act = self._active[self._e_bin]
        eb, es, ew = self._e_bin[act], self._e_samp[act], self._e_w[act]
        fraction = self._fraction_function()
        for w_node, E, I in spectra:
            stokes, gain = self._line_weights(E)
            amp = I * stokes
            if gain is not None:
                amp_g = I * gain
            for c0 in range(0, len(eb), self.entry_chunk):
                b = eb[c0:c0 + self.entry_chunk]
                s = es[c0:c0 + self.entry_chunk]
                l, h = lo[b][:, None], hi[b][:, None]
                val = np.sum(amp[s] * fraction(l, h, E[s]), axis=1)
                if gain is not None:
                    val += np.sum(amp_g[s] * fraction(l, h, -E[s]), axis=1)
                sig += w_node * np.bincount(b, weights=ew[c0:c0 + self.entry_chunk] * val,
                                            minlength=nb)
        self.last_timing["bin_seconds"] = time.perf_counter() - t
        self._cache = {key: sig}
        return sig

    def expected(self, params=None):
        """mu_b = X_b * signal_per_exposure. `params` (optional) replaces the
        Hamiltonian parameters on the CURRENT magnetic structure first: it does not
        re-minimize the ground state, so use it only for changes that keep it."""
        return self.data.exposure * self.signal_per_exposure(params)

    def deviance(self, params=None):
        return poisson_deviance(self.data.counts, self.expected(params))
