"""Energy resolution of a direct-geometry chopper spectrometer, from its moderator,
choppers and flight paths (neutron-data phase 2, energy only).

A neutron is labelled by its emission time t0 and inverse speed u = 1/v_i. It passes a
chopper at distance x (from the moderator) at t0 + x u, reaches the sample at
t0 + L1 u and the detector at t_S + L2 / v_f. The reduction (Mantid, Utsusemi) assumes
the nominal Ei and the nominal time at the sample, so a neutron off the nominal
(tau = dt0, mu = du) is reported at an energy transfer off by, to first order,

    dE = alpha tau + beta mu,   alpha = 2 Ef v_f / L2,   beta = 2 Ei v_i + alpha L1,

with Ef = Ei - E at the MODE energy E (the phase-1 convention). At E = 0 this is
(2 Ei v_i / L2) times the arrival-time error at the detector, tau + (L1 + L2) mu.

The transmitted (tau, mu) have the density

    rho(tau, mu)  ~  M(tau) * prod_c W_c(tau + x_c mu),

M the moderator pulse about the time the choppers are phased to (flat if None), W_c
each chopper's opening (a disk: a trapezoid, slot (x) beam width). Nothing in it is
assumed Gaussian: the line shape -- trapezoid edges, a moderator tail -- is the exact
distribution of alpha tau + beta mu, plus independent extra terms (detector depth, a
Gaussian width). Its mean is removed: the reduction's Ei and time zero are calibrated
on the transmitted beam (`neutron_data.T0EnergyMap` re-centres what is left).

The first-order relation sets sigma(E) and the scale of the table. The table itself maps
every quadrature point through the exact time of flight the reduction uses (`kinematics=
'exact'`, the default), which matters once an unshaped pulse tail spreads the incident
speed by a few % (tests/test_resolution_raytrace.py: about 4 % in width otherwise).

`ChopperEnergyResolution.fraction(lo, hi, centre)` is the probability that a mode at
`centre` is recorded in [lo, hi]: the protocol `forward.ForwardModel` uses, shared with
`forward.GaussianEnergyResolution`. When every part is Gaussian it is that same erf
difference (`forward._bin_fraction`) with the exact sigma(E); otherwise it is read from
a CDF table built once per Ei, on a grid of mode energies, in units of sigma(E).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.special import ndtr

from .neutron_data import NEUTRON_V_PER_SQRT_MEV

logger = logging.getLogger(__name__)

FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))
GAUSS_TAIL = 8.0      # Gaussian components are cut at +-8 sigma on a grid


# --------------------------------------------------------------------------- distributions
class Distribution:
    """A 1-D probability density (times in s, or lengths in m)."""
    is_gaussian = False

    def pdf(self, t):
        raise NotImplementedError

    def support(self):
        """(lo, hi) outside which the density is zero (or negligible)."""
        raise NotImplementedError

    def mean(self):
        raise NotImplementedError

    def var(self):
        raise NotImplementedError


@dataclass
class Gaussian(Distribution):
    sigma: float
    mu: float = 0.0
    is_gaussian = True

    def pdf(self, t):
        z = (np.asarray(t, float) - self.mu) / self.sigma
        return np.exp(-0.5 * z * z) / (self.sigma * np.sqrt(2.0 * np.pi))

    def support(self):
        return self.mu - GAUSS_TAIL * self.sigma, self.mu + GAUSS_TAIL * self.sigma

    def mean(self):
        return self.mu

    def var(self):
        return self.sigma ** 2


@dataclass
class Trapezoid(Distribution):
    """Uniform(a) convolved with Uniform(b) (full widths), centred on `centre`: a disk
    chopper's opening (slot time (x) beam-crossing time). b = 0 is a box, a = b a
    triangle of FWHM a."""
    a: float
    b: float = 0.0
    centre: float = 0.0

    def __post_init__(self):
        if self.a < 0 or self.b < 0 or self.a + self.b <= 0:
            raise ValueError("Trapezoid needs widths a, b >= 0, not both zero")

    def pdf(self, t):
        a, b = max(self.a, self.b), min(self.a, self.b)
        x = np.abs(np.asarray(t, float) - self.centre)
        if b == 0:
            return np.where(x < a / 2, 1.0 / a, np.where(x == a / 2, 0.5 / a, 0.0))
        return np.clip(((a + b) / 2 - x) / (a * b), 0.0, 1.0 / a)

    def support(self):
        h = (self.a + self.b) / 2
        return self.centre - h, self.centre + h

    def mean(self):
        return self.centre

    def var(self):
        return (self.a ** 2 + self.b ** 2) / 12.0


def Uniform(width, centre=0.0):
    return Trapezoid(width, 0.0, centre)


class CounterRotatingOpening(Distribution):
    """Opening in time of a counter-rotating pair of disks: slots of linear widths w1, w2
    moving in opposite directions at speed v each, across a beam of width b. A neutron at
    transverse position y passes at time t if |y - v t| < w1/2, |y + v t| < w2/2 and
    |y| < b/2, so for a beam uniform in y the density is the length of that y-interval:

        L(t) = max(0, min(b/2, w1/2 + v t, w2/2 - v t) - max(-b/2, v t - w1/2, -v t - w2/2)).

    b = w1 = w2 gives a triangle of FWHM w/(2v) (the pair halves a single disk's burst);
    b -> 0 gives a box of width min(w1, w2)/v (it does not). b = 0 is that box."""

    def __init__(self, w1, w2, b, v):
        if min(w1, w2) <= 0 or b < 0 or v <= 0:
            raise ValueError("slot widths and speed must be positive")
        self.w1, self.w2, self.b, self.v = float(w1), float(w2), float(b), float(v)
        if self.b == 0:
            self._box = Uniform(min(w1, w2) / v)
        else:
            self._box = None
            hi = (self.w1 + self.w2) / (4 * self.v)
            t = np.linspace(-hi, hi, 40001)
            self._norm = float(np.trapezoid(self._length(t), t))
            self._t = t

    def _length(self, t):
        vt = self.v * np.asarray(t, float)
        top = np.minimum(np.minimum(self.b / 2, self.w1 / 2 + vt), self.w2 / 2 - vt)
        bot = np.maximum(np.maximum(-self.b / 2, vt - self.w1 / 2), -vt - self.w2 / 2)
        return np.maximum(top - bot, 0.0)

    def pdf(self, t):
        if self._box is not None:
            return self._box.pdf(t)
        return self._length(t) / self._norm

    def support(self):
        if self._box is not None:
            return self._box.support()
        h = (self.w1 + self.w2) / (4 * self.v)
        return -h, h

    def mean(self):
        if self._box is not None:
            return 0.0
        t = self._t
        return float(np.trapezoid(t * self.pdf(t), t))

    def var(self):
        if self._box is not None:
            return self._box.var()
        t = self._t
        m = self.mean()
        return float(np.trapezoid((t - m) ** 2 * self.pdf(t), t))


@dataclass
class IkedaCarpenter(Distribution):
    """Ikeda & Carpenter (1985) moderator pulse, t >= 0 (s); alpha, beta in 1/s:

    f(t) = (alpha/2) { (1-R) (alpha t)^2 e^{-alpha t}
           + R 2 alpha^2 beta / (alpha-beta)^3 [e^{-beta t}
             - e^{-alpha t} (1 + (alpha-beta) t + (alpha-beta)^2 t^2 / 2)] }

    mean 3/alpha + R/beta, variance 3/alpha^2 + R(2-R)/beta^2."""
    alpha: float
    beta: float
    R: float
    shift: float = 0.0

    def __post_init__(self):
        if not (self.alpha > 0 and self.beta > 0 and 0 <= self.R <= 1):
            raise ValueError("IkedaCarpenter needs alpha, beta > 0 and 0 <= R <= 1")
        if abs(self.alpha - self.beta) < 1e-9 * self.alpha:
            raise ValueError("IkedaCarpenter needs alpha != beta")

    def pdf(self, t):
        t = np.asarray(t, float) - self.shift
        a, b, R = self.alpha, self.beta, self.R
        tt = np.maximum(t, 0.0)
        d = a - b
        fast = (1 - R) * (a * tt) ** 2 * np.exp(-a * tt)
        slow = R * 2 * a * a * b / d ** 3 * (
            np.exp(-b * tt) - np.exp(-a * tt) * (1 + d * tt + 0.5 * (d * tt) ** 2))
        return np.where(t >= 0, 0.5 * a * (fast + slow), 0.0)

    def support(self):
        return self.shift, self.shift + max(40.0 / self.alpha, 30.0 / self.beta)

    def mean(self):
        return self.shift + 3.0 / self.alpha + self.R / self.beta

    def var(self):
        return 3.0 / self.alpha ** 2 + self.R * (2 - self.R) / self.beta ** 2

    def peak(self):
        lo, hi = self.support()
        t = np.linspace(lo, lo + 10.0 / min(self.alpha, self.beta), 20001)
        return float(t[np.argmax(self.pdf(t))])


@dataclass
class Tabulated(Distribution):
    """A density given on a grid (linear interpolation, zero outside)."""
    t: np.ndarray
    density: np.ndarray

    def __post_init__(self):
        self.t = np.asarray(self.t, float)
        self.density = np.asarray(self.density, float)
        if np.any(np.diff(self.t) <= 0) or np.any(self.density < 0):
            raise ValueError("Tabulated needs increasing t and a nonnegative density")
        self.density = self.density / np.trapezoid(self.density, self.t)

    def pdf(self, t):
        return np.interp(t, self.t, self.density, left=0.0, right=0.0)

    def support(self):
        return float(self.t[0]), float(self.t[-1])

    def _moment(self, k):
        return float(np.trapezoid(self.t ** k * self.density, self.t))

    def mean(self):
        return self._moment(1)

    def var(self):
        return self._moment(2) - self._moment(1) ** 2


# --------------------------------------------------------------------------- hardware
@dataclass
class DiskChopper:
    """A disk chopper `distance` m from the moderator, at `frequency` Hz, with a slot of
    linear width `slot_width` (m) at the beam radius `radius` (m) and a beam of width
    `beam_width` (m, along the slot's motion) there. Its opening in time is
    Trapezoid(slot/v, beam/v), v = 2 pi radius frequency. A counter-rotating pair
    (`counter_rotating`; the second disk's slot `slot_width2`, default the same) opens
    where the two slots overlap: `CounterRotatingOpening`, which halves the burst only
    when the beam fills the slot."""
    name: str
    distance: float
    frequency: float
    slot_width: float
    radius: float
    beam_width: float = 0.0
    counter_rotating: bool = False
    slot_width2: Optional[float] = None

    @classmethod
    def from_slot_angle(cls, name, distance, frequency, slot_angle_deg, radius,
                        beam_width=0.0, counter_rotating=False, slot_angle2_deg=None):
        w2 = None if slot_angle2_deg is None else radius * np.radians(slot_angle2_deg)
        return cls(name, distance, frequency, radius * np.radians(slot_angle_deg), radius,
                   beam_width, counter_rotating, w2)

    @property
    def speed(self):
        """Speed of one disk at the beam (m/s)."""
        return 2 * np.pi * self.radius * self.frequency

    def window(self) -> Distribution:
        if self.frequency <= 0:
            raise ValueError(f"chopper {self.name} is not running")
        if self.counter_rotating:
            w2 = self.slot_width if self.slot_width2 is None else self.slot_width2
            return CounterRotatingOpening(self.slot_width, w2, self.beam_width, self.speed)
        return Trapezoid(self.slot_width / self.speed, self.beam_width / self.speed)


@dataclass
class TimeWindow:
    """A chopper given directly by its opening-time density (e.g. a Gaussian, or a
    model taken from another program), `distance` m from the moderator."""
    name: str
    distance: float
    density: Distribution

    def window(self) -> Distribution:
        return self.density


Moderator = Union[None, Distribution, Callable[[float], Distribution]]


# --------------------------------------------------------------------------- spectrometer
@dataclass
class ChopperSpectrometer:
    """A direct-geometry spectrometer for the energy resolution: flight paths `l1`
    (moderator to sample) and `l2` (sample to detector), m; the choppers that define the
    resolution (one or two; a pulse-shaping chopper and a monochromating one); the
    moderator pulse about the time the choppers are phased to (a Distribution, a callable
    ei -> Distribution, or None for a pulse that is flat over the openings); a detector
    depth (m, the full width of a uniform spread in L2); an extra Gaussian width (meV)."""
    name: str
    l1: float
    l2: float
    choppers: Sequence[Union[DiskChopper, TimeWindow]]
    moderator: Moderator = None
    detector_depth: float = 0.0
    extra_sigma: float = 0.0

    def __post_init__(self):
        self.choppers = sorted(self.choppers, key=lambda c: c.distance)
        if not 1 <= len(self.choppers) <= 2:
            raise ValueError("give the one or two choppers that set the resolution")
        if self.moderator is None and len(self.choppers) < 2:
            raise ValueError("one chopper and a flat moderator pulse leave the incident "
                             "speed unbounded; give a moderator pulse or a second chopper")
        for c in self.choppers:
            if not 0 < c.distance < self.l1:
                raise ValueError(f"chopper {c.name} is not between moderator and sample")

    def moderator_pulse(self, ei) -> Optional[Distribution]:
        m = self.moderator
        return m(ei) if callable(m) and not isinstance(m, Distribution) else m

    def coefficients(self, ei, e):
        """(alpha, beta) of dE = alpha tau + beta mu at mode energy e (meV/s, meV m/s)."""
        e = np.asarray(e, float)
        ef = ei - e
        if np.any(ef <= 0):
            raise ValueError("mode energy at or above Ei")
        vi = NEUTRON_V_PER_SQRT_MEV * np.sqrt(ei)
        vf = NEUTRON_V_PER_SQRT_MEV * np.sqrt(ef)
        alpha = 2 * ef * vf / self.l2
        beta = 2 * ei * vi + alpha * self.l1
        return alpha, beta

    def depth_coefficient(self, ei, e):
        """dE per metre of extra L2 (meV/m)."""
        return 2 * (ei - np.asarray(e, float)) / self.l2

    def is_gaussian(self, ei):
        m = self.moderator_pulse(ei)
        return (all(c.window().is_gaussian for c in self.choppers)
                and (m is None or m.is_gaussian) and self.detector_depth == 0)

    # ---------------------------------------------------------------- (tau, mu)
    def time_covariance_gaussian(self, ei):
        """Exact 2x2 covariance of (tau, mu) when every part is Gaussian: the precision
        is sum_c a_c a_c^T / var_c with a_c = (1, x_c), plus (1, 0) / var_moderator."""
        lam = np.zeros((2, 2))
        for c in self.choppers:
            w = c.window()
            a = np.array([1.0, c.distance])
            lam += np.outer(a, a) / w.var()
        m = self.moderator_pulse(ei)
        if m is not None:
            lam[0, 0] += 1.0 / m.var()
        return np.linalg.inv(lam)

    def time_points(self, ei, n=400, centre=True):
        """Midpoint quadrature of the transmitted (tau, mu) density: points (N, 2) and
        weights summing to 1, mean removed unless `centre` is False (then tau is the
        emission time about the phasing time and mu the inverse speed about the
        phased one)."""
        m = self.moderator_pulse(ei)
        n = int(n)
        if len(self.choppers) == 2:
            p, q = self.choppers
            wp, wq = p.window(), q.window()
            sp = _midpoints(*wp.support(), n)
            sq = _midpoints(*wq.support(), n)
            SP, SQ = np.meshgrid(sp, sq, indexing="ij")
            dx = q.distance - p.distance
            tau = (q.distance * SP - p.distance * SQ) / dx
            mu = (SQ - SP) / dx
            w = wp.pdf(SP) * wq.pdf(SQ)
        else:
            (c,) = self.choppers
            wc = c.window()
            t0 = _midpoints(*m.support(), n)
            sc = _midpoints(*wc.support(), n)
            T0, SC = np.meshgrid(t0, sc, indexing="ij")
            tau = T0
            mu = (SC - T0) / c.distance
            w = wc.pdf(SC)
        if m is not None:
            w = w * m.pdf(tau)
        w = w.ravel()
        tot = w.sum()
        if not tot > 0:
            raise ValueError("no neutron passes: the choppers and moderator do not overlap")
        w = w / tot
        pts = np.column_stack([tau.ravel(), mu.ravel()])
        keep = w > 0
        pts, w = pts[keep], w[keep]
        if centre:
            pts = pts - w @ pts
        return pts, w

    def time_covariance(self, ei, n=400):
        if self.is_gaussian(ei):
            return self.time_covariance_gaussian(ei)
        pts, w = self.time_points(ei, n)
        return (pts * w[:, None]).T @ pts

    # ---------------------------------------------------------------- energy
    def sigma(self, ei, e, n=400):
        """Standard deviation (meV) of the recorded energy about a mode at e."""
        cov = self.time_covariance(ei, n)
        return self._sigma_from_cov(cov, ei, e)

    def _sigma_from_cov(self, cov, ei, e):
        al, be = self.coefficients(ei, e)
        v = al * al * cov[0, 0] + 2 * al * be * cov[0, 1] + be * be * cov[1, 1]
        v = v + (self.depth_coefficient(ei, e) * self.detector_depth) ** 2 / 12.0
        return np.sqrt(v + self.extra_sigma ** 2)

    def resolution(self, ei, **kw) -> "ChopperEnergyResolution":
        return ChopperEnergyResolution(self, ei, **kw)


def _midpoints(lo, hi, n):
    h = (hi - lo) / n
    return lo + h * (np.arange(n) + 0.5)


# --------------------------------------------------------------------------- the profile
class ChopperEnergyResolution:
    """The energy resolution of `spec` at incident energy `ei`, as the forward model's
    `fraction(lo, hi, centre)`.

    Every part Gaussian: the exact erf difference with sigma(E) from the closed-form
    covariance. Otherwise: the CDF of the recorded-minus-mode energy is tabulated on
    `n_nodes` mode energies spanning `e_range` (default -2 Ei .. 0.95 Ei), each in units
    of its own sigma, and read by linear interpolation in z = x / sigma(E) and between
    the two nearest nodes. A mode outside `e_range` takes the nearest node's profile
    (its weight in bins below Ei is a tail). `n_time` is the quadrature per time axis,
    `n_x` the bins across each node's profile.

    kinematics: 'exact' (default) maps each quadrature point (tau, mu) to its recorded
    energy with the time-of-flight relation the reduction uses, E_rec = Ei - (m/2)(L2 /
    (t_D - t_S0))^2, so a long pulse tail (a few % of the flight time) keeps its true
    shape; 'linear' uses dE = alpha tau + beta mu. sigma(E) is always the linear
    covariance's, the scale the table is stored in.
    """

    def __init__(self, spec: ChopperSpectrometer, ei: float, *, e_range=None,
                 n_nodes=64, n_time=400, n_x=2000, force_table=False, kinematics="exact"):
        if kinematics not in ("exact", "linear"):
            raise ValueError("kinematics must be 'exact' or 'linear'")
        self.kinematics = kinematics
        self.spec = spec
        self.ei = float(ei)
        lo, hi = e_range if e_range is not None else (-2.0 * ei, 0.95 * ei)
        if hi >= ei:
            raise ValueError("e_range must stay below Ei")
        self.e_nodes = np.linspace(lo, hi, int(n_nodes))
        self.gaussian = spec.is_gaussian(ei) and not force_table
        if self.gaussian:
            self.cov = spec.time_covariance_gaussian(ei)
        else:
            # any shape (force_table: a Gaussian one too, to test the table itself)
            # uncentred: the exact mapping needs the speeds the choppers are phased to
            self._raw, self._w = spec.time_points(ei, n_time, centre=False)
            self._pts = self._raw - self._w @ self._raw
            self.cov = (self._pts * self._w[:, None]).T @ self._pts
            self._build_table(int(n_x))

    # ---------------------------------------------------------------- widths
    def _clip(self, e):
        return np.clip(np.asarray(e, float), self.e_nodes[0], self.e_nodes[-1])

    def sigma(self, e):
        """sigma(E) in meV at mode energy e (clipped to e_range)."""
        return self.spec._sigma_from_cov(self.cov, self.ei, self._clip(e))

    def fwhm_gaussian(self, e):
        return FWHM_PER_SIGMA * self.sigma(e)

    # ---------------------------------------------------------------- table
    def _profile_at(self, e, n_x):
        """(x bin edges' upper ends, CDF) of the recorded-minus-mode energy at e, meV."""
        sp = self.spec
        if self.kinematics == "exact":
            s, w = self._exact_shift(e)
        else:
            al, be = sp.coefficients(self.ei, e)
            s, w = self._pts @ np.array([al, be]), self._w
        sig = float(sp._sigma_from_cov(self.cov, self.ei, e))
        dep = abs(float(sp.depth_coefficient(self.ei, e))) * sp.detector_depth
        ext = sp.extra_sigma
        half = np.max(np.abs(s)) + dep / 2 + GAUSS_TAIL * ext
        dx = 2 * half / n_x
        nh = int(np.ceil(half / dx)) + 2
        x = dx * np.arange(-nh, nh + 1)
        mass = _deposit(s, w, dx, nh)
        for width, kind in ((dep, "box"), (ext, "gauss")):
            if width > 0:
                mass = np.convolve(mass, _kernel(width, kind, dx), mode="same")
        mass = np.clip(mass, 0, None)
        cdf = np.cumsum(mass)
        cdf /= cdf[-1]
        return x + dx / 2, cdf, sig

    def _exact_shift(self, e):
        """Recorded-minus-mode energy of every quadrature point by exact time of flight
        (nominal Ei and time at the sample, as reduced), mean removed; weights (points
        whose final energy would be <= 0 are dropped)."""
        sp = self.spec
        V = NEUTRON_V_PER_SQRT_MEV
        u0 = 1.0 / (V * np.sqrt(self.ei))
        tau, mu = self._raw[:, 0], self._raw[:, 1]
        u = u0 + mu
        ef = (1.0 / (u * V)) ** 2 - e
        ok = (ef > 0) & (u > 0)
        w = np.where(ok, self._w, 0.0)
        vf = V * np.sqrt(np.where(ok, ef, 1.0))
        t_det = tau + sp.l1 * u + sp.l2 / vf
        ef_rec = (sp.l2 / (t_det - sp.l1 * u0) / V) ** 2
        s = np.where(ok, self.ei - ef_rec - e, 0.0)
        w = w / w.sum()
        return s - w @ s, w

    def _build_table(self, n_x):
        profs = [self._profile_at(e, n_x) for e in self.e_nodes]
        zmax = max(np.max(np.abs(x)) / sg for x, _, sg in profs)
        self.z = np.linspace(-zmax, zmax, 4 * n_x + 1)
        self.table = np.array([np.interp(self.z * sg, x, c, left=0.0, right=1.0)
                               for x, c, sg in profs])

    def _cdf(self, x, e):
        """P(recorded - mode <= x) for a mode at e (arrays broadcast)."""
        e = self._clip(e)
        sg = self.spec._sigma_from_cov(self.cov, self.ei, e)
        z = np.asarray(x, float) / sg
        fi = (e - self.e_nodes[0]) / (self.e_nodes[1] - self.e_nodes[0])
        k = np.clip(np.floor(fi).astype(int), 0, len(self.e_nodes) - 2)
        t = fi - k
        dz = self.z[1] - self.z[0]
        zi = (z - self.z[0]) / dz
        j = np.clip(np.floor(zi).astype(int), 0, len(self.z) - 2)
        f = np.clip(zi - j, 0.0, 1.0)
        tab = self.table

        def at(kk):
            return tab[kk, j] * (1 - f) + tab[kk, j + 1] * f
        out = (1 - t) * at(k) + t * at(k + 1)
        return np.where(zi <= 0, 0.0, np.where(zi >= len(self.z) - 1, 1.0, out))

    # ---------------------------------------------------------------- protocol
    def fraction(self, lo, hi, centre):
        """Probability that a mode at `centre` is recorded in [lo, hi] (meV)."""
        centre = np.asarray(centre, float)
        if self.gaussian:
            from .forward import _bin_fraction
            return _bin_fraction(lo, hi, centre, self.sigma(centre))
        return self._cdf(np.asarray(hi) - centre, centre) - self._cdf(np.asarray(lo) - centre,
                                                                        centre)

    def cdf(self, x, e):
        """P(recorded - mode <= x) at mode energy e."""
        if self.gaussian:
            return ndtr(np.asarray(x, float) / self.sigma(e))
        return self._cdf(x, e)

    def fwhm(self, e, n_x=4000):
        """The profile's own full width at half maximum at mode energy e (meV)."""
        if self.gaussian:
            return float(self.fwhm_gaussian(e))
        x, c, _ = self._profile_at(float(self._clip(e)), n_x)
        pdf = np.gradient(c, x)
        return _fwhm(x, pdf)


# --------------------------------------------------------------------------- moderators
def fermi_straight_window(frequency, channel_width, length):
    """Opening of a straight-slit Fermi chopper for fast neutrons: a triangle of FWHM
    w / (omega L) (channel width w, slit length L, both m), as a Trapezoid(t, t)."""
    t = channel_width / (2 * np.pi * frequency * length)
    return Trapezoid(t, t)


def wavelength(ei):
    """Neutron wavelength (A) at energy ei (meV)."""
    return np.sqrt(81.8042 / np.asarray(ei, float))


class GaussianPulse:
    """A moderator pulse taken as Gaussian, with FWHM(lambda) interpolated from a table
    (s vs A): the usual way a facility quotes it. Callable: ei -> Gaussian."""

    def __init__(self, lam, fwhm):
        order = np.argsort(lam)
        self.lam = np.asarray(lam, float)[order]
        self.fwhm = np.asarray(fwhm, float)[order]

    def __call__(self, ei):
        return Gaussian(float(np.interp(wavelength(ei), self.lam, self.fwhm)) / FWHM_PER_SIGMA)


class TabulatedPulse:
    """A moderator pulse from an emission-time table brightness(t, E) (e.g. an SNS/McStas
    source file): callable ei -> Tabulated, interpolated linearly in log E between the two
    neighbouring energy bands, and shifted so that t = 0 is the emission time the choppers
    are phased to (`phase`: 'peak', 'mean', or a callable ei -> time in s).

    times: (Nt,) s, or (Ne, Nt) per band; energies: (Ne,) meV; brightness: (Ne, Nt)."""

    def __init__(self, times, energies, brightness, phase="peak"):
        self.energies = np.asarray(energies, float)
        b = np.asarray(brightness, float)
        t = np.asarray(times, float)
        self.times = np.broadcast_to(t, b.shape) if t.ndim == 1 else t
        self.brightness = b
        self.phase = phase
        if np.any(np.diff(self.energies) <= 0):
            raise ValueError("energies must increase")

    def __call__(self, ei):
        le = np.log(self.energies)
        x = np.log(ei)
        k = int(np.clip(np.searchsorted(le, x) - 1, 0, len(le) - 2))
        f = float(np.clip((x - le[k]) / (le[k + 1] - le[k]), 0.0, 1.0))
        t = np.union1d(self.times[k], self.times[k + 1])
        d = ((1 - f) * np.interp(t, self.times[k], self.brightness[k], left=0, right=0)
             + f * np.interp(t, self.times[k + 1], self.brightness[k + 1], left=0, right=0))
        if callable(self.phase):
            t0 = float(self.phase(ei))
        elif self.phase == "mean":
            t0 = float(np.trapezoid(t * d, t) / np.trapezoid(d, t))
        elif self.phase == "peak":
            t0 = float(t[np.argmax(d)])
        else:
            raise ValueError("phase must be 'peak', 'mean' or a callable")
        return Tabulated(t - t0, d)


def read_sns_source_table(path, phase="peak"):
    """Read an SNS moderator source file (the McStas `SNS_source` format: section 2 lists
    emission time (us), energy (eV) and brightness for each energy band) into a
    TabulatedPulse."""
    rows, in2 = [], False
    with open(path) as fh:
        for line in fh:
            if line.startswith("## section 2"):
                in2 = True
                continue
            if not in2 or line.startswith("#"):
                if in2 and line.startswith("## section2end"):
                    break
                continue
            parts = line.split()
            if len(parts) >= 3:
                rows.append([float(p) for p in parts[:3]])
    a = np.array(rows)
    if not len(a):
        raise ValueError(f"{path}: no section-2 emission-time table")
    e_ev = np.unique(a[:, 1])
    nt = len(a) // len(e_ev)
    a = a[: nt * len(e_ev)].reshape(len(e_ev), nt, 3)
    return TabulatedPulse(a[:, :, 0] * 1e-6, a[:, 0, 1] * 1e3, a[:, :, 2], phase=phase)


# --------------------------------------------------------------------------- instruments
# CNCS (SNS BL5). Distances, the chopper-4 disk radius and the moderator FWHM(lambda) are
# Mantid PyChop's CNCS model (scripts/pychop/cncs.yaml). Chopper 4 is a counter-rotating
# pair whose slot is 9, 4.4 or 2 deg on each disk ('High Flux', 'Intermediate', 'High
# Resolution': the angles PyChop's comments and the SNS CNCS McStas model give; PyChop's own
# slot widths are effective values for its Gaussian formula, not the slot), seen by the beam
# at radius 0.25 m (the 0.2825 m disk less half its 65 mm slot height) with the 21 mm beam
# width of the McStas guide ends. Chopper 1 is a straight-slit Fermi chopper (30 channels in
# 50 mm, 17 mm long, 0.15 mm walls) as in the McStas model. Every number can be
# overridden; on real data, calibrate the openings on the incoherent elastic line
# (GAP_STATUS.md, "phase 2, the energy resolution").
CNCS_GEOMETRY = {
    "x_chopper1": 6.413, "x_chopper4": 34.785, "chop_sam": 1.48, "sam_det": 3.5,
    "chopper4_beam_radius": 0.25, "chopper4_beam_width": 0.021,
    "chopper4_slot_deg": {"High Flux": 9.0, "Intermediate": 4.4, "High Resolution": 2.0},
    "fermi_channel": (0.05 - 31 * 0.00015) / 30, "fermi_length": 0.017,
    "fermi_scale": 1.0, "detector_depth": 0.025,
}
CNCS_MODERATOR_FWHM = (     # lambda (A), FWHM (us): PyChop cncs.yaml measured_width
    [0.90445614, 0.97050943, 1.04138664, 1.11744009, 1.19904779, 1.28661538, 1.38057811,
     1.48140303, 1.58959128, 1.70568063, 1.83024809, 1.96391283, 2.10733923, 2.2612402,
     2.42638069, 2.60358156, 2.79372357, 2.99775183, 3.21668046, 3.45159766, 3.70367109,
     3.97415367, 4.26438985, 4.57582225, 4.90999885, 5.26858069, 5.65335009, 6.06621956,
     6.50924128, 6.9846173, 7.49471047, 8.04205622, 8.62937515, 9.25958654, 9.93582285,
     10.66144534, 11.44006072, 12.27553911, 13.17203328, 14.13399926, 15.16621852,
     16.27382172, 17.46231422, 18.73760346, 20.10602826, 21.57439041, 23.14998844,
     24.84065387, 26.65479018, 28.60141458],
    [11.29, 12.85, 14.77, 17.17, 20.19, 23.98, 28.75, 34.74, 42.13, 51.03, 61.31, 72.47,
     83.71, 94.24, 103.81, 112.68, 121.38, 130.4, 139.97, 150.08, 160.57, 171.24, 181.89,
     192.43, 202.77, 212.97, 223.05, 233.12, 243.24, 253.48, 263.89, 274.45, 285.14,
     295.86, 306.49, 316.87, 326.81, 336.06, 344.4, 351.6, 357.47, 361.86, 364.71, 366.0,
     365.84, 364.36, 361.74, 358.23, 354.02, 349.37])


def cncs(variant="Intermediate", f_res=300.0, f1=60.0, *, moderator=None, geometry=None,
         detector_depth=None, extra_sigma=0.0) -> ChopperSpectrometer:
    """CNCS with the counter-rotating chopper 4 (`variant`: 'High Flux', 'Intermediate' or
    'High Resolution', i.e. DoubleDiskMode 1, 3 or the narrow slot, at `f_res` Hz) and the
    Fermi chopper 1 (`f1` Hz).

    moderator: default a Gaussian pulse with PyChop's measured FWHM(lambda); pass
    `read_sns_source_table(...)` for the emission-time shape itself. geometry: overrides
    of CNCS_GEOMETRY (e.g. calibrated slot angles, or `fermi_scale` to widen chopper 1's
    opening). detector_depth: the full width (m) of a uniform spread in L2, default the
    tube diameter."""
    g = dict(CNCS_GEOMETRY)
    g.update(geometry or {})
    if variant not in g["chopper4_slot_deg"]:
        raise ValueError(f"variant must be one of {list(g['chopper4_slot_deg'])}")
    fw = fermi_straight_window(f1, g["fermi_channel"], g["fermi_length"])
    c1 = TimeWindow("chopper 1 (Fermi)", g["x_chopper1"],
                    Trapezoid(g["fermi_scale"] * fw.a, g["fermi_scale"] * fw.b))
    c4 = DiskChopper.from_slot_angle("chopper 4 (counter-rotating pair)", g["x_chopper4"],
                                     f_res, g["chopper4_slot_deg"][variant],
                                     g["chopper4_beam_radius"], g["chopper4_beam_width"],
                                     counter_rotating=True)
    if moderator is None:
        lam, fw = CNCS_MODERATOR_FWHM
        moderator = GaussianPulse(lam, np.asarray(fw) * 1e-6)
    depth = g["detector_depth"] if detector_depth is None else detector_depth
    return ChopperSpectrometer("CNCS " + variant, g["x_chopper4"] + g["chop_sam"],
                               g["sam_det"], [c1, c4], moderator=moderator,
                               detector_depth=depth, extra_sigma=extra_sigma)


def _deposit(s, w, dx, nh):
    """Linear (cloud-in-cell) deposit of weights w at positions s on nodes dx*(-nh..nh)."""
    g = s / dx + nh
    i = np.floor(g).astype(int)
    f = g - i
    n = 2 * nh + 1
    return (np.bincount(i, w * (1 - f), minlength=n + 1)[:n]
            + np.bincount(i + 1, w * f, minlength=n + 1)[:n])


def _kernel(width, kind, dx):
    """A normalised discrete kernel on spacing dx: a box of full width `width`, or a
    Gaussian of sigma `width` (+-8 sigma)."""
    if kind == "box":
        # exact overlap of each grid cell [x - dx/2, x + dx/2] with [-h, h]
        h = width / 2
        n = int(np.ceil(h / dx)) + 1
        centres = dx * np.arange(-n, n + 1)
        k = np.maximum(np.clip(centres + dx / 2, -h, h) - np.clip(centres - dx / 2, -h, h),
                       0.0)
    else:
        n = int(np.ceil(GAUSS_TAIL * width / dx)) + 1
        edges = dx * (np.arange(-n, n + 2) - 0.5)
        k = np.diff(ndtr(edges / width))
    k = np.asarray(k, float)
    return k / k.sum()


def _fwhm(x, pdf):
    peak = np.max(pdf)
    above = np.where(pdf >= peak / 2)[0]
    i0, i1 = above[0], above[-1]

    def cross(i, j):     # interpolate between i (below) and j (above) half maximum
        return x[i] + (peak / 2 - pdf[i]) * (x[j] - x[i]) / (pdf[j] - pdf[i])
    left = cross(i0 - 1, i0) if i0 > 0 else x[0]
    right = cross(i1 + 1, i1) if i1 < len(x) - 1 else x[-1]
    return float(right - left)
