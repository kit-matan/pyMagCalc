"""Q resolution and the Q-E correlation of a direct-geometry spectrometer (neutron-data
phase 2, Level 1): the four-dimensional resolution entering through the forward model's
sample quadrature.

A recorded event differs from the true one by (dQ, dE) = (Q_rec - Q, E_rec - E). These
deviations are functions of the primary deviations of one neutron:
- its emission time and inverse speed (tau, mu): the chopper times of
  `magcalc.resolution`, with their covariance;
- the incident divergence (horizontal, vertical);
- the scattering point in the sample;
- the detection point in the pixel (along the tube, across it, in depth).

`kinematics` is that function, exact: the true event is built from the neutron and
reduced the way a reduction does, with the nominal Ei, the nominal time at the sample,
the sample centre and the pixel centre. Its Jacobian at the nominal event, by central
differences, gives the 4x4 covariance Sigma. The lab frame has x along the beam, z
vertical up and y = z x x.

For a sample at recorded Q, the recorded energy of a mode is
    E_rec = eps(Q - dQ) + b . dQ + r,    b = Sigma_QQ^-1 Sigma_QE,    dQ ~ N(0, Sigma_QQ),
where the residual r carries the energy resolution's own line shape (Level 0),
narrowed to the conditional width sigma_E|Q = sqrt(Sigma_EE - Sigma_EQ b). This is
exact for a Gaussian kernel, and keeps the instrument's line shape otherwise.

The average over dQ is done one of two ways (`QEResolution(method=...)`):
- 'quadrature' (default), deterministic. Write dQ = L z (L L^T = Sigma_QQ, z standard
  normal) and split z = sqrt(1 - a) u + sqrt(a) w, a = `smoothing`. u runs over a
  Gauss-Hermite product grid (`nodes` per axis), or over the 14-point degree-5 rule
  (`nodes="s14"`, about half the model evaluations of the 3^3 grid), where the model is
  evaluated; it carries the dispersion's curvature, a minimum's one-sided tail and a
  crossing's kink.
  w is integrated analytically at each node: a quadratic fitted through all the nodes'
  f = eps(Q - dQ) + b . dQ, f ~ c + g.z + z^T H z / 2, gives the node's mean
  f + a tr(H) / 2 and variance a |g + H z|^2 + a^2 |H|_F^2 / 2, which widen the
  residual in quadrature. Exact for a linear dispersion, and for a quadratic one up to
  the Gaussian treatment of w; no random noise.
- 'sample': one draw dQ = L z per virtual sample (scrambled Sobol). Exact on average
  for any dispersion, but each draw carries only the narrow residual, so on a steep
  dispersion a bin's prediction counts how many draws land in it: noisy until the
  samples per bin number in the thousands.

The pixel each sample was measured in comes from a geometry provider:
- `RunGoniometer`, for count cells: the run's rotation angle;
- `RotationScan`, for histograms that summed the rotations (MDNorm): the two angles
  at which a vertical-axis rotation puts that Q on the detector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence

import numpy as np
from scipy.special import ndtri

from .neutron_data import NEUTRON_V_PER_SQRT_MEV

K_PER_SQRT_MEV = 1.0 / np.sqrt(2.072124)        # |k| in 1/A per sqrt(meV)
V = NEUTRON_V_PER_SQRT_MEV

PRIMARY = ("tau", "mu", "div_h", "div_v", "x_s", "y_s", "z_s",
           "det_along", "det_across", "det_depth")


def _k(e):
    return K_PER_SQRT_MEV * np.sqrt(e)


def rotation_z(psi_deg):
    c, s = np.cos(np.radians(psi_deg)), np.sin(np.radians(psi_deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def kinematics(u, *, ei, omega, n_f, l1, l2):
    """(dQ (N, 3), dE (N,)) of neutrons with primary deviations u (N, 10) in PRIMARY
    order (s, s/m, rad, rad, m, m, m, m, m, m), for true transfers `omega` (N,) meV
    recorded in pixels whose centres lie along the unit vectors n_f (N, 3, lab) from the
    sample centre. One neutron: pass (10,), scalars and (3,)."""
    u = np.atleast_2d(np.asarray(u, float))
    n = len(u)
    omega = np.broadcast_to(np.asarray(omega, float), (n,))
    n_f = np.broadcast_to(np.asarray(n_f, float), (n, 3))
    tau, mu, ah, av, xs, ys, zs, d_al, d_ac, d_dp = u.T
    zhat = np.array([0.0, 0.0, 1.0])
    w = np.cross(zhat, n_f)
    nw = np.linalg.norm(w, axis=1, keepdims=True)
    w = np.where(nw > 1e-12, w / np.where(nw > 1e-12, nw, 1.0), [0.0, 1.0, 0.0])
    u0 = 1.0 / (V * np.sqrt(ei))
    uu = u0 + mu
    ei_true = (1.0 / (uu * V)) ** 2
    ef_true = ei_true - omega
    e_i = np.stack([np.ones(n), ah, av], 1)
    e_i /= np.linalg.norm(e_i, axis=1, keepdims=True)
    r_s = np.stack([xs, ys, zs], 1)
    r_d = (l2 + d_dp)[:, None] * n_f + d_al[:, None] * zhat + d_ac[:, None] * w
    path = r_d - r_s
    l2_true = np.linalg.norm(path, axis=1)
    e_f = path / l2_true[:, None]
    t_s = tau + (l1 + xs) * uu
    t_d = t_s + l2_true / (V * np.sqrt(ef_true))
    ef_rec = (l2 / (t_d - l1 * u0) / V) ** 2
    e_rec = ei - ef_rec
    q_rec = _k(ei) * np.array([1.0, 0.0, 0.0]) - _k(ef_rec)[:, None] * n_f
    q_true = _k(ei_true)[:, None] * e_i - _k(ef_true)[:, None] * e_f
    return q_rec - q_true, e_rec - omega


@dataclass
class GeometrySpread:
    """The spreads of the geometric primary variables, as full widths of uniform
    distributions (variance width^2 / 12) unless said otherwise.

    divergence: (sigma_h, sigma_v) in rad, or a callable ei -> (sigma_h, sigma_v)
        (a guide's divergence grows with the wavelength).
    sample_size: (a, b, c) m, a box in the SAMPLE frame (a, b horizontal, c vertical);
        it turns with the goniometer.
    pixel_height: m along the tube; pixel_width: m across it (horizontal, normal to the
        scattered beam); depth: m along the scattered beam. Use the SAME depth as the
        energy resolution's `detector_depth`, so that the residual width is consistent.
        A tube of diameter D is close to pixel_width = depth = 0.87 D (variance D^2/16).
    """
    divergence: object = (0.0, 0.0)
    sample_size: Sequence[float] = (0.0, 0.0, 0.0)
    pixel_height: float = 0.0
    pixel_width: float = 0.0
    depth: float = 0.0

    def divergence_sigma(self, ei):
        d = self.divergence(ei) if callable(self.divergence) else self.divergence
        return float(d[0]), float(d[1])


def primary_covariance(time_cov, spread: GeometrySpread, ei, sample_rotation):
    """(N, 10, 10) covariance of the primary variables; sample_rotation (N, 3, 3) turns
    the sample box into the lab."""
    R = np.asarray(sample_rotation, float).reshape(-1, 3, 3)
    n = len(R)
    sh, sv = spread.divergence_sigma(ei)
    a, b, c = spread.sample_size
    box = np.diag([a * a, b * b, c * c]) / 12.0
    C = np.zeros((n, 10, 10))
    C[:, :2, :2] = time_cov
    C[:, 2, 2], C[:, 3, 3] = sh * sh, sv * sv
    C[:, 4:7, 4:7] = R @ box @ np.transpose(R, (0, 2, 1))
    C[:, 7, 7] = spread.pixel_height ** 2 / 12.0
    C[:, 8, 8] = spread.pixel_width ** 2 / 12.0
    C[:, 9, 9] = spread.depth ** 2 / 12.0
    return C


def jacobian(*, ei, omega, n_f, l1, l2, steps):
    """(N, 4, 10) Jacobian of (dQ, dE) by central differences of `kinematics`."""
    n_f = np.atleast_2d(np.asarray(n_f, float))
    n = len(n_f)
    omega = np.broadcast_to(np.asarray(omega, float), (n,))
    J = np.zeros((n, 4, 10))
    for i in range(10):
        up = np.zeros((n, 10))
        up[:, i] = steps[i]
        qp, ep = kinematics(up, ei=ei, omega=omega, n_f=n_f, l1=l1, l2=l2)
        qm, em = kinematics(-up, ei=ei, omega=omega, n_f=n_f, l1=l1, l2=l2)
        J[:, :3, i] = (qp - qm) / (2 * steps[i])
        J[:, 3, i] = (ep - em) / (2 * steps[i])
    return J


_STEP_FLOOR = np.array([1e-6, 1e-9, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4])


def covariance(*, ei, omega, n_f, l1, l2, time_cov, spread: GeometrySpread,
               sample_rotation=None, return_jacobian=False):
    """(N, 4, 4) covariance of (dQ_lab, dE) at recorded pixels n_f (N, 3) and energies
    omega (N,), from the Jacobian of `kinematics` and the primary covariances
    (time_cov: 2x2 of (tau, mu)). One pixel: pass (3,) and a scalar; a (4, 4) returns."""
    single = np.ndim(n_f) == 1
    n_f = np.atleast_2d(np.asarray(n_f, float))
    n = len(n_f)
    R = np.broadcast_to(np.eye(3) if sample_rotation is None
                        else np.asarray(sample_rotation, float), (n, 3, 3))
    C = primary_covariance(time_cov, spread, ei, R)
    scale = np.sqrt(np.maximum(np.max(np.diagonal(C, axis1=1, axis2=2), axis=0), 0.0))
    steps = 1e-3 * np.where(scale > 0, scale, _STEP_FLOOR)
    J = jacobian(ei=ei, omega=omega, n_f=n_f, l1=l1, l2=l2, steps=steps)
    S = J @ C @ np.transpose(J, (0, 2, 1))
    if single:
        S, J = S[0], J[0]
    return (S, J) if return_jacobian else S


# --------------------------------------------------------------------------- geometry
class RunGoniometer:
    """Count cells: each sample's run fixes the rotation. Q_lab = R_axis(psi_run) U Q,
    with Q the model-frame Cartesian Q, U the orientation at psi = 0 (model -> lab) and
    a vertical rotation axis. psi: dict run -> angle in degrees."""

    def __init__(self, U, psi: Dict[int, float]):
        self.U = np.asarray(U, float)
        self.psi = dict(psi)

    def rotations(self, data, q_model, sample_index, energy):
        if data.sample_run is None:
            raise ValueError("these data carry no sample_run; read them as count cells")
        runs = data.sample_run[sample_index]
        missing = set(np.unique(runs).tolist()) - set(self.psi)
        if missing:
            raise ValueError(f"no rotation angle for runs {sorted(missing)[:5]}")
        R = np.stack([rotation_z(self.psi[int(r)]) @ self.U for r in runs])
        return R, np.ones(len(runs)), np.arange(len(runs))


class RotationScan:
    """Histograms that summed a vertical-axis rotation scan (MDNorm): for a model-frame
    Q and recorded energy E, the angles psi at which |k_i - R(psi) U Q| = k_f, i.e.
    q_x cos psi - q_y sin psi = c, c = (k_i^2 + Q^2 - k_f^2) / (2 k_i), q = U Q.
    There are two (scattering to either side). A branch counts if psi lies in one of
    `psi_ranges` and `accept(n_f)` (default: every direction; it takes an (N, 3) array
    of pixel directions and returns N booleans) accepts its pixel direction. Valid branches share the sample's weight equally; a sample with none
    keeps the first branch, flagged in `last_unreachable`."""

    def __init__(self, U, psi_ranges, ei, accept: Optional[Callable] = None):
        self.U = np.asarray(U, float)
        self.ranges = [tuple(r) for r in psi_ranges]
        self.ei = float(ei)
        self.accept = accept
        self.last_unreachable = 0

    def branches(self, q_model, energy):
        q = q_model @ self.U.T
        ki = _k(self.ei)
        kf = _k(self.ei - np.asarray(energy, float))
        Q2 = np.sum(q * q, axis=1)
        c = (ki * ki + Q2 - kf * kf) / (2 * ki)
        rho = np.hypot(q[:, 0], q[:, 1])
        phi0 = np.arctan2(q[:, 1], q[:, 0])
        ok = np.abs(c) <= rho
        a = np.arccos(np.clip(c / np.where(rho > 0, rho, 1.0), -1, 1))
        psis = np.degrees(np.stack([-phi0 + a, -phi0 - a], 1))
        psis = (psis + 180.0) % 360.0 - 180.0
        return psis, ok

    def _in_range(self, psi):
        good = np.zeros(psi.shape, bool)
        for lo, hi in self.ranges:
            good |= (psi >= lo) & (psi <= hi)
        return good

    def rotations(self, data, q_model, sample_index, energy):
        q_model = np.asarray(q_model, float)
        n = len(q_model)
        psis, ok = self.branches(q_model, energy)
        ki = _k(self.ei) * np.array([1.0, 0.0, 0.0])
        good = np.zeros((n, 2), bool)
        Rb = np.zeros((n, 2, 3, 3))
        for j in range(2):
            c, s_ = np.cos(np.radians(psis[:, j])), np.sin(np.radians(psis[:, j]))
            Rz = np.zeros((n, 3, 3))
            Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s_, s_, c, 1
            Rb[:, j] = Rz @ self.U
            g = ok & self._in_range(psis[:, j])
            if self.accept is not None and g.any():
                kf = ki - np.einsum("nij,nj->ni", Rb[:, j], q_model)
                n_f = kf / np.linalg.norm(kf, axis=1, keepdims=True)
                g &= np.asarray(self.accept(n_f), bool)
            good[:, j] = g
        none = ~good.any(axis=1)
        self.last_unreachable = int(none.sum())
        good[none, 0] = True                       # keep one branch, flagged above
        nb = good.sum(axis=1)
        src = np.repeat(np.arange(n), nb)
        R = Rb[good]                               # row-major: sample, then branch
        w = np.repeat(1.0 / nb, nb)
        return R, w, src


# --------------------------------------------------------------------------- the kernel
def _gauss_hermite_grid(m):
    """The m^3 product Gauss-Hermite rule for a standard normal in 3-D: nodes (m^3, 3),
    weights summing to 1."""
    x, w = np.polynomial.hermite_e.hermegauss(m)
    w = w / w.sum()
    g = np.stack(np.meshgrid(x, x, x, indexing="ij"), -1).reshape(-1, 3)
    wg = np.einsum("i,j,k->ijk", w, w, w).ravel()
    return g, wg


def _degree5_rule_14():
    """A 14-point rule for a standard normal in 3-D, exact for every polynomial of degree
    <= 5: the 6 points +-r e_i (weight w1) and the 8 points (+-s, +-s, +-s) (weight w2).
    Odd moments vanish by symmetry; the even ones up to degree 4 ask
        6 w1 + 8 w2 = 1,  2 w1 r^2 + 8 w2 s^2 = 1,  2 w1 r^4 + 8 w2 s^4 = 3,  8 w2 s^4 = 1,
    solved by r^2 = 5/2, s^2 = 5, w1 = 4/25, w2 = 1/200 (all weights positive). Its 14
    points determine the 10 coefficients of the quadratic fit."""
    r, s = np.sqrt(2.5), np.sqrt(5.0)
    axis = np.concatenate([np.eye(3), -np.eye(3)]) * r
    corners = np.array([[a, b, c] for a in (1, -1) for b in (1, -1) for c in (1, -1)],
                       float) * s
    z = np.concatenate([axis, corners])
    w = np.concatenate([np.full(6, 4.0 / 25.0), np.full(8, 1.0 / 200.0)])
    return z, w


RULES = {"s14": _degree5_rule_14}


def _quadrature_rule(nodes):
    """The quadrature's (nodes (P, 3), weights (P,)): the nodes^3 Gauss-Hermite grid for an
    int, else the named rule of RULES."""
    if isinstance(nodes, str):
        return RULES[nodes]()
    return _gauss_hermite_grid(nodes)


def _rotations(n, seed):
    """n fixed random rotations of 3-space (QR of Gaussian matrices, sign-fixed)."""
    a = np.random.default_rng(seed).standard_normal((n, 3, 3))
    q, r = np.linalg.qr(a)
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[np.linalg.det(q) < 0, :, 0] *= -1
    return q


def _quadratic_fit_matrix(z):
    """(10, P) least-squares map from values at the nodes z (P, 3) to the coefficients of
    c + g.z + (H11 z1^2 + H22 z2^2 + H33 z3^2) / 2 + H12 z1 z2 + H13 z1 z3 + H23 z2 z3."""
    z1, z2, z3 = z.T
    A = np.stack([np.ones(len(z)), z1, z2, z3, z1 * z1 / 2, z2 * z2 / 2, z3 * z3 / 2,
                  z1 * z2, z1 * z3, z2 * z3], 1)
    return np.linalg.pinv(A)


@dataclass
class QEKernel:
    """Per virtual sample v (a data sample, its energy block, its rotation branch):
    `sample` its data sample, `weight` its share, `dq` (Nv * points, 3) the model-frame
    displacements (subtract from Q; `points` consecutive rows per virtual sample: one
    draw, or the cubature points), `shift` (Nv * points,) meV, `rho` (Nv,) the residual
    width as a fraction of the energy resolution's, `block` (Nv,) its energy block; and
    per data entry (bin, sample) the virtual samples it averages: `entry_ptr`,
    `entry_v`."""
    sample: np.ndarray
    weight: np.ndarray
    dq: np.ndarray
    shift: np.ndarray
    rho: np.ndarray
    block: np.ndarray
    entry_ptr: np.ndarray
    entry_v: np.ndarray
    sigma: np.ndarray = field(default=None)      # (Nv, 4, 4) for inspection
    points: int = 1
    # 'quadrature': the grid's whitened nodes (points, 3), their weights (points,), the
    # quadratic fit's (10, points) least-squares matrix, and the analytic share a
    node_z: np.ndarray = field(default=None)
    node_weight: np.ndarray = field(default=None)
    fit_matrix: np.ndarray = field(default=None)
    smoothing: float = 0.0


class QEResolution:
    """Level 1 resolution for `ForwardModel`: an energy resolution (anything with
    fraction(lo, hi, centre) and sigma(E): `ChopperEnergyResolution` or
    `GaussianEnergyResolution`), the instrument's flight paths and time covariance (from
    `energy.spec` and `energy.cov`, or given), the geometric spreads, and a geometry
    provider.

    energy_blocks: None evaluates the kernel at each entry's own bin energy (count cells:
    one bin per sample); an int groups bins into that many energy blocks per sample
    (MDNorm, where a cell's samples serve every energy bin). method: 'quadrature'
    (default, deterministic; `nodes` per axis of a Gauss-Hermite grid, or "s14" for the
    14-point degree-5 rule; analytic share `smoothing`) or 'sample' (Monte Carlo); see
    the module docstring. seed: the normals of 'sample', one
    scrambled-Sobol point per data sample.
    """

    def __init__(self, energy, geometry, spread: GeometrySpread, *, ei=None, l1=None,
                 l2=None, time_cov=None, energy_blocks=None, seed=0, intrinsic_sigma=0.0,
                 method="quadrature", nodes=3, smoothing=0.3):
        """intrinsic_sigma: a Gaussian width (meV) of the modes themselves (a lifetime or
        an unresolved splitting), added after the Q-E conditioning; 0 for sharp modes."""
        if method not in ("quadrature", "sample"):
            raise ValueError(f"method must be 'quadrature' or 'sample', not {method!r}")
        if isinstance(nodes, str):
            if nodes not in RULES:
                raise ValueError(f"nodes must be an int >= 3 or one of {sorted(RULES)}, "
                                 f"not {nodes!r}")
        elif int(nodes) < 3:
            raise ValueError("nodes must be >= 3 per axis (the quadratic fit needs them)")
        if not 0.0 <= float(smoothing) < 1.0:
            raise ValueError("smoothing must lie in [0, 1)")
        self.method = method
        self.nodes = nodes if isinstance(nodes, str) else int(nodes)
        self.smoothing = float(smoothing)
        self.energy = energy
        self.geometry = geometry
        self.spread = spread
        spec = getattr(energy, "spec", None)
        self.ei = float(ei if ei is not None else energy.ei)
        self.l1 = float(l1 if l1 is not None else spec.l1)
        self.l2 = float(l2 if l2 is not None else spec.l2)
        self.time_cov = np.asarray(time_cov if time_cov is not None else energy.cov, float)
        self.energy_blocks = energy_blocks
        self.seed = int(seed)
        self.intrinsic_sigma = float(intrinsic_sigma)

    def fraction(self, lo, hi, centre, shift=0.0, rho=1.0, var=None):
        """Probability that a mode at `centre` is recorded in [lo, hi] when the recorded
        energy is shifted by `shift` and the residual narrowed to `rho` of the energy
        resolution's width. `var` (meV^2) and, with `intrinsic_sigma` (meV), the mode's
        own Gaussian width are added in quadrature to the residual's: exact for a
        Gaussian residual, and for any other the residual's own shape, widened."""
        extra = self.intrinsic_sigma ** 2
        if var is not None:
            extra = extra + np.asarray(var, float)
        if np.any(extra > 0):
            s0 = np.asarray(self.energy.sigma(np.asarray(centre, float)), float)
            rho = np.sqrt(np.asarray(rho, float) ** 2 + extra / s0 ** 2)
        return self._fraction(lo, hi, centre, shift, rho)

    def _fraction(self, lo, hi, centre, shift=0.0, rho=1.0):
        centre = np.asarray(centre, float)
        lo = np.asarray(lo) - shift
        hi = np.asarray(hi) - shift
        if hasattr(self.energy, "cdf"):
            rho = np.asarray(rho, float)
            return (self.energy.cdf((hi - centre) / rho, centre)
                    - self.energy.cdf((lo - centre) / rho, centre))
        from .forward import _bin_fraction
        return _bin_fraction(lo, hi, centre, rho * self.energy.sigma(centre))

    def prepare(self, data, b_matrix, e_bin_centre, entry_bin, entry_sample,
                crystal_position=None):
        """The kernel for these data (entries = (bin, sample) pairs of the quadrature).
        crystal_position: a sample-frame offset (m) of the scattering crystal from the
        rotation axis; it shifts the kernel's mean."""
        q_model = data.sample_cartesian(b_matrix)
        e_bin = np.asarray(e_bin_centre, float)[entry_bin]
        if self.energy_blocks is None:
            key_e = np.asarray(entry_bin, np.int64)
            uniq, inv = np.unique(np.stack([entry_sample, key_e], 1), axis=0,
                                  return_inverse=True)
            e_blk = np.array([e_bin_centre[b] for b in uniq[:, 1]])
        else:
            lo, hi = float(np.min(e_bin)), float(np.max(e_bin))
            nb = int(self.energy_blocks)
            edges = np.linspace(lo, hi + 1e-12, nb + 1)
            blk = np.clip(np.searchsorted(edges, e_bin, side="right") - 1, 0, nb - 1)
            uniq, inv = np.unique(np.stack([entry_sample, blk], 1), axis=0,
                                  return_inverse=True)
            centres = 0.5 * (edges[:-1] + edges[1:])
            e_blk = centres[uniq[:, 1]]
        inv = inv.ravel()
        base = uniq[:, 0]
        R, w, src = self.geometry.rotations(data, q_model[base], base, e_blk)
        nv = len(src)
        ki = _k(self.ei) * np.array([1.0, 0.0, 0.0])
        q_lab = np.einsum("vij,vj->vi", R, q_model[base[src]])
        kf = ki - q_lab
        n_f = kf / np.linalg.norm(kf, axis=1, keepdims=True)
        omega = e_blk[src]
        S = covariance(ei=self.ei, omega=omega, n_f=n_f, l1=self.l1, l2=self.l2,
                       time_cov=self.time_cov, spread=self.spread, sample_rotation=R)
        Sqq, Sqe, See = S[:, :3, :3], S[:, :3, 3], S[:, 3, 3]
        ev, U = np.linalg.eigh(Sqq)
        ev = np.clip(ev, 0.0, None)
        extra = {}
        if self.method == "quadrature":
            u, wq = _quadrature_rule(self.nodes)
            zn = np.sqrt(1.0 - self.smoothing) * u
            # each DATA sample turns the grid by its own fixed random rotation, so the
            # nodes' comb falls differently for a bin's samples and averages out; the
            # analytic share is isotropic, so the fit is done in the grid's own frame
            rot = _rotations(data.n_samples, self.seed)[base[src]]
            dq_lab = np.einsum("vij,vjk,pk->vpi", U * np.sqrt(ev)[:, None, :], rot, zn)
            extra = dict(node_z=zn, node_weight=wq, fit_matrix=_quadratic_fit_matrix(zn),
                         smoothing=self.smoothing)
        else:
            # normals: one scrambled-Sobol point per DATA sample (consecutive samples of
            # a cell get consecutive points, so each bin's set is well spread; a sample
            # keeps its normal in every energy block and branch, so the model stays
            # smooth)
            from scipy.stats import qmc
            ns = data.n_samples
            m = int(np.ceil(np.log2(max(ns, 2))))
            zz = qmc.Sobol(3, scramble=True, seed=self.seed).random_base2(m)[:ns]
            z = ndtri(np.clip(zz, 1e-12, 1 - 1e-12))[base[src]]
            dq_lab = np.einsum("vij,vj->vi", U * np.sqrt(ev)[:, None, :], z)[:, None, :]
        npts = dq_lab.shape[1]
        tol = 1e-12 * np.maximum(ev.max(axis=1, keepdims=True), 1e-300)
        inv_ev = np.where(ev > tol, 1.0 / np.where(ev > tol, ev, 1.0), 0.0)
        Sinv = np.einsum("vij,vj,vkj->vik", U, inv_ev, U)
        b = np.einsum("vij,vj->vi", Sinv, Sqe)
        var_res = np.clip(See - np.einsum("vi,vi->v", Sqe, b), 0.0, None)
        shift = np.einsum("vi,vpi->vp", b, dq_lab)
        if crystal_position is not None:
            u = np.zeros((nv, 10))
            u[:, 4:7] = R @ np.asarray(crystal_position, float)
            mq, me = kinematics(u, ei=self.ei, omega=omega, n_f=n_f, l1=self.l1,
                                l2=self.l2)
            dq_lab = dq_lab + mq[:, None, :]
            shift = shift + me[:, None]
        # R^T dq_lab: back to the model frame; `npts` consecutive rows per virtual sample
        dq = np.einsum("vji,vpj->vpi", R, dq_lab).reshape(nv * npts, 3)
        shift = shift.reshape(nv * npts)
        rho = np.where(See > 0, np.sqrt(var_res / np.where(See > 0, See, 1.0)), 1.0)
        sig = S
        # entries -> virtual samples (a sample's branches)
        order = np.argsort(src, kind="stable")
        starts = np.searchsorted(src[order], np.arange(len(base)))
        ends = np.searchsorted(src[order], np.arange(len(base)), side="right")
        cnt = (ends - starts)[inv]
        entry_ptr = np.concatenate([[0], np.cumsum(cnt)])
        entry_v = np.concatenate([order[starts[i]:ends[i]] for i in inv]) if len(inv) else \
            np.zeros(0, np.int64)
        return QEKernel(sample=base[src], weight=w, dq=dq, shift=shift, rho=rho,
                        block=uniq[src, 1], entry_ptr=entry_ptr, entry_v=entry_v, sigma=sig,
                        points=npts, **extra)
