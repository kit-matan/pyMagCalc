"""magcalc.qe_resolution: the Q resolution and the Q-E correlation (Level 1).

Oracles:
- the independent ray trace (tests/raytrace_tof.py, `trace_qe`): neutrons through the
  real chopper geometry, a uniform divergence, a uniform sample box and pixel volume,
  exact geometry and reduction. Its 4x4 covariance of (dQ, dE) is matched element by
  element, a strong tilt included;
- first derivatives in closed form: divergence moves Q by k_i per rad, an across-pixel
  offset by k_f / L2 per metre, an offset of the sample along the beam shifts the
  elastic line by (2 E_f v_f / L2)(x u0 - x n_x / v_f);
- the forward model on a LINEAR dispersion eps = e0 + v.Q with a Gaussian kernel: each
  bin is the erf with s^2 = (-v, 1)^T Sigma (-v, 1), exactly (the conditional
  decomposition and every frame and sign in the plumbing);
- a flat dispersion: Level 1 reduces to Level 0;
- the rotation-scan branches satisfy |k_i - R(psi) U Q| = k_f.
"""
import os
import sys

import numpy as np
import pytest
from scipy.special import ndtr

from magcalc.forward import Crystal, ForwardModel
from magcalc.neutron_data import BinnedData
from magcalc.qe_resolution import (GeometrySpread, QEResolution, RotationScan, RunGoniometer,
                                   covariance, kinematics, rotation_z)
from magcalc.resolution import (ChopperSpectrometer, DiskChopper, Gaussian, IkedaCarpenter,
                                TimeWindow)

sys.path.insert(0, os.path.dirname(__file__))
from raytrace_tof import sampler, trace_qe  # noqa: E402

K = 1.0 / np.sqrt(2.072124)
V = 437.393
EI, L1, L2 = 7.7, 30.0, 4.0


def _gaussian_spec():
    return ChopperSpectrometer("g", L1, L2, [TimeWindow("P", 14.2, Gaussian(10e-6)),
                                             TimeWindow("M", 28.4, Gaussian(8e-6))],
                               moderator=Gaussian(40e-6))


def _n_f(tth_deg, up=0.0):
    t = np.radians(tth_deg)
    n = np.array([np.cos(t), np.sin(t), up])
    return n / np.linalg.norm(n)


# ------------------------------------------------------------------ kinematics
def test_kinematics_vanishes_at_the_nominal_event():
    dq, de = kinematics(np.zeros(10), ei=EI, omega=2.0, n_f=_n_f(50), l1=L1, l2=L2)
    assert np.max(np.abs(dq)) < 1e-12 and abs(de[0]) < 1e-12


def test_first_derivatives_in_closed_form():
    n_f, om = _n_f(70.0), 2.5
    _, J = covariance(ei=EI, omega=om, n_f=n_f, l1=L1, l2=L2, time_cov=np.eye(2) * 1e-12,
                      spread=GeometrySpread((1e-3, 1e-3), (1e-3,) * 3, 1e-3, 1e-3, 1e-3),
                      return_jacobian=True)
    ki, kf = K * np.sqrt(EI), K * np.sqrt(EI - om)
    # divergence tilts k_i: Q_true gains k_i * angle along y (z), so dQ = -k_i per rad
    assert J[:3, 2] == pytest.approx([0.0, -ki, 0.0], abs=1e-6)
    assert J[:3, 3] == pytest.approx([0.0, 0.0, -ki], abs=1e-6)
    # an across-pixel offset tilts k_f by d / L2 towards w = z x n_f: dQ = +k_f / L2 w
    w = np.cross([0, 0, 1.0], n_f)
    w /= np.linalg.norm(w)
    assert J[:3, 8] == pytest.approx(kf / L2 * w, abs=1e-6)
    # neither moves the recorded energy to first order
    assert abs(J[3, 2]) < 1e-6 and abs(J[3, 8]) < 1e-6
    # a sample offset x along the beam: dt_S = x u0, L2' = L2 - x n_x
    u0 = 1 / (V * np.sqrt(EI))
    vf = V * np.sqrt(EI - om)
    want = 2 * (EI - om) * vf / L2 * (u0 - n_f[0] / vf)
    assert J[3, 4] == pytest.approx(want, rel=1e-6)


def test_covariance_against_the_ray_trace():
    """Real chopper geometry (a disk and a counter-rotating pair), an Ikeda-Carpenter
    pulse, uniform divergence, sample box and pixel volume, at 2theta = 60 deg."""
    ic = IkedaCarpenter(1 / 30e-6, 1 / 150e-6, 0.5)
    pulse = IkedaCarpenter(ic.alpha, ic.beta, ic.R, shift=-ic.peak())
    spec = ChopperSpectrometer("x", L1, L2, [
        DiskChopper("P", 14.2, 300.0, 0.03, 0.30, 0.01),
        DiskChopper("M", 28.4, 300.0, 0.03, 0.30, 0.02, counter_rotating=True)],
        moderator=pulse, detector_depth=0.02)
    rays = [dict(kind="disk", x=14.2, f=300.0, w=0.03, r=0.30, b=0.01),
            dict(kind="pair", x=28.4, f=300.0, w=0.03, r=0.30, b=0.02)]
    om, n_f = 3.0, _n_f(60.0, 0.05)
    spread = GeometrySpread((0.02 / np.sqrt(12), 0.03 / np.sqrt(12)), (0.01, 0.006, 0.02),
                            0.02, 0.022, 0.02)
    S = covariance(ei=EI, omega=om, n_f=n_f, l1=L1, l2=L2,
                   time_cov=spec.resolution(EI).cov, spread=spread)
    dQ, dE = trace_qe(np.random.default_rng(3), ei=EI, l1=L1, l2=L2, choppers=rays,
                      pulse=sampler(pulse), omega=om, n_f=n_f, div_h=0.02, div_v=0.03,
                      sample=(0.01, 0.006, 0.02), pixel=(0.02, 0.022, 0.02),
                      n_want=200_000)
    C = np.cov(np.column_stack([dQ, dE]).T)
    d = np.sqrt(np.diag(S))
    assert np.max(np.abs((C - S) / np.outer(d, d))) < 1e-2     # measured 3e-3
    assert S[0, 3] / (d[0] * d[3]) > 0.8                      # a strongly tilted kernel


# ------------------------------------------------------------------ forward model
class LinearMode:
    """A toy calculator: one mode eps(Q) = e0 + v.Q of unit intensity."""

    def __init__(self, e0, v):
        self.e0, self.v = e0, np.asarray(v, float)
        self.hamiltonian_params = [0.0]

    def update_hamiltonian_params(self, p):
        pass

    def sqw_batch(self, q, apply_g=False, g_tensors=None):
        q = np.asarray(q, float)
        return (self.e0 + q @ self.v)[:, None], np.ones((len(q), 1))


def _point(q0, edges, n):
    """One Q cell of n identical samples at q0 (model frame = data frame, B = I)."""
    edges = np.asarray(edges, float)
    nb = len(edges) - 1
    return BinnedData(np.zeros(nb), np.ones(nb), edges[:-1], edges[1:],
                      samples_hkl=np.repeat(np.atleast_2d(q0), n, axis=0),
                      cell_ptr=[0, n], bin_ptr=np.arange(nb + 1), bin_cells=np.zeros(nb, int),
                      bin_cell_w=np.ones(nb), sample_run=np.zeros(n, int))


def _setup(v, spread, e_mode=3.0, n=4096):
    n_f = _n_f(55.0)
    q0 = K * np.sqrt(EI) * np.array([1.0, 0, 0]) - K * np.sqrt(EI - e_mode) * n_f
    calc = LinearMode(e_mode - np.dot(v, q0), v)
    energy = _gaussian_spec().resolution(EI)
    qe = QEResolution(energy, RunGoniometer(np.eye(3), {0: 0.0}), spread)
    edges = e_mode + np.linspace(-0.6, 0.6, 25)
    data = _point(q0, edges, n)
    return calc, data, energy, qe, q0, n_f, edges


@pytest.mark.parametrize("v", [[2.0, -1.0, 0.5], [0.0, 3.0, 0.0], [-4.0, 0.0, 1.0]])
def test_linear_dispersion_gives_the_tilted_erf(v):
    """eps = e0 + v.Q, Gaussian kernel: mu_b = Phi((hi - eps0)/s) - Phi((lo - eps0)/s),
    s^2 = Sigma_EE - 2 v.Sigma_QE + v^T Sigma_QQ v at each bin's energy."""
    spread = GeometrySpread((0.012, 0.02), (0.0, 0.0, 0.0), 0.02, 0.02, 0.0)
    calc, data, energy, qe, q0, n_f, edges = _setup(v, spread)
    mu = ForwardModel(calc, data, qe, b_matrix=np.eye(3), temperature=0).expected()
    v = np.asarray(v, float)
    want = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        S = covariance(ei=EI, omega=0.5 * (lo + hi), n_f=n_f, l1=L1, l2=L2,
                       time_cov=energy.cov, spread=spread)
        s = np.sqrt(S[3, 3] - 2 * v @ S[:3, 3] + v @ S[:3, :3] @ v)
        want.append(ndtr((hi - 3.0) / s) - ndtr((lo - 3.0) / s))
    want = np.array(want)
    assert np.max(np.abs(mu - want)) < 4e-3 * want.max()
    # and the tilt matters: without it the widths differ by more than the tolerance
    s0 = float(energy.sigma(3.0))
    plain = ndtr((edges[1:] - 3.0) / s0) - ndtr((edges[:-1] - 3.0) / s0)
    assert np.max(np.abs(mu - plain)) > 3e-2 * want.max()


def test_flat_dispersion_reduces_to_the_energy_resolution():
    spread = GeometrySpread((0.012, 0.02), (0.0, 0.0, 0.0), 0.02, 0.02, 0.0)
    calc, data, energy, qe, q0, n_f, edges = _setup([0.0, 0.0, 0.0], spread)
    mu = ForwardModel(calc, data, qe, b_matrix=np.eye(3), temperature=0).expected()
    plain = ForwardModel(calc, data, energy, b_matrix=np.eye(3), temperature=0).expected()
    assert np.max(np.abs(mu - plain)) < 6e-3 * plain.max()     # Sobol error, 4096 normals
    assert mu.sum() == pytest.approx(plain.sum(), rel=1e-3)


def test_crystal_position_shifts_the_line_by_the_closed_form():
    """A crystal 5 mm downstream of the axis (psi = 0): the elastic-line shift is
    (2 Ef vf / L2)(x u0 - x n_x / vf) to first order, and -x mirrors it."""
    spread = GeometrySpread((0.0, 0.0), (0.0, 0.0, 0.0), 0.0, 0.0, 0.0)
    calc, data, energy, qe, q0, n_f, edges = _setup([0.0, 0.0, 0.0], spread, n=8)
    x = 0.005
    fm_p = ForwardModel(calc, data, qe, b_matrix=np.eye(3), temperature=0,
                        crystals=[Crystal(position=[x, 0, 0])])
    fm_m = ForwardModel(calc, data, qe, b_matrix=np.eye(3), temperature=0,
                        crystals=[Crystal(position=[-x, 0, 0])])
    fm_0 = ForwardModel(calc, data, qe, b_matrix=np.eye(3), temperature=0)
    e = 0.5 * (edges[:-1] + edges[1:])
    u0 = 1 / (V * np.sqrt(EI))
    vf = V * np.sqrt(EI - e)
    want = 2 * (EI - e) * vf / L2 * (x * u0 - x * n_f[0] / vf)
    s0 = fm_0._kernels[0].shift          # the same normals: the difference is the offset
    sp, sm = fm_p._kernels[0].shift - s0, fm_m._kernels[0].shift - s0
    blk = fm_p._kernels[0].block
    assert sp == pytest.approx(want[blk], rel=3e-3)     # 1.0012-1.0024: second order in x
    assert sm == pytest.approx(-sp, rel=6e-3)


# ------------------------------------------------------------------ geometry
def test_rotation_scan_branches_reach_the_detector():
    U = rotation_z(17.0)
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 3)) * [1.0, 1.0, 0.1]
    e = rng.uniform(-1.0, 4.0, 50)
    scan = RotationScan(U, [(-180, 180)], EI)
    psis, ok = scan.branches(q, e)
    ki = K * np.sqrt(EI) * np.array([1.0, 0, 0])
    kf = K * np.sqrt(EI - e)
    n_ok = 0
    for i in range(50):
        if not ok[i]:
            continue
        for j in range(2):
            qlab = rotation_z(psis[i, j]) @ U @ q[i]
            assert np.linalg.norm(ki - qlab) == pytest.approx(kf[i], rel=1e-10)
            n_ok += 1
    assert n_ok > 20
    # a range holding one branch only keeps that one
    one = RotationScan(U, [(psis[np.argmax(ok), 0] - 0.1, psis[np.argmax(ok), 0] + 0.1)], EI)
    data = _point(q[np.argmax(ok)], [0.0, 1.0], 1)
    R, w, src = one.rotations(data, q[[np.argmax(ok)]], np.array([0]),
                              np.array([e[np.argmax(ok)]]))
    assert len(R) == 1 and w[0] == 1.0


def test_run_goniometer_needs_every_run():
    data = _point([1.0, 0.5, 0.0], [0.0, 1.0], 4)
    with pytest.raises(ValueError):
        RunGoniometer(np.eye(3), {1: 0.0}).rotations(data, np.zeros((4, 3)), np.arange(4),
                                                     np.zeros(4))


def test_intrinsic_width_adds_in_quadrature_to_the_residual():
    """A Gaussian intrinsic width s_i, convolved after the conditioning: for a Gaussian
    residual of width rho sigma0 the bin fraction is the erf with sqrt((rho sigma0)^2 + s_i^2)."""
    energy = _gaussian_spec().resolution(EI)
    qe = QEResolution(energy, RunGoniometer(np.eye(3), {0: 0.0}), GeometrySpread(),
                      intrinsic_sigma=0.05)
    lo = np.linspace(2.0, 4.0, 41)[:-1]
    hi = lo + 0.05
    s0 = float(energy.sigma(3.0))
    for rho in (1.0, 0.5):
        s = np.hypot(rho * s0, 0.05)
        want = ndtr((hi - 3.0 - 0.02) / s) - ndtr((lo - 3.0 - 0.02) / s)
        got = qe.fraction(lo, hi, np.full_like(lo, 3.0), 0.02, rho)
        assert np.max(np.abs(got - want)) < 2e-4 * want.max()
