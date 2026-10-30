"""magcalc.resolution against an independent ray-tracing Monte Carlo (tests/raytrace_tof.py).

The ray trace shares only the instrument description. Each neutron gets an emission time
from the pulse and a speed from a flat band. It passes each chopper by that chopper's
geometry: a moving slot, two counter-rotating slots, or a straight Fermi channel at its
rotor angle. It travels with exact kinematics and is reduced with the nominal Ei and time
at the sample. The tabulated profile must reproduce the distribution of its
recorded-minus-true energy.

This checks two things no other test reaches: the time model (which chopper and pulse
times reach the detector) and the exact time-of-flight mapping. The first-order mapping
(`kinematics="linear"`) is shown to fail, as it should, when a long pulse tail passes
unshaped.
"""
import os
import sys

import numpy as np
import pytest

from magcalc.resolution import (CNCS_GEOMETRY, ChopperSpectrometer, DiskChopper,
                                IkedaCarpenter, TimeWindow, fermi_straight_window)

sys.path.insert(0, os.path.dirname(__file__))
from raytrace_tof import sampler, trace  # noqa: E402

G = CNCS_GEOMETRY
W4 = 0.25 * np.radians(4.4)


def _pulse(k=1.0):
    ic = IkedaCarpenter(1 / (30e-6 * k), 1 / (150e-6 * k), 0.5)
    return IkedaCarpenter(ic.alpha, ic.beta, ic.R, shift=-ic.peak())


def _profile_std(r, e):
    x, c, _ = r._profile_at(float(e), 4000)
    p = np.diff(c, prepend=0.0)
    m = p @ x
    return np.sqrt(p @ (x - m) ** 2)


def _compare(r, de, e):
    x = np.linspace(-3, 3, 61) * float(r.sigma(e))
    ecdf = np.searchsorted(np.sort(de), x) / len(de)
    return np.max(np.abs(ecdf - r.cdf(x, e))), de.std() / _profile_std(r, e)


def _cncs_like(pulse, depth=0.025):
    spec = ChopperSpectrometer(
        "CNCS-like", G["x_chopper4"] + G["chop_sam"], G["sam_det"],
        [TimeWindow("F", G["x_chopper1"],
                    fermi_straight_window(60.0, G["fermi_channel"], G["fermi_length"])),
         DiskChopper("4", G["x_chopper4"], 300.0, W4, 0.25, 0.021, counter_rotating=True)],
        moderator=pulse, detector_depth=depth)
    rays = [dict(kind="fermi", x=G["x_chopper1"], f=60.0, w=G["fermi_channel"],
                 L=G["fermi_length"]),
            dict(kind="pair", x=G["x_chopper4"], f=300.0, w=W4, r=0.25, b=0.021)]
    return spec, rays


def test_two_disks_against_the_ray_trace():
    """Fast: a pulse shaper and a monochromating disk, flat pulse, Ei 7.7, E = 2."""
    spec = ChopperSpectrometer("two disks", 30.0, 4.0, [
        DiskChopper("P", 14.2, 300.0, 0.03, 0.30, 0.01),
        DiskChopper("M", 28.4, 300.0, 0.03, 0.30, 0.01)])
    rays = [dict(kind="disk", x=14.2, f=300.0, w=0.03, r=0.30, b=0.01),
            dict(kind="disk", x=28.4, f=300.0, w=0.03, r=0.30, b=0.01)]
    de = trace(np.random.default_rng(1), ei=7.7, l1=30.0, l2=4.0, choppers=rays,
               pulse=lambda rng, n: rng.uniform(-3e-3, 3e-3, n), omega=2.0, n_want=100_000)
    dcdf, ratio = _compare(spec.resolution(7.7), de, 2.0)
    assert dcdf < 8e-3
    assert ratio == pytest.approx(1.0, abs=0.01)


@pytest.mark.slow
@pytest.mark.parametrize("ei,omega", [(12.0, 0.0), (12.0, 6.0), (6.25, -3.0), (3.3, 1.5)])
def test_cncs_like_against_the_ray_trace(ei, omega):
    """A Fermi chopper 1 at 60 Hz, a counter-rotating chopper 4, an Ikeda-Carpenter pulse
    phased at its peak, 25 mm detector depth."""
    pulse = _pulse()
    spec, rays = _cncs_like(pulse)
    de = trace(np.random.default_rng(int(ei * 10 + omega)), ei=ei, l1=spec.l1, l2=spec.l2,
               choppers=rays, pulse=sampler(pulse), omega=omega, depth=0.025,
               n_want=400_000, batch=4_000_000)
    dcdf, ratio = _compare(spec.resolution(ei), de, omega)
    assert dcdf < 3e-3                          # measured 0.8e-3 .. 1.5e-3
    assert ratio == pytest.approx(1.0, abs=3e-3)   # measured 0.998 .. 1.001


@pytest.mark.slow
def test_long_unshaped_tail_needs_exact_kinematics():
    """A counter-rotating pair alone with a long pulse tail: the transmitted speeds spread
    by several %, so the first-order mapping is off by ~4 % in width; the exact mapping
    is not."""
    pulse = _pulse()
    spec = ChopperSpectrometer("pair", 36.265, 3.5, [
        DiskChopper("4", G["x_chopper4"], 300.0, W4, 0.25, 0.021, counter_rotating=True)],
        moderator=pulse)
    rays = [dict(kind="pair", x=G["x_chopper4"], f=300.0, w=W4, r=0.25, b=0.021)]
    de = trace(np.random.default_rng(4), ei=12.0, l1=36.265, l2=3.5, choppers=rays,
               pulse=sampler(pulse), omega=0.0, n_want=400_000, batch=4_000_000)
    dcdf, ratio = _compare(spec.resolution(12.0), de, 0.0)
    assert dcdf < 3e-3 and ratio == pytest.approx(1.0, abs=3e-3)   # 1.4e-3, 1.0009
    dcdf_lin, ratio_lin = _compare(spec.resolution(12.0, kinematics="linear"), de, 0.0)
    assert ratio_lin > 1.03                                          # 1.045
