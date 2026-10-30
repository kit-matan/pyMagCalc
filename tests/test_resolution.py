"""magcalc.resolution: a chopper spectrometer's energy resolution, against closed forms,
exact limits and direct Monte Carlo of the transmitted neutrons.

Oracles (none of them the code's own tabulation). Those built on the first-order relation
dE = alpha tau + beta mu test the tabulation with kinematics="linear"; the default exact
time-of-flight mapping is checked against the ray-tracing Monte Carlo in
test_resolution_raytrace.py.
- the two-chopper Gaussian width in closed form (Windsor's propagation, written out
  below independently of the precision-matrix route the code takes);
- the exact CDFs of Gaussian (x) box and Gaussian (x) trapezoid (iterated integrals of
  Phi), against the FFT-free tabulated profile;
- direct Monte Carlo of (moderator time, chopper times) for a non-Gaussian moderator;
- Mantid PyChop (an independent implementation): given PyChop's own time widths, the
  propagation reproduces its CNCS vanadium widths to rounding, 48 settings x 15 energy
  transfers (`data_pychop_cncs_reference.json`, made by `pychop_cncs_reference.py`);
- exact identities: the phase-1 bin integral, the elastic width ~ Ei^{3/2} at fixed
  openings, the sum over bins, the step limit.
"""
import numpy as np
import pytest
from scipy.special import ndtr

from magcalc.forward import GaussianEnergyResolution, _bin_fraction
from magcalc.neutron_data import NEUTRON_V_PER_SQRT_MEV as V
import json
import os

from magcalc.resolution import (AMATERAS_GEOMETRY, FWHM_PER_SIGMA, ChopperSpectrometer,
                                DiskChopper, Gaussian, GaussianPulse, IkedaCarpenter, Tabulated,
                                TimeWindow, Trapezoid, Uniform, amateras, cncs,
                                fermi_straight_window, read_sns_source_table, wavelength)

HERE = os.path.dirname(__file__)

EI = 7.733
L1, L2, XP, XM = 30.0, 4.0, 7.0, 28.3


def _phi(z):
    return np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)


def _G(z):      # integral of Phi
    return z * ndtr(z) + _phi(z)


def _H(z):      # integral of G
    return 0.5 * (z * z + 1) * ndtr(z) + 0.5 * z * _phi(z)


def _coef(ei, e, l1=L1, l2=L2):
    """Independent restatement: alpha, beta of dE = alpha dt0 + beta du."""
    ef = ei - e
    vi, vf = V * np.sqrt(ei), V * np.sqrt(ef)
    alpha = 2 * ef * vf / l2
    return alpha, 2 * ei * vi + alpha * l1


def _chopper_coefficients(ei, e, xp=XP, xm=XM):
    """dE = cP sP + cM sM for two choppers and a flat moderator."""
    al, be = _coef(ei, e)
    d = xm - xp
    return (al * xm - be) / d, (be - al * xp) / d


# ------------------------------------------------------------------ distributions
def test_ikeda_carpenter_moments_are_the_closed_forms():
    ic = IkedaCarpenter(1 / 20e-6, 1 / 80e-6, 0.4)
    t = np.linspace(*ic.support(), 400001)
    f = ic.pdf(t)
    m = np.trapezoid(t * f, t)
    assert np.trapezoid(f, t) == pytest.approx(1.0, abs=1e-9)
    assert m == pytest.approx(3 * 20e-6 + 0.4 * 80e-6, rel=1e-8)
    assert np.trapezoid((t - m) ** 2 * f, t) == pytest.approx(
        3 * 20e-6 ** 2 + 0.4 * 1.6 * 80e-6 ** 2, rel=1e-7)
    assert ic.mean() == pytest.approx(m, rel=1e-8)
    assert np.all(ic.pdf([-1e-6, -1.0]) == 0)


def test_trapezoid_and_disk_opening():
    tz = Trapezoid(30e-6, 10e-6)
    t = np.linspace(-30e-6, 30e-6, 600001)
    f = tz.pdf(t)
    assert np.trapezoid(f, t) == pytest.approx(1.0, abs=1e-9)
    assert np.trapezoid(t * t * f, t) == pytest.approx(tz.var(), rel=1e-7)
    assert tz.pdf(0.0) == pytest.approx(1 / 30e-6)          # flat top at 1/a
    assert tz.pdf(15e-6) == pytest.approx(0.5 / 30e-6)      # mid-edge at half height
    # a disk: slot 30 mm, beam 10 mm at 0.35 m, 300 Hz: Trapezoid(slot/v, beam/v)
    one = DiskChopper("c", 7.0, 300.0, 0.03, 0.35, 0.01)
    vel = 2 * np.pi * 0.35 * 300
    assert one.window().a == pytest.approx(0.03 / vel)
    assert one.window().b == pytest.approx(0.01 / vel)
    by_angle = DiskChopper.from_slot_angle("c", 7.0, 300.0, np.degrees(0.03 / 0.35), 0.35, 0.01)
    assert by_angle.window().var() == pytest.approx(one.window().var(), rel=1e-12)


def test_counter_rotating_pair_halves_the_burst_only_for_a_full_beam():
    """A pair's slots overlap: with the beam filling the slot the opening is a triangle
    of FWHM w/(2v) (half a single disk's), with a narrow beam a box of w/v (no gain)."""
    v = 2 * np.pi * 0.25 * 300
    w = 0.03
    full = DiskChopper("p", 30.0, 300.0, w, 0.25, w, counter_rotating=True).window()
    assert full.var() == pytest.approx((w / (2 * v)) ** 2 / 6, rel=1e-6)
    assert full.pdf(0.0) == pytest.approx(2 * v / w, rel=1e-6)
    narrow = DiskChopper("p", 30.0, 300.0, w, 0.25, 0.0, counter_rotating=True).window()
    assert narrow.var() == pytest.approx((w / v) ** 2 / 12, rel=1e-12)
    single = DiskChopper("s", 30.0, 300.0, w, 0.25, w).window()
    assert full.var() == pytest.approx(single.var() / 4, rel=1e-6)


@pytest.mark.parametrize("w1,w2,b", [(0.044, 0.044, 0.012), (0.022, 0.044, 0.021),
                                     (0.010, 0.010, 0.025)])
def test_counter_rotating_pair_against_sampled_geometry(w1, w2, b):
    """Direct sampling of (y, t): a neutron passes if it is inside both moving slots."""
    from magcalc.resolution import CounterRotatingOpening
    v = 471.0
    d = CounterRotatingOpening(w1, w2, b, v)
    rng = np.random.default_rng(3)
    n = 2_000_000
    h = 1.1 * (w1 + w2) / (4 * v)
    y, t = rng.uniform(-b / 2, b / 2, n), rng.uniform(-h, h, n)
    tt = t[(np.abs(y - v * t) < w1 / 2) & (np.abs(y + v * t) < w2 / 2)]
    assert tt.var() == pytest.approx(d.var(), rel=5e-3)
    x = np.linspace(-h, h, 21)
    ecdf = np.searchsorted(np.sort(tt), x) / len(tt)
    cdf = np.array([np.trapezoid(d.pdf(np.linspace(-h, xx, 4001)), np.linspace(-h, xx, 4001))
                    for xx in x])
    assert np.max(np.abs(ecdf - cdf)) < 3e-3


def test_tabulated_is_normalised():
    t = np.linspace(0, 1e-4, 1001)
    d = Tabulated(t, np.sin(np.pi * t / 1e-4))
    assert np.trapezoid(d.pdf(t), t) == pytest.approx(1.0, rel=1e-12)
    assert d.mean() == pytest.approx(0.5e-4, rel=1e-9)


# ------------------------------------------------------------------ exact limits
def test_gaussian_fraction_is_the_phase1_bin_integral():
    """T1: the protocol method is the phase-1 arithmetic, bit for bit."""
    r = GaussianEnergyResolution(0.209, (1.0, -1.04, 0.38), "E/Ei", EI, extra_sigma=0.02)
    lo = np.linspace(-1, 6, 141)[:, None]
    c = np.array([[0.0, 0.7, 3.4, 5.5]])
    assert np.array_equal(r.fraction(lo, lo + 0.05, c),
                          _bin_fraction(lo, lo + 0.05, c, r.sigma(c)))


def test_two_gaussian_choppers_give_the_closed_form_width():
    """T2: every part Gaussian -> the exact erf with Windsor's sigma(E)."""
    sp, sm = 9e-6, 6e-6
    spec = ChopperSpectrometer("g", L1, L2, [TimeWindow("P", XP, Gaussian(sp)),
                                             TimeWindow("M", XM, Gaussian(sm))],
                               extra_sigma=0.01)
    r = spec.resolution(EI)
    assert r.gaussian
    e = np.array([-3.0, 0.0, 1.0, 3.4, 6.0])
    cp, cm = _chopper_coefficients(EI, e)
    want = np.sqrt(cp ** 2 * sp ** 2 + cm ** 2 * sm ** 2 + 0.01 ** 2)
    assert r.sigma(e) == pytest.approx(want, rel=1e-12)
    lo = np.linspace(-0.5, 7, 76)[:, None]
    assert r.fraction(lo, lo + 0.1, e[None, :]) == pytest.approx(
        _bin_fraction(lo, lo + 0.1, e[None, :], want[None, :]), abs=1e-14)


def test_gaussian_moderator_and_one_chopper_closed_form():
    """A Fermi-like geometry: Gaussian pulse sigma_m, one chopper at x, Gaussian opening.
    tau ~ N(0, sm^2), s ~ N(0, sc^2) independent, mu = (s - tau)/x."""
    sm, sc, x = 30e-6, 5e-6, 20.0
    spec = ChopperSpectrometer("f", 22.0, 4.0, [TimeWindow("F", x, Gaussian(sc))],
                               moderator=Gaussian(sm))
    r = spec.resolution(15.0)
    e = np.array([0.0, 5.0, 10.0])
    al, be = _coef(15.0, e, 22.0, 4.0)
    want = np.sqrt((al - be / x) ** 2 * sm ** 2 + (be / x) ** 2 * sc ** 2)
    assert r.sigma(e) == pytest.approx(want, rel=1e-12)


def test_table_reproduces_the_erf_when_forced():
    """The tabulation, run on an all-Gaussian instrument, reproduces the exact erf."""
    spec = ChopperSpectrometer("g", L1, L2, [TimeWindow("P", XP, Gaussian(9e-6)),
                                             TimeWindow("M", XM, Gaussian(6e-6))],
                               moderator=Gaussian(40e-6))
    exact = spec.resolution(EI, kinematics="linear")
    table = spec.resolution(EI, force_table=True, kinematics="linear")
    assert not table.gaussian
    assert table.cov == pytest.approx(exact.cov, rel=1e-6)
    x = np.linspace(-0.4, 0.4, 33)
    for e in (0.0, 1.234, 3.4, -2.0):
        assert np.max(np.abs(table.cdf(x, e) - exact.cdf(x, e))) < 2e-5


def test_vanishing_width_is_the_step():
    """T3: widths -> 0 give the step, a centre on an edge counting one half."""
    spec = ChopperSpectrometer("g", L1, L2, [TimeWindow("P", XP, Gaussian(1e-15)),
                                             TimeWindow("M", XM, Gaussian(1e-15))])
    r = spec.resolution(EI)
    lo, hi = np.array([0.0, 1.0, 1.0, 2.0]), np.array([1.0, 2.0, 1.5, 3.0])
    c = np.array([0.5, 1.0, 2.5, 2.0])
    assert r.fraction(lo, hi, c) == pytest.approx([1.0, 0.5, 0.0, 0.5], abs=1e-12)


# ------------------------------------------------------------------ non-Gaussian shapes
def _box_gauss_cdf(x, w, s):
    return s / w * (_G((x + w / 2) / s) - _G((x - w / 2) / s))


def _trap_gauss_cdf(x, a, b, s):
    return s * s / (a * b) * (_H((x + a / 2 + b / 2) / s) - _H((x + a / 2 - b / 2) / s)
                              - _H((x - a / 2 + b / 2) / s) + _H((x - a / 2 - b / 2) / s))


@pytest.mark.parametrize("e", [0.0, 2.0, 3.4])
def test_gaussian_and_box_choppers_give_the_closed_form_profile(e):
    """T4: Gaussian (x) box, the exact CDF (sigma/w)[G((x+w/2)/sigma) - G((x-w/2)/sigma)]."""
    sp, wm = 6e-6, 30e-6
    spec = ChopperSpectrometer("gb", L1, L2, [TimeWindow("P", XP, Gaussian(sp)),
                                              TimeWindow("M", XM, Uniform(wm))])
    r = spec.resolution(EI, kinematics="linear")
    assert not r.gaussian
    cp, cm = _chopper_coefficients(EI, e)
    s, w = abs(cp) * sp, abs(cm) * wm
    x = np.linspace(-1.5, 1.5, 61) * (w / 2 + 3 * s)
    assert np.max(np.abs(r.cdf(x, e) - _box_gauss_cdf(x, w, s))) < 2e-5
    assert r.sigma(e) == pytest.approx(np.sqrt(s * s + w * w / 12), rel=1e-4)


@pytest.mark.parametrize("e", [0.0, 3.4])
def test_trapezoid_with_extra_gaussian_gives_the_closed_form_profile(e):
    """A disk opening (trapezoid a, b) and a Gaussian chopper, plus an extra Gaussian
    width in meV: Gaussian (x) Uniform (x) Uniform, through the second integral of Phi."""
    sp, ta, tb, ext = 4e-6, 30e-6, 12e-6, 0.012
    spec = ChopperSpectrometer("gt", L1, L2, [TimeWindow("P", XP, Gaussian(sp)),
                                              TimeWindow("M", XM, Trapezoid(ta, tb))],
                               extra_sigma=ext)
    r = spec.resolution(EI, kinematics="linear")
    cp, cm = _chopper_coefficients(EI, e)
    s = np.hypot(abs(cp) * sp, ext)
    a, b = abs(cm) * ta, abs(cm) * tb
    x = np.linspace(-1.5, 1.5, 61) * ((a + b) / 2 + 3 * s)
    assert np.max(np.abs(r.cdf(x, e) - _trap_gauss_cdf(x, a, b, s))) < 2e-5


def test_detector_depth_is_a_box_in_energy():
    """A uniform depth d in L2 adds a box of width 2 Ef d / L2: with Gaussian choppers the
    profile is Gaussian (x) box in closed form."""
    sp, sm, d = 6e-6, 4e-6, 0.025
    spec = ChopperSpectrometer("gd", L1, L2, [TimeWindow("P", XP, Gaussian(sp)),
                                              TimeWindow("M", XM, Gaussian(sm))],
                               detector_depth=d)
    r = spec.resolution(EI, kinematics="linear")
    assert not r.gaussian
    for e in (0.0, 3.4):
        cp, cm = _chopper_coefficients(EI, e)
        s = np.hypot(cp * sp, cm * sm)
        w = 2 * (EI - e) * d / L2
        x = np.linspace(-1.2, 1.2, 49) * (w / 2 + 3 * s)
        assert np.max(np.abs(r.cdf(x, e) - _box_gauss_cdf(x, w, s))) < 2e-5


def test_moderator_tail_against_direct_monte_carlo():
    """An Ikeda-Carpenter pulse through two disk choppers: the transmitted density is
    M(tau) W_P(sP) W_M(sM); drawing (sP, sM) from the openings and weighting by M(tau)
    gives the energy distribution directly, without the tabulation."""
    ic = IkedaCarpenter(1 / 25e-6, 1 / 120e-6, 0.5)
    pulse = IkedaCarpenter(ic.alpha, ic.beta, ic.R, shift=-ic.peak())   # phased at the peak
    P = DiskChopper("P", XP, 150.0, 0.03, 0.35, 0.03, counter_rotating=True)
    M = DiskChopper("M", XM, 300.0, 0.02, 0.35, 0.01, counter_rotating=True)
    spec = ChopperSpectrometer("mc", L1, L2, [P, M], moderator=pulse)
    r = spec.resolution(EI, kinematics="linear")
    rng = np.random.default_rng(7)
    n = 400_000

    def draw(dist, size):      # inverse CDF of the opening on a fine grid
        t = np.linspace(*dist.support(), 200001)
        c = np.cumsum(dist.pdf(t))
        return np.interp(rng.uniform(0, c[-1], size), c, t)
    sp, sm = draw(P.window(), n), draw(M.window(), n)
    d = XM - XP
    tau, mu = (XM * sp - XP * sm) / d, (sm - sp) / d
    w = pulse.pdf(tau)
    w /= w.sum()
    for e in (0.0, 3.4):
        al, be = _coef(EI, e)
        de = al * tau + be * mu
        de -= w @ de
        assert np.sqrt(w @ de ** 2) == pytest.approx(float(r.sigma(e)), rel=5e-3)
        x = np.linspace(-2.5, 2.5, 41) * float(r.sigma(e))
        mc = np.array([w[de <= xx].sum() for xx in x])
        assert np.max(np.abs(r.cdf(x, e) - mc)) < 4e-3
    # the tail is real: the profile is skewed, unlike any Gaussian
    x0 = 2.0 * float(r.sigma(0.0))
    assert abs((1 - r.cdf(x0, 0.0)) - r.cdf(-x0, 0.0)) > 1e-3


# ------------------------------------------------------------------ identities
def test_elastic_width_scales_as_ei_to_three_halves():
    """At fixed openings and a flat pulse, dE(0) = (2 Ei v_i / L2) dt_D with dt_D
    independent of Ei: every width scales exactly as Ei^{3/2} (a multi-Ei identity)."""
    P = DiskChopper("P", XP, 300.0, 0.03, 0.35, 0.03, counter_rotating=True)
    M = DiskChopper("M", XM, 300.0, 0.03, 0.35, 0.01, counter_rotating=True)
    spec = ChopperSpectrometer("multi", L1, L2, [P, M])
    eis = np.array([15.143, 7.734, 4.682, 3.135])
    s = np.array([float(spec.resolution(ei, kinematics="linear").sigma(0.0)) for ei in eis])
    assert s / eis ** 1.5 == pytest.approx(np.full(4, s[0] / eis[0] ** 1.5), rel=1e-12)
    f = np.array([spec.resolution(ei, kinematics="linear").fwhm(0.0) for ei in eis])
    assert f / eis ** 1.5 == pytest.approx(np.full(4, f[0] / eis[0] ** 1.5), rel=2e-3)


def test_fractions_over_covering_bins_sum_to_one():
    """T14: resolution only redistributes a mode's weight."""
    ic = IkedaCarpenter(1 / 25e-6, 1 / 120e-6, 0.5)
    spec = ChopperSpectrometer("s", L1, L2, [
        DiskChopper("P", XP, 150.0, 0.03, 0.35, 0.03, True),
        DiskChopper("M", XM, 300.0, 0.02, 0.35, 0.01, True)],
        moderator=IkedaCarpenter(ic.alpha, ic.beta, ic.R, shift=-ic.peak()),
        detector_depth=0.02, extra_sigma=0.005)
    r = spec.resolution(EI)
    edges = np.linspace(-3, 8, 221)
    for c in (0.0, 0.73, 3.4, 6.1):
        fr = r.fraction(edges[:-1], edges[1:], c)
        assert np.all(fr >= -1e-15)
        assert fr.sum() == pytest.approx(1.0, abs=1e-12)


def test_between_nodes_matches_a_profile_built_there():
    spec = ChopperSpectrometer("s", L1, L2, [
        DiskChopper("P", XP, 150.0, 0.03, 0.35, 0.03, True),
        DiskChopper("M", XM, 300.0, 0.02, 0.35, 0.01, True)])
    r = spec.resolution(EI)
    e = 0.5 * (r.e_nodes[40] + r.e_nodes[41])
    x, c, sg = r._profile_at(e, 4000)
    xs = np.linspace(-2.5, 2.5, 51) * sg
    assert np.max(np.abs(r.cdf(xs, e) - np.interp(xs, x, c))) < 3e-4


def test_refuses_what_it_cannot_model():
    with pytest.raises(ValueError):
        ChopperSpectrometer("x", L1, L2, [TimeWindow("F", 20.0, Gaussian(5e-6))])
    with pytest.raises(ValueError):
        ChopperSpectrometer("x", L1, L2, [TimeWindow("F", 40.0, Gaussian(5e-6)),
                                          TimeWindow("G", 20.0, Gaussian(5e-6))])
    spec = ChopperSpectrometer("x", L1, L2, [TimeWindow("P", XP, Gaussian(5e-6)),
                                             TimeWindow("M", XM, Gaussian(5e-6))])
    with pytest.raises(ValueError):
        spec.resolution(EI, e_range=(-1.0, EI))
    with pytest.raises(ValueError):
        DiskChopper("off", 7.0, 0.0, 0.03, 0.35).window()


# ------------------------------------------------------------------ PyChop (Mantid)
def _pychop():
    with open(os.path.join(HERE, "data_pychop_cncs_reference.json")) as f:
        return json.load(f)


def test_propagation_reproduces_pychop_from_its_own_time_widths():
    """T5, exact: PyChop's moderator/first-chopper width, final-chopper width and detector
    term, put through this code's propagation as Gaussians, give PyChop's own vanadium
    FWHM (its sample term removed, which Level 0 does not model) to rounding."""
    ref = _pychop()
    worst = 0.0
    for c in ref["cases"]:
        x0, _, x1, x2, xm = c["distances"]
        tsqmod, tsqchp, _ = c["tsqmod_tsqchp_x0"]
        et = np.array(c["etrans"])
        comp = c["components_s2"]
        v = np.array(c["v_van_s2"]) - np.array(comp.get("sample", np.zeros(len(et))))
        want = 2 * V * np.sqrt((c["ei"] - et) ** 3 * v) / x2
        for i, e in enumerate(et):
            vf = V * np.sqrt(c["ei"] - e)
            depth = np.sqrt(12 * comp["detector"][i]) * vf / FWHM_PER_SIGMA
            spec = ChopperSpectrometer("pychop", x0 + x1, x2, [
                TimeWindow("first", xm, Gaussian(np.sqrt(tsqmod) / FWHM_PER_SIGMA)),
                TimeWindow("final", x0, Gaussian(np.sqrt(tsqchp) / FWHM_PER_SIGMA))],
                detector_depth=depth)
            got = FWHM_PER_SIGMA * float(spec.sigma(c["ei"], e))
            worst = max(worst, abs(got / want[i] - 1))
    assert worst < 1e-9


def test_cncs_is_close_to_pychop():
    """The CNCS model's own geometry (the counter-rotating chopper 4 with its 9, 4.4 and 2
    deg slots, a Fermi chopper 1, the measured pulse widths) against PyChop, whose slot
    widths are effective inputs to its Gaussian formula rather than the slots: within 13 %
    for every variant, Ei and frequency pair at chopper 1 = 60 Hz, |E| <= Ei/2."""
    ref = _pychop()
    n = 0
    for c in ref["cases"]:
        if c["f1"] != 60.0:
            continue
        r = cncs(c["variant"], c["f_res"], c["f1"]).resolution(c["ei"])
        for e, w in zip(c["etrans"], c["fwhm"]):
            if abs(e) <= 0.5 * c["ei"]:
                assert r.fwhm(e) == pytest.approx(w, rel=0.13), (c["variant"], c["ei"], e)
                n += 1
    assert n > 200


def test_cncs_wiring():
    spec = cncs("Intermediate", 300.0, 60.0)
    c1, c4 = spec.choppers
    assert spec.l1 == pytest.approx(36.265) and spec.l2 == 3.5
    w4 = c4.window()
    assert (w4.w1, w4.w2, w4.b) == pytest.approx((0.25 * np.radians(4.4),) * 2 + (0.021,))
    assert w4.v == pytest.approx(2 * np.pi * 0.25 * 300.0)
    assert c1.window().a == pytest.approx(((0.05 - 31 * 0.00015) / 30) / (2 * np.pi * 60 * 0.017))
    pulse = spec.moderator_pulse(12.0)       # PyChop's measured FWHM at 2.611 A
    assert FWHM_PER_SIGMA * pulse.sigma == pytest.approx(
        1e-6 * np.interp(wavelength(12.0), [2.42638069, 2.60358156, 2.79372357],
                         [103.81, 112.68, 121.38]), rel=1e-12)
    with pytest.raises(ValueError):
        cncs("Medium")


def test_sns_source_table(tmp_path):
    """A two-band file in the SNS/McStas format: section 2 is read, bands are interpolated
    in log E, and t = 0 sits at the requested phase."""
    t_us = np.linspace(0.0, 400.0, 81)
    lines = ["## header", "## section 2 key", "## time energy brightness error"]
    for e_ev, tau in ((1e-3, 40.0), (1e-2, 20.0)):
        for t in t_us:
            lines.append(f"  {t:.4e}  {e_ev:.4e}  {t * np.exp(-t / tau):.6e}  0.0")
    lines.append("## section2end")
    p = tmp_path / "src.dat"
    p.write_text("\n".join(lines) + "\n")
    src = read_sns_source_table(p)
    lo = src(1.0)                            # exactly the first band: t e^{-t/40}
    assert lo.t[np.argmax(lo.density)] == pytest.approx(0.0)          # phased at the peak
    assert lo.mean() == pytest.approx(40e-6, rel=2e-2)                # mean 2 tau - peak tau
    mid = read_sns_source_table(p, phase="mean")(np.sqrt(10.0))       # halfway in log E
    assert mid.mean() == pytest.approx(0.0, abs=1e-12)
    # brightness is mixed as given, so the bands weigh as their areas tau^2: 0.8 and 0.2
    w1, w2 = 0.8, 0.2
    assert np.sqrt(mid.var()) == pytest.approx(
        np.sqrt(w1 * 2 * 40e-6 ** 2 + w2 * 2 * 20e-6 ** 2 + w1 * w2 * (80e-6 - 40e-6) ** 2),
        rel=2e-2)
    t0 = read_sns_source_table(p, phase=lambda ei: 1e-4)(1.0)
    assert t0.support()[0] == pytest.approx(-1e-4)


def test_gaussian_pulse_and_fermi_window():
    g = GaussianPulse([1.0, 3.0], [10e-6, 30e-6])
    assert FWHM_PER_SIGMA * g(81.8042 / 4.0).sigma == pytest.approx(20e-6)
    w = fermi_straight_window(60.0, 0.0015, 0.017)
    assert w.a == w.b == pytest.approx(0.0015 / (2 * np.pi * 60 * 0.017))


# ------------------------------------------------------------------ AMATERAS (Nirei et al. 2024)
def _w_eff(w, f, ei, d=0.0040, r=0.300):
    return w - 2 * np.pi * r * f * d / (V * np.sqrt(ei))


def test_disk_thickness_is_the_papers_effective_slit():
    """Nirei et al., J. Neutron Res. (2024), Eqs. (2)-(3): a counter-rotating pair whose beam
    fills the slot opens as a triangle of FWHM w_eff / (2 * 2 pi R f), with w_eff = w -
    2 pi R f d / v_n. Its stated w_eff / w = 0.634 (10 mm slit, 2.0 meV, 300 Hz) gives
    d = 4.0 mm; that d reproduces its pulse widths 74 us and 229 us at 5.9 meV (Eq. 4)."""
    for slit, ei in ((0.030, 7.7), (0.010, 2.0), (0.010, 15.0)):
        pair = DiskChopper("#2", 28.419, 300.0, slit, 0.300, 0.030, True, thickness=0.0040)
        w = pair.window(ei)
        tri = _w_eff(slit, 300.0, ei) / (2 * 2 * np.pi * 0.300 * 300.0)
        assert w.var() == pytest.approx(tri ** 2 / 6, rel=1e-5)        # a triangle of FWHM tri
    assert _w_eff(0.010, 300.0, 2.0) / 0.010 == pytest.approx(0.634, abs=5e-4)
    dch = _w_eff(0.030, 300.0, 5.9) / (2 * 2 * np.pi * 0.3 * 300.0)
    for lps, f, want in ((7.1, 300.0, 74e-6), (14.2, 150.0, 229e-6)):
        dchps = _w_eff(0.030, f, 5.9) / (2 * np.pi * 0.3 * f)
        assert dchps + lps / (30.0 - 1.581 - lps) * (dchps + dch) == pytest.approx(want, abs=1e-6)
    with pytest.raises(ValueError):
        DiskChopper("x", 10.0, 300.0, 0.03, 0.3, 0.03, thickness=0.004).window()


def test_papers_equation_1_is_the_exact_propagation_without_a_pulse_shaper():
    """Eq. (1) of Nirei et al. with only #2 and the moderator: the first-order elastic width
    2 Ei sqrt[(dt_ch/t_ch (1 + L1/L2))^2 + (dt_m/t_ch (1 + L3/L2))^2 + (dL2/L2)^2] equals this
    code's propagation of the same Gaussian widths exactly."""
    L1_, L2_, L3_ = 30.0, 4.0, 1.581
    for ei, dch, dtm, dl2 in ((7.7, 25e-6, 110e-6, 0.013), (3.1, 30e-6, 170e-6, 0.01)):
        t_ch = (L1_ - L3_) / (V * np.sqrt(ei))
        eq1 = 2 * ei * np.sqrt((dch / t_ch * (1 + L1_ / L2_)) ** 2
                               + (dtm / t_ch * (1 + L3_ / L2_)) ** 2 + (dl2 / L2_) ** 2)
        spec = ChopperSpectrometer("eq1", L1_, L2_, [
            TimeWindow("#2", L1_ - L3_, Gaussian(dch / FWHM_PER_SIGMA))],
            moderator=Gaussian(dtm / FWHM_PER_SIGMA),
            extra_sigma=2 * ei * dl2 / L2_ / FWHM_PER_SIGMA)
        assert FWHM_PER_SIGMA * float(spec.sigma(ei, 0.0)) == pytest.approx(eq1, rel=1e-6)


def test_amateras_wiring():
    spec = amateras(f2=300.0, slit2=0.030, f3=300.0)
    names = [c.name for c in spec.choppers]
    assert names == ["#3", "#2"] and spec.l1 == 30.0 and spec.l2 == 4.0
    c3, c2 = spec.choppers
    assert c2.distance == pytest.approx(30.0 - 1.581) and c2.counter_rotating
    assert not c3.counter_rotating and c3.thickness == AMATERAS_GEOMETRY["thickness"]
    assert FWHM_PER_SIGMA * spec.moderator_pulse(5.9).sigma == pytest.approx(124e-6)
    assert [c.name for c in amateras(f1=300.0, f3=150.0).choppers] == ["#1", "#2"]
    r = spec.resolution(7.734)
    assert 0.1 < r.fwhm(0.0) < 0.3
