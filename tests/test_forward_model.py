"""magcalc.forward: the expected count per bin, against exact identities and a closed form.

The closed-form oracle is a ferromagnetic S = 1 chain in a field,
    omega(h) = 2 mu_B B + 2 |J| S (1 - cos 2 pi h),   I(q) = (S/2)(1 + qz^2/|q|^2),
with the moments along -z (antiparallel to the field: the minimum). Each bin's mean
signal is computed independently by SciPy quadrature over the Q box (`quad` over h,
`dblquad` for the polarization factor over (k, l)) with the energy integral as an erf
difference -- including the Bose factor and the energy-gain line -- and the Sobol
estimate must converge to it as the number of samples grows.
"""
import copy
import logging
import os
import sys

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr

import magcalc as mc
from magcalc.constants import MU_B
from magcalc.forward import (Crystal, ForwardModel, GaussianEnergyResolution,
                             crystals_from_domains, poisson_deviance)
from magcalc.generic_model import GenericSpinModel
from magcalc.neutron_data import BinnedData, QFrame, read_count_cells, read_mdnorm_npz
from magcalc.numerical import thermal_bose_prefactor

sys.path.insert(0, os.path.dirname(__file__))
from test_neutron_data import write_cells, write_mdnorm  # noqa: E402

logging.disable(logging.CRITICAL)

A, B_LAT, C_LAT = 3.0, 10.0, 10.0
S, J, FIELD = 1.0, -1.0, 2.0
GAP = 2 * MU_B * FIELD


@pytest.fixture(scope="module")
def chain():
    cfg = {"crystal_structure": {
               "lattice_vectors": [[A, 0, 0], [0, B_LAT, 0], [0, 0, C_LAT]],
               "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S}]},
           "interactions": {"symmetry_rules": [
               {"type": "heisenberg", "distance": A, "value": "J"}]},
           "parameters": {"J": J, "H_mag": FIELD, "H_dir": [0, 0, 1]},
           "parameter_order": ["J", "H_mag", "H_dir"],
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, -1]}}
    m = GenericSpinModel(copy.deepcopy(cfg))
    th, ph = m.generate_magnetic_structure()
    m.set_magnetic_structure(th, ph)
    calc = mc.MagCalc(spin_model_module=m, spin_magnitude=S, cache_mode="none",
                      cache_file_base="fwd_chain", hamiltonian_params=[J, FIELD, [0, 0, 1]])
    e0, e1 = calc.relax_from_current()
    assert e1 >= e0 - 1e-12
    return calc


def _omega(h):
    return GAP + 2 * abs(J) * S * (1 - np.cos(2 * np.pi * h))


def _point_data(hkl, e_edges, exposure=1.0):
    """One Q cell holding one sample (hkl), binned in energy by e_edges."""
    e_edges = np.asarray(e_edges, float)
    nb = len(e_edges) - 1
    return BinnedData(np.zeros(nb), np.full(nb, exposure), e_edges[:-1], e_edges[1:],
                      samples_hkl=np.atleast_2d(hkl), cell_ptr=[0, 1],
                      bin_ptr=np.arange(nb + 1), bin_cells=np.zeros(nb, int),
                      bin_cell_w=np.ones(nb))


RES = GaussianEnergyResolution(fwhm0=0.4, poly=(1.0, -0.5), variable="E/Ei", ei=10.0,
                               extra_sigma=0.05)


# ------------------------------------------------------------------ exact identities
def test_energy_sum_rule_at_zero_temperature(chain, tmp_path):
    """Bins spanning every mode: sum over E of mu_b / X_b = mean over samples of sum_m I_m."""
    p = tmp_path / "m.npz"
    write_mdnorm(p, dims=["[H,0,0]|0.1|0.4|3", "[0,K,0]|-0.1|0.1|1",
                          "[0,0,L]|0.2|0.4|1", "DeltaE|-10|20|60"],
                 shape=(3, 1, 1, 60), empty_cell=False)
    d = read_mdnorm_npz(p, "Z", samples_per_cell=32)
    mu = ForwardModel(chain, d, RES, temperature=0).expected()
    q = d.sample_cartesian(ForwardModel(chain, d, RES).b_matrix)
    _, I = chain.sqw_batch(q)
    for c in range(d.n_cells):
        bins = np.flatnonzero(d.bin_cells == c)
        Ic = I[d.cell_ptr[c]:d.cell_ptr[c + 1]].sum(axis=1).mean()
        assert (mu[bins] / d.exposure[bins]).sum() == pytest.approx(Ic, rel=1e-12)



def test_energy_sum_rule_with_an_instrument_line_shape(chain, tmp_path):
    """The same sum rule with a chopper spectrometer's non-Gaussian profile (trapezoid
    openings, Ikeda-Carpenter tail, detector depth): resolution only redistributes."""
    from magcalc.resolution import ChopperSpectrometer, DiskChopper, IkedaCarpenter
    ic = IkedaCarpenter(1 / 25e-6, 1 / 120e-6, 0.5)
    spec = ChopperSpectrometer("toy", 30.0, 4.0, [
        DiskChopper("P", 7.0, 150.0, 0.03, 0.35, 0.03, True),
        DiskChopper("M", 28.3, 300.0, 0.02, 0.35, 0.01, True)],
        moderator=IkedaCarpenter(ic.alpha, ic.beta, ic.R, shift=-ic.peak()),
        detector_depth=0.02)
    res = spec.resolution(25.0)
    assert not res.gaussian
    p = tmp_path / "m.npz"
    write_mdnorm(p, dims=["[H,0,0]|0.1|0.4|3", "[0,K,0]|-0.1|0.1|1",
                          "[0,0,L]|0.2|0.4|1", "DeltaE|-10|20|60"],
                 shape=(3, 1, 1, 60), empty_cell=False)
    d = read_mdnorm_npz(p, "Z", samples_per_cell=32)
    fm = ForwardModel(chain, d, res, temperature=0)
    mu = fm.expected()
    _, I = chain.sqw_batch(d.sample_cartesian(fm.b_matrix))
    for c in range(d.n_cells):
        bins = np.flatnonzero(d.bin_cells == c)
        Ic = I[d.cell_ptr[c]:d.cell_ptr[c + 1]].sum(axis=1).mean()
        assert (mu[bins] / d.exposure[bins]).sum() == pytest.approx(Ic, rel=1e-12)


def test_a_resolution_with_only_sigma_takes_the_gaussian_path(chain):
    """An object with sigma(E) but no fraction() gets the phase-1 erf: identical numbers."""
    class SigmaOnly:
        def sigma(self, e):
            return RES.sigma(e)
    d = _point_data([0.2, 0.0, 0.3], np.linspace(-6, 6, 49))
    a = ForwardModel(chain, d, RES, temperature=20.0).expected()
    b = ForwardModel(chain, d, SigmaOnly(), temperature=20.0).expected()
    assert np.array_equal(a, b)

def test_mode_on_a_bin_edge_splits_in_half(chain):
    h = 0.2
    w = _omega(h)
    d = _point_data([h, 0.0, 0.3], [w - 1, w, w + 1])
    fm = ForwardModel(chain, d, GaussianEnergyResolution(0.02, variable="E"))
    mu = fm.expected()
    total = chain.sqw_batch(d.sample_cartesian(fm.b_matrix))[1].sum()
    assert mu == pytest.approx([total / 2, total / 2], rel=1e-14)


def test_sharp_mode_stays_in_its_bin(chain):
    h = 0.2
    w = _omega(h)
    edges = w + np.array([-0.3, -0.1, 0.05, 0.3])
    d = _point_data([h, 0.0, 0.3], edges)
    for res in (GaussianEnergyResolution(1e-9, variable="E"),
                GaussianEnergyResolution(0.0, variable="E")):
        mu = ForwardModel(chain, d, res).expected()
        total = chain.sqw_batch(d.sample_cartesian(np.diag(2 * np.pi / np.array(
            [A, B_LAT, C_LAT]))))[1].sum()
        assert mu == pytest.approx([0.0, total, 0.0], abs=1e-14 * total, rel=1e-14)


def test_width_is_taken_at_the_mode_energy(chain):
    """A bin far from the mode sees the tail of a Gaussian whose width is sigma(omega),
    not sigma at the bin."""
    h = 0.15
    w = _omega(h)
    d = _point_data([h, 0.0, 0.3], [w + 0.5, w + 0.7])
    mu = ForwardModel(chain, d, RES).expected()[0]
    I = chain.sqw_batch(d.sample_cartesian(np.diag(2 * np.pi / np.array([A, B_LAT, C_LAT]))))[1].sum()
    s = RES.sigma(w)
    assert mu == pytest.approx(I * (ndtr(0.7 / s) - ndtr(0.5 / s)), rel=1e-12)


# ------------------------------------------------------------------ closed form
def _oracle(h_box, k_box, l_box, lo, hi, T):
    """Mean over the Q box of sum over lines of weight * I * bin fraction, by quadrature."""
    def pol(h):
        qx = 2 * np.pi * h / A

        def f(l, k):
            qy, qz = 2 * np.pi * k / B_LAT, 2 * np.pi * l / C_LAT
            return qz ** 2 / (qx ** 2 + qy ** 2 + qz ** 2)
        val, _ = integrate.dblquad(f, k_box[0], k_box[1], l_box[0], l_box[1],
                                   epsabs=1e-13, epsrel=1e-12)
        return val / ((k_box[1] - k_box[0]) * (l_box[1] - l_box[0]))

    def integrand(h):
        w = _omega(h)
        I = S / 2 * (1 + pol(h))
        n1 = thermal_bose_prefactor(np.array([w]), T)[0]
        n0 = thermal_bose_prefactor(np.array([-w]), T)[0]
        s1, s0 = RES.sigma(w), RES.sigma(-w)
        stokes = ndtr((hi - w) / s1) - ndtr((lo - w) / s1)
        gain = ndtr((hi + w) / s0) - ndtr((lo + w) / s0)
        return I * (n1 * stokes + n0 * gain)

    val, _ = integrate.quad(integrand, h_box[0], h_box[1], epsabs=1e-13, epsrel=1e-12,
                            limit=200)
    return val / (h_box[1] - h_box[0])


def test_fm_chain_bins_converge_to_the_closed_form(chain, tmp_path):
    p = tmp_path / "m.npz"
    T = 12.0
    dims = ["[H,0,0]|0.10|0.25|3", "[0,K,0]|-0.1|0.1|1", "[0,0,L]|0.2|0.5|1",
            "DeltaE|-2|6|16"]
    write_mdnorm(p, dims=dims, shape=(3, 1, 1, 16), empty_cell=False)
    he = np.linspace(0.10, 0.25, 4)
    ee = np.linspace(-2, 6, 17)
    errs = []
    for n in (16, 64, 256, 1024):
        d = read_mdnorm_npz(p, "Z", samples_per_cell=n, seed=7)
        s = ForwardModel(chain, d, RES, temperature=T).signal_per_exposure()
        if n == 16:
            want = np.array([_oracle((he[i], he[i + 1]), (-0.1, 0.1), (0.2, 0.5),
                                     ee[j], ee[j + 1], T)
                             for i, j in zip(d.coords["iH"], d.coords["iE"])])
            assert want.max() > 0.05
        errs.append(np.abs(s - want).max() / want.max())
    assert errs[-1] < 1e-5, errs          # measured: 1.1e-3, 5.4e-4, 2.9e-6, 7.8e-7
    assert errs[-1] < errs[0] / 100, errs


# ------------------------------------------------------------------ crystals, smearing
def test_two_crystals_are_the_weighted_sum(chain, tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p, dims=["[H,0,0]|0.1|0.4|3", "[0,K,0]|-0.2|0.2|2",
                          "[0,0,L]|0.2|0.4|1", "DeltaE|0|6|12"], shape=(3, 2, 1, 12))
    d = read_mdnorm_npz(p, "Z", samples_per_cell=8)
    c1 = Crystal.from_axis_angle([0, 0, 1], 10.0)
    c2 = Crystal.from_axis_angle([0, 0, 1], -25.0)
    f = 0.3
    both = ForwardModel(chain, d, RES, crystals=[Crystal(c1.rotation, 1 - f),
                                                 Crystal(c2.rotation, f)]).expected()
    one = ForwardModel(chain, d, RES, crystals=[c1]).expected()
    two = ForwardModel(chain, d, RES, crystals=[c2]).expected()
    assert both == pytest.approx((1 - f) * one + f * two, rel=1e-12)


def test_crystal_rotation_equals_the_frame_map(chain, tmp_path):
    """Crystal(R) and a data frame with to_model = B^-T R B^T are the same thing."""
    p = tmp_path / "m.npz"
    write_mdnorm(p, dims=["[H,0,0]|0.1|0.4|3", "[0,K,0]|-0.2|0.2|2",
                          "[0,0,L]|0.2|0.4|1", "DeltaE|0|6|12"], shape=(3, 2, 1, 12))
    R = Crystal.from_axis_angle([0.3, 0.1, 1.0], 17.0).rotation
    d = read_mdnorm_npz(p, "Z", samples_per_cell=8)
    Bm = np.diag(2 * np.pi / np.array([A, B_LAT, C_LAT]))
    via_crystal = ForwardModel(chain, d, RES, crystals=[Crystal(R)]).expected()
    d2 = read_mdnorm_npz(p, "Z", samples_per_cell=8,
                         frame=QFrame(np.linalg.inv(Bm.T) @ R @ Bm.T))
    via_frame = ForwardModel(chain, d2, RES).expected()
    assert via_crystal == pytest.approx(via_frame, rel=1e-12)


def test_domains_match_calculate_sqw(chain):
    """crystals_from_domains reproduces calculate_sqw's domain average, mode by mode."""
    spec = [{"axis": [0, 1, 0], "angle": 0, "weight": 2},
            {"axis": [0, 1, 0], "angle": 40, "weight": 1}]
    hkl = np.array([0.17, 0.1, 0.3])
    Bm = np.diag(2 * np.pi / np.array([A, B_LAT, C_LAT]))
    q = hkl @ Bm
    ref = chain.calculate_sqw([q], domains=spec)
    edges = np.linspace(0, 6, 25)
    d = _point_data(hkl, edges)
    mu = ForwardModel(chain, d, RES, crystals=crystals_from_domains(spec)).expected()
    want = np.zeros(len(edges) - 1)
    for E, I in zip(np.real(ref.energies[0]), np.real(ref.intensities[0])):
        s = RES.sigma(E)
        want += I * (ndtr((edges[1:] - E) / s) - ndtr((edges[:-1] - E) / s))
    assert mu == pytest.approx(want, rel=1e-9, abs=1e-15)


def test_mosaic_is_the_gaussian_average_over_angle(chain):
    hkl = [0.17, 0.1, 0.3]
    d = _point_data(hkl, np.linspace(0, 6, 25))
    sig = 4.0
    fm = ForwardModel(chain, d, RES, crystals=[Crystal.from_axis_angle(
        [0, 0, 1], 0.0, mosaic_sigma_deg=sig, n_mosaic=9)])
    mu = fm.expected()
    th = np.linspace(-7 * sig, 7 * sig, 281)
    wts = np.exp(-0.5 * (th / sig) ** 2)
    wts /= wts.sum()
    brute = sum(wt * ForwardModel(chain, d, RES, crystals=[Crystal.from_axis_angle(
        [0, 0, 1], t)]).expected() for t, wt in zip(th, wts))
    assert mu == pytest.approx(brute, rel=1e-6, abs=1e-12)


def test_q_smearing_moves_the_samples(chain, tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p, dims=["[H,0,0]|0.1|0.4|3", "[0,K,0]|-0.2|0.2|2",
                          "[0,0,L]|0.2|0.4|1", "DeltaE|0|6|12"], shape=(3, 2, 1, 12))
    d = read_mdnorm_npz(p, "Z", samples_per_cell=8)
    plain = ForwardModel(chain, d, RES).expected()
    assert ForwardModel(chain, d, RES, q_smear=[0, 0, 0]).expected() == pytest.approx(
        plain, rel=1e-15)
    sig = np.array([0.02, 0.01, 0.0])
    smeared = ForwardModel(chain, d, RES, q_smear=sig).expected()
    d.samples_hkl = d.samples_hkl + sig * d.smear_normals
    assert smeared == pytest.approx(ForwardModel(chain, d, RES).expected(), rel=1e-12)
    assert np.abs(smeared - plain).max() > 1e-6 * plain.max()


# ------------------------------------------------------------------ aggregation, cells
def test_aggregated_mu_is_the_sum_of_member_mu(chain, tmp_path):
    p = tmp_path / "m.npz"
    write_mdnorm(p, dims=["[H,0,0]|0.1|0.4|3", "[0,K,0]|-0.2|0.2|2",
                          "[0,0,L]|0.2|0.4|2", "DeltaE|0|6|12"], shape=(3, 2, 2, 12))
    d = read_mdnorm_npz(p, "Z", samples_per_cell=8)
    lab = d.coords["iK"] * 1000 + d.coords["iE"]
    a = d.aggregate(lab)
    mu_n = ForwardModel(chain, d, RES, temperature=5).expected()
    mu_a = ForwardModel(chain, a, RES, temperature=5).expected()
    _, inv = np.unique(lab, return_inverse=True)
    assert mu_a == pytest.approx(np.bincount(inv, weights=mu_n), rel=1e-12)


def test_all_exposure_in_one_cell_gives_that_cells_value(chain, tmp_path):
    f = tmp_path / "r.npz"
    # Coordinates exact in float32, the export layout's type.
    write_cells(f, 1, H=[0.171875, 0.3], K=[0.046875, 0.1], L=[0.296875, 0.2],
                E=[1.0, 1.0], counts=[2, 0], exposure=[2.5, 0.0], dE=0.5)
    d = read_count_cells([f]).aggregate({"K": [0.0, 0.2]}, samples_per_bin=16)
    fm = ForwardModel(chain, d, RES)
    mu = fm.expected()
    one = _point_data([0.171875, 0.046875, 0.296875], [0.75, 1.25], exposure=2.5)
    assert mu == pytest.approx(ForwardModel(chain, one, RES).expected(), rel=1e-14)


# ------------------------------------------------------------------ small pieces
def test_resolution_law_ascending_powers():
    r = GaussianEnergyResolution(0.104, (1.0, -1.04, 0.38), "E/Ei", 4.681)
    E = np.linspace(0, 4, 9)
    x = E / 4.681
    assert r.fwhm(E) == pytest.approx(0.104 * (1 - 1.04 * x + 0.38 * x ** 2), rel=1e-15)
    with pytest.raises(ValueError):
        GaussianEnergyResolution(1.0, (1.0, -1.0), "E").sigma(np.array([2.0]))


def test_poisson_deviance():
    n = np.array([0.0, 3.0, 5.0])
    mu = np.array([0.5, 2.0, 5.0])
    want = 2 * ((0.5 - 0) + (2 - 3 + 3 * np.log(3 / 2)) + 0.0)
    assert poisson_deviance(n, mu) == pytest.approx(want, rel=1e-14)
    assert poisson_deviance(n, n + (n == 0)) == pytest.approx(2.0)
    assert poisson_deviance([1.0], [0.0]) == np.inf


def test_cached_on_parameters(chain):
    d = _point_data([0.2, 0.0, 0.3], np.linspace(0, 6, 13))
    fm = ForwardModel(chain, d, RES)
    a = fm.expected()
    assert fm.expected() is not None and len(fm._cache) == 1
    b = fm.expected([J * 1.1, FIELD, [0, 0, 1]])
    assert np.abs(a - b).max() > 0
    fm.calc.update_hamiltonian_params([J, FIELD, [0, 0, 1]])
    assert fm.expected() == pytest.approx(a, rel=1e-14)
