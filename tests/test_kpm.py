"""KPM dynamical structure factor (magcalc/sun/kpm.py).

The oracle is the engine's OWN exact diagonalization (`SUNModel.structure_factor`):
KPM expands the SAME spectral function via Chebyshev moments of the para-unitary
dynamical matrix, so with enough moments its S(q,ω) must converge to the exact
diagonalization broadened by the same Gaussian — to machine precision. No external
reference needed: it is pinned to the exact result the engine already trusts.
"""
import numpy as np

_trapezoid = getattr(np, "trapezoid", None) or np.trapz
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.lswt import SUNModel
from magcalc.sun.kpm import kpm_sqw, exact_broadened_sqw, spectral_bound


def _square_afm(S=1.0, J=1.0):
    cfg = {"crystal_structure": {"lattice_vectors": [[1., 0, 0], [0, 1, 0], [0, 0, 10]],
            "atoms_uc": [{"label": "A", "pos": [0, 0, 0], "spin_S": S},
                         {"label": "B", "pos": [0.5, 0.5, 0], "spin_S": S}]},
        "interactions": {"symmetry_rules": [{"type": "heisenberg", "distance": 0.7072,
                                             "value": "J1"}]},
        "parameters": {"J1": J}, "parameter_order": ["J1"],
        "magnetic_structure": {"type": "pattern", "pattern_type": "antiferromagnetic",
                               "direction": [0, 0, 1], "propagation_vector": [0, 0, 0]}}
    gm = GenericSpinModel(cfg)
    sm = SUNModel.from_generic_model(gm, [J])
    B = 2 * np.pi * np.linalg.inv(np.array([[1., 0, 0], [0, 1, 0], [0, 0, 10]])).T
    return sm, B


def test_spectral_bound_encloses_the_spectrum():
    sm, B = _square_afm()
    for hk in [(0.25, 0.0), (0.5, 0.0), (0.3, 0.2)]:
        q = np.array([hk[0], hk[1], 0]) @ B
        gam = spectral_bound(sm.hamiltonian(q))
        maxw = float(np.abs(sm.dispersion(q)).max())
        assert gam > maxw, f"gamma {gam} must enclose max|w| {maxw}"
        assert gam < 2.0 * maxw, "bound should not be wildly loose"


@pytest.mark.slow
def test_kpm_converges_to_exact_diagonalization():
    """As the moment count grows, KPM S(q,ω) → the exact broadened spectrum; the
    error decreases monotonically and hits machine zero at high order."""
    sm, B = _square_afm()
    egrid = np.linspace(0, 8, 400)
    fwhm = 0.3
    for hk in [(0.25, 0.0), (0.5, 0.0), (0.3, 0.2)]:
        q = np.array([hk[0], hk[1], 0]) @ B
        gam = spectral_bound(sm.hamiltonian(q))
        exact = exact_broadened_sqw(sm, q, egrid, fwhm)
        errs = []
        for M in (120, 250, 500):
            r = kpm_sqw(sm, q, egrid, fwhm, n_moments=M, gamma=gam)
            errs.append(_trapezoid(np.abs(r.intensities - exact), egrid)
                        / _trapezoid(np.abs(exact), egrid))
        assert errs[0] > errs[1] > errs[2], f"not converging: {errs}"
        assert errs[2] < 1e-3, f"high-order error too large: {errs[2]}"


@pytest.mark.slow
def test_kpm_conserves_integrated_intensity():
    """∫dω S(q,ω) (the per-q sum rule) must match the exact total to high order."""
    sm, B = _square_afm()
    egrid = np.linspace(0, 8, 600)
    fwhm = 0.3
    for hk in [(0.25, 0.0), (0.5, 0.0)]:
        q = np.array([hk[0], hk[1], 0]) @ B
        exact = _trapezoid(exact_broadened_sqw(sm, q, egrid, fwhm), egrid)
        r = kpm_sqw(sm, q, egrid, fwhm, n_moments=500)
        assert abs(_trapezoid(r.intensities, egrid) - exact) < 1e-3 * exact
