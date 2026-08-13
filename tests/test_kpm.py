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
from magcalc.sun.operators import coherent_from_direction


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


TRI = [[3.0, 0, 0], [-1.5, 3 * np.sqrt(3) / 2, 0], [0, 0, 10.0]]
TRI_OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0])


def _triangular_120(n):
    """The 120-degree triangular AFM (J = 1, s = 1/2) on an n x n supercell, in its
    EXACT ground state (E/site = -0.375) rather than a searched one.

    NON-COLLINEAR and large enough to fold: that combination is what the square AFM
    above cannot test, and it is where the Chebyshev recursion's conjugation stops
    being invisible.
    """
    cfg = {"crystal_structure": {"lattice_vectors": TRI,
                                 "atoms_uc": [{"label": "A", "pos": [0., 0, 0],
                                               "spin_S": 0.5}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": list(o),
                                            "value": 1.0} for o in TRI_OFFS]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    sm = SUNModel.from_generic_model(GenericSpinModel(cfg), [],
                                     supercell=[[n, 0, 0], [0, n, 0], [0, 0, 1]])
    lat = np.array(TRI, float)
    ang = 2 * np.pi * ((np.asarray(sm.pos, float) @ np.linalg.inv(lat))
                       @ np.array([1 / 3, 1 / 3, 0.0]))
    sm.Z = [coherent_from_direction(s, d) for s, d in
            zip(sm.S, np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1))]
    sm._prepare()
    assert abs(sm.energy_per_site() + 0.375) < 1e-10, "not the 120-degree ground state"
    return sm, 2 * np.pi * np.linalg.inv(lat).T


def _ferromagnet():
    """s = 1, square-lattice FM. Its chiral term is non-zero for q out of the plane,
    which is what fixes the (a, b) ORDER of the weight matrix."""
    lat = [[1., 0, 0], [0, 1, 0], [0, 0, 10]]
    cfg = {"crystal_structure": {"lattice_vectors": lat,
                                 "atoms_uc": [{"label": "A", "pos": [0, 0, 0],
                                               "spin_S": 1.0}]},
           "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": o,
                                            "value": -1.0}
                                           for o in ([1, 0, 0], [-1, 0, 0],
                                                     [0, 1, 0], [0, -1, 0])]},
           "magnetic_structure": {"type": "pattern", "pattern_type": "ferromagnetic",
                                  "direction": [0, 0, 1]},
           "parameters": {}, "parameter_order": []}
    return (SUNModel.from_generic_model(GenericSpinModel(cfg), []),
            2 * np.pi * np.linalg.inv(np.array(lat, float)).T)


def test_kpm_matches_exact_on_a_NON_COLLINEAR_supercell():
    """The clean 120-degree state on 81 sites: only 3 of the 81 folded bands carry
    weight, and KPM has to reproduce that.

    Until 2026-08-13 the recursion ran on D-hat rather than conj(D-hat), which put
    ~5 % of the intensity onto bands that carry none. It is invisible for a
    collinear model (D-hat is then real), so every KPM test above passed throughout.
    The symptom was worse than the number suggests: the spurious weight sits at LOW
    energy and looks exactly like disorder broadening of a CLEAN spectrum.
    """
    sm, B = _triangular_120(9)
    egrid = np.arange(0.0, 3.0001, 0.02)
    fwhm = 0.15
    # q off the 1/n grid: on it, the folded Goldstone zero makes the exact
    # Cholesky (not KPM) fail, which is a property of the oracle, not of KPM.
    for q_rlu in ([0.2, 0.1, 0], [0.37, -0.11, 0], [0.13, 0.29, 0]):
        q = np.array(q_rlu, float) @ B
        exact = exact_broadened_sqw(sm, q, egrid, fwhm)
        got = kpm_sqw(sm, q, egrid, fwhm, n_moments=400).intensities
        rel = np.abs(got - exact).max() / exact.max()
        assert rel < 2e-3, f"q={q_rlu}: relative error {rel:.2e}"


def test_kpm_chiral_channel_has_the_right_SIGN():
    """`cross_section: chiral` is ANTISYMMETRIC in (a, b), so it pins the order of
    the weight matrix -- which the symmetric channels (perp, trace, xx, ...) cannot
    see at all. It came back sign-REVERSED on a plain ferromagnet until 2026-08-13
    (relative error exactly 2.0, the signature of a flip)."""
    sm, B = _ferromagnet()
    egrid = np.linspace(0, 10, 300)
    fwhm = 0.3
    for q_rlu in ([0.2, 0.1, 0.4], [0.3, 0.0, 0.6], [0.25, 0.35, 0.5]):
        q = np.array(q_rlu, float) @ B
        exact = exact_broadened_sqw(sm, q, egrid, fwhm, cross_section="chiral")
        assert np.abs(exact).max() > 0.1, "test needs a non-zero chiral signal"
        got = kpm_sqw(sm, q, egrid, fwhm, n_moments=400,
                      cross_section="chiral").intensities
        rel = np.abs(got - exact).max() / np.abs(exact).max()
        assert rel < 1e-3, f"q={q_rlu}: relative error {rel:.2e}"


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
