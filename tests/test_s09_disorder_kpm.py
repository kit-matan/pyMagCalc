"""S09 -- the 120-degree triangular AFM as a REAL-SPACE cell, and disorder + KPM.

The clean part of this tutorial (`config.yaml`) uses the rotating-frame `single_k`
method, which the SU(N)/KPM path does not consume; disorder is a per-bond property of
a large real-space cell. So the port needed the 120-degree order as an explicit
supercell, and that is what is pinned here -- against exact references only:

  * E/site = -0.375 meV, the classical 120-degree energy
    (1/2) * 6 * J * S^2 * cos(120 deg) at J = 1, S = 1/2;
  * the bands are the analytic w(q) = 3 J S sqrt[(1-g_q)(1+2g_q)] of the 120-degree
    state, folded into the three {q-k, q, q+k} channels;
  * S(q,w) on a big supercell EQUALS S(q,w) on the minimal cell -- two cell choices
    for one crystal, so the folded ghost bands must carry exactly zero weight;
  * disorder broadening is measured against the engine's exact diagonalization, and
    the "continuum" statement is pinned to the exact clean bandwidth
    w_max = 3 J S sqrt(9/8) = 1.5910 meV, above which the clean spectrum has no
    weight at all.

THE TRAP THIS FOLDER EXISTS TO RECORD. The old note here read "disorder NARROWED the
KPM width instead of broadening it, which is what expanding about a non-minimum buys
you" -- and the diagnosis was wrong twice over. The placeholder reference state was
indeed wrong (a ferromagnet, E/site = +0.75, on a lattice whose ground state is
-0.375), but fixing it did NOT fix the narrowing. That was a real bug in `sun/kpm.py`
(the Chebyshev recursion ran on D-hat instead of conj(D-hat)), which put spurious
LOW-ENERGY weight on a CLEAN non-collinear supercell -- i.e. it made the clean
spectrum look pre-broadened, so adding disorder appeared to narrow it. See
tests/test_kpm.py.
"""
import copy

import numpy as np
import pytest

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.kpm import exact_broadened_sqw, kpm_sqw
from magcalc.sun.lswt import SUNModel, apply_bond_disorder
from magcalc.sun.operators import coherent_from_direction

A = 3.0
LAT = np.array([[A, 0, 0], [-A / 2, A * np.sqrt(3) / 2, 0], [0, 0, 10.0]])
B = 2 * np.pi * np.linalg.inv(LAT).T
OFFS = ([1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [1, 1, 0], [-1, -1, 0])
K120 = np.array([1 / 3, 1 / 3, 0.0])
E_120 = -0.375                       # (1/2) * 6 * J * S^2 * cos(120 deg)
W_MAX = 3 * 1.0 * 0.5 * np.sqrt(9 / 8)               # 1.59099 meV
# q OFF the 1/n grid of every cell used below: a q on it folds onto Gamma_super,
# where the exact Goldstone zero makes the Cholesky in _bogoliubov fail. That is a
# limit of the ORACLE, not of the models being compared.
QGEN = ([0.2, 0.1, 0], [0.37, -0.11, 0], [0.13, 0.29, 0], [0.44, 0.07, 0])


def _cfg():
    return {"crystal_structure": {"lattice_vectors": LAT.tolist(),
                                  "atoms_uc": [{"label": "A", "pos": [0.0, 0, 0],
                                                "spin_S": 0.5}]},
            "interactions": {"heisenberg": [{"pair": ["A", "A"], "rij_offset": list(o),
                                             "value": 1.0} for o in OFFS]},
            "magnetic_structure": {"type": "pattern",
                                   "pattern_type": "ferromagnetic",
                                   "direction": [0, 0, 1]},
            "parameters": {}, "parameter_order": []}


def _model(supercell):
    return SUNModel.from_generic_model(GenericSpinModel(copy.deepcopy(_cfg())),
                                       params=[], supercell=supercell)


def _impose_120(m):
    """The exact 120-degree state, from the site positions -- not a search."""
    ang = 2 * np.pi * ((np.asarray(m.pos, float) @ np.linalg.inv(LAT)) @ K120)
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros_like(ang)], axis=1)
    m.Z = [coherent_from_direction(s, d) for s, d in zip(m.S, dirs)]
    m._prepare()
    return m


def _w_exact(q_rlu):
    """w(q) = 3 J S sqrt[(1-g_q)(1+2g_q)] for the 120-degree triangular AFM."""
    q = np.asarray(q_rlu, float) @ B
    g = (np.cos(q @ LAT[0]) + np.cos(q @ LAT[1]) + np.cos(q @ (LAT[0] + LAT[1]))) / 3
    return 3 * 1.0 * 0.5 * np.sqrt(max((1 - g) * (1 + 2 * g), 0.0))


# --------------------------------------------------------------- the ground state
def test_sqrt3_cell_hosts_the_exact_120_degree_ground_state():
    """|det| = 3 with k . A in Z for both new lattice vectors. E/site = -0.375 is the
    exact classical value, so this is an identity, not a fitted number."""
    m = _model([[1, 1, 0], [-1, 2, 0], [0, 0, 1]])      # columns a1-a2, a1+2a2
    e = m.minimize_energy(n_restarts=8, seed=0) / m.L
    assert abs(e - E_120) < 1e-10, f"E/site = {e}, expected {E_120}"
    d = m.dipoles
    cos = (d @ d.T) / np.outer(np.linalg.norm(d, axis=1), np.linalg.norm(d, axis=1))
    off = cos[~np.eye(3, dtype=bool)]
    assert np.allclose(off, -0.5, atol=1e-8), f"not 120 degrees apart: {off}"


def test_the_other_determinant_3_cell_CANNOT_host_it():
    """The transposed matrix also has |det| = 3, and is wrong: its lattice vectors
    are a1+a2 and -a1+2a2, and k . (a1+a2) = 2/3 is not an integer, so the
    120-degree state does not fit. The search then returns a frustrated collinear
    state -- with a perfectly plausible-looking spectrum. Commensurability is the
    check, and the energy is how you see it."""
    m = _model([[1, -1, 0], [1, 2, 0], [0, 0, 1]])
    e = m.minimize_energy(n_restarts=12, seed=0) / m.L
    assert e > E_120 + 0.2, f"E/site = {e} -- this cell should NOT reach {E_120}"


def test_supercell_bands_are_the_analytic_120_degree_dispersion():
    """The 3 bands at a chemical q are {w(q-k), w(q), w(q+k)} of the closed form."""
    m = _impose_120(_model([[1, 1, 0], [-1, 2, 0], [0, 0, 1]]))
    assert abs(m.energy_per_site() - E_120) < 1e-12
    for q_rlu in QGEN:
        q = np.array(q_rlu, float)
        got = np.sort(np.real(m.dispersion(q @ B)))
        ref = np.sort([_w_exact(q - K120), _w_exact(q), _w_exact(q + K120)])
        assert np.allclose(got, ref, atol=1e-9), f"q={q_rlu}: {got} vs {ref}"


# ------------------------------------------------------------------ cell identity
def test_a_big_supercell_gives_the_SAME_sqw_as_the_minimal_cell():
    """Two cell choices for one crystal and one spin configuration, so S(q,w) per
    chemical cell is identical: the 78 extra folded bands of the 9x9 cell must carry
    EXACTLY zero weight. This is what makes a supercell spectrum meaningful at all,
    and it is an algebraic identity rather than a physical claim."""
    small = _impose_120(_model([[1, 1, 0], [-1, 2, 0], [0, 0, 1]]))
    big = _impose_120(_model([[9, 0, 0], [0, 9, 0], [0, 0, 1]]))
    assert abs(big.energy_per_site() - E_120) < 1e-12
    egrid = np.arange(0.0, 3.0001, 0.02)
    for q_rlu in QGEN:
        q = np.array(q_rlu, float) @ B
        a = exact_broadened_sqw(small, q, egrid, 0.15)
        b = exact_broadened_sqw(big, q, egrid, 0.15)
        assert np.abs(a - b).max() < 1e-9 * a.max(), f"q={q_rlu}"


def test_the_clean_spectrum_has_no_weight_above_the_exact_bandwidth():
    """w_max = 3 J S sqrt(9/8) is an analytic bound on the clean 120-degree spectrum.
    It is the baseline for the 'bands broaden into a continuum' claim below."""
    m = _impose_120(_model([[9, 0, 0], [0, 9, 0], [0, 0, 1]]))
    egrid = np.arange(0.0, 3.0001, 0.02)
    for q_rlu in QGEN:
        s = exact_broadened_sqw(m, np.array(q_rlu, float) @ B, egrid, 0.15)
        above = s[egrid > W_MAX + 0.3].sum() / s.sum()
        assert above < 1e-6, f"q={q_rlu}: {above:.2e} of the weight above w_max"


# ----------------------------------------------------------------------- disorder
def _disordered(n, sigma, seed):
    m = _impose_120(_model([[n, 0, 0], [0, n, 0], [0, 0, 1]]))
    apply_bond_disorder(m, sigma, seed=seed)
    m.minimize_energy(n_restarts=1, max_iter=2000, tol=1e-15)
    return m


def _width(model, egrid, fwhm=0.15):
    out = []
    for q_rlu in QGEN:
        s = exact_broadened_sqw(model, np.array(q_rlu, float) @ B, egrid, fwhm)
        n = s.sum()
        e1 = (egrid * s).sum() / n
        out.append(np.sqrt(max((egrid ** 2 * s).sum() / n - e1 ** 2, 0.0)))
    return float(np.mean(out))


def test_disorder_BROADENS_and_does_so_monotonically():
    """The tutorial's actual claim. Measured on the exact spectrum, so a KPM
    reconstruction error cannot fake it -- which is exactly how the previous
    'disorder narrows the spectrum' conclusion came about."""
    egrid = np.arange(0.0, 3.0001, 0.02)
    clean = _width(_impose_120(_model([[9, 0, 0], [0, 9, 0], [0, 0, 1]])), egrid)
    widths = [np.mean([_width(_disordered(9, s, seed), egrid) for seed in range(2)])
              for s in (0.1, 0.2)]
    assert clean < widths[0] < widths[1], f"clean {clean}, disordered {widths}"
    assert widths[0] > 1.05 * clean, f"no measurable broadening: {clean} -> {widths[0]}"


def test_disorder_pushes_weight_ABOVE_the_clean_bandwidth():
    """'The discrete bands broaden into a continuum', stated against the analytic
    bound rather than against a recorded number."""
    egrid = np.arange(0.0, 3.0001, 0.02)
    m = _disordered(9, 0.25, seed=0)
    frac = []
    for q_rlu in QGEN:
        s = exact_broadened_sqw(m, np.array(q_rlu, float) @ B, egrid, 0.15)
        frac.append(s[egrid > W_MAX + 0.3].sum() / s.sum())
    assert np.mean(frac) > 0.01, f"only {np.mean(frac):.4f} of the weight above w_max"


def test_kpm_reproduces_the_exact_disordered_spectrum():
    """KPM is the tool the tutorial uses because it is linear in the system size;
    it is only worth using if it agrees with the diagonalization it replaces."""
    m = _disordered(9, 0.1, seed=0)
    egrid = np.arange(0.0, 3.0001, 0.02)
    for q_rlu in QGEN[:2]:
        q = np.array(q_rlu, float) @ B
        exact = exact_broadened_sqw(m, q, egrid, 0.15)
        got = kpm_sqw(m, q, egrid, 0.15, n_moments=400).intensities
        rel = np.abs(got - exact).max() / exact.max()
        assert rel < 2e-3, f"q={q_rlu}: relative error {rel:.2e}"


@pytest.mark.slow
def test_at_sunny_s_disorder_strength_the_120_state_is_NOT_a_minimum():
    """Why this port ships sigma = 0.1 and not Sunny's 1/3.

    H2 >= 0 at every q is the exact criterion for the reference state to be a
    classical minimum. At sigma = 1/3 the relaxed state fails it on most disorder
    realizations -- and KPM cannot notice, because it never diagonalizes. The
    relaxation is not at fault: annealing and a damped CP^(N-1) quench return the
    same state to 8 decimals, and enlarging the cell to 2x, 3x, 4x the disorder
    period does not lower the energy.
    """
    # The criterion is H2 >= 0 at EVERY q, and the unstable modes sit at particular
    # q, so this needs a path rather than the handful of generic points above --
    # scanning only QGEN finds the instability on 1 seed in 3 instead of 2.
    path = np.concatenate([
        np.linspace(np.zeros(3), K120, 14, endpoint=False),
        np.linspace(K120, np.array([0.5, 0.0, 0.0]), 14, endpoint=False),
        np.linspace(np.array([0.5, 0.0, 0.0]), np.zeros(3), 14)])

    def min_eig_h2(model):
        lo = np.inf
        for q_rlu in path:
            H = np.asarray(model.hamiltonian(np.array(q_rlu, float) @ B), complex)
            D = H.shape[0] // 2
            g = np.concatenate([np.ones(D), -np.ones(D)])
            H2 = g[:, None] * H
            lo = min(lo, float(np.linalg.eigvalsh(0.5 * (H2 + H2.conj().T)).min()))
        return lo

    weak = [min_eig_h2(_disordered(9, 0.05, seed)) for seed in range(3)]
    assert min(weak) > -1e-6, f"sigma = 0.05 should stay a minimum: {weak}"
    strong = [min_eig_h2(_disordered(9, 1 / 3, seed)) for seed in range(3)]
    assert sum(x < -1e-4 for x in strong) >= 2, f"expected instability: {strong}"
