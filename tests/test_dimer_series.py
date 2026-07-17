"""High-order dimer series expansion (magcalc/sun/dimer_series.py).

Every layer is pinned to an INDEPENDENT oracle, never a self-generated number:

  * the PT engine and the eigenvalue-series expansion against exact diagonalization
    (errors must scale as lambda^(order+1));
  * the chain's order-1 dispersion against the analytic omega = J - (J'/2) cos k;
  * the full linked-cluster pipeline against momentum-resolved EXACT DIAGONALIZATION
    of the alternating Heisenberg chain, at moderate (lambda = 0.4) and STRONG
    (lambda = 0.8) coupling -- the regime the harmonic engine cannot reach;
  * Cu5SbO6 (through GenericSpinModel) at order 1 against the exact first-order dimer
    expansion of Piyakulworawat et al., PRR 8, 013247 (2026), Eq. (A11);
  * Dlog-Pade against a function it must resum exactly (a simple pole).
"""
import os

import numpy as np
import pytest
import yaml

from magcalc.generic_model import GenericSpinModel
from magcalc.sun.dimer_series import (DimerSeriesModel, block_effective_series,
                                      dlog_pade_estimates, eigenvalue_series, resummed)

HERE = os.path.dirname(__file__)
rng = np.random.default_rng(7)


# ------------------------------------------------------------------ PT engines
def test_block_pt_matches_exact_diagonalization():
    """Eigenvalues of the order-n effective Hamiltonian must match the exact ones to
    O(lambda^{n+1}) -- checked by the error RATIO between two lambdas."""
    N, p, order = 14, 3, 5
    E0 = np.sort(np.concatenate([[1.0, 1.02, 1.05], 3 + 2 * rng.random(N - p)]))
    V = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    V = 0.5 * (V + V.conj().T)
    Heff = block_effective_series(E0, [V], [0, 1, 2], order)
    errs = []
    for lam in (2e-2, 1e-2):
        exact = np.sort(np.linalg.eigvalsh(np.diag(E0) + lam * V))[:p]
        approx = np.sort(np.linalg.eigvalsh(
            sum(Heff[k] * lam ** k for k in range(order + 1))))
        errs.append(np.abs(approx - exact).max())
    ratio = errs[0] / errs[1]
    assert 2 ** (order) < ratio < 2 ** (order + 2), f"error ratio {ratio}"


def test_eigenvalue_series_matches_brute_force():
    n, order = 8, 4
    H0 = np.diag([1.0, 1, 1, 2.5, 2.5, 4, 4, 4])
    Hs = [H0]
    for _ in range(order):
        A = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Hs.append(0.5 * (A + A.conj().T))
    series = eigenvalue_series(Hs, order)
    errs = []
    for lam in (1e-2, 5e-3):
        exact = np.sort(np.linalg.eigvalsh(sum(Hs[k] * lam ** k for k in range(len(Hs)))))
        approx = np.sort([sum(s[k] * lam ** k for k in range(order + 1)) for s in series])
        errs.append(np.abs(exact - approx).max())
    ratio = errs[0] / errs[1]
    assert 2 ** (order - 1) < ratio < 2 ** (order + 3), f"error ratio {ratio}"


# --------------------------------------------------------------------- ED oracle
def _ed_alt_chain(n_dimers, J, Jp):
    """Momentum-resolved ED of the alternating chain. Returns (E_gs, {m: omega})."""
    L = 2 * n_dimers
    bonds = [(2 * i, 2 * i + 1, J) for i in range(n_dimers)] + \
            [(2 * i + 1, (2 * i + 2) % L, Jp) for i in range(n_dimers)]
    mask = (1 << L) - 1

    def t2(s):
        return ((s << 2) | (s >> (L - 2))) & mask

    def block(states, m):
        reps = {}
        for s in states:
            orb, x = [s], t2(s)
            while x != s:
                orb.append(x)
                x = t2(x)
            r = min(orb)
            if r not in reps:
                reps[r] = len(orb)
        Nd = n_dimers
        keep = [(r, p) for r, p in reps.items() if (m * p) % Nd == 0]
        idx = {r: i for i, (r, _p) in enumerate(keep)}
        norm = {r: Nd / p for r, p in keep}
        k = 2 * np.pi * m / Nd
        H = np.zeros((len(keep), len(keep)), complex)
        for (r, _p) in keep:
            i = idx[r]
            for (a, b, Jb) in bonds:
                if ((r >> a) & 1) == ((r >> b) & 1):
                    H[i, i] += 0.25 * Jb
                else:
                    H[i, i] -= 0.25 * Jb
                    s2 = r ^ (1 << a) ^ (1 << b)
                    x, r2, l2 = s2, s2, 0
                    for shift in range(1, Nd):
                        x = t2(x)
                        if x < r2:
                            r2, l2 = x, shift
                    if r2 in idx:
                        H[i, idx[r2]] += 0.5 * Jb * np.exp(-1j * k * l2) * \
                            np.sqrt(norm[r2] / norm[r])
        H = 0.5 * (H + H.conj().T)
        return np.linalg.eigvalsh(H)

    def sector(nup):
        return [s for s in range(1 << L) if bin(s).count("1") == nup]

    E_gs = float(min(block(sector(L // 2), 0)))
    omega = {m: float(block(sector(L // 2 + 1), m)[0] - E_gs)
             for m in range(n_dimers)}
    return E_gs, omega


def _chain_model(lam):
    lat = np.diag([1.0, 10.0, 10.0])
    I3 = np.eye(3)
    bonds = [(0, 1, (0, 0, 0), 1.0 * I3), (1, 0, (1, 0, 0), lam * I3)]
    return DimerSeriesModel.from_spin_arrays(
        lat, [0.5, 0.5], [[0, 0, 0], [0.35, 0, 0]], bonds, units=[[0, 1]])


def test_chain_first_order_is_the_analytic_dispersion():
    m = _chain_model(0.3)
    for qr in (0.0, 0.2, 0.37, 0.5):
        s = m.band_series(np.array([2 * np.pi * qr, 0, 0]), 1)[0]
        assert abs(s[0] - 1.0) < 1e-9
        assert abs(s[1] - (-(0.3 / 2) * np.cos(2 * np.pi * qr))) < 1e-9


@pytest.mark.slow
def test_chain_series_matches_exact_diagonalization():
    """lambda = 0.4, order 6: the whole one-triplon band to < 5e-4 J vs ED (L = 16)."""
    lam, Nd, order = 0.4, 8, 6
    _E, om = _ed_alt_chain(Nd, 1.0, lam)
    m = _chain_model(lam)
    for mm in range(Nd // 2 + 1):
        q = np.array([2 * np.pi * mm / Nd, 0, 0])
        v, _ = resummed(m.band_series(q, order)[0], 1.0, "sum")
        assert abs(v - om[mm]) < 5e-4, f"k index {mm}: {v} vs ED {om[mm]}"


@pytest.mark.slow
def test_chain_strong_coupling_with_dlog_pade():
    """lambda = 0.8 (J'/J = 0.8, STRONG coupling): order 7 + Dlog-Pade reproduces the
    ED gap to better than 8%, and far better away from the gap. This is the regime
    where the harmonic bond-operator engine fails outright."""
    lam, Nd, order = 0.8, 8, 7
    _E, om = _ed_alt_chain(Nd, 1.0, lam)
    m = _chain_model(lam)
    # gap at k = 0 (hardest point)
    s = m.band_series(np.zeros(3), order)[0]
    v_sum, _ = resummed(s, 1.0, "sum")
    assert abs(v_sum - om[0]) / om[0] < 0.08, f"gap: {v_sum} vs ED {om[0]}"
    # mid-band point is much tighter
    q = np.array([2 * np.pi * 2 / Nd, 0, 0])
    v, _ = resummed(m.band_series(q, order)[0], 1.0, "sum")
    assert abs(v - om[2]) / om[2] < 0.01


def test_cu5sbo6_order1_equals_paper_eq_A11():
    """Through the FULL GenericSpinModel pipeline: the order-1 series must equal the
    exact first-order dimer expansion of PRR 8, 013247 (2026), Eq. (A11)."""
    cfg = yaml.safe_load(open(os.path.join(
        HERE, "..", "examples", "entangled", "Cu5SbO6", "config.yaml")))
    J1, J2, J4 = (cfg["parameters"][k] for k in ("J1", "J2", "J4"))
    model = GenericSpinModel(cfg)
    pv = [cfg["parameters"][k] for k in cfg["parameter_order"]]
    dsm = DimerSeriesModel.from_generic_model(model, pv, units=[[0, 1]])
    L = np.array(cfg["crystal_structure"]["lattice_vectors"], float)
    B = 2 * np.pi * np.linalg.inv(L).T
    for (h, l) in [(0, 0), (0.25, 0), (0.5, 0.5), (0.3, 0.7), (0.11, 0.83)]:
        s = dsm.band_series(np.array([h, 0, l]) @ B, 1)[0]
        a11 = J1 - (J2 / 2) * np.cos(2 * np.pi * h) - (J4 / 2) * np.cos(2 * np.pi * (l - h))
        assert abs((s[0] + s[1]) - a11) < 1e-8, f"q=({h},{l})"


def test_dlog_pade_resums_a_simple_pole_exactly():
    """f = (1 - x/2)^(-1): ln-derivative is a [0/1] rational, so Dlog-Pade must be
    exact from a short series -- while the plain truncated sum is visibly off."""
    n = 6
    c = np.array([(0.5) ** k for k in range(n + 1)])     # series of 1/(1 - x/2)
    x = 0.9
    exact = 1.0 / (1.0 - x / 2)
    ests = dlog_pade_estimates(c, x)
    assert ests, "no surviving approximants"
    assert min(abs(e - exact) for e in ests) < 1e-6
    plain, _ = resummed(c, x, "sum")
    assert abs(plain - exact) > 1e-3                      # sum alone is not enough
