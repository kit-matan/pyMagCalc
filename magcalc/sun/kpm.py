"""Kernel Polynomial Method (KPM) for the LSWT dynamical structure factor.

Full diagonalization of the 2D×2D Bogoliubov–de Gennes matrix costs O(D³) per q,
which is prohibitive for the large magnetic cells that arise with quenched disorder
or near-incommensurate order. KPM replaces it by a Chebyshev expansion of the
spectral function, built from iterated matrix–vector products with the dynamical
matrix — no eigensolve. With the dense matrices the engine currently builds this is
O(D²·M) per q for M moments (an eigensolve is O(D³)); a sparse dynamical matrix
would bring it to O(nnz·M).

The subtlety is that the LSWT dynamical matrix is the PARA-unitary (non-Hermitian)
D̂ = g H₂, with H₂ ≻ 0 Hermitian and g = diag(1_D, −1_D); its spectrum is the real
set {±ω_ν}. pyMagCalc already stores exactly this as `SUNModel.hamiltonian(q)`
(HMat = g H₂). Following Lane et al., "Kernel Polynomial Method for Linear Spin Wave
Theory" (arXiv:2312.08349) — the method Sunny 0.8.1's `SpinWaveTheoryKPM` uses — the
one-magnon structure factor is the spectral density

    S^{ab}(q, ω) = Σ_ν conj(M^a_ν) M^b_ν δ(ω − ω_ν),   M^a_ν = (v_a T)_{D+ν},

whose Chebyshev moments are μ_m^{ab} = v_a† T_m(Â) g v_b with  = D̂/γ (spectrum in
[−1,1], γ ≳ max ω). Reconstructing with the coefficients of the broadening kernel,
and a smooth cutoff that keeps the +ω_ν poles (dropping their −ω_ν mirror), gives
S(q,ω) directly from the moments — no diagonalization.

Validated (tests/test_kpm.py) against the engine's OWN exact diagonalization
(`SUNModel.structure_factor`) broadened by the same Gaussian: as M grows the KPM
S(q,ω) converges to the exact broadened spectrum (square-lattice AFM, per-mode and
integrated), and the eigenvalue bound γ is verified to enclose the spectrum.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _observable_vectors(model, q_cart):
    """The neutron observable in the Nambu basis Ψ=(b, b†): v[a] (a=x,y,z), shape
    (3, 2D). Identical construction to `SUNModel.structure_factor`, so the KPM and the
    exact path expand the SAME spectral function."""
    D = model.L * model.M
    v = np.zeros((3, 2 * D), dtype=complex)
    for i in range(model.L):
        # NO intracell position phase: hamiltonian(q) is in the full-position gauge
        # (see SUNModel.structure_factor); only the intra-unit d_k offsets phase.
        sl = slice(i * model.M, (i + 1) * model.M)
        slb = slice(D + i * model.M, D + (i + 1) * model.M)
        for (d_k, idx) in model.moment_terms[i]:
            ph = np.exp(1j * float(np.dot(q_cart, d_k)))
            for a in range(3):
                v[a, sl] += ph * model.t[i, idx[a]]
                v[a, slb] += ph * model.tb[i, idx[a]]
    return v


def spectral_bound(HMat, n_iter=40, pad=1.15, seed=0):
    """Upper bound on max|ω| (spectral radius of the para-unitary D̂ = HMat) via power
    iteration on D̂†D̂. Padded so the true spectrum lies strictly inside [−γ, γ]."""
    n = HMat.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    x /= np.linalg.norm(x)
    H = HMat.conj().T @ HMat
    lam = 0.0
    for _ in range(n_iter):
        y = H @ x
        lam = float(np.real(x.conj() @ y))
        nrm = np.linalg.norm(y)
        if nrm < 1e-300:
            break
        x = y / nrm
    return pad * np.sqrt(max(lam, 0.0))


def _cheb_coefs(f, M, gamma, n_nodes=None):
    """Chebyshev coefficients c_m (m=0..M-1) of f(E) on E ∈ [−γ, γ], via the DCT over
    Chebyshev nodes. f(E) ≈ Σ_m c_m T_m(E/γ)."""
    K = n_nodes or max(4 * M, 64)
    k = np.arange(K)
    xi = np.cos(np.pi * (k + 0.5) / K)          # nodes in [-1,1]
    fk = f(gamma * xi)
    m = np.arange(M)
    # c_m = (2/K) Σ_k f_k cos(m π (k+0.5)/K), with c_0 halved
    Tmk = np.cos(np.pi * np.outer(m, k + 0.5) / K)
    c = (2.0 / K) * (Tmk @ fk)
    c[0] *= 0.5
    return c


@dataclass
class KPMResult:
    energies: np.ndarray
    intensities: np.ndarray     # (n_energies,) S(q,ω)
    n_moments: int
    gamma: float


def kpm_sqw(model, q_cart, energies, fwhm, n_moments=None, tol=0.02,
            cross_section="perp", ion=None, gamma=None):
    """KPM estimate of S(q,ω) at one q, on the given energy grid, Gaussian-broadened
    with the supplied `fwhm`. `n_moments` (or `tol`, which sets M from the bandwidth)
    controls resolution. Returns a KPMResult."""
    from ..numerical import contract_cross_section

    HMat = np.asarray(model.hamiltonian(np.asarray(q_cart, float)), dtype=complex)
    D2 = HMat.shape[0]
    D = D2 // 2
    g = np.concatenate([np.ones(D), -np.ones(D)])
    if gamma is None:
        gamma = spectral_bound(HMat)
    A = HMat / gamma

    energies = np.asarray(energies, float)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    if n_moments is None:
        n_moments = max(int(round(2.0 * gamma / (0.5 * fwhm) * (-np.log10(tol)))), 8)

    v = _observable_vectors(model, q_cart)      # (3, 2D)
    gv = v * g[None, :]                          # g v_b, per observable

    # Chebyshev recursion on β_m^b = T_m(A) (g v_b); moments μ_{m}^{ab} = v_a† β_m^b.
    beta0 = gv.T.copy()                          # (2D, 3)
    beta1 = A @ beta0
    moments = np.zeros((n_moments, 3, 3), dtype=complex)
    moments[0] = v.conj() @ beta0
    if n_moments > 1:
        moments[1] = v.conj() @ beta1
    for m in range(2, n_moments):
        beta2 = 2.0 * (A @ beta1) - beta0
        moments[m] = v.conj() @ beta2
        beta0, beta1 = beta1, beta2

    # Smooth cutoff keeping the +ω_ν poles and dropping their −ω_ν mirror image.
    delta_e = 2.0 * gamma
    smear = max(2.0 * np.sqrt(max(-np.log10(tol), 1.0)) * (delta_e / n_moments), 1e-9)

    def cutoff(E):
        return 0.5 * (np.tanh(E / smear) + 1.0)

    out = np.zeros(len(energies))
    for iw, w in enumerate(energies):
        def f(E, w=w):
            return (np.exp(-0.5 * ((E - w) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
                    * cutoff(E))
        c = _cheb_coefs(f, n_moments, gamma)
        corr = np.tensordot(c, moments, axes=(0, 0)) / getattr(model, "n_cells", 1)
        inten, clamp = contract_cross_section(corr[:, :, None], q_cart, cross_section)
        val = float(np.real(inten[0]))
        out[iw] = max(val, 0.0) if clamp else val

    if ion:
        from ..form_factors import get_form_factor
        out = out * get_form_factor(ion, float(np.linalg.norm(q_cart))) ** 2
    return KPMResult(energies=energies, intensities=out, n_moments=n_moments,
                     gamma=gamma)


def exact_broadened_sqw(model, q_cart, energies, fwhm, cross_section="perp", ion=None):
    """Reference: the engine's exact `structure_factor` (full diagonalization) with the
    same Gaussian broadening on the given grid. The KPM oracle."""
    w, inten = model.structure_factor(q_cart, ion=ion, cross_section=cross_section)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    energies = np.asarray(energies, float)
    out = np.zeros(len(energies))
    for wn, In in zip(w, inten):
        if wn > 0:
            out += In * np.exp(-0.5 * ((energies - wn) / sigma) ** 2) / (
                sigma * np.sqrt(2 * np.pi))
    return out
