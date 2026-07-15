"""Ewald summation of the long-range dipole-dipole interaction.

The dipolar sum is only CONDITIONALLY convergent: a truncated real-space sum depends
on the cutoff and on the (fictitious) sample shape. Ewald splits it into an
exponentially convergent real-space part and a reciprocal-space part, plus an explicit
surface/demagnetization term, and is exact.

What is computed here is the pairwise interaction tensor in reciprocal space,

    A_ij(q)  (3x3 complex),   E = (1/2) sum_ij  mu_i(-q)^T A_ij(q) mu_j(q)

with mu_i = g_i S_i the magnetic moment in Bohr magnetons. Because A_ij(q) is an
infinite lattice sum it CANNOT be written as a finite list of real-space bonds -- it
must be added to the dynamical matrix H(q) directly (see core._ewald_nambu_blocks).

Algorithm and constants follow Sunny 0.8.1 (`precompute_dipole_ewald_at_wavevector`,
src/System/Ewald.jl), against which this is validated (tests/test_ewald.py).

PHASE CONVENTION. Sunny accumulates phases over lattice translations only,
    A^sunny_ij(q) = sum_n T(dr + L n) exp(i q . L n),
whereas pyMagCalc's Fourier transform uses the FULL bond vector (its H(q) carries
exp(i q . (r_j + Ln - r_i))). The two differ by the gauge factor exp(i q . dr) with
dr = r_j - r_i, which is applied here so the result drops straight into pyMagCalc's
H(q). The eigenvalues are identical either way; the eigenvectors (and hence the
intensities) are not, so this must match the host convention.
"""
import logging
from typing import Optional

import numpy as np
from scipy.special import erfc

logger = logging.getLogger(__name__)

# mu0 * muB^2 in meV * Angstrom^3 (no 4pi). Sunny:
#   Units(:meV, :angstrom).vacuum_permeability = 0.6745817653
MU0_MUB2_MEV_A3 = 0.6745817653


def dipole_ewald_at_q(
    latvecs: np.ndarray,
    positions_frac: np.ndarray,
    q_rlu: np.ndarray,
    demag: Optional[np.ndarray] = None,
    accuracy: float = 6.0,
) -> np.ndarray:
    """A_ij(q), shape (n, n, 3, 3), complex, in meV / muB^2.

    latvecs        (3,3) rows are the lattice vectors, Angstrom
    positions_frac (n,3) fractional coordinates
    q_rlu          (3,)  wavevector in reciprocal lattice units
    demag          (3,3) demagnetization tensor, trace 1 (default I/3, i.e. a sphere
                   in vacuum). This is the surface term that makes the conditionally
                   convergent sum well defined; it is what a truncated sum gets wrong.
    accuracy       controls the real/reciprocal cutoffs (6 ~ 13 digits, as in Sunny).
    """
    latvecs = np.asarray(latvecs, dtype=float)
    pos = np.asarray(positions_frac, dtype=float)
    q_rlu = np.asarray(q_rlu, dtype=float)
    n = len(pos)
    if demag is None:
        demag = np.eye(3) / 3.0
    demag = np.asarray(demag, dtype=float)

    # Reciprocal vectors: rows b_i with a_i . b_j = 2 pi delta_ij
    recipvecs = 2.0 * np.pi * np.linalg.inv(latvecs).T
    V = abs(np.linalg.det(latvecs))
    L = V ** (1.0 / 3.0)

    # Splitting parameter: balances the real- and reciprocal-space costs.
    sigma = L / 3.0
    sigma2 = sigma * sigma
    sigma3 = sigma2 * sigma
    rmax = accuracy * np.sqrt(2.0) * sigma
    kmax = accuracy * np.sqrt(2.0) / sigma

    # How many images are needed along each axis to cover rmax / kmax.
    nmax = [int(round(rmax / abs(np.dot(latvecs[i], recipvecs[i] /
                                        np.linalg.norm(recipvecs[i]))))) + 1
            for i in range(3)]
    mmax = [int(round(kmax / abs(np.dot(recipvecs[i], latvecs[i] /
                                        np.linalg.norm(latvecs[i])))))
            for i in range(3)]

    cart = pos @ latvecs
    q_cart = q_rlu @ recipvecs
    # Fold q into the first BZ so the reciprocal window stays centred (Sunny does the
    # same with `q - round(q)`); summing over all m makes this an identity.
    q_fold = q_rlu - np.round(q_rlu)
    q_fold_cart = q_fold @ recipvecs

    n_range = [np.arange(-nmax[i], nmax[i] + 1) for i in range(3)]
    m_range = [np.arange(-mmax[i], mmax[i] + 1) for i in range(3)]
    n_cells = np.array([[a, b, c] for a in n_range[0] for b in n_range[1]
                        for c in n_range[2]], dtype=float)
    m_cells = np.array([[a, b, c] for a in m_range[0] for b in m_range[1]
                        for c in m_range[2]], dtype=float)
    Ln = n_cells @ latvecs                    # (Nn, 3) real-space images
    Km = m_cells @ recipvecs                  # (Nm, 3) reciprocal images

    A = np.zeros((n, n, 3, 3), dtype=complex)
    I3 = np.eye(3)

    for i in range(n):
        for j in range(n):
            dr = cart[j] - cart[i]
            acc = np.zeros((3, 3), dtype=complex)

            # ---------------- real space (erfc-damped, short ranged) -------------
            rvec = dr + Ln                                   # (Nn, 3)
            r2 = np.einsum("ij,ij->i", rvec, rvec)
            sel = (r2 > 1e-14) & (r2 <= rmax * rmax)
            if np.any(sel):
                rv = rvec[sel]
                r = np.sqrt(r2[sel])
                r3 = r2[sel] * r
                rhat = rv / r[:, None]
                erfc0 = erfc(r / (np.sqrt(2.0) * sigma))
                gauss0 = np.sqrt(2.0 / np.pi) * (r / sigma) * np.exp(-r2[sel] / (2 * sigma2))
                # phase over LATTICE TRANSLATIONS only (Sunny's convention)
                phase = np.exp(1j * (Ln[sel] @ q_cart))
                term = (I3[None, :, :] / r3[:, None, None]) * (erfc0 + gauss0)[:, None, None]
                rr = np.einsum("ka,kb->kab", rhat, rhat)
                term -= 3.0 * rr / r3[:, None, None] * (
                    erfc0 + (1.0 + r2[sel] / (3 * sigma2)) * gauss0)[:, None, None]
                acc += (1.0 / (4 * np.pi)) * np.einsum("k,kab->ab", phase, term)

            # ---------------- reciprocal space (long ranged) ---------------------
            k = Km + q_fold_cart                              # (Nm, 3)
            k2 = np.einsum("ij,ij->i", k, k)
            small = k2 <= 1e-16
            if np.any(small):
                # k = 0: the SURFACE term. Net magnetisation lives in this mode, and
                # the demagnetisation tensor is what makes the conditionally
                # convergent dipolar sum well defined. A truncated real-space sum has
                # no such term -- this is exactly what it gets wrong.
                acc += demag / V
            ok = (~small) & (k2 <= kmax * kmax)
            if np.any(ok):
                kk = k[ok]
                k2o = k2[ok]
                phase = np.exp(-1j * (kk @ dr))
                pref = np.exp(-sigma2 * k2o / 2.0) / k2o
                acc += (1.0 / V) * np.einsum(
                    "k,k,kab->ab", phase, pref, np.einsum("ka,kb->kab", kk, kk))

            # ---------------- remove the self energy ------------------------------
            if i == j:
                acc += -I3 / (3.0 * (2.0 * np.pi) ** 1.5 * sigma3)

            # gauge: Sunny phases over lattice translations, pyMagCalc over the FULL
            # bond vector. They differ by exp(i q . dr).
            A[i, j] = acc * np.exp(1j * float(np.dot(q_cart, dr)))

    return A * MU0_MUB2_MEV_A3


def exchange_from_A(A: np.ndarray, g_factors: np.ndarray) -> np.ndarray:
    """Bond matrices J_ij(q) = g_i A_ij(q) g_j acting on SPINS (not moments).

    pyMagCalc's energy is E = (1/2) sum_ordered S_i^T J_ij S_j, and the dipolar energy
    is E = (1/2) sum_ordered mu_i^T A_ij mu_j with mu = g S, so J = g_i A g_j directly.
    g_factors may be scalars per site (isotropic g) or (n,3,3) tensors.
    """
    n = A.shape[0]
    g = np.asarray(g_factors, dtype=float)
    if g.ndim == 1:
        g = np.einsum("i,ab->iab", g, np.eye(3))
    J = np.zeros_like(A)
    for i in range(n):
        for j in range(n):
            J[i, j] = g[i].T @ A[i, j] @ g[j]
    return J
