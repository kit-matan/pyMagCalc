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

WHAT IS COMPUTED ONCE (2026-10-27). The real-space part depends on q only through
exp(i q . L) over the lattice images L: which images fall inside the cutoff, and the
3x3 tensor each contributes to each pair, are q-INDEPENDENT. `DipoleEwald` builds them
once per cell as a sparse (n*n*9, N_images) matrix, so each q costs one sparse
matrix-vector product. The reciprocal part does depend on q (k = G + q), but its pair
dependence factorizes, exp(-i k.(r_j - r_i)) = conj(P_ki) P_kj with P = exp(-i k.r), so
it is a few dense (n x N_k)(N_k x n) products instead of a Python loop over pairs. The
arithmetic is the same as the pair loop it replaces, term for term; only the summation
order differs. It matches Sunny's own `precompute_dipole_ewald_at_wavevector` element by
element to ~1e-15 relative (tests/test_ewald_sunny_tensor.py).
"""
import logging
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.special import erfc

logger = logging.getLogger(__name__)

# mu0 * muB^2 in meV * Angstrom^3 (no 4pi). Sunny:
#   Units(:meV, :angstrom).vacuum_permeability = 0.6745817653
from .constants import MU0_MUB2_MEV_A3  # noqa: F401  (re-exported; see constants.py)

_I3 = np.eye(3)


class DipoleEwald:
    """The Ewald dipolar tensor A_ij(q) for one magnetic cell, precomputed once.

    latvecs        (3,3) rows are the lattice vectors, Angstrom
    positions_frac (n,3) fractional coordinates
    demag          (3,3) demagnetization tensor, trace 1 (default I/3, i.e. a sphere
                   in vacuum). This is the surface term that makes the conditionally
                   convergent sum well defined; it is what a truncated sum gets wrong.
    accuracy       controls the real/reciprocal cutoffs (6 ~ 13 digits, as in Sunny).

    `at_q(q_rlu)` returns A_ij(q), shape (n, n, 3, 3), complex, in meV / muB^2. Build
    one per cell and reuse it: construction holds all the q-independent work.
    """

    def __init__(self, latvecs, positions_frac, demag=None, accuracy: float = 6.0):
        latvecs = np.asarray(latvecs, dtype=float)
        pos = np.asarray(positions_frac, dtype=float)
        self.n = n = len(pos)
        self.demag = (np.eye(3) / 3.0 if demag is None
                      else np.asarray(demag, dtype=float))

        # Reciprocal vectors: rows b_i with a_i . b_j = 2 pi delta_ij
        self.recipvecs = recipvecs = 2.0 * np.pi * np.linalg.inv(latvecs).T
        self.V = V = abs(np.linalg.det(latvecs))
        L = V ** (1.0 / 3.0)

        # Splitting parameter: balances the real- and reciprocal-space costs.
        sigma = L / 3.0
        self.sigma2 = sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        rmax = accuracy * np.sqrt(2.0) * sigma
        kmax = accuracy * np.sqrt(2.0) / sigma
        self.kmax2 = kmax * kmax

        # How many images are needed along each axis to cover rmax / kmax.
        nmax = [int(round(rmax / abs(np.dot(latvecs[i], recipvecs[i] /
                                            np.linalg.norm(recipvecs[i]))))) + 1
                for i in range(3)]
        mmax = [int(round(kmax / abs(np.dot(recipvecs[i], latvecs[i] /
                                            np.linalg.norm(latvecs[i])))))
                for i in range(3)]
        n_range = [np.arange(-nmax[i], nmax[i] + 1) for i in range(3)]
        m_range = [np.arange(-mmax[i], mmax[i] + 1) for i in range(3)]
        n_cells = np.array([[a, b, c] for a in n_range[0] for b in n_range[1]
                            for c in n_range[2]], dtype=float)
        m_cells = np.array([[a, b, c] for a in m_range[0] for b in m_range[1]
                            for c in m_range[2]], dtype=float)
        self.Ln = Ln = n_cells @ latvecs          # (Nn, 3) real-space images
        self.Km = m_cells @ recipvecs             # (Nm, 3) reciprocal images
        self.cart = cart = pos @ latvecs          # (n, 3)

        # ---- real space (erfc-damped, short ranged): q-independent tensors ------
        # Row (i*n + j)*9 + 3a + b, column l: the (a,b) component of pair (i,j)'s
        # image l. Built one source site at a time, so memory is O(n * Nn), not n^2.
        rows, cols, data = [], [], []
        ab = np.arange(9)
        for i in range(n):
            rvec = (cart - cart[i])[:, None, :] + Ln[None, :, :]        # (n, Nn, 3)
            r2 = np.einsum("jlx,jlx->jl", rvec, rvec)
            sel = (r2 > 1e-14) & (r2 <= rmax * rmax)
            jj, ll = np.nonzero(sel)
            if jj.size == 0:
                continue
            rv = rvec[jj, ll]
            r2s = r2[jj, ll]
            r = np.sqrt(r2s)
            r3 = r2s * r
            rhat = rv / r[:, None]
            erfc0 = erfc(r / (np.sqrt(2.0) * sigma))
            gauss0 = np.sqrt(2.0 / np.pi) * (r / sigma) * np.exp(-r2s / (2 * sigma2))
            term = (_I3[None, :, :] / r3[:, None, None]) * (erfc0 + gauss0)[:, None, None]
            rr = np.einsum("ka,kb->kab", rhat, rhat)
            term -= 3.0 * rr / r3[:, None, None] * (
                erfc0 + (1.0 + r2s / (3 * sigma2)) * gauss0)[:, None, None]
            term *= 1.0 / (4 * np.pi)
            pair = i * n + jj
            rows.append((pair[:, None] * 9 + ab[None, :]).ravel())
            cols.append(np.repeat(ll, 9))
            data.append(term.reshape(-1, 9).ravel())
        if rows:
            self._real = csr_matrix((np.concatenate(data),
                                     (np.concatenate(rows), np.concatenate(cols))),
                                    shape=(n * n * 9, len(Ln)))
        else:
            self._real = None

        # ---- remove the self energy (q-independent, diagonal) ---------------------
        self._self = -_I3 / (3.0 * (2.0 * np.pi) ** 1.5 * sigma3)

    def at_q(self, q_rlu) -> np.ndarray:
        n = self.n
        q_rlu = np.asarray(q_rlu, dtype=float)
        q_cart = q_rlu @ self.recipvecs
        A = np.zeros((n, n, 3, 3), dtype=complex)

        # real space: phase over LATTICE TRANSLATIONS only (Sunny's convention)
        if self._real is not None:
            phase = np.exp(1j * (self.Ln @ q_cart))
            A += (self._real @ phase).reshape(n, n, 3, 3)

        # reciprocal space (long ranged). Fold q into the first BZ so the window stays
        # centred (Sunny does the same with `q - round(q)`); summing over all m makes
        # this an identity.
        q_fold_cart = (q_rlu - np.round(q_rlu)) @ self.recipvecs
        k = self.Km + q_fold_cart                                   # (Nm, 3)
        k2 = np.einsum("ij,ij->i", k, k)
        small = k2 <= 1e-16
        if np.any(small):
            # k = 0: the SURFACE term. Net magnetisation lives in this mode, and the
            # demagnetisation tensor is what makes the conditionally convergent
            # dipolar sum well defined. A truncated real-space sum has no such term --
            # this is exactly what it gets wrong.
            A += self.demag / self.V
        ok = (~small) & (k2 <= self.kmax2)
        if np.any(ok):
            kk = k[ok]
            k2o = k2[ok]
            pref = np.exp(-self.sigma2 * k2o / 2.0) / k2o / self.V      # (Nk,)
            P = np.exp(-1j * (kk @ self.cart.T))                        # (Nk, n)
            Pc = P.conj().T                                             # (n, Nk)
            for a in range(3):
                for b in range(a, 3):
                    blk = Pc @ (P * (pref * kk[:, a] * kk[:, b])[:, None])   # (n, n)
                    A[:, :, a, b] += blk
                    if b != a:
                        A[:, :, b, a] += blk

        idx = np.arange(n)
        A[idx, idx] += self._self

        # gauge: Sunny phases over lattice translations, pyMagCalc over the FULL bond
        # vector. They differ by exp(i q . dr), dr = r_j - r_i.
        rq = self.cart @ q_cart
        A *= np.exp(1j * (rq[None, :] - rq[:, None]))[:, :, None, None]
        return A * MU0_MUB2_MEV_A3


def dipole_ewald_at_q(
    latvecs: np.ndarray,
    positions_frac: np.ndarray,
    q_rlu: np.ndarray,
    demag: Optional[np.ndarray] = None,
    accuracy: float = 6.0,
) -> np.ndarray:
    """A_ij(q), shape (n, n, 3, 3), complex, in meV / muB^2, at ONE q.

    A one-off convenience: it builds a `DipoleEwald` and discards it. For more than one
    q on the same cell, build the `DipoleEwald` once and call `at_q` — the build holds
    all the q-independent work. Arguments as in `DipoleEwald`.
    """
    return DipoleEwald(latvecs, positions_frac, demag=demag, accuracy=accuracy).at_q(q_rlu)


def exchange_from_A(A: np.ndarray, g_factors: np.ndarray) -> np.ndarray:
    """Bond matrices J_ij(q) = g_i^T A_ij(q) g_j acting on SPINS (not moments).

    pyMagCalc's energy is E = (1/2) sum_ordered S_i^T J_ij S_j, and the dipolar energy
    is E = (1/2) sum_ordered mu_i^T A_ij mu_j with mu = g S, so J = g_i A g_j directly.
    g_factors may be scalars per site (isotropic g) or (n,3,3) tensors.
    """
    g = np.asarray(g_factors, dtype=float)
    if g.ndim == 1:
        return A * (g[:, None] * g[None, :])[:, :, None, None]
    return np.einsum("iba,ijbc,jcd->ijad", g, A, g)
