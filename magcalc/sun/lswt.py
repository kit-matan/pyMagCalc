"""SU(N) (generalized) spin-wave theory.

Each site carries an N-level local Hilbert space and a coherent state |Z_i>. Choosing a
local basis U_i whose first column is Z_i, an on-site operator A becomes

    A~ = U~ A U,   and, to quadratic order in the N-1 bosons b_m,

    A  ~  A~_00 (1 - sum_m b~_m b_m)
        + sum_m ( A~_0m b_m + A~_m0 b~_m )
        + sum_mn A~_mn b~_m b_n

so that the ONE-boson (linear) part is A~_0m, and the QUADRATIC part is
Q_mn = A~_mn - delta_mn A~_00.

For a bond term  sum_ab J_ab A^a_i B^b_j  the quadratic-in-boson piece is

    <A^a>_i * Q[B^b]_j  +  <B^b>_j * Q[A^a]_i        (on-site "mean field", no phase)
  + (linear at i) x (linear at j)                     (hopping + anomalous, with phase)

which is exactly the dipole construction generalised from 1 boson to N-1.

CONVENTIONS (both verified against the host, see tests):
  * bonds are listed in BOTH directions, and there is NO 1/2 on the hopping -- this is
    how pyMagCalc encodes H = (1/2) sum_ordered;
  * the on-site mean-field term is the q=0 sum (NO phase);
  * the returned H(q) is g*H2, i.e. the Bogoliubov metric is already folded in (the
    host stores it that way -- see core._ewald_nambu), so its eigenvalues come in
    +/- omega pairs.

For S=1/2 (N=2) there is exactly one boson per site and this reduces IDENTICALLY to
dipole LSWT -- the load-bearing test.
"""
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .operators import coherent_from_direction, local_basis, spin_matrices


class SUNModel:
    """A minimal SU(N) LSWT model.

    sites   : list of (S, Z) -- Z is the coherent state (N-vector). Use
              `from_directions` to build the dipole-equivalent reference state.
    bonds   : list of (i, j, dr_cart, J) with J a real 3x3; BOTH directions must be
              listed (as everywhere else in pyMagCalc).
    onsite  : list of (i, A) with A an N_i x N_i Hermitian matrix (crystal field).
    """

    def __init__(
        self,
        spins: Sequence[float],
        coherent_states: Sequence[np.ndarray],
        bonds: Sequence[Tuple[int, int, np.ndarray, np.ndarray]],
        onsite: Optional[Sequence[Tuple[int, np.ndarray]]] = None,
    ):
        self.S = [float(s) for s in spins]
        self.Z = [np.asarray(z, dtype=complex) for z in coherent_states]
        self.bonds = [(int(i), int(j), np.asarray(dr, float), np.asarray(J, float))
                      for (i, j, dr, J) in bonds]
        self.onsite = [(int(i), np.asarray(A, dtype=complex))
                       for (i, A) in (onsite or [])]

        self.L = len(self.S)
        self.Ns = [int(round(2 * s)) + 1 for s in self.S]
        if len(set(self.Ns)) != 1:
            raise NotImplementedError(
                "SU(N) currently requires all sites to have the same N (same spin S).")
        self.N = self.Ns[0]
        self.M = self.N - 1                      # bosons per site
        self._prepare()

    # ---------------------------------------------------------------- setup
    @classmethod
    def from_directions(cls, spins, directions, bonds, onsite=None):
        """Reference state = the spin coherent state pointing along `directions` -- the
        SU(N) state that corresponds to the classical dipole."""
        Z = [coherent_from_direction(s, d) for s, d in zip(spins, directions)]
        return cls(spins, Z, bonds, onsite)

    def _prepare(self):
        """Rotate every local operator into the |Z> basis and cache the pieces."""
        L, M = self.L, self.M
        self.s0 = np.zeros((L, 3), dtype=complex)          # <Z|S^a|Z>
        self.t = np.zeros((L, 3, M), dtype=complex)        # <0|S^a|m>  (coeff of b_m)
        self.tb = np.zeros((L, 3, M), dtype=complex)       # <m|S^a|0>  (coeff of b_m^dag)
        self.Q = np.zeros((L, 3, M, M), dtype=complex)     # A~_mn - delta A~_00
        self.QA = np.zeros((L, M, M), dtype=complex)       # on-site anisotropy

        for i in range(L):
            U = local_basis(self.Z[i])
            Sxyz = spin_matrices(self.S[i])
            for a in range(3):
                St = U.conj().T @ Sxyz[a] @ U
                self.s0[i, a] = St[0, 0]
                self.t[i, a, :] = St[0, 1:]
                self.tb[i, a, :] = St[1:, 0]
                self.Q[i, a] = St[1:, 1:] - np.eye(M) * St[0, 0]

        for (i, A) in self.onsite:
            U = local_basis(self.Z[i])
            At = U.conj().T @ A @ U
            self.QA[i] += At[1:, 1:] - np.eye(M) * At[0, 0]

    # ---------------------------------------------------------------- energy
    def classical_energy(self) -> float:
        """E = (1/2) sum_ordered <S_i> J <S_j> + sum_i <A_i>."""
        E = 0.0
        for (i, j, dr, J) in self.bonds:
            E += 0.5 * float(np.real(self.s0[i] @ J @ self.s0[j]))
        for (i, A) in self.onsite:
            Z = self.Z[i]
            E += float(np.real(Z.conj() @ A @ Z))
        return E

    # ---------------------------------------------------------------- H(q)
    def hamiltonian(self, q_cart: np.ndarray) -> np.ndarray:
        """g * H2(q), shape (2 L M, 2 L M). Eigenvalues come in +/- omega pairs."""
        L, M = self.L, self.M
        D = L * M
        H11 = np.zeros((D, D), dtype=complex)
        H22 = np.zeros((D, D), dtype=complex)
        H12 = np.zeros((D, D), dtype=complex)
        H21 = np.zeros((D, D), dtype=complex)
        q = np.asarray(q_cart, dtype=float)

        for (i, j, dr, J) in self.bonds:
            ph = np.exp(1j * float(np.dot(q, dr)))
            bi, bj = i * M, j * M
            for a in range(3):
                for b in range(3):
                    c = J[a, b]
                    if c == 0.0:
                        continue
                    ti, tbi = self.t[i, a], self.tb[i, a]
                    tj, tbj = self.t[j, b], self.tb[j, b]
                    # inter-site: hopping and anomalous (carry the phase)
                    H11[bi:bi + M, bj:bj + M] += c * np.outer(tbi, tj) * ph
                    H22[bi:bi + M, bj:bj + M] += c * np.outer(ti, tbj) * ph
                    H12[bi:bi + M, bj:bj + M] += c * np.outer(tbi, tbj) * ph
                    H21[bi:bi + M, bj:bj + M] += c * np.outer(ti, tj) * ph
                    # on-site mean field at i, weighted by <S^b>_j. NO phase: it is the
                    # q=0 sum. (Putting the phase here makes a ferromagnet's H(q)
                    # cancel to zero -- the same trap as in the dipole engine.)
                    mf = c * self.Q[i, a] * self.s0[j, b]
                    H11[bi:bi + M, bi:bi + M] += mf
                    H22[bi:bi + M, bi:bi + M] += mf.T

        # single-ion / crystal field
        for i in range(L):
            if np.any(self.QA[i]):
                bi = i * M
                H11[bi:bi + M, bi:bi + M] += self.QA[i]
                H22[bi:bi + M, bi:bi + M] += self.QA[i].T

        g = np.diag([1.0] * D + [-1.0] * D)
        return g @ np.block([[H11, H12], [H21, H22]])

    def dispersion(self, q_cart: np.ndarray) -> np.ndarray:
        """The L*M positive magnon energies at q (ascending)."""
        ev = np.linalg.eigvals(self.hamiltonian(q_cart))
        w = np.sort(np.real(ev))[self.L * self.M:]
        return np.sort(w)

    def max_imaginary(self, q_cart: np.ndarray) -> float:
        ev = np.linalg.eigvals(self.hamiltonian(q_cart))
        return float(np.max(np.abs(np.imag(ev))))
