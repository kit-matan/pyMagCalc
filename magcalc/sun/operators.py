"""Local operator algebra for SU(N) spin-wave theory.

Everything here is an N x N MATRIX in the spin-S representation (N = 2S+1). Note this
is a different object from `magcalc/stevens.py`, which holds the CLASSICAL (large-s)
Stevens polynomials used by dipole LSWT -- those are the s -> inf limit and are not what
SU(N) needs.

Basis convention: |m> ordered by descending S^z eigenvalue, i.e. m = 0 is the S^z = +S
state. This matches Sunny's `spin_matrices(s)`.
"""
from typing import Dict, Tuple

import numpy as np


def spin_matrices(S: float) -> np.ndarray:
    """(3, N, N) complex array: S^x, S^y, S^z in the spin-S representation.

    Basis |m> ordered by DESCENDING S^z (m = 0 is S^z = +S), as in Sunny.
    """
    if abs(2 * S - round(2 * S)) > 1e-9 or S < 0:
        raise ValueError(f"S must be a non-negative (half-)integer, got {S}.")
    N = int(round(2 * S)) + 1
    m = np.array([S - k for k in range(N)])          # descending: S, S-1, ..., -S

    Sz = np.diag(m).astype(complex)
    # S^+ |S, m> = sqrt(S(S+1) - m(m+1)) |S, m+1>. With descending order, raising m by 1
    # moves to the PREVIOUS index.
    Sp = np.zeros((N, N), dtype=complex)
    for k in range(1, N):
        mm = m[k]                                     # the lower state
        Sp[k - 1, k] = np.sqrt(S * (S + 1) - mm * (mm + 1))
    Sm = Sp.conj().T

    Sx = (Sp + Sm) / 2.0
    Sy = (Sp - Sm) / (2.0j)
    return np.stack([Sx, Sy, Sz])


def _stevens_from_spin(S: float, k: int, q: int) -> np.ndarray:
    """O_k^q as an N x N matrix, built from the operator-equivalent definition.

    Uses the standard construction in terms of the ladder operators:
        O_k^q  = (1/2) [ T_k^q + (T_k^q)^dagger ]        for q > 0
        O_k^0  = the polynomial in S^z
        O_k^-q = (1/(2i)) [ T_k^q - (T_k^q)^dagger ]     for q > 0
    Rather than hard-code the (long) table, the operators are obtained by symmetrising
    the corresponding CLASSICAL polynomial over all orderings of the (non-commuting)
    spin matrices. That is exactly the operator-equivalent prescription, and it is
    checked against Sunny's `stevens_matrices(s)` in the tests.
    """
    from itertools import permutations

    import sympy as sp

    from ..stevens import stevens_polynomial

    Sxyz = spin_matrices(S)
    N = Sxyz.shape[1]

    x, y, z = sp.symbols("Sx Sy Sz", commutative=True)
    poly = sp.expand(stevens_polynomial(k, q, x, y, z))

    out = np.zeros((N, N), dtype=complex)
    for term in poly.as_ordered_terms():
        coeff, monom = term.as_coeff_Mul()
        powers = monom.as_powers_dict()
        letters = []
        for sym, idx in ((x, 0), (y, 1), (z, 2)):
            letters += [idx] * int(powers.get(sym, 0))
        if not letters:
            out += complex(coeff) * np.eye(N)
            continue
        # Full symmetrisation over orderings: the operator equivalent of a classical
        # monomial is the symmetrised product of the corresponding matrices.
        perms = set(permutations(letters))
        acc = np.zeros((N, N), dtype=complex)
        for p in perms:
            prod = np.eye(N, dtype=complex)
            for idx in p:
                prod = prod @ Sxyz[idx]
            acc += prod
        out += complex(coeff) * acc / len(perms)
    return out


_STEVENS_CACHE: Dict[Tuple[float, int, int], np.ndarray] = {}


def stevens_matrices(S: float, k: int, q: int) -> np.ndarray:
    """Stevens operator O_k^q as an N x N matrix (k in {2,4,6}, -k <= q <= k)."""
    key = (float(S), int(k), int(q))
    if key not in _STEVENS_CACHE:
        _STEVENS_CACHE[key] = _stevens_from_spin(S, k, q)
    return _STEVENS_CACHE[key]


def coherent_from_direction(S: float, direction: np.ndarray) -> np.ndarray:
    """The spin coherent state |Z> whose expectation <Z|S|Z> points along `direction`
    with |<S>| = S -- i.e. the classical dipole state, as an N-vector.

    This is the SU(N) reference state that reproduces dipole LSWT. It is the maximal-
    weight eigenvector of  n . S  (eigenvalue +S).
    """
    n = np.asarray(direction, dtype=float)
    nrm = np.linalg.norm(n)
    if nrm < 1e-12:
        raise ValueError("direction must be non-zero")
    n = n / nrm
    Sxyz = spin_matrices(S)
    nS = n[0] * Sxyz[0] + n[1] * Sxyz[1] + n[2] * Sxyz[2]
    w, v = np.linalg.eigh(nS)
    Z = v[:, int(np.argmax(w))]              # eigenvalue +S
    # fix the global phase (irrelevant physically, but keeps things reproducible)
    k = int(np.argmax(np.abs(Z)))
    Z = Z * np.exp(-1j * np.angle(Z[k]))
    return Z / np.linalg.norm(Z)


def local_basis(Z: np.ndarray) -> np.ndarray:
    """Unitary U whose FIRST column is the coherent state Z; the remaining N-1 columns
    span the excited local levels (the SU(N) boson flavours)."""
    Z = np.asarray(Z, dtype=complex)
    N = len(Z)
    M = np.zeros((N, N), dtype=complex)
    M[:, 0] = Z / np.linalg.norm(Z)
    # complete to an orthonormal basis
    Q, _ = np.linalg.qr(np.column_stack([M[:, 0], np.eye(N)]))
    U = np.zeros((N, N), dtype=complex)
    U[:, 0] = M[:, 0]
    cols = 1
    for k in range(Q.shape[1]):
        v = Q[:, k]
        v = v - U[:, :cols] @ (U[:, :cols].conj().T @ v)
        nv = np.linalg.norm(v)
        if nv > 1e-8 and cols < N:
            U[:, cols] = v / nv
            cols += 1
        if cols == N:
            break
    return U
