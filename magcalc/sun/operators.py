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

def hermitian_basis(n):
    """An orthonormal basis of Hermitian n x n matrices: tr(G_a G_b) = delta_ab.

    E_jj, (E_jk + E_kj)/sqrt2 and i(E_jk - E_kj)/sqrt2 for j < k -- n^2 of them.
    """
    out = []
    for j in range(n):
        E = np.zeros((n, n), dtype=complex)
        E[j, j] = 1.0
        out.append(E)
    r = 1.0 / np.sqrt(2.0)
    for j in range(n):
        for k in range(j + 1, n):
            A = np.zeros((n, n), dtype=complex)
            A[j, k] = A[k, j] = r
            out.append(A)
            B = np.zeros((n, n), dtype=complex)
            B[j, k] = 1j * r
            B[k, j] = -1j * r
            out.append(B)
    return out


def decompose_pair_operator(D, N1, N2, tol=1e-10):
    """Operator Schmidt decomposition of a two-site operator.

    `D` is Hermitian and acts on the product space in the `np.kron(A, B)` convention
    (A on site i, B on site j). Returns [(A_k, B_k), ...] with each factor Hermitian
    and D = sum_k kron(A_k, B_k) exactly. This turns an arbitrary pair coupling into
    the (n_ops_i, n_ops_j) operator-pair form the SU(N) engine already consumes --
    Sunny's `svd_tensor_expansion`.

    Sunny SVDs the reshaped operator directly and then repairs the factors, because a
    generic SVD basis inside a DEGENERATE singular-value subspace comes back
    non-Hermitian (S_i.S_j at s = 1/2 is exactly that case: three equal singular
    values). Here the operator is first expanded in an orthonormal HERMITIAN basis on
    each site, which makes the coefficient matrix REAL; a real SVD then has real
    singular vectors, so both factors are Hermitian by construction and no repair --
    and no matrix square root of a degenerate block -- is needed. Same Schmidt rank,
    fewer ways to be subtly wrong.

    The reconstruction is asserted, not assumed.
    """
    D = np.asarray(D, dtype=complex)
    if D.shape != (N1 * N2, N1 * N2):
        raise ValueError(
            f"pair operator must be {N1 * N2}x{N1 * N2} for N1={N1}, N2={N2}, "
            f"got {D.shape}.")
    if not np.allclose(D, D.conj().T, atol=1e-9):
        raise ValueError("pair operator must be Hermitian.")

    G1, G2 = hermitian_basis(N1), hermitian_basis(N2)
    # Mt[(i1,j1), (i2,j2)] = D[(i1,i2), (j1,j2)]
    Mt = D.reshape(N1, N2, N1, N2).transpose(0, 2, 1, 3).reshape(N1 * N1, N2 * N2)
    g1 = np.array([g.T.reshape(-1) for g in G1])          # (N1^2, N1^2)
    g2 = np.array([g.T.reshape(-1) for g in G2])          # (N2^2, N2^2)
    C = g1 @ Mt @ g2.T
    if np.abs(C.imag).max() > 1e-9 * max(1.0, np.abs(C).max()):
        raise ValueError(
            "pair operator has complex coefficients in a Hermitian basis; it is not "
            "Hermitian to working precision.")
    C = C.real

    U, S, Vt = np.linalg.svd(C)
    scale = max(float(S[0]) if len(S) else 0.0, 1.0)
    terms = []
    for k, sig in enumerate(S):
        if abs(sig) <= tol * scale:
            continue
        A = sig * sum(U[a, k] * G1[a] for a in range(len(G1)))
        B = sum(Vt[k, b] * G2[b] for b in range(len(G2)))
        terms.append((A, B))

    recon = sum((np.kron(a, b) for a, b in terms),
                np.zeros((N1 * N2, N1 * N2), dtype=complex))
    err = float(np.abs(recon - D).max())
    if err > 1e-8 * scale:
        raise ValueError(
            f"pair-operator decomposition does not reconstruct the input "
            f"(max error {err:.2e}).")
    return terms


def dot_product_operator(S1, S2):
    """S_i . S_j on the product space, kron convention."""
    a = spin_matrices(S1)
    b = spin_matrices(S2)
    n1, n2 = a[0].shape[0], b[0].shape[0]
    out = np.zeros((n1 * n2, n1 * n2), dtype=complex)
    for k in range(3):
        out += np.kron(a[k], b[k])
    return out


def pair_operator_from_spec(spec, S1, S2):
    """Build a two-site operator from a config entry.

    Accepted forms:
      {'matrix': [[...]]}          explicit (2S1+1)(2S2+1) square Hermitian matrix
      {'poly': [c0, c1, c2, ...]}  sum_n c_n (S_i . S_j)^n  (c1 = Heisenberg,
                                   c2 = biquadratic, higher n = ring-exchange-like)
    """
    n1, n2 = int(round(2 * S1 + 1)), int(round(2 * S2 + 1))
    if "matrix" in spec:
        return np.asarray(spec["matrix"], dtype=complex), n1, n2
    if "poly" in spec:
        coeffs = list(spec["poly"])
        dot = dot_product_operator(S1, S2)
        out = np.zeros((n1 * n2, n1 * n2), dtype=complex)
        power = np.eye(n1 * n2, dtype=complex)
        for c in coeffs:
            out = out + complex(c) * power
            power = power @ dot
        return out, n1, n2
    raise ValueError(
        f"pair_operator needs `matrix` or `poly`, got keys {sorted(spec)}.")

