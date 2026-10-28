"""1/S (LSWT) corrections: zero-point energy and ordered-moment reduction.

Linear spin-wave theory is the leading term of a 1/S expansion. The two standard
next-order quantities are:

  * the zero-point energy -- quantum fluctuations lower the ground-state energy below
    the classical value;
  * the ordered-moment reduction -- <S^z_i> = S - dS_i, where dS_i = <a^dag_i a_i> is the
    boson density in the magnon vacuum.

Both are computed here from the SAME dynamical matrix H(q) the dispersion uses, so they
inherit its (validated) conventions. pyMagCalc stores HMat(q) = g * H2(q) with H2
Hermitian and g = diag(I_N, -I_N); its eigenvalues are the +/- omega pairs. Writing the
quadratic Hamiltonian as

    H = E_cl + (1/2) sum_k Psi^dag_k H2(k) Psi_k,   Psi = (a_1..a_N, a^dag_1..a^dag_N),

normal-ordering the (1/2)Psi^dag H2 Psi gives a c-number -(1/2) tr A(k), A = H2[:N,:N], so
the magnon-vacuum energy is

    E_0 = E_cl + (1 / 2 N_k) sum_k [ sum_nu omega_nu(k) - tr A(k) ].

The moment reduction uses the para-unitary (Colpa) Bogoliubov transform T, Psi = T Phi
with Phi = (beta, beta^dag):

    dS_i = <a^dag_i a_i> = (1/N_k) sum_k sum_nu | T[i, N+nu](k) |^2.

Validated against Sunny 0.8.1 (energy_per_site_lswt_correction,
magnetization_lswt_correction_dipole) AND the textbook S=1/2 square-lattice Heisenberg
antiferromagnet: dE = -0.157947 J/site, dS = 0.19657. See tests/test_corrections.py.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy import linalg as la

logger = logging.getLogger(__name__)

_COLPA_JITTER = 1e-9         # tiny diagonal shift if H2 is only semi-definite
_HERMITIAN_TOL = 1e-10       # relative; a less Hermitian H2 takes the per-k loop


@dataclass
class LSWTCorrections:
    energy_correction_per_site: float      # dE_1/S  (add to the classical energy)
    moment_reduction: np.ndarray           # dS_i, one per magnetic-cell site
    n_kpoints: int


def _colpa(H2: np.ndarray):
    """Para-unitary (Colpa) diagonalisation of a bosonic BdG matrix.

    H2 is 2N x 2N Hermitian and (ideally) positive definite. Returns (omega, T) with the
    N positive quasiparticle energies and the transform Psi = T Phi, Phi = (beta, beta^dag).
    """
    n2 = H2.shape[0]
    N = n2 // 2
    g = np.diag([1.0] * N + [-1.0] * N)
    H2 = 0.5 * (H2 + H2.conj().T)
    try:
        K = la.cholesky(H2)                 # H2 = K^dag K, K upper-triangular
    except la.LinAlgError:
        K = la.cholesky(H2 + _COLPA_JITTER * np.eye(n2))
    W = K @ g @ K.conj().T
    W = 0.5 * (W + W.conj().T)
    vals, U = la.eigh(W)                     # ascending
    # Colpa ordering: the N positive-omega modes first (descending), then N negative.
    order = np.concatenate([np.argsort(-vals)[:N], np.argsort(vals)[:N]])
    vals, U = vals[order], U[:, order]
    omega = np.abs(vals[:N])
    E = np.diag(np.sqrt(np.abs(vals)))
    T = la.inv(K) @ U @ E
    return omega, T


def _chunk_sums_per_k(h_stack: np.ndarray, N: int):
    """The original per-k loop, kept for any chunk the batched path cannot take.

    Runs a general eigensolve at every k (to report |Im w|) and the Colpa transform
    with its jitter fallback, exactly as `compute_corrections` always did.
    """
    g = np.diag([1.0] * N + [-1.0] * N)
    sum_zero, dens = 0.0, np.zeros(N)
    n_used = n_failed = 0
    max_imag = 0.0
    for H in h_stack:
        ev = la.eigvals(H)
        max_imag = max(max_imag, float(np.max(np.abs(np.imag(ev)))))
        H2 = g @ H
        A = H2[:N, :N]
        try:
            omega, T = _colpa(H2)
        except la.LinAlgError:
            n_failed += 1                  # H2 not positive definite at this k
            continue
        sum_zero += float(np.sum(omega) - np.real(np.trace(A)))
        # <a^dag_i a_i> = sum_nu |T[i, N+nu]|^2
        dens += np.sum(np.abs(T[:N, N:]) ** 2, axis=1)
        n_used += 1
    return sum_zero, dens, n_used, n_failed, max_imag


def _chunk_sums(h_stack: np.ndarray, N: int):
    """Zero-point and boson-density sums over a chunk of k-points, batched.

    The same Colpa transform as `_colpa`, written for a stack. With H2 = L L^dag
    (L lower triangular, i.e. K = L^dag) and W = L^dag g L = U diag(vals) U^dag:

      * W has exactly N positive and N negative eigenvalues when H2 is positive
        definite (Sylvester), so `eigh`'s ascending order puts the omega_nu last;
      * the beta^dag columns of T = K^-1 U E are L^-dag U[:, :N] sqrt(-vals[:N]),
        a triangular solve; only their first N rows enter <a^dag_i a_i>.

    No general eigensolve is needed: g H2 = g K^dag K is similar to the Hermitian
    K g K^dag = W, so where the Cholesky succeeds its eigenvalues are exactly real
    and |Im w| is zero. The only k-points that can carry an imaginary magnon are
    the ones whose Cholesky fails, and a failure anywhere in the chunk sends the
    whole chunk through `_chunk_sums_per_k`, the original loop.
    """
    gv = np.concatenate([np.ones(N), -np.ones(N)])
    H2 = h_stack * gv[None, :, None]
    H2h = np.conj(np.swapaxes(H2, 1, 2))
    # The similarity argument needs H2 itself to be Hermitian; the per-k loop ran
    # its eigensolve on the raw, unsymmetrized H. Should H2 not be, take that loop.
    if np.max(np.abs(H2 - H2h)) > _HERMITIAN_TOL * max(1.0, float(np.max(np.abs(H2)))):
        return _chunk_sums_per_k(h_stack, N)
    H2 = 0.5 * (H2 + H2h)
    try:
        L = np.linalg.cholesky(H2)
    except np.linalg.LinAlgError:
        return _chunk_sums_per_k(h_stack, N)
    Lh = np.conj(np.swapaxes(L, 1, 2))
    W = (Lh * gv[None, None, :]) @ L
    W = 0.5 * (W + np.conj(np.swapaxes(W, 1, 2)))
    vals, U = np.linalg.eigh(W)
    Tn = np.linalg.solve(Lh, U[:, :, :N] * np.sqrt(np.abs(vals[:, None, :N])))
    tr = np.real(np.trace(H2[:, :N, :N], axis1=1, axis2=2))
    sum_zero = float(np.sum(np.abs(vals[:, N:])) - np.sum(tr))
    dens = np.sum(np.abs(Tn[:, :N, :]) ** 2, axis=(0, 2))
    return sum_zero, dens, len(h_stack), 0, 0.0


def compute_corrections(
    calc,
    k_mesh=(8, 8, 8),
    tol_imag: float = 1e-3,
    n_threads=None,
    chunk_size=None,
) -> LSWTCorrections:
    """1/S corrections for a built MagCalc `calc`.

    k_mesh: Monkhorst-Pack-style grid size per reciprocal axis. Only axes with a real
    (non-flat) reciprocal length are sampled -- a chain samples 1-D, a plane 2-D -- so a
    tuple like (24, 24, 1) is honoured directly and a decoupled axis contributes 1 point.

    MIXED SPINS need nothing here, which is why the `spin_magnitudes` argument this
    used to accept (and never read) is gone. `S` below is the REFERENCE spin, and it
    is the right thing to pass to `_build_h_stack`: gen_HM writes site i's
    Holstein-Primakoff expansion as ratio_i * S_sym, so the per-site lengths are
    already inside H(q). The two outputs are then per-site quantities of the
    Bogoliubov transform itself -- the moment reduction is the boson density
    <a^dag_i a_i>, which carries no explicit S at all. Checked against the
    decoupled-sublattice identity: an S=1 and an S=1/2 chain in one cell return
    exactly the union of their separate results.

    The mesh is processed in chunks of `chunk_size` k-points (default: about 8 MB
    of H(q) each), batched through `_chunk_sums` on `n_threads` threads (default:
    every core). Neither changes the answer beyond rounding: the per-chunk sums are
    combined in mesh order.
    """
    from .core import reciprocal_b_matrix

    if calc.HMat_sym is None:
        raise ValueError("MagCalc must be initialised (HMat_sym is None).")

    N = int(calc.nspins)
    S = float(calc.spin_magnitude)
    B = reciprocal_b_matrix(np.asarray(calc.sm.unit_cell(), dtype=float))

    n1, n2, n3 = (int(x) for x in k_mesh)
    # OFFSET (Monkhorst-Pack) grid: points at (i + 1/2)/n. This deliberately AVOIDS
    # Gamma and the zone edge, where a magnet with a Goldstone mode has omega -> 0 and the
    # moment integrand ~ 1/omega diverges (H2 is only semi-definite there). The integral
    # itself is finite in 2-D/3-D; sampling the singular point directly is what produced
    # the earlier ~1e6 nonsense. An axis with n = 1 puts its single point at the zone
    # centre offset (i.e. a decoupled direction contributes a q = 0 slab, correct).
    grid = [((np.arange(n) + 0.5) / n) for n in (n1, n2, n3)]
    qs = np.array([[a, b, c] for a in grid[0] for b in grid[1] for c in grid[2]])
    q_cart = qs @ B

    nk = len(q_cart)
    if chunk_size is None:
        # ~8 MB of H(q) per chunk. Each thread holds a few working copies of its
        # chunk, so the peak is about (threads x 8 x 8 MB) whatever the mesh; the
        # whole-mesh stack this replaced peaked near 8.6 GB at 64^3 and 16 sites.
        chunk_size = max(16, int(2 ** 19 // (2 * N) ** 2))
    chunks = [q_cart[i:i + chunk_size] for i in range(0, nk, chunk_size)]
    n_threads = int(n_threads) if n_threads else (os.cpu_count() or 1)
    n_threads = max(1, min(n_threads, len(chunks)))

    # Prime the lambdify and Ewald-engine caches once, so the worker threads share
    # them instead of each building its own.
    calc._build_h_stack(q_cart[:1], S)

    def one_chunk(qc):
        return _chunk_sums(calc._build_h_stack(qc, S), N)

    if n_threads == 1:
        parts = [one_chunk(qc) for qc in chunks]
    else:
        # The LAPACK calls release the GIL, so threads run them in parallel; the
        # sums are taken in chunk order below, so the result does not depend on
        # the thread count.
        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            parts = list(ex.map(one_chunk, chunks))

    sum_zero = 0.0                          # sum_k [ sum_nu omega - tr A ]
    dens = np.zeros(N)                      # sum_k <a^dag_i a_i>
    n_used = 0
    n_failed = 0
    max_imag = 0.0
    for z, d, used, failed, imag in parts:
        sum_zero += z
        dens += d
        n_used += used
        n_failed += failed
        max_imag = max(max_imag, imag)
    n_total = nk

    # Imaginary eigenvalues OR a non-positive-definite H2 (Colpa failing) both mean the
    # reference state is not a classical minimum -- the corrections are then meaningless.
    not_a_minimum = (max_imag > tol_imag) or (n_failed > 0)
    if n_used == 0 or (not_a_minimum and n_failed > 0.5 * n_total):
        raise ValueError(
            f"1/S corrections: the magnetic structure is NOT a classical minimum "
            f"(imaginary magnon energies, max |Im w| = {max_imag:.2e}; H(q) not positive "
            f"definite at {n_failed}/{n_total} k-points). Minimise the structure "
            f"first -- corrections about a non-minimum are meaningless.")
    if not_a_minimum:
        logger.warning(
            f"1/S corrections: H(q) has imaginary eigenvalues (max {max_imag:.2e}) or is "
            f"non-positive-definite at {n_failed} k-points; the reference state may not be "
            f"the ground state -- corrections are unreliable.")

    dE = sum_zero / (2.0 * n_used * N)       # per site
    dS = dens / n_used                       # per site
    return LSWTCorrections(energy_correction_per_site=dE,
                           moment_reduction=dS, n_kpoints=n_used)
