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
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from scipy import linalg as la

logger = logging.getLogger(__name__)

_COLPA_JITTER = 1e-9         # tiny diagonal shift if H2 is only semi-definite


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


def compute_corrections(
    calc,
    k_mesh=(8, 8, 8),
    spin_magnitudes: Optional[List[float]] = None,
    tol_imag: float = 1e-3,
) -> LSWTCorrections:
    """1/S corrections for a built MagCalc `calc`.

    k_mesh: Monkhorst-Pack-style grid size per reciprocal axis. Only axes with a real
    (non-flat) reciprocal length are sampled -- a chain samples 1-D, a plane 2-D -- so a
    tuple like (24, 24, 1) is honoured directly and a decoupled axis contributes 1 point.
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

    g = np.diag([1.0] * N + [-1.0] * N)
    h_stack = calc._build_h_stack(q_cart, S)

    sum_zero = 0.0                          # sum_k [ sum_nu omega - tr A ]
    dens = np.zeros(N)                      # sum_k <a^dag_i a_i>
    n_used = 0
    n_failed = 0
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

    # Imaginary eigenvalues OR a non-positive-definite H2 (Colpa failing) both mean the
    # reference state is not a classical minimum -- the corrections are then meaningless.
    not_a_minimum = (max_imag > tol_imag) or (n_failed > 0)
    if n_used == 0 or (not_a_minimum and n_failed > 0.5 * len(h_stack)):
        raise ValueError(
            f"1/S corrections: the magnetic structure is NOT a classical minimum "
            f"(imaginary magnon energies, max |Im w| = {max_imag:.2e}; H(q) not positive "
            f"definite at {n_failed}/{len(h_stack)} k-points). Minimise the structure "
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
