
import logging
from typing import List, Optional, Set, Tuple
import numpy as np
import numpy.typing as npt
from scipy import linalg as la  # Matches magcalc.py usage

logger = logging.getLogger(__name__)

# --- Numerical Constants ---
DEGENERACY_THRESHOLD: float = 1e-8
ZERO_MATRIX_ELEMENT_THRESHOLD: float = 1e-6
ALPHA_MATRIX_ZERO_NORM_WARNING_THRESHOLD: float = 1e-14
EIGENVECTOR_MATCHING_THRESHOLD: float = 1e-4
# PROJECTION_CHECK_TOLERANCE: float = 1e-5 # Not clearly used in extracted block, can add if needed.
I = 1j  # Use pure complex for numerical arrays

def rotation_matrix(
    axis: npt.NDArray[np.float64], angle_deg: float
) -> npt.NDArray[np.float64]:
    """Rodrigues rotation matrix for a rotation of angle_deg about axis.

    axis is a Cartesian 3-vector (need not be normalized). Used for
    domain/twin averaging, where each domain is the crystal rotated in the
    laboratory frame.
    """
    a = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        raise ValueError(f"Rotation axis must be non-zero, got {axis}.")
    a = a / norm
    theta = np.deg2rad(float(angle_deg))
    K = np.array([
        [0.0, -a[2], a[1]],
        [a[2], 0.0, -a[0]],
        [-a[1], a[0], 0.0],
    ])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def gram_schmidt(x: npt.NDArray[np.complex128]) -> npt.NDArray[np.complex128]:
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors using QR decomposition.

    Args:
        x (npt.NDArray[np.complex128]): A matrix where columns represent the vectors
                                       to be orthogonalized.
    Returns:
        npt.NDArray[np.complex128]: A matrix with orthonormal columns spanning the
                                  same space as the input vectors.
    """
    q, r = np.linalg.qr(x, mode="reduced")
    return q

def _diagonalize_and_sort(
    Hmat: npt.NDArray[np.complex128], nspins: int, q_vector_label: str
) -> Tuple[Optional[npt.NDArray[np.complex128]], Optional[npt.NDArray[np.complex128]]]:
    """
    Diagonalize the numerical Hamiltonian matrix and sort eigenvalues/vectors.

    Sorts eigenvalues in ascending order. The corresponding eigenvectors are
    rearranged accordingly. The sorting separates the positive energy (magnon)
    modes from the negative energy modes.

    Args:
        Hmat (npt.NDArray[np.complex128]): The numerical Hamiltonian matrix (2N x 2N).
        nspins (int): The number of spins in the magnetic unit cell (N).
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        Tuple[Optional[npt.NDArray[np.complex128]], Optional[npt.NDArray[np.complex128]]]:
            A tuple containing:
            - Sorted eigenvalues (2N, complex).
            - Sorted eigenvectors (columns, 2N x 2N, complex).
            Returns (None, None) if diagonalization fails.
    """
    try:
        eigvals, eigvecs = la.eig(Hmat)
    except np.linalg.LinAlgError as e:
        logger.error(f"Eigenvalue calculation failed for {q_vector_label}: {e}")
        return None, None
    sort_indices: npt.NDArray[np.int_] = eigvals.argsort()
    eigvecs_tmp1: npt.NDArray[np.complex128] = eigvecs[:, sort_indices][
        :, nspins : 2 * nspins
    ]
    eigvals_tmp1: npt.NDArray[np.complex128] = eigvals[sort_indices][nspins : 2 * nspins]
    eigvecs_tmp2: npt.NDArray[np.complex128] = eigvecs[:, sort_indices][:, 0:nspins]
    eigvals_tmp2: npt.NDArray[np.complex128] = eigvals[sort_indices][0:nspins]
    sort_indices_neg: npt.NDArray[np.int_] = (np.abs(eigvals_tmp2)).argsort()
    eigvecs_tmp3: npt.NDArray[np.complex128] = eigvecs_tmp2[:, sort_indices_neg]
    eigvals_tmp3: npt.NDArray[np.complex128] = eigvals_tmp2[sort_indices_neg]
    eigenvalues_sorted: npt.NDArray[np.complex128] = np.concatenate(
        (eigvals_tmp1, eigvals_tmp3)
    )
    eigenvectors_sorted: npt.NDArray[np.complex128] = np.hstack(
        (eigvecs_tmp1, eigvecs_tmp3)
    )
    return eigenvalues_sorted, eigenvectors_sorted

def _apply_gram_schmidt(
    eigenvalues: npt.NDArray[np.complex128],
    eigenvectors: npt.NDArray[np.complex128],
    degeneracy_threshold: float,
    q_vector_label: str,
) -> npt.NDArray[np.complex128]:
    """
    Apply Gram-Schmidt orthogonalization to blocks of degenerate eigenvectors.

    Iterates through sorted eigenvectors and applies Gram-Schmidt to sets
    of eigenvectors whose corresponding eigenvalues are closer than the
    `degeneracy_threshold`. This ensures orthogonality within degenerate subspaces.

    Args:
        eigenvalues (npt.NDArray[np.complex128]): Sorted eigenvalues.
        eigenvectors (npt.NDArray[np.complex128]): Corresponding sorted eigenvectors (columns).
        degeneracy_threshold (float): The threshold below which eigenvalues are
                                      considered degenerate.
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        npt.NDArray[np.complex128]: The eigenvectors matrix with degenerate blocks
                                  orthogonalized.
    """
    nspins2 = eigenvectors.shape[0]
    orthonormalized_eigenvectors = eigenvectors.copy()
    degeneracy_count: int = 0
    for i in range(1, nspins2):
        if abs(eigenvalues[i] - eigenvalues[i - 1]) < degeneracy_threshold:
            degeneracy_count += 1
        elif degeneracy_count > 0:
            start_idx = i - degeneracy_count - 1
            end_idx = i
            logger.debug(
                f"Applying Gram-Schmidt to block [{start_idx}:{end_idx}] for {q_vector_label}"
            )
            vec_block = orthonormalized_eigenvectors[:, start_idx:end_idx]
            orthonormal_vecs = gram_schmidt(vec_block)
            if orthonormal_vecs.shape[1] == vec_block.shape[1]:
                orthonormalized_eigenvectors[:, start_idx:end_idx] = orthonormal_vecs
            else:
                logger.warning(
                    f"Rank deficiency detected during GS for {q_vector_label} at index {i}. Original block size: {vec_block.shape[1]}, Orthonormal basis size: {orthonormal_vecs.shape[1]}"
                )
                orthonormalized_eigenvectors[
                    :, start_idx : start_idx + orthonormal_vecs.shape[1]
                ] = orthonormal_vecs
                orthonormalized_eigenvectors[
                    :, start_idx + orthonormal_vecs.shape[1] : end_idx
                ] = 0
            degeneracy_count = 0
    if degeneracy_count > 0:
        start_idx = nspins2 - 1 - degeneracy_count
        end_idx = nspins2
        logger.debug(
            f"Applying Gram-Schmidt to final block [{start_idx}:{end_idx}] for {q_vector_label}"
        )
        vec_block = orthonormalized_eigenvectors[:, start_idx:end_idx]
        orthonormal_vecs = gram_schmidt(vec_block)
        if orthonormal_vecs.shape[1] == vec_block.shape[1]:
            orthonormalized_eigenvectors[:, start_idx:end_idx] = orthonormal_vecs
        else:
            logger.warning(
                f"Rank deficiency detected during GS for {q_vector_label} at end of array. Original block size: {vec_block.shape[1]}, Orthonormal basis size: {orthonormal_vecs.shape[1]}"
            )
            orthonormalized_eigenvectors[
                :, start_idx : start_idx + orthonormal_vecs.shape[1]
            ] = orthonormal_vecs
            orthonormalized_eigenvectors[
                :, start_idx + orthonormal_vecs.shape[1] : end_idx
            ] = 0
    return orthonormalized_eigenvectors

def _calculate_alpha_matrix(
    eigenvectors: npt.NDArray[np.complex128],
    G_metric: npt.NDArray[np.float64],
    zero_threshold: float,
    q_vector_label: str,
) -> Optional[npt.NDArray[np.complex128]]:
    """
    Calculate the diagonal alpha matrix used in the transformation T.

    The transformation matrix T relates the original boson operators to the
    diagonal magnon operators. T = V @ alpha, where V is the matrix of
    eigenvectors and alpha is a diagonal matrix.
    alpha_ii = sqrt(g_ii / N_ii), where g is the metric tensor (+1/-1) and
    N_ii = V_i^dagger @ g @ V_i is the pseudo-norm.

    Handles potential issues like near-zero pseudo-norms and sign mismatches
    between g_ii and N_ii, setting alpha_ii to zero in problematic cases.

    Args:
        eigenvectors (npt.NDArray[np.complex128]): Orthonormalized eigenvectors (columns).
        G_metric (npt.NDArray[np.float64]): The diagonal metric tensor [1,..1,-1,..-1].
        zero_threshold (float): Threshold below which norms or alpha values are
                                treated as zero.
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        Optional[npt.NDArray[np.complex128]]: The diagonal alpha matrix (2N x 2N).
                                           Returns None if calculation fails.
    """
    nspins2 = eigenvectors.shape[0]
    alpha_diag_sq: npt.NDArray[np.float64] = np.zeros(nspins2, dtype=float)
    zero_threshold_sq = zero_threshold**2  # Pre-calculate square

    for i in range(nspins2):
        V_i = eigenvectors[:, i]
        norm_sq_N_ii = np.real(np.vdot(V_i, G_metric @ V_i))
        G_ii = G_metric[i, i]

        # Check if norm is below the main threshold for setting alpha to zero
        if abs(norm_sq_N_ii) < zero_threshold_sq:
            # Only warn if the norm is not extremely close to zero
            if abs(norm_sq_N_ii) >= ALPHA_MATRIX_ZERO_NORM_WARNING_THRESHOLD:
                logger.warning(
                    f"Near-zero pseudo-norm N_ii ({norm_sq_N_ii:.2e}) for eigenvector {i} at {q_vector_label}. Setting alpha_ii to 0."
                )
            # Always set alpha to 0 if below the main threshold
            alpha_diag_sq[i] = 0.0
            continue  # Skip the sign mismatch check

        # Check for sign mismatch only if norm is not near zero
        # Use the squared threshold here as well for consistency
        if G_ii * norm_sq_N_ii < -zero_threshold_sq:
            logger.warning(
                f"Sign mismatch between G_ii ({G_ii}) and N_ii ({norm_sq_N_ii:.2e}) for eigenvector {i} at {q_vector_label}. Setting alpha_ii to 0."
            )
            alpha_diag_sq[i] = 0.0
        else:
            # This case implies abs(norm_sq_N_ii) >= zero_threshold_sq
            # AND G_ii * norm_sq_N_ii >= -zero_threshold_sq
            # If G_ii is +1, norm_sq_N_ii must be >= zero_threshold_sq (positive)
            # If G_ii is -1, norm_sq_N_ii must be <= -zero_threshold_sq (negative)
            # So G_ii / norm_sq_N_ii should always be positive here.
            alpha_diag_sq[i] = G_ii / norm_sq_N_ii

    # Ensure no negative values remain due to potential floating point issues near zero
    alpha_diag_sq[alpha_diag_sq < 0] = 0
    alpha_diag: npt.NDArray[np.float64] = np.sqrt(alpha_diag_sq)
    # Set very small resulting alphas to zero
    alpha_diag[np.abs(alpha_diag) < zero_threshold] = 0.0
    alpha_matrix: npt.NDArray[np.complex128] = np.diag(alpha_diag).astype(np.complex128)
    return alpha_matrix


def _match_and_reorder_minus_q(
    eigvecs_p_ortho: npt.NDArray[np.complex128],
    alpha_p: npt.NDArray[np.complex128],
    eigvecs_m_ortho: npt.NDArray[np.complex128],
    eigvals_m_sorted: npt.NDArray[np.complex128],
    alpha_m_sorted: npt.NDArray[np.complex128],
    nspins: int,
    match_tol: float,  # Tolerance for norm comparison in matching (corresponds to 1e-5 in origin)
    zero_tol_comp_phase: float,  # Tolerance for selecting non-zero components for phase factor (corresponds to 1e-5 in origin)
    zero_tol_alpha_final: float,  # Tolerance for truncating final alpha values (corresponds to 1e-6 in origin)
    q_vector_label: str,
) -> Tuple[
    npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]
]:
    """
    Match and reorder eigenvectors and alpha matrix for the -q calculation.

    The calculation of S(q,w) requires relating the solutions at +q and -q.
    This function reorders the eigenvectors (`eigvecs_m_ortho`), eigenvalues
    (`eigvals_m_sorted`), and alpha matrix (`alpha_m_sorted`) obtained from the
    -q diagonalization to match the ordering derived from the +q solution.

    Matching is based on maximizing the projection between the appropriately
    transformed +q eigenvectors and the -q eigenvectors. A phase factor is
    calculated and applied to the reordered -q alpha matrix elements to ensure
    consistency.

    Args:
        eigvecs_p_ortho (npt.NDArray[np.complex128]): Orthonormalized eigenvectors for +q.
        alpha_p (npt.NDArray[np.complex128]): Diagonal alpha matrix for +q.
        eigvecs_m_ortho (npt.NDArray[np.complex128]): Orthonormalized eigenvectors for -q (initial sort).
        eigvals_m_sorted (npt.NDArray[np.complex128]): Eigenvalues for -q (initial sort, corresponds to eigvecs_m_ortho).
        alpha_m_sorted (npt.NDArray[np.complex128]): Diagonal alpha matrix for -q (initial sort).
        nspins (int): Number of spins in the unit cell.
        match_tol (float): Tolerance for eigenvector norm comparison during matching.
        zero_tol_comp_phase (float): Threshold for selecting non-zero components for phase factor calculation.
        zero_tol_alpha_final (float): Threshold below which final alpha matrix elements are set to zero.
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
            - Reordered eigenvectors for -q.
            - Reordered eigenvalues for -q.
            - Reordered and phase-corrected alpha matrix for -q.
    """
    nspins2 = 2 * nspins

    # Initialize outputs
    eigenvectors_minus_q_reordered: npt.NDArray[np.complex128] = np.zeros_like(
        eigvecs_m_ortho, dtype=complex
    )
    eigenvalues_minus_q_reordered: npt.NDArray[np.complex128] = np.zeros_like(
        eigvals_m_sorted, dtype=complex
    )
    # Store diagonal elements of alpha_m_reordered, then form diagonal matrix
    alpha_m_reordered_diag_elements: npt.NDArray[np.complex128] = np.zeros(
        nspins2, dtype=complex
    )

    # Create block-swapped and conjugated version of -q eigenvectors (source for matching)
    # This is Vm_orig_swap_conj in our code, evecmswap in magcalc_origin.py
    Vm_orig_swap_conj: npt.NDArray[np.complex128] = np.conj(
        np.vstack((eigvecs_m_ortho[nspins:nspins2, :], eigvecs_m_ortho[0:nspins, :]))
    )

    def _swap_conj_inverse(v):
        # Undo the block-swap + conjugation (the operation is an involution).
        return np.conj(np.concatenate((v[nspins:nspins2], v[0:nspins])))

    matched_count = 0

    def _match_block(target_indices, candidate_indices, reordered_offset):
        """Match +q eigenvectors against a block of -q eigenvectors.

        For non-degenerate bands the swapped-conjugated -q eigenvector is
        parallel (up to phase) to its +q partner, and this reduces to the
        original best-|overlap| match. Inside a DEGENERATE eigenspace,
        however, the two independent diagonalizations mix the subspace by an
        arbitrary unitary, so no single -q column matches; there the +q
        target is PROJECTED onto the degenerate -q subspace (orthogonalized
        against partners already constructed from the same subspace), which
        is an equally valid -q eigenbasis aligned with +q by construction.
        Without this, degenerate bands (e.g. collinear two-sublattice AFM
        with anisotropy) silently lose all S(q,w) weight.
        """
        nonlocal matched_count

        # Group candidates into degenerate subspaces by eigenvalue.
        groups = []  # each: {'idx': [...], 'eig': value, 'assigned': [vecs]}
        for j in candidate_indices:
            ev = eigvals_m_sorted[j]
            for g in groups:
                if abs(ev - g['eig']) < DEGENERACY_THRESHOLD * max(1.0, abs(g['eig'])):
                    g['idx'].append(j)
                    break
            else:
                groups.append({'idx': [j], 'eig': ev, 'assigned': []})

        for i_p in target_indices:
            target_Vp_col = eigvecs_p_ortho[:, i_p]
            norm_target = np.sqrt(np.real(np.vdot(target_Vp_col, target_Vp_col)))
            if norm_target < zero_tol_comp_phase:
                logger.warning(
                    f"Target +q vector {i_p} has near-zero norm at {q_vector_label}. Skipping match."
                )
                continue

            # Project the target onto each group's remaining subspace.
            best = None  # (norm, group, projected_vector)
            for g in groups:
                if len(g['assigned']) >= len(g['idx']):
                    continue  # subspace exhausted
                proj = np.zeros(nspins2, dtype=complex)
                for j in g['idx']:
                    col = Vm_orig_swap_conj[:, j]
                    proj += np.vdot(col, target_Vp_col) * col
                for assigned in g['assigned']:
                    proj -= np.vdot(assigned, proj) * assigned
                nP = np.sqrt(np.real(np.vdot(proj, proj)))
                if best is None or nP > best[0]:
                    best = (nP, g, proj)

            if best is None or best[0] < norm_target - match_tol:
                logger.warning(
                    f"No matching -q eigenvector subspace found for +q eigenvector "
                    f"index {i_p} at {q_vector_label}"
                    + (f" (best projection {best[0]:.6f})" if best else "")
                )
                continue

            nP, group, proj = best
            v_new_swap_conj = proj / nP  # unit vector aligned with the target
            group['assigned'].append(v_new_swap_conj)

            reordered_m_idx = i_p + reordered_offset
            eigenvectors_minus_q_reordered[:, reordered_m_idx] = _swap_conj_inverse(
                v_new_swap_conj
            )
            eigenvalues_minus_q_reordered[reordered_m_idx] = group['eig']

            # Phase factor between the target and its constructed partner
            # (aligned by construction, so this is ~1; kept for exactness and
            # consistency with the original component-ratio convention).
            comp_Vp_for_phase = target_Vp_col[
                np.abs(target_Vp_col) > zero_tol_comp_phase
            ]
            comp_Vm_swap_for_phase = v_new_swap_conj[
                np.abs(v_new_swap_conj) > zero_tol_comp_phase
            ]
            phase_term = 1.0 + 0.0j
            if (
                len(comp_Vp_for_phase) > 0
                and len(comp_Vm_swap_for_phase) > 0
                and np.abs(comp_Vm_swap_for_phase[0]) > zero_tol_comp_phase
            ):
                phase_term = comp_Vp_for_phase[0] / comp_Vm_swap_for_phase[0]
            else:
                logger.warning(
                    f"Could not determine phase for +q vec {i_p} at {q_vector_label}. Using phase=1."
                )

            alpha_m_reordered_diag_elements[reordered_m_idx] = np.conj(
                alpha_p[i_p, i_p] * phase_term
            )
            matched_count += 1

    # Loop 1: first nspins of +q against the second -q block.
    _match_block(range(nspins), range(nspins, nspins2), +nspins)
    # Loop 2: last nspins of +q against the first -q block.
    _match_block(range(nspins, nspins2), range(nspins), -nspins)

    if matched_count != nspins2:
        logger.warning(
            f"Number of matched -q vectors ({matched_count}) does not equal {nspins2} at {q_vector_label}"
        )

    alpha_matrix_minus_q_reordered = np.diag(alpha_m_reordered_diag_elements)
    alpha_matrix_minus_q_reordered[  # Truncate small alpha values
        np.abs(alpha_matrix_minus_q_reordered) < zero_tol_alpha_final
    ] = 0
    return (
        eigenvectors_minus_q_reordered,
        eigenvalues_minus_q_reordered,
        alpha_matrix_minus_q_reordered,
    )


def _spin_prefactor_vector(
    spin_magnitude: float,
    nspins: int,
    spin_magnitudes: Optional[List[float]] = None,
) -> npt.NDArray[np.float64]:
    """Row scaling sqrt(S_i / 2) for the 3N spin components of K / Kd.

    The Holstein-Primakoff map for site i carries sqrt(S_i/2), so a MIXED-SPIN model
    needs a per-site factor here -- a single global sqrt(S/2) makes every relative
    intensity wrong by sqrt(S_i/S_ref). (The dispersion was already correct: gen_HM
    scales each site's HP expansion by its own spin_S.)

    Applying it to the 3N rows is exact: Ud is block-diagonal in the sites (its 3x3
    block rotates site i's local frame to the lab), so a per-site scalar commutes
    through it.
    """
    if not spin_magnitudes or len(spin_magnitudes) != nspins:
        return np.full(3 * nspins, np.sqrt(spin_magnitude / 2.0))
    return np.repeat(np.sqrt(np.asarray(spin_magnitudes, dtype=float) / 2.0), 3)


def _calculate_K_Kd(
    Ud_numeric: npt.NDArray[np.complex128],
    spin_magnitude: float,
    nspins: int,
    inv_T_p: npt.NDArray[np.complex128],
    inv_T_m_reordered: npt.NDArray[np.complex128],
    zero_threshold: float,
    spin_magnitudes: Optional[List[float]] = None,
) -> Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Calculate the K and Kd matrices for S(q,w) intensity calculation.

    These matrices relate the original spin operators (in global coordinates)
    to the diagonal magnon creation/annihilation operators.
    S^alpha(q) = sum_mu [ K_{alpha, mu} * b_mu + Kd_{alpha, mu} * b_{-mu}^dagger ]

    K = sqrt(S/2) * Ud * Udd * T^{-1}(+q)
    Kd = sqrt(S/2) * Ud * Udd * T^{-1}(-q, reordered)
    where Ud maps local spin axes to global, Udd maps spin components to bosons.

    Args:
        Ud_numeric (npt.NDArray[np.complex128]): Numerical rotation matrix (3N x 3N).
        spin_magnitude (float): Numerical value of spin S.
        nspins (int): Number of spins in the unit cell.
        inv_T_p (npt.NDArray[np.complex128]): Inverse transformation matrix T^{-1} for +q.
        inv_T_m_reordered (npt.NDArray[np.complex128]): Inverse transformation matrix T^{-1} for -q (reordered).
        zero_threshold (float): Threshold below which matrix elements are set to zero.
    Returns:
        Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128]]: K matrix (3N x 2N) and Kd matrix (3N x 2N).
    """
    Udd_local_boson_map: npt.NDArray[np.complex128] = np.zeros(
        (3 * nspins, 2 * nspins), dtype=complex
    )
    for i in range(nspins):
        Udd_local_boson_map[3 * i, i] = 1.0
        Udd_local_boson_map[3 * i, i + nspins] = 1.0
        Udd_local_boson_map[3 * i + 1, i] = 1.0 / I
        Udd_local_boson_map[3 * i + 1, i + nspins] = -1.0 / I
    pref = _spin_prefactor_vector(spin_magnitude, nspins, spin_magnitudes)[:, None]
    K_matrix: npt.NDArray[np.complex128] = (
        pref * (Ud_numeric @ Udd_local_boson_map @ inv_T_p)
    )
    Kd_matrix: npt.NDArray[np.complex128] = (
        pref * (Ud_numeric @ Udd_local_boson_map @ inv_T_m_reordered)
    )
    K_matrix[np.abs(K_matrix) < zero_threshold] = 0
    Kd_matrix[np.abs(Kd_matrix) < zero_threshold] = 0
    return K_matrix, Kd_matrix


def KKdMatrix(
    spin_magnitude: float,
    Hmat_plus_q: npt.NDArray[np.complex128],
    Hmat_minus_q: npt.NDArray[np.complex128],
    Ud_numeric: npt.NDArray[np.complex128],
    q_vector: npt.NDArray[np.float64],
    nspins: int,
    spin_magnitudes: Optional[List[float]] = None,
) -> Tuple[
    npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]
]:
    """
    Calculate K, Kd matrices and eigenvalues for S(q,w) calculation.
    
    Orchestrates diagonalization, sorting, basis matching (-q to +q), and final matrix construction.
    Includes custom Gram-Schmidt logic for -q eigenvectors to ensure continuity with +q basis.
    """
    q_label = f"q={q_vector}"
    nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex128)
    nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex128)
    G_metric = np.diag(np.concatenate([np.ones(nspins), -np.ones(nspins)]))
    
    # --- 1. Diagonalize +q ---
    eigvals_p_sorted, eigvecs_p_sorted = _diagonalize_and_sort(
        Hmat_plus_q, nspins, f"+{q_label}"
    )
    if eigvals_p_sorted is None or eigvecs_p_sorted is None:
        return nan_matrix, nan_matrix, nan_eigs
        
    # --- 2. GS Orthogonalize +q ---
    eigvecs_p_ortho = _apply_gram_schmidt(
        eigvals_p_sorted, eigvecs_p_sorted, DEGENERACY_THRESHOLD, f"+{q_label}"
    )
    
    # --- 3. Alpha +q ---
    alpha_p = _calculate_alpha_matrix(
        eigvecs_p_ortho, G_metric, ZERO_MATRIX_ELEMENT_THRESHOLD, f"+{q_label}"
    )
    if alpha_p is None:
        return nan_matrix, nan_matrix, nan_eigs
        
    # --- 4. Diagonalize -q ---
    eigvals_m_sorted, eigvecs_m_sorted = _diagonalize_and_sort(
        Hmat_minus_q, nspins, f"-{q_label}"
    )
    if eigvals_m_sorted is None or eigvecs_m_sorted is None:
        return nan_matrix, nan_matrix, nan_eigs

    # --- 5. Orthonormalize -q eigenvectors within degenerate blocks ---
    # scipy.linalg.eig returns eigenvectors that are linearly independent but
    # not in general orthonormal — particularly inside a degenerate eigenspace.
    # We orthonormalize each degenerate block via Gram–Schmidt so that the
    # subsequent _calculate_alpha_matrix step (which assumes well-conditioned
    # basis vectors) is numerically stable.
    #
    # Mirrors what _apply_gram_schmidt does for the +q side; alignment between
    # +q and -q modes is then handled by _match_and_reorder_minus_q rather
    # than being baked in here. Earlier "custom GS" attempts to splice +q
    # reference vectors into the -q basis caused inflated alpha values at
    # high-symmetry q-points (vertical streaks in S(Q,w)).
    eigvecs_m_ortho = _apply_gram_schmidt(
        eigvals_m_sorted, eigvecs_m_sorted, DEGENERACY_THRESHOLD, f"-{q_label}"
    )

    # --- 6. Alpha -q ---
    alpha_m_sorted = _calculate_alpha_matrix(
        eigvecs_m_ortho, G_metric, ZERO_MATRIX_ELEMENT_THRESHOLD, f"-{q_label}"
    )
    if alpha_m_sorted is None:
        return nan_matrix, nan_matrix, nan_eigs

    # --- 7. Match and Reorder ---
    (eigvecs_m_final, eigvals_m_reordered, alpha_m_final) = _match_and_reorder_minus_q(
        eigvecs_p_ortho,
        alpha_p,
        eigvecs_m_ortho,
        eigvals_m_sorted,
        alpha_m_sorted,
        nspins,
        EIGENVECTOR_MATCHING_THRESHOLD,
        EIGENVECTOR_MATCHING_THRESHOLD,
        ZERO_MATRIX_ELEMENT_THRESHOLD,
        q_label,
    )

    # --- 8. T Inverse ---
    inv_T_p = eigvecs_p_ortho @ alpha_p
    inv_T_m_reordered = eigvecs_m_final @ alpha_m_final

    # --- 9. K and Kd ---
    # Optimized _calculate_K_Kd with vectorized Udd is needed?
    # Actually _calculate_K_Kd in this file uses loop. We can optimize it here.
    
    # Optimized Udd construction (Vectorized)
    # Udd_local_boson_map: (3*nspins, 2*nspins)
    # Blocks:
    # 3*i, i -> 1
    # 3*i, i+N -> 1
    # 3*i+1, i -> 1/j
    # 3*i+1, i+N -> -1/j
    
    Udd_local_boson_map = np.zeros((3 * nspins, 2 * nspins), dtype=complex)
    indices = np.arange(nspins)
    
    # 3*i rows
    Udd_local_boson_map[3 * indices, indices] = 1.0
    Udd_local_boson_map[3 * indices, indices + nspins] = 1.0
    
    # 3*i+1 rows
    Udd_local_boson_map[3 * indices + 1, indices] = -1j # 1/j = -j
    Udd_local_boson_map[3 * indices + 1, indices + nspins] = 1j # -1/j = j

    pref = _spin_prefactor_vector(spin_magnitude, nspins, spin_magnitudes)[:, None]

    # Matrix multiplications (per-site sqrt(S_i/2) row scaling -- mixed spin)
    K_matrix = pref * (Ud_numeric @ Udd_local_boson_map @ inv_T_p)
    Kd_matrix = pref * (Ud_numeric @ Udd_local_boson_map @ inv_T_m_reordered)

    # Thresholding
    K_matrix[np.abs(K_matrix) < ZERO_MATRIX_ELEMENT_THRESHOLD] = 0
    Kd_matrix[np.abs(Kd_matrix) < ZERO_MATRIX_ELEMENT_THRESHOLD] = 0

    return K_matrix, Kd_matrix, eigvals_p_sorted
