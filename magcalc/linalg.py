
import logging
from typing import Tuple, Optional, Set
import numpy as np
import numpy.typing as npt
from scipy import linalg as la  # Matches magcalc.py usage

logger = logging.getLogger(__name__)

# --- Numerical Constants ---
DEGENERACY_THRESHOLD: float = 1e-12
ZERO_MATRIX_ELEMENT_THRESHOLD: float = 1e-6
ALPHA_MATRIX_ZERO_NORM_WARNING_THRESHOLD: float = 1e-14
EIGENVECTOR_MATCHING_THRESHOLD: float = 1e-5
# PROJECTION_CHECK_TOLERANCE: float = 1e-5 # Not clearly used in extracted block, can add if needed.
I = 1j  # Use pure complex for numerical arrays

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

    matched_original_m_indices: Set[int] = set()  # To track used original -q indices

    # Loop 1: Match first nspins of +q (eigvecs_p_ortho[:, i_p])
    # with last nspins of -q (Vm_orig_swap_conj[:, j_m_orig_idx])
    for i_p in range(nspins):
        target_Vp_col = eigvecs_p_ortho[:, i_p]  # This is e_vec[:, i] in magcalc_origin
        norm_target_Vp_col_sq = np.real(np.vdot(target_Vp_col, target_Vp_col))
        if (
            norm_target_Vp_col_sq < zero_tol_comp_phase**2
        ):  # Check against squared tolerance
            logger.warning(
                f"Target +q vector {i_p} has near-zero norm at {q_vector_label}. Skipping match."
            )
            continue
        
        # --- NEW LOGIC: Collect all projections and pick best ---
        projections = []
        
        for j_m_orig_idx in range(
            nspins, nspins2
        ):  # Original -q indices for the second block (j in magcalc_origin)
            if j_m_orig_idx in matched_original_m_indices:
                continue

            source_Vm_swap_conj_col = Vm_orig_swap_conj[
                :, j_m_orig_idx
            ]  # This is evecmswap[:,j]
            norm_source_Vm_swap_conj_col_sq = np.real(
                np.vdot(source_Vm_swap_conj_col, source_Vm_swap_conj_col)
            )
            if norm_source_Vm_swap_conj_col_sq < zero_tol_comp_phase**2:
                continue
            
            proj = np.abs(np.vdot(target_Vp_col, source_Vm_swap_conj_col))
            threshold = np.sqrt(norm_target_Vp_col_sq * norm_source_Vm_swap_conj_col_sq) - match_tol
            
            if proj > threshold:
                projections.append((proj, j_m_orig_idx))
        
        if projections:
             # Sort by projection strength descending
             projections.sort(key=lambda x: x[0], reverse=True)
             best_match = projections[0]
             j_m_orig_idx = best_match[1]
             
             # Apply success logic
             reordered_m_idx = (
                 i_p + nspins
             )  # As per magcalc_origin.py logic for this loop
             eigenvectors_minus_q_reordered[:, reordered_m_idx] = eigvecs_m_ortho[
                 :, j_m_orig_idx
             ]
             eigenvalues_minus_q_reordered[reordered_m_idx] = eigvals_m_sorted[
                 j_m_orig_idx
             ]

             # Phase factor calculation based on magcalc_origin.py
             source_Vm_swap_conj_col = Vm_orig_swap_conj[:, j_m_orig_idx]
             comp_Vp_for_phase = target_Vp_col[
                 np.abs(target_Vp_col) > zero_tol_comp_phase
             ]
             comp_Vm_swap_for_phase = source_Vm_swap_conj_col[
                 np.abs(source_Vm_swap_conj_col) > zero_tol_comp_phase
             ]
             phase_term = 1.0 + 0.0j
             if (
                 len(comp_Vp_for_phase) > 0
                 and len(comp_Vm_swap_for_phase) > 0
                 and np.abs(comp_Vm_swap_for_phase[0])
                 > zero_tol_comp_phase  # Check divisor
             ):
                 phase_term = comp_Vp_for_phase[0] / comp_Vm_swap_for_phase[0]
             else:
                 logger.warning(
                     f"Could not determine phase for +q vec {i_p} and -q vec {j_m_orig_idx} (original index) at {q_vector_label}. Using phase=1."
                 )

             alpha_m_reordered_diag_elements[reordered_m_idx] = np.conj(
                 alpha_p[i_p, i_p] * phase_term
             )
             matched_original_m_indices.add(j_m_orig_idx)
             
        else:
            logger.warning(
                f"No matching -q eigenvector found for +q eigenvector index {i_p} in first block at {q_vector_label}"
            )

    # Loop 2: Match last nspins of +q (eigvecs_p_ortho[:, i_p])
    # with first nspins of -q (Vm_orig_swap_conj[:, j_m_orig_idx])
    for i_p in range(nspins, nspins2):
        target_Vp_col = eigvecs_p_ortho[:, i_p]
        norm_target_Vp_col_sq = np.real(np.vdot(target_Vp_col, target_Vp_col))
        if norm_target_Vp_col_sq < zero_tol_comp_phase**2:
            logger.warning(
                f"Target +q vector {i_p} has near-zero norm at {q_vector_label}. Skipping match."
            )
            continue

        projections = []
        for j_m_orig_idx in range(nspins):  # Original -q indices for the first block
            if j_m_orig_idx in matched_original_m_indices:
                continue

            source_Vm_swap_conj_col = Vm_orig_swap_conj[:, j_m_orig_idx]
            norm_source_Vm_swap_conj_col_sq = np.real(
                np.vdot(source_Vm_swap_conj_col, source_Vm_swap_conj_col)
            )
            if norm_source_Vm_swap_conj_col_sq < zero_tol_comp_phase**2:
                continue
            
            proj = np.abs(np.vdot(target_Vp_col, source_Vm_swap_conj_col))
            threshold = np.sqrt(norm_target_Vp_col_sq * norm_source_Vm_swap_conj_col_sq) - match_tol
            
            if proj > threshold:
                projections.append((proj, j_m_orig_idx))

        if projections:
             # Sort by projection strength descending
             projections.sort(key=lambda x: x[0], reverse=True)
             best_match = projections[0]
             j_m_orig_idx = best_match[1]

             # Apply success logic
             reordered_m_idx = (
                 i_p - nspins
             )  # As per magcalc_origin.py logic for this loop
             eigenvectors_minus_q_reordered[:, reordered_m_idx] = eigvecs_m_ortho[
                 :, j_m_orig_idx
             ]
             eigenvalues_minus_q_reordered[reordered_m_idx] = eigvals_m_sorted[
                 j_m_orig_idx
             ]
             
             source_Vm_swap_conj_col = Vm_orig_swap_conj[:, j_m_orig_idx]

             # Phase factor calculation
             comp_Vp_for_phase = target_Vp_col[
                 np.abs(target_Vp_col) > zero_tol_comp_phase
             ]
             comp_Vm_swap_for_phase = source_Vm_swap_conj_col[
                 np.abs(source_Vm_swap_conj_col) > zero_tol_comp_phase
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
                     f"Could not determine phase for +q vec {i_p} and -q vec {j_m_orig_idx} (original index) at {q_vector_label}. Using phase=1."
                 )

             alpha_m_reordered_diag_elements[reordered_m_idx] = np.conj(
                 alpha_p[i_p, i_p] * phase_term
             )
             matched_original_m_indices.add(j_m_orig_idx)
             
        else:
            logger.warning(
                f"No matching -q eigenvector found for +q eigenvector index {i_p} in second block at {q_vector_label}"
            )

    if len(matched_original_m_indices) != nspins2:
        logger.warning(
            f"Number of matched original -q vectors ({len(matched_original_m_indices)}) does not equal {nspins2} at {q_vector_label}"
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


def _calculate_K_Kd(
    Ud_numeric: npt.NDArray[np.complex128],
    spin_magnitude: float,
    nspins: int,
    inv_T_p: npt.NDArray[np.complex128],
    inv_T_m_reordered: npt.NDArray[np.complex128],
    zero_threshold: float,
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
    prefactor: float = np.sqrt(spin_magnitude / 2.0)
    K_matrix: npt.NDArray[np.complex128] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_p
    )
    Kd_matrix: npt.NDArray[np.complex128] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_m_reordered
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

    # --- 5. Custom GS / Basis Selection for -q ---
    # Logic: Uses +q basis (via evecswap_p) to guide -q basis choice in degenerate subspaces
    eigvecs_m_processed = eigvecs_m_sorted.copy()
    nspins2 = eigvecs_m_processed.shape[0]

    # evecswap_p: conj(swap(Vp)) - conceptual basis for -q
    evecswap_p = np.conj(
        np.vstack((eigvecs_p_ortho[nspins:nspins2, :], eigvecs_p_ortho[0:nspins, :]))
    )

    current_block_start_idx = 0
    for i in range(1, nspins2 + 1):
        if (
            i == nspins2
            or abs(eigvals_m_sorted[i] - eigvals_m_sorted[current_block_start_idx])
            >= DEGENERACY_THRESHOLD
        ):
            block_size = i - current_block_start_idx
            # If block_size > 1, we have degeneracy. Try to match +q basis.
            if block_size > 1:
                degenerate_block_m = eigvecs_m_sorted[:, current_block_start_idx:i]
                
                # Determine candidate basis part
                if current_block_start_idx < nspins:
                    candidate_basis_p_block = evecswap_p[:, nspins:]
                else:
                    candidate_basis_p_block = evecswap_p[:, :nspins]

                # Project degenerate_block_m onto candidate_basis_p_block
                # Optimize: vectorization? 
                # degenerate_block_m: (2N, K)
                # candidate_basis_p_block: (2N, N) usually
                # Projections: Sum of |<v_m, v_p>| for all k in K
                
                # m_conj_T = degenerate_block_m.conj().T  # (K, 2N)
                # dots = m_conj_T @ candidate_basis_p_block # (K, N)
                # sum_dots = np.sum(dots, axis=0) # (N,) - Sum over K vectors in block
                
                m_conj_T = np.conj(degenerate_block_m.T)
                dots = m_conj_T @ candidate_basis_p_block
                sum_of_projections = np.sum(dots, axis=0) # flattening happens implicitly
                
                projection_magnitudes = np.abs(sum_of_projections)
                sorted_indices_desc = np.argsort(projection_magnitudes)[::-1]

                if len(sorted_indices_desc) >= block_size and projection_magnitudes[sorted_indices_desc[block_size-1]] > EIGENVECTOR_MATCHING_THRESHOLD:
                    selected_indices = sorted_indices_desc[:block_size]
                    if np.sum(projection_magnitudes > EIGENVECTOR_MATCHING_THRESHOLD) > block_size:
                         logger.debug(f"Custom GS: Selecting top {block_size} matches at {q_label}.")
                    
                    new_basis_for_m_block = candidate_basis_p_block[:, selected_indices]
                    eigvecs_m_processed[:, current_block_start_idx:i] = gram_schmidt(new_basis_for_m_block)
                else:
                    logger.warning(f"Custom GS fallback to standard for block {current_block_start_idx}-{i-1} at {q_label}.")
                    eigvecs_m_processed[:, current_block_start_idx:i] = gram_schmidt(degenerate_block_m)
            
            else:
                # No degeneracy, no action needed unless we want to normalize/phase fix? 
                # Standard GS would just normalize.
                # Let's run standard GS on single vector blocks just to be safe/consistent?
                # Actually _diagonalize returns normalized vectors usually.
                pass 

            current_block_start_idx = i

    eigvecs_m_ortho = eigvecs_m_processed

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

    prefactor = np.sqrt(spin_magnitude / 2.0)
    
    # Matrix multiplications
    K_matrix = prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_p
    Kd_matrix = prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_m_reordered

    # Thresholding
    K_matrix[np.abs(K_matrix) < ZERO_MATRIX_ELEMENT_THRESHOLD] = 0
    Kd_matrix[np.abs(Kd_matrix) < ZERO_MATRIX_ELEMENT_THRESHOLD] = 0

    return K_matrix, Kd_matrix, eigvals_p_sorted
