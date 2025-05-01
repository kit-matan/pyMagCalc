#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Spin-Wave Theory (LSWT) Calculator Module.

This module provides the `MagCalc` class and supporting functions to perform
LSWT calculations for magnetic materials. It takes a user-defined spin model
(Hamiltonian, structure, interactions) and computes:

1.  Spin-wave dispersion relations (energy vs. momentum).
2.  Dynamic structure factor S(q, ω) (neutron scattering intensity).

Key features include symbolic manipulation using SymPy, numerical diagonalization
using SciPy/NumPy, parallel processing for performance, and caching of
computationally expensive symbolic results.

@author: Kit Matan and Pharit Piyawongwatthana
@contributor: AI Assistant (Refactoring, Docstrings)
Refactored by AI Assistant
"""
# import spin_model as sm # REMOVED: User-defined spin model will be passed explicitly
import sympy as sp
from sympy import I, lambdify, Add
import numpy as np
from scipy import linalg as la
import timeit
import sys
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import logging
import os

# Type Hinting Imports
import types  # Added for ModuleType hint
from typing import List, Tuple, Dict, Any, Optional, Union, NoReturn
import numpy.typing as npt

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- Numerical Constants ---
DEGENERACY_THRESHOLD: float = 1e-12
ZERO_MATRIX_ELEMENT_THRESHOLD: float = 1e-6
# --- NEW CONSTANT for suppressing alpha matrix norm warnings ---
ALPHA_MATRIX_ZERO_NORM_WARNING_THRESHOLD: float = 1e-14  # Suppress warnings below this
# --- END NEW CONSTANT ---
EIGENVECTOR_MATCHING_THRESHOLD: float = 1e-5
ENERGY_IMAG_PART_THRESHOLD: float = 1e-5
SQW_IMAG_PART_THRESHOLD: float = 1e-4
Q_ZERO_THRESHOLD: float = 1e-10
PROJECTION_CHECK_TOLERANCE: float = 1e-5
# --- End Numerical Constants ---

# --- Helper functions (Keep outside class for easier multiprocessing pickling) ---


def substitute_expr(
    args: Tuple[sp.Expr, Union[Dict, List[Tuple[sp.Expr, sp.Expr]]]],
) -> sp.Expr:
    """
    Perform symbolic substitution on a SymPy expression.

    This helper function is designed to be used with `multiprocessing.Pool.imap`
    to parallelize the substitution process across multiple terms of a larger
    expression.

    Args:
        args: A tuple containing:
            expr (sp.Expr): The SymPy expression to perform substitution on.
            subs_dict (Union[Dict, List[Tuple[sp.Expr, sp.Expr]]]):
                A dictionary or list of (old, new) pairs for substitution.

    Returns:
        sp.Expr: The SymPy expression after substitution.
    """
    expr, subs_dict = args
    result: sp.Expr = expr.subs(subs_dict, simultaneous=True)
    return result


def gram_schmidt(x: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
    """
    Perform Gram-Schmidt orthogonalization on a set of vectors using QR decomposition.

    Args:
        x (npt.NDArray[np.complex_]): A matrix where columns represent the vectors
                                       to be orthogonalized.
    Returns:
        npt.NDArray[np.complex_]: A matrix with orthonormal columns spanning the
                                  same space as the input vectors.
    """
    q, r = np.linalg.qr(x, mode="reduced")
    return q


# --- KKdMatrix Helper Functions (Keep outside class) ---
# _diagonalize_and_sort, _apply_gram_schmidt, _calculate_alpha_matrix,
# _match_and_reorder_minus_q, _calculate_K_Kd
def _diagonalize_and_sort(
    Hmat: npt.NDArray[np.complex_], nspins: int, q_vector_label: str
) -> Tuple[Optional[npt.NDArray[np.complex_]], Optional[npt.NDArray[np.complex_]]]:
    """
    Diagonalize the numerical Hamiltonian matrix and sort eigenvalues/vectors.

    Sorts eigenvalues in ascending order. The corresponding eigenvectors are
    rearranged accordingly. The sorting separates the positive energy (magnon)
    modes from the negative energy modes.

    Args:
        Hmat (npt.NDArray[np.complex_]): The numerical Hamiltonian matrix (2N x 2N).
        nspins (int): The number of spins in the magnetic unit cell (N).
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        Tuple[Optional[npt.NDArray[np.complex_]], Optional[npt.NDArray[np.complex_]]]:
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
    eigvecs_tmp1: npt.NDArray[np.complex_] = eigvecs[:, sort_indices][
        :, nspins : 2 * nspins
    ]
    eigvals_tmp1: npt.NDArray[np.complex_] = eigvals[sort_indices][nspins : 2 * nspins]
    eigvecs_tmp2: npt.NDArray[np.complex_] = eigvecs[:, sort_indices][:, 0:nspins]
    eigvals_tmp2: npt.NDArray[np.complex_] = eigvals[sort_indices][0:nspins]
    sort_indices_neg: npt.NDArray[np.int_] = (np.abs(eigvals_tmp2)).argsort()
    eigvecs_tmp3: npt.NDArray[np.complex_] = eigvecs_tmp2[:, sort_indices_neg]
    eigvals_tmp3: npt.NDArray[np.complex_] = eigvals_tmp2[sort_indices_neg]
    eigenvalues_sorted: npt.NDArray[np.complex_] = np.concatenate(
        (eigvals_tmp1, eigvals_tmp3)
    )
    eigenvectors_sorted: npt.NDArray[np.complex_] = np.hstack(
        (eigvecs_tmp1, eigvecs_tmp3)
    )
    return eigenvalues_sorted, eigenvectors_sorted


def _apply_gram_schmidt(
    eigenvalues: npt.NDArray[np.complex_],
    eigenvectors: npt.NDArray[np.complex_],
    degeneracy_threshold: float,
    q_vector_label: str,
) -> npt.NDArray[np.complex_]:
    """
    Apply Gram-Schmidt orthogonalization to blocks of degenerate eigenvectors.

    Iterates through sorted eigenvectors and applies Gram-Schmidt to sets
    of eigenvectors whose corresponding eigenvalues are closer than the
    `degeneracy_threshold`. This ensures orthogonality within degenerate subspaces.

    Args:
        eigenvalues (npt.NDArray[np.complex_]): Sorted eigenvalues.
        eigenvectors (npt.NDArray[np.complex_]): Corresponding sorted eigenvectors (columns).
        degeneracy_threshold (float): The threshold below which eigenvalues are
                                      considered degenerate.
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        npt.NDArray[np.complex_]: The eigenvectors matrix with degenerate blocks
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


# --- MODIFIED FUNCTION ---
def _calculate_alpha_matrix(
    eigenvectors: npt.NDArray[np.complex_],
    G_metric: npt.NDArray[np.float_],
    zero_threshold: float,
    q_vector_label: str,
) -> Optional[npt.NDArray[np.complex_]]:
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
        eigenvectors (npt.NDArray[np.complex_]): Orthonormalized eigenvectors (columns).
        G_metric (npt.NDArray[np.float_]): The diagonal metric tensor [1,..1,-1,..-1].
        zero_threshold (float): Threshold below which norms or alpha values are
                                treated as zero.
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        Optional[npt.NDArray[np.complex_]]: The diagonal alpha matrix (2N x 2N).
                                           Returns None if calculation fails.
    """
    nspins2 = eigenvectors.shape[0]
    alpha_diag_sq: npt.NDArray[np.float_] = np.zeros(nspins2, dtype=float)
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
    alpha_diag: npt.NDArray[np.float_] = np.sqrt(alpha_diag_sq)
    # Set very small resulting alphas to zero
    alpha_diag[np.abs(alpha_diag) < zero_threshold] = 0.0
    alpha_matrix: npt.NDArray[np.complex_] = np.diag(alpha_diag).astype(np.complex_)
    return alpha_matrix


# --- END MODIFIED FUNCTION ---


def _match_and_reorder_minus_q(
    eigvecs_p_ortho: npt.NDArray[np.complex_],
    alpha_p: npt.NDArray[np.complex_],
    eigvecs_m_ortho: npt.NDArray[np.complex_],
    eigvals_m_sorted: npt.NDArray[np.complex_],
    alpha_m_sorted: npt.NDArray[np.complex_],
    nspins: int,
    match_tol: float,
    zero_tol: float,
    q_vector_label: str,
) -> Tuple[
    npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]
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
        eigvecs_p_ortho (npt.NDArray[np.complex_]): Orthonormalized eigenvectors for +q.
        alpha_p (npt.NDArray[np.complex_]): Diagonal alpha matrix for +q.
        eigvecs_m_ortho (npt.NDArray[np.complex_]): Orthonormalized eigenvectors for -q (initial sort).
        eigvals_m_sorted (npt.NDArray[np.complex_]): Eigenvalues for -q (initial sort).
        alpha_m_sorted (npt.NDArray[np.complex_]): Diagonal alpha matrix for -q (initial sort).
        nspins (int): Number of spins in the unit cell.
        match_tol (float): Tolerance for eigenvector matching projection (|proj|^2 > 1 - tol^2).
        zero_tol (float): Threshold below which vector norms or dot products are treated as zero.
        q_vector_label (str): A string label identifying the q-vector (for logging).

    Returns:
        Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
            - Reordered eigenvectors for -q.
            - Reordered eigenvalues for -q.
            - Reordered and phase-corrected alpha matrix for -q.
    """
    nspins2 = 2 * nspins
    eigenvectors_plus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack((eigvecs_p_ortho[nspins:nspins2, :], eigvecs_p_ortho[0:nspins, :]))
    )
    eigenvectors_minus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack((eigvecs_m_ortho[nspins:nspins2, :], eigvecs_m_ortho[0:nspins, :]))
    )
    eigenvectors_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigvecs_m_ortho, dtype=complex
    )
    eigenvalues_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigvals_m_sorted, dtype=complex
    )
    alpha_matrix_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        alpha_m_sorted, dtype=complex
    )
    matched_indices_m: set[int] = set()
    num_matched_vectors: int = 0
    for i in range(nspins2):
        best_match_j: int = -1
        max_proj_metric: float = -1.0
        vec_i_target: npt.NDArray[np.complex_] = eigenvectors_plus_q_swapped_conj[:, i]
        vec_i_norm_sq: float = np.real(np.dot(np.conj(vec_i_target), vec_i_target))
        if vec_i_norm_sq < zero_tol**2:
            logger.warning(
                f"Target vector {i} has near-zero norm at {q_vector_label}. Skipping match."
            )
            continue
        for j in range(nspins2):
            if j in matched_indices_m:
                continue
            vec_j_source: npt.NDArray[np.complex_] = eigenvectors_minus_q_swapped_conj[
                :, j
            ]
            vec_j_norm_sq: float = np.real(np.dot(np.conj(vec_j_source), vec_j_source))
            if vec_j_norm_sq < zero_tol**2:
                continue
            projection: complex = np.dot(np.conj(vec_i_target), vec_j_source)
            projection_mag_sq: float = np.abs(projection) ** 2
            norm_product_sq = vec_i_norm_sq * vec_j_norm_sq
            if norm_product_sq < zero_tol**4:
                continue
            normalized_projection_mag_sq: float = projection_mag_sq / norm_product_sq
            if (
                normalized_projection_mag_sq > max_proj_metric
                and normalized_projection_mag_sq > (1.0 - match_tol**2)
            ):
                max_proj_metric = normalized_projection_mag_sq
                best_match_j = j
        if best_match_j != -1:
            matched_indices_m.add(best_match_j)
            num_matched_vectors += 1
            orig_i = i + nspins if i < nspins else i - nspins
            target_index = i
            eigenvectors_minus_q_reordered[:, target_index] = eigvecs_m_ortho[
                :, best_match_j
            ]
            eigenvalues_minus_q_reordered[target_index] = eigvals_m_sorted[best_match_j]
            vec_i_plus_q_orig = eigvecs_p_ortho[:, orig_i]
            vec_j_minus_q_source = eigenvectors_minus_q_swapped_conj[:, best_match_j]
            dot_product: complex = np.dot(
                np.conj(vec_j_minus_q_source), vec_i_plus_q_orig
            )
            dot_product_mag: float = np.abs(dot_product)
            phase_factor: complex
            if dot_product_mag < zero_tol:
                logger.warning(
                    f"Near-zero dot product ({dot_product_mag:.2e}) during phase calculation for match ({i} -> {best_match_j}) at {q_vector_label}. Setting phase factor to 1."
                )
                phase_factor = 1.0 + 0.0j
            else:
                phase_factor = dot_product / dot_product_mag
            alpha_matrix_minus_q_reordered[target_index, target_index] = np.conj(
                alpha_p[orig_i, orig_i] * phase_factor
            )
        else:
            logger.warning(
                f"No matching eigenvector found for target vector index {i} at {q_vector_label}"
            )
    if num_matched_vectors != nspins2:
        logger.warning(
            f"Number of matched vectors ({num_matched_vectors}) does not equal {nspins2} at {q_vector_label}"
        )
    alpha_matrix_minus_q_reordered[
        np.abs(alpha_matrix_minus_q_reordered) < zero_tol
    ] = 0
    return (
        eigenvectors_minus_q_reordered,
        eigenvalues_minus_q_reordered,
        alpha_matrix_minus_q_reordered,
    )


def _calculate_K_Kd(
    Ud_numeric: npt.NDArray[np.complex_],
    spin_magnitude: float,
    nspins: int,
    inv_T_p: npt.NDArray[np.complex_],
    inv_T_m_reordered: npt.NDArray[np.complex_],
    zero_threshold: float,
) -> Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
    """
    Calculate the K and Kd matrices for S(q,w) intensity calculation.

    These matrices relate the original spin operators (in global coordinates)
    to the diagonal magnon creation/annihilation operators.
    S^alpha(q) = sum_mu [ K_{alpha, mu} * b_mu + Kd_{alpha, mu} * b_{-mu}^dagger ]

    K = sqrt(S/2) * Ud * Udd * T^{-1}(+q)
    Kd = sqrt(S/2) * Ud * Udd * T^{-1}(-q, reordered)
    where Ud maps local spin axes to global, Udd maps spin components to bosons.

    Args:
        Ud_numeric (npt.NDArray[np.complex_]): Numerical rotation matrix (3N x 3N).
        spin_magnitude (float): Numerical value of spin S.
        nspins (int): Number of spins in the unit cell.
        inv_T_p (npt.NDArray[np.complex_]): Inverse transformation matrix T^{-1} for +q.
        inv_T_m_reordered (npt.NDArray[np.complex_]): Inverse transformation matrix T^{-1} for -q (reordered).
        zero_threshold (float): Threshold below which matrix elements are set to zero.
    Returns:
        Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]]: K matrix (3N x 2N) and Kd matrix (3N x 2N).
    """
    Udd_local_boson_map: npt.NDArray[np.complex_] = np.zeros(
        (3 * nspins, 2 * nspins), dtype=complex
    )
    for i in range(nspins):
        Udd_local_boson_map[3 * i, i] = 1.0
        Udd_local_boson_map[3 * i, i + nspins] = 1.0
        Udd_local_boson_map[3 * i + 1, i] = 1.0 / I
        Udd_local_boson_map[3 * i + 1, i + nspins] = -1.0 / I
    prefactor: float = np.sqrt(spin_magnitude / 2.0)
    K_matrix: npt.NDArray[np.complex_] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_p
    )
    Kd_matrix: npt.NDArray[np.complex_] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_m_reordered
    )
    K_matrix[np.abs(K_matrix) < zero_threshold] = 0
    Kd_matrix[np.abs(Kd_matrix) < zero_threshold] = 0
    return K_matrix, Kd_matrix


# --- Main KKdMatrix Function (Keep outside class) ---
def KKdMatrix(
    spin_magnitude: float,
    Hmat_plus_q: npt.NDArray[np.complex_],
    Hmat_minus_q: npt.NDArray[np.complex_],
    Ud_numeric: npt.NDArray[np.complex_],
    q_vector: npt.NDArray[np.float_],
    nspins: int,
) -> Tuple[
    npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]
]:
    """
    Calculate K, Kd matrices and eigenvalues for S(q,w) calculation.

    This function orchestrates the numerical steps required after obtaining the
    Hamiltonian matrices H(+q) and H(-q):
    1. Diagonalize H(+q) and H(-q).
    2. Sort eigenvalues and eigenvectors.
    3. Apply Gram-Schmidt to degenerate eigenvectors.
    4. Calculate the alpha matrices for +q and -q.
    5. Match and reorder the -q results based on the +q results.
    6. Calculate the inverse transformation matrices T^{-1} = V @ alpha.
    7. Calculate the final K and Kd matrices.

    Args:
        spin_magnitude (float): Numerical value of spin S.
        Hmat_plus_q (npt.NDArray[np.complex_]): Numerical Hamiltonian matrix for +q.
        Hmat_minus_q (npt.NDArray[np.complex_]): Numerical Hamiltonian matrix for -q.
        Ud_numeric (npt.NDArray[np.complex_]): Numerical rotation matrix Ud.
        q_vector (npt.NDArray[np.float_]): The momentum vector q.
        nspins (int): Number of spins in the unit cell.

    Returns:
        Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
            K matrix, Kd matrix, and sorted eigenvalues from the +q calculation. Returns NaN arrays on failure.
    """
    q_label = f"q={q_vector}"
    nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
    nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
    G_metric = np.diag(np.concatenate([np.ones(nspins), -np.ones(nspins)]))
    eigvals_p_sorted, eigvecs_p_sorted = _diagonalize_and_sort(
        Hmat_plus_q, nspins, f"+{q_label}"
    )
    if eigvals_p_sorted is None or eigvecs_p_sorted is None:
        return nan_matrix, nan_matrix, nan_eigs
    eigvecs_p_ortho = _apply_gram_schmidt(
        eigvals_p_sorted, eigvecs_p_sorted, DEGENERACY_THRESHOLD, f"+{q_label}"
    )
    alpha_p = _calculate_alpha_matrix(
        eigvecs_p_ortho, G_metric, ZERO_MATRIX_ELEMENT_THRESHOLD, f"+{q_label}"
    )
    if alpha_p is None:
        return nan_matrix, nan_matrix, nan_eigs
    eigvals_m_sorted, eigvecs_m_sorted = _diagonalize_and_sort(
        Hmat_minus_q, nspins, f"-{q_label}"
    )
    if eigvals_m_sorted is None or eigvecs_m_sorted is None:
        return nan_matrix, nan_matrix, nan_eigs
    eigvecs_m_ortho = _apply_gram_schmidt(
        eigvals_m_sorted, eigvecs_m_sorted, DEGENERACY_THRESHOLD, f"-{q_label}"
    )
    alpha_m_sorted = _calculate_alpha_matrix(
        eigvecs_m_ortho, G_metric, ZERO_MATRIX_ELEMENT_THRESHOLD, f"-{q_label}"
    )
    if alpha_m_sorted is None:
        return nan_matrix, nan_matrix, nan_eigs
    (eigvecs_m_final, eigvals_m_reordered, alpha_m_final) = _match_and_reorder_minus_q(
        eigvecs_p_ortho,
        alpha_p,
        eigvecs_m_ortho,
        eigvals_m_sorted,
        alpha_m_sorted,
        nspins,
        EIGENVECTOR_MATCHING_THRESHOLD,
        ZERO_MATRIX_ELEMENT_THRESHOLD,
        q_label,
    )
    inv_T_p = eigvecs_p_ortho @ alpha_p
    inv_T_m_reordered = eigvecs_m_final @ alpha_m_final
    K_matrix, Kd_matrix = _calculate_K_Kd(
        Ud_numeric,
        spin_magnitude,
        nspins,
        inv_T_p,
        inv_T_m_reordered,
        ZERO_MATRIX_ELEMENT_THRESHOLD,
    )
    return K_matrix, Kd_matrix, eigvals_p_sorted


# --- Worker Functions (Keep outside class) ---
def process_calc_disp(
    args: Tuple[
        sp.Matrix,
        List[sp.Symbol],
        npt.NDArray[np.float_],
        int,
        float,
        Union[List[float], npt.NDArray[np.float_]],
    ],
) -> npt.NDArray[np.float_]:
    """
    Worker function for parallel dispersion calculation at a single q-point.

    This function is designed to be called by `multiprocessing.Pool.imap`.
    It takes the symbolic Hamiltonian, symbols, numerical parameters, and a
    single q-vector. It then:
    1. Lambdifies the symbolic Hamiltonian for fast numerical evaluation.
    2. Substitutes the numerical values (q, S, params) into the lambdified function.
    3. Calculates the eigenvalues of the resulting numerical matrix.
    4. Sorts the eigenvalues and extracts the positive energy modes (magnon energies).

    Args:
        args (Tuple): A tuple containing:
            HMat_sym (sp.Matrix): Symbolic Hamiltonian matrix (2gH).
            full_symbol_list (List[sp.Symbol]): List of all symbols [kx,ky,kz,S,p0,...].
            q_vector (npt.NDArray[np.float_]): The specific q-vector [qx, qy, qz].
            nspins (int): Number of spins in the unit cell.
            spin_magnitude_num (float): Numerical value of S.
            hamiltonian_params_num (Union[List[float], npt.NDArray[np.float_]]): Numerical parameters [p0, p1,...].
    Returns:
        npt.NDArray[np.float_]: Array of calculated magnon energies (N,) for the given q-point. Returns NaN array on failure.
    """
    (
        HMat_sym,
        full_symbol_list,
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
    ) = args
    q_label = f"q={q_vector}"
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)
    try:
        HMat_func = lambdify(full_symbol_list, HMat_sym, modules=["numpy"])
    except Exception:
        logger.exception(f"Error during lambdify at {q_label}.")
        return nan_energies
    try:
        numerical_args = (
            list(q_vector) + [spin_magnitude_num] + list(hamiltonian_params_num)
        )
        HMat_numeric: npt.NDArray[np.complex_] = np.array(
            HMat_func(*numerical_args), dtype=np.complex128
        )
    except Exception:
        logger.exception(f"Error evaluating HMat function at {q_label}.")
        return nan_energies
    eigenvalues: npt.NDArray[np.complex_]
    try:
        eigenvalues = la.eigvals(HMat_numeric)
    except np.linalg.LinAlgError:
        logger.error(f"Eigenvalue calculation failed for {q_label}.")
        return nan_energies
    except Exception:
        logger.exception(
            f"Unexpected error during eigenvalue calculation for {q_label}."
        )
        return nan_energies
    try:
        imag_part_mags: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues))
        if np.any(imag_part_mags > ENERGY_IMAG_PART_THRESHOLD):
            logger.warning(
                f"Significant imaginary part in eigenvalues for {q_label}. Max imag: {np.max(imag_part_mags)}"
            )
        eigenvalues_sorted_real: npt.NDArray[np.float_] = np.real(np.sort(eigenvalues))
        energies = eigenvalues_sorted_real[nspins:]
        if len(energies) != nspins:
            logger.warning(
                f"Unexpected number of positive energies ({len(energies)}) found for {q_label}. Expected {nspins}."
            )
            if len(energies) > nspins:
                energies = energies[:nspins]
            else:
                energies = np.pad(
                    energies, (0, nspins - len(energies)), constant_values=np.nan
                )
        return energies
    except Exception:
        logger.exception(f"Error during eigenvalue sorting/selection for {q_label}.")
        return nan_energies


def process_calc_Sqw(
    args: Tuple[
        sp.Matrix,
        npt.NDArray[np.complex_],
        List[sp.Symbol],
        npt.NDArray[np.float_],
        int,
        float,
        Union[List[float], npt.NDArray[np.float_]],
    ],
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Worker function for parallel S(q,w) calculation at a single q-point.

    This function is designed to be called by `multiprocessing.Pool.imap`.
    It takes the symbolic Hamiltonian, numerical Ud matrix, symbols, parameters,
    and a single q-vector. It then:
    1. Lambdifies the symbolic Hamiltonian.
    2. Evaluates the numerical Hamiltonian matrix at +q and -q.
    3. Calls `KKdMatrix` to perform diagonalization, matching, and calculate K, Kd, and eigenvalues.
    4. Calculates the spin-spin correlation functions using K and Kd.
    5. Applies the neutron scattering polarization factor.
    6. Returns the q-vector, calculated energies, and intensities.

    Args:
        args (Tuple): A tuple containing:
            HMat_sym (sp.Matrix): Symbolic Hamiltonian matrix (2gH).
            Ud_numeric (npt.NDArray[np.complex_]): Numerical rotation matrix Ud.
            full_symbol_list (List[sp.Symbol]): List of all symbols [kx,ky,kz,S,p0,...].
            q_vector (npt.NDArray[np.float_]): The specific q-vector [qx, qy, qz].
            nspins (int): Number of spins in the unit cell.
            spin_magnitude_num (float): Numerical value of S.
            hamiltonian_params_num (Union[List[float], npt.NDArray[np.float_]]): Numerical parameters [p0, p1,...].
    Returns:
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]: q-vector, energies (N,), intensities (N,). Returns NaNs on failure.
    """
    (
        HMat_sym,
        Ud_numeric,
        full_symbol_list,
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
    ) = args
    q_label = f"q={q_vector}"
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)
    nan_intensities: npt.NDArray[np.float_] = np.full((nspins,), np.nan)
    nan_result = (q_vector, nan_energies, nan_intensities)
    try:
        HMat_func = lambdify(full_symbol_list, HMat_sym, modules=["numpy"])
    except Exception:
        logger.exception(f"Error during lambdify at {q_label}.")
        return nan_result
    try:
        numerical_args_base = [spin_magnitude_num] + list(hamiltonian_params_num)
        numerical_args_plus_q = list(q_vector) + numerical_args_base
        numerical_args_minus_q = list(-q_vector) + numerical_args_base
        Hmat_plus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(*numerical_args_plus_q), dtype=np.complex128
        )
        Hmat_minus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(*numerical_args_minus_q), dtype=np.complex128
        )
    except Exception:
        logger.exception(f"Error evaluating HMat function at {q_label}.")
        return nan_result
    try:
        K_matrix, Kd_matrix, eigenvalues = KKdMatrix(
            spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_vector, nspins
        )
        if (
            np.isnan(K_matrix).any()
            or np.isnan(Kd_matrix).any()
            or np.isnan(eigenvalues).any()
        ):
            logger.error(f"NaN encountered in KKdMatrix result for {q_label}.")
            return nan_result
    except Exception:
        logger.exception(f"Unexpected error during KKdMatrix execution for {q_label}.")
        return nan_result
    try:
        imag_energy_mag: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues[0:nspins]))
        if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
            logger.warning(
                f"Significant imaginary part in energy eigenvalues for {q_label}. Max imag: {np.max(imag_energy_mag)}"
            )
        energies: npt.NDArray[np.float_] = np.real(eigenvalues[0:nspins])
        sqw_complex_accumulator: npt.NDArray[np.complex_] = np.zeros(
            nspins, dtype=complex
        )
        for mode_index in range(nspins):
            spin_correlation_matrix: npt.NDArray[np.complex_] = np.zeros(
                (3, 3), dtype=complex
            )
            intensity_one_mode: complex = 0.0 + 0.0j
            for alpha in range(3):
                for beta in range(3):
                    correlation_sum: complex = 0.0 + 0.0j
                    for spin_i in range(nspins):
                        for spin_j in range(nspins):
                            idx_K: int = 3 * spin_i + alpha
                            idx_Kd: int = 3 * spin_j + beta
                            correlation_sum += (
                                K_matrix[idx_K, mode_index]
                                * Kd_matrix[idx_Kd, mode_index + nspins]
                            )
                    spin_correlation_matrix[alpha, beta] = correlation_sum
            q_norm_sq: float = np.dot(q_vector, q_vector)
            if q_norm_sq < Q_ZERO_THRESHOLD:
                for alpha in range(3):
                    intensity_one_mode += spin_correlation_matrix[alpha, alpha]
            else:
                q_normalized: npt.NDArray[np.float_] = q_vector / np.sqrt(q_norm_sq)
                for alpha in range(3):
                    for beta in range(3):
                        delta_ab: float = 1.0 if alpha == beta else 0.0
                        polarization_factor: float = (
                            delta_ab - q_normalized[alpha] * q_normalized[beta]
                        )
                        intensity_one_mode += (
                            polarization_factor * spin_correlation_matrix[alpha, beta]
                        )
            if np.abs(np.imag(intensity_one_mode)) > SQW_IMAG_PART_THRESHOLD:
                logger.warning(
                    f"Significant imaginary part in Sqw for {q_label}, mode {mode_index}: {np.imag(intensity_one_mode)}"
                )
            sqw_complex_accumulator[mode_index] = intensity_one_mode
        intensities: npt.NDArray[np.float_] = np.real(sqw_complex_accumulator)
        intensities[intensities < 0] = 0
        return q_vector, energies, intensities
    except Exception:
        logger.exception(f"Error during intensity calculation for {q_label}.")
        return nan_result


# --- gen_HM Helper Functions ---


def _setup_hp_operators(
    nspins_ouc: int, S_sym: sp.Symbol
) -> Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Matrix]]:
    """
    Create symbolic Holstein-Primakoff boson operators and local spin operators.

    Args:
        nspins_ouc (int): Number of spins in the original unit cell + neighbors (OUC).
        S_sym (sp.Symbol): Symbolic representation of the spin magnitude S.
    Returns:
        Tuple[List[sp.Symbol], List[sp.Symbol], List[sp.Matrix]]: Boson annihilation operators (c), creation operators (cd), and local spin operators (Sx, Sy, Sz) expressed in terms of bosons.
    """
    c_ops = sp.symbols("c0:%d" % nspins_ouc, commutative=False)
    cd_ops = sp.symbols("cd0:%d" % nspins_ouc, commutative=False)
    spin_ops_local = [
        sp.Matrix(
            (
                sp.sqrt(S_sym / 2) * (c_ops[i] + cd_ops[i]),
                sp.sqrt(S_sym / 2) * (c_ops[i] - cd_ops[i]) / I,
                S_sym - cd_ops[i] * c_ops[i],
            )
        )
        for i in range(nspins_ouc)
    ]
    return c_ops, cd_ops, spin_ops_local


def _rotate_spin_operators(
    spin_ops_local: List[sp.Matrix],
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]],
    nspins: int,
    nspins_ouc: int,
) -> List[sp.Matrix]:
    """
    Rotate local spin operators (defined along local z-axis) to the global frame.

    Args:
        spin_ops_local (List[sp.Matrix]): List of local spin operators (3x1 matrices) for each spin in OUC.
        rotation_matrices (List[Union[npt.NDArray, sp.Matrix]]): List of rotation matrices (3x3) for each spin in the magnetic unit cell. These are applied cyclically to the OUC spins.
        nspins (int): Number of spins in the magnetic unit cell.
        nspins_ouc (int): Number of spins in the OUC.
    Returns:
        List[sp.Matrix]: List of global spin operators (3x1 matrices) for each spin in OUC.
    """
    spin_ops_global_ouc = [
        rotation_matrices[j] * spin_ops_local[nspins * i + j]
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]
    return spin_ops_global_ouc


def _prepare_hamiltonian(
    spin_model_module: types.ModuleType,
    spin_ops_global_ouc: List[sp.Matrix],
    params_sym: List[sp.Symbol],
    S_sym: sp.Symbol,
) -> sp.Expr:
    """
    Construct the symbolic Hamiltonian using the user-defined model.

    Retrieves the Hamiltonian expression from the `spin_model_module`, expands it,
    and attempts to filter out terms higher than quadratic in boson operators
    by analyzing powers of the symbolic spin S.

    Args:
        spin_model_module (types.ModuleType): The user-provided spin model module.
        spin_ops_global_ouc (List[sp.Matrix]): List of global spin operators for OUC spins.
        params_sym (List[sp.Symbol]): List of symbolic Hamiltonian parameters.
        S_sym (sp.Symbol): Symbolic spin magnitude.
    Returns:
        sp.Expr: The expanded and filtered symbolic Hamiltonian expression (up to quadratic boson terms).
    """
    hamiltonian_sym = spin_model_module.Hamiltonian(spin_ops_global_ouc, params_sym)
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    # --- Filter Hamiltonian terms (Keep only up to quadratic in boson ops) ---
    # This logic seems specific to how the Hamiltonian is constructed in the model.
    # It assumes higher powers of S correspond to higher orders in boson operators.
    # A more robust approach might involve explicitly checking boson operator powers.
    hamiltonian_S0 = hamiltonian_sym.coeff(S_sym, 0)
    if params_sym:
        # This part seems potentially problematic or overly specific.
        # It keeps the term linear in the *last* parameter from the S^0 part,
        # plus the S^1 and S^2 terms. Revisit if this causes issues.
        hamiltonian_sym = (
            hamiltonian_S0.coeff(params_sym[-1]) * params_sym[-1]
            + hamiltonian_sym.coeff(S_sym, 1) * S_sym
            + hamiltonian_sym.coeff(S_sym, 2) * S_sym**2
        )
    else:
        hamiltonian_sym = (
            hamiltonian_sym.coeff(S_sym, 1) * S_sym
            + hamiltonian_sym.coeff(S_sym, 2) * S_sym**2
        )
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    # --- End Filtering ---
    return hamiltonian_sym


def _define_fourier_substitutions(
    spin_model_module: types.ModuleType,
    k_sym: List[sp.Symbol],
    nspins: int,
    nspins_ouc: int,
    c_ops: List[sp.Symbol],
    cd_ops: List[sp.Symbol],
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
    params_sym: List[sp.Symbol],  # ADDED: Pass symbolic parameters
) -> List[List[sp.Expr]]:
    """
    Define the substitution rules for Fourier transforming boson operator pairs.

    Generates a list of [old_expression, new_expression] pairs to substitute
    real-space boson operator products (e.g., c_i*c_j, cd_i*c_j) with their
    k-space equivalents involving ck, ckd, cmk, cmkd operators and phase factors
    exp(+/-(I * k . dr_ij)). Only pairs corresponding to non-zero interactions
    in the `spin_model_module` are considered. Includes diagonal terms (cd_j*c_j).

    Args:
        spin_model_module (types.ModuleType): The user spin model module.
        k_sym (List[sp.Symbol]): Symbolic momentum vector [kx, ky, kz].
        nspins (int): Number of spins in the magnetic unit cell.
        nspins_ouc (int): Number of spins in the OUC.
        c_ops (List[sp.Symbol]): Real-space annihilation operators (OUC).
        cd_ops (List[sp.Symbol]): Real-space creation operators (OUC).
        ck_ops (List[sp.Symbol]): k-space annihilation operators (UC).
        ckd_ops (List[sp.Symbol]): k-space creation operators (UC).
        cmk_ops (List[sp.Symbol]): -k-space annihilation operators (UC).
        cmkd_ops (List[sp.Symbol]): -k-space creation operators (UC).
        params_sym (List[sp.Symbol]): Symbolic Hamiltonian parameters (used to get interaction matrix).
    Returns:
        List[List[sp.Expr]]: A list of [old_expr, new_expr] substitution pairs for the Fourier transform.
    """
    atom_positions_uc = spin_model_module.atom_pos()
    atom_positions_ouc = spin_model_module.atom_pos_ouc()
    # Use params_sym when calling spin_interactions
    interaction_matrix = spin_model_module.spin_interactions(params_sym)[
        0  # Assuming params don't change interaction pairs
    ]  # Assuming params don't change interaction pairs

    fourier_substitutions = []

    # Revert to iterating over UC spins (i) and OUC spins (j)
    # This matches the original logic and should be more efficient.
    for i in range(nspins):  # Iterate over spins in the magnetic unit cell
        for j in range(nspins_ouc):
            # Check if interaction exists (use OUC indices)
            # interaction_matrix[i_uc, j_ouc]
            if interaction_matrix[i, j] == 0:
                continue

            # Calculate displacement vector between UC atom i and OUC atom j
            # Original code used: atom_positions_uc[i, :] - atom_positions_ouc[j, :]
            # Let's stick to that for consistency.
            disp_vec = atom_positions_uc[i, :] - atom_positions_ouc[j, :]
            disp_vec = atom_positions_ouc[i, :] - atom_positions_ouc[j, :]
            k_dot_dr = sum(k * dr for k, dr in zip(k_sym, disp_vec))
            exp_plus_ikdr = sp.exp(I * k_dot_dr).rewrite(sp.sin)
            exp_minus_ikdr = sp.exp(-I * k_dot_dr).rewrite(sp.sin)

            # Map OUC index 'i' and 'j' to UC index for k-space operators
            # 'i' already refers to the UC index in this loop structure
            j_uc = j % nspins

            # Define substitutions for this pair (i, j)
            sub_list = [
                [
                    cd_ops[i] * cd_ops[j],
                    1
                    / 2
                    * (
                        ckd_ops[i] * cmkd_ops[j_uc] * exp_minus_ikdr
                        + cmkd_ops[i] * ckd_ops[j_uc] * exp_plus_ikdr
                    ),
                ],
                [
                    c_ops[i] * c_ops[j],
                    1
                    / 2
                    * (
                        ck_ops[i] * cmk_ops[j_uc] * exp_plus_ikdr
                        + cmk_ops[i] * ck_ops[j_uc] * exp_minus_ikdr
                    ),
                ],
                [
                    cd_ops[i] * c_ops[j],
                    1
                    / 2
                    * (
                        ckd_ops[i] * ck_ops[j_uc] * exp_minus_ikdr
                        + cmkd_ops[i] * cmk_ops[j_uc] * exp_plus_ikdr
                    ),
                ],
                # c_i * cd_j is handled by commutation rules later if needed,
                # but let's add it for completeness if the Hamiltonian contains it directly.
                [
                    c_ops[i] * cd_ops[j],
                    (
                        ck_ops[i]
                        * ckd_ops[j_uc]
                        * exp_plus_ikdr  # Use 'i' instead of 'i_uc'
                        + cmk_ops[i] * cmkd_ops[j_uc] * exp_minus_ikdr
                    )
                    / 2,
                ],
            ]
            fourier_substitutions.extend(sub_list)

    # Add the diagonal term substitution (present in original code, seems important)
    # This handles terms like cd_j * c_j which remain after HP transform of Sz
    for j in range(nspins_ouc):
        # Only add if the diagonal term involves a spin within the UC
        # The original code implicitly handled this via the outer loop range(nspins)
        # Let's add it explicitly for all j_ouc, then rely on duplicate removal.
        j_uc = j % nspins  # Map OUC index j to UC index
        fourier_substitutions.append(
            [
                cd_ops[j] * c_ops[j],
                1 / 2 * (ckd_ops[j_uc] * ck_ops[j_uc] + cmkd_ops[j_uc] * cmk_ops[j_uc]),
            ]
        )  # Correct closing bracket ']' instead of ')'

    # Remove duplicates (important for efficiency)
    unique_substitutions = []
    seen_keys = set()
    for sub in fourier_substitutions:
        key = sub[0]
        if key not in seen_keys:
            unique_substitutions.append(sub)
            seen_keys.add(key)

    return unique_substitutions


def _define_commutation_substitutions(
    nspins: int,
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
) -> List[List[sp.Expr]]:
    """
    Define substitution rules for applying boson commutation relations in k-space.

    Generates [old, new] pairs to enforce commutation rules like [ck_i, ckd_j] = delta_ij,
    [cmkd_i, cmk_j] = delta_ij, and others ([ck, cmk]=0, [cmkd, ckd]=0). This puts
    the Hamiltonian into normal order (creation operators to the left).

    Args:
        nspins (int): Number of spins in the magnetic unit cell.
        ck_ops, ckd_ops, cmk_ops, cmkd_ops (List[sp.Symbol]): k-space boson operators.
    Returns:
        List[List[sp.Expr]]: Substitution pairs for commutation rules.
    """
    commutation_substitutions = (
        [
            [ck_ops[i] * ckd_ops[j], ckd_ops[j] * ck_ops[i] + (1 if i == j else 0)]
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [cmkd_ops[i] * cmk_ops[j], cmk_ops[j] * cmkd_ops[i] + (1 if i == j else 0)]
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [ck_ops[i] * cmk_ops[j], cmk_ops[j] * ck_ops[i]]
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [cmkd_ops[i] * ckd_ops[j], ckd_ops[j] * cmkd_ops[i]]
            for i in range(nspins)
            for j in range(nspins)
        ]
    )
    return commutation_substitutions


def _define_placeholder_substitutions(
    nspins: int, basis_vector_dagger: List[sp.Symbol], basis_vector: List[sp.Symbol]
) -> Tuple[List[sp.Symbol], List[List[sp.Expr]]]:
    """
    Define placeholder symbols and substitutions to extract the H2 matrix elements.

    Creates unique commutative symbols (e.g., "XdX0", "XdX1", ...) and substitution
    rules to replace the normal-ordered k-space boson pairs (e.g., ckd_i * ck_j)
    with these placeholders. This allows extracting the coefficients of these pairs,
    which form the elements of the quadratic Hamiltonian matrix H2.

    Args:
        nspins (int): Number of spins in the magnetic unit cell.
        basis_vector_dagger (List[sp.Symbol]): Combined list [ckd_ops, cmk_ops].
        basis_vector (List[sp.Symbol]): Combined list [ck_ops, cmkd_ops].
    Returns:
        Tuple[List[sp.Symbol], List[List[sp.Expr]]]: List of placeholder symbols and the corresponding substitution rules.
    """
    nspins2 = 2 * nspins
    placeholder_symbols = [
        sp.Symbol("XdX%d" % (i * nspins2 + j), commutative=True)
        for i in range(nspins2)
        for j in range(nspins2)
    ]
    placeholder_substitutions = [
        [basis_vector_dagger[i] * basis_vector[j], placeholder_symbols[i * nspins2 + j]]
        for i in range(nspins2)
        for j in range(nspins2)
    ]
    return placeholder_symbols, placeholder_substitutions


def _apply_substitutions_parallel(
    hamiltonian_sym: sp.Expr,
    fourier_substitutions: List[List[sp.Expr]],
    commutation_substitutions: List[List[sp.Expr]],
    placeholder_substitutions: List[List[sp.Expr]],
) -> sp.Expr:
    """
    Apply sequential substitutions (Fourier, Commutation, Placeholder) in parallel.

    Uses `multiprocessing.Pool` to apply the substitution rules defined by
    `_define_fourier_substitutions`, `_define_commutation_substitutions`, and
    `_define_placeholder_substitutions` to the terms of the Hamiltonian expression.
    This is often the most time-consuming part of the symbolic calculation.

    Args:
        hamiltonian_sym (sp.Expr): The initial symbolic Hamiltonian (quadratic in bosons).
        fourier_substitutions (List[List[sp.Expr]]): Substitutions for Fourier transform.
        commutation_substitutions (List[List[sp.Expr]]): Substitutions for commutation rules.
        placeholder_substitutions (List[List[sp.Expr]]): Substitutions for placeholders.
    Returns:
        sp.Expr: The Hamiltonian expression with all substitutions applied, containing placeholder symbols.
    """
    logger.info("Applying Fourier transform substitutions...")
    start_time_ft = timeit.default_timer()
    hamiltonian_terms = hamiltonian_sym.as_ordered_terms()
    pool_args_ft = [(expr, fourier_substitutions) for expr in hamiltonian_terms]
    with Pool() as pool:
        results_ft = list(
            tqdm(
                pool.imap(substitute_expr, pool_args_ft),
                total=len(hamiltonian_terms),
                desc="Substituting FT ",
                bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
            )
        )
    # --- Optimization: Remove intermediate expansion ---
    hamiltonian_k_space = Add(*results_ft)
    # --- End Optimization ---
    logger.info(
        f"Fourier transform substitution took: {timeit.default_timer() - start_time_ft:.2f} s"
    )

    logger.info("Applying commutation rule substitutions...")
    start_time_comm = timeit.default_timer()
    hamiltonian_k_terms = hamiltonian_k_space.as_ordered_terms()
    pool_args_comm = [(expr, commutation_substitutions) for expr in hamiltonian_k_terms]
    with Pool() as pool:
        results_comm = list(
            tqdm(
                pool.imap(substitute_expr, pool_args_comm),
                total=len(hamiltonian_k_terms),
                desc="Applying Commutation",
                bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
            )
        )
    hamiltonian_k_commuted = Add(*results_comm).expand()
    logger.info(
        f"Commutation rule substitution took: {timeit.default_timer() - start_time_comm:.2f} s"
    )

    logger.info("Applying placeholder substitutions...")
    start_time_ph = timeit.default_timer()
    hamiltonian_k_comm_terms = hamiltonian_k_commuted.as_ordered_terms()
    pool_args_placeholder = [
        (expr, placeholder_substitutions) for expr in hamiltonian_k_comm_terms
    ]
    with Pool() as pool:
        results_placeholder = list(
            tqdm(
                pool.imap(substitute_expr, pool_args_placeholder),
                total=len(hamiltonian_k_comm_terms),
                desc="Substituting Placeholders",
                bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
            )
        )
    hamiltonian_with_placeholders = Add(*results_placeholder)
    logger.info(
        f"Placeholder substitution took: {timeit.default_timer() - start_time_ph:.2f} s"
    )

    return hamiltonian_with_placeholders


def _extract_h2_matrix(
    hamiltonian_with_placeholders: sp.Expr,
    placeholder_symbols: List[sp.Symbol],
    nspins: int,
) -> sp.Matrix:
    """
    Extract the quadratic Hamiltonian matrix H2 from the placeholder expression.

    After substitutions, the Hamiltonian is expressed as a sum of terms, each
    being a coefficient multiplied by a placeholder symbol (XdXij). This function
    extracts these coefficients to form the H2 matrix. H = C + X^dagger H2 X.

    Args:
        hamiltonian_with_placeholders (sp.Expr): Hamiltonian with placeholder symbols.
        placeholder_symbols (List[sp.Symbol]): The list of placeholder symbols used.
        nspins (int): Number of spins in the unit cell.
    Returns:
        sp.Matrix: The symbolic H2 matrix (2N x 2N).
    """
    nspins2 = 2 * nspins
    H2_elements = [hamiltonian_with_placeholders.coeff(p) for p in placeholder_symbols]
    H2_matrix = sp.Matrix(nspins2, nspins2, H2_elements)
    return H2_matrix


def _build_ud_matrix(
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]], nspins: int
) -> sp.Matrix:
    """
    Construct the block-diagonal rotation matrix Ud.

    Ud transforms vectors of local spin components (for all N spins) to vectors
    of global spin components. It's a block diagonal matrix where each block is
    one of the 3x3 rotation matrices provided by the spin model's `mpr` function.

    Args:
        rotation_matrices (List[Union[npt.NDArray, sp.Matrix]]): List of 3x3 rotation matrices (one per spin in UC).
        nspins (int): Number of spins in the unit cell.
    Returns:
        sp.Matrix: The symbolic block-diagonal Ud matrix (3N x 3N).
    """
    Ud_rotation_matrix_blocks = []
    for i in range(nspins):
        rot_mat = rotation_matrices[i]
        # Ensure it's a SymPy matrix
        if isinstance(rot_mat, np.ndarray):
            rot_mat_sym = sp.Matrix(rot_mat)
        else:
            rot_mat_sym = rot_mat
        Ud_rotation_matrix_blocks.append(rot_mat_sym)
    Ud_rotation_matrix = sp.diag(*Ud_rotation_matrix_blocks)
    return Ud_rotation_matrix


# --- Main gen_HM Function (Calls Helpers) ---
def gen_HM(
    spin_model_module,  # ADDED: Explicitly pass the spin model module
    k_sym: List[sp.Symbol],
    S_sym: sp.Symbol,
    params_sym: List[sp.Symbol],
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Generate the symbolic dynamical matrix (2gH) and rotation matrix (Ud).

    This is the main function for the symbolic part of the LSWT calculation.
    It orchestrates the process:
    1. Get model info (positions, rotations) from `spin_model_module`.
    2. Set up Holstein-Primakoff boson operators.
    3. Rotate local spin operators to global frame.
    4. Construct the Hamiltonian using the model's `Hamiltonian` function.
    5. Filter Hamiltonian to keep only terms up to quadratic in bosons.
    6. Define Fourier transform, commutation, and placeholder substitutions.
    7. Apply substitutions (often using multiprocessing).
    8. Extract the quadratic Hamiltonian matrix H2.
    9. Calculate the dynamical matrix TwogH2 = 2 * g * H2.
    10. Build the block-diagonal rotation matrix Ud.

    Args:
        spin_model_module (module): The user-provided spin model module.
        k_sym (List[sp.Symbol]): Symbolic momentum vector [kx, ky, kz].
        S_sym (sp.Symbol): Symbolic spin magnitude S.
        params_sym (List[sp.Symbol]): List of symbolic Hamiltonian parameters [p0, p1,...].
    Returns:
        Tuple[sp.Matrix, sp.Matrix]: The symbolic dynamical matrix (2gH, 2N x 2N) and the symbolic rotation matrix (Ud, 3N x 3N).
    """
    logger.info("Starting symbolic matrix generation (gen_HM)...")
    start_time_total = timeit.default_timer()

    # --- Get Model Info ---
    try:
        atom_positions_uc = spin_model_module.atom_pos()
        nspins = len(atom_positions_uc)
        atom_positions_ouc = spin_model_module.atom_pos_ouc()
        nspins_ouc = len(atom_positions_ouc)
        rotation_matrices = spin_model_module.mpr(params_sym)
    except Exception as e:
        logger.exception("Error retrieving data from spin_model_module.")
        raise RuntimeError("Failed to get model info for gen_HM.") from e

    logger.info(f"Number of spins in the unit cell: {nspins}")
    nspins2 = 2 * nspins

    # --- Setup Operators ---
    c_ops, cd_ops, spin_ops_local = _setup_hp_operators(nspins_ouc, S_sym)
    spin_ops_global_ouc = _rotate_spin_operators(
        spin_ops_local, rotation_matrices, nspins, nspins_ouc
    )

    # --- Prepare Hamiltonian ---
    hamiltonian_sym = _prepare_hamiltonian(
        spin_model_module, spin_ops_global_ouc, params_sym, S_sym
    )

    # --- Define k-space operators ---
    ck_ops = [sp.Symbol("ck%d" % j, commutative=False) for j in range(nspins)]
    ckd_ops = [sp.Symbol("ckd%d" % j, commutative=False) for j in range(nspins)]
    cmk_ops = [sp.Symbol("cmk%d" % j, commutative=False) for j in range(nspins)]
    cmkd_ops = [sp.Symbol("cmkd%d" % j, commutative=False) for j in range(nspins)]

    # --- Define Substitution Rules ---
    fourier_substitutions = _define_fourier_substitutions(
        spin_model_module,
        k_sym,
        nspins,
        nspins_ouc,
        c_ops,
        cd_ops,
        ck_ops,
        ckd_ops,
        cmk_ops,
        cmkd_ops,
        params_sym,  # Pass params_sym here
    )
    commutation_substitutions = _define_commutation_substitutions(
        nspins, ck_ops, ckd_ops, cmk_ops, cmkd_ops
    )
    basis_vector_dagger = ckd_ops[:nspins] + cmk_ops[:nspins]
    basis_vector = ck_ops[:nspins] + cmkd_ops[:nspins]
    placeholder_symbols, placeholder_substitutions = _define_placeholder_substitutions(
        nspins, basis_vector_dagger, basis_vector
    )

    # --- Apply Substitutions ---
    try:
        hamiltonian_with_placeholders = _apply_substitutions_parallel(
            hamiltonian_sym,
            fourier_substitutions,
            commutation_substitutions,
            placeholder_substitutions,
        )
    except Exception as e:
        logger.exception("Error during symbolic substitution in gen_HM.")
        raise RuntimeError("Symbolic substitution failed.") from e

    # --- Extract H2 Matrix ---
    H2_matrix = _extract_h2_matrix(
        hamiltonian_with_placeholders, placeholder_symbols, nspins
    )

    # --- Calculate TwogH2 ---
    g_metric_tensor_sym = sp.diag(*([1] * nspins + [-1] * nspins))
    dynamical_matrix_TwogH2 = 2 * g_metric_tensor_sym * H2_matrix

    # --- Build Ud Matrix ---
    Ud_rotation_matrix = _build_ud_matrix(rotation_matrices, nspins)

    end_time_total = timeit.default_timer()
    logger.info(
        f"Total run-time for gen_HM: {np.round((end_time_total - start_time_total), 2)} s."
    )

    return dynamical_matrix_TwogH2, Ud_rotation_matrix


# --- MagCalc Class ---
class MagCalc:
    """
    Performs Linear Spin Wave Theory (LSWT) calculations.

    This class handles the setup of symbolic Hamiltonian matrices based on a
    user-provided spin model and calculates spin-wave dispersion relations
    and dynamic structure factors S(q,w) using multiprocessing.

    Attributes:
        spin_magnitude (float): Numerical value of the spin magnitude S.
        hamiltonian_params (List[float]): Numerical Hamiltonian parameters.
        cache_file_base (str): Base name for cache files.
        cache_mode (str): Cache mode ('r' or 'w').
        sm: The imported spin model module.
        nspins (int): Number of spins in the magnetic unit cell.
        k_sym (List[sp.Symbol]): List of momentum symbols [kx, ky, kz].
        S_sym (sp.Symbol): Symbolic spin magnitude 'S'.
        params_sym (Tuple[sp.Symbol, ...]): Tuple of symbolic parameters (p0, p1,...).
        full_symbol_list (List[sp.Symbol]): Combined list of all symbols for HMat lambdification.
        HMat_sym (Optional[sp.Matrix]): Symbolic Hamiltonian matrix (2gH).
        Ud_sym (Optional[sp.Matrix]): Symbolic rotation matrix Ud.
        Ud_numeric (Optional[npt.NDArray[np.complex_]]): Numerical rotation matrix Ud.
    """

    def __init__(
        self,
        spin_magnitude: float,
        hamiltonian_params: Union[List[float], npt.NDArray[np.float_]],
        cache_file_base: str,
        spin_model_module: types.ModuleType,  # Moved before cache_mode
        cache_mode: str = "r",
    ):
        """
        Initializes the MagCalc LSWT calculator.

        Loads or generates the necessary symbolic matrices (Hamiltonian HMat=2gH,
        rotation Ud) based on the provided spin model and parameters. Pre-calculates
        the numerical rotation matrix Ud_numeric.

        Args:
            spin_magnitude (float): The numerical value of the spin magnitude S.
                Must be positive.
            hamiltonian_params (Union[List[float], npt.NDArray[np.float_]]):
                A list or NumPy array containing the numerical values for the
                Hamiltonian parameters expected by the spin_model_module.
            cache_file_base (str): The base filename (without path or extension)
                used for storing/retrieving cached symbolic matrices (HMat, Ud)
                in the 'pckFiles' subdirectory.
            spin_model_module (module): The imported Python module
                containing the spin model definitions (e.g., Hamiltonian, mpr, atom_pos).
                This argument is REQUIRED.
            cache_mode (str, optional): Specifies the cache behavior.
                'r': Read symbolic matrices from cache files. Fails if files
                     don't exist or are invalid. (Default)
                'w': Generate symbolic matrices (potentially slow) and write
                     them to cache files.

        Raises:
            TypeError: If spin_magnitude is not a number or hamiltonian_params
                       is not a list/array of numbers.
            ValueError: If spin_magnitude is not positive, hamiltonian_params is empty,
                        cache_mode is invalid, or cache files are missing in 'r' mode.
            AttributeError: If the spin_model_module is missing required functions.
            RuntimeError: If symbolic matrix generation/loading or Ud_numeric
                          calculation fails for unexpected reasons.
            FileNotFoundError: If cache files are not found in 'r' mode.
            pickle.PickleError: If cache files are corrupted or incompatible.
        """
        logger.info(f"Initializing MagCalc (cache_mode='{cache_mode}')...")

        # --- Input Validation ---
        if not isinstance(spin_magnitude, (int, float)):
            raise TypeError("spin_magnitude must be a number.")
        if spin_magnitude <= 0:
            raise ValueError("spin_magnitude must be positive.")
        self.spin_magnitude = float(spin_magnitude)

        if isinstance(hamiltonian_params, np.ndarray):
            hamiltonian_params = hamiltonian_params.tolist()  # Convert numpy array
        if not isinstance(hamiltonian_params, list) or not hamiltonian_params:
            raise TypeError(
                "hamiltonian_params must be a non-empty list or NumPy array."
            )
        if not all(isinstance(p, (int, float)) for p in hamiltonian_params):
            raise TypeError("All elements in hamiltonian_params must be numbers.")
        self.hamiltonian_params = [float(p) for p in hamiltonian_params]

        if not isinstance(cache_file_base, str) or not cache_file_base:
            raise ValueError("cache_file_base must be a non-empty string.")
        self.cache_file_base = cache_file_base

        if cache_mode not in ["r", "w"]:
            raise ValueError(f"Invalid cache_mode '{cache_mode}'. Use 'r' or 'w'.")
        self.cache_mode = cache_mode

        if not hasattr(spin_model_module, "__name__"):
            raise TypeError("spin_model_module does not appear to be a valid module.")
        self.sm = spin_model_module
        # --- End Input Validation ---

        # --- Check spin_model validity ---
        required_funcs = [
            "atom_pos",
            "atom_pos_ouc",
            "mpr",
            "Hamiltonian",
            "spin_interactions",
        ]
        missing_funcs = [
            f
            for f in required_funcs
            if not hasattr(self.sm, f) or not callable(getattr(self.sm, f))
        ]
        if missing_funcs:
            raise AttributeError(
                f"Required function(s) {missing_funcs} not found or not callable in spin_model_module '{self.sm.__name__}'."
            )

        # --- Setup symbolic variables ---
        try:
            self.nspins = len(self.sm.atom_pos())
            if self.nspins == 0:
                raise ValueError("spin_model.atom_pos() returned an empty list.")
        except Exception as e:
            logger.exception("Error getting nspins from spin_model.atom_pos()")
            raise RuntimeError("Failed to determine nspins from spin model.") from e

        self.kx, self.ky, self.kz = sp.symbols("kx ky kz", real=True)
        self.k_sym: List[sp.Symbol] = [self.kx, self.ky, self.kz]
        self.S_sym: sp.Symbol = sp.Symbol("S", real=True)
        # Ensure correct number of symbols matches parameters provided
        num_params = len(self.hamiltonian_params)
        self.params_sym: Tuple[sp.Symbol, ...] = sp.symbols(
            f"p0:{num_params}", real=True
        )
        if num_params == 1:  # sp.symbols('p0:1') returns a single symbol, not tuple
            self.params_sym = (self.params_sym,)

        self.full_symbol_list: List[sp.Symbol] = (
            self.k_sym + [self.S_sym] + list(self.params_sym)
        )

        # --- Load or Generate Symbolic Matrices ---
        self.HMat_sym: Optional[sp.Matrix] = None
        self.Ud_sym: Optional[sp.Matrix] = None
        # _load_or_generate_matrices raises exceptions on failure
        self._load_or_generate_matrices()

        # --- Pre-calculate numerical Ud ---
        self.Ud_numeric: Optional[npt.NDArray[np.complex_]] = None
        if self.Ud_sym is not None:
            # _calculate_numerical_ud raises exceptions on failure
            self._calculate_numerical_ud()
        else:
            # This case should ideally be caught by _load_or_generate_matrices
            raise RuntimeError("Ud_sym is None after matrix loading/generation.")

        logger.info("MagCalc initialization complete.")

    def _load_or_generate_matrices(self):
        """
        Load symbolic HMat (2gH) and Ud matrices from cache or generate them.

        Handles reading from `.pck` files in `pckFiles/` if `cache_mode='r'`, or calling `gen_HM` and writing the files if `cache_mode='w'`. Ensures the cache directory exists.
        """
        # Ensure cache directory exists
        cache_dir = "pckFiles"
        if not os.path.exists(cache_dir):
            logger.info(f"Creating directory '{cache_dir}' for caching.")
            try:
                os.makedirs(cache_dir, exist_ok=True)  # Allow directory to exist
            except OSError as e:
                logger.error(f"Failed to create cache directory '{cache_dir}': {e}")
                raise

        hm_cache_file: str = os.path.join(cache_dir, self.cache_file_base + "_HM.pck")
        ud_cache_file: str = os.path.join(cache_dir, self.cache_file_base + "_Ud.pck")

        if self.cache_mode == "w":
            logger.info("Generating symbolic matrices (HMat, Ud)...")
            try:
                # gen_HM expects list of param symbols
                self.HMat_sym, self.Ud_sym = gen_HM(
                    self.sm,
                    self.k_sym,
                    self.S_sym,
                    list(self.params_sym),  # Pass self.sm
                )
            except Exception as e:
                logger.exception("Failed to generate symbolic matrices in gen_HM.")
                raise RuntimeError("Symbolic matrix generation failed.") from e

            if not isinstance(self.HMat_sym, sp.Matrix) or not isinstance(
                self.Ud_sym, sp.Matrix
            ):
                raise RuntimeError("gen_HM did not return valid SymPy Matrices.")

            logger.info(f"Writing HMat to {hm_cache_file}")
            try:
                with open(hm_cache_file, "wb") as outHM:
                    pickle.dump(self.HMat_sym, outHM)
            except (IOError, pickle.PicklingError) as e:
                logger.error(f"Error writing HMat cache file: {e}")
                raise
            logger.info(f"Writing Ud to {ud_cache_file}")
            try:
                with open(ud_cache_file, "wb") as outUd:
                    pickle.dump(self.Ud_sym, outUd)
            except (IOError, pickle.PicklingError) as e:
                logger.error(f"Error writing Ud cache file: {e}")
                raise

        elif self.cache_mode == "r":
            logger.info(
                f"Importing symbolic matrices from cache files ({hm_cache_file}, {ud_cache_file})..."
            )
            try:
                with open(hm_cache_file, "rb") as inHM:
                    self.HMat_sym = pickle.load(inHM)
                with open(ud_cache_file, "rb") as inUd:
                    self.Ud_sym = pickle.load(inUd)
            except FileNotFoundError as e:
                logger.error(
                    f"Cache file not found: {e}. Run with cache_mode='w' first."
                )
                raise  # Re-raise specific error
            except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as e:
                logger.error(
                    f"Error loading cache files (may be corrupted or incompatible): {e}"
                )
                raise pickle.PickleError(
                    "Failed to load cache file."
                ) from e  # Raise consistent error type
            except Exception as e:
                logger.exception("An unexpected error occurred loading cache files.")
                raise RuntimeError(
                    "Cache file loading failed."
                ) from e  # General runtime error

        # Final check after load/generate
        if self.HMat_sym is None or self.Ud_sym is None:
            raise RuntimeError(
                "Symbolic matrices HMat_sym or Ud_sym are None after loading/generation."
            )
        if not isinstance(self.HMat_sym, sp.Matrix) or not isinstance(
            self.Ud_sym, sp.Matrix
        ):
            raise TypeError("Loaded cache files do not contain valid SymPy Matrices.")

    def _calculate_numerical_ud(self):
        """
        Calculate the numerical Ud matrix by substituting parameters into Ud_sym.

        Uses the current `self.spin_magnitude` and `self.hamiltonian_params`
        to substitute values into the symbolic `self.Ud_sym` matrix and stores
        the result in `self.Ud_numeric`.
        """
        # Ud_sym existence checked in __init__ before calling this
        logger.info("Calculating numerical Ud matrix...")
        # Ensure correct number of symbols/params match
        if len(self.params_sym) != len(self.hamiltonian_params):
            raise ValueError(
                f"Mismatch between number of symbolic params ({len(self.params_sym)}) and numerical params ({len(self.hamiltonian_params)})."
            )

        param_substitutions_ud: List[Tuple[sp.Symbol, float]] = [
            (self.S_sym, self.spin_magnitude)
        ] + list(zip(self.params_sym, self.hamiltonian_params))

        try:
            # Use evalf(subs=...) for potentially better performance/stability
            Ud_num_sym = self.Ud_sym.evalf(subs=dict(param_substitutions_ud))
            self.Ud_numeric = np.array(Ud_num_sym, dtype=np.complex128)
        except Exception as e:
            logger.exception(
                "Error during substitution/evaluation into symbolic Ud matrix."
            )
            raise RuntimeError("Failed to calculate numerical Ud matrix.") from e

        if self.Ud_numeric is None:  # Should not happen if evalf succeeds
            raise RuntimeError("Ud_numeric calculation resulted in None.")

    # --- NEW METHODS ---
    def update_spin_magnitude(self, new_spin_magnitude: float):
        """
        Update the spin magnitude S and recalculate dependent numerical matrices.

        Args:
            new_spin_magnitude (float): The new numerical value for S. Must be positive.

        Raises:
            TypeError: If new_spin_magnitude is not a number.
            ValueError: If new_spin_magnitude is not positive.
            RuntimeError: If recalculation of Ud_numeric fails.
        """
        logger.info(f"Updating spin magnitude to {new_spin_magnitude}...")
        if not isinstance(new_spin_magnitude, (int, float)):
            raise TypeError("new_spin_magnitude must be a number.")
        if new_spin_magnitude <= 0:
            raise ValueError("new_spin_magnitude must be positive.")

        self.spin_magnitude = float(new_spin_magnitude)
        # Recalculate Ud_numeric as it depends on S
        self._calculate_numerical_ud()
        logger.info("Spin magnitude updated and Ud_numeric recalculated.")

    def update_hamiltonian_params(
        self, new_hamiltonian_params: Union[List[float], npt.NDArray[np.float_]]
    ):
        """
        Update the Hamiltonian parameters and recalculate dependent numerical matrices.

        Args:
            new_hamiltonian_params (Union[List[float], npt.NDArray[np.float_]]):
                The new list or array of numerical Hamiltonian parameters. Must
                have the same length as the original parameters.

        Raises:
            TypeError: If new_hamiltonian_params is not a list/array of numbers.
            ValueError: If the number of new parameters does not match the expected number.
            RuntimeError: If recalculation of Ud_numeric fails.
        """
        logger.info("Updating Hamiltonian parameters...")
        expected_len = len(self.params_sym)

        if isinstance(new_hamiltonian_params, np.ndarray):
            new_hamiltonian_params = (
                new_hamiltonian_params.tolist()
            )  # Convert numpy array
        if not isinstance(new_hamiltonian_params, list):
            raise TypeError("new_hamiltonian_params must be a list or NumPy array.")
        if len(new_hamiltonian_params) != expected_len:
            raise ValueError(
                f"Incorrect number of parameters provided. Expected {expected_len}, got {len(new_hamiltonian_params)}."
            )
        if not all(isinstance(p, (int, float)) for p in new_hamiltonian_params):
            raise TypeError("All elements in new_hamiltonian_params must be numbers.")

        self.hamiltonian_params = [float(p) for p in new_hamiltonian_params]
        # Recalculate Ud_numeric as it depends on parameters
        self._calculate_numerical_ud()
        logger.info("Hamiltonian parameters updated and Ud_numeric recalculated.")

    # --- END NEW METHODS ---

    def calculate_dispersion(
        self,
        q_vectors: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    ) -> Optional[List[npt.NDArray[np.float_]]]:
        """
        Calculate the spin-wave dispersion relation over a list of q-points.

        Args:
            q_vectors (Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]]):
                A list or NumPy array of momentum vectors q = [qx, qy, qz].
                Each vector should be a 1D array/list of 3 numbers.

        Returns:
            Optional[List[npt.NDArray[np.float_]]]: A list containing the
            calculated magnon energies (as a NumPy array) for each corresponding
            input q-vector. Returns None if the calculation fails to start
            (e.g., due to setup errors or pool initialization failure).
            Individual q-points that fail during calculation will have NaN arrays
            in the returned list.
        """
        if self.HMat_sym is None:  # Should be caught by __init__
            logger.error("Symbolic matrix HMat_sym not available.")
            return None

        # --- Input Validation for q_vectors ---
        if isinstance(q_vectors, np.ndarray):
            if q_vectors.ndim == 1 and q_vectors.shape == (3,):  # Single vector case
                q_vectors = [q_vectors]  # Wrap in a list
            elif q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
                raise ValueError("q_vectors NumPy array must be 2D with shape (N, 3).")
            # Convert rows to separate arrays for pool.imap if needed, or keep as list of arrays
            q_vectors_list = [q_vec for q_vec in q_vectors]
        elif isinstance(q_vectors, list):
            if not q_vectors:
                raise ValueError("q_vectors list cannot be empty.")
            if not all(
                isinstance(q, (list, np.ndarray)) and len(q) == 3 for q in q_vectors
            ):
                raise ValueError(
                    "Each element in q_vectors list must be a list/array of length 3."
                )
            # Ensure elements are numpy arrays
            q_vectors_list = [np.array(q, dtype=float) for q in q_vectors]
        else:
            raise TypeError("q_vectors must be a list or NumPy array.")
        # --- End q_vector validation ---

        logger.info("Running dispersion calculation via multiprocessing...")
        start_time: float = timeit.default_timer()

        pool_args: List[Tuple] = [
            (
                self.HMat_sym,
                self.full_symbol_list,
                q_vec,
                self.nspins,
                self.spin_magnitude,
                self.hamiltonian_params,
            )
            for q_vec in q_vectors_list  # Use validated list
        ]
        energies_list: List[npt.NDArray[np.float_]] = []
        try:
            # Use context manager for Pool
            with Pool() as pool:
                energies_list = list(
                    tqdm(
                        pool.imap(process_calc_disp, pool_args),
                        total=len(q_vectors_list),
                        desc="Calculating Dispersion",
                        bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                    )
                )
        except Exception:
            logger.exception(
                "Error during multiprocessing pool execution for dispersion."
            )
            return None

        num_failures = sum(np.isnan(en).any() for en in energies_list)
        if num_failures > 0:
            logger.warning(
                f"Dispersion calculation failed for {num_failures} out of {len(q_vectors_list)} q-points. Check logs for details."
            )

        end_time: float = timeit.default_timer()
        logger.info(
            f"Run-time for dispersion calculation: {np.round((end_time - start_time) / 60, 2)} min."
        )
        return energies_list

    def calculate_sqw(
        self,
        q_vectors: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    ) -> Tuple[
        Optional[Tuple[npt.NDArray[np.float_], ...]],
        Optional[Tuple[npt.NDArray[np.float_], ...]],
        Optional[Tuple[npt.NDArray[np.float_], ...]],
    ]:
        """
        Calculate the dynamical structure factor S(q,w) over a list of q-points.

        Args:
            q_vectors (Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]]):
                A list or NumPy array of momentum vectors q = [qx, qy, qz].
                Each vector should be a 1D array/list of 3 numbers.

        Returns:
            Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple]]: A tuple
            containing three elements:
            1. Tuple of output q-vectors (as NumPy arrays).
            2. Tuple of corresponding magnon energies (as NumPy arrays).
            3. Tuple of corresponding S(q,w) intensities (as NumPy arrays).
            Returns (None, None, None) if the calculation fails to start.
            Individual q-points that fail during calculation will have NaN arrays
            for energies and intensities in the returned tuples.
        """
        if (
            self.HMat_sym is None or self.Ud_numeric is None
        ):  # Should be caught by __init__
            logger.error("Symbolic HMat_sym or numerical Ud_numeric not available.")
            return None, None, None

        # --- Input Validation for q_vectors ---
        if isinstance(q_vectors, np.ndarray):
            if q_vectors.ndim == 1 and q_vectors.shape == (3,):  # Single vector case
                q_vectors = [q_vectors]  # Wrap in a list
            elif q_vectors.ndim != 2 or q_vectors.shape[1] != 3:
                raise ValueError("q_vectors NumPy array must be 2D with shape (N, 3).")
            q_vectors_list = [q_vec for q_vec in q_vectors]
        elif isinstance(q_vectors, list):
            if not q_vectors:
                raise ValueError("q_vectors list cannot be empty.")
            if not all(
                isinstance(q, (list, np.ndarray)) and len(q) == 3 for q in q_vectors
            ):
                raise ValueError(
                    "Each element in q_vectors list must be a list/array of length 3."
                )
            q_vectors_list = [np.array(q, dtype=float) for q in q_vectors]
        else:
            raise TypeError("q_vectors must be a list or NumPy array.")
        # --- End q_vector validation ---

        logger.info("Running S(q,w) calculation via multiprocessing...")
        start_time: float = timeit.default_timer()

        pool_args: List[Tuple] = [
            (
                self.HMat_sym,
                self.Ud_numeric,
                self.full_symbol_list,
                q_vec,
                self.nspins,
                self.spin_magnitude,
                self.hamiltonian_params,
            )
            for q_vec in q_vectors_list  # Use validated list
        ]
        results: List[
            Tuple[
                npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]
            ]
        ] = []
        try:
            with Pool() as pool:
                results = list(
                    tqdm(
                        pool.imap(process_calc_Sqw, pool_args),
                        total=len(q_vectors_list),
                        desc="Calculating S(q,w)",
                        bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                    )
                )
        except Exception:
            logger.exception("Error during multiprocessing pool execution for S(q,w).")
            return None, None, None

        q_vectors_out: Tuple[npt.NDArray[np.float_], ...]
        energies_out: Tuple[npt.NDArray[np.float_], ...]
        intensities_out: Tuple[npt.NDArray[np.float_], ...]
        try:
            if not results:
                raise ValueError("Multiprocessing returned empty results list.")
            if len(results[0]) != 3:
                raise ValueError(
                    f"Multiprocessing returned malformed results: {results[0]}"
                )
            q_vectors_out, energies_out, intensities_out = zip(*results)
        except (ValueError, TypeError):
            logger.exception("Error unpacking results from parallel processing.")
            return None, None, None

        num_failures = sum(np.isnan(en).any() for en in energies_out)
        if num_failures > 0:
            logger.warning(
                f"S(q,w) calculation failed for {num_failures} out of {len(q_vectors_list)} q-points. Check logs for details."
            )

        end_time: float = timeit.default_timer()
        logger.info(
            f"Run-time for S(q,w) calculation: {np.round((end_time - start_time) / 60, 2)} min."
        )
        return q_vectors_out, energies_out, intensities_out

    def save_results(self, filename: str, results_dict: Dict[str, Any]):
        """
        Save calculation results to a compressed NumPy (.npz) file.

        Args:
            filename (str): The name of the file to save the results to.
                            '.npz' extension is recommended.
            results_dict (Dict[str, Any]): A dictionary where keys are string
                names (e.g., 'q_vectors', 'energies', 'intensities') and
                values are the corresponding data (e.g., NumPy array, list/tuple
                of NumPy arrays). Sequences of arrays will be saved directly.

        Raises:
            TypeError: If results_dict is not a dictionary.
            ValueError: If filename is empty.
            IOError: If there is an error writing the file.
            Exception: For other potential errors during saving.
        """
        if not isinstance(results_dict, dict):
            raise TypeError("results_dict must be a dictionary.")
        if not filename:
            raise ValueError("filename cannot be empty.")

        logger.info(f"Saving results to '{filename}'...")
        try:
            # Pass the dictionary directly to savez_compressed
            # It handles saving sequences of arrays appropriately.
            np.savez_compressed(filename, **results_dict)
            logger.info(f"Results successfully saved to '{filename}'.")
        except (IOError, OSError) as e:
            logger.error(f"Failed to save results to '{filename}': {e}")
            raise IOError(f"File saving failed: {e}") from e
        except Exception as e:
            logger.exception(
                f"An unexpected error occurred while saving results to '{filename}'."
            )
            raise


# --- Main execution block (demonstrating update methods) ---
if __name__ == "__main__":
    """
    Example script demonstrating the usage of the MagCalc class.

    1. Imports a spin model (`spin_model.py` by default).
    2. Defines parameters, q-points, and cache settings.
    3. Instantiates `MagCalc`.
    4. Calculates and saves initial dispersion.
    5. Updates parameters and recalculates/saves dispersion and S(q,w).
    """
    # --- Import the specific spin model ---
    # NOTE: Replace 'spin_model_fm' if you intend to use a different model file
    try:
        import spin_model
    except ImportError:
        logger.error(
            "Failed to import 'spin_model'. Please ensure it's in the Python path."
        )
        sys.exit(1)  # Exit if the model cannot be imported
    # --- User Configuration ---
    spin_S_val: float = 1.0
    hamiltonian_params_val: List[float] = [1.0, 0.5, 0.1, 0.1, 0.0]
    cache_file_base_name: str = "my_model_cache"
    cache_operation_mode: str = "r"  # Use 'r' after cache is generated
    output_filename_base: str = "calc_results"

    # Define q points
    q_points_list: List[List[float]] = []
    N_points_per_segment = 20
    logger.info(f"Using {N_points_per_segment} points per segment.")
    q_points_list.extend(
        np.linspace([0, 0, 0], [np.pi, 0, 0], N_points_per_segment, endpoint=False)
    )
    q_points_list.extend(
        np.linspace(
            [np.pi, 0, 0], [np.pi, np.pi, 0], N_points_per_segment, endpoint=False
        )
    )
    q_points_list.extend(
        np.linspace([np.pi, np.pi, 0], [0, 0, 0], N_points_per_segment, endpoint=True)
    )
    q_points_array: npt.NDArray[np.float_] = np.array(q_points_list)
    # --- End User Configuration ---

    logger.info("Starting example calculation using MagCalc class...")

    try:
        # --- Instantiate the Calculator ---
        calculator = MagCalc(
            spin_magnitude=spin_S_val,
            hamiltonian_params=hamiltonian_params_val,
            cache_file_base=cache_file_base_name,
            cache_mode=cache_operation_mode,
            spin_model_module=spin_model,  # <<< Added the missing argument
        )

        # --- Calculate and Save Dispersion (Initial Params) ---
        logger.info("Calculating dispersion (Initial Params)...")
        dispersion_energies = calculator.calculate_dispersion(q_points_array)
        if dispersion_energies is not None:
            disp_filename = f"{output_filename_base}_disp_initial.npz"
            calculator.save_results(
                disp_filename,
                {"q_vectors": q_points_array, "energies": dispersion_energies},
            )
        else:
            logger.error("Initial dispersion calculation failed.")

        # --- Update Parameters ---
        logger.info("Updating parameters for recalculation...")
        new_params = [
            p * 1.1 for p in hamiltonian_params_val
        ]  # Example: Increase params by 10%
        calculator.update_hamiltonian_params(new_params)
        # calculator.update_spin_magnitude(spin_S_val * 1.2) # Example if needed

        # --- Recalculate and Save Dispersion (Updated Params) ---
        logger.info("Calculating dispersion (Updated Params)...")
        dispersion_energies_updated = calculator.calculate_dispersion(q_points_array)
        if dispersion_energies_updated is not None:
            disp_filename_updated = f"{output_filename_base}_disp_updated.npz"
            calculator.save_results(
                disp_filename_updated,
                {"q_vectors": q_points_array, "energies": dispersion_energies_updated},
            )
        else:
            logger.error("Updated dispersion calculation failed.")

        # --- Calculate and Save S(q,w) (Using Updated Params) ---
        logger.info("Calculating S(q,w) (Updated Params)...")
        sqw_results = calculator.calculate_sqw(q_points_array)
        q_vectors_out, energies_sqw, intensities_sqw = sqw_results
        if (
            q_vectors_out is not None
            and energies_sqw is not None
            and intensities_sqw is not None
        ):
            sqw_filename = f"{output_filename_base}_sqw_updated.npz"
            calculator.save_results(
                sqw_filename,
                {
                    "q_vectors": q_vectors_out,
                    "energies": energies_sqw,
                    "intensities": intensities_sqw,
                },
            )
        else:
            logger.error("S(q,w) calculation failed.")

    except (
        FileNotFoundError,
        AttributeError,
        RuntimeError,
        ValueError,
        TypeError,
        pickle.PickleError,
    ) as e:  # Catch specific init/load errors
        logger.error(f"Calculation failed during setup or execution: {e}")
    except Exception as e:
        logger.exception("An unexpected error occurred.")

    logger.info("Example calculation finished.")
