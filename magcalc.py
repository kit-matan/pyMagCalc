#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSWT Calculator Module

@author: Kit Matan and Pharit Piyawongwatthana
Refactored by AI Assistant
"""
import spin_model as sm  # User-defined spin model
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
    """Helper function for multiprocessing substitution."""
    expr, subs_dict = args
    result: sp.Expr = expr.subs(subs_dict, simultaneous=True)
    return result


def gram_schmidt(x: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
    """Performs Gram-Schmidt orthogonalization using QR decomposition."""
    q, r = np.linalg.qr(x, mode="reduced")
    return q


# --- KKdMatrix Helper Functions (Keep outside class) ---
# _diagonalize_and_sort, _apply_gram_schmidt, _calculate_alpha_matrix,
# _match_and_reorder_minus_q, _calculate_K_Kd
def _diagonalize_and_sort(
    Hmat: npt.NDArray[np.complex_], nspins: int, q_vector_label: str
) -> Tuple[Optional[npt.NDArray[np.complex_]], Optional[npt.NDArray[np.complex_]]]:
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
    """Calculates K, Kd, and eigenvalues by orchestrating helper functions."""
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
    """Worker function for dispersion calculation."""
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
    """Worker function for S(q,w) calculation."""
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


# --- gen_HM Function (Keep outside class) ---
def gen_HM(
    k_sym: List[sp.Symbol], S_sym: sp.Symbol, params_sym: List[sp.Symbol]
) -> Tuple[sp.Matrix, sp.Matrix]:
    """Generates symbolic TwogH2 and Ud matrices."""
    # Assumes 'sm' is globally available or correctly imported where needed
    atom_positions_uc: npt.NDArray[np.float_] = sm.atom_pos()
    nspins: int = len(atom_positions_uc)
    atom_positions_ouc: npt.NDArray[np.float_] = sm.atom_pos_ouc()
    nspins_ouc: int = len(atom_positions_ouc)
    logger.info(f"Number of spins in the unit cell: {nspins}")
    c_ops: List[sp.Symbol] = sp.symbols("c0:%d" % nspins_ouc, commutative=False)
    cd_ops: List[sp.Symbol] = sp.symbols("cd0:%d" % nspins_ouc, commutative=False)
    spin_ops_local: List[sp.Matrix] = [
        sp.Matrix(
            (
                sp.sqrt(S_sym / 2) * (c_ops[i] + cd_ops[i]),
                sp.sqrt(S_sym / 2) * (c_ops[i] - cd_ops[i]) / I,
                S_sym - cd_ops[i] * c_ops[i],
            )
        )
        for i in range(nspins_ouc)
    ]
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]] = sm.mpr(params_sym)
    spin_ops_global_ouc: List[sp.Matrix] = [
        rotation_matrices[j] * spin_ops_local[nspins * i + j]
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]
    hamiltonian_sym: sp.Expr = sm.Hamiltonian(spin_ops_global_ouc, params_sym)
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    hamiltonian_S0: sp.Expr = hamiltonian_sym.coeff(S_sym, 0)
    if params_sym:
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
    ck_ops: List[sp.Symbol] = [
        sp.Symbol("ck%d" % j, commutative=False) for j in range(nspins)
    ]
    ckd_ops: List[sp.Symbol] = [
        sp.Symbol("ckd%d" % j, commutative=False) for j in range(nspins)
    ]
    cmk_ops: List[sp.Symbol] = [
        sp.Symbol("cmk%d" % j, commutative=False) for j in range(nspins)
    ]
    cmkd_ops: List[sp.Symbol] = [
        sp.Symbol("cmkd%d" % j, commutative=False) for j in range(nspins)
    ]
    interaction_matrix: npt.NDArray = sm.spin_interactions(params_sym)[0]
    fourier_substitutions: List[List[sp.Expr]] = [
        ent
        for i in range(nspins)
        for j in range(nspins_ouc)
        if interaction_matrix[i, j] != 0
        for disp_vec in [atom_positions_uc[i, :] - atom_positions_ouc[j, :]]
        for k_dot_dr in [
            k_sym[0] * disp_vec[0] + k_sym[1] * disp_vec[1] + k_sym[2] * disp_vec[2]
        ]
        for ent in [
            [
                cd_ops[i] * cd_ops[j],
                1
                / 2
                * (
                    ckd_ops[i % nspins]
                    * cmkd_ops[j % nspins]
                    * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                    + cmkd_ops[i % nspins]
                    * ckd_ops[j % nspins]
                    * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            [
                c_ops[i] * c_ops[j],
                1
                / 2
                * (
                    ck_ops[i % nspins]
                    * cmk_ops[j % nspins]
                    * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                    + cmk_ops[i % nspins]
                    * ck_ops[j % nspins]
                    * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            [
                cd_ops[i] * c_ops[j],
                1
                / 2
                * (
                    ckd_ops[i % nspins]
                    * ck_ops[j % nspins]
                    * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                    + cmkd_ops[i % nspins]
                    * cmk_ops[j % nspins]
                    * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            [
                c_ops[i] * cd_ops[j],
                1
                / 2
                * (
                    ck_ops[i % nspins]
                    * ckd_ops[j % nspins]
                    * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                    + cmk_ops[i % nspins]
                    * cmkd_ops[j % nspins]
                    * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            [
                cd_ops[j] * c_ops[j],
                1
                / 2
                * (
                    ckd_ops[j % nspins] * ck_ops[j % nspins]
                    + cmkd_ops[j % nspins] * cmk_ops[j % nspins]
                ),
            ],
        ]
    ]
    commutation_substitutions: List[List[sp.Expr]] = (
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
    basis_vector_dagger: List[sp.Symbol] = ckd_ops[:nspins] + cmk_ops[:nspins]
    basis_vector: List[sp.Symbol] = ck_ops[:nspins] + cmkd_ops[:nspins]
    placeholder_symbols: List[sp.Symbol] = [
        sp.Symbol("XdX%d" % (i * 2 * nspins + j), commutative=True)
        for i in range(2 * nspins)
        for j in range(2 * nspins)
    ]
    placeholder_substitutions: List[List[sp.Expr]] = [
        [
            basis_vector_dagger[i] * basis_vector[j],
            placeholder_symbols[i * 2 * nspins + j],
        ]
        for i in range(2 * nspins)
        for j in range(2 * nspins)
    ]
    logger.info("Running symbolic substitutions...")
    start_time: float = timeit.default_timer()
    try:
        hamiltonian_terms: List[sp.Expr] = hamiltonian_sym.as_ordered_terms()
        pool_args_ft = [(expr, fourier_substitutions) for expr in hamiltonian_terms]
        with Pool() as pool:
            results_ft: List[sp.Expr] = list(
                tqdm(
                    pool.imap(substitute_expr, pool_args_ft),
                    total=len(hamiltonian_terms),
                    desc="Substituting FT ",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
        hamiltonian_k_space: sp.Expr = Add(*results_ft)
        hamiltonian_k_space = hamiltonian_k_space.expand()
        hamiltonian_k_terms: List[sp.Expr] = hamiltonian_k_space.as_ordered_terms()
        pool_args_comm = [
            (expr, commutation_substitutions) for expr in hamiltonian_k_terms
        ]
        with Pool() as pool:
            results_comm: List[sp.Expr] = list(
                tqdm(
                    pool.imap(substitute_expr, pool_args_comm),
                    total=len(hamiltonian_k_terms),
                    desc="Applying Commutation",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
        hamiltonian_k_commuted: sp.Expr = Add(*results_comm)
        hamiltonian_k_commuted = hamiltonian_k_commuted.expand()
        hamiltonian_k_comm_terms: List[sp.Expr] = (
            hamiltonian_k_commuted.as_ordered_terms()
        )
        pool_args_placeholder = [
            (expr, placeholder_substitutions) for expr in hamiltonian_k_comm_terms
        ]
        with Pool() as pool:
            results_placeholder: List[sp.Expr] = list(
                tqdm(
                    pool.imap(substitute_expr, pool_args_placeholder),
                    total=len(hamiltonian_k_comm_terms),
                    desc="Substituting Placeholders",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
        hamiltonian_with_placeholders: sp.Expr = Add(*results_placeholder)
    except Exception as e:
        logger.exception("Error during symbolic substitution in gen_HM.")
        raise RuntimeError("Symbolic substitution failed.") from e
    end_time: float = timeit.default_timer()
    logger.info(
        f"Run-time for substitution: {np.round((end_time - start_time) / 60, 2)} min."
    )
    H2_elements: sp.Matrix = sp.Matrix(
        [hamiltonian_with_placeholders.coeff(p) for p in placeholder_symbols]
    )
    H2_matrix: sp.Matrix = sp.Matrix(2 * nspins, 2 * nspins, H2_elements)
    g_metric_tensor_sym: sp.Matrix = sp.diag(*([1] * nspins + [-1] * nspins))
    dynamical_matrix_TwogH2: sp.Matrix = 2 * g_metric_tensor_sym * H2_matrix
    Ud_rotation_matrix_blocks: List[sp.Matrix] = []
    for i in range(nspins):
        rot_mat = rotation_matrices[i]
        if isinstance(rot_mat, np.ndarray):
            rot_mat_sym: sp.Matrix = sp.Matrix(rot_mat)
        else:
            rot_mat_sym = rot_mat
        Ud_rotation_matrix_blocks.append(rot_mat_sym)
    Ud_rotation_matrix: sp.Matrix = sp.diag(*Ud_rotation_matrix_blocks)
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
        cache_mode: str = "r",
        spin_model_module=sm,
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
            cache_mode (str, optional): Specifies the cache behavior.
                'r': Read symbolic matrices from cache files. Fails if files
                     don't exist or are invalid. (Default)
                'w': Generate symbolic matrices (potentially slow) and write
                     them to cache files, overwriting existing ones.
            spin_model_module (module, optional): The imported Python module
                containing the spin model definitions (e.g., Hamiltonian, mpr,
                atom_pos). Defaults to the globally imported `sm`.

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
        """Loads symbolic matrices from cache or generates them."""
        # Ensure cache directory exists
        cache_dir = "pckFiles"
        if not os.path.exists(cache_dir):
            logger.info(f"Creating directory '{cache_dir}' for caching.")
            try:
                os.makedirs(cache_dir)
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
                    self.k_sym, self.S_sym, list(self.params_sym)
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
        """Substitutes numerical parameters into Ud_sym."""
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
        Updates the spin magnitude S and recalculates dependent numerical matrices.

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
        Updates the Hamiltonian parameters and recalculates dependent numerical matrices.

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
        Calculates the spin-wave dispersion relation over a list of q-points.

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
        Calculates the dynamical structure factor S(q,w) over a list of q-points.

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
        Saves calculation results to a compressed NumPy (.npz) file.

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
