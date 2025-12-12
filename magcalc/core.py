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

import hashlib  # For numerical cache key generation
import logging
import os  # Added for cpu_count
import json  # For metadata

# Type Hinting Imports
import types  # Added for ModuleType hint
from typing import List, Tuple, Dict, Any, Optional, Union, NoReturn

# --- Imports for Configuration-Driven Model (with fallback for direct script execution) ---
try:
    # This works when magcalc.py is imported as part of the pyMagCalc package
    from .config_loader import load_spin_model_config
    from . import generic_spin_model
except ImportError:
    # This block executes if the relative import fails.
    # This can happen when run as a script or by multiprocessing spawns.
    # Add the script's directory to sys.path to allow direct imports.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    try:
        from config_loader import load_spin_model_config
        import generic_spin_model
    except ImportError as e:
        # If direct import also fails, then there's a more fundamental issue.
        raise ImportError(
            f"Failed to import local modules config_loader or generic_spin_model. Original error: {e}"
        ) from e
import numpy.typing as npt

# --- Modularized Linalg Imports ---
try:
    from .linalg import (
        gram_schmidt,
        _diagonalize_and_sort,
        _apply_gram_schmidt,
        _calculate_alpha_matrix,
        _match_and_reorder_minus_q,
        _calculate_K_Kd,
        DEGENERACY_THRESHOLD,
        ZERO_MATRIX_ELEMENT_THRESHOLD,
        ALPHA_MATRIX_ZERO_NORM_WARNING_THRESHOLD,
        EIGENVECTOR_MATCHING_THRESHOLD
    )
except ImportError:
     from linalg import (
        gram_schmidt,
        _diagonalize_and_sort,
        _apply_gram_schmidt,
        _calculate_alpha_matrix,
        _match_and_reorder_minus_q,
        _calculate_K_Kd,
        DEGENERACY_THRESHOLD,
        ZERO_MATRIX_ELEMENT_THRESHOLD,
        ALPHA_MATRIX_ZERO_NORM_WARNING_THRESHOLD,
        EIGENVECTOR_MATCHING_THRESHOLD
    )

import matplotlib.pyplot as plt  # Added for plotting

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

# --- Numerical Constants ---
# Constants moved to magcalc_linalg.py

ENERGY_IMAG_PART_THRESHOLD: float = 1e-5
SQW_IMAG_PART_THRESHOLD: float = 1e-4
Q_ZERO_THRESHOLD: float = 1e-10
PROJECTION_CHECK_TOLERANCE: float = 1e-5
# --- End Numerical Constants ---

# --- Global variable for worker processes ---
_worker_HMat_func = None


def _init_worker(HMat_sym, full_symbol_list):
    """
    Initializer function for multiprocessing worker.
    Lambdifies the symbolic Hamiltonian once per process.
    """
    global _worker_HMat_func
    # Configure logging for worker (optional, to avoid duplicate logs or silence)
    # logging.basicConfig(level=logging.WARN) 
    
    try:
        # We need to import numpy inside the worker if used in lambdify modules
        import numpy as np 
        _worker_HMat_func = lambdify(full_symbol_list, HMat_sym, modules=["numpy"])
    except Exception as e:
        # We can't log easily here to the main process, but we can print to stderr
        sys.stderr.write(f"Error in worker initialization: {e}\n")
        raise e

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




# --- Main KKdMatrix Function (Keep outside class) ---
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
        Hmat_plus_q (npt.NDArray[np.complex128]): Numerical Hamiltonian matrix for +q.
        Hmat_minus_q (npt.NDArray[np.complex128]): Numerical Hamiltonian matrix for -q.
        Ud_numeric (npt.NDArray[np.complex128]): Numerical rotation matrix Ud.
        q_vector (npt.NDArray[np.float64]): The momentum vector q.
        nspins (int): Number of spins in the unit cell.

    Returns:
        Tuple[npt.NDArray[np.complex128], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
            K matrix, Kd matrix, and sorted eigenvalues from the +q calculation. Returns NaN arrays on failure.
    """
    q_label = f"q={q_vector}"
    nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex128)
    nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex128)
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
    # --- Custom processing for -q eigenvectors (ported from magcalc_origin.py) ---
    # This replaces the simple call to _apply_gram_schmidt for the -q case.
    logger.debug(f"Applying custom Gram-Schmidt / basis selection for -q at {q_label}")
    eigvecs_m_processed = eigvecs_m_sorted.copy()  # Work on a copy
    nspins2 = eigvecs_m_processed.shape[0]

    # evecswap_p is equivalent to evecswap in magcalc_origin.py, derived from +q eigenvectors
    evecswap_p = np.conj(
        np.vstack((eigvecs_p_ortho[nspins:nspins2, :], eigvecs_p_ortho[0:nspins, :]))
    )

    current_block_start_idx = 0
    for i in range(1, nspins2 + 1):  # Iterate up to nspins2 to process the last block
        # Condition to end a block:
        # 1. Reached the end of the array (i == nspins2)
        # 2. Next eigenvalue is different from the start of the current block's eigenvalue
        if (
            i == nspins2
            or abs(eigvals_m_sorted[i] - eigvals_m_sorted[current_block_start_idx])
            >= DEGENERACY_THRESHOLD
        ):
            # Process block from current_block_start_idx to i-1
            if (
                i - 1 > current_block_start_idx
            ):  # Degeneracy of at least 2 (block size > 1)
                block_size = i - current_block_start_idx
                degenerate_block_m = eigvecs_m_sorted[:, current_block_start_idx:i]

                # Determine which part of evecswap_p to compare against
                if current_block_start_idx < nspins:
                    # Positive energy block of -q, compare with conj(positive energy part of +q from evecswap_p)
                    candidate_basis_p_block = evecswap_p[:, nspins:]
                else:
                    # Negative energy block of -q, compare with conj(negative energy part of +q from evecswap_p)
                    candidate_basis_p_block = evecswap_p[:, :nspins]

                # Custom basis selection logic from magcalc_origin.py
                # Calculate sum of projections of each vector in degenerate_block_m onto each vector in candidate_basis_p_block
                # projection_matrix[k, l] = vdot(degenerate_block_m[:,k], candidate_basis_p_block[:,l])
                # sum_of_projections[l] = sum_k vdot(degenerate_block_m[:,k], candidate_basis_p_block[:,l])
                # This is equivalent to sum(conj(degenerate_block_m.T) @ candidate_basis_p_block, axis=0)

                # Original logic: tmpsum = tmpsum + (np.conj(vtmpm1[:, j_deg]).T @ tmpevecswap)
                # This means for each column in tmpevecswap, sum its dot product with all columns in vtmpm1.
                sum_of_projections = np.zeros(
                    candidate_basis_p_block.shape[1], dtype=complex
                )
                for k_deg_block in range(block_size):
                    sum_of_projections += (
                        np.conj(degenerate_block_m[:, k_deg_block])
                        @ candidate_basis_p_block
                    ).flatten()

                # Robust matching: Pick top 'block_size' matches if available
                projection_magnitudes = np.abs(sum_of_projections)
                sorted_indices_desc = np.argsort(projection_magnitudes)[::-1]
                
                # Check if the weakest of the top 'block_size' is strong enough
                match_quality_ok = False
                if len(sorted_indices_desc) >= block_size:
                    if projection_magnitudes[sorted_indices_desc[block_size-1]] > EIGENVECTOR_MATCHING_THRESHOLD:
                         match_quality_ok = True
                
                if match_quality_ok:
                    selected_indices = sorted_indices_desc[:block_size]
                    # Log if we are filtering out extra matches (e.g. got 4, took 2)
                    if np.sum(projection_magnitudes > EIGENVECTOR_MATCHING_THRESHOLD) > block_size:
                         logger.debug(
                             f"Custom GS: Found more matches than block size at {q_label}. "
                             f"Selecting top {block_size} strongest projections."
                         )

                    new_basis_for_m_block = candidate_basis_p_block[:, selected_indices]
                    eigvecs_m_processed[:, current_block_start_idx:i] = gram_schmidt(
                        new_basis_for_m_block
                    )
                else:
                    logger.warning(
                        f"Custom GS for -q: Insufficient matching basis size for block {current_block_start_idx}-{i-1} at {q_label}. "
                        f"Expected {block_size}, but only found {np.sum(projection_magnitudes > EIGENVECTOR_MATCHING_THRESHOLD)} distinct matches > threshold. "
                        f"Applying standard GS to original block."
                    )
                    eigvecs_m_processed[:, current_block_start_idx:i] = gram_schmidt(
                        degenerate_block_m
                    )
            current_block_start_idx = i  # Move to the next block
    # --- End custom processing for -q eigenvectors ---
    eigvecs_m_ortho = eigvecs_m_processed  # Use the result of custom processing

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
        EIGENVECTOR_MATCHING_THRESHOLD,  # zero_tol_comp_phase (use 1e-5 like in origin for selecting components for phase)
        ZERO_MATRIX_ELEMENT_THRESHOLD,  # zero_tol_alpha_final (use 1e-6 like in origin for truncating final alpha)
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
        npt.NDArray[np.float64],
        int,
        float,
        Union[List[float], npt.NDArray[np.float64]],
    ],
) -> Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.complex128]]]:
    """
    Worker function for parallel dispersion calculation at a single q-point.
    Uses pre-initialized _worker_HMat_func.
    """
    (
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
    ) = args
    
    global _worker_HMat_func
    if _worker_HMat_func is None:
        # Fallback or error if not initialized? Should raise.
        raise RuntimeError("Worker not initialized with HMat_func")
        
    # logger.debug(f"Processing dispersion for q-vector: {q_vector}") 
    q_label = f"q={q_vector}"
    nan_energies: npt.NDArray[np.float64] = np.full((nspins,), np.nan)
    HMat_numeric: Optional[npt.NDArray[np.complex128]] = None
    
    try:
        numerical_args = (
            list(q_vector) + [spin_magnitude_num] + list(hamiltonian_params_num)
        )
        HMat_numeric: npt.NDArray[np.complex128] = np.array(
            _worker_HMat_func(*numerical_args), dtype=np.complex128
        )
    except Exception:
        logger.exception(f"Error evaluating HMat function at {q_label}.")
        return nan_energies, None
    eigenvalues: npt.NDArray[np.complex128]
    try:
        eigenvalues = la.eigvals(HMat_numeric)
    except np.linalg.LinAlgError:
        logger.error(f"Eigenvalue calculation failed for {q_label}.")
        return (
            nan_energies,
            HMat_numeric,
        )  # Return HMat even if eigvals failed for inspection
    except Exception:
        logger.exception(
            f"Unexpected error during eigenvalue calculation for {q_label}."
        )
        return (
            nan_energies,
            HMat_numeric,
        )  # Return HMat even if eigvals failed for inspection
    try:
        imag_part_mags: npt.NDArray[np.float64] = np.abs(np.imag(eigenvalues))
        if np.any(imag_part_mags > ENERGY_IMAG_PART_THRESHOLD):
            logger.warning(
                f"Significant imaginary part in eigenvalues for {q_label}. Max imag: {np.max(imag_part_mags)}"
            )
        eigenvalues_sorted_real: npt.NDArray[np.float64] = np.real(np.sort(eigenvalues))
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
        return energies, HMat_numeric
    except Exception:
        logger.exception(f"Error during eigenvalue sorting/selection for {q_label}.")
        return nan_energies, HMat_numeric  # Return HMat even if sorting failed


def process_calc_Sqw(
    args: Tuple[
        npt.NDArray[np.complex128],
        npt.NDArray[np.float64],
        int,
        float,
        Union[List[float], npt.NDArray[np.float64]],
    ],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Worker function for parallel S(q,w) calculation at a single q-point.
    Uses pre-initialized _worker_HMat_func.
    """
    (
        Ud_numeric,
        q_vector,
        nspins,
        spin_magnitude_num,
        hamiltonian_params_num,
    ) = args
    
    global _worker_HMat_func
    if _worker_HMat_func is None:
        raise RuntimeError("Worker not initialized with HMat_func")
        
    # logger.debug(f"Processing S(q,w) for q-vector: {q_vector}")
    q_label = f"q={q_vector}"
    nan_energies: npt.NDArray[np.float64] = np.full((nspins,), np.nan)
    nan_intensities: npt.NDArray[np.float64] = np.full((nspins,), np.nan)
    nan_result = (q_vector, nan_energies, nan_intensities)
    
    try:
        numerical_args_base = [spin_magnitude_num] + list(hamiltonian_params_num)
        numerical_args_plus_q = list(q_vector) + numerical_args_base
        numerical_args_minus_q = list(-q_vector) + numerical_args_base
        
        Hmat_plus_q: npt.NDArray[np.complex128] = np.array(
            _worker_HMat_func(*numerical_args_plus_q), dtype=np.complex128
        )
        Hmat_minus_q: npt.NDArray[np.complex128] = np.array(
            _worker_HMat_func(*numerical_args_minus_q), dtype=np.complex128
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
        imag_energy_mag: npt.NDArray[np.float64] = np.abs(np.imag(eigenvalues[0:nspins]))
        if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
            logger.warning(
                f"Significant imaginary part in energy eigenvalues for {q_label}. Max imag: {np.max(imag_energy_mag)}"
            )
        energies: npt.NDArray[np.float64] = np.real(eigenvalues[0:nspins])
        sqw_complex_accumulator: npt.NDArray[np.complex128] = np.zeros(
            nspins, dtype=complex
        )
        for mode_index in range(nspins):
            spin_correlation_matrix: npt.NDArray[np.complex128] = np.zeros(
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
                q_normalized: npt.NDArray[np.float64] = q_vector / np.sqrt(q_norm_sq)
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
        intensities: npt.NDArray[np.float64] = np.real(sqw_complex_accumulator)
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
    # Check if S_sym is present. If not (substituted), we assume filtering was done by module.
    if not hamiltonian_sym.has(S_sym):
        logger.info("S_sym not found in Hamiltonian (likely substituted). Skipping S-power filtering.")
    else:
        # Legacy filtering logic
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


def _process_hamiltonian_terms(
    hamiltonian_sym: sp.Expr,
    fourier_lookup: Dict[Tuple[str, str], sp.Expr],
    nspins: int,
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
) -> sp.Expr:
    """
    Apply Fourier substitutions and then Normal Order the terms.
    
    Optimized to avoid generic symbolic substitution for commutation rules and F.T.
    """
    logger.info("Applying Fourier transform substitutions...")
    start_time_ft = timeit.default_timer()
    hamiltonian_terms = hamiltonian_sym.as_ordered_terms()

    # DEBUG: Check fourier_lookup
    # logger.info(f"DEBUG: fourier_lookup size: {len(fourier_lookup)}")
    
    # if fourier_lookup:
    #     sample_key = next(iter(fourier_lookup))
    #     sample_val = fourier_lookup[sample_key]
    #     logger.info(f"DEBUG: fourier_lookup sample: {sample_key} -> {sample_val}")
    
    # Use parallel substitution for F.T. with dictionary lookup
    
    # Chunking terms for parallel processing
    chunk_size = max(1, len(hamiltonian_terms) // (os.cpu_count() * 4))
    chunks = [hamiltonian_terms[i:i + chunk_size] for i in range(0, len(hamiltonian_terms), chunk_size)]
    
    pool_args_ft = [
        (sp.Add(*chunk), fourier_lookup) for chunk in chunks
    ]
    
    with Pool() as pool:
        results_ft = list(pool.imap(_fourier_transform_terms, pool_args_ft))
        
    hamiltonian_k_space = Add(*results_ft).expand()
    end_time_ft = timeit.default_timer()
    logger.info(
        f"Fourier transform substitution took: {np.round(end_time_ft - start_time_ft, 2)} s"
    )
    # DEBUG: Check k-space hamiltonian
    ft_terms = hamiltonian_k_space.as_ordered_terms()
    # logger.info(f"DEBUG: hamiltonian_k_space terms count: {len(ft_terms)}")
    if not ft_terms:
        # logger.error("DEBUG: hamiltonian_k_space is ZERO/EMPTY after FT!")
         pass
    else:
        # logger.debug(f"DEBUG: First FT term: {ft_terms[0]}")
         pass

    logger.info("Applying Normal Ordering (Commutation Rules)...")
    start_time_comm = timeit.default_timer()
    
    # The hamiltonian_k_space is now a sum of quadratic terms in k-space ops.
    # We apply normal ordering directly.
    # We can parallelize this too.
    
    k_terms = hamiltonian_k_space.as_ordered_terms()
    
    # Chunking terms for parallel processing
    chunk_size = max(1, len(k_terms) // (os.cpu_count() * 4))
    chunks = [k_terms[i:i + chunk_size] for i in range(0, len(k_terms), chunk_size)]
    
    pool_args_no = [
        (Add(*chunk), ck_ops, ckd_ops, nspins) for chunk in chunks
    ]
    
    with Pool() as pool:
        results_no = list(pool.imap(_normal_order_terms, pool_args_no))
        
    hamiltonian_normal_ordered = Add(*results_no)
    
    # Expand one last time to ensure coeff * Op1 * Op2 structure
    hamiltonian_normal_ordered = hamiltonian_normal_ordered.expand()
    
    logger.info(
        f"Normal ordering took: {timeit.default_timer() - start_time_comm:.2f} s"
    )

    # Placeholder step is REMOVED.
    return hamiltonian_normal_ordered





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


# --- NEW HELPER for config-driven Fourier substitutions ---
def _define_fourier_substitutions_generic(
    k_sym: List[sp.Symbol],
    nspins_uc: int,
    c_ops_ouc: List[sp.Symbol],  # (N_ouc)
    cd_ops_ouc: List[sp.Symbol],  # (N_ouc)
    ck_ops_uc: List[sp.Symbol],  # (N_uc)
    ckd_ops_uc: List[sp.Symbol],  # (N_uc)
    cmk_ops_uc: List[sp.Symbol],  # (N_uc)
    cmkd_ops_uc: List[sp.Symbol],  # (N_uc)
    atom_pos_uc_cart: np.ndarray,  # (N_uc, 3)
    atom_pos_ouc_cart: np.ndarray,  # (N_ouc, 3)
    Jex_sym_matrix: sp.Matrix,  # (N_uc, N_ouc)
) -> List[List[sp.Expr]]:
    """
    Define Fourier substitutions for the configuration-driven model.
    """
    nspins_ouc = len(atom_pos_ouc_cart)
    if len(c_ops_ouc) != nspins_ouc or len(cd_ops_ouc) != nspins_ouc:
        raise ValueError(
            f"Length of OUC boson operators ({len(c_ops_ouc)}) does not match nspins_ouc ({nspins_ouc})."
        )
    if not (
        len(ck_ops_uc) == nspins_uc
        and len(ckd_ops_uc) == nspins_uc
        and len(cmk_ops_uc) == nspins_uc
        and len(cmkd_ops_uc) == nspins_uc
    ):
        raise ValueError(
            f"Length of UC k-space boson operators does not match nspins_uc ({nspins_uc})."
        )
    if Jex_sym_matrix.shape != (nspins_uc, nspins_ouc):
        raise ValueError(
            f"Jex_sym_matrix shape {Jex_sym_matrix.shape} mismatch. Expected ({nspins_uc}, {nspins_ouc})"
        )

    fourier_substitutions = []

    for i_uc in range(nspins_uc):
        for j_ouc in range(nspins_ouc):
            if Jex_sym_matrix[i_uc, j_ouc] == 0:
                continue

            disp_vec = atom_pos_uc_cart[i_uc, :] - atom_pos_ouc_cart[j_ouc, :]
            k_dot_dr = sum(k_comp * dr_comp for k_comp, dr_comp in zip(k_sym, disp_vec))

            exp_plus_ikdr = sp.exp(I * k_dot_dr).rewrite(sp.sin)
            exp_minus_ikdr = sp.exp(-I * k_dot_dr).rewrite(sp.sin)

            idx_op1_real_space = i_uc
            idx_op2_real_space = j_ouc

            idx_op1_k_space = i_uc
            idx_op2_k_space = j_ouc % nspins_uc

            sub_list_for_pair = [
                [
                    cd_ops_ouc[idx_op1_real_space] * cd_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ckd_ops_uc[idx_op1_k_space]
                        * cmkd_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                        + cmkd_ops_uc[idx_op1_k_space]
                        * ckd_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                    ),
                ],
                [
                    c_ops_ouc[idx_op1_real_space] * c_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ck_ops_uc[idx_op1_k_space]
                        * cmk_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                        + cmk_ops_uc[idx_op1_k_space]
                        * ck_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                    ),
                ],
                [
                    cd_ops_ouc[idx_op1_real_space] * c_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ckd_ops_uc[idx_op1_k_space]
                        * ck_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                        + cmkd_ops_uc[idx_op1_k_space]
                        * cmk_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                    ),
                ],
                [
                    c_ops_ouc[idx_op1_real_space] * cd_ops_ouc[idx_op2_real_space],
                    sp.S(1)
                    / 2
                    * (
                        ck_ops_uc[idx_op1_k_space]
                        * ckd_ops_uc[idx_op2_k_space]
                        * exp_plus_ikdr
                        + cmk_ops_uc[idx_op1_k_space]
                        * cmkd_ops_uc[idx_op2_k_space]
                        * exp_minus_ikdr
                    ),
                ],
            ]
            fourier_substitutions.extend(sub_list_for_pair)

    for j_ouc_diag in range(nspins_ouc):
        j_uc_diag = j_ouc_diag % nspins_uc
        fourier_substitutions.append(
            [
                cd_ops_ouc[j_ouc_diag] * c_ops_ouc[j_ouc_diag],
                sp.S(1)
                / 2
                * (
                    ckd_ops_uc[j_uc_diag] * ck_ops_uc[j_uc_diag]
                    + cmkd_ops_uc[j_uc_diag] * cmk_ops_uc[j_uc_diag]
                ),
            ]
        )

    unique_substitutions = []
    seen_keys = set()
    for sub_pair in fourier_substitutions:
        key = sub_pair[0]
        if key not in seen_keys:
            unique_substitutions.append(sub_pair)
            seen_keys.add(key)

    return unique_substitutions


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
    logger.debug(
        f"Hardcoded-path: Initial symbolic Hamiltonian has {len(hamiltonian_sym.as_ordered_terms())} terms."
    )
    # logger.debug(f"Hardcoded-path: Hamiltonian_sym (first 500 chars): {str(hamiltonian_sym)[:500]}")

    # --- Define k-space operators ---
    ck_ops = [sp.Symbol("ck%d" % j, commutative=False) for j in range(nspins)]
    ckd_ops = [sp.Symbol("ckd%d" % j, commutative=False) for j in range(nspins)]
    cmk_ops = [sp.Symbol("cmk%d" % j, commutative=False) for j in range(nspins)]
    cmkd_ops = [sp.Symbol("cmkd%d" % j, commutative=False) for j in range(nspins)]

    # --- Define Substitution Rules ---
    fourier_lookup = _generate_fourier_lookup(
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
    logger.debug(
        f"Hardcoded-path: Number of Fourier substitution rules: {len(fourier_lookup)}"
    )
    
    # Commutation and Placeholder definitions are removed.

    # --- Apply Substitutions & Normal Ordering ---
    try:
        hamiltonian_normal_ordered = _process_hamiltonian_terms(
            hamiltonian_sym,
            fourier_lookup,
            nspins,
            ck_ops,
            ckd_ops,
            cmk_ops,
            cmkd_ops,
        )
    except Exception as e:
        logger.exception("Error during symbolic substitution in gen_HM.")
        raise RuntimeError("Symbolic substitution failed.") from e

    # --- Build TwogH2 Matrix directly ---
    # This replaces _extract_h2_matrix and subsequent multiplication
    dynamical_matrix_TwogH2 = _build_TwogH2_matrix(
        hamiltonian_normal_ordered, 
        nspins, 
        ck_ops, 
        ckd_ops, 
        cmk_ops, 
        cmkd_ops
    )

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
        Ud_numeric (Optional[npt.NDArray[np.complex128]]): Numerical rotation matrix Ud.
    """

    def __init__(
        self,
        spin_magnitude: Optional[float] = None,
        hamiltonian_params: Optional[Union[List[float], npt.NDArray[np.float64]]] = None,
        cache_file_base: Optional[
            str
        ] = None,  # Made optional, will be derived from config if possible
        spin_model_module: Optional[types.ModuleType] = None,
        cache_mode: str = "r",
        Ud_numeric_override: Optional[npt.NDArray[np.complex128]] = None,
        config_filepath: Optional[str] = None,  # For configuration-driven model
    ):
        """
        Initializes the MagCalc LSWT calculator.

        Loads or generates the necessary symbolic matrices (Hamiltonian HMat=2gH,
        rotation Ud) based on the provided spin model and parameters. Pre-calculates
        the numerical rotation matrix Ud_numeric.

        Args:
            spin_magnitude (Optional[float]): The numerical value of the spin magnitude S.
                Must be positive. Required if config_filepath is None.
            hamiltonian_params (Optional[Union[List[float], npt.NDArray[np.float64]]]):
                A list or NumPy array containing the numerical values for the
                Hamiltonian parameters. Required if config_filepath is None.
            cache_file_base (str): The base filename (without path or extension)
                used for storing/retrieving cached symbolic matrices (HMat, Ud)
                in the 'pckFiles' subdirectory.
            spin_model_module (Optional[module]): The imported Python module
                containing the spin model definitions (e.g., Hamiltonian, mpr, atom_pos).
                Required if `config_filepath` is None.
            cache_mode (str, optional): Specifies the cache behavior.
                'r': Read symbolic matrices from cache files. Fails if files
                     don't exist or are invalid. (Default)
                'w': Generate symbolic matrices (potentially slow) and write
                     them to cache files.
                "auto": Reads cache if parameters match current settings; otherwise, (re)generates and writes cache.
            Ud_numeric_override (Optional[npt.NDArray[np.complex128]], optional):
                     them to cache files.
            config_filepath (Optional[str], optional): Path to the YAML configuration
                file. If provided, the spin model is loaded from this file, and
                `spin_model_module`, `spin_magnitude`, and `hamiltonian_params` might
                be overridden or ignored (partially, `spin_magnitude` and `hamiltonian_params` are used as fallback if not in config or for specific cases).
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

        if cache_mode not in ["r", "w", "auto"]:
            raise ValueError(
                f"Invalid cache_mode '{cache_mode}'. Use 'r', 'w', or 'auto'."
            )
        self.cache_mode = cache_mode

        # --- Define Cache Directories ---
        # Root cache directory, one level above the pyMagCalc package
        self.cache_root_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "pyMagCalc_cache")
        )
        self.symbolic_cache_dir = os.path.join(self.cache_root_dir, "symbolic_matrices")
        self.numerical_cache_dir = os.path.join(
            self.cache_root_dir, "numerical_results"
        )

        os.makedirs(self.cache_root_dir, exist_ok=True)
        os.makedirs(self.symbolic_cache_dir, exist_ok=True)
        os.makedirs(self.numerical_cache_dir, exist_ok=True)
        # --- End Cache Directory Definitions ---

        self.raw_config_data: Optional[Dict[str, Any]] = (
            None  # Stores the full loaded config
        )
        self.model_config_data: Optional[Dict[str, Any]] = (
            None  # Stores the relevant model section
        )
        self.config_filepath: Optional[str] = config_filepath  # Store config_filepath
        self.config_data: Optional[Dict[str, Any]] = None  # Ensure attribute exists
        self.p_numerical: Optional[Dict[str, float]] = (
            None  # Stores numerical params by name  # Stores numerical params by name
        )

        if config_filepath:
            logger.info(f"Using configuration file: {config_filepath}")
            self.config_data = load_spin_model_config(config_filepath)
            self.raw_config_data = (
                self.config_data
            )  # Keep a reference to the raw loaded data

            # Determine the actual model configuration section
            current_config_root = self.raw_config_data
            if (
                current_config_root is not None
                and "crystal_structure" not in current_config_root
            ):
                if len(current_config_root) == 1:
                    first_key = list(current_config_root.keys())[0]
                    potential_model_section = current_config_root[first_key]
                    if (
                        isinstance(potential_model_section, dict)
                        and "crystal_structure" in potential_model_section
                    ):
                        logger.info(
                            f"Model configuration found under top-level key: '{first_key}'"
                        )
                        self.model_config_data = potential_model_section
                    else:
                        self.model_config_data = current_config_root  # Will likely fail later, but consistently
                else:
                    self.model_config_data = (
                        current_config_root  # Will likely fail later
                    )
            else:
                self.model_config_data = (
                    current_config_root  # crystal_structure is at the root
                )

            if (
                self.model_config_data is None
            ):  # Should not happen if raw_config_data was not None
                raise ValueError(
                    f"Failed to determine model configuration section from: {config_filepath}"
                )

            # Derive cache_file_base from config_filepath if not provided
            if cache_file_base is None:
                base_name_from_config = os.path.splitext(
                    os.path.basename(config_filepath)
                )[0]
                self.cache_file_base = base_name_from_config + "_cache"
            else:
                self.cache_file_base = cache_file_base
            self.sm = generic_spin_model  # Use the generic spin model module

            # Extract spin_magnitude from config
            crystal_structure_data = self.model_config_data.get("crystal_structure")
            if crystal_structure_data is None:
                raise ValueError(
                    f"Missing required section 'crystal_structure' in model configuration from: {config_filepath}"
                )

            atoms_uc_config = crystal_structure_data.get("atoms_uc", [])
            if not atoms_uc_config:
                raise ValueError(
                    "Configuration error: 'atoms_uc' is missing or empty in 'crystal_structure'."
                )
            if "spin_S" not in atoms_uc_config[0]:
                raise ValueError(
                    "Configuration error: 'spin_S' is missing for the first atom in 'atoms_uc'."
                )
            self.spin_magnitude = float(atoms_uc_config[0]["spin_S"])
            if self.spin_magnitude <= 0:
                raise ValueError(
                    f"Spin magnitude from config ({self.spin_magnitude}) must be positive."
                )

            # Extract numerical Hamiltonian parameters from config
            # Try "model_params" first as per aCVO/config.yaml, then "parameters" as a fallback.
            self.p_numerical = self.model_config_data.get("model_params")
            if self.p_numerical is None:
                self.p_numerical = self.model_config_data.get("parameters")

            if self.p_numerical is None or not isinstance(self.p_numerical, dict):
                raise ValueError(
                    "Configuration error: Neither 'model_params' nor 'parameters' section found in model configuration, or it's not a dictionary."
                )
            if not all(isinstance(v, (int, float)) for v in self.p_numerical.values()):
                raise TypeError(
                    "All values in the parameter section ('model_params' or 'parameters') of config must be numbers."
                )

            # Ordered list of parameter names and symbols
            _parameter_names_ordered = list(self.p_numerical.keys())
            self.params_sym: List[sp.Symbol] = [
                sp.Symbol(name, real=True) for name in _parameter_names_ordered
            ]
            # Ordered list of numerical parameter values
            self.hamiltonian_params: List[float] = [
                float(self.p_numerical[name]) for name in _parameter_names_ordered
            ]

            # Determine nspins from config
            self.nspins = len(atoms_uc_config)
            if self.nspins == 0:  # Should have been caught by atoms_uc_config check
                raise ValueError(
                    "Configuration error: 'atoms_uc' in config led to zero spins."
                )
            logger.info(
                f"Model loaded from config: {self.nspins} spins, S={self.spin_magnitude}, {len(self.hamiltonian_params)} parameters."
            )

        else:  # Traditional spin_model_module approach
            logger.info("Using spin_model_module approach.")
            if spin_model_module is None:
                raise ValueError(
                    "spin_model_module must be provided if config_filepath is not set."
                )
            if cache_file_base is None:
                raise ValueError(
                    "cache_file_base must be provided if config_filepath is not set."
                )
            # Relaxed check: Allow class instances (like GenericSpinModel) that have necessary methods
            if not hasattr(spin_model_module, "Hamiltonian"): 
                raise TypeError(
                    "spin_model_module validation failed: missing 'Hamiltonian' method/attribute."
                )
            self.sm = spin_model_module

            # Validate and set spin_magnitude
            assert (
                spin_magnitude is not None
            ), "spin_magnitude cannot be None when config_filepath is not used"
            if not isinstance(spin_magnitude, (int, float)):
                raise TypeError("spin_magnitude must be a number.")
            if spin_magnitude <= 0:
                raise ValueError("spin_magnitude must be positive.")
            self.spin_magnitude = float(spin_magnitude)

            # Validate and set hamiltonian_params (list of floats)
            assert (
                hamiltonian_params is not None
            ), "hamiltonian_params cannot be None when config_filepath is not used"
            if isinstance(hamiltonian_params, np.ndarray):
                hamiltonian_params_list = hamiltonian_params.tolist()
            elif isinstance(hamiltonian_params, list):
                hamiltonian_params_list = hamiltonian_params
            else:
                raise TypeError("hamiltonian_params must be a list or NumPy array.")
            if not hamiltonian_params_list:  # Allow empty if model needs no params
                logger.warning("hamiltonian_params is empty.")
            if not all(isinstance(p, (int, float)) for p in hamiltonian_params_list):
                raise TypeError("All elements in hamiltonian_params must be numbers.")
            self.hamiltonian_params = [float(p) for p in hamiltonian_params_list]

            self.cache_file_base = cache_file_base
            # Create symbolic parameters (p0, p1, ...)
            num_params = len(self.hamiltonian_params)
            _params_sym_tuple = sp.symbols(f"p0:{num_params}", real=True)
            # _params_sym_tuple is always a sequence when using slice notation "p0:N"
            self.params_sym = list(_params_sym_tuple)

            # For consistency, create p_numerical for the old path too
            self.p_numerical = {
                sym.name: val
                for sym, val in zip(self.params_sym, self.hamiltonian_params)
            }

            # Check spin_model_module validity and get nspins
            self._validate_spin_model_module()
            try:
                self.nspins = len(self.sm.atom_pos())  # type: ignore
                if self.nspins == 0:
                    raise ValueError("spin_model.atom_pos() returned an empty list.")
            except Exception as e:
                logger.exception("Error getting nspins from spin_model.atom_pos()")
                raise RuntimeError("Failed to determine nspins from spin model.") from e

        self.kx, self.ky, self.kz = sp.symbols("kx ky kz", real=True)
        self.k_sym: List[sp.Symbol] = [self.kx, self.ky, self.kz]
        self.S_sym: sp.Symbol = sp.Symbol("S", real=True)

        # self.params_sym is already set in the if/else block above
        # For the traditional path, it's p0, p1...
        # For the config path, it's symbols derived from parameter names.
        if not hasattr(self, "params_sym") or self.params_sym is None:
            # This should not happen if the logic above is correct
            raise RuntimeError("self.params_sym was not set during initialization.")

        self.full_symbol_list: List[sp.Symbol] = (
            self.k_sym + [self.S_sym] + list(self.params_sym)
        )

        # --- Load or Generate Symbolic Matrices ---
        self.HMat_sym: Optional[sp.Matrix] = None
        self.Ud_sym: Optional[sp.Matrix] = None
        # _load_or_generate_matrices raises exceptions on failure
        self._load_or_generate_matrices()

        # --- Pre-calculate numerical Ud ---
        self.Ud_numeric: Optional[npt.NDArray[np.complex128]] = None
        if Ud_numeric_override is not None:
            logger.info("Using externally provided Ud_numeric_override.")
            self.set_external_Ud_numeric(Ud_numeric_override)  # Use the existing setter
        else:
            if self.Ud_sym is not None:
                # _calculate_numerical_ud raises exceptions on failure
                self._calculate_numerical_ud()
            else:
                # This case should ideally be caught by _load_or_generate_matrices
                raise RuntimeError(
                    "Ud_sym is None after matrix loading/generation and no Ud_numeric_override was provided."
                )

        if self.Ud_numeric is None:  # Final check
            raise RuntimeError("Ud_numeric was not set during initialization.")
        logger.info("MagCalc initialization complete.")

        # Initialize attribute for storing intermediate Hamiltonian matrices from dispersion calculation
        self._intermediate_numerical_H_matrices_disp: List[
            Optional[npt.NDArray[np.complex128]]
        ] = []

    def _validate_spin_model_module(self):
        """Checks if the provided spin_model_module has required functions."""
        if self.config_data:  # Skip if using config file
            return

        if self.sm is None: # Should not happen if logic is correct
            raise RuntimeError(
                "Spin model module (self.sm) is not properly set for validation."
            )

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

    def _read_symbolic_cache_metadata(
        self, meta_filepath: str
    ) -> Optional[Dict[str, Any]]:
        """Reads symbolic cache metadata from a JSON file."""
        try:
            with open(meta_filepath, "r") as f:
                metadata = json.load(f)
            logger.debug(f"Successfully read metadata from {meta_filepath}")
            return metadata
        except FileNotFoundError:
            logger.info(f"Metadata file not found: {meta_filepath}")
            return None
        except json.JSONDecodeError:
            logger.warning(
                f"Error decoding JSON from metadata file {meta_filepath}. File might be corrupted."
            )
            return None
        except Exception as e:
            logger.warning(f"Failed to read metadata file {meta_filepath}: {e}")
            return None

    def _write_symbolic_cache_metadata(self, meta_filepath: str):
        """Writes current model parameters to a symbolic cache metadata JSON file."""
        model_source_type = "config" if self.config_data else "module"
        model_identifier = self.config_filepath if self.config_data else self.sm.__name__  # type: ignore

        metadata = {
            "spin_magnitude": self.spin_magnitude,
            "hamiltonian_params": self.hamiltonian_params,
            "model_source_type": model_source_type,
            "model_identifier": model_identifier,
            # "pyMagCalc_version": __version__ # TODO: Add versioning if MagCalc gets one
        }
        try:
            with open(meta_filepath, "w") as f:
                json.dump(metadata, f, indent=4)
            logger.info(f"Symbolic cache metadata saved to {meta_filepath}")
        except Exception as e:
            logger.error(f"Failed to write metadata file {meta_filepath}: {e}")

    def _check_parameter_consistency_with_cache(
        self, cached_meta: Dict[str, Any]
    ) -> bool:
        """Compares current parameters with cached metadata."""
        if cached_meta is None:
            return False

        current_model_identifier = self.config_filepath if self.config_data else self.sm.__name__  # type: ignore

        s_match = np.isclose(cached_meta.get("spin_magnitude"), self.spin_magnitude)
        p_match = np.allclose(
            cached_meta.get("hamiltonian_params", []),
            self.hamiltonian_params,
            equal_nan=True,
        )
        id_match = cached_meta.get("model_identifier") == current_model_identifier

        if not s_match:
            logger.info(
                f"Metadata check: spin_magnitude mismatch (cache: {cached_meta.get('spin_magnitude')}, current: {self.spin_magnitude})."
            )
        if not p_match:
            logger.info(
                f"Metadata check: hamiltonian_params mismatch (cache: {cached_meta.get('hamiltonian_params')}, current: {self.hamiltonian_params})."
            )
        if not id_match:
            logger.info(
                f"Metadata check: model_identifier mismatch (cache: {cached_meta.get('model_identifier')}, current: {current_model_identifier})."
            )

        return s_match and p_match and id_match

    def _generate_and_save_symbolic_matrices(
        self, hm_cache_file: str, ud_cache_file: str, meta_cache_file: str
    ):
        """Generates symbolic matrices and saves them along with metadata."""
        logger.info(
            f"Generating symbolic matrices (HMat, Ud) for {self.cache_file_base}..."
        )
        if self.config_data:
            try:
                self.HMat_sym, self.Ud_sym = self._generate_matrices_from_config()
            except Exception as e:
                logger.exception(
                    "Failed to generate symbolic matrices from configuration."
                )
                raise RuntimeError(
                    "Symbolic matrix generation from config failed."
                ) from e
        else:  # Old path
            try:
                self.HMat_sym, self.Ud_sym = gen_HM(
                    self.sm,  # type: ignore
                    self.k_sym,
                    self.S_sym,
                    list(self.params_sym),
                )
            except Exception as e:
                logger.exception("Failed to generate symbolic matrices in gen_HM.")
                raise RuntimeError("Symbolic matrix generation failed.") from e

        if not isinstance(self.HMat_sym, sp.Matrix) or not isinstance(
            self.Ud_sym, sp.Matrix
        ):
            raise RuntimeError(
                "Symbolic matrix generation did not return valid SymPy Matrices."
            )

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

        self._write_symbolic_cache_metadata(meta_cache_file)

    def _load_or_generate_matrices(self):
        """
        Load symbolic HMat (2gH) and Ud matrices from cache or generate them.
        Handles reading from `.pck` files if `cache_mode='r'` or 'auto' (and valid),
        or calling `gen_HM` and writing the files if `cache_mode='w'` or 'auto' (and regeneration needed).
        """
        hm_cache_file: str = os.path.join(
            self.symbolic_cache_dir, self.cache_file_base + "_HM.pck"
        )
        ud_cache_file: str = os.path.join(  # type: ignore
            self.symbolic_cache_dir, self.cache_file_base + "_Ud.pck"
        )
        meta_cache_file: str = os.path.join(
            self.symbolic_cache_dir, self.cache_file_base + "_meta.json"
        )

        if self.cache_mode == "auto":
            perform_generation = True
            if (
                os.path.exists(hm_cache_file)
                and os.path.exists(ud_cache_file)
                and os.path.exists(meta_cache_file)
            ):
                try:
                    logger.info(
                        f"Auto mode: Found existing symbolic cache and metadata for {self.cache_file_base}."
                    )
                    with open(hm_cache_file, "rb") as f_hm:
                        self.HMat_sym = pickle.load(f_hm)
                    with open(ud_cache_file, "rb") as f_ud:
                        self.Ud_sym = pickle.load(f_ud)

                    cached_meta = self._read_symbolic_cache_metadata(meta_cache_file)
                    if cached_meta and self._check_parameter_consistency_with_cache(
                        cached_meta
                    ):
                        logger.info(
                            f"Auto mode: Symbolic cache is valid for {self.cache_file_base}. Using cached matrices."
                        )
                        perform_generation = False
                    else:
                        logger.info(
                            f"Auto mode: Symbolic cache parameters mismatch or metadata invalid for {self.cache_file_base}. Regenerating."
                        )
                except Exception as e:
                    logger.warning(
                        f"Auto mode: Error loading or validating symbolic cache for {self.cache_file_base}. Regenerating. Error: {e}"
                    )
                    self.HMat_sym = None
                    self.Ud_sym = None
            else:
                logger.info(
                    f"Auto mode: Symbolic cache or metadata not found for {self.cache_file_base}. Generating."
                )

            if perform_generation:
                self._generate_and_save_symbolic_matrices(
                    hm_cache_file, ud_cache_file, meta_cache_file
                )

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
                    f"Cache file not found: {e}. Run with cache_mode='w' or 'auto' first."
                )
                raise
            except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as e:
                logger.error(
                    f"Error loading cache files (may be corrupted or incompatible): {e}"
                )
                raise pickle.PickleError("Failed to load cache file.") from e
            except Exception as e:  # Catch any other unexpected error during loading
                logger.exception("An unexpected error occurred loading cache files.")
                raise RuntimeError("Cache file loading failed.") from e

        elif self.cache_mode == "w":
            self._generate_and_save_symbolic_matrices(
                hm_cache_file, ud_cache_file, meta_cache_file
            )
        else:  # Should have been caught by __init__
            raise ValueError(
                f"Internal error: Unhandled cache_mode '{self.cache_mode}' in _load_or_generate_matrices."
            )
        # Final check after load/generate
        if self.HMat_sym is None or self.Ud_sym is None:
            raise RuntimeError(
                f"Symbolic matrices HMat_sym or Ud_sym are None after loading/generation."
            )
        if not isinstance(self.HMat_sym, sp.Matrix) or not isinstance(
            self.Ud_sym, sp.Matrix  # type: ignore
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
            Ud_num_sym = self.Ud_sym.evalf(subs=dict(param_substitutions_ud))  # type: ignore
            self.Ud_numeric = np.array(Ud_num_sym, dtype=np.complex128)  # type: ignore
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
        self, new_hamiltonian_params: Union[List[float], npt.NDArray[np.float64]]
    ):
        """
        Update the Hamiltonian parameters and recalculate dependent numerical matrices.

        Args:
            new_hamiltonian_params (Union[List[float], npt.NDArray[np.float64]]):
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

    def set_external_Ud_numeric(self, Ud_matrix_numerical: npt.NDArray[np.complex128]):
        """
        Allows setting the Ud_numeric matrix externally.
        This is useful if Ud_numeric is derived from a field-dependent classical ground state
        that is determined outside the initial symbolic generation of Ud_sym.

        Args:
            Ud_matrix_numerical (npt.NDArray[np.complex128]): The externally calculated
                numerical Ud matrix (3N x 3N).
        """
        if (
            not isinstance(Ud_matrix_numerical, np.ndarray)
            or Ud_matrix_numerical.ndim != 2
        ):
            raise TypeError("Ud_matrix_numerical must be a 2D NumPy array.")

        expected_dim = 3 * self.nspins
        if Ud_matrix_numerical.shape != (expected_dim, expected_dim):
            raise ValueError(
                f"Ud_matrix_numerical has incorrect shape {Ud_matrix_numerical.shape}. Expected ({expected_dim}, {expected_dim})."
            )

        self.Ud_numeric = Ud_matrix_numerical.astype(np.complex128)
        logger.info(
            f"External Ud_numeric matrix has been set. Shape: {self.Ud_numeric.shape}"
        )

    def _generate_matrices_from_config(self) -> Tuple[sp.Matrix, sp.Matrix]:
        """
        Generate symbolic HMat (2gH) and Ud matrices using the loaded configuration.
        """
        logger.info("Starting symbolic matrix generation from configuration...")
        start_time_total = timeit.default_timer()

        if not self.model_config_data or self.sm is not generic_spin_model:
            raise RuntimeError(
                "Configuration data not loaded or incorrect spin model module for _generate_matrices_from_config."
            )

        # 1. Get Model Info from Config
        uc_vecs_cart = self.sm.unit_cell_from_config(
            self.config_data["crystal_structure"]
        )  # In generic_spin_model, this takes the crystal_structure dict directly
        atom_pos_uc_cart = self.sm.atom_pos_from_config(
            self.model_config_data["crystal_structure"], uc_vecs_cart
        )
        # self.nspins is already set from config in __init__

        atom_pos_ouc_cart = self.sm.atom_pos_ouc_from_config(
            atom_pos_uc_cart,
            uc_vecs_cart,
            self.model_config_data.get("calculation_settings", {}),
        )
        nspins_ouc = len(atom_pos_ouc_cart)

        symbolic_params_map_for_gsm = {sym.name: sym for sym in self.params_sym}

        rotation_matrices_sym_list = self.sm.mpr_from_config(
            self.model_config_data["crystal_structure"],  # Pass crystal_structure dict
            # Pass symbolic parameters for symbolic Ud_sym generation
            # This assumes mpr_from_config can handle symbolic dict or may need adjustment
            # in generic_spin_model.py if it strictly expects numerical values.
            symbolic_params_map_for_gsm,  # Pass parameters as positional argument
        )

        # 2. Setup Holstein-Primakoff Operators (for OUC)
        c_ops_ouc, cd_ops_ouc, spin_ops_local_ouc = _setup_hp_operators(
            nspins_ouc, self.S_sym
        )

        # 3. Rotate Spin Operators (OUC spins, UC rotations applied cyclically)
        spin_ops_global_ouc = _rotate_spin_operators(
            spin_ops_local_ouc, rotation_matrices_sym_list, self.nspins, nspins_ouc
        )

        # 4. Prepare Hamiltonian from Config
        hamiltonian_sym_config = self.sm.Hamiltonian_from_config(
            spin_ops_global_ouc, symbolic_params_map_for_gsm, self.model_config_data
        )
        # logger.debug(f"Config-driven: Hamiltonian_sym_config (first 500 chars): {str(hamiltonian_sym_config)[:500]}")

        # Hamiltonian_from_config already calls .expand()
        raw_hamiltonian_from_config = hamiltonian_sym_config  # Rename for clarity
        logger.debug(
            f"Config-driven: Raw Hamiltonian from config (first 1000 chars): {str(raw_hamiltonian_from_config)[:1000]}"
        )
        logger.debug(
            f"Config-driven: Raw Hamiltonian from config has {len(raw_hamiltonian_from_config.as_ordered_terms())} terms."
        )

        # --- Filter Hamiltonian terms (Keep only up to quadratic in boson ops) ---
        # For LSWT (H2), we need terms quadratic in boson operators.
        # These typically arise from:
        # 1. S*S type interactions: quadratic boson part is ~ S_sym^1
        # 2. H*S_z type interactions (Zeeman): quadratic boson part is ~ S_sym^0 (no S_sym factor from HP of Sz)
        # We must exclude S_sym^2 terms (quartic bosons) and linear boson terms.

        filtered_terms_config = []
        # Coefficient of S^0 (e.g., Zeeman term H_field*(-cd*c))
        hamiltonian_S0_conf = raw_hamiltonian_from_config.coeff(self.S_sym, 0)
        if hamiltonian_S0_conf != 0:
            filtered_terms_config.append(hamiltonian_S0_conf)

        # Coefficient of S^1 (e.g., J*S_sym*(cd*c + c*c + cd*cd))
        hamiltonian_S1_conf = raw_hamiltonian_from_config.coeff(self.S_sym, 1)
        if hamiltonian_S1_conf != 0:
            filtered_terms_config.append(hamiltonian_S1_conf * self.S_sym)

        # DO NOT include S_sym^2 terms for H2 (LSWT) as they lead to quartic boson terms.

        if not filtered_terms_config:
            logger.warning(
                "Config-driven S-based Hamiltonian filtering resulted in zero terms. Using original raw Hamiltonian."
            )
            hamiltonian_after_S_filter = raw_hamiltonian_from_config
        else:
            hamiltonian_after_S_filter = Add(*filtered_terms_config)

        hamiltonian_after_S_filter = sp.expand(hamiltonian_after_S_filter)
        # --- End Filtering ---

        # --- Explicitly remove terms linear in c_ops_ouc and cd_ops_ouc ---
        logger.debug(
            f"Hamiltonian before explicit linear term removal (first 500): {str(hamiltonian_after_S_filter)[:500]}"
        )
        terms_to_process = hamiltonian_after_S_filter.as_ordered_terms()
        final_H2_terms = []
        linear_terms_removed_count = 0
        all_boson_ops_set = set(c_ops_ouc + cd_ops_ouc)

        for term in terms_to_process:
            boson_ops_in_term = term.free_symbols.intersection(all_boson_ops_set)
            # Heuristic: if a term has exactly one boson operator symbol among its free symbols,
            # consider it a linear term to be removed. This is an approximation.
            if len(boson_ops_in_term) == 1 and not term.has_only_symbol(
                *boson_ops_in_term,
                count_ops=True,
                exact_powers=[(op, 1) for op in boson_ops_in_term],
            ):  # A more refined check might be needed
                logger.debug(f"Removing potential linear term: {term}")
                linear_terms_removed_count += 1
            else:
                final_H2_terms.append(term)

        if linear_terms_removed_count > 0:
            logger.info(
                f"Explicitly removed {linear_terms_removed_count} terms identified as linear in boson operators."
            )
            hamiltonian_sym_config = Add(*final_H2_terms)
            hamiltonian_sym_config = sp.expand(hamiltonian_sym_config)
        else:
            hamiltonian_sym_config = (
                hamiltonian_after_S_filter  # No linear terms removed by this heuristic
            )

        logger.debug(
            f"Config-driven: Hamiltonian_sym_config AFTER filtering (first 1000 chars): {str(hamiltonian_sym_config)[:1000]}"
        )
        logger.debug(
            f"Config-driven: Hamiltonian_sym_config AFTER filtering has {len(hamiltonian_sym_config.as_ordered_terms())} terms."
        )

        logger.debug(
            f"Config-driven: Initial symbolic Hamiltonian has {len(hamiltonian_sym_config.as_ordered_terms())} terms."
        )

        # 5. Define k-space operators (UC scope)
        ck_ops_uc = [sp.Symbol(f"ck{j}", commutative=False) for j in range(self.nspins)]
        ckd_ops_uc = [
            sp.Symbol(f"ckd{j}", commutative=False) for j in range(self.nspins)
        ]
        cmk_ops_uc = [
            sp.Symbol(f"cmk{j}", commutative=False) for j in range(self.nspins)
        ]
        cmkd_ops_uc = [
            sp.Symbol(f"cmkd{j}", commutative=False) for j in range(self.nspins)
        ]

        # 6. Define Fourier Substitutions
        Jex_sym_matrix, _ = (
            self.sm.spin_interactions_from_config(  # DM matrix also returned but not used here
                symbolic_params_map_for_gsm,  # This is a dict of {name: symbol}
                self.model_config_data["interactions"],  # Pass interactions dict
                atom_pos_uc_cart,
                atom_pos_ouc_cart,
                uc_vecs_cart,
            )
        )
        logger.debug(f"Config-driven: Jex_sym_matrix for FT rules:\n{Jex_sym_matrix}")

        fourier_substitutions = _define_fourier_substitutions_generic(
            self.k_sym,
            self.nspins,
            c_ops_ouc,
            cd_ops_ouc,
            ck_ops_uc,
            ckd_ops_uc,
            cmk_ops_uc,
            cmkd_ops_uc,
            atom_pos_uc_cart,
            atom_pos_ouc_cart,
            Jex_sym_matrix,
        )
        logger.debug(
            f"Config-driven: Number of Fourier substitution rules: {len(fourier_substitutions)}"
        )
        # Log a few sample FT rules
        if fourier_substitutions:
            logger.debug("Config-driven: Sample Fourier substitution rules (first 3):")
            for i, rule in enumerate(fourier_substitutions[:3]):
                logger.debug(f"  Rule {i}: {rule[0]} -> {rule[1]}")
        # 7. Define Commutation and Placeholder Substitutions
        commutation_substitutions = _define_commutation_substitutions(
            self.nspins, ck_ops_uc, ckd_ops_uc, cmk_ops_uc, cmkd_ops_uc
        )  # No self needed
        basis_vector_dagger = ckd_ops_uc + cmk_ops_uc
        basis_vector = ck_ops_uc + cmkd_ops_uc
        placeholder_symbols, placeholder_substitutions = (
            _define_placeholder_substitutions(  # No self needed
                self.nspins, basis_vector_dagger, basis_vector
            )
        )

        # 8. Apply Substitutions
        hamiltonian_with_placeholders = _apply_substitutions_parallel(
            hamiltonian_sym_config,
            fourier_substitutions,
            commutation_substitutions,
            placeholder_substitutions,
        )
        H2_matrix = _extract_h2_matrix(
            hamiltonian_with_placeholders, placeholder_symbols, self.nspins
        )  # No self needed

        # 10. Calculate TwogH2
        g_metric_tensor_sym = sp.diag(*([1] * self.nspins + [-1] * self.nspins))
        dynamical_matrix_TwogH2 = 2 * g_metric_tensor_sym * H2_matrix

        # 11. Build Ud Matrix from Config
        Ud_rotation_matrix = _build_ud_matrix(
            rotation_matrices_sym_list, self.nspins
        )  # No self needed

        end_time_total = timeit.default_timer()
        logger.info(
            f"Total run-time for _generate_matrices_from_config: {np.round((end_time_total - start_time_total), 2)} s."
        )

        # --- Sanity Check for HMat_sym ---
        expected_symbols_in_HMat = set(self.k_sym + [self.S_sym] + self.params_sym)
        actual_symbols_in_HMat = dynamical_matrix_TwogH2.free_symbols
        unexpected_symbols = actual_symbols_in_HMat - expected_symbols_in_HMat
        if unexpected_symbols:
            logger.error(
                f"Config-driven HMat_sym (dynamical_matrix_TwogH2) contains unexpected free symbols: {unexpected_symbols}"
            )
            # Detailed element check (optional, can be very verbose)
            # for r_idx in range(dynamical_matrix_TwogH2.rows):
            #     for c_idx in range(dynamical_matrix_TwogH2.cols):
            #         elem_symbols = dynamical_matrix_TwogH2[r_idx, c_idx].free_symbols
            #         if elem_symbols - expected_symbols_in_HMat:
            #             logger.error(f"  Element ({r_idx},{c_idx}) '{dynamical_matrix_TwogH2[r_idx, c_idx]}' has unexpected: {elem_symbols - expected_symbols_in_HMat}")
            raise RuntimeError(
                f"Generated HMat_sym from config contains unexpected symbols: {unexpected_symbols}. This will cause lambdify to fail."
            )
        return dynamical_matrix_TwogH2, Ud_rotation_matrix

    def _generate_numerical_cache_key(
        self, q_vectors_list: List[npt.NDArray[np.float64]], calculation_type: str
    ) -> str:
        """
        Generates a unique cache key for numerical results.
        Args:
            q_vectors_list: List of 1D NumPy arrays representing q-vectors.
            calculation_type: String identifier like "dispersion" or "sqw".
        Returns:
            A hex digest string representing the cache key.
        """
        hasher = hashlib.md5()

        # 1. Symbolic model identifier
        hasher.update(str(self.cache_file_base).encode("utf-8"))
        # 2. Spin magnitude
        hasher.update(str(self.spin_magnitude).encode("utf-8"))
        # 3. Hamiltonian parameters
        hasher.update(str(self.hamiltonian_params).encode("utf-8"))
        # 4. Ud_numeric matrix (critical for spin configuration)
        if self.Ud_numeric is not None:
            hasher.update(self.Ud_numeric.tobytes())
        else:
            # This case should ideally not be reached if __init__ ensures Ud_numeric is set
            hasher.update(b"Ud_numeric_None")
            logger.warning("_generate_numerical_cache_key: self.Ud_numeric is None.")
        # 5. Calculation type
        hasher.update(calculation_type.encode("utf-8"))
        # 6. Q-vectors content and order
        for q_vec in q_vectors_list:
            hasher.update(q_vec.tobytes())

        return calculation_type + "_" + hasher.hexdigest()

    # --- END NEW METHODS ---

    def calculate_dispersion(
        self,
        q_vectors: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> Optional[List[Optional[npt.NDArray[np.float64]]]]:
        """
        Calculate the spin-wave dispersion relation over a list of q-points.

        Args:
            q_vectors (Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]):
                A list or NumPy array of momentum vectors q = [qx, qy, qz].
                Each vector should be a 1D array/list of 3 numbers.

        Returns:
            Optional[List[Optional[npt.NDArray[np.float64]]]]: A list containing the
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

        # --- Numerical Cache Check ---
        cache_key = self._generate_numerical_cache_key(q_vectors_list, "dispersion")
        cache_filepath = os.path.join(self.numerical_cache_dir, cache_key + ".pkl")

        if os.path.exists(cache_filepath):
            try:
                with open(cache_filepath, "rb") as f:
                    cached_energies = pickle.load(f)
                logger.info(
                    f"Loaded dispersion results from numerical cache: {cache_filepath}"
                )
                # Note: _intermediate_numerical_H_matrices_disp will not be populated from cache
                self._intermediate_numerical_H_matrices_disp = [None] * len(
                    q_vectors_list
                )
                return cached_energies
            except Exception as e:
                logger.warning(
                    f"Failed to load from numerical cache {cache_filepath}: {e}. Recalculating."
                )
        # Clear previous intermediate matrices if any
        self._intermediate_numerical_H_matrices_disp = []

        start_time: float = timeit.default_timer()

        pool_args: List[Tuple] = [
            (
                q_vec,
                self.nspins,
                self.spin_magnitude,
                self.hamiltonian_params,
            )
            for q_vec in q_vectors_list  # Use validated list
        ]

        results_from_pool: List[
            Tuple[npt.NDArray[np.float64], Optional[npt.NDArray[np.complex128]]]
        ] = []
        try:
            # Use context manager for Pool
            with Pool(
                initializer=_init_worker,
                initargs=(self.HMat_sym, self.full_symbol_list)
            ) as pool:
                results_from_pool = list(pool.imap(process_calc_disp, pool_args))
        except Exception:
            logger.exception(
                "Error during multiprocessing pool execution for dispersion."
            )
            return None

        # Unpack results and store intermediate matrices
        energies_list: List[Optional[npt.NDArray[np.float64]]] = []
        for res_energies, res_H_matrix in results_from_pool:
            if res_energies is not None and np.all(
                np.isnan(res_energies)
            ):  # if all are nan, treat as None for consistency
                energies_list.append(None)
            else:
                energies_list.append(res_energies)
            self._intermediate_numerical_H_matrices_disp.append(res_H_matrix)

        num_failures = sum(
            1 for en in energies_list if en is None or np.isnan(en).any()
        )
        if num_failures > 0:
            logger.warning(
                f"Dispersion calculation failed for {num_failures} out of {len(q_vectors_list)} q-points. Check logs for details."
            )

        end_time: float = timeit.default_timer()
        logger.info(
            f"Run-time for dispersion calculation: {np.round((end_time - start_time) / 60, 2)} min."
        )

        # --- Save to Numerical Cache ---
        try:
            with open(cache_filepath, "wb") as f:
                pickle.dump(energies_list, f)
            logger.info(
                f"Saved dispersion results to numerical cache: {cache_filepath}"
            )
        except Exception as e:
            logger.warning(f"Failed to save to numerical cache {cache_filepath}: {e}")

        return energies_list

    def calculate_sqw(
        self,
        q_vectors: Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    ) -> Tuple[
        Optional[Tuple[npt.NDArray[np.float64], ...]],
        Optional[Tuple[npt.NDArray[np.float64], ...]],
        Optional[Tuple[npt.NDArray[np.float64], ...]],
    ]:
        """
        Calculate the dynamical structure factor S(q,w) over a list of q-points.

        Args:
            q_vectors (Union[List[npt.NDArray[np.float64]], npt.NDArray[np.float64]]):
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

        # --- Numerical Cache Check ---
        cache_key = self._generate_numerical_cache_key(q_vectors_list, "sqw")
        cache_filepath = os.path.join(self.numerical_cache_dir, cache_key + ".pkl")

        if os.path.exists(cache_filepath):
            try:
                with open(cache_filepath, "rb") as f:
                    cached_q_out, cached_energies, cached_intensities = pickle.load(f)
                logger.info(
                    f"Loaded S(q,w) results from numerical cache: {cache_filepath}"
                )
                return cached_q_out, cached_energies, cached_intensities
            except Exception as e:
                logger.warning(
                    f"Failed to load S(q,w) from numerical cache {cache_filepath}: {e}. Recalculating."
                )
        # --- End Numerical Cache Check ---

        start_time: float = timeit.default_timer()

        pool_args: List[Tuple] = [
            (
                self.Ud_numeric,
                q_vec,
                self.nspins,
                self.spin_magnitude,
                self.hamiltonian_params,
            )
            for q_vec in q_vectors_list  # Use validated list
        ]
        results: List[
            Tuple[
                npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]
            ]
        ] = []
        try:
            with Pool(
                initializer=_init_worker, 
                initargs=(self.HMat_sym, self.full_symbol_list)
            ) as pool:
                results = list(pool.imap(process_calc_Sqw, pool_args))
        except Exception:
            logger.exception("Error during multiprocessing pool execution for S(q,w).")
            return None, None, None

        q_vectors_out: Tuple[npt.NDArray[np.float64], ...]
        energies_out: Tuple[npt.NDArray[np.float64], ...]
        intensities_out: Tuple[npt.NDArray[np.float64], ...]
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

        # --- Save to Numerical Cache ---
        results_to_cache = (q_vectors_out, energies_out, intensities_out)
        try:
            with open(cache_filepath, "wb") as f:
                pickle.dump(results_to_cache, f)
            logger.info(f"Saved S(q,w) results to numerical cache: {cache_filepath}")
        except Exception as e:
            logger.warning(
                f"Failed to save S(q,w) to numerical cache {cache_filepath}: {e}"
            )
        # --- End Save to Numerical Cache ---

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


# --- Plotting Helper Functions ---
def plot_dispersion_from_data(
    q_values: np.ndarray,
    energies_list: List[Optional[npt.NDArray[np.float64]]],
    title: str = "Spin Wave Dispersion",
    q_labels: Optional[List[str]] = None,
    q_ticks_positions: Optional[List[float]] = None,
    save_filename: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plots spin wave dispersion from loaded data.

    Args:
        q_values (np.ndarray): Array of q-values for the x-axis (e.g., path length).
        energies_list (List[Optional[npt.NDArray[np.float64]]]): List of energy arrays,
            one for each q-point. Each array contains energies for different modes.
        title (str): Title of the plot.
        q_labels (Optional[List[str]]): Labels for specific q-points on the x-axis.
        q_ticks_positions (Optional[List[float]]): Positions for q_labels.
        save_filename (Optional[str]): If provided, saves the plot to this file.
        show_plot (bool): If True, displays the plot.
    """
    logger.info(f"Plotting dispersion: {title}")
    plt.figure(figsize=(8, 6))

    num_modes = 0
    if energies_list and energies_list[0] is not None:
        num_modes = energies_list[0].shape[0]

    for mode_idx in range(num_modes):
        mode_energies = []
        valid_q_values_for_mode = []
        for i, q_energy_array in enumerate(energies_list):
            if q_energy_array is not None and mode_idx < len(q_energy_array):
                mode_energies.append(q_energy_array[mode_idx])
                valid_q_values_for_mode.append(q_values[i])
        if mode_energies:
            plt.plot(valid_q_values_for_mode, mode_energies, marker=".", linestyle="-")

    plt.xlabel("q (path length or index)")
    plt.ylabel("Energy (meV)")
    plt.title(title)
    if q_labels and q_ticks_positions:
        plt.xticks(q_ticks_positions, q_labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        logger.info(f"Dispersion plot saved to {save_filename}")
    if show_plot:
        plt.show()
    plt.close()


def plot_sqw_from_data(
    q_values: np.ndarray,
    energies_list: List[Optional[npt.NDArray[np.float64]]],
    intensities_list: List[Optional[npt.NDArray[np.float64]]],
    title: str = "S(q,w) Intensity Map",
    energy_max: Optional[float] = None,
    q_labels: Optional[List[str]] = None,
    q_ticks_positions: Optional[List[float]] = None,
    save_filename: Optional[str] = None,
    show_plot: bool = True,
):
    """
    Plots S(q,w) intensity map from loaded data using a scatter plot.
    Size and color of points represent intensity.
    """
    logger.info(f"Plotting S(q,w) map: {title}")
    plt.figure(figsize=(10, 6))

    all_q = []
    all_e = []
    all_i = []

    for i, (q_val, e_arr, i_arr) in enumerate(
        zip(q_values, energies_list, intensities_list)
    ):
        if e_arr is not None and i_arr is not None:
            for energy, intensity in zip(e_arr, i_arr):
                if (
                    not np.isnan(energy)
                    and not np.isnan(intensity)
                    and intensity > 1e-3
                ):  # Threshold intensity
                    all_q.append(q_val)
                    all_e.append(energy)
                    all_i.append(intensity)

    if not all_q:
        logger.warning("No data to plot for S(q,w).")
        return

    scatter = plt.scatter(
        all_q, all_e, c=all_i, s=np.sqrt(all_i) * 20, cmap="viridis", alpha=0.7
    )  # Scale size for visibility
    plt.colorbar(scatter, label="Intensity (arb. units)")
    plt.xlabel("q (path length or index)")
    plt.ylabel("Energy (meV)")
    plt.title(title)
    if energy_max:
        plt.ylim(0, energy_max)
    if q_labels and q_ticks_positions:
        plt.xticks(q_ticks_positions, q_labels)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    if save_filename:
        plt.savefig(save_filename)
        logger.info(f"S(q,w) map saved to {save_filename}")
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    """
    Example script demonstrating the usage of the MagCalc class.

    1. Imports a spin model (`spin_model.py` by default).
    2. Defines parameters, q-points, and cache settings.
    3. Instantiates `MagCalc`.
    4. Calculates and saves initial dispersion.
    5. Updates parameters and recalculates/saves dispersion and S(q,w).
    6. Optionally loads and plots S(q,w) from file.
    """
    # --- Control Flags for KFe3J Example ---
    CALCULATE_NEW_DATA = True  # Calculate new data or use existing files for plotting
    PLOT_FROM_DISP_FILE = False  # Plot dispersion from a saved file
    PLOT_FROM_SQW_FILE = False  # Plot S(q,w) from a saved file
    SHOW_PLOTS_AFTER_CALC = True  # Show plots immediately after calculation

    # --- Import the KFe3J spin model ---
    try:
        # Assuming KFe3J package is in a directory accessible by Python
        # If KFe3J is in the same directory as pyMagCalc or a subdirectory of project_root_dir
        # and project_root_dir is in sys.path, this should work.
        # For a typical project structure where KFe3J is a sibling to pyMagCalc:
        current_script_dir_for_example = os.path.dirname(os.path.abspath(__file__))
        project_root_dir_for_example = os.path.dirname(
            current_script_dir_for_example
        )  # pyMagCalc's parent
        kfe3j_module_dir = os.path.join(project_root_dir_for_example, "KFe3J")
        if kfe3j_module_dir not in sys.path:
            sys.path.insert(0, kfe3j_module_dir)
        import spin_model as kfe3j_spin_model
    except ImportError:
        logger.error(
            "Failed to import 'KFe3J.spin_model'. Ensure KFe3J directory is accessible."
        )
        sys.exit(1)  # Exit if the model cannot be imported

    # --- KFe3J User Configuration ---
    spin_S_val: float = 2.5  # S for Fe3+
    # Order for KFe3J model: J1, J2, Dy, Dz, H_field
    hamiltonian_params_val: List[float] = [3.23, 0.11, 0.218, -0.195, 0.0]
    cache_file_base_name: str = "kfe3j_example_cache"
    cache_operation_mode: str = "w"  # Use 'w' to generate cache first time, then 'r'
    output_filename_base: str = "kfe3j_example_results"

    # Define q points for KFe3J (example path from test script)
    q_points_list: List[List[float]] = []
    # Gamma to M path (along kx up to 2*pi/sqrt(3))
    N_gamma_m = 10
    for qx_val in np.linspace(
        0, 2 * np.pi / np.sqrt(3), N_gamma_m, endpoint=True
    ):  # Include M
        q_points_list.append([qx_val, 0, 0])

    # M to K path (example, M is (2pi/sqrt(3),0,0), K is (2pi/sqrt(3), 2pi/3,0) )
    # For this example, let's go from M towards K along ky
    N_m_k_segment = 6  # Number of points from M (exclusive) to K (inclusive)
    m_point = np.array([2 * np.pi / np.sqrt(3), 0, 0])
    k_point_approx = np.array([2 * np.pi / np.sqrt(3), 2 * np.pi / 3, 0])

    # Generate points from M (exclusive) to K (inclusive)
    for i in range(1, N_m_k_segment + 1):
        frac = i / N_m_k_segment
        q_pt = m_point + frac * (k_point_approx - m_point)
        q_points_list.append(q_pt.tolist())

    q_points_array: npt.NDArray[np.float64] = np.array(q_points_list)
    # --- End User Configuration ---

    logger.info("Starting example calculation using MagCalc class...")

    # Define q-path labels and positions (example)
    # These should correspond to your q_points_array structure
    # For KFe3J path: Gamma -> M -> K
    q_path_distances = np.zeros(len(q_points_array))
    for i in range(1, len(q_points_array)):
        q_path_distances[i] = q_path_distances[i - 1] + np.linalg.norm(
            q_points_array[i] - q_points_array[i - 1]
        )

    q_special_labels = ["Γ", "M", "K"]
    q_special_positions = [
        q_path_distances[0],  # Gamma
        q_path_distances[N_gamma_m - 1],  # M
        q_path_distances[-1],  # K
    ]

    disp_filename_initial = f"{output_filename_base}_disp_initial.npz"
    disp_filename_updated = f"{output_filename_base}_disp_updated.npz"
    sqw_filename_updated = f"{output_filename_base}_sqw_updated.npz"

    if CALCULATE_NEW_DATA:
        try:
            # --- Instantiate the Calculator ---
            calculator = MagCalc(
                spin_magnitude=spin_S_val,
                hamiltonian_params=hamiltonian_params_val,
                cache_file_base=cache_file_base_name,
                cache_mode=cache_operation_mode,  # Use 'w' to generate cache first time
                spin_model_module=kfe3j_spin_model,
            )

            # --- Calculate and Save Dispersion (Initial Params) ---
            logger.info("Calculating dispersion (Initial Params)...")
            dispersion_energies_initial = calculator.calculate_dispersion(
                q_points_array
            )
            if dispersion_energies_initial is not None:
                calculator.save_results(
                    disp_filename_initial,
                    {
                        "q_values_path": q_path_distances,
                        "energies_list": dispersion_energies_initial,
                        "q_labels": q_special_labels,
                        "q_ticks_positions": q_special_positions,
                    },
                )
                if SHOW_PLOTS_AFTER_CALC:
                    plot_dispersion_from_data(
                        q_path_distances,
                        dispersion_energies_initial,
                        title="Dispersion (Initial Params)",
                        q_labels=q_special_labels,
                        q_ticks_positions=q_special_positions,
                    )
            else:
                logger.error("Initial dispersion calculation failed.")

            # --- Update Parameters ---
            logger.info("Updating parameters for recalculation...")
            new_params = [
                p * 1.1 for p in hamiltonian_params_val
            ]  # Example: Increase params by 10%
            calculator.update_hamiltonian_params(new_params)

            # --- Recalculate and Save Dispersion (Updated Params) ---
            logger.info("Calculating dispersion (Updated Params)...")
            dispersion_energies_updated = calculator.calculate_dispersion(
                q_points_array
            )
            if dispersion_energies_updated is not None:
                calculator.save_results(
                    disp_filename_updated,
                    {
                        "q_values_path": q_path_distances,
                        "energies_list": dispersion_energies_updated,
                        "q_labels": q_special_labels,
                        "q_ticks_positions": q_special_positions,
                    },
                )
                if SHOW_PLOTS_AFTER_CALC:
                    plot_dispersion_from_data(
                        q_path_distances,
                        dispersion_energies_updated,
                        title="Dispersion (Updated Params)",
                        q_labels=q_special_labels,
                        q_ticks_positions=q_special_positions,
                    )
            else:
                logger.error("Updated dispersion calculation failed.")

            # --- Calculate and Save S(q,w) (Using Updated Params) ---
            logger.info("Calculating S(q,w) (Updated Params)...")
            sqw_results = calculator.calculate_sqw(q_points_array)
            q_vectors_out_sqw, energies_sqw, intensities_sqw = sqw_results
            if (
                q_vectors_out_sqw is not None  # q_vectors_out_sqw is a tuple of arrays
                and energies_sqw is not None
                and intensities_sqw is not None
            ):
                # For S(q,w) plotting, we typically use the same q_path_indices if q_vectors_out_sqw matches q_points_array
                calculator.save_results(
                    sqw_filename_updated,
                    {  # Assuming q_vectors_out_sqw corresponds to q_path_distances
                        "q_values_path": q_path_distances,
                        "energies_list": energies_sqw,  # This is a tuple of arrays
                        "intensities_list": intensities_sqw,  # This is a tuple of arrays
                        "q_labels": q_special_labels,
                        "q_ticks_positions": q_special_positions,
                    },
                )
                if SHOW_PLOTS_AFTER_CALC:
                    plot_sqw_from_data(
                        q_path_distances,
                        list(energies_sqw),
                        list(intensities_sqw),
                        title="S(q,w) Map (Updated Params)",
                        energy_max=(
                            np.nanmax(np.hstack(energies_sqw))
                            if energies_sqw and any(e is not None for e in energies_sqw)
                            else 10
                        ),
                        q_labels=q_special_labels,
                        q_ticks_positions=q_special_positions,
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
        ) as e:
            logger.error(
                f"Calculation failed during setup or execution: {e}", exc_info=True
            )
        except Exception as e:
            logger.exception("An unexpected error occurred during calculations.")

    # --- Plotting from saved files ---
    if PLOT_FROM_DISP_FILE:
        logger.info(f"Attempting to plot dispersion from file: {disp_filename_updated}")
        try:
            data = np.load(disp_filename_updated, allow_pickle=True)
            plot_dispersion_from_data(
                data["q_values_path"],
                list(data["energies_list"]),
                title="Dispersion (Loaded from File)",
                q_labels=(
                    data["q_labels"].tolist() if "q_labels" in data else None
                ),  # Convert back to list if saved as array
                q_ticks_positions=(
                    data["q_ticks_positions"].tolist()
                    if "q_ticks_positions" in data
                    else None
                ),
            )
        except FileNotFoundError:
            logger.error(f"Dispersion data file not found: {disp_filename_updated}")
        except Exception as e:
            logger.exception(f"Error plotting dispersion from file: {e}")

    if PLOT_FROM_SQW_FILE:
        logger.info(f"Attempting to plot S(q,w) from file: {sqw_filename_updated}")
        try:
            data = np.load(sqw_filename_updated, allow_pickle=True)
            energies_list_sqw = list(
                data["energies_list"]
            )  # Convert tuple of arrays to list of arrays
            intensities_list_sqw = list(data["intensities_list"])
            plot_sqw_from_data(
                data["q_values_path"],
                energies_list_sqw,
                intensities_list_sqw,
                title="S(q,w) Map (Loaded from File)",  # Ensure energies_list_sqw is not empty before hstack
                energy_max=(
                    np.nanmax(np.hstack(energies_list_sqw))
                    if energies_list_sqw
                    and any(e is not None for e in energies_list_sqw)
                    else 10
                ),
                q_labels=data["q_labels"].tolist() if "q_labels" in data else None,
                q_ticks_positions=(
                    data["q_ticks_positions"].tolist()
                    if "q_ticks_positions" in data
                    else None
                ),
            )
        except FileNotFoundError:
            logger.error(f"S(q,w) data file not found: {sqw_filename_updated}")
        except Exception as e:
            logger.exception(f"Error plotting S(q,w) from file: {e}")

    logger.info("Example calculation finished.")

def _normal_order_terms(args: Tuple[sp.Expr, List[sp.Symbol], List[sp.Symbol], int]) -> sp.Expr:
    """
    Normal order quadratic boson terms in a Hamiltonian expression.
    
    Transforms terms like c_k * c_k_dagger into c_k_dagger * c_k + 1.
    Keeps terms like c_k_dagger * c_k, c_k * c_minus_k, c_k_dagger * c_minus_k_dagger as is.
    Assumes only quadratic terms are present.
    
    Args:
        args: Tuple containing:
            expr_terms (sp.Expr): A sum of terms to process.
            ck_ops: List of ck operators.
            ckd_ops: List of ckd operators.
            nspins: Number of spins.
            
    Returns:
        sp.Expr: The normal ordered expression.
    """
    expr, ck_ops, ckd_ops, nspins = args
    
    # Map operator names to objects for fast lookup
    ck_map = {op.name: (i, 'c') for i, op in enumerate(ck_ops)}
    ckd_map = {op.name: (i, 'cd') for i, op in enumerate(ckd_ops)}
    cmk_map = {f"cmk{i}": (i, 'cm') for i in range(nspins)} # Assuming naming convention
    cmkd_map = {f"cmkd{i}": (i, 'cmd') for i in range(nspins)}
    
    # Combined map
    op_map = {**ck_map, **ckd_map, **cmk_map, **cmkd_map}
    
    terms = expr.as_ordered_terms()
    new_terms = []
    
    for term in terms:
        coeff, ops = term.as_coeff_Mul()
        
        # Identify operators in the term
        # This part assumes simple structure: coeff * op1 * op2 or coeff * op1
        # Complex terms might need rigorous parsing, but usually it's product of 2 non-commuting symbols
        
        non_commuting_factors = term.atoms(sp.Symbol)
        non_commuting_factors = [s for s in non_commuting_factors if not s.is_commutative]
        
        # Sort by position in term is tricky in SymPy as Mul flattens. 
        # But we know the generic F.T. produces specific pairs.
        # Let's rely on nc_parts if available or manual extraction.
        
        nc_part = term / coeff
        
        if not non_commuting_factors:
            new_terms.append(term)
            continue
            
        if len(non_commuting_factors) != 2:
            # Handle Single operator terms or constant terms if any (shouldn't be for LSWT usually)
            new_terms.append(term)
            continue
            
        # We need to find the order. 
        # SymPy Mul args are ordered, but non-commutative multiplication order is preserved in args.
        args_nc = nc_part.args
        if not args_nc: # It's a single symbol
             new_terms.append(term)
             continue
             
        # Extract the two operators in order
        op1 = args_nc[0]
        op2 = args_nc[1]
        
        # Handle powers (e.g. ck**2) - uncommon for distinct indices but possible for same index
        if len(args_nc) == 1 and args_nc[0].is_Pow:
             # e.g. ck0**2
             base, exp = args_nc[0].as_base_exp()
             if exp == 2:
                 op1 = base
                 op2 = base
             else:
                 # This shouldn't happen for quadratic Hamiltonian
                 new_terms.append(term)
                 continue
        elif len(args_nc) > 2:
            # This shouldn't happen for quadratic Hamiltonian
             new_terms.append(term)
             continue

        if op1.name not in op_map or op2.name not in op_map:
             new_terms.append(term)
             continue
             
        idx1, type1 = op_map[op1.name]
        idx2, type2 = op_map[op2.name]
        
        # Commutation Rules:
        # [c_i, cd_j] = delta_ij
        # [cmd_i, cm_j] = delta_ij
        # Others commute or are already normal ordered.
        
        # Case 1: c * cd -> cd * c + delta
        if type1 == 'c' and type2 == 'cd':
            # Swap
            new_term = coeff * op2 * op1
            if idx1 == idx2:
                new_term += coeff # +1 * coeff
            new_terms.append(new_term)
            
        # Case 2: cmd * cm -> cm * cmd + delta
        # Wait, standard normal order is creation left. 
        # cm is annihilation (-k), cmd is creation (-k).
        # So we want cmd * cm.
        # Input might be cm * cmd? No, boson commutators are [c, cd]=1.
        # Operators are c_k, c_-k. 
        # c_-k is an annihilation operator. c_-k^dagger is creation.
        # In code: cmk is c_-k, cmkd is c_-k^dagger.
        # Goal: cmkd on left.
        # If we have cmk * cmkd -> cmkd * cmk + 1
        elif type1 == 'cm' and type2 == 'cmd':
             # Swap
            new_term = coeff * op2 * op1
            if idx1 == idx2:
                new_term += coeff
            new_terms.append(new_term)
            
        # Case 3: c * cm -> cm * c (commuting) - just prefer one order?
        # Usually we don't care about order between c and cm, provided distinct modes.
        # But for Matrix extraction we need consistent basis.
        # Basis is (c_k, c_-k^dagger).
        # Actually H2 is defined as: H = 0.5 * Psi^dagger * H2 * Psi
        # Psi = (c_k, c_-k^dagger)^T
        # Psi^dagger = (c_k^dagger, c_-k)
        # Block structure:
        # A  B
        # B* A*
        # A terms: c_k^dagger * c_k AND c_-k * c_-k^dagger -> c_-k * c_-k^dagger needs reordering?
        # Wait, c_-k * c_-k^dagger = c_-k^dagger * c_-k + 1. 
        # A term comes from: c_k^dagger * A * c_k + c_-k * A * c_-k^dagger
        # = c_k^dagger * A * c_k + c_-k^dagger * A * c_-k + A
        # So yes, we need normal ordering.
        
        else:
            # Already normal ordered or commuting pairs without delta
            new_terms.append(term)
            
    return Add(*new_terms)


def _build_TwogH2_matrix(
    hamiltonian_normal_ordered: sp.Expr, 
    nspins: int,
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
) -> sp.Matrix:
    """
    Extract the dynamical matrix H2 directly from normal ordered Hamiltonian.
    
    H = 0.5 * Psi^dagger * (2gH2) * Psi
    Wait, standard definition:
    H = 0.5 * X^dagger * H_mat * X
    X = [c_1...c_N, c_1_dag...c_N_dag]^T  <-- This is Colpa/standard usage?
    
    In this code's conventions (based on existing logic):
    basis_vector_dagger = ckd_ops + cmk_ops  (Creation parts relative to field?)
    basis_vector = ck_ops + cmkd_ops
    
    Actually, let's look at `_extract_h2_matrix` placeholders logic.
    p0 corresponds to basis_vector_dagger[i] * basis_vector[j]
    
    Basis used in this code seems to be:
    Psi = [ck_1...ck_N, cmkd_1...cmkd_N]^T
    Psi^dagger = [ckd_1...ckd_N, cmk_1...cmk_N]
    
    Terms in Hamiltonian are of form: Psi^dagger_i * M_ij * Psi_j
    
    So we need to map operators to indices in Psi.
    ckd_i -> row i (0 to N-1)
    cmk_i -> row i+N (N to 2N-1)
    
    ck_j -> col j (0 to N-1)
    cmkd_j -> col j+N (N to 2N-1)
    
    """
    
    # Map operator string to row/col index
    # Row operators (from Psi dagger)
    row_map = {}
    for i, op in enumerate(ckd_ops):
        row_map[op.name] = i
    for i, op in enumerate(cmk_ops):
        row_map[op.name] = i + nspins
        
    # Col operators (from Psi)
    col_map = {}
    for i, op in enumerate(ck_ops):
        col_map[op.name] = i
    for i, op in enumerate(cmkd_ops):
        col_map[op.name] = i + nspins
    
    n_dim = 2 * nspins
    H2 = sp.zeros(n_dim, n_dim)
    
    # Parse terms
    # We iterate over terms. Each term should be coeff * RowOp * ColOp
    # We add coeff to H2[row, col]
    
    terms = hamiltonian_normal_ordered.as_ordered_terms()
    
    # DEBUG: Track dropped terms
    dropped_log = []
    
    for term in terms:
        coeff, ops = term.as_coeff_Mul()
        nc_part = term / coeff
        args_nc = nc_part.args
        
        if not args_nc and nc_part.is_Symbol:
             args_nc = [nc_part]
        
    # Map operators to indices
    # Psi = [ck, cmkd]^T
    # Psi.dag = [ckd, cmk]
    # Row index i corresponds to Psi.dag[i]
    # Col index j corresponds to Psi[j]
    
    row_map = {} # Maps op1 to row index
    col_map = {} # Maps op2 to col index
    
    for i in range(nspins):
        # Rows: Psi.dag
        row_map[ckd_ops[i]] = i          # c_k^dagger -> 0..N-1
        row_map[cmk_ops[i]] = i + nspins # c_-k -> N..2N-1
        
        # Cols: Psi
        col_map[ck_ops[i]] = i           # c_k -> 0..N-1
        col_map[cmkd_ops[i]] = i + nspins# c_-k^dagger -> N..2N-1

    # DEBUG: Check maps
    # if len(row_map) > 0:
    #     logger.info(f"DEBUG: row_map sample: {list(row_map.items())[:2]}")
    #     logger.info(f"DEBUG: col_map sample: {list(col_map.items())[:2]}")
    
    dropped_count = 0
    # Build H2 matrix with numerical indices
    H2_matrix = sp.zeros(n_dim, n_dim)
    
    terms_norm = hamiltonian_normal_ordered.as_ordered_terms()
    # logger.info(f"DEBUG: hamiltonian_normal_ordered terms count: {len(terms_norm)}")
    
    added_count = 0
    
    for term in terms_norm:
        # Extract coefficient and operators
        coeff, ops = term.as_coeff_Mul()
        
        # Ensure it's a quadratic term with two operators
        if ops.is_Mul:
             args_ops = ops.args
        elif ops.is_Pow: # op^2
             args_ops = (ops,)
        elif ops.is_Symbol: # single op (should not occur for quadratic H)
             args_ops = (ops,)
        else:
             args_ops = (ops,)
             
        # Filter for non-commutative operators, similar to FT fix
        nc_ops = [o for o in args_ops if not o.is_commutative]
        
        if len(nc_ops) == 1 and nc_ops[0].is_Pow:
             base, exp = nc_ops[0].as_base_exp()
             if exp == 2:
                 op1 = base
                 op2 = base
             else:
                 op1 = None; op2 = None
        elif len(nc_ops) == 2:
             op1 = nc_ops[0]
             op2 = nc_ops[1]
        else:
             op1 = None; op2 = None
             
        if op1 is None or op2 is None:
            # Maybe it's just a number (coeff)?
            if len(nc_ops) == 0:
                 continue # Ignore constant terms
            dropped_log.append(f"Structure mismatch: {term}")
            continue
            
        # Capture commutative factors (e.g. sin(kx), phases)
        # ops contains both commutative and non-commutative parts
        # We need to extract the part that is NOT op1 or op2 (or op1*op2)
        # Robust way: collect all commutative args
        comm_ops = [o for o in args_ops if o.is_commutative]
        comm_factor = sp.Mul(*comm_ops)
        
        # Total value to add
        term_val = coeff * comm_factor

        # Map operators to matrix indices using DISTINCT maps
        idx1 = row_map.get(op1)
        idx2 = col_map.get(op2)

        if idx1 is not None and idx2 is not None:
             H2_matrix[idx1, idx2] += term_val
        else:
             # Check if operators are flipped (e.g. ck * ckd instead of ckd * ck)
             # Since they commute (if different indices), we can swap them.
             # If same index, normal ordering should have handled it, but let's check.
             idx1_rev = row_map.get(op2)
             idx2_rev = col_map.get(op1)
             
             if idx1_rev is not None and idx2_rev is not None:
                 H2_matrix[idx1_rev, idx2_rev] += term_val
             else:
                 dropped_log.append(f"Unparseable ({op1},{op2}): {term}")
                 dropped_count += 1
                 continue

        # Add to matrix (Moved inside if/else)
        # H2_matrix[idx1, idx2] += coeff
        
    if dropped_count > 0:
        logger.warning(f"Construct TwogH2: Dropped {dropped_count} terms that did not match expected structure. First few: {dropped_log[:5]}")
    
    # Calculate TwogH2 = 2 * g * H2
    g_list = [1] * nspins + [-1] * nspins
    TwogH2 = sp.zeros(n_dim, n_dim)
    for i in range(n_dim):
        for j in range(n_dim):
            val = 2 * g_list[i] * H2_matrix[i, j]
            TwogH2[i, j] = val
                
    return TwogH2

def _fourier_transform_terms(
    args: Tuple[sp.Expr, Dict[Tuple[str, str], sp.Expr]],
) -> sp.Expr:
    """
    Apply Fourier Transform substitutions using dictionary lookup.
    
    Parses quadratic terms of form (coeff * Op1 * Op2) and replaces (Op1 * Op2)
    with the corresponding k-space expression found in `ft_lookup`.
    """
    expr, ft_lookup = args
             
    terms = expr.as_ordered_terms()
    new_terms = []
    
    for term in terms:
        coeff, rest = term.as_coeff_Mul()
        
        # Identify operators (non-commutative)
        if rest.is_Mul:
            args_list = rest.args
        else:
            args_list = (rest,)
            
        nc_args = [a for a in args_list if not a.is_commutative]
        c_args = [a for a in args_list if a.is_commutative]
        
        effective_coeff = coeff * sp.Mul(*c_args)
        
        op1 = None
        op2 = None
        
        if len(nc_args) == 2:
            op1 = nc_args[0]
            op2 = nc_args[1]
        elif len(nc_args) == 1 and nc_args[0].is_Pow:
             base, exp = nc_args[0].as_base_exp()
             if exp == 2:
                 op1 = base
                 op2 = base
        
        if op1 and op2:
            key = (op1.name, op2.name)
            if key in ft_lookup:
                new_expr = ft_lookup[key]
                new_terms.append(effective_coeff * new_expr)
            else:
                new_terms.append(term)
        else:
             new_terms.append(term)
             
    return sp.Add(*new_terms)


def _generate_fourier_lookup(
    spin_model_module,
    k_sym: List[sp.Symbol],
    nspins: int,
    nspins_ouc: int,
    c_ops: List[sp.Symbol],
    cd_ops: List[sp.Symbol],
    ck_ops: List[sp.Symbol],
    ckd_ops: List[sp.Symbol],
    cmk_ops: List[sp.Symbol],
    cmkd_ops: List[sp.Symbol],
    params_sym: List[sp.Symbol],
) -> Dict[Tuple[str, str], sp.Expr]:
    """
    Generate dictionary for Fourier Transform substitutions.
    
    Returns:
        Dict[(str, str), sp.Expr]: Map from (OpName1, OpName2) to k-space expression.
    """
    atom_positions_uc = spin_model_module.atom_pos()
    atom_positions_ouc = spin_model_module.atom_pos_ouc()
    interaction_matrix = spin_model_module.spin_interactions(params_sym)[0]

    ft_lookup = {}

    for i in range(nspins):
        for j in range(nspins_ouc):
            if interaction_matrix[i, j] == 0:
                continue

            disp_vec = atom_positions_uc[i, :] - atom_positions_ouc[j, :]
            k_dot_dr = sum(k * dr for k, dr in zip(k_sym, disp_vec))
            
            # Using Rewrite(sin) as in original code
            exp_plus_ikdr = sp.exp(I * k_dot_dr).rewrite(sp.sin)
            exp_minus_ikdr = sp.exp(-I * k_dot_dr).rewrite(sp.sin)

            j_uc = j % nspins

            # cd_i * cd_j
            val1 = 1/2 * (
                ckd_ops[i] * cmkd_ops[j_uc] * exp_minus_ikdr
                + cmkd_ops[i] * ckd_ops[j_uc] * exp_plus_ikdr
            )
            ft_lookup[(cd_ops[i].name, cd_ops[j].name)] = val1
            
            # c_i * c_j
            val2 = 1/2 * (
                ck_ops[i] * cmk_ops[j_uc] * exp_plus_ikdr
                + cmk_ops[i] * ck_ops[j_uc] * exp_minus_ikdr
            )
            ft_lookup[(c_ops[i].name, c_ops[j].name)] = val2
            
            # cd_i * c_j
            val3 = 1/2 * (
                ckd_ops[i] * ck_ops[j_uc] * exp_minus_ikdr
                + cmkd_ops[i] * cmk_ops[j_uc] * exp_plus_ikdr
            )
            ft_lookup[(cd_ops[i].name, c_ops[j].name)] = val3
            
            # c_i * cd_j
            val4 = 1/2 * (
                ck_ops[i] * ckd_ops[j_uc] * exp_plus_ikdr
                + cmk_ops[i] * cmkd_ops[j_uc] * exp_minus_ikdr
            )
            ft_lookup[(c_ops[i].name, cd_ops[j].name)] = val4
            
    # Add the diagonal term substitution (present in original code, seems important)
    for j in range(nspins_ouc):
        j_uc = j % nspins  # Map OUC index j to UC index
        # cd_j * c_j -> 0.5 * (ckd_j * ck_j + cmkd_j * cmk_j)
        val_diag = 1 / 2 * (ckd_ops[j_uc] * ck_ops[j_uc] + cmkd_ops[j_uc] * cmk_ops[j_uc])
        ft_lookup[(cd_ops[j].name, c_ops[j].name)] = val_diag

            
    return ft_lookup
