#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 13:56:34 2018

@author: Kit Matan and Pharit Piyawongwatthana

Python code to calculate spin-wave dispersion and scattering intensity
using Linear Spin Wave Theory (LSWT).
Translated from Mathematica and Octave codes written by Taku J Sato.

This script relies on an external `spin_model.py` file to define the
specific magnetic Hamiltonian, atomic structure, and ground state spin
configuration (via rotation matrices).
"""
# this file contains spin model; see example
# edit spin_model.py
import spin_model as sm
import sympy as sp
from sympy import I
from sympy import lambdify
from sympy import Add
import numpy as np
from scipy import linalg as la
import timeit
import sys
import pickle
from multiprocessing import Pool
from tqdm import tqdm
import logging  # Import logging module

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


# Ensure the pckFiles directory exists
import os

if not os.path.exists("pckFiles"):
    logger.info("Creating directory 'pckFiles' for caching.")
    os.makedirs("pckFiles")

# --- Numerical Constants ---
DEGENERACY_THRESHOLD: float = 1e-12
ZERO_MATRIX_ELEMENT_THRESHOLD: float = 1e-6
EIGENVECTOR_MATCHING_THRESHOLD: float = 1e-5
ENERGY_IMAG_PART_THRESHOLD: float = 1e-5
SQW_IMAG_PART_THRESHOLD: float = 1e-4
Q_ZERO_THRESHOLD: float = 1e-10
PROJECTION_CHECK_TOLERANCE: float = 1e-5
# --- End Numerical Constants ---


def substitute_expr(
    args: Tuple[sp.Expr, Union[Dict, List[Tuple[sp.Expr, sp.Expr]]]],
) -> sp.Expr:
    """Helper function for multiprocessing substitution."""
    expr, subs_dict = args
    result: sp.Expr = expr.subs(subs_dict, simultaneous=True)
    return result


def gen_HM(
    k_sym: List[sp.Symbol], S_sym: sp.Symbol, params_sym: List[sp.Symbol]
) -> Tuple[sp.Matrix, sp.Matrix]:
    """Generates symbolic TwogH2 and Ud matrices."""
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
    hamiltonian_sym = (
        hamiltonian_S0.coeff(params_sym[-1]) * params_sym[-1]
        + hamiltonian_sym.coeff(S_sym, 1) * S_sym
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
    pool_args_comm = [(expr, commutation_substitutions) for expr in hamiltonian_k_terms]
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

    hamiltonian_k_comm_terms: List[sp.Expr] = hamiltonian_k_commuted.as_ordered_terms()
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


def gram_schmidt(x: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
    """Performs Gram-Schmidt orthogonalization using QR decomposition."""
    q, r = np.linalg.qr(x, mode="reduced")
    return q


def check_degeneracy(
    q_vector: npt.NDArray[np.float_],
    index: int,
    degeneracy_count: int,
    projection_magnitudes: npt.NDArray[np.float_],
    eigenvalues_neg_q: npt.NDArray[np.complex_],
    projected_eigenvectors: npt.NDArray[np.complex_],
) -> NoReturn:
    """Logs critical error and exits if degeneracy handling fails (original logic)."""
    logger.critical("Mismatch in degeneracy handling (check_degeneracy).")
    logger.critical(f"  q = {q_vector}")
    logger.critical(
        f"  Eigenvalue indices involved: {index - degeneracy_count} to {index}"
    )
    logger.critical(f"  Degeneracy count (ndeg + 1): {degeneracy_count + 1}")
    logger.critical(
        f"  Projection results (abs): {projection_magnitudes} (Threshold: {PROJECTION_CHECK_TOLERANCE})"
    )
    logger.critical(
        f"  Eigenvalues in range: {np.real(eigenvalues_neg_q[index - degeneracy_count : index + 1])}"
    )
    logger.critical(
        f"  Number of vectors found by projection: {projected_eigenvectors.shape[1]}"
    )
    logger.critical("The program will exit...")
    sys.exit()


# --- Helper Functions for KKdMatrix Refactoring (Phase 2, Step 1) ---


def _diagonalize_and_sort(
    Hmat: npt.NDArray[np.complex_], nspins: int, q_vector_label: str
) -> Tuple[Optional[npt.NDArray[np.complex_]], Optional[npt.NDArray[np.complex_]]]:
    """
    Diagonalizes the Hamiltonian matrix Hmat = g * H2 and sorts eigenvalues/vectors.

    Sorting places positive energy branches first, then negative energy branches
    sorted by magnitude.

    Args:
        Hmat (npt.NDArray[np.complex_]): The matrix to diagonalize (2gH).
        nspins (int): Number of spins in the unit cell.
        q_vector_label (str): Label for logging (e.g., "+q" or "-q").

    Returns:
        Tuple[Optional[npt.NDArray[np.complex_]], Optional[npt.NDArray[np.complex_]]]:
            Sorted eigenvalues and eigenvectors, or (None, None) on failure.
    """
    try:
        eigvals, eigvecs = la.eig(Hmat)
    except np.linalg.LinAlgError as e:
        logger.error(f"Eigenvalue calculation failed for {q_vector_label}: {e}")
        return None, None

    # Sorting logic (same as original KKdMatrix)
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
    Applies Gram-Schmidt orthogonalization to degenerate subspaces using QR decomposition.

    Iterates through sorted eigenvalues and identifies blocks where consecutive
    eigenvalues are closer than `degeneracy_threshold`. Applies QR decomposition
    via the `gram_schmidt` helper to the corresponding eigenvector columns within
    each block to ensure they form an orthonormal basis for that subspace.

    Args:
        eigenvalues (npt.NDArray[np.complex_]): Sorted eigenvalues.
        eigenvectors (npt.NDArray[np.complex_]): Corresponding sorted eigenvectors (columns).
        degeneracy_threshold (float): Threshold to consider eigenvalues degenerate.
        q_vector_label (str): Label for logging (e.g., "+q" or "-q").

    Returns:
        npt.NDArray[np.complex_]: Eigenvectors with degenerate subspaces orthogonalized.
    """
    nspins2 = eigenvectors.shape[0]  # Dimension of the matrix (2 * nspins)
    orthonormalized_eigenvectors = eigenvectors.copy()  # Operate on a copy
    degeneracy_count: int = 0  # Tracks the size of the current degenerate block - 1

    # Iterate through eigenvalues to find blocks of degenerate values
    for i in range(1, nspins2):
        # Check if the current eigenvalue is close to the previous one
        if abs(eigenvalues[i] - eigenvalues[i - 1]) < degeneracy_threshold:
            degeneracy_count += 1  # Extend the current degenerate block
        elif degeneracy_count > 0:
            # End of a degenerate block detected (current eigenvalue is different)
            start_idx = i - degeneracy_count - 1  # Start index of the block
            end_idx = i  # End index (exclusive) of the block
            logger.debug(
                f"Applying Gram-Schmidt to block [{start_idx}:{end_idx}] for {q_vector_label}"
            )
            vec_block = orthonormalized_eigenvectors[:, start_idx:end_idx]

            # Orthonormalize the columns within the degenerate block
            orthonormal_vecs = gram_schmidt(vec_block)  # Uses np.linalg.qr

            # Check if QR decomposition found rank deficiency
            if orthonormal_vecs.shape[1] == vec_block.shape[1]:
                # Full rank: Replace the block with the orthonormalized vectors
                orthonormalized_eigenvectors[:, start_idx:end_idx] = orthonormal_vecs
            else:
                # Rank deficient: Log a warning, replace with the basis found, zero out the rest
                logger.warning(
                    f"Rank deficiency detected during GS for {q_vector_label} at index {i}. "
                    f"Original block size: {vec_block.shape[1]}, "
                    f"Orthonormal basis size: {orthonormal_vecs.shape[1]}"
                )
                # Place the found orthonormal basis vectors
                orthonormalized_eigenvectors[
                    :, start_idx : start_idx + orthonormal_vecs.shape[1]
                ] = orthonormal_vecs
                # Zero out the remaining columns in the block where QR found linear dependence
                orthonormalized_eigenvectors[
                    :, start_idx + orthonormal_vecs.shape[1] : end_idx
                ] = 0
            degeneracy_count = 0  # Reset counter for the next block

    # Handle potential degeneracy extending to the very last eigenvalue
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
                f"Rank deficiency detected during GS for {q_vector_label} at end of array. "
                f"Original block size: {vec_block.shape[1]}, "
                f"Orthonormal basis size: {orthonormal_vecs.shape[1]}"
            )
            orthonormalized_eigenvectors[
                :, start_idx : start_idx + orthonormal_vecs.shape[1]
            ] = orthonormal_vecs
            orthonormalized_eigenvectors[
                :, start_idx + orthonormal_vecs.shape[1] : end_idx
            ] = 0

    return orthonormalized_eigenvectors


def _calculate_alpha_matrix(
    eigenvectors: npt.NDArray[np.complex_],
    G_metric: npt.NDArray[np.float_],
    zero_threshold: float,
    q_vector_label: str,
) -> Optional[npt.NDArray[np.complex_]]:
    """
    Calculates the diagonal alpha normalization matrix.

    Ensures T^-1 G (T^-1)^dagger = G, where T^-1 = eigenvectors @ alpha.

    Args:
        eigenvectors (npt.NDArray[np.complex_]): Orthonormalized eigenvectors (columns).
        G_metric (npt.NDArray[np.float_]): The metric tensor diag(1,..1,-1..-1).
        zero_threshold (float): Threshold for setting small alpha values to zero.
        q_vector_label (str): Label for logging (e.g., "+q" or "-q").

    Returns:
        Optional[npt.NDArray[np.complex_]]: Diagonal alpha matrix, or None on failure.
    """
    try:
        inv_eigenvectors: npt.NDArray[np.complex_] = la.inv(eigenvectors)
        nspins2 = eigenvectors.shape[0]
        alpha_sq_diag: npt.NDArray[np.float_] = np.zeros(nspins2, dtype=float)

        # alpha_ii^2 = | <U_i | G | U_i> | where U_i are rows of V^-1.
        for i in range(nspins2):
            row_i = inv_eigenvectors[i, :]
            alpha_sq_diag[i] = np.real(np.dot(np.conj(row_i), G_metric @ row_i))

        # Ensure positivity and take square root
        alpha_sq_diag[alpha_sq_diag < 0] = 0
        alpha_diag: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag)
        alpha_diag[np.abs(alpha_diag) < zero_threshold] = 0  # Apply zero threshold
        alpha_matrix: npt.NDArray[np.complex_] = np.diag(alpha_diag).astype(np.complex_)
        return alpha_matrix

    except np.linalg.LinAlgError:
        logger.error(
            f"Matrix inversion failed for {q_vector_label}. Eigenvector matrix might be singular."
        )
        return None


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
    Matches -q eigenvectors to +q, reorders, and adjusts phases using dot product.

    Matches columns of conj(swap(eigvecs_m_ortho)) [source] to columns of
    conj(swap(eigvecs_p_ortho)) [target] based on maximum projection.
    Reorders eigvecs_m_ortho, eigvals_m_sorted, alpha_m_sorted according to the match.
    Adjusts the phase of the reordered alpha_m using the phase of the dot product
    between the original +q vector and the matched conj(swap(-q)) vector.

    Args:
        eigvecs_p_ortho: Orthonormalized eigenvectors for +q.
        alpha_p: Alpha matrix for +q.
        eigvecs_m_ortho: Orthonormalized eigenvectors for -q (initial sorting).
        eigvals_m_sorted: Eigenvalues for -q (initial sorting).
        alpha_m_sorted: Alpha matrix for -q (initial sorting).
        nspins: Number of spins.
        match_tol: Tolerance for eigenvector matching projection (|projection|^2 > 1-match_tol^2).
        zero_tol: Tolerance for zero checks (vector norms, dot products).
        q_vector_label: Label for logging (e.g., "q=[0,0,0]").

    Returns:
        Tuple containing the reordered/phase-adjusted:
        - eigenvectors_minus_q_final
        - eigenvalues_minus_q_reordered
        - alpha_matrix_minus_q_final
    """
    nspins2 = 2 * nspins

    # Prepare swapped+conjugated +q vectors (target vectors for matching)
    eigenvectors_plus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack(
            (
                eigvecs_p_ortho[nspins:nspins2, :],  # Lower block
                eigvecs_p_ortho[0:nspins, :],  # Upper block
            )
        )
    )
    # Prepare swapped+conjugated -q vectors (source vectors for matching)
    eigenvectors_minus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack(
            (
                eigvecs_m_ortho[nspins:nspins2, :],  # Lower block
                eigvecs_m_ortho[0:nspins, :],  # Upper block
            )
        )
    )

    # Initialize arrays for reordered results
    eigenvectors_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigvecs_m_ortho, dtype=complex
    )
    eigenvalues_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigvals_m_sorted, dtype=complex
    )
    alpha_matrix_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        alpha_m_sorted, dtype=complex
    )

    matched_indices_m: set[int] = (
        set()
    )  # Keep track of which source vectors (j) are used
    num_matched_vectors: int = 0

    # Match each target vector (i, derived from +q) to a source vector (j, derived from -q)
    for i in range(nspins2):
        best_match_j: int = -1
        max_proj_metric: float = -1.0
        # Target vector 'i' (column i of eigenvectors_plus_q_swapped_conj)
        vec_i_target: npt.NDArray[np.complex_] = eigenvectors_plus_q_swapped_conj[:, i]
        vec_i_norm_sq: float = np.real(np.dot(np.conj(vec_i_target), vec_i_target))

        if vec_i_norm_sq < zero_tol**2:
            logger.warning(
                f"Target vector {i} has near-zero norm at {q_vector_label}. Skipping match."
            )
            continue

        # Compare target 'i' to all unmatched source vectors 'j'
        for j in range(nspins2):
            if j in matched_indices_m:
                continue  # Skip if source 'j' already matched

            # Source vector 'j' (column j of eigenvectors_minus_q_swapped_conj)
            vec_j_source: npt.NDArray[np.complex_] = eigenvectors_minus_q_swapped_conj[
                :, j
            ]
            vec_j_norm_sq: float = np.real(np.dot(np.conj(vec_j_source), vec_j_source))
            if vec_j_norm_sq < zero_tol**2:
                continue  # Skip zero-norm source vectors

            # Calculate normalized projection squared |<target_i|source_j>|^2 / (|target_i|^2 * |source_j|^2)
            projection: complex = np.dot(np.conj(vec_i_target), vec_j_source)
            projection_mag_sq: float = np.abs(projection) ** 2
            # Avoid division by zero just in case norms were exactly zero (though skipped above)
            norm_product_sq = vec_i_norm_sq * vec_j_norm_sq
            if norm_product_sq < zero_tol**4:
                continue
            normalized_projection_mag_sq: float = projection_mag_sq / norm_product_sq

            # Check if this is the best match found so far for target 'i'
            if (
                normalized_projection_mag_sq > max_proj_metric
                and normalized_projection_mag_sq > (1.0 - match_tol**2)
            ):
                max_proj_metric = normalized_projection_mag_sq
                best_match_j = j

        # Process the best match found for target 'i'
        if best_match_j != -1:
            matched_indices_m.add(best_match_j)
            num_matched_vectors += 1

            # Determine the original +q index ('orig_i') corresponding to the target index 'i'
            # This is the reverse of the swap operation used to create the target vectors
            orig_i = i + nspins if i < nspins else i - nspins

            # The target index 'i' becomes the column index in the reordered -q matrices
            target_index = i

            # Assign the matched eigenvector/value from the original -q results
            eigenvectors_minus_q_reordered[:, target_index] = eigvecs_m_ortho[
                :, best_match_j
            ]
            eigenvalues_minus_q_reordered[target_index] = eigvals_m_sorted[best_match_j]

            # --- Robust Phase Calculation ---
            # We need the phase relating the original +q vector (eigvecs_p_ortho[:, orig_i])
            # and the matched source vector (eigenvectors_minus_q_swapped_conj[:, best_match_j]).
            vec_i_plus_q_orig = eigvecs_p_ortho[:, orig_i]
            vec_j_minus_q_source = eigenvectors_minus_q_swapped_conj[
                :, best_match_j
            ]  # Already computed above

            # Calculate the dot product between them
            dot_product: complex = np.dot(
                np.conj(vec_j_minus_q_source), vec_i_plus_q_orig
            )
            dot_product_mag: float = np.abs(dot_product)

            phase_factor: complex
            if dot_product_mag < zero_tol:
                # If dot product is zero (vectors are orthogonal or one is zero), phase is undefined/arbitrary.
                # Setting phase_factor to 1.0 is a neutral choice.
                logger.warning(
                    f"Near-zero dot product ({dot_product_mag:.2e}) during phase calculation for match ({i} -> {best_match_j}) at {q_vector_label}. Setting phase factor to 1."
                )
                phase_factor = 1.0 + 0.0j
            else:
                # Phase factor is the phase of the dot product
                phase_factor = dot_product / dot_product_mag

            # Apply the phase adjustment rule: alpha_m_reordered[i] ~ conj(alpha_p[orig_i] * phase_factor)
            # This ensures the Bogoliubov transformation matrices T(+q) and T(-q) have the correct relative phase.
            alpha_matrix_minus_q_reordered[target_index, target_index] = np.conj(
                alpha_p[orig_i, orig_i] * phase_factor
            )
            # --- End Robust Phase Calculation ---

        else:
            # No suitable match found for target vector 'i'
            logger.warning(
                f"No matching eigenvector found for target vector index {i} at {q_vector_label}"
            )

    # Final checks and cleanup
    if num_matched_vectors != nspins2:
        logger.warning(
            f"Number of matched vectors ({num_matched_vectors}) does not equal {nspins2} at {q_vector_label}"
        )

    # Clean up near-zero elements in the final alpha matrix
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
    Calculates the K and Kd transformation matrices.

    Args:
        Ud_numeric: Numerical rotation matrix.
        spin_magnitude: Numerical spin value S.
        nspins: Number of spins.
        inv_T_p: Inverse Bogoliubov matrix T^-1(+q) = V(+q) @ alpha(+q).
        inv_T_m_reordered: Reordered inverse Bogoliubov matrix T^-1(-q) = V(-q) @ alpha(-q).
        zero_threshold: Threshold for cleaning up small elements.

    Returns:
        Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_]]: K_matrix, Kd_matrix.
    """
    # Mapping from local boson operators (c, c_dag) to local spin operators (Sx, Sy)
    Udd_local_boson_map: npt.NDArray[np.complex_] = np.zeros(
        (3 * nspins, 2 * nspins), dtype=complex
    )
    for i in range(nspins):
        Udd_local_boson_map[3 * i, i] = 1.0
        Udd_local_boson_map[3 * i, i + nspins] = 1.0
        Udd_local_boson_map[3 * i + 1, i] = 1.0 / I
        Udd_local_boson_map[3 * i + 1, i + nspins] = -1.0 / I

    # Combine transformations: S_global = Ud * sqrt(S/2) * Udd * T^-1 * Y
    prefactor: float = np.sqrt(spin_magnitude / 2.0)
    K_matrix: npt.NDArray[np.complex_] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_p
    )
    Kd_matrix: npt.NDArray[np.complex_] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_T_m_reordered
    )

    # Clean up small numerical noise
    K_matrix[np.abs(K_matrix) < zero_threshold] = 0
    Kd_matrix[np.abs(Kd_matrix) < zero_threshold] = 0

    return K_matrix, Kd_matrix


# --- Main KKdMatrix Function (Refactored) ---


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
    Calculates transformation matrices K, Kd and eigenvalues for LSWT by orchestrating helper functions.

    Performs Bogoliubov diagonalization for +q and -q, handles sorting, degeneracy,
    normalization, eigenvector matching, and finally computes K and Kd.

    Args:
        spin_magnitude (float): Numerical value of the spin magnitude S.
        Hmat_plus_q: Numerical TwogH2 matrix evaluated at +q.
        Hmat_minus_q: Numerical TwogH2 matrix evaluated at -q.
        Ud_numeric: Numerical block rotation matrix Ud.
        q_vector: Momentum vector q = [qx, qy, qz].
        nspins: Number of spins in the magnetic unit cell.

    Returns:
        Tuple containing K_matrix, Kd_matrix, and eigenvalues_plus_q_sorted.
        Returns matrices/arrays of NaNs if any intermediate step fails.
    """
    q_label = f"q={q_vector}"  # Label for logging
    nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
    nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
    G_metric = np.diag(np.concatenate([np.ones(nspins), -np.ones(nspins)]))

    # --- Process +q ---
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

    # --- Process -q ---
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

    # --- Match +q and -q results ---
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

    # --- Calculate K and Kd ---
    # Calculate T^-1 = V @ alpha
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


# --- Remaining Functions (process_matrix, process_calc_Sqw, etc.) ---
# These functions remain largely the same, but now call the refactored KKdMatrix


def process_matrix(
    cache_mode: str,
    k_sym: List[sp.Symbol],
    S_sym: sp.Symbol,
    params_sym: List[sp.Symbol],
    cache_file_base: str,
) -> Tuple[sp.Matrix, sp.Matrix]:
    """Manages loading/saving symbolic matrices HMat and Ud."""
    hm_cache_file: str = os.path.join("pckFiles", cache_file_base + "_HM.pck")
    ud_cache_file: str = os.path.join("pckFiles", cache_file_base + "_Ud.pck")
    HMat: sp.Matrix
    Ud: sp.Matrix

    if cache_mode == "w":
        logger.info("Generating symbolic matrices (HMat, Ud)...")
        HMat, Ud = gen_HM(k_sym, S_sym, params_sym)
        logger.info(f"Writing HMat to {hm_cache_file}")
        try:
            with open(hm_cache_file, "wb") as outHM:
                pickle.dump(HMat, outHM)
        except IOError as e:
            logger.error(f"Error writing HMat cache file: {e}")
            raise
        logger.info(f"Writing Ud to {ud_cache_file}")
        try:
            with open(ud_cache_file, "wb") as outUd:
                pickle.dump(Ud, outUd)
        except IOError as e:
            logger.error(f"Error writing Ud cache file: {e}")
            raise

    elif cache_mode == "r":
        logger.info(
            f"Importing symbolic matrices from cache files ({hm_cache_file}, {ud_cache_file})..."
        )
        try:
            with open(hm_cache_file, "rb") as inHM:
                HMat = pickle.load(inHM)
            with open(ud_cache_file, "rb") as inUd:
                Ud = pickle.load(inUd)
        except FileNotFoundError:
            logger.error(
                f"Cache files not found. Run with 'w' option first or check filename '{cache_file_base}'."
            )
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(
                f"Error loading cache files (may be corrupted or incompatible): {e}"
            )
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred loading cache files: {e}")
            raise
    else:
        raise ValueError(f"Invalid mode '{cache_mode}'. Use 'r' (read) or 'w' (write).")

    return HMat, Ud


def process_calc_Sqw(
    args: Tuple[
        sp.Matrix,
        npt.NDArray[np.complex_],
        List[sp.Symbol],
        npt.NDArray[np.float_],
        int,
        float,
    ],
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Worker function for multiprocessing calculation of S(q,w) for a single q-point."""
    HMat_sym, Ud_numeric, k_sym, q_vector, nspins, spin_magnitude_num = args
    sqw_complex_accumulator: npt.NDArray[np.complex_] = np.zeros(nspins, dtype=complex)
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)
    nan_intensities: npt.NDArray[np.float_] = np.full((nspins,), np.nan)

    try:
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        logger.error(f"Error during lambdify at q={q_vector}: {e}")
        return q_vector, nan_energies, nan_intensities

    try:
        Hmat_plus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vector[0], q_vector[1], q_vector[2]), dtype=np.complex128
        )
        Hmat_minus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(-q_vector[0], -q_vector[1], -q_vector[2]), dtype=np.complex128
        )
    except Exception as e:
        logger.error(f"Error evaluating HMat function at q={q_vector}: {e}")
        return q_vector, nan_energies, nan_intensities

    # Call the refactored KKdMatrix orchestrator
    K_matrix: npt.NDArray[np.complex_]
    Kd_matrix: npt.NDArray[np.complex_]
    eigenvalues: npt.NDArray[np.complex_]
    K_matrix, Kd_matrix, eigenvalues = KKdMatrix(
        spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_vector, nspins
    )

    if (
        np.isnan(K_matrix).any()
        or np.isnan(Kd_matrix).any()
        or np.isnan(eigenvalues).any()
    ):
        # Warning already logged within KKdMatrix or its helpers if specific step failed
        # logger.warning(f"NaN encountered in KKdMatrix result for q={q_vector}. Skipping intensity calculation.")
        return q_vector, nan_energies, nan_intensities

    imag_energy_mag: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues[0:nspins]))
    if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
        logger.warning(
            f"Significant imaginary part in energy eigenvalues for q={q_vector}. Max imag: {np.max(imag_energy_mag)}"
        )
    energies: npt.NDArray[np.float_] = np.real(eigenvalues[0:nspins])
    q_output: npt.NDArray[np.float_] = q_vector

    # Intensity calculation loop (remains the same)
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
                f"Significant imaginary part in Sqw for q={q_vector}, mode {mode_index}: {np.imag(intensity_one_mode)}"
            )

        sqw_complex_accumulator[mode_index] = intensity_one_mode

    intensities: npt.NDArray[np.float_] = np.real(sqw_complex_accumulator)
    intensities[intensities < 0] = 0

    return q_output, energies, intensities


def calc_Sqw(
    spin_magnitude: float,
    q_vectors: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    hamiltonian_params: Union[List[float], npt.NDArray[np.float_]],
    cache_file_base: str,
    cache_mode: str,
) -> Tuple[
    Optional[Tuple[npt.NDArray[np.float_], ...]],
    Optional[Tuple[npt.NDArray[np.float_], ...]],
    Optional[Tuple[npt.NDArray[np.float_], ...]],
]:
    """Calculates the dynamical structure factor S(q,w) over a list of q-points."""
    logger.info("Calculating scattering intensity S(q,w)...")
    nspins: int
    try:
        nspins = len(sm.atom_pos())
        if nspins == 0:
            raise ValueError("spin_model.atom_pos() returned an empty list.")
    except AttributeError:
        raise AttributeError("Function 'atom_pos' not found in spin_model.py.")
    except Exception as e:
        raise RuntimeError(f"Error getting nspins from spin_model.atom_pos(): {e}")

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    k_sym: List[sp.Symbol] = [kx, ky, kz]
    S_sym: sp.Symbol = sp.Symbol("S", real=True)
    params_sym: List[sp.Symbol] = sp.symbols(
        "p0:%d" % len(hamiltonian_params), real=True
    )
    HMat_sym: sp.Matrix
    Ud_sym: sp.Matrix

    try:
        HMat_sym, Ud_sym = process_matrix(
            cache_mode, k_sym, S_sym, params_sym, cache_file_base
        )
    except (ValueError, FileNotFoundError, Exception) as e:
        logger.error(f"Failed to get symbolic matrices: {e}")
        return None, None, None

    param_substitutions: List[Tuple[sp.Symbol, float]] = [
        (S_sym, spin_magnitude)
    ] + list(zip(params_sym, hamiltonian_params))
    HMat_num_sym: sp.Matrix
    Ud_num_sym: sp.Matrix
    Ud_numeric: npt.NDArray[np.complex_]
    try:
        HMat_num_sym = HMat_sym.subs(param_substitutions, simultaneous=True).evalf()
        Ud_num_sym = Ud_sym.subs(param_substitutions, simultaneous=True).evalf()
        Ud_numeric = np.array(Ud_num_sym, dtype=np.complex128)
    except Exception as e:
        logger.error(f"Error during substitution of numerical parameters: {e}")
        return None, None, None

    logger.info("Running diagonalization and intensity calculation...")
    start_time: float = timeit.default_timer()

    pool_args: List[Tuple] = [
        (HMat_num_sym, Ud_numeric, k_sym, q_vec, nspins, spin_magnitude)
        for q_vec in q_vectors
    ]
    results: List[
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]
    ] = []
    try:
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(process_calc_Sqw, pool_args),
                    total=len(q_vectors),
                    desc="Calculating S(q,w)",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
    except Exception as e:
        logger.error(f"Error during multiprocessing execution: {e}")
        return None, None, None

    q_vectors_out: Tuple[npt.NDArray[np.float_], ...]
    energies_out: Tuple[npt.NDArray[np.float_], ...]
    intensities_out: Tuple[npt.NDArray[np.float_], ...]
    try:
        if not results or len(results[0]) != 3:
            raise ValueError("Multiprocessing returned empty or malformed results.")
        q_vectors_out, energies_out, intensities_out = zip(*results)
    except ValueError as e:
        logger.error(f"Error unpacking results from parallel processing: {e}")
        return None, None, None

    end_time: float = timeit.default_timer()
    logger.info(
        f"Run-time for S(q,w) calculation: {np.round((end_time - start_time) / 60, 2)} min."
    )

    return q_vectors_out, energies_out, intensities_out


def process_calc_disp(
    args: Tuple[sp.Matrix, List[sp.Symbol], npt.NDArray[np.float_], int],
) -> npt.NDArray[np.float_]:
    """Worker function for multiprocessing calculation of dispersion for a single q-point."""
    HMat_sym, k_sym, q_vector, nspins = args
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)

    try:
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        logger.error(f"Error during lambdify at q={q_vector}: {e}")
        return nan_energies

    try:
        HMat_numeric: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vector[0], q_vector[1], q_vector[2]), dtype=np.complex128
        )
    except Exception as e:
        logger.error(f"Error evaluating HMat function at q={q_vector}: {e}")
        return nan_energies

    eigenvalues: npt.NDArray[np.complex_]
    try:
        eigenvalues = la.eigvals(HMat_numeric)
    except np.linalg.LinAlgError:
        logger.error(f"Eigenvalue calculation failed for q={q_vector}.")
        return nan_energies

    imag_part_mags: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues))
    if np.any(imag_part_mags > ENERGY_IMAG_PART_THRESHOLD):
        logger.warning(
            f"Significant imaginary part in eigenvalues for q={q_vector}. Max imag: {np.max(imag_part_mags)}"
        )

    energies: npt.NDArray[np.float_]
    try:
        eigenvalues_sorted_real: npt.NDArray[np.float_] = np.real(np.sort(eigenvalues))
        energies = eigenvalues_sorted_real[nspins:]
        if len(energies) != nspins:
            logger.warning(
                f"Unexpected number of positive energies ({len(energies)}) found for q={q_vector}. Expected {nspins}."
            )
            if len(energies) > nspins:
                energies = energies[:nspins]
            else:
                energies = np.pad(
                    energies, (0, nspins - len(energies)), constant_values=np.nan
                )
    except Exception as e:
        logger.error(f"Error during eigenvalue sorting/selection for q={q_vector}: {e}")
        return nan_energies

    return energies


def calc_disp(
    spin_magnitude: float,
    q_vectors: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    hamiltonian_params: Union[List[float], npt.NDArray[np.float_]],
    cache_file_base: str,
    cache_mode: str,
) -> Optional[List[npt.NDArray[np.float_]]]:
    """Calculates the spin-wave dispersion relation over a list of q-points."""
    logger.info("Calculating magnon dispersion ...")
    nspins: int
    try:
        nspins = len(sm.atom_pos())
        if nspins == 0:
            raise ValueError("spin_model.atom_pos() returned an empty list.")
    except AttributeError:
        raise AttributeError("Function 'atom_pos' not found in spin_model.py.")
    except Exception as e:
        raise RuntimeError(f"Error getting nspins from spin_model.atom_pos(): {e}")

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    k_sym: List[sp.Symbol] = [kx, ky, kz]
    S_sym: sp.Symbol = sp.Symbol("S", real=True)
    params_sym: List[sp.Symbol] = sp.symbols(
        "p0:%d" % len(hamiltonian_params), real=True
    )
    HMat_sym: sp.Matrix

    try:
        HMat_sym, _ = process_matrix(
            cache_mode, k_sym, S_sym, params_sym, cache_file_base
        )
    except (ValueError, FileNotFoundError, Exception) as e:
        logger.error(f"Failed to get symbolic matrix HMat: {e}")
        return None

    param_substitutions: List[Tuple[sp.Symbol, float]] = [
        (S_sym, spin_magnitude)
    ] + list(zip(params_sym, hamiltonian_params))
    HMat_num_sym: sp.Matrix
    try:
        HMat_num_sym = HMat_sym.subs(param_substitutions, simultaneous=True).evalf()
    except Exception as e:
        logger.error(f"Error during substitution of numerical parameters: {e}")
        return None

    logger.info("Running diagonalization...")
    start_time: float = timeit.default_timer()

    pool_args: List[Tuple] = [
        (HMat_num_sym, k_sym, q_vec, nspins) for q_vec in q_vectors
    ]
    energies_list: List[npt.NDArray[np.float_]] = []
    try:
        with Pool() as pool:
            energies_list = list(
                tqdm(
                    pool.imap(process_calc_disp, pool_args),
                    total=len(q_vectors),
                    desc="Calculating Dispersion",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
    except Exception as e:
        logger.error(f"Error during multiprocessing execution: {e}")

    end_time: float = timeit.default_timer()
    logger.info(
        f"Run-time for dispersion calculation: {np.round((end_time - start_time) / 60, 2)} min."
    )

    return energies_list


# Example usage
if __name__ == "__main__":
    # --- User Configuration ---
    spin_S_val: float = 1.0
    hamiltonian_params_val: List[float] = [1.0, 0.1]
    cache_file_base_name: str = "my_model_cache"
    cache_operation_mode: str = "r"

    q_points_list: List[List[float]] = []
    N_points_per_segment = 20
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

    logger.info("Starting example calculation...")

    # --- Calculate dispersion ---
    dispersion_energies: Optional[List[npt.NDArray[np.float_]]] = calc_disp(
        spin_S_val,
        q_points_array,
        hamiltonian_params_val,
        cache_file_base_name,
        cache_operation_mode,
    )

    if dispersion_energies is not None:
        print("\n--- Dispersion Energies ---")
        for i, q_vec in enumerate(q_points_array):
            if (
                i < len(dispersion_energies)
                and isinstance(dispersion_energies[i], np.ndarray)
                and not np.isnan(dispersion_energies[i]).any()
            ):
                if i < 5 or i >= len(q_points_array) - 5:
                    print(
                        f"q = {np.round(q_vec, 3)}: E = {np.round(dispersion_energies[i], 4)}"
                    )
                elif i == 5:
                    print("...")
            else:
                logger.warning(
                    f"Dispersion calculation failed for q = {np.round(q_vec, 3)}"
                )
    else:
        logger.error("Dispersion calculation failed to start.")

    # --- Calculate S(q,w) ---
    sqw_cache_mode = cache_operation_mode
    logger.info(f"Calculating S(q,w) using cache mode: '{sqw_cache_mode}'")

    sqw_results: Tuple[
        Optional[Tuple[npt.NDArray[np.float_], ...]],
        Optional[Tuple[npt.NDArray[np.float_], ...]],
        Optional[Tuple[npt.NDArray[np.float_], ...]],
    ]
    sqw_results = calc_Sqw(
        spin_S_val,
        q_points_array,
        hamiltonian_params_val,
        cache_file_base_name,
        sqw_cache_mode,
    )
    q_vectors_out, energies_sqw, intensities_sqw = sqw_results

    if (
        q_vectors_out is not None
        and energies_sqw is not None
        and intensities_sqw is not None
    ):
        print("\n--- Scattering Intensities ---")
        for i, q_vec in enumerate(q_vectors_out):
            if (
                i < len(energies_sqw)
                and isinstance(energies_sqw[i], np.ndarray)
                and not np.isnan(energies_sqw[i]).any()
                and i < len(intensities_sqw)
                and isinstance(intensities_sqw[i], np.ndarray)
                and not np.isnan(intensities_sqw[i]).any()
            ):
                if i < 5 or i >= len(q_vectors_out) - 5:
                    print(
                        f"q = {np.round(q_vec, 3)}: E = {np.round(energies_sqw[i], 4)}, S(q,w) = {np.round(intensities_sqw[i], 4)}"
                    )
                elif i == 5:
                    print("...")
            else:
                logger.warning(
                    f"S(q,w) calculation failed for q = {np.round(q_vec, 3)}"
                )
    else:
        logger.error("S(q,w) calculation failed to start.")

    logger.info("Example calculation finished.")
