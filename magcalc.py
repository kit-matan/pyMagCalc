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
# Configure logging to output INFO level messages and above to the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)  # Get a logger instance for this module
# --- End Logging Setup ---


# Ensure the pckFiles directory exists
import os

if not os.path.exists("pckFiles"):
    logger.info("Creating directory 'pckFiles' for caching.")
    os.makedirs("pckFiles")

# --- Numerical Constants ---
# Tolerances used in calculations, especially KKdMatrix and result checking.
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
    """
    Helper function for multiprocessing substitution. Applies a substitution
    dictionary to a SymPy expression.

    Args:
        args (tuple): Contains expr (sympy.Expr) and subs_dict (dict or list).

    Returns:
        sympy.Expr: The expression after substitutions.
    """
    expr, subs_dict = args
    result: sp.Expr = expr.subs(subs_dict, simultaneous=True)
    return result


def gen_HM(
    k_sym: List[sp.Symbol], S_sym: sp.Symbol, params_sym: List[sp.Symbol]
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Generates the symbolic quadratic Hamiltonian matrix (TwogH2) and the
    spin rotation matrix (Ud) in momentum space using Linear Spin Wave Theory.

    Args:
        k_sym (List[sp.Symbol]): Momentum symbols ([kx, ky, kz]).
        S_sym (sp.Symbol): Symbolic spin magnitude.
        params_sym (List[sp.Symbol]): Symbolic Hamiltonian parameters.

    Returns:
        Tuple[sp.Matrix, sp.Matrix]: TwogH2 (dynamical matrix) and Ud (rotation matrix).

    (Detailed description omitted for brevity - see previous version)
    """
    atom_positions_uc: npt.NDArray[np.float_] = sm.atom_pos()
    nspins: int = len(atom_positions_uc)
    atom_positions_ouc: npt.NDArray[np.float_] = (
        sm.atom_pos_ouc()
    )  # Positions including neighbours
    nspins_ouc: int = len(atom_positions_ouc)
    logger.info(f"Number of spins in the unit cell: {nspins}")  # Use logger

    # generate boson spin operators in local coordinate system
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

    # rotate spin operators to global coordinates
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]] = sm.mpr(params_sym)
    spin_ops_global_ouc: List[sp.Matrix] = [
        rotation_matrices[j] * spin_ops_local[nspins * i + j]
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]

    # generate the spin Hamiltonian
    hamiltonian_sym: sp.Expr = sm.Hamiltonian(spin_ops_global_ouc, params_sym)
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    hamiltonian_S0: sp.Expr = hamiltonian_sym.coeff(S_sym, 0)
    hamiltonian_sym = (
        hamiltonian_S0.coeff(params_sym[-1]) * params_sym[-1]
        + hamiltonian_sym.coeff(S_sym, 1) * S_sym
        + hamiltonian_sym.coeff(S_sym, 2) * S_sym**2
    )
    hamiltonian_sym = sp.expand(hamiltonian_sym)

    # Define momentum-space boson operators
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

    # Generate dictionary for Fourier transform substitution
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

    # Apply commutation relations
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

    # Basis vectors for quadratic form
    basis_vector_dagger: List[sp.Symbol] = ckd_ops[:nspins] + cmk_ops[:nspins]
    basis_vector: List[sp.Symbol] = ck_ops[:nspins] + cmkd_ops[:nspins]

    # Commutative placeholders
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

    # Perform substitutions
    logger.info("Running symbolic substitutions...")  # Use logger
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
    )  # Use logger

    # Extract coefficients to form H2 matrix
    H2_elements: sp.Matrix = sp.Matrix(
        [hamiltonian_with_placeholders.coeff(p) for p in placeholder_symbols]
    )
    H2_matrix: sp.Matrix = sp.Matrix(2 * nspins, 2 * nspins, H2_elements)

    # Metric tensor g
    g_metric_tensor_sym: sp.Matrix = sp.diag(*([1] * nspins + [-1] * nspins))

    # Matrix to be diagonalized: TwogH2 = 2 * g * H2
    dynamical_matrix_TwogH2: sp.Matrix = 2 * g_metric_tensor_sym * H2_matrix

    # Rotation matrix Ud (block diagonal)
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
    """
    Performs Gram-Schmidt orthogonalization on the columns of a matrix using QR decomposition.

    Args:
        x (npt.NDArray[np.complex_]): Input matrix.

    Returns:
        npt.NDArray[np.complex_]: Matrix with orthonormal columns.
    """
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
    """
    Helper function to print diagnostic information and exit if an inconsistency
    is found during degeneracy handling in KKdMatrix (original implementation).

    Args:
        q_vector (npt.NDArray[np.float_]): Momentum vector.
        index (int): Loop index where issue occurred.
        degeneracy_count (int): Detected degeneracy count.
        projection_magnitudes (npt.NDArray[np.float_]): Absolute values of projections.
        eigenvalues_neg_q (npt.NDArray[np.complex_]): Eigenvalues being processed.
        projected_eigenvectors (npt.NDArray[np.complex_]): Subset of eigenvectors identified.

    Raises:
        SystemExit: Terminates the program execution.
    """
    # Log critical error before exiting
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
    Calculates transformation matrices K, Kd and eigenvalues for LSWT.

    Args:
        spin_magnitude (float): Numerical value of the spin magnitude.
        Hmat_plus_q (npt.NDArray[np.complex_]): Numerical TwogH2 matrix at +q.
        Hmat_minus_q (npt.NDArray[np.complex_]): Numerical TwogH2 matrix at -q.
        Ud_numeric (npt.NDArray[np.complex_]): Numerical block rotation matrix.
        q_vector (npt.NDArray[np.float_]): Momentum vector q.
        nspins (int): Number of spins in the magnetic unit cell.

    Returns:
        Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
            A tuple containing K_matrix, Kd_matrix, and eigenvalues_plus_q_sorted.

    (Detailed description omitted for brevity - see previous version)
    """
    dEdeg: float = DEGENERACY_THRESHOLD
    zero_tol: float = ZERO_MATRIX_ELEMENT_THRESHOLD
    match_tol: float = EIGENVECTOR_MATCHING_THRESHOLD

    G_metric_tensor: npt.NDArray[np.float_] = np.diag(
        np.concatenate([np.ones(nspins), -np.ones(nspins)])
    )

    # --- Diagonalization for +q ---
    try:
        eigvals_p: npt.NDArray[np.complex_]
        eigvecs_p: npt.NDArray[np.complex_]
        eigvals_p, eigvecs_p = la.eig(Hmat_plus_q)
    except np.linalg.LinAlgError as e:
        logger.error(
            f"Eigenvalue calculation failed for +q = {q_vector}: {e}"
        )  # Use logger
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Sorting eigenvalues and eigenvectors
    sort_indices_p: npt.NDArray[np.int_] = eigvals_p.argsort()
    eigvecs_p_tmp1: npt.NDArray[np.complex_] = eigvecs_p[:, sort_indices_p][
        :, nspins : 2 * nspins
    ]
    eigvals_p_tmp1: npt.NDArray[np.complex_] = eigvals_p[sort_indices_p][
        nspins : 2 * nspins
    ]
    eigvecs_p_tmp2: npt.NDArray[np.complex_] = eigvecs_p[:, sort_indices_p][:, 0:nspins]
    eigvals_p_tmp2: npt.NDArray[np.complex_] = eigvals_p[sort_indices_p][0:nspins]
    sort_indices_p_neg: npt.NDArray[np.int_] = (np.abs(eigvals_p_tmp2)).argsort()
    eigvecs_p_tmp3: npt.NDArray[np.complex_] = eigvecs_p_tmp2[:, sort_indices_p_neg]
    eigvals_p_tmp3: npt.NDArray[np.complex_] = eigvals_p_tmp2[sort_indices_p_neg]
    eigenvalues_plus_q_sorted: npt.NDArray[np.complex_] = np.concatenate(
        (eigvals_p_tmp1, eigvals_p_tmp3)
    )
    eigenvectors_plus_q_sorted: npt.NDArray[np.complex_] = np.hstack(
        (eigvecs_p_tmp1, eigvecs_p_tmp3)
    )

    # Gram-Schmidt [+q]
    degeneracy_count: int = 0
    for i in range(1, 2 * nspins):
        if abs(eigenvalues_plus_q_sorted[i] - eigenvalues_plus_q_sorted[i - 1]) < dEdeg:
            degeneracy_count += 1
        elif degeneracy_count > 0:
            vec_block: npt.NDArray[np.complex_] = eigenvectors_plus_q_sorted[
                :, i - degeneracy_count - 1 : i
            ]
            orthonormal_vecs: npt.NDArray[np.complex_] = gram_schmidt(vec_block)
            if orthonormal_vecs.shape[1] == vec_block.shape[1]:
                eigenvectors_plus_q_sorted[:, i - degeneracy_count - 1 : i] = (
                    orthonormal_vecs
                )
            else:
                logger.warning(
                    f"Rank deficiency detected during GS for +q at index {i}, q={q_vector}"
                )  # Use logger
                eigenvectors_plus_q_sorted[
                    :,
                    i
                    - degeneracy_count
                    - 1 : i
                    - degeneracy_count
                    - 1
                    + orthonormal_vecs.shape[1],
                ] = orthonormal_vecs
                eigenvectors_plus_q_sorted[
                    :, i - degeneracy_count - 1 + orthonormal_vecs.shape[1] : i
                ] = 0
            degeneracy_count = 0
    if degeneracy_count > 0:
        vec_block = eigenvectors_plus_q_sorted[
            :, 2 * nspins - 1 - degeneracy_count : 2 * nspins
        ]
        orthonormal_vecs = gram_schmidt(vec_block)
        if orthonormal_vecs.shape[1] == vec_block.shape[1]:
            eigenvectors_plus_q_sorted[
                :, 2 * nspins - 1 - degeneracy_count : 2 * nspins
            ] = orthonormal_vecs
        else:
            logger.warning(
                f"Rank deficiency detected during GS for +q at end of array, q={q_vector}"
            )  # Use logger
            eigenvectors_plus_q_sorted[
                :,
                2 * nspins
                - 1
                - degeneracy_count : 2 * nspins
                - 1
                - degeneracy_count
                + orthonormal_vecs.shape[1],
            ] = orthonormal_vecs
            eigenvectors_plus_q_sorted[
                :,
                2 * nspins
                - 1
                - degeneracy_count
                + orthonormal_vecs.shape[1] : 2 * nspins,
            ] = 0

    # Alpha matrix [+q]
    try:
        inv_eigenvectors_plus_q: npt.NDArray[np.complex_] = la.inv(
            eigenvectors_plus_q_sorted
        )
        alpha_sq_diag_p: npt.NDArray[np.float_] = np.zeros(2 * nspins, dtype=float)
        for i in range(2 * nspins):
            row_i = inv_eigenvectors_plus_q[i, :]
            alpha_sq_diag_p[i] = np.real(
                np.dot(np.conj(row_i), G_metric_tensor @ row_i)
            )
        alpha_sq_diag_p[alpha_sq_diag_p < 0] = 0
        alpha_diag_p: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag_p)
        alpha_diag_p[np.abs(alpha_diag_p) < zero_tol] = 0
        alpha_matrix_plus_q: npt.NDArray[np.complex_] = np.diag(alpha_diag_p).astype(
            np.complex_
        )
    except np.linalg.LinAlgError:
        logger.error(
            f"Matrix inversion failed for +q = {q_vector}. Eigenvector matrix might be singular."
        )  # Use logger
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    eigenvectors_plus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack(
            (
                eigenvectors_plus_q_sorted[nspins : 2 * nspins, :],
                eigenvectors_plus_q_sorted[0:nspins, :],
            )
        )
    )

    # --- Diagonalization for -q ---
    try:
        eigvals_m: npt.NDArray[np.complex_]
        eigvecs_m: npt.NDArray[np.complex_]
        eigvals_m, eigvecs_m = la.eig(Hmat_minus_q)
    except np.linalg.LinAlgError as e:
        logger.error(
            f"Eigenvalue calculation failed for -q = {-q_vector}: {e}"
        )  # Use logger
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Sorting [-q]
    sort_indices_m: npt.NDArray[np.int_] = eigvals_m.argsort()
    eigvecs_m_tmp1: npt.NDArray[np.complex_] = eigvecs_m[:, sort_indices_m][
        :, nspins : 2 * nspins
    ]
    eigvals_m_tmp1: npt.NDArray[np.complex_] = eigvals_m[sort_indices_m][
        nspins : 2 * nspins
    ]
    eigvecs_m_tmp2: npt.NDArray[np.complex_] = eigvecs_m[:, sort_indices_m][:, 0:nspins]
    eigvals_m_tmp2: npt.NDArray[np.complex_] = eigvals_m[sort_indices_m][0:nspins]
    sort_indices_m_neg: npt.NDArray[np.int_] = (abs(eigvals_m_tmp2)).argsort()
    eigvecs_m_tmp3: npt.NDArray[np.complex_] = eigvecs_m_tmp2[:, sort_indices_m_neg]
    eigvals_m_tmp3: npt.NDArray[np.complex_] = eigvals_m_tmp2[sort_indices_m_neg]
    eigenvalues_minus_q_sorted: npt.NDArray[np.complex_] = np.concatenate(
        (eigvals_m_tmp1, eigvals_m_tmp3)
    )
    eigenvectors_minus_q_sorted: npt.NDArray[np.complex_] = np.hstack(
        (eigvecs_m_tmp1, eigvecs_m_tmp3)
    )

    # Gram-Schmidt [-q]
    degeneracy_count = 0
    for i in range(1, 2 * nspins):
        if (
            abs(eigenvalues_minus_q_sorted[i] - eigenvalues_minus_q_sorted[i - 1])
            < dEdeg
        ):
            degeneracy_count += 1
        elif degeneracy_count > 0:
            vec_block = eigenvectors_minus_q_sorted[:, i - degeneracy_count - 1 : i]
            orthonormal_vecs = gram_schmidt(vec_block)
            if orthonormal_vecs.shape[1] == vec_block.shape[1]:
                eigenvectors_minus_q_sorted[:, i - degeneracy_count - 1 : i] = (
                    orthonormal_vecs
                )
            else:
                logger.warning(
                    f"Rank deficiency detected during GS for -q at index {i}, q={q_vector}"
                )  # Use logger
                eigenvectors_minus_q_sorted[
                    :,
                    i
                    - degeneracy_count
                    - 1 : i
                    - degeneracy_count
                    - 1
                    + orthonormal_vecs.shape[1],
                ] = orthonormal_vecs
                eigenvectors_minus_q_sorted[
                    :, i - degeneracy_count - 1 + orthonormal_vecs.shape[1] : i
                ] = 0
            degeneracy_count = 0
    if degeneracy_count > 0:
        vec_block = eigenvectors_minus_q_sorted[
            :, 2 * nspins - 1 - degeneracy_count : 2 * nspins
        ]
        orthonormal_vecs = gram_schmidt(vec_block)
        if orthonormal_vecs.shape[1] == vec_block.shape[1]:
            eigenvectors_minus_q_sorted[
                :, 2 * nspins - 1 - degeneracy_count : 2 * nspins
            ] = orthonormal_vecs
        else:
            logger.warning(
                f"Rank deficiency detected during GS for -q at end of array, q={q_vector}"
            )  # Use logger
            eigenvectors_minus_q_sorted[
                :,
                2 * nspins
                - 1
                - degeneracy_count : 2 * nspins
                - 1
                - degeneracy_count
                + orthonormal_vecs.shape[1],
            ] = orthonormal_vecs
            eigenvectors_minus_q_sorted[
                :,
                2 * nspins
                - 1
                - degeneracy_count
                + orthonormal_vecs.shape[1] : 2 * nspins,
            ] = 0

    # Alpha matrix [-q]
    try:
        inv_eigenvectors_minus_q: npt.NDArray[np.complex_] = la.inv(
            eigenvectors_minus_q_sorted
        )
        alpha_sq_diag_m: npt.NDArray[np.float_] = np.zeros(2 * nspins, dtype=float)
        for i in range(2 * nspins):
            alpha_sq_diag_m[i] = np.real(
                np.dot(
                    np.conj(inv_eigenvectors_minus_q[i, :]),
                    G_metric_tensor @ inv_eigenvectors_minus_q[i, :],
                )
            )
        alpha_sq_diag_m[alpha_sq_diag_m < 0] = 0
        alpha_diag_m: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag_m)
        alpha_diag_m[np.abs(alpha_diag_m) < zero_tol] = 0
        alpha_matrix_minus_q: npt.NDArray[np.complex_] = np.diag(alpha_diag_m).astype(
            np.complex_
        )
    except np.linalg.LinAlgError:
        logger.error(
            f"Matrix inversion failed for -q = {-q_vector}. Eigenvector matrix might be singular."
        )  # Use logger
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # --- Eigenvector Matching between +q and -q ---
    eigenvectors_minus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack(
            (
                eigenvectors_minus_q_sorted[nspins : 2 * nspins, :],
                eigenvectors_minus_q_sorted[0:nspins, :],
            )
        )
    )
    eigenvectors_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigenvectors_minus_q_sorted, dtype=complex
    )
    eigenvalues_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigenvalues_minus_q_sorted, dtype=complex
    )
    alpha_matrix_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        alpha_matrix_minus_q, dtype=complex
    )

    matched_indices_m: set[int] = set()
    num_matched_vectors: int = 0

    for i in range(2 * nspins):
        best_match_j: int = -1
        max_proj_metric: float = -1.0
        vec_i_plus_q: npt.NDArray[np.complex_] = eigenvectors_plus_q_sorted[:, i]
        vec_i_norm_sq: float = np.real(np.dot(np.conj(vec_i_plus_q), vec_i_plus_q))

        if vec_i_norm_sq < zero_tol**2:
            continue

        for j in range(2 * nspins):
            if j in matched_indices_m:
                continue

            vec_j_minus_q_swapped_conj: npt.NDArray[np.complex_] = (
                eigenvectors_minus_q_swapped_conj[:, j]
            )
            vec_j_norm_sq: float = np.real(
                np.dot(np.conj(vec_j_minus_q_swapped_conj), vec_j_minus_q_swapped_conj)
            )
            if vec_j_norm_sq < zero_tol**2:
                continue

            projection: complex = np.dot(
                np.conj(vec_i_plus_q), vec_j_minus_q_swapped_conj
            )
            projection_mag_sq: float = np.abs(projection) ** 2
            normalized_projection_mag_sq: float = projection_mag_sq / (
                vec_i_norm_sq * vec_j_norm_sq
            )

            if (
                normalized_projection_mag_sq > max_proj_metric
                and normalized_projection_mag_sq > (1.0 - match_tol**2)
            ):
                max_proj_metric = normalized_projection_mag_sq
                best_match_j = j

        if best_match_j != -1:
            matched_indices_m.add(best_match_j)
            num_matched_vectors += 1
            target_index: int = -1
            if i < nspins:
                target_index = i + nspins
            else:
                target_index = i - nspins

            if target_index != -1:
                eigenvectors_minus_q_reordered[:, target_index] = (
                    eigenvectors_minus_q_sorted[:, best_match_j]
                )
                eigenvalues_minus_q_reordered[target_index] = (
                    eigenvalues_minus_q_sorted[best_match_j]
                )

                idx_nonzero_i: npt.NDArray[np.int_] = np.where(
                    np.abs(vec_i_plus_q) > zero_tol
                )[0]
                idx_nonzero_j: npt.NDArray[np.int_] = np.where(
                    np.abs(vec_j_minus_q_swapped_conj) > zero_tol
                )[0]

                if len(idx_nonzero_i) > 0 and len(idx_nonzero_j) > 0:
                    phase_factor: complex = (
                        vec_i_plus_q[idx_nonzero_i[0]]
                        / vec_j_minus_q_swapped_conj[idx_nonzero_j[0]]
                    )
                    alpha_matrix_minus_q_reordered[target_index, target_index] = (
                        np.conj(alpha_matrix_plus_q[i, i] * phase_factor)
                    )
                else:
                    alpha_matrix_minus_q_reordered[target_index, target_index] = 0
            else:
                logger.warning(
                    f"Invalid target index during matching for i={i}, q={q_vector}"
                )  # Use logger
        else:
            logger.warning(
                f"No matching eigenvector found for +q vector index {i} at q={q_vector}"
            )  # Use logger

    if num_matched_vectors != 2 * nspins:
        logger.warning(
            f"Number of matched vectors ({num_matched_vectors}) does not equal 2*nspins at q={q_vector}"
        )  # Use logger

    eigenvectors_minus_q_final = eigenvectors_minus_q_reordered
    alpha_matrix_minus_q_final = alpha_matrix_minus_q_reordered
    alpha_matrix_minus_q_final[np.abs(alpha_matrix_minus_q_final) < zero_tol] = 0

    # --- Calculate K and Kd matrices ---
    inv_bogoliubov_T_plus_q: npt.NDArray[np.complex_] = (
        eigenvectors_plus_q_sorted @ alpha_matrix_plus_q
    )
    inv_bogoliubov_T_minus_q_reordered: npt.NDArray[np.complex_] = (
        eigenvectors_minus_q_final @ alpha_matrix_minus_q_final
    )

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
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_bogoliubov_T_plus_q
    )
    Kd_matrix: npt.NDArray[np.complex_] = (
        prefactor
        * Ud_numeric
        @ Udd_local_boson_map
        @ inv_bogoliubov_T_minus_q_reordered
    )

    K_matrix[np.abs(K_matrix) < zero_tol] = 0
    Kd_matrix[np.abs(Kd_matrix) < zero_tol] = 0

    return K_matrix, Kd_matrix, eigenvalues_plus_q_sorted


def process_matrix(
    cache_mode: str,
    k_sym: List[sp.Symbol],
    S_sym: sp.Symbol,
    params_sym: List[sp.Symbol],
    cache_file_base: str,
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Manages the generation or loading of the symbolic Hamiltonian matrix (HMat=TwogH2)
    and rotation matrix (Ud) from cache files (.pck).

    Args:
        cache_mode (str): 'r' (read) or 'w' (write).
        k_sym (List[sp.Symbol]): Momentum symbols.
        S_sym (sp.Symbol): Symbolic spin magnitude.
        params_sym (List[sp.Symbol]): Symbolic Hamiltonian parameters.
        cache_file_base (str): Base filename for cache files.

    Returns:
        Tuple[sp.Matrix, sp.Matrix]: HMat (TwogH2) and Ud.

    (Detailed description omitted for brevity - see previous version)
    """
    hm_cache_file: str = os.path.join("pckFiles", cache_file_base + "_HM.pck")
    ud_cache_file: str = os.path.join("pckFiles", cache_file_base + "_Ud.pck")
    HMat: sp.Matrix
    Ud: sp.Matrix

    if cache_mode == "w":
        logger.info("Generating symbolic matrices (HMat, Ud)...")  # Use logger
        HMat, Ud = gen_HM(k_sym, S_sym, params_sym)
        logger.info(f"Writing HMat to {hm_cache_file}")  # Use logger
        try:
            with open(hm_cache_file, "wb") as outHM:
                pickle.dump(HMat, outHM)
        except IOError as e:
            logger.error(f"Error writing HMat cache file: {e}")
            raise  # Use logger
        logger.info(f"Writing Ud to {ud_cache_file}")  # Use logger
        try:
            with open(ud_cache_file, "wb") as outUd:
                pickle.dump(Ud, outUd)
        except IOError as e:
            logger.error(f"Error writing Ud cache file: {e}")
            raise  # Use logger

    elif cache_mode == "r":
        logger.info(
            f"Importing symbolic matrices from cache files ({hm_cache_file}, {ud_cache_file})..."
        )  # Use logger
        try:
            with open(hm_cache_file, "rb") as inHM:
                HMat = pickle.load(inHM)
            with open(ud_cache_file, "rb") as inUd:
                Ud = pickle.load(inUd)
        except FileNotFoundError:
            logger.error(
                f"Cache files not found. Run with 'w' option first or check filename '{cache_file_base}'."
            )  # Use logger
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            logger.error(
                f"Error loading cache files (may be corrupted or incompatible): {e}"
            )  # Use logger
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred loading cache files: {e}"
            )  # Use logger
            raise
    else:
        # Error handled by raising ValueError
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
    """
    Worker function for multiprocessing calculation of S(q,w) for a single q-point.

    Args:
        args (Tuple): Contains HMat_sym, Ud_numeric, k_sym, q_vector, nspins, spin_magnitude_num.

    Returns:
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
            q_vector, energies, intensities. Returns NaNs on failure.

    (Detailed description omitted for brevity - see previous version)
    """
    HMat_sym, Ud_numeric, k_sym, q_vector, nspins, spin_magnitude_num = args
    sqw_complex_accumulator: npt.NDArray[np.complex_] = np.zeros(nspins, dtype=complex)
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)
    nan_intensities: npt.NDArray[np.float_] = np.full((nspins,), np.nan)

    try:
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        logger.error(f"Error during lambdify at q={q_vector}: {e}")  # Use logger
        return q_vector, nan_energies, nan_intensities

    try:
        Hmat_plus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vector[0], q_vector[1], q_vector[2]), dtype=np.complex128
        )
        Hmat_minus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(-q_vector[0], -q_vector[1], -q_vector[2]), dtype=np.complex128
        )
    except Exception as e:
        logger.error(
            f"Error evaluating HMat function at q={q_vector}: {e}"
        )  # Use logger
        return q_vector, nan_energies, nan_intensities

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
        logger.warning(
            f"NaN encountered in KKdMatrix result for q={q_vector}. Skipping intensity calculation."
        )  # Use logger
        return q_vector, nan_energies, nan_intensities

    imag_energy_mag: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues[0:nspins]))
    if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
        logger.warning(
            f"Significant imaginary part in energy eigenvalues for q={q_vector}. Max imag: {np.max(imag_energy_mag)}"
        )  # Use logger
    energies: npt.NDArray[np.float_] = np.real(eigenvalues[0:nspins])
    q_output: npt.NDArray[np.float_] = q_vector

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
            )  # Use logger

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
    """
    Calculates the dynamical structure factor S(q,w) over a list of q-points.

    Args:
        spin_magnitude (float): Numerical value of the spin magnitude.
        q_vectors (Union[List, npt.NDArray]): List or array of momentum vectors q.
        hamiltonian_params (Union[List, npt.NDArray]): Numerical Hamiltonian parameter values.
        cache_file_base (str): Base filename for caching.
        cache_mode (str): Cache mode ('r' or 'w').

    Returns:
        Tuple[Optional[Tuple[...]], Optional[Tuple[...]], Optional[Tuple[...]]]:
            (q_vectors_out, energies_out, intensities_out) tuples, or (None, None, None).

    (Detailed description omitted for brevity - see previous version)
    """
    logger.info("Calculating scattering intensity S(q,w)...")  # Use logger
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
        logger.error(f"Failed to get symbolic matrices: {e}")  # Use logger
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
        logger.error(
            f"Error during substitution of numerical parameters: {e}"
        )  # Use logger
        return None, None, None

    logger.info("Running diagonalization and intensity calculation...")  # Use logger
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
        logger.error(f"Error during multiprocessing execution: {e}")  # Use logger
        return None, None, None

    q_vectors_out: Tuple[npt.NDArray[np.float_], ...]
    energies_out: Tuple[npt.NDArray[np.float_], ...]
    intensities_out: Tuple[npt.NDArray[np.float_], ...]
    try:
        if not results or len(results[0]) != 3:
            raise ValueError("Multiprocessing returned empty or malformed results.")
        q_vectors_out, energies_out, intensities_out = zip(*results)
    except ValueError as e:
        logger.error(
            f"Error unpacking results from parallel processing: {e}"
        )  # Use logger
        return None, None, None

    end_time: float = timeit.default_timer()
    logger.info(
        f"Run-time for S(q,w) calculation: {np.round((end_time - start_time) / 60, 2)} min."
    )  # Use logger

    return q_vectors_out, energies_out, intensities_out


def process_calc_disp(
    args: Tuple[sp.Matrix, List[sp.Symbol], npt.NDArray[np.float_], int],
) -> npt.NDArray[np.float_]:
    """
    Worker function for multiprocessing calculation of dispersion for a single q-point.

    Args:
        args (Tuple): Contains HMat_sym, k_sym, q_vector, nspins.

    Returns:
        npt.NDArray[np.float_]: Calculated magnon energies. Returns NaNs on failure.

    (Detailed description omitted for brevity - see previous version)
    """
    HMat_sym, k_sym, q_vector, nspins = args
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)

    try:
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        logger.error(f"Error during lambdify at q={q_vector}: {e}")  # Use logger
        return nan_energies

    try:
        HMat_numeric: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vector[0], q_vector[1], q_vector[2]), dtype=np.complex128
        )
    except Exception as e:
        logger.error(
            f"Error evaluating HMat function at q={q_vector}: {e}"
        )  # Use logger
        return nan_energies

    eigenvalues: npt.NDArray[np.complex_]
    try:
        eigenvalues = la.eigvals(HMat_numeric)
    except np.linalg.LinAlgError:
        logger.error(f"Eigenvalue calculation failed for q={q_vector}.")  # Use logger
        return nan_energies

    imag_part_mags: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues))
    if np.any(imag_part_mags > ENERGY_IMAG_PART_THRESHOLD):
        logger.warning(
            f"Significant imaginary part in eigenvalues for q={q_vector}. Max imag: {np.max(imag_part_mags)}"
        )  # Use logger

    energies: npt.NDArray[np.float_]
    try:
        eigenvalues_sorted_real: npt.NDArray[np.float_] = np.real(np.sort(eigenvalues))
        energies = eigenvalues_sorted_real[nspins:]
        if len(energies) != nspins:
            logger.warning(
                f"Unexpected number of positive energies ({len(energies)}) found for q={q_vector}. Expected {nspins}."
            )  # Use logger
            if len(energies) > nspins:
                energies = energies[:nspins]
            else:
                energies = np.pad(
                    energies, (0, nspins - len(energies)), constant_values=np.nan
                )
    except Exception as e:
        logger.error(
            f"Error during eigenvalue sorting/selection for q={q_vector}: {e}"
        )  # Use logger
        return nan_energies

    return energies


def calc_disp(
    spin_magnitude: float,
    q_vectors: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    hamiltonian_params: Union[List[float], npt.NDArray[np.float_]],
    cache_file_base: str,
    cache_mode: str,
) -> Optional[List[npt.NDArray[np.float_]]]:
    """
    Calculates the spin-wave dispersion relation (energy vs. momentum) over a list of q-points.

    Args:
        spin_magnitude (float): Numerical value of the spin magnitude.
        q_vectors (Union[List, npt.NDArray]): List or array of momentum vectors q.
        hamiltonian_params (Union[List, npt.NDArray]): Numerical Hamiltonian parameter values.
        cache_file_base (str): Base filename for caching.
        cache_mode (str): Cache mode ('r' or 'w').

    Returns:
        Optional[List[npt.NDArray[np.float_]]]: List of energy arrays, or None on failure.

    (Detailed description omitted for brevity - see previous version)
    """
    logger.info("Calculating magnon dispersion ...")  # Use logger
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
        logger.error(f"Failed to get symbolic matrix HMat: {e}")  # Use logger
        return None

    param_substitutions: List[Tuple[sp.Symbol, float]] = [
        (S_sym, spin_magnitude)
    ] + list(zip(params_sym, hamiltonian_params))
    HMat_num_sym: sp.Matrix
    try:
        HMat_num_sym = HMat_sym.subs(param_substitutions, simultaneous=True).evalf()
    except Exception as e:
        logger.error(
            f"Error during substitution of numerical parameters: {e}"
        )  # Use logger
        return None

    logger.info("Running diagonalization...")  # Use logger
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
        logger.error(f"Error during multiprocessing execution: {e}")  # Use logger

    end_time: float = timeit.default_timer()
    logger.info(
        f"Run-time for dispersion calculation: {np.round((end_time - start_time) / 60, 2)} min."
    )  # Use logger

    return energies_list


# Example usage (add within an `if __name__ == "__main__":` block)
if __name__ == "__main__":
    # --- User Configuration ---
    spin_S_val: float = 1.0
    hamiltonian_params_val: List[float] = [1.0, 0.1]  # Example J, D
    cache_file_base_name: str = "my_model_cache"
    cache_operation_mode: str = "r"  # 'w' for first run, 'r' after

    # Define q points (example path Gamma-X-M-Gamma for square lattice)
    q_points_list: List[List[float]] = []
    N_points_per_segment = 20  # Number of points between high-symmetry points
    # Gamma to X
    q_points_list.extend(
        np.linspace([0, 0, 0], [np.pi, 0, 0], N_points_per_segment, endpoint=False)
    )
    # X to M
    q_points_list.extend(
        np.linspace(
            [np.pi, 0, 0], [np.pi, np.pi, 0], N_points_per_segment, endpoint=False
        )
    )
    # M to Gamma
    q_points_list.extend(
        np.linspace([np.pi, np.pi, 0], [0, 0, 0], N_points_per_segment, endpoint=True)
    )  # Include endpoint Gamma
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
        # Keep print for final user output in example
        print("\n--- Dispersion Energies ---")
        for i, q_vec in enumerate(q_points_array):
            if (
                i < len(dispersion_energies)
                and isinstance(dispersion_energies[i], np.ndarray)
                and not np.isnan(dispersion_energies[i]).any()
            ):
                # Only print first few and last few points for brevity if many points
                if i < 5 or i >= len(q_points_array) - 5:
                    print(
                        f"q = {np.round(q_vec, 3)}: E = {np.round(dispersion_energies[i], 4)}"
                    )
                elif i == 5:
                    print("...")
            else:
                # Use logger for calculation failures within the loop
                logger.warning(
                    f"Dispersion calculation failed for q = {np.round(q_vec, 3)}"
                )
        # Add plotting code here if desired (e.g., using matplotlib)
    else:
        logger.error("Dispersion calculation failed to start.")  # Use logger

    # --- Calculate S(q,w) ---
    # Ensure cache_mode='r' if dispersion was just calculated with 'w'
    # If cache_operation_mode was 'w', change it to 'r' here if desired
    sqw_cache_mode = cache_operation_mode  # Or explicitly set to 'r'
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
        sqw_cache_mode,  # Use potentially updated mode
    )
    q_vectors_out, energies_sqw, intensities_sqw = sqw_results

    if (
        q_vectors_out is not None
        and energies_sqw is not None
        and intensities_sqw is not None
    ):
        # Keep print for final user output in example
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
                # Only print first few and last few points for brevity
                if i < 5 or i >= len(q_vectors_out) - 5:
                    print(
                        f"q = {np.round(q_vec, 3)}: E = {np.round(energies_sqw[i], 4)}, S(q,w) = {np.round(intensities_sqw[i], 4)}"
                    )
                elif i == 5:
                    print("...")
            else:
                # Use logger for calculation failures within the loop
                logger.warning(
                    f"S(q,w) calculation failed for q = {np.round(q_vec, 3)}"
                )
        # Add plotting code here if desired
    else:
        logger.error("S(q,w) calculation failed to start.")  # Use logger

    logger.info("Example calculation finished.")
