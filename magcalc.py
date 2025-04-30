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

# Type Hinting Imports
from typing import List, Tuple, Dict, Any, Optional, Union, NoReturn
import numpy.typing as npt

# Ensure the pckFiles directory exists
import os

if not os.path.exists("pckFiles"):
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
    # Use simultaneous=True for potentially better handling of overlapping substitutions
    # and ensuring consistent application based on the original expression.
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
    print("Number of spins in the unit cell: ", nspins)

    # generate boson spin operators in local coordinate system,
    # with Z as quantization axis (Holstein-Primakoff, linear approx)
    c_ops: List[sp.Symbol] = sp.symbols(
        "c0:%d" % nspins_ouc, commutative=False
    )  # Annihilation ops for all spins (incl. neighbours)
    cd_ops: List[sp.Symbol] = sp.symbols(
        "cd0:%d" % nspins_ouc, commutative=False
    )  # Creation ops
    # Note: S+/- = sqrt(S/2)*(c+cd) and sqrt(S/2)*(c-cd)/I differs slightly from
    # standard S+ = sqrt(2S)a, S- = sqrt(2S)adagger. This definition implies
    # c = (Sx_local + iSy_local)/sqrt(2S), cd = (Sx_local - iSy_local)/sqrt(2S) up to rotation.
    spin_ops_local: List[sp.Matrix] = [
        sp.Matrix(
            (
                sp.sqrt(S_sym / 2) * (c_ops[i] + cd_ops[i]),  # Sx_local
                sp.sqrt(S_sym / 2) * (c_ops[i] - cd_ops[i]) / I,  # Sy_local
                S_sym
                - cd_ops[i]
                * c_ops[i],  # Sz_local (linear approx: S_sym - cdag*c -> S_sym)
            )
        )
        for i in range(nspins_ouc)
    ]

    # rotate spin operators to global coordinates using matrices from spin_model.mpr
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]] = sm.mpr(
        params_sym
    )  # Rotation matrices R such that S_global = R * S_local
    # spin_ops_global_ouc contains spin operators for all atoms needed (unit cell + neighbours)
    # ordered by unit cell index then spin index within cell
    spin_ops_global_ouc: List[sp.Matrix] = [
        rotation_matrices[j] * spin_ops_local[nspins * i + j]
        for i in range(
            int(nspins_ouc / nspins)
        )  # Loop over unit cells (incl. neighbours)
        for j in range(nspins)  # Loop over spins within a cell
    ]

    # generate the spin Hamiltonian using the model definition (spin_model.Hamiltonian)
    # This function should return a sympy expression in terms of the provided spin operators.
    hamiltonian_sym: sp.Expr = sm.Hamiltonian(spin_ops_global_ouc, params_sym)
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    # Extract terms based on powers of S_sym.
    # This aims to isolate the quadratic boson part relevant for LSWT.
    # The exact logic depends on how Hamiltonian() is defined and expanded.
    hamiltonian_S0: sp.Expr = hamiltonian_sym.coeff(
        S_sym, 0
    )  # Terms independent of S (constant or quadratic in bosons)
    # Assuming params_sym[-1] is magnetic field H, keep the Zeeman term linear in H.
    # Keep terms linear and quadratic in S (which become quadratic and constant in bosons, respectively).
    hamiltonian_sym = (
        hamiltonian_S0.coeff(params_sym[-1])
        * params_sym[-1]  # Zeeman part from S^0 term? Check convention.
        + hamiltonian_sym.coeff(S_sym, 1)
        * S_sym  # Terms linear in S -> quadratic in bosons
        + hamiltonian_sym.coeff(S_sym, 2)
        * S_sym**2  # Terms quadratic in S -> constant boson term (ignored later)
    )
    hamiltonian_sym = sp.expand(hamiltonian_sym)  # Expand again after selection

    # Define momentum-space boson operators (only need for nspins in the unit cell for H2)
    # ck_j = 1/sqrt(N_cells) * sum_R c_{R,j} * exp(-ik.R)
    # cmk_j = 1/sqrt(N_cells) * sum_R c_{R,j} * exp(ik.R) (for -k)
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
    # This maps real-space bilinear boson terms (c_i*c_j, cdag_i*c_j etc.)
    # to momentum-space terms involving ck, ckd, cmk, cmkd operators.
    interaction_matrix: npt.NDArray = sm.spin_interactions(params_sym)[
        0
    ]  # Get interaction matrix J_ij (which pairs interact)
    fourier_substitutions: List[List[sp.Expr]] = [
        ent
        for i in range(
            nspins
        )  # Index for spin within the reference unit cell (cell R=0)
        for j in range(
            nspins_ouc
        )  # Index for potentially interacting spin (atom j' in cell R')
        if interaction_matrix[i, j]
        != 0  # Only consider interacting pairs based on spin_model
        # Calculate displacement vector r_i - r_j = (r_{0,i} - r_{R',j'})
        for disp_vec in [atom_positions_uc[i, :] - atom_positions_ouc[j, :]]
        # Calculate phase factor k . (r_i - r_j)
        for k_dot_dr in [
            k_sym[0] * disp_vec[0] + k_sym[1] * disp_vec[1] + k_sym[2] * disp_vec[2]
        ]
        # Define substitution rules for each type of bilinear term
        for ent in [
            # Example: cd_i * cd_j -> involves terms like ckd_i * cmkd_j * exp(-ik.dr)
            # The exact form depends on the FT definition. Using modulo nspins maps
            # the index j (which can be > nspins) back to the unit cell index for ck/cmk ops.
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
            # On-site term (i=j): cd_j * c_j -> (ckd_j*ck_j + cmkd_j*cmk_j)/2
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
    # Note: The validity of these specific FT rules should be checked against the
    # chosen definitions of c, cd, ck, cmk and standard LSWT derivations.

    # Apply commutation relations to bring terms into standard quadratic form:
    # Order: ckd*ck, cmk*cmkd, cmk*ck, ckd*cmkd
    commutation_substitutions: List[List[sp.Expr]] = (
        [
            [
                ck_ops[i] * ckd_ops[j],
                ckd_ops[j] * ck_ops[i] + (1 if i == j else 0),
            ]  # Use [ck_i, ckd_j] = delta_ij
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [
                cmkd_ops[i] * cmk_ops[j],
                cmk_ops[j] * cmkd_ops[i] + (1 if i == j else 0),
            ]  # Use [cmk_j, cmkd_i] = delta_ij
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [ck_ops[i] * cmk_ops[j], cmk_ops[j] * ck_ops[i]]  # [ck_i, cmk_j] = 0
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [cmkd_ops[i] * ckd_ops[j], ckd_ops[j] * cmkd_ops[i]]  # [cmkd_i, ckd_j] = 0
            for i in range(nspins)
            for j in range(nspins)
        ]
    )

    # Create basis vector X = [ck0..ckN-1, cmkd0..cmkdN-1] (N=nspins)
    # and Xd = [ckd0..ckdN-1, cmk0..cmkN-1] (dagger)
    # The quadratic Hamiltonian should be H = const + Xd H2 X
    basis_vector_dagger: List[sp.Symbol] = (
        ckd_ops[:nspins] + cmk_ops[:nspins]
    )  # Row vector [ckd_0..N-1, cmk_0..N-1]
    basis_vector: List[sp.Symbol] = (
        ck_ops[:nspins] + cmkd_ops[:nspins]
    )  # Column vector [ck_0..N-1, cmkd_0..N-1]^T (conceptually)

    # Use commutative placeholders (placeholder_symbols) to extract coefficients of non-commutative products.
    # This trick allows using sympy's .coeff() method which works on commutative expressions.
    # placeholder_symbols_ij represents the placeholder for the product basis_vector_dagger[i] * basis_vector[j].
    placeholder_symbols: List[sp.Symbol] = [
        sp.Symbol("XdX%d" % (i * 2 * nspins + j), commutative=True)
        for i in range(2 * nspins)
        for j in range(2 * nspins)
    ]
    # Substitution rule: Replace non-commutative product basis_vector_dagger[i]*basis_vector[j] with commutative placeholder_symbols_ij
    placeholder_substitutions: List[List[sp.Expr]] = [
        [
            basis_vector_dagger[i] * basis_vector[j],
            placeholder_symbols[i * 2 * nspins + j],
        ]
        for i in range(2 * nspins)
        for j in range(2 * nspins)
    ]

    # Perform substitutions using multiprocessing for speed
    print("Running substitution ...")
    start_time: float = timeit.default_timer()

    # 1. Substitute Fourier transform rules into the original Hamiltonian hamiltonian_sym
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

    # 2. Substitute commutation relation rules to order terms correctly
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

    # 3. Substitute commutative placeholders (placeholder_symbols) for final coefficient extraction
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
    # Now the Hamiltonian is expressed in terms of commutative placeholder symbols
    hamiltonian_with_placeholders: sp.Expr = Add(*results_placeholder)

    end_time: float = timeit.default_timer()
    print(
        f"Run-time for substitution: {np.round((end_time - start_time) / 60, 2)} min."
    )

    # Extract coefficients of the placeholder symbols placeholder_symbols_ij to form the H2 matrix
    # H2_matrix[i, j] = coefficient of basis_vector_dagger[i] * basis_vector[j]
    H2_elements: sp.Matrix = sp.Matrix(
        [hamiltonian_with_placeholders.coeff(p) for p in placeholder_symbols]
    )
    H2_matrix: sp.Matrix = sp.Matrix(2 * nspins, 2 * nspins, H2_elements)

    # Define the metric tensor g = diag(1, ..., 1, -1, ..., -1) used in Bogoliubov transformation
    g_metric_tensor_sym: sp.Matrix = sp.diag(*([1] * nspins + [-1] * nspins))

    # The matrix to be diagonalized via Bogoliubov transformation is dynamical_matrix_TwogH2 = 2 * g * H2_matrix
    # This form arises from the equations of motion or canonical transformation approach.
    dynamical_matrix_TwogH2: sp.Matrix = 2 * g_metric_tensor_sym * H2_matrix

    # Create Ud_rotation_matrix matrix: block diagonal matrix of spin rotation operators rotation_matrices[i]
    # This matrix transforms spin components from the local quantization frame to the global frame.
    # Ud = block_diag(R_0, R_1, ..., R_{N-1})
    Ud_rotation_matrix_blocks: List[sp.Matrix] = []
    for i in range(nspins):
        rot_mat = rotation_matrices[i]
        # Ensure it's a sympy Matrix for symbolic consistency
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
    # Using QR decomposition is generally more numerically stable than manual Gram-Schmidt.
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
    # This function indicates a potential issue in the complex degeneracy handling/matching logic.
    print("Error: Mismatch in degeneracy handling (check_degeneracy).")
    print(f"  q = {q_vector}")
    print(f"  Eigenvalue indices involved: {index - degeneracy_count} to {index}")
    print(f"  Degeneracy count (ndeg + 1): {degeneracy_count + 1}")
    print(
        f"  Projection results (abs): {projection_magnitudes} (Threshold: {PROJECTION_CHECK_TOLERANCE})"
    )
    print(
        f"  Eigenvalues in range: {np.real(eigenvalues_neg_q[index - degeneracy_count : index + 1])}"
    )
    print(f"  Number of vectors found by projection: {projected_eigenvectors.shape[1]}")
    print("The program will exit...")
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
    # Use constants defined at the top
    dEdeg: float = DEGENERACY_THRESHOLD
    zero_tol: float = ZERO_MATRIX_ELEMENT_THRESHOLD
    match_tol: float = EIGENVECTOR_MATCHING_THRESHOLD

    # Metric tensor g = diag(1, ..., 1, -1, ..., -1) required for Bogoliubov transformation
    G_metric_tensor: npt.NDArray[np.float_] = np.diag(
        np.concatenate([np.ones(nspins), -np.ones(nspins)])
    )

    # --- Diagonalization for +q ---
    # Solves Hmat_plus_q * v = w * v
    try:
        eigvals_p: npt.NDArray[np.complex_]
        eigvecs_p: npt.NDArray[np.complex_]
        eigvals_p, eigvecs_p = la.eig(Hmat_plus_q)
    except np.linalg.LinAlgError as e:
        print(f"Error: Eigenvalue calculation failed for +q = {q_vector}: {e}")
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Sorting eigenvalues and eigenvectors:
    # The goal is to group eigenvalues into positive and negative branches,
    # typically corresponding to E_k and -E_{-k}.
    # The specific sorting here separates the initially sorted array into two halves,
    # assumes one half is mostly positive and the other mostly negative,
    # then sorts the negative half by absolute value.
    sort_indices_p: npt.NDArray[np.int_] = eigvals_p.argsort()
    eigvecs_p_tmp1: npt.NDArray[np.complex_] = eigvecs_p[:, sort_indices_p][
        :, nspins : 2 * nspins
    ]  # Upper half of sorted vecs
    eigvals_p_tmp1: npt.NDArray[np.complex_] = eigvals_p[sort_indices_p][
        nspins : 2 * nspins
    ]  # Upper half of sorted vals
    eigvecs_p_tmp2: npt.NDArray[np.complex_] = eigvecs_p[:, sort_indices_p][
        :, 0:nspins
    ]  # Lower half of sorted vecs
    eigvals_p_tmp2: npt.NDArray[np.complex_] = eigvals_p[sort_indices_p][
        0:nspins
    ]  # Lower half of sorted vals
    sort_indices_p_neg: npt.NDArray[np.int_] = (
        np.abs(eigvals_p_tmp2)
    ).argsort()  # Sort lower half by magnitude
    eigvecs_p_tmp3: npt.NDArray[np.complex_] = eigvecs_p_tmp2[:, sort_indices_p_neg]
    eigvals_p_tmp3: npt.NDArray[np.complex_] = eigvals_p_tmp2[sort_indices_p_neg]
    # Final sorted eigenvalues/vectors: positive branch first, then negative sorted by |E|
    eigenvalues_plus_q_sorted: npt.NDArray[np.complex_] = np.concatenate(
        (eigvals_p_tmp1, eigvals_p_tmp3)
    )
    eigenvectors_plus_q_sorted: npt.NDArray[np.complex_] = np.hstack(
        (eigvecs_p_tmp1, eigvecs_p_tmp3)
    )

    # Gram-Schmidt Orthogonalization for degenerate subspaces [+q]
    # This handles cases where eigenvalues are numerically very close.
    degeneracy_count: int = 0
    for i in range(1, 2 * nspins):
        if abs(eigenvalues_plus_q_sorted[i] - eigenvalues_plus_q_sorted[i - 1]) < dEdeg:
            degeneracy_count += 1
        elif degeneracy_count > 0:
            # Apply Gram-Schmidt to the columns corresponding to the degenerate block
            vec_block: npt.NDArray[np.complex_] = eigenvectors_plus_q_sorted[
                :, i - degeneracy_count - 1 : i
            ]
            orthonormal_vecs: npt.NDArray[np.complex_] = gram_schmidt(vec_block)
            # Replace original block with orthonormalized vectors
            if orthonormal_vecs.shape[1] == vec_block.shape[1]:
                eigenvectors_plus_q_sorted[:, i - degeneracy_count - 1 : i] = (
                    orthonormal_vecs
                )
            else:  # Handle potential rank deficiency found by QR
                print(
                    f"Warning: Rank deficiency detected during GS for +q at index {i}, q={q_vector}"
                )
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
    # Check degeneracy at the very end of the array
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
            print(
                f"Warning: Rank deficiency detected during GS for +q at end of array, q={q_vector}"
            )
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

    # Determine Alpha matrix (normalization factors) [+q]
    # The Bogoliubov transformation matrix T satisfies T G T^dagger = G.
    # Its inverse T^-1 relates the original basis X to the diagonal basis Y: X = T^-1 Y.
    # The eigenvector matrix V from la.eig(gH) is related to T^-1 by T^-1 = V * alpha,
    # where alpha is a diagonal normalization matrix ensuring T^-1 G (T^-1)^dagger = G.
    # alpha^2 = diag( abs(real( V^-1 G (V^-1)^dagger )) ) -- careful with indices/conj.
    # Easier: alpha_ii^2 = | <V_i | G | V_i> | where V_i are columns of V? No.
    # Correct: alpha_ii^2 = | <U_i | G | U_i> | where U_i are rows of V^-1.
    try:
        inv_eigenvectors_plus_q: npt.NDArray[np.complex_] = la.inv(
            eigenvectors_plus_q_sorted
        )
        alpha_sq_diag_p: npt.NDArray[np.float_] = np.zeros(2 * nspins, dtype=float)
        # Calculate diagonal elements <U_i | G | U_i>
        for i in range(2 * nspins):
            row_i = inv_eigenvectors_plus_q[i, :]
            alpha_sq_diag_p[i] = np.real(
                np.dot(np.conj(row_i), G_metric_tensor @ row_i)
            )

        # Ensure positivity and take square root
        alpha_sq_diag_p[alpha_sq_diag_p < 0] = 0
        alpha_diag_p: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag_p)
        alpha_diag_p[np.abs(alpha_diag_p) < zero_tol] = 0  # Apply zero threshold
        alpha_matrix_plus_q: npt.NDArray[np.complex_] = np.diag(alpha_diag_p).astype(
            np.complex_
        )
    except np.linalg.LinAlgError:
        print(
            f"Error: Matrix inversion failed for +q = {q_vector}. Eigenvector matrix might be singular."
        )
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Prepare swapped+conjugated eigenvector matrix from +q results.
    # This is used as the target for matching the -q results.
    # Based on property T(-k) = conj(swap(T(k))) for the Bogoliubov matrix.
    eigenvectors_plus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack(
            (
                eigenvectors_plus_q_sorted[
                    nspins : 2 * nspins, :
                ],  # Lower block (cmkd part)
                eigenvectors_plus_q_sorted[0:nspins, :],  # Upper block (ck part)
            )
        )
    )

    # --- Diagonalization for -q ---
    # Repeat diagonalization and sorting for Hmat_minus_q
    try:
        eigvals_m: npt.NDArray[np.complex_]
        eigvecs_m: npt.NDArray[np.complex_]
        eigvals_m, eigvecs_m = la.eig(Hmat_minus_q)
    except np.linalg.LinAlgError as e:
        print(f"Error: Eigenvalue calculation failed for -q = {-q_vector}: {e}")
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Apply the same sorting logic as for +q
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

    # Gram-Schmidt Orthogonalization [-q]
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
                print(
                    f"Warning: Rank deficiency detected during GS for -q at index {i}, q={q_vector}"
                )
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
            print(
                f"Warning: Rank deficiency detected during GS for -q at end of array, q={q_vector}"
            )
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

    # Determine alpha matrix [-q]
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
        print(
            f"Error: Matrix inversion failed for -q = {-q_vector}. Eigenvector matrix might be singular."
        )
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # --- Eigenvector Matching between +q and -q ---
    # This is crucial for calculating S(q,w) correctly. We need to ensure that the
    # eigenvectors/values from the -q diagonalization are ordered and phased consistently
    # with the +q results, based on the expected theoretical relationship.
    # We match columns of eigenvectors_minus_q_swapped_conj (derived from -q results)
    # against columns of eigenvectors_plus_q_sorted (from +q results) using projections.
    eigenvectors_minus_q_swapped_conj: npt.NDArray[np.complex_] = np.conj(
        np.vstack(
            (
                eigenvectors_minus_q_sorted[nspins : 2 * nspins, :],
                eigenvectors_minus_q_sorted[0:nspins, :],
            )
        )
    )
    # Initialize arrays to store the reordered -q results
    eigenvectors_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigenvectors_minus_q_sorted, dtype=complex
    )
    eigenvalues_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        eigenvalues_minus_q_sorted, dtype=complex
    )
    alpha_matrix_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        alpha_matrix_minus_q, dtype=complex
    )

    matched_indices_m: set[int] = (
        set()
    )  # Tracks which columns of original -q results have been matched
    num_matched_vectors: int = 0

    # Loop through each eigenvector from the +q calculation
    for i in range(2 * nspins):
        best_match_j: int = -1
        max_proj_metric: float = -1.0  # Stores max projection^2 / (norm1^2 * norm2^2)
        vec_i_plus_q: npt.NDArray[np.complex_] = eigenvectors_plus_q_sorted[:, i]
        vec_i_norm_sq: float = np.real(np.dot(np.conj(vec_i_plus_q), vec_i_plus_q))

        if vec_i_norm_sq < zero_tol**2:
            continue  # Skip if vector is zero

        # Compare vec_i_plus_q with all unmatched swapped/conjugated -q eigenvectors
        for j in range(2 * nspins):
            if j in matched_indices_m:
                continue  # Skip if already matched

            vec_j_minus_q_swapped_conj: npt.NDArray[np.complex_] = (
                eigenvectors_minus_q_swapped_conj[:, j]
            )
            vec_j_norm_sq: float = np.real(
                np.dot(np.conj(vec_j_minus_q_swapped_conj), vec_j_minus_q_swapped_conj)
            )
            if vec_j_norm_sq < zero_tol**2:
                continue

            # Calculate normalized projection squared (cosine squared of angle between vectors)
            projection: complex = np.dot(
                np.conj(vec_i_plus_q), vec_j_minus_q_swapped_conj
            )
            projection_mag_sq: float = np.abs(projection) ** 2
            normalized_projection_mag_sq: float = projection_mag_sq / (
                vec_i_norm_sq * vec_j_norm_sq
            )

            # If projection is close to 1 (vectors are parallel), consider it the best match so far
            if (
                normalized_projection_mag_sq > max_proj_metric
                and normalized_projection_mag_sq > (1.0 - match_tol**2)
            ):
                max_proj_metric = normalized_projection_mag_sq
                best_match_j = j

        # If a good match (best_match_j) was found for vec_i_plus_q
        if best_match_j != -1:
            matched_indices_m.add(best_match_j)
            num_matched_vectors += 1

            # Determine the target index for the reordered -q matrices.
            # This specific swapping (i -> i+nspins, i+nspins -> i) is crucial
            # for the structure needed by K and Kd calculation later.
            target_index: int = -1
            if i < nspins:
                target_index = (
                    i + nspins
                )  # Map first half of +q to second half of reordered -q
            else:
                target_index = (
                    i - nspins
                )  # Map second half of +q to first half of reordered -q

            if target_index != -1:
                # Place the matched eigenvector/value from original -q results into the reordered arrays
                eigenvectors_minus_q_reordered[:, target_index] = (
                    eigenvectors_minus_q_sorted[:, best_match_j]
                )
                eigenvalues_minus_q_reordered[target_index] = (
                    eigenvalues_minus_q_sorted[best_match_j]
                )

                # Adjust the phase of the corresponding alpha factor.
                # We need alpha_matrix_minus_q_reordered[target, target] to have a phase consistent
                # with conj(alpha_matrix_plus_q[i, i] * phase_factor), where phase_factor relates
                # vec_i_plus_q and vec_j_minus_q_swapped_conj.
                # The original code uses the ratio of the first non-zero elements as the phase factor.
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
                else:  # Should not happen if vectors are non-zero
                    alpha_matrix_minus_q_reordered[target_index, target_index] = 0
            else:
                print(
                    f"Warning: Invalid target index during matching for i={i}, q={q_vector}"
                )
        else:
            # This indicates a failure in the matching process, likely due to numerical issues or incorrect assumptions.
            print(
                f"Warning: No matching eigenvector found for +q vector index {i} at q={q_vector}"
            )

    # Verify that all vectors were successfully matched
    if num_matched_vectors != 2 * nspins:
        print(
            f"Warning: Number of matched vectors ({num_matched_vectors}) does not equal 2*nspins at q={q_vector}"
        )

    # Use the reordered and phase-corrected matrices for the -q results
    eigenvectors_minus_q_final = eigenvectors_minus_q_reordered
    alpha_matrix_minus_q_final = alpha_matrix_minus_q_reordered
    alpha_matrix_minus_q_final[np.abs(alpha_matrix_minus_q_final) < zero_tol] = (
        0  # Clean up small values
    )

    # --- Calculate K and Kd matrices ---
    # These matrices relate the global spin operators S_alpha(q) to the Bogoliubov operators b_l(q), b_l^dagger(-q)
    # S_alpha(q) = sum_l ( K[alpha, l] * b_l(q) + Kd[alpha, l] * b_l^dagger(-q) )

    # Calculate T^-1 = V * alpha for both +q and the reordered -q
    inv_bogoliubov_T_plus_q: npt.NDArray[np.complex_] = (
        eigenvectors_plus_q_sorted @ alpha_matrix_plus_q
    )
    inv_bogoliubov_T_minus_q_reordered: npt.NDArray[np.complex_] = (
        eigenvectors_minus_q_final @ alpha_matrix_minus_q_final
    )

    # Define the matrix Udd_local_boson_map that relates local spin components (Sx, Sy)
    # to the local boson operators (c, cd) used in the Holstein-Primakoff definition:
    # [Sx_i, Sy_i, Sz_i]^T ~ sqrt(S/2) * [[1, 1], [1/i, -1/i], [0, 0]] * [c_i, cd_i]^T
    Udd_local_boson_map: npt.NDArray[np.complex_] = np.zeros(
        (3 * nspins, 2 * nspins), dtype=complex
    )
    for i in range(nspins):
        Udd_local_boson_map[3 * i, i] = 1.0  # Sx_i ~ c_i contribution
        Udd_local_boson_map[3 * i, i + nspins] = 1.0  # Sx_i ~ cd_i contribution
        Udd_local_boson_map[3 * i + 1, i] = 1.0 / I  # Sy_i ~ c_i contribution
        Udd_local_boson_map[3 * i + 1, i + nspins] = (
            -1.0 / I
        )  # Sy_i ~ cd_i contribution
        # Sz component is typically S - cdag*c, ignored in linear calculation of K/Kd for transverse fluctuations

    # Combine the transformations: S_global = Ud * S_local ~ Ud * sqrt(S/2) * Udd * X = Ud * sqrt(S/2) * Udd * T^-1 * Y
    # The K matrix relates S_global to the b_k part of Y (first nspins columns of T^-1)
    # The Kd matrix relates S_global to the b_{-k}^dagger part of Y (last nspins columns of T^-1)
    # The calculation uses the full T^-1 matrices based on the original code structure.
    prefactor: float = np.sqrt(spin_magnitude / 2.0)
    K_matrix: npt.NDArray[np.complex_] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_bogoliubov_T_plus_q
    )
    Kd_matrix: npt.NDArray[np.complex_] = (
        prefactor
        * Ud_numeric
        @ Udd_local_boson_map
        @ inv_bogoliubov_T_minus_q_reordered
    )  # Uses matched T^-1(-q)

    # Clean up small numerical noise
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
        print("Generating symbolic matrices (HMat, Ud)...")
        HMat, Ud = gen_HM(k_sym, S_sym, params_sym)  # Call gen_HM to generate
        print(f"Writing HMat to {hm_cache_file}")
        try:
            with open(hm_cache_file, "wb") as outHM:
                pickle.dump(HMat, outHM)
        except IOError as e:
            print(f"Error writing HMat cache file: {e}")
            raise
        print(f"Writing Ud to {ud_cache_file}")
        try:
            with open(ud_cache_file, "wb") as outUd:
                pickle.dump(Ud, outUd)
        except IOError as e:
            print(f"Error writing Ud cache file: {e}")
            raise

    elif cache_mode == "r":
        print(
            f"Importing symbolic matrices from cache files ({hm_cache_file}, {ud_cache_file})..."
        )
        try:
            with open(hm_cache_file, "rb") as inHM:
                HMat = pickle.load(inHM)  # Load from file
            with open(ud_cache_file, "rb") as inUd:
                Ud = pickle.load(inUd)  # Load from file
        except FileNotFoundError:
            print(
                f"Error: Cache files not found. Run with 'w' option first or check filename '{cache_file_base}'."
            )
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading cache files (may be corrupted or incompatible): {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred loading cache files: {e}")
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
        # Create fast numerical function from symbolic matrix
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        print(f"Error during lambdify at q={q_vector}: {e}")
        return q_vector, nan_energies, nan_intensities

    try:
        # Evaluate numerical matrix at +q and -q
        Hmat_plus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vector[0], q_vector[1], q_vector[2]), dtype=np.complex128
        )
        Hmat_minus_q: npt.NDArray[np.complex_] = np.array(
            HMat_func(-q_vector[0], -q_vector[1], -q_vector[2]), dtype=np.complex128
        )
    except Exception as e:
        print(f"Error evaluating HMat function at q={q_vector}: {e}")
        return q_vector, nan_energies, nan_intensities

    # Perform Bogoliubov diagonalization and get K, Kd matrices
    K_matrix: npt.NDArray[np.complex_]
    Kd_matrix: npt.NDArray[np.complex_]
    eigenvalues: npt.NDArray[np.complex_]
    K_matrix, Kd_matrix, eigenvalues = KKdMatrix(
        spin_magnitude_num, Hmat_plus_q, Hmat_minus_q, Ud_numeric, q_vector, nspins
    )

    # Check for errors from KKdMatrix
    if (
        np.isnan(K_matrix).any()
        or np.isnan(Kd_matrix).any()
        or np.isnan(eigenvalues).any()
    ):
        print(
            f"Warning: NaN encountered in KKdMatrix result for q={q_vector}. Skipping intensity calculation."
        )
        return q_vector, nan_energies, nan_intensities

    # Extract positive energies and check imaginary part using constant
    imag_energy_mag: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues[0:nspins]))
    if np.any(imag_energy_mag > ENERGY_IMAG_PART_THRESHOLD):
        print(
            f"Warning: Significant imaginary part in energy eigenvalues for q={q_vector}. Max imag: {np.max(imag_energy_mag)}"
        )
    energies: npt.NDArray[np.float_] = np.real(eigenvalues[0:nspins])
    q_output: npt.NDArray[np.float_] = q_vector  # Use input q_vector as output q

    # Calculate intensity for each mode (corresponding to each positive energy)
    for mode_index in range(nspins):
        spin_correlation_matrix: npt.NDArray[np.complex_] = np.zeros(
            (3, 3), dtype=complex
        )  # Stores <S_alpha(q) S_beta(-q)>_omega_l
        intensity_one_mode: complex = 0.0 + 0.0j

        # Calculate S_alpha,beta(q, omega_l) using K and Kd
        # This corresponds to the contribution of mode l to the correlation function.
        for alpha in range(3):  # Global coord index (x=0, y=1, z=2)
            for beta in range(3):  # Global coord index
                correlation_sum: complex = 0.0 + 0.0j
                # Sum contributions over all spins i, j in the unit cell
                for spin_i in range(nspins):
                    for spin_j in range(nspins):
                        idx_K: int = (
                            3 * spin_i + alpha
                        )  # Row index for spin i, component alpha
                        idx_Kd: int = (
                            3 * spin_j + beta
                        )  # Row index for spin j, component beta
                        # The structure K[..., l] * Kd[..., l+nspins] comes from the specific
                        # definition relating S_alpha to b_k and b_{-k}^dagger via K and Kd.
                        correlation_sum += (
                            K_matrix[idx_K, mode_index]  # Contribution from b_l(q)
                            * Kd_matrix[
                                idx_Kd, mode_index + nspins
                            ]  # Contribution from b_l^dagger(-q)
                        )
                spin_correlation_matrix[alpha, beta] = correlation_sum

        # Apply geometric polarization factor for neutron scattering: (delta_ab - q_a*q_b / |q|^2)
        q_norm_sq: float = np.dot(q_vector, q_vector)
        # Use constant to check for q=0
        if q_norm_sq < Q_ZERO_THRESHOLD:
            # At q=0, factor is delta_ab. Sum trace(spin_correlation_matrix).
            for alpha in range(3):
                intensity_one_mode += spin_correlation_matrix[alpha, alpha]
        else:
            q_normalized: npt.NDArray[np.float_] = q_vector / np.sqrt(q_norm_sq)
            # Sum over alpha, beta: (delta_ab - q_alpha * q_beta) * S_alpha,beta
            for alpha in range(3):
                for beta in range(3):
                    delta_ab: float = 1.0 if alpha == beta else 0.0
                    polarization_factor: float = (
                        delta_ab - q_normalized[alpha] * q_normalized[beta]
                    )
                    intensity_one_mode += (
                        polarization_factor * spin_correlation_matrix[alpha, beta]
                    )

        # Check imaginary part of calculated intensity using constant
        if np.abs(np.imag(intensity_one_mode)) > SQW_IMAG_PART_THRESHOLD:
            print(
                f"Warning: Significant imaginary part in Sqw for q={q_vector}, mode {mode_index}: {np.imag(intensity_one_mode)}"
            )

        sqw_complex_accumulator[mode_index] = intensity_one_mode

    # Final intensity is the real part, ensure non-negative
    intensities: npt.NDArray[np.float_] = np.real(sqw_complex_accumulator)
    intensities[intensities < 0] = 0  # Intensity must be non-negative

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
    print("Calculating scattering intensity S(q,w)...")
    nspins: int
    try:
        nspins = len(sm.atom_pos())
        if nspins == 0:
            raise ValueError("spin_model.atom_pos() returned an empty list.")
    except AttributeError:
        raise AttributeError("Function 'atom_pos' not found in spin_model.py.")
    except Exception as e:
        raise RuntimeError(f"Error getting nspins from spin_model.atom_pos(): {e}")

    # Define symbolic variables
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    k_sym: List[sp.Symbol] = [kx, ky, kz]
    S_sym: sp.Symbol = sp.Symbol("S", real=True)
    params_sym: List[sp.Symbol] = sp.symbols(
        "p0:%d" % len(hamiltonian_params), real=True
    )
    HMat_sym: sp.Matrix
    Ud_sym: sp.Matrix

    # Get symbolic matrices (generate or load from cache)
    try:
        HMat_sym, Ud_sym = process_matrix(
            cache_mode, k_sym, S_sym, params_sym, cache_file_base
        )
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to get symbolic matrices: {e}")
        return None, None, None

    # Substitute numerical values into symbolic matrices
    param_substitutions: List[Tuple[sp.Symbol, float]] = [
        (S_sym, spin_magnitude)
    ] + list(zip(params_sym, hamiltonian_params))
    HMat_num_sym: sp.Matrix
    Ud_num_sym: sp.Matrix
    Ud_numeric: npt.NDArray[np.complex_]
    try:
        HMat_num_sym = HMat_sym.subs(param_substitutions, simultaneous=True).evalf()
        Ud_num_sym = Ud_sym.subs(param_substitutions, simultaneous=True).evalf()
        Ud_numeric = np.array(
            Ud_num_sym, dtype=np.complex128
        )  # Convert Ud to numerical numpy array
    except Exception as e:
        print(f"Error during substitution of numerical parameters: {e}")
        return None, None, None

    print("Running diagonalization and intensity calculation...")
    start_time: float = timeit.default_timer()

    # Prepare arguments for parallel processing
    pool_args: List[Tuple] = [
        (HMat_num_sym, Ud_numeric, k_sym, q_vec, nspins, spin_magnitude)
        for q_vec in q_vectors
    ]
    results: List[
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]
    ] = []
    # Execute calculations in parallel
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
        print(f"Error during multiprocessing execution: {e}")
        return None, None, None

    # Unpack results
    q_vectors_out: Tuple[npt.NDArray[np.float_], ...]
    energies_out: Tuple[npt.NDArray[np.float_], ...]
    intensities_out: Tuple[npt.NDArray[np.float_], ...]
    try:
        if not results or len(results[0]) != 3:
            raise ValueError("Multiprocessing returned empty or malformed results.")
        q_vectors_out, energies_out, intensities_out = zip(*results)
    except ValueError as e:
        print(f"Error unpacking results from parallel processing: {e}")
        return None, None, None

    end_time: float = timeit.default_timer()
    print(
        f"Run-time for S(q,w) calculation: {np.round((end_time - start_time) / 60, 2)} min."
    )

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
        # Create fast numerical function
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        print(f"Error during lambdify at q={q_vector}: {e}")
        return nan_energies

    try:
        # Evaluate numerical matrix at q
        HMat_numeric: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vector[0], q_vector[1], q_vector[2]), dtype=np.complex128
        )
    except Exception as e:
        print(f"Error evaluating HMat function at q={q_vector}: {e}")
        return nan_energies

    # Calculate eigenvalues (faster than eig if only values are needed)
    eigenvalues: npt.NDArray[np.complex_]
    try:
        eigenvalues = la.eigvals(HMat_numeric)
    except np.linalg.LinAlgError:
        print(f"Error: Eigenvalue calculation failed for q={q_vector}.")
        return nan_energies

    # Check imaginary part using constant
    imag_part_mags: npt.NDArray[np.float_] = np.abs(np.imag(eigenvalues))
    if np.any(imag_part_mags > ENERGY_IMAG_PART_THRESHOLD):
        print(
            f"Warning: Significant imaginary part in eigenvalues for q={q_vector}. Max imag: {np.max(imag_part_mags)}"
        )

    # Sort eigenvalues and extract positive energy branches
    energies: npt.NDArray[np.float_]
    try:
        eigenvalues_sorted_real: npt.NDArray[np.float_] = np.real(
            np.sort(eigenvalues)
        )  # Sort ascending by real part
        # Physical magnon energies correspond to the positive eigenvalues.
        # Assuming the Bogoliubov transformation yields pairs +/-E, the positive
        # energies should be the upper half of the sorted real parts.
        energies = eigenvalues_sorted_real[nspins:]
        # Verify correct number of energies obtained
        if len(energies) != nspins:
            print(
                f"Warning: Unexpected number of positive energies ({len(energies)}) found for q={q_vector}. Expected {nspins}."
            )
            # Pad with NaN if fewer energies found, truncate if more found
            if len(energies) > nspins:
                energies = energies[:nspins]
            else:
                energies = np.pad(
                    energies, (0, nspins - len(energies)), constant_values=np.nan
                )
    except Exception as e:
        print(f"Error during eigenvalue sorting/selection for q={q_vector}: {e}")
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
    print("Calculating magnon dispersion ...")
    nspins: int
    try:
        nspins = len(sm.atom_pos())
        if nspins == 0:
            raise ValueError("spin_model.atom_pos() returned an empty list.")
    except AttributeError:
        raise AttributeError("Function 'atom_pos' not found in spin_model.py.")
    except Exception as e:
        raise RuntimeError(f"Error getting nspins from spin_model.atom_pos(): {e}")

    # Define symbolic variables
    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    k_sym: List[sp.Symbol] = [kx, ky, kz]
    S_sym: sp.Symbol = sp.Symbol("S", real=True)
    params_sym: List[sp.Symbol] = sp.symbols(
        "p0:%d" % len(hamiltonian_params), real=True
    )
    HMat_sym: sp.Matrix

    # Get symbolic matrix HMat (Ud not needed for dispersion)
    try:
        # The Ud matrix is also returned but ignored using '_'
        HMat_sym, _ = process_matrix(
            cache_mode, k_sym, S_sym, params_sym, cache_file_base
        )
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to get symbolic matrix HMat: {e}")
        return None

    # Substitute numerical values for spin and parameters
    param_substitutions: List[Tuple[sp.Symbol, float]] = [
        (S_sym, spin_magnitude)
    ] + list(zip(params_sym, hamiltonian_params))
    HMat_num_sym: sp.Matrix
    try:
        # .evalf() attempts to evaluate any remaining symbolic constants numerically
        HMat_num_sym = HMat_sym.subs(param_substitutions, simultaneous=True).evalf()
    except Exception as e:
        print(f"Error during substitution of numerical parameters: {e}")
        return None

    print("Running diagonalization ...")
    start_time: float = timeit.default_timer()

    # Prepare arguments for parallel processing
    pool_args: List[Tuple] = [
        (HMat_num_sym, k_sym, q_vec, nspins) for q_vec in q_vectors
    ]
    energies_list: List[npt.NDArray[np.float_]] = []
    # Execute in parallel using imap for progress bar compatibility
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
        print(f"Error during multiprocessing execution: {e}")
        # energies_list will contain results up to the point of failure

    end_time: float = timeit.default_timer()
    print(
        f"Run-time for dispersion calculation: {np.round((end_time - start_time) / 60, 2)} min."
    )

    return energies_list


# Example usage (add within an `if __name__ == "__main__":` block)
if __name__ == "__main__":
    # Define parameters (example)
    spin_S_val: float = 1.0
    hamiltonian_params_val: List[float] = [1.0, 0.1]  # Example J, D
    cache_file_base_name: str = "my_model_cache"
    # Use 'w' for the very first run to generate cache files.
    # Use 'r' for subsequent runs to load from cache.
    cache_operation_mode: str = "r"  # Or 'w'

    # Define q points (example path Gamma-X-M-Gamma)
    q_points_list: List[List[float]] = []
    # Add specific q-vectors, e.g., from a path generator function
    q_points_list.append([0.0, 0.0, 0.0])
    q_points_list.append([np.pi, 0.0, 0.0])
    q_points_list.append([np.pi, np.pi, 0.0])
    q_points_list.append([0.0, 0.0, 0.0])
    q_points_array: npt.NDArray[np.float_] = np.array(q_points_list)

    # --- Calculate dispersion ---
    dispersion_energies: Optional[List[npt.NDArray[np.float_]]] = calc_disp(
        spin_S_val,
        q_points_array,
        hamiltonian_params_val,
        cache_file_base_name,
        cache_operation_mode,
    )

    if dispersion_energies is not None:
        print("\nDispersion Energies:")
        for i, q_vec in enumerate(q_points_array):
            # Check if the result for this q-point is valid
            if (
                i < len(dispersion_energies)
                and isinstance(dispersion_energies[i], np.ndarray)
                and not np.isnan(dispersion_energies[i]).any()
            ):
                print(
                    f"q = {np.round(q_vec, 3)}: E = {np.round(dispersion_energies[i], 4)}"
                )
            else:
                print(f"q = {np.round(q_vec, 3)}: Calculation failed.")
        # Add plotting code here if desired
    else:
        print("Dispersion calculation failed to start.")

    # --- Calculate S(q,w) ---
    # Note: Ensure cache_operation_mode='r' if dispersion was just calculated with 'w'
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
        cache_operation_mode,
    )
    q_vectors_out, energies_sqw, intensities_sqw = sqw_results

    if (
        q_vectors_out is not None
        and energies_sqw is not None
        and intensities_sqw is not None
    ):
        print("\nScattering Intensities:")
        for i, q_vec in enumerate(q_vectors_out):
            # Check if the result for this q-point is valid
            if (
                i < len(energies_sqw)
                and isinstance(energies_sqw[i], np.ndarray)
                and not np.isnan(energies_sqw[i]).any()
                and i < len(intensities_sqw)
                and isinstance(intensities_sqw[i], np.ndarray)
                and not np.isnan(intensities_sqw[i]).any()
            ):
                print(
                    f"q = {np.round(q_vec, 3)}: E = {np.round(energies_sqw[i], 4)}, S(q,w) = {np.round(intensities_sqw[i], 4)}"
                )
            else:
                print(f"q = {np.round(q_vec, 3)}: Calculation failed.")
        # Add plotting code here if desired
    else:
        print("S(q,w) calculation failed to start.")
