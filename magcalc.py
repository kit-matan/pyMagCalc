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

# Threshold for considering eigenvalues degenerate in KKdMatrix
DEGENERACY_THRESHOLD: float = 1e-12

# Threshold for considering matrix elements or normalization factors effectively zero
ZERO_MATRIX_ELEMENT_THRESHOLD: float = 1e-6

# Threshold for matching eigenvectors between +q and -q based on projection (cosine squared)
# Match is accepted if normalized projection squared > (1 - EIGENVECTOR_MATCHING_THRESHOLD**2)
EIGENVECTOR_MATCHING_THRESHOLD: float = 1e-5

# Threshold for considering the imaginary part of eigenvalues (energies) negligible
ENERGY_IMAG_PART_THRESHOLD: float = 1e-5

# Threshold for considering the imaginary part of S(q,w) negligible
SQW_IMAG_PART_THRESHOLD: float = 1e-4

# Threshold for considering q-vector magnitude effectively zero (to avoid division by zero)
Q_ZERO_THRESHOLD: float = 1e-10

# Tolerance used in the original check_degeneracy function (may be deprecated if refactored)
# Represents a threshold for projection sums.
PROJECTION_CHECK_TOLERANCE: float = 1e-5
# --- End Numerical Constants ---


def substitute_expr(
    args: Tuple[sp.Expr, Union[Dict, List[Tuple[sp.Expr, sp.Expr]]]],
) -> sp.Expr:
    """
    Helper function for multiprocessing substitution. Applies a substitution
    dictionary to a SymPy expression.

    Designed to be used with `multiprocessing.Pool.imap` or similar.

    Args:
        args (tuple): A tuple containing:
            expr (sympy.Expr): The SymPy expression to perform substitution on.
            subs_dict (dict or list): The dictionary or list of tuples defining
                                      the substitutions (e.g., {old: new, ...}
                                      or [(old1, new1), (old2, new2), ...]).

    Returns:
        sympy.Expr: The SymPy expression after applying the substitutions.
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

    This function performs the core symbolic setup:
    1. Defines boson operators using Holstein-Primakoff approximation (linearized).
    2. Rotates local spin operators to global coordinates using matrices from `spin_model.mpr`.
    3. Constructs the symbolic Hamiltonian using `spin_model.Hamiltonian`.
    4. Performs a Fourier transform from real-space to momentum-space boson operators.
    5. Applies commutation relations to bring the Hamiltonian into quadratic form
       (ckd*ck, cmk*cmkd, cmk*ck, ckd*cmkd).
    6. Extracts coefficients to build the dynamical matrix `H2`.
    7. Constructs the matrix `TwogH2 = 2 * g * H2` which needs diagonalization.
    8. Constructs the symbolic block rotation matrix `Ud`.

    Args:
        k_sym (List[sp.Symbol]): List of sympy symbols representing momentum components ([kx, ky, kz]).
                             Typically `sympy.symbols('kx ky kz', real=True)`.
        S_sym (sp.Symbol): Symbolic representation of the spin magnitude
                       (e.g., `sympy.Symbol('S', real=True)`).
        params_sym (List[sp.Symbol]): List of sympy symbols representing Hamiltonian parameters
                                  (e.g., `sympy.symbols('p0:N', real=True)`). The order must
                                  match the usage in `spin_model.py`.

    Returns:
        Tuple[sp.Matrix, sp.Matrix]: A tuple containing:
            TwogH2 (sp.Matrix): The symbolic dynamical matrix (2gH) for LSWT
                                diagonalization. Size (2*nspins, 2*nspins).
                                This matrix, when evaluated numerically and
                                diagonalized, yields the spin-wave energies.
            Ud (sp.Matrix): The symbolic block matrix representing the spin
                            rotation operators from local to global coordinates.
                            Size (3*nspins, 3*nspins). Used for calculating
                            scattering intensity.

    Notes:
        - Relies heavily on functions defined in `spin_model.py`:
          `atom_pos()`, `atom_pos_ouc()`, `mpr()`, `Hamiltonian()`, `spin_interactions()`.
          Ensure these functions are correctly implemented in `spin_model.py`.
        - The `mpr()` function from `spin_model.py` may depend on `params_sym`.
        - The `Hamiltonian()` function from `spin_model.py` defines the interactions.
        - The `spin_interactions()` function likely returns interaction matrices needed
          to determine which pairs interact, used for setting up the Fourier transform.
        - The symbolic calculations, especially substitutions involving large
          Hamiltonians or many spins, can be very time-consuming. Results are
          typically cached using `process_matrix`.
        - The definition `spin_ops_local` uses a specific form of Holstein-Primakoff
          related to `c = (Sx_local + iSy_local)/sqrt(2S)`.
        - The Fourier transform implementation assumes specific relationships between
          real-space and momentum-space operators.
        - The `placeholder_symbols` substitution trick is used to extract coefficients of non-commutative
          operator products using SymPy's commutative coefficient extraction.
    """
    atom_positions_uc: npt.NDArray[np.float_] = sm.atom_pos()
    nspins: int = len(atom_positions_uc)
    atom_positions_ouc: npt.NDArray[np.float_] = sm.atom_pos_ouc()
    nspins_ouc: int = len(atom_positions_ouc)
    print("Number of spins in the unit cell: ", nspins)

    # generate boson spin operators in local coordinate system,
    # with Z as quantization axis (Holstein-Primakoff, linear approx)
    c_ops: List[sp.Symbol] = sp.symbols("c0:%d" % nspins_ouc, commutative=False)
    cd_ops: List[sp.Symbol] = sp.symbols("cd0:%d" % nspins_ouc, commutative=False)
    # Note: S+/- = sqrt(S/2)*(c+cd) and sqrt(S/2)*(c-cd)/I differs slightly from
    # standard S+ = sqrt(2S)a, S- = sqrt(2S)adagger. This definition implies
    # c = (Sx_local + iSy_local)/sqrt(2S), cd = (Sx_local - iSy_local)/sqrt(2S) up to rotation.
    spin_ops_local: List[sp.Matrix] = [
        sp.Matrix(
            (
                sp.sqrt(S_sym / 2) * (c_ops[i] + cd_ops[i]),  # Sx_local
                sp.sqrt(S_sym / 2) * (c_ops[i] - cd_ops[i]) / I,  # Sy_local
                S_sym - cd_ops[i] * c_ops[i],
            )
        )  # Sz_local (linear approx: S_sym)
        for i in range(nspins_ouc)
    ]

    # rotate spin operators to global coordinates
    rotation_matrices: List[Union[npt.NDArray, sp.Matrix]] = sm.mpr(
        params_sym
    )  # the rotation matrices can depend on the Hamiltonian parameters
    # spin_ops_global_ouc contains spin operators for all atoms needed (unit cell + neighbours)
    # ordered by unit cell index then spin index within cell
    spin_ops_global_ouc: List[sp.Matrix] = [
        rotation_matrices[j] * spin_ops_local[nspins * i + j]
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]

    # generate the spin Hamiltonian using the model definition
    hamiltonian_sym: sp.Expr = sm.Hamiltonian(spin_ops_global_ouc, params_sym)
    hamiltonian_sym = sp.expand(hamiltonian_sym)
    # Extract terms based on powers of S_sym.
    # This part seems intended to separate classical and quantum parts, potentially
    # keeping only terms up to quadratic in boson operators (linear spin wave).
    # The exact logic might depend on how Hamiltonian() is defined.
    # Assuming it correctly isolates the relevant quadratic boson terms for LSWT.
    hamiltonian_S0: sp.Expr = hamiltonian_sym.coeff(S_sym, 0)
    # params_sym[-1] is assumed to be magnetic field H; ensure this convention is followed.
    # The line below seems to reconstruct the Hamiltonian from coefficients,
    # potentially simplifying or selecting terms. Needs careful check against theory.
    hamiltonian_sym = (
        hamiltonian_S0.coeff(params_sym[-1]) * params_sym[-1]
        + hamiltonian_sym.coeff(S_sym, 1) * S_sym
        + hamiltonian_sym.coeff(S_sym, 2) * S_sym**2
    )
    hamiltonian_sym = sp.expand(hamiltonian_sym)

    # Define momentum-space boson operators (only need for nspins in the unit cell for H2)
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
    # This maps real-space bilinear boson terms to momentum-space terms
    interaction_matrix: npt.NDArray = sm.spin_interactions(params_sym)[
        0
    ]  # Assumes spin_interactions returns interaction matrix (numpy array)
    fourier_substitutions: List[List[sp.Expr]] = [
        ent
        for i in range(nspins)  # Index for spin within the reference unit cell
        for j in range(
            nspins_ouc
        )  # Index for potentially interacting spin (incl. neighbours)
        if interaction_matrix[i, j]
        != 0  # Only consider interacting pairs defined in spin_model
        for disp_vec in [
            atom_positions_uc[i, :] - atom_positions_ouc[j, :]
        ]  # Displacement vector r_i - r_j
        for k_dot_dr in [
            k_sym[0] * disp_vec[0] + k_sym[1] * disp_vec[1] + k_sym[2] * disp_vec[2]
        ]  # k . (ri - rj)
        for ent in [
            # cd_i * cd_j -> (ckd_i*cmkd_j*exp(-ikdr) + cmkd_i*ckd_j*exp(ikdr))/2
            # Note: Indices i, j here refer to the original c_ops/cd_ops indices (0..nspins_ouc-1)
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
            # c_i * c_j -> (ck_i*cmk_j*exp(ikdr) + cmk_i*ck_j*exp(-ikdr))/2
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
            # cd_i * c_j -> (ckd_i*ck_j*exp(-ikdr) + cmkd_i*cmk_j*exp(ikdr))/2
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
            # c_i * cd_j -> (ck_i*ckd_j*exp(ikdr) + cmk_i*cmkd_j*exp(-ikdr))/2
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
            # On-site term: cd_j * c_j -> (ckd_j*ck_j + cmkd_j*cmk_j)/2
            # This assumes j is in the reference cell for the simple form.
            # The loop structure needs to correctly handle indices i, j relative to cells.
            # Using modulo nspins assumes periodicity.
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
    # Note: The Fourier transform rules used here might need verification against
    # standard LSWT literature depending on the exact definition of c, cd, ck, cmk.

    # Apply commutation relations to bring terms into standard quadratic form:
    # ckd*ck, cmk*cmkd, cmk*ck, ckd*cmkd
    commutation_substitutions: List[List[sp.Expr]] = (
        [
            [
                ck_ops[i] * ckd_ops[j],
                ckd_ops[j] * ck_ops[i] + (1 if i == j else 0),
            ]  # [ck_i, ckd_j] = delta_ij
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [
                cmkd_ops[i] * cmk_ops[j],
                cmk_ops[j] * cmkd_ops[i] + (1 if i == j else 0),
            ]  # [cmk_j, cmkd_i] = delta_ij -> cmkd*cmk = cmk*cmkd + delta
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
    )  # Basis vector (dagger part)
    basis_vector: List[sp.Symbol] = ck_ops[:nspins] + cmkd_ops[:nspins]  # Basis vector

    # Use commutative placeholders (placeholder_symbols) to extract coefficients of non-commutative products.
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

    # Perform substitutions using multiprocessing
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
    hamiltonian_k_space: sp.Expr = Add(
        *results_ft
    )  # Summing up terms after substitution
    hamiltonian_k_space = hamiltonian_k_space.expand()  # Expand after summing

    # 2. Substitute commutation relation rules to order terms
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
    hamiltonian_k_commuted = hamiltonian_k_commuted.expand()  # Expand after summing

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
    hamiltonian_with_placeholders: sp.Expr = Add(*results_placeholder)
    # No need to expand again if placeholder_symbols symbols are the highest level

    end_time: float = timeit.default_timer()
    print(
        f"Run-time for substitution: {np.round((end_time - start_time) / 60, 2)} min."
    )

    # Extract coefficients of the placeholder symbols placeholder_symbols_ij to form the H2 matrix
    # H2_matrix[i, j] = coefficient of basis_vector_dagger[i] * basis_vector[j] (represented by placeholder_symbols_ij)
    H2_elements: sp.Matrix = sp.Matrix(
        [hamiltonian_with_placeholders.coeff(p) for p in placeholder_symbols]
    )
    H2_matrix: sp.Matrix = sp.Matrix(2 * nspins, 2 * nspins, H2_elements)

    # Define the metric tensor g = diag(1, ..., 1, -1, ..., -1) used in Bogoliubov transformation
    g_metric_tensor_sym: sp.Matrix = sp.diag(
        *([1] * nspins + [-1] * nspins)
    )  # Sympy equivalent

    # The matrix to be diagonalized via Bogoliubov transformation is dynamical_matrix_TwogH2 = 2 * g * H2_matrix
    # Note: Factor of 2 might differ depending on Hamiltonian definition conventions.
    # Check the source theory or derivation if results seem off by a factor of 2.
    dynamical_matrix_TwogH2: sp.Matrix = 2 * g_metric_tensor_sym * H2_matrix

    # Create Ud_rotation_matrix matrix: block diagonal matrix of spin rotation operators rotation_matrices[i]
    # Ud_rotation_matrix transforms [Sx0,Sy0,Sz0, Sx1,Sy1,Sz1, ...] from local to global coordinates.
    # Size (3*nspins, 3*nspins).
    Ud_rotation_matrix_blocks: List[sp.Matrix] = []
    for i in range(nspins):
        # Ensure rotation_matrices[i] is a sympy Matrix if it contains symbols
        rot_mat = rotation_matrices[i]
        if isinstance(rot_mat, np.ndarray):
            rot_mat_sym: sp.Matrix = sp.Matrix(rot_mat)
        else:
            rot_mat_sym = rot_mat  # Assume it's already a sympy Matrix
        Ud_rotation_matrix_blocks.append(rot_mat_sym)

    # Construct the block diagonal matrix using sympy.diag
    Ud_rotation_matrix: sp.Matrix = sp.diag(
        *Ud_rotation_matrix_blocks
    )  # Size (3*nspins, 3*nspins)

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
    print("Error: Mismatch in degeneracy handling (check_degeneracy).")
    print(f"  q = {q_vector}")
    print(f"  Eigenvalue indices involved: {index - degeneracy_count} to {index}")
    print(f"  Degeneracy count (ndeg + 1): {degeneracy_count + 1}")
    # Use the constant here
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

    # Metric tensor g = diag(1, ..., 1, -1, ..., -1)
    G_metric_tensor: npt.NDArray[np.float_] = np.diag(
        np.concatenate([np.ones(nspins), -np.ones(nspins)])
    )

    # --- Diagonalization for +q ---
    try:
        eigvals_p: npt.NDArray[np.complex_]
        eigvecs_p: npt.NDArray[np.complex_]
        eigvals_p, eigvecs_p = la.eig(
            Hmat_plus_q
        )  # Eigenvalues eigvals_p, Right eigenvectors eigvecs_p (columns)
    except np.linalg.LinAlgError as e:
        print(f"Error: Eigenvalue calculation failed for +q = {q_vector}: {e}")
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Sorting eigenvalues and eigenvectors:
    sort_indices_p: npt.NDArray[np.int_] = eigvals_p.argsort()  # Initial sort
    eigvecs_p_tmp1: npt.NDArray[np.complex_] = eigvecs_p[:, sort_indices_p][
        :, nspins : 2 * nspins
    ]  # Assumed positive energy vectors
    eigvals_p_tmp1: npt.NDArray[np.complex_] = eigvals_p[sort_indices_p][
        nspins : 2 * nspins
    ]  # Assumed positive energy values
    eigvecs_p_tmp2: npt.NDArray[np.complex_] = eigvecs_p[:, sort_indices_p][
        :, 0:nspins
    ]  # Assumed negative energy vectors
    eigvals_p_tmp2: npt.NDArray[np.complex_] = eigvals_p[sort_indices_p][
        0:nspins
    ]  # Assumed negative energy values
    sort_indices_p_neg: npt.NDArray[np.int_] = (
        np.abs(eigvals_p_tmp2)
    ).argsort()  # Sort negative branch by magnitude
    eigvecs_p_tmp3: npt.NDArray[np.complex_] = eigvecs_p_tmp2[:, sort_indices_p_neg]
    eigvals_p_tmp3: npt.NDArray[np.complex_] = eigvals_p_tmp2[sort_indices_p_neg]
    # Combine: positive energies first, then negative energies sorted by magnitude
    eigenvalues_plus_q_sorted: npt.NDArray[np.complex_] = np.concatenate(
        (eigvals_p_tmp1, eigvals_p_tmp3)
    )
    eigenvectors_plus_q_sorted: npt.NDArray[np.complex_] = np.hstack(
        (eigvecs_p_tmp1, eigvecs_p_tmp3)
    )  # Shape (2*nspins, 2*nspins)

    # Gram-Schmidt Orthogonalization for degenerate subspaces [+q]
    degeneracy_count: int = 0
    for i in range(1, 2 * nspins):
        # Use constant
        if abs(eigenvalues_plus_q_sorted[i] - eigenvalues_plus_q_sorted[i - 1]) < dEdeg:
            degeneracy_count += 1
        elif degeneracy_count > 0:
            # Orthogonalize the degenerate block found [i-degeneracy_count-1 : i]
            vec_block: npt.NDArray[np.complex_] = eigenvectors_plus_q_sorted[
                :, i - degeneracy_count - 1 : i
            ]
            orthonormal_vecs: npt.NDArray[np.complex_] = gram_schmidt(vec_block)
            if orthonormal_vecs.shape[1] == vec_block.shape[1]:
                eigenvectors_plus_q_sorted[:, i - degeneracy_count - 1 : i] = (
                    orthonormal_vecs
                )
            else:  # Handle rank deficiency case if necessary
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
                ] = 0  # Zero out remaining columns
            degeneracy_count = 0
    # Handle degeneracy at the end of the array
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
    # inv_bogoliubov_T_plus_q = eigenvectors_plus_q_sorted * alpha_matrix_plus_q
    try:
        inv_eigenvectors_plus_q: npt.NDArray[np.complex_] = la.inv(
            eigenvectors_plus_q_sorted
        )
        alpha_sq_diag_p: npt.NDArray[np.float_] = np.zeros(2 * nspins, dtype=float)
        for i in range(2 * nspins):
            alpha_sq_diag_p[i] = np.real(
                np.dot(
                    np.conj(inv_eigenvectors_plus_q[i, :]),
                    G_metric_tensor @ inv_eigenvectors_plus_q[i, :],
                )
            )

        # Handle potential small negative values due to numerical noise before sqrt
        alpha_sq_diag_p[alpha_sq_diag_p < 0] = 0
        alpha_diag_p: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag_p)
        # Use constant
        alpha_diag_p[np.abs(alpha_diag_p) < zero_tol] = 0  # Truncate small values
        alpha_matrix_plus_q: npt.NDArray[np.complex_] = np.diag(alpha_diag_p).astype(
            np.complex_
        )  # Shape (2*nspins, 2*nspins)
    except np.linalg.LinAlgError:
        print(
            f"Error: Matrix inversion failed for +q = {q_vector}. Eigenvector matrix might be singular."
        )
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Prepare swapped eigenvector matrix for matching with -q results
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
    sort_indices_m_neg: npt.NDArray[np.int_] = (
        abs(eigvals_m_tmp2)
    ).argsort()  # Sort negative branch by magnitude
    eigvecs_m_tmp3: npt.NDArray[np.complex_] = eigvecs_m_tmp2[:, sort_indices_m_neg]
    eigvals_m_tmp3: npt.NDArray[np.complex_] = eigvals_m_tmp2[sort_indices_m_neg]
    eigenvalues_minus_q_sorted: npt.NDArray[np.complex_] = np.concatenate(
        (eigvals_m_tmp1, eigvals_m_tmp3)
    )
    eigenvectors_minus_q_sorted: npt.NDArray[np.complex_] = np.hstack(
        (eigvecs_m_tmp1, eigvecs_m_tmp3)
    )  # Shape (2*nspins, 2*nspins)

    # Gram-Schmidt Orthogonalization [-q]
    degeneracy_count = 0
    for i in range(1, 2 * nspins):
        # Use constant
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
        # Use constant
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
    # Goal: Ensure consistency between T(+q) and T(-q) for calculating K and Kd
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
    )  # Keep track if needed
    alpha_matrix_minus_q_reordered: npt.NDArray[np.complex_] = np.zeros_like(
        alpha_matrix_minus_q, dtype=complex
    )

    matched_indices_m: set[int] = (
        set()
    )  # Keep track of matched columns in eigenvectors_minus_q_sorted
    num_matched_vectors: int = 0

    # Loop through +q eigenvectors (columns of eigenvectors_plus_q_sorted)
    for i in range(2 * nspins):
        best_match_j: int = -1
        max_proj_metric: float = -1.0
        vec_i_plus_q: npt.NDArray[np.complex_] = eigenvectors_plus_q_sorted[:, i]
        vec_i_norm_sq: float = np.real(np.dot(np.conj(vec_i_plus_q), vec_i_plus_q))

        # Use constant
        if vec_i_norm_sq < zero_tol**2:
            continue  # Skip zero vectors

        # Loop through potential matching -q eigenvectors (columns of eigenvectors_minus_q_swapped_conj)
        for j in range(2 * nspins):
            if j in matched_indices_m:
                continue  # Already matched this -q vector

            vec_j_minus_q_swapped_conj: npt.NDArray[np.complex_] = (
                eigenvectors_minus_q_swapped_conj[:, j]
            )
            vec_j_norm_sq: float = np.real(
                np.dot(np.conj(vec_j_minus_q_swapped_conj), vec_j_minus_q_swapped_conj)
            )
            # Use constant
            if vec_j_norm_sq < zero_tol**2:
                continue

            # Calculate projection magnitude squared | <vec_i_plus_q | vec_j_minus_q_swapped_conj> |^2
            projection: complex = np.dot(
                np.conj(vec_i_plus_q), vec_j_minus_q_swapped_conj
            )
            projection_mag_sq: float = np.abs(projection) ** 2
            # Normalize projection magnitude squared for comparison (cosine squared)
            normalized_projection_mag_sq: float = projection_mag_sq / (
                vec_i_norm_sq * vec_j_norm_sq
            )

            # Check if this is the best match so far for vector i
            # Use constant for matching threshold check
            if (
                normalized_projection_mag_sq > max_proj_metric
                and normalized_projection_mag_sq > (1.0 - match_tol**2)
            ):
                max_proj_metric = normalized_projection_mag_sq
                best_match_j = j

        # If a good match was found for vector i
        if best_match_j != -1:
            matched_indices_m.add(best_match_j)
            num_matched_vectors += 1

            # Determine the target index for reordering based on original code's logic
            target_index: int = -1
            if i < nspins:  # Corresponds to positive energy branch of +q
                target_index = (
                    i + nspins
                )  # Place in the second block (dagger part) of reordered -q matrix
            else:  # Corresponds to negative energy branch of +q
                target_index = (
                    i - nspins
                )  # Place in the first block (non-dagger part) of reordered -q matrix

            if target_index != -1:
                # Assign the matched eigenvector from -q calculation to the target position
                eigenvectors_minus_q_reordered[:, target_index] = (
                    eigenvectors_minus_q_sorted[:, best_match_j]
                )
                eigenvalues_minus_q_reordered[target_index] = (
                    eigenvalues_minus_q_sorted[best_match_j]
                )

                # Adjust the phase of the corresponding alpha_matrix_minus_q diagonal element
                # Find first non-zero element for phase comparison (original approach)
                # Use constant
                idx_nonzero_i: npt.NDArray[np.int_] = np.where(
                    np.abs(vec_i_plus_q) > zero_tol
                )[0]
                idx_nonzero_j: npt.NDArray[np.int_] = np.where(
                    np.abs(vec_j_minus_q_swapped_conj) > zero_tol
                )[0]

                if len(idx_nonzero_i) > 0 and len(idx_nonzero_j) > 0:
                    # Calculate phase factor based on first non-zero elements
                    phase_factor: complex = (
                        vec_i_plus_q[idx_nonzero_i[0]]
                        / vec_j_minus_q_swapped_conj[idx_nonzero_j[0]]
                    )
                    # Apply phase adjustment to the diagonal element of alpha_matrix_minus_q
                    alpha_matrix_minus_q_reordered[target_index, target_index] = (
                        np.conj(alpha_matrix_plus_q[i, i] * phase_factor)
                    )
                else:  # Handle cases where one vector might be zero (shouldn't happen if norms > 0)
                    alpha_matrix_minus_q_reordered[target_index, target_index] = 0
            else:
                print(
                    f"Warning: Invalid target index during matching for i={i}, q={q_vector}"
                )
        else:
            # No match found for vec_i_plus_q - this indicates a problem
            print(
                f"Warning: No matching eigenvector found for +q vector index {i} at q={q_vector}"
            )

    # Check if all vectors were matched
    if num_matched_vectors != 2 * nspins:
        print(
            f"Warning: Number of matched vectors ({num_matched_vectors}) does not equal 2*nspins at q={q_vector}"
        )

    # Use reordered matrices for -q results
    eigenvectors_minus_q_final = eigenvectors_minus_q_reordered
    alpha_matrix_minus_q_final = alpha_matrix_minus_q_reordered
    # Use constant
    alpha_matrix_minus_q_final[np.abs(alpha_matrix_minus_q_final) < zero_tol] = (
        0  # Truncate small values
    )

    # --- Calculate K and Kd matrices ---
    # Inverse Bogoliubov transformation matrix T^-1 = eigenvector_matrix * alpha_matrix
    inv_bogoliubov_T_plus_q: npt.NDArray[np.complex_] = (
        eigenvectors_plus_q_sorted @ alpha_matrix_plus_q
    )
    inv_bogoliubov_T_minus_q_reordered: npt.NDArray[np.complex_] = (
        eigenvectors_minus_q_final @ alpha_matrix_minus_q_final
    )

    # Matrix relating local Sxy to boson operators c, cd based on spin_ops_local definition
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
        # Sz component is ignored in linear spin wave theory for S_alpha calculation

    # Calculate K and Kd matrices
    # S_global = Ud_numeric * S_local
    # S_local ~ sqrt(spin_magnitude/2) * Udd_local_boson_map * X (where X = [c0..N-1, cd0..N-1])
    # X = T^-1 * Y (where Y = [b_k0..N-1, b_{-k}^dagger0..N-1])
    # S_global = Ud_numeric * sqrt(spin_magnitude/2) * Udd_local_boson_map * T^-1 * Y
    prefactor: float = np.sqrt(spin_magnitude / 2.0)
    K_matrix: npt.NDArray[np.complex_] = (
        prefactor * Ud_numeric @ Udd_local_boson_map @ inv_bogoliubov_T_plus_q
    )
    Kd_matrix: npt.NDArray[np.complex_] = (
        prefactor
        * Ud_numeric
        @ Udd_local_boson_map
        @ inv_bogoliubov_T_minus_q_reordered
    )  # Uses matched invTm

    # Truncate small values using constant
    K_matrix[np.abs(K_matrix) < zero_tol] = 0
    Kd_matrix[np.abs(Kd_matrix) < zero_tol] = 0

    # Return K, Kd, and the sorted eigenvalues from the +q calculation
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

    # Calculate intensity for each mode
    for mode_index in range(nspins):
        spin_correlation_matrix: npt.NDArray[np.complex_] = np.zeros(
            (3, 3), dtype=complex
        )
        intensity_one_mode: complex = 0.0 + 0.0j

        # Calculate S_alpha,beta(q, omega_l) using K and Kd
        for alpha in range(3):  # Global coord index (x=0, y=1, z=2)
            for beta in range(3):  # Global coord index
                correlation_sum: complex = 0.0 + 0.0j
                for spin_i in range(nspins):  # Spin index in unit cell
                    for spin_j in range(nspins):  # Spin index in unit cell
                        idx_K: int = 3 * spin_i + alpha
                        idx_Kd: int = 3 * spin_j + beta
                        # Assumes S_alpha,beta ~ K[alpha_i, l] * Kd[beta_j, l+nspins] convention
                        correlation_sum += (
                            K_matrix[idx_K, mode_index]
                            * Kd_matrix[idx_Kd, mode_index + nspins]
                        )
                spin_correlation_matrix[alpha, beta] = correlation_sum

        # Apply geometric polarization factor
        q_norm_sq: float = np.dot(q_vector, q_vector)
        # Use constant
        if q_norm_sq < Q_ZERO_THRESHOLD:
            # At q=0, polarization factor is delta_ab, sum trace(SS)
            for alpha in range(3):
                intensity_one_mode += spin_correlation_matrix[alpha, alpha]
        else:
            q_normalized: npt.NDArray[np.float_] = q_vector / np.sqrt(q_norm_sq)
            # Sum over (delta_ab - qa*qb) * S_ab
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

    # Calculate eigenvalues
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
        energies = eigenvalues_sorted_real[
            nspins:
        ]  # Assume positive energies are the upper half
        # Check if the correct number of energies was obtained
        if len(energies) != nspins:
            print(
                f"Warning: Unexpected number of positive energies ({len(energies)}) found for q={q_vector}. Expected {nspins}."
            )
            # Handle mismatch (e.g., pad with NaN)
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
        HMat_sym, _ = process_matrix(
            cache_mode, k_sym, S_sym, params_sym, cache_file_base
        )
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to get symbolic matrix HMat: {e}")
        return None

    # Substitute numerical values
    param_substitutions: List[Tuple[sp.Symbol, float]] = [
        (S_sym, spin_magnitude)
    ] + list(zip(params_sym, hamiltonian_params))
    HMat_num_sym: sp.Matrix
    try:
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
    # Execute in parallel
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
