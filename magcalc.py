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
    k: List[sp.Symbol], S: sp.Symbol, params: List[sp.Symbol]
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
        k (List[sp.Symbol]): List of sympy symbols representing momentum components ([kx, ky, kz]).
                             Typically `sympy.symbols('kx ky kz', real=True)`.
        S (sp.Symbol): Symbolic representation of the spin magnitude
                       (e.g., `sympy.Symbol('S', real=True)`).
        params (List[sp.Symbol]): List of sympy symbols representing Hamiltonian parameters
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
        - The `mpr()` function from `spin_model.py` may depend on `params`.
        - The `Hamiltonian()` function from `spin_model.py` defines the interactions.
        - The `spin_interactions()` function likely returns interaction matrices needed
          to determine which pairs interact, used for setting up the Fourier transform.
        - The symbolic calculations, especially substitutions involving large
          Hamiltonians or many spins, can be very time-consuming. Results are
          typically cached using `process_matrix`.
        - The definition `Sabn_local` uses a specific form of Holstein-Primakoff
          related to `c = (Sx_local + iSy_local)/sqrt(2S)`.
        - The Fourier transform implementation assumes specific relationships between
          real-space and momentum-space operators.
        - The `XdX` substitution trick is used to extract coefficients of non-commutative
          operator products using SymPy's commutative coefficient extraction.
    """
    apos: npt.NDArray[np.float_] = sm.atom_pos()  # Assuming positions are floats
    nspins: int = len(apos)
    apos_ouc: npt.NDArray[np.float_] = sm.atom_pos_ouc()
    nspins_ouc: int = len(apos_ouc)
    print("Number of spins in the unit cell: ", nspins)

    # generate boson spin operators in local coordinate system,
    # with Z as quantization axis (Holstein-Primakoff, linear approx)
    c: List[sp.Symbol] = sp.symbols("c0:%d" % nspins_ouc, commutative=False)
    cd: List[sp.Symbol] = sp.symbols("cd0:%d" % nspins_ouc, commutative=False)
    # Note: S+/- = sqrt(S/2)*(c+cd) and sqrt(S/2)*(c-cd)/I differs slightly from
    # standard S+ = sqrt(2S)a, S- = sqrt(2S)adagger. This definition implies
    # c = (Sx_local + iSy_local)/sqrt(2S), cd = (Sx_local - iSy_local)/sqrt(2S) up to rotation.
    Sabn_local: List[sp.Matrix] = [
        sp.Matrix(
            (
                sp.sqrt(S / 2) * (c[i] + cd[i]),  # Sx_local
                sp.sqrt(S / 2) * (c[i] - cd[i]) / I,  # Sy_local
                S - cd[i] * c[i],
            )
        )  # Sz_local (linear approx: S)
        for i in range(nspins_ouc)
    ]

    # rotate spin operators to global coordinates
    mp: List[Union[npt.NDArray, sp.Matrix]] = sm.mpr(
        params
    )  # the rotation matrices can depend on the Hamiltonian parameters
    # Sabn contains spin operators for all atoms needed (unit cell + neighbours)
    # ordered by unit cell index then spin index within cell
    Sabn: List[sp.Matrix] = [
        mp[j] * Sabn_local[nspins * i + j]
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]

    # generate the spin Hamiltonian using the model definition
    HM: sp.Expr = sm.Hamiltonian(Sabn, params)
    HM = sp.expand(HM)
    # Extract terms based on powers of S.
    # This part seems intended to separate classical and quantum parts, potentially
    # keeping only terms up to quadratic in boson operators (linear spin wave).
    # The exact logic might depend on how Hamiltonian() is defined.
    # Assuming it correctly isolates the relevant quadratic boson terms for LSWT.
    HM_S0: sp.Expr = HM.coeff(S, 0)
    # params[-1] is assumed to be magnetic field H; ensure this convention is followed.
    # The line below seems to reconstruct the Hamiltonian from coefficients,
    # potentially simplifying or selecting terms. Needs careful check against theory.
    # Original line: HM = HM_S0.coeff(params[-1]) * params[-1] + HM.coeff(S ** 1.0) * S + HM.coeff(S) * S + HM.coeff(S ** 2) * S ** 2
    # Simplified assumption: Keep terms linear and quadratic in S (relevant for LSWT)
    # Let's assume the original code correctly extracts the quadratic boson part.
    # The S**1.0 and S might be redundant if S is a symbol.
    HM = (
        HM_S0.coeff(params[-1]) * params[-1]
        + HM.coeff(S, 1) * S
        + HM.coeff(S, 2) * S**2
    )
    HM = sp.expand(HM)

    # Define momentum-space boson operators
    ck: List[sp.Symbol] = [
        sp.Symbol("ck%d" % j, commutative=False)
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]
    ckd: List[sp.Symbol] = [
        sp.Symbol("ckd%d" % j, commutative=False)
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]
    cmk: List[sp.Symbol] = [
        sp.Symbol("cmk%d" % j, commutative=False)
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]
    cmkd: List[sp.Symbol] = [
        sp.Symbol("cmkd%d" % j, commutative=False)
        for i in range(int(nspins_ouc / nspins))
        for j in range(nspins)
    ]

    # Generate dictionary for Fourier transform substitution
    # This maps real-space bilinear boson terms to momentum-space terms
    Jex: npt.NDArray = sm.spin_interactions(params)[
        0
    ]  # Assumes spin_interactions returns interaction matrix (numpy array)
    fourier_dict: List[List[sp.Expr]] = [
        ent
        for i in range(nspins)  # Index for spin within the reference unit cell
        for j in range(
            nspins_ouc
        )  # Index for potentially interacting spin (incl. neighbours)
        if Jex[i, j] != 0  # Only consider interacting pairs defined in spin_model
        for dr in [apos[i, :] - apos_ouc[j, :]]  # Displacement vector r_i - r_j
        for k_dot_dr in [k[0] * dr[0] + k[1] * dr[1] + k[2] * dr[2]]  # k . (ri - rj)
        for ent in [
            # cd_i * cd_j -> (ckd_i*cmkd_j*exp(-ikdr) + cmkd_i*ckd_j*exp(ikdr))/2
            [
                cd[i] * cd[j],
                1
                / 2
                * (
                    ckd[i] * cmkd[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                    + cmkd[i] * ckd[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            # c_i * c_j -> (ck_i*cmk_j*exp(ikdr) + cmk_i*ck_j*exp(-ikdr))/2
            [
                c[i] * c[j],
                1
                / 2
                * (
                    ck[i] * cmk[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                    + cmk[i] * ck[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            # cd_i * c_j -> (ckd_i*ck_j*exp(-ikdr) + cmkd_i*cmk_j*exp(ikdr))/2
            [
                cd[i] * c[j],
                1
                / 2
                * (
                    ckd[i] * ck[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                    + cmkd[i] * cmk[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            # c_i * cd_j -> (ck_i*ckd_j*exp(ikdr) + cmk_i*cmkd_j*exp(-ikdr))/2
            [
                c[i] * cd[j],
                1
                / 2
                * (
                    ck[i] * ckd[j] * sp.exp(I * k_dot_dr).rewrite(sp.sin)
                    + cmk[i] * cmkd[j] * sp.exp(-I * k_dot_dr).rewrite(sp.sin)
                ),
            ],
            # On-site term: cd_j * c_j -> (ckd_j*ck_j + cmkd_j*cmk_j)/2
            # This assumes j is in the reference cell for the simple form.
            # The loop structure needs to correctly handle indices i, j relative to cells.
            # Assuming the original implementation correctly maps indices.
            [cd[j] * c[j], 1 / 2 * (ckd[j] * ck[j] + cmkd[j] * cmk[j])],
        ]
    ]
    # Note: The Fourier transform rules used here might need verification against
    # standard LSWT literature depending on the exact definition of c, cd, ck, cmk.

    # Apply commutation relations to bring terms into standard quadratic form:
    # ckd*ck, cmk*cmkd, cmk*ck, ckd*cmkd
    comm_dict: List[List[sp.Expr]] = (
        [
            [
                ck[i] * ckd[j],
                ckd[j] * ck[i] + (1 if i == j else 0),
            ]  # [ck_i, ckd_j] = delta_ij
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [
                cmkd[i] * cmk[j],
                cmk[j] * cmkd[i] + (1 if i == j else 0),
            ]  # [cmk_j, cmkd_i] = delta_ij -> cmkd*cmk = cmk*cmkd + delta
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [ck[i] * cmk[j], cmk[j] * ck[i]]  # [ck_i, cmk_j] = 0
            for i in range(nspins)
            for j in range(nspins)
        ]
        + [
            [cmkd[i] * ckd[j], ckd[j] * cmkd[i]]  # [cmkd_i, ckd_j] = 0
            for i in range(nspins)
            for j in range(nspins)
        ]
    )

    # Create basis vector X = [ck0..ckN-1, cmkd0..cmkdN-1] (N=nspins)
    # and Xd = [ckd0..ckdN-1, cmk0..cmkN-1] (dagger)
    # The quadratic Hamiltonian should be H = const + Xd H2 X
    Xd: List[sp.Symbol] = ckd[:nspins] + cmk[:nspins]  # Basis vector (dagger part)
    X: List[sp.Symbol] = ck[:nspins] + cmkd[:nspins]  # Basis vector

    # Use commutative placeholders (XdX) to extract coefficients of non-commutative products.
    # XdX_ij represents the placeholder for the product Xd[i] * X[j].
    XdX: List[sp.Symbol] = [
        sp.Symbol("XdX%d" % (i * 2 * nspins + j), commutative=True)
        for i in range(2 * nspins)
        for j in range(2 * nspins)
    ]
    # Substitution rule: Replace non-commutative product Xd[i]*X[j] with commutative XdX_ij
    XdX_subs: List[List[sp.Expr]] = [
        [Xd[i] * X[j], XdX[i * 2 * nspins + j]]
        for i in range(2 * nspins)
        for j in range(2 * nspins)
    ]

    # Perform substitutions using multiprocessing
    print("Running substitution ...")
    st: float = timeit.default_timer()

    # 1. Substitute Fourier transform rules into the original Hamiltonian HM
    HM_terms: List[sp.Expr] = HM.as_ordered_terms()
    pool_args_ft = [(expr, fourier_dict) for expr in HM_terms]
    with Pool() as pool:
        results_ft: List[sp.Expr] = list(
            tqdm(
                pool.imap(substitute_expr, pool_args_ft),
                total=len(HM_terms),
                desc="Substituting FT ",
                bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
            )
        )
    HMk: sp.Expr = Add(*results_ft)  # Summing up terms after substitution
    HMk = HMk.expand()  # Expand after summing

    # 2. Substitute commutation relation rules to order terms
    HMk_terms: List[sp.Expr] = HMk.as_ordered_terms()
    pool_args_comm = [(expr, comm_dict) for expr in HMk_terms]
    with Pool() as pool:
        results_comm: List[sp.Expr] = list(
            tqdm(
                pool.imap(substitute_expr, pool_args_comm),
                total=len(HMk_terms),
                desc="Applying Commutation",
                bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
            )
        )
    HMk_comm: sp.Expr = Add(*results_comm)
    HMk_comm = HMk_comm.expand()  # Expand after summing

    # 3. Substitute commutative placeholders (XdX) for final coefficient extraction
    HMk_comm_terms: List[sp.Expr] = HMk_comm.as_ordered_terms()
    pool_args_xdx = [(expr, XdX_subs) for expr in HMk_comm_terms]
    with Pool() as pool:
        results_xdx: List[sp.Expr] = list(
            tqdm(
                pool.imap(substitute_expr, pool_args_xdx),
                total=len(HMk_comm_terms),
                desc="Substituting XdX ",
                bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
            )
        )
    HMk_comm_XdX: sp.Expr = Add(*results_xdx)
    # No need to expand again if XdX symbols are the highest level

    et: float = timeit.default_timer()
    print(f"Run-time for substitution: {np.round((et - st) / 60, 2)} min.")

    # Extract coefficients of the placeholder symbols XdX_ij to form the H2 matrix
    # H2[i, j] = coefficient of Xd[i] * X[j] (represented by XdX_ij)
    H2_matrix_elements: sp.Matrix = sp.Matrix([HMk_comm_XdX.coeff(x) for x in XdX])
    H2: sp.Matrix = sp.Matrix(2 * nspins, 2 * nspins, H2_matrix_elements)

    # Define the metric tensor g = diag(1, ..., 1, -1, ..., -1) used in Bogoliubov transformation
    # g = np.bmat([[np.eye(nspins), np.zeros((nspins, nspins))],
    #              [np.zeros((nspins, nspins)), -np.eye(nspins)]]) # Original numpy construction
    g_sp: sp.Matrix = sp.diag(*([1] * nspins + [-1] * nspins))  # Sympy equivalent

    # The matrix to be diagonalized via Bogoliubov transformation is TwogH2 = 2 * g * H2
    # Note: Factor of 2 might differ depending on Hamiltonian definition conventions.
    # Check the source theory or derivation if results seem off by a factor of 2.
    TwogH2: sp.Matrix = 2 * g_sp * H2

    # Create Ud matrix: block diagonal matrix of spin rotation operators mp[i]
    # Ud transforms [Sx0,Sy0,Sz0, Sx1,Sy1,Sz1, ...] from local to global coordinates.
    # Size (3*nspins, 3*nspins).
    # The original construction was complex; using sp.diag is simpler.
    Ud_blocks: List[sp.Matrix] = []
    for i in range(nspins):
        # Ensure mp[i] is a sympy Matrix if it contains symbols
        mp_i = mp[i]
        if isinstance(mp_i, np.ndarray):
            mp_sym: sp.Matrix = sp.Matrix(mp_i)
        else:
            mp_sym = mp_i  # Assume it's already a sympy Matrix
        Ud_blocks.append(mp_sym)

    # Construct the block diagonal matrix using sympy.diag
    Ud: sp.Matrix = sp.diag(*Ud_blocks)  # Size (3*nspins, 3*nspins)

    return TwogH2, Ud


def gram_schmidt(x: npt.NDArray[np.complex_]) -> npt.NDArray[np.complex_]:
    """
    Performs Gram-Schmidt orthogonalization on the columns of a matrix using QR decomposition.

    This provides a numerically stable way to obtain an orthonormal basis for the
    column space of the input matrix `x`.

    Args:
        x (npt.NDArray[np.complex_]): Input matrix whose columns are vectors to be orthogonalized.
                                      Shape (M, N), where N is the number of vectors.

    Returns:
        npt.NDArray[np.complex_]: Matrix with orthonormal columns spanning the same space as x.
                                  Shape (M, K), where K <= N is the rank of x.
                                  `np.linalg.qr` mode='reduced' ensures the output has shape (M, K).
    """
    # np.linalg.qr handles potential linear dependence more robustly than manual GS.
    # mode='reduced' ensures the output Q has orthonormal columns spanning the column space of x.
    q, r = np.linalg.qr(x, mode="reduced")
    return q


def check_degeneracy(
    q: npt.NDArray[np.float_],
    i: int,
    ndeg: int,
    tmpsum_abs: npt.NDArray[np.float_],
    e_valm: npt.NDArray[np.complex_],
    newevec: npt.NDArray[np.complex_],
) -> NoReturn:
    """
    Helper function to print diagnostic information and exit if an inconsistency
    is found during degeneracy handling in KKdMatrix (original implementation).

    This typically indicates a mismatch between the expected number of vectors
    found via projection and the calculated degeneracy count. This function might
    be removed or modified if the degeneracy handling in KKdMatrix is refactored.

    Args:
        q (npt.NDArray[np.float_]): Momentum vector [qx, qy, qz] where the issue occurred.
        i (int): Index within the eigenvalue/vector loop where the issue occurred.
        ndeg (int): Detected degeneracy count (number of nearly equal eigenvalues).
        tmpsum_abs (npt.NDArray[np.float_]): Absolute values of projections used in the check.
        e_valm (npt.NDArray[np.complex_]): Array of eigenvalues being processed.
        newevec (npt.NDArray[np.complex_]): Subset of eigenvectors identified via projection.

    Raises:
        SystemExit: Terminates the program execution.
    """
    print("Error: Mismatch in degeneracy handling (check_degeneracy).")
    print(f"  q = {q}")
    print(f"  Eigenvalue indices involved: {i - ndeg} to {i}")
    print(f"  Degeneracy count (ndeg + 1): {ndeg + 1}")
    print(f"  Projection results (abs): {tmpsum_abs}")
    print(f"  Eigenvalues in range: {np.real(e_valm[i - ndeg : i + 1])}")
    print(f"  Number of vectors found by projection: {newevec.shape[1]}")
    print("The program will exit...")
    sys.exit()


def KKdMatrix(
    Sp: float,
    Hkp: npt.NDArray[np.complex_],
    Hkm: npt.NDArray[np.complex_],
    Ud: npt.NDArray[np.complex_],
    q: npt.NDArray[np.float_],
    nspins: int,
) -> Tuple[
    npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]
]:
    """
    Calculates transformation matrices K, Kd and eigenvalues for LSWT.

    Performs Bogoliubov diagonalization of the quadratic Hamiltonian H = Xd*H2*X
    for both +q (Hkp = 2gH2(+q)) and -q (Hkm = 2gH2(-q)). It handles eigenvalue
    sorting, degeneracy using Gram-Schmidt orthogonalization, and crucially matches
    eigenvectors between +q and -q calculations to ensure correct phase relationships
    for calculating scattering intensities.

    The matrices K and Kd relate the global spin components S = [Sx0, Sy0, Sz0, ...]
    to the Bogoliubov quasiparticle operators Y = [b_k0..N-1, b_{-k}dagger0..N-1] such that
    S_alpha(q) = sum_l ( K[alpha, l] * b_l(q) + Kd[alpha, l] * b_l^dagger(-q) ).
    (Note: Indexing and exact relation depends on the derivation conventions).

    Args:
        Sp (float): Numerical value of the spin magnitude.
        Hkp (npt.NDArray[np.complex_]): Numerical Hamiltonian matrix TwogH2 evaluated at +q.
                                        Shape (2*nspins, 2*nspins).
        Hkm (npt.NDArray[np.complex_]): Numerical Hamiltonian matrix TwogH2 evaluated at -q.
                                        Shape (2*nspins, 2*nspins).
        Ud (npt.NDArray[np.complex_]): Numerical block rotation matrix (from local to global).
                                       Shape (3*nspins, 3*nspins).
        q (npt.NDArray[np.float_]): Momentum vector q = [qx, qy, qz]. Shape (3,).
        nspins (int): Number of spins in the magnetic unit cell.

    Returns:
        Tuple[npt.NDArray[np.complex_], npt.NDArray[np.complex_], npt.NDArray[np.complex_]]:
            A tuple containing:
            K (npt.NDArray[np.complex_]): Transformation matrix relating global spin operators
                                          to Bogoliubov creation/annihilation operators.
                                          Shape (3*nspins, 2*nspins).
            Kd (npt.NDArray[np.complex_]): Transformation matrix, related to K and the -q calculation.
                                           Shape (3*nspins, 2*nspins).
            e_val (npt.NDArray[np.complex_]): Sorted eigenvalues from the +q diagonalization.
                                              Contains 2*nspins values, typically ordered with
                                              positive energies first. Shape (2*nspins,).

    Notes:
        - The diagonalization `la.eig(Hkp)` solves Hkp * v = w * v.
        - The sorting logic assumes eigenvalues come in +/- pairs and aims to place
          the positive energy branches first.
        - Gram-Schmidt is applied to handle numerical degeneracies.
        - The 'alpha' matrix (`al`, `alm`) relates the eigenvector matrix `v` to the
          inverse Bogoliubov transformation matrix `T^-1 = v * alpha`. It ensures
          proper normalization: `T^-1 G (T^-1)^dagger = G`.
        - The eigenvector matching between +q and -q is crucial for phase consistency
          when calculating K and Kd, which are needed for S(q,w). The matching relies
          on the property that eigenvectors of H(-k) are related to the swapped complex
          conjugate of eigenvectors of H(k).
        - The `Udd` matrix relates local spin components (Sx, Sy) to the local
          boson operators (c, cd) used in the Holstein-Primakoff definition.
        - Tolerances (`dEdeg`, `torr`, `zero_tol`, `match_tol`) are used for numerical comparisons.
          These might need adjustment depending on the problem's numerical stability.
    """
    dEdeg: float = 1e-12  # Degeneracy threshold for eigenvalues
    torr: float = 1e-5  # Tolerance for projection checks (original use)
    zero_tol: float = 1e-6  # Tolerance for setting small matrix elements to zero
    match_tol: float = 1e-5  # Tolerance for eigenvector matching projection

    # Metric tensor g = diag(1, ..., 1, -1, ..., -1)
    G: npt.NDArray[np.float_] = np.diag(
        np.concatenate([np.ones(nspins), -np.ones(nspins)])
    )

    # --- Diagonalization for +q ---
    try:
        w: npt.NDArray[np.complex_]
        v: npt.NDArray[np.complex_]
        w, v = la.eig(Hkp)  # Eigenvalues w, Right eigenvectors v (columns)
    except np.linalg.LinAlgError as e:
        print(f"Error: Eigenvalue calculation failed for +q = {q}: {e}")
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Sorting eigenvalues and eigenvectors:
    es: npt.NDArray[np.int_] = w.argsort()
    vtmp1: npt.NDArray[np.complex_] = v[:, es][:, nspins : 2 * nspins]
    etmp1: npt.NDArray[np.complex_] = w[es][nspins : 2 * nspins]
    vtmp2: npt.NDArray[np.complex_] = v[:, es][:, 0:nspins]
    etmp2: npt.NDArray[np.complex_] = w[es][0:nspins]
    ess: npt.NDArray[np.int_] = (np.abs(etmp2)).argsort()
    vtmp3: npt.NDArray[np.complex_] = vtmp2[:, ess]
    etmp3: npt.NDArray[np.complex_] = etmp2[ess]
    e_val: npt.NDArray[np.complex_] = np.concatenate((etmp1, etmp3))
    e_vec: npt.NDArray[np.complex_] = np.hstack((vtmp1, vtmp3))

    # Gram-Schmidt Orthogonalization for degenerate subspaces [+q]
    ndeg: int = 0
    for i in range(1, 2 * nspins):
        if abs(e_val[i] - e_val[i - 1]) < dEdeg:
            ndeg += 1
        elif ndeg > 0:
            v_block: npt.NDArray[np.complex_] = e_vec[:, i - ndeg - 1 : i]
            q_ortho: npt.NDArray[np.complex_] = gram_schmidt(v_block)
            if q_ortho.shape[1] == v_block.shape[1]:
                e_vec[:, i - ndeg - 1 : i] = q_ortho
            else:
                print(
                    f"Warning: Rank deficiency detected during GS for +q at index {i}, q={q}"
                )
                e_vec[:, i - ndeg - 1 : i - ndeg - 1 + q_ortho.shape[1]] = q_ortho
                e_vec[:, i - ndeg - 1 + q_ortho.shape[1] : i] = 0
            ndeg = 0
    if ndeg > 0:
        v_block = e_vec[:, 2 * nspins - 1 - ndeg : 2 * nspins]
        q_ortho = gram_schmidt(v_block)
        if q_ortho.shape[1] == v_block.shape[1]:
            e_vec[:, 2 * nspins - 1 - ndeg : 2 * nspins] = q_ortho
        else:
            print(
                f"Warning: Rank deficiency detected during GS for +q at end of array, q={q}"
            )
            e_vec[
                :, 2 * nspins - 1 - ndeg : 2 * nspins - 1 - ndeg + q_ortho.shape[1]
            ] = q_ortho
            e_vec[:, 2 * nspins - 1 - ndeg + q_ortho.shape[1] : 2 * nspins] = 0

    # Determine Alpha matrix (normalization factors) [+q]
    try:
        Td: npt.NDArray[np.complex_] = la.inv(e_vec)
        alpha_sq_diag: npt.NDArray[np.float_] = np.zeros(2 * nspins, dtype=float)
        for i in range(2 * nspins):
            alpha_sq_diag[i] = np.real(np.dot(np.conj(Td[i, :]), G @ Td[i, :]))

        alpha_sq_diag[alpha_sq_diag < 0] = 0
        al_diag: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag)
        al_diag[np.abs(al_diag) < zero_tol] = 0
        al: npt.NDArray[np.complex_] = np.diag(al_diag).astype(
            np.complex_
        )  # Keep complex
    except np.linalg.LinAlgError:
        print(
            f"Error: Matrix inversion failed for +q = {q}. Eigenvector matrix might be singular."
        )
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Prepare swapped eigenvector matrix for matching with -q results
    evecswap: npt.NDArray[np.complex_] = np.conj(
        np.vstack((e_vec[nspins : 2 * nspins, :], e_vec[0:nspins, :]))
    )

    # --- Diagonalization for -q ---
    try:
        wm: npt.NDArray[np.complex_]
        vm: npt.NDArray[np.complex_]
        wm, vm = la.eig(Hkm)
    except np.linalg.LinAlgError as e:
        print(f"Error: Eigenvalue calculation failed for -q = {-q}: {e}")
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # Apply the same sorting logic as for +q
    esm: npt.NDArray[np.int_] = wm.argsort()
    vtmpm1: npt.NDArray[np.complex_] = vm[:, esm][:, nspins : 2 * nspins]
    etmpm1: npt.NDArray[np.complex_] = wm[esm][nspins : 2 * nspins]
    vtmpm2: npt.NDArray[np.complex_] = vm[:, esm][:, 0:nspins]
    etmpm2: npt.NDArray[np.complex_] = wm[esm][0:nspins]
    essm: npt.NDArray[np.int_] = (abs(etmpm2)).argsort()
    vtmpm3: npt.NDArray[np.complex_] = vtmpm2[:, essm]
    etmpm3: npt.NDArray[np.complex_] = etmpm2[essm]
    e_valm: npt.NDArray[np.complex_] = np.concatenate((etmpm1, etmpm3))
    e_vecm: npt.NDArray[np.complex_] = np.hstack((vtmpm1, vtmpm3))

    # Gram-Schmidt Orthogonalization [-q]
    ndeg = 0
    for i in range(1, 2 * nspins):
        if abs(e_valm[i] - e_valm[i - 1]) < dEdeg:
            ndeg += 1
        elif ndeg > 0:
            v_block = e_vecm[:, i - ndeg - 1 : i]
            q_ortho = gram_schmidt(v_block)
            if q_ortho.shape[1] == v_block.shape[1]:
                e_vecm[:, i - ndeg - 1 : i] = q_ortho
            else:
                print(
                    f"Warning: Rank deficiency detected during GS for -q at index {i}, q={q}"
                )
                e_vecm[:, i - ndeg - 1 : i - ndeg - 1 + q_ortho.shape[1]] = q_ortho
                e_vecm[:, i - ndeg - 1 + q_ortho.shape[1] : i] = 0
            ndeg = 0
    if ndeg > 0:
        v_block = e_vecm[:, 2 * nspins - 1 - ndeg : 2 * nspins]
        q_ortho = gram_schmidt(v_block)
        if q_ortho.shape[1] == v_block.shape[1]:
            e_vecm[:, 2 * nspins - 1 - ndeg : 2 * nspins] = q_ortho
        else:
            print(
                f"Warning: Rank deficiency detected during GS for -q at end of array, q={q}"
            )
            e_vecm[
                :, 2 * nspins - 1 - ndeg : 2 * nspins - 1 - ndeg + q_ortho.shape[1]
            ] = q_ortho
            e_vecm[:, 2 * nspins - 1 - ndeg + q_ortho.shape[1] : 2 * nspins] = 0

    # Determine alpha matrix [-q]
    try:
        Tdm: npt.NDArray[np.complex_] = la.inv(e_vecm)
        alpha_sq_diag_m: npt.NDArray[np.float_] = np.zeros(2 * nspins, dtype=float)
        for i in range(2 * nspins):
            alpha_sq_diag_m[i] = np.real(np.dot(np.conj(Tdm[i, :]), G @ Tdm[i, :]))

        alpha_sq_diag_m[alpha_sq_diag_m < 0] = 0
        alm_diag: npt.NDArray[np.float_] = np.sqrt(alpha_sq_diag_m)
        alm_diag[np.abs(alm_diag) < zero_tol] = 0
        alm: npt.NDArray[np.complex_] = np.diag(alm_diag).astype(np.complex_)
    except np.linalg.LinAlgError:
        print(
            f"Error: Matrix inversion failed for -q = {-q}. Eigenvector matrix might be singular."
        )
        nan_matrix = np.full((3 * nspins, 2 * nspins), np.nan, dtype=np.complex_)
        nan_eigs = np.full((2 * nspins,), np.nan, dtype=np.complex_)
        return nan_matrix, nan_matrix, nan_eigs

    # --- Eigenvector Matching between +q and -q ---
    evecmswap: npt.NDArray[np.complex_] = np.conj(
        np.vstack((e_vecm[nspins : 2 * nspins, :], e_vecm[0:nspins, :]))
    )
    tmpm: npt.NDArray[np.complex_] = np.zeros_like(e_vecm, dtype=complex)
    tmpevalm: npt.NDArray[np.complex_] = np.zeros_like(e_valm, dtype=complex)
    tmpalm: npt.NDArray[np.complex_] = np.zeros_like(alm, dtype=complex)

    matched_indices_m: set[int] = set()
    Ntmpm: int = 0

    for i in range(2 * nspins):
        best_match_j: int = -1
        max_proj_metric: float = -1.0
        vec_i: npt.NDArray[np.complex_] = e_vec[:, i]
        vec_i_norm_sq: float = np.real(np.dot(np.conj(vec_i), vec_i))

        if vec_i_norm_sq < zero_tol**2:
            continue

        for j in range(2 * nspins):
            if j in matched_indices_m:
                continue

            vec_j_swap_conj: npt.NDArray[np.complex_] = evecmswap[:, j]
            vec_j_norm_sq: float = np.real(
                np.dot(np.conj(vec_j_swap_conj), vec_j_swap_conj)
            )
            if vec_j_norm_sq < zero_tol**2:
                continue

            proj: complex = np.dot(np.conj(vec_i), vec_j_swap_conj)
            proj_mag_sq: float = np.abs(proj) ** 2
            norm_proj_mag_sq: float = proj_mag_sq / (vec_i_norm_sq * vec_j_norm_sq)

            if norm_proj_mag_sq > max_proj_metric and norm_proj_mag_sq > (
                1.0 - match_tol**2
            ):
                max_proj_metric = norm_proj_mag_sq
                best_match_j = j

        if best_match_j != -1:
            matched_indices_m.add(best_match_j)
            Ntmpm += 1
            target_index: int = -1
            if i < nspins:
                target_index = i + nspins
            else:
                target_index = i - nspins

            if target_index != -1:
                tmpm[:, target_index] = e_vecm[:, best_match_j]
                tmpevalm[target_index] = e_valm[
                    best_match_j
                ]  # Note: Original code used e_valm, check if intended

                idx_nz_i: npt.NDArray[np.int_] = np.where(np.abs(vec_i) > zero_tol)[0]
                idx_nz_j: npt.NDArray[np.int_] = np.where(
                    np.abs(vec_j_swap_conj) > zero_tol
                )[0]

                if len(idx_nz_i) > 0 and len(idx_nz_j) > 0:
                    phase_factor: complex = (
                        vec_i[idx_nz_i[0]] / vec_j_swap_conj[idx_nz_j[0]]
                    )
                    tmpalm[target_index, target_index] = np.conj(
                        al[i, i] * phase_factor
                    )
                else:
                    tmpalm[target_index, target_index] = 0
            else:
                print(f"Warning: Invalid target index during matching for i={i}, q={q}")
        else:
            print(
                f"Warning: No matching eigenvector found for +q vector index {i} at q={q}"
            )

    if Ntmpm != 2 * nspins:
        print(
            f"Warning: Number of matched vectors ({Ntmpm}) does not equal 2*nspins at q={q}"
        )

    e_vecm = tmpm
    alm = tmpalm
    alm[np.abs(alm) < zero_tol] = 0

    # --- Calculate K and Kd matrices ---
    invT: npt.NDArray[np.complex_] = e_vec @ al
    invTm: npt.NDArray[np.complex_] = e_vecm @ alm

    Udd: npt.NDArray[np.complex_] = np.zeros((3 * nspins, 2 * nspins), dtype=complex)
    for i in range(nspins):
        Udd[3 * i, i] = 1.0
        Udd[3 * i, i + nspins] = 1.0
        Udd[3 * i + 1, i] = 1.0 / I
        Udd[3 * i + 1, i + nspins] = -1.0 / I

    prefactor: float = np.sqrt(Sp / 2.0)
    K: npt.NDArray[np.complex_] = prefactor * Ud @ Udd @ invT
    Kd: npt.NDArray[np.complex_] = prefactor * Ud @ Udd @ invTm

    K[np.abs(K) < zero_tol] = 0
    Kd[np.abs(Kd) < zero_tol] = 0

    return K, Kd, e_val


def process_matrix(
    rd_or_wr: str, k: List[sp.Symbol], S: sp.Symbol, params: List[sp.Symbol], file: str
) -> Tuple[sp.Matrix, sp.Matrix]:
    """
    Manages the generation or loading of the symbolic Hamiltonian matrix (HMat=TwogH2)
    and rotation matrix (Ud) from cache files (.pck).

    Ensures the 'pckFiles' directory exists. If 'w' mode is selected, it calls
    `gen_HM` to generate the symbolic matrices and saves them using pickle.
    If 'r' mode is selected, it loads the matrices from the specified cache files.

    Args:
        rd_or_wr (str): Mode selector:
                        'r' to read from cache files.
                        'w' to generate matrices and write to cache files.
        k (List[sp.Symbol]): List of sympy symbols for momentum ([kx, ky, kz]).
        S (sp.Symbol): Symbolic spin magnitude (e.g., `sympy.Symbol('S')`).
        params (List[sp.Symbol]): List of sympy symbols for Hamiltonian parameters (e.g., `[p0, p1]`).
        file (str): Base filename used for the cache files (e.g., 'my_model').
                    Cache files will be named 'pckFiles/{file}_HM.pck' and
                    'pckFiles/{file}_Ud.pck'.

    Returns:
        Tuple[sp.Matrix, sp.Matrix]: A tuple containing:
            HMat (sp.Matrix): The symbolic TwogH2 matrix.
            Ud (sp.Matrix): The symbolic Ud matrix.

    Raises:
        ValueError: If `rd_or_wr` is not 'r' or 'w'.
        FileNotFoundError: If 'r' mode is selected and cache files are not found.
        pickle.UnpicklingError: If cache files are corrupted or incompatible.
        Exception: Other potential errors during file I/O or pickling.

    Notes:
        - The cache files store the *symbolic* matrices generated by `gen_HM`.
        - Using 'r' significantly speeds up subsequent runs after the initial
          generation ('w'), as `gen_HM` can be very slow.
        - **Important:** If `spin_model.py` or the definition of `S` or `params`
          used in `gen_HM` changes, the cache files become invalid. You *must*
          run with `rd_or_wr='w'` again or manually delete the old `.pck` files.
          (Future improvement: Add hashing to detect changes automatically).
    """
    hm_cache_file: str = os.path.join("pckFiles", file + "_HM.pck")
    ud_cache_file: str = os.path.join("pckFiles", file + "_Ud.pck")
    HMat: sp.Matrix
    Ud: sp.Matrix

    if rd_or_wr == "w":
        print("Generating symbolic matrices (HMat, Ud)...")
        HMat, Ud = gen_HM(k, S, params)
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

    elif rd_or_wr == "r":
        print(
            f"Importing symbolic matrices from cache files ({hm_cache_file}, {ud_cache_file})..."
        )
        try:
            with open(hm_cache_file, "rb") as inHM:
                HMat = pickle.load(inHM)
            with open(ud_cache_file, "rb") as inUd:
                Ud = pickle.load(inUd)
        except FileNotFoundError:
            print(
                f"Error: Cache files not found. Run with 'w' option first or check filename '{file}'."
            )
            raise
        except (pickle.UnpicklingError, EOFError) as e:
            print(f"Error loading cache files (may be corrupted or incompatible): {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred loading cache files: {e}")
            raise
    else:
        raise ValueError(f"Invalid mode '{rd_or_wr}'. Use 'r' (read) or 'w' (write).")

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

    Takes the symbolic Hamiltonian (with parameters substituted), numerical Ud matrix,
    and a specific momentum vector q. It evaluates the Hamiltonian at +q and -q,
    calls KKdMatrix to perform diagonalization and get transformation matrices,
    and then calculates the neutron scattering intensity S(q,w) for each mode,
    including the geometric polarization factor.

    Args:
        args (Tuple): A tuple containing:
            HMat_sym (sp.Matrix): Symbolic TwogH2 matrix (already substituted with
                                  numerical parameters, depends only on kx, ky, kz).
            Ud_num (npt.NDArray[np.complex_]): Numerical Ud matrix (float or complex).
                                               Shape (3*nspins, 3*nspins).
            k_sym (List[sp.Symbol]): List of sympy symbols for momentum ([kx, ky, kz]).
            q_vec (npt.NDArray[np.float_]): Specific momentum vector q = [qx, qy, qz] for this
                                            calculation. Shape (3,).
            nspins (int): Number of spins in the magnetic unit cell.
            Sp_num (float): Numerical value of the spin magnitude.

    Returns:
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]:
            A tuple containing:
            qout (npt.NDArray[np.float_]): The input momentum vector q_vec. Shape (3,).
            En (npt.NDArray[np.float_]): Calculated magnon energies (real part of positive
                                         eigenvalues from KKdMatrix) for this q. Shape (nspins,).
                                         Contains NaN if calculation failed.
            Sqwout (npt.NDArray[np.float_]): Calculated scattering intensity (real part)
                                             for each mode at this q. Shape (nspins,).
                                             Contains NaN if calculation failed.

    Notes:
        - Uses `lambdify` to create a fast numerical function from the symbolic HMat.
        - Calls `KKdMatrix` which performs the complex diagonalization and matching.
        - Calculates the spin-spin correlation function `SS[alpha, beta]` for each mode `l`.
        - Applies the neutron scattering geometric polarization factor `(delta_ab - q_a*q_b)`.
        - Warns if the imaginary part of the calculated S(q,w) for any mode exceeds 1e-4.
        - Returns real part of S(q,w) as intensity should be real.
    """
    HMat_sym, Ud_num, k_sym, q_vec, nspins, Sp_num = args
    Sqwout0: npt.NDArray[np.complex_] = np.zeros(nspins, dtype=complex)
    nan_en: npt.NDArray[np.float_] = np.full((nspins,), np.nan)
    nan_sqw: npt.NDArray[np.float_] = np.full((nspins,), np.nan)

    try:
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        print(f"Error during lambdify at q={q_vec}: {e}")
        return q_vec, nan_en, nan_sqw

    try:
        Hkp: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vec[0], q_vec[1], q_vec[2]), dtype=np.complex128
        )
        Hkm: npt.NDArray[np.complex_] = np.array(
            HMat_func(-q_vec[0], -q_vec[1], -q_vec[2]), dtype=np.complex128
        )
    except Exception as e:
        print(f"Error evaluating HMat function at q={q_vec}: {e}")
        return q_vec, nan_en, nan_sqw

    K: npt.NDArray[np.complex_]
    Kd: npt.NDArray[np.complex_]
    evals: npt.NDArray[np.complex_]
    K, Kd, evals = KKdMatrix(Sp_num, Hkp, Hkm, Ud_num, q_vec, nspins)

    if np.isnan(K).any() or np.isnan(Kd).any() or np.isnan(evals).any():
        print(
            f"Warning: NaN encountered in KKdMatrix result for q={q_vec}. Skipping intensity calculation."
        )
        return q_vec, nan_en, nan_sqw

    imag_energy_mag: npt.NDArray[np.float_] = np.abs(np.imag(evals[0:nspins]))
    if np.any(imag_energy_mag > 1e-5):
        print(
            f"Warning: Significant imaginary part in energy eigenvalues for q={q_vec}. Max imag: {np.max(imag_energy_mag)}"
        )
    En: npt.NDArray[np.float_] = np.real(evals[0:nspins])
    qout: npt.NDArray[np.float_] = q_vec

    for l in range(nspins):
        SS: npt.NDArray[np.complex_] = np.zeros((3, 3), dtype=complex)
        Sqw_mode: complex = 0.0 + 0.0j

        for alpha in range(3):
            for beta in range(3):
                correlation_sum: complex = 0.0 + 0.0j
                for i in range(nspins):
                    for j in range(nspins):
                        idx_K: int = 3 * i + alpha
                        idx_Kd: int = 3 * j + beta
                        correlation_sum += K[idx_K, l] * Kd[idx_Kd, l + nspins]
                SS[alpha, beta] = correlation_sum

        q_norm_sq: float = np.dot(q_vec, q_vec)
        if q_norm_sq < 1e-10:
            for alpha in range(3):
                Sqw_mode += SS[alpha, alpha]
        else:
            qt: npt.NDArray[np.float_] = q_vec / np.sqrt(q_norm_sq)
            for alpha in range(3):
                for beta in range(3):
                    delta_ab: float = 1.0 if alpha == beta else 0.0
                    polarization_factor: float = delta_ab - qt[alpha] * qt[beta]
                    Sqw_mode += polarization_factor * SS[alpha, beta]

        if np.abs(np.imag(Sqw_mode)) > 1e-4:
            print(
                f"Warning: Significant imaginary part in Sqw for q={q_vec}, mode {l}: {np.imag(Sqw_mode)}"
            )

        Sqwout0[l] = Sqw_mode

    Sqwout: npt.NDArray[np.float_] = np.real(Sqwout0)
    Sqwout[Sqwout < 0] = 0

    return qout, En, Sqwout


def calc_Sqw(
    Sp: float,
    q: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    p: Union[List[float], npt.NDArray[np.float_]],
    file: str,
    rd_or_wr: str,
) -> Tuple[
    Optional[Tuple[npt.NDArray[np.float_], ...]],
    Optional[Tuple[npt.NDArray[np.float_], ...]],
    Optional[Tuple[npt.NDArray[np.float_], ...]],
]:
    """
    Calculates the dynamical structure factor S(q,w) over a list of q-points.

    Orchestrates the calculation by:
    1. Getting symbolic matrices HMat (TwogH2) and Ud using `process_matrix` (reads
       from cache 'r' or generates 'w').
    2. Substituting numerical values for spin magnitude (Sp) and Hamiltonian
       parameters (p) into the symbolic matrices.
    3. Using `multiprocessing.Pool` to parallelize the calculation for each
       momentum vector `q` in the input list by calling `process_calc_Sqw`.
    4. Collecting and returning the results.

    Args:
        Sp (float): Numerical value of the spin magnitude.
        q (Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]]):
            List or array of momentum vectors q to calculate. Each q should be
            a list or array [qx, qy, qz].
        p (Union[List[float], npt.NDArray[np.float_]]):
            List of numerical Hamiltonian parameter values. Order must match the
            symbolic parameters used in `spin_model.py` and `gen_HM`.
        file (str): Base filename for caching passed to `process_matrix`.
        rd_or_wr (str): Cache mode ('r' or 'w') passed to `process_matrix`.

    Returns:
        Tuple[Optional[Tuple[npt.NDArray[np.float_], ...]], Optional[Tuple[npt.NDArray[np.float_], ...]], Optional[Tuple[npt.NDArray[np.float_], ...]]]:
            A tuple containing:
            qout (Optional[Tuple[npt.NDArray[np.float_], ...]]): Tuple of the input momentum vectors q.
            En (Optional[Tuple[npt.NDArray[np.float_], ...]]): Tuple of numpy arrays containing calculated magnon energies.
            Sqwout (Optional[Tuple[npt.NDArray[np.float_], ...]]): Tuple of numpy arrays containing calculated intensities.
            Returns (None, None, None) if initial matrix loading or substitution fails.

    Raises:
        ValueError: If `spin_model.atom_pos()` returns an empty list or if `rd_or_wr`
                    is invalid (via `process_matrix`).
        FileNotFoundError: If `rd_or_wr='r'` and cache files are missing (via `process_matrix`).
        Exception: Can propagate errors from `process_matrix`, substitution, or multiprocessing.
    """
    print("Calculating scattering intensity S(q,w)...")
    nspins: int
    try:
        nspins = len(sm.atom_pos())
        if nspins == 0:
            raise ValueError(
                "spin_model.atom_pos() returned an empty list. Cannot proceed."
            )
    except AttributeError:
        raise AttributeError("Function 'atom_pos' not found in spin_model.py.")
    except Exception as e:
        raise RuntimeError(f"Error getting nspins from spin_model.atom_pos(): {e}")

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    k_sym: List[sp.Symbol] = [kx, ky, kz]
    S_sym: sp.Symbol = sp.Symbol("S", real=True)
    params_sym: List[sp.Symbol] = sp.symbols("p0:%d" % len(p), real=True)
    HMat_sym: sp.Matrix
    Ud_sym: sp.Matrix

    try:
        HMat_sym, Ud_sym = process_matrix(rd_or_wr, k_sym, S_sym, params_sym, file)
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to get symbolic matrices: {e}")
        return None, None, None

    param_subs: List[Tuple[sp.Symbol, float]] = [(S_sym, Sp)] + list(zip(params_sym, p))
    HMat_num_sym: sp.Matrix
    Ud_num_sym: sp.Matrix
    Ud_num: npt.NDArray[np.complex_]
    try:
        HMat_num_sym = HMat_sym.subs(param_subs, simultaneous=True).evalf()
        Ud_num_sym = Ud_sym.subs(param_subs, simultaneous=True).evalf()
        Ud_num = np.array(Ud_num_sym, dtype=np.complex128)
    except Exception as e:
        print(f"Error during substitution of numerical parameters: {e}")
        return None, None, None

    print("Running diagonalization and intensity calculation...")
    st: float = timeit.default_timer()

    pool_args: List[Tuple] = [
        (HMat_num_sym, Ud_num, k_sym, q_vec, nspins, Sp) for q_vec in q
    ]
    results: List[
        Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_], npt.NDArray[np.float_]]
    ] = []
    try:
        with Pool() as pool:
            results = list(
                tqdm(
                    pool.imap(process_calc_Sqw, pool_args),
                    total=len(q),
                    desc="Calculating S(q,w)",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
    except Exception as e:
        print(f"Error during multiprocessing execution: {e}")
        return None, None, None

    qout_t: Tuple[npt.NDArray[np.float_], ...]
    En_t: Tuple[npt.NDArray[np.float_], ...]
    Sqwout_t: Tuple[npt.NDArray[np.float_], ...]
    try:
        if not results or len(results[0]) != 3:
            raise ValueError("Multiprocessing returned empty or malformed results.")
        qout_t, En_t, Sqwout_t = zip(*results)
    except ValueError as e:
        print(f"Error unpacking results from parallel processing: {e}")
        return None, None, None  # Indicate failure more clearly

    et: float = timeit.default_timer()
    print(f"Run-time for S(q,w) calculation: {np.round((et - st) / 60, 2)} min.")

    return qout_t, En_t, Sqwout_t


def process_calc_disp(
    args: Tuple[sp.Matrix, List[sp.Symbol], npt.NDArray[np.float_], int],
) -> npt.NDArray[np.float_]:
    """
    Worker function for multiprocessing calculation of dispersion for a single q-point.

    Takes the symbolic Hamiltonian (with parameters substituted) and a specific
    momentum vector q. It evaluates the Hamiltonian at q, diagonalizes it using
    `scipy.linalg.eigvals`, sorts the eigenvalues, and returns the positive energy
    branches (magnon energies).

    Args:
        args (Tuple): A tuple containing:
            HMat_sym (sp.Matrix): Symbolic TwogH2 matrix (already substituted with
                                  numerical parameters, depends only on kx, ky, kz).
            k_sym (List[sp.Symbol]): List of sympy symbols for momentum ([kx, ky, kz]).
            q_vec (npt.NDArray[np.float_]): Specific momentum vector q = [qx, qy, qz] for this
                                            calculation. Shape (3,).
            nspins (int): Number of spins in the magnetic unit cell.

    Returns:
        npt.NDArray[np.float_]: Calculated magnon energies (real part of sorted positive
                                eigenvalues) for this q. Shape (nspins,). Returns array
                                of NaNs if calculation fails.

    Notes:
        - Uses `lambdify` for fast numerical evaluation of the Hamiltonian matrix.
        - Uses `scipy.linalg.eigvals` which is generally faster than `eig` if only
          eigenvalues are needed.
        - Sorts the eigenvalues by real part. Assumes the upper half correspond to
          the positive physical magnon energies.
        - Warns if any eigenvalue has an imaginary part with magnitude greater than 1e-5.
    """
    HMat_sym, k_sym, q_vec, nspins = args
    nan_energies: npt.NDArray[np.float_] = np.full((nspins,), np.nan)

    try:
        HMat_func = lambdify(k_sym, HMat_sym, modules=["numpy"])
    except Exception as e:
        print(f"Error during lambdify at q={q_vec}: {e}")
        return nan_energies

    try:
        HMat_k: npt.NDArray[np.complex_] = np.array(
            HMat_func(q_vec[0], q_vec[1], q_vec[2]), dtype=np.complex128
        )
    except Exception as e:
        print(f"Error evaluating HMat function at q={q_vec}: {e}")
        return nan_energies

    eigval: npt.NDArray[np.complex_]
    try:
        eigval = la.eigvals(HMat_k)
    except np.linalg.LinAlgError:
        print(f"Error: Eigenvalue calculation failed for q={q_vec}.")
        return nan_energies

    imag_part_mags: npt.NDArray[np.float_] = np.abs(np.imag(eigval))
    if np.any(imag_part_mags > 1e-5):
        print(
            f"Warning: Significant imaginary part in eigenvalues for q={q_vec}. Max imag: {np.max(imag_part_mags)}"
        )

    energies: npt.NDArray[np.float_]
    try:
        eigval_sorted_real: npt.NDArray[np.float_] = np.real(np.sort(eigval))
        energies = eigval_sorted_real[nspins:]
        if len(energies) != nspins:
            print(
                f"Warning: Unexpected number of positive energies ({len(energies)}) found for q={q_vec}. Expected {nspins}."
            )
            if len(energies) > nspins:
                energies = energies[:nspins]
            else:  # Pad with NaN if fewer than expected energies found
                energies = np.pad(
                    energies, (0, nspins - len(energies)), constant_values=np.nan
                )
    except Exception as e:
        print(f"Error during eigenvalue sorting/selection for q={q_vec}: {e}")
        return nan_energies

    return energies


def calc_disp(
    Sp: float,
    q: Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]],
    p: Union[List[float], npt.NDArray[np.float_]],
    file: str,
    rd_or_wr: str,
) -> Optional[List[npt.NDArray[np.float_]]]:
    """
    Calculates the spin-wave dispersion relation (energy vs. momentum) over a list of q-points.

    Orchestrates the calculation by:
    1. Getting symbolic matrix HMat (TwogH2) using `process_matrix` (reads
       from cache 'r' or generates 'w'). Ud is also retrieved but not used here.
    2. Substituting numerical values for spin magnitude (Sp) and Hamiltonian
       parameters (p) into the symbolic HMat.
    3. Using `multiprocessing.Pool` to parallelize the calculation for each
       momentum vector `q` in the input list by calling `process_calc_disp`.
    4. Collecting and returning the calculated energies.

    Args:
        Sp (float): Numerical value of the spin magnitude.
        q (Union[List[npt.NDArray[np.float_]], npt.NDArray[np.float_]]):
            List or array of momentum vectors q to calculate. Each q should be
            a list or array [qx, qy, qz].
        p (Union[List[float], npt.NDArray[np.float_]]):
            List of numerical Hamiltonian parameter values. Order must match the
            symbolic parameters.
        file (str): Base filename for caching passed to `process_matrix`.
        rd_or_wr (str): Cache mode ('r' or 'w') passed to `process_matrix`.

    Returns:
        Optional[List[npt.NDArray[np.float_]]]:
            A list of numpy arrays. En[i] contains the calculated magnon
            energies (nspins values) for the i-th input momentum vector q[i].
            Returns None if initial matrix loading or substitution fails.
            Individual elements in the list can be NaN arrays if calculation
            failed for specific q-points.

    Raises:
        ValueError: If `spin_model.atom_pos()` returns an empty list or if `rd_or_wr`
                    is invalid (via `process_matrix`).
        FileNotFoundError: If `rd_or_wr='r'` and cache files are missing (via `process_matrix`).
        Exception: Can propagate errors from `process_matrix`, substitution, or multiprocessing.
    """
    print("Calculating magnon dispersion ...")
    nspins: int
    try:
        nspins = len(sm.atom_pos())
        if nspins == 0:
            raise ValueError(
                "spin_model.atom_pos() returned an empty list. Cannot proceed."
            )
    except AttributeError:
        raise AttributeError("Function 'atom_pos' not found in spin_model.py.")
    except Exception as e:
        raise RuntimeError(f"Error getting nspins from spin_model.atom_pos(): {e}")

    kx, ky, kz = sp.symbols("kx ky kz", real=True)
    k_sym: List[sp.Symbol] = [kx, ky, kz]
    S_sym: sp.Symbol = sp.Symbol("S", real=True)
    params_sym: List[sp.Symbol] = sp.symbols("p0:%d" % len(p), real=True)
    HMat_sym: sp.Matrix

    try:
        HMat_sym, _ = process_matrix(rd_or_wr, k_sym, S_sym, params_sym, file)
    except (ValueError, FileNotFoundError, Exception) as e:
        print(f"Failed to get symbolic matrix HMat: {e}")
        return None

    param_subs: List[Tuple[sp.Symbol, float]] = [(S_sym, Sp)] + list(zip(params_sym, p))
    HMat_num_sym: sp.Matrix
    try:
        HMat_num_sym = HMat_sym.subs(param_subs, simultaneous=True).evalf()
    except Exception as e:
        print(f"Error during substitution of numerical parameters: {e}")
        return None

    print("Running diagonalization ...")
    st: float = timeit.default_timer()

    pool_args: List[Tuple] = [(HMat_num_sym, k_sym, q_vec, nspins) for q_vec in q]
    En: List[npt.NDArray[np.float_]] = []
    try:
        with Pool() as pool:
            En = list(
                tqdm(
                    pool.imap(process_calc_disp, pool_args),
                    total=len(q),
                    desc="Calculating Dispersion",
                    bar_format="{percentage:3.0f}%|{bar}| {elapsed}<{remaining}",
                )
            )
    except Exception as e:
        print(f"Error during multiprocessing execution: {e}")
        # En will contain results up to the point of failure

    et: float = timeit.default_timer()
    print(f"Run-time for dispersion calculation: {np.round((et - st) / 60, 2)} min.")

    return En


# Example usage (add within an `if __name__ == "__main__":` block)
# if __name__ == "__main__":
#     # Define parameters (example)
#     spin_S_val: float = 1.0
#     hamiltonian_params_val: List[float] = [1.0, 0.1] # Example J, D
#     cache_file_base: str = "my_model_cache"
#     # Use 'w' for the very first run to generate cache files.
#     # Use 'r' for subsequent runs to load from cache.
#     cache_mode: str = 'r' # Or 'w'
#
#     # Define q points (example path Gamma-X-M-Gamma)
#     q_points_list: List[List[float]] = []
#     # Add specific q-vectors, e.g., from a path generator function
#     q_points_list.append([0.0, 0.0, 0.0])
#     q_points_list.append([np.pi, 0.0, 0.0])
#     q_points_list.append([np.pi, np.pi, 0.0])
#     q_points_list.append([0.0, 0.0, 0.0])
#     q_points_array: npt.NDArray[np.float_] = np.array(q_points_list)
#
#     # --- Calculate dispersion ---
#     energies_list: Optional[List[npt.NDArray[np.float_]]] = calc_disp(
#         spin_S_val, q_points_array, hamiltonian_params_val, cache_file_base, cache_mode
#     )
#
#     if energies_list is not None:
#         print("\nDispersion Energies:")
#         for i, q_vec in enumerate(q_points_array):
#             # Check if the result for this q-point is valid
#             if i < len(energies_list) and isinstance(energies_list[i], np.ndarray) and not np.isnan(energies_list[i]).any():
#                 print(f"q = {np.round(q_vec, 3)}: E = {np.round(energies_list[i], 4)}")
#             else:
#                 print(f"q = {np.round(q_vec, 3)}: Calculation failed.")
#         # Add plotting code here if desired
#     else:
#         print("Dispersion calculation failed to start.")
#
#     # --- Calculate S(q,w) ---
#     # Note: Ensure cache_mode='r' if dispersion was just calculated with 'w'
#     sqw_results: Tuple[Optional[Tuple[npt.NDArray[np.float_], ...]], Optional[Tuple[npt.NDArray[np.float_], ...]], Optional[Tuple[npt.NDArray[np.float_], ...]]]
#     sqw_results = calc_Sqw(
#         spin_S_val, q_points_array, hamiltonian_params_val, cache_file_base, cache_mode
#     )
#     q_vectors_out, energies_sqw, intensities = sqw_results
#
#     if q_vectors_out is not None and energies_sqw is not None and intensities is not None:
#         print("\nScattering Intensities:")
#         for i, q_vec in enumerate(q_vectors_out):
#              # Check if the result for this q-point is valid
#              if i < len(energies_sqw) and isinstance(energies_sqw[i], np.ndarray) and not np.isnan(energies_sqw[i]).any() and \
#                 i < len(intensities) and isinstance(intensities[i], np.ndarray) and not np.isnan(intensities[i]).any():
#                  print(f"q = {np.round(q_vec, 3)}: E = {np.round(energies_sqw[i], 4)}, S(q,w) = {np.round(intensities[i], 4)}")
#              else:
#                  print(f"q = {np.round(q_vec, 3)}: Calculation failed.")
#         # Add plotting code here if desired
#     else:
#          print("S(q,w) calculation failed to start.")
