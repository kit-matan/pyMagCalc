#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018
Modified on Mon 19 Feb 2024
Further revised based on plan for field-dependent LSWT.

@author: Kit Matan

This file contains the information about the spin structure for alpha-Cu2V2O7 (CVO)
that will be used to calculate spin-waves by MagCal.py.
It includes classical energy minimization for an applied magnetic field.
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize
from tqdm.auto import tqdm  # Added for progress bar
import logging

# Define logging for this module
logger = logging.getLogger(__name__)
if (
    not logger.hasHandlers()
):  # Avoid adding multiple handlers if imported multiple times
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Flag to ensure the integration message is logged only once per run
_spin_interactions_integration_logged = False


# Global 'al' list for CVO (16 spins)
al = [
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    -1,
]  # 1 along +a, -1 along -a

# Lattice parameters (replace with more precise values if available)
LA = 20.645
LB = 8.383
LC = 6.442


def unit_cell():
    """Defines the unit cell vectors."""
    va = np.array([LA, 0, 0])
    vb = np.array([0, LB, 0])
    vc = np.array([0, 0, LC])
    return np.array([va, vb, vc])


def atom_pos():
    """
    Returns Cartesian coordinates of the 16 Cu atoms in the CVO magnetic unit cell.
    Ensure these are the positions for the cell used in LSWT.
    """
    # Fractional coordinates from the original CVO structure
    # (These might need adjustment if your magnetic cell is different from crystallographic)
    x, y, z = 0.16572, 0.3646, 0.7545
    positions_frac = [
        [x, y, z],
        [1 - x, 1 - y, z],
        [x, y + 0.5, z - 0.5],
        [1 - x, -y + 0.5, z - 0.5],
        [x + 0.5, y, z - 0.5],
        [-x + 0.5, 1 - y, z - 0.5],
        [x + 0.5, y + 0.5, z],
        [-x + 0.5, -y + 0.5, z],
        [
            x + 0.25,
            -y + 0.25 + 1,
            z + 0.25,
        ],  # Corrected y for periodicity based on original
        [-x + 0.25, y + 0.25, z + 0.25],
        [x + 0.25, -y + 0.75, z + 0.75 - 1],  # Corrected z for periodicity
        [-x + 0.25, y + 0.75, z + 0.75 - 1],  # Corrected z for periodicity
        [x + 0.75, -y + 0.25 + 1, z + 0.75 - 1],  # Corrected y, z
        [-x + 0.75, y + 0.25, z + 0.75 - 1],  # Corrected z
        [x + 0.75, -y + 0.75, z + 0.25],
        [-x + 0.75, y + 0.75 - 1, z + 0.25],  # Corrected y
    ]
    r_pos = [
        np.array([pos[0] * LA, pos[1] * LB, pos[2] * LC]) for pos in positions_frac
    ]
    return np.array(r_pos)


def atom_pos_ouc():
    """Defines positions for spins in the original unit cell + neighbors (OUC)."""
    r_pos_ouc_list = []
    uc_vecs = unit_cell()
    apos_uc = atom_pos()
    r_pos_ouc_list.extend(apos_uc)  # Start with the primary unit cell
    # Add neighbors in a 3x3x3 block of unit cells
    for i_offset in range(-1, 2):
        for j_offset in range(-1, 2):
            for k_offset in range(-1, 2):
                if i_offset == 0 and j_offset == 0 and k_offset == 0:
                    continue  # Skip the primary cell itself, already added
                translation_vec = (
                    i_offset * uc_vecs[0]
                    + j_offset * uc_vecs[1]
                    + k_offset * uc_vecs[2]
                )
                r_pos_ouc_list.extend(apos_uc + translation_vec)
    return np.array(r_pos_ouc_list)


# Helper functions from spin_model.py, to be used by the new spin_interactions
def is_vector_parallel_to_another(vector1, vector2):
    """
    Checks if vector1 is parallel to vector2.
    """
    v1 = np.array(vector1, dtype=float)
    v2 = np.array(vector2, dtype=float)
    # Check if cross product is close to zero vector
    cross_product = np.cross(v1, v2)
    return np.allclose(cross_product, 0)


def is_right_neighbor(atom1, atom2):
    """
    Checks if atom2 is to the right of atom1.
    Exact replica of the function from spin_model.py.
    """
    if atom1[1] < atom2[1]:  # Check b-axis (y-coordinate)
        return True
    elif atom1[1] > atom2[1]:
        return False
    else:
        # If y-coordinates are equal, original spin_model.py had a problematic check.
        # For distinct atoms with same y, this condition implies they are not "right" or "left" in y.
        # If they are the same atom (already excluded by distance checks), it's also false.
        # The original ValueError was likely for an unexpected state.
        # To match original behavior if it ever returned under this branch:
        return False


def get_nearest_neighbor_distances(
    atom_positions, outside_unit_cell_atom_positions, num_neighbors=None
):
    """
    Return a list of unique non-zero distances, sorted.
    This version iterates all pairs to find unique distances, like in spin_model.py.
    """
    distances = []
    # Iterate through all spins in the unit cell and all spins in OUC
    for uc_atom_pos_single in atom_positions:
        for ouc_atom_pos_single in outside_unit_cell_atom_positions:
            distance = round(
                np.linalg.norm(uc_atom_pos_single - ouc_atom_pos_single),
                4,  # Rounding to 4 decimal places
            )
            if distance > 1e-4:  # Exclude self-interaction (distance = 0)
                if distance not in distances:
                    distances.append(distance)
    distances = sorted(distances)
    # Return enough distances for the original logic to select from.
    # The original spin_model.py took [1:4] from a list of up to 10.
    return distances[:num_neighbors] if num_neighbors is not None else distances


def spin_interactions(params_sym_or_num):
    """
    Defines interaction matrices J_ex, G_ex (anisotropic exchange), DM, and extracts H_field.
    params_sym_or_num: List of symbolic or numerical parameters [J1, J2, J3, G1, Dx, H_field].
    """
    if len(params_sym_or_num) != 6:
        raise ValueError("Expected 6 parameters: J1, J2, J3, G1, Dx, H_field")
    J1, J2, J3, G1, Dx, H_field = params_sym_or_num

    n_uc = len(atom_pos())
    n_ouc = len(atom_pos_ouc())

    Jex = sp.zeros(n_uc, n_ouc)
    Gex = sp.zeros(n_uc, n_ouc)
    is_symbolic = any(hasattr(p, "free_symbols") for p in params_sym_or_num)

    apos_uc = atom_pos()
    apos_ouc = atom_pos_ouc()
    uc_vectors = unit_cell()

    # Get all unique neighbor distances, sorted, by iterating all pairs.
    # Original spin_model.py used get_nearest_neighbor_distances(..., 10)[1:4]
    all_unique_distances = get_nearest_neighbor_distances(
        apos_uc, apos_ouc, num_neighbors=None  # Get all unique distances
    )

    if len(all_unique_distances) < 4:
        logger.warning(
            f"Found fewer than 4 unique neighbor distances: {all_unique_distances}. "
            "J1, J2, J3 assignments might be incorrect or incomplete based on original logic."
        )
        # Define default large distances if not enough unique distances are found
        dist_J1_val = (
            all_unique_distances[1] if len(all_unique_distances) > 1 else np.inf
        )
        dist_J2_val = (
            all_unique_distances[2] if len(all_unique_distances) > 2 else np.inf
        )
        dist_J3_val = (
            all_unique_distances[3] if len(all_unique_distances) > 3 else np.inf
        )
    else:
        # Select 2nd, 3rd, and 4th unique non-zero distances for J1, J2, J3 respectively
        dist_J1_val = all_unique_distances[1]
        dist_J2_val = all_unique_distances[2]
        dist_J3_val = all_unique_distances[3]

    logger.debug(
        f"Using distances for J1: {dist_J1_val:.4f}, J2: {dist_J2_val:.4f}, J3: {dist_J3_val:.4f}"
    )

    DM1_sym_vec = sp.Matrix([[Dx, 0, 0]])
    DMnull_sym_vec = sp.Matrix([[0, 0, 0]])

    if is_symbolic:
        DMmat_to_fill = sp.Matrix(
            [[DMnull_sym_vec for _ in range(n_ouc)] for _ in range(n_uc)]
        )
    else:
        DMmat_to_fill_np = np.zeros((n_uc, n_ouc, 3), dtype=float)

    for i in range(n_uc):
        for j_idx in range(n_ouc):
            if np.allclose(apos_uc[i], apos_ouc[j_idx]):
                continue

            dist_ij = la.norm(apos_uc[i] - apos_ouc[j_idx])

            current_dm_sym = DMnull_sym_vec
            current_dm_num = np.array([0.0, 0.0, 0.0])

            # Using np.round for distance comparison to match original spin_model.py
            if np.round(dist_ij, 2) == np.round(dist_J1_val, 2):
                Jex[i, j_idx] = J1
                Gex[i, j_idx] = G1

                bond_vector = apos_uc[i] - apos_ouc[j_idx]
                if is_vector_parallel_to_another(
                    bond_vector, [0, uc_vectors[1, 1], uc_vectors[2, 2]]
                ):
                    dm_sign_factor = -1.0
                else:
                    dm_sign_factor = 1.0

                if is_right_neighbor(apos_uc[i], apos_ouc[j_idx]):
                    dm_sign_factor *= -1.0

                if is_symbolic:
                    current_dm_sym = dm_sign_factor * DM1_sym_vec
                else:
                    current_dm_num = dm_sign_factor * np.array(
                        [float(Dx), 0, 0], dtype=float
                    )

            elif np.round(dist_ij, 2) == np.round(dist_J2_val, 2):
                Jex[i, j_idx] = J2
            elif np.round(dist_ij, 2) == np.round(dist_J3_val, 2):
                Jex[i, j_idx] = J3

            if is_symbolic:
                DMmat_to_fill[i, j_idx] = current_dm_sym
            else:
                DMmat_to_fill_np[i, j_idx, :] = current_dm_num

    global _spin_interactions_integration_logged
    if not _spin_interactions_integration_logged:
        logger.info(
            "CVO spin_interactions logic (adapted from spin_model.py) has been integrated into this module."
        )
        _spin_interactions_integration_logged = True

    DM_to_return = DMmat_to_fill if is_symbolic else DMmat_to_fill_np
    return Jex, Gex, DM_to_return, H_field


def Hamiltonian(Sabn_global_ouc, params_sym_or_num):
    """
    Constructs the symbolic spin Hamiltonian.
    Sabn_global_ouc: List of global spin operators (3x1 SymPy matrices) for OUC.
    params_sym_or_num: List of symbolic or numerical parameters.
    """
    n_uc = len(atom_pos())
    n_ouc = len(atom_pos_ouc())

    Jex, Gex, DM_values, H_field_val = spin_interactions(params_sym_or_num)
    is_symbolic_params = any(hasattr(p, "free_symbols") for p in params_sym_or_num)

    H_total = sp.S(0)

    for i in range(n_uc):
        for j_ouc_idx in range(n_ouc):
            if np.allclose(atom_pos()[i], atom_pos_ouc()[j_ouc_idx]):
                continue

            if Jex[i, j_ouc_idx] != 0:
                H_total += (
                    0.5
                    * Jex[i, j_ouc_idx]
                    * (Sabn_global_ouc[i].dot(Sabn_global_ouc[j_ouc_idx]))
                )

            Dx_ij, Dy_ij, Dz_ij = sp.S(0), sp.S(0), sp.S(0)
            if is_symbolic_params:
                current_dm_element = DM_values[i, j_ouc_idx]
                if (
                    isinstance(current_dm_element, sp.MatrixBase)
                    and not current_dm_element.is_zero_matrix
                ):
                    Dx_ij = current_dm_element[0, 0]
                    Dy_ij = current_dm_element[0, 1]
                    Dz_ij = current_dm_element[0, 2]
            else:
                if np.any(np.abs(DM_values[i, j_ouc_idx, :]) > 1e-9):
                    Dx_ij = DM_values[i, j_ouc_idx, 0]
                    Dy_ij = DM_values[i, j_ouc_idx, 1]
                    Dz_ij = DM_values[i, j_ouc_idx, 2]

            if not (Dx_ij == 0 and Dy_ij == 0 and Dz_ij == 0):
                Si, Sj = Sabn_global_ouc[i], Sabn_global_ouc[j_ouc_idx]
                H_total += 0.5 * (
                    Dx_ij * (Si[1] * Sj[2] - Si[2] * Sj[1])
                    + Dy_ij * (Si[2] * Sj[0] - Si[0] * Sj[2])
                    + Dz_ij * (Si[0] * Sj[1] - Si[1] * Sj[0])
                )

            if Gex[i, j_ouc_idx] != 0:
                H_total += (
                    0.5
                    * Gex[i, j_ouc_idx]
                    * (
                        Sabn_global_ouc[i][0] * Sabn_global_ouc[j_ouc_idx][0]
                        - Sabn_global_ouc[i][1] * Sabn_global_ouc[j_ouc_idx][1]
                        - Sabn_global_ouc[i][2] * Sabn_global_ouc[j_ouc_idx][2]
                    )
                )

    effective_H_coeff = (
        2.0 * 5.7883818066e-2
    )  # g*mu_B in meV/T, if H_field_val is H_Tesla
    for i in range(n_uc):
        H_total -= effective_H_coeff * Sabn_global_ouc[i][2] * H_field_val
    return H_total.expand()


# --- Classical Energy Minimization Functions ---
def classical_energy(
    theta_angles_rad, params_numerical, S_numerical, current_al_preference_list
):
    n_uc = len(atom_pos())
    n_ouc = len(atom_pos_ouc())

    if len(theta_angles_rad) != n_uc:
        logger.error(
            f"classical_energy: Mismatch in theta_angles_rad ({len(theta_angles_rad)}) and n_uc ({n_uc})"
        )
        return np.inf
    if len(current_al_preference_list) != n_uc:
        logger.error(
            f"classical_energy: Mismatch in al_preference ({len(current_al_preference_list)}) and n_uc ({n_uc})"
        )
        return np.inf

    Jex_num, Gex_num, DM_num_values, H_field_val_num = spin_interactions(
        params_numerical
    )
    if hasattr(Jex_num, "evalf"):
        Jex_num = np.array(Jex_num.evalf(), dtype=float)
    if hasattr(Gex_num, "evalf"):
        Gex_num = np.array(Gex_num.evalf(), dtype=float)

    classical_spins_uc = np.zeros((n_uc, 3))
    for i in range(n_uc):
        theta_i = theta_angles_rad[i]
        phi_i = 0.0 if current_al_preference_list[i] == 1 else np.pi
        classical_spins_uc[i, 0] = S_numerical * np.sin(theta_i) * np.cos(phi_i)
        classical_spins_uc[i, 1] = S_numerical * np.sin(theta_i) * np.sin(phi_i)
        classical_spins_uc[i, 2] = S_numerical * np.cos(theta_i)

    total_energy = 0.0
    apos_uc_coords = atom_pos()
    apos_ouc_coords = atom_pos_ouc()

    for i in range(n_uc):
        S_i = classical_spins_uc[i, :]
        for j_ouc in range(n_ouc):
            if np.allclose(apos_uc_coords[i], apos_ouc_coords[j_ouc]):
                continue

            j_uc_equivalent = j_ouc % n_uc
            S_j_ouc = classical_spins_uc[j_uc_equivalent, :]

            J_ij = float(Jex_num[i, j_ouc])
            G_ij = float(Gex_num[i, j_ouc])
            DM_ij_vec = DM_num_values[i, j_ouc, :]

            if abs(J_ij) > 1e-9:
                total_energy += 0.5 * J_ij * np.dot(S_i, S_j_ouc)

            if np.any(np.abs(DM_ij_vec) > 1e-9):
                Dx_c, Dy_c, Dz_c = DM_ij_vec[0], DM_ij_vec[1], DM_ij_vec[2]
                total_energy += 0.5 * (
                    Dx_c * (S_i[1] * S_j_ouc[2] - S_i[2] * S_j_ouc[1])
                    + Dy_c * (S_i[2] * S_j_ouc[0] - S_i[0] * S_j_ouc[2])
                    + Dz_c * (S_i[0] * S_j_ouc[1] - S_i[1] * S_j_ouc[0])
                )

            if abs(G_ij) > 1e-9:
                total_energy += (
                    0.5
                    * G_ij
                    * (S_i[0] * S_j_ouc[0] - S_i[1] * S_j_ouc[1] - S_i[2] * S_j_ouc[2])
                )

    effective_H_coeff = 2.0 * 5.7883818066e-2
    for i in range(n_uc):
        total_energy -= effective_H_coeff * classical_spins_uc[i, 2] * H_field_val_num

    return total_energy


def find_ground_state_orientations(
    params_numerical, S_numerical, initial_theta_guess_rad=None
):
    n_spins_uc = len(atom_pos())
    current_al = al

    options = {"maxiter": 5000, "ftol": 2.22e-09, "gtol": 1e-06, "disp": False}
    if initial_theta_guess_rad is None:
        H_field_val = params_numerical[-1]
        if abs(H_field_val) > 1e-9:
            initial_theta_guess_rad = np.full(
                n_spins_uc, np.pi / 2.0 - 0.1 * np.sign(H_field_val + 1e-12)
            )
        else:
            initial_theta_guess_rad = np.full(n_spins_uc, np.pi / 2.0)

    bounds_theta = [(1e-9, np.pi - 1e-9)] * n_spins_uc

    logger.info(
        f"Starting classical energy minimization for H={params_numerical[-1]:.4f} with {n_spins_uc} spins..."
    )
    logger.debug(f"Initial theta guess (radians): {initial_theta_guess_rad}")

    with tqdm(
        total=options.get("maxiter", 5000),
        desc="Classical Minimization (sm_cvo_Hc)",
        unit="iter",
        leave=False,
    ) as pbar:
        minimization_result = minimize(
            classical_energy,
            initial_theta_guess_rad,
            args=(params_numerical, S_numerical, current_al),
            method="L-BFGS-B",
            bounds=bounds_theta,
            tol=1e-8,
            options=options,
            callback=lambda xk: pbar.update(1),
        )

    if minimization_result.success:
        optimal_theta_angles_rad = np.clip(minimization_result.x, 0, np.pi)
        final_energy = minimization_result.fun
        logger.info(
            f"Classical minimization successful. Final energy: {final_energy:.6f} meV/formula_unit (approx). Iterations: {minimization_result.nit}."
        )
        optimal_theta_degrees = np.degrees(optimal_theta_angles_rad)
        canting_angles_degrees = 90.0 - optimal_theta_degrees
        logger.info(f"Optimal theta angles (degrees from +c): {optimal_theta_degrees}")
        logger.info(
            f"Canting angles from a-b plane (degrees towards +z): {canting_angles_degrees}"
        )
        return optimal_theta_angles_rad, final_energy
    else:
        logger.error(f"Classical minimization failed: {minimization_result.message}")
        logger.debug(f"Final function value: {minimization_result.fun}")
        logger.debug(f"Final jacobian: {minimization_result.jac}")
        return None, np.inf


# --- Magnetic Principal Axes Rotation Matrices (mpr) ---
def mpr(params_sym_or_num, S_val_for_mpr=None, field_optimized_theta_angles=None):
    n_spins = len(atom_pos())
    rot_matrices = []
    current_al_list = al
    is_symbolic = any(hasattr(p, "free_symbols") for p in params_sym_or_num)

    if S_val_for_mpr is None:
        S_val_for_mpr = sp.Symbol("S") if is_symbolic else 0.5

    for i in range(n_spins):
        if (
            field_optimized_theta_angles is not None
        ):  # Field applied, use optimized thetas
            if len(field_optimized_theta_angles) != n_spins:
                raise ValueError("field_optimized_theta_angles length mismatch.")
            theta_i_rad_eff = field_optimized_theta_angles[i]
            phi_i_rad_eff = 0.0 if current_al_list[i] == 1 else np.pi

            cos_theta = np.cos(theta_i_rad_eff)
            sin_theta = np.sin(theta_i_rad_eff)

            Ry_neg_theta_mat_num = np.array(
                [
                    [
                        cos_theta,
                        0,
                        -sin_theta,
                    ],
                    [0, 1, 0],
                    [sin_theta, 0, cos_theta],
                ]
            )

            if abs(phi_i_rad_eff - np.pi) < 1e-6:
                Rz_pi_mat_num = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
                rot_mat_final = sp.Matrix(Rz_pi_mat_num @ Ry_neg_theta_mat_num)
            else:
                rot_mat_final = sp.Matrix(Ry_neg_theta_mat_num)
        else:  # Zero-field case: Replicate spin_model.py's mpr logic: Ry(al[i]*pi/2)
            angle = current_al_list[i] * (sp.pi / 2.0 if is_symbolic else np.pi / 2.0)
            cos_angle = (
                sp.cos(angle)
                if is_symbolic or isinstance(angle, sp.Basic)
                else np.cos(angle)
            )
            sin_angle = (
                sp.sin(angle)
                if is_symbolic or isinstance(angle, sp.Basic)
                else np.sin(angle)
            )
            rot_mat_final = sp.Matrix(
                [[cos_angle, 0, sin_angle], [0, 1, 0], [-sin_angle, 0, cos_angle]]
            )
        rot_matrices.append(rot_mat_final)
    return rot_matrices


# Helper function to be called by sw_CVO.py
def get_field_optimized_state_for_lswt(numerical_params_list, S_numerical_val):
    logger.info("Attempting to find field-optimized classical ground state for CVO...")
    optimal_thetas_rad, final_energy = find_ground_state_orientations(
        numerical_params_list, S_numerical_val
    )

    if optimal_thetas_rad is None:
        logger.error(
            "Classical minimization failed. Cannot generate field-optimized Ud matrix."
        )
        return None, None, np.inf

    mpr_matrices_optimized_numeric_sympy = mpr(
        params_sym_or_num=numerical_params_list,
        S_val_for_mpr=S_numerical_val,
        field_optimized_theta_angles=optimal_thetas_rad,
    )

    nspins_uc = len(atom_pos())
    Ud_blocks_numeric = []
    for i in range(nspins_uc):
        rot_mat_num = np.array(
            mpr_matrices_optimized_numeric_sympy[i].evalf(), dtype=np.complex128
        )
        Ud_blocks_numeric.append(rot_mat_num)

    from scipy.linalg import block_diag

    Ud_numeric_optimized = block_diag(*Ud_blocks_numeric)
    logger.info(
        f"Successfully generated field-optimized Ud_numeric. Shape: {Ud_numeric_optimized.shape}"
    )

    return optimal_thetas_rad, Ud_numeric_optimized, final_energy


# Example usage for standalone testing of classical minimization
if __name__ == "__main__":
    logger.info(
        "Running standalone test for classical minimization in spin_model_cvo_Hc.py..."
    )

    example_S_val = 0.5
    J1_test = 2.49
    J2_test = 1.12 * J1_test
    J3_test = 2.03 * J1_test
    G1_test = 0.28
    Dx_test = 2.67

    example_params_h0 = [J1_test, J2_test, J3_test, G1_test, Dx_test, 0.0]  # H=0
    logger.info(f"\n--- Test Case: H=0 ---")
    logger.info(f"Parameters: {example_params_h0}")
    optimal_angles_h0, energy_h0 = find_ground_state_orientations(
        example_params_h0, example_S_val
    )
    if optimal_angles_h0 is not None:
        logger.info(
            f"Optimal theta angles at H=0 (degrees): {np.degrees(optimal_angles_h0)}"
        )
        logger.info(f"Energy at H=0: {energy_h0:.6f} meV")

    example_params_h_test = [
        J1_test,
        J2_test,
        J3_test,
        G1_test,
        Dx_test,
        14.0,
    ]  # Example H=14T
    logger.info(f"\n--- Test Case: H={example_params_h_test[5]} ---")
    logger.info(f"Parameters: {example_params_h_test}")
    optimal_angles_h_test, energy_h_test = find_ground_state_orientations(
        example_params_h_test, example_S_val
    )
    if optimal_angles_h_test is not None:
        logger.info(
            f"Optimal theta angles at H={example_params_h_test[5]} (degrees): {np.degrees(optimal_angles_h_test)}"
        )
        logger.info(f"Energy at H={example_params_h_test[5]}: {energy_h_test:.6f} meV")
        canting_degrees_h_test = 90.0 - np.degrees(optimal_angles_h_test)
        logger.info(
            f"Canting towards +z (degrees, 90 - theta): {canting_degrees_h_test}"
        )
        logger.info(f"Mean canting (degrees): {np.mean(canting_degrees_h_test):.4f}")

    logger.info("Standalone test finished.")
