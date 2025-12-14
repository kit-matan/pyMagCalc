#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018
Modified on Mon 19 Feb 2024
Further modified based on classical_minimizer_cvo.py to handle applied field Hc.

@author: Kit Matan
@contributor: AI Assistant

This file contains the information about the spin structure that will be used
to calculate spin-waves by MagCal.py.
It includes classical energy minimization to find the ground state spin
configuration in an applied magnetic field.

alpha-Cu2V2O7
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
from scipy.optimize import minimize
from tqdm.auto import tqdm  # Added for progress bar
import logging
import sympy
import sys
import os  # Added for path operations for theta cache
import hashlib  # Added for theta cache key generation
import pickle  # Added for saving/loading theta cache

# --- Basic Logging Setup ---
# Configure logging early, before any custom log messages are emitted.
logging.basicConfig(
    level=logging.INFO,  # Use INFO for general operation, DEBUG for more verbosity
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# --- End Logging Setup ---
logger = logging.getLogger(__name__)

# This list defines the preferred azimuthal orientation of spins.
# 1 means phi=0 (positive x-direction preference in xz-plane),
# -1 means phi=pi (negative x-direction preference in xz-plane).
AL_SPIN_PREFERENCE = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]
logger.debug(f"{len(AL_SPIN_PREFERENCE)=}")
# Module-level cache for optimal theta angles to be used by mpr
_cached_optimal_thetas_for_mpr = None


def unit_cell():
    """Unit cell vectors for alpha-Cu2V2O7."""
    va = np.array([20.645, 0, 0])
    vb = np.array([0, 8.383, 0])
    vc = np.array([0, 0, 6.442])
    uc = [va, vb, vc]
    return np.array(uc)


def atom_pos():
    """Atomic positions for V atoms in alpha-Cu2V2O7 within the unit cell."""
    la_uc = 20.645
    lb_uc = 8.383
    lc_uc = 6.442
    x = 0.16572
    y = 0.3646
    z = 0.7545

    # Fractional coordinates
    positions_fractional = [
        [x, y, z],
        [1 - x, 1 - y, z],
        [x, y + 1 / 2, z - 1 / 2],
        [1 - x, -y + 1 / 2, z - 1 / 2],
        [x + 1 / 2, y, z - 1 / 2],
        [-x + 1 / 2, 1 - y, z - 1 / 2],
        [x + 1 / 2, y + 1 / 2, z],
        [-x + 1 / 2, -y + 1 / 2, z],
        [x + 1 / 4, -y + 1 / 4 + 1, z + 1 / 4],
        [-x + 1 / 4, y + 1 / 4, z + 1 / 4],
        [x + 1 / 4, -y + 3 / 4, z + 3 / 4 - 1],
        [-x + 1 / 4, y + 3 / 4, z + 3 / 4 - 1],
        [x + 3 / 4, -y + 1 / 4 + 1, z + 3 / 4 - 1],
        [-x + 3 / 4, y + 1 / 4, z + 3 / 4 - 1],
        [x + 3 / 4, -y + 3 / 4, z + 1 / 4],
        [-x + 3 / 4, y + 3 / 4 - 1, z + 1 / 4],
    ]
    # Convert to Cartesian coordinates
    r_pos_cartesian = [
        np.array([pos[0] * la_uc, pos[1] * lb_uc, pos[2] * lc_uc])
        for pos in positions_fractional
    ]
    return np.array(r_pos_cartesian)


def atom_pos_ouc():
    """Atomic positions including nearest neighbors outside the unit cell."""
    r_pos_ouc_list = []
    uc_vectors = unit_cell()
    apos_uc = atom_pos()
    r_pos_ouc_list.extend(apos_uc)  # Add atoms in the unit cell first
    # Iterate over neighboring unit cells (-1 to 1 in each direction)
    for i_cell in range(-1, 2):
        for j_cell in range(-1, 2):
            for k_cell in range(-1, 2):
                if i_cell == 0 and j_cell == 0 and k_cell == 0:
                    continue  # Skip the original unit cell atoms already added
                # Displacement vector for the neighboring cell
                displacement = (
                    i_cell * uc_vectors[0]
                    + j_cell * uc_vectors[1]
                    + k_cell * uc_vectors[2]
                )
                # Add displaced atomic positions
                r_pos_ouc_list.extend(apos_uc + displacement)
    return np.array(r_pos_ouc_list)


def symbolic_rot_mat_y(angle_sym):
    """Symbolic rotation matrix around y-axis."""
    return sp.Matrix(
        [
            [sp.cos(angle_sym), 0, sp.sin(angle_sym)],
            [0, 1, 0],
            [-sp.sin(angle_sym), 0, sp.cos(angle_sym)],
        ]
    )


def mpr(p_symbolic):
    """
    Provides symbolic rotation matrices for MagCalc's gen_HM.
    ALWAYS returns the zero-field configuration (spins along +/- a-axis).
    The field effect is handled by Ud_numeric_override in MagCalc.
    p_symbolic: list of symbolic Hamiltonian parameters [J1, ..., H]
    """
    rot_m = []
    global _cached_optimal_thetas_for_mpr

    if _cached_optimal_thetas_for_mpr is None:
        logger.info(
            "mpr: _cached_optimal_thetas_for_mpr not set. "
            "Using default zero-field symbolic rotations (spins along +/-a axis)."
        )
        for al_val in AL_SPIN_PREFERENCE:
            symbolic_angle_for_Ry = -al_val * sp.pi / 2
            rot_m.append(symbolic_rot_mat_y(symbolic_angle_for_Ry))
    else:
        logger.info(
            "mpr: Using _cached_optimal_thetas_for_mpr to construct rotation matrices."
        )
        if len(_cached_optimal_thetas_for_mpr) != len(AL_SPIN_PREFERENCE):
            logger.error(
                "mpr: Mismatch between length of _cached_optimal_thetas_for_mpr "
                f"({len(_cached_optimal_thetas_for_mpr)}) and AL_SPIN_PREFERENCE "
                f"({len(AL_SPIN_PREFERENCE)}). Falling back to zero-field rotations."
            )
            # Fallback to zero-field symbolic rotations
            for al_val in AL_SPIN_PREFERENCE:
                symbolic_angle_for_Ry = -al_val * sp.pi / 2
                rot_m.append(symbolic_rot_mat_y(symbolic_angle_for_Ry))
            return rot_m

        for i in range(len(AL_SPIN_PREFERENCE)):
            theta_val_num = _cached_optimal_thetas_for_mpr[i]
            phi_val_num = 0.0 if AL_SPIN_PREFERENCE[i] == 1 else np.pi

            # _get_rotation_matrix_from_theta_phi returns a NumPy complex array.
            # The rotation matrix for real angles should be real.
            numerical_rot_matrix_np = _get_rotation_matrix_from_theta_phi(
                theta_val_num, phi_val_num
            )
            if np.any(np.abs(np.imag(numerical_rot_matrix_np)) > 1e-9):
                logger.warning(
                    f"mpr: Numerical rotation matrix for spin {i} has significant imaginary part."
                )
            # Convert the real part of the NumPy matrix to a SymPy Matrix of SymPy Floats
            rot_m.append(sp.Matrix(np.real(numerical_rot_matrix_np).tolist()))
    return rot_m


def _get_rotation_matrix_from_theta_phi(theta, phi):
    """
    Returns a 3x3 rotation matrix Ud_i for a single spin.
    This matrix transforms S_local (quantized along local z') to S_global.
    The third column of Ud_i is the direction of the spin in global coordinates:
    (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
    """
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi), np.sin(phi)

    # Spin direction (local z' axis in global coords)
    z_prime_global = np.array([sin_t * cos_p, sin_t * sin_p, cos_t])

    # Create orthonormal basis (x', y', z') in global coords
    # Handle cases where spin is along global z to avoid issues with cross product
    if np.abs(sin_t) < 1e-9:  # Spin is along global z or -z
        x_prime_global = np.array([cos_p, sin_p, 0])  # x' in xy plane, rotated by phi
        if cos_t > 0:  # along +z
            y_prime_global = np.array([-sin_p, cos_p, 0])  # y' = z x x' (global z x x')
        else:  # along -z
            y_prime_global = np.array(
                [sin_p, -cos_p, 0]
            )  # y' = z x x' (global -z x x') to maintain right-handedness
    else:  # General case
        # Prefer global_x_axis to form y_prime_global = z_prime_global x global_x_axis
        # unless z_prime_global is parallel to global_x_axis
        global_x_axis = np.array([1.0, 0.0, 0.0])
        if (
            np.abs(np.dot(z_prime_global, global_x_axis)) > 1.0 - 1e-9
        ):  # z_prime is along x_axis
            global_y_axis = np.array(
                [0.0, 1.0, 0.0]
            )  # Use global y if z_prime is along x
            y_prime_global = np.cross(z_prime_global, global_y_axis)
        else:
            y_prime_global = np.cross(z_prime_global, global_x_axis)

        y_prime_global /= np.linalg.norm(y_prime_global)
        x_prime_global = np.cross(y_prime_global, z_prime_global)
        # x_prime_global should be normalized if y_prime_global and z_prime_global are ortho and normalized.
        # np.linalg.norm(x_prime_global) # For checking, should be ~1

    # The rotation matrix Ud_i has x_prime_global, y_prime_global, z_prime_global as its columns
    # S_global = Ud_i * S_local
    rot_matrix = np.array([x_prime_global, y_prime_global, z_prime_global]).T
    return rot_matrix.astype(complex)


def is_vector_parallel_to_another(vector1, vector2):
    """
    Checks if vector1 is parallel to vector2.

    Args:
        vector1: A numpy array representing the first vector.
        vector2: A numpy array representing the second vector.

    Returns:
        True if vector1 is parallel to vector2, False otherwise.
    """
    return np.allclose(np.cross(vector1, vector2), 0)


def is_right_neighbor(atom1, atom2):
    """
    Checks if atom2 is to the right of atom1 in a straight-line crystal structure.

    Args:
        atom1: A numpy array representing the Cartesian coordinates of the first atom.
        atom2: A numpy array representing the Cartesian coordinates of the second atom.

    Returns:
        True if atom2 is to the right of atom1, False otherwise.
    """
    if atom1[1] < atom2[1]:  # Check b-axis
        return True
    elif atom1[1] > atom2[1]:
        return False
    else:
        # If y-coordinates are equal, check other axes for uniqueness
        # (assuming no overlapping atoms)
        if atom1[0] != atom2[0] or atom1[2] != atom2[2]:
            raise ValueError(
                "Atoms cannot have the same x-coordinate and different other coordinates."
            )
        else:
            return False  # Both atoms are at the same position


def get_nearest_neighbor_distances(
    atom_positions, outside_unit_cell_atom_positions, num_neighbors=3
):
    """Return the distances to the nearest Fe neighbors."""
    # List the distance between the first three nearest neighnor Fe atoms and do not list the same distance twice
    distances = []
    for i in range(len(atom_positions)):
        for j in range(len(outside_unit_cell_atom_positions)):
            distance = round(
                np.linalg.norm(atom_positions[i] - outside_unit_cell_atom_positions[j]),
                4,
            )
            if distance not in distances:
                distances.append(distance)

    distances = sorted(distances)  # Sort the distances from low to high
    return distances[:num_neighbors]


def spin_interactions(p):
    # generate J exchange interactions
    J1, J2, J3, G1, Dx, Dy, D3, H = p
    apos = atom_pos()
    nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    nspin_ouc = len(apos_ouc)
    neighbor_dist_list = get_nearest_neighbor_distances(apos, apos_ouc, 10)[1:4]
    Jex = sp.zeros(nspin, nspin_ouc)
    Gex = sp.zeros(nspin, nspin_ouc)

    # Create a matrix to be filled with DM vectors.
    # This will be a matrix of SymPy Matrix objects if parameters are symbolic.
    DMmat = np.empty((nspin, nspin_ouc), dtype=object)
    DMnull = sp.Matrix([0, 0, 0])

    for i in range(nspin):
        for j in range(nspin_ouc):
            dist_ij = la.norm(apos[i] - apos_ouc[j])

            if np.round(dist_ij, 2) == np.round(neighbor_dist_list[0], 2):
                Jex[i, j] = J1
                Gex[i, j] = G1

                # --- DM vector logic from plot_magnetic_structure.py ---
                # This logic must be compatible with symbolic parameters (Dx, Dy).
                bond_direction_vec = apos_ouc[j] - apos[i]

                # a-component's sign is determined by the bond's orientation along the b-axis.
                sign = 1.0
                if is_right_neighbor(apos[i], apos_ouc[j]):
                    sign = -1.0
                dm_a_component = sign * Dx

                # bc-component is perpendicular to the bond in the bc-plane.
                bond_vec_bc_y = bond_direction_vec[1]
                bond_vec_bc_z = bond_direction_vec[2]

                # The norm is a numerical value, not symbolic.
                norm_bond_vec_bc_val = np.sqrt(bond_vec_bc_y**2 + bond_vec_bc_z**2)

                # Avoid division by zero for bonds purely along the a-axis (if any)
                if norm_bond_vec_bc_val > 1e-9:
                    # Normalized bc-projection of the bond vector (numerical)
                    norm_y = bond_vec_bc_y / norm_bond_vec_bc_val
                    norm_z = bond_vec_bc_z / norm_bond_vec_bc_val

                    # Direction perpendicular to bond in bc-plane: (y, z) -> (-z, y)
                    dm_vec_bc_direction_y = -norm_z
                    dm_vec_bc_direction_z = norm_y

                    # Magnitude from Dy, sign from spin preference
                    dm_bc_component_y = (
                        AL_SPIN_PREFERENCE[i] * Dy * dm_vec_bc_direction_y
                    )
                    dm_bc_component_z = (
                        AL_SPIN_PREFERENCE[i] * Dy * dm_vec_bc_direction_z
                    )
                else:
                    dm_bc_component_y = 0
                    dm_bc_component_z = 0

                # Construct the final DM vector (can be symbolic)
                final_dm_vec = sp.Matrix(
                    [dm_a_component, dm_bc_component_y, dm_bc_component_z]
                )
                DMmat[i, j] = final_dm_vec

            elif np.round(dist_ij, 2) == np.round(neighbor_dist_list[1], 2):
                Jex[i, j] = J2
                DMmat[i, j] = DMnull
            elif np.round(dist_ij, 2) == np.round(neighbor_dist_list[2], 2):
                Jex[i, j] = J3
                # The sign of the DM vector's a-component should depend on the
                # bond orientation to respect crystal symmetry, similar to J1.
                sign = 1.0
                if is_right_neighbor(apos[i], apos_ouc[j]):
                    sign = -1.0
                DMmat[i, j] = sp.Matrix([sign * D3, 0, 0])
            else:
                DMmat[i, j] = DMnull

    return Jex, Gex, DMmat, H


def Hamiltonian(Sxyz_ops, p_sym):
    """Define the spin Hamiltonian for alpha-Cu2V2O7."""
    gamma = 2.0
    mu_B = 5.7883818066e-2

    Jex_sym, Gex_sym, DM_sym_mat, H_sym = spin_interactions(p_sym)
    # H_sym is p_sym[-1]

    HM_expr = sp.sympify(0)
    apos_uc = atom_pos()
    Nspin_uc = len(apos_uc)
    Nspin_ouc = len(Sxyz_ops)

    for i in range(Nspin_uc):
        for j in range(Nspin_ouc):
            # Check if any interaction exists to avoid adding zero terms
            is_J_non_zero = Jex_sym[i, j] != 0
            is_G_non_zero = Gex_sym[i, j] != 0
            DM_vec_ij = DM_sym_mat[i, j]
            is_DM_non_zero = (
                isinstance(DM_vec_ij, sp.MatrixBase) and not DM_vec_ij.is_zero_matrix
            )
            if is_J_non_zero or is_G_non_zero or is_DM_non_zero:
                # Heisenberg term
                if is_J_non_zero:
                    HM_expr += (
                        0.5
                        * Jex_sym[i, j]
                        * (
                            Sxyz_ops[i][0] * Sxyz_ops[j][0]
                            + Sxyz_ops[i][1] * Sxyz_ops[j][1]
                            + Sxyz_ops[i][2] * Sxyz_ops[j][2]
                        )
                    )
                # DM interaction term
                if is_DM_non_zero:
                    HM_expr += 0.5 * (
                        DM_vec_ij[0]
                        * (
                            Sxyz_ops[i][1] * Sxyz_ops[j][2]
                            - Sxyz_ops[i][2] * Sxyz_ops[j][1]
                        )
                        + DM_vec_ij[1]
                        * (
                            Sxyz_ops[i][2] * Sxyz_ops[j][0]
                            - Sxyz_ops[i][0] * Sxyz_ops[j][2]
                        )
                        + DM_vec_ij[2]
                        * (
                            Sxyz_ops[i][0] * Sxyz_ops[j][1]
                            - Sxyz_ops[i][1] * Sxyz_ops[j][0]
                        )
                    )
                # Anisotropic exchange term
                if is_G_non_zero:
                    HM_expr += (
                        0.5
                        * Gex_sym[i, j]
                        * (
                            Sxyz_ops[i][0] * Sxyz_ops[j][0]
                            - Sxyz_ops[i][1] * Sxyz_ops[j][1]
                            - Sxyz_ops[i][2] * Sxyz_ops[j][2]
                        )
                    )

        # Zeeman term: H is applied along the global z-axis
        # Energy is -mu.B = -(-g*mu_B*S).B = +g*mu_B*S.B
        HM_expr += gamma * mu_B * Sxyz_ops[i][2] * H_sym

    HM_expr = HM_expr.expand()
    return HM_expr



# --- Functions for Classical Energy Minimization ---
# NOTE: The legacy specialized minimization functions (classical_energy_fast, 
# find_ground_state_orientations, get_field_optimized_state_for_lswt) have been 
# REMOVED in favor of the generalized MagCalc.minimize_energy method.

def set_magnetic_structure(theta_angles, phi_angles=None):
    """
    Sets the magnetic structure (theta and phi angles) to be used by mpr.
    
    Args:
        theta_angles (list or np.ndarray): Theta angles for each spin (in radians).
        phi_angles (list or np.ndarray, optional): Phi angles for each spin (in radians).
                                     If None, uses AL_SPIN_PREFERENCE to determine phi.
    """
    global _cached_optimal_thetas_for_mpr
    global _cached_optimal_phis_for_mpr
    
    if len(theta_angles) != len(AL_SPIN_PREFERENCE):
         logger.error(f"Length of theta_angles ({len(theta_angles)}) does not match spins ({len(AL_SPIN_PREFERENCE)}).")
         return
         
    _cached_optimal_thetas_for_mpr = np.array(theta_angles, dtype=float)
    
    if phi_angles is not None:
         if len(phi_angles) != len(AL_SPIN_PREFERENCE):
              logger.warning("Length of phi_angles mismatch. Ignoring provided phis.")
              _cached_optimal_phis_for_mpr = None
         else:
              _cached_optimal_phis_for_mpr = np.array(phi_angles, dtype=float)
    else:
         _cached_optimal_phis_for_mpr = None
    
    logger.info("Magnetic structure updated in spin_model_hc module.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("This module defines the spin model for aCVO (H//c).")
    logger.info("Use MagCalc.minimize_energy() to find the ground state, then call set_magnetic_structure().")
