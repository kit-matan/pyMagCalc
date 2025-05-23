#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Spin Model Module for MagCalc.

This module provides functions to define a spin model (crystal structure,
atomic positions, spin orientations, interactions) based on a configuration
file. It aims to be general and applicable to various magnetic materials.
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
import logging
from typing import Dict, List, Tuple, Any, Union

logger = logging.getLogger(__name__)

# Module-level storage for atom label-to-index mapping (for UC atoms)
_atom_label_to_index_uc: Dict[str, int] = {}
_atom_index_to_label_uc: Dict[int, str] = {}
DIST_TOL = 1e-6  # Distance tolerance for matching atom positions


def _init_atom_maps(config_crystal_structure: Dict[str, Any]):
    """Initializes the atom label to index maps for unit cell atoms."""
    global _atom_label_to_index_uc, _atom_index_to_label_uc
    _atom_label_to_index_uc = {
        atom["label"]: i
        for i, atom in enumerate(config_crystal_structure.get("atoms_uc", []))
    }
    _atom_index_to_label_uc = {
        i: atom["label"]
        for i, atom in enumerate(config_crystal_structure.get("atoms_uc", []))
    }
    if not _atom_label_to_index_uc:
        logger.warning("_init_atom_maps: No atoms found in atoms_uc to create maps.")


def unit_cell_from_config(config_crystal_structure: Dict[str, Any]) -> np.ndarray:
    """
    Calculates Cartesian unit cell vectors from lattice parameters in the config.

    Args:
        config_crystal_structure (Dict[str, Any]): The 'crystal_structure'
            section of the loaded configuration.

    Returns:
        np.ndarray: A 3x3 NumPy array where rows are the Cartesian unit cell
                    vectors va, vb, vc.

    Raises:
        ValueError: If 'lattice_parameters' are missing or in an invalid format.
    """
    if "unit_cell_vectors" in config_crystal_structure:
        logger.info("Using directly provided Cartesian unit_cell_vectors from config.")
        uc_vectors = np.array(
            config_crystal_structure["unit_cell_vectors"], dtype=float
        )
        if uc_vectors.shape == (3, 3):
            return uc_vectors
        else:
            raise ValueError("Provided 'unit_cell_vectors' must be a 3x3 matrix.")

    lp = config_crystal_structure.get("lattice_parameters")
    if lp is None:
        raise ValueError(
            "Missing 'lattice_parameters' or 'unit_cell_vectors' in crystal_structure config."
        )

    if isinstance(lp, dict):
        a, b, c = lp["a"], lp["b"], lp["c"]
        alpha_deg, beta_deg, gamma_deg = lp["alpha"], lp["beta"], lp["gamma"]
    elif isinstance(lp, list) and len(lp) == 6:
        a, b, c = lp[0], lp[1], lp[2]
        alpha_deg, beta_deg, gamma_deg = lp[3], lp[4], lp[5]
    else:
        raise ValueError(
            "Invalid 'lattice_parameters' format. Must be dict or list of 6 values."
        )

    alpha, beta, gamma = np.radians([alpha_deg, beta_deg, gamma_deg])

    cos_a, cos_b, cos_g = np.cos(alpha), np.cos(beta), np.cos(gamma)
    sin_g = np.sin(gamma)

    if abs(sin_g) < 1e-9:  # Avoid division by zero for collinear a,b
        raise ValueError(
            "Lattice parameter gamma cannot be 0 or 180 degrees for this conversion method if c is not along z."
        )

    # Standard formula for Cartesian lattice vectors from parameters
    # (see https://en.wikipedia.org/wiki/Fractional_coordinates)
    # Or International Tables for Crystallography, Vol. A, Section 5.1.
    va = np.array([a, 0, 0], dtype=float)
    vb = np.array([b * cos_g, b * sin_g, 0], dtype=float)

    # Volume term for vc_z, simplified using cell volume V
    # V = a*b*c * sqrt(1 - cos_a^2 - cos_b^2 - cos_g^2 + 2*cos_a*cos_b*cos_g)
    # More robustly:
    val = (cos_a - cos_b * cos_g) / sin_g
    vc_x = c * cos_b
    vc_y = c * val
    vc_z = c * np.sqrt(
        np.sin(beta) ** 2 - val**2
    )  # This is simplified from V/(a*b*sin_g)
    # A more direct formula for vc_z:
    # vc_z = c * np.sqrt(1 - cos_b**2 - ((cos_a - cos_b*cos_g)/sin_g)**2)
    # Let's use the one from Wikipedia which is common:
    # omega = V / (a*b*c) = sqrt(1 - cos_a^2 - cos_b^2 - cos_g^2 + 2*cos_a*cos_b*cos_g)
    # vc_z = c * omega / sin_g (This is not quite right, omega is part of V)
    # Correct form for vc_z from cell matrix:
    # M = [[a, b*cos_g, c*cos_b],
    #      [0, b*sin_g, c*(cos_a - cos_b*cos_g)/sin_g],
    #      [0, 0,       c*sqrt(sin_g^2 - cos_b^2 - cos_a^2 + 2*cos_a*cos_b*cos_g)/sin_g]]
    # The last term is V / (a*b*sin_g)
    # Let's use a simpler, widely cited set of formulas:
    # va = [a, 0, 0]
    # vb = [b*cos(gamma), b*sin(gamma), 0]
    # vc = [c*cos(beta), c*(cos(alpha) - cos(beta)*cos(gamma))/sin(gamma), c*sqrt(sin(gamma)^2 - cos(beta)^2 - ((cos(alpha) - cos(beta)*cos(gamma)))^2)/sin(gamma)]
    # The z-component of vc is V / (a*b*sin(gamma)) where V is cell volume.
    # V = a*b*c * sqrt(1 - cos(alpha)^2 - cos(beta)^2 - cos(gamma)^2 + 2*cos(alpha)*cos(beta)*cos(gamma))
    # So vc_z = c * sqrt(1 - cos_a^2 - cos_b^2 - cos_g^2 + 2*cos_a*cos_b*cos_g) / sin_g
    # This is only if sin_g is not zero.

    # Using the formula from https://en.wikipedia.org/wiki/Crystal_structure#Unit_cell
    # (assuming standard setting where a is along x, b is in xy plane)
    v_c_x = c * cos_b
    v_c_y = c * (cos_a - cos_b * cos_g) / sin_g
    v_c_z = (
        c * np.sqrt(sin_g**2 - cos_b**2 - ((cos_a - cos_b * cos_g)) ** 2) / sin_g
    )  # Incorrect formula
    # Corrected z component for vc:
    # From https://www.ruppweb.org/Xray/tutorial/spcdescr.htm
    # Or from https://github.com/materialsproject/pymatgen/blob/master/pymatgen/core/lattice.py#L208
    term_for_vc_z_sq = (
        1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    )
    if term_for_vc_z_sq < 0:  # Should not happen for valid lattice parameters
        logger.warning(
            f"Term for vc_z calculation is negative ({term_for_vc_z_sq:.3e}), lattice parameters might be invalid. Setting vc_z to 0."
        )
        vc_z_val = 0.0
    else:
        vc_z_val = (c / sin_g) * np.sqrt(term_for_vc_z_sq)

    vc = np.array([v_c_x, v_c_y, vc_z_val], dtype=float)

    return np.array([va, vb, vc])


def atom_pos_from_config(
    config_crystal_structure: Dict[str, Any], unit_cell_vectors_cartesian: np.ndarray
) -> np.ndarray:
    """
    Returns Cartesian atomic positions for atoms in the unit cell.

    Args:
        config_crystal_structure (Dict[str, Any]): The 'crystal_structure'
            section of the loaded configuration.
        unit_cell_vectors_cartesian (np.ndarray): 3x3 array of Cartesian unit
            cell vectors.

    Returns:
        np.ndarray: NumPy array of (N_atoms_uc, 3) Cartesian coordinates.

    Raises:
        ValueError: If 'atoms_uc' is missing or an atom spec is invalid.
    """
    _init_atom_maps(config_crystal_structure)  # Initialize label-index maps

    atoms_uc_list = config_crystal_structure.get("atoms_uc")
    if not atoms_uc_list:
        raise ValueError("Missing 'atoms_uc' list in crystal_structure config.")

    atom_positions_cartesian = []
    for i, atom_spec in enumerate(atoms_uc_list):
        if (
            not isinstance(atom_spec, dict)
            or "pos" not in atom_spec
            or "label" not in atom_spec
        ):
            raise ValueError(
                f"Invalid atom specification at index {i} in 'atoms_uc'. Missing 'pos' or 'label'."
            )

        frac_coords = np.array(atom_spec["pos"], dtype=float)
        if frac_coords.shape != (3,):
            raise ValueError(
                f"Atom '{atom_spec['label']}' has invalid 'pos' shape. Must be 3D fractional coordinates."
            )

        cart_coords = frac_coords @ unit_cell_vectors_cartesian
        atom_positions_cartesian.append(cart_coords)

    return np.array(atom_positions_cartesian)


def atom_pos_ouc_from_config(
    atom_positions_uc_cartesian: np.ndarray,
    unit_cell_vectors_cartesian: np.ndarray,
    config_calc_settings: Dict[str, Any],
) -> np.ndarray:
    """
    Generates Cartesian atomic positions including neighbors (OUC - Outside Unit Cell).

    Args:
        atom_positions_uc_cartesian (np.ndarray): Cartesian coordinates of UC atoms.
        unit_cell_vectors_cartesian (np.ndarray): Cartesian unit cell vectors.
        config_calc_settings (Dict[str, Any]): 'calculation_settings' from config,
            especially 'neighbor_shells'.

    Returns:
        np.ndarray: NumPy array of (N_atoms_ouc, 3) Cartesian coordinates.
                    The first N_atoms_uc entries are the original UC atoms.
    """
    r_pos_ouc_list = list(atom_positions_uc_cartesian)  # Start with UC atoms

    neighbor_shells = config_calc_settings.get(
        "neighbor_shells", [1, 1, 1]
    )  # Default to 3D +/-1 shell
    if not (
        isinstance(neighbor_shells, list)
        and len(neighbor_shells) == 3
        and all(isinstance(n, int) for n in neighbor_shells)
    ):
        logger.warning(
            f"Invalid 'neighbor_shells' format: {neighbor_shells}. Using default [1,1,1]."
        )
        neighbor_shells = [1, 1, 1]

    nx_range = range(-neighbor_shells[0], neighbor_shells[0] + 1)
    ny_range = range(-neighbor_shells[1], neighbor_shells[1] + 1)
    nz_range = range(-neighbor_shells[2], neighbor_shells[2] + 1)

    for i_offset in nx_range:
        for j_offset in ny_range:
            for k_offset in nz_range:
                if i_offset == 0 and j_offset == 0 and k_offset == 0:
                    continue  # Skip the original unit cell atoms, already added

                displacement = (
                    i_offset * unit_cell_vectors_cartesian[0]
                    + j_offset * unit_cell_vectors_cartesian[1]
                    + k_offset * unit_cell_vectors_cartesian[2]
                )
                r_pos_ouc_list.extend(atom_positions_uc_cartesian + displacement)

    # Remove duplicate positions that might arise from large neighbor_shells or specific symmetries
    # Using np.unique preserves the order of first appearance for unique rows.
    unique_ouc_pos, indices = np.unique(
        np.array(r_pos_ouc_list, dtype=float), axis=0, return_index=True
    )
    # Sort by the original index to maintain the UC atoms at the beginning, followed by neighbors.
    return unique_ouc_pos[np.argsort(indices)]


def _get_rotation_matrix_from_theta_phi(theta_rad: float, phi_rad: float) -> sp.Matrix:
    """
    Helper to get a SymPy rotation matrix Ud_i for a single spin.
    Ud_i transforms S_local (quantized along local z') to S_global.
    The third column of Ud_i is the direction of the spin in global coordinates:
    (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)).
    This version returns a SymPy matrix for direct use in symbolic Hamiltonian.
    """
    cos_t, sin_t = sp.cos(theta_rad), sp.sin(theta_rad)
    cos_p, sin_p = sp.cos(phi_rad), sp.sin(phi_rad)

    # Standard Z-Y-X Euler rotation U = Rz(phi)Ry(theta)Rx(psi=0)
    # Ud = Rz(phi)Ry(theta)
    # Ud = [[cp*ct, -sp, cp*st], [sp*ct, cp, sp*st], [-st, 0, ct]]
    # The 3rd column [cp*st, sp*st, ct] is the spin direction vector.
    # This matrix rotates the COORDINATE SYSTEM. We want to rotate the VECTOR.
    # S_global = Rz(phi)Ry(theta) * [0,0,S_val_local_z]
    # So the matrix Ud itself should be Rz(phi)Ry(theta).

    rot_matrix_sympy = sp.Matrix(
        [
            [cos_p * cos_t, -sin_p, cos_p * sin_t],
            [sin_p * cos_t, cos_p, sin_p * sin_t],
            [-sin_t, sp.S(0), cos_t],
        ]
    )
    return rot_matrix_sympy


def mpr_from_config(
    config_crystal_structure: Dict[str, Any], numerical_parameters: Dict[str, float]
) -> List[sp.Matrix]:
    """
    Generates symbolic rotation matrices based on classical spin orientations in the config.

    Args:
        config_crystal_structure (Dict[str, Any]): The 'crystal_structure' section from config,
            expected to contain `atoms_uc` with `magmom_classical` for each atom.
        numerical_parameters (Dict[str, float]): Dictionary of numerical values for parameters
            (e.g., field strength if magmom depends on it, though typically mpr is field-independent
             or uses a base field configuration).

    Returns:
        List[sp.Matrix]: List of SymPy 3x3 rotation matrices, one for each atom in `atoms_uc`.
    """
    rot_matrices = []
    atoms_uc_config = config_crystal_structure.get("atoms_uc")
    if not atoms_uc_config:
        raise ValueError(
            "mpr_from_config: 'atoms_uc' not found in config_crystal_structure."
        )

    for i, atom_spec in enumerate(atoms_uc_config):
        label = atom_spec.get("label", f"atom_{i}")
        magmom_spec = atom_spec.get("magmom_classical")

        if magmom_spec is None:
            logger.warning(
                f"magmom_classical not specified for atom '{label}'. Defaulting to spin along +z (theta=0, phi=0)."
            )
            theta_rad, phi_rad = 0.0, 0.0
        elif (
            isinstance(magmom_spec, list) and len(magmom_spec) == 3
        ):  # Cartesian [mx,my,mz]
            m_vec = np.array(magmom_spec, dtype=float)
            m_norm = np.linalg.norm(m_vec)
            if m_norm < 1e-9:  # Near-zero moment
                logger.warning(
                    f"Atom '{label}' has near-zero classical magmom {magmom_spec}. Defaulting to +z."
                )
                theta_rad, phi_rad = 0.0, 0.0
            else:
                m_vec_norm = m_vec / m_norm
                theta_rad = np.arccos(
                    np.clip(m_vec_norm[2], -1.0, 1.0)
                )  # Polar angle from z-axis
                phi_rad = np.arctan2(
                    m_vec_norm[1], m_vec_norm[0]
                )  # Azimuthal angle from x-axis
        elif (
            isinstance(magmom_spec, list) and len(magmom_spec) == 2
        ):  # Spherical [theta_deg, phi_deg]
            theta_rad = np.radians(magmom_spec[0])
            phi_rad = np.radians(magmom_spec[1])
        elif isinstance(magmom_spec, str):
            # --- Implement parsing for special strings here ---
            # Example: KFe3J-like angles based on atom label or index
            # This part needs to be customized or made more general.
            logger.warning(
                f"Special string '{magmom_spec}' for magmom_classical of atom '{label}' is not yet fully implemented. Defaulting to +z."
            )
            # Example for KFe3J (needs AL_SPIN_PREFERENCE or similar from config)
            # if label == "Fe1": phi_rad = np.radians(-120)
            # elif label == "Fe2": phi_rad = np.radians(0)
            # elif label == "Fe3": phi_rad = np.radians(120)
            # else: phi_rad = 0.0
            # theta_rad = np.pi/2 # Assume in xy-plane for this example
            theta_rad, phi_rad = 0.0, 0.0  # Fallback
        else:
            raise ValueError(
                f"Invalid 'magmom_classical' format for atom '{label}': {magmom_spec}"
            )

        rot_matrices.append(_get_rotation_matrix_from_theta_phi(theta_rad, phi_rad))

    return rot_matrices


def spin_interactions_from_config(
    symbolic_params_map: Dict[str, sp.Symbol],
    config_interactions: Dict[str, Any],
    atom_pos_uc_cartesian: np.ndarray,
    atom_pos_ouc_cartesian: np.ndarray,
    unit_cell_vectors_cartesian: np.ndarray,
) -> Tuple[sp.Matrix, sp.Matrix]:  # Jex, DM_matrix (matrix of DM vectors)
    """
    Generates symbolic Jex and DM_matrix (matrix of DM vectors) from config.
    This is a placeholder and needs full implementation.
    """
    # This function will be complex and is a key part of the refactoring.
    # For now, returning empty/zero matrices.
    N_atom_uc = len(atom_pos_uc_cartesian)
    N_atom_ouc = len(atom_pos_ouc_cartesian)

    Jex_sym = sp.zeros(N_atom_uc, N_atom_ouc)
    DMnull_sym = sp.Matrix([[sp.S(0), sp.S(0), sp.S(0)]])  # 1x3
    # Create a list of lists of DMnull_sym, then convert to SymPy Matrix
    dm_matrix_list_of_lists = [
        [DMnull_sym for _ in range(N_atom_ouc)] for _ in range(N_atom_uc)
    ]
    DM_matrix_sym = sp.Matrix(dm_matrix_list_of_lists)

    # Process Heisenberg interactions
    for heis_inter_def in config_interactions.get("heisenberg", []):
        label_i, label_j = heis_inter_def["pair"]
        idx_i_uc = _atom_label_to_index_uc.get(label_i)
        idx_j_uc_config = _atom_label_to_index_uc.get(
            label_j
        )  # Index of atom j in its own cell

        if idx_i_uc is None or idx_j_uc_config is None:
            logger.warning(
                f"Unknown atom label in Heisenberg pair: {label_i}, {label_j}. Skipping interaction."
            )
            continue

        J_val_config = heis_inter_def["J"]
        if isinstance(J_val_config, str):  # Symbolic J
            J_sympy_val = symbolic_params_map.get(J_val_config, sp.Symbol(J_val_config))
        else:  # Numerical J
            J_sympy_val = sp.S(J_val_config)

        offset_frac = np.array(heis_inter_def.get("rij_offset", [0, 0, 0]), dtype=float)
        offset_cart = offset_frac @ unit_cell_vectors_cartesian

        pos_j_target_cart = atom_pos_uc_cartesian[idx_j_uc_config] + offset_cart

        idx_j_ouc = -1
        for k_ouc, pos_ouc_k_cart in enumerate(atom_pos_ouc_cartesian):
            if np.allclose(pos_j_target_cart, pos_ouc_k_cart, atol=DIST_TOL):
                idx_j_ouc = k_ouc
                break

        if idx_j_ouc != -1:
            Jex_sym[idx_i_uc, idx_j_ouc] = J_sympy_val
        else:
            logger.warning(
                f"Could not find OUC match for Heisenberg interaction: {label_i} to {label_j} with offset {offset_frac}. Target pos: {pos_j_target_cart}"
            )

    # Process DM interactions
    for dm_inter_def in config_interactions.get("dm_interaction", []):
        label_i, label_j = dm_inter_def["pair"]
        idx_i_uc = _atom_label_to_index_uc.get(label_i)
        idx_j_uc_config = _atom_label_to_index_uc.get(label_j)

        if idx_i_uc is None or idx_j_uc_config is None:
            logger.warning(
                f"Unknown atom label in DM pair: {label_i}, {label_j}. Skipping interaction."
            )
            continue

        D_vector_config_comps = dm_inter_def[
            "D_vector"
        ]  # List of 3 components (symbolic or numeric)
        D_vec_sympy_comps_list = []
        for comp_str_or_num in D_vector_config_comps:
            if isinstance(comp_str_or_num, str):
                D_vec_sympy_comps_list.append(
                    symbolic_params_map.get(comp_str_or_num, sp.Symbol(comp_str_or_num))
                )
            else:
                D_vec_sympy_comps_list.append(sp.S(comp_str_or_num))
        D_vector_sympy = sp.Matrix([D_vec_sympy_comps_list])  # 1x3 SymPy Matrix

        offset_frac = np.array(dm_inter_def.get("rij_offset", [0, 0, 0]), dtype=float)
        offset_cart = offset_frac @ unit_cell_vectors_cartesian
        pos_j_target_cart = atom_pos_uc_cartesian[idx_j_uc_config] + offset_cart

        idx_j_ouc = -1  # find idx_j_ouc as above
        for k_ouc, pos_ouc_k_cart in enumerate(atom_pos_ouc_cartesian):
            if np.allclose(pos_j_target_cart, pos_ouc_k_cart, atol=DIST_TOL):
                idx_j_ouc = k_ouc
                break

        if idx_j_ouc != -1:
            DM_matrix_sym[idx_i_uc, idx_j_ouc] = D_vector_sympy
        else:
            logger.warning(
                f"Could not find OUC match for DM interaction: {label_i} to {label_j} with offset {offset_frac}. Target pos: {pos_j_target_cart}"
            )

    return Jex_sym, DM_matrix_sym


def Hamiltonian_from_config(
    Sxyz_operators_ouc: List[sp.Matrix],  # List of 3x1 SymPy matrices
    symbolic_parameters: Dict[str, sp.Symbol],  # e.g. {'J1': Symbol('J1_sym')}
    config_data: Dict[str, Any],
) -> sp.Expr:
    """
    Defines the symbolic spin Hamiltonian based on the loaded configuration.
    This is a placeholder and needs full implementation.
    """
    # Get geometric and interaction matrices
    uc_vecs = unit_cell_from_config(config_data["crystal_structure"])
    atom_pos_uc = atom_pos_from_config(config_data["crystal_structure"], uc_vecs)
    # Ensure _atom_label_to_index_uc and _atom_index_to_label_uc are populated by atom_pos_from_config

    # OUC positions are needed for spin_interactions, but Sxyz_operators_ouc is already for OUC
    # The number of OUC atoms is len(Sxyz_operators_ouc)
    # We need atom_pos_ouc to correctly map interactions.
    atom_pos_ouc = atom_pos_ouc_from_config(
        atom_pos_uc, uc_vecs, config_data.get("calculation_settings", {})
    )

    Jex_sym, DM_matrix_sym = spin_interactions_from_config(
        symbolic_parameters,  # Pass symbolic parameters map
        config_data["interactions"],
        atom_pos_uc,
        atom_pos_ouc,
        uc_vecs,
    )

    HM_expr = sp.S(0)
    gamma = 2.0  # g-factor
    mu_B = 5.7883818066e-2  # Bohr magneton in meV/T

    N_atoms_in_uc = len(atom_pos_uc)
    N_atoms_in_ouc = len(Sxyz_operators_ouc)  # Should match len(atom_pos_ouc)

    for i in range(N_atoms_in_uc):  # Iterate over atoms in the magnetic unit cell
        S_i = Sxyz_operators_ouc[
            i
        ]  # Spin operator for the i-th atom in UC (which is also i-th in OUC list)

        for j_ouc in range(N_atoms_in_ouc):  # Iterate over all OUC spins
            S_j_ouc = Sxyz_operators_ouc[j_ouc]

            # Heisenberg term
            if Jex_sym[i, j_ouc] != 0:
                HM_expr += 0.5 * Jex_sym[i, j_ouc] * (S_i.dot(S_j_ouc))

            # DM term
            DM_vec_ij = DM_matrix_sym[i, j_ouc]  # This is a 1x3 SymPy Matrix
            # Check if DM_vec_ij is a Matrix and then if it's not a zero matrix
            # or if it's simply SymPy's Zero (sp.S.Zero)
            is_dm_term_present = (
                isinstance(DM_vec_ij, sp.Matrix) and not DM_vec_ij.is_zero_matrix
            )
            if is_dm_term_present:
                # DM_vec_ij is 1x3, S_i.cross(S_j_ouc) is 3x1. Dot product is correct.
                HM_expr += 0.5 * DM_vec_ij.dot(S_i.cross(S_j_ouc))

        # Single-Ion Anisotropy (SIA) Term
        atom_i_label = _atom_index_to_label_uc.get(i)
        if atom_i_label:
            for sia_def in config_data["interactions"].get("single_ion_anisotropy", []):
                if sia_def.get("atom_label") == atom_i_label:
                    if sia_def.get("type") == "uniaxial_K_Sz_sq":
                        K_val_str = sia_def.get("K")
                        if K_val_str:
                            K_sym = symbolic_parameters.get(
                                K_val_str, sp.Symbol(K_val_str)
                            )
                            HM_expr += (
                                K_sym * S_i[2] ** 2
                            )  # Assumes S_i[2] is S_z_local after mpr rotation
                    # Add other SIA types here if needed
                    # E.g., D_Sz_sq_global: D_const * (S_i.dot(global_axis_vec))**2

        # Zeeman Term
        H_field_config = config_data["interactions"].get("applied_field", {})
        if H_field_config:  # Only add Zeeman if section exists in config
            H_vec_config_comps = H_field_config.get("H_vector", [0, 0, 0])
            H_mag_symbol_name = H_field_config.get("H_magnitude_symbol")

            actual_H_vector_sympy_comps = [sp.S(0)] * 3

            if H_mag_symbol_name:
                # Field is given by a direction vector and a symbolic magnitude
                H_mag_val_sym = symbolic_parameters.get(
                    H_mag_symbol_name, sp.Symbol(H_mag_symbol_name)
                )
                direction_H_numeric = np.array(H_vec_config_comps, dtype=float)
                norm_dir = np.linalg.norm(direction_H_numeric)
                if norm_dir > DIST_TOL:  # Use DIST_TOL for float comparison
                    unit_dir_H = direction_H_numeric / norm_dir
                    for k_ax in range(3):
                        actual_H_vector_sympy_comps[k_ax] = (
                            unit_dir_H[k_ax] * H_mag_val_sym
                        )
                elif H_mag_val_sym.is_zero is False and (
                    not hasattr(H_mag_val_sym, "is_Number")
                    or not H_mag_val_sym.is_Number
                    or H_mag_val_sym != 0
                ):  # Check if symbol is not explicitly zero
                    logger.warning(
                        f"Applied field symbol '{H_mag_symbol_name}' has non-zero symbolic magnitude but zero direction vector. Field term will be zero unless '{H_mag_symbol_name}' evaluates to zero."
                    )
            else:
                # H_vector components are direct values (can be symbols if they are keys in symbolic_parameters, or numbers)
                for k_ax in range(3):
                    comp = H_vec_config_comps[k_ax]
                    if isinstance(comp, str):  # Component is a symbolic parameter name
                        actual_H_vector_sympy_comps[k_ax] = symbolic_parameters.get(
                            comp, sp.Symbol(comp)
                        )
                    else:  # Component is a numerical value
                        actual_H_vector_sympy_comps[k_ax] = sp.S(comp)

            # Add Zeeman term if H is non-zero (symbolically or numerically)
            if any(
                comp != 0 for comp in actual_H_vector_sympy_comps
            ):  # Check if any component is non-zero
                HM_expr -= (
                    gamma
                    * mu_B
                    * (
                        S_i[0] * actual_H_vector_sympy_comps[0]
                        + S_i[1] * actual_H_vector_sympy_comps[1]
                        + S_i[2] * actual_H_vector_sympy_comps[2]
                    )
                )

    return HM_expr.expand()
