#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generic Spin Model Module for MagCalc.
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
import logging
from typing import Dict, List, Tuple, Any, Union
from itertools import product

import os, sys
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_atom_label_to_index_uc: Dict[str, int] = {}
_atom_index_to_label_uc: Dict[int, str] = {}
DIST_TOL = 1e-6


def _init_atom_maps(config_crystal_structure: Dict[str, Any]):
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
    if "unit_cell_vectors" in config_crystal_structure or "lattice_vectors" in config_crystal_structure:
        key = "unit_cell_vectors" if "unit_cell_vectors" in config_crystal_structure else "lattice_vectors"
        logger.debug(f"Using directly provided Cartesian {key} from config.")
        uc_vectors = np.array(
            config_crystal_structure[key], dtype=float
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
    if abs(sin_g) < 1e-9:
        raise ValueError("Lattice parameter gamma results in sin(gamma) close to zero.")
    va = np.array([a, 0, 0], dtype=float)
    vb = np.array([b * cos_g, b * sin_g, 0], dtype=float)
    vc_x = c * cos_b
    vc_y = c * (cos_a - cos_b * cos_g) / sin_g
    term_for_vc_z_sq_content = (
        1.0 - cos_a**2 - cos_b**2 - cos_g**2 + 2.0 * cos_a * cos_b * cos_g
    )
    if term_for_vc_z_sq_content < -1e-9:
        raise ValueError(
            f"Invalid lattice parameters: term for vc_z calculation is negative ({term_for_vc_z_sq_content:.3e})."
        )
    if term_for_vc_z_sq_content < 0:
        term_for_vc_z_sq_content = 0
    vc_z_val = (c / sin_g) * np.sqrt(term_for_vc_z_sq_content)
    vc = np.array([vc_x, vc_y, vc_z_val], dtype=float)
    return np.array([va, vb, vc])


def atom_pos_from_config(
    config_crystal_structure: Dict[str, Any], unit_cell_vectors_cartesian: np.ndarray
) -> np.ndarray:
    _init_atom_maps(config_crystal_structure)
    atoms_uc_list = config_crystal_structure.get("atoms_uc")
    if not atoms_uc_list:
        raise ValueError("Missing 'atoms_uc' list.")
    atom_positions_cartesian = []
    for i, atom_spec in enumerate(atoms_uc_list):
        if (
            not isinstance(atom_spec, dict)
            or "pos" not in atom_spec
            or "label" not in atom_spec
        ):
            raise ValueError(f"Invalid atom spec at index {i}.")
        frac_coords = np.array(atom_spec["pos"], dtype=float)
        if frac_coords.shape != (3,):
            raise ValueError(f"Atom '{atom_spec['label']}' pos not 3D.")
        cart_coords = frac_coords @ unit_cell_vectors_cartesian
        atom_positions_cartesian.append(cart_coords)
    return np.array(atom_positions_cartesian)


def atom_pos_ouc_from_config(
    atom_positions_uc_cartesian: np.ndarray,
    unit_cell_vectors_cartesian: np.ndarray,
    config_calc_settings: Dict[str, Any],
) -> np.ndarray:
    r_pos_ouc_list = list(atom_positions_uc_cartesian.copy())
    neighbor_shells = config_calc_settings.get("neighbor_shells", [1, 1, 0])
    if not (
        isinstance(neighbor_shells, list)
        and len(neighbor_shells) == 3
        and all(isinstance(n, int) for n in neighbor_shells)
    ):
        logger.warning(f"Invalid 'neighbor_shells': {neighbor_shells}. Using [1,1,0].")
        neighbor_shells = [1, 1, 0]
    generated_neighbors_for_ouc = []
    u_range = range(-neighbor_shells[0], neighbor_shells[0] + 1)
    v_range = range(-neighbor_shells[1], neighbor_shells[1] + 1)
    w_range = range(-neighbor_shells[2], neighbor_shells[2] + 1)
    for u_offset in u_range:
        for v_offset in v_range:
            for w_offset in w_range:
                if u_offset == 0 and v_offset == 0 and w_offset == 0:
                    continue
                displacement = (
                    u_offset * unit_cell_vectors_cartesian[0]
                    + v_offset * unit_cell_vectors_cartesian[1]
                    + w_offset * unit_cell_vectors_cartesian[2]
                )
                for atom_uc_pos in atom_positions_uc_cartesian:
                    generated_neighbors_for_ouc.append(atom_uc_pos + displacement)
    r_pos_ouc_list.extend(generated_neighbors_for_ouc)
    return np.array(r_pos_ouc_list, dtype=float)


def mpr_from_config(
    config_dict: Dict[str, Any], symbolic_params_map: Dict[str, sp.Symbol]
) -> List[sp.Matrix]:
    """
    Generate rotation matrices (Ud blocks) for each atom in the unit cell.
    Reads from 'minimization' -> 'initial_configuration' or 'magnetic_structure'.
    Defaults to Identity (Global Z alignment).
    """
    atoms_uc = config_dict.get("crystal_structure", {}).get("atoms_uc", [])
    nspins = len(atoms_uc)
    rot_m_list = [sp.eye(3) for _ in range(nspins)]

    # Helper to convert direction vector to rotation matrix
    def dir_to_rot(direction):
        dv = np.array(direction, dtype=float)
        norm = np.linalg.norm(dv)
        if norm < 1e-9:
            return sp.eye(3)
        dv = dv / norm
        theta = np.arccos(dv[2])
        phi = np.arctan2(dv[1], dv[0])
        
        ct, st = sp.cos(theta), sp.sin(theta)
        cp, sp_val = sp.cos(phi), sp.sin(phi)
        return sp.Matrix([
            [ct*cp, -sp_val, st*cp],
            [ct*sp_val,  cp, st*sp_val],
            [  -st,      0,    ct]
        ])

    # 1. Check for magnetic_structure
    mag_struct = config_dict.get("magnetic_structure")
    if mag_struct:
        stype = mag_struct.get("type")
        if stype == "pattern":
            ptype = mag_struct.get("pattern_type")
            directions = mag_struct.get("directions", [])
            if ptype == "antiferromagnetic" and len(directions) >= 2:
                for i in range(nspins):
                    dir_vec = directions[i % 2]
                    rot_m_list[i] = dir_to_rot(dir_vec)
            elif directions:
                for i in range(min(nspins, len(directions))):
                    rot_m_list[i] = dir_to_rot(directions[i])
        return rot_m_list

    # 2. Check for initial configuration in minimization
    minimization = config_dict.get("minimization", {})
    initial_config = minimization.get("initial_configuration")
    
    if initial_config and isinstance(initial_config, list):
        for entry in initial_config:
            idx = entry.get("atom_index")
            if idx is None or not isinstance(idx, int) or idx < 0 or idx >= nspins:
                continue
            theta = entry.get("theta", 0.0)
            phi = entry.get("phi", 0.0)
            
            ct, st = sp.cos(theta), sp.sin(theta)
            cp, sp_val = sp.cos(phi), sp.sin(phi)
            rot_m_list[idx] = sp.Matrix([
                [ct*cp, -sp_val, st*cp],
                [ct*sp_val, cp, st*sp_val],
                [-st, 0, ct]
            ])

    return rot_m_list


def spin_interactions_from_config(
    symbolic_params_map: Dict[str, sp.Symbol],
    config_interactions: List[Dict[str, Any]],
    atom_pos_uc_cartesian: np.ndarray,
    atom_pos_ouc_cartesian: np.ndarray,
    unit_cell_vectors_cartesian: np.ndarray,
) -> Tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    N_atom_uc = len(atom_pos_uc_cartesian)
    N_atom_ouc = len(atom_pos_ouc_cartesian)
    Jex_sym = sp.zeros(N_atom_uc, N_atom_ouc)

    # Initialize DM and Anisotropic Exchange matrices
    _dm_matrix_python_list = []
    _kex_matrix_python_list = []
    default_null_vector = sp.Matrix([sp.S(0), sp.S(0), sp.S(0)])

    for _ in range(N_atom_uc):
        _dm_matrix_python_list.append(
            [default_null_vector.copy() for _ in range(N_atom_ouc)]
        )
        _kex_matrix_python_list.append(
            [default_null_vector.copy() for _ in range(N_atom_ouc)]
        )

    sympify_locals = {**symbolic_params_map, "sqrt": sp.sqrt, "S": sp.S}

    # Pre-process interactions if list
    if isinstance(config_interactions, list):
        grouped = {}
        for inter in config_interactions:
            itype = inter.get("type")
            if itype:
                if itype not in grouped:
                    grouped[itype] = []
                grouped[itype].append(inter)
        config_interactions = grouped

    # Helper for OUC matching
    def find_j_ouc(label_j, offset_frac):
        idx_j_uc = _atom_label_to_index_uc.get(label_j)
        if idx_j_uc is None:
            return None
        offset_cart = np.array(offset_frac, dtype=float) @ unit_cell_vectors_cartesian
        pos_j_target_cart = atom_pos_uc_cartesian[idx_j_uc] + offset_cart
        for k_ouc, pos_ouc_k_cart in enumerate(atom_pos_ouc_cartesian):
            if np.allclose(pos_j_target_cart, pos_ouc_k_cart, atol=DIST_TOL):
                return k_ouc
        return None

    # Heisenberg
    for heis_inter_def in config_interactions.get("heisenberg", []):
        J_val_config = heis_inter_def.get("J") or heis_inter_def.get("value")
        J_sympy_val = symbolic_params_map.get(J_val_config, sp.Symbol(J_val_config)) if isinstance(J_val_config, str) else sp.S(J_val_config)

        if "pair" in heis_inter_def:
            label_i, label_j = heis_inter_def["pair"]
            offset_frac = heis_inter_def.get("rij_offset", [0, 0, 0])
            idx_i_uc = _atom_label_to_index_uc.get(label_i)
            idx_j_ouc = find_j_ouc(label_j, offset_frac)

            if idx_i_uc is not None and idx_j_ouc is not None:
                Jex_sym[idx_i_uc, idx_j_ouc] = J_sympy_val
            else:
                logger.error(f"Heisenberg expansion skip: {label_i}-{label_j} offset {offset_frac}")
        elif "distance" in heis_inter_def:
            target_dist = float(heis_inter_def["distance"])
            # Match all pairs within distance tolerance
            for i_uc in range(N_atom_uc):
                for j_ouc in range(N_atom_ouc):
                    dist = np.linalg.norm(atom_pos_uc_cartesian[i_uc] - atom_pos_ouc_cartesian[j_ouc])
                    if abs(dist - target_dist) < DIST_TOL:
                        Jex_sym[i_uc, j_ouc] = J_sympy_val

    # DM Interaction (Symmetry Propagated or Manual)
    dm_list = config_interactions.get("dm_interaction", []) + config_interactions.get("dm_manual", [])
    for dm_inter_def in dm_list:
        if "pair" in dm_inter_def:
            label_i, label_j = dm_inter_def["pair"]
            offset_frac = dm_inter_def.get("rij_offset", [0, 0, 0])
        else:
            # Handle atom_i, atom_j, offset_j style (manual)
            idx_i = dm_inter_def.get("atom_i")
            idx_j = dm_inter_def.get("atom_j")
            label_i = _atom_index_to_label_uc.get(idx_i) if isinstance(idx_i, int) else None
            label_j = _atom_index_to_label_uc.get(idx_j) if isinstance(idx_j, int) else None
            offset_frac = dm_inter_def.get("offset_j", [0, 0, 0])

        idx_i_uc = _atom_label_to_index_uc.get(label_i)
        idx_j_ouc = find_j_ouc(label_j, offset_frac)

        if idx_i_uc is not None and idx_j_ouc is not None:
            D_vals = dm_inter_def.get("D_vector") or dm_inter_def.get("value")
            D_vec_sym = sp.Matrix([
                sp.sympify(c, locals=sympify_locals) if isinstance(c, str) else sp.S(c)
                for c in D_vals
            ])
            _dm_matrix_python_list[idx_i_uc][idx_j_ouc] = D_vec_sym
        else:
            logger.error(f"DM expansion skip: {label_i}-{label_j} offset {offset_frac}")

    # Anisotropic Exchange / Interaction Matrix
    kex_list = config_interactions.get("anisotropic_exchange", []) + config_interactions.get("interaction_matrix", [])
    for kex_inter_def in kex_list:
        if "pair" not in kex_inter_def: continue
        label_i, label_j = kex_inter_def["pair"]
        offset_frac = kex_inter_def.get("rij_offset", [0, 0, 0])
        idx_i_uc = _atom_label_to_index_uc.get(label_i)
        idx_j_ouc = find_j_ouc(label_j, offset_frac)

        if idx_i_uc is not None and idx_j_ouc is not None:
            K_vals = kex_inter_def.get("value")
            if isinstance(K_vals, list) and len(K_vals) > 0 and isinstance(K_vals[0], list):
                 # Full 3x3 matrix
                 K_mat_sym = sp.Matrix([
                    [sp.sympify(c, locals=sympify_locals) if isinstance(c, str) else sp.S(c) for c in row]
                    for row in K_vals
                 ])
                 _kex_matrix_python_list[idx_i_uc][idx_j_ouc] = K_mat_sym
            else:
                # Vector [Jxx, Jyy, Jzz]
                K_vec_sym = sp.Matrix([
                    sp.sympify(c, locals=sympify_locals) if isinstance(c, str) else sp.S(c)
                    for c in K_vals
                ])
                _kex_matrix_python_list[idx_i_uc][idx_j_ouc] = K_vec_sym

    # Kitaev Interaction
    for kitaev_def in config_interactions.get("kitaev", []):
        if "pair" not in kitaev_def: continue
        label_i, label_j = kitaev_def["pair"]
        offset_frac = kitaev_def.get("rij_offset", [0, 0, 0])
        idx_i_uc = _atom_label_to_index_uc.get(label_i)
        idx_j_ouc = find_j_ouc(label_j, offset_frac)

        if idx_i_uc is not None and idx_j_ouc is not None:
            K_val = kitaev_def.get("K") or kitaev_def.get("value")
            bond_dir = kitaev_def.get("bond_direction", "x").lower()
            K_sym = symbolic_params_map.get(K_val, sp.Symbol(K_val)) if isinstance(K_val, str) else sp.S(K_val)
            
            K_vec = [sp.S(0)] * 3
            if bond_dir == 'x': K_vec[0] = K_sym
            elif bond_dir == 'y': K_vec[1] = K_sym
            elif bond_dir == 'z': K_vec[2] = K_sym
            
            _kex_matrix_python_list[idx_i_uc][idx_j_ouc] = sp.Matrix(K_vec)

    DM_matrix_sym = sp.Matrix(N_atom_uc, N_atom_ouc, lambda r, c: _dm_matrix_python_list[r][c])
    Kex_matrix_sym = sp.Matrix(N_atom_uc, N_atom_ouc, lambda r, c: _kex_matrix_python_list[r][c])

    return Jex_sym, DM_matrix_sym, Kex_matrix_sym


def Hamiltonian_from_config(
    Sxyz_operators_ouc: List[sp.Matrix],
    symbolic_parameters: Dict[str, sp.Symbol],
    config_data: Dict[str, Any],
) -> sp.Expr:
    uc_vecs = unit_cell_from_config(config_data["crystal_structure"])
    atom_pos_uc = atom_pos_from_config(config_data["crystal_structure"], uc_vecs)
    atom_pos_ouc = atom_pos_ouc_from_config(
        atom_pos_uc, uc_vecs, config_data.get("calculation_settings", {})
    )
    if len(Sxyz_operators_ouc) != len(atom_pos_ouc):
        raise ValueError(
            f"Sxyz_operators_ouc len {len(Sxyz_operators_ouc)} != atom_pos_ouc len {len(atom_pos_ouc)}."
        )
    
    interactions_config = config_data.get("interactions", {})
    if isinstance(interactions_config, list):
        grouped = {}
        for inter in interactions_config:
            itype = inter.get("type")
            if itype:
                if itype not in grouped:
                    grouped[itype] = []
                grouped[itype].append(inter)
        interactions_config = grouped

    Jex_sym, DM_matrix_sym, Kex_matrix_sym = spin_interactions_from_config(
        symbolic_parameters,
        interactions_config,
        atom_pos_uc,
        atom_pos_ouc,
        uc_vecs,
    )
    HM_expr = sp.S(0)
    phys_consts = config_data.get("physical_constants", {})
    gamma_val = float(phys_consts.get("g_factor", 2.0))
    mu_B_val = float(phys_consts.get("bohr_magneton_meV_T", 5.7883818066e-2))

    N_atoms_in_uc = len(atom_pos_uc)
    N_atoms_total_ouc = len(Sxyz_operators_ouc)
    for i in range(N_atoms_in_uc):
        S_i = Sxyz_operators_ouc[i]
        for j_ouc in range(N_atoms_total_ouc):
            S_j_ouc = Sxyz_operators_ouc[j_ouc]
            
            # Heisenberg
            if Jex_sym[i, j_ouc] != 0:
                HM_expr += (
                    0.5 * Jex_sym[i, j_ouc] * (S_i.T * S_j_ouc)[0, 0]
                )

            # DM Interaction
            DM_vec_ij = DM_matrix_sym[i, j_ouc]
            if hasattr(DM_vec_ij, "is_zero_matrix") and DM_vec_ij.is_zero_matrix is not True:
                Dx_ij, Dy_ij, Dz_ij = DM_vec_ij[0], DM_vec_ij[1], DM_vec_ij[2]
                term_DM = (
                    Dx_ij * (S_i[1] * S_j_ouc[2] - S_i[2] * S_j_ouc[1])
                    + Dy_ij * (S_i[2] * S_j_ouc[0] - S_i[0] * S_j_ouc[2])
                    + Dz_ij * (S_i[0] * S_j_ouc[1] - S_i[1] * S_j_ouc[0])
                )
                HM_expr += 0.5 * term_DM

            # Anisotropic Exchange / Kitaev
            K_mat_ij = Kex_matrix_sym[i, j_ouc]
            if hasattr(K_mat_ij, "is_zero_matrix") and K_mat_ij.is_zero_matrix is not True:
                if K_mat_ij.shape == (3, 3):
                    term = 0.5 * (S_i.T * K_mat_ij * S_j_ouc)[0, 0]
                    HM_expr += term
                else:
                    term = 0.5 * (
                        K_mat_ij[0] * S_i[0] * S_j_ouc[0] +
                        K_mat_ij[1] * S_i[1] * S_j_ouc[1] +
                        K_mat_ij[2] * S_i[2] * S_j_ouc[2]
                    )
                    HM_expr += term
                print(f"DEBUG: HM_from_config - Added K term, cumulative count: {len(HM_expr.as_ordered_terms())}")

        atom_i_label = _atom_index_to_label_uc.get(i)
        if atom_i_label:
            sia_list = interactions_config.get("single_ion_anisotropy", []) + interactions_config.get("sia", [])
            for sia_def in sia_list:
                if sia_def.get("atom_label") == atom_i_label:
                    stype = sia_def.get("type", "sia")
                    if stype == "uniaxial_K_Sz_sq_global":
                        K_val_str = sia_def.get("K") or sia_def.get("K_global") or sia_def.get("value")
                        if K_val_str:
                            K_sym = symbolic_parameters.get(K_val_str, sp.Symbol(K_val_str)) if isinstance(K_val_str, str) else sp.S(K_val_str)
                            HM_expr += K_sym * S_i[2, 0] ** 2
                    elif stype == "sia":
                        K_val = sia_def.get("K") or sia_def.get("value")
                        if K_val is not None:
                            K_sym = symbolic_parameters.get(K_val, sp.Symbol(K_val)) if isinstance(K_val, str) else sp.S(K_val)
                            axis = sia_def.get("axis", [0, 0, 1])
                            axis_vec = np.array(axis, dtype=float)
                            norm = np.linalg.norm(axis_vec)
                            if norm > 1e-9: axis_vec /= norm
                            axis_sym = sp.Matrix(axis_vec)
                            
                            S_dot_n = (S_i.T * axis_sym)[0, 0]
                            HM_expr += K_sym * S_dot_n ** 2

        H_field_config = interactions_config.get("applied_field", {})
        if H_field_config:
            if isinstance(H_field_config, list): H_field_config = H_field_config[0]
            H_vec_config = H_field_config.get("H_vector", [0, 0, 0])
            H_mag_symbol_name = H_field_config.get("H_magnitude_symbol")
            actual_H_vector_sympy_comps = [sp.S(0)] * 3

            if H_mag_symbol_name:
                H_mag_val_sym = symbolic_parameters.get(
                    H_mag_symbol_name, sp.Symbol(H_mag_symbol_name)
                )
                
                if isinstance(H_vec_config, str):
                    h_dir_val = symbolic_parameters.get(H_vec_config, sp.Symbol(H_vec_config))
                    if isinstance(h_dir_val, (list, tuple, np.ndarray, sp.Matrix)):
                        for k_ax in range(3):
                            actual_H_vector_sympy_comps[k_ax] = h_dir_val[k_ax] * H_mag_val_sym
                elif isinstance(H_vec_config, (list, tuple)):
                    direction_H_numeric = np.array(H_vec_config, dtype=float)
                    norm_dir = np.linalg.norm(direction_H_numeric)
                    if norm_dir > DIST_TOL:
                        unit_dir_H = direction_H_numeric / norm_dir
                        for k_ax in range(3):
                            actual_H_vector_sympy_comps[k_ax] = (
                                unit_dir_H[k_ax] * H_mag_val_sym
                            )
            else:
                if isinstance(H_vec_config, str):
                    h_vec_val = symbolic_parameters.get(H_vec_config, sp.Symbol(H_vec_config))
                    if isinstance(h_vec_val, (list, tuple, np.ndarray, sp.Matrix)):
                         actual_H_vector_sympy_comps = [sp.S(v) for v in h_vec_val]
                else:
                    for k_ax in range(3):
                        comp = H_vec_config[k_ax]
                        actual_H_vector_sympy_comps[k_ax] = (
                            symbolic_parameters.get(comp, sp.Symbol(comp))
                            if isinstance(comp, str)
                            else sp.S(comp)
                        )

            if any(c != 0 for c in actual_H_vector_sympy_comps):
                H_field_col_vec = sp.Matrix(actual_H_vector_sympy_comps)
                Zeeman_dot_product = (S_i.T * H_field_col_vec)[0, 0]
                HM_expr += gamma_val * mu_B_val * Zeeman_dot_product
    return HM_expr.expand()


# --- Standard Interface Wrappers for Runner/Core Compatibility ---
config: Dict[str, Any] = {}


def _params_list_to_dict(params_list):
    if isinstance(params_list, dict):
        return params_list
    p_dict = config.get("model_params") or config.get("parameters") or {}
    
    if config.get("parameter_order"):
        keys = config.get("parameter_order")
        keys = [k for k in keys if k in p_dict]
        remaining_keys = [k for k in p_dict.keys() if k not in keys]
        keys.extend(remaining_keys)
    else:
        keys = list(p_dict.keys())
    
    result = {}
    p_idx = 0
    for k in keys:
        if p_idx >= len(params_list):
            break
            
        val = p_dict[k]
        if isinstance(val, (list, tuple, np.ndarray)):
            ndim = len(val)
            if isinstance(params_list[p_idx], (list, tuple, np.ndarray, sp.Matrix)):
                result[k] = params_list[p_idx]
                p_idx += 1
            else:
                result[k] = params_list[p_idx : p_idx + ndim]
                p_idx += ndim
        else:
            result[k] = params_list[p_idx]
            p_idx += 1
            
    return result


def unit_cell():
    if not config:
        raise RuntimeError("generic_spin_model: config not injected.")
    return unit_cell_from_config(config.get("crystal_structure", {}))


def atom_pos():
    if not config:
        raise RuntimeError("generic_spin_model: config not injected.")
    uc = unit_cell()
    return atom_pos_from_config(config.get("crystal_structure", {}), uc)


def atom_pos_ouc():
    if not config:
        raise RuntimeError("generic_spin_model: config not injected.")
    uc = unit_cell()
    pos_uc = atom_pos()
    return atom_pos_ouc_from_config(pos_uc, uc, config.get("calculation_settings", {}))


def mpr(symbolic_params):
    if not config:
        raise RuntimeError("generic_spin_model: config not injected.")
    return mpr_from_config(config, _params_list_to_dict(symbolic_params))


def Hamiltonian(Sxyz_operators_ouc: List[sp.Matrix], symbolic_parameters: Union[List[sp.Symbol], Dict[str, sp.Symbol]]) -> sp.Expr:
    if not config:
        raise RuntimeError("generic_spin_model: config not injected.")
    return Hamiltonian_from_config(Sxyz_operators_ouc, _params_list_to_dict(symbolic_parameters), config)


def spin_interactions(symbolic_params):
    if not config:
        raise RuntimeError("generic_spin_model: config not injected.")
    uc = unit_cell()
    pos_uc = atom_pos()
    pos_ouc = atom_pos_ouc()
    return spin_interactions_from_config(_params_list_to_dict(symbolic_params), config.get("interactions", {}), pos_uc, pos_ouc, uc)


def set_magnetic_structure(thetas, phis):
    if not config:
        return
    nspins = len(thetas)
    if "minimization" not in config:
        config["minimization"] = {}
    new_conf = []
    for i in range(nspins):
        new_conf.append({
            "atom_index": i,
            "theta": float(thetas[i]),
            "phi": float(phis[i])
        })
    config["minimization"]["initial_configuration"] = new_conf
