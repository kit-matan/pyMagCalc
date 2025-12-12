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
    config_crystal_structure: Dict[str, Any], symbolic_params_map: Dict[str, sp.Symbol]
) -> List[sp.Matrix]:
    J1 = symbolic_params_map.get("J1", sp.Symbol("J1"))
    J2 = symbolic_params_map.get("J2", sp.Symbol("J2"))
    Dy = symbolic_params_map.get("Dy", sp.Symbol("Dy"))
    Dz = symbolic_params_map.get("Dz", sp.Symbol("Dz"))
    _sym_sqrt3 = sp.sqrt(sp.S(3))
    Z_expr = J2 - _sym_sqrt3 / 3 * Dz
    term_factor1 = sp.S(2) / (sp.S(3) * J1)
    term_factor2 = -_sym_sqrt3 * Dy
    term_factor3_inv = sp.S(1) + Z_expr / J1
    arg_asin = term_factor1 * term_factor2 / term_factor3_inv
    ca_expr = sp.Abs(sp.asin(arg_asin) / 2)
    mp_rot = sp.Matrix(
        [
            [sp.cos(ca_expr), 0, -sp.sin(ca_expr)],
            [0, 1, 0],
            [sp.sin(ca_expr), 0, sp.cos(ca_expr)],
        ]
    )
    _np_pi = np.pi
    al_float_based = [-1 / 3 * _np_pi, _np_pi, 1 / 3 * _np_pi]
    rot_m_list = []
    for ag_val_float in al_float_based:
        M_ag = sp.Matrix(
            [
                [0, sp.sin(ag_val_float), sp.cos(ag_val_float)],
                [0, -sp.cos(ag_val_float), sp.sin(ag_val_float)],
                [1, 0, 0],
            ]
        )
        rot_m_list.append(M_ag * mp_rot)
    return rot_m_list


def spin_interactions_from_config(
    symbolic_params_map: Dict[str, sp.Symbol],
    config_interactions: Dict[str, Any],
    atom_pos_uc_cartesian: np.ndarray,
    atom_pos_ouc_cartesian: np.ndarray,
    unit_cell_vectors_cartesian: np.ndarray,
) -> Tuple[sp.Matrix, sp.Matrix]:
    N_atom_uc = len(atom_pos_uc_cartesian)
    N_atom_ouc = len(atom_pos_ouc_cartesian)
    Jex_sym = sp.zeros(N_atom_uc, N_atom_ouc)

    for heis_inter_def in config_interactions.get("heisenberg", []):
        label_i, label_j = heis_inter_def["pair"]
        idx_i_uc = _atom_label_to_index_uc.get(label_i)
        idx_j_uc_config = _atom_label_to_index_uc.get(label_j)
        if idx_i_uc is None or idx_j_uc_config is None:
            logger.warning(
                f"Unknown atom label in Heisenberg: {label_i},{label_j}. Skip."
            )
            continue
        J_val_config = heis_inter_def["J"]
        J_sympy_val = symbolic_params_map.get(J_val_config, sp.Symbol(J_val_config))
        offset_frac = np.array(heis_inter_def.get("rij_offset", [0, 0, 0]), dtype=float)
        offset_cart = offset_frac @ unit_cell_vectors_cartesian
        pos_j_target_cart = atom_pos_uc_cartesian[idx_j_uc_config] + offset_cart
        idx_j_ouc = -1
        for k_ouc, pos_ouc_k_cart in enumerate(atom_pos_ouc_cartesian):
            if np.allclose(pos_j_target_cart, pos_ouc_k_cart, atol=DIST_TOL):
                idx_j_ouc = k_ouc
                break
        if idx_j_ouc != -1:
            if (
                Jex_sym[idx_i_uc, idx_j_ouc] != 0
                and Jex_sym[idx_i_uc, idx_j_ouc] != J_sympy_val
            ):
                logger.warning(
                    f"JEX OVERWRITE: Jex[{idx_i_uc},{idx_j_ouc}] from {Jex_sym[idx_i_uc, idx_j_ouc]} to {J_sympy_val}"
                )
            Jex_sym[idx_i_uc, idx_j_ouc] = J_sympy_val
        else:
            logger.error(
                f"CRITICAL JEX OUC MATCH FAIL: {label_i}-{label_j} offset {offset_frac}."
            )

    _dm_matrix_python_list = []
    default_null_dm_vector = sp.Matrix([sp.S(0), sp.S(0), sp.S(0)])
    for _ in range(N_atom_uc):
        _dm_matrix_python_list.append(
            [default_null_dm_vector.copy() for _ in range(N_atom_ouc)]
        )

    sympify_locals = {**symbolic_params_map, "sqrt": sp.sqrt, "S": sp.S}

    for dm_inter_def in config_interactions.get("dm_interaction", []):
        label_i, label_j = dm_inter_def["pair"]
        idx_i_uc_cfg = _atom_label_to_index_uc.get(label_i)
        idx_j_uc_config = _atom_label_to_index_uc.get(label_j)
        if idx_i_uc_cfg is None or idx_j_uc_config is None:
            logger.warning(f"Unknown atom label in DM: {label_i},{label_j}. Skip.")
            continue
        D_vector_config_comps = dm_inter_def["D_vector"]
        D_vec_sympy_comps_list = [
            sp.sympify(c, locals=sympify_locals) if isinstance(c, str) else sp.S(c)
            for c in D_vector_config_comps
        ]
        D_vector_sympy = sp.Matrix(D_vec_sympy_comps_list)
        offset_frac = np.array(dm_inter_def.get("rij_offset", [0, 0, 0]), dtype=float)
        offset_cart = offset_frac @ unit_cell_vectors_cartesian
        pos_j_target_cart = atom_pos_uc_cartesian[idx_j_uc_config] + offset_cart
        idx_j_ouc_cfg = -1
        for k_ouc, pos_ouc_k_cart in enumerate(atom_pos_ouc_cartesian):
            if np.allclose(pos_j_target_cart, pos_ouc_k_cart, atol=DIST_TOL):
                idx_j_ouc_cfg = k_ouc
                break
        if idx_j_ouc_cfg != -1:
            if (
                not _dm_matrix_python_list[idx_i_uc_cfg][idx_j_ouc_cfg].is_zero_matrix
                and _dm_matrix_python_list[idx_i_uc_cfg][idx_j_ouc_cfg]
                != D_vector_sympy
            ):
                logger.warning(
                    f"DM OVERWRITE: DM[{idx_i_uc_cfg},{idx_j_ouc_cfg}] from {_dm_matrix_python_list[idx_i_uc_cfg][idx_j_ouc_cfg]} to {D_vector_sympy}"
                )
            _dm_matrix_python_list[idx_i_uc_cfg][idx_j_ouc_cfg] = D_vector_sympy
        else:
            logger.error(
                f"CRITICAL DM OUC MATCH FAIL: {label_i}-{label_j} offset {offset_frac}."
            )

    DM_matrix_sym = sp.Matrix(
        N_atom_uc, N_atom_ouc, lambda r, c: _dm_matrix_python_list[r][c]
    )

    return Jex_sym, DM_matrix_sym


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
    Jex_sym, DM_matrix_sym = spin_interactions_from_config(
        symbolic_parameters,
        config_data["interactions"],
        atom_pos_uc,
        atom_pos_ouc,
        uc_vecs,
    )
    HM_expr = sp.S(0)
    phys_consts = config_data.get("physical_constants", {})
    # Use sp.Float to match spin_model.py if it uses Python floats
    gamma_val = float(phys_consts.get("g_factor", 2.0))
    mu_B_val = float(phys_consts.get("bohr_magneton_meV_T", 5.7883818066e-2))

    N_atoms_in_uc = len(atom_pos_uc)
    N_atoms_total_ouc = len(Sxyz_operators_ouc)
    for i in range(N_atoms_in_uc):
        S_i = Sxyz_operators_ouc[i]
        for j_ouc in range(N_atoms_total_ouc):
            S_j_ouc = Sxyz_operators_ouc[j_ouc]
            if Jex_sym[i, j_ouc] != 0:
                HM_expr += (
                    0.5 * Jex_sym[i, j_ouc] * (S_i.T * S_j_ouc)[0, 0]
                )  # Using float 0.5

            DM_vec_ij = DM_matrix_sym[i, j_ouc]
            if isinstance(DM_vec_ij, sp.Matrix) and not DM_vec_ij.is_zero_matrix:
                Dx_ij = DM_vec_ij[0, 0]
                Dy_ij = DM_vec_ij[1, 0]
                Dz_ij = DM_vec_ij[2, 0]

                term_DM = (
                    Dx_ij * (S_i[1, 0] * S_j_ouc[2, 0] - S_i[2, 0] * S_j_ouc[1, 0])
                    + Dy_ij * (S_i[2, 0] * S_j_ouc[0, 0] - S_i[0, 0] * S_j_ouc[2, 0])
                    + Dz_ij * (S_i[0, 0] * S_j_ouc[1, 0] - S_i[1, 0] * S_j_ouc[0, 0])
                )
                HM_expr += 0.5 * term_DM  # Using float 0.5, sign is POSITIVE

        atom_i_label = _atom_index_to_label_uc.get(i)
        if atom_i_label:
            for sia_def in config_data["interactions"].get("single_ion_anisotropy", []):
                if sia_def.get("atom_label") == atom_i_label:
                    if sia_def.get("type") == "uniaxial_K_Sz_sq_global":
                        K_val_str = sia_def.get("K_global")
                        if K_val_str:
                            HM_expr += (
                                symbolic_parameters.get(K_val_str, sp.Symbol(K_val_str))
                                * S_i[2, 0] ** 2
                            )

        H_field_config = config_data["interactions"].get("applied_field", {})
        if H_field_config:
            H_vec_config_comps = H_field_config.get("H_vector", [0, 0, 0])
            H_mag_symbol_name = H_field_config.get("H_magnitude_symbol")
            actual_H_vector_sympy_comps = [sp.S(0)] * 3
            if H_mag_symbol_name:
                H_mag_val_sym = symbolic_parameters.get(
                    H_mag_symbol_name, sp.Symbol(H_mag_symbol_name)
                )
                direction_H_numeric = np.array(H_vec_config_comps, dtype=float)
                norm_dir = np.linalg.norm(direction_H_numeric)
                if norm_dir > DIST_TOL:
                    unit_dir_H = direction_H_numeric / norm_dir
                    for k_ax in range(3):
                        actual_H_vector_sympy_comps[k_ax] = (
                            unit_dir_H[k_ax] * H_mag_val_sym
                        )
                elif not H_mag_val_sym.is_zero:
                    logger.warning(
                        f"Field symbol '{H_mag_symbol_name}' has zero direction. Field term is zero."
                    )
            else:
                for k_ax in range(3):
                    comp = H_vec_config_comps[k_ax]
                    actual_H_vector_sympy_comps[k_ax] = (
                        symbolic_parameters.get(comp, sp.Symbol(comp))
                        if isinstance(comp, str)
                        else sp.S(comp)
                    )
            if any(c != 0 for c in actual_H_vector_sympy_comps):
                H_field_col_vec = sp.Matrix(actual_H_vector_sympy_comps)
                Zeeman_dot_product = (S_i.T * H_field_col_vec)[0, 0]
                # Using float for gamma and mu_B to match spin_model.py
                HM_expr += gamma_val * mu_B_val * Zeeman_dot_product
    return HM_expr.expand()
