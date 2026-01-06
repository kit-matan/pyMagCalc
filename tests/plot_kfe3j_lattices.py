#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to visualize and compare KFe3J lattices from:
1. Configuration-driven approach (generic_spin_model.py + kfe3j_config.yaml)
2. Hardcoded approach (KFe3J/spin_model.py)
"""
import os
import sys
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# --- Add project root to sys.path for correct package imports ---
try:
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root_dir is the parent directory of 'pyMagCalc'
    pyMagCalc_dir = os.path.abspath(os.path.join(current_script_dir, ".."))
    
    # Add pyMagCalc_dir to sys.path to allow 'import magcalc'
    if pyMagCalc_dir not in sys.path:
        sys.path.insert(0, pyMagCalc_dir)

    project_root_dir = os.path.dirname(pyMagCalc_dir)
    # if project_root_dir not in sys.path:
    #     sys.path.insert(0, project_root_dir)

    # Add KFe3J module path
    kfe3j_module_path = os.path.join(project_root_dir, "KFe3J")
    if kfe3j_module_path not in sys.path:
        sys.path.insert(0, kfe3j_module_path)

except NameError:  # __file__ is not defined (e.g. in interactive interpreter)
    print("Warning: __file__ not defined, sys.path modification might be incomplete.")
    pass

from magcalc.config_loader import load_spin_model_config
from magcalc.generic_model import GenericSpinModel



def plot_lattice_3d(
    ax,
    uc_vectors_cart,
    atom_pos_cart,
    atom_labels,
    spin_vectors_cart=None,
    title="Lattice Structure",
    plot_ouc_atoms=None,
    dm_vectors_plot_data=None,  # New parameter for DM vectors
    ouc_atom_labels=None,
):
    """
    Plots the unit cell, atoms, and optionally spin vectors in 3D.
    """
    ax.set_title(title)

    # Draw unit cell boundaries
    origin = np.array([0, 0, 0])
    v1, v2, v3 = uc_vectors_cart
    vertices = [origin, v1, v2, v3, v1 + v2, v1 + v3, v2 + v3, v1 + v2 + v3]
    edges = [
        (origin, v1),
        (origin, v2),
        (origin, v3),
        (v1, v1 + v2),
        (v1, v1 + v3),
        (v2, v1 + v2),
        (v2, v2 + v3),
        (v3, v1 + v3),
        (v3, v2 + v3),
        (v1 + v2, v1 + v2 + v3),
        (v1 + v3, v1 + v2 + v3),
        (v2 + v3, v1 + v2 + v3),
    ]
    for p1, p2 in edges:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], "k-", alpha=0.5)

    # Plot atoms in UC
    ax.scatter(
        atom_pos_cart[:, 0],
        atom_pos_cart[:, 1],
        atom_pos_cart[:, 2],
        s=100,
        c="blue",
        label="UC Atoms",
    )
    for i, label in enumerate(atom_labels):
        ax.text(
            atom_pos_cart[i, 0],
            atom_pos_cart[i, 1],
            atom_pos_cart[i, 2],
            f" {label}",
            color="black",
        )

    # Plot OUC atoms if provided
    if plot_ouc_atoms is not None:
        ax.scatter(
            plot_ouc_atoms[:, 0],
            plot_ouc_atoms[:, 1],
            plot_ouc_atoms[:, 2],
            s=50,
            c="lightblue",
            alpha=0.5,
            label="OUC Atoms",
        )
        if ouc_atom_labels:
            for i, label in enumerate(ouc_atom_labels):
                ax.text(
                    plot_ouc_atoms[i, 0],
                    plot_ouc_atoms[i, 1],
                    plot_ouc_atoms[i, 2],
                    f" {label}",
                    color="gray",
                    fontsize=8,
                )

    # Plot spin vectors
    if spin_vectors_cart is not None:
        for i in range(len(atom_pos_cart)):
            start_point = atom_pos_cart[i]
            end_point = spin_vectors_cart[
                i
            ]  # Spin vectors are directions, scale for plotting
            ax.quiver(
                start_point[0],
                start_point[1],
                start_point[2],
                end_point[0],
                end_point[1],
                end_point[2],
                length=1.0,
                normalize=True,
                color="red",
                arrow_length_ratio=0.3,
            )

    # Plot DM vectors
    if dm_vectors_plot_data:
        for dm_data in dm_vectors_plot_data:
            midpoint = dm_data["midpoint"]
            vector = dm_data["vector"]
            color = dm_data.get("color", "purple")  # Default color for DM vectors
            # Normalize DM vector for consistent arrow length, then scale
            norm_vector = vector / (
                np.linalg.norm(vector) + 1e-9
            )  # Add epsilon to avoid division by zero
            ax.quiver(
                midpoint[0],
                midpoint[1],
                midpoint[2],
                norm_vector[0],
                norm_vector[1],
                norm_vector[2],
                length=0.5,  # Visual length of DM vector arrows
                color=color,
                arrow_length_ratio=0.4,
                label="DM Vector" if dm_data is dm_vectors_plot_data[0] else None,
            )  # Label only once

    # Axis limits will be set externally for consistency between subplots
    # However, we can still set labels and view angle here.
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    ax.legend()
    ax.view_init(elev=20.0, azim=-35)


if __name__ == "__main__":
    # --- Parameters for KFe3J ---
    S_val = 2.5
    # J1, J2, Dy, Dz, H_field
    params_numerical_kfe3j = [3.23, 0.11, 0.218, -0.195, 0.0]
    params_sym_kfe3j_map = {
        "J1": sp.Symbol("J1"),
        "J2": sp.Symbol("J2"),
        "Dy": sp.Symbol("Dy"),
        "Dz": sp.Symbol("Dz"),
        "H_field": sp.Symbol("H_field"),
    }
    params_subs_kfe3j = list(zip(params_sym_kfe3j_map.values(), params_numerical_kfe3j))
    params_subs_kfe3j_dict = dict(params_subs_kfe3j)

    # --- 1. Configuration-driven KFe3J ---
    print("Loading Configuration-driven KFe3J model...")
    # Config file is KFe3J_config.yaml and located in pyMagCalc_dir (one level up from current_script_dir)
    config_file_path = os.path.join(pyMagCalc_dir, "examples", "KFe3J", "config_kfe3j.yaml")
    print(f"Attempting to load config from: {config_file_path}")

    config_data = load_spin_model_config(config_file_path)

    # Instantiate GenericSpinModel
    gsm_model = GenericSpinModel(config_data, base_path=pyMagCalc_dir)

    uc_vectors_conf_cart = gsm_model.unit_cell()
    atom_pos_conf_uc_cart = gsm_model.atom_pos()
    atom_labels_conf_uc = [
        atom["label"] for atom in config_data["crystal_structure"]["atoms_uc"]
    ]

    # OUC atoms for visualization
    atom_pos_conf_ouc_cart = gsm_model.atom_pos_ouc()
    
    # For OUC labels, we need to map back to UC labels based on periodicity
    # This is a simplified labeling for OUC, true labels would require more logic
    ouc_labels_conf = []
    for i in range(len(atom_pos_conf_ouc_cart)):
        ouc_labels_conf.append(atom_labels_conf_uc[i % len(atom_labels_conf_uc)])

    # Spins for config-driven
    # GenericSpinModel expects a list of parameter values in the correct order, NOT a map
    # We need to reconstruct the parameter list expected by GSM based on its internal parameter_order
    # OR we need to modify our usage to support named params if GSM supports it (it partially does)
    # The safest way is to use the symbolic params list we created, hoping order matches if we passed config properly.
    # Actually, GSM.mpr takes 'p' which is a list.
    # Let's see what params GSM expects.
    # gsm_model.params_sym is set internally based on config.
    
    # We will build the numerical parameter list for GSM
    # GSM uses config['parameter_order'] or keys of config['parameters']
    # We'll just build a list of symbols for now to get symbolic output
    
    # Re-using our manual map might be risky if ordering differs. 
    # Let's rely on mapping by NAME.
    
    # GSM expects 'p' to be a list of values/symbols corresponding to its internal parameter list.
    # However, we want to control the symbols. 
    # Let's inspect gsm_model.params_sym
    # But gsm_model logic for params is internal.
    
    # Wait, GenericSpinModel.spin_interactions(p) takes a LIST 'p'.
    # We need to construct 'p' such that it matches the order in gsm_model.
    # BUT, we want to inject OUR symbols so we can subs them.
    
    # Workaround: Manually build a list of our symbols in the order GSM wants.
    gsm_param_names = gsm_model.config.get('parameter_order', gsm_model.config.get('parameters').keys())
    gsm_p_list = []
    for k in gsm_param_names:
        if k in params_sym_kfe3j_map:
            gsm_p_list.append(params_sym_kfe3j_map[k])
        elif k != 'S': # S is handled separately usually or excluded
            gsm_p_list.append(sp.Symbol(k)) # Fallback

    rotation_matrices_conf_sym = gsm_model.mpr(gsm_p_list)
    
    spin_vectors_conf_cart = []
    base_spin_local = sp.Matrix([0, 0, S_val])
    for rot_mat_sym in rotation_matrices_conf_sym:
        rot_mat_num = rot_mat_sym.subs(params_subs_kfe3j_dict).evalf()
        spin_global_sym = rot_mat_num * base_spin_local
        spin_vectors_conf_cart.append(np.array(spin_global_sym, dtype=float).flatten())
    spin_vectors_conf_cart = np.array(spin_vectors_conf_cart)

    # Extract DM vectors for config-driven
    _Jex_conf, DM_matrix_conf_sym, _Kex_conf = gsm_model.spin_interactions(gsm_p_list)
    
    dm_vectors_conf_plot_data = []
    for i_uc in range(len(atom_pos_conf_uc_cart)):
        for j_ouc in range(len(atom_pos_conf_ouc_cart)):
            D_vec_sym = DM_matrix_conf_sym[i_uc][j_ouc]
            if D_vec_sym is not None and (
                D_vec_sym != sp.Matrix([0, 0, 0]) and not (hasattr(D_vec_sym, 'is_zero_matrix') and D_vec_sym.is_zero_matrix)
            ):  # Check if it's not a zero matrix
                D_vec_num = np.array(
                    D_vec_sym.subs(params_subs_kfe3j_dict).evalf(), dtype=float
                ).flatten()
                if (
                    np.linalg.norm(D_vec_num) > 1e-6
                ):  # Only plot non-negligible DM vectors
                    pos_i = atom_pos_conf_uc_cart[i_uc]
                    pos_j = atom_pos_conf_ouc_cart[j_ouc]
                    midpoint = (pos_i + pos_j) / 2
                    dm_vectors_conf_plot_data.append(
                        {"midpoint": midpoint, "vector": D_vec_num, "color": "purple"}
                    )

    # --- 2. Hardcoded KFe3J (REMOVED - Module not available) ---
    print("Skipping Hardcoded KFe3J model comparison (module not found).")
    
    # --- Set Fixed Axis Limits ---
    fixed_axis_limit = (-1.5, 1.5)
    xlims = fixed_axis_limit
    ylims = fixed_axis_limit
    zlims = fixed_axis_limit

    # --- Plotting ---
    fig = plt.figure(figsize=(9, 9))

    # Plot Config-driven
    ax1 = fig.add_subplot(111, projection="3d")
    plot_lattice_3d(
        ax1,
        uc_vectors_conf_cart,
        atom_pos_conf_uc_cart,
        atom_labels_conf_uc,
        spin_vectors_conf_cart,
        title="KFe3J (Config-driven)",
        plot_ouc_atoms=atom_pos_conf_ouc_cart,
        dm_vectors_plot_data=dm_vectors_conf_plot_data,
        ouc_atom_labels=ouc_labels_conf,
    )
    ax1.set_xlim(xlims)
    ax1.set_ylim(ylims)
    ax1.set_zlim(zlims)

    # plt.tight_layout()
    # plt.show()

    print("Plotting script finished (Config-driven only).")
