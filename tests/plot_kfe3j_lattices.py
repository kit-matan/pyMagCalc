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
    project_root_dir = os.path.dirname(pyMagCalc_dir)
    if project_root_dir not in sys.path:
        sys.path.insert(0, project_root_dir)

    # Add KFe3J module path
    kfe3j_module_path = os.path.join(project_root_dir, "KFe3J")
    if kfe3j_module_path not in sys.path:
        sys.path.insert(0, kfe3j_module_path)

except NameError:  # __file__ is not defined (e.g. in interactive interpreter)
    print("Warning: __file__ not defined, sys.path modification might be incomplete.")
    pass

from pyMagCalc.config_loader import load_spin_model_config
from pyMagCalc import generic_spin_model as gsm
import spin_model as kfe3j_hardcoded_model  # From KFe3J directory


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
    config_file_path = os.path.join(pyMagCalc_dir, "KFe3J_config.yaml")
    print(f"Attempting to load config from: {config_file_path}")

    config_data = load_spin_model_config(config_file_path)

    uc_vectors_conf_cart = gsm.unit_cell_from_config(config_data["crystal_structure"])
    atom_pos_conf_uc_cart = gsm.atom_pos_from_config(
        config_data["crystal_structure"], uc_vectors_conf_cart
    )
    atom_labels_conf_uc = [
        atom["label"] for atom in config_data["crystal_structure"]["atoms_uc"]
    ]

    # OUC atoms for visualization
    atom_pos_conf_ouc_cart = gsm.atom_pos_ouc_from_config(
        atom_pos_conf_uc_cart,
        uc_vectors_conf_cart,
        config_data.get("calculation_settings", {}),
    )
    # For OUC labels, we need to map back to UC labels based on periodicity
    # This is a simplified labeling for OUC, true labels would require more logic
    ouc_labels_conf = []
    for i in range(len(atom_pos_conf_ouc_cart)):
        ouc_labels_conf.append(atom_labels_conf_uc[i % len(atom_labels_conf_uc)])

    # Spins for config-driven
    # mpr_from_config expects symbolic parameters map
    rotation_matrices_conf_sym = gsm.mpr_from_config(
        config_data["crystal_structure"], params_sym_kfe3j_map
    )
    spin_vectors_conf_cart = []
    base_spin_local = sp.Matrix([0, 0, S_val])
    for rot_mat_sym in rotation_matrices_conf_sym:
        rot_mat_num = rot_mat_sym.subs(params_subs_kfe3j_dict).evalf()
        spin_global_sym = rot_mat_num * base_spin_local
        spin_vectors_conf_cart.append(np.array(spin_global_sym, dtype=float).flatten())
    spin_vectors_conf_cart = np.array(spin_vectors_conf_cart)

    # Extract DM vectors for config-driven
    # We need Jex_sym and DM_matrix_sym from spin_interactions_from_config
    # atom_labels_uc was already extracted as atom_labels_conf_uc
    # uc_vectors_cart is uc_vectors_conf_cart
    # atom_pos_ouc_cart is atom_pos_conf_ouc_cart
    _Jex_conf, DM_matrix_conf_sym = gsm.spin_interactions_from_config(
        params_sym_kfe3j_map,
        config_data["interactions"],
        atom_pos_conf_uc_cart,
        # atom_labels_conf_uc, # This was removed to match a 5-argument function signature if gsm.spin_interactions_from_config expects 5 args.
        atom_pos_conf_ouc_cart,
        uc_vectors_conf_cart,
    )
    dm_vectors_conf_plot_data = []
    for i_uc in range(len(atom_pos_conf_uc_cart)):
        for j_ouc in range(len(atom_pos_conf_ouc_cart)):
            D_vec_sym = DM_matrix_conf_sym[i_uc, j_ouc]
            if (
                D_vec_sym != sp.Matrix([0, 0, 0]) and not D_vec_sym.is_zero_matrix
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

    # --- 2. Hardcoded KFe3J ---
    print("Loading Hardcoded KFe3J model...")
    uc_basis_hardcoded = kfe3j_hardcoded_model.unit_cell()  # These are basis vectors
    atom_pos_hardcoded_frac = kfe3j_hardcoded_model.atom_pos()
    atom_labels_hardcoded_uc = [
        f"Fe{i+1}" for i in range(len(atom_pos_hardcoded_frac))
    ]  # Assuming Fe1, Fe2, Fe3

    # Convert fractional to Cartesian for hardcoded model
    # For plotting, let's use the same Cartesian cell definition as the config for direct comparison of atom placement
    # If we wanted to plot the primitive cell of hardcoded, we'd use uc_basis_hardcoded
    # For now, let's assume the hardcoded fractional positions are relative to a cell
    # that can be mapped to the config's Cartesian cell for visualization.
    # The hardcoded model's unit_cell() defines a primitive cell.
    # The config uses a conventional hexagonal cell.
    # To plot hardcoded atoms in the context of the *config's cell*:
    # This requires careful mapping or redefinition.
    # For simplicity, let's plot the hardcoded model in its own primitive cell.
    atom_pos_hardcoded_uc_cart = np.array(
        [
            f[0] * uc_basis_hardcoded[0]
            + f[1] * uc_basis_hardcoded[1]
            + f[2] * uc_basis_hardcoded[2]
            for f in atom_pos_hardcoded_frac
        ]
    )

    # OUC for hardcoded
    atom_pos_hardcoded_ouc_frac = kfe3j_hardcoded_model.atom_pos_ouc()
    atom_pos_hardcoded_ouc_cart = np.array(
        [
            f[0] * uc_basis_hardcoded[0]
            + f[1] * uc_basis_hardcoded[1]
            + f[2] * uc_basis_hardcoded[2]
            for f in atom_pos_hardcoded_ouc_frac
        ]
    )
    ouc_labels_hardcoded = []
    for i in range(len(atom_pos_hardcoded_ouc_cart)):
        ouc_labels_hardcoded.append(
            atom_labels_hardcoded_uc[i % len(atom_labels_hardcoded_uc)]
        )

    # Spins for hardcoded
    rotation_matrices_hardcoded_sym = kfe3j_hardcoded_model.mpr(
        list(params_sym_kfe3j_map.values())
    )  # Pass symbolic params
    spin_vectors_hardcoded_cart = []
    for rot_mat_sym in rotation_matrices_hardcoded_sym:
        rot_mat_num = rot_mat_sym.subs(params_subs_kfe3j_dict).evalf()
        spin_global_sym = rot_mat_num * base_spin_local
        spin_vectors_hardcoded_cart.append(
            np.array(spin_global_sym, dtype=float).flatten()
        )
    spin_vectors_hardcoded_cart = np.array(spin_vectors_hardcoded_cart)

    # Extract DM vectors for hardcoded
    _Jex_hardcoded, DM_hardcoded_sym_mat = kfe3j_hardcoded_model.spin_interactions(
        list(params_sym_kfe3j_map.values())
    )
    dm_vectors_hardcoded_plot_data = []
    # N_atom_hardcoded = len(atom_pos_hardcoded_uc_cart)
    # N_atom_ouc_hardcoded = len(atom_pos_hardcoded_ouc_cart)

    for i_uc in range(DM_hardcoded_sym_mat.rows):  # Iterate up to N_atom (UC)
        for j_ouc in range(DM_hardcoded_sym_mat.cols):  # Iterate up to N_atom_ouc
            # DM_hardcoded_sym_mat elements are already substituted by kfe3j_hardcoded_model.spin_interactions
            # if numerical parameters were passed. Here we passed symbolic, so we need to sub.
            D_vec_sym_hardcoded = DM_hardcoded_sym_mat[i_uc, j_ouc]
            if (
                D_vec_sym_hardcoded != sp.Matrix([0, 0, 0])
                and not D_vec_sym_hardcoded.is_zero_matrix
            ):
                D_vec_num_hardcoded = np.array(
                    D_vec_sym_hardcoded.subs(params_subs_kfe3j_dict).evalf(),
                    dtype=float,
                ).flatten()
                if np.linalg.norm(D_vec_num_hardcoded) > 1e-6:
                    pos_i = atom_pos_hardcoded_uc_cart[i_uc]
                    pos_j = atom_pos_hardcoded_ouc_cart[
                        j_ouc
                    ]  # j_ouc is the index in the OUC list
                    midpoint = (pos_i + pos_j) / 2
                    dm_vectors_hardcoded_plot_data.append(
                        {
                            "midpoint": midpoint,
                            "vector": D_vec_num_hardcoded,
                            "color": "darkgreen",
                        }
                    )

    # --- Compare DM Matrices Numerically ---
    print("\n--- Comparing DM Matrices Numerically ---")
    if DM_matrix_conf_sym.shape == DM_hardcoded_sym_mat.shape:
        print(f"DM Matrix shapes match: {DM_matrix_conf_sym.shape}")
        mismatches_found = 0
        for r in range(DM_matrix_conf_sym.rows):
            for c in range(DM_matrix_conf_sym.cols):
                dm_vec_conf_sym = DM_matrix_conf_sym[r, c]
                dm_vec_hard_sym = DM_hardcoded_sym_mat[r, c]

                # Convert to numerical by substituting parameters
                # Ensure the elements are actual SymPy matrices before evalf
                if isinstance(dm_vec_conf_sym, sp.MatrixBase):
                    dm_vec_conf_num = np.array(
                        dm_vec_conf_sym.subs(params_subs_kfe3j_dict).evalf(),
                        dtype=float,
                    ).flatten()
                else:  # Handle cases where it might be 0 or other non-matrix type if Jex was 0
                    dm_vec_conf_num = (
                        np.array([0.0, 0.0, 0.0])
                        if dm_vec_conf_sym == 0
                        else np.array(dm_vec_conf_sym, dtype=float).flatten()
                    )

                if isinstance(dm_vec_hard_sym, sp.MatrixBase):
                    dm_vec_hard_num = np.array(
                        dm_vec_hard_sym.subs(params_subs_kfe3j_dict).evalf(),
                        dtype=float,
                    ).flatten()
                else:
                    dm_vec_hard_num = (
                        np.array([0.0, 0.0, 0.0])
                        if dm_vec_hard_sym == 0
                        else np.array(dm_vec_hard_sym, dtype=float).flatten()
                    )

                if not np.allclose(dm_vec_conf_num, dm_vec_hard_num, atol=1e-7):
                    if mismatches_found < 5:  # Print first few mismatches
                        print(f"Mismatch at DM_matrix[{r},{c}]:")
                        print(f"  Config-driven: {dm_vec_conf_num}")
                        print(f"  Hardcoded:     {dm_vec_hard_num}")
                        print(f"  Difference:    {dm_vec_conf_num - dm_vec_hard_num}")
                    mismatches_found += 1

        if mismatches_found == 0:
            print("SUCCESS: Numerically evaluated DM matrices are identical.")
        else:
            print(
                f"FAILURE: Found {mismatches_found} mismatches in numerically evaluated DM matrices."
            )
    else:
        print("FAILURE: DM Matrix shapes do NOT match.")
        print(f"  Config-driven DM_matrix shape: {DM_matrix_conf_sym.shape}")
        print(f"  Hardcoded DM_matrix shape:     {DM_hardcoded_sym_mat.shape}")
        print("Cannot perform element-wise numerical comparison.")

    # --- Set Fixed Axis Limits for Both Plots ---
    fixed_axis_limit = (-1.5, 1.5)
    xlims = fixed_axis_limit
    ylims = fixed_axis_limit
    zlims = fixed_axis_limit

    # --- Plotting ---
    fig = plt.figure(figsize=(18, 9))

    # Plot Config-driven
    ax1 = fig.add_subplot(121, projection="3d")
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

    # Plot Hardcoded
    # Note: The hardcoded model defines a primitive cell. We plot this primitive cell.
    # The config uses a larger conventional cell. Direct overlay might be confusing without transformation.
    ax2 = fig.add_subplot(122, projection="3d")
    plot_lattice_3d(
        ax2,
        uc_basis_hardcoded,
        atom_pos_hardcoded_uc_cart,
        atom_labels_hardcoded_uc,
        spin_vectors_hardcoded_cart,
        title="KFe3J (Hardcoded - Primitive Cell)",
        plot_ouc_atoms=atom_pos_hardcoded_ouc_cart,
        dm_vectors_plot_data=dm_vectors_hardcoded_plot_data,
        ouc_atom_labels=ouc_labels_hardcoded,
    )
    ax2.set_xlim(xlims)
    ax2.set_ylim(ylims)
    ax2.set_zlim(zlims)

    plt.tight_layout()
    plt.show()

    print("Plotting script finished.")
