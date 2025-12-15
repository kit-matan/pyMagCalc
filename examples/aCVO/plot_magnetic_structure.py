#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualizes the magnetic structure of alpha-Cu2V2O7.

This script generates a 3D plot of one magnetic unit cell, showing:
- The positions of the 16 Cu atoms, labeled by their index.
- The ground-state spin vectors.
- The nearest-neighbor J1 exchange bonds.
- The Dzyaloshinskii-Moriya (DM) vectors for each J1 bond.

It uses the structural information and interaction logic from `spin_model_hc.py`.

@author: Kit Matan
@contributor: Gemini Code Assist
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

# --- Add pyMagCalc directory to sys.path to import the spin model ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

try:
    from aCVO.spin_model import (
        atom_pos,
        atom_pos_ouc,
        unit_cell,
        get_nearest_neighbor_distances,
        is_vector_parallel_to_another,
        is_right_neighbor,
        AL_SPIN_PREFERENCE,
    )
except ImportError as e:
    print(f"Error importing from spin_model.py: {e}")
    print(
        "Please ensure 'spin_model.py' is in the 'aCVO' directory and the project structure is correct."
    )
    sys.exit(1)


class Arrow3D(FancyArrowPatch):
    """A custom arrow patch for 3D plots."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        """Project the 3D arrow to 2D."""
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def visualize_magnetic_structure():
    """
    Generates and displays a 3D plot of the aCVO magnetic structure.
    """
    print("Generating magnetic structure visualization for aCVO...")

    # --- 1. Define Model Parameters and Get Structure ---
    # Use representative parameters to calculate DM vectors
    Dx, Dy = 2.67, -2.0
    S = 0.5

    apos_uc = atom_pos()
    apos_ouc = atom_pos_ouc()
    uc_vectors = unit_cell()
    n_uc = len(apos_uc)

    # --- 2. Define Ground State Spin Vectors ---
    spin_vectors = np.zeros((n_uc, 3))
    for i in range(n_uc):
        # Spins are along +/- a-axis (global x-axis)
        spin_vectors[i, 0] = S * AL_SPIN_PREFERENCE[i]

    # --- 3. Identify J1 Bonds and DM vectors ---
    # Get the distance corresponding to the J1 interaction
    all_distances = get_nearest_neighbor_distances(apos_uc, apos_ouc, num_neighbors=10)
    dist_j1 = all_distances[
        1
    ]  # J1 is the second unique distance (after self-interaction)

    j1_bonds = []
    dm_vectors = []

    for i in range(n_uc):  # Iterate through atoms in the unit cell
        for j in range(len(apos_ouc)):  # Iterate through all atoms in the extended cell
            dist_ij = np.linalg.norm(apos_uc[i] - apos_ouc[j])

            # Check if this is a J1 bond
            if np.isclose(dist_ij, dist_j1):
                # To avoid plotting each bond and its corresponding DM vector twice
                # (e.g., D_ij for bond i->j and D_ji for bond j->i), we establish
                # a canonical representation for each bond. We only plot the bond
                # that has a lexicographically "positive" direction vector.
                bond_direction_vec = apos_ouc[j] - apos_uc[i]

                # Skip if the bond direction is lexicographically negative.
                # The reverse bond (e.g., from j' to i') will have a positive
                # vector and will be selected for plotting instead.
                # A small tolerance is used for floating point comparisons.
                if bond_direction_vec[0] < -1e-9:
                    continue
                if (
                    np.isclose(bond_direction_vec[0], 0)
                    and bond_direction_vec[1] < -1e-9
                ):
                    continue
                if (
                    np.isclose(bond_direction_vec[0], 0)
                    and np.isclose(bond_direction_vec[1], 0)
                    and bond_direction_vec[2] < -1e-9
                ):
                    continue

                j1_bonds.append((i, j))

                # --- Calculate the DM vector for this specific bond ---
                # The a-component's sign is determined by the bond's orientation along the b-axis.
                sign = 1.0
                if is_right_neighbor(apos_uc[i], apos_ouc[j]):
                    sign = -1.0
                dm_a_component = sign * Dx

                # The bc-component of the DM vector is perpendicular to the bond in the bc-plane.
                # Project bond vector onto the bc-plane.
                bond_vec_bc = np.array(
                    [0, bond_direction_vec[1], bond_direction_vec[2]]
                )
                # Normalize the bc-plane projection of the bond vector.
                norm_bond_vec_bc = bond_vec_bc / np.linalg.norm(bond_vec_bc)

                # The direction perpendicular to the bond in the bc-plane is found
                # by swapping components and negating one: (y, z) -> (-z, y).
                dm_vec_bc_direction = np.array(
                    [0, -norm_bond_vec_bc[2], norm_bond_vec_bc[1]]
                )

                # The magnitude of the bc-component is determined by Dy, and its sign
                # by the spin preference of the starting atom.
                dm_bc_component = AL_SPIN_PREFERENCE[i] * Dy * dm_vec_bc_direction

                # Construct the final DM vector from its components.
                final_dm_vec = np.array(
                    [dm_a_component, dm_bc_component[1], dm_bc_component[2]]
                )
                dm_vectors.append(final_dm_vec)

    # --- 4. Create the 3D Plot ---
    fig = plt.figure(figsize=(14, 12))
    ax = fig.add_subplot(111, projection="3d")

    # Plot atoms and their index labels
    ax.scatter(
        apos_uc[:, 0],
        apos_uc[:, 1],
        apos_uc[:, 2],
        s=150,
        c="skyblue",
        alpha=0.8,
        label="Cu Atoms",
    )
    for i in range(n_uc):
        ax.text(
            apos_uc[i, 0] + 0.3,
            apos_uc[i, 1] + 0.3,
            apos_uc[i, 2],
            f"{i}",
            color="black",
        )

    # Plot spin vectors using custom 3D arrows
    spin_length = 2.0
    for i in range(n_uc):
        start_pos = apos_uc[i]
        spin_vec = spin_vectors[i]
        # Normalize the spin vector to control arrow length uniformly
        norm_spin_vec = (
            spin_vec / np.linalg.norm(spin_vec)
            if np.linalg.norm(spin_vec) > 0
            else spin_vec
        )
        end_pos = start_pos + norm_spin_vec * spin_length

        color = "red" if spin_vec[0] > 0 else "blue"
        arrow = Arrow3D(
            [start_pos[0], end_pos[0]],
            [start_pos[1], end_pos[1]],
            [start_pos[2], end_pos[2]],
            mutation_scale=15,
            lw=1.5,
            arrowstyle="-|>",
            color=color,
        )
        ax.add_artist(arrow)
    # Dummy plots for legend
    ax.plot(
        [],
        [],
        [],
        color="red",
        marker=">",
        linestyle="None",
        markersize=10,
        label="Spin (+a)",
    )
    ax.plot(
        [],
        [],
        [],
        color="blue",
        marker="<",
        linestyle="None",
        markersize=10,
        label="Spin (-a)",
    )

    # Define reference vector for one of the chain directions, used for classification
    ref_vec_chain1 = np.array([0, uc_vectors[1, 1], uc_vectors[2, 2]])

    # Plot J1 bonds with direction arrows in the middle
    chain1_plotted = False
    chain2_plotted = False
    for i_bond, (i, j) in enumerate(j1_bonds):
        pos1 = apos_uc[i]
        pos2 = apos_ouc[j]
        bond_vec = pos2 - pos1
        midpoint = (pos1 + pos2) / 2

        # Determine chain type to assign color and legend label.
        # The logic mirrors the DM vector sign determination in spin_model_hc.py.
        bond_label = ""
        check_vec = pos1 - pos2  # Vector direction consistent with spin_model_hc.py
        if is_vector_parallel_to_another(check_vec, ref_vec_chain1):
            bond_color = "gray"
            if not chain1_plotted:
                bond_label = "J1 Bond (Chain 1: ~[0,1,1])"
                chain1_plotted = True
        else:  # The other chain direction is assumed to be ~[0,1,-1]
            bond_color = "olive"
            if not chain2_plotted:
                bond_label = "J1 Bond (Chain 2: ~[0,1,-1])"
                chain2_plotted = True

        # Plot the bond as a simple line
        ax.plot(
            [pos1[0], pos2[0]],
            [pos1[1], pos2[1]],
            [pos1[2], pos2[2]],
            color=bond_color,
            linestyle="-",
            label=bond_label,
        )

        # Add a 3D arrow at the midpoint to indicate direction
        arrow_length = 1.2
        norm_bond_vec = bond_vec / np.linalg.norm(bond_vec)
        start_arrow = midpoint - norm_bond_vec * arrow_length / 2
        end_arrow = midpoint + norm_bond_vec * arrow_length / 2

        arrow = Arrow3D(
            [start_arrow[0], end_arrow[0]],
            [start_arrow[1], end_arrow[1]],
            [start_arrow[2], end_arrow[2]],
            mutation_scale=12,
            lw=1,
            arrowstyle="-|>",
            color="black",
        )
        ax.add_artist(arrow)

    # Plot DM vectors at the midpoint of each bond
    dm_scale_factor = 0.5  # Adjust for better visualization
    for i_dm, (i, j) in enumerate(j1_bonds):
        pos1 = apos_uc[i]
        pos2 = apos_ouc[j]
        midpoint = (pos1 + pos2) / 2
        dm_vec = dm_vectors[i_dm]
        dm_length = np.linalg.norm(dm_vec) * dm_scale_factor
        if dm_length > 1e-6:  # Only plot if vector is not null
            norm_dm_vec = dm_vec / np.linalg.norm(dm_vec)
            start_arrow = midpoint
            end_arrow = start_arrow + norm_dm_vec * dm_length

            arrow = Arrow3D(
                [start_arrow[0], end_arrow[0]],
                [start_arrow[1], end_arrow[1]],
                [start_arrow[2], end_arrow[2]],
                mutation_scale=10,
                lw=1,
                arrowstyle="-|>",
                color="purple",
            )
            ax.add_artist(arrow)

    # Add a dummy plot for the DM vector legend
    ax.plot([], [], [], color="purple", lw=2, label="DM Vector")

    # --- 5. Finalize Plot ---
    ax.set_title("Magnetic Structure of α-Cu₂V₂O₇ Unit Cell", fontsize=16)
    ax.set_xlabel("a-axis (Å)", fontsize=12)
    ax.set_ylabel("b-axis (Å)", fontsize=12)
    ax.set_zlabel("c-axis (Å)", fontsize=12)

    # Set aspect ratio to be equal
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    z_lim = ax.get_zlim()
    x_range = abs(x_lim[1] - x_lim[0])
    y_range = abs(y_lim[1] - y_lim[0])
    z_range = abs(z_lim[1] - z_lim[0])
    max_range = np.array([x_range, y_range, z_range]).max() / 2.0
    mid_x = np.mean(x_lim)
    mid_y = np.mean(y_lim)
    mid_z = np.mean(z_lim)
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.legend()
    ax.view_init(elev=20, azim=-75)  # Adjust viewing angle
    plt.tight_layout()

    # Save the figure
    output_filename = os.path.join(SCRIPT_DIR, "../plots/aCVO_magnetic_structure.png")
    plt.savefig(output_filename, dpi=300)
    print(f"Plot saved to {output_filename}")

    plt.show()


if __name__ == "__main__":
    visualize_magnetic_structure()
