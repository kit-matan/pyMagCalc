#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018

@author: Kit Matan

this file contains the information about the spin model that will be used
to calculate spin-waves by magcalc.py

KFe3(OH)6(SO4)2

Edit this file for your system
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
from itertools import product
from scipy.optimize import differential_evolution, dual_annealing
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting if needed

def unit_cell():
    """a unit cell for the kagome lattice"""
    va = np.array([np.sqrt(3) / 2, -1 / 2, 0])
    vb = np.array([0, 1, 0])
    vc = np.array([0, 0, 1])
    uc = [va, vb, vc]
    return np.array(uc)

def atom_pos():
    """atomic positions for the kagome lattice"""
    atom1 = np.array([0, 0, 0])
    atom2 = np.array([np.sqrt(3) / 4, -1 / 4, 0])
    atom3 = np.array([0, 1 / 2, 0])
    r_pos = [atom1, atom2, atom3]
    return np.array(r_pos)

def atom_pos_ouc():
    """atomic positions outside the unit cell"""
    uc = unit_cell()
    apos = atom_pos()
    apos_len = len(apos)
    r_pos_ouc = [apos[0], apos[1], apos[2]] + [apos[k] + i * uc[0] + j * uc[1] for i, j in product(range(-1, 2), repeat=2) if i != 0 or j != 0 for k in range(apos_len)]
    return np.array(r_pos_ouc)

def spin_interactions_numerical(p):
    """Generate spin interactions (numerical version)"""
    J1, J2, Dy, Dz, H = p
    apos = atom_pos()
    N_atom = len(apos)
    apos_ouc = atom_pos_ouc()
    N_atom_ouc = len(apos_ouc)
    Jex = np.zeros((N_atom, N_atom_ouc))
    for i in range(N_atom):
        for j in range(N_atom_ouc):
            if np.abs(la.norm(apos[i] - apos_ouc[j]) - 0.5) < 0.001:
                Jex[i, j] = J1
            elif np.abs(la.norm(apos[i] - apos_ouc[j]) - np.sqrt(3.0) / 2.0) < 0.001:
                Jex[i, j] = J2
            else:
                Jex[i, j] = 0

    DM = np.zeros((N_atom, N_atom_ouc, 3))
    DMvec1 = np.array([Dy, 0, -Dz])
    DMvec2 = np.array([Dy / 2, np.sqrt(3) / 2 * Dy, Dz])
    DMvec3 = np.array([-Dy / 2, np.sqrt(3) / 2 * Dy, -Dz])
    DMnull = np.array([0, 0, 0])

    DM[0, 1] = -DMvec2
    DM[0, 2] = -DMvec1
    DM[0, 7] = -DMvec2
    DM[0, 14] = -DMvec1
    DM[1, 0] = DMvec2
    DM[1, 14] = DMvec3
    DM[1, 21] = DMvec2
    DM[1, 23] = DMvec3
    DM[2, 0] = DMvec1
    DM[2, 7] = -DMvec3
    DM[2, 15] = DMvec1
    DM[2, 16] = -DMvec3

    return Jex, DM

def energy_in_xy_plane_fixed_length(phi_angles, pr):
    """Energy function with spin vectors constrained to the xy-plane and fixed length."""
    Jex, DM = spin_interactions_numerical(pr)
    HM = 0
    gamma = 2.0
    mu = 5.7883818066e-2
    H = pr[-1]
    apos = atom_pos()
    nspins = len(apos)
    apos_ouc = atom_pos_ouc()
    nspins_ouc = len(apos_ouc)
    spin_length = 0.5

    # Spin vectors in the xy-plane with fixed length: [0.5*cos(phi), 0.5*sin(phi), 0]
    spin_vectors_unit_cell = np.zeros((nspins, 3))
    for i in range(nspins):
        phi = phi_angles[i]
        spin_vectors_unit_cell[i] = [spin_length * np.cos(phi), spin_length * np.sin(phi), 0]

    for i in range(nspins):
        spin_i = spin_vectors_unit_cell[i]
        for j_ouc in range(nspins_ouc):
            if Jex[i, j_ouc] != 0 or np.any(DM[i, j_ouc] != 0):
                # Find the vector connecting atom i in the central unit cell to atom j_ouc
                diff_vec = apos_ouc[j_ouc] - apos[i]

                # Find the corresponding atom index within the original unit cell
                atom_index_j = -1
                translation = np.round(np.linalg.solve(unit_cell().T, diff_vec)).astype(int)
                for k in range(nspins):
                    if np.allclose(apos[k], apos_ouc[j_ouc] - translation @ unit_cell()):
                        atom_index_j = k
                        break

                if atom_index_j != -1:
                    spin_j = spin_vectors_unit_cell[atom_index_j]
                    if Jex[i, j_ouc] != 0:
                        HM += 0.5 * Jex[i, j_ouc] * np.dot(spin_i, spin_j)
                    HM += 0.5 * np.dot(DM[i, j_ouc], np.cross(spin_i, spin_j))

        HM += gamma * mu * spin_i[2] * H

    return HM

def objective_in_xy_plane_fixed_length(phi_angles, pr):
    """Objective function for minimization with spins in the xy-plane and fixed length."""
    return energy_in_xy_plane_fixed_length(phi_angles, pr)

def plot_magnetic_structure_xy(atom_positions, spin_configuration):
    """Plots the magnetic structure in the xy-plane."""
    fig, ax = plt.subplots()
    ax.scatter(atom_positions[:, 0], atom_positions[:, 1], color='blue', label='Atoms')

    for i, pos in enumerate(atom_positions):
        spin = spin_configuration[i]
        ax.arrow(pos[0], pos[1], spin[0] * 0.4, spin[1] * 0.4, head_width=0.05, head_length=0.1, fc='red', ec='red', label='Spins' if i == 0 else "")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Magnetic Structure in the xy-plane (Spin Length = 0.5)')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

def plot_magnetic_structure_3d_xy(atom_positions, spin_configuration):
    """Plots the magnetic structure in 3D (but spins are in xy-plane)."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(atom_positions[:, 0], atom_positions[:, 1], atom_positions[:, 2], color='blue', label='Atoms')

    for i, pos in enumerate(atom_positions):
        spin = spin_configuration[i]
        ax.quiver(pos[0], pos[1], pos[2], spin[0] * 0.4, spin[1] * 0.4, spin[2] * 0.4, length=0.5, color='red', label='Spins' if i == 0 else "")

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Magnetic Structure in 3D (Spins in xy-plane, Length = 0.5)')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Set parameters for the Hamiltonian
    params = [3.23, 0.11, 0.218, -0.195, 0]
    atom_positions = atom_pos()
    num_atoms = len(atom_positions)
    spin_length = 0.5
    bounds = [(0, 2 * np.pi)] * num_atoms

    # Try differential evolution for global optimization (often a good balance of speed and effectiveness)
    print("\nTrying global optimization method: differential_evolution")
    result_de = differential_evolution(objective_in_xy_plane_fixed_length, bounds, args=(params,), maxiter=200, popsize=20, tol=0.01)

    if result_de.success:
        min_energy_de = result_de.fun
        optimal_phi_angles_de = result_de.x
        optimal_spins_de = np.zeros((num_atoms, 3))
        for i in range(num_atoms):
            phi = optimal_phi_angles_de[i]
            optimal_spins_de[i] = [spin_length * np.cos(phi), spin_length * np.sin(phi), 0]
        print(f"  Minimum energy (differential_evolution):", min_energy_de)
        print(f"  Optimal spin configuration (differential_evolution):\n", optimal_spins_de)
        plot_magnetic_structure_xy(atom_positions, optimal_spins_de)
        plot_magnetic_structure_3d_xy(atom_positions, optimal_spins_de)
    else:
        print(f"  Optimization failed (differential_evolution):", result_de.message)

    # You can also try dual annealing, which can be effective but might be slower
    print("\nTrying global optimization method: dual_annealing")
    result_da = dual_annealing(objective_in_xy_plane_fixed_length, bounds, args=(params,), maxiter=200, seed=1234)

    if result_da.success:
        min_energy_da = result_da.fun
        optimal_phi_angles_da = result_da.x
        optimal_spins_da = np.zeros((num_atoms, 3))
        for i in range(num_atoms):
            phi = optimal_phi_angles_da[i]
            optimal_spins_da[i] = [spin_length * np.cos(phi), spin_length * np.sin(phi), 0]
        print(f"  Minimum energy (dual_annealing):", min_energy_da)
        print(f"  Optimal spin configuration (dual_annealing):\n", optimal_spins_da)
        plot_magnetic_structure_xy(atom_positions, optimal_spins_da)
        plot_magnetic_structure_3d_xy(atom_positions, optimal_spins_da)
    else:
        print(f"  Optimization failed (dual_annealing):", result_da.message)