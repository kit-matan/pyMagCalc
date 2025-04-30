#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018
Modified on Mon 19 Feb 2024

@author: Kit Matan

this file contains the information about the spin structure that will be used
to calculate spin-waves by MagCal.py

alpha-Cu2V2O7
"""
import sympy as sp
import numpy as np
from numpy import linalg as la


def unit_cell():
    """a unit cell for the kagome lattice"""
    va = np.array([20.645, 0, 0])
    vb = np.array([0, 8.383, 0])
    vc = np.array([0, 0, 6.442])
    uc = [va, vb, vc]
    return np.array(uc)


def atom_pos():
    """atomic positions for the kagome lattice"""
    la = 20.645
    lb = 8.383
    lc = 6.442
    x = 0.16572
    y = 0.3646
    z = 0.7545

    positions = [
        [x, y, z],
        [1 - x, 1 - y, z],
        [x, y + 1/2, z - 1/2],
        [1 - x, -y + 1/2, z - 1/2],
        [x + 1/2, y, z - 1/2],
        [-x + 1/2, 1 - y, z - 1/2],
        [x + 1/2, y + 1/2, z],
        [-x + 1/2, -y + 1/2, z],
        [x + 1/4, -y + 1/4 + 1, z + 1/4],
        [-x + 1/4, y + 1/4, z + 1/4],
        [x + 1/4, -y + 3/4, z + 3/4 - 1],
        [-x + 1/4, y + 3/4, z + 3/4 - 1],
        [x + 3/4, -y + 1/4 + 1, z + 3/4 - 1],
        [-x + 3/4, y + 1/4, z + 3/4 - 1],
        [x + 3/4, -y + 3/4, z + 1/4],
        [-x + 3/4, y + 3/4 - 1, z + 1/4]
    ]

    r_pos = [np.array([pos[0]*la, pos[1]*lb, pos[2]*lc]) for pos in positions]
    return np.array(r_pos)


def atom_pos_ouc():
    """atomic positions outside the unit cell"""
    r_pos_ouc = []
    uc = unit_cell()
    apos = atom_pos()
    r_pos_ouc.extend(apos)
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k == 0:  # first set of spins are in the unit cell
                    pass
                else:
                    r_pos_ij = apos + i * uc[0] + j * uc[1] + k * uc[2]
                    r_pos_ouc.extend(r_pos_ij)
    return np.array(r_pos_ouc)


def rot_mat(atom_list, p):
    """"rotation matrix to transform spins to global coordinates"""
    # p is not used in this case since we assume that the spin structure of alpha-Cu2V2O6
    # does not depend on Hamiltonian parameters
    # rotation matrix for the spins about the x-axis
    rot_m = []
    for ag in atom_list:
        omp = sp.Matrix([[1, 0, 0],
                 [0, np.cos(ag*np.pi), -np.sin(ag*np.pi)],
                 [0, np.sin(ag*np.pi), np.cos(ag*np.pi)]])
        rot_m.append(omp)
    return rot_m


def mpr(p):
    al = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # angle in a unit of pi, 0 along +c, 1 along -c
    mp = rot_mat(al, p)
    return mp


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
      raise ValueError("Atoms cannot have the same x-coordinate and different other coordinates.")
    else:
      return False  # Both atoms are at the same position


def get_nearest_neighbor_distances(atom_positions, outside_unit_cell_atom_positions, num_neighbors=3):
    """Return the distances to the nearest Fe neighbors."""
    # List the distance between the first three nearest neighnor Fe atoms and do not list the same distance twice
    distances = []
    for i in range(len(atom_positions)):
        for j in range(len(outside_unit_cell_atom_positions)):
            distance = round(np.linalg.norm(atom_positions[i] - outside_unit_cell_atom_positions[j]), 4)
            if distance not in distances:
                distances.append(distance)

    distances = sorted(distances)  # Sort the distances from low to high
    return distances[:num_neighbors]


def spin_interactions(p):
    # generate J exchange interactions
    J1, J2, J3, G1, Dx, H = p
    apos = atom_pos()
    nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    nspin_ouc = len(apos_ouc)
    lattice_constants = unit_cell()

    neighbor_dist_list = get_nearest_neighbor_distances(apos, apos_ouc, 10)[1:4] 
    Jex = sp.zeros(nspin, nspin_ouc)
    Gex = sp.zeros(nspin, nspin_ouc)
    # create zeros matrix to fill in the DM vectors for each pair of spins in the unit cell and outside the unit cell
    DMmat = sp.zeros(nspin, nspin_ouc)
    DM1 = sp.MatrixSymbol('DM1', 1, 3)
    DMnull = sp.MatrixSymbol('DMnull', 1, 3)
    
    DMmat = sp.zeros(nspin, nspin_ouc)
    for i in range(nspin):
        for j in range(nspin_ouc):
            DMmat[i, j] = DMnull
    
    for i in range(nspin):
        for j in range(nspin_ouc):
            if np.round(la.norm(apos[i]-apos_ouc[j]), 2) == np.round(neighbor_dist_list[0], 2):
                Jex[i, j] = J1
                Gex[i, j] = G1
                # check if spins_ouc is along the positive x-axis
                if is_vector_parallel_to_another(apos[i]-apos_ouc[j], [0, lattice_constants[1,1], lattice_constants[2,2]]):
                    DMmat[i,j] = -DM1
                    # check if spins_ouc is to the right of spins
                    if is_right_neighbor(apos[i], apos_ouc[j]):
                        DMmat[i,j] = -DMmat[i,j]
                    else:
                        DMmat[i,j] = DMmat[i,j]
                # check if spins_ouc is along the negative x-axis
                else:
                    DMmat[i,j] = DM1
                    # check if spins_ouc is to the right of spins
                    if is_right_neighbor(apos[i], apos_ouc[j]):
                        DMmat[i,j] = -DMmat[i,j]
                    else:
                        DMmat[i,j] = DMmat[i,j]
            elif np.round(la.norm(apos[i]-apos_ouc[j]), 2) == np.round(neighbor_dist_list[1], 2):
                Jex[i, j] = J2
                Gex[i, j] = 0.0
                DMmat[i,j] = DMnull
            elif np.round(la.norm(apos[i]-apos_ouc[j]), 2) == np.round(neighbor_dist_list[2], 2):
                Jex[i, j] = J3
                Gex[i, j] = 0.0
                DMmat[i,j] = DMnull
            else:
                Jex[i, j] = 0.0
                Gex[i, j] = 0.0
                DMmat[i,j] = DMnull
    
    DM = DMmat.subs({DM1: sp.Matrix([Dx, 0,  0]), DMnull: sp.Matrix([0, 0, 0])}).doit()
                
    return Jex, Gex, DM, H       

def Hamiltonian(Sxyz, p):
    "Define the spin Hamiltonain for your system" 
    gamma = 2.0
    mu = 5.7883818066e-2
    Jex, Gex, DM, H = spin_interactions(p)
    HM = 0
    apos = atom_pos()
    Nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    Nspin_ouc = len(apos_ouc)
    for i in range(Nspin):
        for j in range(Nspin_ouc):
            if Jex[i, j] != 0:
                HM = HM + 1/2 * Jex[i, j] * (Sxyz[i][0] * Sxyz[j][0] +
                                             Sxyz[i][1] * Sxyz[j][1] + Sxyz[i][2] * Sxyz[j][2]) + \
                          1/2 * (DM[i, j][0] * (Sxyz[i][1] * Sxyz[j][2] - Sxyz[i][2] * Sxyz[j][1]) +
                                 DM[i, j][1] * (Sxyz[i][2] * Sxyz[j][0] - Sxyz[i][0] * Sxyz[j][2]) +
                                 DM[i, j][2] * (Sxyz[i][0] * Sxyz[j][1] - Sxyz[i][1] * Sxyz[j][0])) + \
                          1/2 * Gex[i, j] * (Sxyz[i][0] * Sxyz[j][0] -
                                             Sxyz[i][1] * Sxyz[j][1] - Sxyz[i][2] * Sxyz[j][2])
        HM = HM + gamma * mu * Sxyz[i][0] * H
    HM = HM.expand()
    return HM
