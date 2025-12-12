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
import read_cif as rc

# Define atoms globally
atoms = rc.read_cif_file('KFe3J.cif')

def atom_pos():
    """atomic positions for Fe"""
    atom_positions = rc.get_atom_positions_in_unit_cell(atoms, 'Fe')
    atom_positions.sort(key=lambda pos: pos[2])  # Sort the positions based on the z-axis value
    min_z = atom_positions[0][2]  # Get the smallest z-axis value

    # Filter the Fe positions to only include those with the smallest z-axis value
    atom_positions_planar = np.array([pos for pos in atom_positions if pos[2] == min_z])
    atom_positions_planar[:, 2] = 0  # Set the z-axis of atom_positions_planar to zero
    
    return atom_positions_planar / atoms.cell.cellpar()[0]


def atom_pos_ouc():
    """atomic positions outside the unit cell"""
    atom_positions = rc.get_atom_positions_in_unit_cell(atoms, 'Fe')
    outside_unit_cell_atom_positions = rc.get_atom_positions_outside_unit_cell(atoms, atom_positions)
    atom_positions.sort(key=lambda pos: pos[2])  # Sort the positions based on the z-axis value
    min_z = atom_positions[0][2]  # Get the smallest z-axis 

    # Filter the Fe positions to only include those with the smallest z-axis value
    outside_unit_cell_atom_positions_planar = np.array([pos for pos in outside_unit_cell_atom_positions if pos[2] == min_z])
    outside_unit_cell_atom_positions_planar[:, 2] = 0  # Set the z-axis of atom_positions_planar to zero
    
    return outside_unit_cell_atom_positions_planar / atoms.cell.cellpar()[0]


def rot_mat(atom_list, p):
    """rotation matrix to transform spins to global coordinates
       Inputs:
           atom_list: list of angles with respect to the x-axis
           p: list of parameters [J1, J2, Dy, Dz, H]"""
    
    J1, J2, Dy, Dz, H = p

    Z = J2 - sp.sqrt(3)/3 * Dz
    ca = np.abs(sp.asin(2/(3*J1) * (-sp.sqrt(3)*Dy)/(1+Z/J1))/2)

    # rotation matrix for the canting
    mp_rot = sp.Matrix([[sp.cos(ca), 0, -sp.sin(ca)], [0, 1, 0], [sp.sin(ca), 0, sp.cos(ca)]])
    rot_m = [sp.Matrix([[0, sp.sin(ag), sp.cos(ag)],
                        [0, -sp.cos(ag), sp.sin(ag)],
                        [1, 0, 0]]) * mp_rot for ag in atom_list]
    return rot_m


def mpr(p):
    """rotation matrix for the positive chirality
       Input:
           p: list of parameters [J1, J2, Dy, Dz, H]"""
    # positive chirality
    al = [-1 / 3 * np.pi, np.pi, 1 / 3 * np.pi]  # angle with respect to the x-axis
    # negative chirality
    # al = [1 / 3 * np.pi, np.pi, -1 / 3 * np.pi]  # angle with respect to the x-axis
    mp = rot_mat(al, p)
    return mp


def spin_interactions(p):
    """Generate spin interactions
       Input:
           p: list of parameters [J1, J2, Dy, Dz, H]""" 
    # Exchange interactions J's
    J1, J2, Dy, Dz, H = p
    apos = atom_pos()
    N_atom = len(apos)
    apos_ouc = atom_pos_ouc()
    N_atom_ouc = len(apos_ouc)
    Jex = sp.zeros(N_atom, N_atom_ouc)
    for i in range(N_atom):
        for j in range(N_atom_ouc):
            if np.abs(la.norm(apos[i] - apos_ouc[j]) - 0.5) < 0.001:
                Jex[i, j] = J1
            elif np.abs(la.norm(apos[i] - apos_ouc[j]) - np.sqrt(3.0) / 2.0) < 0.001:
                Jex[i, j] = J2
            else:
                Jex[i, j] = 0

    # generate DM interactions !!! currently defined manually !!!
    DMvec1 = sp.MatrixSymbol('DMvec1', 1, 3)
    DMvec2 = sp.MatrixSymbol('DMvec2', 1, 3)
    DMvec3 = sp.MatrixSymbol('DMvec3', 1, 3)
    DMnull = sp.MatrixSymbol('DMnull', 1, 3)
    DMmat = sp.zeros(N_atom, N_atom_ouc)
    for i in range(N_atom):
        for j in range(N_atom_ouc):
            DMmat[i, j] = DMnull
    
    DMmat[0, 1] = -DMvec2
    DMmat[0, 2] = -DMvec1
    DMmat[0, 13] = -DMvec2
    DMmat[0, 23] = -DMvec1
    DMmat[1, 0] = DMvec2
    DMmat[1, 2] = DMvec3
    DMmat[1, 15] = DMvec2
    DMmat[1, 26] = DMvec3
    DMmat[2, 0] = DMvec1
    DMmat[2, 1] = -DMvec3
    DMmat[2, 6] = DMvec1
    DMmat[2, 4] = -DMvec3

    DM = DMmat.subs({DMvec1: sp.Matrix([Dy, 0, -Dz]),
                     DMvec2: sp.Matrix([Dy / 2, sp.sqrt(3) / 2 * Dy, Dz]),
                     DMvec3: sp.Matrix([-Dy / 2, sp.sqrt(3) / 2 * Dy, -Dz]),
                     DMnull: sp.Matrix([0, 0, 0])}).doit()
    return Jex, DM


def Hamiltonian(Sxyz, pr):
    """Define the spin Hamiltonian for your system
       Inputs:
           Sxyz: list of spin operators
           pr: list of parameters [J1, J2, Dy, Dz, H]"""
    Jex, DM = spin_interactions(pr)
    HM = 0
    gamma = 2.0
    mu = 5.7883818066e-2
    H = pr[-1]
    apos = atom_pos()
    nspins = len(apos)
    apos_ouc = atom_pos_ouc()
    nspins_ouc = len(apos_ouc)
    for i in range(nspins):
        for j in range(nspins_ouc):
            if Jex[i, j] != 0:
                HM = HM + 1 / 2 * Jex[i, j] * (Sxyz[i][0] * Sxyz[j][0] +
                                               Sxyz[i][1] * Sxyz[j][1] + Sxyz[i][2] * Sxyz[j][2]) + \
                     1 / 2 * (DM[i, j][0] * (Sxyz[i][1] * Sxyz[j][2] - Sxyz[i][2] * Sxyz[j][1]) +
                              DM[i, j][1] * (Sxyz[i][2] * Sxyz[j][0] - Sxyz[i][0] * Sxyz[j][2]) +
                              DM[i, j][2] * (Sxyz[i][0] * Sxyz[j][1] - Sxyz[i][1] * Sxyz[j][0]))
        HM = HM + gamma * mu * Sxyz[i][2] * H

    return HM
