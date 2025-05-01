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
    r_pos_ouc = [apos[0], apos[1], apos[2]] + [
        apos[k] + i * uc[0] + j * uc[1]
        for i, j in product(range(-1, 2), repeat=2)
        if i != 0 or j != 0
        for k in range(apos_len)
    ]
    return np.array(r_pos_ouc)


def rot_mat(atom_list, p):
    """rotation matrix to transform spins to global coordinates
    Inputs:
        atom_list: list of angles with respect to the x-axis
        p: list of parameters [J1, J2, Dy, Dz, H]"""
    J1, J2, Dy, Dz, H = p

    Z = J2 - sp.sqrt(3) / 3 * Dz
    ca = np.abs(sp.asin(2 / (3 * J1) * (-sp.sqrt(3) * Dy) / (1 + Z / J1)) / 2)

    # rotation matrix for the canting
    mp_rot = sp.Matrix(
        [[sp.cos(ca), 0, -sp.sin(ca)], [0, 1, 0], [sp.sin(ca), 0, sp.cos(ca)]]
    )
    rot_m = [
        sp.Matrix(
            [[0, sp.sin(ag), sp.cos(ag)], [0, -sp.cos(ag), sp.sin(ag)], [1, 0, 0]]
        )
        * mp_rot
        for ag in atom_list
    ]
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
    N_atom_ouc = len(apos_ouc)  # Number of atoms including neighbors
    Jex = sp.zeros(N_atom, N_atom_ouc)

    # Define expected distances and tolerance
    J1_dist = 0.5
    J2_dist = np.sqrt(3.0) / 2.0
    dist_tol = 0.001  # Tolerance for distance matching

    for i in range(N_atom):
        for j in range(N_atom_ouc):
            d = la.norm(apos[i] - apos_ouc[j])  # Calculate distance
            if abs(d - J1_dist) < dist_tol:  # Check if distance matches J1 distance
                Jex[i, j] = J1
            elif abs(d - J2_dist) < dist_tol:  # Check if distance matches J2 distance
                Jex[i, j] = J2
            else:
                Jex[i, j] = 0

    # generate DM interactions !!! currently defined manually !!!
    DMvec1 = sp.MatrixSymbol("DMvec1", 1, 3)
    DMvec2 = sp.MatrixSymbol("DMvec2", 1, 3)
    DMvec3 = sp.MatrixSymbol("DMvec3", 1, 3)
    DMnull = sp.MatrixSymbol("DMnull", 1, 3)
    DMmat = sp.zeros(N_atom, N_atom_ouc)
    for i in range(N_atom):
        for j in range(N_atom_ouc):
            DMmat[i, j] = DMnull
    DMmat[0, 1] = -DMvec2
    DMmat[0, 2] = -DMvec1
    DMmat[0, 7] = -DMvec2
    DMmat[0, 14] = -DMvec1
    DMmat[1, 0] = DMvec2
    DMmat[1, 14] = DMvec3
    DMmat[1, 21] = DMvec2
    DMmat[1, 23] = DMvec3
    DMmat[2, 0] = DMvec1
    DMmat[2, 7] = -DMvec3
    DMmat[2, 15] = DMvec1
    DMmat[2, 16] = -DMvec3
    DM = DMmat.subs(
        {
            DMvec1: sp.Matrix([Dy, 0, -Dz]),
            DMvec2: sp.Matrix([Dy / 2, sp.sqrt(3) / 2 * Dy, Dz]),
            DMvec3: sp.Matrix([-Dy / 2, sp.sqrt(3) / 2 * Dy, -Dz]),
            DMnull: sp.Matrix([0, 0, 0]),
        }
    ).doit()
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
                HM = (
                    HM
                    + 1
                    / 2
                    * Jex[i, j]
                    * (
                        Sxyz[i][0] * Sxyz[j][0]
                        + Sxyz[i][1] * Sxyz[j][1]
                        + Sxyz[i][2] * Sxyz[j][2]
                    )
                    + 1
                    / 2
                    * (
                        DM[i, j][0]
                        * (Sxyz[i][1] * Sxyz[j][2] - Sxyz[i][2] * Sxyz[j][1])
                        + DM[i, j][1]
                        * (Sxyz[i][2] * Sxyz[j][0] - Sxyz[i][0] * Sxyz[j][2])
                        + DM[i, j][2]
                        * (Sxyz[i][0] * Sxyz[j][1] - Sxyz[i][1] * Sxyz[j][0])
                    )
                )
        HM = HM + gamma * mu * Sxyz[i][2] * H

    return HM
