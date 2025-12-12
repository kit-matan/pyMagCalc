#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018

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
    rot_m = []
    for ag in atom_list:
        omp = sp.Matrix([[np.cos(ag*np.pi/2), 0, np.sin(ag*np.pi/2)],
                      [0, 1, 0], [-np.sin(ag*np.pi/2), 0, np.cos(np.pi/2)]])
        rot_m.append(omp)
    return rot_m


def mpr(p):
    al = [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]  # angle with respect to the x-axis
    mp = rot_mat(al, p)
    return mp


def spin_interactions(p):
    # generate J exchange interactions
    J1 = p[0]
    J2 = p[1]
    J3 = p[2]
    G1 = p[3]
    G2 = p[4]
    G3 = p[5]
    Dx = p[6]
    Dy = p[7]
    Dz = p[8]
    Da = p[9]
    Db = p[10]
    D3x = p[11]
    D3y = p[12]
    D3z = p[13]
    H = p[14]
    apos = atom_pos()
    Nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    Nspin_ouc = len(apos_ouc)
    neighbor_dist_list = np.array([np.abs(la.norm(apos[0]-apos_ouc[9])),
                             np.abs(la.norm(apos[0]-apos_ouc[7])),
                             np.abs(la.norm(apos[0]-apos_ouc[5]))])
    Jex = sp.zeros(Nspin, Nspin_ouc)
    Gex = sp.zeros(Nspin, Nspin_ouc)
    for i in range(Nspin):
        for j in range(Nspin_ouc):
            if np.round(la.norm(apos[i]-apos_ouc[j]), 2) == np.round(neighbor_dist_list[0], 2):
                Jex[i, j] = J1
                Gex[i, j] = G1
            elif np.round(la.norm(apos[i]-apos_ouc[j]), 2) == np.round(neighbor_dist_list[1], 2):
                Jex[i, j] = J2
                Gex[i, j] = G2
            elif np.round(la.norm(apos[i]-apos_ouc[j]), 2) == np.round(neighbor_dist_list[2], 2):
                Jex[i, j] = J3
                Gex[i, j] = G3
            else:
                Jex[i, j] = 0.0
                Gex[i, j] = 0.0
                
    # generate DM interactions !!! currently defined manually !!!
    DMvec1 = sp.MatrixSymbol('DMvec1', 1, 3)
    DMvec2 = sp.MatrixSymbol('DMvec2', 1, 3)
    DMvec3 = sp.MatrixSymbol('DMvec3', 1, 3)
    DMvec4 = sp.MatrixSymbol('DMvec4', 1, 3)
    DMvec5 = sp.MatrixSymbol('DMvec5', 1, 3)
    DMvec6 = sp.MatrixSymbol('DMvec6', 1, 3)
    DMJ31 = sp.MatrixSymbol('DMJ31', 1, 3)
    DMJ32 = sp.MatrixSymbol('DMJ32', 1, 3)
    DMJ33 = sp.MatrixSymbol('DMJ33', 1, 3)
    DMJ34 = sp.MatrixSymbol('DMJ34', 1, 3)
    DMnull = sp.MatrixSymbol('DMnull', 1, 3)
 
    DMmat = sp.zeros(Nspin, Nspin_ouc)
    for i in range(Nspin):
        for j in range(Nspin_ouc):
            DMmat[i, j] = DMnull

    DMmat[0, 5] = -DMJ31
    DMmat[0, 7] = DMvec5
    DMmat[0, 9] = -DMvec4
    DMmat[0, 187] = DMvec3
    DMmat[0, 229] = DMJ32
    DMmat[1, 4] = -DMJ32
    DMmat[1, 6] = DMvec5
    DMmat[1, 12] = -DMvec2
    DMmat[1, 14] = DMvec1
    DMmat[1, 228] = DMJ31
    DMmat[2, 5] = DMvec5
    DMmat[2, 11] = -DMvec4
    DMmat[2, 217] = DMvec3
    DMmat[2, 247] = -DMJ31
    DMmat[2, 263] = DMJ32
    DMmat[3, 4] = DMvec5
    DMmat[3, 166] = -DMJ32
    DMmat[3, 182] = DMJ31
    DMmat[3, 188] = DMvec1
    DMmat[3, 222] = -DMvec2
    DMmat[4, 1] = DMJ32
    DMmat[4, 3] = -DMvec5
    DMmat[4, 13] = -DMvec4
    DMmat[4, 209] = -DMJ31
    DMmat[4, 223] = DMvec3
    DMmat[5, 0] = DMJ31
    DMmat[5, 2] = -DMvec5
    DMmat[5, 10] = DMvec1
    DMmat[5, 208] = -DMJ32
    DMmat[5, 216] = -DMvec2
    DMmat[6, 1] = -DMvec5
    DMmat[6, 13] = DMvec3
    DMmat[6, 259] = -DMJ31
    DMmat[6, 271] = -DMvec4
    DMmat[6, 275] = DMJ32
    DMmat[7, 0] = -DMvec5
    DMmat[7, 10] = -DMvec2
    DMmat[7, 178] = -DMJ32
    DMmat[7, 184] = DMvec1
    DMmat[7, 194] = DMJ31
    DMmat[8, 13] = -DMJ33
    DMmat[8, 229] = DMvec2
    DMmat[8, 237] = DMJ34
    DMmat[8, 263] = -DMvec1
    DMmat[8, 271] = DMvec6
    DMmat[9, 0] = DMvec4
    DMmat[9, 92] = -DMJ34
    DMmat[9, 94] = DMvec6
    DMmat[9, 108] = DMJ33
    DMmat[9, 226] = -DMvec3
    DMmat[10, 5] = -DMvec1
    DMmat[10, 7] = DMvec2
    DMmat[10, 13] = DMvec6
    DMmat[10, 15] = DMJ34
    DMmat[10, 223] = -DMJ33 
    DMmat[11, 2] = DMvec4
    DMmat[11, 92] = DMvec6
    DMmat[11, 126] = -DMJ34 
    DMmat[11, 142] = DMJ33
    DMmat[11, 256] = -DMvec3 
    DMmat[12, 1] = DMvec2
    DMmat[12, 259] = -DMvec1 
    DMmat[12, 345] = -DMJ33 
    DMmat[12, 361] = DMJ34
    DMmat[12, 363] = -DMvec6 
    DMmat[13, 4] = DMvec4
    DMmat[13, 6] = -DMvec3
    DMmat[13, 8] = DMJ33
    DMmat[13, 10] = -DMvec6
    DMmat[13, 216] = -DMJ34 
    DMmat[14, 1] = -DMvec1
    DMmat[14, 227] = DMvec3
    DMmat[14, 315] = -DMJ33 
    DMmat[14, 331] = DMJ34
    DMmat[14, 361] = -DMvec6 
    DMmat[15, 10] = -DMJ34
    DMmat[15, 182] = DMvec4
    DMmat[15, 184] = -DMvec6 
    DMmat[15, 228] = -DMvec3 
    DMmat[15, 234] = DMJ33

    DM = DMmat.subs({DMvec1: sp.Matrix([Dx, Dy,  Dz]),
                     DMvec2: sp.Matrix([Dx, -Dy, -Dz]),
                     DMvec3: sp.Matrix([Dx, -Dy,  Dz]),
                     DMvec4: sp.Matrix([Dx, Dy, -Dz]),
                     DMvec5: sp.Matrix([Da, Db,  0]),
                     DMvec6: sp.Matrix([-Da, Db,  0]),
                     DMJ31 : sp.Matrix([D3x, D3y, D3z]),
                     DMJ32 : sp.Matrix([D3x, D3y, -D3z]),
                     DMJ33 : sp.Matrix([-D3x, D3y, -D3z]),
                     DMJ34 : sp.Matrix([-D3x, D3y, D3z]),
                     DMnull: sp.Matrix([0, 0, 0])}).doit()
    
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
