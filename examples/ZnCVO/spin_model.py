#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018

@author: Ganatee Gitgeatpong and Kit Matan

this file contains the information about the spin structure that will be used
to calculate spin-waves by MagCal.py

Sep 2021: PURE: Code modified for ZnCVO-bCVO system
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
import math

def unit_cell():
    """a unit cell of bCVO"""
    beta = 110.251999
    #betarad = beta*np.pi/180
    aaxis = 7.689
    baxis = 8.0289
    caxis = 10.1065
    cx = caxis*np.cos(math.radians(beta))
    cz = caxis*np.sin(math.radians(beta))
    uc = []
    va = np.array([aaxis, 0, 0])
    vb = np.array([0, baxis, 0])
    vc = np.array([cx, 0, cz])
    uc.append(va)
    uc.append(vb)
    uc.append(vc)
    return np.array(uc)


def atom_pos():
    """atomic positions"""
    r_pos = []
    u = 0.30976
    v = 0.07364
    w = 0.51407
    a = 7.689
    b = 8.0289
    c = 10.1065
    beta = 110.251999
    # x = a*u + c*w*np.cos(math.radians(beta))
    # y = b*v
    # z = c*w*np.sin(math.radians(beta))
    i = 0
    j = 0
    k = 0
    atom1 = np.array([a*(i+u) + c*(k+w)*np.cos(math.radians(beta)), b*(j+v), c*(k+w)*np.sin(math.radians(beta))]) #1 down
    atom2 = np.array([a*(i+1-u) + c*(k+1-w)*np.cos(math.radians(beta)), b*(j+1-v), c*(k+1-w)*np.sin(math.radians(beta))]) #5 up
    atom3 = np.array([a*(i+1-u) + c*(k+1-w+1/2)*np.cos(math.radians(beta)), b*(j+v), c*(k+1-w+1/2)*np.sin(math.radians(beta))]) #3 down
    atom4 = np.array([a*(i+u) + c*(k-1+w+1/2)*np.cos(math.radians(beta)), b*(j+1-v), c*(k-1+w+1/2)*np.sin(math.radians(beta))]) #6 up --#1 ***J5
    atom5 = np.array([a*(i+u+1/2) + c*(k+w)*np.cos(math.radians(beta)), b*(j+v+1/2), c*(k+w)*np.sin(math.radians(beta))]) #2 down
    atom6 = np.array([a*(i-u+1/2) + c*(k+1-w)*np.cos(math.radians(beta)), b*(j-v+1/2), c*(k+1-w)*np.sin(math.radians(beta))]) #7 up --#1 ***J6
    atom7 = np.array([a*(i-u+1/2) + c*(k+1-w+1/2)*np.cos(math.radians(beta)), b*(j+v+1/2), c*(k+1-w+1/2)*np.sin(math.radians(beta))]) #4 down --#7 ***J5
    atom8 = np.array([a*(i+u+1/2) + c*(k-1+w+1/2)*np.cos(math.radians(beta)), b*(j-v+1/2), c*(k-1+w+1/2)*np.sin(math.radians(beta))]) #8 up

    # atom1 = np.array([-1.98639, 4.60570, 9.34830])      d
    # atom2 = np.array([1.85811,   0.59125,   9.34830])   d
    # atom3 = np.array([-0.23720,   3.42320,   4.60745])  u
    # atom4 = np.array([0.58334,   0.59125,   4.87426])   d
    # atom5 = np.array([3.60730,   7.43765,   4.60745])   u
    # atom6 = np.array([4.42784,   4.60570,   4.87426])   d
    # atom7 = np.array([2.33252,   7.43765,   0.13341])   u
    # atom8 = np.array([6.17702,   3.42320,   0.13341])   u
    # _space_group_symop_operation_xyz
    # 'x, y, z'                  atom5   d plane A
    # '-x, -y, -z'               atom4   u
    # '-x, y, -z+1/2'            atom2   d
    # 'x, -y, z+1/2'             atom7   u plane A
    # 'x+1/2, y+1/2, z'          atom6   d
    # '-x+1/2, -y+1/2, -z'       atom3   u plane A
    # '-x+1/2, y+1/2, -z+1/2'    atom1   d plane A
    # 'x+1/2, -y+1/2, z+1/2'     atom8   u
   
    r_pos.append(atom1)
    r_pos.append(atom2)
    r_pos.append(atom3)
    r_pos.append(atom4)
    r_pos.append(atom5)
    r_pos.append(atom6)
    r_pos.append(atom7)
    r_pos.append(atom8)
    return np.array(r_pos)

def atom_pos_ouc():
    """atomic positions outside the unit cell"""
    r_pos_ouc = []
    uc = unit_cell()
    apos = atom_pos()
    u = 0.30976
    v = 0.07364
    w = 0.51407
    a = 7.689
    b = 8.0289
    c = 10.1065
    beta = 110.251999
    r_pos_ouc.append(apos[0])
    r_pos_ouc.append(apos[1])
    r_pos_ouc.append(apos[2])
    r_pos_ouc.append(apos[3])
    r_pos_ouc.append(apos[4])
    r_pos_ouc.append(apos[5])
    r_pos_ouc.append(apos[6])
    r_pos_ouc.append(apos[7])
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i == 0 and j == 0 and k ==0:  # first set of spins are in the unit cell
                    pass
                else:
                    r_pos1_ij = np.array([a*(i+u) + c*(k+w)*np.cos(math.radians(beta)), b*(j+v), c*(k+w)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos1_ij)                       
                    r_pos2_ij = np.array([a*(i+1-u) + c*(k+1-w)*np.cos(math.radians(beta)), b*(j+1-v), c*(k+1-w)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos2_ij)
                    r_pos3_ij = np.array([a*(i+1-u) + c*(k+1-w+1/2)*np.cos(math.radians(beta)), b*(j+v), c*(k+1-w+1/2)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos3_ij)
                    r_pos4_ij = np.array([a*(i+u) + c*(k-1+w+1/2)*np.cos(math.radians(beta)), b*(j+1-v), c*(k-1+w+1/2)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos4_ij)
                    r_pos5_ij = np.array([a*(i+u+1/2) + c*(k+w)*np.cos(math.radians(beta)), b*(j+v+1/2), c*(k+w)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos5_ij)
                    r_pos6_ij = np.array([a*(i-u+1/2) + c*(k+1-w)*np.cos(math.radians(beta)), b*(j-v+1/2), c*(k+1-w)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos6_ij)
                    r_pos7_ij = np.array([a*(i-u+1/2) + c*(k+1-w+1/2)*np.cos(math.radians(beta)), b*(j+v+1/2), c*(k+1-w+1/2)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos7_ij)
                    r_pos8_ij = np.array([a*(i+u+1/2) + c*(k-1+w+1/2)*np.cos(math.radians(beta)), b*(j-v+1/2), c*(k-1+w+1/2)*np.sin(math.radians(beta))])
                    r_pos_ouc.append(r_pos8_ij)
    return np.array(r_pos_ouc)


def rot_mat(atom_list, p):
    """"rotation matrix to transform spins to global coordinates"""
    # p is not used in this case since we assume that the spin structure of alpha-Cu2V2O6
    # does not depend on Hamiltonian parameters
    # PURE: rotate spin in c-direction around the y-axis by +/-(beta-90) degree
    rot_m = []
    for ag in atom_list:
        omp = sp.Matrix([[np.cos(math.radians(ag)), 0, -np.sin(math.radians(ag))], [0, 1, 0], 
                [np.sin(math.radians(ag)), 0, np.cos(math.radians(ag))]])
        rot_m.append(omp)
    return rot_m


def mpr(p):
    # PURE: spins are along c-axis making angle beta-90 deg with respect to z-axis
    beta = 110.251999
    up = beta - 90 # angle from +z to +c
    down = beta + 90 # angle from +z to -c
    # al = [-90, 90, -90, 90]
    #al = [0, 0, 0, 0, 0, 0, 0, 0]
    al = [up, down, up, down, up, down, up, down]
    mp = rot_mat(al, p)
    return mp


def spin_interactions(p):
    # generate J exchange interactions
    J1 = p[0]
    J2 = p[1]
    J3 = p[2]
    J4 = p[3]
    J5 = p[4]
    J6 = p[5]
    J7 = p[6]
    G = p[7]
    H = p[8]
    apos = atom_pos()
    Nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    Nspin_ouc = len(apos_ouc)
    # neighbor_dist_list = np.array([np.abs(la.norm(apos[0]-apos_ouc[5])),
                         # np.abs(la.norm(apos[5]-apos_ouc[6])),
                         # np.abs(la.norm(apos[5]-apos_ouc[6])),
                         # np.abs(la.norm(apos[5]-apos_ouc[6])),
                         # np.abs(la.norm(apos[5]-apos_ouc[6])),
                         # np.abs(la.norm(apos[5]-apos_ouc[6])),
                         # np.abs(la.norm(apos[1]-apos_ouc[132])),
                         # ])
    Jex = sp.zeros(Nspin, Nspin_ouc)
    Gex = sp.zeros(Nspin, Nspin_ouc)
    for i in range(Nspin):
        for j in range(Nspin_ouc):
            if 2.85 < np.round(la.norm(apos[i]-apos_ouc[j]), 2) < 3.1:
                Jex[i, j] = J1
                Gex[i, j] = G * J1
                # print(i, j, np.round(la.norm(apos[i]-r_pos_ouc[j]), 2))
            elif 3.15 < np.round(la.norm(apos[i]-apos_ouc[j]), 2) < 3.35:
                Jex[i, j] = J2
                Gex[i, j] = G * J2
            elif 4.6 < np.round(la.norm(apos[i] - apos_ouc[j]), 2) < 4.7:
                Jex[i, j] = J3
                Gex[i, j] = G * J3
                # print(i, j, np.round(la.norm(apos[i]-r_pos_ouc[j]), 2))
            elif 4.8 < np.round(la.norm(apos[i] - apos_ouc[j]), 2) < 4.9:
                Jex[i, j] = J4
                Gex[i, j] = G * J4
                # print(i, j, np.round(la.norm(apos[i]-r_pos_ouc[j]), 2))
            elif 5.1 < np.round(la.norm(apos[i]-apos_ouc[j]), 2) < 5.21:
                Jex[i, j] = J5
                Gex[i, j] = G * J5
                # print(i, j, np.round(la.norm(apos[i]-r_pos_ouc[j]), 2))
            elif 5.23 < np.round(la.norm(apos[i]-apos_ouc[j]), 2) < 5.33:
                Jex[i, j] = J6
                Gex[i, j] = G * J6
                # print(i, j, np.round(la.norm(apos[i]-r_pos_ouc[j]), 2))
            elif 7.3 < np.round(la.norm(apos[i]-apos_ouc[j]), 2) < 7.36:
                Jex[i, j] = J7
                Gex[i, j] = G * J7
            else:
                Jex[i, j] = 0.0
                Gex[i, j] = 0.0
                   
    return Jex, Gex, H

def Hamiltonian(Sxyz, p):
    " Define the spin Hamiltonain for your system "
    gamma = 2.0
    beta = 110.251999
    mu = 5.7883818066e-2
    Jex, Gex, H = spin_interactions(p)
    #D = p[8]
    HM = 0
    apos = atom_pos()
    Nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    Nspin_ouc = len(apos_ouc)
    # print('Spins in the unit cell: ', Nspin)
    # print('Spins in the neighboring unit cells: ', Nspin_ouc)
    for i in range(Nspin):
        for j in range(Nspin_ouc):
            if Jex[i, j] != 0:
                HM = HM + 1/2 * Jex[i, j] * (Sxyz[i][0] * Sxyz[j][0] +
                                             Sxyz[i][1] * Sxyz[j][1] + Sxyz[i][2] * Sxyz[j][2]) + \
                         1/2 * Gex[i, j] * np.sin(math.radians(beta)) * (Sxyz[i][2] * Sxyz[j][2] -
                                             Sxyz[i][0] * Sxyz[j][0] - Sxyz[i][1] * Sxyz[j][1]) + \
                         1/2 * Gex[i, j] * np.cos(math.radians(beta)) * (Sxyz[i][0] * Sxyz[j][0] -
                                             Sxyz[i][1] * Sxyz[j][1] - Sxyz[i][2] * Sxyz[j][2])
        HM = HM + gamma * mu * Sxyz[i][0] * H
    HM = HM.expand()
    return HM
