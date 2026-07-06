#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 01:05:39 2018

@author: Kit Matan

this file contains the information about the spin structure that will be used
to calculate spin-waves by magcalc.py

Cs2Cu3SnF12 P21/n
"""
import sympy as sp
import numpy as np
from numpy import linalg as la


def uc_tran(vabc):
    """transform a vector in P21/n to a vector in ta kagome plane"""
    theta = 58.2780 / 180 * np.pi
    phi = 39.4430 / 180 * np.pi
    uc = [7.89032, 7.08465, 10.56426]  # unit cell in P21/n
    vxyz = np.zeros(3)
    vxyz[0] = vabc[1] * uc[1]
    vxyz[1] = vabc[0] * uc[0] * np.cos(theta) + vabc[2] * uc[2] * np.cos(phi)
    vxyz[2] = -vabc[0] * uc[0] * np.sin(theta) + vabc[2] * uc[2] * np.sin(phi)
    vxyz[np.abs(vxyz) < 1e-3] = 0
    return vxyz


def unit_cell():
    """a unit cell for the kagome lattice"""
    v_ip = uc_tran([1.0, 1.0, 1.0])
    v_op = uc_tran([0.0, 0.5, 1.0])
    va = np.array([v_ip[0], 0.0, 0.0])
    vb = np.array([0.0, v_ip[1], 0.0])
    vc = np.array([0.0, 0.0, 3 * v_op[2]])  # 3 kagome planes
    uc = [va, vb, vc]
    return uc


def atom_pos():
    """atomic positions for the kagome lattice"""
    atom1 = np.array(uc_tran([0.25364, 0.77059, 0.24238]))
    atom2 = np.array(uc_tran([0.24636, 0.27059, 0.25762]))
    atom3 = np.array(uc_tran([0.5, 0.0, 0.5]))
    atom4 = np.array(uc_tran([0.74636, 0.22941, 0.75762]))
    atom5 = np.array(uc_tran([0.75364, 0.72941, 0.74238]))
    atom6 = np.array(uc_tran([1.0, 0.5, 1.0]))
    return np.array([atom1, atom2, atom3, atom4, atom5, atom6])


def atom_pos_ouc():
    """atomic positions outside the unit cell"""
    r_pos_ouc = []
    uc = unit_cell()
    apos = atom_pos()
    for i in range(len(apos)):
        r_pos_ouc.append(apos[i])
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:  # first three spins are in the unit cell
                pass
            else:
                for k in range(len(apos)):
                    r_pos_ij = apos[k] + i * uc[0] + j * uc[1]
                    r_pos_ouc.append(r_pos_ij)
    return np.array(r_pos_ouc)

def rot_n(u,v,w, theta):
    rotn1 = np.asmatrix([[u ** 2 + (1 - u ** 2) * np.cos(theta), u * v * (1 - np.cos(theta)) - w * np.sin(theta),
              u * w * (1 - np.cos(theta)) + v * np.sin(theta)],
             [u * v * (1 - np.cos(theta)) + w * np.sin(theta), v ** 2 + (1 - v ** 2) * np.cos(theta),
              v * w * (1 - np.cos(theta)) - u * np.sin(theta)],
             [u * w * (1 - np.cos(theta)) - v * np.sin(theta), v * w * (1 - np.cos(theta)) + u * np.sin(theta),
              w ** 2 + (1 - w ** 2) * np.cos(theta)]])
    return rotn1


def crystal2cart(vec):
    ag = 7.721
    cart_vec = np.asmatrix([[vec[0, 0] - vec[2, 0] * np.sin(ag / 180 * np.pi)], [vec[1, 0]], [vec[2, 0] * np.cos(ag / 180 * np.pi)]])
    return cart_vec


def calc_angles(vec):
    norm = np.linalg.norm(vec)
    theta = np.arccos(vec[2, 0] / norm)
    if vec[0,0] == 0:
        phi = 90
    else:
        phi = np.arctan2(vec[1, 0], vec[0, 0])
    return theta, phi  # unit in radian


def spin_direction(crystal_vec):
    # rotation matrix around vector n
    # define vector n rotation around for y-axis 58.278 degrees
    u1 = 0.0
    v1 = 1.0
    w1 = 0.0
    theta1 = 58.278 / 180 * np.pi
    rot1 = rot_n(u1, v1, w1, theta1)
    # define vector n for the second rotation around z 90 degrees (counter-clockwise)
    u2 = 0.0
    v2 = 0.0
    w2 = 1.0
    theta2 = 90 / 180 * np.pi
    rot2 = rot_n(u2, v2, w2, theta2)
    rot = rot2 * rot1
    cart_vec = crystal2cart(crystal_vec)
    cart_vec_prime = rot * cart_vec
    th, ph = calc_angles(cart_vec_prime)
    return th, ph

def rot_mat(spin_list, p):
    """rotation matrix to transform spins to global coordinates"""
    J11 = p[0]
    J12 = p[1]
    J13 = p[2]
    dy = p[4]
    # Javg = (J11 + J12 + J13) / 3
    # ca = 1 / 2 * sp.acos(1 - 2 / (9 * Javg ** 2) * 3 * (dy * Javg) ** 2)
    # rotation matrix for the canting
    # define vector n rotation around for y-axis th degrees
    u1 = 0.0
    v1 = 1.0
    w1 = 0.0
    # define vector n for the second rotation around z 90 degrees (counter-clockwise)
    u2 = 0.0
    v2 = 0.0
    w2 = 1.0
    rot_m = []
    for ag in spin_list:
        rot1 = rot_n(u1, v1, w1, ag[0])
        rot2 = rot_n(u2, v2, w2, ag[1])
        rot = rot2 * rot1
        rot = sp.Matrix(rot)
        rot_m.append(rot)
    return rot_m


def mpr(p):
    """Baked negative-chirality 120-degree ground state.

    Obtained once by classical energy minimization at the variant-B
    parameters (E = -19.61 meV); see examples/CCSF/README.md. Returning it
    directly avoids re-minimizing on every run and keeps the LSWT expansion
    about the true classical ground state.
    """
    th = [1.57280669, 1.56877137, 1.51638102, 1.57280104, 1.56877714, 1.62522784]
    ph = [4.17880377, 2.10235963, 6.01060422, 4.17879820, 2.10235593, 0.27055166]
    al = [[t, f] for t, f in zip(th, ph)]
    return rot_mat(al, p)


def spin_interactions(p):
    """generate exchange interactions and spin network"""
    J11 = p[0]
    J12 = p[1]
    J13 = p[2]
    J2 = p[3]
    Dy = p[4]
    Dz = p[5]
    apos = atom_pos()
    nspins = len(apos)
    apos_ouc = atom_pos_ouc()
    nspins_ouc = len(apos_ouc)
    Jex = sp.zeros(nspins, nspins_ouc)
    dist_nnn = la.norm(apos[0] - apos[2])
    for i in range(nspins):
        for j in range(nspins_ouc):
            if np.abs(la.norm(apos[i]-apos_ouc[j])-dist_nnn) < 1.0:
                Jex[i, j] = J2
            else:
                Jex[i, j] = 0
    Jex[0, 1] = J12
    Jex[0, 29] = J13
    Jex[0, 43] = J12
    Jex[0, 44] = J11
    Jex[1, 0] = J12
    Jex[1, 2] = J13
    Jex[1, 12] = J12
    Jex[1, 29] = J11
    Jex[2, 1] = J13
    Jex[2, 3] = J11
    Jex[2, 12] = J11
    Jex[2, 16] = J13
    Jex[3, 2] = J11
    Jex[3, 4] = J12
    Jex[3, 5] = J13
    Jex[3, 16] = J12
    Jex[4, 3] = J12
    Jex[4, 5] = J11
    Jex[4, 44] = J13
    Jex[4, 45] = J12
    Jex[5, 3] = J13
    Jex[5, 4] = J11
    Jex[5, 30] = J13
    Jex[5, 31] = J11

    # generate DM interactions 
    DMvec1 = sp.MatrixSymbol('DMvec1', 1, 3)
    DMvec2 = sp.MatrixSymbol('DMvec2', 1, 3)
    DMvec3 = sp.MatrixSymbol('DMvec3', 1, 3)
    DMnull = sp.MatrixSymbol('DMnull', 1, 3)
    DMmat = sp.zeros(nspins, nspins_ouc)

    for i in range(nspins):
        for j in range(nspins_ouc):
            DMmat[i, j] = DMnull

    DMmat[0, 1] = DMvec1
    DMmat[0, 29] = -DMvec2
    DMmat[0, 43] = DMvec1
    DMmat[0, 44] = -DMvec2
    DMmat[1, 0] = -DMvec1
    DMmat[1, 2] = DMvec3
    DMmat[1, 12] = -DMvec1
    DMmat[1, 29] = DMvec3
    DMmat[2, 1] = -DMvec3
    DMmat[2, 3] = DMvec2
    DMmat[2, 12] = DMvec2
    DMmat[2, 16] = -DMvec3
    DMmat[3, 2] = -DMvec2
    DMmat[3, 4] = DMvec1
    DMmat[3, 5] = -DMvec2
    DMmat[3, 16] = DMvec1
    DMmat[4, 3] = -DMvec1
    DMmat[4, 5] = DMvec3
    DMmat[4, 44] = DMvec3
    DMmat[4, 45] = -DMvec1
    DMmat[5, 3] = DMvec2
    DMmat[5, 4] = -DMvec3
    DMmat[5, 30] = DMvec2
    DMmat[5, 31] = -DMvec3
    
    DM = DMmat.subs({DMvec1: sp.Matrix([0, Dy, -Dz]),
                     DMvec2: sp.Matrix([sp.sqrt(3) / 2 * Dy, -Dy / 2, -Dz]),
                     DMvec3: sp.Matrix([-sp.sqrt(3) / 2 * Dy, -Dy / 2, -Dz]),
                     DMnull: sp.Matrix([0, 0, 0])}).doit()
    return Jex, DM


def Hamiltonian(Sxyz, pr):
    """Define the spin Hamiltonain for your system"""
    Jex, DM = spin_interactions(pr)
    HM = 0
    gamma = 2.0
    muB = 5.7883818066e-2
    H = pr[len(pr) - 1]
    apos = atom_pos()
    Nspin = len(apos)
    apos_ouc = atom_pos_ouc()
    Nspin_ouc = len(apos_ouc)
    for i in range(Nspin):
        for j in range(Nspin_ouc):
            if Jex[i, j] != 0:
                HM = HM + 1 / 2 * Jex[i, j] * (Sxyz[i][0] * Sxyz[j][0] +
                                               Sxyz[i][1] * Sxyz[j][1] + Sxyz[i][2] * Sxyz[j][2]) + \
                        1/2 * (DM[i, j][0] * Jex[i, j] * (Sxyz[i][1] * Sxyz[j][2] - Sxyz[i][2] * Sxyz[j][1]) +
                               DM[i, j][1] * Jex[i, j] * (Sxyz[i][2] * Sxyz[j][0] - Sxyz[i][0] * Sxyz[j][2]) +
                               DM[i, j][2] * Jex[i, j] * (Sxyz[i][0] * Sxyz[j][1] - Sxyz[i][1] * Sxyz[j][0]))
        HM = HM + gamma * muB * Sxyz[i][2] * H
    HM = HM.expand()
    return HM

def ion_list():
    """Magnetic ion at each site, for the Cu(2+) magnetic form factor in S(Q,w)."""
    return ['Cu2+'] * 6
