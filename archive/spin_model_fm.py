#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple 1D Ferromagnet Spin Model for Testing MagCalc

Chain along x-axis, lattice constant a=1.
One atom per unit cell at the origin.
Nearest-neighbor Heisenberg interaction J.
Optional external field H along z.
"""
import sympy as sp
import numpy as np
from numpy import linalg as la
from itertools import product


def unit_cell():
    """Unit cell vectors (simple cubic, a=1)."""
    va = np.array([1.0, 0.0, 0.0])
    vb = np.array([0.0, 1.0, 0.0])  # Dummy vectors for y, z
    vc = np.array([0.0, 0.0, 1.0])
    return np.array([va, vb, vc])


def atom_pos():
    """Atomic position (one atom at origin)."""
    return np.array([[0.0, 0.0, 0.0]])


def atom_pos_ouc():
    """Atom positions including nearest neighbors along x."""
    apos = atom_pos()
    uc = unit_cell()
    # Include self, left neighbor, right neighbor
    r_pos_ouc = [apos[0], apos[0] - uc[0], apos[0] + uc[0]]
    return np.array(r_pos_ouc)


def mpr(p):
    """Rotation matrix (identity for simple FM along z)."""
    # Parameters p = [J, H] are not used for rotation here
    return [sp.eye(3)]  # List containing one identity matrix


def spin_interactions(p):
    """Generate spin interactions (nearest neighbor J).
    Input:
        p: list of parameters [J, H]"""
    J, H_field = p  # Unpack parameters
    apos = atom_pos()
    N_atom = len(apos)  # Should be 1
    apos_ouc = atom_pos_ouc()
    N_atom_ouc = len(apos_ouc)  # Should be 3
    Jex = sp.zeros(N_atom, N_atom_ouc)

    dist_tol = 0.001  # Tolerance for distance matching
    nn_dist = 1.0  # Nearest neighbor distance

    for i in range(N_atom):  # i=0
        for j in range(N_atom_ouc):  # j=0 (self), 1 (left), 2 (right)
            if i == 0 and j == 0:
                continue  # No self-interaction
            d = la.norm(apos[i] - apos_ouc[j])
            if abs(d - nn_dist) < dist_tol:
                Jex[i, j] = J

    # No DM interactions
    DM = sp.zeros(N_atom, N_atom_ouc)
    return Jex, DM


def Hamiltonian(Sxyz, pr):
    """Define the spin Hamiltonian for 1D FM.
    Inputs:
        Sxyz: list of spin operators (length = N_atom_ouc = 3)
        pr: list of parameters [J, H]"""
    Jex, DM = spin_interactions(pr)
    J_val, H_val = pr
    HM = 0
    nspins = 1  # Only 1 spin in the unit cell interacts with others
    nspins_ouc = len(Sxyz)  # Number of spins passed (should be 3)
    gamma = 2.0
    mu = 5.7883818066e-2  # Bohr magneton in meV/T

    # Heisenberg term (interaction of spin 0 with neighbors 1 and 2)
    for i in range(nspins):  # i=0
        for j in range(nspins_ouc):
            if Jex[i, j] != 0:
                # REMOVED 1/2 factor - assume magcalc handles factors internally
                HM += Jex[i, j] * (Sxyz[i].T @ Sxyz[j])[0, 0]

    # Zeeman term (field on spin 0 only)
    # Note: Factor of 1/2 removed from J term, assuming standard H = Sum J Si.Sj
    # Adjust gamma/mu factor if needed, here simplified
    HM -= gamma * mu * H_val * Sxyz[0][2]  # Add gamma*mu factor

    return HM
