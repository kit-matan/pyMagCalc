#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan

Create an intensity contour map of Q and energy for spin-waves in KFe3(OH)6(SO4)2
"""
import sys
import os

# Adjust sys.path to correctly locate the magcalc package
# Get the directory of the current script (examples/KFe3J)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc
import matplotlib.pyplot as plt
import numpy as np
from timeit import default_timer
from matplotlib.colors import LogNorm
import spin_model as sm  # Import the spin model module
import pickle


def plot_map(calculator: mc.MagCalc, newcalc: int):
    """Spin-wave intensity map S(Q, omega)
    Inputs:
        calculator: An initialized MagCalc instance.
        newcalc: 1 for new calculation, 0 for reading from file

    Note: The 'wr' (cache_mode) is now handled by the calculator instance
          during its initialization.
    """

    intv = 0.05

    qsx = np.arange(
        -np.pi / np.sqrt(3) - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv
    )
    qsy = np.arange(-np.pi - intv / 2, 2 * np.pi + intv / 2, intv)
    q = []
    for i in range(len(qsx)):
        q.append(np.array([qsx[i], 0, 0]))
    for i in range(len(qsy)):
        q.append(np.array([0, qsy[i], 0]))

    q_vectors_array = np.array(q)  # Convert list of arrays to a 2D NumPy array

    # Define cache directory path
    cache_dir = os.path.join(project_root_dir, "cache", "data")

    if newcalc == 1:  # New calculation
        # Use the calculator instance's method
        res = calculator.calculate_sqw(q_vectors_array)
        qout, En, Sqwout = res.q_vectors, res.energies, res.intensities
        with open(os.path.join(cache_dir, "KFe3J_EQmap_En.pck"), "wb") as outEn:
            outEn.write(pickle.dumps(En))  # En is now a tuple of arrays
        with open(os.path.join(cache_dir, "KFe3J_EQmap_Sqw.pck"), "wb") as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open(os.path.join(cache_dir, "KFe3J_EQmap_En.pck"), "rb") as inEn:
            En = pickle.loads(inEn.read())  # Load energies
        with open(os.path.join(cache_dir, "KFe3J_EQmap_Sqw.pck"), "rb") as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())  # Load Sqw

    # En and Sqwout are now tuples of arrays, convert back to lists if needed for slicing
    En_list = list(En)
    Sqwout_list = list(Sqwout)

    En_kx = En_list[: len(qsx)]  # Separate energies for each q path
    En_ky = En_list[len(qsx) :]  # Separate energies for each q path
    Sqwout_kx = Sqwout_list[: len(qsx)]
    Sqwout_ky = Sqwout_list[len(qsx) :]

    Ex = np.arange(0, 22.5, 0.05)
    wid = 0.2
    intMat_kx = np.zeros((len(Ex), len(qsx)))
    fint_kx = 0
    for i in range(len(Ex)):
        for j in range(len(qsx)):
            for band in range(len(En_kx[0])):
                fint_kx = fint_kx + Sqwout_kx[j][band] * 1.0 / np.pi * wid / 2 / (
                    (Ex[i] - En_kx[j][band]) ** 2 + (wid / 2) ** 2
                )
            intMat_kx[i, j] = fint_kx
            fint_kx = 0

    intMat_ky = np.zeros((len(Ex), len(qsy)))
    fint_ky = 0
    for i in range(len(Ex)):
        for j in range(len(qsy)):
            for band in range(len(En_ky[0])):
                fint_ky = fint_ky + Sqwout_ky[j][band] * 1.0 / np.pi * wid / 2 / (
                    (Ex[i] - En_ky[j][band]) ** 2 + (wid / 2) ** 2
                )
            intMat_ky[i, j] = fint_ky
            fint_ky = 0

    qsyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    qsyn = np.flip(qsyn, 0)
    qs = np.concatenate((qsx, qsyn))
    intMat_ky = np.flip(intMat_ky, 1)
    intMat = np.concatenate([intMat_kx, intMat_ky], axis=-1)

    qs = np.array(qs)
    Ex = np.array(Ex)
    intMat = np.array(intMat)  # Convert to numpy array for better handling

    # Sort the data
    sort_index_qs = np.argsort(qs)
    sort_index_Ex = np.argsort(Ex)

    qs = qs[sort_index_qs]
    Ex = Ex[sort_index_Ex]
    intMat = intMat[:, sort_index_qs]
    intMat = intMat[sort_index_Ex, :]

    plt.pcolormesh(
        qs,
        Ex,
        intMat,
        norm=LogNorm(vmin=intMat.min(), vmax=intMat.max()),
        cmap="PuBu_r",
        shading="auto",
    )
    plt.xlim([-np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3) + 3 * np.pi])
    plt.ylim([0, 20])
    plt.xticks([])
    plt.text(-0.1, -1, r"$\Gamma$", fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) - 0.1, -1, "M", fontsize=12)
    plt.text(
        2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3 - 0.1, -1, "K", fontsize=12
    )
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 0.1, -1, r"$\Gamma$", fontsize=12)
    plt.ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title("Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()
    # KFe3Jarosite
    S = 5.0 / 2.0  # Spin value
    p = [
        3.23,
        0.11,
        0.218,
        -0.195,
        0,
    ]  # Parameters (J1, J2, Dy, Dz, H) as in EQmap_KFe3J.py
    cache_file_base_name = "my_model_cache"
    cache_operation_mode = "w"

    calculator = mc.MagCalc(
        spin_magnitude=S,
        hamiltonian_params=p,
        cache_file_base=cache_file_base_name,
        cache_mode=cache_operation_mode,
        spin_model_module=sm,  # Pass the imported spin model module
    )
    # calculate_and_save_sqw(calculator, S, p, "w")

    plot_map(
        calculator, newcalc=1
    )  # Pass the calculator instance, set newcalc=1 to calculate
    et_main = default_timer()
    print("Total run-time: ", np.round((et_main - st_main) / 60, 2), " min.")
