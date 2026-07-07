#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Ganatee Gitgeatpong and Kit Matan
This work is based on the paper PRB 106, 214438 (2022).
"""
import os
import sys

# Adjust sys.path to correctly locate the magcalc package (if not already in path)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import spin_model as sm
import numpy as np
from timeit import default_timer
import magcalc as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import pickle


def plot_Sqw_EHmap(S, p, newcalc, wr):
    # calculate spin-wave intensity S(Q,\omega)
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382
    beta = 110.251999

    qsx = np.arange(0 - np.e / 1e5, 1 + 0.01, 0.01)
    q = []
    # calculation of S(Q,\omega) along [100]
    # Note that qx is not exactly along [100].
    for i in range(len(qsx)):
        qx = qsx[i] * astr * np.cos(math.radians(beta-90))
        qy = 2 * bstr
        qz = qsx[i] * cstr * np.sin(math.radians(beta-90))
        q1 = np.array([qx, qy, qz])
        q.append(q1)
    
    # Cache setup
    cache_dir = os.path.join(project_root_dir, 'cache', 'data')
    os.makedirs(cache_dir, exist_ok=True)
    en_file = os.path.join(cache_dir, 'ZnCVO_EQmap_En.pck')
    sqw_file = os.path.join(cache_dir, 'ZnCVO_EQmap_Sqw.pck')

    # Determine if we need to calculate
    should_calculate = (newcalc == 1) or not (os.path.exists(en_file) and os.path.exists(sqw_file))

    if should_calculate:
        # Check symbolic cache existence to avoid error in 'r' mode
        sym_hm_file = os.path.join(project_root_dir, 'cache', 'symbolic_matrices', 'ZnCVO_EHmap_HM.pck')
        cache_mode = wr
        if not os.path.exists(sym_hm_file) and cache_mode == 'r':
            print(f"Symbolic cache {sym_hm_file} missing, switching MagCalc mode to 'w'")
            cache_mode = 'w'

        # Initialize MagCalc
        calc = mc.MagCalc(spin_magnitude=S, hamiltonian_params=p, cache_file_base='ZnCVO_EHmap', 
                          spin_model_module=sm, cache_mode=cache_mode)
        
        res = calc.calculate_sqw(q)
        qout, En, Sqwout = res.q_vectors, res.energies, res.intensities
        
        with open(en_file, 'wb') as outEn:
            pickle.dump(En, outEn)
        with open(sqw_file, 'wb') as outSqwout:
            pickle.dump(Sqwout, outSqwout)
    else:
        with open(en_file, 'rb') as inEn:
            En = pickle.load(inEn)
        with open(sqw_file, 'rb') as inSqwout:
            Sqwout = pickle.load(inSqwout)

    Ex = np.arange(0, 13, 0.01)
    wid = 0.2
    intMat_kx = np.zeros((len(Ex), len(qsx)))
    fint_kx = 0
    for i in range(len(Ex)):
        for j in range(len(qsx)):
            for band in range(len(En[0])):
                fint_kx = fint_kx + Sqwout[j][band] * 1.0 / np.pi * \
                          wid / 2 / ((Ex[i] - En[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_kx[i, j] = fint_kx
            fint_kx = 0

    X, Y = np.meshgrid(qsx, Ex)
    plt.pcolormesh(X, Y, intMat_kx, norm=LogNorm(vmin=intMat_kx.min(), vmax=intMat_kx.max()), cmap='PuBu_r')
    plt.xlim([0, 1])
    plt.ylim([0, 13])
    plt.xlabel('(H, 2, 0) (r.l.u.)', fontsize=12)
    plt.ylabel(r'$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 13, 2.0))
    plt.xticks(np.arange(0, 1 + 0.25, 0.25))
    plt.title('Spin-waves in Zn$_{0.15}$Cu$_{1.85}$V$_2$O$_7$')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    S = 1.0 / 2.0  # spin value
    p = [8.497751, 0, 0, 0, 5.261605, 1.873546, 0.5095509, 0.00447892, 0]
    plot_Sqw_EHmap(S, p, 1, 'r')
    et = default_timer()
    print('Total run-time: ', np.round((et - st) / 60, 2), ' min.')
