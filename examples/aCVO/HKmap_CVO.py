#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: kmatan
"""
import numpy as np
from timeit import default_timer
import os
import sys

# Adjust sys.path to correctly locate the magcalc package (if not already in path)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc
import spin_model_ha as sm
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import time


def plot_HKmap(p, newcalc, E_intv):
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    S = 1.0 / 2.0  # spin value
    a = 20.645
    b = 8.383
    c = 6.442
    astr = 2 * np.pi / a
    bstr = 2 * np.pi / b
    cstr = 2 * np.pi / c

    # E_intv = [7, 9]

    shift = np.e / 1000
    step_size = 0.1

    kx = np.arange(-1.5 - shift, 1.5 - shift + step_size, step_size)
    ky = np.arange(0.0 - shift, 8.0 - shift + step_size, step_size)
    kz = 0.0
    q = []
    for i in range(len(kx)):
        for j in range(len(ky)):
            qz = kx[i] * cstr
            qy = ky[j] * bstr
            qx = kz * astr
            q1 = np.array([qx, qy, qz])
            q.append(q1)
            
    q_arr = np.array(q)

    print("A number of k-points is", len(q))
    
    # Cache setup
    cache_dir = os.path.join(project_root_dir, 'cache', 'data')
    os.makedirs(cache_dir, exist_ok=True)
    en_file = os.path.join(cache_dir, 'CVO_HKmap_En.pck')
    sqw_file = os.path.join(cache_dir, 'CVO_HKmap_Sqw.pck')
    
    # Determine cache mode
    cache_mode = 'r'
    if newcalc == 1 or not os.path.exists(en_file):
        cache_mode = 'w'
    
    # Auto-check symbolic cache
    sym_cache_base = 'HKmap_CVO_base'
    sym_hm_file = os.path.join(project_root_dir, 'cache', 'symbolic_matrices', f'{sym_cache_base}_HM.pck')
    if not os.path.exists(sym_hm_file) and cache_mode == 'r':
        print("Symbolic cache missing, switching to 'w'")
        cache_mode = 'w'

    # Initialize MagCalc
    calculator = mc.MagCalc(
        spin_magnitude=S,
        hamiltonian_params=p,
        cache_file_base=sym_cache_base,
        spin_model_module=sm,
        cache_mode=cache_mode
    )

    if cache_mode == 'w':
        qout, En, Sqwout = calculator.calculate_sqw(q_arr)
        with open(en_file, 'wb') as outEn:
            pickle.dump(En, outEn)
        with open(sqw_file, 'wb') as outSqwout:
            pickle.dump(Sqwout, outSqwout)
    else:
        with open(en_file, 'rb') as inEn:
            En = pickle.load(inEn)
        with open(sqw_file, 'rb') as inSqwout:
            Sqwout = pickle.load(inSqwout)

    intMat = np.zeros((len(ky), len(kx)))
    for i in range(len(kx)):
        for j in range(len(ky)):
            for band in range(len(En[0])):
                if E_intv[1] > En[(i * len(ky) + j)][band] > E_intv[0]:
                    intMat[j, i] = intMat[j, i] + Sqwout[(i * len(ky) + j)][band]
                else:
                    intMat[j, i] = intMat[j, i]

    X, Y = np.meshgrid(kx, ky)
    # intMat[intMat <= 0] = 1e-5
    plt.pcolormesh(X, Y, intMat, vmin=intMat.min(), vmax=intMat.max(), shading='auto')
    # plt.pcolormesh(X, Y, intMat, norm=LogNorm(vmin=intMat.min(), vmax=intMat.max()), cmap='PuBu_r')
    plt.xlim([min(kx), max(kx)])
    plt.ylim([min(ky), max(ky)])
    plt.title('Spin-waves for alpha-Cu$_2$V$_2$O$_7$')
    plt.colorbar()
    et=default_timer()
    print('Total run-time: ', np.round((et-st)/60, 2), ' mins.')
    # plt.savefig('figures/CVO_HKmap.eps', format='eps', dpi=1000)
    plt.show()


if __name__ == '__main__':
    print("Start: ", time.asctime(time.localtime()))
    p = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 2.67, 0.0]
    plot_HKmap(p, 1, [6, 7])
    print("End: ", time.asctime(time.localtime()))
