# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan
"""
import numpy as np
from timeit import default_timer
import os
import sys
import pickle

# Adjust sys.path to correctly locate the magcalc package
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc
import spin_model_ha as sm # Using the CVO model (H || a)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == '__main__':
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    S = 1.0 / 2.0  # spin value
    # p = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 2.67, 3.0]
    p = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 2.67, 0.0]
    shift = 1e-5

    la = 20.645
    lb = 8.383
    lc = 6.442

    emin = 0
    emax = 12
    estep = 0.05
    qmin = -5
    qmax = 5
    qstep = 0.02

    qsy = np.arange(qmin + shift, qmax + shift + qstep, qstep)
    q = []
    for i in range(len(qsy)):
        q1 = np.array([0, qsy[i] * (2 * np.pi / lb), 0])
        q.append(q1)

    """
    qsy = np.arange(qmin + shift, qmax + shift + qstep, qstep)
    q = []
    for i in range(len(qsy)):
        qx = qsy[i] * (2 * np.pi / la)
        qy = (2.0 + 0.05) * (2 * np.pi / lb)
        qz = 0.0 * (2 * np.pi / lc)
        q1 = np.array([qx, qy, qz])
        q.append(q1)
    """

    q_vectors_array = np.array(q)

    # Define cache paths
    cache_dir = os.path.join(project_root_dir, "cache", "data")
    cache_file_base = "EQmap_CVO_cache" # Unique base for this script's result
    # We will use manual pickle saving for the map data matching the style of previous fixes,
    # or just rely on MagCalc's internal cache if we trusted it fully, but for plotting we need the raw data arrays.
    # The previous KFe3J fix manually saved/loaded .pck files. Let's do that for consistency with the requested "fix this script".
    
    en_file = os.path.join(cache_dir, "CVO_EQmap_En.pck")
    sqw_file = os.path.join(cache_dir, "CVO_EQmap_Sqw.pck")
    
    calc_mode = 'w'
    if not os.path.exists(en_file):
        print(f"Cache file {en_file} not found. Switching to new calculation mode.")
        calc_mode = 'w'

    # Initialize MagCalc
    # Note: using 'w' for cache_mode in MagCalc RE-GENERATES symbolic matrices if they depend on p (they don't usually for structure).
    # But strictly, if we are just running this script, we want the results.
    calculator = mc.MagCalc(
        spin_magnitude=S,
        hamiltonian_params=p,
        cache_file_base="CVO_model_cache", # Reuse standard CVO cache for symbolic stuff
        cache_mode=calc_mode, 
        spin_model_module=sm
    )

    if calc_mode == 'w':
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        qout_ky, En_ky, Sqwout_ky = calculator.calculate_sqw(q_vectors_array)
        # Save to specific files for this plotting script
        with open(en_file, "wb") as f:
            pickle.dump(En_ky, f)
        with open(sqw_file, "wb") as f:
            pickle.dump(Sqwout_ky, f)
    else:
        # qout is just q_vectors_array
        with open(en_file, "rb") as f:
            En_ky = pickle.load(f)
        with open(sqw_file, "rb") as f:
            Sqwout_ky = pickle.load(f)

    Ex = np.arange(emin, emax, estep)
    wid = 0.2
    intMat_ky = np.zeros((len(Ex), len(qsy)))
    fint_ky = 0
    for i in range(len(Ex)):
        for j in range(len(qsy)):
            for band in range(len(En_ky[0])):
                fint_ky = fint_ky + Sqwout_ky[j][band] * 1.0 / np.pi * \
                          wid / 2 / ((Ex[i] - En_ky[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_ky[i, j] = fint_ky
            fint_ky = 0

    X, Y = np.meshgrid(qsy, Ex)
    # plt.pcolormesh(X, Y, intMat_ky, norm=LogNorm(vmin=intMat_ky.min(), vmax=intMat_ky.max()), cmap='PuBu_r')
    plt.pcolormesh(X, Y, intMat_ky, vmin=intMat_ky.min(), vmax=intMat_ky.max() / 5, cmap='PuBu_r', shading='auto')
    plt.xlim([qmin, qmax])
    plt.ylim([emin, emax])
    plt.ylabel(r'$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(emin, emax + 0.5, 2.0))
    plt.xticks(np.arange(qmin, qmax, 1))
    plt.title('Spin-waves in a-Cu$_2$V$_2$O$_7$')
    plt.colorbar()
    et = default_timer()
    print('Total run-time: ', np.round((et - st) / 60, 2), ' min.')
    plt.show()

# %%
