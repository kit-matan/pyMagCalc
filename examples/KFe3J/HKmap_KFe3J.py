# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan
"""
import numpy as np
import sys
import os

# Adjust sys.path to correctly locate the magcalc package
# Get the directory of the current script (examples/KFe3J)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from timeit import default_timer
import magcalc as mc
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import spin_model as sm  # Import the spin model module


def plot_hkmap(calculator, p, S, wr, newcalc, E_intv, qstep):
    """Spinwave intensity S(Q,\\omega) 2D Q-map
        Inputs:
            calculator: An initialized MagCalc instance.
            p: list of parameters
            S: spin value
            wr: 'w' for write to file, 'r' for read from file
            newcalc: 1 for new calculation, 0 for read from file
            E_intv: energy interval for integration"""
    qsx = np.arange(0 - qstep / 2, 4 + qstep / 2, qstep) * np.pi
    qsy = np.arange(0 - qstep / 2, 4 + qstep / 2, qstep) * np.pi
    q = []
    for i in range(len(qsx)):
        for j in range(len(qsy)):
            q1 = np.array([qsx[i], qsy[j], 0])
            q.append(q1)

    print("A total number of q points: ", len(q))
    q_vectors_array = np.array(q)

    # Define cache directory path
    cache_dir = os.path.join(project_root_dir, "cache", "data")

    if newcalc == 1:
        res = calculator.calculate_sqw(q_vectors_array)
        qout, En, Sqwout = res.q_vectors, res.energies, res.intensities
        with open(os.path.join(cache_dir, 'KFe3J_HKmap_En.pck'), 'wb') as outEn:
            outEn.write(pickle.dumps(En))
        with open(os.path.join(cache_dir, 'KFe3J_HKmap_Sqw.pck'), 'wb') as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open(os.path.join(cache_dir, 'KFe3J_HKmap_En.pck'), 'rb') as inEn:
            En = pickle.loads(inEn.read())
        with open(os.path.join(cache_dir, 'KFe3J_HKmap_Sqw.pck'), 'rb') as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())

    # print(len(qsy), len(En), len(Sqwout))
    intMat = np.zeros((len(qsy), len(qsx)))
    for i in range(len(qsy)):
        for j in range(len(qsx)):
            # Original code used i*len(qsx)+j for En and i*len(qsy)+j for Sqwout.
            # Given qsx and qsy are of the same length in this script, these indices are identical.
            # This indexing corresponds to y_idx * N_x + x_idx.
            # The q list was built x-major (x_idx * N_y + y_idx).
            # This means the plotting loop effectively transposes the data or accesses it in a y-major fashion.
            # We preserve this original indexing logic.
            idx = i * len(qsx) + j # This index is used for both En and Sqwout as len(qsx) == len(qsy)

            bands_en = En[idx]
            bands_sqw = Sqwout[idx]

            for band in range(len(bands_en)):
                if bands_en[band] < E_intv[1] and bands_en[band] > E_intv[0]:
                    intMat[i, j] = intMat[i, j] + bands_sqw[band]
                # else:
                #     intMat[i, j] = intMat[i, j] # This line is redundant as intMat is initialized to zeros
    
    print("A number of Qy and Qx point: ", qsy.shape[0], qsx.shape[0])
    print("A matrix size: ", intMat.shape)
    X, Y = np.meshgrid(qsx / np.pi, qsy / np.pi)
    plt.pcolor(X, Y, intMat, cmap='PuBu_r')
    # plt.pcolormesh(X, Y, intMat, norm=LogNorm(vmin=intMat.min() + 1e-1, vmax=intMat.max()), cmap='PuBu_r')
    plt.title('Spinwave intensity Q-map for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()
    # KFe3 Jarosite
    S = 5.0 / 2.0  # spin value
    p = [3.23, 0.11, 0.218, -0.195, 0]
    # CCSF
    # S = 1.0 / 2.0
    # p = [12.8, -1.23, 0.063 * 12.8, -0.25 * 12.8, 0]

    e_inv = [8,9]
    
    ############# DO NOT CHANGE ######################
    q_step = 0.01

    # Check if cache files exist, otherwise force new calculation
    cache_dir = os.path.join(project_root_dir, "cache", "data")
    cache_file = os.path.join(cache_dir, 'KFe3J_HKmap_En.pck')
    
    calc_mode = 0  # Default to read
    if not os.path.exists(cache_file):
        print(f"Cache file {cache_file} not found. Switching to new calculation mode.")
        calc_mode = 1

    # Initialize MagCalc
    cache_file_base_name = "KFe3J_HKmap" # Using a specific name for this map
    cache_operation_mode = "w" if calc_mode == 1 else "r"
    
    calculator = mc.MagCalc(
        spin_magnitude=S,
        hamiltonian_params=p,
        cache_file_base=cache_file_base_name,
        cache_mode=cache_operation_mode,
        spin_model_module=sm,
    )
        
    plot_hkmap(calculator, p, S, 'r', calc_mode, e_inv, q_step)
    ##################################################
    
    et_main = default_timer()
    print('Total run-time: ', np.round((et_main-st_main) / 60, 2), ' min.')

# %%
