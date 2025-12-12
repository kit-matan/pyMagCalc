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


def plot_hkmap(p, S, wr, newcalc, E_intv, qstep):
    """Spinwave intensity S(Q,\omega) 2D Q-map
        Inputs:
            p: list of parameters
            S: spin value
            nspins: number of spins in a unit cell
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

    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, 'KFe3J', wr)
        with open('pckFiles/KFe3J_HKmap_En.pck', 'wb') as outEn:
            outEn.write(pickle.dumps(En))
        with open('pckFiles/KFe3J_HKmap_Sqw.pck', 'wb') as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open('pckFiles/KFe3J_HKmap_En.pck', 'rb') as inEn:
            En = pickle.loads(inEn.read())
        with open('pckFiles/KFe3J_HKmap_Sqw.pck', 'rb') as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())

    # print(len(qsy), len(En), len(Sqwout))
    intMat = np.zeros((len(qsy), len(qsx)))
    for i in range(len(qsy)):
        for j in range(len(qsx)):
            for band in range(len(En[0])):
                if En[i*len(qsx)+j][band] < E_intv[1] and En[i*len(qsx)+j][band] > E_intv[0]:
                    intMat[i, j] = intMat[i, j] + Sqwout[i*len(qsy)+j][band]
                else:
                    intMat[i, j] = intMat[i, j]
    
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
    plot_hkmap(p, S, 'r', 0, e_inv, q_step)
    ##################################################
    
    et_main = default_timer()
    print('Total run-time: ', np.round((et_main-st_main) / 60, 2), ' min.')

# %%
