# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:11:17 2018

@author: Kit Matan

fit the spin-wave data of KFe3J using lmfit package 

The data and results are from PRL 96, 247201 (2006).
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

import magcalc as mc
from numpy import loadtxt
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
from lmfit import Model


def sw_KFe3J(x, J1, J2, Dy, Dz, H):
    """Calculate the spin-wave dispersion for KFe3(OH)6(SO4)2
        Inputs:
            x: list of k-vectors
            J1: exchange constant between nearest-neighbor
            J2: exchange constant between second nearest-neighbor
            Dy: DM interaction constant along y
            Dz: DM interaction constant along z
            H: magnetic field along z"""

    S = 5.0/2.0
    p = np.array([J1, J2, Dy, Dz, H])
    k = []
    for i in range(len(x[:, 0])):
        if x[i, 2] == 2:
            qx = x[i, 0]
            qy = 0
            qz = 0
        elif x[i, 2] == 1:
            qx = 0
            qy = x[i, 0]
            qz = 0
        else:
            print('Wrong k-vector index!')
            sys.exit()
        qi = np.array([qx, qy, qz])
        k.append(qi)
    En_k = mc.calc_disp(S, k, p, 'KFe3J', 'r')
    En = []
    for i in range(len(x[:, 0])):
        Eni = En_k[i][int(x[i, 1])]
        En.append(Eni)
    return En


def fit_sw(p, x, y, dy):
    """Fit the spin-wave data using lmfit
        Inputs:
            p: initial parameters
            x: list of k-vectors
            y: list of energy
            dy: list of error of energy
        Outputs:
            pfit: fitted parameters
            result: fitting result"""
    sw_model = Model(sw_KFe3J)
    params = sw_model.make_params(J1=p[0], J2=p[1], Dy=p[2], Dz=p[3], H=p[4])
    params.add('H', value=0.0, vary=False)
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1./dy)
    pfit = [result.params['J1'].value, result.params['J2'].value, 
            result.params['Dy'].value, result.params['Dz'].value, 
            result.params['H'].value]
    return pfit, result


if __name__ == "__main__":
    st = default_timer()
    data = loadtxt(os.path.join(current_script_dir, 'sw_KFe3J.txt'), comments="#", delimiter=',', unpack=False, dtype=float)
    # p = [3.23189, 0.109754, 0.2014636998, -0.193990331, 0]
    p = [3.1783, 0.11415, 0.20403, -0.19568, 0.0]
    x = np.zeros((len(data[:, 0]), 3))
    x[:, 0] = data[:, 0] * 2
    dk = data[:, 1]
    y = data[:, 2]
    dy = data[:, 3]
    x[:, 1] = data[:, 4]
    x[:, 2] = data[:, 5]

    pfit, result = fit_sw(p, x, y, dy)

    # Plot the fitting result with the data
    S = 5.0 / 2.0
    qsx = np.arange(0, 2 * np.pi / np.sqrt(3) + 0.05, 0.05)
    qsy = np.arange(0, 2 * np.pi + 0.05, 0.05)
    q = []
    for i in range(len(qsx)):
        q1 = np.array([qsx[i], 0, 0])
        q.append(q1)
    for i in range(len(qsy)):
        q1 = np.array([0, qsy[i], 0])
        q.append(q1)
        
    En = mc.calc_disp(S, q, pfit, 'KFe3J', 'r')

    Ekx1 = [En[i][0] for i in range(len(qsx))]
    Ekx2 = [En[i][1] for i in range(len(qsx))]
    Ekx3 = [En[i][2] for i in range(len(qsx))]
    Eky1 = [En[len(qsx) + i][0] for i in range(len(qsy))]
    Eky2 = [En[len(qsx) + i][1] for i in range(len(qsy))]
    Eky3 = [En[len(qsx) + i][2] for i in range(len(qsy))]

    # plot the spin-wave dispersion from the fitting result
    qsyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    plt.plot(qsx, Ekx1, 'r-')
    plt.plot(qsx, Ekx2, 'g-')
    plt.plot(qsx, Ekx3, 'b-')
    plt.plot(qsyn, Eky1, 'r-')
    plt.plot(qsyn, Eky2, 'g-')
    plt.plot(qsyn, Eky3, 'b-')

    # plot the data
    kx = data[data[:, 5] == 2, 0] * 2
    dkx = data[data[:, 5] == 2, 1] * 2
    ky = data[data[:, 5] == 1, 0] * 2
    dky = data[data[:, 5] == 1, 1] * 2
    Enx = data[data[:, 5] == 2, 2]
    dEnx = data[data[:, 5] == 2, 3]
    Eny = data[data[:, 5] == 1, 2]
    dEny = data[data[:, 5] == 1, 3]
    kyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - ky
    plt.errorbar(kx, Enx, xerr=dkx, yerr=dEnx, fmt='ko')
    plt.errorbar(kyn, Eny, xerr=dky, yerr=dEny, fmt='ko')

    plt.plot([2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3)], [-1, 25], 'k:')
    plt.plot([2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3, 2 * np.pi /
              np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3], [-1, 25], 'k:')
    plt.xlim([0, 2 * np.pi / np.sqrt(3) + 2 * np.pi])
    plt.ylim([0, 20])
    plt.xticks([])
    plt.text(0, -1, r'$\Gamma$', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) - 0.1, -1, 'M')
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3 - 0.1, -1, 'K', fontsize=12)
    plt.text(0, -1, r'$\Gamma$', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 0.1, -1, r'$\Gamma$', fontsize=12)
    plt.ylabel(r'$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title('Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    et = default_timer()
    print('Total run-time: ', np.round((et-st) / 60, 2), ' min.')
    plt.show()

# %%
