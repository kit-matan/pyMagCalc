#%%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018

@author: Kit Matan

Calculate and plot the spin-wave dispersion for KFe3(OH)6(SO4)2
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import magcalc as mc


def plot_disp(p, S, wr):
    """Plot spin-wave dispersion for KFe3(OH)6(SO4)2
    Inputs:
        p: list of parameters
        S: spin value
        nspins: number of spins in a unit cell
        wr: 'w' for write to file, 'r' for read from file"""
    
    intv = 0.05
    qsx = np.arange(0 - intv / 2 , 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    qsy = np.arange(0 - intv / 2, 2 * np.pi + intv / 2 ,intv)
    #qsy = np.arange(0 - intv / 2 , 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    #qsx = np.arange(0 - intv / 2, 2 * np.pi + intv / 2 ,intv)
    q = []
    for i in range(len(qsx)):
        q1 = np.array([qsx[i], 0, 0])
        q.append(q1)
    for i in range(len(qsy)):
        q1 = np.array([0, qsy[i], 0])
        q.append(q1)
    En = mc.calc_disp(S, q, p, 'KFe3J', wr)

    Ekx1 = [En[i][0] for i in range(len(qsx))]
    Ekx2 = [En[i][1] for i in range(len(qsx))]
    Ekx3 = [En[i][2] for i in range(len(qsx))]
    Eky1 = [En[len(qsx) + i][0] for i in range(len(qsy))]
    Eky2 = [En[len(qsx) + i][1] for i in range(len(qsy))]
    Eky3 = [En[len(qsx) + i][2] for i in range(len(qsy))]

    # plot the spin-wave dispersion
    qsyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    plt.plot(qsx, Ekx1, 'r-', qsx, Ekx2, 'g-', qsx, Ekx3, 'b-', qsyn, Eky1, 'r-', qsyn, Eky2, 'g-', qsyn, Eky3, 'b-')
    plt.plot([2 * np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3)], [-1, 25], 'k:')
    plt.plot([2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3, 2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3], [-1, 25], 'k:')
    plt.xlim([0, 2 * np.pi / np.sqrt(3) + 2 * np.pi])
    plt.ylim([0, 20])
    plt.xticks([])
    plt.text(0, -1, r'$\Gamma$', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) - 0.1, -1, 'M')
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3 - 0.1, -1, 'K', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 0.1, -1, r'$\Gamma$', fontsize=12)
    plt.ylabel(r'$\hbar\omega$ (meV)', fontsize=12)    
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title('Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()
    # KFe3Jarosite
    S = 5.0 / 2.0  # spin value
    p = [3.23, 0.11, 0.218, -0.195, 0]
    # CCSF
    # S = 1.0 / 2.0
    # p = [12.8, -1.23, 0.063 * 12.8, -0.25 * 12.8, 0]
    plot_disp(p, S, 'w')
    et_main = default_timer()
    print('Total run-time: ', np.round((et_main-st_main) / 60, 2), ' min.')

# %%
