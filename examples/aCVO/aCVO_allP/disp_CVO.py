# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018

@author: Kit Matan
"""
import spin_model_allP as sm
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import magcalc as mc

if __name__ == "__main__":
    st = default_timer()
    S = 1.0 / 2.0  # spin value
    p = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 0, 0, 2.67, 0, 0, 0, 0, 0, 0, 0, 0.0]
    Nspin = len(sm.atom_pos())  # number of spins in a unit cell

    qsy = np.arange(1, 3 + 0.02, 0.02)
    q = []
    for i in range(len(qsy)):
        qx = 0
        qy = qsy[i] * (2 * np.pi / 8.383)
        qz = 0
        q1 = [qx, qy, qz]
        q.append(q1)
    En_ky = mc.calc_disp(S, q, p, Nspin, 'CVO', 'w')

    Eky1 = [En_ky[i][0] for i in range(len(En_ky))]
    Eky2 = [En_ky[i][1] for i in range(len(En_ky))]
    Eky3 = [En_ky[i][2] for i in range(len(En_ky))]
    Eky4 = [En_ky[i][3] for i in range(len(En_ky))]
    Eky5 = [En_ky[i][4] for i in range(len(En_ky))]
    Eky6 = [En_ky[i][5] for i in range(len(En_ky))]
    Eky7 = [En_ky[i][6] for i in range(len(En_ky))]
    Eky8 = [En_ky[i][7] for i in range(len(En_ky))]
    Eky9 = [En_ky[i][8] for i in range(len(En_ky))]
    Eky10 = [En_ky[i][9] for i in range(len(En_ky))]
    Eky11 = [En_ky[i][10] for i in range(len(En_ky))]
    Eky12 = [En_ky[i][11] for i in range(len(En_ky))]
    Eky13 = [En_ky[i][12] for i in range(len(En_ky))]
    Eky14 = [En_ky[i][13] for i in range(len(En_ky))]
    Eky15 = [En_ky[i][14] for i in range(len(En_ky))]
    Eky16 = [En_ky[i][15] for i in range(len(En_ky))]

    # plot the spin-waves dispersion
    plt.plot(qsy, Eky1, 'r-')
    plt.plot(qsy, Eky2, 'g-')
    plt.plot(qsy, Eky3, 'b-')
    plt.plot(qsy, Eky4, 'r-')
    plt.plot(qsy, Eky5, 'g-')
    plt.plot(qsy, Eky6, 'b-')
    plt.plot(qsy, Eky7, 'r-')
    plt.plot(qsy, Eky8, 'g-')
    plt.plot(qsy, Eky9, 'b-')
    plt.plot(qsy, Eky10, 'r-')
    plt.plot(qsy, Eky11, 'g-')
    plt.plot(qsy, Eky12, 'b-')
    plt.plot(qsy, Eky13, 'r-')
    plt.plot(qsy, Eky14, 'g-')
    plt.plot(qsy, Eky15, 'b-')
    plt.plot(qsy, Eky16, 'b-')
    plt.xlim([1, 3])
    # plt.ylim([0, 10])
    plt.ylabel(r'$\hbar\omega$ (meV)', fontsize=12)
    # plt.yticks(np.arange(0, 10.5, 2))
    plt.xticks(np.arange(1.0, 3.0, 0.25))
    plt.title('Spin-waves a-Cu$_2$V$_2$O$_7$')
    et = default_timer()
    print('Total run-time: ', np.round((et-st)/60, 2), ' min.')
    plt.show()

# %%
