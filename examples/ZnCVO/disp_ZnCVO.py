#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018

@author: Ganatee Gitgeatpong and Kit Matan
This work is based on the paper PRB 106, 214438 (2022).
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import magcalc as mc
import math
import pandas as pd

def plot_dispersion(p, wr):
    '''Plot the spin-wave dispersion of ZnCVO
        Input:
        p: a list of parameters'''
    S = 1 / 2 # spin quantum number
    beta = 110.251999
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382

    qsx = np.arange(-0.5, 1.5 + 0.01, 0.01) # along the [100] direction
    qsy = np.arange(0.5, 3.5 + 0.01, 0.01) # along the [010] direction 
    
    qH = []
    qK = []
    # Calculate the spin-wave dispersion along the [100] direction
    # Note that qx is not along [100]
    for i in range(len(qsx)):
        qx1 = qsx[i] * astr * np.cos(math.radians(beta-90))
        qy1 = 2 * bstr
        qz1 = qsx[i] * cstr * np.sin(math.radians(beta-90))
        q1 = [qx1, qy1, qz1]
        qH.append(q1)
    En_kx = mc.calc_disp(S, qH, p, 'ZnCVO', wr)
    # Calculate the spin-wave dispersion along the [010] direction
    for i in range(len(qsy)):
        q2 = [0, qsy[i] * bstr, 0]
        qK.append(q2)
    En_ky = mc.calc_disp(S, qK, p, 'ZnCVO', 'r')

    # Extract the data from the list along the [100] direction
    Ekx1 = [En_kx[i][0] for i in range(len(En_kx))]
    Ekx2 = [En_kx[i][1] for i in range(len(En_kx))]
    Ekx3 = [En_kx[i][2] for i in range(len(En_kx))]
    Ekx4 = [En_kx[i][3] for i in range(len(En_kx))]
    Ekx5 = [En_kx[i][4] for i in range(len(En_kx))]
    Ekx6 = [En_kx[i][5] for i in range(len(En_kx))]
    Ekx7 = [En_kx[i][6] for i in range(len(En_kx))]
    Ekx8 = [En_kx[i][7] for i in range(len(En_kx))]
    # Extract the data from the list along the [010] direction
    Eky1 = [En_ky[i][0] for i in range(len(En_ky))]
    Eky2 = [En_ky[i][1] for i in range(len(En_ky))]
    Eky3 = [En_ky[i][2] for i in range(len(En_ky))]
    Eky4 = [En_ky[i][3] for i in range(len(En_ky))]
    Eky5 = [En_ky[i][4] for i in range(len(En_ky))]
    Eky6 = [En_ky[i][5] for i in range(len(En_ky))]
    Eky7 = [En_ky[i][6] for i in range(len(En_ky))]
    Eky8 = [En_ky[i][7] for i in range(len(En_ky))]

    '''
    # Export the data to csv file along the kx direction
    dat_qsx = pd.DataFrame(qsx)
    dat_Ekx1 = pd.DataFrame(Ekx1)
    dat_Ekx2 = pd.DataFrame(Ekx2)
    dat_Ekx3 = pd.DataFrame(Ekx3)
    dat_Ekx4 = pd.DataFrame(Ekx4)
    dat_Ekx5 = pd.DataFrame(Ekx5)
    dat_Ekx6 = pd.DataFrame(Ekx6)
    dat_Ekx7 = pd.DataFrame(Ekx7)
    dat_Ekx8 = pd.DataFrame(Ekx8)
    result = pd.concat([dat_qsx, dat_Ekx1, dat_Ekx2, dat_Ekx3, dat_Ekx4, dat_Ekx5, dat_Ekx6, dat_Ekx7, dat_Ekx8], axis=1, ignore_index=True)
    result.to_csv('Ekx_extend.csv')

    # Export the data to csv file along the ky direction
    dat_qsy = pd.DataFrame(qsy)
    dat_Eky1 = pd.DataFrame(Eky1)
    dat_Eky2 = pd.DataFrame(Eky2)
    dat_Eky3 = pd.DataFrame(Eky3)
    dat_Eky4 = pd.DataFrame(Eky4)
    dat_Eky5 = pd.DataFrame(Eky5)
    dat_Eky6 = pd.DataFrame(Eky6)
    dat_Eky7 = pd.DataFrame(Eky7)
    dat_Eky8 = pd.DataFrame(Eky8)
    result = pd.concat([dat_qsy, dat_Eky1, dat_Eky2, dat_Eky3, dat_Eky4, dat_Eky5, dat_Eky6, dat_Eky7, dat_Eky8], axis=1, ignore_index=True)
    result.to_csv('Eky_extend.csv')
    '''

    # plot the spin-waves dispersion
    fig, ((axh,axk)) = plt.subplots(1,2)
    # along the [100] direction
    axh.plot(qsx, Ekx1, 'r-')
    axh.plot(qsx, Ekx2, 'g-')
    axh.plot(qsx, Ekx3, 'b-')
    axh.plot(qsx, Ekx4, 'k-')
    axh.plot(qsx, Ekx5, 'm-.')
    axh.plot(qsx, Ekx6, 'y-.')
    axh.plot(qsx, Ekx7, 'c-.')
    axh.plot(qsx, Ekx8, 'k--')
    # along the [010] direction
    axk.plot(qsy, Eky1, 'r-')
    axk.plot(qsy, Eky2, 'g-')
    axk.plot(qsy, Eky3, 'b-')
    axk.plot(qsy, Eky4, 'k-')
    axk.plot(qsy, Eky5, 'm-.')
    axk.plot(qsy, Eky6, 'y-.')
    axk.plot(qsy, Eky7, 'c-.')
    axk.plot(qsy, Eky8, 'k--')
    # plt.title('Spin-waves Zn$_{0.15}$Cu$_{1.85}$V$_2$O$_7$')
    plt.show()
    

if __name__ == "__main__":
    st = default_timer()
    # p = [J1, J2, J3, J4, J5, J6, J7, G, D, H]
    p = [8.497751, 0, 0, 0, 5.261605, 1.873546, 0.5095509, 0.00447892, 0]
    plot_dispersion(p, 'w')
    et = default_timer()
    print('Total run-time: ', np.round((et-st)/60, 2), ' min.')
