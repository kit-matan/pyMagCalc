#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:11:17 2018

@author: Kit Matan

fit the spin-wave data of alpha-Cu2V2O7
The data and results are from PRL 119, 047201 (2017).
"""
import numpy as np
import magcalc as mc
from numpy import loadtxt
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
from lmfit import Model


def sw_CVO(x, J1, J2, J3, G1, Dx, H):
    
    S = 1.0 / 2.0
    p = [J1, J2, J3, G1, Dx, H]
    k = []
    for i in range(len(x[:, 0])):
        if x[i, 2] == 1:
            qx = x[i, 0] * 2 * np.pi / 20.645
            qy = 1.75 * 2 * np.pi / 8.383
            qz = 0
        elif x[i, 2] == 2:
            qx = 0
            qy = x[i, 0] * 2 * np.pi / 8.383
            qz = 0
        elif x[i, 2] == 3:
            qx = 0
            qy = 2 * 2 * np.pi / 8.383
            qz = x[i, 0] * 2 * np.pi / 6.442
        else:
            print('Wrong k-vector index!')
            sys.exit()
        q1 = np.array([qx, qy, qz])
        k.append(q1)
    En_k = mc.calc_disp(S, k, p, 'CVO', 'r')
    En = []
    for i in range(len(x[:, 0])):
        En1 = En_k[i][int(x[i, 1]-1)]
        En.append(En1)
    # print(np.abs(En-x[:, 3]))
    return En


if __name__ == "__main__":
    st = default_timer()
    data = loadtxt('data/sw_aCVO.txt', comments="#", delimiter=',', unpack=False, dtype=float)
    #p = [2.65522, 2.97384, 5.39009, 0.293822, 2.86330, 0]
    p = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 2.67, 0.0]
    x = np.zeros((len(data[:, 0]), 4))
    x[:, 0] = data[:, 0]
    y = data[:, 1]
    dy = data[:, 2]
    x[:, 1] = data[:, 3]
    x[:, 2] = data[:, 4]
    x[:, 3] = y

    sw_model = Model(sw_CVO)
    
    # fit the data using lmfit
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], G1=p[3], Dx=p[4], H=p[5])
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('G1', value=p[3], vary=False)
    params.add('Dx', value=p[4], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J1, J2, and J3.')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value, 
         result.params['J3'].value, result.params['G1'].value,
         result.params['Dx'].value, result.params['H'].value]
    print(result.fit_report())

    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], G1=p[3], Dx=p[4], H=p[5])
    params.add('J1', value=p[0], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('G1', value=p[3], vary=False)
    params.add('Dx', value=p[4], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J1, J2, and J3.')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
         result.params['J3'].value, result.params['G1'].value,
         result.params['Dx'].value, result.params['H'].value]
    print(result.fit_report())

    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], G1=p[3], Dx=p[4], H=p[5])
    params.add('J1', value=p[0], vary=False)
    params.add('J2', value=p[1], vary=False)
    params.add('G1', value=p[3], vary=False)
    params.add('Dx', value=p[4], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J1, J2, and J3.')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
         result.params['J3'].value, result.params['G1'].value,
         result.params['Dx'].value, result.params['H'].value]
    print(result.fit_report())
    
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], G1=p[3], Dx=p[4], H=p[5])
    params.add('J1', value=p[0], vary=False)
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting G1 and Dx.')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value, 
         result.params['J3'].value, result.params['G1'].value,
         result.params['Dx'].value, result.params['H'].value]
    print(result.fit_report())
    
    pfit = [result.params['J1'].value, result.params['J2'].value, 
            result.params['J3'].value, result.params['G1'].value,
            result.params['Dx'].value, result.params['H'].value]
        
    S = 1.0 / 2.0
    qsy = np.arange(1, 3 + 0.02, 0.02)
    q = []
    for i in range(len(qsy)):
        qx = 0
        qy = qsy[i] * (2 * np.pi / 8.383)
        qz = 0
        q1 = [qx, qy, qz]
        q.append(q1)
    En_ky = mc.calc_disp(S, q, pfit, 'CVO', 'r')

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

    # plot the spin-wave dispersion
    plt.plot(qsy, Eky1,  'r-')
    plt.plot(qsy, Eky2,  'g-')
    plt.plot(qsy, Eky3,  'b-')
    plt.plot(qsy, Eky4,  'r-')
    plt.plot(qsy, Eky5,  'g-')
    plt.plot(qsy, Eky6,  'b-')
    plt.plot(qsy, Eky7,  'r-')
    plt.plot(qsy, Eky8,  'g-')
    plt.plot(qsy, Eky9,  'b-')
    plt.plot(qsy, Eky10, 'r-')
    plt.plot(qsy, Eky11, 'g-')
    plt.plot(qsy, Eky12, 'b-')
    plt.plot(qsy, Eky13, 'r-')
    plt.plot(qsy, Eky14, 'g-')
    plt.plot(qsy, Eky15, 'b-')
    plt.plot(qsy, Eky16, 'b-')

    # plot the data
    kx = data[data[:, 4] == 1, 0]
    ky = data[data[:, 4] == 2, 0]
    kz = data[data[:, 4] == 3, 0]
    Enx = data[data[:, 4] == 1, 1]
    Eny = data[data[:, 4] == 2, 1]
    dEny = data[data[:, 4] == 2, 2]
    Enz = data[data[:, 4] == 3, 1]
    # plot the data only along (0, K, 0)
    plt.errorbar(ky, Eny, yerr=dEny, fmt='ko')

    plt.xlim([1, 3])
    plt.ylim([0, 10])
    plt.ylabel('$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 10.5, 2))
    plt.xticks(np.arange(1.0, 3.0, 0.25))
    plt.title('Spin-waves a-Cu$_2$V$_2$O$_7$')
    et = default_timer()
    print('Total run-time: ', np.round((et-st)/60, 2), ' min.')
    plt.show()
