#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 15:11:17 2018

@author: Ganatee Gitgeatpong and Kit Matan
This work is based on the paper PRB 106, 214438 (2022).

fit the spin-wave data of Zn-doped Cu2V2O7
"""
import KFe3J.spin_model_sf as sm
import numpy as np
import magcalc as mc
from numpy import loadtxt
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
from lmfit import Model
import math


def sw_ZnCVO(x, J1, J2, J3, J4, J5, J6, J7, G, H):
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382
    beta = 110.251999

    S = 1.0 / 2.0
    p = [J1, J2, J3, J4, J5, J6, J7, G, H]
    k = []
    for i in range(len(x[:, 0])):
        if x[i, 2] == 1:
            qx = 0
            qy = x[i, 0] * bstr
            qz = 0
        elif x[i, 2] == 2:
            qx = x[i, 0] * astr * np.cos(math.radians(beta-90))
            qy = 2 * bstr
            qz = x[i, 0] * cstr * np.sin(math.radians(beta-90))
        else:
            print('Wrong k-vector index!')
            sys.exit()
        q1 = np.array([qx, qy, qz])
        k.append(q1)
    En_k = mc.calc_disp(S, k, p, 'ZnCVO', 'r')
    En = []
    for i in range(len(x[:, 0])):
        En1 = En_k[i][int(x[i, 1]-1)]
        En.append(En1)
    # print(np.abs(En-x[:, 3]))
    return En


if __name__ == "__main__":
    st = default_timer()
    data = loadtxt('data/sw_ZnCVO.txt', comments="#", delimiter=',', unpack=False, dtype=float)
    p = [8.710832, 0, 0, 0, 5.048879, 2.141771, 0.4964815, 0.00435000, 0]
    x = np.zeros((len(data[:, 0]), 4))
    x[:, 0] = data[:, 0]
    y = data[:, 1]
    dy = data[:, 2]
    x[:, 1] = data[:, 3]
    x[:, 2] = data[:, 4]
    x[:, 3] = y

    # fit the data using lmfit
    # fit J_1
    sw_model = Model(sw_ZnCVO)
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], J4=p[3], J5=p[4], J6=p[5], J7=p[6], G=p[7], H=p[8])
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('J4', value=p[3], vary=False)
    params.add('J5', value=p[4], vary=False)
    params.add('J6', value=p[5], vary=False)
    params.add('J7', value=p[6], vary=False)
    params.add('G', value=p[7], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J1')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
         result.params['J3'].value, result.params['J4'].value,
         result.params['J5'].value, result.params['J6'].value,
         result.params['J7'].value, result.params['G'].value,
         result.params['H'].value]
    print(result.fit_report())

    # fit J_5
    sw_model = Model(sw_ZnCVO)
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], J4=p[3], J5=p[4], J6=p[5], J7=p[6], G=p[7], H=p[8])
    params.add('J1', value=p[0], vary=False)
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('J4', value=p[3], vary=False)
    params.add('J6', value=p[5], vary=False)
    params.add('J7', value=p[6], vary=False)
    params.add('G', value=p[7], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J5')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
         result.params['J3'].value, result.params['J4'].value,
         result.params['J5'].value, result.params['J6'].value,
         result.params['J7'].value, result.params['G'].value,
         result.params['H'].value]
    print(result.fit_report())
    
    # fit J_6
    sw_model = Model(sw_ZnCVO)
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], J4=p[3], J5=p[4], J6=p[5], J7=p[6], G=p[7], H=p[8])
    params.add('J1', value=p[0], vary=False)
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('J4', value=p[3], vary=False)
    params.add('J5', value=p[4], vary=False)
    params.add('J7', value=p[6], vary=False)
    params.add('G', value=p[7], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J6')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
         result.params['J3'].value, result.params['J4'].value,
         result.params['J5'].value, result.params['J6'].value,
         result.params['J7'].value, result.params['G'].value,
         result.params['H'].value]
    print(result.fit_report())
    
    # fit J_7
    sw_model = Model(sw_ZnCVO)
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], J4=p[3], J5=p[4], J6=p[5], J7=p[6], G=p[7], H=p[8])
    params.add('J1', value=p[0], vary=False)
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('J4', value=p[3], vary=False)
    params.add('J5', value=p[4], vary=False)
    params.add('J6', value=p[5], vary=False)
    params.add('G', value=p[7], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting J7')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
        result.params['J3'].value, result.params['J4'].value,
        result.params['J5'].value, result.params['J6'].value,
        result.params['J7'].value, result.params['G'].value,
        result.params['H'].value]
    print(result.fit_report())
    
    # fit G
    sw_model = Model(sw_ZnCVO)
    params = sw_model.make_params(J1=p[0], J2=p[1], J3=p[2], J4=p[3], J5=p[4], J6=p[5], J7=p[6], G=p[7], H=p[8])
    params.add('J1', value=p[0], vary=False)
    params.add('J2', value=p[1], vary=False)
    params.add('J3', value=p[2], vary=False)
    params.add('J4', value=p[3], vary=False)
    params.add('J5', value=p[4], vary=False)
    params.add('J6', value=p[5], vary=False)
    params.add('J7', value=p[6], vary=False)
    params.add('H', value=0.0, vary=False)
    print('Fitting G')
    result = sw_model.fit(y, params, method='leastsq', x=x, weights=1.0/dy)
    p = [result.params['J1'].value, result.params['J2'].value,
         result.params['J3'].value, result.params['J4'].value,
         result.params['J5'].value, result.params['J6'].value,
         result.params['J7'].value, result.params['G'].value,
         result.params['H'].value]
    print(result.fit_report())
  
    pfit = [result.params['J1'].value, result.params['J2'].value,
            result.params['J3'].value, result.params['J4'].value,
            result.params['J5'].value, result.params['J6'].value,
            result.params['J7'].value, result.params['G'].value,
            result.params['H'].value]
    
    S = 1.0 / 2.0
    Nspin = 8
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382
    beta = 110.251999
    qsx = np.arange(0, 1 + 0.02, 0.02)
    qsy = np.arange(1, 3 + 0.02, 0.02)
    qH = []
    qK = []
    # calculate dispersion along H
    for i in range(len(qsx)):
        qx1 = qsx[i] * astr * np.cos(math.radians(beta-90))
        qy1 = 2 * bstr
        qz1 = qsx[i] * cstr * np.sin(math.radians(beta-90))
        q1 = [qx1, qy1, qz1]
        qH.append(q1)
    En_kx = mc.calc_disp(S, qH, pfit, 'ZnCVO', 'r')
    # calculate dispersion along K
    for i in range(len(qsy)):
        q2 = [0, qsy[i] * bstr, 0]
        qK.append(q2)
    En_ky = mc.calc_disp(S, qK, p, 'ZnCVO', 'r')

    Ekx1 = [En_kx[i][0] for i in range(len(En_kx))]
    Ekx2 = [En_kx[i][1] for i in range(len(En_kx))]
    Ekx3 = [En_kx[i][2] for i in range(len(En_kx))]
    Ekx4 = [En_kx[i][3] for i in range(len(En_kx))]
    Ekx5 = [En_kx[i][4] for i in range(len(En_kx))]
    Ekx6 = [En_kx[i][5] for i in range(len(En_kx))]
    Ekx7 = [En_kx[i][6] for i in range(len(En_kx))]
    Ekx8 = [En_kx[i][7] for i in range(len(En_kx))]

    Eky1 = [En_ky[i][0] for i in range(len(En_ky))]
    Eky2 = [En_ky[i][1] for i in range(len(En_ky))]
    Eky3 = [En_ky[i][2] for i in range(len(En_ky))]
    Eky4 = [En_ky[i][3] for i in range(len(En_ky))]
    Eky5 = [En_ky[i][4] for i in range(len(En_ky))]
    Eky6 = [En_ky[i][5] for i in range(len(En_ky))]
    Eky7 = [En_ky[i][6] for i in range(len(En_ky))]
    Eky8 = [En_ky[i][7] for i in range(len(En_ky))]
    # import experimental data
    kx = data[data[:, 4] == 2, 0]
    ky = data[data[:, 4] == 1, 0]
    Enx = data[data[:, 4] == 2, 1]
    Eny = data[data[:, 4] == 1, 1]
    dEnx = data[data[:, 4] == 2, 2]
    dEny = data[data[:, 4] == 1, 2]

    # plot the data and dispersion
    # along H
    fig, ((axh,axk)) = plt.subplots(1,2)
    axh.plot(qsx, Ekx1, 'r-')
    axh.plot(qsx, Ekx2, 'g-')
    axh.plot(qsx, Ekx3, 'b-')
    axh.plot(qsx, Ekx4, 'k-')
    axh.plot(qsx, Ekx5, 'm-.')
    axh.plot(qsx, Ekx6, 'y-.')
    axh.plot(qsx, Ekx7, 'c-.')
    axh.plot(qsx, Ekx8, 'k--')
    axh.errorbar(kx, Enx, yerr=dEnx, fmt='ko')
    # along K
    axk.plot(qsy, Eky1, 'r-')
    axk.plot(qsy, Eky2, 'g-')
    axk.plot(qsy, Eky3, 'b-')
    axk.plot(qsy, Eky4, 'k-')
    axk.plot(qsy, Eky5, 'm-.')
    axk.plot(qsy, Eky6, 'y-.')
    axk.plot(qsy, Eky7, 'c-.')
    axk.plot(qsy, Eky8, 'k--')
    axk.errorbar(ky, Eny, yerr=dEny, fmt='ko')
    plt.title('Spin-waves Zn$_{0.15}$Cu$_{1.85}$V$_2$O$_7$')

    et = default_timer()
    print('Total run-time: ', np.round((et-st)/60, 2), ' min.')
    plt.show()
