#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Ganatee Gitgeatpong and Kit Matan
This work is based on the paper PRB 106, 214438 (2022).
"""
import KFe3J.spin_model_sf as sm
import numpy as np
from timeit import default_timer
import magcalc as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle

def plot_Sqw_EKmap(S, p, newcalc, wr):
    # spin-wave intensity S(Q,\omega)
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382

    qsy = np.arange(1 - np.e / 1e5, 3 + 0.01, 0.01)
    q = []
    for i in range(len(qsy)):
        q1 = np.array([0, qsy[i] * bstr, 0])
        q.append(q1)
            
    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, 'ZnCVO', wr)
        with open('pckFiles/ZnCVO_EKmap_En.pck', 'wb') as outEn:
            outEn.write(pickle.dumps(En))
        with open('pckFiles/ZnCVO_EKmap_Sqw.pck', 'wb') as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open('pckFiles/ZnCVO_EKmap_En.pck', 'rb') as inEn:
            En = pickle.loads(inEn.read())
        with open('pckFiles/ZnCVO_EKmap_Sqw.pck', 'rb') as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())

    Ex = np.arange(0, 13, 0.01)
    wid = 0.2
    intMat_ky = np.zeros((len(Ex), len(qsy)))
    fint_ky = 0
    for i in range(len(Ex)):
        for j in range(len(qsy)):
            for band in range(len(En[0])):
                fint_ky = fint_ky + Sqwout[j][band] * 1.0 / np.pi * \
                          wid / 2 / ((Ex[i] - En[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_ky[i, j] = fint_ky
            fint_ky = 0

    X, Y = np.meshgrid(qsy, Ex)
    plt.pcolormesh(X, Y, intMat_ky, norm=LogNorm(vmin=intMat_ky.min(), vmax=intMat_ky.max()), cmap='PuBu_r')
    plt.xlim([1, 3])
    plt.ylim([0, 13])
    plt.xlabel('(0,K,0) (r.l.u.)', fontsize=12)
    plt.ylabel('$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 13, 2.0))
    plt.xticks(np.arange(1, 3 + 0.25, 0.25))
    plt.title('Spin-waves in Zn$_{0.15}$Cu$_{1.85}$V$_2$O$_7$')
    plt.colorbar()    
    plt.show()

if __name__ == '__main__':
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382
    S = 1.0 / 2.0  # spin value
    p = [8.497751, 0, 0, 0, 5.261605, 1.873546, 0.5095509, 0.00447892, 0]
    plot_Sqw_EKmap(S, p, 1, 'r')
    et = default_timer()
    print('Total run-time: ', np.round((et - st) / 60, 2), ' min.')

