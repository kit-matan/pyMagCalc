#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Ganatee Gitgeatpong and Kit Matan
This work is based on the paper PRB 106, 214438 (2022).
"""
import numpy as np
from timeit import default_timer
import magcalc as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import math
import pickle


def plot_Sqw_EHmap(S, p, newcalc, wr):
    # calculate spin-wave intensity S(Q,\omega)
    astr = 0.87101208
    bstr = 0.78257113
    cstr = 0.66266382
    beta = 110.251999

    qsx = np.arange(0 - np.e / 1e5, 1 + 0.01, 0.01)
    q = []
    # calculation of S(Q,\omega) along [100]
    # Note that qx is not exactly along [100].
    for i in range(len(qsx)):
        qx = qsx[i] * astr * np.cos(math.radians(beta-90))
        qy = 2 * bstr
        qz = qsx[i] * cstr * np.sin(math.radians(beta-90))
        q1 = np.array([qx, qy, qz])
        q.append(q1)
    
    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, 'ZnCVO', wr)
        with open('pckFiles/ZnCVO_EQmap_En.pck', 'wb') as outEn:
            outEn.write(pickle.dumps(En))
        with open('pckFiles/ZnCVO_EQmap_Sqw.pck', 'wb') as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open('pckFiles/ZnCVO_EQmap_En.pck', 'rb') as inEn:
            En = pickle.loads(inEn.read())
        with open('pckFiles/ZnCVO_EQmap_Sqw.pck', 'rb') as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())

    Ex = np.arange(0, 13, 0.01)
    wid = 0.2
    intMat_kx = np.zeros((len(Ex), len(qsx)))
    fint_kx = 0
    for i in range(len(Ex)):
        for j in range(len(qsx)):
            for band in range(len(En[0])):
                fint_kx = fint_kx + Sqwout[j][band] * 1.0 / np.pi * \
                          wid / 2 / ((Ex[i] - En[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_kx[i, j] = fint_kx
            fint_kx = 0

    X, Y = np.meshgrid(qsx, Ex)
    plt.pcolormesh(X, Y, intMat_kx, norm=LogNorm(vmin=intMat_kx.min(), vmax=intMat_kx.max()), cmap='PuBu_r')
    plt.xlim([0, 1])
    plt.ylim([0, 13])
    plt.xlabel('(H, 2, 0) (r.l.u.)', fontsize=12)
    plt.ylabel('$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 13, 2.0))
    plt.xticks(np.arange(0, 1 + 0.25, 0.25))
    plt.title('Spin-waves in Zn$_{0.15}$Cu$_{1.85}$V$_2$O$_7$')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    S = 1.0 / 2.0  # spin value
    p = [8.497751, 0, 0, 0, 5.261605, 1.873546, 0.5095509, 0.00447892, 0]
    plot_Sqw_EHmap(S, p, 1, 'r')
    et = default_timer()
    print('Total run-time: ', np.round((et - st) / 60, 2), ' min.')
