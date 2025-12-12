#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: kmatan
"""
import numpy as np
from timeit import default_timer
import magcalc as mc
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle
import time


def plot_HKmap(p, newcalc, E_intv):
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    S = 1.0 / 2.0  # spin value
    a = 20.645
    b = 8.383
    c = 6.442
    astr = 2 * np.pi / a
    bstr = 2 * np.pi / b
    cstr = 2 * np.pi / c

    # E_intv = [7, 9]

    shift = np.e / 1000
    step_size = 0.1

    kx = np.arange(-1.5 - shift, 1.5 - shift + step_size, step_size)
    ky = np.arange(0.0 - shift, 8.0 - shift + step_size, step_size)
    kz = 0.0
    q = []
    for i in range(len(kx)):
        for j in range(len(ky)):
            qz = kx[i] * cstr
            qy = ky[j] * bstr
            qx = kz * astr
            q1 = np.array([qx, qy, qz])
            q.append(q1)

    print("A number of k-points is", len(q))

    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, 'CVO', 'r')
        with open('pckFiles/CVO_HKmap_En.pck', 'wb') as outEn:
                outEn.write(pickle.dumps(En))
        with open('pckFiles/CVO_HKmap_Sqw.pck', 'wb') as outSqwout:
                outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open('pckFiles/CVO_HKmap_En.pck', 'rb') as inEn:
                En = pickle.loads(inEn.read())
        with open('pckFiles/CVO_HKmap_Sqw.pck', 'rb') as inSqwout:
                Sqwout = pickle.loads(inSqwout.read())

    intMat = np.zeros((len(ky), len(kx)))
    for i in range(len(kx)):
        for j in range(len(ky)):
            for band in range(len(En[0])):
                if E_intv[1] > En[(i * len(ky) + j)][band] > E_intv[0]:
                    intMat[j, i] = intMat[j, i] + Sqwout[(i * len(ky) + j)][band]
                else:
                    intMat[j, i] = intMat[j, i]

    X, Y = np.meshgrid(kx, ky)
    # intMat[intMat <= 0] = 1e-5
    plt.pcolormesh(X, Y, intMat, vmin=intMat.min(), vmax=intMat.max(), shading='auto')
    # plt.pcolormesh(X, Y, intMat, norm=LogNorm(vmin=intMat.min(), vmax=intMat.max()), cmap='PuBu_r')
    plt.xlim([min(kx), max(kx)])
    plt.ylim([min(ky), max(ky)])
    plt.title('Spin-waves for alpha-Cu$_2$V$_2$O$_7$')
    plt.colorbar()
    et=default_timer()
    print('Total run-time: ', np.round((et-st)/60, 2), ' mins.')
    # plt.savefig('figures/CVO_HKmap.eps', format='eps', dpi=1000)
    plt.show()


if __name__ == '__main__':
    print("Start: ", time.asctime(time.localtime()))
    p = [2.49, 1.12 * 2.49, 2.03 * 2.49, 0.28, 2.67, 0.0]
    plot_HKmap(p, 1, [6, 7])
    print("End: ", time.asctime(time.localtime()))
