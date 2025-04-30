# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan

Create an intensity contour map of Q and energy for spin-waves in KFe3(OH)6(SO4)2
"""
import numpy as np
from timeit import default_timer
import magcalc as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pickle


def plot_map(p, S, wr, newcalc):
    """Spin-wave intensity map S(Q, omega)
        Inputs:
            p: list of parameters
            S: spin value
            nspins: number of spins in a unit cell
            wr: 'w' for write to file, 'r' for read from file
            newcalc: 1 for new calculation, 0 for reading from file"""

    intv = 0.05
    qsx = np.arange(-np.pi / np.sqrt(3) - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    qsy = np.arange(-np.pi - intv / 2, 2 * np.pi + intv / 2, intv)
    # qsy = np.arange(-np.pi / np.sqrt(3) - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    # qsx = np.arange(-np.pi - intv / 2, 2 * np.pi + intv / 2, intv)
    q = []
    for i in range(len(qsx)):
        q.append(np.array([qsx[i], 0, 0]))
    for i in range(len(qsy)):
        q.append(np.array([0, qsy[i], 0]))

    if newcalc == 1:
        qout, En, Sqwout = mc.calc_Sqw(S, q, p, 'KFe3J', wr)
        with open('pckFiles/KFe3J_EQmap_En.pck', 'wb') as outEn:
            outEn.write(pickle.dumps(En))
        with open('pckFiles/KFe3J_EQmap_Sqw.pck', 'wb') as outSqwout:
            outSqwout.write(pickle.dumps(Sqwout))
    else:
        with open('pckFiles/KFe3J_EQmap_En.pck', 'rb') as inEn:
            En = pickle.loads(inEn.read())
        with open('pckFiles/KFe3J_EQmap_Sqw.pck', 'rb') as inSqwout:
            Sqwout = pickle.loads(inSqwout.read())

    En_kx = En[:len(qsx)]
    En_ky = En[len(qsx):]
    Sqwout_kx = Sqwout[:len(qsx)]
    Sqwout_ky = Sqwout[len(qsx):]

    Ex = np.arange(0, 22.5, 0.05)
    wid = 0.2
    intMat_kx = np.zeros((len(Ex), len(qsx)))
    fint_kx = 0
    for i in range(len(Ex)):
        for j in range(len(qsx)):
            for band in range(len(En_kx[0])):
                fint_kx = fint_kx + Sqwout_kx[j][band] * 1.0 / np.pi * \
                    wid / 2 / ((Ex[i] - En_kx[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_kx[i, j] = fint_kx
            fint_kx = 0

    intMat_ky = np.zeros((len(Ex), len(qsy)))
    fint_ky = 0
    for i in range(len(Ex)):
        for j in range(len(qsy)):
            for band in range(len(En_ky[0])):
                fint_ky = fint_ky + Sqwout_ky[j][band] * 1.0 / np.pi * \
                    wid / 2 / ((Ex[i] - En_ky[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_ky[i, j] = fint_ky
            fint_ky = 0

    qsyn = 2 * np.pi + 2 * np.pi / np.sqrt(3) - qsy
    qsyn = np.flip(qsyn, 0)
    qs = np.concatenate((qsx, qsyn))
    intMat_ky = np.flip(intMat_ky, 1)
    intMat = np.concatenate([intMat_kx, intMat_ky], axis=-1)

    qs = np.array(qs)
    Ex = np.array(Ex)
    intMat = np.array(intMat)

    # Sort the data
    sort_index_qs = np.argsort(qs)
    sort_index_Ex = np.argsort(Ex)

    qs = qs[sort_index_qs]
    Ex = Ex[sort_index_Ex]
    intMat = intMat[:, sort_index_qs]
    intMat = intMat[sort_index_Ex, :]

    plt.pcolormesh(qs, Ex, intMat, norm=LogNorm(vmin=intMat.min(), vmax=intMat.max()), cmap='PuBu_r', shading='auto')
    plt.xlim([-np.pi / np.sqrt(3), 2 * np.pi / np.sqrt(3) + 3 * np.pi])
    plt.ylim([0, 20])
    plt.xticks([])
    plt.text(-0.1, -1, r'$\Gamma$', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) - 0.1, -1, 'M', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 4 * np.pi / 3 - 0.1, -1, 'K', fontsize=12)
    plt.text(2 * np.pi / np.sqrt(3) + 2 * np.pi - 0.1, -1, r'$\Gamma$', fontsize=12)
    plt.ylabel(r'$\hbar\omega$ (meV)', fontsize=12)
    plt.yticks(np.arange(0, 21, 5.0))
    plt.title('Spin-waves for KFe$_3$(OH)$_6$(SO$_4$)$_2$')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    st_main = default_timer()
    # KFe3Jarosite
    S = 5.0 / 2.0  # spin value
    # CCSF
    # S = 1.0 / 2.0
    # p = [12.8, -1.23, 0.063 * 12.8, -0.25 * 12.8, 0]
    p = [3.23, 0.11, 0.218, -0.195, 0]
    plot_map(p, S, 'r', 1)
    et_main = default_timer()
    print('Total run-time: ', np.round((et_main-st_main) / 60, 2), ' min.')

# %%
