import os
import sys

# Adjust sys.path to correctly locate the magcalc package (if not already in path)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import spin_model as sm
import numpy as np
import magcalc as mc
from numpy import loadtxt
from timeit import default_timer
import matplotlib.pyplot as plt
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
    # x is passed by lmfit usually as numpy array.
    n_points = len(x)
    for i in range(n_points):
        # Safely access columns assuming x shape is (N, >=3)
        # column 0 is q value, column 2 is direction index (1=0k0, 2=h0l?)
        # Logic from original code:
        # if x[i, 2] == 1: qx=0, qy=val*bstr, qz=0 (along K)
        # if x[i, 2] == 2: qx=val*astr*.., qy=2*bstr, qz=val*cstr*.. (along H)
        
        q_idx = int(x[i, 2]) if hasattr(x[i, 2], '__int__') else int(x[i, 2])
        val = x[i, 0]
        
        if q_idx == 1:
            qx = 0
            qy = val * bstr
            qz = 0
        elif q_idx == 2:
            qx = val * astr * np.cos(math.radians(beta-90))
            qy = 2 * bstr
            qz = val * cstr * np.sin(math.radians(beta-90))
        else:
            print(f'Wrong k-vector index! {q_idx}')
            sys.exit()
        q1 = np.array([qx, qy, qz])
        k.append(q1)
    
    k_arr = np.array(k)
    
    # Initialize MagCalc
    # Use consistent cache base 'ZnCVO_lmfit'.
    # For fitting, we want 'r' mode mostly. But if symbolic missing, auto-switch to 'w' once.
    # To avoid repeated checking inside the fit loop which is called many times, we rely on 'r'.
    # Assuming 'w' run happens once initially or manually.
    # However, to be robust for first run, we try 'r', if fails we might need 'w'.
    # But inside a fit loop, re-initializing MagCalc might be costly if it re-loads cache every time?
    # MagCalc class re-init does load cache. Ideally we instantiated valid calc outside.
    # But lmfit passes params to function. The params change!
    # MagCalc checks params_changed.
    
    # We will use a standard robust check:
    cache_base = 'ZnCVO_lmfit'
    cache_mode = 'r'
    
    # Check if symbolic cache exists to avoid error in 'r' mode (only check once efficiently?)
    # Since we can't easily persist state across calls without a global or class, we check file existence.
    sym_hm_file = os.path.join(project_root_dir, 'cache', 'symbolic_matrices', f'{cache_base}_HM.pck')
    if not os.path.exists(sym_hm_file):
        cache_mode = 'w'
        
    calc = mc.MagCalc(spin_magnitude=S, hamiltonian_params=p, cache_file_base=cache_base, 
                      spin_model_module=sm, cache_mode=cache_mode)

    res = calc.calculate_dispersion(k_arr)
    energies = res.energies if res else None
        
    energies = np.array(energies)
    
    En = []
    for i in range(n_points):
        band_idx = int(x[i, 1] - 1)
        # bound check?
        if band_idx < 0 or band_idx >= energies.shape[1]:
             # fallback or error?
             En1 = 0.0 # Should not happen if data correct
        else:
             En1 = energies[i, band_idx]
        En.append(En1)
    # print(np.abs(En-x[:, 3]))
    return np.array(En)


if __name__ == "__main__":
    st = default_timer()
    # Data is now in examples/data
    data_path = os.path.join(project_root_dir, 'examples', 'data', 'sw_ZnCVO.txt')
    data = loadtxt(data_path, comments="#", delimiter=',', unpack=False, dtype=float)
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

    # Initialize calc for final plot
    # Use 'w' to force fresh calculation with best fit params? 
    # Or 'r', assuming if params changed significantly MagCalc detects it?
    # MagCalc detects param changes if cache_file_base is same but params different -> actually it overwrites 
    # if parameters in cache don't match? No, typical MagCalc implementation checks existing cache params.
    # But since we use same base 'ZnCVO_lmfit', the file will be overwritten multiple times during fit?
    # Actually, MagCalc saves numerical results with hash of params. 
    # Symbolic cache is parameter-independent (usually).
    # So we can safely use same base for symbolic.
    
    # We use 'r' here, but symbolic must exist (created during fit if needed).
    calc_final = mc.MagCalc(spin_magnitude=S, hamiltonian_params=pfit, cache_file_base='ZnCVO_lmfit', 
                            spin_model_module=sm, cache_mode='r')
    
    res_kx = calc_final.calculate_dispersion(np.array(qH))
    En_kx = res_kx.energies if res_kx else []

    # calculate dispersion along K
    for i in range(len(qsy)):
        q2 = [0, qsy[i] * bstr, 0]
        qK.append(q2)
        
    res_ky = calc_final.calculate_dispersion(np.array(qK))
    En_ky = res_ky.energies if res_ky else []

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
