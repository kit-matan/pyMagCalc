# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan
"""
import numpy as np
from timeit import default_timer
import os
import sys
import pickle

# Adjust sys.path to correctly locate the magcalc package
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))
if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc
import spin_model_ha as sm # Using the CVO model (H || a)
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

if __name__ == '__main__':
    # spin-wave intensity S(Q,\omega)
    st = default_timer()
    import yaml

    # Load configuration
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    config_file = os.path.join(script_dir, "config_cvo_ha.yaml")
    
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found.")
        sys.exit(1)
        
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        
    # Parse Model Params
    mp = config.get('model_params', {})
    S = mp.get('S', 0.5)
    J1 = mp.get('J1', 2.49)
    J2 = mp.get('J2', 1.12 * 2.49) # Fallback to calculation if not in config, but config has it.
    J3 = mp.get('J3', 2.03 * 2.49)
    G1 = mp.get('G1', 0.28)
    Dx = mp.get('Dx', 2.67)
    H = mp.get('H', 0.0)
    
    # Ha model params order: [J1, J2, J3, G1, Dx, H]
    p = [J1, J2, J3, G1, Dx, H]
    
    # Parse SQW Map Params
    sqw_cfg = config.get('sqw_map_calc', {})
    lat_p = sqw_cfg.get('lattice_params', {})
    la = lat_p.get('la', 20.645)
    lb = lat_p.get('lb', 8.383)
    lc = lat_p.get('lc', 6.442)
    
    qmin = sqw_cfg.get('q_scan_min_rlu', -5.0)
    qmax = sqw_cfg.get('q_scan_max_rlu', 5.0)
    qstep = sqw_cfg.get('q_scan_step_rlu', 0.02)
    scan_dir = sqw_cfg.get('scan_direction', '0k0')
    
    ebins = sqw_cfg.get('energy_bins', {})
    emin = ebins.get('min', 0.0)
    emax = ebins.get('max', 12.0)
    estep = ebins.get('step', 0.05)
    
    wid = sqw_cfg.get('lorentzian_width', 0.2)
    
    shift = 1e-5
    qsy = np.arange(qmin + shift, qmax + shift + qstep, qstep)
    q = []

    # Generate Q-vectors based on scan direction
    # Assuming standard 0k0 scan logic similar to original script
    if scan_dir == '0k0':
        # fixed_h and fixed_l are in valid config too?
        fixed_h = sqw_cfg.get('map_fixed_h_rlu', 0.0)
        fixed_l = sqw_cfg.get('map_fixed_l_rlu', 0.0)
        for i in range(len(qsy)):
            # Conversion to inverse Angstroms? Original script used:
            # q1 = np.array([0, qsy[i] * (2 * np.pi / lb), 0])
            # This implies fixed_h=0, fixed_l=0.
            # Using config values:
            qx = fixed_h * (2 * np.pi / la)
            qy = qsy[i] * (2 * np.pi / lb)
            qz = fixed_l * (2 * np.pi / lc)
            q.append(np.array([qx, qy, qz]))
    else:
        print(f"Warning: Scan direction {scan_dir} not fully implemented in this update. Defaulting to 0k0 logic.")
        # Fallback to original logic if needed, but 0k0 is the target.
        for i in range(len(qsy)):
            q1 = np.array([0, qsy[i] * (2 * np.pi / lb), 0])
            q.append(q1)

    q_vectors_array = np.array(q)

    # Define cache paths
    calc_sets = config.get('calculation_settings', {})
    base_prefix = calc_sets.get('cache_file_base_prefix', 'cvo_sw')
    # Augment base with H
    cache_base_name = f"{base_prefix}_H{H:.2f}"
    
    cache_dir = os.path.join(project_root_dir, "cache", "data")
    en_file = os.path.join(cache_dir, f"{cache_base_name}_EQmap_En.pck")
    sqw_file = os.path.join(cache_dir, f"{cache_base_name}_EQmap_Sqw.pck")
    
    calc_mode = calc_sets.get('cache_mode', 'w')
    
    # Auto-switch to 'w' if cache files don't exist
    if not os.path.exists(en_file) and calc_mode == 'r':
        print(f"Cache file {en_file} not found. Switching to new calculation mode.")
        calc_mode = 'w'

    # Initialize MagCalc
    # Check for symbolic cache logic? MagCalc handles it if we pass correct base.
    # We should ensure symbolic cache exists or force 'w'.
    sym_hm_file = os.path.join(project_root_dir, 'cache', 'symbolic_matrices', f'{cache_base_name}_HM.pck')
    if not os.path.exists(sym_hm_file) and calc_mode == 'r':
         print("Symbolic cache missing, forcing 'w'")
         calc_mode = 'w'

    calculator = mc.MagCalc(
        spin_magnitude=S,
        hamiltonian_params=p,
        cache_file_base=cache_base_name,
        cache_mode=calc_mode, 
        spin_model_module=sm
    )

    if calc_mode == 'w':
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        qout_ky, En_ky, Sqwout_ky = calculator.calculate_sqw(q_vectors_array)
        # Save to specific files for this plotting script
        with open(en_file, "wb") as f:
            pickle.dump(En_ky, f)
        with open(sqw_file, "wb") as f:
            pickle.dump(Sqwout_ky, f)
    else:
        # qout is just q_vectors_array
        with open(en_file, "rb") as f:
            En_ky = pickle.load(f)
        with open(sqw_file, "rb") as f:
            Sqwout_ky = pickle.load(f)

    Ex = np.arange(emin, emax, estep)
    # wid read from config earlier
    intMat_ky = np.zeros((len(Ex), len(qsy)))
    fint_ky = 0
    for i in range(len(Ex)):
        for j in range(len(qsy)):
            for band in range(len(En_ky[0])):
                fint_ky = fint_ky + Sqwout_ky[j][band] * 1.0 / np.pi * \
                          wid / 2 / ((Ex[i] - En_ky[j][band]) ** 2 + (wid / 2) ** 2)
            intMat_ky[i, j] = fint_ky
            fint_ky = 0

    plot_cfg = config.get('plotting', {}).get('sqw_map_plot', {})
    vmax_scale = plot_cfg.get('vmax_scale', 5.0)
    
    X, Y = np.meshgrid(qsy, Ex)
    plt.pcolormesh(X, Y, intMat_ky, vmin=intMat_ky.min(), vmax=intMat_ky.max() / vmax_scale, cmap='PuBu_r', shading='auto')
    plt.xlim([qmin, qmax])
    plt.ylim([emin, emax])
    plt.ylabel(plot_cfg.get('ylabel', r'$\hbar\omega$ (meV)'), fontsize=12)
    plt.yticks(np.arange(emin, emax + 0.5, 2.0))
    plt.xticks(np.arange(qmin, qmax, 1))
    
    title_prefix = config.get('plotting', {}).get('figure_title_prefix', 'Spin-waves')
    plt.title(f'{title_prefix} (H={H} T)')
    plt.colorbar(label=plot_cfg.get('colorbar_label', 'Intensity'))
    
    et = default_timer()
    print('Total run-time: ', np.round((et - st) / 60, 2), ' min.')
    
    if config.get('plotting', {}).get('show_plot', True):
        plt.show()

# %%
