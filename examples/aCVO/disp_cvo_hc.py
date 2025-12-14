#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and plot the spin-wave dispersion for CVO (H || c).
Based on disp_CVO.py but using config_cvo_hc.yaml and spin_model_hc.
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import yaml
import sys
import os
import logging

# Adjust sys.path to correctly locate the magcalc package
# Get the directory of the current script (aCVO)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of pyMagCalc)
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc

# Import the specific spin model for CVO (H || c)
try:
    import spin_model_hc as spin_model_cvo
except ImportError:
    print("Error: Could not import spin_model_hc.py.")
    sys.exit(1)

if __name__ == "__main__":
    st = default_timer()

    # --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # --- Load Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = os.path.join(script_dir, "config_cvo_hc.yaml")

    config = {}
    try:
        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_filename}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_filename}. Exiting.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}. Exiting.")
        sys.exit(1)

    # --- Extract Configuration (Root Keys) ---
    model_p_config = config.get("model_params", {})
    calc_p_config = config.get("calculation", {}) # HC config uses 'calculation' usually?
    if not calc_p_config:
        calc_p_config = config.get("calculation_settings", {})

    plotting_p_config = config.get("plotting", {})
    
    # HC config might not have 'dispersion_calc' section if it was just for EQmap.
    # But disp_cvo_hc.py needs it. We will use defaults if missing or try to read q_scan.
    disp_config = config.get("dispersion_calc", {})
    
    # If disp_config is empty, maybe try to use q_scan from map config as a fallback?
    # Or just use hardcoded defaults for now if the user hasn't added dispersion_calc to config_cvo_hc.yaml yet.
    # config_cvo_c.yaml (Step 952 content) had q_scan, but not dispersion_calc.
    # We will prioritize 'dispersion_calc' key, fallback to 'q_scan' logic if needed, or defaults.
    
    S_val = model_p_config.get("S", 0.5)
    J1 = model_p_config.get("J1", 2.49)
    # Check if J2/J3 are explicit or ratios. config_cvo_c.yaml (Step 952) had explicit values.
    J2 = model_p_config.get("J2", 2.7888) 
    J3 = model_p_config.get("J3", 5.0547)
    G1 = model_p_config.get("G1", 0.28)
    Dx = model_p_config.get("Dx", 2.67)
    Dy = model_p_config.get("Dy", 0.0) # HC model needs Dy
    D3 = model_p_config.get("D3", 0.0) # HC model needs D3
    H_field = model_p_config.get("H", 0.0)
    
    # Params for HC model: [J1, J2, J3, G1, Dx, Dy, D3, H]
    params_val = [J1, J2, J3, G1, Dx, Dy, D3, H_field]

    cache_mode = calc_p_config.get("cache_mode", "r")
    # Base prefix: config_cvo_c.yaml (Step 952) uses 'cache_file_base' directly.
    # But EQmap appended H. Let's do the same for uniqueness.
    base_prefix = calc_p_config.get("cache_file_base", "cvo_sw_hc")
    cache_file_base = f"{base_prefix}_H{H_field:.2f}"
    
    cache_dir = os.path.join(project_root_dir, "cache", "data")
    os.makedirs(cache_dir, exist_ok=True)
    disp_data_file = os.path.join(cache_dir, f"{cache_file_base}_disp_data.npz")

    logger.info(f"Initializing MagCalc with cache base: {cache_file_base}, mode: {cache_mode}")
    
    # Robust symbolic cache check
    sym_hm_file = os.path.join(project_root_dir, 'cache', 'symbolic_matrices', f'{cache_file_base}_HM.pck')
    if not os.path.exists(sym_hm_file) and cache_mode == 'r':
         logger.warning("Symbolic cache missing, forcing 'w'")
         cache_mode = 'w'

    # --- Classical Minimization (Generalized) ---
    if abs(H_field) > 1e-9:
        if hasattr(spin_model_cvo, 'set_magnetic_structure'):
            logger.info(f"H={H_field} T. Performing classical energy minimization via MagCalc...")
            try:
                # Init temp MagCalc
                calc_gs = mc.MagCalc(
                    spin_magnitude=S_val,
                    hamiltonian_params=params_val,
                    cache_file_base=cache_file_base,
                    cache_mode=cache_mode if cache_mode != 'w' else 'auto', 
                    spin_model_module=spin_model_cvo
                )
                # Minimize
                nspins = calc_gs.nspins
                x0 = np.zeros(2 * nspins)
                # Use module's preference if available for guess
                AL_SPIN_PREFERENCE = getattr(spin_model_cvo, 'AL_SPIN_PREFERENCE', [1]*nspins)
                for i in range(nspins):
                    x0[2*i] = np.pi/2.0 - 0.05 
                    phi_pref = 0.0 if AL_SPIN_PREFERENCE[i] == 1 else np.pi
                    x0[2*i+1] = phi_pref

                min_res = calc_gs.minimize_energy(x0=x0)
                if min_res.success:
                    logger.info(f"Minimization successful. Energy: {min_res.fun:.6f} meV")
                    spin_model_cvo.set_magnetic_structure(min_res.x[0::2], min_res.x[1::2])
                    if cache_mode == 'r':
                        logger.warning("Switching cache_mode to 'w' to update Ud with new structure.")
                        cache_mode = 'w'
                else:
                    logger.error(f"Minimization failed: {min_res.message}")
            except Exception as e:
                logger.error(f"Minimization error: {e}")
        else:
             logger.info("Minimization skipped: Model missing 'set_magnetic_structure'.")
    else:
        logger.info("H=0. Skipping minimization.")

    try:
        calculator = mc.MagCalc(
            spin_magnitude=S_val,
            hamiltonian_params=params_val,
            cache_file_base=cache_file_base,
            cache_mode=cache_mode,
            spin_model_module=spin_model_cvo,
        )
        nspins = calculator.nspins
        logger.info(f"MagCalc initialized. nspins: {nspins}")

    except Exception as e:
        logger.error(f"Failed to initialize MagCalc: {e}", exc_info=True)
        sys.exit(1)

    # --- Q-point generation ---
    # Defaults if section missing
    q_path_type = disp_config.get("q_path_type", "0k0")
    q_min = disp_config.get("q_min_rlu", 1.0)
    q_max = disp_config.get("q_max_rlu", 3.0)
    n_points = disp_config.get("q_num_points", 101)
    
    qsy_plot_axis = np.linspace(q_min, q_max, n_points)
    
    q_list_calc = []
    
    # Lattice params from config (check root 'lattice_params' first as in config_cvo_c.yaml)
    lat_p = config.get("lattice_params", {})
    la = lat_p.get("la", 20.645)
    lb = lat_p.get("lb", 8.383)
    lc = lat_p.get("lc", 6.442)

    kx_fac = 2 * np.pi / la
    ky_fac = 2 * np.pi / lb
    kz_fac = 2 * np.pi / lc
    
    fixed_h = disp_config.get("fixed_h_rlu", 0.0)
    fixed_k = disp_config.get("fixed_k_rlu", 0.0)
    fixed_l = disp_config.get("fixed_l_rlu", 0.0)

    if q_path_type == "0k0":
        for q_val in qsy_plot_axis:
            q_list_calc.append(np.array([fixed_h * kx_fac, q_val * ky_fac, fixed_l * kz_fac]))
    elif q_path_type == "h00":
         for q_val in qsy_plot_axis:
            q_list_calc.append(np.array([q_val * kx_fac, fixed_k * ky_fac, fixed_l * kz_fac]))
    elif q_path_type == "00l":
         for q_val in qsy_plot_axis:
            q_list_calc.append(np.array([fixed_h * kx_fac, fixed_k * ky_fac, q_val * kz_fac]))
    else:
        # Default 0k0 fallback
        for q_val in qsy_plot_axis:
             q_list_calc.append(np.array([0, q_val * ky_fac, 0]))

    q_vectors_array_calc = np.array(q_list_calc)

    # --- Calculation ---
    logger.info("Calculating dispersion...")
    dispersion_result = calculator.calculate_dispersion(q_vectors_array_calc)
    dispersion_energies = dispersion_result.energies
    
    if dispersion_energies is not None:
         # Save
         results_to_save = {
            "q_vectors_calc": q_vectors_array_calc,
            "q_plot_axis": qsy_plot_axis,
            "energies": dispersion_energies,
         }
         try:
             calculator.save_results(disp_data_file, results_to_save)
             logger.info(f"Saved results to {disp_data_file}")
         except Exception as e:
             logger.error(f"Failed to save results: {e}")
             
         # --- Plotting ---
         if plotting_p_config.get("show_plot", True):
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            
            for band_idx in range(nspins):
                 band_data = [dispersion_energies[i][band_idx] for i in range(len(dispersion_energies))]
                 ax.plot(qsy_plot_axis, band_data, color=colors[band_idx % len(colors)], linestyle="-")
            
            # Limits (try to get from disp_config or defaults)
            ylim = [0, 12] # Default
            ax.set_ylim(ylim)
            ax.set_xlim([q_min, q_max])
            
            ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
            ax.set_xlabel(f"Q ({q_path_type}) (r.l.u.)", fontsize=12)
            
            title = plotting_p_config.get("title", f"Spin Waves (H={H_field} T)")
            ax.set_title(title)
            
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            fig.tight_layout()
            
            # Assuming 'plotting' keys might differ, we just show for now
            plt.show()
    else:
         logger.error("Calculation returned None.")
         sys.exit(1)
                
    st_end = default_timer()
    logger.info(f"Total time: {st_end - st:.2f} s")
