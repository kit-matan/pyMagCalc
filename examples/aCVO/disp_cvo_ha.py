#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate and plot the spin-wave dispersion for CVO (H || a).
Based on disp_CVO.py but using config_cvo_ha.yaml and spin_model_ha.
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

# Import the specific spin model for CVO (H || a)
try:
    import spin_model_ha as spin_model_cvo
except ImportError:
    print("Error: Could not import spin_model_ha.py.")
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
    config_filename = os.path.join(script_dir, "config_cvo_ha.yaml")

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
    # config_cvo_ha.yaml has keys at root level
    model_p_config = config.get("model_params", {})
    calc_p_config = config.get("calculation_settings", {}) # Note: Key might be calculation_settings or calculation
    # Checking config_cvo_ha.yaml view (Step 963): key is 'calculation_settings'
    # Checking config_cvo_c.yaml view (Step 982/952): key is 'calculation'
    # I should be careful. config_cvo_ha used 'calculation_settings'.
    
    # Let's try to support both or fallback
    if "calculation_settings" in config:
        calc_p_config = config["calculation_settings"]
    elif "calculation" in config:
        calc_p_config = config["calculation"]
        
    plotting_p_config = config.get("plotting", {})
    disp_config = config.get("dispersion_calc", {}) # Specific to config_cvo_ha
    
    # If using older config style (like cvo_c), output paths might be nested differently.
    # config_cvo_ha has 'dispersion_calc' section.
    
    S_val = model_p_config.get("S", 0.5)
    J1 = model_p_config.get("J1", 2.49)
    J2 = model_p_config.get("J2", 2.7888) # Default fallback
    J3 = model_p_config.get("J3", 5.0547)
    G1 = model_p_config.get("G1", 0.28)
    Dx = model_p_config.get("Dx", 2.67)
    H_field = model_p_config.get("H", 0.0)
    
    # Params for HA model: [J1, J2, J3, G1, Dx, H]
    params_val = [J1, J2, J3, G1, Dx, H_field]

    cache_mode = calc_p_config.get("cache_mode", "r")
    # Base prefix in config_cvo_ha is 'cache_file_base_prefix'
    base_prefix = calc_p_config.get("cache_file_base_prefix", "cvo_sw")
    cache_file_base = f"{base_prefix}_H{H_field:.2f}"
    
    # Output file logic
    # config_cvo_ha doesn't seem to have explicit 'output' section with filenames for .npz
    # It has 'plotting' -> 'plot_filename'.
    # We should define a sensible location for .npz
    cache_dir = os.path.join(project_root_dir, "cache", "data")
    os.makedirs(cache_dir, exist_ok=True)
    disp_data_file = os.path.join(cache_dir, f"{cache_file_base}_disp_data.npz")

    logger.info(f"Initializing MagCalc with cache base: {cache_file_base}, mode: {cache_mode}")
    
    # Check for symbolic cache logic?
    sym_hm_file = os.path.join(project_root_dir, 'cache', 'symbolic_matrices', f'{cache_file_base}_HM.pck')
    if not os.path.exists(sym_hm_file) and cache_mode == 'r':
         logger.warning("Symbolic cache missing, forcing 'w'")
         cache_mode = 'w'

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
    # Use dispersion_calc settings
    q_path_type = disp_config.get("q_path_type", "0k0")
    q_min = disp_config.get("q_min_rlu", 1.0)
    q_max = disp_config.get("q_max_rlu", 3.0)
    n_points = disp_config.get("q_num_points", 101)
    
    # Q-plot axis (r.l.u.)
    qsy_plot_axis = np.linspace(q_min, q_max, n_points)
    
    q_list_calc = []
    
    # Constants for conversion
    # Need to match what spin_model expect. Usually spin_model expects inverse Angstroms?
    # Original disp_CVO used 2*pi/8.383 for 0k0.
    lat_p = config.get("sqw_map_calc", {}).get("lattice_params", {}) # Fallback to looking there
    if not lat_p:
         # Fallback defaults
         la, lb, lc = 20.645, 8.383, 6.442
    else:
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
        logger.warning(f"Unknown q_path_type {q_path_type}, defaulting to 0k0 logic.")
        for q_val in qsy_plot_axis:
            q_list_calc.append(np.array([0, q_val * ky_fac, 0]))

    q_vectors_array_calc = np.array(q_list_calc)

    # --- Calculation ---
    disp_enabled = disp_config.get("enabled", True)
    if disp_enabled:
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
        else:
             logger.error("Calculation returned None.")
             sys.exit(1)
             
        # --- Plotting ---
        if plotting_p_config.get("show_plot", True) or plotting_p_config.get("save_plot", False):
            disp_plot_cfg = plotting_p_config.get("dispersion_plot", {})
            
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            prop_cycle = plt.rcParams["axes.prop_cycle"]
            colors = prop_cycle.by_key()["color"]
            
            # Plot bands
            for band_idx in range(nspins):
                 # Extract band
                 band_data = [dispersion_energies[i][band_idx] for i in range(len(dispersion_energies))]
                 ax.plot(qsy_plot_axis, band_data, color=colors[band_idx % len(colors)], linestyle="-")
            
            # Limits
            ylim = disp_plot_cfg.get("ylim", [0, 12])
            ax.set_ylim(ylim)
            ax.set_xlim([q_min, q_max])
            
            ylabel = disp_plot_cfg.get("ylabel", r"$\hbar\omega$ (meV)")
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel(f"Q ({q_path_type}) (r.l.u.)", fontsize=12)
            
            title_prefix = plotting_p_config.get("figure_title_prefix", "Spin Waves")
            ax.set_title(f"{title_prefix} (H={H_field} T)")
            
            ax.grid(axis="y", linestyle="--", alpha=0.7)
            fig.tight_layout()
            
            if plotting_p_config.get("save_plot", False):
                fn = plotting_p_config.get("plot_filename", "disp_plot.png")
                # Prepend 'disp_' to filename if it looks generic
                if "combined" in fn: fn = fn.replace("combined", "disp")
                save_path = os.path.join(script_dir, fn)
                plt.savefig(save_path)
                logger.info(f"Plot saved to {save_path}")
            
            if plotting_p_config.get("show_plot", True):
                plt.show()
                
    st_end = default_timer()
    logger.info(f"Total time: {st_end - st:.2f} s")
