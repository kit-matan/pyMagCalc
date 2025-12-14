#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import yaml
import numpy as np
from timeit import default_timer

# Add current directory to path if running closely
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import magcalc as mc
from magcalc.config_loader import load_spin_model_config


# Helper for generating Q-points
def generate_q_points(config_section, full_config):
    q_path_type = config_section.get("q_path_type", "0k0")
    q_min = config_section.get("q_min_rlu", 0.0)
    q_max = config_section.get("q_max_rlu", 1.0)
    n_points = config_section.get("q_num_points", 51)
    
    lat_p = full_config.get("lattice_params", {})
    la = lat_p.get("la", 1.0) # Default 1.0 if not provided
    lb = lat_p.get("lb", 1.0)
    lc = lat_p.get("lc", 1.0)
    
    kx_fac = 2 * np.pi / la
    ky_fac = 2 * np.pi / lb
    kz_fac = 2 * np.pi / lc
    
    fixed_h = config_section.get("fixed_h_rlu", 0.0)
    fixed_k = config_section.get("fixed_k_rlu", 0.0)
    fixed_l = config_section.get("fixed_l_rlu", 0.0)

    q_axis = np.linspace(q_min, q_max, n_points)
    q_list = []
    
    if q_path_type == "0k0":
        for q_val in q_axis:
            q_list.append(np.array([fixed_h * kx_fac, q_val * ky_fac, fixed_l * kz_fac]))
    elif q_path_type == "h00":
        for q_val in q_axis:
            q_list.append(np.array([q_val * kx_fac, fixed_k * ky_fac, fixed_l * kz_fac]))
    elif q_path_type == "00l":
        for q_val in q_axis:
            q_list.append(np.array([fixed_h * kx_fac, fixed_k * ky_fac, q_val * kz_fac]))
    else:
        logger.warning(f"Unknown q_path_type {q_path_type}, defaulting to 0k0 logic.")
        for q_val in q_axis:
            q_list.append(np.array([0, q_val * ky_fac, 0]))
            
    return np.array(q_list), q_axis

def main():
    parser = argparse.ArgumentParser(description="MagCalc: Spin Wave Calculation CLI")
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("magcalc_cli")

    config_path = args.config_file
    if not os.path.exists(config_path):
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logger.info(f"Loading configuration from {config_path}")
    
    # Load config manually to extract high level settings
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML config: {e}")
        sys.exit(1)

    # Initialize MagCalc
    try:
        model_params = config.get("model_params", {})
        calc_settings = config.get("calculation_settings", {})
        spin_model_name = config.get("spin_model_module")
        sm_module = None
        if spin_model_name:
             try:
                 import importlib
                 if os.getcwd() not in sys.path:
                     sys.path.insert(0, os.getcwd())
                 sm_module = importlib.import_module(spin_model_name)
             except ImportError as e:
                 logger.warning(f"Could not import specified spin_model_module '{spin_model_name}': {e}")

        # Fallback/Default loading logic
        cache_base = calc_settings.get("cache_file_base", "magcalc_run")
        cache_mode = calc_settings.get("cache_mode", "r")
        
        params = []
        if isinstance(model_params, list):
            params = model_params
        elif isinstance(model_params, dict):
            # Best to require 'hamiltonian_params' explicitly in config for CLI generic run
            params = config.get("hamiltonian_params", [])
            if not params:
                 params = list(model_params.values()) 
                 logger.warning("Using values() of model_params dict as hamiltonian_params. Order is not guaranteed!")

        calculator = mc.MagCalc(
            spin_magnitude=model_params.get("S", 0.5), # Default 0.5
            hamiltonian_params=params, 
            cache_file_base=cache_base,
            cache_mode=cache_mode,
            spin_model_module=sm_module,
            config_filepath=config_path 
        )
        
        logger.info(f"MagCalc initialized. nspins: {calculator.nspins}")

        # Dispatch Tasks
        # 1. Dispersion
        disp_config = config.get("dispersion_calc", config.get("dispersion"))
        if disp_config and disp_config.get("enabled", False):
            logger.info("Running Dispersion Calculation...")
            try:
                q_vectors, q_axis = generate_q_points(disp_config, config) # Accessing global helper function
                result = calculator.calculate_dispersion(q_vectors)
                energies = result.energies
                
                # Save results
                output_file = disp_config.get("output_file", f"{cache_base}_dispersion.npz")
                if not os.path.isabs(output_file):
                     output_file = os.path.join(os.path.dirname(config_path), output_file)

                calculator.save_results(output_file, {
                    "q_vectors": q_vectors,
                    "q_axis": q_axis,
                    "energies": energies
                })
                logger.info(f"Dispersion results saved to {output_file}")
                
            except Exception as e:
                logger.exception(f"Dispersion calculation failed: {e}")

        # 2. S(q,w)
        sqw_config = config.get("sqw_map", config.get("sqw"))
        if sqw_config and sqw_config.get("enabled", False):
             logger.info("Running S(q,w) Calculation...")
             try:
                q_vectors, q_axis = generate_q_points(sqw_config, config)
                use_stream = sqw_config.get("use_streaming", False)
                
                output_file = sqw_config.get("output_file", f"{cache_base}_sqw.npz")
                if not os.path.isabs(output_file):
                     output_file = os.path.join(os.path.dirname(config_path), output_file)

                if use_stream:
                    logger.info("Using streaming generator for S(q,w)...")
                    q_out_list = []
                    en_out_list = []
                    int_out_list = []
                    
                    for res in calculator.calculate_sqw_generator(q_vectors):
                        q_out_list.append(res[0])
                        en_out_list.append(res[1])
                        int_out_list.append(res[2])
                        
                    calculator.save_results(output_file, {
                        "q_vectors": np.array(q_out_list),
                        "energies": np.array(en_out_list),
                        "intensities": np.array(int_out_list)
                    })
                else:
                    result = calculator.calculate_sqw(q_vectors)
                    calculator.save_results(output_file, {
                        "q_vectors": result.q_vectors,
                        "energies": result.energies,
                        "intensities": result.intensities
                    })
                
                logger.info(f"S(q,w) results saved to {output_file}")
             except Exception as e:
                logger.exception(f"S(q,w) calculation failed: {e}")

        logger.info("MagCalc CLI completed successfully.")

    except Exception as e:
        logger.exception(f"Runtime error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
