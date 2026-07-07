#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
disp_cvo.py
Calculates and plots spin-wave dispersion for CVO (alpha-Cu2V2O7).
Reads parameters from config.yaml.
Dedicated script for Dispersion task only.
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import sys
import os
import yaml
import logging
import magcalc as mc
import tkinter as tk

# Initialize logging as in sw_CVO.py
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Q-point Generation ---
def generate_dispersion_q_vectors_cvo(plotting_config, tasks_config):
    """Generates q-vectors for CVO dispersion plot based on config."""
    q_limits_disp = plotting_config.get("q_limits_disp", [1, 3])
    q_step_disp = tasks_config.get(
        "disp_q_step", 0.02
    )
    qsy_plot_axis = np.arange(
        q_limits_disp[0], q_limits_disp[1] + q_step_disp, q_step_disp
    )

    q_list_calc = []
    lb_cvo = plotting_config.get("lattice_b_cvo", 8.383)
    qy_conversion_factor = 2 * np.pi / lb_cvo
    for qy_rlu_val in qsy_plot_axis:
        q_list_calc.append(np.array([0, qy_rlu_val * qy_conversion_factor, 0]))
    return np.array(q_list_calc), qsy_plot_axis

# --- Calculation ---
def calculate_and_save_dispersion_cvo(
    calculator: mc.MagCalc, q_vectors_array, q_plot_axis, output_filename
):
    logger.info("Calculating CVO dispersion...")
    try:
        result = calculator.calculate_dispersion(q_vectors_array)
        if result is not None:
            En = result.energies
            logger.info(f"Saving CVO dispersion results to {output_filename}...")
            results_to_save = {
                "q_vectors_calc": q_vectors_array,
                "q_plot_axis": q_plot_axis,
                "energies": En,
            }
            calculator.save_results(output_filename, results_to_save)
            logger.info("CVO dispersion results saved successfully.")
            return True
        else:
            logger.error(
                "CVO dispersion calculation returned None, cannot save results."
            )
            return False
    except Exception as e:
        logger.error(
            f"Error during CVO dispersion calculation or saving: {e}", exc_info=True
        )
        return False

# --- Plotting ---
def plot_dispersion_from_file_cvo(filename, config, ax):
    plotting_p_config = config["cvo_model"]["plotting"]
    logger.info(f"Loading CVO dispersion data from {filename} for plotting...")
    try:
        data = np.load(filename, allow_pickle=True)
        qsy_for_plot = data["q_plot_axis"]
        En_loaded_tuples = data["energies"]
        En_loaded = [np.array(e) for e in En_loaded_tuples]
        logger.info("CVO dispersion data loaded successfully.")
    except Exception as e:
        logger.error(
            f"Error loading CVO dispersion data from {filename}: {e}", exc_info=True
        )
        return

    nspins = config["cvo_model"]["nspins_for_plot"]
    
    if len(En_loaded) != len(qsy_for_plot):
         # Mismatch handling
         min_len = min(len(En_loaded), len(qsy_for_plot))
         qsy_for_plot = qsy_for_plot[:min_len]
         En_loaded = En_loaded[:min_len]

    if not En_loaded:
        return

    Eky_bands = [
        [En_loaded[i][band_idx] for i in range(len(qsy_for_plot))]
        for band_idx in range(nspins)
    ]

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]
    for band_idx in range(nspins):
        ax.plot(
            qsy_for_plot,
            Eky_bands[band_idx],
            color=colors[band_idx % len(colors)],
            linestyle="-",
        )

    ax.set_xlim(plotting_p_config.get("q_limits_disp", [1, 3]))
    ax.set_ylim(plotting_p_config.get("energy_limits_disp", [0, 10]))
    ax.set_xlabel(r"q$_y$ (r.l.u.)", fontsize=12)
    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    ax.set_title(plotting_p_config.get("disp_title", "CVO Spin Wave Dispersion"))
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def main():
    st_main = default_timer()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(script_dir, "config.yaml")
    config_filename = sys.argv[1] if len(sys.argv) > 1 else default_config_path

    config = {}
    try:
        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_filename}")
    except Exception as e:
        logger.error(f"Error loading config: {e}. Exiting.")
        sys.exit(1)

    cvo_config = config.get("cvo_model", {})
    model_p_config = cvo_config.get("model_params", {})
    calc_p_config = cvo_config.get("calculation", {})
    output_p_config = cvo_config.get("output", {})
    plotting_p_config = cvo_config.get("plotting", {})
    tasks_config = cvo_config.get("tasks", {})

    # Extract Parameters
    J1 = model_p_config.get("J1", 2.49)
    J2_ratio = model_p_config.get("J2_ratio", 1.12)
    J3_ratio = model_p_config.get("J3_ratio", 2.03)
    G1 = model_p_config.get("G1", 0.28)
    Dx = model_p_config.get("Dx", 2.67)
    Dy = model_p_config.get("Dy", -2.0)
    D3 = model_p_config.get("D3", 0.0)
    H_field_legacy = model_p_config.get("H", 0.0)
    H_dir = model_p_config.get("H_dir")
    H_mag_conf = model_p_config.get("H_mag")

    if H_dir is not None and H_mag_conf is not None:
         final_H_dir = H_dir
         final_H_mag = H_mag_conf
    else:
         final_H_mag = H_field_legacy
         # Default Z if only scalar provided, or let Minimizer decide?
         # Assuming user has migrated to modern config for vector field support.
         final_H_dir = [0, 0, 1] 

    logger.info(f"Magnetic Field Config: Mag={final_H_mag}, Dir={final_H_dir}")

    # Load Spin Model
    config_dir = os.path.dirname(os.path.abspath(config_filename))
    python_model_rel = cvo_config.get('python_model_file', 'spin_model.py')
    
    cvo_spin_model = None
    try:
        import importlib.util
        model_path = os.path.join(config_dir, python_model_rel)
        spec = importlib.util.spec_from_file_location("cvo_spin_model", model_path)
        cvo_spin_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cvo_spin_model)
        logger.info(f"Loaded spin model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load spin model: {e}")
        sys.exit(1)

    params_val = [
        J1,
        J2_ratio * J1,
        J3_ratio * J1,
        G1,
        Dx,
        Dy,
        D3,
        final_H_dir,
        final_H_mag,
    ]

    cache_mode = calc_p_config.get("cache_mode", "r")
    cache_file_base = calc_p_config.get("cache_file_base", "CVO_model_cache")

    disp_data_file = os.path.join(
        script_dir, output_p_config.get("disp_data_filename", "CVO_disp_data.npz")
    )
    
    # Update config for plotter
    config["cvo_model"]["output"]["disp_data_filename"] = disp_data_file

    # --- Minimization ---
    should_minimize = tasks_config.get("run_minimization", True)
    H_mag_check = abs(params_val[-1])

    if should_minimize and round(H_mag_check, 4) > 0.0:
        if hasattr(cvo_spin_model, 'set_magnetic_structure'):
            logger.info("Minimizing energy to find ground state...")
            try:
                calc_gs = mc.MagCalc(
                    spin_magnitude=model_p_config.get("S", 0.5),
                    hamiltonian_params=params_val,
                    cache_file_base=cache_file_base, 
                    cache_mode='auto',
                    spin_model_module=cvo_spin_model
                )
                
                nspins = calc_gs.nspins
                x0 = np.zeros(2 * nspins)
                # Smart initialization based on AL_SPIN_PREFERENCE
                al_prefs = getattr(cvo_spin_model, 'AL_SPIN_PREFERENCE', [1] * nspins)
                for i in range(nspins):
                    x0[2*i] = np.pi/2.0 - 0.05 
                    pref = al_prefs[i] if i < len(al_prefs) else 1
                    x0[2*i+1] = 0.0 if pref == 1 else np.pi

                min_res = calc_gs.minimize_energy(x0=x0)
                if min_res.success:
                     logger.info(f"Ground state found. E={min_res.fun:.4f} meV")
                     cvo_spin_model.set_magnetic_structure(min_res.x[0::2], min_res.x[1::2])
                     if cache_mode == 'r': cache_mode = 'w'
                else:
                     logger.warning("Minimization failed. Using default structure.")
            except Exception as e:
                logger.error(f"Minimization error: {e}")

    # --- Initialize Calculator ---
    try:
        calculator = mc.MagCalc(
            spin_magnitude=model_p_config.get("S", 0.5),
            hamiltonian_params=params_val,
            cache_file_base=cache_file_base,
            cache_mode=cache_mode,
            spin_model_module=cvo_spin_model
        )
        config["cvo_model"]["nspins_for_plot"] = calculator.nspins
    except Exception as e:
        logger.error(f"MagCalc Init Failed: {e}")
        sys.exit(1)

    # --- Run Dispersion ---
    plot_requested = tasks_config.get("plot_dispersion", False)
    # Check if we should calculate new or just load
    calc_new = tasks_config.get("calculate_dispersion_new", True)
    
    # If run_dispersion is explicit in logic (often implies both calc + plot or just master switch)
    # Here we assume this script IS 'run_dispersion'.
    
    if calc_new:
        q_vectors_disp, qsy_plot_axis_disp = generate_dispersion_q_vectors_cvo(plotting_p_config, tasks_config)
        calculate_and_save_dispersion_cvo(calculator, q_vectors_disp, qsy_plot_axis_disp, disp_data_file)
    
    if plot_requested:
        if os.path.exists(disp_data_file):
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_dispersion_from_file_cvo(disp_data_file, config, ax)
            
            p_filename = plotting_p_config.get("disp_plot_filename", "../plots/CVO_disp_plot.png")
            full_path = os.path.abspath(os.path.join(script_dir, p_filename))
            plt.savefig(full_path)
            logger.info(f"Plot saved to {full_path}")
            
            if plotting_p_config.get("show_plot", True):
                plt.show()
        else:
            logger.warning("No data found to plot.")

if __name__ == "__main__":
    main()
