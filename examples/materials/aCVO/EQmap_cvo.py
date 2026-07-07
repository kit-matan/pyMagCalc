#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EQmap_cvo.py
Calculates and plots S(Q,w) intensity map for CVO (alpha-Cu2V2O7).
Reads parameters from config.yaml.
Dedicated script for S(Q,w) task only.
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os
import yaml
import logging
import magcalc as mc
import tkinter as tk

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# --- Q-point Generation ---
def generate_sqw_q_vectors_cvo(plotting_config, tasks_config):
    """Generates q-vectors for CVO S(Q,w) map based on config, including shift."""
    qmin_rlu_config = plotting_config.get("q_limits_sqw", [-5, 5])[0]
    qmax_rlu_config = plotting_config.get("q_limits_sqw", [-5, 5])[1]
    qstep_rlu = tasks_config.get("sqw_q_step", 0.02)
    lb_cvo = plotting_config.get("lattice_b_cvo", 8.383)
    qy_conversion_factor = 2 * np.pi / lb_cvo

    q_shift = plotting_config.get("q_shift_sqw", 0.0)
    logger.info(f"Applying q-shift of {q_shift:.6f} r.l.u.")
    
    qmin_rlu_shifted = qmin_rlu_config + q_shift
    qmax_rlu_shifted = qmax_rlu_config + q_shift

    qsy_calc_axis_shifted = np.arange(
        qmin_rlu_shifted, qmax_rlu_shifted + qstep_rlu, qstep_rlu
    )
    q_list_calc = []
    for qy_rlu_shifted_val in qsy_calc_axis_shifted:
        q_list_calc.append(np.array([0, qy_rlu_shifted_val * qy_conversion_factor, 0]))
    return np.array(q_list_calc), qsy_calc_axis_shifted

# --- Calculation ---
def calculate_and_save_sqw_cvo(
    calculator: mc.MagCalc, q_vectors_array, q_plot_axis_shifted, output_filename
):
    logger.info("Calculating CVO S(Q,w)...")
    try:
        result = calculator.calculate_sqw(q_vectors_array)
        if result is not None:
             qout = result.q_vectors
             En = result.energies
             Sqwout = result.intensities
             
             logger.info(f"Saving results to {output_filename}...")
             results_to_save = {
                "q_vectors_calc": qout,
                "q_plot_axis": q_plot_axis_shifted,
                "energies": En,
                "sqw_values": Sqwout,
             }
             calculator.save_results(output_filename, results_to_save)
             return True
        else:
             logger.error("Calculation returned None.")
             return False
    except Exception as e:
        logger.error(f"Error in S(Q,w) calculation: {e}", exc_info=True)
        return False

# --- Plotting ---
def plot_sqw_map_from_file_cvo(filename, config, ax, fig):
    plotting_p_config = config["cvo_model"]["plotting"]
    tasks_p_config = config["cvo_model"]["tasks"]
    logger.info(f"Loading data from {filename}...")
    try:
        data = np.load(filename, allow_pickle=True)
        qsy_plot_loaded_shifted = data["q_plot_axis"]
        En_loaded_tuples = data["energies"]
        Sqwout_loaded_tuples = data["sqw_values"]
        En_loaded = [np.array(e) for e in En_loaded_tuples]
        Sqwout_loaded = [np.array(s) for s in Sqwout_loaded_tuples]
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return

    emin_plot = plotting_p_config.get("energy_limits_sqw", [0, 12])[0]
    emax_plot = plotting_p_config.get("energy_limits_sqw", [0, 12])[1]
    estep_plot = tasks_p_config.get("sqw_energy_step", 0.05)
    wid_plot = tasks_p_config.get("sqw_resolution_width", 0.2)

    Ex_plot_axis = np.arange(emin_plot, emax_plot + estep_plot, estep_plot)
    intMat = np.zeros((len(Ex_plot_axis), len(qsy_plot_loaded_shifted)))

    if not En_loaded: return

    # Convolution
    for i_e, ex_val in enumerate(Ex_plot_axis):
        for j_q, _ in enumerate(qsy_plot_loaded_shifted):
            fint_val = 0
            if (En_loaded[j_q] is not None and len(En_loaded[j_q]) > 0):
                for band_idx in range(len(En_loaded[j_q])):
                    fint_val += (
                        Sqwout_loaded[j_q][band_idx] * wid_plot / 2 / np.pi
                        / ((ex_val - En_loaded[j_q][band_idx]) ** 2 + (wid_plot / 2) ** 2)
                    )
            intMat[i_e, j_q] = fint_val

    X_mesh, Y_mesh = np.meshgrid(qsy_plot_loaded_shifted, Ex_plot_axis)
    
    # Dynamic Vmin/Vmax
    min_val_for_plot = 1e-6
    max_intensity = np.max(intMat) if np.any(intMat > 0) else 1.0
    if np.any(intMat > 0):
         pos_vals = intMat[intMat > 0]
         if np.min(pos_vals) > min_val_for_plot: min_val_for_plot = np.min(pos_vals)
    
    norm_choice = LogNorm(vmin=min_val_for_plot, vmax=max_intensity)
    pcm = ax.pcolormesh(X_mesh, Y_mesh, intMat, norm=norm_choice, cmap="PuBu_r", shading="auto")

    if qsy_plot_loaded_shifted.size > 0:
        ax.set_xlim([np.min(qsy_plot_loaded_shifted), np.max(qsy_plot_loaded_shifted)])
        
    ax.set_ylim(plotting_p_config.get("energy_limits_sqw", [0, 12]))
    ax.set_xlabel(r"q$_y$ (r.l.u.)", fontsize=12)
    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    ax.set_title(plotting_p_config.get("sqw_title", "CVO S(Q,ω) Intensity Map"))
    if fig:
        fig.colorbar(pcm, ax=ax, label="S(Q,ω) (arb. units)")


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

    # Extract Parameters (Same logic as disp_cvo)
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
    sqw_data_file = os.path.join(
        script_dir, output_p_config.get("sqw_data_filename", "CVO_sqw_data.npz")
    )
    config["cvo_model"]["output"]["sqw_data_filename"] = sqw_data_file

    # --- Minimization (Can be common utility, but duplicated for standalone script robustness) ---
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
    except Exception as e:
        logger.error(f"MagCalc Init Failed: {e}")
        sys.exit(1)

    # --- Run S(Q,w) ---
    plot_requested = tasks_config.get("plot_sqw_map", False)
    calc_new = tasks_config.get("calculate_sqw_map_new", True)
    
    if calc_new:
        q_vectors_sqw, qsy_plot_axis_sqw = generate_sqw_q_vectors_cvo(plotting_p_config, tasks_config)
        calculate_and_save_sqw_cvo(calculator, q_vectors_sqw, qsy_plot_axis_sqw, sqw_data_file)
    
    if plot_requested:
        if os.path.exists(sqw_data_file):
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_sqw_map_from_file_cvo(sqw_data_file, config, ax, fig)
            
            p_filename = plotting_p_config.get("sqw_plot_filename", "../plots/CVO_sqw_plot.png")
            full_path = os.path.abspath(os.path.join(script_dir, p_filename))
            plt.savefig(full_path)
            logger.info(f"Plot saved to {full_path}")
            
            if plotting_p_config.get("show_plot", True):
                plt.show()
        else:
            logger.warning("No data found to plot.")

if __name__ == "__main__":
    main()
