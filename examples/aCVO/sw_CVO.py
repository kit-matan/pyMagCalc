#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined script for CVO-like material:
Calculates and plots spin-wave dispersion and S(Q,omega) intensity map.
Reads parameters from config.yaml.
Includes handling for applied magnetic field to find classical ground state.
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Original import that causes issues when sw_CVO.py is run directly
# import magcalc as mc

import sys
import os

# Adjust sys.path to correctly locate the magcalc package
# Get the directory of the current script (aCVO)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of pyMagCalc)
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

import magcalc as mc

import yaml

# Import the CVO spin model that now includes minimization
# spin_model_module will be imported dynamically later based on config
cvo_spin_model = None

import logging
import tkinter as tk  # For screen size detection

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_screen_size_inches():
    """Gets screen size in inches and screen DPI."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window
        screen_width_pixels = root.winfo_screenwidth()
        screen_height_pixels = root.winfo_screenheight()

        # Get DPI. winfo_fpixels('1i') returns pixels per inch.
        dpi = root.winfo_fpixels("1i")
        root.destroy()

        if dpi <= 0:  # Fallback DPI if detection fails
            logger.warning(
                "Could not determine screen DPI accurately, using default 100."
            )
            dpi = 100.0
        else:
            logger.debug(f"Detected screen DPI: {dpi:.2f}")

        screen_width_inches = screen_width_pixels / dpi
        screen_height_inches = screen_height_pixels / dpi
        logger.debug(
            f"Detected screen size: {screen_width_pixels}x{screen_height_pixels} pixels; "
            f"{screen_width_inches:.2f}x{screen_height_inches:.2f} inches."
        )
        return screen_width_inches, screen_height_inches, dpi
    except Exception as e:
        logger.warning(
            f"Could not get screen size using tkinter: {e}. Plot will use default figsize."
        )
        return None, None, None


# --- Q-point Generation ---
def generate_dispersion_q_vectors_cvo(plotting_config, tasks_config):
    """Generates q-vectors for CVO dispersion plot based on config."""
    q_limits_disp = plotting_config.get("q_limits_disp", [1, 3])
    q_step_disp = tasks_config.get(
        "disp_q_step", 0.02
    )  # Use disp_q_step from tasks_config
    qsy_plot_axis = np.arange(
        q_limits_disp[0], q_limits_disp[1] + q_step_disp, q_step_disp
    )

    q_list_calc = []
    # Lattice parameter b for CVO, ensure this is consistent with spin_model_cvo_Hc.py if used there
    lb_cvo = plotting_config.get("lattice_b_cvo", 8.383)
    qy_conversion_factor = 2 * np.pi / lb_cvo
    for qy_rlu_val in qsy_plot_axis:
        q_list_calc.append(np.array([0, qy_rlu_val * qy_conversion_factor, 0]))
    return np.array(q_list_calc), qsy_plot_axis


def generate_sqw_q_vectors_cvo(plotting_config, tasks_config):
    """Generates q-vectors for CVO S(Q,w) map based on config, including shift."""
    qmin_rlu_config = plotting_config.get("q_limits_sqw", [-5, 5])[0]
    qmax_rlu_config = plotting_config.get("q_limits_sqw", [-5, 5])[1]
    qstep_rlu = tasks_config.get("sqw_q_step", 0.02)
    lb_cvo = plotting_config.get("lattice_b_cvo", 8.383)
    qy_conversion_factor = 2 * np.pi / lb_cvo

    q_shift = plotting_config.get(
        "q_shift_sqw", np.e / 1e5
    )  # Make q_shift configurable
    logger.info(
        f"Applying q-shift of {q_shift:.6f} r.l.u. to S(Q,w) calculation range."
    )
    qmin_rlu_shifted = qmin_rlu_config + q_shift
    qmax_rlu_shifted = qmax_rlu_config + q_shift

    qsy_calc_axis_shifted = np.arange(
        qmin_rlu_shifted, qmax_rlu_shifted + qstep_rlu, qstep_rlu
    )
    q_list_calc = []
    for qy_rlu_shifted_val in qsy_calc_axis_shifted:
        q_list_calc.append(np.array([0, qy_rlu_shifted_val * qy_conversion_factor, 0]))
    return np.array(q_list_calc), qsy_calc_axis_shifted


# --- Calculation and Saving Functions ---
def calculate_and_save_dispersion_cvo(
    calculator: mc.MagCalc, q_vectors_array, q_plot_axis, output_filename
):
    logger.info("Calculating CVO dispersion...")
    try:
        En = calculator.calculate_dispersion(q_vectors_array)
        if En is not None:
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


def calculate_and_save_sqw_cvo(
    calculator: mc.MagCalc, q_vectors_array, q_plot_axis_shifted, output_filename
):
    logger.info("Calculating CVO S(Q,w)...")
    try:
        qout, En, Sqwout = calculator.calculate_sqw(q_vectors_array)
        if En is not None and Sqwout is not None:
            logger.info(f"Saving CVO S(Q,w) results to {output_filename}...")
            results_to_save = {
                "q_vectors_calc": qout,
                "q_plot_axis": q_plot_axis_shifted,
                "energies": En,
                "sqw_values": Sqwout,
            }
            calculator.save_results(output_filename, results_to_save)
            logger.info("CVO S(Q,w) results saved successfully.")
            return True
        else:
            logger.error(
                "CVO S(Q,w) calculation returned None for En or Sqwout, cannot save results."
            )
            return False
    except Exception as e:
        logger.error(
            f"Error during CVO S(Q,w) calculation or saving: {e}", exc_info=True
        )
        return False


# --- Plotting Functions ---
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
        logger.warning(
            "Mismatch in q-points and energies for CVO dispersion. Plotting may be incorrect."
        )
        min_len = min(len(En_loaded), len(qsy_for_plot))
        qsy_for_plot = qsy_for_plot[:min_len]
        En_loaded = En_loaded[:min_len]

    if (
        not En_loaded
        or not hasattr(En_loaded[0], "__len__")
        or len(En_loaded[0]) < nspins
    ):
        logger.error(
            f"Loaded energy data does not have enough bands (expected {nspins}). Cannot plot."
        )
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
    ax.set_xticks(plotting_p_config.get("q_ticks_disp", np.arange(1.0, 3.01, 0.25)))
    ax.set_xlabel(r"q$_y$ (r.l.u.)", fontsize=12)
    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    y_lim_plot = ax.get_ylim()
    ax.set_yticks(np.arange(np.floor(y_lim_plot[0]), np.ceil(y_lim_plot[1]) + 1, 2))
    ax.set_title(plotting_p_config.get("disp_title", "CVO Spin Wave Dispersion"))
    ax.grid(axis="y", linestyle="--", alpha=0.7)


def plot_sqw_map_from_file_cvo(filename, config, ax, fig):
    plotting_p_config = config["cvo_model"]["plotting"]
    tasks_p_config = config["cvo_model"]["tasks"]
    logger.info(f"Loading CVO S(Q,w) data from {filename} for plotting...")
    try:
        data = np.load(filename, allow_pickle=True)
        qsy_plot_loaded_shifted = data["q_plot_axis"]
        En_loaded_tuples = data["energies"]
        Sqwout_loaded_tuples = data["sqw_values"]
        En_loaded = [np.array(e) for e in En_loaded_tuples]
        Sqwout_loaded = [np.array(s) for s in Sqwout_loaded_tuples]
        logger.info("CVO S(Q,w) data loaded successfully.")
    except Exception as e:
        logger.error(
            f"Error loading CVO S(Q,w) data from {filename}: {e}", exc_info=True
        )
        return

    emin_plot = plotting_p_config.get("energy_limits_sqw", [0, 12])[0]
    emax_plot = plotting_p_config.get("energy_limits_sqw", [0, 12])[1]
    estep_plot = tasks_p_config.get("sqw_energy_step", 0.05)
    wid_plot = tasks_p_config.get("sqw_resolution_width", 0.2)

    Ex_plot_axis = np.arange(emin_plot, emax_plot + estep_plot, estep_plot)
    intMat = np.zeros((len(Ex_plot_axis), len(qsy_plot_loaded_shifted)))

    if len(En_loaded) != len(qsy_plot_loaded_shifted) or len(Sqwout_loaded) != len(
        qsy_plot_loaded_shifted
    ):
        logger.error(
            "Mismatch between q-points and loaded S(Q,w) data for CVO. Cannot plot."
        )
        return

    for i_e, ex_val in enumerate(Ex_plot_axis):
        for j_q, _ in enumerate(qsy_plot_loaded_shifted):
            fint_val = 0
            if (
                En_loaded[j_q] is not None
                and Sqwout_loaded[j_q] is not None
                and len(En_loaded[j_q]) > 0
            ):
                for band_idx in range(len(En_loaded[j_q])):
                    fint_val += (
                        Sqwout_loaded[j_q][band_idx]
                        * 1.0
                        / np.pi
                        * wid_plot
                        / 2
                        / (
                            (ex_val - En_loaded[j_q][band_idx]) ** 2
                            + (wid_plot / 2) ** 2
                        )
                    )
            intMat[i_e, j_q] = fint_val

    X_mesh, Y_mesh = np.meshgrid(qsy_plot_loaded_shifted, Ex_plot_axis)

    logger.debug(f"CVO Intensity matrix (intMat) shape: {intMat.shape}")
    logger.debug(
        f"CVO Min value in intMat: {np.min(intMat)}, Max value: {np.max(intMat)}"
    )
    min_val_for_plot = 1e-6
    if np.any(intMat > 0):
        min_pos_val = np.min(intMat[intMat > 0])
        logger.debug(f"CVO Min positive value in intMat: {min_pos_val}")
        if min_pos_val >= 1e-6:
            min_val_for_plot = min_pos_val
    else:
        logger.debug("All values in CVO intMat are zero or negative.")

    max_intensity = np.max(intMat) if np.any(intMat > 0) else 1.0
    if max_intensity <= min_val_for_plot:
        max_intensity = min_val_for_plot * 100
    if max_intensity < min_val_for_plot * 1.1:
        logger.warning(
            f"CVO Max intensity ({max_intensity:.2e}) is too close to min for log ({min_val_for_plot:.2e})."
        )
        if np.any(intMat > 0):
            max_intensity = np.max(intMat[intMat > 0]) * 1.1
        if max_intensity <= min_val_for_plot:
            max_intensity = min_val_for_plot * 100
        else:
            min_val_for_plot = 1e-6
            max_intensity = 1e-4  # Fallback if all else fails

    logger.debug(
        f"CVO Plotting with LogNorm: vmin={min_val_for_plot:.2e}, vmax={max_intensity:.2e}"
    )
    norm_choice = LogNorm(vmin=min_val_for_plot, vmax=max_intensity)

    pcm = ax.pcolormesh(
        X_mesh, Y_mesh, intMat, norm=norm_choice, cmap="PuBu_r", shading="auto"
    )

    if qsy_plot_loaded_shifted.size > 0:
        ax.set_xlim([np.min(qsy_plot_loaded_shifted), np.max(qsy_plot_loaded_shifted)])
        logger.debug(
            f"CVO Setting x-axis limits (xlim) to: [{np.min(qsy_plot_loaded_shifted):.2f}, {np.max(qsy_plot_loaded_shifted):.2f}] based on shifted data."
        )

        # Use q_ticks_sqw from config if available
        q_ticks_from_config = plotting_p_config.get("q_ticks_sqw")
        if q_ticks_from_config is not None:
            ax.set_xticks(q_ticks_from_config)
        else:  # Fallback if not specified in config
            q_tick_min = np.min(qsy_plot_loaded_shifted)
            q_tick_max = np.max(qsy_plot_loaded_shifted)
            start_tick = np.ceil(q_tick_min)
            end_tick = np.floor(q_tick_max)
            q_ticks_actual_range = (
                np.arange(start_tick, end_tick + 1, 1.0)
                if start_tick <= end_tick
                else np.array([q_tick_min, q_tick_max])
            )
            ax.set_xticks(q_ticks_actual_range)
    else:
        qmin_rlu_config = plotting_p_config.get("q_limits_sqw", [-5, 5])[0]
        qmax_rlu_config = plotting_p_config.get("q_limits_sqw", [-5, 5])[1]
        # Use q_ticks_sqw from config if available, else fallback
        q_ticks_from_config = plotting_p_config.get("q_ticks_sqw")
        if q_ticks_from_config is not None:
            ax.set_xticks(q_ticks_from_config)
        else:  # Fallback if not specified in config
            ax.set_xticks(np.arange(qmin_rlu_config, qmax_rlu_config + 1, 1))

    ax.set_ylim(plotting_p_config.get("energy_limits_sqw", [0, 12]))
    ax.set_xlabel(r"q$_y$ (r.l.u.)", fontsize=12)
    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    ax.set_yticks(np.arange(emin_plot, emax_plot + 1, 2.0))
    ax.set_title(plotting_p_config.get("sqw_title", "CVO S(Q,ω) Intensity Map"))
    if fig:
        fig.colorbar(pcm, ax=ax, label="S(Q,ω) (arb. units)")


# --- Main Execution ---
if __name__ == "__main__":
    st_main = default_timer()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming config.yaml is in the same directory as the script
    default_config_path = os.path.join(script_dir, "config.yaml")
    config_filename = sys.argv[1] if len(sys.argv) > 1 else default_config_path

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

    cvo_config = config.get("cvo_model", {})
    model_p_config = cvo_config.get("model_params", {})
    calc_p_config = cvo_config.get("calculation", {})
    output_p_config = cvo_config.get("output", {})
    plotting_p_config = cvo_config.get("plotting", {})
    tasks_config = cvo_config.get("tasks", {})

    S_val = model_p_config.get("S", 0.5)
    J1 = model_p_config.get("J1", 2.49)
    J2_ratio = model_p_config.get("J2_ratio", 1.12)
    J3_ratio = model_p_config.get("J3_ratio", 2.03)
    G1 = model_p_config.get("G1", 0.28)
    Dx = model_p_config.get("Dx", 2.67)
    Dy = model_p_config.get("Dy", -2.0)  # DM component for J1
    D3 = model_p_config.get("D3", 0.0)  # DM component for J3
    H_field = model_p_config.get("H", 0.0)
    spin_model_type = cvo_config.get("spin_model_type", "ha").lower()

    if spin_model_type == "hc":
        try:
            import spin_model_hc as cvo_spin_model
            logger.info("Using spin model: spin_model_hc (H || c)")
        except ImportError:
            logger.error("Could not import spin_model_hc.py. Exiting.")
            sys.exit(1)
        
        # params for hc: J1, J2, J3, G1, Dx, Dy, D3, H
        params_val = [
            J1,
            J2_ratio * J1,
            J3_ratio * J1,
            G1,
            Dx,
            Dy,
            D3,
            H_field,
        ]
    elif spin_model_type == "ha":
        try:
            import spin_model_ha as cvo_spin_model
            logger.info("Using spin model: spin_model_ha (H || a)")
        except ImportError:
            logger.error("Could not import spin_model_ha.py. Exiting.")
            sys.exit(1)

        # params for ha: J1, J2, J3, G1, Dx, H
        params_val = [
            J1,
            J2_ratio * J1,
            J3_ratio * J1,
            G1,
            Dx,
            H_field,
        ]
    else:
        logger.error(f"Unknown spin_model_type: {spin_model_type}. Use 'ha' or 'hc'.")
        sys.exit(1)

    cache_mode = calc_p_config.get("cache_mode", "r")
    cache_file_base = calc_p_config.get("cache_file_base", "CVO_model_cache")

    disp_data_file = os.path.join(
        script_dir, output_p_config.get("disp_data_filename", "CVO_disp_data.npz")
    )
    sqw_data_file = os.path.join(
        script_dir, output_p_config.get("sqw_data_filename", "CVO_sqw_data.npz")
    )

    for key in ["disp_plot_filename", "sqw_plot_filename", "combined_plot_filename"]:
        default_name = f"CVO_{key.replace('_filename','_plot')}.png"
        if key == "combined_plot_filename":
            default_name = "CVO_combined_plot.png"
        filename = plotting_p_config.get(key)
        if filename and not os.path.isabs(filename):
            plotting_p_config[key] = os.path.join(script_dir, filename)
        elif not filename:
            plotting_p_config[key] = os.path.join(script_dir, default_name)

    config["cvo_model"]["output"]["disp_data_filename"] = disp_data_file
    config["cvo_model"]["output"]["sqw_data_filename"] = sqw_data_file
    config["cvo_model"]["plotting"] = plotting_p_config

    # --- Perform Classical Minimization if H_field is non-zero (BEFORE MagCalc init) ---
    Ud_numeric_for_magcalc = None
    H_field_val_from_config = params_val[-1]  # Assuming H is the last param

    # Perform classical minimization only if:
    # 1. The magnetic field is non-zero.
    # 2. cache_mode is not 'r' (i.e., we are in a mode that might regenerate symbolic matrices).
    if abs(H_field_val_from_config) > 1e-9 and cache_mode != "r":
        if hasattr(cvo_spin_model, 'get_field_optimized_state_for_lswt'):
            logger.info(
                f"Non-zero H_field ({H_field_val_from_config}) detected. "
                "Performing classical energy minimization for CVO before MagCalc initialization."
            )
            optimal_thetas, Ud_numeric_optimized, final_classical_energy = (
                cvo_spin_model.get_field_optimized_state_for_lswt(params_val, S_val)
            )
            if Ud_numeric_optimized is not None:
                logger.info(
                    f"Classical ground state found with energy: {final_classical_energy:.6f} meV."
                )
                Ud_numeric_for_magcalc = Ud_numeric_optimized
                # Log the canting angles
                if optimal_thetas is not None:
                    canting_angles_degrees_sw = 90.0 - np.degrees(optimal_thetas)
                    logger.info(
                        f"Canting angles from a-b plane (degrees towards +z): {canting_angles_degrees_sw}"
                    )
                    logger.info(
                        f"Mean canting angle (degrees): {np.mean(canting_angles_degrees_sw):.4f}"
                    )
            else:
                logger.error(
                    "Failed to determine field-optimized classical ground state. "
                    "MagCalc will proceed with Ud derived from zero-field mpr."
                )
        else:
             logger.warning(
                f"Non-zero H_field ({H_field_val_from_config}) detected, but 'get_field_optimized_state_for_lswt' "
                "is missing in spin_model_ha. Proceeding without explicit classical minimization in this script. "
                "Ensure the model handles field correctly or that mpr is sufficient."
            )
    else:
        if abs(H_field_val_from_config) <= 1e-9:
            logger.info(
                "H_field is zero. Classical energy minimization skipped. "
                "MagCalc will use spin structure from mpr (likely zero-field or based on any existing angle cache in spin_model_hc)."
            )
        elif cache_mode == "r":
            logger.info(
                f"cache_mode is 'r'. Classical energy minimization skipped for H_field={H_field_val_from_config}. "
                "MagCalc will use spin structure from mpr (likely based on cached symbolic matrices or any existing angle cache in spin_model_hc)."
            )

    try:
        calculator = mc.MagCalc(
            spin_magnitude=S_val,
            hamiltonian_params=params_val,
            cache_file_base=cache_file_base,
            cache_mode=cache_mode,
            spin_model_module=cvo_spin_model,  # Use the field-capable CVO model
            # Ud_numeric_override is no longer needed if mpr provides field-specific rotations
            Ud_numeric_override=None,
        )
        config["cvo_model"]["nspins_for_plot"] = calculator.nspins

    except Exception as e:
        logger.error(f"Failed to initialize MagCalc for CVO: {e}", exc_info=True)
        sys.exit(1)

    # --- Process Dispersion Task ---
    disp_plotting_requested = tasks_config.get(
        "run_dispersion", False
    ) and tasks_config.get("plot_dispersion", False)
    disp_calc_requested = tasks_config.get(
        "run_dispersion", False
    ) and tasks_config.get("calculate_dispersion_new", False)

    if tasks_config.get("run_dispersion", False):
        logger.info("--- Starting CVO Dispersion Workflow ---")
        q_vectors_disp, qsy_plot_axis_disp = generate_dispersion_q_vectors_cvo(
            plotting_p_config, tasks_config
        )
        if disp_calc_requested:
            calculate_and_save_dispersion_cvo(
                calculator, q_vectors_disp, qsy_plot_axis_disp, disp_data_file
            )
        else:
            logger.info("Task: Skipping new CVO dispersion calculation.")
    else:
        logger.info("--- Skipping CVO Dispersion Workflow ---")

    # --- Process S(Q,w) Map Task ---
    sqw_plotting_requested = tasks_config.get(
        "run_sqw_map", False
    ) and tasks_config.get("plot_sqw_map", False)
    sqw_calc_requested = tasks_config.get("run_sqw_map", False) and tasks_config.get(
        "calculate_sqw_map_new", False
    )

    if tasks_config.get("run_sqw_map", False):
        logger.info("--- Starting CVO S(Q,w) Map Workflow ---")
        q_vectors_sqw, qsy_plot_axis_sqw_shifted = generate_sqw_q_vectors_cvo(
            plotting_p_config, tasks_config
        )
        if sqw_calc_requested:
            calculate_and_save_sqw_cvo(
                calculator, q_vectors_sqw, qsy_plot_axis_sqw_shifted, sqw_data_file
            )
        else:
            logger.info("Task: Skipping new CVO S(Q,w) calculation.")
    else:
        logger.info("--- Skipping CVO S(Q,w) Map Workflow ---")

    # --- Plotting Stage ---
    disp_data_exists = os.path.exists(disp_data_file)
    sqw_data_exists = os.path.exists(sqw_data_file)

    fig_to_show_save = None
    ax_disp, ax_sqw = None, None

    plot_disp = disp_plotting_requested and disp_data_exists
    plot_sqw = sqw_plotting_requested and sqw_data_exists

    # --- Determine Figsizes ---
    default_figsize_combined = (10, 13)
    default_figsize_disp_only = (8, 6)
    default_figsize_sqw_only = (10, 6)

    final_figsize_combined = default_figsize_combined
    final_figsize_disp_only = default_figsize_disp_only
    final_figsize_sqw_only = default_figsize_sqw_only

    adjust_to_screen = plotting_p_config.get("adjust_height_to_screen", False)
    screen_height_fraction = plotting_p_config.get("screen_height_fraction", 0.85)

    if adjust_to_screen:
        logger.debug("Attempting to adjust CVO plot height to screen size...")
        screen_w_in, screen_h_in, screen_dpi = get_screen_size_inches()
        if screen_h_in is not None and screen_h_in > 0:
            target_fig_height_inches = screen_h_in * screen_height_fraction

            orig_w, orig_h = default_figsize_combined
            aspect_ratio = orig_w / orig_h
            final_figsize_combined = (
                target_fig_height_inches * aspect_ratio,
                target_fig_height_inches,
            )
            logger.debug(
                f"Adjusted CVO combined figsize to: ({final_figsize_combined[0]:.2f}, {final_figsize_combined[1]:.2f}) inches."
            )

            orig_w, orig_h = default_figsize_disp_only
            aspect_ratio = orig_w / orig_h
            final_figsize_disp_only = (
                target_fig_height_inches
                * aspect_ratio
                * (orig_h / default_figsize_combined[1]),
                target_fig_height_inches * (orig_h / default_figsize_combined[1]),
            )
            logger.debug(
                f"Adjusted CVO dispersion-only figsize to: ({final_figsize_disp_only[0]:.2f}, {final_figsize_disp_only[1]:.2f}) inches."
            )

            orig_w, orig_h = default_figsize_sqw_only
            aspect_ratio = orig_w / orig_h
            final_figsize_sqw_only = (
                target_fig_height_inches
                * aspect_ratio
                * (orig_h / default_figsize_combined[1]),
                target_fig_height_inches * (orig_h / default_figsize_combined[1]),
            )
            logger.debug(
                f"Adjusted CVO S(Q,w)-only figsize to: ({final_figsize_sqw_only[0]:.2f}, {final_figsize_sqw_only[1]:.2f}) inches."
            )
        else:
            logger.warning(
                "Failed to get screen height for CVO plot, using default figsizes."
            )

    if plot_disp and plot_sqw:
        logger.info("Plotting both CVO Dispersion and S(Q,w) map.")
        fig_to_show_save, (ax_disp, ax_sqw) = plt.subplots(
            2, 1, figsize=final_figsize_combined, sharex=False
        )
        plot_dispersion_from_file_cvo(disp_data_file, config, ax_disp)
        plot_sqw_map_from_file_cvo(sqw_data_file, config, ax_sqw, fig_to_show_save)
        fig_to_show_save.suptitle("CVO Model Calculations", fontsize=16)
        fig_to_show_save.tight_layout(rect=[0, 0.03, 1, 0.95])
    elif plot_disp:
        logger.info("Plotting CVO Dispersion only.")
        fig_to_show_save, ax_disp = plt.subplots(1, 1, figsize=final_figsize_disp_only)
        plot_dispersion_from_file_cvo(disp_data_file, config, ax_disp)
        fig_to_show_save.tight_layout()
    elif plot_sqw:
        logger.info("Plotting CVO S(Q,w) map only.")
        fig_to_show_save, ax_sqw = plt.subplots(1, 1, figsize=final_figsize_sqw_only)
        plot_sqw_map_from_file_cvo(sqw_data_file, config, ax_sqw, fig_to_show_save)
        fig_to_show_save.tight_layout()

    if fig_to_show_save:
        if plotting_p_config.get("save_plot"):
            plot_filepath = None
            if plot_disp and plot_sqw:
                plot_filepath = plotting_p_config.get("combined_plot_filename")
            elif plot_disp:
                plot_filepath = plotting_p_config.get("disp_plot_filename")
            elif plot_sqw:
                plot_filepath = plotting_p_config.get("sqw_plot_filename")

            if plot_filepath:
                plt.savefig(plot_filepath)
                logger.info(f"CVO plot saved to {plot_filepath}")
            else:
                logger.warning("CVO plot filename not determined for saving.")

        if plotting_p_config.get("show_plot"):
            plt.show()
        plt.close(fig_to_show_save)

    if disp_plotting_requested and not disp_data_exists:
        logger.warning(
            f"CVO dispersion data not available from {disp_data_file}. Cannot plot dispersion."
        )
    if sqw_plotting_requested and not sqw_data_exists:
        logger.warning(
            f"CVO S(Q,w) data not available from {sqw_data_file}. Cannot plot S(Q,w) map."
        )

    et_main = default_timer()
    logger.info(
        f"Total run-time for CVO script: {np.round((et_main - st_main), 2)} sec ({np.round((et_main - st_main) / 60, 2)} min)."
    )
