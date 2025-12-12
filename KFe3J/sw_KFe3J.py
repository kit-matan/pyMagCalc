#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018 (disp_KFe3J.py)
Created on Mon Aug 13 01:22:57 2018 (EQmap_KFe3J.py)

Combined script by Gemini Code Assist

Calculates and plots spin-wave dispersion and S(Q,omega) intensity map
for KFe3(OH)6(SO4)2 using MagCalc.
Reads parameters from config.yaml.
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Original import that causes issues when sw_CVO.py is run directly
# import magcalc as mc

import sys
import os

# Adjust sys.path to correctly locate the pyMagCalc package
# Get the directory of the current script (aCVO)
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the project root directory (parent of pyMagCalc)
project_root_dir = os.path.dirname(os.path.dirname(current_script_dir))

if project_root_dir not in sys.path:
    sys.path.insert(0, project_root_dir)

from pyMagCalc import magcalc as mc

import spin_model as kfe3j_spin_model
# try:
#     import auto_KFe3J as kfe3j_spin_model
# except ImportError:
#     sys.path.append(os.path.dirname(__file__))
#     import auto_KFe3J as kfe3j_spin_model
import yaml
import logging
import tkinter as tk  # For screen size detection


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

try:
    from generic_model import GenericSpinModel
except ImportError:
    # Add parent dir to path to find generic_model if needed
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
    from generic_model import GenericSpinModel
logger = logging.getLogger(__name__)


def get_screen_size_inches():
    """Gets screen size in inches and screen DPI."""
    try:
        root = tk.Tk()
        root.withdraw()  # Hide the main Tkinter window
        screen_width_pixels = root.winfo_screenwidth()
        screen_height_pixels = root.winfo_screenheight()

        # Get DPI. winfo_fpixels('1i') returns pixels per inch.
        # This might vary by axis, take x-axis DPI or average if necessary.
        # For simplicity, assuming square pixels and using x-axis DPI.
        dpi = root.winfo_fpixels("1i")
        root.destroy()

        if dpi <= 0:  # Fallback DPI if detection fails
            logger.warning(
                "Could not determine screen DPI accurately, using default 100."
            )
            dpi = 100.0
        else:
            logger.info(f"Detected screen DPI: {dpi:.2f}")

        screen_width_inches = screen_width_pixels / dpi
        screen_height_inches = screen_height_pixels / dpi
        logger.info(
            f"Detected screen size: {screen_width_pixels}x{screen_height_pixels} pixels; {screen_width_inches:.2f}x{screen_height_inches:.2f} inches."
        )
        return screen_width_inches, screen_height_inches, dpi
    except Exception as e:
        logger.warning(
            f"Could not get screen size using tkinter: {e}. Plot will use default figsize."
        )
        return None, None, None


# --- Q-point generation (consistent with original scripts) ---
# These are specific to the KFe3J plots and kept as per original scripts
# If more general q-paths are needed, config['q_path'] could be used with a new function.
def generate_dispersion_q_vectors():
    """Generates q-vectors for dispersion plot as in disp_KFe3J.py."""
    intv = 0.05
    qsx_disp = np.arange(0 - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv)
    qsy_disp = np.arange(0 - intv / 2, 2 * np.pi + intv / 2, intv)
    q_list_disp = []
    for qx_val in qsx_disp:
        q_list_disp.append(np.array([qx_val, 0, 0]))
    for qy_val in qsy_disp:
        q_list_disp.append(np.array([0, qy_val, 0]))
    return np.array(q_list_disp), qsx_disp, qsy_disp


def generate_sqw_q_vectors():
    """Generates q-vectors for S(Q,w) map as in EQmap_KFe3J.py."""
    intv = 0.05
    qsx_sqw = np.arange(
        -np.pi / np.sqrt(3) - intv / 2, 2 * np.pi / np.sqrt(3) + intv / 2, intv
    )
    qsy_sqw = np.arange(-np.pi - intv / 2, 2 * np.pi + intv / 2, intv)
    q_list_sqw = []
    for qx_val in qsx_sqw:
        q_list_sqw.append(np.array([qx_val, 0, 0]))
    for qy_val in qsy_sqw:
        q_list_sqw.append(np.array([0, qy_val, 0]))
    return np.array(q_list_sqw), qsx_sqw, qsy_sqw


# --- Dispersion Calculation and Plotting ---
def calculate_and_save_dispersion(
    calculator: mc.MagCalc, q_vectors_array, output_filename
):
    """Calculates spin-wave dispersion and saves results."""
    logger.info("Calculating dispersion...")
    try:
        En = calculator.calculate_dispersion(q_vectors_array)
        if En is not None:
            logger.info(f"Saving dispersion results to {output_filename}...")
            results_to_save = {"q_vectors": q_vectors_array, "energies": En}
            # Assuming MagCalc.save_results can handle this dictionary structure
            # If not, use np.savez_compressed directly:
            # np.savez_compressed(output_filename, **results_to_save)
            calculator.save_results(
                output_filename, results_to_save
            )  # Or use np.savez_compressed
            logger.info("Dispersion results saved successfully.")
            return True
        else:
            logger.error("Dispersion calculation returned None, cannot save results.")
            return False
    except Exception as e:
        logger.error(
            f"Error during dispersion calculation or saving: {e}", exc_info=True
        )
        return False


def plot_dispersion_from_file(filename, config, ax):
    """Loads dispersion data from a .npz file and plots it."""
    logger.info(f"Loading dispersion data from {filename} for plotting...")
    try:
        data = np.load(filename, allow_pickle=True)
        q_vectors_array = data["q_vectors"]
        En = data["energies"]
        logger.info("Dispersion data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Dispersion data file not found: {filename}. Cannot plot.")
        return
    except Exception as e:
        logger.error(
            f"Error loading dispersion data from {filename}: {e}", exc_info=True
        )
        return

    # Regenerate qsx_disp, qsy_disp lengths for plotting logic
    # This assumes the q-path structure from generate_dispersion_q_vectors
    # A more robust way would be to save len_qsx_disp and len_qsy_disp if q-gen changes
    split_index = -1
    for i in range(1, len(q_vectors_array)):
        if (
            abs(q_vectors_array[i, 1]) > 1e-9 and abs(q_vectors_array[i - 1, 1]) < 1e-9
        ) or (
            abs(q_vectors_array[i, 0]) < 1e-9
            and abs(q_vectors_array[i - 1, 0]) > 1e-9
            and abs(q_vectors_array[i - 1, 1]) < 1e-9
        ):
            if (
                abs(q_vectors_array[i - 1, 1]) < 1e-9
            ):  # Check if previous was on qx path
                split_index = i
                break

    if split_index == -1:  # Fallback if specific q-path structure changes
        # This fallback assumes qx path comes first, then qy.
        # Find first q-vector with y != 0, assuming it's the start of qy path
        non_zero_y_indices = np.where(np.abs(q_vectors_array[:, 1]) > 1e-9)[0]
        if len(non_zero_y_indices) > 0 and non_zero_y_indices[0] > 0:
            # Check if the point before it had y=0 and x!=0 (or x=0 if it's Gamma start)
            prev_idx = non_zero_y_indices[0] - 1
            if abs(q_vectors_array[prev_idx, 1]) < 1e-9:
                split_index = non_zero_y_indices[0]
            else:  # Could be that qx path was all zeros (e.g. single Gamma point)
                split_index = 0  # effectively all qy path or just qx path
        else:  # All qx path or all qy path (if qx part was empty)
            if np.all(np.abs(q_vectors_array[:, 1]) < 1e-9):  # All qx
                split_index = len(q_vectors_array)
            else:  # All qy or mixed in a way not caught
                split_index = len(q_vectors_array) // 2
                logger.warning(
                    f"Could not reliably determine split for dispersion plot. Using fallback split_index: {split_index}"
                )

    qsx_disp_loaded = q_vectors_array[:split_index, 0]
    qsy_disp_loaded = q_vectors_array[split_index:, 1]
    len_qsx_disp = len(qsx_disp_loaded)
    len_qsy_disp = len(qsy_disp_loaded)

    if len(En) != len(q_vectors_array):
        logger.error(f"Mismatch in loaded dispersion data. Cannot plot.")
        return

    Ekx1 = [
        En[i][0] if En[i] is not None and len(En[i]) > 0 else np.nan
        for i in range(len_qsx_disp)
    ]
    Ekx2 = [
        En[i][1] if En[i] is not None and len(En[i]) > 1 else np.nan
        for i in range(len_qsx_disp)
    ]
    Ekx3 = [
        En[i][2] if En[i] is not None and len(En[i]) > 2 else np.nan
        for i in range(len_qsx_disp)
    ]
    Eky1 = [
        (
            En[len_qsx_disp + i][0]
            if En[len_qsx_disp + i] is not None and len(En[len_qsx_disp + i]) > 0
            else np.nan
        )
        for i in range(len_qsy_disp)
    ]
    Eky2 = [
        (
            En[len_qsx_disp + i][1]
            if En[len_qsx_disp + i] is not None and len(En[len_qsx_disp + i]) > 1
            else np.nan
        )
        for i in range(len_qsy_disp)
    ]
    Eky3 = [
        (
            En[len_qsx_disp + i][2]
            if En[len_qsx_disp + i] is not None and len(En[len_qsx_disp + i]) > 2
            else np.nan
        )
        for i in range(len_qsy_disp)
    ]

    # X-axis transformation for the second segment (M-K-Gamma') to match disp_KFe3J.py
    # This plots the qsy segment reversed and shifted.
    qsyn_disp_plot = qsx_disp_loaded[-1] + (qsy_disp_loaded[-1] - qsy_disp_loaded)

    ax.plot(
        qsx_disp_loaded,
        Ekx1,
        "r-",
        qsx_disp_loaded,
        Ekx2,
        "g-",
        qsx_disp_loaded,
        Ekx3,
        "b-",
        qsyn_disp_plot,
        Eky1,
        "r-",
        qsyn_disp_plot,
        Eky2,
        "g-",
        qsyn_disp_plot,
        Eky3,
        "b-",
        marker=".",
        markersize=2,
    )

    # Fixed x-coordinates for high-symmetry points based on disp_KFe3J.py
    X_M_disp = 2 * np.pi / np.sqrt(3)
    Y_MAX_disp = 2 * np.pi

    # Vertical line at M
    ax.plot(
        [X_M_disp, X_M_disp],
        [-1, config["plotting"]["energy_limits_disp"][1] + 5],
        "k:",
    )
    # Vertical line at K (midpoint of the plotted qsyn_disp_plot segment, as in disp_KFe3J.py)
    # Ensure qsyn_disp_plot is not empty before accessing its middle element
    if len(qsyn_disp_plot) > 0:
        k_line_coord = qsyn_disp_plot[len(qsyn_disp_plot) // 2]
        ax.plot(
            [k_line_coord, k_line_coord],
            [-1, config["plotting"]["energy_limits_disp"][1] + 5],
            "k:",
        )

    ax.set_xlim([0, X_M_disp + Y_MAX_disp])  # Matches disp_KFe3J.py
    ax.set_ylim(config["plotting"]["energy_limits_disp"])
    ax.set_xticks([])
    ax.text(0, -1, r"$\Gamma$", fontsize=12)
    ax.text(X_M_disp - 0.1, -1, "M", fontsize=12)
    ax.text(
        X_M_disp + Y_MAX_disp - (4 * np.pi / 3.0) - 0.1, -1, "K", fontsize=12
    )  # K label from disp_KFe3J.py
    ax.text(X_M_disp + Y_MAX_disp - 0.1, -1, r"$\Gamma$", fontsize=12)  # Gamma' label

    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    ax.set_yticks(
        np.arange(
            config["plotting"]["energy_limits_disp"][0],
            config["plotting"]["energy_limits_disp"][1] + 1,
            5.0,
        )
    )
    ax.set_title(config["plotting"]["disp_title"])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    # save/show/close handled by the main caller


# --- S(Q,w) Map Calculation and Plotting ---
def calculate_and_save_sqw(calculator: mc.MagCalc, q_vectors_array, output_filename):
    """Calculates S(Q,w) and saves results."""
    logger.info("Calculating S(Q,w)...")
    try:
        qout, En, Sqwout = calculator.calculate_sqw(q_vectors_array)
        if En is not None and Sqwout is not None:
            logger.info(f"Saving S(Q,w) results to {output_filename}...")
            # qout should be same as q_vectors_array, save it for completeness
            np.savez_compressed(
                output_filename, q_vectors=qout, energies=En, sqw_values=Sqwout
            )
            logger.info("S(Q,w) results saved successfully.")
            return True
        else:
            logger.error(
                "S(Q,w) calculation returned None for En or Sqwout, cannot save results."
            )
            return False
    except Exception as e:
        logger.error(f"Error during S(Q,w) calculation or saving: {e}", exc_info=True)
        return False


def plot_sqw_map_from_file(filename, config, qsx_sqw_orig, qsy_sqw_orig, ax, fig):
    """Loads S(Q,w) data from a .npz file and plots the intensity map on given axes."""
    logger.info(f"Loading S(Q,w) data from {filename} for plotting...")
    try:
        data = np.load(filename, allow_pickle=True)
        # q_vectors = data["q_vectors"] # q_vectors loaded but qsx_sqw_orig, qsy_sqw_orig used for shaping
        En_loaded = data["energies"]
        Sqwout_loaded = data["sqw_values"]
        logger.info("S(Q,w) data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"S(Q,w) data file not found: {filename}. Cannot plot.")
        return
    except Exception as e:
        logger.error(f"Error loading S(Q,w) data from {filename}: {e}", exc_info=True)
        return

    len_qsx_sqw = len(qsx_sqw_orig)
    # En_loaded and Sqwout_loaded are expected to be flat arrays of arrays/lists
    En_kx = En_loaded[:len_qsx_sqw]
    En_ky = En_loaded[len_qsx_sqw:]
    Sqwout_kx = Sqwout_loaded[:len_qsx_sqw]
    Sqwout_ky = Sqwout_loaded[len_qsx_sqw:]

    Ex_plot = np.arange(
        0, config["plotting"]["energy_limits_sqw"][1] + 0.05, 0.05
    )  # Ensure range covers limit
    wid = 0.2  # Resolution width, could be configurable

    intMat_kx = np.zeros((len(Ex_plot), len(qsx_sqw_orig)))
    for i in range(len(Ex_plot)):
        for j in range(len(qsx_sqw_orig)):
            fint_kx_val = 0
            if En_kx[j] is not None and Sqwout_kx[j] is not None:
                for band in range(len(En_kx[j])):  # Number of bands
                    fint_kx_val += (
                        Sqwout_kx[j][band]
                        * 1.0
                        / np.pi
                        * wid
                        / 2
                        / ((Ex_plot[i] - En_kx[j][band]) ** 2 + (wid / 2) ** 2)
                    )
            intMat_kx[i, j] = fint_kx_val

    intMat_ky = np.zeros((len(Ex_plot), len(qsy_sqw_orig)))
    for i in range(len(Ex_plot)):
        for j in range(len(qsy_sqw_orig)):
            fint_ky_val = 0
            if En_ky[j] is not None and Sqwout_ky[j] is not None:
                for band in range(len(En_ky[j])):
                    fint_ky_val += (
                        Sqwout_ky[j][band]
                        * 1.0
                        / np.pi
                        * wid
                        / 2
                        / ((Ex_plot[i] - En_ky[j][band]) ** 2 + (wid / 2) ** 2)
                    )
            intMat_ky[i, j] = fint_ky_val

    # Q-axis transformation for plotting (as in EQmap_KFe3J.py)
    # This specific transformation creates the Gamma-M ... K-Gamma path view
    qsy_transformed = qsx_sqw_orig[-1] + (
        qsy_sqw_orig[-1] - qsy_sqw_orig
    )  # Shift and flip
    qsy_transformed = np.flip(qsy_transformed, 0)

    qs_plot = np.concatenate((qsx_sqw_orig, qsy_transformed))
    intMat_ky_flipped = np.flip(intMat_ky, 1)
    intMat_plot = np.concatenate([intMat_kx, intMat_ky_flipped], axis=-1)

    # Sort if necessary, though concatenation order should be correct
    sort_index_qs = np.argsort(qs_plot)
    qs_plot_sorted = qs_plot[sort_index_qs]
    intMat_plot_sorted = intMat_plot[:, sort_index_qs]

    # Ensure Ex_plot is sorted (it should be by arange)
    # intMat_plot_sorted = intMat_plot_sorted[np.argsort(Ex_plot), :] # Ex_plot is already sorted

    # Guard against all-zero intMat for LogNorm
    min_val = (
        np.min(intMat_plot_sorted[intMat_plot_sorted > 0])
        if np.any(intMat_plot_sorted > 0)
        else 1e-5
    )
    max_val = np.max(intMat_plot_sorted) if np.any(intMat_plot_sorted > 0) else 1.0
    if max_val <= min_val:
        max_val = min_val + 1.0

    pcm = ax.pcolormesh(
        qs_plot_sorted,
        Ex_plot,
        intMat_plot_sorted,
        norm=LogNorm(vmin=min_val, vmax=max_val),
        cmap="PuBu_r",
        shading="auto",
    )

    # X-axis limits and labels based on the transformed q-axis
    ax.set_xlim(
        [qsx_sqw_orig[0], qsy_transformed[-1]]
    )  # Full range of the constructed path
    ax.set_ylim(config["plotting"]["energy_limits_sqw"])
    ax.set_xticks([])

    # Text labels for high-symmetry points (approximate positions on the transformed axis)
    # These are illustrative and depend on the specific q-ranges and transformations
    ax.text(qsx_sqw_orig[0], -1, r"$\Gamma$", fontsize=12)  # Start of qsx path
    # M point is at the end of qsx path
    m_point_x = qsx_sqw_orig[-1]
    ax.text(m_point_x - 0.1, -1, "M", fontsize=12)
    # K point is somewhere in the middle of the transformed qsy part
    k_point_x = (
        m_point_x + qsy_transformed[len(qsy_transformed) // 2]
    ) / 1.7  # Heuristic
    ax.text(k_point_x, -1, "K", fontsize=12)
    ax.text(
        qsy_transformed[-1] - 0.1, -1, r"$\Gamma$", fontsize=12
    )  # End of transformed qsy path

    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
    ax.set_yticks(
        np.arange(
            config["plotting"]["energy_limits_sqw"][0],
            config["plotting"]["energy_limits_sqw"][1] + 1,
            5.0,
        )
    )
    ax.set_title(config["plotting"]["sqw_title"])
    if fig:  # Add colorbar to the figure if fig is provided
        fig.colorbar(pcm, ax=ax, label="S(Q,ω) (arb. units)")
    # save/show/close handled by the main caller


# --- Main Execution ---
def main():
    st_main = default_timer()

    # --- Load Configuration ---
    # Assuming config.yaml is in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Point to the configuration file (YAML)
    config_file_path = os.path.join(script_dir, "config.yaml")

    # Allow overriding config path via command line argument
    config_filename = sys.argv[1] if len(sys.argv) > 1 else config_file_path

    config = {}
    try:
        with open(config_filename, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_filename}")
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_filename}. Exiting.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing configuration file {config_filename}: {e}. Exiting."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}. Exiting.")
        sys.exit(1)

    # Extract parameters
    model_p_config = config.get("model_params", {})
    calc_p_config = config.get("calculation", {})
    output_p_config = config.get("output", {})
    plotting_p_config = config.get("plotting", {})
    tasks_config = config.get("tasks", {})

    S_val = model_p_config.get("S", 2.5)
    # Updated for Auto-Gen Model (J1-J6, H)
    # params_val = [
    #     model_p_config.get("J1", 0.0),
    # ...
    # ]
    
    # --- DECLARATIVE MODEL LOGIC ---
    interactions_config = config.get("interactions")
    spin_model_to_use = kfe3j_spin_model # Default fallback
    
    
    if interactions_config:
        logger.info("Declarative 'interactions' found in config. Using GenericSpinModel.")
        spin_model_to_use = GenericSpinModel(config, base_path=script_dir)
        
        # Build params_val
        params_val = []
        
        # Check if 'parameters' list defines explicit order and naming
        param_names = config.get('parameters')
        if param_names:
            logger.info(f"Using explicit parameter list from config: {param_names}")
            for name in param_names:
                # Look up value in model_params
                val = model_p_config.get(name, 0.0)
                params_val.append(float(val))
        else:
            # Fallback to old inference method (discouraged for complex manual models)
            for interaction in interactions_config:
                itype = interaction.get('type')
                val = interaction.get('value')
                # ... legacy inferrence logic (omitted for brevity, assume 'parameters' exists)
                if itype == 'heisenberg':
                     params_val.append(float(val))
                # ...
            # Append H
            H_val = model_p_config.get("H", 0.0)
            params_val.append(H_val)

        logger.info(f"Constructed parameters for GenericModel: {params_val}")
        
    else:
        # Fallback to Manual Model logic
        # Ensure params_val are in the correct order expected by the spin_model
        # (J1, J2, Dy, Dz, H) - this order is assumed by original scripts
        params_val = [
            model_p_config.get("J1", 0.0),
            model_p_config.get("J2", 0.0),
            model_p_config.get("Dy", 0.0),
            model_p_config.get("Dz", 0.0),
            model_p_config.get("H", 0.0),
        ]
    cache_mode = calc_p_config.get("cache_mode", "auto")
    cache_file_base = calc_p_config.get("cache_file_base", "KFe3J_model_cache")

    disp_data_file = os.path.join(
        script_dir, output_p_config.get("disp_data_filename", "KFe3J_disp_data.npz")
    )
    sqw_data_file = os.path.join(
        script_dir, output_p_config.get("sqw_data_filename", "KFe3J_sqw_data.npz")
    )

    # Update plotting filenames to be relative to script dir if not absolute
    for key in ["disp_plot_filename", "sqw_plot_filename", "combined_plot_filename"]:
        default_name = f"KFe3J_{key.replace('_filename','_plot')}.png"
        if key == "combined_plot_filename":
            default_name = "KFe3J_combined_plot.png"

        filename = plotting_p_config.get(key)
        if filename and not os.path.isabs(filename):
            plotting_p_config[key] = os.path.join(script_dir, filename)
        elif not filename:  # if key is missing or None/empty
            plotting_p_config[key] = os.path.join(script_dir, default_name)

    config["output"][
        "disp_data_filename"
    ] = disp_data_file  # Store full path back for functions
    config["output"]["sqw_data_filename"] = sqw_data_file
    config["plotting"] = plotting_p_config  # Store updated plotting paths

    # --- Initialize MagCalc ---
    logger.info(
        f"Initializing MagCalc with cache base: {cache_file_base}, mode: {cache_mode}"
    )
    try:
        calculator = mc.MagCalc(
            spin_magnitude=S_val,
            hamiltonian_params=params_val,
            cache_file_base=cache_file_base,  # This is for MagCalc's internal cache
            cache_mode=cache_mode,
            spin_model_module=spin_model_to_use,
        )
    except Exception as e:
        logger.error(f"Failed to initialize MagCalc: {e}", exc_info=True)
        sys.exit(1)

    # --- Process Dispersion Task ---
    if tasks_config.get("run_dispersion", False):
        logger.info("--- Starting Dispersion Workflow ---")
        q_vectors_disp, qsx_disp, qsy_disp = generate_dispersion_q_vectors()
        if tasks_config.get("calculate_dispersion_new", False):
            logger.info("Task: Calculate new dispersion data.")
            calc_success = calculate_and_save_dispersion(
                calculator, q_vectors_disp, disp_data_file
            )
            if not calc_success:
                logger.warning(
                    "Dispersion calculation/saving failed. Plotting might not be possible from new data."
                )
        else:
            logger.info(
                "Task: Skipping new dispersion calculation. Will attempt to load existing data for plotting."
            )
        # Plotting for dispersion is handled later if requested
    else:
        logger.info("--- Skipping Dispersion Workflow ---")

    # --- Process S(Q,w) Map Task ---
    if tasks_config.get("run_sqw_map", False):
        logger.info("--- Starting S(Q,w) Map Workflow ---")
        q_vectors_sqw, qsx_sqw, qsy_sqw = generate_sqw_q_vectors()
        if tasks_config.get("calculate_sqw_map_new", False):
            logger.info("Task: Calculate new S(Q,w) data.")
            calc_success_sqw = calculate_and_save_sqw(
                calculator, q_vectors_sqw, sqw_data_file
            )
            if not calc_success_sqw:
                logger.warning(
                    "S(Q,w) calculation/saving failed. Plotting might not be possible from new data."
                )
        else:
            logger.info(
                "Task: Skipping new S(Q,w) calculation. Will attempt to load existing data for plotting."
            )
        # Plotting for S(Q,w) is handled later if requested
    else:
        logger.info("--- Skipping S(Q,w) Map Workflow ---")

    # --- Plotting Stage ---
    disp_plotting_requested = tasks_config.get(
        "run_dispersion", False
    ) and tasks_config.get("plot_dispersion", False)
    sqw_plotting_requested = tasks_config.get(
        "run_sqw_map", False
    ) and tasks_config.get("plot_sqw_map", False)

    disp_data_available = os.path.exists(disp_data_file) or (
        disp_plotting_requested and tasks_config.get("calculate_dispersion_new", False)
    )
    sqw_data_available = os.path.exists(sqw_data_file) or (
        sqw_plotting_requested and tasks_config.get("calculate_sqw_map_new", False)
    )

    fig_to_show_save = None
    ax_disp, ax_sqw = None, None

    # --- Determine Figsizes ---
    # Default figsizes from the script's original hardcoded values
    default_figsize_combined = (10, 13)  # width, height
    default_figsize_disp_only = (8, 6)
    default_figsize_sqw_only = (10, 6)

    # Initialize final figsizes with defaults
    final_figsize_combined = default_figsize_combined
    final_figsize_disp_only = default_figsize_disp_only
    final_figsize_sqw_only = default_figsize_sqw_only

    adjust_to_screen = plotting_p_config.get("adjust_height_to_screen", False)
    screen_height_fraction = plotting_p_config.get("screen_height_fraction", 0.85)

    if adjust_to_screen:
        logger.info("Attempting to adjust plot height to screen size...")
        screen_w_in, screen_h_in, screen_dpi = get_screen_size_inches()
        if screen_h_in is not None and screen_h_in > 0:
            target_fig_height_inches = screen_h_in * screen_height_fraction

            # Adjust combined plot
            orig_w, orig_h = default_figsize_combined
            aspect_ratio = orig_w / orig_h
            final_figsize_combined = (
                target_fig_height_inches * aspect_ratio,
                target_fig_height_inches,
            )
            logger.info(
                f"Adjusted combined figsize to: ({final_figsize_combined[0]:.2f}, {final_figsize_combined[1]:.2f}) inches."
            )

            # Adjust dispersion only plot
            orig_w, orig_h = default_figsize_disp_only
            aspect_ratio = orig_w / orig_h
            final_figsize_disp_only = (
                target_fig_height_inches
                * aspect_ratio
                * (orig_h / default_figsize_combined[1]),
                target_fig_height_inches * (orig_h / default_figsize_combined[1]),
            )  # Scale a bit for single plots
            logger.info(
                f"Adjusted dispersion-only figsize to: ({final_figsize_disp_only[0]:.2f}, {final_figsize_disp_only[1]:.2f}) inches."
            )

            # Adjust S(Q,w) only plot
            orig_w, orig_h = default_figsize_sqw_only
            aspect_ratio = orig_w / orig_h
            final_figsize_sqw_only = (
                target_fig_height_inches
                * aspect_ratio
                * (orig_h / default_figsize_combined[1]),
                target_fig_height_inches * (orig_h / default_figsize_combined[1]),
            )
            logger.info(
                f"Adjusted S(Q,w)-only figsize to: ({final_figsize_sqw_only[0]:.2f}, {final_figsize_sqw_only[1]:.2f}) inches."
            )
        else:
            logger.warning("Failed to get screen height, using default figsizes.")

    if (
        disp_plotting_requested
        and disp_data_available
        and sqw_plotting_requested
        and sqw_data_available
    ):
        logger.info("Plotting both Dispersion and S(Q,w) map on subplots.")
        fig_to_show_save, (ax_disp, ax_sqw) = plt.subplots(
            2, 1, figsize=final_figsize_combined, sharex=False
        )
        plot_dispersion_from_file(disp_data_file, config, ax_disp)

        # qsx_sqw, qsy_sqw are needed for plot_sqw_map_from_file
        # They are generated if run_sqw_map is true, even if calculate_sqw_map_new is false.
        # If run_sqw_map was false, but we somehow want to plot existing sqw data, we need to generate them.
        # However, sqw_plotting_requested already checks run_sqw_map.
        _, qsx_sqw_for_plot, qsy_sqw_for_plot = (
            generate_sqw_q_vectors()
        )  # Generate for plotting context
        plot_sqw_map_from_file(
            sqw_data_file,
            config,
            qsx_sqw_for_plot,
            qsy_sqw_for_plot,
            ax_sqw,
            fig_to_show_save,
        )

        fig_to_show_save.suptitle(
            "KFe$_3$(OH)$_6$(SO$_4$)$_2$ Calculations", fontsize=16
        )
        fig_to_show_save.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust for suptitle and colorbar

    elif disp_plotting_requested and disp_data_available:
        logger.info("Plotting Dispersion only.")
        fig_to_show_save, ax_disp = plt.subplots(1, 1, figsize=final_figsize_disp_only)
        plot_dispersion_from_file(disp_data_file, config, ax_disp)
        fig_to_show_save.tight_layout()

    elif sqw_plotting_requested and sqw_data_available:
        logger.info("Plotting S(Q,w) map only.")
        fig_to_show_save, ax_sqw = plt.subplots(1, 1, figsize=final_figsize_sqw_only)
        _, qsx_sqw_for_plot, qsy_sqw_for_plot = generate_sqw_q_vectors()
        plot_sqw_map_from_file(
            sqw_data_file,
            config,
            qsx_sqw_for_plot,
            qsy_sqw_for_plot,
            ax_sqw,
            fig_to_show_save,
        )
        fig_to_show_save.tight_layout()

    # Save and/or show the figure if one was created
    if fig_to_show_save:
        if plotting_p_config.get("save_plot"):
            plot_filepath = None
            if (
                disp_plotting_requested
                and disp_data_available
                and sqw_plotting_requested
                and sqw_data_available
            ):
                plot_filepath = plotting_p_config.get("combined_plot_filename")
            elif disp_plotting_requested and disp_data_available:
                plot_filepath = plotting_p_config.get("disp_plot_filename")
            elif sqw_plotting_requested and sqw_data_available:
                plot_filepath = plotting_p_config.get("sqw_plot_filename")

            if plot_filepath:  # Path should be absolute from config loading
                plt.savefig(plot_filepath)
                logger.info(f"Plot saved to {plot_filepath}")
            else:
                logger.warning("Plot filename not determined for saving.")

        if plotting_p_config.get("show_plot"):
            plt.show()
        plt.close(fig_to_show_save)  # Close the figure

    # Log warnings if plotting was requested but data was not available
    if disp_plotting_requested and not disp_data_available:
        logger.warning(
            f"Dispersion data not available from {disp_data_file}. Cannot plot dispersion."
        )
    if sqw_plotting_requested and not sqw_data_available:
        logger.warning(
            f"S(Q,w) data not available from {sqw_data_file}. Cannot plot S(Q,w) map."
        )

    et_main = default_timer()
    logger.info(
        f"Total run-time: {np.round((et_main - st_main), 2)} sec ({np.round((et_main - st_main) / 60, 2)} min)."
    )


if __name__ == "__main__":
    main()
