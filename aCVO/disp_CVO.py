#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:18:18 2018

@author: Kit Matan

Calculate and plot the spin-wave dispersion for a CVO-like material.
(Updated to use MagCalc class and load configuration from YAML)
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
import magcalc as mc
import yaml
import sys
import os

# Assume a spin model module for CVO exists
# You will need to create spin_model_cvo.py if it doesn't exist
# and ensure it's in your Python path or the same directory as this script.
try:
    import spin_model_cvo
except ImportError:
    print("Error: Could not import spin_model_cvo.py.")
    print("Please ensure it exists and is in your Python path or the aCVO directory.")
    sys.exit(1)

import logging

if __name__ == "__main__":
    st = default_timer()

    # --- Configure logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    # --- End logging config ---

    # --- Load Configuration ---
    # Assuming config.yaml is in the same directory as the aCVO directory
    # Corrected path: config.yaml should be in the parent of the parent of the script_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Config file is in the same directory as the script
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
    except yaml.YAMLError as e:
        logger.error(
            f"Error parsing configuration file {config_filename}: {e}. Exiting."
        )
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error loading configuration: {e}. Exiting.")
        sys.exit(1)

    # --- Extract CVO specific configuration ---
    cvo_config = config.get("cvo_model", {})
    if not cvo_config:
        logger.error("Section 'cvo_model' not found in config.yaml. Exiting.")
        sys.exit(1)

    model_p_config = cvo_config.get("model_params", {})
    calc_p_config = cvo_config.get("calculation", {})
    output_p_config = cvo_config.get("output", {})
    plotting_p_config = cvo_config.get("plotting", {})
    tasks_config = cvo_config.get("tasks", {})

    S_val = model_p_config.get("S", 0.5)
    # Reconstruct parameters list as in the original script
    J1 = model_p_config.get("J1", 2.49)
    J2_ratio = model_p_config.get("J2_ratio", 1.12)
    J3_ratio = model_p_config.get("J3_ratio", 2.03)
    G1 = model_p_config.get("G1", 0.28)
    Dx = model_p_config.get("Dx", 2.67)
    H_field = model_p_config.get("H", 0.0)  # Renamed to avoid conflict with H matrix
    params_val = [J1, J2_ratio * J1, J3_ratio * J1, G1, Dx, H_field]

    cache_mode = calc_p_config.get("cache_mode", "r")
    cache_file_base = calc_p_config.get("cache_file_base", "CVO_model_cache")
    disp_data_file = os.path.join(
        script_dir, output_p_config.get("disp_data_filename", "CVO_disp_data.npz")
    )

    # Update plotting filename to be relative to script dir if not absolute
    plot_fn_key = "disp_plot_filename"
    default_plot_fn = "CVO_disp_plot.png"
    # Ensure plotting_p_config[plot_fn_key] exists or use default
    current_plot_fn = plotting_p_config.get(plot_fn_key, default_plot_fn)
    if not os.path.isabs(current_plot_fn):
        plotting_p_config[plot_fn_key] = os.path.join(script_dir, current_plot_fn)
    # If the key was missing and default_plot_fn was used, ensure it's set in plotting_p_config
    elif plot_fn_key not in plotting_p_config:
        plotting_p_config[plot_fn_key] = os.path.join(script_dir, default_plot_fn)

    config["cvo_model"]["plotting"] = plotting_p_config  # Store updated plotting paths

    # --- Initialize MagCalc ---
    logger.info(
        f"Initializing MagCalc with cache base: {cache_file_base}, mode: {cache_mode}"
    )
    try:
        calculator = mc.MagCalc(
            spin_magnitude=S_val,
            hamiltonian_params=params_val,
            cache_file_base=cache_file_base,
            cache_mode=cache_mode,
            spin_model_module=spin_model_cvo,  # Use the CVO spin model
        )
        nspins = calculator.nspins  # Get nspins from the calculator
        logger.info(
            f"MagCalc initialized. Number of spins in magnetic UC for LSWT: {nspins}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize MagCalc: {e}", exc_info=True)
        sys.exit(1)

    # --- Q-point generation (as in original script) ---
    # qsy represents q-points in r.l.u. for plotting
    qsy_plot_axis = np.arange(1, 3 + 0.02, 0.02)  # This is the axis for plotting
    q_list_calc = []  # q-points for calculation in radians/inv_angstrom

    # Assuming the original script's qy unit was 2*pi/b where b=8.383 Angstrom
    # This factor converts r.l.u. to calculation units.
    # Ensure your spin_model_cvo.py uses consistent lattice parameters if this is critical.
    # Or, make this conversion factor configurable.
    qy_conversion_factor = 2 * np.pi / 8.383
    for qy_rlu_val in qsy_plot_axis:
        qx_calc = 0
        qy_calc = qy_rlu_val * qy_conversion_factor
        qz_calc = 0
        q_list_calc.append(np.array([qx_calc, qy_calc, qz_calc]))
    q_vectors_array_calc = np.array(q_list_calc)
    # --- End Q-point generation ---

    # --- Dispersion Calculation ---
    disp_plotting_requested = tasks_config.get(
        "run_dispersion", False
    ) and tasks_config.get("plot_dispersion", False)
    disp_data_available = os.path.exists(disp_data_file)

    if tasks_config.get("run_dispersion", False):
        logger.info("--- Starting Dispersion Workflow ---")
        if tasks_config.get("calculate_dispersion_new", False):
            logger.info("Task: Calculate new dispersion data.")
            dispersion_energies = calculator.calculate_dispersion(q_vectors_array_calc)
            if dispersion_energies is not None:
                logger.info(f"Saving dispersion results to {disp_data_file}...")
                # Save q_vectors_array_calc (for potential reloading) and qsy_plot_axis (for plotting)
                results_to_save = {
                    "q_vectors_calc": q_vectors_array_calc,
                    "q_plot_axis": qsy_plot_axis,
                    "energies": dispersion_energies,
                }
                calculator.save_results(disp_data_file, results_to_save)
                logger.info("Dispersion results saved successfully.")
                disp_data_available = True  # Data is now available
            else:
                logger.warning(
                    "Dispersion calculation returned None, cannot save results."
                )
                disp_data_available = False  # Calculation failed, data not available
        else:
            logger.info(
                "Task: Skipping new dispersion calculation. Will attempt to load existing data for plotting."
            )
    else:
        logger.info("--- Skipping Dispersion Calculation Workflow ---")

    # --- Plotting Stage ---
    fig_to_show_save = None

    if disp_plotting_requested:
        if not disp_data_available and not tasks_config.get(
            "calculate_dispersion_new", False
        ):
            logger.warning(
                f"Dispersion data file {disp_data_file} not found and calculation was skipped. Cannot plot dispersion."
            )
        elif not disp_data_available and tasks_config.get(
            "calculate_dispersion_new", False
        ):
            logger.warning(
                f"Dispersion calculation was requested but failed or did not produce data. Cannot plot."
            )
        elif disp_data_available:  # Data exists or was just calculated successfully
            logger.info("Task: Plot dispersion.")
            logger.info(
                f"Loading dispersion data from {disp_data_file} for plotting..."
            )
            try:
                data = np.load(disp_data_file, allow_pickle=True)
                # Use loaded qsy_plot_axis if available, else fallback to the one generated in this run
                qsy_for_plot = data.get("q_plot_axis", qsy_plot_axis)
                En_loaded = data["energies"]
                logger.info("Dispersion data loaded successfully.")
            except (
                FileNotFoundError
            ):  # Should be caught by disp_data_available, but as a safeguard
                logger.error(
                    f"Dispersion data file not found: {disp_data_file}. Cannot plot."
                )
                En_loaded = None
            except Exception as e:
                logger.error(
                    f"Error loading dispersion data from {disp_data_file}: {e}",
                    exc_info=True,
                )
                En_loaded = None

            if En_loaded is not None:
                # MagCalc returns nspins positive energy modes.
                # Original script plotted 16 bands, implying nspins=8 for CVO.
                # This should match calculator.nspins.

                # Ensure loaded data matches expected q-points for plotting
                if len(En_loaded) != len(qsy_for_plot):
                    logger.warning(
                        f"Mismatch in number of q-points in loaded data ({len(En_loaded)}) and plotting axis ({len(qsy_for_plot)}). Plotting may be incorrect."
                    )
                    # Attempt to reconcile or use the shorter length
                    plot_len = min(len(En_loaded), len(qsy_for_plot))
                    qsy_final_plot_axis = qsy_for_plot[:plot_len]
                    En_for_plot = En_loaded[:plot_len]
                else:
                    plot_len = len(qsy_for_plot)
                    qsy_final_plot_axis = qsy_for_plot
                    En_for_plot = En_loaded

                if plot_len > 0 and (len(En_for_plot[0]) != nspins):
                    logger.warning(
                        f"Mismatch in number of bands. Loaded data has {len(En_for_plot[0])} bands, expected {nspins} from MagCalc. Plotting available bands."
                    )
                    num_bands_to_plot = min(len(En_for_plot[0]), nspins)
                elif plot_len == 0:
                    logger.error("No q-points to plot.")
                    num_bands_to_plot = 0
                else:  # Data shape matches expectations
                    num_bands_to_plot = nspins

                if num_bands_to_plot > 0:
                    Eky_bands = [
                        [
                            (
                                En_for_plot[i][band_idx]
                                if len(En_for_plot[i]) > band_idx
                                else np.nan
                            )
                            for i in range(plot_len)
                        ]
                        for band_idx in range(num_bands_to_plot)
                    ]

                    fig_to_show_save, ax = plt.subplots(1, 1, figsize=(8, 6))
                    # Cycle through more colors if nspins > 8
                    prop_cycle = plt.rcParams["axes.prop_cycle"]
                    colors = prop_cycle.by_key()["color"]

                    for band_idx in range(num_bands_to_plot):
                        ax.plot(
                            qsy_final_plot_axis,
                            Eky_bands[band_idx],
                            color=colors[band_idx % len(colors)],
                            linestyle="-",
                        )

                    # Plotting limits and labels from config or defaults
                    ax.set_xlim(plotting_p_config.get("q_limits_disp", [1, 3]))
                    ax.set_ylim(plotting_p_config.get("energy_limits_disp", [0, 10]))
                    ax.set_xticks(
                        plotting_p_config.get(
                            "q_ticks_disp", np.arange(1.0, 3.01, 0.25)
                        )
                    )  # Ensure last tick is inclusive
                    ax.set_xlabel("q$_y$ (r.l.u.)", fontsize=12)  # Added x-axis label
                    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
                    # Adjust y-ticks based on actual y-limits from config
                    y_lim_plot = ax.get_ylim()
                    ax.set_yticks(np.arange(y_lim_plot[0], y_lim_plot[1] + 1, 2))
                    ax.set_title(
                        plotting_p_config.get("disp_title", "CVO Spin Wave Dispersion")
                    )
                    ax.grid(axis="y", linestyle="--", alpha=0.7)
                    fig_to_show_save.tight_layout()
                else:
                    logger.error("No bands to plot based on loaded data or nspins.")
            else:  # En_loaded is None
                logger.warning(
                    "Dispersion data not available or could not be loaded. Cannot plot."
                )
        # End of disp_data_available check
    else:  # Not disp_plotting_requested
        logger.info("--- Skipping Dispersion Plotting ---")

    # Save and/or show the figure if one was created
    if fig_to_show_save:
        if plotting_p_config.get("save_plot"):
            plot_filepath = plotting_p_config.get(
                "disp_plot_filename"
            )  # Already made absolute
            if plot_filepath:
                try:
                    plt.savefig(plot_filepath)
                    logger.info(f"Plot saved to {plot_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save plot to {plot_filepath}: {e}")
            else:  # Should not happen if default is set
                logger.warning("Plot filename not determined for saving.")

        if plotting_p_config.get("show_plot"):
            plt.show()
        plt.close(fig_to_show_save)  # Close the figure
    # --- End Plotting Stage ---

    et = default_timer()
    logger.info(
        f"Total run-time: {np.round((et - st), 2)} sec ({np.round((et-st)/60, 2)} min)."
    )
