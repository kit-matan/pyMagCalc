# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018

@author: Kit Matan

Create an intensity contour map of Q and energy for spin-waves in a CVO-like material.
(Updated to use MagCalc class and load configuration from YAML)
"""
import numpy as np
from timeit import default_timer
import magcalc as mc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import yaml
import sys
import os

# Assume a spin model module for CVO exists
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_config_path = os.path.join(
        script_dir, "config.yaml"
    )  # Assumes config.yaml is in the same directory
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
    J1 = model_p_config.get("J1", 2.49)
    J2_ratio = model_p_config.get("J2_ratio", 1.12)
    J3_ratio = model_p_config.get("J3_ratio", 2.03)
    G1 = model_p_config.get("G1", 0.28)
    Dx = model_p_config.get("Dx", 2.67)
    H_field = model_p_config.get("H", 0.0)
    params_val = [J1, J2_ratio * J1, J3_ratio * J1, G1, Dx, H_field]

    cache_mode = calc_p_config.get("cache_mode", "r")
    cache_file_base = calc_p_config.get("cache_file_base", "CVO_model_cache")
    sqw_data_file = os.path.join(
        script_dir, output_p_config.get("sqw_data_filename", "CVO_sqw_data.npz")
    )

    plot_fn_key = "sqw_plot_filename"
    default_plot_fn = "CVO_sqw_plot.png"
    current_plot_fn = plotting_p_config.get(plot_fn_key, default_plot_fn)
    if not os.path.isabs(current_plot_fn):
        plotting_p_config[plot_fn_key] = os.path.join(script_dir, current_plot_fn)
    elif plot_fn_key not in plotting_p_config:
        plotting_p_config[plot_fn_key] = os.path.join(script_dir, default_plot_fn)
    config["cvo_model"]["plotting"] = plotting_p_config

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
            spin_model_module=spin_model_cvo,
        )
        nspins = calculator.nspins
        logger.info(
            f"MagCalc initialized. Number of spins in magnetic UC for LSWT: {nspins}"
        )
    except Exception as e:
        logger.error(f"Failed to initialize MagCalc: {e}", exc_info=True)
        sys.exit(1)

    # --- Q-point generation for S(Q,w) map ---
    # Parameters from original script / config
    qmin_rlu_config = plotting_p_config.get("q_limits_sqw", [-5, 5])[0]
    qmax_rlu_config = plotting_p_config.get("q_limits_sqw", [-5, 5])[1]
    qstep_rlu = tasks_config.get("sqw_q_step", 0.02)
    lb = 8.383  # Lattice parameter b, from original script
    qy_conversion_factor = 2 * np.pi / lb

    # Define and apply the requested shift
    # "e/1e-3" means np.e / (1 * 10^3) = np.e * 1000
    q_shift = np.e / 1e5
    logger.info(
        f"Applying q-shift of {q_shift:.6f} r.l.u. to the S(Q,w) map calculation range."
    )

    qmin_rlu_shifted = qmin_rlu_config + q_shift
    qmax_rlu_shifted = qmax_rlu_config + q_shift

    # qsy_plot_axis will contain the shifted r.l.u. values
    qsy_plot_axis = np.arange(qmin_rlu_shifted, qmax_rlu_shifted + qstep_rlu, qstep_rlu)

    q_list_calc = []
    for (
        qy_rlu_shifted_val
    ) in qsy_plot_axis:  # Iterate over already shifted r.l.u. values
        q_list_calc.append(np.array([0, qy_rlu_shifted_val * qy_conversion_factor, 0]))
    q_vectors_array_calc = np.array(q_list_calc)
    # --- End Q-point generation ---

    # --- S(Q,w) Calculation ---
    sqw_plotting_requested = tasks_config.get(
        "run_sqw_map", False
    ) and tasks_config.get("plot_sqw_map", False)
    sqw_data_available = os.path.exists(sqw_data_file)

    if tasks_config.get("run_sqw_map", False):
        logger.info("--- Starting S(Q,w) Map Workflow ---")
        if tasks_config.get("calculate_sqw_map_new", False):
            logger.info("Task: Calculate new S(Q,w) data.")
            qout_calc, En_calc, Sqwout_calc = calculator.calculate_sqw(
                q_vectors_array_calc
            )
            if (
                qout_calc is not None
                and En_calc is not None
                and Sqwout_calc is not None
            ):
                logger.info(f"Saving S(Q,w) results to {sqw_data_file}...")
                results_to_save = {
                    "q_vectors_calc": qout_calc,  # Save calculated q-vectors
                    "q_plot_axis": qsy_plot_axis,  # Save r.l.u. axis for plotting
                    "energies": En_calc,
                    "sqw_values": Sqwout_calc,
                }
                calculator.save_results(sqw_data_file, results_to_save)
                logger.info("S(Q,w) results saved successfully.")
                sqw_data_available = True
            else:
                logger.warning("S(Q,w) calculation returned None, cannot save results.")
                sqw_data_available = False
        else:
            logger.info(
                "Task: Skipping new S(Q,w) calculation. Will attempt to load existing data for plotting."
            )
    else:
        logger.info("--- Skipping S(Q,w) Map Calculation Workflow ---")

    # --- Plotting Stage ---
    fig_to_show_save = None

    if sqw_plotting_requested:
        if not sqw_data_available and not tasks_config.get(
            "calculate_sqw_map_new", False
        ):
            logger.warning(
                f"S(Q,w) data file {sqw_data_file} not found and calculation was skipped. Cannot plot."
            )
        elif not sqw_data_available and tasks_config.get(
            "calculate_sqw_map_new", False
        ):
            logger.warning(
                f"S(Q,w) calculation was requested but failed or did not produce data. Cannot plot."
            )
        elif sqw_data_available:
            logger.info("Task: Plot S(Q,w) map.")
            logger.info(f"Loading S(Q,w) data from {sqw_data_file} for plotting...")
            try:
                data = np.load(sqw_data_file, allow_pickle=True)
                qsy_plot_loaded = data.get(
                    "q_plot_axis", qsy_plot_axis
                )  # Use loaded qsy if available
                En_loaded_tuples = data["energies"]  # This is a tuple of arrays
                Sqwout_loaded_tuples = data["sqw_values"]  # This is a tuple of arrays
                logger.info("S(Q,w) data loaded successfully.")

                # Convert tuples of arrays to lists of arrays for easier processing if needed
                En_loaded = [np.array(e) for e in En_loaded_tuples]
                Sqwout_loaded = [np.array(s) for s in Sqwout_loaded_tuples]

            except FileNotFoundError:
                logger.error(
                    f"S(Q,w) data file not found: {sqw_data_file}. Cannot plot."
                )
                En_loaded, Sqwout_loaded = None, None
            except Exception as e:
                logger.error(
                    f"Error loading S(Q,w) data from {sqw_data_file}: {e}",
                    exc_info=True,
                )
                En_loaded, Sqwout_loaded = None, None

            if En_loaded is not None and Sqwout_loaded is not None:
                emin_plot = plotting_p_config.get("energy_limits_sqw", [0, 12])[0]
                emax_plot = plotting_p_config.get("energy_limits_sqw", [0, 12])[1]
                estep_plot = tasks_config.get("sqw_energy_step", 0.05)
                wid_plot = tasks_config.get("sqw_resolution_width", 0.2)

                Ex_plot_axis = np.arange(emin_plot, emax_plot + estep_plot, estep_plot)
                intMat = np.zeros((len(Ex_plot_axis), len(qsy_plot_loaded)))

                if len(En_loaded) != len(qsy_plot_loaded) or len(Sqwout_loaded) != len(
                    qsy_plot_loaded
                ):
                    logger.error(
                        "Mismatch between q-points and loaded S(Q,w) data. Cannot plot."
                    )
                else:
                    for i_e, ex_val in enumerate(Ex_plot_axis):
                        for j_q, _ in enumerate(qsy_plot_loaded):
                            fint_val = 0
                            if (
                                En_loaded[j_q] is not None
                                and Sqwout_loaded[j_q] is not None
                                and len(En_loaded[j_q]) > 0
                            ):
                                for band_idx in range(
                                    len(En_loaded[j_q])
                                ):  # Iterate over bands
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

                    fig_to_show_save, ax = plt.subplots(1, 1, figsize=(10, 6))
                    X_mesh, Y_mesh = np.meshgrid(qsy_plot_loaded, Ex_plot_axis)

                    logger.info(f"Intensity matrix (intMat) shape: {intMat.shape}")
                    logger.info(
                        f"Min value in intMat: {np.min(intMat)}, Max value: {np.max(intMat)}"
                    )
                    if np.any(intMat > 0):
                        logger.info(
                            f"Min positive value in intMat: {np.min(intMat[intMat > 0])}"
                        )
                        # Determine a reasonable vmin for LogNorm, avoiding zero or very small numbers
                        min_val_for_plot = np.min(intMat[intMat > 0])
                        if (
                            min_val_for_plot < 1e-6
                        ):  # If min positive is too small, set a floor
                            min_val_for_plot = 1e-6
                    else:
                        logger.info("All values in intMat are zero or negative.")
                        min_val_for_plot = 1e-6  # Default if no positive data

                    max_intensity = np.max(intMat) if np.any(intMat > 0) else 1.0

                    # Ensure vmax is somewhat larger than vmin for LogNorm to work well
                    if max_intensity <= min_val_for_plot:
                        max_intensity = (
                            min_val_for_plot * 100
                        )  # Or some other factor, ensure it's significantly larger

                    # If, after adjustments, max_intensity is still effectively zero or too close to min_val_for_plot
                    if (
                        max_intensity < min_val_for_plot * 1.1
                    ):  # Check if max is barely above min
                        logger.warning(
                            f"Max intensity ({max_intensity:.2e}) is too close to or less than min intensity for log ({min_val_for_plot:.2e}). Plot might be blank or uninformative."
                        )
                        # Attempt to show something, even if it's just a flat color
                        if np.any(intMat > 0):  # If there's any positive data at all
                            max_intensity = (
                                np.max(intMat[intMat > 0]) * 1.1
                            )  # Try to make vmax slightly larger
                            if max_intensity <= min_val_for_plot:  # Final fallback
                                max_intensity = min_val_for_plot * 100
                        else:  # All data is zero or negative
                            min_val_for_plot = 1e-6
                            max_intensity = 1e-4

                    logger.info(
                        f"Plotting with LogNorm: vmin={min_val_for_plot:.2e}, vmax={max_intensity:.2e}"
                    )

                    norm_choice = LogNorm(vmin=min_val_for_plot, vmax=max_intensity)

                    pcm = ax.pcolormesh(
                        X_mesh,
                        Y_mesh,
                        intMat,
                        norm=norm_choice,  # Apply LogNorm
                        cmap="PuBu_r",
                        shading="auto",
                    )

                    # Define the x-axis range explicitly based on the X_mesh used for pcolormesh.
                    # X_mesh's columns correspond to qsy_plot_loaded.
                    # Ensure qsy_plot_loaded has data before trying to get min/max.
                    # This was already good, but let's be explicit that xlim matches X_mesh.
                    if qsy_plot_loaded.size > 0:
                        ax.set_xlim([np.min(qsy_plot_loaded), np.max(qsy_plot_loaded)])
                        logger.info(
                            f"Setting x-axis limits (xlim) to: [{np.min(qsy_plot_loaded):.2f}, {np.max(qsy_plot_loaded):.2f}] based on shifted data."
                        )
                    # If it's empty (e.g. error in loading/calc), fall back to config or default.

                    ax.set_ylim(plotting_p_config.get("energy_limits_sqw", [0, 12]))
                    ax.set_xlabel(
                        r"q$_y$ (r.l.u.)", fontsize=12
                    )  # Also made this a raw string for consistency
                    ax.set_ylabel(r"$\hbar\omega$ (meV)", fontsize=12)
                    ax.set_yticks(np.arange(emin_plot, emax_plot + 1, 2.0))

                    # Set x-ticks based on the *actual shifted* Q-range of the data being plotted.
                    if qsy_plot_loaded.size > 0:
                        q_tick_min = np.min(qsy_plot_loaded)
                        q_tick_max = np.max(qsy_plot_loaded)

                        # Set x-ticks with a step of 1 r.l.u. within the shifted data range
                        # Ensure q_tick_min is rounded appropriately for arange if needed,
                        # or use ceil/floor to ensure the range is covered.
                        # For simplicity, we'll start from ceil(min) and go to floor(max).
                        start_tick = np.ceil(q_tick_min)
                        end_tick = np.floor(q_tick_max)

                        # Ensure there's at least one tick if the range is very small,
                        # or if start_tick > end_tick after rounding.
                        q_ticks_actual_range = (
                            np.arange(start_tick, end_tick + 1, 1.0)
                            if start_tick <= end_tick
                            else np.array([q_tick_min, q_tick_max])
                        )
                        ax.set_xticks(q_ticks_actual_range)
                        # Optional: Format tick labels if they become too long/messy
                        # ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
                    else:  # Fallback if qsy_plot_loaded is empty
                        ax.set_xticks(
                            np.arange(qmin_rlu_config, qmax_rlu_config + 1, 1)
                        )

                    ax.set_title(
                        plotting_p_config.get("sqw_title", "CVO S(Q,ω) Intensity Map")
                    )
                    fig_to_show_save.colorbar(pcm, label="S(Q,ω) (arb. units)")
                    fig_to_show_save.tight_layout()
            else:
                logger.warning(
                    "S(Q,w) data not available or could not be loaded. Cannot plot."
                )
    else:
        logger.info("--- Skipping S(Q,w) Map Plotting ---")

    # Save and/or show the figure
    if fig_to_show_save:
        if plotting_p_config.get("save_plot"):
            plot_filepath = plotting_p_config.get("sqw_plot_filename")
            if plot_filepath:
                try:
                    plt.savefig(plot_filepath)
                    logger.info(f"Plot saved to {plot_filepath}")
                except Exception as e:
                    logger.error(f"Failed to save plot to {plot_filepath}: {e}")
            else:
                logger.warning("Plot filename not determined for saving.")
        if plotting_p_config.get("show_plot"):
            plt.show()
        plt.close(fig_to_show_save)

    et = default_timer()
    logger.info(
        f"Total run-time: {np.round((et - st), 2)} sec ({np.round((et-st)/60, 2)} min)."
    )

# %%
