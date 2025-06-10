#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plots the minimum of the spin-wave dispersion for the aCVO model
along the q-path (0,1,0) to (0,2,0) r.l.u. (specifically qy from 1.5 to 2.0 r.l.u.)
as a function of the applied magnetic field H.
Also plots the mean canting angle.
"""

import os
import sys
import yaml
import numpy as np
import matplotlib.pyplot as plt
import timeit
import logging

# --- Add pyMagCalc directory to sys.path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# The directory containing the pyMagCalc package
PROJECT_ROOT_DIR = os.path.dirname(
    os.path.dirname(SCRIPT_DIR)
)  # This should be '.../research/magcalc/'
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# --- Import MagCalc and the specific CVO spin model ---
try:
    from pyMagCalc import magcalc as mc
    import spin_model_hc as cvo_model_module
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        f"Ensure the pyMagCalc package is in {PROJECT_ROOT_DIR} and spin_model_hc.py is in {SCRIPT_DIR} or accessible."
    )
    sys.exit(1)

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def plot_min_energy_vs_H_cvo():
    """
    Main function to calculate and plot minimum dispersion energy and canting angle vs. H.
    """
    script_start_time = timeit.default_timer()
    logger.info("Starting script to plot minimum energy vs. H for CVO.")

    # --- 1. Load Configuration ---
    config_filename = "config.yaml"
    config_filepath = os.path.join(SCRIPT_DIR, config_filename)

    if not os.path.exists(config_filepath):
        logger.error(f"Configuration file not found at {config_filepath}. Exiting.")
        return

    try:
        with open(config_filepath, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_filepath}")
    except Exception as e:
        logger.error(
            f"Error loading or parsing configuration file {config_filepath}: {e}. Exiting."
        )
        return

    cvo_config = config.get("cvo_model", {})
    if not cvo_config:
        logger.error(f"'cvo_model' section not found in {config_filepath}. Exiting.")
        return

    model_params_config = cvo_config.get("model_params", {})
    calc_settings_config = cvo_config.get("calculation", {})
    plotting_config = cvo_config.get("plotting", {})

    if not model_params_config:
        logger.error(f"'model_params' section not found in 'cvo_model'. Exiting.")
        return

    try:
        S_val = float(model_params_config["S"])
        J1 = float(model_params_config["J1"])
        J2_ratio = float(model_params_config["J2_ratio"])
        J3_ratio = float(model_params_config["J3_ratio"])
        G1 = float(model_params_config["G1"])
        Dx = float(model_params_config["Dx"])
        H_initial_config = float(model_params_config.get("H", 0.0))
    except KeyError as e:
        logger.error(
            f"Missing parameter {e} in 'model_params' section of config. Exiting."
        )
        return
    except ValueError as e:
        logger.error(
            f"Invalid numerical value for a parameter in 'model_params': {e}. Exiting."
        )
        return

    base_model_params = [J1, J1 * J2_ratio, J1 * J3_ratio, G1, Dx, H_initial_config]
    H_param_index = 5

    lb_cvo = plotting_config.get("lattice_b_cvo", 8.383)
    magcalc_cache_file_base_prefix = calc_settings_config.get(
        "cache_file_base_prefix", "CVO_model_cache"
    )
    magcalc_cache_mode = calc_settings_config.get("cache_mode", "auto")

    # --- 2. Define Scan Parameters ---
    H_values = np.linspace(0, 20, 21)  # 0 to 20 T, 1 T steps
    logger.info(
        f"Scanning H from {H_values[0]:.1f} T to {H_values[-1]:.1f} T in {len(H_values)} steps."
    )

    qy_rlu_min = 1.5
    qy_rlu_max = 2.0
    num_qy_points = int((qy_rlu_max - qy_rlu_min) / 0.01) + 1  # Approx 0.01 rlu step
    qy_values_rlu = np.linspace(qy_rlu_min, qy_rlu_max, num_qy_points)

    qy_conversion_factor = 2 * np.pi / lb_cvo
    q_vectors_cartesian_for_scan = np.array(
        [[0, qy_val * qy_conversion_factor, 0] for qy_val in qy_values_rlu]
    )
    logger.info(
        f"Calculating dispersion over {len(q_vectors_cartesian_for_scan)} q-points along (0,qy,0) "
        f"with qy in [{qy_rlu_min:.2f},{qy_rlu_max:.2f}] r.l.u."
    )

    # --- 3. Main Loop Over Magnetic Field Values ---
    results_list = []

    for current_H_field in H_values:
        loop_start_time = timeit.default_timer()
        logger.info(f"Processing H = {current_H_field:.2f} T...")

        current_model_params_for_minimization = list(base_model_params)
        current_model_params_for_minimization[H_param_index] = current_H_field

        logger.debug(
            f"  Calling get_field_optimized_state_for_lswt with params: {current_model_params_for_minimization}"
        )
        opt_thetas, Ud_numeric_optimized, classical_E = (
            cvo_model_module.get_field_optimized_state_for_lswt(
                current_model_params_for_minimization, S_val
            )
        )

        if Ud_numeric_optimized is None:
            logger.error(
                f"  Classical minimization failed for H = {current_H_field:.2f} T. Skipping."
            )
            results_list.append((current_H_field, np.nan, np.nan))
            continue

        mean_canting_angle = np.nan
        if opt_thetas is not None:
            mean_canting_angle = np.mean(90.0 - np.degrees(opt_thetas))
            logger.info(
                f"  Classical state for H={current_H_field:.2f} T: E={classical_E:.4f} meV, Mean Canting={mean_canting_angle:.2f} deg"
            )
            # Also log individual canting angles and theta angles
            canting_angles_degrees = 90.0 - np.degrees(opt_thetas)
            logger.info(
                f"  Optimal Theta Angles (degrees from +c): {np.degrees(opt_thetas)}"
            )
            logger.info(
                f"  Canting angles from a-b plane (degrees towards +z): {canting_angles_degrees}"
            )

        current_magcalc_cache_base = (
            f"{magcalc_cache_file_base_prefix}_S{S_val}_H{current_H_field:.2f}"
        )
        logger.debug(
            f"  Initializing MagCalc with cache base: {current_magcalc_cache_base}"
        )
        try:
            calculator = mc.MagCalc(
                spin_model_module=cvo_model_module,
                spin_magnitude=S_val,
                hamiltonian_params=current_model_params_for_minimization,
                cache_file_base=current_magcalc_cache_base,
                cache_mode=magcalc_cache_mode,
                Ud_numeric_override=None,
            )
        except Exception as e_mc_init:
            logger.error(
                f"  Error initializing MagCalc for H={current_H_field:.2f} T: {e_mc_init}. Skipping.",
                exc_info=True,
            )
            results_list.append((current_H_field, np.nan, mean_canting_angle))
            continue

        logger.debug(f"  Calculating dispersion for H={current_H_field:.2f} T...")
        dispersion_energies_list = calculator.calculate_dispersion(
            q_vectors_cartesian_for_scan
        )

        current_min_E = np.inf
        if dispersion_energies_list:
            all_positive_energies_for_H = []
            for energies_at_q in dispersion_energies_list:
                if energies_at_q is not None and energies_at_q.size > 0:
                    positive_energies = energies_at_q[
                        energies_at_q >= -1e-6
                    ]  # Allow small numerical noise
                    if positive_energies.size > 0:
                        all_positive_energies_for_H.extend(positive_energies)
            if all_positive_energies_for_H:
                current_min_E = np.nanmin(all_positive_energies_for_H)
            else:
                current_min_E = np.nan
                logger.warning(
                    f"  No valid (>=0) dispersion energies found for H={current_H_field:.2f} T."
                )
        else:
            current_min_E = np.nan
            logger.warning(
                f"  Dispersion calculation returned empty or None for H={current_H_field:.2f} T."
            )

        results_list.append((current_H_field, current_min_E, mean_canting_angle))
        loop_duration = timeit.default_timer() - loop_start_time
        logger.info(
            f"  Finished H = {current_H_field:.2f} T. Min Energy = {current_min_E:.4f} meV. Time: {loop_duration:.2f}s"
        )

    # --- 4. Plotting the Results ---
    H_values_plot = np.array([res[0] for res in results_list])
    min_energies_plot = np.array([res[1] for res in results_list])
    mean_canting_plot = np.array([res[2] for res in results_list])

    fig, ax1 = plt.subplots(figsize=(10, 7))
    color1 = "tab:red"
    ax1.set_xlabel("Magnetic Field H (T)")
    ax1.set_ylabel("Minimum Spin-Wave Energy (meV)", color=color1)
    ax1.plot(
        H_values_plot,
        min_energies_plot,
        marker="o",
        linestyle="-",
        color=color1,
        label="Min Energy",
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle=":", alpha=0.7)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.set_ylabel("Mean Canting Angle (degrees from a-b plane)", color=color2)
    ax2.plot(
        H_values_plot,
        mean_canting_plot,
        marker="s",
        linestyle="--",
        color=color2,
        label="Mean Canting Angle",
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle(
        f"CVO: Min Magnon Energy & Canting vs. H along (0,qy,0), qy in [{qy_rlu_min:.2f},{qy_rlu_max:.2f}] rlu",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")

    plot_filename = os.path.join(SCRIPT_DIR, "CVO_min_energy_canting_vs_H_plot.png")
    try:
        plt.savefig(plot_filename)
        logger.info(f"Plot saved to {plot_filename}")
    except Exception as e:
        logger.error(f"Failed to save plot to {plot_filename}: {e}")

    if plotting_config.get("show_plot", True):
        plt.show()
    plt.close(fig)

    # --- 5. Saving Data ---
    data_filename = os.path.join(SCRIPT_DIR, "CVO_min_energy_canting_vs_H_data.npz")
    try:
        np.savez_compressed(
            data_filename,
            H_values=H_values_plot,
            min_energies=min_energies_plot,
            mean_canting_angles=mean_canting_plot,
        )
        logger.info(f"Data saved to {data_filename}")
    except Exception as e:
        logger.error(f"Failed to save data to {data_filename}: {e}")

    total_script_duration = timeit.default_timer() - script_start_time
    logger.info(
        f"Total script execution time: {total_script_duration:.2f} seconds ({total_script_duration/60:.2f} minutes)."
    )


if __name__ == "__main__":
    plot_min_energy_vs_H_cvo()
