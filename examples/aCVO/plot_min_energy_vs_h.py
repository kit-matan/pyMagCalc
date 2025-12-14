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
    import magcalc as mc
    import spin_model_hc as cvo_model_module
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        f"Ensure the magcalc package is in {PROJECT_ROOT_DIR} and spin_model_hc.py is in {SCRIPT_DIR} or accessible."
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
        Dy = float(model_params_config.get("Dy", -2.0))  # DM component for J1
        D3 = float(model_params_config.get("D3", 0.0))  # DM component for J3
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

    # The model expects 8 parameters: J1, J2, J3, G1, Dx, Dy, D3, H
    base_model_params = [
        J1,
        J1 * J2_ratio,
        J1 * J3_ratio,
        G1,
        Dx,
        Dy,
        D3,
        H_initial_config,
    ]
    H_param_index = 7  # H is the 8th parameter, so its index is 7

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

    # Define q=(0,2,0) r.l.u. for splitting calculation
    q_020_rlu = np.array([0, 2.0, 0])
    q_020_cartesian = np.array([0, q_020_rlu[1] * qy_conversion_factor, 0])

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
            results_list.append(
                (current_H_field, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            )  # Added NaNs for splitting
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
        res_scan = calculator.calculate_dispersion(
            q_vectors_cartesian_for_scan
        )
        dispersion_energies_list = res_scan.energies if res_scan else None

        current_min_E = np.inf
        q_at_min_E_rlu = np.nan
        if dispersion_energies_list:
            min_E_so_far = np.inf
            min_E_q_idx = -1
            for i, energies_at_q in enumerate(dispersion_energies_list):
                if energies_at_q is not None and energies_at_q.size > 0:
                    positive_energies = energies_at_q[
                        energies_at_q >= -1e-6
                    ]  # Allow small numerical noise
                    if positive_energies.size > 0:
                        local_min = np.nanmin(positive_energies)
                        if local_min < min_E_so_far:
                            min_E_so_far = local_min
                            min_E_q_idx = i
            if min_E_q_idx != -1:
                current_min_E = min_E_so_far
                q_at_min_E_rlu = qy_values_rlu[min_E_q_idx]
            else:
                current_min_E = np.nan
                logger.warning(
                    f"  No valid (>=0) dispersion energies found for H={current_H_field:.2f} T."
                )

        # --- Calculate splitting at (0,2,0) ---
        splitting_at_020 = np.nan
        e1_at_020 = np.nan
        e2_at_020 = np.nan
        logger.debug(
            f"  Calculating dispersion at q=(0,2,0) for H={current_H_field:.2f} T..."
        )
        res_020 = calculator.calculate_dispersion([q_020_cartesian])
        disp_at_020_list = res_020.energies if res_020 else None

        if disp_at_020_list and disp_at_020_list[0] is not None:
            energies_at_020_raw = disp_at_020_list[0]
            # Filter out very small or negative energies if they are non-physical, then sort
            valid_energies_at_020 = np.sort(
                energies_at_020_raw[energies_at_020_raw >= -1e-6]
            )

            if len(valid_energies_at_020) >= 2:
                e1_at_020 = valid_energies_at_020[0]
                e2_at_020 = valid_energies_at_020[1]
                splitting_at_020 = e2_at_020 - e1_at_020
                logger.info(
                    f"  Splitting at (0,2,0) for H={current_H_field:.2f} T: {splitting_at_020:.4f} meV (E1={e1_at_020:.4f}, E2={e2_at_020:.4f})"
                )
            elif len(valid_energies_at_020) == 1:
                e1_at_020 = valid_energies_at_020[0]
                logger.warning(
                    f"  Only one valid energy branch ({e1_at_020:.4f} meV) found at (0,2,0) for H={current_H_field:.2f} T. Cannot calculate splitting."
                )
            else:
                logger.warning(
                    f"  Less than two valid energy branches found at (0,2,0) for H={current_H_field:.2f} T. Cannot calculate splitting."
                )
        else:
            logger.warning(
                f"  Dispersion calculation at (0,2,0) failed or returned insufficient branches for H={current_H_field:.2f} T."
            )

        results_list.append(
            (
                current_H_field,
                current_min_E,
                mean_canting_angle,
                splitting_at_020,
                e1_at_020,
                e2_at_020,
                q_at_min_E_rlu,
            )
        )
        loop_duration = timeit.default_timer() - loop_start_time
        logger.info(
            f"  Finished H = {current_H_field:.2f} T. Min E = {current_min_E:.4f} meV at qy={q_at_min_E_rlu:.3f}. Time: {loop_duration:.2f}s"
        )

    # --- 4. Plotting the Results ---
    H_values_plot = np.array([res[0] for res in results_list], dtype=float)
    min_energies_plot = np.array([res[1] for res in results_list], dtype=float)
    mean_canting_plot = np.array([res[2] for res in results_list], dtype=float)
    splitting_plot = np.array([res[3] for res in results_list], dtype=float)
    e1_020_plot = np.array([res[4] for res in results_list], dtype=float)
    e2_020_plot = np.array([res[5] for res in results_list], dtype=float)
    q_min_loc_plot = np.array([res[6] for res in results_list], dtype=float)

    # Create a figure with two subplots
    fig, (ax1, ax3_bottom) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True
    )  # sharex for common H-axis

    # --- Top Subplot: Min Energy and Canting Angle ---
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

    ax2_top = ax1.twinx()
    color2 = "tab:blue"
    ax2_top.set_ylabel("Mean Canting Angle (degrees from a-b plane)", color=color2)
    ax2_top.plot(
        H_values_plot,
        mean_canting_plot,
        marker="s",
        linestyle="--",
        color=color2,
        label="Mean Canting Angle",
    )
    ax2_top.tick_params(axis="y", labelcolor=color2)

    # Third y-axis for the location of the minimum energy
    ax3_top = ax1.twinx()
    ax3_top.spines["right"].set_position(("axes", 1.18))  # Offset the new spine
    color3 = "tab:green"
    ax3_top.set_ylabel("qy of Min Energy (r.l.u.)", color=color3)
    ax3_top.plot(
        H_values_plot,
        q_min_loc_plot,
        marker="d",
        linestyle=":",
        color=color3,
        label="qy of Min E",
    )
    ax3_top.tick_params(axis="y", labelcolor=color3)

    ax1.set_title(
        f"CVO: Min Magnon Energy & Canting vs. H along (0,qy,0), qy in [{qy_rlu_min:.2f},{qy_rlu_max:.2f}] rlu",
        fontsize=14,
    )
    lines_top, labels_top = ax1.get_legend_handles_labels()
    lines2_top, labels2_top = ax2_top.get_legend_handles_labels()
    lines3_top, labels3_top = ax3_top.get_legend_handles_labels()
    ax1.legend(
        lines_top + lines2_top + lines3_top,
        labels_top + labels2_top + labels3_top,
        loc="upper left",
    )

    # --- Bottom Subplot: Splitting at (0,2,0) ---
    color_e1 = "darkgreen"
    color_e2 = "purple"
    ax3_bottom.plot(
        H_values_plot,
        e1_020_plot,
        marker="^",
        linestyle=":",
        color=color_e1,
        label="E1 at (0,2,0)",
    )
    ax3_bottom.plot(
        H_values_plot,
        e2_020_plot,
        marker="v",
        linestyle=":",
        color=color_e2,
        label="E2 at (0,2,0)",
    )
    ax3_bottom.set_ylabel("Energy at (0,2,0) (meV)", color="black")
    ax3_bottom.tick_params(axis="y", labelcolor="black")
    ax3_bottom.grid(True, linestyle=":", alpha=0.7)

    ax4_bottom = ax3_bottom.twinx()
    color_split = "orangered"
    ax4_bottom.plot(
        H_values_plot,
        splitting_plot,
        marker="x",
        linestyle="-",
        color=color_split,
        label="Splitting E2-E1 at (0,2,0)",
    )
    ax4_bottom.set_ylabel("Energy Splitting at (0,2,0) (meV)", color=color_split)
    ax4_bottom.tick_params(axis="y", labelcolor=color_split)
    ax3_bottom.set_xlabel("Magnetic Field H (T)")
    ax3_bottom.set_title(
        "CVO: Energy Branches and Splitting at q=(0,2,0) r.l.u. vs. H", fontsize=14
    )
    lines_bottom, labels_bottom = ax3_bottom.get_legend_handles_labels()
    lines2_bottom, labels2_bottom = ax4_bottom.get_legend_handles_labels()
    ax4_bottom.legend(
        lines_bottom + lines2_bottom, labels_bottom + labels2_bottom, loc="upper right"
    )

    fig.tight_layout(
        rect=[0, 0.03, 0.88, 0.97]
    )  # Adjust for overall suptitle if added, or just general spacing

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
    # Save to proj_root/cache/data/
    cache_data_dir = os.path.join(PROJECT_ROOT_DIR, "cache", "data")
    os.makedirs(cache_data_dir, exist_ok=True)
    data_filename = os.path.join(cache_data_dir, "CVO_min_energy_canting_vs_H_data.npz")
    try:
        np.savez_compressed(
            data_filename,
            H_values=H_values_plot,
            min_energies=min_energies_plot,
            mean_canting_angles=mean_canting_plot,
            splitting_at_020=splitting_plot,
            q_location_of_min_energy=q_min_loc_plot,
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
