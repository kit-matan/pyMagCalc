# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 01:22:57 2018
Modified for new MagCalc class structure.

@author: Kit Matan
@contributor: AI Assistant
"""
import numpy as np
from timeit import default_timer
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging
import yaml  # Added
import os  # Added
import sys

# --- Custom Module Imports ---
# Ensure the paths are correct for your project structure
try:
    # Assuming magcalc.py, classical_minimizer_cvo.py, and spin_model_cvo_Hc.py
    # are in the same directory or accessible via PYTHONPATH.
    import magcalc as mc
    import classical_minimizer_cvo as cm_cvo
    import spin_model_cvo_Hc as sm_cvo
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(
        "Please ensure magcalc.py, classical_minimizer_cvo.py, and spin_model_cvo_Hc.py are accessible."
    )
    sys.exit(1)
# --- End Custom Module Imports ---

# --- Basic Logging Setup ---
logger = logging.getLogger(__name__)
# --- End Logging Setup ---

if __name__ == "__main__":
    st = default_timer()

    # --- Load Configuration from YAML ---
    script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
    config_file_path = os.path.join(script_dir, "config_cvo_c.yaml")
    try:
        with open(config_file_path, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)

    # --- Configure Logging ---
    log_level_str = config.get("logging_level", "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- Parameters from Config ---
    S_val = config["spin_value"]
    p_hamiltonian = config["hamiltonian_params"]

    cache_cfg = config["cache"]
    cache_mode = cache_cfg["mode"]
    cache_file_prefix = cache_cfg.get("base_name_prefix", "cvo_Hc")
    cache_file_base = f"{cache_file_prefix}_S{S_val}_H{p_hamiltonian[5]}"

    lattice_cfg = config["lattice_params"]
    la = lattice_cfg["la"]
    lb = lattice_cfg["lb"]
    lc = lattice_cfg["lc"]

    energy_map_cfg = config["energy_map"]
    emin = energy_map_cfg["emin"]
    emax = energy_map_cfg["emax"]
    estep = energy_map_cfg["estep"]

    q_scan_cfg = config["q_scan"]
    qmin_rlu = q_scan_cfg["q_min"]
    qmax_rlu = q_scan_cfg["q_max"]
    qstep_rlu = q_scan_cfg["q_step"]
    scan_direction = q_scan_cfg["direction"]
    fixed_h_rlu = q_scan_cfg.get("fixed_h", 0.0)
    fixed_k_rlu = q_scan_cfg.get("fixed_k", 0.0)
    fixed_l_rlu = q_scan_cfg.get("fixed_l", 0.0)

    plot_cfg = config["plot_params"]
    lorentzian_width = plot_cfg["lorentzian_width"]
    vmax_scale = plot_cfg.get("vmax_scale", 5.0)
    # --- End Parameters from Config ---

    shift = 1e-5  # Small shift to avoid singularities at exact high-symmetry points

    if "discrete_q_points" in q_scan_cfg and q_scan_cfg["discrete_q_points"]:
        q_scan_values_r_l_u = np.array(q_scan_cfg["discrete_q_points"])
        logger.info(f"Using discrete q-points: {q_scan_values_r_l_u}")
    else:
        q_scan_values_r_l_u = np.arange(
            qmin_rlu + shift, qmax_rlu + shift + qstep_rlu, qstep_rlu
        )
    q_vectors_list = []

    # Determine scan parameters and plot labels
    kx_factor = 2 * np.pi / la
    ky_factor = 2 * np.pi / lb
    kz_factor = 2 * np.pi / lc

    scan_description_for_title = ""
    xlabel = f"{plot_cfg.get('xlabel_base', 'Q (r.l.u.)')}"

    if scan_direction == "h00":
        for q_val in q_scan_values_r_l_u:
            q_vec = np.array(
                [q_val * kx_factor, fixed_k_rlu * ky_factor, fixed_l_rlu * kz_factor]
            )
            q_vectors_list.append(q_vec)
        scan_description_for_title = f"(h,{fixed_k_rlu:.2f},{fixed_l_rlu:.2f})"
        xlabel += f" along H"
    elif scan_direction == "0k0":
        for q_val in q_scan_values_r_l_u:
            q_vec = np.array(
                [fixed_h_rlu * kx_factor, q_val * ky_factor, fixed_l_rlu * kz_factor]
            )
            q_vectors_list.append(q_vec)
        scan_description_for_title = f"({fixed_h_rlu:.2f},k,{fixed_l_rlu:.2f})"
        xlabel += f" along K"
    elif scan_direction == "00l":
        for q_val in q_scan_values_r_l_u:
            q_vec = np.array(
                [fixed_h_rlu * kx_factor, fixed_k_rlu * ky_factor, q_val * kz_factor]
            )
            q_vectors_list.append(q_vec)
        scan_description_for_title = f"({fixed_h_rlu:.2f},{fixed_k_rlu:.2f},l)"
        xlabel += f" along L"
    elif scan_direction == "hk0_scan_h":
        for q_val in q_scan_values_r_l_u:
            q_vec = np.array(
                [q_val * kx_factor, fixed_k_rlu * ky_factor, fixed_l_rlu * kz_factor]
            )
            q_vectors_list.append(q_vec)
        scan_description_for_title = f"(h,{fixed_k_rlu:.2f},{fixed_l_rlu:.2f})"
        xlabel += f" along H"
    elif scan_direction == "hk0_scan_k":
        for q_val in q_scan_values_r_l_u:
            q_vec = np.array(
                [fixed_h_rlu * kx_factor, q_val * ky_factor, fixed_l_rlu * kz_factor]
            )
            q_vectors_list.append(q_vec)
        scan_description_for_title = f"({fixed_h_rlu:.2f},k,{fixed_l_rlu:.2f})"
        xlabel += f" along K"
    # Add more specific scan directions as needed from your config options
    else:
        logger.error(
            f"Unknown or unsupported scan direction '{scan_direction}' in config."
        )
        sys.exit(1)

    if not q_vectors_list:
        logger.error("No Q vectors generated. Check q_scan configuration.")
        q_vectors_list.append(q_vec)
        sys.exit(1)

    logger.info("Finding classical ground state spin orientations...")
    optimal_theta_angles = cm_cvo.find_ground_state_orientations(
        p_hamiltonian, S_val, sm_cvo
    )

    if optimal_theta_angles is None:
        logger.error("Failed to find classical ground state. Exiting.")
        sys.exit(1)

    logger.info("Initializing MagCalc...")
    calculator = mc.MagCalc(
        spin_magnitude=S_val,
        hamiltonian_params=p_hamiltonian,
        cache_file_base=cache_file_base,
        spin_model_module=sm_cvo,
        cache_mode=cache_mode,
        optimal_spin_angles=optimal_theta_angles,
    )

    logger.info("Calculating S(Q,w)...")
    q_vectors_out, energies_out, intensities_out = calculator.calculate_sqw(
        q_vectors_list
    )
    logger.info("Calculated S(Q,w). Performing checks on raw data...")
    logger.debug(f"Shape of q_vectors_out: {np.shape(q_vectors_out)}")
    logger.debug(f"Shape of energies_out: {np.shape(energies_out)}")
    logger.debug(f"Shape of intensities_out: {np.shape(intensities_out)}")

    if (
        energies_out is not None
        and len(energies_out) > 0
        and intensities_out is not None
        and len(intensities_out) > 0
    ):
        # Log stats for a few representative Q-points
        q_indices_to_log = sorted(
            list(
                set(
                    [
                        0,
                        1,
                        len(energies_out) // 2,
                        len(energies_out) - 2,
                        len(energies_out) - 1,
                    ]
                )
            )
        )
        for q_idx in q_indices_to_log:
            if 0 <= q_idx < len(energies_out):
                q_val_rlu = (
                    q_scan_values_r_l_u[q_idx]
                    if q_idx < len(q_scan_values_r_l_u)
                    else "N/A"
                )
                logger.debug(f"--- Q-point index {q_idx} (q_r_l_u ~ {q_val_rlu}) ---")
                logger.debug(
                    f"Energies sample: {energies_out[q_idx][:5]}...{energies_out[q_idx][-5:]}"
                )  # Sample first and last 5
                logger.debug(
                    f"Intensities sample: {intensities_out[q_idx][:5]}...{intensities_out[q_idx][-5:]}"
                )
                if np.any(
                    energies_out[q_idx] < -1e-6
                ):  # Check for negative energies (allowing for small numerical noise around 0)
                    logger.warning(
                        f"Negative energies found at q_idx {q_idx} (q_r_l_u ~ {q_val_rlu}): {energies_out[q_idx][energies_out[q_idx] < -1e-6]}"
                    )
                if np.any(
                    intensities_out[q_idx] < 0
                ):  # Should ideally not happen if I_band > 0 is used later, but good to check raw output
                    logger.warning(
                        f"Negative intensities found at q_idx {q_idx} (q_r_l_u ~ {q_val_rlu}): {intensities_out[q_idx][intensities_out[q_idx] < 0]}"
                    )

    Ex_bins = np.arange(emin, emax, estep)
    intensity_map = np.zeros((len(Ex_bins), len(q_scan_values_r_l_u)))

    for i_ex, energy_bin_center in enumerate(Ex_bins):
        for j_q in range(
            len(q_scan_values_r_l_u)
        ):  # Iterate using index for q_scan_values_r_l_u
            energies_for_q = energies_out[
                j_q
            ]  # energies_out corresponds to q_vectors_list
            intensities_for_q = intensities_out[
                j_q
            ]  # intensities_out corresponds to q_vectors_list
            accumulated_intensity = 0.0
            for band_idx in range(len(energies_for_q)):
                E_band = energies_for_q[band_idx]
                I_band = intensities_for_q[band_idx]
                if not np.isnan(E_band) and not np.isnan(I_band) and I_band > 0:
                    accumulated_intensity += (
                        I_band
                        * (1.0 / np.pi)
                        * (lorentzian_width / 2.0)
                        / (
                            (energy_bin_center - E_band) ** 2
                            + (lorentzian_width / 2.0) ** 2
                        )
                    )
            intensity_map[i_ex, j_q] = accumulated_intensity

    X_mesh, Y_mesh = np.meshgrid(q_scan_values_r_l_u, Ex_bins)

    plot_vmin = intensity_map.min()
    plot_vmax = intensity_map.max() / vmax_scale
    if plot_vmax <= plot_vmin:
        plot_vmax = plot_vmin + 1e-9  # Avoids issues with zero intensity

    # For LogNorm, ensure vmin is positive and sensible if data can be zero
    # if plot_cfg.get('log_scale', False):
    #     safe_min_val = np.max([1e-5, intensity_map[intensity_map > 0].min()]) if np.any(intensity_map > 0) else 1e-5
    #     plt.pcolormesh(X_mesh, Y_mesh, intensity_map, norm=LogNorm(vmin=safe_min_val, vmax=intensity_map.max()), cmap='PuBu_r', shading='auto')
    # else:
    plt.pcolormesh(
        X_mesh,
        Y_mesh,
        intensity_map,
        vmin=plot_vmin,
        vmax=plot_vmax,
        cmap="PuBu_r",
        shading="auto",
    )

    plt.xlim([qmin_rlu, qmax_rlu])
    plt.ylim([emin, emax])
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(plot_cfg.get("ylabel", r"$\hbar\omega$ (meV)"), fontsize=12)
    plt.yticks(np.arange(emin, emax + 0.5, 2.0))
    plt.xticks(
        np.arange(qmin_rlu, qmax_rlu + 1, 1.0 if (qmax_rlu - qmin_rlu) > 2 else 0.5)
    )
    title_str = f"{plot_cfg.get('title_prefix', 'Spin Waves in a-Cu$_2$V$_2$O$_7$')} (H={p_hamiltonian[5]} T, scan: {scan_description_for_title})"
    plt.title(title_str)
    plt.colorbar(label=plot_cfg.get("colorbar_label", "Intensity (arb. units)"))

    et = default_timer()
    logger.info(f"Total run-time: {np.round((et - st) / 60, 2)} min.")
    plt.tight_layout()  # Adjust layout
    plt.show()

# %%
